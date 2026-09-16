"""Teacher semantic labeling pipeline and compact-JSON SFT data build."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from collections import Counter


"""DeepSeek teacher labeling pipeline for the semantic dataset.

Five resumable stages build 2 Hz observation windows and Fossen-lite evidence,
assemble deployable-observation prompts, call the DeepSeek teacher, parse the
eight yes/no action-direction labels into the 9D semantic vector, and audit the
result (distribution, prompt-leakage checks, parquet alignment).  The pipeline
never overwrites an existing output directory."""
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
SEMANTIC_ORDER = [
    "CRUISE",
    "ALIGN_GOAL",
    "AVOID_LEFT",
    "AVOID_RIGHT",
    "AVOID_UP",
    "AVOID_DOWN",
    "BRAKE",
    "REVERSE",
    "STOP_HOLD",
]
DS_LABELS = SEMANTIC_ORDER[:8]
SOURCE_DATASETS = {
    "v1": "teacher_rollouts",
    "v2": "teacher_rollouts_custom_a",
    "v3": "teacher_rollouts_custom_b",
}
SFT_BALANCED_JSONL = DATASETS_ROOT / "sft_balanced" / "llm_sft_balanced.jsonl"
SFT_STRICT_JSONL = DATASETS_ROOT / "sft_strict" / "llm_sft_strict.jsonl"
CONTROL_HZ = 10
STRIDE_STEPS = 5
HISTORY_OFFSETS_STEPS = [-15, -10, -5, 0]
HISTORY_TIME_OFFSETS_S = [-1.5, -1.0, -0.5, 0.0]
WINDOW_SPAN_S = 1.5
ACTION_COLUMNS = [
    "teacher_action_forward",
    "teacher_action_left",
    "teacher_action_up",
    "teacher_action_pitch",
    "teacher_action_yaw",
]
K_LAT = 0.25
K_VERT = 0.25
SAFE_FRONT_M = 12.0
SAFE_SIDE_M = 8.0
SAFE_VERTICAL_M = 8.0
THRESHOLDS = {
    "lateral_proxy_norm_m_s": 1.50,
    "vertical_proxy_norm_m_s": 1.30,
    "progress_loss_norm_m_s": 0.30,
    "drag_proxy_norm_m2_s2": 2.25,
    "progress_good_m_per_s": 0.08,
}
def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
def _magnitude_bin(norm_abs: float) -> str:
    value = abs(float(norm_abs))
    if value < 0.10:
        return "none"
    if value < 0.30:
        return "weak"
    if value < 0.60:
        return "moderate"
    return "strong"
def _signed_bin(value: float, denom: float, positive_name: str, negative_name: str) -> str:
    norm_abs = abs(value) / max(abs(denom), 1e-9)
    mag = _magnitude_bin(norm_abs)
    if mag == "none":
        return "neutral"
    direction = positive_name if value >= 0.0 else negative_name
    return f"{direction}_{mag}"
def _action_vector(row: dict[str, Any]) -> np.ndarray:
    return np.array([_as_float(row, column) for column in ACTION_COLUMNS], dtype=np.float64)
def _velocity_body(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            _as_float(row, "linear_velocity_body_x"),
            _as_float(row, "linear_velocity_body_y"),
            _as_float(row, "linear_velocity_body_z"),
        ],
        dtype=np.float64,
    )
def _angular_rate_body(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            _as_float(row, "angular_velocity_body_x"),
            _as_float(row, "angular_velocity_body_y"),
            _as_float(row, "angular_velocity_body_z"),
        ],
        dtype=np.float64,
    )
def _goal_direction(row: dict[str, Any]) -> np.ndarray:
    vector = np.array(
        [
            _as_float(row, "goal_relative_body_x"),
            _as_float(row, "goal_relative_body_y"),
            _as_float(row, "goal_relative_body_z"),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return vector / norm
def _progress_rate(history_rows: list[dict[str, Any]]) -> float:
    if len(history_rows) < 2:
        return 0.0
    first, last = history_rows[0], history_rows[-1]
    return (_as_float(first, "distance_to_goal_m") - _as_float(last, "distance_to_goal_m")) / WINDOW_SPAN_S
def _previous_action(step_to_row: dict[int, dict[str, Any]], step_id: int) -> list[float]:
    previous = step_to_row.get(step_id - 1)
    if previous is None:
        return [0.0] * 5
    return _round_list(_action_vector(previous).tolist(), 4)
def _sample_id(source: str, episode_id: int, anchor_step: int) -> str:
    return f"teacher_{source}_e{int(episode_id):05d}_s{int(anchor_step):05d}"
def _build_window_records(
    transitions: pd.DataFrame,
    source: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for episode_id, group in transitions.groupby("episode_id", sort=True):
        group = group.sort_values("step_id")
        step_to_row = {int(row["step_id"]): row for row in group.to_dict("records")}
        steps = [int(step) for step in group["step_id"].tolist()]

        anchor_candidates = [
            step for step in steps if step % STRIDE_STEPS == 0 and all(s in step_to_row for s in (step - 15, step - 10, step - 5, step))
        ]

        for anchor_step in anchor_candidates:
            history_rows = [step_to_row[anchor_step + offset] for offset in HISTORY_OFFSETS_STEPS]
            anchor_row = history_rows[-1]
            records.append(
                {
                    "sample_id": _sample_id(source, episode_id, anchor_step),
                    "source_dataset": source,
                    "episode_id": int(episode_id),
                    "anchor_step": anchor_step,
                    "window_steps": [anchor_step + offset for offset in HISTORY_OFFSETS_STEPS],
                    "window_time_offsets_s": HISTORY_TIME_OFFSETS_S,
                    "history_rows": history_rows,
                    "step_to_row": step_to_row,
                    "episode_outcome": str(anchor_row.get("episode_outcome", "")),
                }
            )
    return records
def _fossen_lite_trace(history_rows: list[dict[str, Any]], step_to_row: dict[int, dict[str, Any]]) -> dict[str, Any]:
    anchor = history_rows[-1]
    anchor_step = int(anchor["step_id"])
    prev_action = _previous_action(step_to_row, anchor_step)

    v = _velocity_body(anchor)
    lat_raw = float(v[1])
    vert_raw = float(v[2])

    lat_drift = lat_raw - K_LAT * prev_action[1]
    vert_drift = vert_raw - K_VERT * prev_action[2]

    drag_x = float(v[0] * abs(v[0]))
    drag_y = float(v[1] * abs(v[1]))
    drag_z = float(v[2] * abs(v[2]))
    drag_norm = float(math.sqrt(drag_x**2 + drag_y**2 + drag_z**2))

    observed_progress = _progress_rate(history_rows)
    expected_progress = _clamp(prev_action[0]) * THRESHOLDS["progress_good_m_per_s"]
    progress_loss = max(0.0, expected_progress - observed_progress)

    front = _as_float(anchor, "front_clearance_m")
    left = _as_float(anchor, "left_clearance_m")
    right = _as_float(anchor, "right_clearance_m")
    up = _as_float(anchor, "up_clearance_m")
    down = _as_float(anchor, "down_clearance_m")

    lat_bin = _signed_bin(lat_drift, THRESHOLDS["lateral_proxy_norm_m_s"], "body_y_positive", "body_y_negative")
    vert_bin = _signed_bin(vert_drift, THRESHOLDS["vertical_proxy_norm_m_s"], "body_z_positive", "body_z_negative")
    loss_bin = _magnitude_bin(progress_loss / THRESHOLDS["progress_loss_norm_m_s"])
    drag_bin = _magnitude_bin(drag_norm / THRESHOLDS["drag_proxy_norm_m2_s2"])

    return {
        "anchor_step": anchor_step,
        "previous_executed_action": _round_list(prev_action, 4),
        "observed_progress_rate_m_s": _round_float(observed_progress, 5),
        "expected_progress_proxy_m_s": _round_float(expected_progress, 5),
        "lateral_drift_proxy": {
            "formula": "lateral_drift_proxy = v_body_y - k_lat * previous_lateral_action",
            "constants": {"k_lat": K_LAT},
            "raw_values": {"v_body_y_m_s": _round_float(lat_raw, 5), "previous_lateral_action": _round_float(prev_action[1], 5)},
            "value_m_s": _round_float(lat_drift, 5),
            "bin": lat_bin,
            "interpretation": (
                "positive above threshold means drifting right; support AVOID_LEFT only when left clearance margin is positive; "
                "negative below threshold means drifting left; support AVOID_RIGHT only when right clearance margin is positive."
            ),
        },
        "vertical_drift_proxy": {
            "formula": "vertical_drift_proxy = v_body_z - k_vert * previous_vertical_action",
            "constants": {"k_vert": K_VERT},
            "raw_values": {"v_body_z_m_s": _round_float(vert_raw, 5), "previous_vertical_action": _round_float(prev_action[2], 5)},
            "value_m_s": _round_float(vert_drift, 5),
            "bin": vert_bin,
            "interpretation": (
                "negative below threshold means drifting down; support AVOID_UP only when up clearance margin is positive; "
                "positive above threshold means drifting up; support AVOID_DOWN only when down clearance margin is positive."
            ),
        },
        "drag_proxy": {
            "formula": "drag_proxy_i = v_i * abs(v_i); drag_proxy_norm = sqrt(sum_i drag_proxy_i^2)",
            "constants": {},
            "raw_values": {"v_body_m_s": _round_list(v.tolist(), 5)},
            "value_m2_s2": _round_list([drag_x, drag_y, drag_z], 5),
            "norm_m2_s2": _round_float(drag_norm, 5),
            "bin": drag_bin,
            "interpretation": "high drag proxy with weak forward progress supports BRAKE.",
        },
        "progress_loss_proxy": {
            "formula": "progress_loss_proxy = max(0, expected_progress_proxy - observed_progress)",
            "constants": {"progress_good_m_per_s": THRESHOLDS["progress_good_m_per_s"]},
            "raw_values": {
                "expected_progress_proxy_m_s": _round_float(expected_progress, 5),
                "observed_progress_rate_m_s": _round_float(observed_progress, 5),
            },
            "value_m_s": _round_float(progress_loss, 5),
            "bin": loss_bin,
            "interpretation": "high progress loss supports ALIGN_GOAL; with low front clearance also supports BRAKE or REVERSE.",
        },
        "clearance_margin": {
            "formula": "clearance_margin = clearance - safe_clearance",
            "constants": {"safe_front_m": SAFE_FRONT_M, "safe_side_m": SAFE_SIDE_M, "safe_vertical_m": SAFE_VERTICAL_M},
            "raw_values": {"front_m": _round_float(front, 5), "left_m": _round_float(left, 5), "right_m": _round_float(right, 5), "up_m": _round_float(up, 5), "down_m": _round_float(down, 5)},
            "value_m": {
                "front": _round_float(front - SAFE_FRONT_M, 5),
                "left": _round_float(left - SAFE_SIDE_M, 5),
                "right": _round_float(right - SAFE_SIDE_M, 5),
                "up": _round_float(up - SAFE_VERTICAL_M, 5),
                "down": _round_float(down - SAFE_VERTICAL_M, 5),
            },
            "interpretation": "avoidance labels activate only when the target direction has positive margin; prefer larger margin when compensating drift.",
        },
    }
def _format_fossen_trace_for_prompt(fossen: dict[str, Any]) -> str:
    lat = fossen["lateral_drift_proxy"]
    vert = fossen["vertical_drift_proxy"]
    drag = fossen["drag_proxy"]
    loss = fossen["progress_loss_proxy"]
    clear = fossen["clearance_margin"]
    lines = [
        f"lateral_drift_proxy: formula={lat['formula']}; k_lat={lat['constants']['k_lat']}; "
        f"substitution: v_body_y={lat['raw_values']['v_body_y_m_s']:+.5f} m/s, previous_lateral_action={lat['raw_values']['previous_lateral_action']:+.5f}; "
        f"value={lat['value_m_s']:+.5f} m/s; bin={lat['bin']}.",
        f"vertical_drift_proxy: formula={vert['formula']}; k_vert={vert['constants']['k_vert']}; "
        f"substitution: v_body_z={vert['raw_values']['v_body_z_m_s']:+.5f} m/s, previous_vertical_action={vert['raw_values']['previous_vertical_action']:+.5f}; "
        f"value={vert['value_m_s']:+.5f} m/s; bin={vert['bin']}.",
        f"drag_proxy: formula={drag['formula']}; substitution: v_body={drag['raw_values']['v_body_m_s']}; "
        f"value={drag['value_m2_s2']} m^2/s^2; norm={drag['norm_m2_s2']:.5f}; bin={drag['bin']}.",
        f"progress_loss_proxy: formula={loss['formula']}; substitution: expected={loss['raw_values']['expected_progress_proxy_m_s']:.5f}, observed={loss['raw_values']['observed_progress_rate_m_s']:+.5f}; "
        f"value={loss['value_m_s']:.5f} m/s; bin={loss['bin']}.",
        f"clearance_margin: formula={clear['formula']}; safe=[front={clear['constants']['safe_front_m']},side={clear['constants']['safe_side_m']},vertical={clear['constants']['safe_vertical_m']}]; "
        f"value_m={clear['value_m']}.",
    ]
    return "\n".join("  " + line for line in lines)
DS_LABEL_SET = set(DS_LABELS)
ACTION_DIRECTION_SYSTEM_PROMPT = (
    "You are an onboard semantic intent labeling expert for an AUV. "
    "Use only deployable observation summaries and locally computed Fossen-lite evidence. "
    "The Fossen-lite values were computed locally. Do not recompute them. "
    "Use only the information explicitly provided in the user message. "
    "Your task is bounded semantic reasoning: decide which action-direction semantic intent labels should be active. "
    "Output valid JSON only."
)
STRICT_JSON_SCHEMA_TEXT = """Return JSON only with exactly these keys:
CRUISE, ALIGN_GOAL, AVOID_LEFT, AVOID_RIGHT, AVOID_UP, AVOID_DOWN, BRAKE, REVERSE.

Allowed values are only "yes" or "no".
Do not output explanations, numbers, markdown, code fences, headings, confidence, main intent, secondary intent, or extra keys."""
SEMANTIC_DEFINITIONS_TEXT = """Semantic label definitions:
- CRUISE: continue nominal forward navigation when progress is good and no urgent braking or reversing is needed.
- ALIGN_GOAL: correct heading or attitude toward the goal when goal direction is off-axis.
- AVOID_LEFT: the recommended semantic maneuver direction is leftward.
- AVOID_RIGHT: the recommended semantic maneuver direction is rightward.
- AVOID_UP: the recommended semantic maneuver direction is upward.
- AVOID_DOWN: the recommended semantic maneuver direction is downward.
- BRAKE: reduce forward motion when front risk, drag, or progress loss suggests unsafe continuation.
- REVERSE: command reverse when front clearance is critically low or braking is insufficient."""
DECISION_CONSTRAINTS_TEXT = """Decision constraints:
- Multiple labels may be yes.
- Opposite avoidance labels should not both be yes unless the evidence is ambiguous.
- If progress_loss_proxy is none and front_margin is positive, BRAKE should usually be no.
- If goal direction is off-axis, ALIGN_GOAL should usually be yes.
- AVOID_LEFT/RIGHT/UP/DOWN indicate the recommended action direction, not the disturbance source direction.
- Use drift proxies as physical evidence only; do not automatically invert left drift into AVOID_RIGHT or right drift into AVOID_LEFT.
- Previous action direction and available clearance should be used as action-direction evidence when deciding avoidance labels."""
FORBIDDEN_PROMPT_MARKERS = [
    "route_id",
    "current_speed_m_s",
    "current_offset_deg",
    "current_class_id",
    "relative_water_velocity",
    "current_velocity_body",
    "waypoint",
    "next_waypoint",
    "distance_to_next_waypoint",
    "oracle_",
    "path_tangent",
    "planned_",
    "reward_",
    "cross_track",
    "physics_evidence",
    "dvl_body_velocity_m_s=[",
    "imu_angular_rate_rad_s=[",
    "goal_dir_body=[",
    "clearance_m=[",
    "previous_executed_action=[",
]
def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse failed: {path}:{line_index}: {exc}") from exc
    return rows
def _round_float(value: float, digits: int = 4) -> float:
    return _round_float(value, digits)
def _round_list(values: list[float] | np.ndarray, digits: int = 4) -> list[float]:
    return [_round_float(float(value), digits) for value in list(values)]
def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _as_float(row, key, default)
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def _sample_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return _sha256_text(system_prompt + "\n---USER---\n" + user_prompt)
def _mag_bin(value: float, weak: float, medium: float, strong: float) -> str:
    av = abs(float(value))
    if av < weak:
        return "none"
    if av < medium:
        return "weak"
    if av < strong:
        return "medium"
    return "strong"
def _goal_direction_bin(goal: np.ndarray) -> str:
    x, y, z = [float(v) for v in goal.tolist()]
    if x > 0.75 and abs(y) < 0.15 and abs(z) < 0.15:
        return "nearly_aligned"
    forward = "forward" if x >= 0.0 else "behind"
    lateral = "left" if y > 0.15 else "right" if y < -0.15 else "center"
    vertical = "up" if z > 0.15 else "down" if z < -0.15 else "level"
    return f"{forward}_{lateral}_{vertical}"
def _goal_distance_trend(history_rows: list[dict[str, Any]]) -> str:
    first = _as_float(history_rows[0], "distance_to_goal_m")
    last = _as_float(history_rows[-1], "distance_to_goal_m")
    delta = first - last
    if delta > 0.20:
        return "decreasing"
    if delta < -0.20:
        return "increasing"
    return "flat"
def _observed_progress_bin(progress_rate_m_s: float) -> str:
    if progress_rate_m_s >= 0.30:
        return "positive_good"
    if progress_rate_m_s >= THRESHOLDS["progress_good_m_per_s"]:
        return "positive_weak"
    if progress_rate_m_s >= -0.02:
        return "none"
    return "negative"
def _front_clearance_bin(front_m: float) -> str:
    margin = front_m - SAFE_FRONT_M
    if margin <= 0.0:
        return "critical"
    if margin <= 2.0:
        return "low_risk"
    if margin <= 5.0:
        return "low_safe"
    if margin <= 18.0:
        return "medium_safe"
    return "high_safe"
def _side_clearance_bin(value_m: float, safe_m: float) -> str:
    margin = value_m - safe_m
    if margin <= 0.0:
        return "low"
    if margin <= 8.0:
        return "medium_safe"
    if margin <= 30.0:
        return "high"
    return "very_high"
def _signed_motion_bin(value: float, positive_name: str, negative_name: str, weak: float, medium: float, strong: float) -> str:
    mag = _mag_bin(value, weak=weak, medium=medium, strong=strong)
    if mag == "none":
        return "neutral"
    direction = positive_name if value >= 0.0 else negative_name
    return f"{direction}_{mag}"
def _forward_motion_bin(vx: float) -> str:
    return _signed_motion_bin(vx, "forward", "reverse", weak=0.05, medium=0.45, strong=0.90)
def _lateral_motion_bin(vy: float) -> str:
    return _signed_motion_bin(vy, "right_drift", "left_drift", weak=0.05, medium=0.35, strong=0.70)
def _vertical_motion_bin(vz: float) -> str:
    return _signed_motion_bin(vz, "upward", "downward", weak=0.05, medium=0.35, strong=0.75)
def _previous_forward_action_bin(action: float) -> str:
    if action <= -0.20:
        return "reverse"
    if action <= -0.05:
        return "braking"
    if abs(action) < 0.10:
        return "neutral"
    if action >= 0.65:
        return "forward_high"
    return "forward_low"
def _previous_lateral_action_bin(action: float) -> str:
    if abs(action) < 0.10:
        return "neutral"
    direction = "left" if action > 0.0 else "right"
    level = "high" if abs(action) >= 0.50 else "low"
    return f"{direction}_{level}"
def _previous_vertical_action_bin(action: float) -> str:
    if abs(action) < 0.10:
        return "neutral"
    direction = "up" if action > 0.0 else "down"
    level = "high" if abs(action) >= 0.50 else "low"
    return f"{direction}_{level}"
def _margin_bin(margin_m: float) -> str:
    if margin_m <= 0.0:
        return "negative_or_zero"
    if margin_m <= 3.0:
        return "positive_low"
    if margin_m <= 10.0:
        return "positive_medium"
    if margin_m <= 30.0:
        return "positive_high"
    return "positive_very_high"
def _drift_bin_from_value(value: float, kind: str) -> str:
    if kind == "lateral":
        return _signed_motion_bin(value, "right_drift", "left_drift", weak=0.05, medium=0.35, strong=0.70)
    if kind == "vertical":
        return _signed_motion_bin(value, "upward", "downward", weak=0.05, medium=0.25, strong=0.60)
    raise ValueError(f"unknown drift kind: {kind}")
def _build_raw_history(history_rows: list[dict[str, Any]], step_to_row: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    raw_frames: list[dict[str, Any]] = []
    for index, row in enumerate(history_rows):
        step_id = int(row["step_id"])
        raw_frames.append(
            {
                "time_offset_s": HISTORY_TIME_OFFSETS_S[index],
                "step_id": step_id,
                "body_velocity_m_s": _round_list(_velocity_body(row), 5),
                "imu_angular_rate_rad_s": _round_list(_angular_rate_body(row), 5),
                "goal_dir_body": _round_list(_goal_direction(row), 5),
                "goal_distance_m": _round_float(_as_float(row, "distance_to_goal_m"), 5),
                "progress_to_goal_m": _round_float(_as_float(row, "progress_to_goal_m"), 5),
                "clearance_m": {
                    "front": _round_float(_as_float(row, "front_clearance_m"), 5),
                    "left": _round_float(_as_float(row, "left_clearance_m"), 5),
                    "right": _round_float(_as_float(row, "right_clearance_m"), 5),
                    "up": _round_float(_as_float(row, "up_clearance_m"), 5),
                    "down": _round_float(_as_float(row, "down_clearance_m"), 5),
                },
                "previous_executed_action": _previous_action(step_to_row, step_id),
            }
        )
    return raw_frames
def _build_observation_summary(history_rows: list[dict[str, Any]], step_to_row: dict[int, dict[str, Any]]) -> dict[str, Any]:
    anchor = history_rows[-1]
    anchor_step = int(anchor["step_id"])
    velocity = _velocity_body(anchor)
    previous_action = _previous_action(step_to_row, anchor_step)
    progress_rate = _progress_rate(history_rows)
    front = _as_float(anchor, "front_clearance_m")
    left = _as_float(anchor, "left_clearance_m")
    right = _as_float(anchor, "right_clearance_m")
    up = _as_float(anchor, "up_clearance_m")
    down = _as_float(anchor, "down_clearance_m")

    return {
        "observation_window": "4_frames_at_2hz",
        "current_frame": "t+0.0s",
        "goal_direction": _goal_direction_bin(_goal_direction(anchor)),
        "goal_distance_trend": _goal_distance_trend(history_rows),
        "observed_progress": _observed_progress_bin(progress_rate),
        "front_clearance": _front_clearance_bin(front),
        "left_clearance": _side_clearance_bin(left, SAFE_SIDE_M),
        "right_clearance": _side_clearance_bin(right, SAFE_SIDE_M),
        "up_clearance": _side_clearance_bin(up, SAFE_VERTICAL_M),
        "down_clearance": _side_clearance_bin(down, SAFE_VERTICAL_M),
        "body_forward_motion": _forward_motion_bin(float(velocity[0])),
        "body_lateral_motion": _lateral_motion_bin(float(velocity[1])),
        "body_vertical_motion": _vertical_motion_bin(float(velocity[2])),
        "previous_forward_action": _previous_forward_action_bin(float(previous_action[0])),
        "previous_lateral_action": _previous_lateral_action_bin(float(previous_action[1])),
        "previous_vertical_action": _previous_vertical_action_bin(float(previous_action[2])),
    }
def _format_observation_summary_for_prompt(summary: dict[str, Any]) -> str:
    keys = [
        "observation_window",
        "current_frame",
        "goal_direction",
        "goal_distance_trend",
        "observed_progress",
        "front_clearance",
        "left_clearance",
        "right_clearance",
        "up_clearance",
        "down_clearance",
        "body_forward_motion",
        "body_lateral_motion",
        "body_vertical_motion",
        "previous_forward_action",
        "previous_lateral_action",
        "previous_vertical_action",
    ]
    return "\n".join(f"- {key}: {summary.get(key, '')}" for key in keys)
def _add_local_substitutions(fossen: dict[str, Any]) -> dict[str, Any]:
    lat = fossen["lateral_drift_proxy"]
    vert = fossen["vertical_drift_proxy"]
    drag = fossen["drag_proxy"]
    loss = fossen["progress_loss_proxy"]

    lat["substitution"] = (
        f"{lat['raw_values']['v_body_y_m_s']:+.5f} - {lat['constants']['k_lat']:.2f} * "
        f"{lat['raw_values']['previous_lateral_action']:+.5f}"
    )
    vert["substitution"] = (
        f"{vert['raw_values']['v_body_z_m_s']:+.5f} - {vert['constants']['k_vert']:.2f} * "
        f"{vert['raw_values']['previous_vertical_action']:+.5f}"
    )
    drag["substitution"] = f"v_body={drag['raw_values']['v_body_m_s']} -> v_i * abs(v_i)={drag['value_m2_s2']}"
    loss["substitution"] = (
        f"max(0, {loss['raw_values']['expected_progress_proxy_m_s']:.5f} - "
        f"{loss['raw_values']['observed_progress_rate_m_s']:+.5f})"
    )
    return fossen
def _compact_fossen_evidence(fossen: dict[str, Any]) -> dict[str, Any]:
    lat = fossen["lateral_drift_proxy"]
    vert = fossen["vertical_drift_proxy"]
    drag = fossen["drag_proxy"]
    loss = fossen["progress_loss_proxy"]
    clear = fossen["clearance_margin"]
    clear_values = clear["value_m"]

    lat_value = float(lat["value_m_s"])
    vert_value = float(vert["value_m_s"])
    drag_norm = float(drag["norm_m2_s2"])
    loss_value = float(loss["value_m_s"])
    margins = {name: _margin_bin(float(value)) for name, value in clear_values.items()}
    if loss["bin"] == "none":
        loss_interpretation = "observed progress is sufficient; progress loss alone does not support BRAKE or REVERSE."
    else:
        loss_interpretation = "progress loss supports ALIGN_GOAL; with low front margin it can support BRAKE or REVERSE."

    return {
        "lateral_drift_proxy": {
            "formula": "v_body_y - k_lat * previous_lateral_action",
            "value": f"{lat_value:+.5f} m/s",
            "bin": _drift_bin_from_value(lat_value, "lateral"),
            "interpretation": "lateral drift evidence only; the final AVOID_LEFT or AVOID_RIGHT label must indicate the recommended action direction, not an automatic opposite-drift compensation.",
        },
        "vertical_drift_proxy": {
            "formula": "v_body_z - k_vert * previous_vertical_action",
            "value": f"{vert_value:+.5f} m/s",
            "bin": _drift_bin_from_value(vert_value, "vertical"),
            "interpretation": "vertical drift evidence only; the final AVOID_UP or AVOID_DOWN label must indicate the recommended action direction, not an automatic opposite-drift compensation.",
        },
        "drag_proxy_norm": {
            "formula": "sqrt((v_x|v_x|)^2 + (v_y|v_y|)^2 + (v_z|v_z|)^2)",
            "value": f"{drag_norm:.5f}",
            "bin": drag["bin"],
            "interpretation": "large value means strong resistance or aggressive relative motion.",
        },
        "progress_loss_proxy": {
            "formula": "max(0, expected_progress_proxy - observed_progress)",
            "value": f"{loss_value:.5f} m/s",
            "bin": loss["bin"],
            "interpretation": loss_interpretation,
        },
        "clearance_margin": {
            "front": margins["front"],
            "left": margins["left"],
            "right": margins["right"],
            "up": margins["up"],
            "down": margins["down"],
            "interpretation": "avoidance labels should be consistent with positive clearance margin in the recommended action direction.",
        },
    }
def _format_compact_fossen_for_prompt(compact: dict[str, Any]) -> str:
    lat = compact["lateral_drift_proxy"]
    vert = compact["vertical_drift_proxy"]
    drag = compact["drag_proxy_norm"]
    loss = compact["progress_loss_proxy"]
    clear = compact["clearance_margin"]
    return "\n".join(
        [
            "These values were computed locally from deployable observations. Do not recompute them.",
            "",
            "1. lateral_drift_proxy",
            f"formula: {lat['formula']}",
            f"value: {lat['value']}",
            f"bin: {lat['bin']}",
            f"interpretation: {lat['interpretation']}",
            "",
            "2. vertical_drift_proxy",
            f"formula: {vert['formula']}",
            f"value: {vert['value']}",
            f"bin: {vert['bin']}",
            f"interpretation: {vert['interpretation']}",
            "",
            "3. drag_proxy_norm",
            f"formula: {drag['formula']}",
            f"value: {drag['value']}",
            f"bin: {drag['bin']}",
            f"interpretation: {drag['interpretation']}",
            "",
            "4. progress_loss_proxy",
            f"formula: {loss['formula']}",
            f"value: {loss['value']}",
            f"bin: {loss['bin']}",
            f"interpretation: {loss['interpretation']}",
            "",
            "5. clearance_margin",
            f"front: {clear['front']}",
            f"left: {clear['left']}",
            f"right: {clear['right']}",
            f"up: {clear['up']}",
            f"down: {clear['down']}",
            f"interpretation: {clear['interpretation']}",
        ]
    )
def stage_build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"B output directory must not exist: {output_dir}")
    output_dir.mkdir(parents=True)

    raw_window_rows: list[dict[str, Any]] = []
    fossen_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    selected_by_counts: Counter[str] = Counter()

    for source in args.sources:
        if source not in SOURCE_DATASETS:
            raise ValueError(f"unknown source: {source}")
        transitions_path = DATASETS_ROOT / SOURCE_DATASETS[source] / "raw_rollouts" / "transitions.parquet"
        if not transitions_path.is_file():
            raise FileNotFoundError(transitions_path)

        transitions = pd.read_parquet(transitions_path)
        windows = _build_window_records(transitions, source)
        if args.limit_per_source > 0:
            rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in source))
            rng.shuffle(windows)
            windows = windows[: args.limit_per_source]
        source_counts[source] = len(windows)

        for window in windows:
            sample_id = window["sample_id"]
            history_rows = window["history_rows"]
            step_to_row = window["step_to_row"]

            fossen = _add_local_substitutions(_fossen_lite_trace(history_rows, step_to_row))
            compact_fossen = _compact_fossen_evidence(fossen)
            summary = _build_observation_summary(history_rows, step_to_row)

            selected_by: list[str] = ["full_raw"]
            for tag in selected_by:
                selected_by_counts[tag] += 1

            raw_window_rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source,
                    "episode_id": int(window["episode_id"]),
                    "anchor_step": int(window["anchor_step"]),
                    "window_steps": [int(step) for step in window["window_steps"]],
                    "window_time_offsets_s": HISTORY_TIME_OFFSETS_S,
                    "episode_outcome": window["episode_outcome"],
                    "selected_by": selected_by,
                    "raw_history": _build_raw_history(history_rows, step_to_row),
                }
            )

            fossen_rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source,
                    "episode_id": int(window["episode_id"]),
                    "anchor_step": int(window["anchor_step"]),
                    "full_fossen_lite_trace": fossen,
                    "formatted_full_trace_local_only": _format_fossen_trace_for_prompt(fossen),
                    "compact_fossen_for_prompt": compact_fossen,
                }
            )
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source,
                    "episode_id": int(window["episode_id"]),
                    "anchor_step": int(window["anchor_step"]),
                    "semanticized_observation_summary": summary,
                    "formatted_summary_for_prompt": _format_observation_summary_for_prompt(summary),
                    "episode_outcome": window["episode_outcome"],
                }
            )

        del transitions
        print(f"[BUILD] {source}: windows={len(windows)}")

    computed_dir = output_dir / "computed_traces"
    _write_jsonl(computed_dir / "raw_observation_windows.jsonl", raw_window_rows)
    _write_jsonl(computed_dir / "fossen_lite_traces.jsonl", fossen_rows)
    _write_jsonl(computed_dir / "observation_summaries.jsonl", summary_rows)
    _write_json(
        computed_dir / "build_report.json",
        {
            "stage": "build_c_action_direction",
            "created_at": _now_iso(),
            "source_window_counts": dict(sorted(source_counts.items())),
            "selected_by_counts": dict(sorted(selected_by_counts.items())),
            "total_windows": len(raw_window_rows),
            "window_offsets_s": HISTORY_TIME_OFFSETS_S,
            "control_hz": CONTROL_HZ,
            "stride_steps": STRIDE_STEPS,
            "safe_clearance_m": {"front": SAFE_FRONT_M, "side": SAFE_SIDE_M, "vertical": SAFE_VERTICAL_M},

            "ds_sees_raw_arrays": False,
            "ds_sees_compact_fossen_only": True,
        },
    )
    print(f"[BUILD] total windows={len(raw_window_rows)} output={output_dir}")
def _build_user_prompt(summary_text: str, compact_fossen_text: str) -> str:
    return "\n\n".join(
        [
            "Deployable observation summary:",
            summary_text,
            "Fossen-lite computed evidence:",
            compact_fossen_text,
            SEMANTIC_DEFINITIONS_TEXT,
            DECISION_CONSTRAINTS_TEXT,
            STRICT_JSON_SCHEMA_TEXT,
        ]
    )
def stage_prompts(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    computed_dir = output_dir / "computed_traces"
    windows_path = computed_dir / "raw_observation_windows.jsonl"
    summaries_path = computed_dir / "observation_summaries.jsonl"
    traces_path = computed_dir / "fossen_lite_traces.jsonl"
    if not windows_path.is_file() or not summaries_path.is_file() or not traces_path.is_file():
        raise FileNotFoundError("run build stage first")

    windows = {row["sample_id"]: row for row in _load_jsonl(windows_path)}
    summaries = {row["sample_id"]: row for row in _load_jsonl(summaries_path)}
    traces = {row["sample_id"]: row for row in _load_jsonl(traces_path)}
    sample_ids = [row["sample_id"] for row in _load_jsonl(windows_path)]

    source_filter = set(args.sources or [])
    pending_ids = [sid for sid in sample_ids if (not source_filter or windows[sid]["source_dataset"] in source_filter)]
    if args.limit > 0:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(pending_ids)
        pending_ids = pending_ids[: args.limit]

    requests: list[dict[str, Any]] = []
    leakage_hits: list[dict[str, Any]] = []
    for request_index, sample_id in enumerate(pending_ids):
        window = windows[sample_id]
        summary = summaries[sample_id]
        trace = traces[sample_id]
        summary_text = summary["formatted_summary_for_prompt"]
        compact_text = _format_compact_fossen_for_prompt(trace["compact_fossen_for_prompt"])
        user_prompt = _build_user_prompt(summary_text, compact_text)
        hits = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in user_prompt]
        if hits:
            leakage_hits.append({"sample_id": sample_id, "forbidden_markers_hit": hits})
        prompt_sha256 = _sample_prompt_hash(ACTION_DIRECTION_SYSTEM_PROMPT, user_prompt)
        requests.append(
            {
                "request_index": request_index,
                "sample_id": sample_id,
                "source_dataset": window["source_dataset"],
                "episode_id": int(window["episode_id"]),
                "anchor_step": int(window["anchor_step"]),
                "window_steps": window["window_steps"],
                "system_prompt": ACTION_DIRECTION_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "model": "<filled-at-run>",
                "prompt_sha256": prompt_sha256,
                "prompt_version": "deepseek_action_direction_v1",
            }
        )

    requests_dir = output_dir / "prompts"
    _write_jsonl(requests_dir / "ds_requests.jsonl", requests)
    _write_json(
        requests_dir / "ds_request_manifest.json",
        {
            "stage": "prompts_c_action_direction",
            "created_at": _now_iso(),
            "total_requests": len(requests),
            "sources": sorted(source_filter) if source_filter else sorted(SOURCE_DATASETS),
            "prompt_version": "deepseek_action_direction_v1",
            "system_prompt": ACTION_DIRECTION_SYSTEM_PROMPT,
            "strict_json_schema_in_user_prompt": True,
            "ds_sees_raw_arrays": False,
            "ds_sees_compact_fossen_only": True,
            "forbidden_input_leakage_hits": len(leakage_hits),
            "leakage_examples": leakage_hits[:20],
        },
    )
    print(f"[PROMPTS] written={len(requests)} leakage_hits={len(leakage_hits)} path={requests_dir / 'ds_requests.jsonl'}")
def _extract_json_text(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
def _parse_yes_no(raw: str) -> tuple[dict[str, str] | None, str]:
    if raw is None or not str(raw).strip():
        return None, "empty_response"
    text = _extract_json_text(str(raw))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "not_json"
    if not isinstance(data, dict):
        return None, "not_json_object"
    keys = set(data.keys())
    if keys != DS_LABEL_SET:
        if keys < DS_LABEL_SET:
            return None, "missing_key"
        return None, "extra_key"
    result: dict[str, str] = {}
    for label in DS_LABELS:
        value = str(data[label]).strip().lower()
        if value not in {"yes", "no"}:
            return None, "invalid_value"
        result[label] = value
    return result, "ok"
def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text", part.get("content", ""))
                if text:
                    pieces.append(str(text))
        return "".join(pieces)
    return str(content)
def _read_valid_done(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    done: set[tuple[str, str]] = set()
    for row in _load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        prompt_sha = str(row.get("prompt_sha256", ""))
        raw = str(row.get("raw_response", ""))
        labels, status = _parse_yes_no(raw)
        if sample_id and prompt_sha and row.get("parse_status") == "ok" and labels is not None and status == "ok":
            done.add((sample_id, prompt_sha))
    return done
def _call_deepseek_one(
    request_row: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_s: float,
    max_retries: int,
    use_response_format: bool,
    thinking: str,
    save_full_body_on_fail: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": request_row["system_prompt"]},
        {"role": "user", "content": request_row["user_prompt"]},
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 180,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    if thinking != "omit":
        payload["thinking"] = {"type": thinking}

    retry_rows: list[dict[str, Any]] = []
    raw_response = ""
    parse_error = "not_started"
    http_status = 0
    latency_ms = 0.0
    attempts_used = 0
    last_body: dict[str, Any] | None = None
    last_message_keys: list[str] = []
    last_reasoning_preview = ""
    last_body_preview = ""

    for attempt in range(1, max_retries + 1):
        attempts_used = attempt
        started = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            latency_ms = (time.perf_counter() - started) * 1000.0
            http_status = int(response.status_code)
            last_body_preview = response.text[:4000]
            if response.status_code != 200:
                parse_error = f"http_{response.status_code}"
                retry_rows.append(
                    {
                        "sample_id": request_row["sample_id"],
                        "request_index": request_row.get("request_index"),
                        "attempt": attempt,
                        "error": parse_error,
                        "detail": response.text[:500],
                        "created_at": _now_iso(),
                    }
                )
                time.sleep(min(2.0**attempt, 30.0))
                continue
            body = response.json()
            last_body = body
            message = body.get("choices", [{}])[0].get("message", {}) or {}
            last_message_keys = sorted(str(key) for key in message.keys()) if isinstance(message, dict) else []
            if isinstance(message, dict):
                raw_response = _message_content_to_text(message.get("content", ""))
                last_reasoning_preview = _message_content_to_text(message.get("reasoning_content", ""))[:1000]
            else:
                raw_response = ""
                last_reasoning_preview = ""
            labels, status = _parse_yes_no(raw_response)
            if labels is not None and status == "ok":
                parse_error = ""
                break
            parse_error = status
            retry_rows.append(
                {
                    "sample_id": request_row["sample_id"],
                    "request_index": request_row.get("request_index"),
                    "attempt": attempt,
                    "error": status,
                    "raw_response_preview": raw_response[:500],
                    "message_keys": last_message_keys,
                    "reasoning_content_preview": last_reasoning_preview,
                    "created_at": _now_iso(),
                }
            )
            time.sleep(min(2.0**attempt, 30.0))
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            parse_error = f"{type(exc).__name__}: {exc}"
            retry_rows.append(
                {
                    "sample_id": request_row["sample_id"],
                    "request_index": request_row.get("request_index"),
                    "attempt": attempt,
                    "error": parse_error,
                    "created_at": _now_iso(),
                }
            )
            time.sleep(min(2.0**attempt, 30.0))

    labels, final_status = _parse_yes_no(raw_response)
    ok = labels is not None and final_status == "ok"
    record = {
        "request_index": request_row.get("request_index"),
        "sample_id": request_row["sample_id"],
        "request_id": f"req_{int(time.time() * 1000):x}_{request_row.get('request_index', 0)}",
        "source_dataset": request_row.get("source_dataset", ""),
        "episode_id": request_row.get("episode_id"),
        "anchor_step": request_row.get("anchor_step"),
        "prompt_sha256": request_row["prompt_sha256"],
        "prompt_version": request_row.get("prompt_version", "deepseek_action_direction_v1"),
        "raw_response": raw_response,
        "model": model,
        "created_at": _now_iso(),
        "http_status": http_status,
        "attempts": attempts_used,
        "latency_ms": round(latency_ms, 3),
        "parse_status": "ok" if ok else "failed",
        "parse_error": "" if ok else (final_status if final_status != "empty_response" else parse_error or "empty_response"),
        "message_keys": last_message_keys,
        "reasoning_content_preview": "" if ok else last_reasoning_preview,
    }
    if save_full_body_on_fail and not ok:
        record["response_body"] = last_body if last_body is not None else last_body_preview
    return record, retry_rows
def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]
def stage_run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("missing DeepSeek API key (--api-key or DEEPSEEK_API_KEY env)")

    output_dir = Path(args.output_dir).resolve()
    requests_path = output_dir / "prompts" / "ds_requests.jsonl"
    raw_path = output_dir / "ds_outputs" / "raw_responses.jsonl"
    retry_path = output_dir / "ds_outputs" / "retry_records.jsonl"
    if not requests_path.is_file():
        raise FileNotFoundError(requests_path)

    request_rows = _load_jsonl(requests_path)
    valid_done = _read_valid_done(raw_path)
    pending = [row for row in request_rows if (row["sample_id"], row["prompt_sha256"]) not in valid_done]
    pending.sort(key=lambda row: int(row.get("request_index", 0)))
    if args.max_samples > 0:
        pending = pending[: args.max_samples]

    current_concurrency = max(1, int(args.concurrency))
    batch_size = args.run_batch_size if args.run_batch_size > 0 else max(current_concurrency * 20, current_concurrency)
    print(
        f"[RUN] pending={len(pending)} valid_done={len(valid_done)} model={args.model} "
        f"concurrency={current_concurrency} batch_size={batch_size}"
    )

    ok_count = 0
    fail_count = 0
    rate_limit_count = 0
    empty_count = 0
    api_error_count = 0
    processed = 0

    for batch in _chunks(pending, batch_size):
        batch_results: list[dict[str, Any]] = []
        batch_retries: list[dict[str, Any]] = []
        with futures.ThreadPoolExecutor(max_workers=current_concurrency) as executor:
            future_to_row = {
                executor.submit(
                    _call_deepseek_one,
                    row,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    use_response_format=not args.no_response_format,
                    thinking=args.thinking,
                    save_full_body_on_fail=args.save_full_body_on_fail,
                ): row
                for row in batch
            }
            for future in futures.as_completed(future_to_row):
                record, retry_rows = future.result()
                batch_results.append(record)
                batch_retries.extend(retry_rows)

        for record in sorted(batch_results, key=lambda row: int(row.get("request_index", 0))):
            _append_jsonl(raw_path, record)
            processed += 1
            if record.get("parse_status") == "ok":
                ok_count += 1
            else:
                fail_count += 1
                err = str(record.get("parse_error", ""))
                if err == "empty_response":
                    empty_count += 1
                elif err == "http_429":
                    rate_limit_count += 1
                else:
                    api_error_count += 1
        for retry in sorted(batch_retries, key=lambda row: (int(row.get("request_index", 0)), int(row.get("attempt", 0)))):
            _append_jsonl(retry_path, retry)

        batch_total = max(len(batch_results), 1)
        batch_empty = sum(1 for row in batch_results if row.get("parse_error") == "empty_response")
        batch_rate_limit = sum(1 for row in batch_results if row.get("parse_error") == "http_429")
        batch_failed = sum(1 for row in batch_results if row.get("parse_status") != "ok")
        if current_concurrency > 1 and (batch_empty / batch_total > 0.01 or batch_rate_limit / batch_total > 0.05 or batch_failed / batch_total > 0.10):
            current_concurrency = max(1, current_concurrency // 2)
            batch_size = args.run_batch_size if args.run_batch_size > 0 else max(current_concurrency * 20, current_concurrency)
            print(f"[RUN] auto-backoff: concurrency={current_concurrency} sleep={args.backoff_sleep_s}s")
            time.sleep(args.backoff_sleep_s)

        if processed % args.progress_every == 0 or processed >= len(pending):
            print(
                f"[RUN] progress={processed}/{len(pending)} ok={ok_count} fail={fail_count} "
                f"empty={empty_count} rate_limit={rate_limit_count} api_other={api_error_count}"
            )
        if args.delay_s > 0:
            time.sleep(args.delay_s)

    _write_json(
        output_dir / "ds_outputs" / "run_report.json",
        {
            "stage": "run_c_action_direction",
            "created_at": _now_iso(),
            "model": args.model,
            "processed_this_session": processed,
            "ok_this_session": ok_count,
            "failed_this_session": fail_count,
            "empty_response_this_session": empty_count,
            "rate_limit_this_session": rate_limit_count,
            "api_other_failed_this_session": api_error_count,
            "initial_concurrency": int(args.concurrency),
            "final_concurrency": current_concurrency,
            "max_retries": args.max_retries,
            "response_format_enabled": not args.no_response_format,
            "thinking": args.thinking,
            "save_full_body_on_fail": args.save_full_body_on_fail,
        },
    )
    print(f"[RUN] done processed={processed} ok={ok_count} fail={fail_count}")
def stage_parse(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    raw_path = output_dir / "ds_outputs" / "raw_responses.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    latest_by_sample: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(raw_path):
        latest_by_sample[str(record["sample_id"])] = record

    yes_no_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    failure_types: Counter[str] = Counter()

    for sample_id, record in sorted(latest_by_sample.items(), key=lambda item: int(item[1].get("request_index", 0))):
        raw = str(record.get("raw_response", ""))
        labels, status = _parse_yes_no(raw)
        if record.get("parse_status") != "ok" or labels is None or status != "ok":
            err = str(record.get("parse_error") or status)
            failure_rows.append(
                {
                    "sample_id": sample_id,
                    "request_index": record.get("request_index"),
                    "parse_status": "failed",
                    "parse_error": err,
                    "raw_response": raw,
                }
            )
            failure_types[err] += 1
            continue
        semantic_9 = [1 if labels[label] == "yes" else 0 for label in DS_LABELS] + [0]
        yes_no_rows.append({"sample_id": sample_id, "request_index": record.get("request_index"), **labels})
        semantic_rows.append(
            {
                "sample_id": sample_id,
                "request_index": record.get("request_index"),
                "semantic_yes_no": labels,
                "semantic_9": semantic_9,
                "semantic_order": SEMANTIC_ORDER,
                "stop_hold_policy": "fixed_zero",
            }
        )

    ds_outputs = output_dir / "ds_outputs"
    _write_jsonl(ds_outputs / "parsed_yes_no.jsonl", yes_no_rows)
    _write_jsonl(ds_outputs / "parsed_semantic_9.jsonl", semantic_rows)
    _write_jsonl(ds_outputs / "parse_failures.jsonl", failure_rows)
    _write_json(
        ds_outputs / "parse_report.json",
        {
            "stage": "parse_c_action_direction",
            "created_at": _now_iso(),
            "total_latest_responses": len(latest_by_sample),
            "parsed_ok": len(yes_no_rows),
            "parse_failures": len(failure_rows),
            "failure_type_counts": dict(sorted(failure_types.items())),
            "failure_rate": round(len(failure_rows) / max(len(latest_by_sample), 1), 6),
        },
    )
    print(f"[PARSE] ok={len(yes_no_rows)} failures={len(failure_rows)} types={dict(failure_types)}")
def _label_counts_from_yes_no(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for label in DS_LABELS:
            if row.get(label) == "yes":
                counts[label] += 1
    return counts
def _combo_key(row: dict[str, Any]) -> str:
    active = [label for label in DS_LABELS if row.get(label) == "yes"]
    return "|".join(active) if active else "<EMPTY>"
def stage_audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    requests_path = output_dir / "prompts" / "ds_requests.jsonl"
    windows_path = output_dir / "computed_traces" / "raw_observation_windows.jsonl"
    raw_path = output_dir / "ds_outputs" / "raw_responses.jsonl"
    yes_no_path = output_dir / "ds_outputs" / "parsed_yes_no.jsonl"
    semantic_path = output_dir / "ds_outputs" / "parsed_semantic_9.jsonl"
    failure_path = output_dir / "ds_outputs" / "parse_failures.jsonl"
    for path in (requests_path, windows_path, raw_path, yes_no_path, semantic_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    requests_rows = _load_jsonl(requests_path)
    windows = {row["sample_id"]: row for row in _load_jsonl(windows_path)}
    raw_rows = _load_jsonl(raw_path)
    yes_no_rows = _load_jsonl(yes_no_path)
    semantic_rows = _load_jsonl(semantic_path)
    failure_rows = _load_jsonl(failure_path) if failure_path.is_file() else []
    semantic_lookup = {row["sample_id"]: row["semantic_9"] for row in semantic_rows}

    label_counts = _label_counts_from_yes_no(yes_no_rows)
    combo_counts: Counter[str] = Counter(_combo_key(row) for row in yes_no_rows)
    source_counts: Counter[str] = Counter()
    selected_by_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    all_one_count = 0
    lr_conflict = 0
    ud_conflict = 0

    for row in yes_no_rows:
        window = windows.get(row["sample_id"], {})
        source_counts[str(window.get("source_dataset", "<missing>"))] += 1
        for tag in window.get("selected_by", []):
            selected_by_counts[str(tag)] += 1
        old_split = str(window.get("old_split", ""))
        if old_split:
            split_counts[old_split] += 1
        active = {label for label in DS_LABELS if row.get(label) == "yes"}
        if active == DS_LABEL_SET:
            all_one_count += 1
        if "AVOID_LEFT" in active and "AVOID_RIGHT" in active:
            lr_conflict += 1
        if "AVOID_UP" in active and "AVOID_DOWN" in active:
            ud_conflict += 1

    leakage_hits: list[dict[str, Any]] = []
    for request in requests_rows:
        user_prompt = str(request.get("user_prompt", ""))
        hits = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in user_prompt]
        if hits:
            leakage_hits.append({"sample_id": request["sample_id"], "forbidden_markers_hit": hits})

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    alignment_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    request_lookup = {row["sample_id"]: row for row in requests_rows}
    for row in yes_no_rows:
        sample_id = row["sample_id"]
        window = windows.get(sample_id, {})
        semantic_9 = semantic_lookup.get(sample_id)
        alignment_rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": window.get("source_dataset"),
                "episode_id": window.get("episode_id"),
                "anchor_step": window.get("anchor_step"),
                "semantic_9": semantic_9,
                "selected_by": window.get("selected_by", []),
            }
        )
        teacher_rows.append(
            {
                "sample_id": sample_id,
                "messages": [
                    {"role": "system", "content": ACTION_DIRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": request_lookup.get(sample_id, {}).get("user_prompt", "")},
                    {"role": "assistant", "content": json.dumps({label: row[label] for label in DS_LABELS}, ensure_ascii=False, sort_keys=True)},
                ],
                "target": {
                    "semantic_yes_no": {label: row[label] for label in DS_LABELS},
                    "semantic_9": semantic_9,
                    "semantic_order": SEMANTIC_ORDER,
                    "stop_hold_policy": "fixed_zero",
                },
                "metadata": {
                    "source_dataset": window.get("source_dataset"),
                    "episode_id": window.get("episode_id"),
                    "anchor_step": window.get("anchor_step"),
                    "selected_by": window.get("selected_by", []),
                    "prompt_version": "deepseek_action_direction_v1",
                },
            }
        )

    table = pa.Table.from_pylist(alignment_rows)
    pq.write_table(table, labels_dir / "semantic_alignment_deepseek.parquet", compression="zstd")
    _write_jsonl(labels_dir / "deepseek_teacher_semantic.jsonl", teacher_rows)

    total = len(yes_no_rows) + len(failure_rows)
    audit = {
        "audit_version": "deepseek_action_direction_audit_v1",
        "created_at": _now_iso(),
        "total_samples": total,
        "requests": len(requests_rows),
        "raw_responses": len(raw_rows),
        "parsed_ok": len(yes_no_rows),
        "parse_failures": len(failure_rows),
        "parse_failure_rate": round(len(failure_rows) / max(total, 1), 6),
        "label_distribution": dict(sorted(label_counts.items())),
        "combo_distribution": dict(sorted(combo_counts.items())),
        "source_distribution": dict(sorted(source_counts.items())),
        "selected_by_distribution": dict(sorted(selected_by_counts.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "empty_label_rate": round(combo_counts.get("<EMPTY>", 0) / max(len(yes_no_rows), 1), 6),
        "all_one_label_rate": round(all_one_count / max(len(yes_no_rows), 1), 6),
        "left_right_simultaneous_count": lr_conflict,
        "left_right_simultaneous_rate": round(lr_conflict / max(len(yes_no_rows), 1), 6),
        "up_down_simultaneous_count": ud_conflict,
        "up_down_simultaneous_rate": round(ud_conflict / max(len(yes_no_rows), 1), 6),
        "forbidden_input_leakage_hits": len(leakage_hits),
        "leakage_examples": leakage_hits[:20],

        "ds_sees_raw_arrays": False,
        "ds_sees_compact_fossen_only": True,
        "model": raw_rows[-1].get("model", "") if raw_rows else "",
    }
    audit_dir = output_dir / "audit"
    _write_json(audit_dir / "ds_qc_report.json", audit)
    _write_json(audit_dir / "label_distribution.json", audit)
    _write_json(audit_dir / "combo_distribution.json", audit)
    _write_json(audit_dir / "leakage_check_report.json", {"leakage_hits": leakage_hits, "forbidden_markers": FORBIDDEN_PROMPT_MARKERS})
    _write_json(
        audit_dir / "sample_review_manifest.json",
        {
            "review_note": "Manual review checklist: request prompt, raw_response, parsed yes/no, semantic_9, local full Fossen trace.",
            "random_sample_ids": [row["sample_id"] for row in yes_no_rows[:10]],
        },
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "dataset_id": output_dir.name,
            "created_at": _now_iso(),
            "semantic_order": SEMANTIC_ORDER,
            "label_policy": "deepseek_teacher_yes_no_c_action_direction_prompt",
            "stop_hold_policy": "fixed_zero",

            "source_datasets": SOURCE_DATASETS,
            "model": audit["model"],
            "totals": {
                "requests": len(requests_rows),
                "raw_responses": len(raw_rows),
                "parsed_ok": len(yes_no_rows),
                "parse_failures": len(failure_rows),
            },
            "outputs": {
                "raw_observation_windows": str(output_dir / "computed_traces" / "raw_observation_windows.jsonl"),
                "fossen_lite_traces": str(output_dir / "computed_traces" / "fossen_lite_traces.jsonl"),
                "observation_summaries": str(output_dir / "computed_traces" / "observation_summaries.jsonl"),
                "prompts": str(requests_path),
                "raw_responses": str(raw_path),
                "parsed_yes_no": str(yes_no_path),
                "parsed_semantic_9": str(semantic_path),
                "parse_failures": str(failure_path),
    "labels_parquet": str(labels_dir / "semantic_alignment_deepseek.parquet"),
    "teacher_jsonl": str(labels_dir / "deepseek_teacher_semantic.jsonl"),
                "audit_dir": str(audit_dir),
            },
        },
    )
    print(f"[AUDIT] parsed={len(yes_no_rows)} failures={len(failure_rows)} leakage_hits={len(leakage_hits)}")
    print(f"[AUDIT] label_distribution={json.dumps(dict(sorted(label_counts.items())), ensure_ascii=False)}")
def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek teacher action-direction compact-prompt pipeline.")
    parser.add_argument("--output-dir", type=str, required=True, help="B version output directory; build stage requires it not to exist")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_build = sub.add_parser("build", help="construct local raw windows + full Fossen traces + semantic summaries")
    p_build.add_argument("--sources", type=str, nargs="+", default=["v1", "v2", "v3"])
    p_build.add_argument("--limit-per-source", type=int, default=0, help="0 = all anchors")
    p_build.add_argument("--seed", type=int, default=20260805)

    p_prompt = sub.add_parser("prompts", help="assemble compact B prompt requests")
    p_prompt.add_argument("--limit", type=int, default=0, help="0 = all built windows")
    p_prompt.add_argument("--seed", type=int, default=20260805)
    p_prompt.add_argument("--sources", type=str, nargs="+", default=["v1", "v2", "v3"])

    p_run = sub.add_parser("run", help="call DeepSeek API with strict validation and resumable done set")
    p_run.add_argument("--api-key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))
    p_run.add_argument("--base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p_run.add_argument("--model", type=str, default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    p_run.add_argument("--max-samples", type=int, default=0, help="0 = all pending requests")
    p_run.add_argument("--concurrency", type=int, default=1, help="use 1/4/8/16/32 ramp; script auto-backs off on failures")
    p_run.add_argument("--run-batch-size", type=int, default=0, help="0 = concurrency * 20")
    p_run.add_argument("--delay-s", type=float, default=0.0, help="sleep after each written batch")
    p_run.add_argument("--backoff-sleep-s", type=float, default=60.0)
    p_run.add_argument("--timeout-s", type=float, default=120.0)
    p_run.add_argument("--max-retries", type=int, default=3)
    p_run.add_argument("--progress-every", type=int, default=50)
    p_run.add_argument("--no-response-format", action="store_true", help="disable OpenAI-compatible JSON response_format")
    p_run.add_argument(
        "--thinking",
        type=str,
        choices=["disabled", "enabled", "omit"],
        default="disabled",
        help="DeepSeek thinking control; disabled is recommended for strict JSON labeling",
    )
    p_run.add_argument(
        "--save-full-body-on-fail",
        action="store_true",
        help="store full API response body only for failed rows, useful when content is empty",
    )

    sub.add_parser("parse", help="parse raw responses into yes/no and semantic_9")
    sub.add_parser("audit", help="write labels, parquet alignment, audit reports, and manifest")
    sub.add_parser("sft", help="build compact-JSON SFT supervision rows")
    return parser
def main() -> None:
    args = _make_parser().parse_args()
    if args.stage == "build":
        stage_build(args)
    elif args.stage == "prompts":
        stage_prompts(args)
    elif args.stage == "run":
        stage_run(args)
    elif args.stage == "parse":
        stage_parse(args)
    elif args.stage == "audit":
        stage_audit(args)
    elif args.stage == "sft":
        sft_main()
    else:
        raise ValueError(f"unknown stage: {args.stage}")


"""Compact-JSON SFT data builder for the student LLM.

Reads the audited teacher labels together with the observation windows and
Fossen-lite traces, filters invalid label combinations, and emits the eight-field
yes/no JSON supervision rows used for LoRA adaptation of the student model."""
ROOT = Path(os.environ.get("CSPI_WORKSPACE", str(Path(__file__).resolve().parents[2])))
LABEL_ROOT = Path(
    os.environ.get(
        "CSPI_LABEL_ROOT",
        str(ROOT / "datasets" / "teacher_labels"),
    )
)
LABEL_JSONL = LABEL_ROOT / "labels" / "deepseek_teacher_semantic.jsonl"
WINDOW_JSONL = LABEL_ROOT / "computed_traces" / "raw_observation_windows.jsonl"
FOSSEN_JSONL = LABEL_ROOT / "computed_traces" / "fossen_lite_traces.jsonl"
ARCHIVE_ROOT = Path(
    os.environ.get(
        "CSPI_SFT_ROOT",
        str(ROOT / "datasets" / "sft_compact_json"),
    )
)
ARCHIVE_JSONL = ARCHIVE_ROOT / "llm_sft_compact_json.jsonl"
ARCHIVE_REPORT = ARCHIVE_ROOT / "compact_json_report.json"
ARCHIVE_PREVIEW = ARCHIVE_ROOT / "sample_preview.json"
LLAMAFACTORY_DATA = Path(os.environ.get("CSPI_LLAMAFACTORY_DATA", str(ARCHIVE_ROOT / "llamafactory_data")))
LLAMAFACTORY_JSONL = LLAMAFACTORY_DATA / "compact_json.jsonl"
OUTPUT_KEYS = SEMANTIC_ORDER[:8]
SYSTEM_PROMPT = (
    "You are an AUV onboard semantic module. Given a 4-frame observation window, "
    "output one 8-field JSON object. Each value must be \"yes\" or \"no\". Do not explain."
)
OUTPUT_FORMAT = (
    '{"CRUISE":"yes/no","ALIGN_GOAL":"yes/no","AVOID_LEFT":"yes/no",'
    '"AVOID_RIGHT":"yes/no","AVOID_UP":"yes/no","AVOID_DOWN":"yes/no",'
    '"BRAKE":"yes/no","REVERSE":"yes/no"}'
)
LABEL_MEANING = (
    "CRUISE=nominal forward motion; ALIGN_GOAL=align toward the goal; "
    "AVOID_LEFT=move left; AVOID_RIGHT=move right; AVOID_UP=move up; "
    "AVOID_DOWN=move down; BRAKE=reduce forward motion; REVERSE=move backward."
)
def sft_load_jsonl_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:

    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"missing sample_id in {path}")
            rows[sample_id] = row
    return rows
def sft_label_from_sign(value: float, positive_label: str, negative_label: str, neutral: str, weak: float, strong: float) -> str:

    abs_value = abs(float(value))
    if abs_value < weak:
        return neutral
    strength = "strong" if abs_value >= strong else "medium"
    direction = positive_label if value > 0 else negative_label
    return f"{strength}_{direction}"
def sft_goal_bin(goal_dir: list[float]) -> str:

    x, y, z = (list(goal_dir) + [0.0, 0.0, 0.0])[:3]
    if x >= 0.75:
        forward = "forward"
    elif x >= 0.35:
        forward = "diagonal"
    else:
        forward = "side"

    if y > 0.15:
        lateral = "left"
    elif y < -0.15:
        lateral = "right"
    else:
        lateral = "center"

    if z > 0.12:
        vertical = "up"
    elif z < -0.12:
        vertical = "down"
    else:
        vertical = "level"
    return f"{forward}-{lateral}-{vertical}"
def sft_progress_bin(progress_m: float) -> str:

    value = float(progress_m)
    if value >= 0.12:
        return "good"
    if value >= 0.03:
        return "weak_positive"
    if value > -0.02:
        return "flat"
    return "negative"
def sft_clearance_bin(value_m: float, axis: str) -> str:

    value = float(value_m)
    if axis == "front":
        if value >= 35.0:
            return "very_high"
        if value >= 20.0:
            return "high"
        if value >= 12.0:
            return "safe"
        if value >= 6.0:
            return "low"
        return "critical"
    if value >= 35.0:
        return "very_high"
    if value >= 16.0:
        return "high"
    if value >= 8.0:
        return "safe"
    if value >= 4.0:
        return "low"
    return "critical"
def sft_motion_bins(frame: dict[str, Any]) -> dict[str, str]:

    velocity = list(frame.get("body_velocity_m_s", [0.0, 0.0, 0.0]))
    action = list(frame.get("previous_executed_action", [0.0, 0.0, 0.0]))
    velocity = (velocity + [0.0, 0.0, 0.0])[:3]
    action = (action + [0.0, 0.0, 0.0])[:3]
    return {
        "forward_motion": sft_label_from_sign(velocity[0], "forward", "reverse", "near_zero", 0.08, 0.45),
        "lateral_motion": sft_label_from_sign(velocity[1], "right_drift", "left_drift", "near_zero", 0.06, 0.35),
        "vertical_motion": sft_label_from_sign(velocity[2], "up_drift", "down_drift", "near_zero", 0.06, 0.35),
        "forward_action": sft_label_from_sign(action[0], "forward", "reverse", "near_zero", 0.15, 0.65),
        "lateral_action": sft_label_from_sign(action[1], "left", "right", "near_zero", 0.15, 0.65),
        "vertical_action": sft_label_from_sign(action[2], "up", "down", "near_zero", 0.15, 0.65),
    }
def sft_format_frame(frame: dict[str, Any]) -> str:

    clear = frame.get("clearance_m", {}) if isinstance(frame.get("clearance_m"), dict) else {}
    motion = sft_motion_bins(frame)
    return (
        f"t{float(frame.get('time_offset_s', 0.0)):+.1f}s: "
        f"goal={sft_goal_bin(frame.get('goal_dir_body', [0.0, 0.0, 0.0]))}; "
        f"progress={sft_progress_bin(float(frame.get('progress_to_goal_m', 0.0)))}; "
        f"front_clearance={sft_clearance_bin(clear.get('front', 0.0), 'front')}; "
        f"left_clearance={sft_clearance_bin(clear.get('left', 0.0), 'side')}; "
        f"right_clearance={sft_clearance_bin(clear.get('right', 0.0), 'side')}; "
        f"up_clearance={sft_clearance_bin(clear.get('up', 0.0), 'vertical')}; "
        f"down_clearance={sft_clearance_bin(clear.get('down', 0.0), 'vertical')}; "
        f"forward_motion={motion['forward_motion']}; "
        f"lateral_motion={motion['lateral_motion']}; "
        f"vertical_motion={motion['vertical_motion']}; "
        f"forward_action={motion['forward_action']}; "
        f"lateral_action={motion['lateral_action']}; "
        f"vertical_action={motion['vertical_action']}."
    )
def sft_goal_distance_trend(history: list[dict[str, Any]]) -> str:

    distances = [float(row.get("goal_distance_m", 0.0)) for row in history]
    if len(distances) < 2:
        return "unknown"
    delta = distances[-1] - distances[0]
    if delta <= -0.5:
        return "decreasing"
    if delta >= 0.5:
        return "increasing"
    return "nearly_flat"
def sft_format_fossen(fossen_row: dict[str, Any]) -> str:

    compact = fossen_row.get("compact_fossen_for_prompt", {})
    lat = compact.get("lateral_drift_proxy", {})
    vert = compact.get("vertical_drift_proxy", {})
    drag = compact.get("drag_proxy_norm", {})
    loss = compact.get("progress_loss_proxy", {})
    margin = compact.get("clearance_margin", {})
    safety = (
        f"front_{margin.get('front', 'unknown')},"
        f"left_{margin.get('left', 'unknown')},"
        f"right_{margin.get('right', 'unknown')},"
        f"up_{margin.get('up', 'unknown')},"
        f"down_{margin.get('down', 'unknown')}"
    )
    return (
        f"Fossen-lite: lateral_drift={lat.get('bin', 'unknown')}; "
        f"vertical_drift={vert.get('bin', 'unknown')}; "
        f"drag_disturbance={drag.get('bin', 'unknown')}; "
        f"progress_loss={loss.get('bin', 'unknown')}; "
        f"safety_margin={safety}."
    )
def sft_build_user_prompt(window_row: dict[str, Any], fossen_row: dict[str, Any]) -> str:

    history = list(window_row.get("raw_history", []))
    if len(history) != 4:
        raise ValueError(f"sample {window_row.get('sample_id')} raw_history length != 4")
    frame_lines = "\n".join(sft_format_frame(frame) for frame in history)
    return "\n\n".join(
        [
            f"Output format: {OUTPUT_FORMAT}",
            "Observation window: 2 Hz, 4 frames, covering the latest 1.5 s.",
            f"Label meaning: {LABEL_MEANING}",
            "Note: AVOID directions are recommended maneuver directions, not disturbance source directions.",
            f"Window trend: goal_distance={sft_goal_distance_trend(history)}.",
            frame_lines,
            sft_format_fossen(fossen_row),
        ]
    )
def sft_assistant_json_from_semantic(semantic_9: list[int]) -> str:

    payload = {label: ("yes" if int(semantic_9[index]) == 1 else "no") for index, label in enumerate(OUTPUT_KEYS)}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
def sft_invalid_reason(semantic_9: list[int]) -> str:

    if len(semantic_9) < 9:
        return "bad_semantic_length"
    active_first8 = [int(value) for value in semantic_9[:8]]
    if not any(active_first8):
        return "empty_label"
    if int(semantic_9[2]) == 1 and int(semantic_9[3]) == 1:
        return "left_right_conflict"
    if int(semantic_9[4]) == 1 and int(semantic_9[5]) == 1:
        return "up_down_conflict"
    return ""
def sft_build_sft_row(label_row: dict[str, Any], window_row: dict[str, Any], fossen_row: dict[str, Any]) -> dict[str, Any]:

    target = label_row.get("target", {})
    semantic_9 = [int(value) for value in target.get("semantic_9", [])]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sft_build_user_prompt(window_row, fossen_row)},
            {"role": "assistant", "content": sft_assistant_json_from_semantic(semantic_9)},
        ],
        "metadata": {
            "sample_id": label_row.get("sample_id"),
            "source_dataset": label_row.get("metadata", {}).get("source_dataset"),
            "episode_id": label_row.get("metadata", {}).get("episode_id"),
            "anchor_step": label_row.get("metadata", {}).get("anchor_step"),
            "teacher": "deepseek-chat",
            "teacher_version": "action_direction",
            "sft_version": "compact_json_en",
            "target_semantic_9": semantic_9,
            "semantic_order": SEMANTIC_ORDER,
        },
        "target": {
            "semantic_9": semantic_9,
            "semantic_order": SEMANTIC_ORDER,
            "stop_hold_policy": "fixed_zero",
        },
    }
def sft_main() -> None:

    for path in (LABEL_JSONL, WINDOW_JSONL, FOSSEN_JSONL):
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = sft_load_jsonl_by_sample_id(LABEL_JSONL)
    windows = sft_load_jsonl_by_sample_id(WINDOW_JSONL)
    fossen = sft_load_jsonl_by_sample_id(FOSSEN_JSONL)

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    LLAMAFACTORY_DATA.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = Counter()
    label_distribution = Counter()
    combo_distribution = Counter()
    source_distribution = Counter()
    previews: list[dict[str, Any]] = []

    with ARCHIVE_JSONL.open("w", encoding="utf-8", newline="\n") as out:
        for sample_id in sorted(labels):
            label_row = labels[sample_id]
            semantic_9 = [int(value) for value in label_row.get("target", {}).get("semantic_9", [])]
            reason = sft_invalid_reason(semantic_9)
            if reason:
                skipped[reason] += 1
                continue
            if sample_id not in windows or sample_id not in fossen:
                skipped["missing_window_or_fossen"] += 1
                continue

            row = sft_build_sft_row(label_row, windows[sample_id], fossen[sample_id])
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1

            active = [SEMANTIC_ORDER[index] for index, value in enumerate(semantic_9) if int(value) == 1]
            for label in active:
                label_distribution[label] += 1
            combo_distribution["|".join(active) if active else "<EMPTY>"] += 1
            source_distribution[str(row["metadata"].get("source_dataset", "unknown"))] += 1
            if len(previews) < 3:
                previews.append(row)

    shutil.copyfile(ARCHIVE_JSONL, LLAMAFACTORY_JSONL)

    report = {
        "dataset_version": "compact_json_en",
        "source_labels": str(LABEL_JSONL),
        "archive_jsonl": str(ARCHIVE_JSONL),
        "llamafactory_jsonl": str(LLAMAFACTORY_JSONL),
        "raw_rows": len(labels),
        "kept_rows": kept,
        "skipped_rows": sum(skipped.values()),
        "skip_reasons": dict(sorted(skipped.items())),
        "label_distribution": dict(label_distribution),
        "top_25_combos": combo_distribution.most_common(25),
        "source_distribution": dict(source_distribution),
        "output_format": OUTPUT_FORMAT,
        "system_prompt": SYSTEM_PROMPT,
    }
    ARCHIVE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ARCHIVE_PREVIEW.write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SFT] raw_rows={len(labels)} kept={kept} skipped={sum(skipped.values())} copied={LLAMAFACTORY_JSONL}")
    print(f"[SFT] skip_reasons={dict(sorted(skipped.items()))}")
    print(f"[SFT] report={ARCHIVE_REPORT}")


if __name__ == "__main__":
    main()