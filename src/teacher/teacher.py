"""Physics-guided teacher prompts, constraint rules, and the student-LLM interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import numpy as np
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


"""Prompt protocol and label constants for the compact-JSON semantic interface.

The student LLM emits one JSON object with eight yes/no keys per 4-frame
observation window.  These constants define the fixed key order shared by the
distillation dataset, the on-line prompt, and the 9D multi-hot semantic vector
consumed by the controller."""
SEMANTIC_LABELS: tuple[str, ...] = (
    "CRUISE",
    "ALIGN_GOAL",
    "AVOID_LEFT",
    "AVOID_RIGHT",
    "AVOID_UP",
    "AVOID_DOWN",
    "BRAKE",
    "REVERSE",
    "STOP_HOLD",
)
OUTPUT_KEYS: tuple[str, ...] = SEMANTIC_LABELS[:8]
SEMANTIC_INDEX: dict[str, int] = {label: index for index, label in enumerate(SEMANTIC_LABELS)}
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


"""Deployable physics-rule semantic fallback and safety bins.

The rule layer provides a deterministic warm-start intent before the student
LLM is enabled and a hold-last fallback whenever an LLM response is invalid.
It only reads deployable observations: the 28D inspection state, the ray/depth
clearance window, and the previous executed action.  No route identity,
current velocity, waypoint, or privilege information is used here."""
inspection_DVL_VELOCITY_SCALES_M_S = np.asarray((2.0, 1.5, 1.0), dtype=np.float32)
def _sector_quantile(frame: np.ndarray, rows: slice, cols: slice) -> float:

    sector = np.asarray(frame[rows, cols], dtype=np.float32).reshape(-1)
    finite = sector[np.isfinite(sector)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.quantile(finite, 0.05), 0.0, 1.0))
def depth_sector_stats(ray_depth: np.ndarray) -> dict[str, float]:

    frame = np.asarray(ray_depth, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[-1]
    if frame.shape != (45, 80):
        raise ValueError(f"ray_depth latest frame must be (45,80), got {frame.shape}")
    h, w = frame.shape
    return {
        "front": _sector_quantile(frame, slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3)),
        "left": _sector_quantile(frame, slice(None), slice(0, w // 3)),
        "right": _sector_quantile(frame, slice(None), slice(2 * w // 3, w)),
        "up": _sector_quantile(frame, slice(0, h // 3), slice(None)),
        "down": _sector_quantile(frame, slice(2 * h // 3, h), slice(None)),
    }
def restore_inspection_physical_velocity(state_28: Sequence[float]) -> tuple[float, float, float]:

    state = np.asarray(state_28, dtype=np.float32).reshape(-1)
    if state.size < 3:
        raise ValueError(f"semantic physical velocity restore needs at least 3 values, got {state.size}")
    semantic_velocity = state[:3].astype(np.float32, copy=False) * inspection_DVL_VELOCITY_SCALES_M_S
    physical_velocity = np.asarray(
        (-semantic_velocity[0], -semantic_velocity[1], semantic_velocity[2]),
        dtype=np.float32,
    )
    return tuple(float(value) for value in physical_velocity)
def goal_bin(goal_dir: Sequence[float]) -> str:

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
def label_from_sign(value: float, positive_label: str, negative_label: str, neutral: str, weak: float, strong: float) -> str:

    abs_value = abs(float(value))
    if abs_value < weak:
        return neutral
    strength = "strong" if abs_value >= strong else "medium"
    direction = positive_label if value > 0 else negative_label
    return f"{strength}_{direction}"
def progress_bin(delta_distance: float) -> str:

    value = float(delta_distance)
    if value >= 0.010:
        return "good"
    if value >= 0.002:
        return "weak_positive"
    if value > -0.002:
        return "flat"
    return "negative"
def clearance_bin(value: float, axis: str) -> str:

    value = float(np.clip(value, 0.0, 1.0))
    if value >= 0.65:
        return "very_high"
    if value >= 0.45:
        return "high"
    if value >= (0.26 if axis == "front" else 0.22):
        return "safe"
    if value >= 0.12:
        return "low"
    return "critical"
def margin_token(value: float, axis: str) -> str:

    safe = 0.26 if axis == "front" else 0.22
    margin = float(np.clip(value, 0.0, 1.0)) - safe
    if margin >= 0.40:
        return f"{axis}_positive_very_high"
    if margin >= 0.18:
        return f"{axis}_positive_high"
    if margin >= 0.0:
        return f"{axis}_positive_low"
    if margin >= -0.10:
        return f"{axis}_negative_low"
    return f"{axis}_negative_high"
def motion_bins(frame: FrameSnapshot) -> dict[str, str]:

    vx, vy, vz = frame.body_velocity
    action = frame.previous_action
    return {
        "forward_motion": label_from_sign(vx, "forward", "reverse", "near_zero", 0.08, 0.45),
        "lateral_motion": label_from_sign(vy, "right_drift", "left_drift", "near_zero", 0.06, 0.35),
        "vertical_motion": label_from_sign(vz, "up_drift", "down_drift", "near_zero", 0.06, 0.35),
        "forward_action": label_from_sign(action[0], "forward", "reverse", "near_zero", 0.15, 0.65),
        "lateral_action": label_from_sign(action[1], "left", "right", "near_zero", 0.15, 0.65),
        "vertical_action": label_from_sign(action[2], "up", "down", "near_zero", 0.15, 0.65),
    }
def _resolve_rule_conflicts(tokens: Sequence[str]) -> tuple[str, ...]:

    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = str(token).strip().upper()
        if token in OUTPUT_KEYS and token not in seen:
            ordered.append(token)
            seen.add(token)
    if "REVERSE" in seen and "CRUISE" in seen:
        ordered = [token for token in ordered if token != "CRUISE"]
        seen.discard("CRUISE")
    if "AVOID_LEFT" in seen and "AVOID_RIGHT" in seen:
        drop = "AVOID_RIGHT" if ordered.index("AVOID_LEFT") < ordered.index("AVOID_RIGHT") else "AVOID_LEFT"
        ordered = [token for token in ordered if token != drop]
        seen.discard(drop)
    if "AVOID_UP" in seen and "AVOID_DOWN" in seen:
        drop = "AVOID_DOWN" if ordered.index("AVOID_UP") < ordered.index("AVOID_DOWN") else "AVOID_UP"
        ordered = [token for token in ordered if token != drop]
        seen.discard(drop)
    return tuple(ordered)
def rule_semantic_from_observation(
    state_28: np.ndarray,
    ray_depth: np.ndarray,
    previous_action: np.ndarray | None,
) -> tuple[str, ...]:

    state = np.asarray(state_28, dtype=np.float32).reshape(-1)
    if state.size != 28:
        raise ValueError(f"semantic rule semantic needs 28D inspection state, got {state.size}D")
    action = state[23:28].astype(np.float32, copy=True) if previous_action is None else np.asarray(previous_action, dtype=np.float32).reshape(5)
    clear = depth_sector_stats(ray_depth)
    body_vx, body_vy, body_vz = (float(value) for value in state[0:3])
    goal_y, goal_z, goal_distance = float(state[20]), float(state[21]), float(state[22])
    tokens: list[str] = []

    if action[0] < -0.55:
        tokens.append("REVERSE")
    elif action[0] < -0.20 or (action[0] > 0.35 and body_vx < 0.02):
        tokens.append("BRAKE")
    elif action[0] > 0.10:
        tokens.append("CRUISE")

    if abs(goal_y) > 0.06 or abs(goal_z) > 0.06 or goal_distance > 0.10:
        tokens.append("ALIGN_GOAL")
    if body_vy > 0.12:
        tokens.append("AVOID_LEFT")
    elif body_vy < -0.12:
        tokens.append("AVOID_RIGHT")
    if body_vz > 0.12:
        tokens.append("AVOID_DOWN")
    elif body_vz < -0.12:
        tokens.append("AVOID_UP")

    if clear["front"] < 0.18:
        tokens.append("BRAKE")
        tokens.append("AVOID_LEFT" if clear["left"] >= clear["right"] else "AVOID_RIGHT")
    if clear["down"] < 0.16 and clear["up"] > clear["down"]:
        tokens.append("AVOID_UP")
    if clear["up"] < 0.16 and clear["down"] > clear["up"]:
        tokens.append("AVOID_DOWN")
    if clear["front"] >= 0.18 and "BRAKE" not in tokens and "REVERSE" not in tokens:
        tokens.insert(0, "CRUISE")
    if not tokens:
        tokens.extend(("CRUISE", "ALIGN_GOAL"))
    return _resolve_rule_conflicts(tokens)
def resolve_text_conflicts(tokens: Sequence[str]) -> tuple[str, ...]:

    ordered: list[str] = []
    seen: set[str] = set()
    for raw_token in tokens:
        token = str(raw_token).strip().upper()
        if token not in SEMANTIC_INDEX or token in seen:
            continue
        ordered.append(token)
        seen.add(token)

    if "STOP_HOLD" in seen:
        return ("STOP_HOLD",)
    if "REVERSE" in seen and "CRUISE" in seen:
        ordered = [token for token in ordered if token != "CRUISE"]
        seen.discard("CRUISE")
    if "AVOID_LEFT" in seen and "AVOID_RIGHT" in seen:
        first = "AVOID_LEFT" if ordered.index("AVOID_LEFT") < ordered.index("AVOID_RIGHT") else "AVOID_RIGHT"
        drop = "AVOID_RIGHT" if first == "AVOID_LEFT" else "AVOID_LEFT"
        ordered = [token for token in ordered if token != drop]
        seen.discard(drop)
    if "AVOID_UP" in seen and "AVOID_DOWN" in seen:
        first = "AVOID_UP" if ordered.index("AVOID_UP") < ordered.index("AVOID_DOWN") else "AVOID_DOWN"
        drop = "AVOID_DOWN" if first == "AVOID_UP" else "AVOID_UP"
        ordered = [token for token in ordered if token != drop]
        seen.discard(drop)
    return tuple(ordered)
def depth_sector_stats_base(ray_depth: np.ndarray) -> dict[str, float]:

    frame = np.asarray(ray_depth, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[-1]
    if frame.shape != (45, 80):
        raise ValueError(f"ray_depth latest frame should be (45,80), got {frame.shape}")
    h, w = frame.shape
    return {
        "front": float(np.nanmin(frame)),
        "left": float(np.nanmin(frame[:, : w // 3])),
        "right": float(np.nanmin(frame[:, 2 * w // 3 :])),
        "up": float(np.nanmin(frame[: h // 3, :])),
        "down": float(np.nanmin(frame[2 * h // 3 :, :])),
    }
def bin_clearance(value: float) -> str:

    if value < 0.10:
        return "critical"
    if value < 0.20:
        return "low"
    if value < 0.35:
        return "medium"
    return "open"
def bin_signed(value: float, deadband: float = 0.08) -> str:

    if value > deadband:
        return "positive"
    if value < -deadband:
        return "negative"
    return "neutral"
def bin_magnitude(value: float, low: float = 0.04, high: float = 0.16) -> str:

    magnitude = abs(float(value))
    if magnitude < low:
        return "low"
    if magnitude < high:
        return "medium"
    return "high"
def rule_semantic_from_observation_base(
    state_28: np.ndarray,
    ray_depth: np.ndarray,
    previous_action: np.ndarray | None,
) -> tuple[str, ...]:

    state = np.asarray(state_28, dtype=np.float32).reshape(-1)
    if state.size != 28:
        raise ValueError(f"rule semantic requires a 28D state, got {state.size}D.")
    if previous_action is None:
        action = state[23:28].astype(np.float32, copy=True)
    else:
        action = np.asarray(previous_action, dtype=np.float32).reshape(5)
    sectors = depth_sector_stats_base(ray_depth)
    tokens: list[str] = []

    body_vx = float(state[0])
    body_vy = float(state[1])
    body_vz = float(state[2])
    goal_y = float(state[20])
    goal_z = float(state[21])
    goal_distance = float(state[22])

    if action[0] < -0.55:
        tokens.append("REVERSE")
    elif action[0] < -0.20 or (action[0] > 0.35 and body_vx < 0.02):
        tokens.append("BRAKE")
    elif action[0] > 0.10:
        tokens.append("CRUISE")

    if abs(goal_y) > 0.06 or abs(goal_z) > 0.06 or goal_distance > 0.10:
        tokens.append("ALIGN_GOAL")

    if body_vy > 0.12:
        tokens.append("AVOID_LEFT")
    elif body_vy < -0.12:
        tokens.append("AVOID_RIGHT")
    if body_vz > 0.12:
        tokens.append("AVOID_DOWN")
    elif body_vz < -0.12:
        tokens.append("AVOID_UP")

    if sectors["front"] < 0.18:
        tokens.append("BRAKE")
        if sectors["left"] > sectors["right"] + 0.03:
            tokens.append("AVOID_LEFT")
        elif sectors["right"] > sectors["left"] + 0.03:
            tokens.append("AVOID_RIGHT")
    if sectors["right"] < 0.16 and sectors["left"] > sectors["right"]:
        tokens.append("AVOID_LEFT")
    if sectors["left"] < 0.16 and sectors["right"] > sectors["left"]:
        tokens.append("AVOID_RIGHT")
    if sectors["down"] < 0.16 and sectors["up"] > sectors["down"]:
        tokens.append("AVOID_UP")
    if sectors["up"] < 0.16 and sectors["down"] > sectors["up"]:
        tokens.append("AVOID_DOWN")

    if sectors["front"] >= 0.18 and "BRAKE" not in tokens and "REVERSE" not in tokens:
        tokens.insert(0, "CRUISE")

    if not tokens:
        tokens.extend(("CRUISE", "ALIGN_GOAL"))
    return resolve_text_conflicts(tokens)


"""Physics-guided semantic teacher: snapshot capture, prompt construction,
JSON/text parsing, and batched student-LLM inference.

The on-line module receives only deployable observations and returns the fixed
9D multi-hot semantic vector that conditions the dual-stream SAC controller.
Invalid responses are reported with ``valid=False`` so that callers keep the
last committed semantic state instead of injecting a fabricated intent."""
@dataclass(frozen=True)
class SemanticParseResult:

    tokens: tuple[str, ...]
    vector: np.ndarray
    valid: bool
    raw_text: str
    error: str = ""
@dataclass(frozen=True)
class FrameSnapshot:

    body_velocity: tuple[float, float, float]
    goal_direction: tuple[float, float, float]
    goal_distance: float
    previous_action: tuple[float, float, float, float, float]
    clearance: dict[str, float]
def semantic_tokens_to_vector(tokens: Iterable[str]) -> np.ndarray:

    vector = np.zeros(len(SEMANTIC_LABELS), dtype=np.float32)
    for token in tokens:
        index = SEMANTIC_INDEX.get(str(token).strip().upper())
        if index is not None:
            vector[index] = 1.0
    vector[SEMANTIC_INDEX["STOP_HOLD"]] = 0.0
    return vector
def semantic_vector_to_tokens(vector: Sequence[float]) -> tuple[str, ...]:

    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != len(SEMANTIC_LABELS):
        raise ValueError(f"semantic vector must be 9D, got {values.size}D")
    return tuple(label for label, value in zip(SEMANTIC_LABELS, values, strict=True) if float(value) >= 0.5)
def semantic_text(tokens: Sequence[str]) -> str:

    return "|".join(tokens) if tokens else "<EMPTY>"
def semantic_vector_to_matrix_text(vector: Sequence[float]) -> str:

    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != len(SEMANTIC_LABELS):
        raise ValueError(f"semantic vector must be 9D, got {values.size}D")
    return "[" + ",".join(str(int(float(value) >= 0.5)) for value in values.tolist()) + "]"
def parse_semantic_json_text(text: str) -> SemanticParseResult:

    raw_text = str(text or "").strip()
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match is None:
        return SemanticParseResult((), np.zeros(9, dtype=np.float32), False, raw_text, "missing_json")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return SemanticParseResult((), np.zeros(9, dtype=np.float32), False, raw_text, f"json_error:{exc.msg}")
    if not isinstance(payload, dict):
        return SemanticParseResult((), np.zeros(9, dtype=np.float32), False, raw_text, "json_not_object")
    keys = set(str(key) for key in payload.keys())
    expected = set(OUTPUT_KEYS)
    if keys != expected:
        return SemanticParseResult((), np.zeros(9, dtype=np.float32), False, raw_text, f"bad_keys:{sorted(keys)}")

    tokens: list[str] = []
    for label in OUTPUT_KEYS:
        value = str(payload.get(label, "")).strip().lower()
        if value not in {"yes", "no"}:
            return SemanticParseResult((), np.zeros(9, dtype=np.float32), False, raw_text, f"bad_value:{label}={value}")
        if value == "yes":
            tokens.append(label)
    if not tokens:
        return SemanticParseResult((), np.zeros(9, dtype=np.float32), False, raw_text, "empty_yes_set")
    vector = semantic_tokens_to_vector(tokens)
    return SemanticParseResult(tuple(tokens), vector, True, raw_text, "")
def capture_frame_snapshot(
    state_28: np.ndarray,
    ray_depth: np.ndarray,
    previous_action: np.ndarray | None,
) -> FrameSnapshot:

    state = np.asarray(state_28, dtype=np.float32).reshape(-1)
    if state.size != 28:
        raise ValueError(f"semantic frame capture needs 28D inspection state, got {state.size}D")
    if previous_action is None:
        action = state[23:28].astype(np.float32, copy=True)
    else:
        action = np.asarray(previous_action, dtype=np.float32).reshape(5)
    return FrameSnapshot(
        body_velocity=restore_inspection_physical_velocity(state),
        goal_direction=tuple(float(value) for value in state[19:22]),
        goal_distance=float(state[22]),
        previous_action=tuple(float(value) for value in action),
        clearance=depth_sector_stats(ray_depth),
    )
def _format_frame(time_label: str, frame: FrameSnapshot, progress_token: str) -> str:

    motion = motion_bins(frame)
    clear = frame.clearance
    return (
        f"{time_label}: "
        f"goal={goal_bin(frame.goal_direction)}; "
        f"progress={progress_token}; "
        f"front_clearance={clearance_bin(clear.get('front', 0.0), 'front')}; "
        f"left_clearance={clearance_bin(clear.get('left', 0.0), 'side')}; "
        f"right_clearance={clearance_bin(clear.get('right', 0.0), 'side')}; "
        f"up_clearance={clearance_bin(clear.get('up', 0.0), 'vertical')}; "
        f"down_clearance={clearance_bin(clear.get('down', 0.0), 'vertical')}; "
        f"forward_motion={motion['forward_motion']}; "
        f"lateral_motion={motion['lateral_motion']}; "
        f"vertical_motion={motion['vertical_motion']}; "
        f"forward_action={motion['forward_action']}; "
        f"lateral_action={motion['lateral_action']}; "
        f"vertical_action={motion['vertical_action']}."
    )
def _goal_distance_trend(history: Sequence[FrameSnapshot]) -> str:

    if len(history) < 2:
        return "unknown"
    delta = float(history[-1].goal_distance - history[0].goal_distance)
    if delta <= -0.020:
        return "decreasing"
    if delta >= 0.020:
        return "increasing"
    return "nearly_flat"
def _fossen_line(history: Sequence[FrameSnapshot]) -> str:

    latest = history[-1]
    previous = history[-2] if len(history) >= 2 else latest
    vx, vy, vz = latest.body_velocity
    action = latest.previous_action
    lateral_proxy = vy - 0.25 * action[1]
    vertical_proxy = vz - 0.25 * action[2]
    drag_norm = float(np.sqrt((vx * abs(vx)) ** 2 + (vy * abs(vy)) ** 2 + (vz * abs(vz)) ** 2))
    observed_progress = float(previous.goal_distance - latest.goal_distance)
    expected_progress = max(0.0, 0.02 * max(float(action[0]), 0.0))
    progress_loss = max(0.0, expected_progress - observed_progress)

    def fossen_signed(value: float, positive: str, negative: str, weak: float, strong: float) -> str:
        magnitude = abs(float(value))
        if magnitude < weak:
            return "neutral"
        strength = "strong" if magnitude >= strong else "medium"
        direction = positive if value > 0 else negative
        return f"{direction}_{strength}"

    lateral = fossen_signed(lateral_proxy, "right_drift", "left_drift", 0.06, 0.35)
    vertical = fossen_signed(vertical_proxy, "upward", "downward", 0.06, 0.35)
    drag = "none" if drag_norm < 0.03 else "weak" if drag_norm < 0.16 else "medium" if drag_norm < 0.60 else "strong"
    loss = "none" if progress_loss < 0.002 else "weak" if progress_loss < 0.010 else "moderate" if progress_loss < 0.030 else "high"
    safety = ",".join(
        [
            margin_token(latest.clearance.get("front", 0.0), "front"),
            margin_token(latest.clearance.get("left", 0.0), "left"),
            margin_token(latest.clearance.get("right", 0.0), "right"),
            margin_token(latest.clearance.get("up", 0.0), "up"),
            margin_token(latest.clearance.get("down", 0.0), "down"),
        ]
    )
    return (
        f"Fossen-lite: lateral_drift={lateral}; vertical_drift={vertical}; "
        f"drag_disturbance={drag}; progress_loss={loss}; safety_margin={safety}."
    )
def frame_debug_fields(history: Sequence[FrameSnapshot]) -> dict[str, object]:

    frames = list(history)[-4:]
    if not frames:
        raise ValueError("semantic debug history is empty")
    latest = frames[-1]
    motion = motion_bins(latest)
    action = latest.previous_action
    vx, vy, vz = latest.body_velocity
    lateral_proxy = float(vy - 0.25 * action[1])
    vertical_proxy = float(vz - 0.25 * action[2])

    def signed_proxy(value: float, positive: str, negative: str, weak: float, strong: float) -> str:
        magnitude = abs(float(value))
        if magnitude < weak:
            return "neutral"
        strength = "strong" if magnitude >= strong else "medium"
        return f"{positive if value > 0 else negative}_{strength}"

    return {
        "front_clearance": float(latest.clearance.get("front", 0.0)),
        "left_clearance": float(latest.clearance.get("left", 0.0)),
        "right_clearance": float(latest.clearance.get("right", 0.0)),
        "up_clearance": float(latest.clearance.get("up", 0.0)),
        "down_clearance": float(latest.clearance.get("down", 0.0)),
        "forward_motion": motion["forward_motion"],
        "lateral_motion": motion["lateral_motion"],
        "vertical_motion": motion["vertical_motion"],
        "forward_action": motion["forward_action"],
        "lateral_action": motion["lateral_action"],
        "vertical_action": motion["vertical_action"],
        "fossen_lateral": signed_proxy(lateral_proxy, "right_drift", "left_drift", 0.06, 0.35),
        "fossen_vertical": signed_proxy(vertical_proxy, "upward", "downward", 0.06, 0.35),
        "body_velocity_x_m_s": float(vx),
        "body_velocity_y_m_s": float(vy),
        "body_velocity_z_m_s": float(vz),
    }
def build_json_semantic_prompt(history: Sequence[FrameSnapshot]) -> str:

    frames = list(history)[-4:]
    if not frames:
        raise ValueError("semantic prompt history is empty")
    while len(frames) < 4:
        frames.insert(0, frames[0])
    time_labels = ["t-1.5s", "t-1.0s", "t-0.5s", "t+0.0s"]
    progress_tokens = ["flat"]
    for previous, current in zip(frames[:-1], frames[1:], strict=True):
        progress_tokens.append(progress_bin(float(previous.goal_distance - current.goal_distance)))
    frame_lines = "\n".join(
        _format_frame(label, frame, progress)
        for label, frame, progress in zip(time_labels, frames, progress_tokens, strict=True)
    )
    return "\n\n".join(
        [
            f"Output format: {OUTPUT_FORMAT}",
            "Observation window: 2 Hz, 4 frames, covering the latest 1.5 s.",
            f"Label meaning: {LABEL_MEANING}",
            "Note: AVOID directions are recommended maneuver directions, not disturbance source directions.",
            f"Window trend: goal_distance={_goal_distance_trend(frames)}.",
            frame_lines,
            _fossen_line(frames),
        ]
    )
def build_json_semantic_messages(history: Sequence[FrameSnapshot]) -> list[dict[str, str]]:

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_json_semantic_prompt(history)},
    ]
class QwenJsonSemanticBatcher:

    def __init__(
        self,
        base_model_path: str | Path,
        lora_adapter_path: str | Path,
        *,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        qwen_batch_size: int = 8,
        max_new_tokens: int = 96,
    ) -> None:
        self.base_model_path = str(Path(base_model_path))
        self.lora_adapter_path = str(Path(lora_adapter_path))
        self.device = str(device)
        self.torch_dtype = str(torch_dtype)
        self.qwen_batch_size = int(qwen_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:

        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if self.torch_dtype.lower() in {"bf16", "bfloat16"} else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
        model = PeftModel.from_pretrained(model, self.lora_adapter_path)
        generation_config = model.generation_config
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config = generation_config
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:

        assert self._tokenizer is not None
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def predict_texts(self, histories: Sequence[Sequence[FrameSnapshot]]) -> tuple[list[str], dict[str, object]]:

        self._load()
        assert self._tokenizer is not None and self._model is not None
        import torch

        outputs: list[str] = []
        latencies_ms: list[float] = []
        prompt_latencies_ms: list[float] = []
        messages = [build_json_semantic_messages(history) for history in histories]
        for start in range(0, len(messages), self.qwen_batch_size):
            chunk = messages[start : start + self.qwen_batch_size]
            encoded_texts = [self._apply_chat_template(message) for message in chunk]
            inputs = self._tokenizer(
                encoded_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self._model.device)
            started = time.perf_counter()
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            prompt_latencies_ms.extend([elapsed_ms / max(len(chunk), 1) for _ in chunk])
            prompt_len = inputs["input_ids"].shape[1]
            decoded = self._tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            outputs.extend([text.strip() for text in decoded])
        metrics = {
            "qwen_chunks": float(max(len(latencies_ms), 1)),
            "qwen_latency_ms_sum": float(np.sum(latencies_ms)) if latencies_ms else 0.0,
            "qwen_latency_ms_max": float(np.max(latencies_ms)) if latencies_ms else 0.0,
            "qwen_chunk_latency_ms": [float(value) for value in latencies_ms],
            "qwen_prompt_latency_ms": [float(value) for value in prompt_latencies_ms],
            "qwen_prompt_version": "commit_hold_compact_json_online",
        }
        return outputs, metrics
def semantic_tokens_to_vector_base(tokens: Iterable[str]) -> np.ndarray:

    vector = np.zeros(len(SEMANTIC_LABELS), dtype=np.float32)
    for token in tokens:
        index = SEMANTIC_INDEX.get(str(token).strip().upper())
        if index is not None:
            vector[index] = 1.0
    return vector
def semantic_vector_to_tokens_base(vector: Sequence[float]) -> tuple[str, ...]:

    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != len(SEMANTIC_LABELS):
        raise ValueError(f"semantic vector must be 9D, got {values.size}D.")
    return tuple(label for label, active in zip(SEMANTIC_LABELS, values, strict=True) if float(active) >= 0.5)
def semantic_vector_to_matrix_text_base(vector: Sequence[float]) -> str:

    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != len(SEMANTIC_LABELS):
        raise ValueError(f"semantic vector must be 9D, got {values.size}D.")
    return "[" + ",".join(str(int(float(item) >= 0.5)) for item in values.tolist()) + "]"
def parse_semantic_text(text: str) -> SemanticParseResult:

    raw_text = str(text or "")
    normalized = raw_text.upper().strip()
    pieces = re.split(r"[|\s,;, ; ]+", normalized)
    tokens = resolve_text_conflicts([piece for piece in pieces if piece])
    return SemanticParseResult(
        tokens=tokens,
        vector=semantic_tokens_to_vector_base(tokens),
        valid=bool(tokens),
        raw_text=raw_text,
    )
def build_text_semantic_prompt(
    state_28: np.ndarray,
    ray_depth: np.ndarray,
    previous_action: np.ndarray | None,
    semantic_age_s: float,
) -> str:

    state = np.asarray(state_28, dtype=np.float32).reshape(-1)
    if state.size != 28:
        raise ValueError(f"student prompt requires a 28D state, got {state.size}D.")
    if previous_action is None:
        action = state[23:28].astype(np.float32, copy=True)
    else:
        action = np.asarray(previous_action, dtype=np.float32).reshape(5)
    sectors = depth_sector_stats_base(ray_depth)
    body_vx = float(state[0])
    body_vy = float(state[1])
    body_vz = float(state[2])
    angular_x = float(state[15])
    angular_y = float(state[16])
    angular_z = float(state[17])
    depth_proxy = float(state[18])
    goal_direction_y = float(state[20])
    goal_direction_z = float(state[21])
    goal_distance = float(state[22])
    forward_effort = float(action[0])
    lateral_effort = float(action[1])
    vertical_effort = float(action[2])

    lateral_drift_proxy = bin_signed(body_vy - 0.25 * lateral_effort)
    vertical_drift_proxy = bin_signed(body_vz - 0.25 * vertical_effort)
    yaw_disturbance_proxy = bin_signed(angular_z, deadband=0.05)
    drag_proxy_y = bin_magnitude(body_vy * abs(body_vy))
    drag_proxy_z = bin_magnitude(body_vz * abs(body_vz))
    progress_loss_proxy = "moderate" if forward_effort > 0.35 and body_vx < 0.04 else "low"
    disturbance_needed = (
        lateral_drift_proxy != "neutral"
        or vertical_drift_proxy != "neutral"
        or yaw_disturbance_proxy != "neutral"
        or progress_loss_proxy != "low"
    )

    return (
        "System:\n"
        "You are an onboard semantic intent module for an AUV.\n"
        "Return only allowed intent labels joined by \"|\".\n"
        "Allowed labels:\n"
        "CRUISE, ALIGN_GOAL, AVOID_LEFT, AVOID_RIGHT, AVOID_UP, AVOID_DOWN, BRAKE, REVERSE, STOP_HOLD.\n\n"
        "Observation window:\n"
        f"goal_direction_y: {bin_signed(goal_direction_y)}\n"
        f"goal_direction_z: {bin_signed(goal_direction_z)}\n"
        f"goal_distance: {bin_magnitude(goal_distance, low=0.10, high=0.40)}\n"
        f"body_velocity_x: {bin_signed(body_vx)}\n"
        f"body_velocity_y: {bin_signed(body_vy)}\n"
        f"body_velocity_z: {bin_signed(body_vz)}\n"
        f"angular_rate_x: {bin_signed(angular_x, deadband=0.05)}\n"
        f"angular_rate_y: {bin_signed(angular_y, deadband=0.05)}\n"
        f"angular_rate_z: {yaw_disturbance_proxy}\n"
        f"depth_proxy: {bin_magnitude(depth_proxy, low=0.20, high=0.70)}\n"
        f"front_clearance: {bin_clearance(sectors['front'])}\n"
        f"left_clearance: {bin_clearance(sectors['left'])}\n"
        f"right_clearance: {bin_clearance(sectors['right'])}\n"
        f"up_clearance: {bin_clearance(sectors['up'])}\n"
        f"down_clearance: {bin_clearance(sectors['down'])}\n"
        f"previous_forward_action: {bin_signed(forward_effort)}\n"
        f"previous_lateral_action: {bin_signed(lateral_effort)}\n"
        f"previous_vertical_action: {bin_signed(vertical_effort)}\n"
        f"semantic_age_s: {float(semantic_age_s):.1f}\n\n"
        "Fossen-lite cues:\n"
        f"lateral_drift_proxy: {lateral_drift_proxy}\n"
        f"vertical_drift_proxy: {vertical_drift_proxy}\n"
        f"yaw_disturbance_proxy: {yaw_disturbance_proxy}\n"
        f"drag_proxy_y: {drag_proxy_y}\n"
        f"drag_proxy_z: {drag_proxy_z}\n"
        f"progress_loss_proxy: {progress_loss_proxy}\n"
        f"disturbance_compensation_needed: {str(disturbance_needed).lower()}.\n\n"
        "Output:\n"
    )
class QwenTextSemanticBatcher:

    def __init__(
        self,
        base_model_path: str | Path,
        lora_adapter_path: str | Path,
        *,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        qwen_batch_size: int = 8,
        max_new_tokens: int = 24,
    ) -> None:
        self.base_model_path = str(Path(base_model_path))
        self.lora_adapter_path = str(Path(lora_adapter_path))
        self.device = str(device)
        self.torch_dtype = str(torch_dtype)
        self.qwen_batch_size = int(qwen_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:

        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if self.torch_dtype.lower() in {"bf16", "bfloat16"} else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, self.lora_adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def predict_texts(self, prompts: Sequence[str]) -> tuple[list[str], dict[str, object]]:

        self._load()
        assert self._tokenizer is not None and self._model is not None
        import torch

        outputs: list[str] = []
        latencies_ms: list[float] = []
        prompt_latencies_ms: list[float] = []
        for start in range(0, len(prompts), self.qwen_batch_size):
            chunk = list(prompts[start : start + self.qwen_batch_size])
            messages = [[{"role": "user", "content": prompt}] for prompt in chunk]
            encoded_texts = [
                self._tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for message in messages
            ]
            inputs = self._tokenizer(
                encoded_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,
            ).to(self._model.device)
            started = time.perf_counter()
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            allocated_prompt_ms = elapsed_ms / max(len(chunk), 1)
            prompt_latencies_ms.extend([allocated_prompt_ms for _ in chunk])
            prompt_len = inputs["input_ids"].shape[1]
            decoded = self._tokenizer.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
            )
            outputs.extend([text.strip() for text in decoded])

        metrics = {
            "qwen_chunks": float(max(len(latencies_ms), 1)),
            "qwen_latency_ms_sum": float(np.sum(latencies_ms)) if latencies_ms else 0.0,
            "qwen_latency_ms_max": float(np.max(latencies_ms)) if latencies_ms else 0.0,
            "qwen_chunk_latency_ms": [float(value) for value in latencies_ms],
            "qwen_prompt_latency_ms": [float(value) for value in prompt_latencies_ms],
        }
        return outputs, metrics
