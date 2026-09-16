"""Route and current scenario definitions used by the inspection task."""

from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

ROUTES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

VALIDATION_CURRENT_SPEEDS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
CURRENT_CASES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.1, 90.0),
    (0.1, -90.0),
    (0.2, 90.0),
    (0.2, -90.0),
    (0.3, 90.0),
    (0.3, -90.0),
)
CASES_PER_CHECKPOINT = len(ROUTES) * len(CURRENT_CASES)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_SOURCE_MANIFEST = WORKSPACE_ROOT / "outputs" / "oracle_paths" / "manifest.json"
DEFAULT_ROUTE_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "custom_paths_r5r6"


CUSTOM_ROUTE_CONTROL_POINTS: dict[int, list[tuple[float, float, float]]] = {
    5: [
        (-80.0, 55.0, 56.34),
        (-50.0, 42.0, 54.0),
        (-20.0, 25.0, 50.0),
        (10.0, 5.0, 46.0),
        (35.0, -20.0, 46.0),
        (60.0, -38.0, 50.0),
        (95.0, -55.0, 56.34),
    ],
    6: [
        (-80.0, -55.0, 56.34),
        (-55.0, -42.0, 56.34),
        (-30.0, -20.0, 53.50),
        (-5.0, 10.0, 53.50),
        (-18.0, 18.0, 53.50),
        (25.0, 30.0, 58.50),
        (55.0, 43.0, 58.50),
        (95.0, 55.0, 56.34),
    ],
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _as_points(values: list[tuple[float, float, float]]) -> np.ndarray:

    points = np.asarray(values, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(f"control points must be [N,3] with at least two points, got shape={points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("control points contain NaN/Inf.")
    return points


def _path_length(points: np.ndarray) -> float:

    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:

    spacing = float(spacing_m)
    if spacing <= 0.0 or not math.isfinite(spacing):
        raise ValueError(f"spacing_m must be positive, got {spacing_m}")

    segment = points[1:] - points[:-1]
    segment_lengths = np.linalg.norm(segment, axis=1)
    keep = segment_lengths > 1.0e-6
    if not bool(np.all(keep)):
        points = np.concatenate((points[:1], points[1:][keep]), axis=0)
        segment = points[1:] - points[:-1]
        segment_lengths = np.linalg.norm(segment, axis=1)
    total = float(segment_lengths.sum())
    if total <= 1.0e-6:
        raise ValueError("total path length too short to build a strong-path.")

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.arange(0.0, total, spacing, dtype=np.float64)
    if targets.size == 0 or abs(float(targets[-1]) - total) > 1.0e-6:
        targets = np.concatenate((targets, np.asarray([total], dtype=np.float64)))

    sampled: list[np.ndarray] = []
    for distance in targets:
        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        index = min(max(index, 0), len(segment_lengths) - 1)
        local = float(distance - cumulative[index])
        ratio = local / max(float(segment_lengths[index]), 1.0e-6)
        sampled.append(points[index] + ratio * segment[index])
    return np.asarray(sampled, dtype=np.float32)


def _route_summary(route_id: int, control: np.ndarray, smoothed: np.ndarray, waypoints: np.ndarray, npz_path: Path) -> dict[str, Any]:

    euclidean = float(np.linalg.norm(control[-1] - control[0]))
    length = _path_length(smoothed)
    return {
        "route_id": int(route_id),
        "status": "provisional",
        "source": "manual_custom_r5r6",
        "control_points": [[float(v) for v in row] for row in control],
        "raw_grid_path_points": int(control.shape[0]),
        "waypoints": int(waypoints.shape[0]),
        "planned_length_m": length,
        "reconstructed_length_m": length,
        "euclidean_length_m": euclidean,
        "length_ratio": float(length / max(euclidean, 1.0e-6)),
        "min_clearance_m": 0.0,
        "certified_clearance_lower_bound_m": None,
        "clearance_certification": "not_run",
        "npz": str(npz_path.resolve()),
    }


def _write_custom_route(output_dir: Path, route_id: int, control_points: list[tuple[float, float, float]], point_spacing_m: float, waypoint_spacing_m: float) -> dict[str, Any]:

    control = _as_points(control_points)
    smoothed = _resample_polyline(control, point_spacing_m)
    waypoints = _resample_polyline(control, waypoint_spacing_m)
    clearance = np.full((smoothed.shape[0],), np.nan, dtype=np.float32)

    route_dir = output_dir / f"route_{route_id}"
    route_dir.mkdir(parents=True, exist_ok=True)
    npz_path = route_dir / f"custom_route_{route_id}.npz"
    np.savez_compressed(
        npz_path,
        raw_grid_path=control.astype(np.float32),
        smoothed_path=smoothed.astype(np.float32),
        downsampled_waypoints=waypoints.astype(np.float32),
        clearance_m=clearance,
    )

    summary = _route_summary(route_id, control, smoothed, waypoints, npz_path)
    _atomic_write_json(route_dir / f"custom_route_{route_id}.json", summary)
    return summary


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists; pass --overwrite to rebuild it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
