"""Deterministic reference reward used by the inspection task."""

from __future__ import annotations

import math
from typing import Any

def reference_reward(
    initial_dsafe: float,
    *,
    safe_distance_progress_scale: float = 8.0,
    success_reward: float = 300.0,
) -> float:
    return float(safe_distance_progress_scale) * float(initial_dsafe) + float(success_reward)
