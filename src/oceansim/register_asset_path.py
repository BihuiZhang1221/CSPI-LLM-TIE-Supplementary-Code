from __future__ import annotations

import json
from pathlib import Path


def register_asset_path(asset_path: str) -> Path:
    """Write the OceanSim asset path into the sensor utility directory."""

    target = Path(asset_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"asset path does not exist: {target}")
    destination = Path(__file__).resolve().parent / "utils" / "asset_path.json"
    destination.write_text(json.dumps({"asset_path": str(target)}, indent=2), encoding="utf-8")
    return destination
