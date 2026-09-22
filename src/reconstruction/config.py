from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "source_path": "./third_party/vggt",
        "checkpoint": "./models/vggt/model.pt",
        "device": "cuda",
        "allow_cpu_fallback": False,
        "dtype": "bfloat16",
    },
    "frames": {
        "num_frames": 60,
        "frame_interval": None,
        "start_sec": 0.0,
        "end_sec": None,
        "resize_max": 1024,
        "remove_blur": True,
        "blur_threshold": 80.0,
        "jpeg_quality": 95,
    },
    "inference": {"query_points": [], "conf_threshold": 5.0},
    "export": {"save_pointmaps": True, "max_points": 2000000, "colmap": True},
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("The YAML config root must be a mapping")
        _merge(config, loaded)
    return config


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()
