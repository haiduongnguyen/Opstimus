from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "",
        "task_type": "anomaly_detection_rca",
        "tags": [],
    },
    "dataset": {},
    "preprocessing": {
        "scaler": "standard",
    },
    "detector": {
        "name": "",
        "params": {},
    },
    "threshold": {
        "strategy": "model_default",
    },
    "rca": {
        "top_k": 5,
    },
    "output_dir": "artifacts/default_run",
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    config = _deep_merge(DEFAULT_PIPELINE_CONFIG, raw_config)
    config["config_path"] = str(path)
    config["config_id"] = path.with_suffix("").as_posix()

    if not config["dataset"].get("name"):
        raise ValueError("Config must define dataset.name")
    if not config["detector"].get("name"):
        raise ValueError("Config must define detector.name")
    if not config["experiment"].get("name"):
        config["experiment"]["name"] = path.stem

    return config
