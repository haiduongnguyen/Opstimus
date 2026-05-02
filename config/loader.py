from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from config.catalog import DATASET_PRESETS, DETECTOR_PRESETS


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


def _derive_output_dir(path: Path) -> str:
    try:
        config_index = path.parts.index("config")
        relative = Path(*path.parts[config_index + 1 :]).with_suffix("")
    except (ValueError, IndexError):
        relative = path.with_suffix("")
    return str(Path("artifacts") / relative).replace("\\", "/")


def _merge_tags(*tag_groups: list[str]) -> list[str]:
    merged_tags: list[str] = []
    for tags in tag_groups:
        for tag in tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
    return merged_tags


def _expand_dataset(raw_dataset: Any) -> dict[str, Any]:
    if isinstance(raw_dataset, str):
        if raw_dataset not in DATASET_PRESETS:
            supported = ", ".join(sorted(DATASET_PRESETS))
            raise ValueError(f"Unsupported dataset preset: {raw_dataset}. Supported presets: {supported}")
        return deepcopy(DATASET_PRESETS[raw_dataset]["dataset"])
    if isinstance(raw_dataset, dict):
        return raw_dataset
    raise ValueError("Config dataset must be either a string preset or an object")


def _expand_detector(raw_detector: Any) -> dict[str, Any]:
    if isinstance(raw_detector, str):
        if raw_detector not in DETECTOR_PRESETS:
            supported = ", ".join(sorted(DETECTOR_PRESETS))
            raise ValueError(f"Unsupported detector preset: {raw_detector}. Supported presets: {supported}")
        return deepcopy(DETECTOR_PRESETS[raw_detector]["detector"])
    if isinstance(raw_detector, dict):
        return raw_detector
    raise ValueError("Config detector must be either a string preset or an object")


def _normalize_threshold(raw_threshold: Any) -> dict[str, Any]:
    if raw_threshold in (None, "", "model_default", "default"):
        return {"strategy": "model_default"}
    if isinstance(raw_threshold, dict):
        return raw_threshold
    if isinstance(raw_threshold, (int, float)):
        return {"strategy": "value", "value": float(raw_threshold)}
    if isinstance(raw_threshold, str):
        if ":" not in raw_threshold:
            return {"strategy": raw_threshold}
        strategy, raw_value = raw_threshold.split(":", 1)
        strategy = strategy.strip()
        raw_value = raw_value.strip()
        if strategy == "percentile":
            return {"strategy": "percentile", "percentile": float(raw_value)}
        if strategy == "stddev":
            return {"strategy": "stddev", "std_factor": float(raw_value)}
        if strategy == "value":
            return {"strategy": "value", "value": float(raw_value)}
    raise ValueError("Unsupported threshold config. Use object, number, or string like 'percentile:97'.")


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    normalized_raw = deepcopy(raw_config)
    normalized_raw["dataset"] = _expand_dataset(raw_config.get("dataset", {}))
    normalized_raw["detector"] = _expand_detector(raw_config.get("detector", {}))
    normalized_raw["threshold"] = _normalize_threshold(raw_config.get("threshold"))
    if raw_config.get("dataset_overrides"):
        normalized_raw["dataset"] = _deep_merge(normalized_raw["dataset"], raw_config["dataset_overrides"])
    if raw_config.get("params"):
        normalized_raw["detector"] = _deep_merge(normalized_raw["detector"], {"params": raw_config["params"]})
    if raw_config.get("detector_overrides"):
        normalized_raw["detector"] = _deep_merge(normalized_raw["detector"], raw_config["detector_overrides"])

    config = _deep_merge(DEFAULT_PIPELINE_CONFIG, normalized_raw)
    config["config_path"] = str(path)
    config["config_id"] = path.with_suffix("").as_posix()

    if not config["dataset"].get("name"):
        raise ValueError("Config must define dataset.name")
    if not config["detector"].get("name"):
        raise ValueError("Config must define detector.name")
    if not config["experiment"].get("name"):
        config["experiment"]["name"] = path.stem
    if "output_dir" not in raw_config:
        config["output_dir"] = _derive_output_dir(path)

    dataset_key = raw_config.get("dataset") if isinstance(raw_config.get("dataset"), str) else None
    detector_key = raw_config.get("detector") if isinstance(raw_config.get("detector"), str) else None

    experiment = deepcopy(DEFAULT_PIPELINE_CONFIG["experiment"])
    if dataset_key and dataset_key in DATASET_PRESETS:
        experiment = _deep_merge(experiment, DATASET_PRESETS[dataset_key].get("experiment", {}))
    if detector_key and detector_key in DETECTOR_PRESETS:
        detector_experiment = DETECTOR_PRESETS[detector_key].get("experiment", {})
        experiment = _deep_merge(experiment, detector_experiment)
    experiment = _deep_merge(experiment, raw_config.get("experiment", {}))
    experiment["tags"] = _merge_tags(
        DEFAULT_PIPELINE_CONFIG["experiment"].get("tags", []),
        DATASET_PRESETS.get(dataset_key, {}).get("experiment", {}).get("tags", []),
        DETECTOR_PRESETS.get(detector_key, {}).get("experiment", {}).get("tags", []),
        raw_config.get("experiment", {}).get("tags", []),
    )

    if not experiment.get("name"):
        experiment["name"] = path.stem

    config["experiment"] = experiment

    return config
