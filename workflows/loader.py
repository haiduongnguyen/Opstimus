from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from workflows.profiles import DATASET_PRESETS


DEFAULT_WORKFLOW_CONFIG: dict[str, Any] = {
    "mode": "train",
    "workflow": {
        "name": "",
        "tags": [],
    },
    "dataset": {},
    "preprocessing": {
        "scaler": "standard",
    },
    "benchmark": {
        "model_profile": "",
        "models": [],
        "threshold_profiles": ["default"],
    },
    "inference": {
        "model_source": "profile",
        "model_profile": "",
        "models": [],
        "saved_run_dir": "",
        "threshold_profiles": ["default"],
    },
    "selection": {
        "metric": "f1",
        "higher_is_better": True,
    },
    "rca": {
        "enabled": True,
        "top_k": 5,
    },
    "deployment": {
        "save_best_model": True,
        "enable_local_dashboard": True,
    },
    "output_dir": "",
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_tags(*tag_groups: list[str]) -> list[str]:
    merged_tags: list[str] = []
    for tags in tag_groups:
        for tag in tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
    return merged_tags


def _derive_output_dir(path: Path) -> str:
    return str(Path("artifacts") / "workflows" / path.stem).replace("\\", "/")


def _expand_dataset(raw_dataset: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(raw_dataset, str):
        if raw_dataset not in DATASET_PRESETS:
            supported = ", ".join(sorted(DATASET_PRESETS))
            raise ValueError(f"Unsupported dataset preset: {raw_dataset}. Supported presets: {supported}")
        preset = deepcopy(DATASET_PRESETS[raw_dataset])
        return preset["dataset"], preset
    if isinstance(raw_dataset, dict):
        return raw_dataset, {}
    raise ValueError("Workflow dataset must be a string preset or an object")


def load_workflow_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    dataset_config, dataset_preset = _expand_dataset(raw_config.get("dataset", {}))
    if raw_config.get("dataset_overrides"):
        dataset_config = _deep_merge(dataset_config, raw_config["dataset_overrides"])

    config = _deep_merge(DEFAULT_WORKFLOW_CONFIG, raw_config)
    config["dataset"] = dataset_config
    config["config_path"] = str(path)
    config["config_id"] = path.with_suffix("").as_posix()

    if config["mode"] not in {"train", "inference"}:
        raise ValueError("Workflow mode must be 'train' or 'inference'")

    if not config["workflow"].get("name"):
        config["workflow"]["name"] = path.stem

    config["workflow"]["tags"] = _merge_tags(
        dataset_preset.get("tags", []),
        config["workflow"].get("tags", []),
    )
    config["dataset_type"] = dataset_preset.get("dataset_type")

    if not config["output_dir"]:
        config["output_dir"] = _derive_output_dir(path)

    if config["mode"] == "train" and not config["benchmark"].get("model_profile") and not config["benchmark"].get("models"):
        config["benchmark"]["model_profile"] = dataset_preset.get("default_train_profile", "")

    if config["mode"] == "inference" and not config["inference"].get("model_profile") and not config["inference"].get("models"):
        config["inference"]["model_profile"] = dataset_preset.get("default_inference_profile", "")

    return config
