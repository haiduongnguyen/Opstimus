from __future__ import annotations

from pathlib import Path

from workflows.inference import run_inference_workflow
from workflows.loader import load_workflow_config
from workflows.train import run_train_workflow


def run_workflow(config_path: str | Path) -> dict:
    config = load_workflow_config(config_path)
    if config["mode"] == "train":
        return run_train_workflow(config)
    if config["mode"] == "inference":
        return run_inference_workflow(config)
    raise ValueError(f"Unsupported workflow mode: {config['mode']}")
