from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from datasets import build_dataset
from workflows.executor import execute_detector_run, execute_saved_model_run
from workflows.profiles import MODEL_PROFILES, THRESHOLD_PROFILES


def _derive_dashboard_run_id(run_output_dir: str | Path) -> str | None:
    run_path = Path(run_output_dir)
    parts = run_path.parts
    if "artifacts" in parts:
        artifacts_index = parts.index("artifacts")
        return Path(*parts[artifacts_index + 1 :]).as_posix()
    return None


def _resolve_inference_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_models = config["inference"].get("models", [])
    if explicit_models:
        return explicit_models

    profile_name = config["inference"].get("model_profile")
    if profile_name not in MODEL_PROFILES:
        supported = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unsupported inference model_profile: {profile_name}. Supported profiles: {supported}")
    return MODEL_PROFILES[profile_name]


def _resolve_threshold_specs(profile_names: list[str]) -> list[tuple[str, dict[str, Any]]]:
    resolved: list[tuple[str, dict[str, Any]]] = []
    for profile_name in profile_names:
        if profile_name not in THRESHOLD_PROFILES:
            supported = ", ".join(sorted(THRESHOLD_PROFILES))
            raise ValueError(f"Unsupported threshold profile: {profile_name}. Supported profiles: {supported}")
        resolved.append((profile_name, THRESHOLD_PROFILES[profile_name]))
    return resolved


def _row(run_id: str, result: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    summary = result.get("summary", {})
    metrics = summary.get("metrics", {})
    rca_metrics = summary.get("rca", {}).get("metrics", {})
    return {
        "run_id": run_id,
        "status": "success" if error is None else "failed",
        "run_dir": result.get("output_dir"),
        "detector_name": summary.get("detector", {}).get("name"),
        "threshold_strategy": summary.get("threshold", {}).get("strategy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "rca_hit_at_5": rca_metrics.get("hit_at_5"),
        "error": error,
    }


def run_inference_workflow(config: dict[str, Any]) -> dict[str, Any]:
    workflow_dir = Path(config["output_dir"])
    runs_dir = workflow_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = build_dataset(config["dataset"]).load()
    if config["inference"].get("use_test_only", True):
        dataset_bundle.train_features = None

    rows: list[dict[str, Any]] = []
    threshold_specs = _resolve_threshold_specs(config["inference"].get("threshold_profiles", ["default"]))
    recommended_run_id = None

    if config["inference"].get("model_source", "profile") == "saved_run":
        saved_run_dir = config["inference"].get("saved_run_dir")
        if not saved_run_dir:
            raise ValueError("inference.saved_run_dir is required when model_source='saved_run'")
        run_id = "saved_model"
        threshold_config = threshold_specs[0][1] if threshold_specs else None
        try:
            result = execute_saved_model_run(
                workflow_config=config,
                dataset_bundle=dataset_bundle,
                saved_run_dir=saved_run_dir,
                threshold_config=threshold_config,
                output_dir=runs_dir / run_id,
            )
            rows.append(_row(run_id, result))
            recommended_run_id = run_id
        except Exception as exc:
            rows.append(_row(run_id, {}, error=str(exc)))
    else:
        model_specs = _resolve_inference_models(config)
        for model_spec in model_specs:
            for threshold_name, threshold_config in threshold_specs:
                run_id = model_spec["name"] if threshold_name == "default" else f"{model_spec['name']}__{threshold_name}"
                try:
                    result = execute_detector_run(
                        workflow_config=config,
                        dataset_bundle=dataset_bundle,
                        detector_config=model_spec,
                        threshold_config=threshold_config,
                        output_dir=runs_dir / run_id,
                    )
                    rows.append(_row(run_id, result))
                    if recommended_run_id is None:
                        recommended_run_id = run_id
                except Exception as exc:
                    rows.append(_row(run_id, {}, error=str(exc)))

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(workflow_dir / "leaderboard.csv", index=False)

    workflow_summary = {
        "generated_at": datetime.now().isoformat(),
        "mode": "inference",
        "workflow": config["workflow"],
        "dataset": config["dataset"],
        "inference": config["inference"],
        "num_runs": len(rows),
        "num_success": int((leaderboard["status"] == "success").sum()) if not leaderboard.empty else 0,
        "num_failed": int((leaderboard["status"] == "failed").sum()) if not leaderboard.empty else 0,
        "recommended_run_id": recommended_run_id,
        "dashboard_hint": (
            f"http://127.0.0.1:8765/?run={_derive_dashboard_run_id(runs_dir / recommended_run_id)}"
            if recommended_run_id and _derive_dashboard_run_id(runs_dir / recommended_run_id)
            else None
        ),
    }

    with (workflow_dir / "workflow_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(workflow_summary, handle, indent=2)

    return workflow_summary
