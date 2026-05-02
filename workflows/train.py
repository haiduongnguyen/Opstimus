from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from datasets import build_dataset
from workflows.executor import execute_detector_run
from workflows.profiles import MODEL_PROFILES, THRESHOLD_PROFILES


def _resolve_model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_models = config["benchmark"].get("models", [])
    if explicit_models:
        return explicit_models

    profile_name = config["benchmark"].get("model_profile")
    if profile_name not in MODEL_PROFILES:
        supported = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unsupported benchmark model_profile: {profile_name}. Supported profiles: {supported}")
    return MODEL_PROFILES[profile_name]


def _resolve_threshold_specs(profile_names: list[str]) -> list[tuple[str, dict[str, Any]]]:
    resolved: list[tuple[str, dict[str, Any]]] = []
    for profile_name in profile_names:
        if profile_name not in THRESHOLD_PROFILES:
            supported = ", ".join(sorted(THRESHOLD_PROFILES))
            raise ValueError(f"Unsupported threshold profile: {profile_name}. Supported profiles: {supported}")
        resolved.append((profile_name, THRESHOLD_PROFILES[profile_name]))
    return resolved


def _run_id(detector_name: str, threshold_name: str) -> str:
    return detector_name if threshold_name == "default" else f"{detector_name}__{threshold_name}"


def _extract_metric(summary: dict[str, Any], metric_name: str):
    metrics = summary.get("metrics", {})
    if metric_name in metrics:
        return metrics.get(metric_name)
    rca_metrics = summary.get("rca", {}).get("metrics", {})
    return rca_metrics.get(metric_name)


def _build_row(run_id: str, result: dict[str, Any], error: str | None = None) -> dict[str, Any]:
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


def _copy_best_run(best_run_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in best_run_dir.iterdir():
        target = destination_dir / item.name
        if item.is_file():
            shutil.copy2(item, target)


def run_train_workflow(config: dict[str, Any]) -> dict[str, Any]:
    workflow_dir = Path(config["output_dir"])
    runs_dir = workflow_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = build_dataset(config["dataset"]).load()
    if dataset_bundle.train_features is None:
        raise ValueError("Train workflow requires dataset with train_features")

    rows: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []
    model_specs = _resolve_model_specs(config)
    threshold_specs = _resolve_threshold_specs(config["benchmark"].get("threshold_profiles", ["default"]))

    for model_spec in model_specs:
        for threshold_name, threshold_config in threshold_specs:
            run_id = _run_id(model_spec["name"], threshold_name)
            run_output_dir = runs_dir / run_id
            try:
                result = execute_detector_run(
                    workflow_config=config,
                    dataset_bundle=dataset_bundle,
                    detector_config=model_spec,
                    threshold_config=threshold_config,
                    output_dir=run_output_dir,
                )
                rows.append(_build_row(run_id, result))
                run_results.append({"run_id": run_id, **result})
            except Exception as exc:
                rows.append(_build_row(run_id, {}, error=str(exc)))

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(workflow_dir / "leaderboard.csv", index=False)

    successful_results = [result for result in run_results if result["summary"].get("metrics") or result["summary"].get("rca")]
    selection_metric = config["selection"]["metric"]
    higher_is_better = config["selection"].get("higher_is_better", True)
    best_run = None
    best_score = None
    for result in successful_results:
        score = _extract_metric(result["summary"], selection_metric)
        if score is None:
            continue
        if best_score is None or (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
            best_score = score
            best_run = result

    if best_run is None and successful_results:
        best_run = successful_results[0]

    best_model_dir = workflow_dir / "best_model"
    if best_run is not None and config["deployment"].get("save_best_model", True):
        _copy_best_run(Path(best_run["output_dir"]), best_model_dir)

    workflow_summary = {
        "generated_at": datetime.now().isoformat(),
        "mode": "train",
        "workflow": config["workflow"],
        "dataset": config["dataset"],
        "selection": config["selection"],
        "benchmark": config["benchmark"],
        "num_runs": len(rows),
        "num_success": int((leaderboard["status"] == "success").sum()) if not leaderboard.empty else 0,
        "num_failed": int((leaderboard["status"] == "failed").sum()) if not leaderboard.empty else 0,
        "best_run_id": best_run["run_id"] if best_run is not None else None,
        "best_run_dir": best_run["output_dir"] if best_run is not None else None,
        "best_metric": selection_metric,
        "best_metric_value": best_score,
        "dashboard_hint": f"http://127.0.0.1:8765/?run={Path(best_run['output_dir']).relative_to('artifacts').as_posix()}" if best_run is not None else None,
    }

    with (workflow_dir / "workflow_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(workflow_summary, handle, indent=2)

    return workflow_summary
