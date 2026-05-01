from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_pipeline_config
from pipelines.runner import run_pipeline


def discover_config_paths(config_root: str | Path) -> list[Path]:
    root = Path(config_root)
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def _extract_leaderboard_row(config_path: Path, summary: dict[str, Any], status: str, error: str | None = None) -> dict[str, Any]:
    experiment = summary.get("experiment", {})
    dataset = summary.get("dataset", {})
    detector = summary.get("detector", {})
    metrics = summary.get("metrics", {})
    rca = summary.get("rca", {})
    rca_metrics = rca.get("metrics", {}) if isinstance(rca, dict) else {}

    return {
        "config_path": str(config_path),
        "status": status,
        "experiment_name": experiment.get("name"),
        "task_type": experiment.get("task_type"),
        "tags": ",".join(experiment.get("tags", [])) if isinstance(experiment.get("tags"), list) else experiment.get("tags"),
        "dataset_name": dataset.get("name"),
        "dataset_type": dataset.get("dataset_type"),
        "detector_name": detector.get("name"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "rca_hit_at_5": rca_metrics.get("hit_at_5"),
        "error": error,
    }


def run_batch(config_root: str | Path, output_dir: str | Path = "artifacts/batch_runs") -> dict[str, Any]:
    config_paths = discover_config_paths(config_root)
    batch_output_dir = Path(output_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for config_path in config_paths:
        try:
            config = load_pipeline_config(config_path)
            summary = run_pipeline(config_path)
            rows.append(_extract_leaderboard_row(Path(config_path), summary, status="success"))
            run_summaries.append(
                {
                    "config_path": str(config_path),
                    "output_dir": config.get("output_dir"),
                    "summary": summary,
                }
            )
        except Exception as exc:
            summary = {
                "dataset": {},
                "detector": {},
            }
            rows.append(
                _extract_leaderboard_row(
                    Path(config_path),
                    summary,
                    status="failed",
                    error=str(exc),
                )
            )

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(batch_output_dir / "leaderboard.csv", index=False)

    batch_summary = {
        "generated_at": datetime.now().isoformat(),
        "config_root": str(config_root),
        "num_configs": len(config_paths),
        "num_success": int((leaderboard["status"] == "success").sum()) if not leaderboard.empty else 0,
        "num_failed": int((leaderboard["status"] == "failed").sum()) if not leaderboard.empty else 0,
        "runs": run_summaries,
    }

    with (batch_output_dir / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(batch_summary, handle, indent=2)

    return {
        "leaderboard_path": str(batch_output_dir / "leaderboard.csv"),
        "batch_summary_path": str(batch_output_dir / "batch_summary.json"),
        "num_configs": batch_summary["num_configs"],
        "num_success": batch_summary["num_success"],
        "num_failed": batch_summary["num_failed"],
    }
