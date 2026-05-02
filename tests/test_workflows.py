from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from workflows import load_workflow_config, run_workflow


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class WorkflowTests(unittest.TestCase):
    def test_loader_expands_dataset_preset_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = _write_json(
                Path(temp_dir) / "train_sklearn.json",
                {
                    "mode": "train",
                    "workflow": {"name": "loader_case"},
                    "dataset": "sklearn_breast_cancer",
                    "benchmark": {
                        "models": [
                            {
                                "name": "isolation_forest",
                                "params": {"n_estimators": 20, "contamination": 0.35, "random_state": 42},
                            }
                        ],
                        "threshold_profiles": ["default"],
                    },
                },
            )

            config = load_workflow_config(config_path)
            self.assertEqual(config["mode"], "train")
            self.assertEqual(config["dataset"]["name"], "sklearn_breast_cancer")
            self.assertEqual(config["dataset_type"], "tabular")
            self.assertEqual(config["workflow"]["name"], "loader_case")
            self.assertIn("sklearn", config["workflow"]["tags"])
            self.assertTrue(config["output_dir"].endswith("artifacts/workflows/train_sklearn"))

    def test_train_workflow_creates_leaderboard_and_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "artifacts" / "workflows" / "train_case"
            config_path = _write_json(
                Path(temp_dir) / "train_case.json",
                {
                    "mode": "train",
                    "workflow": {"name": "train_case"},
                    "dataset": "sklearn_breast_cancer",
                    "benchmark": {
                        "models": [
                            {
                                "name": "isolation_forest",
                                "params": {"n_estimators": 30, "contamination": 0.35, "random_state": 42},
                            },
                            {
                                "name": "lof",
                                "params": {"n_neighbors": 35},
                            },
                        ],
                        "threshold_profiles": ["default"],
                    },
                    "selection": {"metric": "f1", "higher_is_better": True},
                    "output_dir": str(output_dir),
                },
            )

            summary = run_workflow(config_path)
            self.assertEqual(summary["mode"], "train")
            self.assertEqual(summary["num_runs"], 2)
            self.assertEqual(summary["num_success"], 2)
            self.assertIsNotNone(summary["best_run_id"])

            leaderboard_path = output_dir / "leaderboard.csv"
            workflow_summary_path = output_dir / "workflow_summary.json"
            best_model_summary = output_dir / "best_model" / "summary.json"
            best_model_reference = output_dir / "best_model" / "reference_profile.json"

            self.assertTrue(leaderboard_path.exists())
            self.assertTrue(workflow_summary_path.exists())
            self.assertTrue(best_model_summary.exists())
            self.assertTrue(best_model_reference.exists())

            leaderboard = pd.read_csv(leaderboard_path)
            self.assertSetEqual(set(leaderboard["status"]), {"success"})
            self.assertIn(summary["best_run_id"], leaderboard["run_id"].tolist())

    def test_inference_profile_mode_runs_on_csv_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = load_breast_cancer(as_frame=True)
            frame = dataset.data.head(120).copy()
            frame["label"] = (dataset.target.head(120) == 0).astype(int)
            csv_path = Path(temp_dir) / "sample.csv"
            frame.to_csv(csv_path, index=False)

            output_dir = Path(temp_dir) / "artifacts" / "workflows" / "inference_case"
            config_path = _write_json(
                Path(temp_dir) / "inference_case.json",
                {
                    "mode": "inference",
                    "workflow": {"name": "inference_case"},
                    "dataset": "csv_data_only",
                    "dataset_overrides": {
                        "data_path": str(csv_path),
                        "label_col": "label",
                    },
                    "inference": {
                        "model_source": "profile",
                        "models": [
                            {
                                "name": "isolation_forest",
                                "params": {"n_estimators": 25, "contamination": 0.2, "random_state": 42},
                            }
                        ],
                        "threshold_profiles": ["default"],
                        "use_test_only": True,
                    },
                    "output_dir": str(output_dir),
                },
            )

            summary = run_workflow(config_path)
            self.assertEqual(summary["mode"], "inference")
            self.assertEqual(summary["num_runs"], 1)
            self.assertEqual(summary["num_success"], 1)

            run_summary_path = output_dir / "runs" / "isolation_forest" / "summary.json"
            predictions_path = output_dir / "runs" / "isolation_forest" / "predictions.csv"
            self.assertTrue(run_summary_path.exists())
            self.assertTrue(predictions_path.exists())

    def test_inference_saved_run_loads_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_output_dir = Path(temp_dir) / "artifacts" / "workflows" / "train_saved_source"
            train_config_path = _write_json(
                Path(temp_dir) / "train_saved_source.json",
                {
                    "mode": "train",
                    "workflow": {"name": "train_saved_source"},
                    "dataset": "sklearn_breast_cancer",
                    "benchmark": {
                        "models": [
                            {
                                "name": "isolation_forest",
                                "params": {"n_estimators": 30, "contamination": 0.35, "random_state": 42},
                            }
                        ],
                        "threshold_profiles": ["default"],
                    },
                    "selection": {"metric": "f1", "higher_is_better": True},
                    "output_dir": str(train_output_dir),
                },
            )
            train_summary = run_workflow(train_config_path)
            self.assertEqual(train_summary["num_success"], 1)

            inference_output_dir = Path(temp_dir) / "artifacts" / "workflows" / "inference_saved_target"
            inference_config_path = _write_json(
                Path(temp_dir) / "inference_saved_target.json",
                {
                    "mode": "inference",
                    "workflow": {"name": "inference_saved_target"},
                    "dataset": "sklearn_breast_cancer",
                    "inference": {
                        "model_source": "saved_run",
                        "saved_run_dir": str(train_output_dir / "best_model"),
                        "threshold_profiles": ["default"],
                        "use_test_only": True,
                    },
                    "output_dir": str(inference_output_dir),
                },
            )

            inference_summary = run_workflow(inference_config_path)
            self.assertEqual(inference_summary["mode"], "inference")
            self.assertEqual(inference_summary["num_success"], 1)
            saved_run_summary = json.loads(
                (inference_output_dir / "runs" / "saved_model" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_run_summary["detector"]["name"], "isolation_forest")
            self.assertIn("loaded_from", saved_run_summary)


if __name__ == "__main__":
    unittest.main()
