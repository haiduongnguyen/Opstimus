from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from datasets import DatasetBundle, get_dataset_definition
from detection.autoencoder import AutoencoderDetector
from detection.isolation_forest import IsolationForestDetector
from detection.lof import LocalOutlierFactorDetector
from evaluation import evaluate_detection, write_report
from preprocessing.scaling import Scaler
from rca import analyze_root_causes, build_reference_profile
from thresholding import apply_threshold_strategy


def build_detector(detector_config: dict[str, Any]):
    detector_name = detector_config["name"]
    detector_params = detector_config.get("params", {})

    if detector_name == "isolation_forest":
        return IsolationForestDetector(**detector_params)
    if detector_name == "lof":
        return LocalOutlierFactorDetector(**{"novelty": True, **detector_params})
    if detector_name == "autoencoder":
        return AutoencoderDetector(**detector_params)

    raise ValueError(f"Unsupported detector: {detector_name}")


def get_model_file_path(output_dir: Path, detector_name: str) -> Path:
    extension = ".keras" if detector_name == "autoencoder" else ".pkl"
    return output_dir / f"{detector_name}{extension}"


def _build_summary(
    workflow_config: dict[str, Any],
    dataset_bundle: DatasetBundle,
    detector_config: dict[str, Any],
    threshold_config: dict[str, Any],
) -> dict[str, Any]:
    dataset_definition = get_dataset_definition(dataset_bundle.name)
    return {
        "workflow": workflow_config["workflow"],
        "mode": workflow_config["mode"],
        "dataset": {
            "name": dataset_bundle.name,
            "dataset_type": dataset_definition["dataset_type"],
            **dataset_bundle.metadata,
        },
        "detector": detector_config,
        "preprocessing": workflow_config["preprocessing"],
        "threshold": dict(threshold_config),
    }


def _save_reference_profile(output_dir: Path, reference_profile: dict[str, dict[str, float]]) -> None:
    with (output_dir / "reference_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(reference_profile, handle, indent=2)


def _select_inference_reference(test_features: pd.DataFrame, predictions) -> pd.DataFrame:
    normal_rows = test_features.loc[pd.Series(predictions).astype(int) == 0]
    if len(normal_rows) >= max(10, int(len(test_features) * 0.1)):
        return normal_rows.reset_index(drop=True)
    return test_features.reset_index(drop=True)


def execute_detector_run(
    workflow_config: dict[str, Any],
    dataset_bundle: DatasetBundle,
    detector_config: dict[str, Any],
    threshold_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detector = build_detector(detector_config)
    scaler = Scaler()

    if dataset_bundle.train_features is None:
        reference_train_features = dataset_bundle.test_features.reset_index(drop=True)
        train_array = scaler.fit_transform(reference_train_features)
        test_array = train_array
    else:
        reference_train_features = dataset_bundle.train_features.reset_index(drop=True)
        train_array = scaler.fit_transform(reference_train_features)
        test_array = scaler.transform(dataset_bundle.test_features)

    detector.fit(train_array)
    raw_scores = detector.score(test_array)
    threshold_result = apply_threshold_strategy(
        detector_name=detector_config["name"],
        detector=detector,
        X_test=test_array,
        scores=raw_scores,
        threshold_config=threshold_config,
    )
    scores = threshold_result["scores"]
    predictions = threshold_result["predictions"]

    reference_profile = build_reference_profile(reference_train_features)
    rca_reference = reference_train_features
    if dataset_bundle.train_features is None:
        rca_reference = _select_inference_reference(dataset_bundle.test_features, predictions)
        reference_profile = build_reference_profile(rca_reference)

    summary = _build_summary(workflow_config, dataset_bundle, detector_config, threshold_config)
    summary["threshold"]["resolved_value"] = threshold_result["threshold_value"]

    if dataset_bundle.test_labels is not None:
        summary["metrics"] = evaluate_detection(
            labels=dataset_bundle.test_labels.to_numpy(),
            predictions=predictions,
            scores=scores,
        )

    if workflow_config.get("rca", {}).get("enabled", True):
        rca_analysis = analyze_root_causes(
            train_features=rca_reference,
            test_features=dataset_bundle.test_features,
            anomaly_mask=predictions,
            top_k=workflow_config.get("rca", {}).get("top_k", 5),
            interpretation_label_path=dataset_bundle.metadata.get("interpretation_label_path"),
            reference_profile=reference_profile,
        )
    else:
        rca_analysis = {
            "global_ranking": pd.DataFrame(),
            "segment_rankings": pd.DataFrame(),
            "event_matches": [],
            "rca_metrics": {},
        }

    summary["rca"] = {
        "global_ranking": rca_analysis["global_ranking"].to_dict(orient="records"),
        "metrics": rca_analysis["rca_metrics"],
    }

    predictions_frame = pd.DataFrame({
        "anomaly_score": scores,
        "prediction": predictions.astype(int),
    })
    if dataset_bundle.test_labels is not None:
        predictions_frame["label"] = dataset_bundle.test_labels.to_numpy()

    joblib.dump(scaler.scaler, output_path / "scaler.pkl")
    detector.save_model(get_model_file_path(output_path, detector_config["name"]))
    _save_reference_profile(output_path, reference_profile)

    predictions_frame.to_csv(output_path / "predictions.csv", index=False)
    if not rca_analysis["global_ranking"].empty:
        rca_analysis["global_ranking"].to_csv(output_path / "root_causes.csv", index=False)
    if not rca_analysis["segment_rankings"].empty:
        rca_analysis["segment_rankings"].to_csv(output_path / "root_cause_segments.csv", index=False)
    if rca_analysis["event_matches"]:
        pd.DataFrame(rca_analysis["event_matches"]).to_csv(output_path / "root_cause_event_matches.csv", index=False)
    write_report(output_path, summary, predictions_frame)

    return {
        "summary": summary,
        "output_dir": str(output_path),
        "reference_profile": reference_profile,
    }


def execute_saved_model_run(
    workflow_config: dict[str, Any],
    dataset_bundle: DatasetBundle,
    saved_run_dir: str | Path,
    threshold_config: dict[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    saved_path = Path(saved_run_dir)
    with (saved_path / "summary.json").open("r", encoding="utf-8") as handle:
        saved_summary = json.load(handle)

    detector_config = saved_summary["detector"]
    detector = build_detector(detector_config)
    detector.load_model(get_model_file_path(saved_path, detector_config["name"]))

    scaler = Scaler()
    scaler.scaler = joblib.load(saved_path / "scaler.pkl")
    reference_profile = None
    reference_profile_path = saved_path / "reference_profile.json"
    if reference_profile_path.exists():
        reference_profile = json.loads(reference_profile_path.read_text(encoding="utf-8"))

    test_array = scaler.transform(dataset_bundle.test_features)
    raw_scores = detector.score(test_array)
    resolved_threshold = threshold_config or saved_summary.get("threshold", {"strategy": "model_default"})
    threshold_result = apply_threshold_strategy(
        detector_name=detector_config["name"],
        detector=detector,
        X_test=test_array,
        scores=raw_scores,
        threshold_config=resolved_threshold,
    )
    scores = threshold_result["scores"]
    predictions = threshold_result["predictions"]

    if reference_profile is None:
        reference_profile = build_reference_profile(_select_inference_reference(dataset_bundle.test_features, predictions))

    summary = _build_summary(workflow_config, dataset_bundle, detector_config, resolved_threshold)
    summary["threshold"]["resolved_value"] = threshold_result["threshold_value"]
    summary["loaded_from"] = str(saved_path)

    if dataset_bundle.test_labels is not None:
        summary["metrics"] = evaluate_detection(
            labels=dataset_bundle.test_labels.to_numpy(),
            predictions=predictions,
            scores=scores,
        )

    if workflow_config.get("rca", {}).get("enabled", True):
        rca_analysis = analyze_root_causes(
            train_features=None,
            test_features=dataset_bundle.test_features,
            anomaly_mask=predictions,
            top_k=workflow_config.get("rca", {}).get("top_k", 5),
            interpretation_label_path=dataset_bundle.metadata.get("interpretation_label_path"),
            reference_profile=reference_profile,
        )
    else:
        rca_analysis = {
            "global_ranking": pd.DataFrame(),
            "segment_rankings": pd.DataFrame(),
            "event_matches": [],
            "rca_metrics": {},
        }

    summary["rca"] = {
        "global_ranking": rca_analysis["global_ranking"].to_dict(orient="records"),
        "metrics": rca_analysis["rca_metrics"],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_frame = pd.DataFrame({
        "anomaly_score": scores,
        "prediction": predictions.astype(int),
    })
    if dataset_bundle.test_labels is not None:
        predictions_frame["label"] = dataset_bundle.test_labels.to_numpy()

    predictions_frame.to_csv(output_path / "predictions.csv", index=False)
    if not rca_analysis["global_ranking"].empty:
        rca_analysis["global_ranking"].to_csv(output_path / "root_causes.csv", index=False)
    if not rca_analysis["segment_rankings"].empty:
        rca_analysis["segment_rankings"].to_csv(output_path / "root_cause_segments.csv", index=False)
    if rca_analysis["event_matches"]:
        pd.DataFrame(rca_analysis["event_matches"]).to_csv(output_path / "root_cause_event_matches.csv", index=False)
    write_report(output_path, summary, predictions_frame)

    return {
        "summary": summary,
        "output_dir": str(output_path),
        "reference_profile": reference_profile,
    }
