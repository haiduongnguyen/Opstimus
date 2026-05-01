from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import load_pipeline_config
from datasets import build_dataset, get_dataset_definition
from detection.autoencoder import AutoencoderDetector
from detection.isolation_forest import IsolationForestDetector
from detection.lof import LocalOutlierFactorDetector
from evaluation import evaluate_detection, write_report
from preprocessing.scaling import Scaler
from rca import analyze_root_causes
from thresholding import apply_threshold_strategy


def _build_detector(config: dict[str, Any]):
    detector_config = config["detector"]
    detector_name = detector_config["name"]
    detector_params = detector_config.get("params", {})

    if detector_name == "isolation_forest":
        return IsolationForestDetector(**detector_params)
    if detector_name == "lof":
        detector_params = {"novelty": True, **detector_params}
        return LocalOutlierFactorDetector(**detector_params)
    if detector_name == "autoencoder":
        return AutoencoderDetector(**detector_params)

    raise ValueError(f"Unsupported detector: {detector_name}")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    config = load_pipeline_config(config_path)
    dataset_definition = get_dataset_definition(config["dataset"]["name"])
    dataset = build_dataset(config["dataset"]).load()
    detector = _build_detector(config)

    scaler = Scaler()
    train_array = scaler.fit_transform(dataset.train_features)
    test_array = scaler.transform(dataset.test_features)

    detector.fit(train_array)
    raw_scores = detector.score(test_array)
    threshold_result = apply_threshold_strategy(
        detector_name=config["detector"]["name"],
        detector=detector,
        X_test=test_array,
        scores=raw_scores,
        threshold_config=config["threshold"],
    )
    scores = threshold_result["scores"]
    predictions = threshold_result["predictions"]

    summary: dict[str, Any] = {
        "experiment": config["experiment"],
        "dataset": {
            "name": dataset.name,
            "dataset_type": dataset_definition["dataset_type"],
            **dataset.metadata,
        },
        "detector": config["detector"],
        "preprocessing": config["preprocessing"],
        "threshold": config["threshold"],
    }
    summary["threshold"]["resolved_value"] = threshold_result["threshold_value"]

    if dataset.test_labels is not None:
        summary["metrics"] = evaluate_detection(
            labels=dataset.test_labels.to_numpy(),
            predictions=predictions,
            scores=scores,
        )

    rca_analysis = analyze_root_causes(
        train_features=dataset.train_features,
        test_features=dataset.test_features,
        anomaly_mask=predictions,
        top_k=config.get("rca", {}).get("top_k", 5),
        interpretation_label_path=dataset.metadata.get("interpretation_label_path"),
    )

    summary["rca"] = {
        "global_ranking": rca_analysis["global_ranking"].to_dict(orient="records"),
        "metrics": rca_analysis["rca_metrics"],
    }

    output_dir = Path(config.get("output_dir", "artifacts/default_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler.scaler, scaler_path)

    model_path = output_dir / config["detector"]["name"]
    if config["detector"]["name"] == "autoencoder":
        model_path = model_path.with_suffix(".keras")
    else:
        model_path = model_path.with_suffix(".pkl")
    detector.save_model(model_path)

    predictions_frame = pd.DataFrame({
        "anomaly_score": scores,
        "prediction": predictions.astype(int),
    })
    if dataset.test_labels is not None:
        predictions_frame["label"] = dataset.test_labels.to_numpy()

    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    rca_analysis["global_ranking"].to_csv(output_dir / "root_causes.csv", index=False)
    if not rca_analysis["segment_rankings"].empty:
        rca_analysis["segment_rankings"].to_csv(output_dir / "root_cause_segments.csv", index=False)
    if rca_analysis["event_matches"]:
        pd.DataFrame(rca_analysis["event_matches"]).to_csv(output_dir / "root_cause_event_matches.csv", index=False)
    write_report(output_dir, summary, predictions_frame)

    return summary
