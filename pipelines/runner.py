from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from datasets import CreditCardDataset, SMDDataset
from detection.autoencoder import AutoencoderDetector
from detection.isolation_forest import IsolationForestDetector
from detection.lof import LocalOutlierFactorDetector
from evaluation import evaluate_detection, write_report
from preprocessing.scaling import Scaler
from rca import rank_root_causes


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_dataset(config: dict[str, Any]):
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]

    if dataset_name == "credit_card":
        return CreditCardDataset(
            root=dataset_config["path"],
            label_col=dataset_config.get("label_col", "Class"),
            drop_columns=dataset_config.get("drop_columns", ["Time"]),
            test_size=dataset_config.get("test_size", 0.3),
            random_state=dataset_config.get("random_state", 42),
        )

    if dataset_name == "smd":
        return SMDDataset(
            train_path=dataset_config["train_path"],
            test_path=dataset_config["test_path"],
            label_path=dataset_config["label_path"],
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


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
    config = _load_config(config_path)
    dataset = _build_dataset(config).load()
    detector = _build_detector(config)

    scaler = Scaler()
    train_array = scaler.fit_transform(dataset.train_features)
    test_array = scaler.transform(dataset.test_features)

    detector.fit(train_array)
    scores = detector.score(test_array)
    predictions = detector.predict(test_array)

    summary: dict[str, Any] = {
        "dataset": {
            "name": dataset.name,
            **dataset.metadata,
        },
        "detector": config["detector"],
    }

    if dataset.test_labels is not None:
        summary["metrics"] = evaluate_detection(
            labels=dataset.test_labels.to_numpy(),
            predictions=predictions,
            scores=scores,
        )

    feature_ranking = rank_root_causes(
        train_features=dataset.train_features,
        test_features=dataset.test_features,
        anomaly_mask=predictions,
        top_k=config.get("rca", {}).get("top_k", 5),
    )

    summary["rca"] = feature_ranking.to_dict(orient="records")

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
    feature_ranking.to_csv(output_dir / "root_causes.csv", index=False)
    write_report(output_dir, summary, predictions_frame)

    return summary
