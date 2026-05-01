from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_scores(detector_name: str, scores) -> np.ndarray:
    normalized = np.asarray(scores, dtype=float)

    if detector_name == "lof":
        # sklearn LOF score_samples is higher for more normal points.
        return -normalized

    return normalized


def _resolve_threshold(strategy: str, scores: np.ndarray, threshold_config: dict[str, Any]) -> float:
    if strategy == "percentile":
        percentile = threshold_config.get("percentile", 95.0)
        return float(np.percentile(scores, percentile))

    if strategy == "stddev":
        mean = float(scores.mean())
        std = float(scores.std())
        std_factor = float(threshold_config.get("std_factor", 3.0))
        return mean + std_factor * std

    if strategy == "value":
        if "value" not in threshold_config:
            raise ValueError("threshold.value must be provided when strategy='value'")
        return float(threshold_config["value"])

    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def apply_threshold_strategy(detector_name: str, detector, X_test, scores, threshold_config: dict[str, Any]) -> dict[str, Any]:
    strategy = threshold_config.get("strategy", "model_default")
    normalized_scores = _normalize_scores(detector_name, scores)

    if strategy == "model_default":
        predictions = np.asarray(detector.predict(X_test)).astype(int)
        return {
            "scores": normalized_scores,
            "predictions": predictions,
            "strategy": strategy,
            "threshold_value": None,
        }

    threshold_value = _resolve_threshold(strategy, normalized_scores, threshold_config)
    predictions = (normalized_scores > threshold_value).astype(int)

    return {
        "scores": normalized_scores,
        "predictions": predictions,
        "strategy": strategy,
        "threshold_value": float(threshold_value),
    }
