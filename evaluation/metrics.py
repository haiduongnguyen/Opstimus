from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_detection(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)

    metrics: dict[str, Any] = {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, zero_division=0, output_dict=True),
    }

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))

    return metrics
