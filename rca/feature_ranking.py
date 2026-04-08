from __future__ import annotations

import pandas as pd


def rank_root_causes(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    anomaly_mask,
    top_k: int = 5,
) -> pd.DataFrame:
    anomaly_mask = pd.Series(anomaly_mask).astype(bool)

    if anomaly_mask.sum() == 0:
        return pd.DataFrame(
            [{"feature": "none", "contribution_score": 0.0, "detail": "No anomalies predicted"}]
        )

    baseline = train_features.mean(axis=0)
    anomaly_mean = test_features.loc[anomaly_mask].mean(axis=0)
    contribution = (anomaly_mean - baseline).abs().sort_values(ascending=False)

    ranking = contribution.head(top_k).reset_index()
    ranking.columns = ["feature", "contribution_score"]
    ranking["detail"] = "Absolute mean shift between train baseline and predicted anomalies"
    return ranking
