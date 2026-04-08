from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class InterpretationEvent:
    start: int
    end: int
    channels: list[str]


def _mask_to_segments(anomaly_mask) -> list[tuple[int, int]]:
    mask = pd.Series(anomaly_mask).astype(bool).reset_index(drop=True)
    segments: list[tuple[int, int]] = []
    start = None

    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            segments.append((start, index - 1))
            start = None

    if start is not None:
        segments.append((start, len(mask) - 1))

    return segments


def _parse_interpretation_labels(path: str | Path | None) -> list[InterpretationEvent]:
    if not path:
        return []

    events: list[InterpretationEvent] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        window, channels_str = line.split(":")
        start_str, end_str = window.split("-")
        channels = [f"channel_{int(channel_id) - 1}" for channel_id in channels_str.split(",") if channel_id]
        events.append(
            InterpretationEvent(
                start=int(start_str),
                end=int(end_str),
                channels=channels,
            )
        )
    return events


def _compute_contribution_matrix(train_features: pd.DataFrame, test_features: pd.DataFrame) -> pd.DataFrame:
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    non_zero_std = train_std[train_std > 0]
    robust_floor = max(float(non_zero_std.quantile(0.1)) if not non_zero_std.empty else 0.0, 1e-3)
    stabilized_std = train_std.clip(lower=robust_floor)
    return (test_features - train_mean).abs().divide(stabilized_std, axis=1)


def rank_root_causes(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    anomaly_mask,
    top_k: int = 5,
) -> pd.DataFrame:
    analysis = analyze_root_causes(
        train_features=train_features,
        test_features=test_features,
        anomaly_mask=anomaly_mask,
        top_k=top_k,
    )
    return analysis["global_ranking"]


def analyze_root_causes(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    anomaly_mask,
    top_k: int = 5,
    interpretation_label_path: str | Path | None = None,
) -> dict[str, Any]:
    anomaly_mask = pd.Series(anomaly_mask).astype(bool).reset_index(drop=True)

    if anomaly_mask.sum() == 0:
        empty_ranking = pd.DataFrame(
            [{"feature": "none", "contribution_score": 0.0, "detail": "No anomalies predicted"}]
        )
        return {
            "global_ranking": empty_ranking,
            "segment_rankings": pd.DataFrame(),
            "event_matches": [],
            "rca_metrics": {},
        }

    contribution_matrix = _compute_contribution_matrix(train_features, test_features)
    anomaly_contribution = contribution_matrix.loc[anomaly_mask]
    global_scores = anomaly_contribution.mean(axis=0).sort_values(ascending=False)

    global_ranking = global_scores.head(top_k).reset_index()
    global_ranking.columns = ["feature", "contribution_score"]
    global_ranking["detail"] = "Mean absolute stabilized z-score deviation on predicted anomalies"

    segment_rows: list[dict[str, Any]] = []
    segments = _mask_to_segments(anomaly_mask)
    for segment_id, (start, end) in enumerate(segments):
        segment_scores = contribution_matrix.iloc[start : end + 1].mean(axis=0).sort_values(ascending=False)
        for rank, (feature, score) in enumerate(segment_scores.head(top_k).items(), start=1):
            segment_rows.append(
                {
                    "segment_id": segment_id,
                    "start": start,
                    "end": end,
                    "rank": rank,
                    "feature": feature,
                    "contribution_score": float(score),
                }
            )

    segment_rankings = pd.DataFrame(segment_rows)

    interpretation_events = _parse_interpretation_labels(interpretation_label_path)
    event_matches: list[dict[str, Any]] = []
    hit_at_k = 0

    if interpretation_events and not segment_rankings.empty:
        grouped_segments = {
            segment_id: group.sort_values("rank")
            for segment_id, group in segment_rankings.groupby("segment_id")
        }
        for event in interpretation_events:
            matched_segment_id = None
            matched_prediction: list[str] = []
            for segment_id, group in grouped_segments.items():
                start = int(group["start"].iloc[0])
                end = int(group["end"].iloc[0])
                overlaps = not (end < event.start or start > event.end)
                if overlaps:
                    matched_segment_id = segment_id
                    matched_prediction = group["feature"].tolist()
                    break

            hit = any(feature in event.channels for feature in matched_prediction[:top_k])
            if hit:
                hit_at_k += 1

            event_matches.append(
                {
                    "start": event.start,
                    "end": event.end,
                    "ground_truth_channels": ",".join(event.channels),
                    "matched_segment_id": matched_segment_id,
                    "predicted_top_features": ",".join(matched_prediction[:top_k]),
                    "hit_at_k": hit,
                }
            )

    rca_metrics = {}
    if event_matches:
        rca_metrics = {
            f"hit_at_{top_k}": hit_at_k / len(event_matches),
            "num_interpretation_events": len(event_matches),
            "matched_events": sum(1 for event in event_matches if event["matched_segment_id"] is not None),
        }

    return {
        "global_ranking": global_ranking,
        "segment_rankings": segment_rankings,
        "event_matches": event_matches,
        "rca_metrics": rca_metrics,
    }
