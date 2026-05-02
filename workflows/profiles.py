from __future__ import annotations

from typing import Any


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "smd.machine_1_1": {
        "dataset": {
            "name": "smd",
            "train_path": "data/raw/SMD_data/ServerMachineDataset/train/machine-1-1.txt",
            "test_path": "data/raw/SMD_data/ServerMachineDataset/test/machine-1-1.txt",
            "label_path": "data/raw/SMD_data/ServerMachineDataset/test_label/machine-1-1.txt",
            "interpretation_label_path": "data/raw/SMD_data/ServerMachineDataset/interpretation_label/machine-1-1.txt",
        },
        "dataset_type": "time_series",
        "tags": ["smd", "machine_1_1", "time_series"],
        "default_train_profile": "time_series_baseline",
        "default_inference_profile": "time_series_inference_fast",
    },
    "sklearn_breast_cancer": {
        "dataset": {
            "name": "sklearn_breast_cancer",
            "path": "sklearn://breast_cancer",
            "anomaly_label": 0,
            "normal_only_train": True,
            "test_size": 0.3,
            "random_state": 42,
        },
        "dataset_type": "tabular",
        "tags": ["sklearn", "breast_cancer", "tabular"],
        "default_train_profile": "tabular_baseline",
        "default_inference_profile": "tabular_inference_fast",
    },
    "credit_card": {
        "dataset": {
            "name": "credit_card",
            "path": "data/raw/creditcard/creditcard.csv",
            "label_col": "Class",
            "drop_columns": ["Time"],
            "test_size": 0.3,
            "random_state": 42,
        },
        "dataset_type": "tabular",
        "tags": ["credit_card", "tabular"],
        "default_train_profile": "tabular_baseline",
        "default_inference_profile": "tabular_inference_fast",
    },
    "skab.other_1": {
        "dataset": {
            "name": "skab",
            "train_path": "data/raw/SKAB/data/anomaly-free/anomaly-free.csv",
            "test_path": "data/raw/SKAB/data/other/1.csv",
            "label_col": "anomaly",
            "changepoint_col": "changepoint",
            "timestamp_col": "datetime",
            "csv_separator": ";",
        },
        "dataset_type": "time_series",
        "tags": ["skab", "other_1", "time_series"],
        "default_train_profile": "time_series_baseline",
        "default_inference_profile": "time_series_inference_fast",
    },
    "csv_data_only": {
        "dataset": {
            "name": "csv_data_only",
            "data_path": "",
            "label_col": None,
            "drop_columns": [],
        },
        "dataset_type": "tabular",
        "tags": ["csv", "tabular", "data_only"],
        "default_train_profile": "tabular_baseline",
        "default_inference_profile": "tabular_inference_fast",
    },
}


MODEL_PROFILES: dict[str, list[dict[str, Any]]] = {
    "tabular_baseline": [
        {"name": "isolation_forest", "params": {"n_estimators": 100, "contamination": 0.03, "random_state": 42}},
        {"name": "lof", "params": {"n_neighbors": 35}},
        {"name": "autoencoder", "params": {"epochs": 20, "batch_size": 64}},
    ],
    "time_series_baseline": [
        {"name": "isolation_forest", "params": {"n_estimators": 100, "contamination": 0.05, "random_state": 42}},
        {"name": "lof", "params": {"n_neighbors": 35}},
        {"name": "autoencoder", "params": {"epochs": 15, "batch_size": 128}},
    ],
    "tabular_inference_fast": [
        {"name": "isolation_forest", "params": {"n_estimators": 100, "contamination": 0.03, "random_state": 42}},
        {"name": "lof", "params": {"n_neighbors": 35}},
    ],
    "time_series_inference_fast": [
        {"name": "isolation_forest", "params": {"n_estimators": 100, "contamination": 0.05, "random_state": 42}},
    ],
}


THRESHOLD_PROFILES: dict[str, dict[str, Any]] = {
    "default": {"strategy": "model_default"},
    "percentile_95": {"strategy": "percentile", "percentile": 95},
    "percentile_97": {"strategy": "percentile", "percentile": 97},
    "stddev_3": {"strategy": "stddev", "std_factor": 3},
}
