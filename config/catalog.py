from __future__ import annotations

from typing import Any


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "credit_card": {
        "dataset": {
            "name": "credit_card",
            "path": "data/raw/creditcard/creditcard.csv",
            "label_col": "Class",
            "drop_columns": ["Time"],
            "test_size": 0.3,
            "random_state": 42,
        },
        "experiment": {
            "task_type": "tabular_ad_rca",
            "tags": ["credit_card", "tabular"],
        },
    },
    "smd.machine_1_1": {
        "dataset": {
            "name": "smd",
            "train_path": "data/raw/SMD_data/ServerMachineDataset/train/machine-1-1.txt",
            "test_path": "data/raw/SMD_data/ServerMachineDataset/test/machine-1-1.txt",
            "label_path": "data/raw/SMD_data/ServerMachineDataset/test_label/machine-1-1.txt",
            "interpretation_label_path": "data/raw/SMD_data/ServerMachineDataset/interpretation_label/machine-1-1.txt",
        },
        "experiment": {
            "task_type": "time_series_ad_rca",
            "tags": ["smd", "machine_1_1", "time_series"],
        },
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
        "experiment": {
            "task_type": "tabular_ad_rca",
            "tags": ["sklearn", "breast_cancer", "tabular"],
        },
    },
}


DETECTOR_PRESETS: dict[str, dict[str, Any]] = {
    "isolation_forest": {
        "detector": {
            "name": "isolation_forest",
            "params": {
                "n_estimators": 100,
                "random_state": 42,
            },
        },
        "experiment": {
            "tags": ["isolation_forest"],
        },
    },
    "lof": {
        "detector": {
            "name": "lof",
            "params": {},
        },
        "experiment": {
            "tags": ["lof"],
        },
    },
    "autoencoder": {
        "detector": {
            "name": "autoencoder",
            "params": {},
        },
        "experiment": {
            "tags": ["autoencoder"],
        },
    },
}
