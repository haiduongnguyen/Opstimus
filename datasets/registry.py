from __future__ import annotations

from typing import Any

from datasets.credit_card import CreditCardDataset
from datasets.smd import SMDDataset


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "credit_card": {
        "dataset_type": "tabular",
        "builder": CreditCardDataset,
    },
    "smd": {
        "dataset_type": "time_series",
        "builder": SMDDataset,
    },
}


def get_dataset_definition(dataset_name: str) -> dict[str, Any]:
    if dataset_name not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}")
    return DATASET_REGISTRY[dataset_name]


def build_dataset(dataset_config: dict[str, Any]):
    dataset_name = dataset_config["name"]
    builder = get_dataset_definition(dataset_name)["builder"]

    if dataset_name == "credit_card":
        return builder(
            root=dataset_config["path"],
            label_col=dataset_config.get("label_col", "Class"),
            drop_columns=dataset_config.get("drop_columns", ["Time"]),
            test_size=dataset_config.get("test_size", 0.3),
            random_state=dataset_config.get("random_state", 42),
        )

    if dataset_name == "smd":
        return builder(
            train_path=dataset_config["train_path"],
            test_path=dataset_config["test_path"],
            label_path=dataset_config["label_path"],
            interpretation_label_path=dataset_config.get("interpretation_label_path"),
        )

    raise ValueError(f"Dataset builder not implemented for: {dataset_name}")
