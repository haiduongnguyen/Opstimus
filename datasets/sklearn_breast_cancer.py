from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from datasets.base import BaseDataset, DatasetBundle


class SklearnBreastCancerDataset(BaseDataset):
    def __init__(
        self,
        root: str | Path = "sklearn://breast_cancer",
        anomaly_label: int = 0,
        normal_only_train: bool = True,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        super().__init__(name="sklearn_breast_cancer", root=root)
        self.anomaly_label = anomaly_label
        self.normal_only_train = normal_only_train
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> DatasetBundle:
        dataset = load_breast_cancer(as_frame=True)
        features = dataset.data.copy()
        raw_labels = dataset.target.copy()
        labels = (raw_labels == self.anomaly_label).astype(int)

        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        if self.normal_only_train:
            train_features = train_features.loc[train_labels == 0]

        return DatasetBundle(
            name=self.name,
            train_features=train_features.reset_index(drop=True),
            test_features=test_features.reset_index(drop=True),
            test_labels=test_labels.reset_index(drop=True),
            metadata={
                "source_path": str(self.root),
                "num_features": features.shape[1],
                "num_train_rows": len(train_features),
                "num_test_rows": len(test_features),
                "anomaly_label": self.anomaly_label,
                "normal_only_train": self.normal_only_train,
            },
        )
