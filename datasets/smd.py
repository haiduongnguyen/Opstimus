from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.base import BaseDataset, DatasetBundle


class SMDDataset(BaseDataset):
    def __init__(
        self,
        train_path: str | Path,
        test_path: str | Path,
        label_path: str | Path,
    ) -> None:
        super().__init__(name="smd", root=Path(train_path).parent)
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.label_path = Path(label_path)

    def load(self) -> DatasetBundle:
        train_features = pd.read_csv(self.train_path, header=None)
        test_features = pd.read_csv(self.test_path, header=None)
        test_labels = pd.read_csv(self.label_path, header=None).iloc[:, 0]

        column_names = [f"channel_{index}" for index in range(train_features.shape[1])]
        train_features.columns = column_names
        test_features.columns = column_names

        return DatasetBundle(
            name=self.name,
            train_features=train_features,
            test_features=test_features,
            test_labels=test_labels,
            metadata={
                "train_path": str(self.train_path),
                "test_path": str(self.test_path),
                "label_path": str(self.label_path),
                "num_channels": train_features.shape[1],
                "num_train_rows": len(train_features),
                "num_test_rows": len(test_features),
            },
        )
