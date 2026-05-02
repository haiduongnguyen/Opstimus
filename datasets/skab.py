from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.base import BaseDataset, DatasetBundle


class SKABDataset(BaseDataset):
    def __init__(
        self,
        train_path: str | Path,
        test_path: str | Path,
        label_col: str = "anomaly",
        changepoint_col: str = "changepoint",
        timestamp_col: str = "datetime",
        csv_separator: str = ";",
    ) -> None:
        super().__init__(name="skab", root=Path(train_path).parent)
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.label_col = label_col
        self.changepoint_col = changepoint_col
        self.timestamp_col = timestamp_col
        self.csv_separator = csv_separator

    def load(self) -> DatasetBundle:
        train_frame = pd.read_csv(self.train_path, sep=self.csv_separator)
        test_frame = pd.read_csv(self.test_path, sep=self.csv_separator)

        train_features = train_frame.drop(columns=[self.timestamp_col, self.label_col, self.changepoint_col], errors="ignore")
        test_labels = test_frame[self.label_col].astype(int).reset_index(drop=True)
        changepoints = test_frame[self.changepoint_col].astype(int).reset_index(drop=True) if self.changepoint_col in test_frame.columns else None
        test_features = test_frame.drop(columns=[self.timestamp_col, self.label_col, self.changepoint_col], errors="ignore")

        return DatasetBundle(
            name=self.name,
            train_features=train_features.reset_index(drop=True),
            test_features=test_features.reset_index(drop=True),
            test_labels=test_labels,
            metadata={
                "train_path": str(self.train_path),
                "test_path": str(self.test_path),
                "timestamp_col": self.timestamp_col,
                "label_col": self.label_col,
                "changepoint_col": self.changepoint_col,
                "num_channels": test_features.shape[1],
                "num_train_rows": len(train_features),
                "num_test_rows": len(test_features),
                "num_anomaly_points": int(test_labels.sum()),
                "num_changepoints": int(changepoints.sum()) if changepoints is not None else None,
            },
        )
