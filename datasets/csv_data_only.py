from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.base import BaseDataset, DatasetBundle


class CSVDataOnlyDataset(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        label_col: str | None = None,
        drop_columns: list[str] | None = None,
    ) -> None:
        super().__init__(name="csv_data_only", root=root)
        self.label_col = label_col
        self.drop_columns = drop_columns or []

    def load(self) -> DatasetBundle:
        data = pd.read_csv(self.root)
        labels = None
        if self.label_col and self.label_col in data.columns:
            labels = data[self.label_col].reset_index(drop=True)

        features = data.drop(columns=[column for column in [self.label_col, *self.drop_columns] if column], errors="ignore")

        return DatasetBundle(
            name=self.name,
            train_features=None,
            test_features=features.reset_index(drop=True),
            test_labels=labels,
            metadata={
                "source_path": str(self.root),
                "num_features": features.shape[1],
                "num_test_rows": len(features),
                "data_mode": "data_only",
            },
        )
