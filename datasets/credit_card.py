from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.base import BaseDataset, DatasetBundle


class CreditCardDataset(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        label_col: str = "Class",
        drop_columns: list[str] | None = None,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        super().__init__(name="credit_card", root=root)
        self.label_col = label_col
        self.drop_columns = drop_columns or ["Time"]
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> DatasetBundle:
        data = pd.read_csv(self.root)
        labels = data[self.label_col]
        features = data.drop(columns=[self.label_col, *self.drop_columns], errors="ignore")

        train_features, test_features, _, test_labels = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

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
            },
        )
