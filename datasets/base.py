from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    train_features: Optional[pd.DataFrame]
    test_features: pd.DataFrame
    test_labels: Optional[pd.Series]
    metadata: dict


class BaseDataset(ABC):
    def __init__(self, name: str, root: str | Path) -> None:
        self.name = name
        self.root = Path(root)

    @abstractmethod
    def load(self) -> DatasetBundle:
        raise NotImplementedError
