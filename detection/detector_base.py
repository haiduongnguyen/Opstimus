# detection/detector_base.py
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseDetector(ABC):

    @abstractmethod
    def fit(self, X: Any) -> None:
        pass

    @abstractmethod
    def score(self, X: Any) -> np.ndarray:
        """Return anomaly scores"""
        pass

    @abstractmethod
    def predict(self, X: Any, threshold: Any = None) -> np.ndarray:
        pass
