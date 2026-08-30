from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        ...
