"""Unsupervised anomaly detectors with a uniform fit/score interface.

Every detector exposes:
    fit(X_train)          -> self          (NO labels, ever)
    score(X)              -> np.ndarray     (higher = more anomalous)

This lets the runner treat Deep SVDD and the Anomal-E GNN identically.
"""
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
        """Return anomaly scores; higher means more anomalous."""
        ...
