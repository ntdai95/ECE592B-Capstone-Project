from abc import ABC, abstractmethod


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def fit(self, X):
        ...

    @abstractmethod
    def score(self, X):
        ...
