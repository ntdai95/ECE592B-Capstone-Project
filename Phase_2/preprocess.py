from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from .data import Dataset


class LogStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_cols_: np.ndarray | None = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.log_cols_ = (X.min(axis=0) >= 0)
        Xt = X.copy()
        Xt[:, self.log_cols_] = np.log1p(Xt[:, self.log_cols_])
        self.scaler_.fit(Xt)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xt = X.copy()
        Xt[:, self.log_cols_] = np.log1p(np.clip(Xt[:, self.log_cols_], a_min=0, a_max=None))
        return self.scaler_.transform(Xt).astype(np.float32)


class IdentityScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


def make_scaler(name: str):
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                   subsample=1_000_000, random_state=0)
    if name == "robust":
        return RobustScaler()
    if name == "standard":
        return StandardScaler()
    if name == "log_standard":
        return LogStandardScaler()
    if name == "none":
        return IdentityScaler()
    raise ValueError(f"Unknown scaler {name!r}")


def find_zero_variance(ds: Dataset, tol: float = 1e-12) -> list[str]:
    stds = ds.X.std(axis=0)
    return [n for n, s in zip(ds.feature_names, stds) if s < tol]


def drop_features(ds: Dataset, drop: list[str]) -> Dataset:
    if not drop:
        return ds
    keep = [i for i, n in enumerate(ds.feature_names) if n not in set(drop)]
    return replace(ds, X=ds.X[:, keep], feature_names=[ds.feature_names[i] for i in keep])


def apply_scaler(ds: Dataset, scaler) -> Dataset:
    return replace(ds, X=scaler.transform(ds.X).astype(np.float32))
