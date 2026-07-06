"""Phase-2 preprocessing: proper feature scaling (the key fix from the EDA).

WHY THIS EXISTS
---------------
The EDA (see ``results/eda_report.txt`` / ``docs/phase2_plan.md`` Section 1.1)
showed that the provided ``normalized_original_data.csv`` is only *partially*
normalized: ~32 of 164 features are unscaled and the ``stream_jitter_*_var``
columns reach 6.7e10 while most features sit near 0-1. Distance/reconstruction
models (k-means, autoencoder, Deep SVDD, the GNN's downstream detector) are then
dominated by those few huge columns -- the likely reason the baselines failed.

This module rescales every feature so each one contributes comparably.

LEAKAGE POLICY
--------------
* In the experiment pipeline the scaler is **fit on the training split only** and
  applied to val/test (see ``run_phase2.py``). Each seed refits its own scaler.

SCALERS
-------
* ``quantile``     QuantileTransformer -> Gaussian (default; robust to heavy tails)
* ``robust``       RobustScaler (median / IQR)
* ``standard``     StandardScaler (z-score)
* ``log_standard`` log1p on non-negative columns, then StandardScaler
* ``none``         identity (used to reproduce the broken baseline)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from .data import Dataset


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------
class LogStandardScaler(BaseEstimator, TransformerMixin):
    """log1p on non-negative columns (tame heavy tails), then z-score everything."""

    def __init__(self):
        self.log_cols_: np.ndarray | None = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.log_cols_ = (X.min(axis=0) >= 0)  # only log non-negative features
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
    """No-op scaler (to reproduce the unscaled baseline behavior)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


def make_scaler(name: str):
    if name == "quantile":
        # n_quantiles capped at n_samples internally; normal output keeps things bounded-ish
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


# ---------------------------------------------------------------------------
# Dataset-level helpers used by the pipeline
# ---------------------------------------------------------------------------
def find_zero_variance(ds: Dataset, tol: float = 1e-12) -> list[str]:
    """Names of constant features (safe, label-free drop)."""
    stds = ds.X.std(axis=0)
    return [n for n, s in zip(ds.feature_names, stds) if s < tol]


def drop_features(ds: Dataset, drop: list[str]) -> Dataset:
    if not drop:
        return ds
    keep = [i for i, n in enumerate(ds.feature_names) if n not in set(drop)]
    return replace(ds, X=ds.X[:, keep], feature_names=[ds.feature_names[i] for i in keep])


def apply_scaler(ds: Dataset, scaler) -> Dataset:
    """Return a copy of ``ds`` with scaled features (scaler already fitted)."""
    return replace(ds, X=scaler.transform(ds.X).astype(np.float32))
