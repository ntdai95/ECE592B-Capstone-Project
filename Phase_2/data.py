"""Data loading, label encoding, and stratified train/val/test splitting.

Design rules that keep the experiment honestly *unsupervised*:
  * Training receives features only (`X_train`) -- labels are never passed to a
    detector's `fit`.
  * Labels for the val/test splits exist solely for threshold selection and the
    final metric reporting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Dataset:
    """Container bundling a feature matrix with its multi-class + binary labels."""

    X: np.ndarray              # (n, d) float32 features
    y_bin: np.ndarray          # (n,) 0 = benign, 1 = attack
    y_multi: np.ndarray        # (n,) string labels (benign / ddos / ...)
    ids: np.ndarray            # (n,) original packet ids (join key for the graph)
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]


def load_feature_table(feature_set: str = "normalized", nrows: int | None = None) -> pd.DataFrame:
    """Load one of the preprocessed feature CSVs by logical name.

    Parameters
    ----------
    feature_set : {"normalized", "for_data"}
    nrows : optional row cap (handy for quick smoke tests on a laptop).
    """
    if feature_set not in C.FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {list(C.FEATURE_SETS)}")
    path = C.FEATURE_SETS[feature_set]
    df = pd.read_csv(path, nrows=nrows)
    return df


def to_dataset(df: pd.DataFrame) -> Dataset:
    """Split a raw dataframe into features / labels / ids and sanitize values."""
    feature_names = [c for c in df.columns if c not in (C.ID_COL, C.LABEL_COL)]
    X = df[feature_names].to_numpy(dtype=np.float32, copy=True)

    # Defensive: Phase-1 imputed missing values, but guard against residual NaN/inf
    # so a single bad cell can never crash a downstream model.
    n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
    if n_bad:
        # NaN -> 0; +/-inf -> large finite sentinels so an "extreme" value stays
        # extreme after scaling instead of being mapped to neutral 0 (review L3).
        X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)

    y_multi = df[C.LABEL_COL].astype(str).to_numpy()
    y_bin = (y_multi != C.BENIGN_LABEL).astype(np.int64)
    ids = df[C.ID_COL].to_numpy()
    return Dataset(X=X, y_bin=y_bin, y_multi=y_multi, ids=ids, feature_names=feature_names)


def stratified_split(
    ds: Dataset,
    fracs: tuple[float, float, float] = C.SPLIT_FRACS,
    seed: int = 0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Stratified train/val/test split that preserves the per-class proportions.

    Stratification is on the *multi-class* label so each of the five attacks keeps
    its ~3%/5 share in every split (important given the heavy imbalance).
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, "split fractions must sum to 1"
    rng = np.random.default_rng(seed)
    n = len(ds)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(ds.y_multi):
        cls_idx = np.where(ds.y_multi == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_tr = int(round(fracs[0] * n_cls))
        n_va = int(round(fracs[1] * n_cls))
        train_idx.append(cls_idx[:n_tr])
        val_idx.append(cls_idx[n_tr:n_tr + n_va])
        test_idx.append(cls_idx[n_tr + n_va:])

    def gather(idx_parts: list[np.ndarray]) -> Dataset:
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        return Dataset(
            X=ds.X[idx],
            y_bin=ds.y_bin[idx],
            y_multi=ds.y_multi[idx],
            ids=ds.ids[idx],
            feature_names=ds.feature_names,
        )

    return gather(train_idx), gather(val_idx), gather(test_idx)


def subsample(ds: Dataset, n: int, seed: int = 0) -> Dataset:
    """Stratified down-sample to ~n rows, preserving per-class proportions.

    Used for fast smoke tests so the subset still contains every attack type
    (the raw files are benign-first, so a head() slice would be all benign).
    """
    if n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    frac = n / len(ds)
    keep = []
    for cls in np.unique(ds.y_multi):
        cls_idx = np.where(ds.y_multi == cls)[0]
        k = max(1, int(round(frac * len(cls_idx))))
        keep.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return Dataset(
        X=ds.X[idx], y_bin=ds.y_bin[idx], y_multi=ds.y_multi[idx],
        ids=ds.ids[idx], feature_names=ds.feature_names,
    )


def load_splits(
    feature_set: str = "normalized",
    seed: int = 0,
    nrows: int | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Convenience: load a feature set and return (train, val, test)."""
    df = load_feature_table(feature_set, nrows=nrows)
    ds = to_dataset(df)
    return stratified_split(ds, seed=seed)
