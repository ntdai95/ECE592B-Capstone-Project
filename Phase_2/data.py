from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Dataset:
    X: np.ndarray
    y_bin: np.ndarray
    y_multi: np.ndarray
    ids: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]


def load_feature_table(feature_set: str = "normalized", nrows: int | None = None) -> pd.DataFrame:
    if feature_set not in C.FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {list(C.FEATURE_SETS)}")
    path = C.FEATURE_SETS[feature_set]
    df = pd.read_csv(path, nrows=nrows)
    return df


def to_dataset(df: pd.DataFrame) -> Dataset:
    feature_names = [c for c in df.columns if c not in (C.ID_COL, C.LABEL_COL)]
    X = df[feature_names].to_numpy(dtype=np.float32, copy=True)

    n_bad = int(np.isnan(X).sum() + np.isinf(X).sum())
    if n_bad:
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
    df = load_feature_table(feature_set, nrows=nrows)
    ds = to_dataset(df)
    return stratified_split(ds, seed=seed)
