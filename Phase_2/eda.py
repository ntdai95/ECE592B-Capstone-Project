from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import load_feature_table, to_dataset


def _guess_scaler(X: np.ndarray) -> str:
    gmin, gmax, gmean = X.min(), X.max(), X.mean()
    if gmin >= -1e-6 and gmax <= 1 + 1e-6:
        return "min-max to [0, 1]"
    if gmin >= -1 - 1e-6 and gmax <= 1 + 1e-6:
        return "min-max / mean-centered in [-1, 1]"
    if abs(gmean) < 0.3 and gmax > 2:
        return "z-score / standardization (unbounded)"
    return "unknown / custom"


def data_quality_report(feature_set: str) -> None:
    df = load_feature_table(feature_set)
    ds = to_dataset(df)
    X = ds.X
    print(f"\n{'=' * 70}\n[1] DATA QUALITY — {feature_set}: "
          f"{X.shape[0]} rows x {X.shape[1]} features\n{'=' * 70}")
    print(f"global  min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}  "
          f"std={X.std():.4f}")
    print(f"likely scaler : {_guess_scaler(X)}")

    stds = X.std(axis=0)
    zero_var = [n for n, s in zip(ds.feature_names, stds) if s < 1e-9]
    near_zero = [n for n, s in zip(ds.feature_names, stds) if 1e-9 <= s < 1e-4]
    print(f"zero-variance features : {len(zero_var)}  -> {zero_var[:12]}"
          + (" ..." if len(zero_var) > 12 else ""))
    print(f"near-zero-variance     : {len(near_zero)}")

    miss = df.isna().sum()
    print(f"columns with missing values (post Phase-1): {(miss > 0).sum()}")
    if (miss > 0).any():
        print(miss[miss > 0].sort_values(ascending=False).head(8).to_string())

    dups = df.drop(columns=[C.ID_COL]).duplicated().sum()
    print(f"duplicate feature rows (ignoring id): {dups} ({dups / len(df) * 100:.2f}%)")


def detectability_report(feature_set: str, benign_sample: int = 20000, seed: int = 0) -> None:
    df = load_feature_table(feature_set)
    ds = to_dataset(df)
    rng = np.random.default_rng(seed)
    print(f"\n{'=' * 70}\n[2] DETECTABILITY (single-feature ROC-AUC vs benign) — "
          f"{feature_set}\n{'=' * 70}")
    print("Per attack: how separable is it from benign using the BEST single "
          "feature?\n(AUC≈1.0 => trivially detectable; AUC≈0.5 => invisible to "
          "that feature)\n")

    benign_idx = np.where(ds.y_multi == C.BENIGN_LABEL)[0]
    benign_idx = rng.choice(benign_idx, size=min(benign_sample, len(benign_idx)), replace=False)
    Xb = ds.X[benign_idx]

    for atk in C.ATTACK_LABELS:
        atk_idx = np.where(ds.y_multi == atk)[0]
        Xa = ds.X[atk_idx]
        X = np.vstack([Xb, Xa])
        y = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xa))])
        aucs = np.full(X.shape[1], 0.5)
        for j in range(X.shape[1]):
            col = X[:, j]
            if col.std() < 1e-12:
                continue
            a = roc_auc_score(y, col)
            aucs[j] = max(a, 1 - a)
        order = np.argsort(aucs)[::-1]
        n_strong = int((aucs >= 0.80).sum())
        best = ", ".join(f"{ds.feature_names[j]}({aucs[j]:.2f})" for j in order[:5])
        print(f"  {atk:13s} best={aucs[order[0]]:.3f}  #feat(AUC>=0.80)={n_strong:3d}  "
              f"| top: {best}")


def dimensionality_report(feature_set: str, sample: int = 40000, seed: int = 0) -> None:
    df = load_feature_table(feature_set)
    ds = to_dataset(df)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(sample, len(ds)), replace=False)
    X = StandardScaler().fit_transform(ds.X[idx])
    p = PCA(n_components=min(X.shape[1], 100), random_state=seed).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    print(f"\n{'=' * 70}\n[3] DIMENSIONALITY (PCA on standardized features) — "
          f"{feature_set}\n{'=' * 70}")
    for thr in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, thr) + 1)
        print(f"  components for {int(thr * 100)}% variance: {k} / {X.shape[1]}")
    print(f"  variance in top-2 PCs: {cum[1] * 100:.1f}%  "
          f"(used for the EDA scatter plot)")


def topology_report() -> None:
    ids = pd.read_csv(C.IDS_CSV)
    labels = load_feature_table("normalized")[[C.ID_COL, C.LABEL_COL]]
    g = ids.merge(labels, on=C.ID_COL, how="inner")
    print(f"\n{'=' * 70}\n[4] GRAPH TOPOLOGY — {len(g)} packets\n{'=' * 70}")

    nodes = pd.unique(pd.concat([g["src_ip"], g["dst_ip"]], ignore_index=True))
    pairs = g.groupby(["src_ip", "dst_ip"]).size()
    print(f"global (IP nodes): {len(nodes)} hosts, {len(pairs)} directed host-pairs, "
          f"max packets on one edge={pairs.max()}\n")

    print(f"  {'class':13s} {'pkts':>7s} {'uSrc':>6s} {'uDst':>6s} "
          f"{'fanIn(max)':>10s} {'fanOut(max)':>11s} {'pkts/edge(mx)':>13s} {'topDst%':>8s}")
    for cls in [C.BENIGN_LABEL] + C.ATTACK_LABELS:
        sub = g[g[C.LABEL_COL] == cls]
        if sub.empty:
            continue
        fan_in = sub.groupby("dst_ip")["src_ip"].nunique()
        fan_out = sub.groupby("src_ip")["dst_ip"].nunique()
        mult = sub.groupby(["src_ip", "dst_ip"]).size()
        top_dst_share = sub["dst_ip"].value_counts(normalize=True).iloc[0] * 100
        print(f"  {cls:13s} {len(sub):>7d} {sub['src_ip'].nunique():>6d} "
              f"{sub['dst_ip'].nunique():>6d} {fan_in.max():>10d} {fan_out.max():>11d} "
              f"{mult.max():>13d} {top_dst_share:>7.1f}%")
    print("\nReading: high fan-in => distributed (DDoS); high pkts/edge => volumetric "
          "to one target (DoS); high top-Dst% => traffic concentrated on a host.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller samples for speed")
    args = ap.parse_args()
    bn = 8000 if args.quick else 20000
    dim = 15000 if args.quick else 40000

    for fs in C.FEATURE_SETS:
        data_quality_report(fs)
    for fs in C.FEATURE_SETS:
        detectability_report(fs, benign_sample=bn)
    for fs in C.FEATURE_SETS:
        dimensionality_report(fs, sample=dim)
    topology_report()


if __name__ == "__main__":
    main()
