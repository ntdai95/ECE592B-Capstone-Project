import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from evaluation import FLOW_DATA_DIR, OUT
SEED = 1
OUTLIER_CONTAMINATION = 0.05


def load_flow_data():
    df = pd.read_csv(FLOW_DATA_DIR / "original_data.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


def merge_anomaly_scores(df):
    sc = pd.read_csv(FLOW_DATA_DIR / "task3_1_flow_anomaly_scores.csv")
    sc = sc.drop_duplicates(subset="Flow ID", keep="first")
    df = df.merge(sc, on="Flow ID", how="left")
    df["has_anomaly_score"] = df["flow_anomaly_score"].notna().astype(int)
    df["flow_anomaly_score"] = df["flow_anomaly_score"].fillna(0.0)
    df["flow_anomaly_flag"] = df["flow_anomaly_flag"].fillna(0).astype(int)
    return df


def merge_context_features(df):
    path = f"{OUT}/context_features.parquet"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} missing, run `python build_context_features.py` first.")
    ctx = pd.read_parquet(path)
    df = df.merge(ctx, on="Flow ID", how="left")
    ctx_cols = [c for c in ctx.columns if c != "Flow ID"]
    print(f"context features: {len(ctx_cols)} cols, "
          f"coverage {df[ctx_cols[0]].notna().mean()*100:.1f}%")
    return df


def merge_capture_metadata(df):
    ids = pd.read_csv(FLOW_DATA_DIR / "flow_data_ids.csv")
    ids.columns = [c.strip() for c in ids.columns]
    ids = ids.drop_duplicates(subset="Flow ID", keep="first")
    df = df.merge(ids[["Flow ID", "Timestamp", "Dst Port", "Protocol"]], on="Flow ID", how="left")
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed", dayfirst=True)
    df["capture_day"] = ts.dt.date.astype(str)
    return df


def make_features_and_target(df):
    y_type = df["label"].astype(str)
    y = (y_type != "benign").astype(int)
    meta_cols = ["label", "Flow ID", "Timestamp", "Dst Port", "Protocol", "capture_day"]
    X = df.drop(columns=[c for c in meta_cols if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.astype(np.float32), y.values, y_type.values, list(X.columns), df["capture_day"].values


def split(X, y, y_type, seed=SEED):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y_type)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=y_type[tmp])
    return tr, va, te


def preprocess_from_train(X, tr, flow_features):
    """Fit every data-dependent preprocessing step on training rows only."""
    med = X.iloc[tr].median(numeric_only=True)
    med = med.fillna(0.0)
    clean = X.fillna(med).astype(np.float32).copy()

    # This fixed element-wise transform has no fitted state. It matches the
    # transform used by the original flow feature pipeline.
    clean.loc[:, flow_features] = np.log1p(clean[flow_features].clip(lower=0))

    detector = IsolationForest(
        contamination=OUTLIER_CONTAMINATION,
        random_state=42,
        n_jobs=1,
    )
    inlier = detector.fit_predict(clean.iloc[tr][flow_features]) == 1
    clean_tr = np.asarray(tr)[inlier]

    scaler = StandardScaler()
    scaler.fit(clean.iloc[clean_tr][flow_features])
    clean.loc[:, flow_features] = scaler.transform(clean[flow_features])
    return clean.astype(np.float32), clean_tr, med, int((~inlier).sum())


def main():
    df = load_flow_data()
    flow_features = [c for c in df.columns if c not in ["Flow ID", "label"]]
    df = merge_anomaly_scores(df)
    df = merge_context_features(df)
    df = merge_capture_metadata(df)
    X, y, y_type, cols, cap_day = make_features_and_target(df)
    tr, va, te = split(X, y, y_type)
    train_before_outliers = len(tr)
    X, tr, med, outliers_removed = preprocess_from_train(X, tr, flow_features)
    Xv = X.values
    np.savez_compressed(
        f"{OUT}/splits.npz",
        X_tr=Xv[tr], X_va=Xv[va], X_te=Xv[te],
        y_tr=y[tr], y_va=y[va], y_te=y[te],
        t_tr=y_type[tr], t_va=y_type[va], t_te=y_type[te],
        day_tr=cap_day[tr], day_va=cap_day[va], day_te=cap_day[te],
        idx_tr=tr, idx_va=va, idx_te=te,
    )
    meta = {
        "rows": int(len(y)), "features": cols, "n_features": len(cols),
        "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
        "train_attack": int(y[tr].sum()), "val_attack": int(y[va].sum()),
        "test_attack": int(y[te].sum()), "seed": SEED,
        "anomaly_score_coverage": float((X["has_anomaly_score"] == 1).mean()),
        "preprocessing": {
            "order": "split, train-median imputation, log1p, train-only outlier removal, train-only scaling",
            "imputation": "feature medians fitted on training split only; no label use",
            "outliers": "IsolationForest fitted without labels on training split; only training outliers removed",
            "outlier_contamination": OUTLIER_CONTAMINATION,
            "train_before_outliers": int(train_before_outliers),
            "train_outliers_removed": outliers_removed,
            "scaling": "StandardScaler fitted on retained training rows only",
        },
        "context_features": int(sum(c.startswith("ctx_") for c in cols)),
        "context_note": ("causal per-source/per-destination connection-window "
                         "statistics over the full 886,621-flow capture; "
                         "label-free, no future information"),
        "capture_days_per_label": {
            str(k): sorted(map(str, v)) for k, v in
            pd.DataFrame({"l": y_type, "d": cap_day}).groupby("l")["d"].unique().items()
        },
    }
    with open(f"{OUT}/prep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(f"{OUT}/train_medians.json", "w") as f:
        json.dump({c: float(med.get(c, 0.0)) for c in cols}, f)
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
