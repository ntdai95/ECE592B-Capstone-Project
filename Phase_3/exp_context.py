"""Feature-group ablation: train the same XGBoost varying only the feature set,
under two split regimes. A gain that appears under RANDOM but vanishes under
SESSION is leakage, not detection.

  arms:    base | base+context | no_anomaly (full minus the 3 Task 3.1 columns)
  regimes: RANDOM (stratified 60/20/20) | SESSION (capture-day-disjoint)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import prep
from evaluation import thr_max_recall, FPR_BUDGET

from evaluation import OUT
SEED = 1
ANOMALY_COLS = ["flow_anomaly_score", "flow_anomaly_flag", "has_anomaly_score"]

DAY_PLAN = {
    "benign":       ("2022-10-07", "2022-10-08"),
    "brute_force":  ("2022-10-17", "2022-10-14"),
    "ddos":         ("2022-09-14", "2022-11-07"),
    "dns_spoofing": ("2022-11-16", "2022-11-15"),
    "dos":          ("2022-08-08", "2022-08-09"),
}


def load_frame():
    df = prep.load_flow_data()
    df = prep.merge_anomaly_scores(df)
    df = prep.merge_capture_metadata(df)
    ctx = pd.read_parquet(f"{OUT}/context_features.parquet")
    df = df.merge(ctx, on="Flow ID", how="left")
    ctx_cols = [c for c in ctx.columns if c != "Flow ID"]
    print(f"frame {df.shape}, context coverage "
          f"{df[ctx_cols[0]].notna().mean()*100:.1f}%")
    return df, ctx_cols


def fit_score(X, y, tr, va, te, seed=SEED):
    med = X.iloc[tr].median(numeric_only=True).fillna(0.0)
    Xf = X.fillna(med).astype(np.float32).to_numpy()
    spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    m = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=spw, tree_method="hist",
                      eval_metric="aucpr", random_state=seed, n_jobs=-1)
    m.fit(Xf[tr], y[tr])
    return m.predict_proba(Xf[va])[:, 1], m.predict_proba(Xf[te])[:, 1]


def at_budget(y_va, va_score, y_te, t_te, score, budget=FPR_BUDGET):
    """Threshold from the validation scores, applied to test. Sizing it on the
    test benign scores would be an oracle threshold."""
    thr = thr_max_recall(y_va, va_score, budget)
    pred = (score >= thr).astype(int)
    per = {c: round(float((pred[t_te == c] == 1).mean()) * 100, 2)
           for c in sorted(set(t_te)) if c != "benign"}
    return {
        "recall": round(float(pred[y_te == 1].mean()) * 100, 2),
        "fpr": round(float(pred[y_te == 0].mean()) * 100, 3),
        "pr_auc": round(float(average_precision_score(y_te, score)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, score)), 4),
        "per_attack": per,
    }


def random_split(df):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.40, random_state=SEED,
                               stratify=df["label"].to_numpy())
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED,
                              stratify=df["label"].to_numpy()[tmp])
    return tr, va, te


def session_split(df):
    """Train on one capture day per class, test on another. The threshold set is
    carved out of train, so no test-day information reaches the operating point."""
    from sklearn.model_selection import train_test_split
    day = df["capture_day"].to_numpy()
    lab = df["label"].to_numpy()
    held = np.zeros(len(df), bool)
    for cls, (d_tr, d_te) in DAY_PLAN.items():
        held |= (lab == cls) & (day == d_te)
    keep = np.zeros(len(df), bool)
    for cls, (d_tr, d_te) in DAY_PLAN.items():
        keep |= (lab == cls) & np.isin(day, [d_tr, d_te])
    tr_all = np.where(keep & ~held)[0]
    te = np.where(keep & held)[0]
    tr, va = train_test_split(tr_all, test_size=0.20, random_state=SEED,
                              stratify=lab[tr_all])
    return tr, va, te


def main():
    df, ctx_cols = load_frame()
    X_all, y, t, _, _ = prep.make_features_and_target(df)
    base_cols = [c for c in X_all.columns if c not in ctx_cols]
    y = np.asarray(y)
    t = np.asarray(t)

    all_cols = X_all.columns.tolist()
    arms = [
        ("base", base_cols),
        ("base+context", all_cols),
        ("no_anomaly", [c for c in all_cols if c not in ANOMALY_COLS]),
    ]

    results = {}
    for regime, splitter in [("RANDOM", random_split), ("SESSION", session_split)]:
        tr, va, te = splitter(df)
        print(f"\n=== {regime}: train {len(tr):,} val {len(va):,} test {len(te):,} ===")
        for name, cols in arms:
            s_va, s = fit_score(X_all[cols], y, tr, va, te)
            r = at_budget(y[va], s_va, y[te], t[te], s)
            results[f"{regime}|{name}"] = r
            print(f"{name:<14} recall {r['recall']:>6.2f}%  fpr {r['fpr']:.2f}%  "
                  f"pr_auc {r['pr_auc']:.4f}  {r['per_attack']}")

    with open(f"{OUT}/exp_context_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}/exp_context_results.json")


if __name__ == "__main__":
    main()
