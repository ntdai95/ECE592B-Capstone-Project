import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from evaluation import FLOW_DATA_DIR, FPR_BUDGET, OUT, thr_max_recall

SEED = 1

DAY_PLAN = {
    "benign":       ("2022-10-07", "2022-10-08"),
    "brute_force":  ("2022-10-17", "2022-10-14"),
    "ddos":         ("2022-09-14", "2022-11-07"),
    "dns_spoofing": ("2022-11-16", "2022-11-15"),
    "dos":          ("2022-08-08", "2022-08-09"),
}
XGB_KW = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
              tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=SEED)


def load_frame(drop_context=False):
    meta = json.load(open(f"{OUT}/prep_meta.json"))
    df = pd.read_csv(FLOW_DATA_DIR / "normalized_original_data.csv")
    df.columns = [c.strip() for c in df.columns]
    ids = pd.read_csv(FLOW_DATA_DIR / "flow_data_ids.csv")
    ids.columns = [c.strip() for c in ids.columns]
    ids = ids.drop_duplicates(subset="Flow ID", keep="first")
    df = df.merge(ids[["Flow ID", "Timestamp"]], on="Flow ID", how="left")
    sc = pd.read_csv(FLOW_DATA_DIR / "task3_1_flow_anomaly_scores.csv")
    sc = sc.drop_duplicates(subset="Flow ID", keep="first")
    df = df.merge(sc, on="Flow ID", how="left")
    df["has_anomaly_score"] = df["flow_anomaly_score"].notna().astype(int)
    df["flow_anomaly_score"] = df["flow_anomaly_score"].fillna(0.0)
    df["flow_anomaly_flag"] = df["flow_anomaly_flag"].fillna(0).astype(int)
    ctx = pd.read_parquet(f"{OUT}/context_features.parquet")
    df = df.merge(ctx, on="Flow ID", how="left")
    day = pd.to_datetime(df["Timestamp"], errors="coerce",
                         format="mixed", dayfirst=True).dt.date.astype(str).values
    feat = [c for c in meta["features"] if c in df.columns]
    if drop_context:
        keep = [c for c in feat if not c.startswith("ctx_")]
        print(f"dropping {len(feat) - len(keep)} context features -> {len(keep)} base features")
        feat = keep
    X = df[feat].replace([np.inf, -np.inf], np.nan)
    return X, df["label"].astype(str).values, day, feat


def build_masks(lab, day, benign_mode, attack_mode, rng):
    tr = np.zeros(len(lab), bool)
    te = np.zeros(len(lab), bool)
    for cls, (a, b) in DAY_PLAN.items():
        mode = benign_mode if cls == "benign" else attack_mode
        m = lab == cls
        if mode == "day":
            tr |= m & (day == a)
            te |= m & (day == b)
        else:
            i = np.where(m)[0]
            p = rng.permutation(len(i))
            k = int(0.7 * len(i))
            tr[i[p[:k]]] = True
            te[i[p[k:]]] = True
    return tr, te


def run_condition(tag, X, lab, tr, te):
    tr_idx = np.where(tr)[0]
    lab_tr = lab[tr]
    fit_i, thr_i = train_test_split(
        np.arange(len(tr_idx)), test_size=0.2, random_state=SEED,
        stratify=lab_tr if len(set(lab_tr)) > 1 else None)
    fit_rows, thr_rows = tr_idx[fit_i], tr_idx[thr_i]

    med = X.iloc[fit_rows].median(numeric_only=True).fillna(0.0)
    Xfit = X.iloc[fit_rows].fillna(med).astype(np.float32).values
    Xthr = X.iloc[thr_rows].fillna(med).astype(np.float32).values
    Xte = X.iloc[te].fillna(med).astype(np.float32).values
    yfit = (lab[fit_rows] != "benign").astype(int)
    ythr = (lab[thr_rows] != "benign").astype(int)
    yte = (lab[te] != "benign").astype(int)
    spw = float((yfit == 0).sum() / max((yfit == 1).sum(), 1))
    model = XGBClassifier(**XGB_KW, scale_pos_weight=spw).fit(Xfit, yfit)

    s_thr = model.predict_proba(Xthr)[:, 1]
    s = model.predict_proba(Xte)[:, 1]

    thr = thr_max_recall(ythr, s_thr, FPR_BUDGET)
    pred = (s >= thr).astype(int)
    achieved_fpr = float(pred[yte == 0].mean())

    tte = lab[te]
    attacks = sorted(set(tte) - {"benign"})
    det = {c: round(float((pred[tte == c] == 1).mean()) * 100, 2) for c in attacks}
    aucs = {}
    bm = tte == "benign"
    for c in attacks:
        cm = tte == c
        aucs[c] = round(float(roc_auc_score(
            np.r_[np.zeros(int(bm.sum())), np.ones(int(cm.sum()))],
            np.r_[s[bm], s[cm]])), 4)
    out = {
        "condition": tag,
        "n_train": int(len(fit_rows)), "n_threshold": int(len(thr_rows)),
        "n_test": int(te.sum()),
        "threshold_source": "held-out 20% of train",
        "pr_auc": round(float(average_precision_score(yte, s)), 4),
        "roc_auc": round(float(roc_auc_score(yte, s)), 4),
        "recall_at_budget": round(float(pred[yte == 1].mean()), 4),
        "achieved_test_fpr": round(achieved_fpr, 4),
        "fpr_budget": FPR_BUDGET,
        "per_attack_detection": det,
        "per_class_auc": aucs,
    }
    print(f"\n  {tag}")
    print(f"    PR-AUC={out['pr_auc']:.3f}  recall={out['recall_at_budget']*100:.1f}%  "
          f"achieved FPR={achieved_fpr*100:.2f}% (budget {FPR_BUDGET*100:.0f}%)"
          f"{'  <-- BUDGET BREACHED' if achieved_fpr > FPR_BUDGET * 1.5 else ''}")
    print(f"    detection: " + "  ".join(f"{k}={v}" for k, v in det.items()))
    print(f"    AUC      : " + "  ".join(f"{k}={v}" for k, v in aucs.items()))
    return out


def main():
    ap = argparse.ArgumentParser(description="Capture-session held-out evaluation")
    ap.add_argument("--features", default="all", choices=["all", "base"],
                    help="'base' drops the ctx_* context features")
    args = ap.parse_args()
    base = args.features == "base"

    X, lab, day, feat = load_frame(drop_context=base)
    print(f"features={len(feat)}  rows={len(lab)}  (xss excluded: single capture day)")
    print("\ncapture-day plan (train -> test):")
    for c, (a, b) in DAY_PLAN.items():
        print(f"  {c:<14} {a} (n={int(((lab==c)&(day==a)).sum()):>6})  ->  "
              f"{b} (n={int(((lab==c)&(day==b)).sum()):>6})")

    rng = np.random.RandomState(SEED)
    results = []
    for tag, bmode, amode in [
        ("C3 benign=random, attacks=random  (reference)", "random", "random"),
        ("C1 benign=DAY held out, attacks=random", "day", "random"),
        ("C2 benign=random, attacks=DAY held out", "random", "day"),
        ("C0 benign=DAY, attacks=DAY  (honest generalisation)", "day", "day"),
    ]:
        tr, te = build_masks(lab, day, bmode, amode, rng)
        results.append(run_condition(tag, X, lab, tr, te))

    out = {"day_plan": DAY_PLAN, "excluded": ["xss"], "conditions": results}
    if base:
        out["feature_set"] = "base (context features excluded)"
    name = "session_holdout_base_results.json" if base else "session_holdout_results.json"
    with open(f"{OUT}/{name}", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}/{name}")


if __name__ == "__main__":
    main()
