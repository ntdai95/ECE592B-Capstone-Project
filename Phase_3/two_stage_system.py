"""Combined two-stage system: stage 1 (Phase 2 autoencoder) -> stage 2 (the best
Phase 3 flow classifier).

The cascade the brief describes: stage 1 flags packet-level alerts, stage 2
re-checks the flows behind them, and an item is finally called an attack only if
stage 1 alerts it AND stage 2 confirms it. Items stage 1 never alerts stay benign.

Stage 2 is the leaderboard's highest-recall model (auto-selected from
results.json), scored on the alert flows in the exact feature space it was trained
on: the alert flows were normalised once in flow_based_feature_engineering.py with
the same fitted scaler as the training data, so nothing is re-scaled here. The
context and Task 3.1 columns are joined raw by Flow ID (as in prep.py) and any
gaps filled with the training medians.

Inputs, all already written by the pipeline:
  - results.json["phase2_autoencoder"]["test"]     stage 1 test confusion
  - phase2_alert_corresponding_flows.csv            normalised alert flows
  - context_features.parquet, task3_1 scores        the two engineered blocks
  - prep_meta.json, train_medians.json              feature order + imputation
  - model_*.joblib                                  the fitted stage-2 models

Alerts with no matching flow keep stage 1's decision, so the combined counts sum
back to the full stage-1 test set.
"""
import json

import joblib
import numpy as np
import pandas as pd

from evaluation import FLOW_DATA_DIR, OUT, RESULTS_DIR, load_results, save_result

ALERT_CSV = FLOW_DATA_DIR / "phase2_alert_corresponding_flows.csv"
ANOMALY_COLS = ["flow_anomaly_score", "flow_anomaly_flag", "has_anomaly_score"]
NICE = {
    "multiclass_xgb": "Multiclass XGBoost", "xgboost": "XGBoost",
    "voting_soft": "Soft Voting (RF+XGB+MLP)", "random_forest": "Random Forest",
    "mlp": "Neural Network (MLP)", "linear_svm": "Linear SVM",
}


def best_model(results):
    """The leaderboard model with the highest tuned test recall."""
    cands = [(k, results[k]["test_tuned"]["recall"]) for k in NICE if k in results]
    return max(cands, key=lambda x: x[1])[0]


def attack_scores(model_key, X):
    """Attack probability from a fitted model, in the training feature space."""
    if model_key == "random_forest":
        return joblib.load(f"{OUT}/model_rf.joblib").predict_proba(X)[:, 1]
    if model_key == "xgboost":
        return joblib.load(f"{OUT}/model_xgb.joblib").predict_proba(X)[:, 1]
    if model_key == "mlp":
        return joblib.load(f"{OUT}/model_mlp.joblib").predict_proba(X)[:, 1]
    if model_key == "linear_svm":
        m = joblib.load(f"{OUT}/model_svm.joblib")
        return m["cal"].predict_proba(m["svm"].decision_function(X).reshape(-1, 1))[:, 1]
    if model_key == "voting_soft":
        rf = joblib.load(f"{OUT}/model_rf.joblib")
        xgb = joblib.load(f"{OUT}/model_xgb.joblib")
        mlp = joblib.load(f"{OUT}/model_mlp.joblib")
        return np.column_stack([rf.predict_proba(X)[:, 1], xgb.predict_proba(X)[:, 1],
                                mlp.predict_proba(X)[:, 1]]).mean(axis=1)
    if model_key == "multiclass_xgb":
        return 1 - joblib.load(f"{OUT}/model_multiclass.joblib").predict_proba(X)[:, 0]
    raise ValueError(f"unknown model {model_key}")


def build_alert_features(features, medians):
    """Alert-flow feature matrix in prep.py's exact column order and space."""
    df = pd.read_csv(ALERT_CSV)
    df.columns = [c.strip() for c in df.columns]

    ctx = pd.read_parquet(f"{OUT}/context_features.parquet")
    df = df.merge(ctx, on="Flow ID", how="left")
    sc = FLOW_DATA_DIR / "task3_1_flow_anomaly_scores.csv"
    if sc.exists():
        df = df.merge(pd.read_csv(sc).drop_duplicates("Flow ID"), on="Flow ID", how="left")
    df["has_anomaly_score"] = df.get("flow_anomaly_score", pd.Series(np.nan, index=df.index)).notna().astype(int)
    df["flow_anomaly_score"] = df.get("flow_anomaly_score", 0.0)
    df["flow_anomaly_flag"] = df.get("flow_anomaly_flag", 0)

    for c in features:
        if c not in df.columns:
            df[c] = medians.get(c, 0.0)
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(medians)).fillna(0.0)
    return df, X.astype(np.float32).values


def combined_counts(stage1, packet_label, stage2_attack):
    tn1, fp1, fn1, tp1 = stage1["tn"], stage1["fp"], stage1["fn"], stage1["tp"]
    is_benign = packet_label == "benign"
    n_fp_mapped = int(is_benign.sum())
    n_tp_mapped = int((~is_benign).sum())
    cleared = int((is_benign & ~stage2_attack).sum())     # benign alerts stage 2 clears
    retained = int((~is_benign & stage2_attack).sum())    # attack alerts stage 2 keeps

    # Alerts with no matching flow keep stage 1's decision (still an alert).
    unmapped_fp = max(fp1 - n_fp_mapped, 0)
    unmapped_tp = max(tp1 - n_tp_mapped, 0)

    tp = retained + unmapped_tp
    fp = (n_fp_mapped - cleared) + unmapped_fp
    fn = fn1 + (n_tp_mapped - retained)
    tn = tn1 + cleared
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, unmapped=unmapped_fp + unmapped_tp)


def metrics(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run():
    results = load_results()
    stage1_entry = results.get("phase2_autoencoder")
    if not stage1_entry:
        print("[two-stage] phase2_autoencoder not in results.json; run Phase 2 first.")
        return None
    if not ALERT_CSV.exists():
        print(f"[two-stage] {ALERT_CSV.name} not found; run flow_based_feature_engineering.py first.")
        return None

    model_key = best_model(results)
    threshold = results[model_key]["threshold_max_recall"]
    features = json.loads((RESULTS_DIR / "prep_meta.json").read_text())["features"]
    medians = json.loads((RESULTS_DIR / "train_medians.json").read_text())

    alerts, X = build_alert_features(features, medians)
    stage2_attack = attack_scores(model_key, X) >= threshold
    packet_label = alerts["packet_label"].astype(str).values

    stage1 = stage1_entry["test"]
    c = combined_counts(stage1, packet_label, stage2_attack)
    m = metrics(c)
    fp1 = stage1["fp"]
    fp_reduction = round(100 * (fp1 - m["fp"]) / fp1, 2) if fp1 else 0.0

    out = {
        "stage": "two_stage_system",
        "stage1_detector": "autoencoder",
        "stage2_detector": model_key,
        "combined": m,
        "stage1_recall": round(stage1["recall"], 4),
        "stage1_fp": fp1,
        "fp_reduction_pct_vs_phase2": fp_reduction,
        "alerts_unmapped": c["unmapped"],
    }
    save_result("two_stage_system", out)

    name = NICE.get(model_key, model_key)
    print("\n" + "=" * 78)
    print(f"  FINAL TWO-STAGE SYSTEM  (stage 1 autoencoder -> stage 2 {name})")
    print("=" * 78)
    print(f"  Overall accuracy : {m['accuracy'] * 100:.2f}%")
    print(f"  Precision        : {m['precision'] * 100:.2f}%     Recall: {m['recall'] * 100:.2f}%"
          f"     F1: {m['f1'] * 100:.2f}%")
    print(f"  Combined FPR     : {m['fpr'] * 100:.2f}%")
    print(f"  Confusion (attack=+): TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  False positives  : {fp1} (stage 1 alone) -> {m['fp']} (after stage 2)"
          f"   = {fp_reduction:.2f}% reduction")
    print(f"  Recall           : {stage1['recall'] * 100:.2f}% (stage 1) -> "
          f"{m['recall'] * 100:.2f}% (two-stage)")
    if c["unmapped"]:
        print(f"  Note: {c['unmapped']} alert(s) had no matching flow and kept stage 1's decision.")
    print("=" * 78)
    return out


if __name__ == "__main__":
    run()
