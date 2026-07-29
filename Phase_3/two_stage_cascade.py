"""
Run the two-stage IDS cascade on the seed-1 packet/flow samples.

Stage 1: Isolation Forest on packet features (Phase 2)
Stage 2: XGBoost on flows linked from Phase-2 alerts (Phase 3)
Final alert = Phase-2 alert AND Phase-3 attack prediction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(Path(__file__).resolve().parent))

from flow_preprocessing import (
    ATTACK_COL,
    BINARY_COL,
    FLOW_KEY_COL,
    TARGET_COL,
    build_flow_preprocessor,
)
from packet_flow_link import add_packet_flow_ids, load_flows_for_ids, resolve_packet_flow_ids


ATTACK_COL_PACKET = "attack_type"
BINARY_COL_PACKET = "binary_label"


def evaluate_binary(y_true, y_pred, scores=None, name="model"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alerts": int(y_pred.sum()),
    }
    if scores is not None:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["pr_auc"] = float(average_precision_score(y_true, scores))
    return result


def build_packet_features(df: pd.DataFrame) -> pd.DataFrame:
    base_features = [
        "time_since_previously_displayed_frame",
        "port_class_dst",
        "l4_tcp",
        "l4_udp",
        "ttl",
        "eth_size",
        "tcp_window_size",
        "payload_entropy",
        "payload_length",
        "jitter",
        "dns_len_qry",
        "dns_interval",
        "dns_len_ans",
        "http_content_len",
        "http_response_code",
        "handshake_cipher_suites_length",
        "handshake_extensions_length",
        "handshake_sig_hash_alg_len",
        "icmp_data_size",
        "l3_ip_dst_count",
        "average_p",
        "var_p",
        "iqr_p",
    ]

    rolling_features = []
    for group in ["stream", "src_ip", "channel"]:
        rolling_features += [f"{group}_{window}_count" for window in [1, 5, 30]]
        rolling_features += [
            f"{group}_{window}_{stat}" for window in [5, 30] for stat in ["mean", "var"]
        ]
    rolling_features += [
        "stream_jitter_5_mean",
        "stream_jitter_5_var",
        "stream_jitter_30_mean",
        "stream_jitter_30_var",
    ]

    feature_cols = [col for col in base_features + rolling_features if col in df.columns]
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").copy()

    src_port = pd.to_numeric(df["src_port"], errors="coerce").fillna(-1)
    dst_port = pd.to_numeric(df["dst_port"], errors="coerce").fillna(-1)
    packet_size = pd.to_numeric(df["eth_size"], errors="coerce").replace(0, np.nan)
    payload_size = pd.to_numeric(df["payload_length"], errors="coerce").fillna(0)
    dns_query = pd.to_numeric(df["dns_len_qry"], errors="coerce").fillna(0)
    dns_answer = pd.to_numeric(df["dns_len_ans"], errors="coerce").fillna(0)
    http_length = pd.to_numeric(df["http_content_len"], errors="coerce").fillna(0)
    http_code = pd.to_numeric(df["http_response_code"], errors="coerce").fillna(0)
    tls_length = pd.to_numeric(df["handshake_extensions_length"], errors="coerce").fillna(0)
    icmp_size = pd.to_numeric(df["icmp_data_size"], errors="coerce").fillna(-1)

    X_raw["payload_ratio"] = (payload_size / packet_size).clip(0, 1).fillna(0)
    X_raw["has_payload"] = (payload_size > 0).astype(int)
    X_raw["uses_dns_port"] = ((src_port == 53) | (dst_port == 53)).astype(int)
    X_raw["has_dns_data"] = ((dns_query > 0) | (dns_answer > 0)).astype(int)
    X_raw["uses_http_port"] = ((src_port == 80) | (dst_port == 80)).astype(int)
    X_raw["uses_https_port"] = ((src_port == 443) | (dst_port == 443)).astype(int)
    X_raw["has_http_data"] = ((http_length > 0) | (http_code > 0)).astype(int)
    X_raw["has_tls_handshake"] = (tls_length > 0).astype(int)
    X_raw["is_icmp"] = (icmp_size >= 0).astype(int)
    X_raw["src_port_ephemeral"] = (src_port >= 49152).astype(int)
    X_raw["dst_port_ephemeral"] = (dst_port >= 49152).astype(int)

    def burst_ratio(short_col, long_col):
        short_count = pd.to_numeric(df[short_col], errors="coerce").fillna(0)
        long_count = pd.to_numeric(df[long_col], errors="coerce").fillna(0)
        recent_rate = short_count / 5
        previous_rate = (long_count - short_count).clip(lower=0) / 25
        return (recent_rate + 1) / (previous_rate + 1)

    for group in ["stream", "src_ip", "channel"]:
        short = f"{group}_5_count"
        long = f"{group}_30_count"
        if short in df.columns and long in df.columns:
            X_raw[f"{group}_burst_ratio"] = burst_ratio(short, long)

    return X_raw.replace([np.inf, -np.inf], np.nan)


def preprocess_packet_splits(X_raw, train_idx, val_idx, test_idx):
    X_clean = X_raw.copy()
    zero_fill_cols = [col for col in X_clean.columns if col.endswith("_var") or "jitter" in col]
    X_clean[zero_fill_cols] = X_clean[zero_fill_cols].fillna(0)

    X_train_df = X_clean.iloc[train_idx].copy()
    X_val_df = X_clean.iloc[val_idx].copy()
    X_test_df = X_clean.iloc[test_idx].copy()

    drop_cols = [
        col
        for col in X_train_df.columns
        if X_train_df[col].isna().all() or X_train_df[col].nunique(dropna=True) <= 1
    ]
    X_train_df = X_train_df.drop(columns=drop_cols)
    X_val_df = X_val_df.drop(columns=drop_cols)
    X_test_df = X_test_df.drop(columns=drop_cols)

    log_cols = []
    for col in X_train_df.columns:
        values = X_train_df[col].dropna()
        if len(values) > 0 and values.min() >= 0 and values.nunique() > 2 and abs(values.skew()) > 2:
            log_cols.append(col)
    for split in (X_train_df, X_val_df, X_test_df):
        split[log_cols] = np.log1p(split[log_cols].clip(lower=0))

    correlation = X_train_df.corr(numeric_only=True).abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    correlated_cols = [col for col in upper.columns if (upper[col] > 0.98).any()]
    X_train_df = X_train_df.drop(columns=correlated_cols)
    X_val_df = X_val_df.drop(columns=correlated_cols)
    X_test_df = X_test_df.drop(columns=correlated_cols)

    preprocess = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    X_train = preprocess.fit_transform(X_train_df).astype(np.float32)
    X_val = preprocess.transform(X_val_df).astype(np.float32)
    X_test = preprocess.transform(X_test_df).astype(np.float32)
    return X_train, X_val, X_test, preprocess, list(X_train_df.columns)


def select_threshold(y_val, scores, max_fpr=0.015):
    """Pick score threshold maximizing validation F1 with FPR <= max_fpr."""
    # Higher anomaly score => more anomalous for our inverted IsolationForest score.
    candidates = np.unique(np.quantile(scores, np.linspace(0.90, 0.999, 40)))
    best = None
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 1.0
        if fpr > max_fpr:
            continue
        f1 = f1_score(y_val, pred, zero_division=0)
        row = {"threshold": float(threshold), "f1": float(f1), "fpr": float(fpr)}
        if best is None or row["f1"] > best["f1"]:
            best = row
    if best is None:
        # Fallback: top 2% alert rate
        threshold = float(np.quantile(scores, 0.98))
        best = {"threshold": threshold, "f1": None, "fpr": None}
    return best


def train_phase2(packet_df, seed=1):
    y_true = (packet_df[BINARY_COL_PACKET] == "attack").astype(int).to_numpy()
    X_raw = build_packet_features(packet_df)

    all_idx = np.arange(len(packet_df))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.40, random_state=seed, stratify=packet_df[ATTACK_COL_PACKET]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=packet_df.iloc[temp_idx][ATTACK_COL_PACKET]
    )

    X_train, X_val, X_test, preprocess, feature_names = preprocess_packet_splits(
        X_raw, train_idx, val_idx, test_idx
    )

    model = IsolationForest(
        n_estimators=400,
        max_samples=4096,
        max_features=0.5,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    start = time.perf_counter()
    model.fit(X_train)
    train_time = time.perf_counter() - start

    # decision_function: larger => more normal. Invert so larger => more anomalous.
    val_scores = -model.decision_function(X_val)
    test_scores = -model.decision_function(X_test)
    threshold_info = select_threshold(y_true[val_idx], val_scores)
    threshold = threshold_info["threshold"]

    val_pred = (val_scores >= threshold).astype(int)
    test_pred = (test_scores >= threshold).astype(int)

    return {
        "model": model,
        "preprocess": preprocess,
        "feature_names": feature_names,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "y_true": y_true,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "train_time": train_time,
    }


def train_phase3_xgboost(flow_dir: Path, seed=1):
    meta = json.loads((flow_dir / "flow_split_metadata.json").read_text())
    feature_cols = meta["feature_columns"]

    train_df = pd.read_csv(flow_dir / "flow_train.csv", low_memory=False)
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].to_numpy()

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    best_path = PROJECT_ROOT / "reports" / "phase3" / "xgboost_best_parameters.json"
    if best_path.exists():
        best_params = json.loads(best_path.read_text())["best_params"]
        model_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    else:
        model_params = {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "reg_lambda": 0.5,
            "gamma": 0.0,
        }

    pipeline = Pipeline(
        [
            ("preprocessor", build_flow_preprocessor(scale=False)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=seed,
                    n_jobs=-1,
                    **model_params,
                ),
            ),
        ]
    )

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return pipeline, feature_cols, train_time, model_params


def per_attack_detection(packet_df, idx, y_true, y_pred):
    subset = packet_df.iloc[idx][[ATTACK_COL_PACKET]].copy()
    subset["y_true"] = y_true[idx]
    subset["y_pred"] = y_pred
    rows = []
    for attack, group in subset.groupby(ATTACK_COL_PACKET):
        if attack == "Benign":
            continue
        total = int((group["y_true"] == 1).sum())
        detected = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
        rows.append(
            {
                "attack_type": attack,
                "attack_packets": total,
                "detected": detected,
                "detection_rate": detected / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("attack_type")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-sample", default="data/processed_data/samples/packet_sample_seed1.csv")
    parser.add_argument("--flow-raw-dir", default="data/raw_data")
    parser.add_argument("--flow-processed-dir", default="data/processed_data/phase3")
    parser.add_argument("--output-dir", default="reports/cascade")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--phase3-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading packet sample...")
    packet_df = pd.read_csv(args.packet_sample, low_memory=False)
    packet_df = add_packet_flow_ids(packet_df)

    print("Training Phase 2 Isolation Forest...")
    phase2 = train_phase2(packet_df, seed=args.seed)
    p2_val = evaluate_binary(
        phase2["y_true"][phase2["val_idx"]],
        phase2["val_pred"],
        phase2["val_scores"],
        "Phase2-IF-Validation",
    )
    p2_test = evaluate_binary(
        phase2["y_true"][phase2["test_idx"]],
        phase2["test_pred"],
        phase2["test_scores"],
        "Phase2-IF-Test",
    )
    print("Phase 2 test:", {k: p2_test[k] for k in ["precision", "recall", "f1", "fpr", "fp", "tp"]})

    print("Training Phase 3 XGBoost on flow splits...")
    xgb_pipeline, feature_cols, p3_train_time, model_params = train_phase3_xgboost(
        Path(args.flow_processed_dir), seed=args.seed
    )

    test_idx = phase2["test_idx"]
    test_packets = packet_df.iloc[test_idx].copy().reset_index(drop=True)
    test_packets["phase2_score"] = phase2["test_scores"]
    test_packets["phase2_alert"] = phase2["test_pred"]
    test_packets["y_true"] = phase2["y_true"][test_idx]

    alerted = test_packets[test_packets["phase2_alert"] == 1].copy()
    print(f"Phase 2 test alerts: {len(alerted):,}")

    candidate_ids = set(alerted["flow_id_fwd"]).union(set(alerted["flow_id_rev"]))
    print(f"Looking up {len(candidate_ids):,} candidate Flow IDs in raw flow files...")
    start_lookup = time.perf_counter()
    linked_flows = load_flows_for_ids(args.flow_raw_dir, candidate_ids)
    lookup_time = time.perf_counter() - start_lookup
    print(f"Linked aggregated flows: {len(linked_flows):,} (lookup {lookup_time:.1f}s)")

    available_ids = set(linked_flows["Flow ID"].astype(str)) if len(linked_flows) else set()
    alerted["resolved_flow_id"] = resolve_packet_flow_ids(alerted, available_ids)
    link_rate = float(alerted["resolved_flow_id"].notna().mean()) if len(alerted) else 0.0
    print(f"Alerted packets with matching flow: {link_rate:.1%}")

    if len(linked_flows):
        flow_features = linked_flows.set_index("Flow ID")
        # Ensure required feature columns exist
        for col in feature_cols:
            if col not in flow_features.columns:
                flow_features[col] = np.nan
        X_linked = flow_features.loc[flow_features.index.intersection(available_ids), feature_cols]
        flow_proba = pd.Series(
            xgb_pipeline.predict_proba(X_linked)[:, 1],
            index=X_linked.index,
            name="phase3_proba",
        )
        flow_pred = (flow_proba >= args.phase3_threshold).astype(int)
    else:
        flow_proba = pd.Series(dtype=float)
        flow_pred = pd.Series(dtype=int)

    alerted["phase3_proba"] = alerted["resolved_flow_id"].map(flow_proba)
    alerted["phase3_pred"] = alerted["resolved_flow_id"].map(flow_pred)

    # Cascade policy:
    # - no Phase-2 alert => benign
    # - Phase-2 alert + linked flow => Phase-3 decision
    # - Phase-2 alert + missing flow => keep Phase-2 alert (fail-open on unmatched)
    cascade_pred = np.zeros(len(test_packets), dtype=int)
    alert_positions = np.flatnonzero(test_packets["phase2_alert"].to_numpy() == 1)
    for local_i, packet_i in enumerate(alert_positions):
        row = alerted.iloc[local_i]
        if pd.isna(row["resolved_flow_id"]):
            cascade_pred[packet_i] = 1
        else:
            cascade_pred[packet_i] = int(row["phase3_pred"]) if not pd.isna(row["phase3_pred"]) else 1

    # Score for ROC: use phase3 proba when available else phase2 score scaled
    cascade_scores = test_packets["phase2_score"].to_numpy(dtype=float).copy()
    for local_i, packet_i in enumerate(alert_positions):
        row = alerted.iloc[local_i]
        if not pd.isna(row["phase3_proba"]):
            cascade_scores[packet_i] = float(row["phase3_proba"])

    y_test = test_packets["y_true"].to_numpy()
    combined = evaluate_binary(y_test, cascade_pred, cascade_scores, "TwoStage-Cascade-Test")
    phase2_only = evaluate_binary(
        y_test, test_packets["phase2_alert"].to_numpy(), test_packets["phase2_score"].to_numpy(), "Phase2-Only-Test"
    )

    fp_reduction = None
    if phase2_only["fp"] > 0:
        fp_reduction = (phase2_only["fp"] - combined["fp"]) / phase2_only["fp"] * 100.0

    attack_p2 = per_attack_detection(packet_df, test_idx, phase2["y_true"], phase2["test_pred"])
    attack_combined = per_attack_detection(packet_df, test_idx, phase2["y_true"], cascade_pred)
    attack_p2 = attack_p2.rename(columns={"detection_rate": "phase2_detection_rate", "detected": "phase2_detected"})
    attack_combined = attack_combined.rename(
        columns={"detection_rate": "cascade_detection_rate", "detected": "cascade_detected"}
    )
    attack_compare = attack_p2.merge(
        attack_combined[["attack_type", "cascade_detected", "cascade_detection_rate"]],
        on="attack_type",
        how="outer",
    )

    metrics_df = pd.DataFrame([p2_val, phase2_only, combined])
    summary = {
        "seed": args.seed,
        "packet_sample": args.packet_sample,
        "phase2_threshold": phase2["threshold"],
        "phase2_threshold_info": phase2["threshold_info"],
        "phase2_train_time_sec": phase2["train_time"],
        "phase3_train_time_sec": p3_train_time,
        "phase3_params": model_params,
        "phase3_threshold": args.phase3_threshold,
        "flow_lookup_time_sec": lookup_time,
        "phase2_test_alerts": int(len(alerted)),
        "unique_candidate_flow_ids": len(candidate_ids),
        "linked_flows": int(len(linked_flows)),
        "alert_flow_link_rate": link_rate,
        "false_positive_reduction_pct": fp_reduction,
        "phase2_test": phase2_only,
        "cascade_test": combined,
    }

    metrics_df.to_csv(output_dir / "cascade_metrics.csv", index=False)
    attack_compare.to_csv(output_dir / "cascade_per_attack.csv", index=False)
    (output_dir / "cascade_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    predictions = test_packets[
        [
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "flow_id_fwd",
            "flow_id_rev",
            ATTACK_COL_PACKET,
            BINARY_COL_PACKET,
            "y_true",
            "phase2_score",
            "phase2_alert",
        ]
    ].copy()
    predictions["cascade_pred"] = cascade_pred
    predictions["cascade_score"] = cascade_scores
    # attach resolved flow / phase3 where alerted
    alerted_out = alerted[
        ["flow_id_fwd", "resolved_flow_id", "phase3_proba", "phase3_pred"]
    ].reset_index(drop=True)
    alerted_out["test_row"] = alert_positions
    predictions = predictions.reset_index(drop=True)
    predictions.loc[alert_positions, "resolved_flow_id"] = alerted_out["resolved_flow_id"].to_numpy()
    predictions.loc[alert_positions, "phase3_proba"] = alerted_out["phase3_proba"].to_numpy()
    predictions.loc[alert_positions, "phase3_pred"] = alerted_out["phase3_pred"].to_numpy()
    predictions.to_csv(output_dir / "cascade_test_predictions.csv", index=False)

    joblib.dump(
        {
            "phase2_model": phase2["model"],
            "phase2_preprocess": phase2["preprocess"],
            "phase2_threshold": phase2["threshold"],
            "phase3_pipeline": xgb_pipeline,
            "phase3_feature_cols": feature_cols,
            "phase3_threshold": args.phase3_threshold,
        },
        output_dir / "cascade_models.joblib",
    )

    print("\n=== Cascade results (test) ===")
    print(f"Phase 2 only  FP={phase2_only['fp']}  precision={phase2_only['precision']:.4f}  recall={phase2_only['recall']:.4f}  f1={phase2_only['f1']:.4f}")
    print(f"Two-stage     FP={combined['fp']}  precision={combined['precision']:.4f}  recall={combined['recall']:.4f}  f1={combined['f1']:.4f}")
    if fp_reduction is not None:
        print(f"FP reduction: {fp_reduction:.1f}%")
    print(f"Alert→flow link rate: {link_rate:.1%}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
