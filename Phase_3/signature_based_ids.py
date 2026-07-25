import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.utils.class_weight import compute_sample_weight

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

FLOW_DATA_DIR = PROJECT_ROOT / "data" / "processed_data" / "flow-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_data" / "task_33_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 1

# Label mapping for multi-class classification
LABEL_MAP = {
    'benign': 0,
    'dos': 1,
    'brute_force': 2,
    'ddos': 3,
    'dns_spoofing': 4,
    'xss': 5
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_supervised_flow_data(task_31_scores_file=None):
    # Load normalized data set
    data_path = FLOW_DATA_DIR / "normalized_original_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing supervised data file at {data_path}")

    df = pd.read_csv(data_path)

    # Optional Task 3.1 Feature Integration (Ablation Test)
    if task_31_scores_file and Path(task_31_scores_file).exists():
        print(f"[task-3.3] Merging Task 3.1 flow anomaly features from {task_31_scores_file}")
        t31_df = pd.read_csv(task_31_scores_file)
        df = df.merge(t31_df, on="Flow ID", how="left")
        df = df.fillna(0)

    # Isolate strictly statistical feature columns (excluding identifiers and target label)
    feature_cols = [c for c in df.columns if c not in ["Flow ID", "label"]]
    X = df[feature_cols]
    y = df["label"].map(LABEL_MAP)

    print(f"[task-3.3] Dataset loaded: {X.shape[0]:,} rows across {len(feature_cols)} features.")
    return df, X, y, feature_cols


def train_xgboost_classifier(X_train, y_train, X_val, y_val):
    sample_weights = compute_sample_weight("balanced", y_train)

    # Define hyperparameter grid for tuning
    max_depths = [4, 6, 8]
    gammas = [0.0, 0.1, 0.3]

    best_score = float("inf")
    best_model = None
    best_params = {}

    print("[task-3.3] Tuning XGBoost hyperparameters (gamma, max_depth)...")
    start_time = time.time()

    for depth in max_depths:
        for g in gammas:
            print(f"Testing max_depth={depth}, gamma={g}...")
            
            candidate_model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=len(LABEL_MAP),
                n_estimators=300,
                max_depth=depth,
                gamma=g,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss",
                early_stopping_rounds=20
            )

            candidate_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Get optimal score achieved on validation set
            val_loss = candidate_model.best_score if hasattr(candidate_model, "best_score") else float("inf")

            if val_loss < best_score:
                best_score = val_loss
                best_model = candidate_model
                best_params = {'max_depth': depth, 'gamma': g}

    train_duration = time.time() - start_time
    print(f"[task-3.3] Tuning completed in {train_duration:.2f} seconds.")
    print(f"[task-3.3] Optimal Parameters Found: max_depth={best_params['max_depth']}, gamma={best_params['gamma']} (Validation Loss: {best_score:.4f})")
    
    return best_model


def evaluate_supervised_stage(model, X_test, y_test):
    # Evaluate model
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    inference_time = time.time() - start_time

    latency_per_sample = (inference_time / len(X_test)) * 1000
    print(f"XGBoost Evaluation")
    print(f"Inference Latency: {latency_per_sample:.4f} ms/sample")

    target_names = [INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Task 3.3 Supervised XGBoost Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgboost_confusion_matrix.png", dpi=300)
    plt.close()

    # Save Multi-Class ROC Curves
    plt.figure(figsize=(9, 7))
    for i, class_name in enumerate(target_names):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task 3.3 XGBoost Multi-Class ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgboost_roc_curves.png", dpi=300)
    plt.close()


def reclassify_phase2_alerts(model, feature_cols, task_31_scores_file=None):
    # Re-evaluate phase 2 alerts
    alerts_file = FLOW_DATA_DIR / "phase2_alert_corresponding_flows.csv"
    if not alerts_file.exists():
        raise FileNotFoundError(f"Missing alert correspondence file at {alerts_file}")

    print(f"Alert Reclassification")
    alerts_df = pd.read_csv(alerts_file)
    print(f"Loaded {len(alerts_df):,} Phase 2 alert flow records for re-evaluation.")

    # Optional Task 3.1 Feature Integration for Alerts
    if task_31_scores_file and Path(task_31_scores_file).exists():
        t31_df = pd.read_csv(task_31_scores_file)
        alerts_df = alerts_df.merge(t31_df, on="Flow ID", how="left").fillna(0)

    X_alerts_raw = alerts_df[feature_cols].copy()

    # Clean infinite or NaN values in raw flow records
    X_alerts_raw = X_alerts_raw.replace([np.inf, -np.inf], np.nan)
    X_alerts_raw = X_alerts_raw.fillna(X_alerts_raw.mean())

    # Apply preprocessing pipeline used in Task 3.2
    X_alerts_log = np.log1p(X_alerts_raw.clip(lower=0))
    scaler = StandardScaler()
    X_alerts_scaled = scaler.fit_transform(X_alerts_log)

    # Perform Stage 2 Re-Classification
    start_time = time.time()
    alerts_df["stage2_pred_code"] = model.predict(X_alerts_scaled)
    alerts_df["stage2_pred_label"] = alerts_df["stage2_pred_code"].map(INV_LABEL_MAP)
    inference_duration = time.time() - start_time

    # Calculate False Positive Reduction & Attack Retention Metrics
    benign_phase2_fps = alerts_df[alerts_df["packet_label"] == "benign"]
    attack_phase2_alerts = alerts_df[alerts_df["packet_label"] != "benign"]

    total_fps = len(benign_phase2_fps)
    if total_fps > 0:
        cleared_fps = (benign_phase2_fps["stage2_pred_label"] == "benign").sum()
        fp_reduction_pct = (cleared_fps / total_fps) * 100.0
        print(f"Phase 2 Benign False Positives Flagged: {total_fps:,}")
        print(f"False Positives Cleared by Stage 2 XGBoost: {cleared_fps:,}")
        print(f"--> False Positive Reduction Rate: {fp_reduction_pct:.2f}%")
    else:
        print("No benign false positives found in the Phase 2 alert set.")

    if len(attack_phase2_alerts) > 0:
        retained_attacks = (attack_phase2_alerts["stage2_pred_label"] != "benign").sum()
        retention_rate = (retained_attacks / len(attack_phase2_alerts)) * 100.0
        print(f"True Attacks Retained by Stage 2: {retained_attacks:,}/{len(attack_phase2_alerts):,} ({retention_rate:.2f}%)")

    # Export results
    output_path = OUTPUT_DIR / "reclassified_phase2_alerts.csv"
    alerts_df.to_csv(output_path, index=False)
    print(f"Saved complete re-classification breakdown to {output_path}")


def run_experiment(task_31_scores_file, run_name):
    print(f"Model Version: {run_name.upper()}")

    # Load preprocessed flow dataset
    df, X, y, feature_cols = load_supervised_flow_data(task_31_scores_file)

    # Perform Stratified Train / Validation / Test Split (60% Train, 20% Val, 20% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    # Train XGBoost Model with Hyperparameter Tuning
    model = train_xgboost_classifier(X_train, y_train, X_val, y_val)

    # Evaluate Supervised Model Performance
    evaluate_supervised_stage(model, X_test, y_test)

    # Re-Classify Phase 2 Alerts to Compute False Positive Reduction Rate
    reclassify_phase2_alerts(model, feature_cols, task_31_scores_file)


def main():
    task_31_scores_file = FLOW_DATA_DIR / "task3_1_flow_anomaly_scores.csv" 

    # Baseline that doesn't use the 3.1 features
    run_experiment(task_31_scores_file=None, run_name="Baseline (No Task 3.1 Features)")

    # Augmented with the 3.1 features
    run_experiment(task_31_scores_file=task_31_scores_file, run_name="Augmented Model (With Task 3.1 Features)")


if __name__ == "__main__":
    main()