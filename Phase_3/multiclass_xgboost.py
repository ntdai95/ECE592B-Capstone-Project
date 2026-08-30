import json
import time

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight

from evaluation import RESULTS_DIR, PROJECT_ROOT, Timer, evaluate_and_save, load_splits

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_data" / "task_33_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 1

LABEL_MAP = {
    'benign': 0,
    'dos': 1,
    'brute_force': 2,
    'ddos': 3,
    'dns_spoofing': 4,
    'xss': 5
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

ANOMALY_COLS = ["flow_anomaly_score", "flow_anomaly_flag", "has_anomaly_score"]


def load_supervised_flow_data(drop_anomaly=False):
    d = load_splits()
    feature_cols = json.loads((RESULTS_DIR / "prep_meta.json").read_text())["features"]
    keep = [i for i, c in enumerate(feature_cols) if c not in ANOMALY_COLS]
    if drop_anomaly:
        feature_cols = [feature_cols[i] for i in keep]

    def feats(a):
        return a[:, keep] if drop_anomaly else a

    def y_multi(t):
        return np.array([LABEL_MAP[c] for c in t])

    data = {
        "X_train": feats(d["X_tr"]), "y_train": y_multi(d["t_tr"]),
        "X_val": feats(d["X_va"]), "y_val": y_multi(d["t_va"]),
        "X_test": feats(d["X_te"]), "y_test": y_multi(d["t_te"]),
        "y_va_bin": d["y_va"], "y_te_bin": d["y_te"], "t_te": d["t_te"],
    }
    print(f"[task-3.3] Split loaded: {len(data['X_train']):,} train / "
          f"{len(data['X_val']):,} val / {len(data['X_test']):,} test "
          f"across {len(feature_cols)} features.")
    return data


def xgb_classifier(depth, gamma):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABEL_MAP),
        n_estimators=300,
        max_depth=depth,
        gamma=gamma,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
    )


def train_xgboost_classifier(X_train, y_train, X_val, y_val):
    sample_weights = compute_sample_weight("balanced", y_train)

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
            candidate_model = xgb_classifier(depth, g)
            candidate_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_loss = candidate_model.best_score if hasattr(candidate_model, "best_score") else float("inf")
            if val_loss < best_score:
                best_score = val_loss
                best_model = candidate_model
                best_params = {'max_depth': depth, 'gamma': g}

    print(f"[task-3.3] Tuning completed in {time.time() - start_time:.2f} seconds.")
    print(f"[task-3.3] Optimal Parameters Found: max_depth={best_params['max_depth']}, "
          f"gamma={best_params['gamma']} (Validation Loss: {best_score:.4f})")
    return best_model


def evaluate_supervised_stage(model, X_test, y_test, tag):
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print("XGBoost Evaluation")
    print(f"Inference Latency: {(time.time() - start_time) / len(X_test) * 1000:.4f} ms/sample")

    target_names = [INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Task 3.3 Supervised XGBoost Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"multiclass_xgb_confusion_matrix{tag}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 7))
    for i, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task 3.3 XGBoost Multi-Class ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"multiclass_xgb_roc_curves{tag}.png", dpi=300)
    plt.close()


def run_experiment(drop_anomaly, run_name):
    print(f"Model Version: {run_name.upper()}")
    tag = "__no_anomaly" if drop_anomaly else ""

    data = load_supervised_flow_data(drop_anomaly)

    with Timer() as t:
        model = train_xgboost_classifier(data["X_train"], data["y_train"],
                                         data["X_val"], data["y_val"])

    va = 1 - model.predict_proba(data["X_val"])[:, LABEL_MAP["benign"]]
    te = 1 - model.predict_proba(data["X_test"])[:, LABEL_MAP["benign"]]
    evaluate_and_save(f"multiclass_xgb{tag}", data["y_va_bin"], va,
                      data["y_te_bin"], te, data["t_te"], t.dt)

    evaluate_supervised_stage(model, data["X_test"], data["y_test"], tag)
    joblib.dump(model, f"{RESULTS_DIR}/model_multiclass{tag}.joblib", compress=3)


def main():
    run_experiment(drop_anomaly=False, run_name="Full feature set (with Task 3.1 features)")
    run_experiment(drop_anomaly=True, run_name="Without Task 3.1 features")


if __name__ == "__main__":
    main()
