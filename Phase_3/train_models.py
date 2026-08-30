import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from evaluation import OUT, Timer, evaluate_and_save, load_splits

ANOMALY_COLS = ["flow_anomaly_score", "flow_anomaly_flag", "has_anomaly_score"]

TAG = ""


def key(name):
    return f"{name}{TAG}"


def model_path(stem):
    return f"{OUT}/{stem}{TAG}.joblib"


def drop_anomaly_columns(d):
    features = json.loads(open(f"{OUT}/prep_meta.json").read())["features"]
    keep = [i for i, c in enumerate(features) if c not in ANOMALY_COLS]
    dropped = [c for c in features if c in ANOMALY_COLS]
    print(f"dropping {len(dropped)} Task 3.1 columns: {', '.join(dropped)}")
    out = {k: d[k] for k in d.files}
    for k in ["X_tr", "X_va", "X_te"]:
        out[k] = out[k][:, keep]
    return out


def val_halves(d, seed=1):
    idx = np.arange(len(d["y_va"]))
    return train_test_split(idx, test_size=0.5, random_state=seed,
                            stratify=d["t_va"])


def train_rf(d):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, max_features=0.3,
        min_samples_split=2, min_samples_leaf=2,
        class_weight="balanced", bootstrap=True, n_jobs=-1, random_state=1,
    )
    with Timer() as t:
        model.fit(d["X_tr"], d["y_tr"])
    va = model.predict_proba(d["X_va"])[:, 1]
    te = model.predict_proba(d["X_te"])[:, 1]
    evaluate_and_save(key("random_forest"), d["y_va"], va, d["y_te"], te, d["t_te"], t.dt)
    joblib.dump(model, model_path("model_rf"), compress=3)


def train_xgb(d):
    spw = float((d["y_tr"] == 0).sum() / (d["y_tr"] == 1).sum())
    model = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_lambda=1.0, scale_pos_weight=spw, tree_method="hist",
        eval_metric="aucpr", n_jobs=-1, random_state=1,
    )
    with Timer() as t:
        model.fit(d["X_tr"], d["y_tr"])
    va = model.predict_proba(d["X_va"])[:, 1]
    te = model.predict_proba(d["X_te"])[:, 1]
    evaluate_and_save(key("xgboost"), d["y_va"], va, d["y_te"], te, d["t_te"], t.dt)
    joblib.dump(model, model_path("model_xgb"), compress=3)


def train_svm(d):
    fit_i, thr_i = val_halves(d)
    Xf, yf = d["X_va"][fit_i], d["y_va"][fit_i]
    best, best_ap = None, -1
    with Timer() as t:
        for C in [0.01, 0.1, 1.0]:
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("svm", LinearSVC(C=C, class_weight="balanced", dual=False, random_state=1)),
            ])
            pipe.fit(d["X_tr"], d["y_tr"])
            ap = average_precision_score(yf, pipe.decision_function(Xf))
            print(f"C={C}: val PR-AUC={ap:.4f}")
            if ap > best_ap:
                best, best_ap = pipe, ap
        cal = LogisticRegression()
        cal.fit(best.decision_function(Xf).reshape(-1, 1), yf)

    def score(X):
        return cal.predict_proba(best.decision_function(X).reshape(-1, 1))[:, 1]

    va = score(d["X_va"][thr_i])
    te = score(d["X_te"])
    evaluate_and_save(key("linear_svm"), d["y_va"][thr_i], va, d["y_te"], te, d["t_te"], t.dt)
    joblib.dump({"svm": best, "cal": cal}, model_path("model_svm"), compress=3)


def train_mlp(d):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
            max_iter=60, early_stopping=True, n_iter_no_change=8,
            validation_fraction=0.1, random_state=1,
        )),
    ])
    with Timer() as t:
        model.fit(d["X_tr"], d["y_tr"])
    va = model.predict_proba(d["X_va"])[:, 1]
    te = model.predict_proba(d["X_te"])[:, 1]
    evaluate_and_save(key("mlp"), d["y_va"], va, d["y_te"], te, d["t_te"], t.dt)
    joblib.dump(model, model_path("model_mlp"), compress=3)


def base_model_scores(d):
    rf = joblib.load(model_path("model_rf"))
    xgb = joblib.load(model_path("model_xgb"))
    mlp = joblib.load(model_path("model_mlp"))

    def probs(X):
        return np.column_stack([
            rf.predict_proba(X)[:, 1],
            xgb.predict_proba(X)[:, 1],
            mlp.predict_proba(X)[:, 1],
        ])
    return probs(d["X_va"]), probs(d["X_te"])


def train_ensemble(d):
    with Timer() as t:
        P_va, P_te = base_model_scores(d)
    va_vote, te_vote = P_va.mean(axis=1), P_te.mean(axis=1)
    evaluate_and_save(key("voting_soft"), d["y_va"], va_vote, d["y_te"], te_vote, d["t_te"], t.dt)


TRAINERS = {
    "rf": train_rf,
    "xgb": train_xgb,
    "svm": train_svm,
    "mlp": train_mlp,
    "ensemble": train_ensemble,
}


def main():
    ap = argparse.ArgumentParser(description="Phase 3 flow-level classifiers")
    ap.add_argument("--models", nargs="+", default=["all"],
                    choices=list(TRAINERS) + ["all"])
    ap.add_argument("--drop-anomaly", action="store_true",
                    help="retrain without the Task 3.1 columns, into __no_anomaly keys")
    args = ap.parse_args()

    names = list(TRAINERS) if args.models == ["all"] else [
        n for n in TRAINERS if n in args.models
    ]
    d = load_splits()
    if args.drop_anomaly:
        global TAG
        TAG = "__no_anomaly"
        d = drop_anomaly_columns(d)
    for name in names:
        print(f"\n=== {name} ===")
        TRAINERS[name](d)


if __name__ == "__main__":
    main()
