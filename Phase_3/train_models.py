"""Train the Phase 3 flow-level classifiers.

Every model is scored through the shared operating-point rule in common.py, so
no model can grade itself on a different threshold. Results accumulate into one
results.json keyed by model name.

Order matters: the ensembles read the four fitted base models off disk, and the
blend reads the binary XGBoost, so `--models all` runs them in dependency order.

    python Phase_3/train_models.py --models all
    python Phase_3/train_models.py --models rf xgb
    python Phase_3/train_models.py --drop-anomaly

--drop-anomaly retrains the same eight models with the three Task 3.1 columns
removed and nothing else changed, saving under "<model>__no_anomaly" keys so the
contribution of stage one is measured per model rather than argued from XGBoost
alone. The joblib filenames are suffixed too, so an ablation run cannot
overwrite the models the leaderboard and the audits read.
"""
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

from common import OUT, Timer, evaluate_and_save, load_splits

CLASSES = ["benign", "brute_force", "ddos", "dns_spoofing", "dos", "xss"]

ANOMALY_COLS = ["flow_anomaly_score", "flow_anomaly_flag", "has_anomaly_score"]

TAG = ""


def key(name):
    return f"{name}{TAG}"


def model_path(stem):
    return f"{OUT}/{stem}{TAG}.joblib"


def drop_anomaly_columns(d):
    """Return the splits with the three Task 3.1 columns removed.

    Column order in splits.npz matches prep_meta.json["features"], so the
    indices are taken from there rather than assumed.
    """
    features = json.loads(open(f"{OUT}/prep_meta.json").read())["features"]
    keep = [i for i, c in enumerate(features) if c not in ANOMALY_COLS]
    dropped = [c for c in features if c in ANOMALY_COLS]
    print(f"dropping {len(dropped)} Task 3.1 columns: {', '.join(dropped)}")
    out = {k: d[k] for k in d.files}
    for k in ["X_tr", "X_va", "X_te"]:
        out[k] = out[k][:, keep]
    return out


def val_halves(d, seed=1):
    """Split validation into a fitting half and a threshold half.

    Anything fitted on the validation set - the SVM calibrator, the stacking
    meta-learner - makes its own validation scores in-sample. A threshold read
    off those scores sits below the true operating point and the FPR budget can
    then overrun on test. Keeping the two roles on disjoint halves removes that.
    """
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
    """Light C sweep, then Platt-style sigmoid calibration of decision_function.

    The C sweep and the calibrator both fit on the first validation half; the
    threshold is read off the second, which neither has seen.
    """
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
    svm = joblib.load(model_path("model_svm"))

    def probs(X):
        s = svm["cal"].predict_proba(svm["svm"].decision_function(X).reshape(-1, 1))[:, 1]
        return np.column_stack([
            rf.predict_proba(X)[:, 1],
            xgb.predict_proba(X)[:, 1],
            mlp.predict_proba(X)[:, 1],
            s,
        ])
    return probs(d["X_va"]), probs(d["X_te"])


def train_ensemble(d):
    """voting_soft averages the probabilities of RF, XGB and MLP, the strongest
    three. stacking_lr fits a logistic-regression meta-learner on the
    validation-set predictions of all four.

    voting_soft fits nothing on validation, so it keeps the whole set. The
    meta-learner fits on the first half and is thresholded on the second.
    """
    with Timer() as t:
        P_va, P_te = base_model_scores(d)
    va_vote, te_vote = P_va[:, :3].mean(axis=1), P_te[:, :3].mean(axis=1)
    evaluate_and_save(key("voting_soft"), d["y_va"], va_vote, d["y_te"], te_vote, d["t_te"], t.dt)
    fit_i, thr_i = val_halves(d)
    meta = LogisticRegression(class_weight="balanced")
    meta.fit(P_va[fit_i], d["y_va"][fit_i])
    va_st = meta.predict_proba(P_va[thr_i])[:, 1]
    te_st = meta.predict_proba(P_te)[:, 1]
    evaluate_and_save(key("stacking_lr"), d["y_va"][thr_i], va_st, d["y_te"], te_st, d["t_te"], t.dt)
    joblib.dump(meta, model_path("model_stack_meta"), compress=3)


def train_multiclass(d):
    """Multiclass XGBoost (6-way softmax) plus an equal-weight blend with the
    shared binary model.

    A binary attack/benign objective satisfies most of its loss on the two
    volumetric classes (DDoS, DoS) and leaves the boundary for the hard classes
    loosely fitted. A 6-way softmax forces class-discriminative structure; the
    attack score is then 1 - P(benign). Per-class AUC improves for every hard
    class (brute force .962->.969, DNS spoofing .894->.908, XSS .947->.960), and
    averaging the two scores gives the same recall as the binary model for fewer
    false positives.
    """
    cmap = {c: i for i, c in enumerate(CLASSES)}
    y_tr_m = np.array([cmap[c] for c in d["t_tr"]])

    w = np.ones(len(y_tr_m))
    w[y_tr_m > 0] = float((y_tr_m == 0).sum() / (y_tr_m > 0).sum())

    model = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        reg_lambda=1.0, tree_method="hist",
        objective="multi:softprob", num_class=len(CLASSES),
        n_jobs=-1, random_state=1,
    )
    with Timer() as t:
        model.fit(d["X_tr"], y_tr_m, sample_weight=w)

    P_va = model.predict_proba(d["X_va"])
    P_te = model.predict_proba(d["X_te"])
    va, te = 1 - P_va[:, 0], 1 - P_te[:, 0]
    evaluate_and_save(key("multiclass_xgb"), d["y_va"], va, d["y_te"], te, d["t_te"], t.dt)
    joblib.dump(model, model_path("model_multiclass"), compress=3)

    xgb = joblib.load(model_path("model_xgb"))
    bva = 0.5 * va + 0.5 * xgb.predict_proba(d["X_va"])[:, 1]
    bte = 0.5 * te + 0.5 * xgb.predict_proba(d["X_te"])[:, 1]
    evaluate_and_save(key("blend_binary_multiclass"), d["y_va"], bva, d["y_te"], bte, d["t_te"], t.dt)


TRAINERS = {
    "rf": train_rf,
    "xgb": train_xgb,
    "svm": train_svm,
    "mlp": train_mlp,
    "ensemble": train_ensemble,
    "multiclass": train_multiclass,
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
