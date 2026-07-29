"""Shared evaluation utilities for Phase 3.

Every training script imports its metrics and its threshold rule from here, so
the operating point is defined in one place and no model can score itself on a
different rule. Results for all models accumulate in a single results.json.

Operating point: the brief fixes a hard budget of FPR <= 1%. Choosing the
threshold by maximising F1 subject to that budget stops far short of the budget
under a 3% attack base rate and suppresses the hard classes, so the
Neyman-Pearson point is used instead: the lowest threshold whose validation FPR
still respects the budget. Both points are computed and saved.

The count of permitted validation false positives is sized by its
Clopper-Pearson 99% upper bound rather than budget * n_benign, so the budget
holds on test rather than only on the sample it was tuned on. Thresholds come
from the benign validation scores alone, which parameterises FPR directly.

Per-class ROC-AUC against benign is recorded alongside per-class detection
rate, since detection rate at a fixed FPR conflates ranking quality with base
rate.
"""
import json
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

FLOW_DATA_DIR = PROJECT_ROOT / "data" / "processed_data" / "flow-data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT = str(RESULTS_DIR)
RESULTS = RESULTS_DIR / "results.json"
FPR_BUDGET = 0.01


def load_splits():
    d = np.load(f"{OUT}/splits.npz", allow_pickle=True)
    return d


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {}


def save_result(name, res):
    all_res = load_results()
    all_res[name] = res
    RESULTS.write_text(json.dumps(all_res, indent=2))


def metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "fpr": round(fp / (fp + tn), 4),
        "fnr": round(fn / (fn + tp), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def safe_fp_allowance(n_benign, budget=FPR_BUDGET, confidence=0.99):
    """Largest number of validation false positives whose upper confidence bound
    on the true FPR still sits inside the budget.

    Taking the naive k = budget * n_benign fits the threshold to the validation
    benign tail and the budget then leaks on test. The Clopper-Pearson upper
    bound keeps the hard FPR <= 1% requirement intact on unseen data at the cost
    of a few tenths of a point of recall.
    """
    from scipy.stats import beta
    k = int(np.floor(budget * n_benign))
    while k > 0:
        ucb = beta.ppf(confidence, k + 1, n_benign - k)
        if ucb <= budget:
            return k
        k -= 1
    return 0


def thr_max_recall(y_val, val_score, budget=FPR_BUDGET):
    """Neyman-Pearson operating point: lowest threshold that keeps FPR within
    budget, with the allowance sized by its upper confidence bound."""
    ben = np.sort(val_score[y_val == 0])[::-1]
    k = safe_fp_allowance(len(ben), budget)
    if k < 1:
        return float(ben[0]) + 1.0
    return float(ben[k - 1]) + 1e-12


def thr_max_f1(y_val, val_score, budget=FPR_BUDGET):
    """Alternative rule kept for the trade-off discussion: maximise F1 subject
    to FPR <= budget."""
    ben = np.sort(val_score[y_val == 0])[::-1]
    best, best_f1 = float(ben[0]) + 1.0, -1.0
    for k in range(1, safe_fp_allowance(len(ben), budget) + 1):
        t = float(ben[k - 1]) + 1e-12
        pred = (val_score >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        if tp == 0:
            continue
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def per_attack(t_te, y_pred):
    out = {}
    for cls in np.unique(t_te):
        mask = t_te == cls
        if cls == "benign":
            out[cls] = round(float((y_pred[mask] == 0).mean()) * 100, 2)
        else:
            out[cls] = round(float((y_pred[mask] == 1).mean()) * 100, 2)
    return out


def per_class_auc(t_te, te_score):
    """ROC-AUC of each attack class against benign: ranking quality, free of
    base rate."""
    out = {}
    ben = t_te == "benign"
    for cls in np.unique(t_te):
        if cls == "benign":
            continue
        m = t_te == cls
        y = np.r_[np.zeros(int(ben.sum())), np.ones(int(m.sum()))]
        s = np.r_[te_score[ben], te_score[m]]
        out[cls] = round(float(roc_auc_score(y, s)), 4)
    return out


def detection_vs_budget(t_te, y_te, te_score, y_va, va_score,
                        budgets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)):
    """How each class's detection rate trades off against the FPR allowance.

    Every threshold comes from the validation scores under the same rule the
    headline operating point uses. Taking them from the test benign quantiles
    would be an oracle threshold and would report detection rates at an exactly
    achieved FPR that a deployed sensor could not reproduce. The achieved test
    FPR is recorded per row so the gap stays visible.
    """
    rows = {}
    for b in budgets:
        thr = thr_max_recall(y_va, va_score, b)
        pred = (te_score >= thr).astype(int)
        row = {c: round(float((pred[t_te == c] == 1).mean()) * 100, 2)
               for c in np.unique(t_te) if c != "benign"}
        row["overall_recall"] = round(float(pred[y_te == 1].mean()) * 100, 2)
        row["achieved_test_fpr"] = round(float(pred[y_te == 0].mean()) * 100, 3)
        rows[f"{b*100:g}%"] = row
    return rows


def evaluate_and_save(name, y_va, va_score, y_te, te_score, t_te, train_time):
    res = {"model": name, "train_time_sec": round(train_time, 1), "fpr_budget": FPR_BUDGET}

    res["test_default"] = metrics(y_te, (te_score >= 0.5).astype(int), te_score)
    res["per_attack_default"] = per_attack(t_te, (te_score >= 0.5).astype(int))

    thr_r = thr_max_recall(y_va, va_score)
    pred_r = (te_score >= thr_r).astype(int)
    res["threshold_max_recall"] = float(thr_r)
    res["test_tuned"] = metrics(y_te, pred_r, te_score)
    res["per_attack_tuned"] = per_attack(t_te, pred_r)

    thr_f = thr_max_f1(y_va, va_score)
    pred_f = (te_score >= thr_f).astype(int)
    res["threshold_max_f1"] = float(thr_f)
    res["test_maxf1"] = metrics(y_te, pred_f, te_score)
    res["per_attack_maxf1"] = per_attack(t_te, pred_f)

    res["val_default"] = metrics(y_va, (va_score >= 0.5).astype(int), va_score)
    res["per_class_auc"] = per_class_auc(t_te, te_score)
    res["detection_vs_budget"] = detection_vs_budget(t_te, y_te, te_score, y_va, va_score)

    save_result(name, res)
    t = res["test_tuned"]
    print(f"[{name}] max-recall@FPR<=1%: F1={t['f1']} P={t['precision']} R={t['recall']} "
          f"FPR={t['fpr']*100:.2f}% | PR-AUC={t['pr_auc']}")
    print(f"          per-attack: {res['per_attack_tuned']}")
    print(f"          per-class AUC: {res['per_class_auc']}")
    return res


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
