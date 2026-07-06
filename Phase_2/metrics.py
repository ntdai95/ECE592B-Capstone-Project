"""Evaluation metrics + threshold selection for the anomaly detectors.

Score convention everywhere in this project: **higher score = more anomalous**.
A sample is flagged as an attack when ``score >= threshold``.

The required Phase-2 metrics (from the assignment) are all produced here:
precision, recall, F1, per-attack detection rate, FPR, FNR, AUC-ROC, PR-AUC,
and the confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from . import config as C


@dataclass
class EvalResult:
    threshold: float
    precision: float
    recall: float            # == overall attack detection rate (1 - FNR)
    f1: float
    fpr: float
    fnr: float
    accuracy: float
    auc_roc: float
    pr_auc: float
    confusion: np.ndarray            # [[TN, FP], [FN, TP]]
    per_attack_dr: dict[str, float]  # detection rate for each attack type
    extra: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        """Flatten into a single dict suitable for a results dataframe row."""
        row = {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "pr_auc": self.pr_auc,
        }
        row.update({f"dr_{k}": v for k, v in self.per_attack_dr.items()})
        return row


def pick_threshold(
    scores: np.ndarray,
    y_bin: np.ndarray,
    target_recall: float = C.TARGET_RECALL,
    max_fpr: float = C.MAX_FPR,
    strategy: str = "recall_first",
) -> float:
    """Choose an operating threshold on a *labeled validation* set.

    strategy:
      * "recall_first"  -> smallest threshold that reaches ``target_recall``
                           (minimizes FNR, the project's stated priority).
      * "fpr_capped"    -> threshold giving the highest recall while FPR <= max_fpr.
      * "youden"        -> classic Youden's J (max tpr - fpr) balance point.
    """
    fpr, tpr, thr = roc_curve(y_bin, scores)
    # roc_curve's first threshold is +inf; drop it for a usable operating point.
    fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]

    if strategy == "recall_first":
        ok = np.where(tpr >= target_recall)[0]
        if len(ok):
            # among thresholds meeting the recall target, take the one with lowest FPR
            best = ok[np.argmin(fpr[ok])]
            return float(thr[best])
        # fall back to the max-recall threshold if target unreachable
        return float(thr[np.argmax(tpr)])

    if strategy == "fpr_capped":
        ok = np.where(fpr <= max_fpr)[0]
        if len(ok):
            best = ok[np.argmax(tpr[ok])]
            return float(thr[best])
        return float(thr[np.argmin(fpr)])

    if strategy == "youden":
        return float(thr[np.argmax(tpr - fpr)])

    raise ValueError(f"Unknown strategy {strategy!r}")


def evaluate(
    scores: np.ndarray,
    y_bin: np.ndarray,
    y_multi: np.ndarray,
    threshold: float,
) -> EvalResult:
    """Compute the full Phase-2 metric suite at a fixed threshold."""
    y_pred = (scores >= threshold).astype(np.int64)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_bin, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_bin, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / cm.sum()

    # Ranking-quality metrics are threshold-independent.
    auc = roc_auc_score(y_bin, scores) if len(np.unique(y_bin)) > 1 else float("nan")
    pr_auc = average_precision_score(y_bin, scores) if len(np.unique(y_bin)) > 1 else float("nan")

    # Per-attack detection rate: of all packets of attack X, how many were flagged.
    per_attack = {}
    for atk in C.ATTACK_LABELS:
        mask = y_multi == atk
        per_attack[atk] = float(y_pred[mask].mean()) if mask.any() else float("nan")

    return EvalResult(
        threshold=float(threshold),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        fpr=float(fpr),
        fnr=float(fnr),
        accuracy=float(accuracy),
        auc_roc=float(auc),
        pr_auc=float(pr_auc),
        confusion=cm,
        per_attack_dr=per_attack,
    )


def aggregate_results(results: list[dict]) -> dict:
    """Mean +/- std across seeds for a list of ``EvalResult.summary_row()`` dicts."""
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))
    return agg
