import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = RESULTS_DIR / "results.json"


def _as_percent(d):
    vals = [float(v) for v in d.values()]
    scale = 100.0 if vals and max(vals) <= 1.0 else 1.0
    return {_CLASS_NAMES.get(k, k): float(v) * scale for k, v in d.items()}


_CLASS_NAMES = {
    "Benign": "benign",
    "DDoS-HTTP Flood": "ddos",
    "DoS-HTTP Flood": "dos",
    "DNS Spoofing": "dns_spoofing",
    "Brute Force": "brute_force",
    "XSS": "xss",
}


def save_phase2_result(detector, tn, fp, fn, tp, roc_auc, pr_auc,
                       per_attack_detection, threshold_rule="",
                       selection_rule="", **params):
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    per_attack_detection = _as_percent(per_attack_detection)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    res = {
        "stage": "phase2_packet",
        "detector": detector,
        "params": params,
        "threshold_rule": threshold_rule,
        "selection_rule": selection_rule,
        "test": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "fnr": round(fn / (fn + tp), 4) if fn + tp else 0.0,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "alerts": tp + fp,
        },
        "per_attack_detection": {k: round(v, 2)
                                 for k, v in per_attack_detection.items()},
    }

    all_res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_res[f"phase2_{detector}"] = res
    RESULTS.write_text(json.dumps(all_res, indent=2))
    print(f"saved phase2_{detector} -> {RESULTS}")
    return res
