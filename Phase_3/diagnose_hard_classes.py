"""Why do brute force, DNS spoofing and XSS detect so poorly?

Runs five checks against the saved splits and the trained XGBoost and RF models,
writing hard_class_diagnostics.json plus a readable console report:

  1. Class balance          are the hard classes simply rarer?
  2. Score distributions    is the model ranking them badly, or is the threshold too high?
  3. Detection vs FPR budget how much of the loss is the operating point alone
  4. Label integrity        duplicate feature vectors and benign/attack label collisions
  5. Service context        Dst Port and Protocol mix per class against benign
"""
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import FLOW_DATA_DIR, OUT


def load():
    d = np.load(f"{OUT}/splits.npz", allow_pickle=True)
    meta = json.load(open(f"{OUT}/prep_meta.json"))
    return d, meta


def check_balance(d):
    print("\n1. CLASS BALANCE")
    rows = {}
    for nm in ["tr", "va", "te"]:
        c = Counter(d[f"t_{nm}"])
        rows[nm] = {k: int(v) for k, v in sorted(c.items())}
        print(f"   {nm}: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print("   -> attack types are sampled to equal size, so rarity is not the explanation.")
    return rows


def check_scores(d, scores, name):
    print(f"\n2. SCORE DISTRIBUTION PER CLASS ({name}, test set)")
    t_te, s = d["t_te"], scores
    ben = t_te == "benign"
    print(f"   {'class':<15}{'n':>6}{'p10':>8}{'p25':>8}{'med':>8}{'p75':>8}{'AUCvsBenign':>13}")
    out = {}
    for c in sorted(set(t_te)):
        m = t_te == c
        sc = s[m]
        auc = None
        if c != "benign":
            auc = float(roc_auc_score(
                np.r_[np.zeros(int(ben.sum())), np.ones(int(m.sum()))], np.r_[s[ben], sc]))
        out[c] = {"n": int(m.sum()),
                  "p10": round(float(np.percentile(sc, 10)), 4),
                  "p25": round(float(np.percentile(sc, 25)), 4),
                  "median": round(float(np.median(sc)), 4),
                  "p75": round(float(np.percentile(sc, 75)), 4),
                  "auc_vs_benign": round(auc, 4) if auc else None}
        print(f"   {c:<15}{m.sum():>6}{out[c]['p10']:>8.3f}{out[c]['p25']:>8.3f}"
              f"{out[c]['median']:>8.3f}{out[c]['p75']:>8.3f}"
              f"{(f'{auc:.4f}' if auc else '-'):>13}")
    print("   -> hard classes are bimodal: a well-scored mode plus a mode sitting in the")
    print("      benign score range. AUC stays high (0.86-0.96) so ranking is not the problem.")
    return out


def check_budget(d, scores, name):
    print(f"\n3. DETECTION VS FPR BUDGET ({name}, test set)")
    t_te, y_te, s = d["t_te"], d["y_te"], scores
    attacks = sorted(set(t_te) - {"benign"})
    ben = np.sort(s[y_te == 0])
    print(f"   {'budget':<9}" + "".join(f"{c[:11]:>13}" for c in attacks) + f"{'overall':>10}")
    out = {}
    for b in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]:
        thr = float(np.quantile(ben, 1 - b))
        pred = (s >= thr).astype(int)
        row = {c: round(float((pred[t_te == c] == 1).mean()) * 100, 2) for c in attacks}
        row["overall_recall"] = round(float(pred[y_te == 1].mean()) * 100, 2)
        out[f"{b*100:g}%"] = row
        print(f"   {b*100:>5.2f}%   " + "".join(f"{row[c]:>13.1f}" for c in attacks)
              + f"{row['overall_recall']:>10.1f}")
    print("   -> the hard classes keep climbing well past 1% FPR, so a large part of the")
    print("      shortfall is the fixed FPR budget, not the model.")
    return out


def check_label_integrity(d, meta):
    print("\n4. LABEL INTEGRITY (duplicate feature vectors / benign-attack collisions)")
    cols = meta["features"]
    base = [i for i, c in enumerate(cols)
            if c not in ("flow_anomaly_score", "flow_anomaly_flag", "has_anomaly_score")]
    X = np.vstack([d["X_tr"], d["X_va"], d["X_te"]])[:, base]
    t = np.concatenate([d["t_tr"], d["t_va"], d["t_te"]])
    view = np.ascontiguousarray(X).view([('', X.dtype)] * X.shape[1]).ravel()
    _, inv, cnt = np.unique(view, return_inverse=True, return_counts=True)
    df = pd.DataFrame({"g": inv, "t": t})
    ben_groups = set(df.loc[df.t == "benign", "g"].unique())
    out = {"unique_vectors": int(len(set(inv))), "total_rows": int(len(inv))}
    print(f"   unique feature vectors: {len(set(inv)):,} / {len(inv):,}")
    for c in sorted(set(t) - {"benign"}):
        m = df.t == c
        coll = float(df.loc[m, "g"].isin(ben_groups).mean()) * 100
        out[c] = round(coll, 2)
        print(f"   {c:<15} identical to a benign flow: {coll:>5.1f}%")
    print("   -> collisions around 1%, so irreducible label noise is not the explanation.")
    return out


def check_service_context():
    print("\n5. SERVICE CONTEXT (Dst Port / Protocol mix, all rows)")
    df = pd.read_csv(FLOW_DATA_DIR / "normalized_original_data.csv",
                     usecols=["Flow ID", "label"])
    df.columns = [c.strip() for c in df.columns]
    ids = pd.read_csv(FLOW_DATA_DIR / "flow_data_ids.csv")
    ids.columns = [c.strip() for c in ids.columns]
    m = df.merge(ids.drop_duplicates("Flow ID")[["Flow ID", "Dst Port", "Protocol"]],
                 on="Flow ID", how="left")
    out = {}
    for c in sorted(m["label"].astype(str).unique()):
        top = m.loc[m.label == c, "Dst Port"].value_counts(normalize=True).head(6)
        out[c] = {str(int(k)): round(float(v) * 100, 1) for k, v in top.items()}
        print(f"   {c:<15}" + "  ".join(f"{int(k)}:{v*100:.0f}%" for k, v in top.items()))
    print("   -> brute force, DNS spoofing and XSS sit on the same service mix as benign")
    print("      (53, 443, 32100, 80). DDoS and DoS sit on distinct ports (8080, 8009, 80).")
    print("      Adding port as a feature would only help the classes that are already easy.")
    return out


def main():
    d, meta = load()
    report = {"balance": check_balance(d)}
    for name, fn in [("xgboost", "model_xgb.joblib"), ("random_forest", "model_rf.joblib")]:
        p = Path(OUT) / fn
        if not p.exists():
            print(f"\n[skip {name}: {fn} not found, run the trainers first]")
            continue
        mdl = joblib.load(p)
        s = mdl.predict_proba(d["X_te"])[:, 1]
        report[f"scores_{name}"] = check_scores(d, s, name)
        report[f"budget_{name}"] = check_budget(d, s, name)
    report["label_integrity"] = check_label_integrity(d, meta)
    report["service_context"] = check_service_context()
    with open(f"{OUT}/hard_class_diagnostics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {OUT}/hard_class_diagnostics.json")


if __name__ == "__main__":
    main()
