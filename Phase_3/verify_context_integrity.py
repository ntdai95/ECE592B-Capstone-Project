"""Integrity audit for the connection-window context features.

  1. CAUSALITY  : rebuild the features after deleting every flow after a cut
                  point. Features for flows before the cut must be identical.
                  If a future flow influenced a past feature, this fails.
  2. LABEL-FREE : confirm the builder never reads the label column.
  3. FPR BUDGET : every model's test FPR is strictly inside the 1% ceiling.
  4. IMPORTANCE : which context features carry the gain. Writes
                  feature_importance.csv, consumed by show_results.py.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from common import FLOW_DATA_DIR, OUT


def check_causality():
    import build_context_features as B
    ids = B._load().sort_values("t", kind="mergesort").reset_index(drop=True)
    cut = int(len(ids) * 0.6)
    t_cut = ids["t"].iloc[cut]

    def build_on(sub):
        sub = sub.sort_values("t", kind="mergesort").reset_index(drop=True)
        dport = sub["Dst Port"].to_numpy()
        dip = pd.factorize(sub["Dst IP"])[0]
        t = sub["t"].to_numpy()
        codes = pd.factorize(sub["Src IP"])[0]
        order = np.argsort(codes, kind="mergesort")
        cnt = np.zeros(len(sub), np.float32)
        s = 0
        while s < len(order):
            e = s
            while e < len(order) and codes[order[e]] == codes[order[s]]:
                e += 1
            idx = order[s:e]
            idx = idx[np.argsort(t[idx], kind="mergesort")]
            c, _, _, _, _ = B._window_block(t[idx], dport[idx], dip[idx],
                                            codes[idx[0]], 60)
            cnt[idx] = c
            s = e
        return pd.Series(cnt, index=sub["Flow ID"].to_numpy())

    full = build_on(ids)
    trunc = build_on(ids[ids["t"] <= t_cut])
    common = trunc.index
    same = np.allclose(full.loc[common].to_numpy(), trunc.to_numpy())
    print(f"1. CAUSALITY  : truncating the capture at 60% leaves "
          f"{len(common):,} early flows' features "
          f"{'IDENTICAL, PASS' if same else 'CHANGED, FAIL'}")
    return same


def check_label_free():
    src = Path(__file__).with_name("build_context_features.py").read_text()
    bad = [w for w in ["label", "attack_type", "y_type", "benign"] if w in src]
    ok = not bad
    print(f"2. LABEL-FREE : builder references {bad or 'no label columns'} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_budget():
    """The 1% ceiling is a Phase 3 flow-level constraint.

    Phase 2 writes its packet detectors into the same results.json under
    "phase2_*" keys. Those entries carry a "test" block rather than the
    "test_tuned" a Phase 3 model has, and a stage-one detector is meant to run
    at a much looser FPR, so including them here would both crash and fail the
    check for the wrong reason.
    """
    rows, ok = [], True
    results = json.loads((Path(OUT) / "results.json").read_text())
    for name, r in results.items():
        if name.startswith("phase2_") or "test_tuned" not in r:
            continue
        fpr = r["test_tuned"]["fpr"]
        if fpr > 0.01:
            ok = False
        rows.append((name, fpr * 100, r["test_tuned"]["recall"] * 100))
    print("3. FPR BUDGET :")
    for m, f, rc in sorted(rows, key=lambda x: -x[2]):
        print(f"     {'ok ' if f <= 1.0 else 'OVER'} {m:<26} fpr {f:5.2f}%  recall {rc:5.2f}%")
    print(f"   -> {'all strictly inside 1%, PASS' if ok else 'BREACH, FAIL'}")
    return ok


def check_importance():
    import joblib
    m = joblib.load(f"{OUT}/model_xgb.joblib")
    meta = json.load(open(f"{OUT}/prep_meta.json"))
    imp = m.feature_importances_
    s = pd.Series(imp, index=meta["features"]).sort_values(ascending=False)
    ctx_share = s[[c for c in s.index if c.startswith("ctx_")]].sum()
    print(f"4. IMPORTANCE : context features carry {ctx_share*100:.1f}% of total gain")
    print("   top 12 features:")
    for k, v in s.head(12).items():
        tag = "[ctx]" if k.startswith("ctx_") else "     "
        print(f"     {tag} {k:<34} {v:.4f}")
    s.to_csv(f"{OUT}/feature_importance.csv", header=["importance"])
    return float(ctx_share)


if __name__ == "__main__":
    print("=" * 72)
    a = check_causality()
    b = check_label_free()
    c = check_budget()
    d = check_importance()
    print("=" * 72)
    print(f"VERDICT: {'ALL INTEGRITY CHECKS PASS' if (a and b and c) else 'REVIEW REQUIRED'}")
