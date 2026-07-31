"""Print the whole pipeline's results from the saved artifacts, and write the
two summary CSVs. Read-only with respect to the models, about one second.

Nothing is retrained. This formats results.json and the audit JSON files
already sitting in results/, and emits leaderboard.csv and
per_attack_detection.csv from them.

Both stages report here. The Phase 2 packet detectors write into the same
results.json under "phase2_*" keys, so packet-level and flow-level detection
print in one run instead of the packet numbers living only in notebook output.

    python show_results.py            # everything
    python show_results.py --brief    # leaderboard + per-attack only

To regenerate the underlying numbers, run `python run_all.py` (about 20 min).
"""
import argparse
import csv
import json
from pathlib import Path

from evaluation import RESULTS_DIR as OUT
W = 124

NICE = {
    "multiclass_xgb": "Multiclass XGBoost",
    "xgboost": "XGBoost",
    "voting_soft": "Soft Voting (RF+XGB+MLP)",
    "random_forest": "Random Forest",
    "mlp": "Neural Network (MLP)",
    "linear_svm": "Linear SVM",
}


def head(n, title):
    print()
    print("=" * W)
    print(f"  {n}. {title.upper()}")
    print("=" * W)


def rule():
    print("  " + "-" * (W - 4))


def load_results():
    f = OUT / "results.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    rows = [(name, data[key], data[key]["test_tuned"])
            for key, name in NICE.items() if key in data]
    rows.sort(key=lambda x: -x[2]["recall"])
    return rows


def load_phase2():
    """Phase 2 entries are keyed "phase2_*" and carry a "test" block, not the
    "test_tuned"/"per_attack_tuned" pair a Phase 3 model has, so load_results()
    skips them and they are read separately here."""
    f = OUT / "results.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    return sorted((k[len("phase2_"):], v) for k, v in data.items()
                  if k.startswith("phase2_"))


def t_phase2(p2):
    head(1, "Phase 2 packet-level detection  (test set)")
    print("  Unsupervised detectors from the Phase_2 notebooks, recorded by phase2_results.py.")
    print("  Stage 1 of the pipeline: high recall on raw packets, and the false alarms it")
    print("  cannot avoid are what the Phase 3 flow classifier is there to remove.")
    rule()
    print(f"  {'Detector':<16}{'Setting':<32}{'Prec':>7}{'Recall':>8}{'F1':>7}"
          f"{'FPR%':>7}{'PR-AUC':>8}{'ROC-AUC':>9}{'Alerts':>8}{'FP':>7}{'FN':>6}")
    rule()
    for name, r in p2:
        t = r["test"]
        setting = ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items()
                            if not isinstance(v, dict))
        print(f"  {name:<16}{setting[:31]:<32}{t['precision']:>7.3f}{t['recall']:>8.3f}"
              f"{t['f1']:>7.3f}{t['fpr'] * 100:>7.2f}{t['pr_auc']:>8.3f}"
              f"{t['roc_auc']:>9.3f}{t['alerts']:>8}{t['fp']:>7}{t['fn']:>6}")
    rule()

    cls = sorted({c for _, r in p2 for c in r.get("per_attack_detection", {})})
    if cls:
        print("\n  Per-attack detection rate (%)")
        rule()
        print(f"  {'Detector':<16}" + "".join(f"{c:>14}" for c in cls))
        rule()
        for name, r in p2:
            pa = r.get("per_attack_detection", {})
            print(f"  {name:<16}"
                  + "".join(f"{pa[c]:>14.2f}" if c in pa else f"{'n/a':>14}" for c in cls))
        rule()

    for name, r in p2:
        if r.get("selection_rule"):
            print(f"  {name}: {r['selection_rule']}")
        if r.get("threshold_rule"):
            print(f"  {name}: {r['threshold_rule']}")
        for k, v in r.get("params", {}).items():
            if not (k.startswith("previous_") and isinstance(v, dict)):
                continue
            t = r["test"]
            print(f"  {name}: {k[len('previous_'):]} gave recall {v['recall'] * 100:.2f}% on "
                  f"{v['alerts']} alerts; the current setting reaches "
                  f"{t['recall'] * 100:.2f}% on {t['alerts']} "
                  f"({t['alerts'] - v['alerts']:+d} alerts, {v['fn'] - t['fn']} fewer missed).")

    t_phase2_sweeps(p2)


def t_phase2_sweeps(p2):
    """Print each detector's validation hyperparameter sweep, if it recorded one,
    so the selected setting can be read against its neighbours."""
    for name, r in p2:
        sweep = r.get("params", {}).get("sweep")
        if not isinstance(sweep, dict) or not sweep:
            continue
        cols = sorted({c for v in sweep.values() for c in v})
        print(f"\n  {name}: validation sweep, * = selected on PR-AUC")
        rule()
        print(f"  {'setting':<26}" + "".join(f"{c:>13}" for c in cols))
        rule()
        for lab, v in sweep.items():
            print(f"  {lab:<26}"
                  + "".join(f"{v[c]:>13.4f}" if c in v else f"{'n/a':>13}" for c in cols))
        rule()
        best = {c: max(v[c] for v in sweep.values() if c in v) for c in cols}
        top = {c: next(l for l, v in sweep.items() if v.get(c) == best[c]) for c in cols}
        if len({top[c] for c in cols}) > 1:
            print("  " + ";  ".join(f"best {c} at {top[c].rstrip(' *')}" for c in cols))
            print("  The metrics do not agree on a winner, which is why the selection metric is")
            print("  stated rather than assumed. PR-AUC is the one that tracks alert precision.")


def t_leaderboard(rows):
    head(2, "Phase 3 model leaderboard  (test set)")
    print("  Operating point: MAX RECALL subject to validation FPR <= 1% (Neyman-Pearson),")
    print("  with the false-positive allowance sized by its Clopper-Pearson 99% upper bound.")
    rule()
    print(f"  {'#':<4}{'Model':<30}{'Prec':>7}{'Recall':>8}{'F1':>7}"
          f"{'FPR%':>7}{'PR-AUC':>8}{'ROC-AUC':>9}{'FP':>6}{'FN':>6}")
    rule()
    for i, (name, r, t) in enumerate(rows):
        star = "*" if i == 0 else " "
        print(f"  {str(i + 1) + '.':<4}{name:<30}{t['precision']:>7.3f}{t['recall']:>8.3f}"
              f"{t['f1']:>7.3f}{t['fpr'] * 100:>7.2f}{t['pr_auc']:>8.3f}"
              f"{t['roc_auc']:>9.3f}{t['fp']:>6}{t['fn']:>6} {star}")
    rule()
    with open(OUT / "leaderboard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "f1", "precision", "recall", "fpr_pct",
                    "pr_auc", "roc_auc", "false_pos", "false_neg"])
        for name, _, t in rows:
            w.writerow([name, t["f1"], t["precision"], t["recall"],
                        round(t["fpr"] * 100, 2), t["pr_auc"], t["roc_auc"],
                        t["fp"], t["fn"]])

    n90 = sum(1 for _, _, t in rows if t["recall"] >= 0.90)
    mx = max(t["fpr"] for _, _, t in rows) * 100
    best = rows[0]
    print(f"  BEST: {best[0]}, recall {best[2]['recall'] * 100:.1f}% at "
          f"{best[2]['fpr'] * 100:.2f}% FPR ({best[2]['fp']} false alarms, "
          f"{best[2]['fn']} attacks missed)")
    print(f"  {n90} of {len(rows)} models exceed 90% recall.  Max FPR {mx:.2f}%, "
          f"{'all inside' if mx <= 1 else 'BREACH of'} the 1.00% ceiling.")


def t_per_attack(rows):
    head(3, "Per-attack detection rate (%)")
    print("  benign = % correctly passed (1 - FPR);  attacks = % detected")
    cols = ["benign", "brute_force", "ddos", "dns_spoofing", "dos", "xss"]
    rule()
    print(f"  {'Model':<30}" + "".join(f"{c:>14}" for c in cols))
    rule()
    for name, r, _ in rows:
        pa = r["per_attack_tuned"]
        print(f"  {name:<30}" + "".join(f"{pa.get(c, float('nan')):>14.2f}" for c in cols))
    rule()

    auc_cols = [c for c in cols if c != "benign"]
    print("\n  Per-class ROC-AUC against benign (ranking quality, threshold free)")
    rule()
    print(f"  {'Model':<30}" + "".join(f"{c:>14}" for c in auc_cols))
    rule()
    for name, r, _ in rows:
        ac = r.get("per_class_auc", {})
        print(f"  {name:<30}"
              + "".join(f"{ac[c]:>14.4f}" if c in ac else f"{'n/a':>14}" for c in auc_cols))
    rule()

    with open(OUT / "per_attack_detection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "operating_point"] + cols)
        for key, label in [("per_attack_tuned", "max-recall @ FPR<=1%"),
                           ("per_attack_maxf1", "max-F1 @ FPR<=1%"),
                           ("per_attack_default", "default 0.5")]:
            for name, r, _ in rows:
                pa = r.get(key)
                if pa:
                    w.writerow([name, label] + [pa.get(c, "") for c in cols])


def t_context():
    f = OUT / "exp_context_results.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    head(4, "Feature-group ablation: context features and the Task 3.1 scores")
    print("  Source: exp_context.py. One model (XGBoost), same seed, same protocol,")
    print("  the only change is the feature set. Scored under both split regimes, because")
    print("  a gain that appears under RANDOM and vanishes under SESSION is leakage.")
    print("  no_anomaly = the full set minus the three Task 3.1 columns.")
    cls = ["brute_force", "ddos", "dns_spoofing", "dos", "xss"]
    rule()
    print(f"  {'Regime':<10}{'Features':<16}{'Recall':>8}{'FPR%':>7}{'PR-AUC':>9}"
          + "".join(f"{c:>14}" for c in cls))
    rule()
    for regime in ["RANDOM", "SESSION"]:
        for name in ["base", "base+context", "no_anomaly"]:
            r = d.get(f"{regime}|{name}")
            if not r:
                continue
            pa = r["per_attack"]
            print(f"  {regime:<10}{name:<16}{r['recall']:>7.1f}%{r['fpr']:>7.2f}{r['pr_auc']:>9.3f}"
                  + "".join(f"{pa[c]:>14.2f}" if c in pa else f"{'n/a':>14}" for c in cls))
        b, c = d.get(f"{regime}|base"), d.get(f"{regime}|base+context")
        if b and c:
            print(f"  {'':<10}{'delta':<16}{c['recall'] - b['recall']:>+7.1f} {'':>6}"
                  f"{c['pr_auc'] - b['pr_auc']:>+9.3f}"
                  + "".join(f"{c['per_attack'][k] - b['per_attack'][k]:>+14.2f}"
                            if k in c["per_attack"] and k in b["per_attack"]
                            else f"{'n/a':>14}" for k in cls))
        rule()
    b, c = d.get("SESSION|base"), d.get("SESSION|base+context")
    if b and c:
        print(f"  The gain survives the session-disjoint split "
              f"({b['recall']:.1f}% -> {c['recall']:.1f}% recall), so it is not purely")
        print("  an artefact of the random split. XSS is absent from SESSION, one capture day.")

    print("\n  Does Task 3.1 carry the model?  (full set -> full set minus the 3 anomaly columns)")
    rule()
    for regime in ["RANDOM", "SESSION"]:
        f_, n_ = d.get(f"{regime}|base+context"), d.get(f"{regime}|no_anomaly")
        if not (f_ and n_):
            continue
        print(f"  {regime:<10}recall {f_['recall']:>6.1f}% -> {n_['recall']:>6.1f}% "
              f"({n_['recall'] - f_['recall']:+.1f})   "
              f"PR-AUC {f_['pr_auc']:.3f} -> {n_['pr_auc']:.3f} "
              f"({n_['pr_auc'] - f_['pr_auc']:+.3f})")
    rule()
    print("  The Task 3.1 columns rank 2nd and 4th by gain importance, but gain rewards a")
    print("  feature for splitting cleanly, not for covering many rows. Coverage is 7.4% of")
    print("  flows and uneven by class, so read the recall delta above, not the importance")
    print("  rank, as the answer to what Phase 2 contributes to the flow classifier.")


def t_anomaly_ablation():
    """Per-model Task 3.1 ablation, from the "<model>__no_anomaly" keys: every
    model retrained without the three anomaly columns, so the contribution is
    measured across architectures rather than argued from one."""
    f = OUT / "results.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    pairs = [(n, d[k]["test_tuned"], d[k + "__no_anomaly"]["test_tuned"])
             for k, n in NICE.items() if k in d and k + "__no_anomaly" in d]
    if not pairs:
        return
    head(5, "Task 3.1 ablation across every model")
    print("  Each model retrained with the three Phase 2 anomaly columns removed and nothing")
    print("  else changed, then scored through the same operating-point rule. 'with' is the")
    print("  leaderboard model; 'without' is its ablated twin.")
    rule()
    print(f"  {'Model':<30}{'recall with':>12}{'without':>10}{'delta':>8}"
          f"{'PR-AUC with':>13}{'without':>10}{'delta':>8}{'FPR% w/o':>10}")
    rule()
    for name, w, o in sorted(pairs, key=lambda p: -p[1]["recall"]):
        print(f"  {name:<30}{w['recall'] * 100:>11.1f}%{o['recall'] * 100:>9.1f}%"
              f"{(o['recall'] - w['recall']) * 100:>+8.1f}"
              f"{w['pr_auc']:>13.3f}{o['pr_auc']:>10.3f}"
              f"{o['pr_auc'] - w['pr_auc']:>+8.3f}{o['fpr'] * 100:>10.2f}")
    rule()
    dr = [(o["recall"] - w["recall"]) * 100 for _, w, o in pairs]
    dp = [o["pr_auc"] - w["pr_auc"] for _, w, o in pairs]
    print(f"  Mean change from dropping the columns: recall {sum(dr) / len(dr):+.2f} points, "
          f"PR-AUC {sum(dp) / len(dp):+.4f}")
    print(f"  {sum(1 for x in dr if x > 0)} of {len(dr)} models score HIGHER without them.")
    if len(pairs) < len(NICE):
        print(f"  PARTIAL: {len(pairs)} of {len(NICE)} models have an ablated twin. Rerun")
        print("  `python train_models.py --drop-anomaly` to complete the table before quoting it.")
        return

    # Split the models by how well they do WITH the columns, because the answer
    # turns out to depend on that and reporting one average would hide it.
    strong = [(n, o["recall"] - w["recall"]) for n, w, o in pairs if w["recall"] >= 0.85]
    weak = [(n, o["recall"] - w["recall"]) for n, w, o in pairs if w["recall"] < 0.85]
    if strong and weak:
        s = [x * 100 for _, x in strong]
        k = [x * 100 for _, x in weak]
        print(f"\n  The answer depends on the model, so it is split here rather than averaged.")
        print(f"  {len(strong)} models above 85% recall: change ranges {min(s):+.1f} to {max(s):+.1f} "
              f"points and is not")
        print(f"  consistent in sign ({sum(1 for x in s if x > 0)} improve, "
              f"{sum(1 for x in s if x <= 0)} worsen). Rerunning the ablation moves")
        print("  individual models by a comparable amount, so that band is spread, not effect.")
        print(f"  {len(weak)} models below 85% recall: "
              f"{', '.join(f'{n} {x * 100:+.1f}' for n, x in weak)}.")
        print("  Consistently negative, and several times larger than the band above.")
        print("  So the columns do carry signal, but it is signal a strong learner already")
        print("  recovers from the flow and context features on its own. Only the models that")
        print("  cannot extract it themselves still need it. That is why the headline models are")
        print("  unaffected while the linear and neural baselines lose several points, and it is")
        print("  the honest answer to 'did stage one help stage two': yes, redundantly.")
    print("  Gain importance ranks these columns near the top regardless, because gain rewards a")
    print("  feature for splitting cleanly, not for covering many rows. Coverage is 7.4% of flows")
    print("  and uneven by class, so read this table, not the importance rank.")


def t_importance():
    f = OUT / "feature_importance.csv"
    if not f.exists():
        return
    rows = [(r[0], float(r[1])) for i, r in enumerate(csv.reader(f.open())) if i > 0]
    tot = sum(v for _, v in rows)
    ctx = sum(v for n, v in rows if n.startswith("ctx_"))
    head(6, "Feature importance, top 12 (binary XGBoost, gain)")
    rule()
    top = rows[0][1]
    for i, (n, v) in enumerate(rows[:12], 1):
        tag = "[ctx]" if n.startswith("ctx_") else "     "
        bar = "#" * max(1, int(round(v / top * 44)))
        print(f"  {i:>3}. {tag} {n:<28}{v:>8.4f}  {bar}")
    rule()
    print(f"  Context features carry {ctx / tot * 100:.1f}% of total importance "
          f"({sum(1 for n, _ in rows if n.startswith('ctx_'))} of {len(rows)} features).")


def t_budget(rows):
    if not rows:
        return
    name, best, _ = rows[0]
    curve = best.get("detection_vs_budget")
    if not curve:
        return
    head(7, f"Detection vs false-positive budget ({name})")
    print("  Every threshold comes from the validation scores under the same rule as the")
    print("  headline operating point, so these are budgets a deployed sensor could set.")
    print("  The achieved test FPR is shown alongside, to keep the gap visible.")
    cols = list(next(iter(curve.values())))
    hdr = {"overall_recall": "RECALL", "achieved_test_fpr": "test FPR%"}
    rule()
    print(f"  {'budget':<10}" + "".join(f"{hdr.get(c, c):>14}" for c in cols))
    rule()
    for budget, vals in curve.items():
        mark = "  <-- chosen" if budget == "1%" else ""
        print(f"  {budget:<10}" + "".join(f"{vals[c]:>14.2f}" for c in cols) + mark)
    rule()


def t_session():
    f3 = OUT / "session_holdout_results.json"
    fb = OUT / "session_holdout_base_results.json"
    if not f3.exists():
        return
    v3 = {c["condition"][:2]: c for c in json.loads(f3.read_text())["conditions"]}
    head(8, "Validity check: capture-session held-out evaluation")
    print("  Benign was recorded only on 2022-10-07/08 and each attack on its own days, so a")
    print("  random split lets a model score by recognising the capture session. XSS excluded")
    print("  (single capture day). Capture window is predictable from flow features at AUC 0.922.")
    cls = ["brute_force", "ddos", "dns_spoofing", "dos"]
    rule()
    print(f"  {'Condition':<47}{'PR-AUC':>8}{'Recall':>8}{'FPR%':>8}"
          + "".join(f"{c:>13}" for c in cls))
    rule()
    lbl = {"C3": "C3  benign random   attacks random   (ref)",
           "C1": "C1  benign BY DAY   attacks random",
           "C2": "C2  benign random   attacks BY DAY",
           "C0": "C0  benign BY DAY   attacks BY DAY    (HONEST)"}
    for tag in ["C3", "C1", "C2", "C0"]:
        c = v3[tag]
        pa = c["per_attack_detection"]
        print(f"  {lbl[tag]:<47}{c['pr_auc']:>8.3f}{c['recall_at_budget'] * 100:>7.1f}%"
              f"{c['achieved_test_fpr'] * 100:>7.2f}%"
              + "".join(f"{pa[k]:>13.2f}" for k in cls))
    rule()
    print("  Recall is quoted with the FPR it actually cost. The threshold is calibrated on a")
    print("  held-out slice of the TRAINING capture day; where benign traffic shifts between")
    print("  days the 1% budget does not survive the move, and a recall figure from a row that")
    print("  breached the budget is not comparable to the leaderboard.")

    if fb.exists():
        bs = {c["condition"][:2]: c for c in json.loads(fb.read_text())["conditions"]}
        print("\n  Does the gain survive the honest split?  (base features -> + context)")
        rule()
        print(f"  {'':<16}{'C3 base':>10}{'C3 ctx':>9}{'delta':>8}   "
              f"{'C0 base':>10}{'C0 ctx':>9}{'delta':>8}")
        rule()
        for k in ["dos", "ddos", "dns_spoofing", "brute_force"]:
            b3 = bs["C3"]["per_attack_detection"][k]; c3 = v3["C3"]["per_attack_detection"][k]
            b0 = bs["C0"]["per_attack_detection"][k]; c0 = v3["C0"]["per_attack_detection"][k]
            print(f"  {k:<16}{b3:>10.1f}{c3:>9.1f}{c3 - b3:>+8.1f}   "
                  f"{b0:>10.1f}{c0:>9.1f}{c0 - b0:>+8.1f}")
        for label, key, fmt in [("OVERALL RECALL", "recall_at_budget", 100),
                                ("achieved FPR %", "achieved_test_fpr", 100),
                                ("PR-AUC", "pr_auc", 1)]:
            b3 = bs["C3"][key] * fmt; c3 = v3["C3"][key] * fmt
            b0 = bs["C0"][key] * fmt; c0 = v3["C0"][key] * fmt
            d = 1 if fmt == 100 else 3
            print(f"  {label:<16}{b3:>10.{d}f}{c3:>9.{d}f}{c3 - b3:>+8.{d}f}   "
                  f"{b0:>10.{d}f}{c0:>9.{d}f}{c0 - b0:>+8.{d}f}")
        rule()
        print("  Read the C0 columns by PR-AUC, not by recall. Both C0 rows breached the FPR")
        print("  budget, and by different amounts, so their recalls sit at different operating")
        print("  points and the recall delta is not a like-for-like comparison. PR-AUC is")
        print("  threshold-free: 0.149 -> 0.594 under the honest split against 0.784 -> 0.926")
        print("  under the random one. The context features help MORE where the calendar")
        print("  shortcut has been taken away, which is the opposite of what leakage looks like.")

    c0 = v3["C0"]
    print(f"\n  HEADLINE CAVEAT: under a session-disjoint split this pipeline reaches "
          f"{c0['recall_at_budget'] * 100:.1f}% recall")
    print(f"  only by spending {c0['achieved_test_fpr'] * 100:.1f}% FPR, {c0['achieved_test_fpr'] / 0.01:.0f}x "
          f"its budget. PR-AUC falls from {v3['C3']['pr_auc']:.3f} to {c0['pr_auc']:.3f}.")
    print("  What fails first is not ranking but calibration: an operating point fitted on one")
    print("  capture day does not hold its false-positive budget on another. Report the")
    print("  leaderboard WITH this table.")


def t_two_stage():
    """Combined system: stage 1 autoencoder alerts re-checked by stage 2."""
    f = OUT / "results.json"
    if not f.exists():
        return
    d = json.loads(f.read_text()).get("two_stage_system")
    if not d:
        return
    m = d["combined"]
    stage2 = NICE.get(d.get("stage2_detector", ""), d.get("stage2_detector", "flow classifier"))
    print()
    print("=" * W)
    print(f"  FINAL TWO-STAGE SYSTEM  (stage 1 autoencoder -> stage 2 {stage2})")
    print("=" * W)
    print("  Final decision: attack only if stage 1 alerts AND stage 2 confirms.")
    rule()
    print(f"  Overall accuracy {m['accuracy'] * 100:6.2f}%    precision {m['precision'] * 100:6.2f}%"
          f"    recall {m['recall'] * 100:6.2f}%    F1 {m['f1'] * 100:6.2f}%    FPR {m['fpr'] * 100:5.2f}%")
    print(f"  Confusion (attack=+): TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  False positives: {d['stage1_fp']} (stage 1 alone) -> {m['fp']} (after stage 2) "
          f"= {d['fp_reduction_pct_vs_phase2']:.2f}% reduction")
    print(f"  Recall: {d['stage1_recall'] * 100:.2f}% (stage 1) -> "
          f"{m['recall'] * 100:.2f}% (two-stage)")
    rule()


def render(brief=False):
    """Print every results table from the saved artifacts. Called by run_all.py."""
    rows = load_results()
    if not rows:
        raise SystemExit("No results.json found. Run `python run_all.py` first.")

    p2 = load_phase2()
    meta = OUT / "prep_meta.json"
    print("=" * W)
    print("  MULTI-STAGE IoT IDS RESULTS  (saved artifacts, nothing retrained)")
    print("  Stage 1 packet-level (Phase 2) -> Stage 2 flow-level (Phase 3)")
    if meta.exists():
        m = json.loads(meta.read_text())
        nctx = len([f for f in m["features"] if f.startswith("ctx_")])
        print(f"  Phase 3: {len(m['features'])} features ({nctx} multi-flow context) | "
              f"60/20/20 stratified split | seed 1")
    if not p2:
        print("  Phase 2: no packet-level results recorded. Run the Phase_2 notebooks to the")
        print("           save_phase2_result cell to add them.")
    print("=" * W)

    if p2:
        t_phase2(p2)
    t_leaderboard(rows)
    t_per_attack(rows)
    t_two_stage()
    if not brief:
        t_context()
        t_anomaly_ablation()
        t_importance()
        t_budget(rows)
        t_session()

    print()
    print("=" * W)
    print(f"  Files: {OUT}")
    print("  Regenerate everything: python run_all.py     Audit: python verify_context_integrity.py")
    print("=" * W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true", help="leaderboard + per-attack only")
    render(brief=ap.parse_args().brief)


if __name__ == "__main__":
    main()
