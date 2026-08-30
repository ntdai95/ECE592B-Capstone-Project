from __future__ import annotations

import argparse
import json
import time

import torch
import numpy as np
import pandas as pd

from . import config as C
from . import plots
from .data import load_feature_table, stratified_split, subsample, to_dataset
from .fusion import fuse
from .metrics import aggregate_results, evaluate, pick_threshold
from .preprocess import apply_scaler, drop_features, find_zero_variance, make_scaler

ALL_MODELS = [
    "deep_svdd",
    "anomal_e",
    "fusion",
]

FUSION_PARTS = ["anomal_e", "deep_svdd"]


def build_detector(name: str, seed: int, params: dict | None = None):
    params = params or {}
    if name == "deep_svdd":
        from .detectors.deepsvdd import DeepSVDD
        return DeepSVDD(seed=seed, **params)
    if name == "anomal_e":
        from .detectors.anomal_e import AnomalE
        return AnomalE(seed=seed, **params)
    raise ValueError(f"Unknown model {name!r}")


def _scores_for(name, det, train, val, test):
    if name == "anomal_e":
        val_s, test_s = det.fit_score_transductive(train, val, test)
        return val_s, test_s, det.last_test_emb
    det.fit(train.X)
    return det.score(val.X), det.score(test.X), None


def _eval_two_points(val_scores, test_scores, val, test, primary_strategy):
    thr_p = pick_threshold(val_scores, val.y_bin, strategy=primary_strategy)
    res = evaluate(test_scores, test.y_bin, test.y_multi, thr_p)

    thr_b = pick_threshold(val_scores, val.y_bin, strategy="youden")
    res_b = evaluate(test_scores, test.y_bin, test.y_multi, thr_b)

    val_recall = evaluate(val_scores, val.y_bin, val.y_multi, thr_p).recall
    extra = {
        "target_met": bool(val_recall >= C.TARGET_RECALL),
        "degenerate": bool(res.fpr > 0.5),
        "bal_recall": res_b.recall, "bal_fnr": res_b.fnr, "bal_fpr": res_b.fpr,
        "bal_f1": res_b.f1, "bal_accuracy": res_b.accuracy,
    }
    return res, extra


def run_seed(models, feature_set, seed, ds, strategy, scaler_name, best_params=None):
    best_params = best_params or {}
    train, val, test = stratified_split(ds, seed=seed)
    zv = find_zero_variance(train)
    train, val, test = (drop_features(train, zv), drop_features(val, zv), drop_features(test, zv))
    scaler = make_scaler(scaler_name).fit(train.X)
    train, val, test = (apply_scaler(train, scaler), apply_scaler(val, scaler),
                        apply_scaler(test, scaler))
    rows, cache, emb_cache = [], {}, {}

    for name in [m for m in models if m != "fusion"]:
        t0 = time.time()
        det = build_detector(name, seed, best_params.get(name))
        val_scores, test_scores, test_emb = _scores_for(name, det, train, val, test)
        res, extra = _eval_two_points(val_scores, test_scores, val, test, strategy)
        row = {"model": name, "feature_set": feature_set, "seed": seed,
               "fit_sec": round(time.time() - t0, 2), **res.summary_row(), **extra}
        rows.append(row)
        cache[name] = (val_scores, test_scores, val, test, res)
        if test_emb is not None:
            emb_cache["anomal_e"] = test_emb
        flag = "" if extra["target_met"] and not extra["degenerate"] else "  <-- check FPR/target"
        print(f"  [{name:11s}] auc={res.auc_roc:.3f} pr_auc={res.pr_auc:.3f} | "
              f"recall={res.recall:.3f} fnr={res.fnr:.3f} fpr={res.fpr:.3f} "
              f"| bal: fnr={extra['bal_fnr']:.3f} fpr={extra['bal_fpr']:.3f} "
              f"acc={extra['bal_accuracy']:.3f} ({row['fit_sec']}s){flag}")

    if "fusion" in models:
        parts = [p for p in FUSION_PARTS if p in cache]
        if len(parts) >= 2:
            val_s = [cache[p][0] for p in parts]
            test_s = [cache[p][1] for p in parts]
            f_val = fuse(val_s, val_s, combine="max")
            f_test = fuse(test_s, val_s, combine="max")
            res, extra = _eval_two_points(f_val, f_test, val, test, strategy)
            rows.append({"model": "fusion", "feature_set": feature_set, "seed": seed,
                         "fit_sec": 0.0, **res.summary_row(), **extra})
            cache["fusion"] = (f_val, f_test, val, test, res)
            print(f"  [{'fusion':11s}] auc={res.auc_roc:.3f} pr_auc={res.pr_auc:.3f} | "
                  f"recall={res.recall:.3f} fnr={res.fnr:.3f} fpr={res.fpr:.3f} "
                  f"| bal: fnr={extra['bal_fnr']:.3f} fpr={extra['bal_fpr']:.3f} "
                  f"acc={extra['bal_accuracy']:.3f} (parts={parts})")
        else:
            print(f"  [fusion] skipped (need >=2 of {FUSION_PARTS})")

    return rows, cache, emb_cache


def make_figures(cache, emb_cache, feature_set):
    for name, (val_s, test_s, val, test, res) in cache.items():
        tag = f"{name}_{feature_set}"
        plots.plot_roc(test_s, test.y_bin, name, f"roc_{tag}.png")
        plots.plot_pr(test_s, test.y_bin, name, f"pr_{tag}.png")
        plots.plot_confusion(res.confusion, name, f"cm_{tag}.png")
        plots.plot_per_attack_dr(res.per_attack_dr, name, f"dr_{tag}.png")

    if "anomal_e" in emb_cache:
        _, _, _, test, _ = cache["anomal_e"]
        plots.plot_embedding(emb_cache["anomal_e"], test.y_multi, "Anomal-E",
                             f"umap_anomal_e_{feature_set}.png")


def main():
    ap = argparse.ArgumentParser(description="Phase 2 unsupervised anomaly detection")
    ap.add_argument("--feature-set", default="normalized", choices=list(C.FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--quick", type=int, default=None, help="cap rows for a fast smoke test")
    ap.add_argument("--strategy", default="recall_first",
                    choices=["recall_first", "fpr_capped", "youden"])
    ap.add_argument("--scaler", default="quantile",
                    choices=["quantile", "robust", "standard", "log_standard", "none"],
                    help="feature scaler (fit on train only); 'none' reproduces the broken baseline")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-tuned", action="store_true",
                    help="ignore tuned hyperparameters (results/best_params_*.json)")
    args = ap.parse_args()

    C.ensure_dirs()
    models = ALL_MODELS if args.models == ["all"] else args.models

    best_params = {}
    bp_file = C.RESULTS_DIR / f"best_params_{args.feature_set}_{args.scaler}.json"
    if bp_file.exists() and not args.no_tuned:
        best_params = {k: v for k, v in json.loads(bp_file.read_text()).items() if v}
        best_params = {k: {kk: tuple(vv) if isinstance(vv, list) else vv
                           for kk, vv in p.items()} for k, p in best_params.items()}
        print(f"Using tuned hyperparameters from {bp_file.name}: "
              f"{ {k: best_params[k] for k in best_params} }")
    else:
        print("Using default hyperparameters (no tuned params loaded)")

    print(f"Loading feature set '{args.feature_set}'"
          + (f" (quick={args.quick} rows, stratified)" if args.quick else "") + " ...")
    df = load_feature_table(args.feature_set)
    ds = to_dataset(df)
    if args.quick:
        ds = subsample(ds, args.quick, seed=0)
    print(f"Loaded {len(ds)} packets, {ds.X.shape[1]} features, "
          f"{ds.y_bin.mean() * 100:.2f}% attacks")
    print(f"Scaler: {args.scaler} | zero-variance drop + scaling fit on train split only")

    all_rows, first_cache, first_emb = [], None, None
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        rows, cache, emb_cache = run_seed(models, args.feature_set, seed, ds,
                                          args.strategy, args.scaler, best_params)
        all_rows.extend(rows)
        if first_cache is None:
            first_cache, first_emb = cache, emb_cache

    df_res = pd.DataFrame(all_rows)
    df_res.insert(1, "scaler", args.scaler)
    tag = f"{args.feature_set}_{args.scaler}"
    out_csv = C.RESULTS_DIR / f"results_{tag}.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved per-seed metrics -> {out_csv}")

    print("\n=== mean +/- std across seeds ===")
    summary_rows = []
    for name in df_res["model"].unique():
        sub = df_res[df_res["model"] == name].to_dict("records")
        agg = aggregate_results(sub)
        summary_rows.append({"model": name, **agg})
        print(f"{name:11s} recall={agg['recall_mean']:.3f}±{agg['recall_std']:.3f} "
              f"fnr={agg['fnr_mean']:.3f} fpr={agg['fpr_mean']:.3f} "
              f"f1={agg['f1_mean']:.3f} auc={agg['auc_roc_mean']:.3f}")
    pd.DataFrame(summary_rows).to_csv(C.RESULTS_DIR / f"summary_{tag}.csv", index=False)

    if not args.no_figures and first_cache is not None:
        print("\nGenerating figures ...")
        make_figures(first_cache, first_emb or {}, tag)
        print(f"Figures -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
