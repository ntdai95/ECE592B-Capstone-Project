from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .data import load_feature_table, stratified_split, subsample, to_dataset
from .metrics import aggregate_results, evaluate, pick_threshold
from .preprocess import apply_scaler, drop_features, find_zero_variance, make_scaler


GRIDS = {
    "deep_svdd": {
        "hidden_dims": [(64, 32), (128, 64, 32), (256, 128, 32)],
        "lr": [1e-3, 1e-2],
        "epochs": [40],
        "batch_size": [1024],
        "weight_decay": [1e-6],
    },
    "anomal_e": {
        "hidden": [64],
        "n_layers": [1, 2],
        "lr": [1e-3],
        "epochs": [60],
    },
}

DEFAULT_MODELS = ["anomal_e", "deep_svdd"]


def _grid(space: dict):
    keys = list(space)
    for combo in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, combo))


def _empty_like(ds):
    return replace(ds, X=ds.X[:0], y_bin=ds.y_bin[:0], y_multi=ds.y_multi[:0], ids=ds.ids[:0])


def _val_scores_deep_svdd(params, train, val, seed):
    from .detectors.deepsvdd import DeepSVDD
    det = DeepSVDD(seed=seed, **params).fit(train.X)
    return det.score(val.X)


def _val_scores_anomal_e(params, train, val, seed):
    from .detectors.anomal_e import AnomalE
    det = AnomalE(seed=seed, **params)
    val_scores, _ = det.fit_score_transductive(train, val, _empty_like(train))
    return val_scores


SCORERS = {
    "deep_svdd": lambda p, tr, va, seed: _val_scores_deep_svdd(p, tr, va, seed),
    "anomal_e": lambda p, tr, va, seed: _val_scores_anomal_e(p, tr, va, seed),
}


def _score_validation(scores: np.ndarray, val) -> dict:
    thr = pick_threshold(scores, val.y_bin, strategy="recall_first")
    res = evaluate(scores, val.y_bin, val.y_multi, thr)
    return {
        "val_auc": roc_auc_score(val.y_bin, scores),
        "val_pr_auc": average_precision_score(val.y_bin, scores),
        "val_threshold": res.threshold,
        "val_precision": res.precision,
        "val_recall": res.recall,
        "val_fnr": res.fnr,
        "val_fpr": res.fpr,
        "val_f1": res.f1,
        "val_accuracy": res.accuracy,
        "val_target_met": bool(res.recall >= C.TARGET_RECALL),
    }


def tune(models, feature_set, scaler_name, seeds=(0, 1, 2), quick=None,
         selection_metric="val_auc"):
    unknown = [m for m in models if m not in GRIDS]
    if unknown:
        raise ValueError(f"Unknown model(s) for tuning: {unknown}; choose from {list(GRIDS)}")
    seeds = list(seeds)
    df = load_feature_table(feature_set)
    ds = to_dataset(df)
    if quick:
        ds = subsample(ds, quick, seed=seeds[0])

    prepared = {}
    for seed in seeds:
        train, val, _test = stratified_split(ds, seed=seed)
        zv = find_zero_variance(train)
        train, val = drop_features(train, zv), drop_features(val, zv)
        scaler = make_scaler(scaler_name).fit(train.X)
        prepared[seed] = (apply_scaler(train, scaler), apply_scaler(val, scaler))
    n_tr, n_va = len(prepared[seeds[0]][0]), len(prepared[seeds[0]][1])
    print(f"Tuning on feature_set={feature_set} scaler={scaler_name} "
          f"(train~{n_tr}, val~{n_va}); seeds={seeds}; "
          f"selection = mean {selection_metric} across {len(seeds)} seeds\n")

    tag = f"{feature_set}_{scaler_name}"
    best_path = C.RESULTS_DIR / f"best_params_{tag}.json"
    best_params = json.loads(best_path.read_text()) if best_path.exists() else {}
    rows = []
    for name in models:
        space = GRIDS[name]
        configs = list(_grid(space))
        print(f"== {name}: {len(configs)} configs x {len(seeds)} seeds ==")
        best_score, best = -np.inf, None
        for cfg in configs:
            t0 = time.time()
            per_seed = []
            try:
                for seed in seeds:
                    train, val = prepared[seed]
                    scores = SCORERS[name](cfg, train, val, seed)
                    per_seed.append(_score_validation(scores, val))
            except Exception as e:
                print(f"   {cfg} -> FAILED ({e})")
                continue
            dt = time.time() - t0
            agg = aggregate_results(per_seed)
            rows.append({"model": name, "sec": round(dt, 1), "n_seeds": len(seeds), **agg, **cfg})
            score = agg[f"{selection_metric}_mean"]
            marker = ""
            if score > best_score:
                best_score, best, marker = score, cfg, "  <-- best so far"
            print(f"   {cfg} -> {selection_metric}={score:.4f}"
                  f"+-{agg[f'{selection_metric}_std']:.4f} "
                  f"val_fnr={agg['val_fnr_mean']:.3f} "
                  f"val_fpr={agg['val_fpr_mean']:.3f} ({dt:.0f}s){marker}")
        if best is None:
            raise RuntimeError(f"All tuning configs failed for {name}")
        best_params[name] = best
        print(f"   BEST {name}: {best}  (mean {selection_metric}={best_score:.4f})\n")
        C.ensure_dirs()
        with open(best_path, "w") as f:
            json.dump(best_params, f, indent=2)
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"tuning_{tag}.csv", index=False)

    print(f"Saved best params -> {best_path}")
    print(f"Saved full grid   -> {C.RESULTS_DIR / f'tuning_{tag}.csv'}")
    return best_params


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning (val ROC-AUC)")
    ap.add_argument("--feature-set", default="normalized", choices=list(C.FEATURE_SETS))
    ap.add_argument("--scaler", default="quantile",
                    choices=["quantile", "robust", "standard", "log_standard", "none"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="models to tune; default is the user's Phase-2 scope: anomal_e deep_svdd")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds to average over for stable model selection")
    ap.add_argument("--quick", type=int, default=None)
    ap.add_argument("--selection-metric", default="val_auc",
                    choices=["val_auc", "val_pr_auc", "val_f1", "val_recall", "val_accuracy"],
                    help="metric used to choose best_params JSON; val_auc is threshold-independent")
    args = ap.parse_args()
    tune(args.models, args.feature_set, args.scaler, seeds=args.seeds, quick=args.quick,
         selection_metric=args.selection_metric)


if __name__ == "__main__":
    main()
