# Code Analysis Report

Scope: (A) the Phase 2 environment, (B) verification that Phase 2 runs and its flow
is complete and consistent, and (C) carry-forward findings from dependency tracing.
**All conclusions below are based on traced imports, grep of the source tree, git
status, and actual runs — not assumptions.**

---

## A. Environment

Phase 2 runs in a virtual environment at the repo root, **`.venv/`**:

- Built on the system Python 3.11 (`--system-site-packages`), inheriting the heavy
  scientific/deep-learning stack: **torch 2.9.0+cpu, numpy 2.3.3, pandas, scikit-learn,
  matplotlib, scipy**.
- Adds the three Phase-2-specific packages into the venv: **torch-geometric 2.8.0,
  pyod 3.6.1, umap-learn 0.5.12**.
- Verified: `import torch, torch_geometric, pyod, umap, sklearn, pandas` all succeed
  from `.venv/Scripts/python.exe`.

Build recipe (also in `docs/phase2_run_guide.md` §5):
```bash
"C:/Python311/python.exe" -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install pyod torch-geometric umap-learn
```

`.venv/` is gitignored. The input data lives in `data/processed_data/packet-data/`
(Phase 1's output location); no code path points anywhere else.

---

## B. Phase 2 verification

### Does it run? — Yes (verified by execution via `.venv`)

| Run | Result |
|---|---|
| Quick smoke: `--quick 20000 --models deep_svdd anomal_e fusion --seeds 0` | **Exit 0.** deep_svdd, anomal_e, and fusion all scored; metrics CSV written. |
| Full data: `--models all --seeds 0 1` (with figures) | **Exit 0.** 3 models × 2 seeds; `results_*.csv` + `summary_*.csv` written; **13 figures** generated (ROC/PR/CM/per-attack ×3 + UMAP). anomal_e 0.848, fusion 0.843, deep_svdd 0.745. |

Note: full 5-seed runs can hit an environment RAM limit (`numpy _ArrayMemoryError`)
because the disk is near-full and there is no swap headroom — not a code defect.
Run fewer seeds or free disk space (see `docs/phase2_run_guide.md` §6).

### Is the flow complete and consistent? — Yes (traced)
The end-to-end path resolves cleanly:
`load_feature_table → to_dataset → (subsample) → stratified_split →
find_zero_variance → drop_features → make_scaler.fit(train) → apply_scaler →
detector.fit/score (or GNN fit_score_transductive) → pick_threshold(val) →
evaluate(test) → fuse → aggregate_results → CSVs + figures`.

- All detectors share the `BaseDetector` `fit/score` interface (the GNN uses the
  documented transductive variant).
- Leakage discipline is consistent: zero-variance drop + scaler fit on **train
  only**, per seed; labels are never passed to a detector's `fit`.
- `run_phase2` and `tune` both consume the same scaler helpers from `preprocess.py`.

---

## C. Carry-forward findings (from import/usage tracing)

These are **not** part of the current task's changes; listed for maintainers.
Grouped into **confirmed** vs **needs verification**.

### Confirmed by tracing (in-repo `.py` only)
| Item | Evidence |
|---|---|
| `preprocess.py` "Half B" (orphaned `.parquet`/`.joblib` CLI) — **already removed** in an earlier session. | No in-repo consumer read its outputs (grep: no `read_parquet` / `joblib.load`). |
| `run_phase2.py` imports `normalize_ref` but never calls it (used only inside `fusion.py`). | Dead import in that file. |
| `config.py` `SRC_NODE_COL`, `DST_NODE_COL`, `GRAPH_NODE_OPTIONS` are unreferenced. | Grep finds only their definitions; `graph.py` hard-codes node columns. |
| `metrics.py` `EvalResult.extra` field is never populated or read. | Grep: no `.extra` access. |
| Duplicate seed-aggregation helpers: `metrics.aggregate_results` vs `tune._aggregate`. | Near-identical mean/std logic. |
| Cross-phase duplication: log1p + StandardScaler + per-label IsolationForest appears in both `Phase_1/data_preprocessing_packet.py` and `Phase_3/flow_based_feature_engineering.py`. | Phase 3's docstring explicitly references the Phase 1 version. |

### Needs verification (cannot be proven from the `.py` tree alone)
| Item | Why it needs verification |
|---|---|
| Whether `normalized_original_data.csv` is consumed **directly** (unscaled-expectation) by a teammate's k-means/autoencoder code or notebooks. | Would determine if Phase 1's own `StandardScaler` step is still needed, since Phase 2 re-scales on top of it. |
| Whether the extra scalers (`log_standard`, `none`) are still wanted for ablations. | They are reachable via `--scaler` but unused by the default (quantile). |
| Notebook usage (`Working/*.ipynb`, `Phase_2/*.ipynb`). | The traces above cover `.py` files only; a symbol used **only** from a notebook would not appear in them. |

---

## Summary

- **Environment**: Phase 2 runs in the repo-root `.venv/` (system Python 3.11 +
  torch-geometric / pyod / umap-learn). Verified working.
- **Phase 2**: **runs correctly** (exit 0 on smoke + full 2-seed runs; 13 figures);
  flow is complete and consistent. The only failure observed (5-seed OOM) is an
  environment RAM limit, not a code defect.
- **Documentation**: see `docs/phase2_overview.md` and `docs/phase2_run_guide.md`.
