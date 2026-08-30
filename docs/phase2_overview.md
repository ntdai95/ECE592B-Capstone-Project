# Phase 2 — Overview

Packet-level **unsupervised anomaly detection** (network intrusion detection) for
the ECE 597 IoT capstone. Phase 2 takes the preprocessed packet feature tables
produced by Phase 1 and runs a set of anomaly detectors that flag attack traffic
**without using labels at training time**. Labels are used only to (a) pick an
operating threshold on a validation split and (b) compute the final metrics.

---

## 1. High-level: what Phase 2 does

1. Loads a preprocessed feature table (`normalized_original_data.csv` or the
   RF-selected `for_data.csv`) plus the graph topology table (`packet_data_ids.csv`).
2. For each random seed: splits the data (stratified train/val/test), drops
   zero-variance columns, and **re-scales features fit on the train split only**
   (leakage-safe; default = quantile transform).
3. Fits each detector on the training features only and scores val/test.
4. Chooses a decision threshold on the **validation** split, then evaluates on the
   **test** split (reports two operating points: a recall-first point and a
   balanced Youden point).
5. Combines the two models' scores via **score-level fusion**.
6. Aggregates metrics (mean ± std) across seeds and writes CSVs + figures.

### The models
| Model | Role | Owner |
|---|---|---|
| **Deep SVDD** | Headline deep one-class method (AE pretrain + hypersphere objective) | **This project (yours)** |
| **Anomal-E** | GNN centerpiece — transductive E-GraphSAGE + Deep Graph Infomax, downstream Isolation Forest on edge embeddings | **This project (yours)** |
| **Fusion** | Quantile-normalized **max** of `anomal_e + deep_svdd` | **This project (yours)** |
| k-means, autoencoder | **Intentionally NOT here** — teammate (Adam)'s scope | Separate |

> The three models in this deliverable are **Deep SVDD**, **Anomal-E** (the GNN),
> and their **Fusion**. Fusion combines the two models' normalized anomaly scores
> with a max ("OR") rule so a packet flagged by either track becomes an alert.

---

## 2. Key components and modules (`Phase_2/`)

| File | Responsibility |
|---|---|
| `config.py` | Central paths (`DATA_DIR`, `RESULTS_DIR`), column names (`id`, `label`), feature-set registry, seeds, split fractions, threshold targets. `ROOT` resolves to the repo root (`parents[1]`). |
| `data.py` | `Dataset` dataclass (X, y_bin, y_multi, ids, feature_names); `load_feature_table`, `to_dataset` (NaN/inf sanitizing), `stratified_split`, `subsample` (quick mode), `load_splits`. |
| `preprocess.py` | Feature scaling used by the pipeline: `make_scaler` (quantile / robust / standard / log_standard / none), `find_zero_variance`, `drop_features`, `apply_scaler`. **All fit on train only.** |
| `detectors/__init__.py` | `BaseDetector` ABC — uniform `fit(X)` / `score(X)` interface (higher score = more anomalous). |
| `detectors/deepsvdd.py` | `DeepSVDD` — autoencoder pretraining then one-class hypersphere objective (Ruff et al., 2018). |
| `detectors/anomal_e.py` | `AnomalE` — transductive GNN; `fit_score_transductive(train, val, test)` builds one graph over all packets, trains DGI, fits Isolation Forest on train edge embeddings. |
| `detectors/graph.py` | `build_graph`, `load_ids_table` — turns packets into a host communication graph (nodes = IPs/MACs, edges = packets). |
| `fusion.py` | `normalize_ref` (quantile/ECDF normalization against a reference distribution), `fuse` (max/mean combine). |
| `metrics.py` | `EvalResult`, `pick_threshold` (recall_first / fpr_capped / youden), `evaluate` (precision, recall, F1, FPR, FNR, AUC-ROC, PR-AUC, per-attack DR, confusion), `aggregate_results`. |
| `plots.py` | ROC / PR / confusion / per-attack-DR / UMAP-embedding figures → `results/figures/`. |
| `eda.py` | Diagnostic EDA (data quality, per-attack detectability, PCA dimensionality, graph topology). |
| `tune.py` | Grid-search hyperparameter tuning with **multi-seed** selection on validation ROC-AUC. |
| `run_phase2.py` | **Orchestrator / entry point** — ties everything together. |

---

## 3. Data flow / processing pipeline

```
Phase 1 output (data/processed_data/packet-data/)
    normalized_original_data.csv   (full feature set)
    for_data.csv                   (RF-selected ablation set)
    packet_data_ids.csv            (graph topology: src/dst ip/mac/port per packet id)
              │
              ▼
run_phase2.main()
  load_feature_table ──► to_dataset ──► (optional) subsample  [--quick]
              │
              ▼   for each seed:
  stratified_split ──► find_zero_variance (train) ──► drop_features
              │                                          │
              ▼                                          ▼
     make_scaler(...).fit(train.X)  ──►  apply_scaler(train/val/test)
              │
              ▼   for each model:
   ┌───────────────────────────────────────────────────────────────┐
   │ deep_svdd:      det.fit(train.X) → det.score(val/test)        │
   │ anomal_e (GNN): fit_score_transductive(train,val,test)        │
   │                 (one graph over all splits; IForest           │
   │                  fit on TRAIN edges only)                     │
   └───────────────────────────────────────────────────────────────┘
              │
              ▼
  pick_threshold(val) ──► evaluate(test)   [recall_first + youden points]
              │
              ▼
  fusion: fuse([anomal_e, deep_svdd]) with quantile-normalized max
              │
              ▼
  aggregate_results (mean ± std across seeds)
              │
              ▼
  results/results_<fs>_<scaler>.csv   results/summary_<fs>_<scaler>.csv
  results/figures/*.png  (ROC/PR/CM/per-attack ×3 models + Anomal-E UMAP)
```

**Leakage discipline:** zero-variance dropping and the scaler are both fit on the
**train split only** and applied to val/test; each seed refits its own scaler.
Labels never touch a detector's `fit`. For the transductive GNN, node embeddings
use edge *structure/features* (never labels), and the downstream Isolation Forest
is fit on train edges only, with the threshold chosen on validation.

---

## 4. Relationship to Phase 1

- **Coupling is a file-artifact contract only — there is no Python import between
  the phases.** Phase 1 (`Phase_1/data_preprocessing_packet.py`) writes the three
  CSVs into `data/processed_data/packet-data/`; Phase 2 reads them by path via
  `config.py`.
- The contract is: the two feature CSVs must contain an `id` column, a `label`
  column (`benign` + attack names), and numeric feature columns; `packet_data_ids.csv`
  must contain `id, src_ip, dst_ip, src_mac, dst_mac, src_port, dst_port`.
- Phase 1 already applies a global `StandardScaler`; Phase 2 **re-scales**
  (default quantile, fit-on-train) because the provided normalization was only
  partial. Because the quantile transform is rank-based, Phase 2's scaling is the
  one that actually governs the detectors.
- **Phase 3** (`Phase_3/flow_based_feature_engineering.py`) is a separate,
  flow-based track and is independent of Phase 2.

See `docs/phase2_run_guide.md` for how to run it.
