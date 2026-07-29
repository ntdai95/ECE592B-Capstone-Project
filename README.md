# Multi-Stage IoT Intrusion Detection System

A two-stage IDS evaluated on **CIC IoT-DIAD 2024**. Stage one is unsupervised
anomaly detection at the packet level; stage two is a supervised flow-level
classifier that consumes the stage-one anomaly score as a feature.

This repository contains the Phase 3 (flow-level) pipeline, its results, and the
validity analysis that accompanies them.

---

## Headline result

Best model, held-out test set, threshold tuned on validation for FPR ≤ 1%:

| Model | Precision | Recall | F1 | FPR | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| **Multiclass XGBoost** | 0.750 | **0.925** | 0.828 | 0.96% | 0.944 | 275 | 67 |
| Blend (binary + multiclass) | 0.760 | 0.919 | 0.832 | 0.91% | 0.946 | 259 | 72 |
| Soft Voting (RF+XGB+MLP) | 0.748 | 0.917 | 0.824 | 0.96% | 0.940 | 275 | 74 |
| Stacking (LR meta) | 0.779 | 0.912 | 0.840 | 0.81% | 0.941 | 231 | 78 |
| XGBoost | 0.758 | 0.910 | 0.827 | 0.91% | 0.942 | 259 | 80 |
| Random Forest | 0.768 | 0.901 | 0.830 | 0.85% | 0.930 | 242 | 88 |
| Neural Network (MLP) | 0.747 | 0.782 | 0.764 | 0.83% | 0.833 | 236 | 194 |
| Linear SVM | 0.681 | 0.499 | 0.576 | 0.73% | 0.584 | 208 | 446 |

Six of eight models exceed 90% recall inside the 1% false-positive ceiling.

**Read this alongside the caveat below.** Under a capture-session-disjoint split
the pipeline keeps its recall only by spending 32% FPR, 32 times its budget, and
PR-AUC falls from 0.927 to 0.630. What fails first is not the model's ranking but
its calibration. That analysis is part of the contribution, not a footnote.

**Reproducibility.** Every stage is seeded, but XGBoost's `hist` tree method with
`n_jobs=-1` sums gradients in thread-completion order, so a machine with a
different core count reproduces these figures to roughly ±0.5 points of recall and
±0.1 points of FPR rather than exactly. Model ordering, the budget compliance and
every conclusion below are stable across machines; the third decimal is not.

---

## The capture-session caveat

Benign traffic in IoT-DIAD was recorded only on 2022-10-07/08, while each attack
class occupies its own separate capture days. A model can therefore score well by
recognising the capture session rather than the attack. Capture window is
predictable from the flow features alone at ROC-AUC 0.922.

`eval_session_holdout.py` runs four conditions to isolate this:

| Condition | Split | PR-AUC | Recall | Achieved FPR |
|---|---|---|---|---|
| C3 reference | benign random, attacks random | 0.927 | 90.5% | 0.97% |
| C1 benign shift | benign **by day**, attacks random | 0.602 | 93.1% | **32.43%** |
| C2 attack shift | benign random, attacks **by day** | 0.903 | 86.1% | 0.89% |
| **C0 honest** | benign **by day**, attacks **by day** | 0.630 | 90.7% | **32.13%** |

Every threshold here is calibrated on a held-out slice of the *training* capture
day, never on the test day. Reading recall alone would suggest C0 beats C3, which
is nonsense; the achieved FPR column is what makes the rows comparable.

The pattern is sharper than a general "it generalises worse" result. The two
conditions that hold their budget (C3, C2) are exactly the two where benign
traffic is split randomly. Both conditions that blow it (C1, C0) are the ones
where the benign capture day changes. Shifting the *attack* days costs recall
(C2: 90.5% → 86.1%) but leaves calibration intact; shifting the *benign* day
leaves ranking largely intact but destroys the operating point. For a deployed
sensor that is the more serious failure: the model still ranks attacks above
benign, but the threshold that bought 1% FPR on one day buys 32% on the next.

The random-split numbers are correct under the stated protocol and are what is
comparable to published IoT-DIAD baselines. They should not be reported alone.

---

## What makes this pipeline different

**Multi-flow context features.** Brute force, XSS and DNS spoofing are multi-flow
phenomena: a single password-guessing connection is indistinguishable from a
single normal login, so per-flow models plateau on those classes. 23 causal
connection-window features were added (`build_context_features.py`), following
the KDD'99 `count` / `srv_count` / `same_srv_rate` family. Per source IP and per
destination IP, over trailing 2s and 60s windows: connection count, distinct
destination ports, distinct destination IPs, same-service rate, same-destination
rate, plus three statistics over the previous 100 connections.

`exp_context.py` isolates their contribution by training one model (XGBoost) on
the base features and on base + context, under both split regimes:

| Regime | Base PR-AUC | + context | Δ | Base recall | + context |
|---|---|---|---|---|---|
| Random split | 0.787 | **0.942** | +0.155 | 72.4% | 91.0% |
| Session-disjoint | 0.141 | **0.630** | +0.489 | 54.0% | 90.7% |

PR-AUC leads the table because it is threshold-free. Under the session-disjoint
regime neither arm holds the 1% budget (14.0% and 32.1% achieved FPR), so the two
recalls sit at different operating points and their difference is not a
like-for-like comparison; PR-AUC is.

The gain is larger under the honest split than the random one, which is the
argument that these features measure attack behaviour rather than the calendar —
the opposite of what leakage looks like. Brute force is the exception and gains
less under the session split than the random one; it is reported as such rather
than dropped. Context features carry 33.3% of total model importance. Verified as
legitimate rather than leaky by `verify_context_integrity.py`:

1. **Causal** — rebuilding features after truncating the capture at 60% leaves all
   531,975 earlier flows' features bit-identical. No future information enters.
2. **Label-free** — the builder reads only Flow ID, IP, port and timestamp.
3. **Budget** — every model's test FPR is inside the 1% ceiling (0.73–0.96%).

Windows are computed over the full 886,621-flow capture because a live sensor
observes all prior traffic regardless of which flows land in the train/test
partition. The split controls what the classifier is *fitted* on, not what the
sensor saw. Raw destination port remains excluded as a feature; it enters only in
aggregate form.

**What the Task 3.1 anomaly scores actually contribute.** The three columns
carried over from stage one (`flow_anomaly_score`, `flow_anomaly_flag`,
`has_anomaly_score`) sit high in the XGBoost gain ranking — `flow_anomaly_score`
places 3rd of 102 — which reads as stage one carrying the classifier. It does
not. The `no_anomaly` arm of `exp_context.py` drops all three and holds
everything else fixed:

| Regime | Full set | Minus Task 3.1 | Δ recall | Δ PR-AUC |
|---|---|---|---|---|
| Random split | 91.0% | 90.9% | −0.1 | −0.005 |
| Session-disjoint | 90.7% | 90.8% | +0.1 | −0.043 |

One model is a weak basis for that claim, so `train_models.py --drop-anomaly`
retrains **all eight** leaderboard models with the three columns removed and
nothing else changed, saving under `<model>__no_anomaly` keys. The answer turns
out to depend on the model, so it is reported split rather than averaged:

| Model | Recall with | Without | Δ | PR-AUC with | Without | Δ |
|---|---|---|---|---|---|---|
| Multiclass XGBoost | 92.5% | 91.1% | −1.3 | 0.944 | 0.939 | −0.005 |
| Blend (binary + multiclass) | 91.9% | 91.5% | −0.5 | 0.946 | 0.941 | −0.005 |
| Soft Voting (RF+XGB+MLP) | 91.7% | 90.9% | −0.8 | 0.940 | 0.934 | −0.006 |
| Stacking (LR meta) | 91.2% | 91.5% | +0.2 | 0.941 | 0.936 | −0.004 |
| XGBoost | 91.0% | 90.9% | −0.1 | 0.942 | 0.937 | −0.005 |
| Random Forest | 90.1% | 90.8% | +0.7 | 0.930 | 0.930 | −0.000 |
| Neural Network (MLP) | 78.2% | 77.1% | −1.1 | 0.833 | 0.824 | −0.008 |
| Linear SVM | 49.9% | 46.9% | **−3.0** | 0.584 | 0.555 | −0.029 |

For the six models above 85% recall the change spans −1.3 to +0.7 and is not
consistent in sign. Rerunning the ablation moves individual models by a
comparable amount — XGBoost's ablated arm scored 91.7% on one run and 90.9% on
the next from the same seed on the same machine — so that band is run-to-run
spread, not effect. For the two weakest models it is consistently negative and
several times larger.

So the columns do carry signal. It is signal a strong learner already recovers
from the flow and context features on its own, and only the models that cannot
extract it themselves still depend on it. The honest answer to whether stage one
helps stage two is yes, redundantly.

Their gain ranking is separately unstable in a way the flow and context features
are not: `has_anomaly_score` placed 2nd on one machine and 45th on another from
the same seed. Gain rewards a feature for splitting cleanly, not for covering
many rows, and these cover 7.4% of flows. Coverage is further uneven in a way
traffic volume does not explain (XSS 40.9%, brute force 36.7%, benign 7.1%, DoS
0.8%, DDoS 0.5%), so `has_anomaly_score` is closer to an indicator of which flows
the stage-one sampling happened to reach than a behavioural signal. Read the
ablation table, not the importance rank.

**Neyman–Pearson operating point.** The brief fixes FPR ≤ 1% as a hard constraint,
so the threshold maximises recall subject to that constraint rather than
maximising F1. Maximising F1 under a 3% attack base rate stops at roughly 0.15%
FPR, spending about a fifth of the allowance. Both operating points are computed
and saved so the trade-off can be shown.

**Clopper–Pearson false-positive allowance.** Allowing a flat `1% x n_benign`
false positives fits the threshold to the validation benign tail and the budget
then leaks on test: re-scoring all eight models under the flat rule breaches the
ceiling on five of them, from 1.03% (stacking) to 1.10% (soft voting), and lands
random forest and MLP within three hundredths of a point of it. The allowance is
instead the largest count whose Clopper–Pearson 99% upper bound still sits inside
1%. That costs between 0.3 and 1.1 points of recall and no model exceeds the
budget. The breach is not a rounding artefact — it is the threshold being fitted
to the shape of the validation benign tail, so it recurs on every rerun.

---

## Repository layout

```
.
├── Phase_1/                        # packet-level preprocessing and EDA
├── Phase_2/                        # unsupervised packet-level detectors
├── Phase_3/                        # flow-level pipeline, notebooks and cascade
├── data/                           # not in the repository, git-ignored
│   ├── raw_data/
│   └── processed_data/
│       ├── packet-data/
│       └── flow-data/
├── results/                        # not in the repository, git-ignored
├── requirements.txt
├── README.md
└── LICENSE
```

`data/` and `results/` are git-ignored: the datasets are not redistributable and
every result file is regenerated by the pipeline. See "Running it" below.

### Phase 1 and Phase 2 code

| Script | Purpose |
|---|---|
| `Phase_1/data_preprocessing_packet.py` | Packet-level parsing, cleaning and feature extraction |
| `Phase_2/run_phase2.py` | Phase 2 orchestrator: run detectors, select thresholds, evaluate, report |
| `Phase_2/config.py` | Paths, column conventions and shared constants |
| `Phase_2/data.py`, `Phase_2/preprocess.py` | Loading, label encoding, stratified splits, feature scaling |
| `Phase_2/eda.py`, `Phase_2/plots.py` | Exploratory analysis driving model selection, and the report visuals |
| `Phase_2/metrics.py` | Evaluation metrics and threshold selection for the anomaly detectors |
| `Phase_2/detectors/anomal_e.py`, `Phase_2/detectors/graph.py` | Anomal-E, the self-supervised edge-feature GNN, and its communication-graph construction |
| `Phase_2/detectors/deepsvdd.py` | Deep SVDD, one-class deep anomaly detection |
| `Phase_2/fusion.py` | Score-level fusion of the GNN track and the feature-space track |
| `Phase_2/tune.py` | Hyperparameter tuning for Anomal-E and Deep SVDD |
| `Phase_2/phase2_results.py` | Records packet-level detector results into `results.json` under the `phase2_*` keys |

The `Phase_2/` notebooks (`kmeans.ipynb`, `isolation_forest.ipynb`,
`autoencoders.ipynb`, and the `Unsupervised_Learning_*` pair) hold the
feature-space detector track: the k-means and Isolation Forest sweeps and the
autoencoder reconstruction-error baseline. `kmeans.ipynb` and
`isolation_forest.ipynb` each end by writing their test result and
hyperparameter sweep into `results/results.json`, which is what feeds the Phase 2
tables above; both need the packet sample under `data/samples/` to run.

### Phase 3 code

| Script | Purpose |
|---|---|
| `run_all.py` | Full pipeline, then prints every results table |
| `show_results.py` | Prints all results from saved files and writes the two CSVs, retrains nothing |
| `flow_based_feature_engineering.py` | Builds the flow table and the Phase 2 → Phase 3 link |
| `build_context_features.py` | The 23 causal connection-window features |
| `prep.py` | Merge, split, impute, save `splits.npz` |
| `common.py` | Paths, metrics, threshold rules, Clopper–Pearson allowance, `results.json` |
| `train_models.py` | All eight models, `--models rf xgb svm mlp ensemble multiclass`, `--drop-anomaly` for the Task 3.1 ablation |
| `eval_session_holdout.py` | Four-condition capture-day analysis, `--features all\|base` |
| `exp_context.py` | Context features under random vs session split |
| `verify_context_integrity.py` | The causality / label-free / budget audit |
| `diagnose_hard_classes.py` | Per-class score distributions and AUC |
| `flow_preprocessing.py`, `flow_sampling.py`, `packet_flow_link.py` | Shared helpers for the notebooks and the cascade |
| `generate_flow_sample.py`, `prepare_flow_dataset.py`, `inspect_dataset.py` | Notebook dataset preparation |
| `two_stage_cascade.py` | Packet stage → flow stage end-to-end run |
| `signature_based_ids.py` | Rule-based signature detector used as the non-learned comparison point |
| `Phase3_1_Flow_Anomaly_Detection.ipynb` | Task 3.1: the flow-level Isolation Forest / k-means anomaly scores consumed as the three anomaly columns |

Everything the pipeline writes lands in `results/`: `results.json` (all eight
models, all three operating points), `leaderboard.csv`, `per_attack_detection.csv`,
`feature_importance.csv`, `hard_class_diagnostics.json`, `exp_context_results.json`,
`prep_meta.json` and `session_holdout*.json`.

---

## Running it

### 1. Data

The datasets are not in this repository (~1.6 GB, and the source is not
redistributable). Download **CIC IoT-DIAD 2024** from the Canadian Institute for
Cybersecurity and unpack it as:

```
data/raw_data/flow-based-features/     # the per-attack *.pcap_Flow.csv exports
data/raw_data/packet-based-features/
```

Run Phase 1 and Phase 2 first, then `Phase_3/flow_based_feature_engineering.py`,
which writes the three files the Phase 3 pipeline consumes:

```
data/processed_data/flow-data/normalized_original_data.csv
data/processed_data/flow-data/flow_data_ids.csv
data/processed_data/flow-data/task3_1_flow_anomaly_scores.csv
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python Phase_3/run_all.py                    # full rebuild, ~30 min, prints all tables
python Phase_3/run_all.py --skip-existing    # only run stages whose output is missing
python Phase_3/show_results.py               # just print saved results, ~1 second
python Phase_3/verify_context_integrity.py   # the legitimacy audit on its own
```

The last stage, `verify_context_integrity.py`, is the slowest single step after
the training runs: its causality check rebuilds the context features from
scratch over the truncated capture in order to compare them.

`run_all.py` must start with `build_context_features.py`. Skipping it drops the
pipeline back to the 79-feature per-flow set and `prep.py` will refuse to run.

Each stage can also be run on its own, provided the stages it depends on have
already produced their output. In order:

| Stage | Writes into `results/` |
|---|---|
| `build_context_features.py` | `context_features.parquet` |
| `prep.py` | `splits.npz`, `prep_meta.json` |
| `train_models.py` | `results.json` → all eight models |
| `train_models.py --drop-anomaly` | `results.json` → the eight `__no_anomaly` twins |
| `diagnose_hard_classes.py` | `hard_class_diagnostics.json` |
| `eval_session_holdout.py` | `session_holdout_results.json` |
| `eval_session_holdout.py --features base` | `session_holdout_base_results.json` |
| `exp_context.py` | `exp_context_results.json` |
| `verify_context_integrity.py` | `feature_importance.csv` |

`train_models.py` runs the six trainers in dependency order, since the ensembles
load the four fitted base models off disk and the blend loads the binary XGBoost.
A single family can be rerun with `--models xgb`.

Every model writes into one `results.json`, keyed by model name, so a rerun of a
single trainer updates its own key and leaves the others alone. `--skip-existing`
checks for the key rather than for a file.

`show_results.py` writes `leaderboard.csv` and `per_attack_detection.csv` as it
prints, and `run_all.py` calls it last, so a full run leaves every file listed in
the layout above on disk.

Rerun the whole pipeline rather than individual stages when you want a
publishable set of numbers. The ensembles load the saved `.joblib` models, so
retraining one model on its own leaves the ensembles built on the previous one.

`verify_context_integrity.py` is both the legitimacy audit and the last pipeline
stage: it prints the causality, label-free and FPR-budget checks, and writes the
feature importances that section 4 of `show_results.py` reads. It loads
`model_xgb.joblib`, so the training stages must have run first.

### 4. The notebooks

The notebooks in `Phase_3/` are the exploratory track that preceded the pipeline
scripts: one notebook per model family. They are kept because they carry the
tuning searches and diagnostic plots the pipeline scripts do not.

They read a differently prepared dataset and will not run against
`data/processed_data/flow-data/`. Build their inputs first, from the raw
per-attack CIC exports (`BenignTraffic.pcap_Flow.csv`,
`DDoS-HTTP_Flood-.pcap_Flow.csv`, `DoS-HTTP_Flood.pcap_Flow.csv`,
`DNS_Spoofing.pcap_Flow.csv`, `DictionaryBruteForce.pcap_Flow.csv`,
`XSS.pcap_Flow.csv`):

```bash
python Phase_3/inspect_dataset.py \
       --data-dir data/raw_data/flow-based-features \
       --out-dir results/inventory
python Phase_3/generate_flow_sample.py \
       --data-dir data/raw_data/flow-based-features \
       --output-path data/processed_data/samples/flow_sample_seed1.csv --seed 1
python Phase_3/prepare_flow_dataset.py \
       --input-path data/processed_data/samples/flow_sample_seed1.csv \
       --output-dir data/processed_data/phase3
```

That writes `flow_train.csv`, `flow_validation.csv`, `flow_test.csv` and
`flow_split_metadata.json` into `data/processed_data/phase3/`, which every model
notebook loads.

`two_stage_cascade.ipynb` only renders results — produce them first with
`python Phase_3/two_stage_cascade.py`, which additionally needs a packet sample.

These splits are stratified 70/15/15 at seed 1 and share no flow key across
train, validation and test, but they are *not* the same splits the pipeline
uses, so notebook and pipeline numbers are not directly comparable.

---

## Reproducibility notes

Every split and model is seeded with `random_state=1`, and repeated runs on the
same machine reproduce the tables above exactly. Across machines expect movement
in the third decimal, since the tree libraries are sensitive to thread count.
One observed run differed by up to 0.007 recall on a single model. That is well
inside the spread that matters here, but it does mean a rebuild can produce
figures that differ slightly from the ones committed to this repository.

Preprocessing is fitted on the training split only. Median imputation was
originally fitted over the whole dataset before splitting, which leaked test
information into preprocessing; this was corrected.

The base-features and base+context runs use the same split, seed and protocol,
so the feature-set comparison is like for like. Both are produced in one run by
`exp_context.py` rather than being carried forward from an earlier version of
the pipeline, so the comparison is regenerated from source every time.

Every figure quoted in this README was verified against its source file in
`results/`: the headline table against `leaderboard.csv`, the
capture-session table against `session_holdout_results.json`, the context
comparison against `exp_context_results.json`, and the dataset and feature
counts against `prep_meta.json`.

---

## Dataset

Canadian Institute for Cybersecurity, **CIC IoT-DIAD 2024**.
195,953 flows after preprocessing, 3.0% attack. Classes: benign, brute force,
DDoS, DNS spoofing, DoS, XSS.
