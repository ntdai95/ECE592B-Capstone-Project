# Multi-Stage IoT Intrusion Detection System

A two-stage IDS evaluated on **CIC IoT-DIAD 2024**. Stage one is unsupervised
anomaly detection at the packet level; stage two is a supervised flow-level
classifier that consumes the stage-one anomaly score as a feature.

The whole project runs from a single entry point, `main.py`, which sequences
Phase 1 (packet preprocessing), Phase 2 (unsupervised packet detection) and
Phase 3 (supervised flow classification and its validity analysis).

```bash
pip install -r requirements.txt
python main.py                  # run everything
python main.py --skip-existing  # only run stages whose output is missing
python main.py --phase 3        # run one phase in isolation
```

---

## Headline result

All six models have now been regenerated on the same held-out test set through
the corrected split-first pipeline. Imputation, label-free outlier handling,
and scaling are fitted on training rows only; each threshold is tuned on the
validation set for FPR <= 1%. Historical columns are the invalid pre-fix values
and are retained only to make the effect of the correction explicit:

| Model (clean recall rank) | Clean precision | Old precision | Clean recall | Old recall | Clean F1 | Old F1 | Clean FPR | Old FPR | Clean PR-AUC | Old PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Binary XGBoost** | **0.7610** | 0.7849 | **0.9950** | 0.9949 | **0.8624** | 0.8775 | **0.94%** | 0.85% | **0.9893** | 0.9957 |
| 2. Multiclass XGBoost | 0.7508 | 0.7918 | 0.9867 | 0.9958 | 0.8527 | 0.8822 | 0.98% | 0.82% | 0.9880 | 0.9947 |
| 3. Soft Voting (RF+XGB+MLP) | 0.7732 | 0.7870 | 0.9858 | 0.9924 | 0.8667 | 0.8779 | 0.87% | 0.84% | 0.9842 | 0.9947 |
| 4. Random Forest | 0.7574 | 0.7754 | 0.9833 | 0.9941 | 0.8557 | 0.8713 | 0.94% | 0.90% | 0.9826 | 0.9929 |
| 5. Neural Network (MLP) | 0.7615 | 0.7879 | 0.8942 | 0.9630 | 0.8225 | 0.8667 | 0.84% | 0.81% | 0.8825 | 0.9681 |
| 6. Linear SVM | 0.6204 | 0.6783 | 0.4617 | 0.5059 | 0.5294 | 0.5796 | 0.85% | 0.75% | 0.5824 | 0.6622 |

The former results were invalid because imputation and outlier removal used the
label before splitting, and scaling was fit on the full dataset. Four of the six
clean models exceed 90% recall, and all six held-out FPRs remain below 1%.

**The selection changed.** The contaminated recall ordering was Multiclass
XGBoost > binary XGBoost > Random Forest > Soft Voting > MLP > Linear SVM. The
clean ordering is binary XGBoost > Multiclass XGBoost > Soft Voting > Random
Forest > MLP > Linear SVM. Under the stated highest-recall selection rule,
binary XGBoost now leads multiclass XGBoost by 0.83 percentage points, so the
old claim that multiclass XGBoost was the selected model is no longer valid.

The six models are: the multiclass XGBoost signature detector
(`multiclass_xgboost.py`), and five binary classifiers (`train_models.py`) — Random Forest, binary XGBoost,
Linear SVM, a 128→64 MLP, and a soft-voting ensemble of RF+XGB+MLP.

**Read this alongside the caveat below.** For the selected binary XGBoost, a
capture-session-disjoint split keeps high recall only by spending 25.35% FPR,
over 25 times its budget, and PR-AUC falls from 0.9811 to 0.1250. What fails
first is not the model's ranking but its calibration.

---

## The capture-session caveat

Benign traffic in IoT-DIAD was recorded only on 2022-10-07/08, while each attack
class occupies its own separate capture days. A model can therefore score well by
recognising the capture session rather than the attack — the FPR blowout below is
the direct evidence: the same threshold that holds 1% FPR on one benign day costs
25–29% on another.

`eval_session_holdout.py` now makes the binary objective explicit and writes the
model-identified artifact `session_holdout_binary_xgb_results.json`. Its four
conditions are:

| Condition | Split | Binary XGB PR-AUC | Binary XGB recall | Binary XGB FPR |
|---|---|---:|---:|---:|
| C3 reference | benign random, attacks random | 0.9811 | 98.12% | 1.03% |
| C1 benign shift | benign **by day**, attacks random | 0.4848 | 98.61% | **28.80%** |
| C2 attack shift | benign random, attacks **by day** | 0.9637 | 94.25% | 0.83% |
| **C0 honest** | benign **by day**, attacks **by day** | **0.1250** | 95.25% | **25.35%** |

The retained legacy `session_holdout_results.json` contains the same values
previously cited as multiclass. The explicit binary rerun matches all of its
condition metrics, counts, per-attack detection rates, and per-class AUCs
exactly. Moreover, repository history shows that the evaluator already formed a
binary target and trained a binary classifier. The legacy file therefore cannot
serve as independent evidence of a multiclass holdout; a separate multiclass
four-condition run is **not verified** and its values must not be contrasted
with the binary table as if they came from a different model.

Every threshold here is calibrated on a held-out slice of the *training* capture
day, never on the test day, so recall is always quoted with the FPR it actually
cost. The two conditions that hold their budget (C3, C2) are exactly the two
where benign traffic is split randomly; both that blow it (C1, C0) are the ones
where the benign capture day changes. For a deployed sensor that is the more
serious failure: the model still ranks attacks above benign, but the threshold
that bought 1% FPR on one day costs 25–29% on the next.

Even the random-split result must not be reported alone; the session-disjoint
result is the stronger test of generalisation.

---

## What makes this pipeline different

**One shared, altered dataset for all six models.** Every model — the multiclass
signature detector and the five binary classifiers — trains on the same feature
table and the same stratified 60/20/20 split, so they are directly comparable and
no model grades itself on a different rule. The table is the flow-level sample
(200k benign + a few thousand attack flows, aggregated per Flow ID) plus two
engineered blocks: the 23 connection-window context features and the three
Task 3.1 anomaly columns.

**Multi-flow context features.** Brute force, XSS and DNS spoofing are multi-flow
phenomena: a single password-guessing connection is indistinguishable from a
single normal login, so per-flow models plateau on those classes. 23 causal
connection-window features (`build_context_features.py`) follow the KDD'99
`count` / `srv_count` / `same_srv_rate` family — per source and destination IP,
over trailing 2s and 60s windows: connection count, distinct destination ports,
distinct destination IPs, same-service rate, same-destination rate, plus three
statistics over the previous 100 connections.

`exp_context.py` isolates their contribution by training XGBoost on base features
vs base + context, under both split regimes:

> **Historical only:** this context ablation predates the preprocessing fix and
> has **NOT BEEN REGENERATED**. Do not cite it as a clean result.

| Regime | Base PR-AUC | + context | Δ | Base recall | + context |
|---|---|---|---|---|---|
| Random split | 0.776 | **0.996** | +0.220 | 70.3% | 99.5% |
| Session-disjoint | 0.178 | **0.242** | +0.064 | 60.8% | 98.0% |

Recall improves substantially under both regimes (+29.2 points random, +37.2
points session-disjoint), but the PR-AUC gain this run is concentrated in the
random split. That is a different picture from an earlier run of this pipeline,
where the honest-split PR-AUC gain exceeded the random-split one — the strongest
form of the "not leakage" argument does not hold on this draw of the dataset, and
this README will not claim it does. What still holds: `verify_context_integrity.py`
audits the builder itself as causal (rebuilding after truncating the capture at
60% leaves earlier flows' features bit-identical) and label-free (it reads only
Flow ID, IP, port and timestamp, never the label column), so the computation is
not leaking future or label information. The capture-session confound above is a
property of this specific benign/attack sample, not a bug in how these features
are built — read the recall gain as the honest headline for what context adds,
not the PR-AUC gain.

**Task 3.1 ablation across every model.** The three anomaly columns carried over
from stage one sit high in the XGBoost gain ranking, which reads as stage one
carrying the classifier. `train_models.py --drop-anomaly` and
`xgboost.py --drop-anomaly` retrain every model with those columns removed and
nothing else changed, saving under `<model>__no_anomaly` keys. The strong models
barely move; only the weak linear and neural baselines lose several points. So
the columns carry signal, but it is signal a strong learner already recovers from
the flow and context features on its own.

**Neyman–Pearson operating point + Clopper–Pearson allowance.** The brief fixes
FPR ≤ 1% as a hard constraint, so the threshold maximises recall subject to that
constraint (`evaluation.py`). The permitted false-positive count is the largest
whose Clopper–Pearson 99% upper bound still sits inside 1%, so the budget holds on
test rather than only on the validation sample it was tuned on.

**Two-stage system (`two_stage_system.py`).** The combined cascade re-checks the
autoencoder's alerts with stage 2 and calls an item an attack only if stage 1
alerts it *and* stage 2 confirms it, then reports overall accuracy, precision,
recall and the false-positive reduction versus Phase 2. Stage 2 is the
leaderboard's highest-recall model, selected automatically (now binary
XGBoost). The saved cascade result predates this completed six-model rerun,
still names multiclass XGBoost, and has not been regenerated under the
split-first preprocessing protocol. It is not part of the clean six-model table
and must not be cited as a current clean result.

---

## Repository layout

```
.
├── main.py                          # single entry point: runs Phase 1 → 2 → 3
├── Phase_1/
│   └── data_preprocessing_packet.py # packet-level parsing, cleaning, sampling
├── Phase_2/                         # unsupervised packet-level detectors
├── Phase_3/                         # flow-level pipeline and validity analysis
├── notebooks/                       # exploratory notebooks + converted scripts
├── docs/                            # Phase 2 write-ups
├── data/                            # not in the repository, git-ignored
├── results/                         # not in the repository, git-ignored
├── requirements.txt
└── README.md
```

`data/` and `results/` are git-ignored: the datasets are not redistributable and
every result file is regenerated by the pipeline.

### Phase 1 and Phase 2

| Script | Purpose |
|---|---|
| `Phase_1/data_preprocessing_packet.py` | Packet-level parsing, cleaning, sampling (Task 1.1/1.2) |
| `Phase_2/kmean_autoencoder.py` | k-means + autoencoder trained on benign only; exports the alert rows Phase 3 consumes |
| `Phase_2/run_phase2.py` | Orchestrates the Deep SVDD / Anomal-E / fusion detector suite |
| `Phase_2/detectors/deepsvdd.py`, `detectors/anomal_e.py`, `detectors/graph.py` | Deep SVDD, the self-supervised edge-feature GNN, and its graph construction |
| `Phase_2/fusion.py` | Score-level fusion of the GNN and feature-space tracks |
| `Phase_2/config.py`, `data.py`, `preprocess.py`, `metrics.py`, `eda.py`, `plots.py`, `tune.py`, `phase2_results.py` | Config, loading, scaling, metrics, EDA, figures, tuning, results recording |

### Phase 3

| Script | Purpose |
|---|---|
| `flow_based_feature_engineering.py` | Builds the aggregated flow table, the Phase 2 → Phase 3 link, and the altered dataset |
| `flow_anomaly_detection.py` | Task 3.1: flow-level anomaly scores (Isolation Forest / k-means) → the three anomaly columns |
| `build_context_features.py` | The 23 causal connection-window context features |
| `prep.py` | Merge flow + context + anomaly features, split, impute (train-only), save `splits.npz` |
| `evaluation.py` | Shared metrics, threshold rules, Clopper–Pearson allowance, `results.json` |
| `multiclass_xgboost.py` | Task 3.3 multiclass XGBoost signature detector (per-class report, confusion matrix, ROC) |
| `train_models.py` | The five binary models: RF, XGBoost, Linear SVM, MLP, Soft Voting; `--drop-anomaly` for the Task 3.1 ablation |
| `two_stage_system.py` | Combined cascade: stage 1 alerts re-checked by the best stage-2 model; overall accuracy + FP reduction |
| `eval_session_holdout.py` | Binary-XGBoost four-condition capture-day analysis; `--features all\|base` |
| `exp_context.py` | Context features under random vs session split |
| `verify_context_integrity.py` | Causality / label-free / budget audit |
| `diagnose_hard_classes.py` | Per-class score distributions and AUC |
| `run_all.py` | Runs the Phase 3 stages in order; `show_results.py` prints every table |

### Notebooks

`notebooks/` holds the two exploratory notebooks kept for their tuning searches
and diagnostic plots — `Phase3_1_Flow_Anomaly_Detection.ipynb` (Task 3.1) and
`Unsupervised_Learning_..._BenighOnlyTraining_v2.ipynb` (Phase 2) — whose runnable
`.py` versions live in `Phase_3/flow_anomaly_detection.py` and
`Phase_2/kmean_autoencoder.py`.

---

## Running it

### 1. Data

The datasets are not in this repository (~1.6 GB; the source is not
redistributable). Download **CIC IoT-DIAD 2024** from the Canadian Institute for
Cybersecurity and unpack the packet-level and flow-level files as:

```
data/raw_data/flow-based-features/     # the per-attack *.pcap_Flow.csv exports
data/raw_data/packet-based-features/
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py                  # full pipeline, prints every results table at the end
python main.py --skip-existing  # only run stages whose output is missing
python main.py --phase 3        # just the flow-level phase
```

`main.py` runs, in order: Phase 1 preprocessing → the Phase 2 autoencoder (which
exports the Phase 3 alert rows) and detector suite → Phase 3 flow feature
engineering → Task 3.1 anomaly scores → context features, the shared split, the
six classifiers, the Task 3.1 ablation, the validity analysis, and the printed
tables. Each stage is a standalone script and can be run on its own once the
stages it depends on have produced their output.

Every result lands in `results/`: `results.json` (all models, all operating
points), `leaderboard.csv`, `per_attack_detection.csv`, `feature_importance.csv`,
`hard_class_diagnostics.json`, `exp_context_results.json`, `prep_meta.json` and
`session_holdout*.json` (including the model-identified binary-XGBoost run).

To reprint saved results without retraining: `python Phase_3/show_results.py`.

---

## Reproducibility notes

Every split and model is seeded with `random_state=1`, and repeated runs on the
same machine, on the same downloaded copy of the dataset, reproduce the tables
above closely. Across machines expect movement in the third decimal, since the
tree libraries are sensitive to thread count.

A fresh download of the dataset is a different story: re-running this pipeline
against an independently downloaded copy of CIC IoT-DIAD 2024 moved the headline
numbers well past third-decimal noise (recall climbed from the low 90s into the
high 90s, and the capture-session PR-AUC gap widened). Phase 1's random,
seeded packet sample matched a prior run's counts exactly, so the packet-level
captures are stable; the flow-level `.pcap_Flow.csv` exports are a separate
download from CIC and are the more likely source of the difference. The
integrity audit (causality, label-free, FPR budget) passes either way — what
moves is the data, not the method.

Preprocessing is fitted on the training split only — median imputation over the
whole dataset before splitting would leak test information into preprocessing.
The base-features and base+context runs use the same split, seed and protocol, so
the feature-set comparison is like for like, and both are produced in one run by
`exp_context.py` rather than carried forward from an earlier version.

---

## Dataset

Canadian Institute for Cybersecurity, **CIC IoT-DIAD 2024**.
~196,000 flows after preprocessing, ~3% attack. Classes: benign, brute force,
DDoS-HTTP Flood, DNS spoofing, DoS-HTTP Flood, XSS.
