# Preprocessing leakage fix and clean rerun

Date: 2026-09-23

## Defect and fix

The previous flow dataset was imputed with per-label means, filtered by a
per-label IsolationForest over the full dataset, and scaled on the full dataset
before `prep.py` created train/validation/test splits. This leaked the target and
held-out distribution.

The corrected order is:

1. build and sample unprocessed aggregated flows;
2. create the train/validation/test split;
3. fit feature medians on training rows only, without labels;
4. apply the fixed `log1p` transform;
5. fit one label-free IsolationForest on training rows and remove training
   outliers only;
6. fit StandardScaler on the retained training rows and transform all splits.

The session-holdout evaluation repeats this fit independently inside each of
its four conditions. Its threshold subset and test sessions are transform-only.

## Headline multiclass XGBoost

Threshold selection is unchanged: maximize validation recall subject to the 1%
FPR allowance, then evaluate once on the held-out test set.

| Run | Precision | Recall | F1 | FPR | PR-AUC | Test rows |
|---|---:|---:|---:|---:|---:|---:|
| Contaminated | 0.7918 | 0.9958 | 0.8822 | 0.0082 | 0.9947 | 39,188 |
| Clean | 0.7508 | 0.9867 | 0.8527 | 0.0098 | 0.9880 | 41,200 |
| Clean - contaminated | -0.0410 | -0.0091 | -0.0295 | +0.0016 | -0.0067 | +2,012 |

Clean confusion counts: TN=39,607, FP=393, FN=16, TP=1,184. The selected
hyperparameters were `max_depth=8`, `gamma=0.1`; grid-search validation log loss
was 0.0177.

## Four-condition session holdout (binary XGBoost)

These session-holdout values belong to the binary XGBoost evaluator, not the
multiclass headline immediately above. An explicit `binary:logistic` rerun
reproduced every clean condition metric exactly; no independently verified
multiclass four-condition artifact is available.

| Condition | Old PR-AUC | Clean PR-AUC | Old recall | Clean recall | Old FPR | Clean FPR | Clean precision | Clean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3: benign random, attacks random | 0.9945 | 0.9811 | 0.9937 | 0.9812 | 0.0082 | 0.0103 | 0.6967 | 0.8149 |
| C1: benign day, attacks random | 0.7182 | 0.4848 | 0.9944 | 0.9861 | 0.3551 | 0.2880 | 0.0496 | 0.0945 |
| C2: benign random, attacks day | 0.9815 | 0.9637 | 0.9721 | 0.9425 | 0.0079 | 0.0083 | 0.7843 | 0.8562 |
| C0: benign day, attacks day | 0.2425 | 0.1250 | 0.9800 | 0.9525 | 0.3379 | 0.2535 | 0.0708 | 0.1318 |

The random-to-session-disjoint PR-AUC drop remains and grows from 0.7520 in the
contaminated run to 0.8561 in the clean run.

## What moved the metrics

- **Outlier handling was the dominant structural change.** The contaminated
  pipeline deleted 10,000 benign rows (5%) and exactly 12 rows from every attack
  class (1%) before splitting, including from validation and test. The clean
  label-free detector removed 6,180 training rows only. Because attack flows are
  often global outliers, this removed 854 of the initial 3,600 training attack
  rows while leaving all validation/test rows in place.
- **Per-label imputation was genuine target leakage but affected little data in
  this sample.** There were 486 missing/non-finite cells across 345 rows (0.17%
  of rows). Its isolated metric contribution was not separately rerun.
- **StandardScaler did not move the XGBoost result.** An empirical affine-scale
  ablation produced the same thresholded predictions and all reported metrics;
  the maximum test probability difference was 4.17e-7. This matches the expected
  scale invariance of tree splits. Fitting it on all data was still procedurally
  wrong and is now corrected.

## Reproduction

```bash
python Phase_3/flow_based_feature_engineering.py
python Phase_3/prep.py
python -c "import sys; sys.path.insert(0, 'Phase_3'); import multiclass_xgboost as m; m.run_experiment(False, 'Clean full feature set')"
python Phase_3/eval_session_holdout.py
```

Resume-safe phrasing: **Built and leakage-audited a binary XGBoost IoT IDS;
under split-first preprocessing it achieved 99.50% recall at 0.94% FPR and
0.9893 PR-AUC on a held-out random test set, while capture-session testing
reduced PR-AUC from 0.9811 to 0.1250 and drove FPR to 25.35% at 95.25% recall,
exposing capture-session shift.**
