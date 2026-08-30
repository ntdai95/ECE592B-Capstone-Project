# Phase 2 — Run Guide

Step-by-step instructions to run Phase 2 (`Phase_2/` package). All commands are
run **from the repository root** (`ECE592B-Capstone-Project/`).

---

## 1. Prerequisites

### 1a. Input data must exist
Phase 2 reads three files from `data/processed_data/packet-data/`:

| File | Purpose | Approx size |
|---|---|---|
| `normalized_original_data.csv` | Full feature set (default) | ~411 MB |
| `for_data.csv` | RF-selected ablation feature set | ~365 MB |
| `packet_data_ids.csv` | Graph topology for the GNN | ~33 MB |

If they are missing, regenerate them with Phase 1:
```bash
python Phase_1/data_preprocessing_packet.py
```
(Phase 1 needs the raw captures under `data/raw_data/packet-based-features/`.)

### 1b. Python environment
The project uses a virtual environment at the **repo root: `.venv/`**.
It is built on the system Python 3.11 and carries the Phase-2 dependencies
(torch, numpy, pandas, scikit-learn, matplotlib inherited from the base Python;
torch-geometric, pyod, umap-learn installed into the venv).

Interpreter path (Windows):
```
.venv/Scripts/python.exe
```

If `.venv/` does not exist yet, create it (see §5).

---

## 2. Quick start (smoke test, ~1–2 min)

Confirms everything works on a 20k-row stratified subsample, one seed, no figures:

```bash
# from the repo root, using the project venv
MPLCONFIGDIR="$PWD/.mplcache" \
.venv/Scripts/python.exe -m Phase_2.run_phase2 \
    --quick 20000 --models deep_svdd anomal_e fusion \
    --seeds 0 --no-figures
```

PowerShell equivalent:
```powershell
$env:MPLCONFIGDIR = "$PWD\.mplcache"
& ".venv\Scripts\python.exe" -m Phase_2.run_phase2 `
    --quick 20000 --models deep_svdd anomal_e fusion `
    --seeds 0 --no-figures
```

Expected: a per-model line for each of `deep_svdd`, `anomal_e`, and `fusion`
with `auc=...`, then `Saved per-seed metrics -> results/...csv`.

---

## 3. Common run recipes

```bash
PY=".venv/Scripts/python.exe"

# All three models (Deep SVDD + GNN + Fusion), full data, default seeds:
$PY -m Phase_2.run_phase2 --models all

# All models + figures, TWO seeds (fits in RAM on this machine):
$PY -m Phase_2.run_phase2 --models all --seeds 0 1

# The two individual models only (no fusion):
$PY -m Phase_2.run_phase2 --models deep_svdd anomal_e

# Ablation on the RF-selected feature set:
$PY -m Phase_2.run_phase2 --feature-set for_data --models all --seeds 0 1

# Reproduce the "broken" unscaled run (no feature scaling):
$PY -m Phase_2.run_phase2 --scaler none --models deep_svdd anomal_e --seeds 0

# Hyperparameter tuning (multi-seed selection on validation AUC):
$PY -m Phase_2.tune --models anomal_e deep_svdd --seeds 0 1 2

# Diagnostic EDA:
$PY -m Phase_2.eda --quick
```

Models: `deep_svdd`, `anomal_e`, `fusion` (`--models all` runs all three; fusion
combines `anomal_e` + `deep_svdd`).

Key flags (`run_phase2.py`): `--feature-set {normalized,for_data}`,
`--models <names…|all>`, `--seeds <ints>`, `--quick <N>`,
`--strategy {recall_first,fpr_capped,youden}`,
`--scaler {quantile,robust,standard,log_standard,none}`,
`--no-figures`, `--no-tuned`.

---

## 4. Expected outputs

Written to `results/` (all gitignored):

| Output | Description |
|---|---|
| `results/results_<featureset>_<scaler>.csv` | Per-model, per-seed metrics |
| `results/summary_<featureset>_<scaler>.csv` | Mean ± std across seeds, per model |
| `results/figures/roc_*.png`, `pr_*.png`, `cm_*.png`, `dr_*.png` | ROC, PR, confusion, per-attack detection rate (one set per model) |
| `results/figures/umap_anomal_e_*.png` | UMAP of the GNN's edge embeddings |

A full all-models run produces **13 figures** (4 × 3 models + 1 UMAP).
Console output prints each model's AUC/PR-AUC and both operating points, then the
mean ± std summary. Representative full-data numbers from a full re-run against a
freshly downloaded copy of the dataset (30 Aug 2026, 5 seeds): Anomal-E ≈ 0.64 AUC,
Deep SVDD ≈ 0.68, fusion ≈ 0.70. These move with the specific dataset download and
random sample drawn — see the top-level README's reproducibility notes.

---

## 5. Building the environment (`.venv`)

The project's `.venv/` was created by inheriting the system Python 3.11 (which
already has torch, numpy, pandas, scikit-learn, matplotlib) and adding only the
three overlay packages Phase 2 needs. This is fast and small (no multi-GB torch
download):

```bash
# from the repo root, using the base Python 3.11 that already has torch/numpy
"C:/Python311/python.exe" -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install pyod torch-geometric umap-learn
```

Verify:
```bash
.venv/Scripts/python.exe -c "import torch, torch_geometric, pyod, umap, sklearn, pandas; print('env OK')"
```

**Fully self-contained alternative:** if you do not have a system Python with
torch, create a plain venv and install everything (needs ~3 GB free disk for the
torch wheel):
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
Note: `requirements.txt` pins `numpy<2.3`; the `--system-site-packages` recipe
above intentionally keeps the already-working system numpy (2.3.x) instead, which
is compatible with the installed torch 2.9. Use the self-contained route only if
you are starting from a clean Python without torch.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `numpy ... _ArrayMemoryError: Unable to allocate ...` | **Out of RAM** on the full dataset across many seeds (no swap because disk is near-full). Not a code bug. | Run fewer seeds (`--seeds 0 1`), free disk space (so the OS has swap headroom), or run seeds in separate invocations and combine the CSVs. |
| `ModuleNotFoundError: No module named 'Phase_2'` | Not run from the repo root, or not run with `-m`. | `cd` to the repo root and use `python -m Phase_2.run_phase2`. |
| `FileNotFoundError: normalized_original_data.csv` | Input CSVs missing from `data/processed_data/packet-data/`. | Regenerate via `python Phase_1/data_preprocessing_packet.py`, or restore the data files. |
| `ModuleNotFoundError: torch` / `torch_geometric` / `pyod` | Wrong interpreter (bare system Python instead of the project venv). | Use `.venv/Scripts/python.exe` (build it via §5 if missing). |
| Matplotlib cache / permission warnings | Default matplotlib cache dir unwritable. | Set `MPLCONFIGDIR="$PWD/.mplcache"` as shown above (or pass `--no-figures`). |
| `No space left on device` / write failures | Disk is near 100% full. | Clear space; the input CSVs alone are ~800 MB. |
| GNN (`anomal_e`) slow (~60–75 s/seed) or high RAM | Full-graph message passing is inherently heavier. | Expected; use `--quick` for fast iteration. `max_mp_edges` (in `anomal_e.py`) caps message-passing edges. |
| GNN AUC noticeably lower under `--quick` | Subsampling thins the communication graph, weakening topology signal. | Expected — use the full dataset for reportable GNN numbers. |
