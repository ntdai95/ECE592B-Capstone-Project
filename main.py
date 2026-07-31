"""
Runs the whole project end to end, in dependency order:

  Phase 1   packet-level preprocessing and sampling
  Phase 2   unsupervised packet-level anomaly detection (Anomaly-Based IDS)
              - autoencoder benign-only detector, which also exports the alert
                rows Phase 3 consumes
              - Deep SVDD / Anomal-E / score-level fusion detector suite
  Phase 3   supervised flow-level refinement (Signature-Based IDS)
              - flow feature engineering + the unified altered dataset
                (flow features + 23 connection-window context features)
              - Task 3.1 flow-level anomaly scores
              - context features, the shared train/val/test split, the six
                classifiers, the Task 3.1 ablation, the validity analysis, and
                the printed results tables

    python main.py                  # run everything
    python main.py --skip-existing  # only run stages whose output is missing
    python main.py --phase 3        # run a single phase (1, 2 or 3)

Each stage is a separate script so it can also be run on its own; this file just
sequences them and stops at the first failure. Every result is written under
results/ and data/processed_data/, both git-ignored.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (phase, label, command, output marker that proves the stage already ran)
STAGES = [
    (1, "Phase 1  packet preprocessing and sampling",
     [PY, "Phase_1/data_preprocessing_packet.py"],
     "data/processed_data/packet-data/for_data.csv"),

    (2, "Phase 2  k-means + autoencoder benign-only detector + Phase 3 alert export",
     [PY, "Phase_2/kmean_autoencoder.py"],
     "data/processed_data/packet-data/phase2_autoencoder_alert_rows_for_phase3.csv"),

    (2, "Phase 2  packet detectors (Deep SVDD / Anomal-E / fusion)",
     [PY, "-m", "Phase_2.run_phase2", "--feature-set", "normalized", "--models", "all"],
     "results/results_normalized.csv"),

    (3, "Phase 3.2  flow feature engineering + unified dataset",
     [PY, "Phase_3/flow_based_feature_engineering.py"],
     "data/processed_data/flow-data/normalized_original_data.csv"),

    (3, "Phase 3.1  flow-level anomaly scores",
     [PY, "Phase_3/flow_anomaly_detection.py"],
     "data/processed_data/flow-data/task3_1_flow_anomaly_scores.csv"),

    (3, "Phase 3.3  context, splits, six models, ablation, analysis, results",
     [PY, "Phase_3/run_all.py"],
     "results/leaderboard.csv"),
]


def run_stage(label, cmd, marker, skip_existing):
    if skip_existing and (ROOT / marker).exists():
        print(f"[skip] {label}  (found {marker})")
        return
    print(f"\n{'=' * 78}\n[run ] {label}\n{'=' * 78}", flush=True)
    t0 = time.time()
    if skip_existing and cmd[-1].endswith("run_all.py"):
        cmd = cmd + ["--skip-existing"]
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"\nFAILED at: {label}  (exit {r.returncode})")
    print(f"[done] {label}  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="Multi-stage IoT IDS pipeline")
    ap.add_argument("--skip-existing", action="store_true",
                    help="only run stages whose output file is missing")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                    help="run only this phase")
    args = ap.parse_args()

    os.chdir(ROOT)
    stages = [s for s in STAGES if args.phase is None or s[0] == args.phase]
    print(f"Running {len(stages)} stage(s)"
          + (f" for phase {args.phase}" if args.phase else "") + ".")
    t0 = time.time()
    for _, label, cmd, marker in stages:
        run_stage(label, cmd, marker, args.skip_existing)
    print(f"\nPipeline complete in {time.time() - t0:.0f}s. "
          f"Results in {ROOT / 'results'}.")


if __name__ == "__main__":
    main()
