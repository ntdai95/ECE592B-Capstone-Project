import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    (1, "Phase 1  packet preprocessing and sampling",
     [PY, "Phase_1/data_preprocessing_packet.py"],
     "data/processed_data/packet-data/for_data.csv"),

    (2, "Phase 2  k-means + autoencoder benign-only detector + Phase 3 alert export",
     [PY, "Phase_2/kmean_autoencoder.py"],
     "data/processed_data/packet-data/phase2_autoencoder_alert_rows_for_phase3.csv"),

    (2, "Phase 2  packet detectors (Deep SVDD / Anomal-E / fusion)",
     [PY, "-m", "Phase_2.run_phase2", "--feature-set", "normalized", "--models", "all", "--scaler", "quantile"],
     "results/results_normalized_quantile.csv"),

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
