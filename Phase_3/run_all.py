import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from evaluation import RESULTS_DIR as OUT

CODE = Path(__file__).resolve().parent

STAGES = [
    (["build_context_features.py"], "context_features.parquet"),
    (["prep.py"], "splits.npz"),
    (["train_models.py"], "results.json#voting_soft"),
    (["multiclass_xgboost.py"], "results.json#multiclass_xgb__no_anomaly"),
    (["train_models.py", "--drop-anomaly"], "results.json#voting_soft__no_anomaly"),
    (["diagnose_hard_classes.py"], "hard_class_diagnostics.json"),
    (["eval_session_holdout.py"], "session_holdout_results.json"),
    (["eval_session_holdout.py", "--features", "base"], "session_holdout_base_results.json"),
    (["exp_context.py"], "exp_context_results.json"),
    (["verify_context_integrity.py"], "feature_importance.csv"),
    (["two_stage_system.py"], "results.json#two_stage_system"),
]


def already_done(marker):
    name, _, key = marker.partition("#")
    path = OUT / name
    if not path.exists():
        return False
    if not key:
        return True
    try:
        return key in json.loads(path.read_text())
    except (ValueError, OSError):
        return False


def run_stage(argv, marker, skip_existing):
    label = " ".join(argv)
    if skip_existing and already_done(marker):
        print(f"  [skip] {label} (found {marker})")
        return
    print(f"  [run ] {label} ...", flush=True)
    t0 = time.time()
    cmd = [sys.executable, str(CODE / argv[0])] + argv[1:]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(f"FAILED at {label}")
    print(f"         done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse saved results, only run missing stages")
    args = ap.parse_args()
    print("Phase 3 full pipeline")
    t0 = time.time()
    for argv, marker in STAGES:
        run_stage(argv, marker, args.skip_existing)
    print(f"\nAll stages complete in {time.time() - t0:.0f}s")

    import show_results
    show_results.render()


if __name__ == "__main__":
    main()
