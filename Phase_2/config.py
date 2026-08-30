from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_data" / "packet-data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"

NORMALIZED_CSV = DATA_DIR / "normalized_original_data.csv"
FOR_DATA_CSV = DATA_DIR / "for_data.csv"
IDS_CSV = DATA_DIR / "packet_data_ids.csv"

FEATURE_SETS = {
    "normalized": NORMALIZED_CSV,
    "for_data": FOR_DATA_CSV,
}

ID_COL = "id"
LABEL_COL = "label"
BENIGN_LABEL = "benign"
ATTACK_LABELS = ["ddos", "dos", "dns_spoofing", "xss", "brute_force"]

SEEDS = [0, 1, 2, 3, 4]
SPLIT_FRACS = (0.6, 0.2, 0.2)
TARGET_RECALL = 0.95
MAX_FPR = 0.05
CONTAMINATION = 0.03


def ensure_dirs():
    for d in (RESULTS_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
