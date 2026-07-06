"""Central configuration: file paths, column conventions, and shared constants.

Everything that another module might need to know about *where the data lives*
or *how it is laid out* is defined here so the rest of the code stays portable.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]              # repo root (Phase_2/ -> root)
DATA_DIR = ROOT / "data" / "processed_data" / "packet-data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"

# Headline (clean unsupervised) feature set vs. RF-selected ablation set.
NORMALIZED_CSV = DATA_DIR / "normalized_original_data.csv"   # full feature set, no leakage
FOR_DATA_CSV = DATA_DIR / "for_data.csv"                     # RF-selected ablation feats
IDS_CSV = DATA_DIR / "packet_data_ids.csv"                   # topology for the GNN

FEATURE_SETS = {
    "normalized": NORMALIZED_CSV,   # default headline
    "for_data": FOR_DATA_CSV,       # ablation (label-informed selection)
}

# --- Column conventions ------------------------------------------------------
ID_COL = "id"
LABEL_COL = "label"
BENIGN_LABEL = "benign"
ATTACK_LABELS = ["ddos", "dos", "dns_spoofing", "xss", "brute_force"]

# Columns in packet_data_ids.csv used to build the communication graph.
SRC_NODE_COL = "src_ip"
DST_NODE_COL = "dst_ip"
GRAPH_NODE_OPTIONS = {
    "ip": ("src_ip", "dst_ip"),
    "mac": ("src_mac", "dst_mac"),
    "ipport": ("src_ip", "dst_ip"),  # combined with ports in graph.py
}

# --- Evaluation defaults -----------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]            # repeat experiments for mean +/- std
SPLIT_FRACS = (0.6, 0.2, 0.2)      # train / val / test
TARGET_RECALL = 0.95               # FNR-first operating point
MAX_FPR = 0.05                     # cap when maximizing recall
CONTAMINATION = 0.03               # ~ known attack rate (used only for model internals)


def ensure_dirs() -> None:
    """Create the results directory tree if it does not exist."""
    for d in (RESULTS_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
