from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


TARGET_COL = "target"
BINARY_COL = "binary_label"
ATTACK_COL = "attack_type"
FLOW_KEY_COL = "flow_key"

NON_FEATURE_COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "attack_type",
    "binary_label", "target", "source_file", "flow_key"
]

# These values are undefined when a flow has too few packets or no packets
# in one direction. Zero is the natural replacement in those cases.
STRUCTURAL_ZERO_COLUMNS = [
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Fwd IAT Std",
    "Bwd IAT Mean", "Bwd IAT Std",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg"
]


def load_flow_dataset(path):
    df = pd.read_csv(Path(path), low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_flow_dataset(df):
    required = [FLOW_KEY_COL, BINARY_COL, ATTACK_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[df[BINARY_COL].isin(["benign", "attack"])].copy()
    df = df.drop_duplicates(subset=FLOW_KEY_COL, keep="first").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    for col in STRUCTURAL_ZERO_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[TARGET_COL] = df[BINARY_COL].map({"benign": 0, "attack": 1}).astype("int8")
    return df


def split_flow_dataset(df, validation_size=0.15, test_size=0.15, seed=1):
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[ATTACK_COL], random_state=seed
    )

    validation_share = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_val, test_size=validation_share,
        stratify=train_val[ATTACK_COL], random_state=seed
    )

    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def get_flow_feature_columns(df):
    return [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in NON_FEATURE_COLUMNS
    ]


def build_flow_preprocessor(scale=False):
    steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("remove_constant", VarianceThreshold())
    ]

    if scale:
        steps.append(("scaler", RobustScaler()))

    return Pipeline(steps)


def check_split_overlap(train, validation, test):
    train_keys = set(train[FLOW_KEY_COL])
    validation_keys = set(validation[FLOW_KEY_COL])
    test_keys = set(test[FLOW_KEY_COL])

    return {
        "train_validation": len(train_keys & validation_keys),
        "train_test": len(train_keys & test_keys),
        "validation_test": len(validation_keys & test_keys)
    }
