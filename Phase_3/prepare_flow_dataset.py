import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(Path(__file__).resolve().parent))

from flow_preprocessing import (
    ATTACK_COL, BINARY_COL, FLOW_KEY_COL, TARGET_COL,
    check_split_overlap, clean_flow_dataset, get_flow_feature_columns,
    load_flow_dataset, split_flow_dataset
)


def class_counts(df):
    return {str(k): int(v) for k, v in df[ATTACK_COL].value_counts().items()}


def save_split(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/processed_data/samples/flow_sample_seed1.csv")
    parser.add_argument("--output-dir", default="data/processed_data/phase3")
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df_raw = load_flow_dataset(args.input_path)
    df = clean_flow_dataset(df_raw)
    train, validation, test = split_flow_dataset(
        df, validation_size=args.validation_size,
        test_size=args.test_size, seed=args.seed
    )

    feature_cols = get_flow_feature_columns(train)
    overlap = check_split_overlap(train, validation, test)

    save_split(train, output_dir / "flow_train.csv")
    save_split(validation, output_dir / "flow_validation.csv")
    save_split(test, output_dir / "flow_test.csv")

    metadata = {
        "input_path": str(args.input_path),
        "seed": args.seed,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "raw_rows": len(df_raw),
        "clean_rows": len(df),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "excluded_columns": [col for col in df.columns if col not in feature_cols],
        "target_column": TARGET_COL,
        "target_mapping": {"benign": 0, "attack": 1},
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test)
        },
        "binary_counts": {
            "train": {str(k): int(v) for k, v in train[BINARY_COL].value_counts().items()},
            "validation": {str(k): int(v) for k, v in validation[BINARY_COL].value_counts().items()},
            "test": {str(k): int(v) for k, v in test[BINARY_COL].value_counts().items()}
        },
        "attack_counts": {
            "train": class_counts(train),
            "validation": class_counts(validation),
            "test": class_counts(test)
        },
        "missing_feature_values": {
            "train": int(train[feature_cols].isna().sum().sum()),
            "validation": int(validation[feature_cols].isna().sum().sum()),
            "test": int(test[feature_cols].isna().sum().sum())
        },
        "flow_key_overlap": overlap
    }

    metadata_path = output_dir / "flow_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Loaded rows: {len(df_raw):,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Model features: {len(feature_cols)}")
    print("")
    print("Split sizes")
    print(f"  Train:      {len(train):,}")
    print(f"  Validation: {len(validation):,}")
    print(f"  Test:       {len(test):,}")
    print("")
    print("Attack types by split")
    print("  Train:", class_counts(train))
    print("  Validation:", class_counts(validation))
    print("  Test:", class_counts(test))
    print("")
    print("Flow-key overlap:", overlap)
    print(f"Saved files to: {output_dir}")


if __name__ == "__main__":
    main()
