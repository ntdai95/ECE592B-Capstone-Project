from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd


ATTACK_PATTERNS = [
    ("ddos-http_flood", "DDoS-HTTP Flood"),
    ("ddos-http-flood", "DDoS-HTTP Flood"),
    ("dos-http_flood", "DoS-HTTP Flood"),
    ("dos-http-flood", "DoS-HTTP Flood"),
    ("dns_spoofing", "DNS Spoofing"),
    ("dns-spoofing", "DNS Spoofing"),
    ("dictionarybruteforce", "Brute Force"),
    ("dictionary_bruteforce", "Brute Force"),
    ("bruteforce", "Brute Force"),
    ("brute_force", "Brute Force"),
    ("xss", "XSS"),
    ("benign", "Benign"),
]


def to_builtin(obj):
    """Convert numpy/pandas objects into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if pd.isna(obj):
        return None
    return obj


def count_csv_rows_fast(csv_path):
    """
    Fast row counter for large CSV files.
    Assumes one record per line, which is normal for these network CSV files.
    """
    line_count = 0
    with open(csv_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            line_count += block.count(b"\n")

    # If file does not end with newline, count the final line.
    try:
        with open(csv_path, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                line_count += 1
    except OSError:
        pass

    # Subtract header row.
    return max(line_count - 1, 0)


def normalize_file_name(path):
    """
    Normalize file names so small differences like '-' vs '_' do not matter.
    Example:
    DDoS-HTTP_Flood-.csv              -> ddos_http_flood
    DDoS-HTTP_Flood-.pcap_Flow.csv    -> ddos_http_flood
    BenignTraffic.pcap_Flow.csv       -> benigntraffic
    """
    name = path.name.lower()
    name = name.replace(".pcap_flow.csv", "")
    name = name.replace(".csv", "")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def infer_traffic_level(path):
    """
    In this dataset:
    files ending with .pcap_Flow.csv are flow-level files.
    files without .pcap_Flow are packet-level files.
    """
    name = path.name.lower()

    if name.endswith(".pcap_flow.csv"):
        return "flow-level"

    return "packet-level"


def infer_attack_type(path):
    name = normalize_file_name(path)

    if "benign" in name:
        return "Benign"

    if "ddos" in name and "http" in name and "flood" in name:
        return "DDoS-HTTP Flood"

    if name.startswith("dos") and "http" in name and "flood" in name:
        return "DoS-HTTP Flood"

    if "dns" in name and "spoofing" in name:
        return "DNS Spoofing"

    if "dictionarybruteforce" in name or "bruteforce" in name or "brute_force" in name:
        return "Brute Force"

    if "xss" in name:
        return "XSS"

    return "Unknown"


def infer_binary_label(attack_type):
    if attack_type == "Benign":
        return "benign"
    if attack_type == "Unknown":
        return "unknown"
    return "attack"


def find_possible_columns(columns):
    lower_map = {col: str(col).strip().lower() for col in columns}

    patterns = {
        "label_columns": ["label", "class", "attack", "category"],
        "flow_id_columns": ["flow id", "flow_id", "flowid"],
        "source_ip_columns": ["src ip", "source ip", "src_ip", "source_ip"],
        "destination_ip_columns": ["dst ip", "destination ip", "dst_ip", "destination_ip"],
        "source_port_columns": ["src port", "source port", "sport", "src_port"],
        "destination_port_columns": ["dst port", "destination port", "dport", "dst_port"],
        "protocol_columns": ["protocol", "proto"],
        "timestamp_columns": ["timestamp", "time", "date"],
    }

    result = {}
    for group_name, keywords in patterns.items():
        result[group_name] = [
            col for col, lower_col in lower_map.items()
            if any(keyword in lower_col for keyword in keywords)
        ]

    return result


def read_csv_sample(csv_path, sample_rows):
    try:
        df = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False, encoding="latin1")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def profile_csv_file(csv_path, sample_rows=5000, head_rows=3, full_profile=False, chunksize=100000):
    start = time.time()

    file_size_mb = csv_path.stat().st_size / (1024 ** 2)
    row_count_fast = count_csv_rows_fast(csv_path)

    sample_df = read_csv_sample(csv_path, sample_rows)

    columns = list(sample_df.columns)
    duplicate_columns = sample_df.columns[sample_df.columns.duplicated()].tolist()
    unnamed_columns = [c for c in columns if str(c).lower().startswith("unnamed")]

    possible_columns = find_possible_columns(columns)

    sample_missing = sample_df.isna().sum().sort_values(ascending=False)
    sample_missing_top = sample_missing[sample_missing > 0].head(20).to_dict()

    sample_dtypes = {c: str(t) for c, t in sample_df.dtypes.items()}

    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()

    constant_sample_columns = [
        c for c in columns
        if sample_df[c].nunique(dropna=False) <= 1
    ]

    details = {
        "file_name": csv_path.name,
        "relative_path": str(csv_path),
        "file_size_mb": round(file_size_mb, 2),
        "traffic_level": infer_traffic_level(csv_path),
        "attack_type": infer_attack_type(csv_path),
        "binary_label": infer_binary_label(infer_attack_type(csv_path)),
        "row_count_fast": row_count_fast,
        "num_columns": len(columns),
        "columns": columns,
        "duplicate_columns": duplicate_columns,
        "unnamed_columns": unnamed_columns,
        "sample_rows_used": len(sample_df),
        "head": sample_df.head(head_rows).to_dict(orient="records"),
        "sample_dtypes": sample_dtypes,
        "num_numeric_columns_sample": len(numeric_cols),
        "num_non_numeric_columns_sample": len(object_cols),
        "non_numeric_columns_sample": object_cols,
        "sample_missing_top": sample_missing_top,
        "constant_sample_columns_count": len(constant_sample_columns),
        "constant_sample_columns_first_30": constant_sample_columns[:30],
        "possible_important_columns": possible_columns,
    }

    if full_profile:
        print(f"  Running full profile for {csv_path.name} ...")

        exact_rows = 0
        missing_counts = None
        inf_counts = None

        try:
            chunk_iter = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
        except UnicodeDecodeError:
            chunk_iter = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, encoding="latin1")

        for chunk in chunk_iter:
            chunk.columns = [str(c).strip() for c in chunk.columns]
            exact_rows += len(chunk)

            chunk_missing = chunk.isna().sum()
            if missing_counts is None:
                missing_counts = chunk_missing
            else:
                missing_counts = missing_counts.add(chunk_missing, fill_value=0)

            num_chunk = chunk.select_dtypes(include=[np.number])
            chunk_inf = np.isinf(num_chunk).sum()
            if inf_counts is None:
                inf_counts = chunk_inf
            else:
                inf_counts = inf_counts.add(chunk_inf, fill_value=0)

        missing_counts = missing_counts.sort_values(ascending=False)
        inf_counts = inf_counts.sort_values(ascending=False)

        details["exact_rows_by_pandas_chunks"] = exact_rows
        details["full_missing_columns_top_30"] = missing_counts[missing_counts > 0].head(30).to_dict()
        details["full_infinite_columns_top_30"] = inf_counts[inf_counts > 0].head(30).to_dict()

    details["profile_time_seconds"] = round(time.time() - start, 2)
    return details


def build_schema_comparison(all_details):
    by_level = {}

    for item in all_details:
        level = item["traffic_level"]
        by_level.setdefault(level, []).append(item)

    comparison = {}

    for level, items in by_level.items():
        column_sets = [set(item["columns"]) for item in items]

        if not column_sets:
            continue

        common_columns = set.intersection(*column_sets)
        union_columns = set.union(*column_sets)

        comparison[level] = {
            "num_files": len(items),
            "common_columns_count": len(common_columns),
            "union_columns_count": len(union_columns),
            "common_columns": sorted(common_columns),
            "union_columns": sorted(union_columns),
            "schema_is_identical_across_files": all(cols == column_sets[0] for cols in column_sets),
            "per_file_missing_from_union": {
                item["file_name"]: sorted(list(union_columns - set(item["columns"])))
                for item in items
            },
        }

    return comparison


def write_text_report(all_details, schema_comparison, output_path):
    lines = []
    lines.append("DATASET INVENTORY REPORT")
    lines.append("=" * 80)
    lines.append("")

    for i, item in enumerate(all_details, start=1):
        lines.append(f"[{i}] {item['file_name']}")
        lines.append("-" * 80)
        lines.append(f"Path: {item['relative_path']}")
        lines.append(f"Traffic level: {item['traffic_level']}")
        lines.append(f"Inferred attack type: {item['attack_type']}")
        lines.append(f"Inferred binary label: {item['binary_label']}")
        lines.append(f"File size: {item['file_size_mb']} MB")
        lines.append(f"Rows, fast count: {item['row_count_fast']}")
        lines.append(f"Columns: {item['num_columns']}")
        lines.append("")
        lines.append("Header:")
        lines.append(", ".join(item["columns"]))
        lines.append("")
        lines.append("Possible important columns:")
        for key, value in item["possible_important_columns"].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append("Top sample missing columns:")
        lines.append(str(item["sample_missing_top"]))
        lines.append("")
        lines.append("First rows:")
        for row in item["head"]:
            lines.append(str(row))
        lines.append("")
        lines.append("")

    lines.append("SCHEMA COMPARISON")
    lines.append("=" * 80)
    lines.append(json.dumps(to_builtin(schema_comparison), indent=2))

    output_path.write_text("\n".join(lines), encoding="utf-8")

def validate_expected_files(all_details):
    expected_attack_types = {
        "Benign",
        "DDoS-HTTP Flood",
        "DoS-HTTP Flood",
        "DNS Spoofing",
        "XSS",
        "Brute Force",
    }

    for level in ["packet-level", "flow-level"]:
        found = {
            item["attack_type"]
            for item in all_details
            if item["traffic_level"] == level
        }

        missing = expected_attack_types - found
        extra = found - expected_attack_types

        print(f"\nValidation for {level}:")
        print(f"  Found: {sorted(found)}")

        if missing:
            print(f"  Missing: {sorted(missing)}")
        else:
            print("  Missing: None")

        if extra:
            print(f"  Extra/unknown: {sorted(extra)}")
        else:
            print("  Extra/unknown: None")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the folder containing the dataset CSV files."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/dataset_inventory",
        help="Folder where the inventory reports will be saved."
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Rows to read from each CSV for quick dtype/missing/header inspection."
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=3,
        help="Number of first rows to include in the readable report."
    )
    parser.add_argument(
        "--full-profile",
        action="store_true",
        help="Read every CSV in chunks to compute exact missing/infinite counts. Slower."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Chunk size used when --full-profile is enabled."
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    print(f"Found {len(csv_files)} CSV files.")
    print(f"Output folder: {out_dir}")
    print("")

    all_details = []

    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Profiling: {csv_path.name}")
        details = profile_csv_file(
            csv_path=csv_path,
            sample_rows=args.sample_rows,
            head_rows=args.head_rows,
            full_profile=args.full_profile,
            chunksize=args.chunksize,
        )
        all_details.append(details)

        print(f"  Level: {details['traffic_level']}")
        print(f"  Attack type: {details['attack_type']}")
        print(f"  Size: {details['file_size_mb']} MB")
        print(f"  Rows: {details['row_count_fast']}")
        print(f"  Columns: {details['num_columns']}")
        print(f"  Time: {details['profile_time_seconds']} sec")
        print("")

    schema_comparison = build_schema_comparison(all_details)

    summary_rows = []
    for item in all_details:
        summary_rows.append({
            "file_name": item["file_name"],
            "relative_path": item["relative_path"],
            "traffic_level": item["traffic_level"],
            "attack_type": item["attack_type"],
            "binary_label": item["binary_label"],
            "file_size_mb": item["file_size_mb"],
            "row_count_fast": item["row_count_fast"],
            "num_columns": item["num_columns"],
            "num_numeric_columns_sample": item["num_numeric_columns_sample"],
            "num_non_numeric_columns_sample": item["num_non_numeric_columns_sample"],
            "duplicate_columns": "; ".join(item["duplicate_columns"]),
            "unnamed_columns": "; ".join(item["unnamed_columns"]),
            "constant_sample_columns_count": item["constant_sample_columns_count"],
            "profile_time_seconds": item["profile_time_seconds"],
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = out_dir / "dataset_inventory_summary.csv"
    details_json_path = out_dir / "dataset_inventory_details.json"
    schema_json_path = out_dir / "schema_comparison.json"
    text_report_path = out_dir / "dataset_inventory_report.txt"

    summary_df.to_csv(summary_csv_path, index=False)

    details_json_path.write_text(
        json.dumps(to_builtin(all_details), indent=2),
        encoding="utf-8"
    )

    schema_json_path.write_text(
        json.dumps(to_builtin(schema_comparison), indent=2),
        encoding="utf-8"
    )

    write_text_report(all_details, schema_comparison, text_report_path)

    print("Done.")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved details JSON: {details_json_path}")
    print(f"Saved schema JSON: {schema_json_path}")
    print(f"Saved text report: {text_report_path}")

    schema_comparison = build_schema_comparison(all_details)
    validate_expected_files(all_details)


if __name__ == "__main__":
    main()