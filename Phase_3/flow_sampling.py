from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


ATTACK_FILE_NAMES = {
    "DDoS-HTTP Flood": "DDoS-HTTP_Flood-.pcap_Flow.csv",
    "DoS-HTTP Flood": "DoS-HTTP_Flood.pcap_Flow.csv",
    "DNS Spoofing": "DNS_Spoofing.pcap_Flow.csv",
    "Brute Force": "DictionaryBruteForce.pcap_Flow.csv",
    "XSS": "XSS.pcap_Flow.csv",
}

BENIGN_FILE_NAME = "BenignTraffic.pcap_Flow.csv"

ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]

SUM_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Fwd Act Data Pkts",
]

MAX_COLUMNS = [
    "Fwd Packet Length Max",
    "Bwd Packet Length Max",
    "Flow IAT Max",
    "Fwd IAT Max",
    "Bwd IAT Max",
    "Packet Length Max",
    "Active Max",
    "Idle Max",
]

MIN_COLUMNS = [
    "Fwd Packet Length Min",
    "Bwd Packet Length Min",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Bwd IAT Min",
    "Packet Length Min",
    "Fwd Seg Size Min",
    "Active Min",
    "Idle Min",
]

MEAN_COLUMNS = [
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",
]

FIRST_COLUMNS = ["FWD Init Win Bytes", "Bwd Init Win Bytes"]

STAT_GROUPS = [
    ("Total Fwd Packet", "Fwd Packet Length Mean", "Fwd Packet Length Std"),
    ("Total Bwd packets", "Bwd Packet Length Mean", "Bwd Packet Length Std"),
    ("Flow IAT Count", "Flow IAT Mean", "Flow IAT Std"),
    ("Fwd IAT Count", "Fwd IAT Mean", "Fwd IAT Std"),
    ("Bwd IAT Count", "Bwd IAT Mean", "Bwd IAT Std"),
    ("Packet Length Count", "Packet Length Mean", "Packet Length Std"),
]


def _read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1", **kwargs)


def _available_flow_ids(path, chunksize):
    flow_ids = set()

    for chunk in _read_csv(path, usecols=["Flow ID"], chunksize=chunksize):
        values = chunk["Flow ID"].dropna().astype(str)
        flow_ids.update(values)

    return np.array(sorted(flow_ids), dtype=object)


def _allocate_attack_counts(total_rows, minimum_each, capacities, rng):
    attack_types = list(capacities)
    minimums = {name: min(minimum_each, capacities[name]) for name in attack_types}
    minimum_total = sum(minimums.values())

    if total_rows < minimum_total:
        raise ValueError(f"attack rows must be at least {minimum_total} for the available classes")

    total_rows = min(total_rows, sum(capacities.values()))
    counts = minimums.copy()
    remaining = total_rows - minimum_total

    while remaining > 0:
        available = [name for name in attack_types if counts[name] < capacities[name]]

        if not available:
            break

        chosen = rng.choice(available)
        counts[chosen] += 1
        remaining -= 1

    return counts


def _load_selected_segments(path, selected_ids, chunksize):
    selected_ids = set(selected_ids)
    chunks = []

    for chunk in _read_csv(path, chunksize=chunksize):
        chunk["Flow ID"] = chunk["Flow ID"].astype(str)
        chunk = chunk[chunk["Flow ID"].isin(selected_ids)]

        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No selected flows were found in {path.name}")

    return pd.concat(chunks, ignore_index=True)


def _prepare_segments(df):
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.drop(columns=["Label"], errors="ignore")
    raw_timestamp = df["Timestamp"].copy()
    df["Timestamp"] = pd.to_datetime(raw_timestamp, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    missing_timestamp = df["Timestamp"].isna() & raw_timestamp.notna()

    if missing_timestamp.any():
        df.loc[missing_timestamp, "Timestamp"] = pd.to_datetime(raw_timestamp[missing_timestamp], errors="coerce", dayfirst=True)

    protected = set(ID_COLUMNS + ["Timestamp"])
    numeric_columns = [column for column in df.columns if column not in protected]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    total_packets = df.get("Total Fwd Packet", 0).fillna(0) + df.get("Total Bwd packets", 0).fillna(0)
    df["Flow IAT Count"] = (total_packets - 1).clip(lower=0)
    df["Fwd IAT Count"] = (df.get("Total Fwd Packet", 0).fillna(0) - 1).clip(lower=0)
    df["Bwd IAT Count"] = (df.get("Total Bwd packets", 0).fillna(0) - 1).clip(lower=0)
    df["Packet Length Count"] = total_packets + 1

    return df.sort_values(["Flow ID", "Timestamp"], kind="stable")


def _combine_mean_std(df, group_column, count_column, mean_column, std_column):
    columns = [group_column, count_column, mean_column, std_column]
    temp = df[columns].copy()
    temp[count_column] = temp[count_column].fillna(0).clip(lower=0)
    temp[mean_column] = temp[mean_column].fillna(0)
    temp[std_column] = temp[std_column].fillna(0)

    count = temp[count_column]
    temp["_weighted_mean"] = count * temp[mean_column]
    temp["_second_moment"] = np.maximum(count - 1, 0) * temp[std_column].pow(2) + count * temp[mean_column].pow(2)

    grouped = temp.groupby(group_column, sort=False).agg(
        _count=(count_column, "sum"),
        _weighted_mean=("_weighted_mean", "sum"),
        _second_moment=("_second_moment", "sum"),
    )

    grouped[mean_column] = grouped["_weighted_mean"] / grouped["_count"].replace(0, np.nan)
    numerator = grouped["_second_moment"] - grouped["_count"] * grouped[mean_column].pow(2)
    grouped[std_column] = np.sqrt((numerator.clip(lower=0) / (grouped["_count"] - 1).replace(0, np.nan)).clip(lower=0))

    return grouped[[mean_column, std_column]]


def aggregate_flow_segments(df):
    df = _prepare_segments(df)
    group_column = "Flow ID"
    aggregation = {}

    for column in ID_COLUMNS:
        if column != group_column and column in df:
            aggregation[column] = "first"

    if "Timestamp" in df:
        aggregation["Timestamp"] = "min"

    for column in SUM_COLUMNS:
        if column in df:
            aggregation[column] = "sum"

    for column in MAX_COLUMNS:
        if column in df:
            aggregation[column] = "max"

    for column in MIN_COLUMNS:
        if column in df:
            aggregation[column] = "min"

    for column in MEAN_COLUMNS:
        if column in df:
            aggregation[column] = "mean"

    for column in FIRST_COLUMNS:
        if column in df:
            aggregation[column] = "first"

    aggregated = df.groupby(group_column, sort=False).agg(aggregation)
    aggregated["segment_count"] = df.groupby(group_column, sort=False).size()

    for count_column, mean_column, std_column in STAT_GROUPS:
        required = {count_column, mean_column, std_column}

        if required.issubset(df.columns):
            combined = _combine_mean_std(df, group_column, count_column, mean_column, std_column)
            aggregated[[mean_column, std_column]] = combined[[mean_column, std_column]]

    if "Packet Length Std" in aggregated:
        aggregated["Packet Length Variance"] = aggregated["Packet Length Std"].pow(2)

    duration_seconds = aggregated["Flow Duration"].replace(0, np.nan) / 1_000_000
    total_packets = aggregated["Total Fwd Packet"] + aggregated["Total Bwd packets"]
    total_bytes = aggregated["Total Length of Fwd Packet"] + aggregated["Total Length of Bwd Packet"]

    aggregated["Flow Bytes/s"] = total_bytes / duration_seconds
    aggregated["Flow Packets/s"] = total_packets / duration_seconds
    aggregated["Fwd Packets/s"] = aggregated["Total Fwd Packet"] / duration_seconds
    aggregated["Bwd Packets/s"] = aggregated["Total Bwd packets"] / duration_seconds
    aggregated["Down/Up Ratio"] = aggregated["Total Bwd packets"] / aggregated["Total Fwd Packet"].replace(0, np.nan)
    aggregated["Average Packet Size"] = total_bytes / total_packets.replace(0, np.nan)
    aggregated["Fwd Segment Size Avg"] = aggregated["Total Length of Fwd Packet"] / aggregated["Total Fwd Packet"].replace(0, np.nan)
    aggregated["Bwd Segment Size Avg"] = aggregated["Total Length of Bwd Packet"] / aggregated["Total Bwd packets"].replace(0, np.nan)

    return aggregated.reset_index()


def _sample_file(path, flow_ids, count, rng, chunksize, attack_type):
    selected_ids = rng.choice(flow_ids, size=count, replace=False)
    segments = _load_selected_segments(path, selected_ids, chunksize)
    flows = aggregate_flow_segments(segments)

    flows["attack_type"] = attack_type
    flows["binary_label"] = "benign" if attack_type == "Benign" else "attack"
    flows["source_file"] = path.name
    flows["flow_key"] = flows["source_file"] + "::" + flows["Flow ID"].astype(str)

    return flows, len(segments)


def generate_random_flow_dataset(data_dir, output_path, benign_rows=200000, attack_min_rows=4000, attack_max_rows=6200, min_attack_each=200, seed=None, chunksize=100000):
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if attack_min_rows > attack_max_rows:
        raise ValueError("attack_min_rows cannot be larger than attack_max_rows")

    rng = np.random.default_rng(seed)
    benign_path = data_dir / BENIGN_FILE_NAME
    attack_paths = {name: data_dir / file_name for name, file_name in ATTACK_FILE_NAMES.items()}
    required_paths = [benign_path, *attack_paths.values()]
    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError("Missing flow files:\n" + "\n".join(missing_paths))

    print("Scanning unique flow IDs...")
    benign_ids = _available_flow_ids(benign_path, chunksize)
    attack_ids = {name: _available_flow_ids(path, chunksize) for name, path in attack_paths.items()}

    actual_benign_rows = min(benign_rows, len(benign_ids))

    if actual_benign_rows < benign_rows:
        warnings.warn(
            f"Requested {benign_rows:,} benign flows, but only {len(benign_ids):,} unique benign flows are available. "
            f"Using all {actual_benign_rows:,} without replacement."
        )

    requested_attack_rows = int(rng.integers(attack_min_rows, attack_max_rows + 1))
    attack_capacities = {name: len(ids) for name, ids in attack_ids.items()}
    attack_counts = _allocate_attack_counts(requested_attack_rows, min_attack_each, attack_capacities, rng)

    print(f"Sampling {actual_benign_rows:,} benign flows...")
    frames = []
    source_segment_counts = {}
    benign_flows, benign_segments = _sample_file(benign_path, benign_ids, actual_benign_rows, rng, chunksize, "Benign")
    frames.append(benign_flows)
    source_segment_counts[benign_path.name] = benign_segments

    for attack_type, count in attack_counts.items():
        print(f"Sampling {count:,} {attack_type} flows...")
        attack_flows, attack_segments = _sample_file(attack_paths[attack_type], attack_ids[attack_type], count, rng, chunksize, attack_type)
        frames.append(attack_flows)
        source_segment_counts[attack_paths[attack_type].name] = attack_segments

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    if dataset["flow_key"].duplicated().any():
        raise ValueError("Duplicate flow keys remain after aggregation")

    dataset.to_csv(output_path, index=False)

    metadata = {
        "output_path": str(output_path),
        "seed": seed,
        "requested_benign_rows": benign_rows,
        "actual_benign_rows": int((dataset["binary_label"] == "benign").sum()),
        "requested_attack_rows": requested_attack_rows,
        "actual_attack_rows": int((dataset["binary_label"] == "attack").sum()),
        "attack_counts": {name: int(count) for name, count in dataset["attack_type"].value_counts().items() if name != "Benign"},
        "total_rows": len(dataset),
        "available_unique_flows": {
            "Benign": len(benign_ids),
            **{name: len(ids) for name, ids in attack_ids.items()},
        },
        "source_segment_counts": source_segment_counts,
        "segments_collapsed": int(sum(source_segment_counts.values()) - len(dataset)),
        "columns": list(dataset.columns),
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved flow dataset: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Rows: {len(dataset):,}")
    print(f"Segments collapsed: {metadata['segments_collapsed']:,}")

    return dataset, metadata
