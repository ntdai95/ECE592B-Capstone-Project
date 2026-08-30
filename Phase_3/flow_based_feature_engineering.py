import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


FILE_GROUPS = {
    'benign':       ['BenignTraffic.pcap_Flow.csv', 'BenignTraffic1.pcap_Flow.csv',
                     'BenignTraffic2.pcap_Flow.csv', 'BenignTraffic3.pcap_Flow.csv'],
    'dos':          ['DoS-HTTP_Flood.pcap_Flow.csv', 'DoS-HTTP_Flood1.pcap_Flow.csv'],
    'brute_force':  ['DictionaryBruteForce.pcap_Flow.csv'],
    'ddos':         ['DDoS-HTTP_Flood-.pcap_Flow.csv'],
    'dns_spoofing': ['DNS_Spoofing.pcap_Flow.csv'],
    'xss':          ['XSS.pcap_Flow.csv'],
}

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

FLOW_DATA_FILE_PATH = PROJECT_ROOT / "data" / "raw_data" / "flow-based-features"
PACKET_DATA_DIR = PROJECT_ROOT / "data" / "processed_data" / "packet-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_data" / "flow-data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
PACKET_IDENTIFIERS_FILE = PACKET_DATA_DIR / "packet_data_ids.csv"
PACKET_PROTOCOL_FLAGS_FILE = PACKET_DATA_DIR / "normalized_original_data.csv"
PHASE2_ALERTS_FILE = PACKET_DATA_DIR / "phase2_autoencoder_alert_rows_for_phase3.csv"

PROTOCOL_NUMBERS = {"tcp": 6, "udp": 17}

def load_all_datasets(path, filenames):
    flow_data = []
    for label, files in filenames.items():
        flow_df = []
        for file in files:
            flow_df.append(pd.read_csv(path / file))

        flow_df = pd.concat(flow_df, ignore_index=True)
        flow_df = flow_df.drop(columns=["Label"])
        flow_df['label'] = label
        flow_data.append(flow_df)

    return pd.concat(flow_data, ignore_index=True)

def aggregate_flows(flow_df):
    identifier_columns = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]
    identifier_df = flow_df[["Flow ID"] + identifier_columns].drop_duplicates("Flow ID")
    flow_df = flow_df.drop(columns=identifier_columns)
    columns_aggregation_methods = {}
    for column_name in flow_df.columns:
        if column_name not in ["Flow ID", "label"]:
            if "Max" in column_name:
                columns_aggregation_methods[column_name] = "max"
            elif "Min" in column_name:
                columns_aggregation_methods[column_name] = "min"
            else:
                keywords = ["Total", "Count", "Subflow", "Flags", "Duration", "Header Length", "Pkts"]
                columns_aggregation_methods[column_name] = "mean"
                for word in keywords:
                    if word in column_name:
                        columns_aggregation_methods[column_name] = "sum"
                        break

    return flow_df.groupby(["Flow ID", "label"], as_index=False).agg(columns_aggregation_methods), identifier_df

def load_packet_identifiers(packet_ids=None):
    identifiers_df = pd.read_csv(PACKET_IDENTIFIERS_FILE)
    protocol_df = pd.read_csv(PACKET_PROTOCOL_FLAGS_FILE, usecols=["id", "l4_tcp", "l4_udp", "label"])
    packet_identifiers_df = identifiers_df.merge(protocol_df, on="id", how="inner")
    if packet_ids is not None:
        packet_identifiers_df = packet_identifiers_df[packet_identifiers_df["id"].isin(packet_ids)]

    packet_identifiers_df["protocol"] = np.select([packet_identifiers_df["l4_tcp"] > 0, packet_identifiers_df["l4_udp"] > 0],
                                                  [PROTOCOL_NUMBERS["tcp"], PROTOCOL_NUMBERS["udp"]], default=np.nan)

    unmapped = packet_identifiers_df["protocol"].isna().sum()
    print(f"[flow-correspondence] {unmapped:,}/{len(packet_identifiers_df):,} packets have no TCP/UDP transport and cannot be matched to a flow")
    return packet_identifiers_df.dropna(subset=["protocol"])

def build_flow_ids(packet_identifiers_df):
    protocol = packet_identifiers_df["protocol"].astype(int).astype(str)
    src_ip = packet_identifiers_df["src_ip"].fillna('nan').astype(str)
    dst_ip = packet_identifiers_df["dst_ip"].fillna('nan').astype(str)
    src_port = packet_identifiers_df["src_port"].fillna('nan').astype(str)
    dst_port = packet_identifiers_df["dst_port"].fillna('nan').astype(str)
    packet_identifiers_df = packet_identifiers_df.copy()
    packet_identifiers_df["flow_id_forward"] = src_ip + "-" + dst_ip + "-" + src_port + "-" + dst_port + "-" + protocol
    packet_identifiers_df["flow_id_reverse"] = dst_ip + "-" + src_ip + "-" + dst_port + "-" + src_port + "-" + protocol
    return packet_identifiers_df

def build_aggregated_flows():
    flow_df = load_all_datasets(FLOW_DATA_FILE_PATH, FILE_GROUPS)
    aggregated_df, identifier_df = aggregate_flows(flow_df)
    identifier_df.to_csv(OUTPUT_DIR / "flow_data_ids.csv", index=False)
    return aggregated_df

def locate_corresponding_flows(packet_identifiers_df, aggregated_flows_df):
    packet_identifiers_df = build_flow_ids(packet_identifiers_df)
    packet_identifiers_df = packet_identifiers_df.rename(columns={"label": "packet_label"})
    forward_matches = packet_identifiers_df.merge(aggregated_flows_df, left_on="flow_id_forward", right_on="Flow ID", how="inner")
    still_unmatched = packet_identifiers_df[~packet_identifiers_df["id"].isin(forward_matches["id"])]
    reverse_matches = still_unmatched.merge(aggregated_flows_df, left_on="flow_id_reverse", right_on="Flow ID", how="inner")
    matched_df = pd.concat([forward_matches, reverse_matches], ignore_index=True)
    matched_df["label_agrees"] = matched_df["packet_label"] == matched_df["label"]
    matched_df = (matched_df.sort_values("label_agrees", ascending=False).drop_duplicates(subset="id", keep="first").drop(columns="label_agrees"))
    matched_df = matched_df.rename(columns={"id": "packet_id"})
    print(f"[flow-correspondence] Matched {len(matched_df):,}/{len(packet_identifiers_df):,} packets to a flow-level record")
    return matched_df

def random_sampling(df, n_samples, random_state=None):
    if len(df) < n_samples:
        return df.copy()

    return df.sample(n=n_samples, random_state=random_state)

def subsample_by_label(flow_df, n_benign=200_000, n_attack=1200, random_state=RANDOM_STATE):
    label_dfs = {label: flow_df[flow_df['label'] == label] for label in flow_df['label'].unique()}
    sampled = []
    for label, sub_df in label_dfs.items():
        n = n_benign if label == 'benign' else n_attack
        sampled.append(random_sampling(sub_df, n, random_state=random_state))

    sampled_df = pd.concat(sampled, ignore_index=True)
    print(f"Sampled DataFrame shape: {sampled_df.shape}")
    print(f"Number of Label Groups: {sampled_df['label'].unique()}")
    print(f"Number of Samples from each Label:\n{sampled_df['label'].value_counts()}")
    return sampled_df

def impute_missing_feature_values(flow_df):
    for column_name in flow_df.columns:
        if column_name not in ["Flow ID", "label"]:
            flow_df[column_name] = flow_df[column_name].replace([np.inf, -np.inf], np.nan)
            flow_df[column_name] = flow_df.groupby("label")[column_name].transform(lambda x: x.fillna(x.mean()))

    return flow_df

def remove_outliers_per_label(flow_df, contamination_benign=0.05, contamination_attack=0.01):
    normal_rows = []
    feature_columns = []
    for column_name in flow_df.columns:
        if column_name not in ["Flow ID", "label"]:
            feature_columns.append(column_name)

    for label in flow_df["label"].unique():
        flow_by_label_df = flow_df[flow_df["label"] == label]
        if label == "benign":
            isolation_forest = IsolationForest(contamination=contamination_benign, random_state=RANDOM_STATE)
        else:
            isolation_forest = IsolationForest(contamination=contamination_attack, random_state=RANDOM_STATE)

        prediction = isolation_forest.fit_predict(flow_by_label_df[feature_columns])
        normal_rows.append(flow_by_label_df[prediction == 1])

    return pd.concat(normal_rows, ignore_index=True)

def log_transform_skewed(X):
    X = X.copy()
    X[X.columns] = np.log1p(X[X.columns].clip(lower=0))
    print(f"[feature-eng] log1p applied to {len(X.columns)} heavy-tailed columns")
    return X

def normalization():
    pipeline = Pipeline([('scaler', StandardScaler())])
    return pipeline

def PCA_reduction():
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95))])
    return pca_pipeline

def pca_contribution(pca, X):
    weights = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])])
    weights.to_csv(OUTPUT_DIR / "pca_feature_contribution.csv")

def normalize_alert_flows(alert_flows, feature_columns, fill_means, scaler):
    meta_cols = [c for c in ["packet_id", "Flow ID", "packet_label"] if c in alert_flows.columns]
    A = alert_flows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(fill_means)
    A = log_transform_skewed(A)
    scaled = pd.DataFrame(scaler.transform(A), columns=feature_columns)
    for c in meta_cols:
        scaled[c] = alert_flows[c].values
    scaled.to_csv(OUTPUT_DIR / "phase2_alert_corresponding_flows.csv", index=False)
    print(f"Saved {len(scaled):,} normalized Phase 2 alert flows")


def main():
    aggregated_flows_df = build_aggregated_flows()

    flow_df = subsample_by_label(aggregated_flows_df)
    flow_df = impute_missing_feature_values(flow_df)
    feature_columns = [c for c in flow_df.columns if c not in ["Flow ID", "label"]]
    fill_means = flow_df[feature_columns].mean()
    flow_df[feature_columns] = log_transform_skewed(flow_df[feature_columns])
    flow_df = remove_outliers_per_label(flow_df)
    flow_ids = flow_df["Flow ID"].values
    X = flow_df[feature_columns]
    y = flow_df["label"]

    scaler = normalization().fit(X)
    full_ds = pd.DataFrame(scaler.transform(X), columns=feature_columns)
    full_ds["Flow ID"] = flow_ids
    full_ds["label"] = y.values
    full_ds.to_csv(OUTPUT_DIR / "normalized_original_data.csv", index=False)

    pca_pipeline = PCA_reduction()
    pca_arr = pca_pipeline.fit_transform(X)
    pca_ds = pd.DataFrame(pca_arr, columns=[f"PC{i+1}" for i in range(pca_arr.shape[1])])
    pca_ds["Flow ID"] = flow_ids
    pca_ds["label"] = y.values
    pca_ds.to_csv(OUTPUT_DIR / "pca_data.csv", index=False)
    pca_contribution(pca_pipeline.named_steps["pca"], X)

    phase2_alert_ids = pd.read_csv(PHASE2_ALERTS_FILE, usecols=["id"])["id"]
    alerted = load_packet_identifiers(packet_ids=phase2_alert_ids)
    alert_flows = locate_corresponding_flows(alerted, aggregated_flows_df)
    normalize_alert_flows(alert_flows, feature_columns, fill_means, scaler)


if __name__ == "__main__":
    main()
