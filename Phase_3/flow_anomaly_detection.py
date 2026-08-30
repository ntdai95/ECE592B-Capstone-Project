import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

RANDOM_STATE = 42

FLOW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data" / "flow-based-features"
FILE_GROUPS = {
    "benign":       ["BenignTraffic.pcap_Flow.csv", "BenignTraffic1.pcap_Flow.csv",
                      "BenignTraffic2.pcap_Flow.csv", "BenignTraffic3.pcap_Flow.csv"],
    "dos":          ["DoS-HTTP_Flood.pcap_Flow.csv", "DoS-HTTP_Flood1.pcap_Flow.csv"],
    "ddos":         ["DDoS-HTTP_Flood-.pcap_Flow.csv"],
    "brute_force":  ["DictionaryBruteForce.pcap_Flow.csv"],
    "dns_spoofing": ["DNS_Spoofing.pcap_Flow.csv"],
    "xss":          ["XSS.pcap_Flow.csv"],
}

KMEANS_K_VALUES = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25]
KMEANS_PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]
AE_BOTTLENECK_SIZES = [4, 8, 12, 16]
AE_PERCENTILES = [50, 60, 70, 80, 85, 90, 92, 94, 95, 96, 98, 99]
WIDENED_K_VALUES = [2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

FINAL_BOTTLENECK_SIZE = 12
FINAL_THRESHOLD_PERCENTILE = 90

NON_FEATURE_COLUMNS = [
    "packet_id", "flow_id_forward", "flow_id_reverse",
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "stream", "src_mac", "dst_mac", "src_ip", "dst_ip", "src_port", "dst_port",
    "device_mac", "eth_src_oui", "eth_dst_oui", "l4_tcp", "l4_udp", "protocol",
    "packet_label",
]


def build_flow_ids(df, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col):
    protocol = df[protocol_col].astype(int).astype(str)
    src_ip = df[src_ip_col].fillna('nan').astype(str)
    dst_ip = df[dst_ip_col].fillna('nan').astype(str)
    src_port = df[src_port_col].astype(int).astype(str)
    dst_port = df[dst_port_col].astype(int).astype(str)

    df = df.copy()
    df["flow_id_forward"] = src_ip + "-" + dst_ip + "-" + src_port + "-" + dst_port + "-" + protocol
    df["flow_id_reverse"] = dst_ip + "-" + src_ip + "-" + dst_port + "-" + src_port + "-" + protocol
    return df


def aggregate_flows(flow_df):
    identifier_columns = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]
    identifier_df = flow_df[["Flow ID"] + identifier_columns].drop_duplicates("Flow ID")
    flow_df = flow_df.drop(columns=identifier_columns + ["Label"])

    columns_aggregation_methods = {}
    for column_name in flow_df.columns:
        if column_name in ["Flow ID", "label"]:
            continue
        if "Max" in column_name:
            columns_aggregation_methods[column_name] = "max"
        elif "Min" in column_name:
            columns_aggregation_methods[column_name] = "min"
        else:
            columns_aggregation_methods[column_name] = "mean"
            for word in ["Total", "Count", "Subflow", "Flags", "Duration", "Header Length", "Pkts"]:
                if word in column_name:
                    columns_aggregation_methods[column_name] = "sum"
                    break

    aggregated_df = flow_df.groupby(["Flow ID", "label"], as_index=False).agg(columns_aggregation_methods)
    return aggregated_df, identifier_df


def locate_corresponding_flows(packet_df, aggregated_flows_df):
    packet_df = packet_df.rename(columns={"label": "packet_label"})

    forward_matches = packet_df.merge(
        aggregated_flows_df, left_on="flow_id_forward", right_on="Flow ID", how="inner"
    )
    still_unmatched = packet_df[~packet_df["id"].isin(forward_matches["id"])]

    reverse_matches = still_unmatched.merge(
        aggregated_flows_df, left_on="flow_id_reverse", right_on="Flow ID", how="inner"
    )

    matched_df = pd.concat([forward_matches, reverse_matches], ignore_index=True)
    matched_df["label_agrees"] = matched_df["packet_label"] == matched_df["label"]
    matched_df = (
        matched_df.sort_values("label_agrees", ascending=False)
        .drop_duplicates(subset="id", keep="first")
        .drop(columns="label_agrees")
    )
    matched_df = matched_df.rename(columns={"id": "packet_id"})

    print(f"Matched {len(matched_df):,}/{len(packet_df):,} packets to a flow-level record")
    return matched_df


def impute_missing(df, feature_columns, reference_means):
    df = df.copy()
    for col in feature_columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(reference_means[col])
    return df


def log_transform(df, feature_columns):
    df = df.copy()
    df[feature_columns] = np.log1p(df[feature_columns].clip(lower=0))
    return df


def kmeans_anomaly_scores(kmeans_model, X):
    distances = kmeans_model.transform(X)
    return distances.min(axis=1)


def evaluate_predictions(y_true_binary, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "fnr": fnr}


def build_autoencoder(input_dim, bottleneck_size):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(48, activation="relu"),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(bottleneck_size, activation="relu"),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(48, activation="relu"),
        keras.layers.Dense(input_dim, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def ae_anomaly_scores(model, X):
    reconstructions = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructions), axis=1)


def distance_to_ideal(row):
    return np.sqrt((1 - row["recall"])**2 + row["fpr"]**2)


def sweep_kmeans(X_train, X_val, val_labels_binary, k_values, percentiles):
    rows = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_train)

        train_scores = kmeans_anomaly_scores(kmeans, X_train)
        val_scores = kmeans_anomaly_scores(kmeans, X_val)

        for percentile in percentiles:
            threshold = np.percentile(train_scores, percentile)
            val_predictions = (val_scores > threshold).astype(int)
            metrics = evaluate_predictions(val_labels_binary, val_predictions)
            rows.append({"k": k, "threshold_percentile": percentile, "threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def sweep_autoencoder(X_train, X_val, val_labels_binary, bottleneck_sizes, percentiles):
    rows = []
    for bottleneck_size in bottleneck_sizes:
        autoencoder = build_autoencoder(X_train.shape[1], bottleneck_size)
        autoencoder.fit(X_train, X_train, epochs=30, batch_size=256, validation_split=0.1, verbose=0, shuffle=True)

        train_scores = ae_anomaly_scores(autoencoder, X_train)
        val_scores = ae_anomaly_scores(autoencoder, X_val)

        for percentile in percentiles:
            threshold = np.percentile(train_scores, percentile)
            val_predictions = (val_scores > threshold).astype(int)
            metrics = evaluate_predictions(val_labels_binary, val_predictions)
            rows.append({"bottleneck_size": bottleneck_size, "threshold_percentile": percentile, "threshold": threshold, **metrics})
        print(f"bottleneck_size={bottleneck_size} done")
    return pd.DataFrame(rows)


def main():
    tf.random.set_seed(RANDOM_STATE)

    PACKET_DATA_DIR = PROJECT_ROOT / "data" / "processed_data" / "packet-data"
    packet_ids_df = pd.read_csv(PACKET_DATA_DIR / "packet_data_ids.csv")

    PROTOCOL_FLAGS_FILE = PACKET_DATA_DIR / "normalized_original_data.csv"
    protocol_df = pd.read_csv(PROTOCOL_FLAGS_FILE, usecols=["id", "l4_tcp", "l4_udp", "label"])

    packet_identifiers_df = packet_ids_df.merge(protocol_df, on="id", how="inner")
    packet_identifiers_df["protocol"] = np.select(
        [packet_identifiers_df["l4_tcp"] > 0, packet_identifiers_df["l4_udp"] > 0],
        [6, 17],
        default=np.nan,
    )
    print(packet_identifiers_df["protocol"].value_counts(dropna=False))

    matchable_df = packet_identifiers_df.dropna(subset=["protocol"]).copy()
    matchable_df["protocol"] = matchable_df["protocol"].astype(int)
    print(matchable_df["label"].value_counts())

    matchable_df = build_flow_ids(
        matchable_df,
        src_ip_col="src_ip", dst_ip_col="dst_ip",
        src_port_col="src_port", dst_port_col="dst_port",
        protocol_col="protocol",
    )

    flow_frames = []
    for label, filenames in FILE_GROUPS.items():
        for filename in filenames:
            df = pd.read_csv(FLOW_DATA_DIR / filename, low_memory=False)
            df["label"] = label
            flow_frames.append(df)

    raw_flow_df = pd.concat(flow_frames, ignore_index=True)
    print("Total combined shape:", raw_flow_df.shape)

    aggregated_flows_df, flow_identifier_df = aggregate_flows(raw_flow_df)
    print("Aggregated flows shape:", aggregated_flows_df.shape)

    matched_df = locate_corresponding_flows(matchable_df, aggregated_flows_df)
    print(matched_df["packet_label"].value_counts())

    feature_columns = [c for c in matched_df.columns if c not in NON_FEATURE_COLUMNS + ["label"]]

    benign_df = matched_df[matched_df["packet_label"] == "benign"]
    attack_df = matched_df[matched_df["packet_label"] != "benign"]

    benign_train, benign_temp = train_test_split(benign_df, test_size=0.4, random_state=RANDOM_STATE)
    benign_val, benign_test = train_test_split(benign_temp, test_size=0.5, random_state=RANDOM_STATE)

    attack_val, attack_test = train_test_split(
        attack_df, test_size=0.5, random_state=RANDOM_STATE, stratify=attack_df["packet_label"]
    )

    val_df = pd.concat([benign_val, attack_val], ignore_index=True)
    test_df = pd.concat([benign_test, attack_test], ignore_index=True)

    print(f"Train (benign only): {len(benign_train):,}")
    print(f"Val: {len(val_df):,} ({val_df['packet_label'].value_counts().to_dict()})")
    print(f"Test: {len(test_df):,} ({test_df['packet_label'].value_counts().to_dict()})")

    train_means = benign_train[feature_columns].replace([np.inf, -np.inf], np.nan).mean()
    benign_train = impute_missing(benign_train, feature_columns, train_means)
    val_df = impute_missing(val_df, feature_columns, train_means)
    test_df = impute_missing(test_df, feature_columns, train_means)

    iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    train_predictions = iso_forest.fit_predict(benign_train[feature_columns])
    benign_train_clean = benign_train[train_predictions == 1]
    print(f"Removed {sum(train_predictions == -1):,} outliers from training set")

    benign_train_clean = log_transform(benign_train_clean, feature_columns)
    val_df = log_transform(val_df, feature_columns)
    test_df = log_transform(test_df, feature_columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(benign_train_clean[feature_columns])
    X_val = scaler.transform(val_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    val_labels_binary = (val_df["packet_label"] != "benign").astype(int).values

    kmeans_results_wide_df = sweep_kmeans(X_train, X_val, val_labels_binary, KMEANS_K_VALUES, KMEANS_PERCENTILES)
    eligible_wide = kmeans_results_wide_df[(kmeans_results_wide_df["recall"] >= 0.90) & (kmeans_results_wide_df["fpr"] <= 0.15)]
    print(f"Eligible configs (Recall >= 90%, FPR <= 15%): {len(eligible_wide)}")

    ae_results_df = sweep_autoencoder(X_train, X_val, val_labels_binary, AE_BOTTLENECK_SIZES, AE_PERCENTILES)
    eligible_ae = ae_results_df[(ae_results_df["recall"] >= 0.90) & (ae_results_df["fpr"] <= 0.15)]
    print(f"Eligible configs (Recall >= 90%, FPR <= 15%): {len(eligible_ae)}")

    kmeans_best_by_auc, best_kmeans_auc = None, -1
    for k in KMEANS_K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_train)
        val_scores = kmeans_anomaly_scores(kmeans, X_val)
        auc = roc_auc_score(val_labels_binary, val_scores)
        if auc > best_kmeans_auc:
            best_kmeans_auc, kmeans_best_by_auc = auc, k
    print(f"Best K-Means AUC-ROC: {best_kmeans_auc:.4f} (k={kmeans_best_by_auc})")

    ae_best_by_auc, best_ae_auc = None, -1
    for bottleneck_size in AE_BOTTLENECK_SIZES:
        autoencoder = build_autoencoder(X_train.shape[1], bottleneck_size)
        autoencoder.fit(X_train, X_train, epochs=30, batch_size=256, validation_split=0.1, verbose=0, shuffle=True)
        val_scores = ae_anomaly_scores(autoencoder, X_val)
        auc = roc_auc_score(val_labels_binary, val_scores)
        if auc > best_ae_auc:
            best_ae_auc, ae_best_by_auc = auc, bottleneck_size
    print(f"Best Autoencoder AUC-ROC: {best_ae_auc:.4f} (bottleneck_size={ae_best_by_auc})")

    best_bottleneck = 12
    autoencoder = build_autoencoder(X_train.shape[1], best_bottleneck)
    autoencoder.fit(X_train, X_train, epochs=30, batch_size=256, validation_split=0.1, verbose=0, shuffle=True)

    val_scores = ae_anomaly_scores(autoencoder, X_val)
    fpr_curve, tpr_curve, thresholds_curve = roc_curve(val_labels_binary, val_scores)
    candidates = pd.DataFrame({"threshold": thresholds_curve, "recall": tpr_curve, "fpr": fpr_curve})
    eligible_curve = candidates[(candidates["recall"] >= 0.90) & (candidates["fpr"] <= 0.15)]
    print(f"Eligible thresholds along the full ROC curve: {len(eligible_curve)}")

    ae_results_df["distance_to_ideal"] = ae_results_df.apply(distance_to_ideal, axis=1)
    best_ae_row = ae_results_df.sort_values("distance_to_ideal").iloc[0]
    print("Best Autoencoder config:\n", best_ae_row)

    kmeans_results_wide_df["distance_to_ideal"] = kmeans_results_wide_df.apply(distance_to_ideal, axis=1)
    best_kmeans_row = kmeans_results_wide_df.sort_values("distance_to_ideal").iloc[0]
    print("\nBest K-Means config:\n", best_kmeans_row)

    kmeans_results_widened_df = sweep_kmeans(X_train, X_val, val_labels_binary, WIDENED_K_VALUES, KMEANS_PERCENTILES)
    eligible_widened = kmeans_results_widened_df[(kmeans_results_widened_df["recall"] >= 0.90) & (kmeans_results_widened_df["fpr"] <= 0.15)]
    print(f"Eligible configs (Recall >= 90%, FPR <= 15%): {len(eligible_widened)}")

    kmeans_results_widened_df["distance_to_ideal"] = kmeans_results_widened_df.apply(distance_to_ideal, axis=1)
    best_kmeans_widened = kmeans_results_widened_df.sort_values("distance_to_ideal").iloc[0]
    print("\nBest K-Means config (widened search):\n", best_kmeans_widened)

    print("\nAUC by k (checking for plateau):")
    for k in WIDENED_K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_train)
        val_scores = kmeans_anomaly_scores(kmeans, X_val)
        auc = roc_auc_score(val_labels_binary, val_scores)
        print(f"  k={k}: AUC={auc:.4f}")

    final_bottleneck_size = FINAL_BOTTLENECK_SIZE
    final_threshold_percentile = FINAL_THRESHOLD_PERCENTILE

    final_autoencoder = build_autoencoder(X_train.shape[1], final_bottleneck_size)
    final_autoencoder.fit(X_train, X_train, epochs=30, batch_size=256, validation_split=0.1, verbose=0, shuffle=True)

    train_scores_final = ae_anomaly_scores(final_autoencoder, X_train)
    final_threshold = np.percentile(train_scores_final, final_threshold_percentile)

    test_scores = ae_anomaly_scores(final_autoencoder, X_test)
    test_labels_binary = (test_df["packet_label"] != "benign").astype(int).values
    test_predictions = (test_scores > final_threshold).astype(int)

    test_metrics = evaluate_predictions(test_labels_binary, test_predictions)
    test_auc = roc_auc_score(test_labels_binary, test_scores)

    print(f"=== FINAL TEST SET RESULTS (Autoencoder, bottleneck={final_bottleneck_size}, percentile={final_threshold_percentile}) ===")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    print(f"Comparison to Phase 2 baseline (0.848): {test_auc - 0.848:+.4f}")

    full_scores = ae_anomaly_scores(final_autoencoder, scaler.transform(
        log_transform(impute_missing(matched_df, feature_columns, train_means), feature_columns)[feature_columns]
    ))

    export_df = matched_df[["packet_id", "Flow ID", "packet_label"]].copy()
    export_df["flow_anomaly_score"] = full_scores

    clean_export_df = export_df.groupby("Flow ID", as_index=False)["flow_anomaly_score"].mean()
    clean_export_df["flow_anomaly_flag"] = (clean_export_df["flow_anomaly_score"] > final_threshold).astype(int)

    output_dir = PROJECT_ROOT / "data" / "processed_data" / "flow-data"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_export_df.to_csv(output_dir / "task3_1_flow_anomaly_scores.csv", index=False)
    print(clean_export_df.shape)

    category_results = []
    for category in ["ddos", "dos", "xss", "dns_spoofing", "brute_force"]:
        category_mask = (test_df["packet_label"] == category) | (test_df["packet_label"] == "benign")
        category_scores = test_scores[category_mask.values]
        category_pred = (category_scores > final_threshold).astype(int)

        attack_only_mask = test_df.loc[category_mask, "packet_label"] == category
        attack_recall = category_pred[attack_only_mask.values].mean()

        category_results.append({"category": category, "n_test_samples": attack_only_mask.sum(), "recall": attack_recall})

    print(pd.DataFrame(category_results))


if __name__ == "__main__":
    main()
