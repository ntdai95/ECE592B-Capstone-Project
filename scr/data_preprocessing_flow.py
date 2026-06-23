import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


FLOW_DATA_FILE_PATH = "data/raw_data/flow-based-features"
OUTPUT_DATA_FILE_PATH = "data/processed_data/flow-data"
FLOW_DATA_FILENAMES = {"benign": ["BenignTraffic.pcap_Flow.csv", "BenignTraffic1.pcap_Flow.csv", 
                                  "BenignTraffic2.pcap_Flow.csv", "BenignTraffic3.pcap_Flow.csv"],
                       "ddos": ["DDoS-HTTP_Flood-.pcap_Flow.csv"], 
                       "brute_force": ["DictionaryBruteForce.pcap_Flow.csv"], 
                       "dns_spoofing": ["DNS_Spoofing.pcap_Flow.csv"], 
                       "dos": ["DoS-HTTP_Flood.pcap_Flow.csv", "DoS-HTTP_Flood1.pcap_Flow.csv"], 
                       "xss": ["XSS.pcap_Flow.csv"]}

# Load Flow-level Data 
def load_all_datasets(path, filenames):
    flow_data = []
    for label, files in filenames.items():
        flow_df = []
        for file in files:
            flow_df.append(pd.read_csv(path + "/" + file))
        
        flow_df = pd.concat(flow_df, ignore_index=True)
        flow_df = flow_df.drop(columns=["Label"])
        flow_df['label'] = label
        flow_data.append(flow_df)
    
    return pd.concat(flow_data, ignore_index=True)

# Aggregate Flows into one unified record per Flow ID
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

# Based on Task 3.2 copying function for random sampling in Task 1.1 from the data_preprocessing.py (for packet-level data).
def random_sampling(df, n_samples, random_state=None):
    if len(df) >= n_samples:
        sampled_df = df.sample(n=n_samples, random_state=random_state)
        return sampled_df
    else:
        raise ValueError("Don't have enough samples in the dataframe.")

def subsample_by_label(flow_df, n_benign=200_000, n_attack=1200, random_state=42):
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

# Handle missing feature values by imputing the mean for continuous, and the mode for categorical (no categorical column)
def impute_missing_feature_values(flow_df):
    for column_name in flow_df.columns:
        if column_name not in ["Flow ID", "label"]:
            flow_df[column_name] = flow_df[column_name].replace([np.inf, -np.inf], np.nan)
            # Duplicated Flow Ids have been removed in the aggregate_flows() function above
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
            iso = IsolationForest(contamination=contamination_benign, random_state=42)
        else:
            iso = IsolationForest(contamination=contamination_attack, random_state=42)
        
        prediction = iso.fit_predict(flow_by_label_df[feature_columns])
        normal_rows.append(flow_by_label_df[prediction == 1])
    
    return pd.concat(normal_rows, ignore_index=True)

## Normalization/Scaling and Dimension Reduction 
# copying the implemented function for scaling and PCA from the data_preprocessing.py (for packet-level data).
def normalization():
    pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])
    return pipeline

def PCA_reduction():
    pca_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=0.95))
    ])

    return pca_pipeline

def pca_contribution(pca, X):
    weights = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    weights.to_csv(OUTPUT_DATA_FILE_PATH + "/pca_feature_contribution.csv")

def main():
    flow_df = load_all_datasets(FLOW_DATA_FILE_PATH, FLOW_DATA_FILENAMES)
    # Based on Phase 3 aggregating flows into one unified record per Flow ID (and label)
    flow_df, identifier_df = aggregate_flows(flow_df)
    identifier_df.to_csv(OUTPUT_DATA_FILE_PATH + "/flow_data_ids.csv", index=False)
    flow_df = subsample_by_label(flow_df)
    flow_df = impute_missing_feature_values(flow_df)
    flow_df = remove_outliers_per_label(flow_df)

    flow_ids = flow_df["Flow ID"].values
    feature_columns = []
    for column_name in flow_df.columns:
        if column_name not in ["Flow ID", "label"]:
            feature_columns.append(column_name)

    X = flow_df[feature_columns]
    y = flow_df["label"]

    ## Normalized Original Dataset (no dimension reduction)
    # copying the calling of implemented function for scaling and PCA from the data_preprocessing.py (for packet-level data).
    full_ds = pd.DataFrame(normalization().fit_transform(X), columns=X.columns)
    full_ds["Flow ID"] = flow_ids
    full_ds["label"] = y.values
    full_ds.to_csv(OUTPUT_DATA_FILE_PATH + "/normalized_original_data.csv", index=False)

    # PCA Reduced Data
    pca_pipeline = PCA_reduction()
    pca_arr = pca_pipeline.fit_transform(X)
    pca_ds = pd.DataFrame(pca_arr, columns=[f"PC{i+1}" for i in range(pca_arr.shape[1])])
    pca_ds["Flow ID"] = flow_ids
    pca_ds["label"] = y.values
    pca_ds.to_csv(OUTPUT_DATA_FILE_PATH + "/pca_data.csv", index=False)

    pca = pca_pipeline.named_steps["pca"]
    pca_contribution(pca, X)


if __name__ == "__main__":
    # how to run (from root folder): python -m scr.data_preprocessing_flow
    main()