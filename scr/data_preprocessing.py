import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# File names
BENIGN_DATA_FILE_NAMES = ["BenignTraffic.csv",
                           "BenignTraffic1.csv",
                           "BenignTraffic2.csv",
                           "BenignTraffic3.csv",
                          ]

DOS_DATA_FILE_NAMES = ["DoS-HTTP_Flood.csv",
                       "DoS-HTTP_Flood1.csv"
                      ]

BRUTE_FORCE_FILE_NAME = ["DictionaryBruteForce.csv"]

DDOS_FILE_NAME = ["DDoS-HTTP_Flood-.csv"]

DNS_SPOOFING_FILE_NAME = ["DNS_Spoofing.csv"]

XSS_FILE_NAME = ["XSS.csv"]

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
PACKET_DATA_FILE_PATH = PROJECT_ROOT / "data" / "raw_data" / "packet-based-features"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_data" / "packet-data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Datasets
def load_packet_data(path, file_name):
    if len(file_name) > 1:
        packet_df = []
        for file in file_name:
            df = pd.read_csv(path/file)
            packet_df.append(df)

        packet_df = pd.concat(packet_df, ignore_index=True)
    else:
        packet_df = pd.read_csv(path / file_name[0])

    return packet_df

def load_all_datasets(path):
    sources = {
        'benign': BENIGN_DATA_FILE_NAMES,
        'dos': DOS_DATA_FILE_NAMES,
        'brute_force': BRUTE_FORCE_FILE_NAME,
        'ddos': DDOS_FILE_NAME,
        'dns_spoofing': DNS_SPOOFING_FILE_NAME,
        'xss': XSS_FILE_NAME,
    }
    dfs = []
    for label, file_names in sources.items():
        df = load_packet_data(path=path, file_name=file_names)
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Task 1.1. Create Function for Random Sampling
def random_sampling(df, n_samples, random_state=None):
    if len(df) >= n_samples:
        sampled_df = df.sample(n=n_samples, random_state=random_state)
        return sampled_df
    else:
        raise ValueError("Don't have enough samples in the dataframe.")

def subsample_by_label(packet_df, n_benign=200_000, n_attack=1200, random_state=42):
    label_dfs = {label: packet_df[packet_df['label'] == label] for label in packet_df['label'].unique()}
    sampled = []
    for label, sub_df in label_dfs.items():
        n = n_benign if label == 'benign' else n_attack
        sampled.append(random_sampling(sub_df, n, random_state=random_state))
    sampled_df = pd.concat(sampled, ignore_index=True)

    print(f"Sampled DataFrame shape: {sampled_df.shape}")
    print(f"Number of Label Groups: {sampled_df['label'].unique()}")
    print(f"Number of Samples from each Label:\n{sampled_df['label'].value_counts()}")
    return sampled_df

# Task 1.2: Data Preprocessing
## Handle Missing Values
def binary_flags(df, flag_cols):
    df[flag_cols] = df[flag_cols].replace(['none', 'None', 'NaN'], np.nan).infer_objects(copy=False)
    for col in flag_cols:
        df[f"has_{col}"] = df[col].notna().astype(int)

    new_df = df.drop(columns=flag_cols)
    return new_df

## Extract Identifiers
def extract_identifiers(df, id_cols, output_path=None):
    """Assign a permanent row id, split off identifier columns into a lookup table."""
    df = df.reset_index(drop=True)
    df['id'] = df.index
    identifiers_lookup = df[['id'] + id_cols].copy()
    if output_path:
        identifiers_lookup.to_csv(output_path, index=False)
    return df, identifiers_lookup

## Outlier Detection using Isolation Forest
class IsolationForestFilter(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.5, random_state=42):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        self.iso_ = IsolationForest(contamination=self.contamination,
                                    random_state=self.random_state)
        self.iso_.fit(X)
        return self

    def transform(self, X):
        preds = self.iso_.predict(X)
        mask = preds == 1
        if isinstance(X, pd.DataFrame):
            return X.loc[mask]
        return X[mask]


def _impute_per_label(df, feature_cols, fill_zero_cols, mean_impute_cols, mode_impute_cols):
    """Fill NaNs within each label so attack and benign keep their own means."""
    chunks = []
    labels = df['label'].unique()

    for label in labels:
        sub = df.loc[df['label'] == label, feature_cols + ['id']].copy()

        sub[fill_zero_cols] = sub[fill_zero_cols].fillna(0)
        for c in mean_impute_cols:
            sub[c] = sub[c].fillna(sub[c].mean())
        for c in mode_impute_cols:
            mode_val = sub[c].mode(dropna=True)
            fill = mode_val.iloc[0] if not mode_val.empty else ''
            sub[c] = sub[c].fillna(fill).map(str)

        sub['label'] = label
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True)

def _encode_categoricals(df, mode_impute_cols):
    """Global one-hot encoding so every label sees the same category columns."""
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    arr = ohe.fit_transform(df[mode_impute_cols])
    encoded = pd.DataFrame(arr,
                           columns=ohe.get_feature_names_out(mode_impute_cols),
                           index=df.index)
    return pd.concat([df.drop(columns=mode_impute_cols), encoded], axis=1)

def _remove_outliers_per_label(df, contamination_benign=0.05, contamination_attack=0.01):
    """Isolation Forest per label."""
    final_df = []
    for label in df['label'].unique():
        sub = df[df['label'] == label].drop(columns='label')
        c = contamination_benign if label == 'benign' else contamination_attack
        kept = IsolationForestFilter(contamination=c).fit_transform(sub).copy()
        kept['label'] = label
        final_df.append(kept)
    return pd.concat(final_df, ignore_index=True)

def clean_all_labels(sampled_df, fill_zero_cols, mean_impute_cols, mode_impute_cols, id_cols):
    """
    Stage 1: Impute missing values per label
    Stage 2: One-Hot Encoding on the entire DF
    State 3: Outliers Removal using Isolation Forest per label
        contamination = 0.05 for 'benign'
        contamination = 0.01 for 'attack'
    """
    drop_cols = id_cols + ['inter_arrival_time', 'label']
    feature_cols = [c for c in sampled_df.columns if c not in drop_cols and c != 'id']
    df = _impute_per_label(sampled_df, feature_cols, fill_zero_cols, mean_impute_cols, mode_impute_cols)
    df = _encode_categoricals(df, mode_impute_cols)
    df = _remove_outliers_per_label(df)
    return df


## Normalization/Scaling and Dimension Reduction
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
    weights.to_csv(OUTPUT_DIR/"pca_feature_contribution.csv")


def FOR_reduction(X, y, threshold=0.95, n_estimators=300, random_state=42):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=random_state
    )
    rf.fit(X_scaled, y)

    # Rank features by importance, pick the top N that cover threshold cumulatively
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    n_keep = (importances.cumsum() < threshold).sum() + 1
    kept = importances.head(n_keep).index
    dropped = importances.drop(kept)

    # Keep the original column order in the returned matrix
    mask = X.columns.isin(kept)
    return X_scaled[:, mask], X.columns[mask], rf, dropped


def for_contribution(rf, X):
    contributions = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    contributions.to_csv(OUTPUT_DIR/"for_feature_importance.csv")

def main():
    packet_df = load_all_datasets(PACKET_DATA_FILE_PATH)
    sampled_df = subsample_by_label(packet_df)

    # Create ID column for downstream lookup
    # Identifier columns
    id_cols = [
        'stream',
        'src_mac',
        'dst_mac',
        'src_ip',
        'dst_ip',
        'src_port',
        'dst_port',
        'device_mac',
        'eth_src_oui',
        'eth_dst_oui'
    ]

    sampled_df, _ = extract_identifiers(
        sampled_df, id_cols, output_path=OUTPUT_DIR / 'packet_data_ids.csv'
    )

    # Create Binary Flags for Categorical Columns
    flag_cols = [
        'tls_server',
        'http_request_method',
        'http_host',
        'user_agent',
        'dns_server',
        'http_uri',
        'http_content_type',
    ]

    df = binary_flags(sampled_df, flag_cols)


    # Fill Missing Values and Remove Outliers
    fill_zero_cols = [
        'dns_query_type', 'jitter',
        'stream_1_var', 'src_ip_1_count', 'src_ip_1_mean', 'src_ip_1_var',
        'src_ip_mac_1_var', 'channel_1_var',
        'stream_jitter_1_sum', 'stream_jitter_1_mean', 'stream_jitter_1_var'
    ]

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for c in ['id', 'inter_arrival_time'] + id_cols + flag_cols:
        if c in numeric_cols:
            numeric_cols.remove(c)
    mean_impute_cols = [col for col in numeric_cols if col not in fill_zero_cols]

    categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                        if c not in id_cols + ['label']]

    clean_df = clean_all_labels(df,
                                fill_zero_cols,
                                mean_impute_cols,
                                categorical_cols,
                                id_cols)

    # Keep id out of the feature matrix so it survives scaling/PCA/FOR and we can
    # join back to packet_data_ids.csv in Phase 3.
    ids = clean_df['id'].astype(int)
    y = clean_df['label']
    X = clean_df.drop(columns=['id', 'label'])

    # Normalized Original Dataset (no dimension reduction)
    full_ds = pd.DataFrame(normalization().fit_transform(X), columns=X.columns)
    full_ds['id'] = ids.values
    full_ds['label'] = y.values
    full_ds.to_csv(OUTPUT_DIR/"normalized_original_data.csv", index=False)

    # PCA Reduced Data
    pca_pipeline = PCA_reduction()
    pca_arr = pca_pipeline.fit_transform(X)
    pca_ds = pd.DataFrame(pca_arr, columns=[f"PC{i+1}" for i in range(pca_arr.shape[1])])
    pca_ds['id'] = ids.values
    pca_ds['label'] = y.values
    pca_ds.to_csv(OUTPUT_DIR/"pca_data.csv", index=False)

    pca = pca_pipeline.named_steps['pca']
    pca_contribution(pca, X)

    # Random Forest feature selection (95% cumulative importance)
    for_arr, selected_cols, rf, dropped = FOR_reduction(X, y, threshold=0.95)

    for_ds = pd.DataFrame(for_arr, columns=selected_cols)
    for_ds['id'] = ids.values
    for_ds['label'] = y.values
    for_ds.to_csv(OUTPUT_DIR / "for_data.csv", index=False)

    for_contribution(rf, X)
    dropped.to_csv(OUTPUT_DIR / "for_dropped_features.csv", header=['importance'])


if __name__ == "__main__":
    main()
