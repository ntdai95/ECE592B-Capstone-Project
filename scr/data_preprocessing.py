import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

def fill_na(fill_zero_cols, mean_impute_cols, mode_impute_cols):
    imputer = ColumnTransformer([
        ('zero', SimpleImputer(strategy='constant', fill_value=0), fill_zero_cols),
        ('mean', SimpleImputer(strategy='mean'), mean_impute_cols),
        ('cat', Pipeline([
            ('mode', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), mode_impute_cols),
    ], remainder='passthrough')

    return imputer

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


def clean_all_labels(sampled_df, fill_zero_cols, mean_impute_cols, mode_impute_cols, id_cols,
                      contamination_benign=0.05, contamination_attack=0.01):
    """Impute + encode globally, then run outlier removal within each label."""
    drop_cols = id_cols + ['inter_arrival_time', 'label']
    feature_cols = [c for c in sampled_df.columns if c not in drop_cols and c != 'id']

    X = sampled_df[feature_cols + ['id']].copy()

    # Some categorical cols in the raw CSVs mix ints with strings (e.g. http_response_code
    # carries 200 and 'NeedManualLabel'). OneHotEncoder rejects mixed types, so coerce
    # non-null values to str while keeping NaN intact for the imputer.
    for c in mode_impute_cols:
        X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else v)

    imputer = fill_na(fill_zero_cols, mean_impute_cols, mode_impute_cols)
    X_imp = imputer.fit_transform(X)
    out_cols = [c.split('__', 1)[-1] for c in imputer.get_feature_names_out()]
    X_imp = pd.DataFrame(X_imp, columns=out_cols)
    X_imp['label'] = sampled_df['label'].values

    processed = []
    for label in X_imp['label'].unique():
        sub = X_imp[X_imp['label'] == label].drop(columns='label')
        contamination = contamination_benign if label == 'benign' else contamination_attack
        kept = IsolationForestFilter(contamination=contamination).fit_transform(sub)
        if not isinstance(kept, pd.DataFrame):
            kept = pd.DataFrame(kept, columns=sub.columns)
        else:
            kept = kept.copy()
        kept['label'] = label
        processed.append(kept)

    return pd.concat(processed, ignore_index=True)

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

def LDA_reduction():
    lda_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('lda', LDA())
    ])

    return lda_pipeline

def lda_contribution(lda, X):
    # Per-feature weights for each of the discriminant directions
    weights = pd.DataFrame(
        lda.scalings_,
        index=X.columns,
        columns=[f"LD{i+1}" for i in range(lda.scalings_.shape[1])]
    )
    weights.to_csv(OUTPUT_DIR/"lda_feature_contribution.csv")

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

    # Keep id out of the feature matrix so it survives scaling/PCA/LDA and we can
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

    # LDA Reduced Data
    lda_pipeline = LDA_reduction()
    lda_arr = lda_pipeline.fit_transform(X, y)
    lda_ds = pd.DataFrame(lda_arr, columns=[f"LD{i+1}" for i in range(lda_arr.shape[1])])
    lda_ds['id'] = ids.values
    lda_ds['label'] = y.values
    lda_ds.to_csv(OUTPUT_DIR/"lda_data.csv", index=False)

    lda = lda_pipeline.named_steps['lda']
    lda_contribution(lda, X)


if __name__ == "__main__":
    main()
