import ipaddress
import re

import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# File names
FILE_GROUPS = {
    'benign':       ['BenignTraffic.csv', 'BenignTraffic1.csv',
                     'BenignTraffic2.csv', 'BenignTraffic3.csv'],
    'dos':          ['DoS-HTTP_Flood.csv', 'DoS-HTTP_Flood1.csv'],
    'brute_force':  ['DictionaryBruteForce.csv'],
    'ddos':         ['DDoS-HTTP_Flood-.csv'],
    'dns_spoofing': ['DNS_Spoofing.csv'],
    'xss':          ['XSS.csv'],
}

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
PACKET_DATA_FILE_PATH = PROJECT_ROOT / "data" / "raw_data" / "packet-based-features"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_data" / "packet-data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Packet-level relabeling
#
# The captures are labeled per-pcap, NOT per-packet (see raw data README:
# "The data has not been labeled on a row basis"). Each attack capture also
# recorded the background traffic of the ~45 testbed IoT devices, so most
# rows in an "attack" file are behaviorally benign. We therefore relabel per packet 
# using the attack endpoints, identified empirically from the captures:
#   dos:          192.168.137.66  sends >98k HTTP-flood requests at victims
#   ddos:         192.168.137.139 is the victim; 13 sources flood it on :80
#   brute_force:  192.168.137.65  runs SSH(:22)/RTSP(:554) dictionary attempts
#   xss:          192.168.137.147 is the sole source of URIs with XSS payloads
#   dns_spoofing: dc:a6:32:dc:27:d5 (Raspberry Pi) ARP-spoofs the gateway IP
#                 (192.168.137.1) and forges DNS answers -> match by MAC,
#                 since the MITM attacker impersonates other hosts' IPs.
ATTACK_RULES = {
    'dos':          {'ips':  ['192.168.137.66']},
    'ddos':         {'ips':  ['192.168.137.139']},
    'brute_force':  {'ips':  ['192.168.137.65']},
    'xss':          {'ips':  ['192.168.137.147']},
    'dns_spoofing': {'macs': ['dc:a6:32:dc:27:d5']},
}

def relabel_packets(df, group):
    if group not in ATTACK_RULES:
        df['label'] = group
        return df
    rule = ATTACK_RULES[group]
    mask = pd.Series(False, index=df.index)
    for ip in rule.get('ips', []):
        mask |= (df['src_ip'] == ip) | (df['dst_ip'] == ip)
    for mac in rule.get('macs', []):
        mask |= (df['src_mac'] == mac) | (df['dst_mac'] == mac)
    df['label'] = np.where(mask, group, 'benign')
    print(f"[relabel] {group}: {mask.sum():,}/{len(df):,} packets kept as "
          f"attack ({mask.mean():.1%}); rest -> benign background")
    return df

# Load Datasets
def load_packet_data(path, file_name):
    return pd.concat([pd.read_csv(path / f, low_memory=False) for f in file_name],
                     ignore_index=True)

def load_all_datasets(path):
    dfs = []
    for label, file_names in FILE_GROUPS.items():
        df = load_packet_data(path=path, file_name=file_names)
        df = relabel_packets(df, label)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Task 1.1. Create Function for Random Sampling
def random_sampling(df, n_samples, random_state=None):
    if len(df) < n_samples:
        print(f"[sampling] Warning: requested {n_samples} rows but only "
              f"{len(df)} available; using all of them.")
        return df.copy()
    return df.sample(n=n_samples, random_state=random_state)

def subsample_by_label(packet_df, n_benign=200000, n_attack=1200, random_state=42):
    sampled = []
    for label, sub_df in packet_df.groupby('label'):
        n = n_benign if label == 'benign' else n_attack
        sampled.append(random_sampling(sub_df, n, random_state=random_state))
    sampled_df = pd.concat(sampled, ignore_index=True)

    print(f"Sampled DataFrame shape: {sampled_df.shape}")
    print(f"Number of Label Groups: {sampled_df['label'].unique()}")
    print(f"Number of Samples from each Label:\n{sampled_df['label'].value_counts()}")
    return sampled_df

# Feature Engineering
def _num(s):
    return pd.to_numeric(s, errors='coerce')

URI_SPECIAL_CHARS = r"""[<>'"%;()=&]"""

def engineer_content_features(df):
    """
    Content-level features for application-layer attacks.
    """
    uri = df['http_uri'].astype(str).replace('none', '')
    df['uri_length'] = uri.str.len()
    df['uri_special_chars'] = uri.str.count(URI_SPECIAL_CHARS)
    df['uri_param_count'] = uri.str.count('&') + uri.str.contains(r'\?').astype(int)
    df['uri_depth'] = uri.str.count('/')

    qry = _num(df['dns_len_qry']).fillna(0)
    ans = _num(df['dns_len_ans']).fillna(0)
    df['dns_is_answer'] = (ans > 0).astype(int)
    df['dns_ans_qry_ratio'] = ans / (qry + 1.0)  # spoofed answers are oversized

    payload = _num(df['payload_length']).fillna(0).clip(lower=0)
    eth = _num(df['eth_size']).fillna(0)
    df['payload_ratio'] = payload / (eth + 1.0)  # payload share of the frame
    return df

def engineer_volumetric_features(df):
    """
    Scale-invariant traffic-shape ratios for flood attacks.
    """
    for scope in ['stream', 'src_ip', 'channel']:
        c5  = _num(df[f'{scope}_5_count']).fillna(0)
        c60 = _num(df[f'{scope}_60_count']).fillna(0)
        m5  = _num(df[f'{scope}_5_mean']).fillna(0)
        v5  = _num(df[f'{scope}_5_var']).fillna(0)
        df[f'{scope}_burst_persistence'] = c60 / (c5 + 1.0)  # sustained flood >> 1
        df[f'{scope}_fano_5'] = v5 / (m5.abs() + 1e-6)       # floods: near-0 (constant pkt size)
        df[f'{scope}_rate_5'] = c5 / 5.0                     # packets/second
    return df

# Endpoint semantics + repetition features (brute-force oriented)
# Ports 554 (RTSP) and 8009 are included: the dictionary attack in this
# dataset targets them alongside SSH (22).
LOGIN_OR_WEB_PORTS = frozenset({21, 22, 23, 80, 443, 554, 8009, 8080, 8443})
EPHEMERAL_PORT_START = 49152

def _ip_property_flag(ip_series, attr):
    """int8 flag for an ipaddress property, evaluated once per unique value."""
    def flag(v):
        try:
            return int(getattr(ipaddress.ip_address(str(v)), attr))
        except ValueError:
            return 0
    uniq = ip_series.unique()
    mapping = {v: flag(v) for v in uniq}
    return ip_series.map(mapping).astype('int8')

def engineer_endpoint_semantics(df):
    """Low-dimensional IP/port semantics instead of one-hot raw identifiers."""
    df['src_ip_is_private'] = _ip_property_flag(df['src_ip'], 'is_private')
    df['dst_ip_is_private'] = _ip_property_flag(df['dst_ip'], 'is_private')
    df['dst_ip_is_multicast'] = _ip_property_flag(df['dst_ip'], 'is_multicast')

    src_port = _num(df['src_port']).fillna(0).astype('int32')
    dst_port = _num(df['dst_port']).fillna(0).astype('int32')
    df['dst_port_is_login_or_web'] = dst_port.isin(LOGIN_OR_WEB_PORTS).astype('int8')
    df['dst_port_is_well_known'] = ((dst_port > 0) & (dst_port < 1024)).astype('int8')
    df['src_port_is_ephemeral'] = (src_port >= EPHEMERAL_PORT_START).astype('int8')
    df['has_valid_ports'] = ((src_port > 0) & (dst_port > 0)).astype('int8')
    return df

def _keys(df, cols):
    """Grouping keys as strings so NaN identifiers don't drop rows."""
    return [df[c].astype(str) for c in cols]

def _rolling_group_count(df, cols, window):
    """Packets from the same group among the last `window` rows (inclusive).
    """
    n = len(df)
    group_id = df.groupby(_keys(df, cols), sort=False).ngroup().to_numpy(np.int64)
    enc = group_id * (n + window) + np.arange(n, dtype=np.int64)
    enc_sorted = np.sort(enc)
    left = np.searchsorted(enc_sorted, enc - window + 1, side='left')
    rank = np.searchsorted(enc_sorted, enc, side='right')
    return pd.Series((rank - left).astype('int32'), index=df.index)

def engineer_repetition_features(df):
    """Backward-looking repetition/diversity counts (brute-force signature:
    one source hammering one destination:port, especially login services).

    Counts run over the sampled row order (no timestamp survives
    preprocessing).
    """
    eps = 1e-9
    src = df.groupby(_keys(df, ['src_ip']), sort=False).cumcount() + 1
    src_dst = df.groupby(_keys(df, ['src_ip', 'dst_ip']), sort=False).cumcount() + 1
    src_dst_port = df.groupby(_keys(df, ['src_ip', 'dst_ip', 'dst_port']), sort=False).cumcount() + 1
    df['src_packets_seen_so_far'] = src
    df['src_dst_packets_seen_so_far'] = src_dst
    df['src_dst_port_packets_seen_so_far'] = src_dst_port
    df['stream_packets_seen_so_far'] = df.groupby(_keys(df, ['stream']), sort=False).cumcount() + 1

    # Cumulative fan-out: is this source spreading across many targets/ports?
    def cum_unique(first_seen_cols, group_col):
        first = ~df.duplicated(first_seen_cols)
        return first.groupby(df[group_col].astype(str)).cumsum().astype('int32')

    df['src_unique_dst_ips_seen_so_far'] = cum_unique(['src_ip', 'dst_ip'], 'src_ip')
    df['src_unique_dst_ports_seen_so_far'] = cum_unique(['src_ip', 'dst_port'], 'src_ip')
    df['stream_unique_dst_ips_seen_so_far'] = cum_unique(['stream', 'dst_ip'], 'stream')
    df['stream_unique_dst_ports_seen_so_far'] = cum_unique(['stream', 'dst_port'], 'stream')

    # Repetition vs diversity ratios
    df['src_dst_port_repeat_ratio'] = src_dst_port / (src + eps)
    df['src_dst_repeat_ratio'] = src_dst / (src + eps)
    df['src_dst_port_login_attempt_score'] = src_dst_port * df['dst_port_is_login_or_web']
    df['src_port_diversity_ratio'] = df['src_unique_dst_ports_seen_so_far'] / (src + eps)
    df['src_dst_diversity_ratio'] = df['src_unique_dst_ips_seen_so_far'] / (src + eps)

    # Short-horizon activity counts
    df['src_count_last_100_packets'] = _rolling_group_count(df, ['src_ip'], 100)
    df['src_dst_count_last_100_packets'] = _rolling_group_count(df, ['src_ip', 'dst_ip'], 100)
    df['src_dst_port_count_last_100_packets'] = _rolling_group_count(df, ['src_ip', 'dst_ip', 'dst_port'], 100)
    df['src_dst_port_count_last_500_packets'] = _rolling_group_count(df, ['src_ip', 'dst_ip', 'dst_port'], 500)
    return df

def propagate_stream_content(df):
    """
    Every packet inherits the worst content signal seen in its conversation.

    Most packets of an attack stream are bare SYN/ACKs with no payload —
    content-identical to benign traffic. Propagating the per-stream max of
    the content features onto every sibling packet makes the whole attack
    conversation separable, not just the one packet carrying the payload.
    """
    keys = ['stream', 'src_ip', 'dst_ip']
    grp = df.groupby(keys)
    for col in ['uri_special_chars', 'uri_length', 'dns_ans_qry_ratio']:
        df[f'stream_max_{col}'] = grp[col].transform('max')
    # Multiple DNS answers in one conversation = the spoofing race signature
    # (forged answer + the real one both arrive). Requires dns_is_answer from
    # engineer_content_features(), so keep the call order in main().
    df['stream_dns_answer_count'] = grp['dns_is_answer'].transform('sum')
    return df


CAPTURE_ARTIFACT_COLS = ['min_et', 'q1', 'min_e', 'var_e', 'q1_e', 'most_freq_spot']

def drop_capture_artifacts(df):
    return df.drop(columns=[c for c in CAPTURE_ARTIFACT_COLS if c in df.columns])

# Heavy-tailed counts/sizes: a DoS packet can have stream_5_count in the
# thousands vs single digits for benign. Without log compression these few
# columns dominate every Euclidean distance (k-means) and reconstruction
# loss (autoencoder), and the content features become invisible.
LOG_COL_PATTERN = re.compile(
    r"(_count$|_var$|_sum$|_len$|_len_|length$|size$|_p$|_rate_5$"
    r"|_burst_persistence$|_fano_5$|jitter$"
    r"|_seen_so_far$|_last_\d+_packets$|_attempt_score$)"
)

def log_transform_skewed(X):
    cols = [c for c in X.columns if LOG_COL_PATTERN.search(c)]
    X = X.copy()
    X[cols] = np.log1p(X[cols].clip(lower=0))
    print(f"[feature-eng] log1p applied to {len(cols)} heavy-tailed columns")
    return X


# Task 1.2: Data Preprocessing
## Handle Missing Values
def binary_flags(df, flag_cols):
    df[flag_cols] = df[flag_cols].replace(['none', 'None', 'NaN'], np.nan).infer_objects(copy=False)
    for col in flag_cols:
        df[f"has_{col}"] = df[col].notna().astype(int)
    return df.drop(columns=flag_cols)

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
    for label, group in df.groupby('label'):
        sub = group[feature_cols + ['id']].copy()
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
    """Global one-hot encoding so every label sees the same category columns.

    min_frequency groups categories seen in <0.1% of packets (dozens of rare
    highest_layer protocols) into one 'infrequent' column. Otherwise ~60
    near-constant dummies flood the feature space and dilute the distance
    metric k-means relies on.
    """
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist',
                        sparse_output=False, min_frequency=0.001)
    arr = ohe.fit_transform(df[mode_impute_cols])
    encoded = pd.DataFrame(arr,
                           columns=ohe.get_feature_names_out(mode_impute_cols),
                           index=df.index)
    return pd.concat([df.drop(columns=mode_impute_cols), encoded], axis=1)

def _remove_outliers_per_label(df, contamination_benign=0.05, contamination_attack=0.01):
    """Isolation Forest per label."""
    final_df = []
    for label, group in df.groupby('label'):
        sub = group.drop(columns='label')
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
    """log1p (applied upstream) + z-scoring.

    StandardScaler instead of RobustScaler: many columns here are zero-
    inflated, so their IQR is 0 and RobustScaler leaves them unscaled at
    raw magnitude. After log1p the tails are tame enough for z-scoring,
    which gives every feature comparable weight in k-means distances and
    autoencoder reconstruction loss.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return pipeline

ALWAYS_KEEP = [
    'payload_entropy', 'payload_ratio', 'ttl',
    'uri_special_chars', 'dns_is_answer', 'dns_ans_qry_ratio',
    'stream_max_dns_ans_qry_ratio', 'stream_dns_answer_count',
]

def _FOR_reduction_per_attack(X, y, threshold=0.95, n_estimators=300, random_state=42,
                              always_keep=None):
    """
    Feature Selection using Forest/Random Forest method.
    Selecting the most important features that explain 95% of the prediction cummulatively.
    Each type of attack is compared with benign to select the specific features that are important to that attack.
    Finally, take the union of all selected features, discard any features that do not contribute to detecting any type of attack.
    """
    labels = y.unique()
    attacks = labels[labels != 'benign']

    kept_cols = []
    contributions = []
    for attack in attacks:
        mask = (y == 'benign') | (y == attack)
        rf = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=random_state,
            class_weight='balanced_subsample'
        )
        rf.fit(X[mask], y[mask])

        # Rank features by importance, pick the top N that cover threshold cumulatively
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        n_keep = (importances.cumsum() < threshold).sum() + 1
        kept_cols.append(importances.head(n_keep).index)
        contributions.append(importances.rename(attack))

    whitelist = set(ALWAYS_KEEP) | set(always_keep or [])
    kept_cols = sorted(set().union(*kept_cols)              # union across attacks, no redundant features
                       | {c for c in whitelist if c in X.columns})
    dropped_cols = [c for c in X.columns if c not in kept_cols]

    X_final = StandardScaler().fit_transform(X[kept_cols])

    pd.concat(contributions, axis=1).to_csv(OUTPUT_DIR / "for_feature_importance.csv")
    pd.Series(dropped_cols, name='dropped').to_csv(OUTPUT_DIR / "for_dropped_features.csv", index=False)

    return X_final, kept_cols

    
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

    # Feature engineering
    pre_engineering_cols = set(sampled_df.columns)
    sampled_df = engineer_content_features(sampled_df)
    sampled_df = engineer_volumetric_features(sampled_df)
    sampled_df = engineer_endpoint_semantics(sampled_df)   # these three need id_cols,
    sampled_df = engineer_repetition_features(sampled_df)  # so they run before
    sampled_df = propagate_stream_content(sampled_df)      # clean_all_labels drops them
    sampled_df = drop_capture_artifacts(sampled_df)
    engineered_cols = sorted(set(sampled_df.columns) - pre_engineering_cols)
    print(f"[feature-eng] {len(engineered_cols)} engineered features (RF-exempt)")

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

    # Compress heavy tails before any scaling / feature selection
    X = log_transform_skewed(X)

    # Normalized Original Dataset (no dimension reduction)
    full_ds = pd.DataFrame(normalization().fit_transform(X), columns=X.columns)
    full_ds['id'] = ids.values
    full_ds['label'] = y.values
    full_ds.to_csv(OUTPUT_DIR/"normalized_original_data.csv", index=False)


    # Random Forest feature selection (95% cumulative importance)
    for_arr, selected_cols = _FOR_reduction_per_attack(X, y, threshold=0.95,
                                                       always_keep=engineered_cols)

    for_ds = pd.DataFrame(for_arr, columns=selected_cols)
    for_ds['id'] = ids.values
    for_ds['label'] = y.values
    for_ds.to_csv(OUTPUT_DIR / "for_data.csv", index=False)


if __name__ == "__main__":
    main()
