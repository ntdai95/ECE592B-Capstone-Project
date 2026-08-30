from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class GraphArrays:
    edge_index: np.ndarray
    edge_attr: np.ndarray
    n_nodes: int
    node_in_dim: int


def load_ids_table():
    cols = ["id", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_port", "dst_port"]
    return pd.read_csv(C.IDS_CSV, usecols=cols)


def _host_col(sub, name, prefix):
    vals = sub[name].to_numpy(dtype=object)
    miss = pd.isna(vals)
    if miss.any():
        rows = np.nonzero(miss)[0]
        for r in rows:
            vals[r] = f"UNK_{prefix}_{r}"
    return vals.astype(str)


def _port_col(sub, name):
    s = pd.to_numeric(sub[name], errors="coerce").astype("Int64")
    return s.astype(str).str.replace("<NA>", "NA", regex=False).to_numpy(dtype=object)


def _endpoints(sub, node_type):
    if node_type == "ip":
        return _host_col(sub, "src_ip", "s"), _host_col(sub, "dst_ip", "d")
    if node_type == "mac":
        return _host_col(sub, "src_mac", "s"), _host_col(sub, "dst_mac", "d")
    if node_type == "ipport":
        s = np.char.add(np.char.add(_host_col(sub, "src_ip", "s").astype(str), ":"),
                        _port_col(sub, "src_port").astype(str))
        d = np.char.add(np.char.add(_host_col(sub, "dst_ip", "d").astype(str), ":"),
                        _port_col(sub, "dst_port").astype(str))
        return s.astype(object), d.astype(object)
    raise ValueError(f"Unknown node_type {node_type!r}")


def build_graph(X, ids, ids_df, node_type="ip", node_in_dim=16):
    sub = ids_df.set_index("id").reindex(ids)
    src, dst = _endpoints(sub, node_type)

    nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    n = len(src)
    src_idx = inv[:n].astype(np.int64)
    dst_idx = inv[n:].astype(np.int64)

    edge_index = np.vstack([src_idx, dst_idx])
    edge_attr = X.astype(np.float32, copy=False)
    return GraphArrays(
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_nodes=len(nodes),
        node_in_dim=node_in_dim,
    )
