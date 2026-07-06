"""Communication-graph construction for the GNN track.

Turns a packet feature table into a graph where:
  * nodes = hosts (IPs by default; MAC or IP:port as ablations)
  * edges = packets (each packet is one directed edge src -> dst)
  * edge features = the scaled packet feature vector
  * node features = all-ones (the E-GraphSAGE trick: all signal lives on edges)

Topology comes from ``packet_data_ids.csv`` joined on the packet ``id``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C


@dataclass
class GraphArrays:
    edge_index: np.ndarray   # (2, E) int64 -- [src_nodes; dst_nodes]
    edge_attr: np.ndarray    # (E, F)  float32 -- packet features (scoring order)
    n_nodes: int
    node_in_dim: int         # dimension of the all-ones node feature


def load_ids_table() -> pd.DataFrame:
    cols = ["id", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_port", "dst_port"]
    return pd.read_csv(C.IDS_CSV, usecols=cols)


def _host_col(sub: pd.DataFrame, name: str, prefix: str) -> np.ndarray:
    """String node ids for a host column.

    Missing values (e.g. ARP/L2 packets have no IP) get a UNIQUE placeholder per
    row -- never a shared 'UNK' node -- so they don't collapse into one giant
    artificial hub that would distort the graph (review fix M1).
    """
    vals = sub[name].to_numpy(dtype=object)
    miss = pd.isna(vals)
    if miss.any():
        rows = np.nonzero(miss)[0]
        for r in rows:
            vals[r] = f"UNK_{prefix}_{r}"
    return vals.astype(str)


def _port_col(sub: pd.DataFrame, name: str) -> np.ndarray:
    """Port as a clean integer string (avoids '443.0'); missing -> 'NA' (L2)."""
    s = pd.to_numeric(sub[name], errors="coerce").astype("Int64")
    return s.astype(str).str.replace("<NA>", "NA", regex=False).to_numpy(dtype=object)


def _endpoints(sub: pd.DataFrame, node_type: str) -> tuple[np.ndarray, np.ndarray]:
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


def build_graph(
    X: np.ndarray,
    ids: np.ndarray,
    ids_df: pd.DataFrame,
    node_type: str = "ip",
    node_in_dim: int = 16,
) -> GraphArrays:
    """Build graph arrays for the given packets.

    The Anomal-E pipeline is transductive: this is called ONCE over all packets
    (train+val+test together, see ``AnomalE.fit_score_transductive``) so a host's
    true fan-in/out topology is preserved rather than fragmented across splits. A
    node index space is built over whatever ``ids`` are passed; every host uses the
    same all-ones node feature (all signal lives on the edges).
    """
    sub = ids_df.set_index("id").reindex(ids)  # align topology to this split's rows
    src, dst = _endpoints(sub, node_type)

    # Build a single shared node vocabulary across src + dst.
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
