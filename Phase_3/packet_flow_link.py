"""Link packet-level rows to CICFlowMeter flow records via Flow ID."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flow_sampling import ATTACK_FILE_NAMES, BENIGN_FILE_NAME, _read_csv, aggregate_flow_segments


PROTOCOL_TCP = 6
PROTOCOL_UDP = 17


def packet_protocol(row_or_tcp, udp=None):
    """Map packet L4 flags to CICFlowMeter protocol numbers."""
    if udp is None:
        tcp = int(row_or_tcp.get("l4_tcp", 0) or 0)
        udp = int(row_or_tcp.get("l4_udp", 0) or 0)
    else:
        tcp = int(row_or_tcp or 0)
        udp = int(udp or 0)

    if tcp == 1:
        return PROTOCOL_TCP
    if udp == 1:
        return PROTOCOL_UDP
    return 0


def build_flow_id(src_ip, dst_ip, src_port, dst_port, protocol):
    return (
        f"{src_ip}-{dst_ip}-"
        f"{int(src_port)}-{int(dst_port)}-"
        f"{int(protocol)}"
    )


def add_packet_flow_ids(packet_df: pd.DataFrame) -> pd.DataFrame:
    """Add forward and reverse Flow ID candidates to a packet dataframe."""
    df = packet_df.copy()
    tcp = pd.to_numeric(df.get("l4_tcp", 0), errors="coerce").fillna(0).astype(int)
    udp = pd.to_numeric(df.get("l4_udp", 0), errors="coerce").fillna(0).astype(int)
    protocol = np.where(tcp == 1, PROTOCOL_TCP, np.where(udp == 1, PROTOCOL_UDP, 0))

    src_ip = df["src_ip"].astype(str)
    dst_ip = df["dst_ip"].astype(str)
    src_port = pd.to_numeric(df["src_port"], errors="coerce").fillna(-1).astype(int)
    dst_port = pd.to_numeric(df["dst_port"], errors="coerce").fillna(-1).astype(int)

    df["protocol"] = protocol
    df["flow_id_fwd"] = (
        src_ip + "-" + dst_ip + "-" + src_port.astype(str) + "-" + dst_port.astype(str) + "-" + pd.Series(protocol, index=df.index).astype(str)
    )
    df["flow_id_rev"] = (
        dst_ip + "-" + src_ip + "-" + dst_port.astype(str) + "-" + src_port.astype(str) + "-" + pd.Series(protocol, index=df.index).astype(str)
    )
    return df


def list_raw_flow_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    names = [BENIGN_FILE_NAME, *ATTACK_FILE_NAMES.values()]
    paths = [data_dir / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing flow files:\n" + "\n".join(missing))
    return paths


def load_flows_for_ids(
    data_dir: str | Path,
    flow_ids: set[str] | list[str],
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load and aggregate raw flow segments whose Flow ID is in ``flow_ids``."""
    wanted = set(map(str, flow_ids))
    if not wanted:
        return pd.DataFrame()

    frames = []
    for path in list_raw_flow_files(data_dir):
        matched_chunks = []
        for chunk in _read_csv(path, chunksize=chunksize):
            chunk["Flow ID"] = chunk["Flow ID"].astype(str)
            hit = chunk[chunk["Flow ID"].isin(wanted)]
            if not hit.empty:
                matched_chunks.append(hit)

        if not matched_chunks:
            continue

        segments = pd.concat(matched_chunks, ignore_index=True)
        flows = aggregate_flow_segments(segments)
        flows["source_file"] = path.name
        flows["flow_key"] = flows["source_file"] + "::" + flows["Flow ID"].astype(str)
        frames.append(flows)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Flow ID"], keep="first")
    return combined.reset_index(drop=True)


def resolve_packet_flow_ids(packet_df: pd.DataFrame, available_flow_ids: set[str]) -> pd.Series:
    """Pick the matching Flow ID for each packet (forward preferred, else reverse)."""
    fwd = packet_df["flow_id_fwd"].astype(str)
    rev = packet_df["flow_id_rev"].astype(str)
    resolved = fwd.where(fwd.isin(available_flow_ids), other=pd.NA)
    resolved = resolved.fillna(rev.where(rev.isin(available_flow_ids), other=pd.NA))
    return resolved
