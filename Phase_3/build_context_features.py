"""Connection-window (multi-flow context) features, following the KDD'99
count / srv_count / same_srv_rate family. Per source and destination IP, over
trailing 2s and 60s windows, plus statistics over the previous 100 connections.

Computed over the full capture (what a live detector observes) and joined back to
the modelling rows by Flow ID. Strictly causal: only flows at or before the
current flow's timestamp contribute, so no future information enters a feature.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation import FLOW_DATA_DIR, OUT

WINDOWS = [2, 60]
LAST_N = 100


def _load():
    ids = pd.read_csv(FLOW_DATA_DIR / "flow_data_ids.csv")
    ids.columns = [c.strip() for c in ids.columns]
    ids = ids.drop_duplicates(subset="Flow ID", keep="first").reset_index(drop=True)
    ts = pd.to_datetime(ids["Timestamp"], errors="coerce",
                        format="mixed", dayfirst=True)
    ids["t"] = ts.astype("int64") // 10**9
    ids = ids.dropna(subset=["t"])
    return ids


def _window_block(t, dport, dip, key, wsec):
    """Two-pointer causal sweep within one key group. Returns exact counts and
    distinct counts over the trailing wsec-second window."""
    n = len(t)
    cnt = np.zeros(n, np.float32)
    n_port = np.zeros(n, np.float32)
    n_dip = np.zeros(n, np.float32)
    same_srv = np.zeros(n, np.float32)
    same_dst = np.zeros(n, np.float32)

    port_ct, dip_ct = {}, {}
    lo = 0
    for hi in range(n):
        p, d = dport[hi], dip[hi]
        port_ct[p] = port_ct.get(p, 0) + 1
        dip_ct[d] = dip_ct.get(d, 0) + 1
        while t[hi] - t[lo] > wsec:
            pl, dl = dport[lo], dip[lo]
            port_ct[pl] -= 1
            if port_ct[pl] == 0:
                del port_ct[pl]
            dip_ct[dl] -= 1
            if dip_ct[dl] == 0:
                del dip_ct[dl]
            lo += 1
        c = hi - lo + 1
        cnt[hi] = c
        n_port[hi] = len(port_ct)
        n_dip[hi] = len(dip_ct)
        same_srv[hi] = port_ct.get(p, 0) / c
        same_dst[hi] = dip_ct.get(d, 0) / c
    return cnt, n_port, n_dip, same_srv, same_dst


def _last_n_block(dport, dip, n_back=LAST_N):
    """Host-based: over the previous n_back connections from this source,
    what fraction share this flow's service / destination."""
    n = len(dport)
    same_srv = np.zeros(n, np.float32)
    same_dst = np.zeros(n, np.float32)
    n_port = np.zeros(n, np.float32)
    for i in range(n):
        lo = max(0, i - n_back + 1)
        w_p = dport[lo:i + 1]
        w_d = dip[lo:i + 1]
        m = len(w_p)
        same_srv[i] = (w_p == dport[i]).sum() / m
        same_dst[i] = (w_d == dip[i]).sum() / m
        n_port[i] = len(set(w_p.tolist()))
    return same_srv, same_dst, n_port


def build():
    ids = _load()
    print(f"loaded {len(ids):,} flows, {ids['Src IP'].nunique():,} sources")

    ids = ids.sort_values("t", kind="mergesort").reset_index(drop=True)
    dport = ids["Dst Port"].to_numpy()
    dip = pd.factorize(ids["Dst IP"])[0]
    t = ids["t"].to_numpy()

    feats = {"Flow ID": ids["Flow ID"].to_numpy()}

    for keyname, keycol in [("src", "Src IP"), ("dst", "Dst IP")]:
        codes = pd.factorize(ids[keycol])[0]
        order = np.argsort(codes, kind="mergesort")
        for w in WINDOWS:
            acc = {k: np.zeros(len(ids), np.float32) for k in
                   ["cnt", "nport", "ndip", "samesrv", "samedst"]}
            start = 0
            while start < len(order):
                end = start
                while end < len(order) and codes[order[end]] == codes[order[start]]:
                    end += 1
                idx = order[start:end]
                idx = idx[np.argsort(t[idx], kind="mergesort")]
                c, npo, nd, ss, sd = _window_block(
                    t[idx], dport[idx], dip[idx], codes[idx[0]], w)
                acc["cnt"][idx] = c
                acc["nport"][idx] = npo
                acc["ndip"][idx] = nd
                acc["samesrv"][idx] = ss
                acc["samedst"][idx] = sd
                start = end
            for k, v in acc.items():
                feats[f"ctx_{keyname}_{k}_{w}s"] = v
            print(f"  {keyname} window {w}s done")

    codes = pd.factorize(ids["Src IP"])[0]
    order = np.argsort(codes, kind="mergesort")
    a = {k: np.zeros(len(ids), np.float32) for k in ["samesrv", "samedst", "nport"]}
    start = 0
    while start < len(order):
        end = start
        while end < len(order) and codes[order[end]] == codes[order[start]]:
            end += 1
        idx = order[start:end]
        idx = idx[np.argsort(t[idx], kind="mergesort")]
        ss, sd, npo = _last_n_block(dport[idx], dip[idx])
        a["samesrv"][idx] = ss
        a["samedst"][idx] = sd
        a["nport"][idx] = npo
        start = end
    for k, v in a.items():
        feats[f"ctx_src_{k}_last{LAST_N}"] = v
    print("  last-N host block done")

    out = pd.DataFrame(feats)
    out.to_parquet(f"{OUT}/context_features.parquet", index=False)
    print(f"wrote {out.shape[1]-1} context features for {len(out):,} flows")
    return out


if __name__ == "__main__":
    build()
