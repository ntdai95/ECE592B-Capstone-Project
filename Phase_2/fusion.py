"""Score-level fusion of the GNN track (Anomal-E) and the feature-space track.

Rationale: volumetric attacks (DDoS/DoS) surface in the graph/topology track while
stealthy attacks (DNS spoofing, XSS, brute force) surface in the feature track.
Combining their normalized scores with a max ("OR") rule means a packet flagged by
*either* track becomes an alert -- directly minimizing FNR.
"""
from __future__ import annotations

import numpy as np


def normalize_ref(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map ``scores`` to [0,1] via the reference's empirical CDF (quantile
    normalisation). Unlike min-max, one outlier cannot compress the range, so
    heterogeneous detector scores (squared distance, tail probability, isolation
    score) become comparable and outlier-robust before max-fusion.
    """
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return np.zeros_like(scores, dtype=float)
    ref_sorted = np.sort(ref)
    q = np.searchsorted(ref_sorted, scores, side="right") / ref_sorted.size
    return np.clip(q, 0.0, 1.0)


def fuse(score_lists: list[np.ndarray], ref_lists: list[np.ndarray], combine: str = "max") -> np.ndarray:
    """Combine detectors' scores after per-detector quantile normalization.

    ``ref_lists`` are the reference scores (validation) used to fix each detector's
    normalization, so test scores are mapped through the same CDF.
    """
    norm = [normalize_ref(s, r) for s, r in zip(score_lists, ref_lists)]
    stacked = np.vstack(norm)
    if combine == "max":
        return stacked.max(axis=0)
    if combine == "mean":
        return stacked.mean(axis=0)
    raise ValueError(f"Unknown combine {combine!r}")
