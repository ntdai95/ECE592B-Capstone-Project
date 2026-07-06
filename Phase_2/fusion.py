"""Score-level fusion of the GNN track (Anomal-E) and the feature-space track.

Rationale: volumetric attacks (DDoS/DoS) surface in the graph/topology track while
stealthy attacks (DNS spoofing, XSS, brute force) surface in the feature track.
Combining their normalized scores with a max ("OR") rule means a packet flagged by
*either* track becomes an alert -- directly minimizing FNR.
"""
from __future__ import annotations

import numpy as np


def normalize_ref(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rank/quantile-normalize ``scores`` to [0,1] using the reference's empirical CDF.

    This replaces min-max normalization (review fix H2): min-max is set by a single
    extreme value, so one outlier compresses everyone else and a noisy member can
    dominate a max-fusion. Mapping each score to its quantile within the validation
    distribution makes heterogeneous detectors (squared distance, tail probability,
    isolation score) comparable and outlier-robust.
    """
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return np.zeros_like(scores, dtype=float)
    ref_sorted = np.sort(ref)
    # fraction of reference values <= each score  ->  empirical CDF in [0, 1]
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
