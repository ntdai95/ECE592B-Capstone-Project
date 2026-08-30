import numpy as np


def normalize_ref(scores, ref):
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return np.zeros_like(scores, dtype=float)
    ref_sorted = np.sort(ref)
    q = np.searchsorted(ref_sorted, scores, side="right") / ref_sorted.size
    return np.clip(q, 0.0, 1.0)


def fuse(score_lists, ref_lists, combine="max"):
    norm = [normalize_ref(s, r) for s, r in zip(score_lists, ref_lists)]
    stacked = np.vstack(norm)
    if combine == "max":
        return stacked.max(axis=0)
    if combine == "mean":
        return stacked.mean(axis=0)
    raise ValueError(f"Unknown combine {combine!r}")
