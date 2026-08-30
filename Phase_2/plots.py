import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from . import config as C


def _save(fig, name):
    C.ensure_dirs()
    path = C.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc(scores, y_bin, title, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_bin, scores, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_title(f"ROC — {title}")
    return _save(fig, name)


def plot_pr(scores, y_bin, title, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_bin, scores, ax=ax, name=title)
    ax.set_title(f"Precision–Recall — {title}")
    return _save(fig, name)


def plot_confusion(cm, title, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "attack"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion — {title}")
    return _save(fig, name)


def plot_per_attack_dr(per_attack_dr, title, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    atks = list(per_attack_dr.keys())
    vals = [per_attack_dr[a] for a in atks]
    bars = ax.bar(atks, vals, color="#c0392b")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Detection rate")
    ax.set_title(f"Per-attack detection rate — {title}")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    return _save(fig, name)


def plot_embedding(emb, y_multi, title, name, method="umap", max_points=8000):
    rng = np.random.default_rng(0)
    benign_idx = np.where(y_multi == C.BENIGN_LABEL)[0]
    attack_idx = np.where(y_multi != C.BENIGN_LABEL)[0]
    keep_benign = rng.choice(benign_idx, size=min(max_points, len(benign_idx)), replace=False)
    idx = np.concatenate([keep_benign, attack_idx])
    rng.shuffle(idx)
    E, lab = emb[idx], y_multi[idx]

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=0)
            E2 = reducer.fit_transform(E)
        except Exception:
            method = "pca"
    if method != "umap":
        from sklearn.decomposition import PCA

        E2 = PCA(n_components=2, random_state=0).fit_transform(E)

    fig, ax = plt.subplots(figsize=(7, 6))
    classes = [C.BENIGN_LABEL] + C.ATTACK_LABELS
    palette = ["#bdc3c7", "#e74c3c", "#e67e22", "#8e44ad", "#2980b9", "#16a085"]
    for cls, col in zip(classes, palette):
        m = lab == cls
        if m.any():
            ax.scatter(E2[m, 0], E2[m, 1], s=6, c=col, label=cls, alpha=0.6,
                       edgecolors="none")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title(f"{method.upper()} of embeddings — {title}")
    ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, name)
