"""Anomal-E: self-supervised, edge-feature GNN for NIDS, transductive.

Pipeline (Caville et al., 2022, arXiv:2207.06819):
  1. One host graph over all packets; each packet is an edge carrying its scaled
     feature vector, hosts are nodes.
  2. E-GraphSAGE (Lo et al., 2021) node embeddings, then an edge embedding
     concat(h_src, h_dst) per packet.
  3. Train the encoder self-supervised with Deep Graph Infomax: a discriminator
     separates real edge embeddings from ones built on a feature-permuted graph.
     No labels used.
  4. Fit Isolation Forest on the TRAIN edge embeddings only, score val/test.

Transductive by design: node embeddings use edge structure/features (never
labels), and stealthy-attack fan-in/out is only visible on the full graph -- a
random per-split graph fragments each host's degree and the signal disappears.
Label discipline is kept where labels could leak: the downstream detector fits on
train edges, the threshold is chosen on validation. max_mp_edges caps the
message-passing graph for memory; edge embeddings are still formed for all packets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import build_graph, load_ids_table


def _make_modules():
    """Lazily build the torch/PyG modules (so the file imports without torch)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing

    class EGraphSAGELayer(MessagePassing):
        """One E-GraphSAGE layer: messages fuse neighbor node + edge features."""

        def __init__(self, node_in: int, edge_dim: int, out: int):
            super().__init__(aggr="mean")
            self.lin_msg = nn.Linear(node_in + edge_dim, out)
            self.lin_upd = nn.Linear(node_in + out, out)

        def forward(self, x, edge_index, edge_attr):
            agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            h = F.relu(self.lin_upd(torch.cat([x, agg], dim=1)))
            return F.normalize(h, p=2, dim=1)

        def message(self, x_j, edge_attr):
            return F.relu(self.lin_msg(torch.cat([x_j, edge_attr], dim=1)))

    class EGraphSAGE(nn.Module):
        def __init__(self, node_in: int, edge_dim: int, hidden: int, n_layers: int = 2):
            super().__init__()
            self.layers = nn.ModuleList()
            dims = [node_in] + [hidden] * n_layers
            for i in range(n_layers):
                self.layers.append(EGraphSAGELayer(dims[i], edge_dim, dims[i + 1]))
            self.out_dim = hidden * 2

        def node_embeddings(self, x, edge_index, edge_attr):
            h = x
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return h

    class Discriminator(nn.Module):
        """Bilinear DGI discriminator scoring an embedding against a graph summary."""

        def __init__(self, dim: int):
            super().__init__()
            self.bilinear = nn.Bilinear(dim, dim, 1)

        def forward(self, z, summary):
            return self.bilinear(z, summary.expand_as(z)).squeeze(-1)

    return torch, nn, F, EGraphSAGE, Discriminator


class AnomalE:
    name = "anomal_e"

    def __init__(
        self,
        node_type: str = "ip",
        hidden: int = 64,
        n_layers: int = 2,
        node_in_dim: int = 16,
        epochs: int = 60,
        lr: float = 1e-3,
        max_mp_edges: int = 90_000,    # cap edges for node-embedding message passing (CPU RAM)
        loss_edges: int = 20_000,      # #edges sampled per epoch for the DGI loss
        contamination: float = 0.03,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.node_type = node_type
        self.hidden = hidden
        self.n_layers = n_layers
        self.node_in_dim = node_in_dim
        self.epochs = epochs
        self.lr = lr
        self.max_mp_edges = max_mp_edges
        self.loss_edges = loss_edges
        self.contamination = contamination
        self.seed = seed
        self.verbose = verbose
        self._ids_df: pd.DataFrame | None = None
        self.encoder = None
        self.downstream = None
        self.last_test_emb: np.ndarray | None = None  # cached for UMAP figure

    def _ids_table(self) -> pd.DataFrame:
        if self._ids_df is None:
            self._ids_df = load_ids_table()
        return self._ids_df

    def fit_score_transductive(self, train, val, test):
        """Train DGI on the full graph; return (val_scores, test_scores).

        ``train``/``val``/``test`` are scaled Datasets that together partition the
        full dataset. Node embeddings come from the full graph; the downstream
        Isolation Forest is fit on TRAIN edges only.
        """
        torch, nn, F, EGraphSAGE, Discriminator = _make_modules()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # --- assemble the full edge set with split membership -----------------
        X_all = np.vstack([train.X, val.X, test.X]).astype(np.float32)
        ids_all = np.concatenate([train.ids, val.ids, test.ids])
        n_tr, n_va = len(train.X), len(val.X)
        tr_sl = slice(0, n_tr)
        va_sl = slice(n_tr, n_tr + n_va)
        te_sl = slice(n_tr + n_va, len(X_all))

        g = build_graph(X_all, ids_all, self._ids_table(), self.node_type, self.node_in_dim)
        edge_index_all = torch.tensor(g.edge_index, dtype=torch.long)   # (2, E_all)
        edge_attr_all = torch.tensor(g.edge_attr, dtype=torch.float32)
        x = torch.ones((g.n_nodes, self.node_in_dim), dtype=torch.float32)

        # Message-passing graph (bidirectional) -- optionally subsampled for memory.
        E_all = edge_index_all.shape[1]
        if E_all > self.max_mp_edges:
            sel = torch.tensor(rng.choice(E_all, size=self.max_mp_edges, replace=False))
            mp_ei0, mp_ea0 = edge_index_all[:, sel], edge_attr_all[sel]
        else:
            mp_ei0, mp_ea0 = edge_index_all, edge_attr_all
        rev = torch.stack([mp_ei0[1], mp_ei0[0]])
        mp_ei = torch.cat([mp_ei0, rev], dim=1)
        mp_ea = torch.cat([mp_ea0, mp_ea0], dim=0)

        self.encoder = EGraphSAGE(self.node_in_dim, edge_attr_all.shape[1], self.hidden, self.n_layers)
        disc = Discriminator(self.encoder.out_dim)
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(disc.parameters()), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()

        def edge_emb(h, ei):
            return torch.cat([h[ei[0]], h[ei[1]]], dim=1)

        # Sample which edges to compute the DGI loss on (from ALL edges).
        self.encoder.train()
        for ep in range(self.epochs):
            opt.zero_grad()
            h_pos = self.encoder.node_embeddings(x, mp_ei, mp_ea)            # real
            perm = torch.randperm(mp_ea.shape[0])
            h_neg = self.encoder.node_embeddings(x, mp_ei, mp_ea[perm])      # corrupted

            n_loss = min(self.loss_edges, E_all)
            le = torch.tensor(rng.choice(E_all, size=n_loss, replace=False))
            pos = edge_emb(h_pos, edge_index_all[:, le])
            neg = edge_emb(h_neg, edge_index_all[:, le])
            summary = torch.sigmoid(pos.mean(dim=0, keepdim=True))

            loss = bce(disc(pos, summary), torch.ones(n_loss)) + \
                bce(disc(neg, summary), torch.zeros(n_loss))
            loss.backward()
            opt.step()
            if self.verbose and (ep % 10 == 0 or ep == self.epochs - 1):
                print(f"[Anomal-E] epoch {ep + 1}/{self.epochs} dgi_loss={loss.item():.4f}")

        # --- final embeddings for ALL edges, then split-aware scoring ---------
        from pyod.models.iforest import IForest

        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder.node_embeddings(x, mp_ei, mp_ea)
            emb_all = edge_emb(h, edge_index_all).cpu().numpy()

        self.downstream = IForest(n_estimators=200, contamination=self.contamination,
                                  random_state=self.seed)
        self.downstream.fit(emb_all[tr_sl])               # fit on TRAIN edges only
        val_scores = self.downstream.decision_function(emb_all[va_sl])
        # test may be empty during hyperparameter tuning (test excluded -> no leakage)
        if (te_sl.stop - te_sl.start) > 0:
            test_scores = self.downstream.decision_function(emb_all[te_sl])
        else:
            test_scores = np.empty(0, dtype=float)
        self.last_test_emb = emb_all[te_sl]
        return val_scores, test_scores
