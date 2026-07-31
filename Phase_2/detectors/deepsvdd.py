"""Deep SVDD: one-class deep anomaly detection (Ruff et al., 2018).

Learns an encoder that pulls normal traffic into a tight hypersphere around a
fixed center c; the anomaly score is the squared distance to c. Bias-free linear
layers and unbounded activations avoid the trivial hypersphere-collapse solution,
and c is fixed from an initial forward pass rather than learned.

The encoder is optionally warm-started as an autoencoder before c is fixed.
Without it, c comes from a randomly initialised encoder and a bad seed can
collapse the run (one seed dropped to AUC ~0.45 against ~0.75). Set
pretrain_epochs=0 to recover the random-init behaviour.
"""
from __future__ import annotations

import numpy as np

from . import BaseDetector


class DeepSVDD(BaseDetector):
    name = "deep_svdd"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        epochs: int = 40,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        pretrain_epochs: int = 50,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.seed = seed
        self.verbose = verbose
        self.net = None
        self.c = None  # hypersphere center

    def _build(self, in_dim: int):
        """Encoder phi: bias-free Linear + LeakyReLU stack (unbounded activations)."""
        import torch.nn as nn

        layers = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h, bias=False), nn.LeakyReLU(0.1)]
            prev = h
        layers += [nn.Linear(prev, self.hidden_dims[-1], bias=False)]
        return nn.Sequential(*layers)

    def _build_decoder(self, in_dim: int):
        """Mirror decoder used only for AE pretraining (discarded afterwards).

        Maps the embedding (dim = hidden_dims[-1]) back up through the reversed
        hidden widths to the input dimension.
        """
        import torch.nn as nn

        rev = list(reversed(self.hidden_dims))
        layers = []
        prev = self.hidden_dims[-1]
        for h in rev[1:]:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        layers += [nn.Linear(prev, in_dim)]
        return nn.Sequential(*layers)

    def _pretrain(self, ae, loader, Xt):
        """Train encoder+decoder on reconstruction (MSE) to warm-start the encoder."""
        import torch

        mse = torch.nn.MSELoss()
        opt = torch.optim.Adam(ae.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ae.train()
        for ep in range(self.pretrain_epochs):
            tot = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                loss = mse(ae(xb), xb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if self.verbose and (ep % 10 == 0 or ep == self.pretrain_epochs - 1):
                print(f"[DeepSVDD-AE] epoch {ep + 1}/{self.pretrain_epochs} "
                      f"recon={tot / len(Xt):.5f}")

    def fit(self, X: np.ndarray) -> "DeepSVDD":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xt = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt), batch_size=self.batch_size, shuffle=True)
        self.net = self._build(X.shape[1])

        # AE pretraining warm-starts the encoder; the decoder is discarded.
        if self.pretrain_epochs > 0:
            decoder = self._build_decoder(X.shape[1])
            ae = torch.nn.Sequential(self.net, decoder)
            self._pretrain(ae, loader, Xt)

        # Fix the center c from a forward pass of the pretrained encoder.
        self.net.eval()
        with torch.no_grad():
            reps = self.net(Xt)
            c = reps.mean(dim=0)
            c[(abs(c) < 1e-6)] = 1e-6  # keep near-zero dims off the origin
        self.c = c

        # Deep SVDD objective: pull normal points toward c.
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net.train()
        for ep in range(self.epochs):
            tot = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                z = self.net(xb)
                loss = ((z - self.c) ** 2).sum(dim=1).mean()
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if self.verbose:
                print(f"[DeepSVDD] epoch {ep + 1}/{self.epochs} loss={tot / len(Xt):.5f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(X, dtype=torch.float32))
            dist = ((z - self.c) ** 2).sum(dim=1)
        return dist.cpu().numpy()
