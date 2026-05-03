"""
Adaptive Hop-Selection (AHS) — the "Suspicion Gater".

For every node we run a tiny MLP on its 1-hop embedding to score how
suspicious it looks. Nodes below a configurable threshold skip the deeper
k-hop branches, which is where MD-GCN spends most of its compute.

The output is a soft mask in [0, 1] per (node, hop). At inference we hard-
threshold; during training we keep it soft so gradients flow into the gater.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuspicionGater(nn.Module):
    """MLP that maps a node embedding to a per-hop "expand or prune" score.

    Output shape: (N, max_hop - 1). The first hop is always taken; only the
    deeper hops (k >= 2) are gated, since the gater itself runs on the 1-hop
    embedding.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, max_hop: int = 3) -> None:
        super().__init__()
        if max_hop < 2:
            raise ValueError("AHS only makes sense when max_hop >= 2")
        self.max_hop = max_hop
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_hop - 1),
        )

    def forward(self, h1: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h1))

    def hard_mask(self, h1: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            return (self.forward(h1) > threshold).float()

    @staticmethod
    def sparsity_loss(gate: torch.Tensor, target_keep: float = 0.1) -> torch.Tensor:
        """Encourage the gater to expand on roughly `target_keep` fraction of nodes.

        Without this regularizer the model would learn to gate everything open
        (free signal) and lose the latency benefit. We penalize the squared
        deviation of the actual keep-rate from the target.
        """
        keep_rate = gate.mean()
        return (keep_rate - target_keep).pow(2)
