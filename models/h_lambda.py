from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .ahs import SuspicionGater
from .hierarchical import EdgeLayer, FogLayer, CloudLayer


class HLAMBDa(nn.Module):
    def __init__( self, in_dim: int, hidden_dim: int, num_classes: int, max_hop: int = 3, gate_threshold: float = 0.5, target_keep: float = 0.15, dropout: float = 0.3) -> None:
        super().__init__()
        self.max_hop = max_hop
        self.gate_threshold = gate_threshold
        self.target_keep = target_keep

        self.edge_layer = EdgeLayer(in_dim, hidden_dim)
        self.gater = SuspicionGater(hidden_dim, hidden_dim=32, max_hop=max_hop)
        self.fog_layer = FogLayer(hidden_dim, max_hop=max_hop, dropout=dropout)
        self.cloud_layer = CloudLayer(hidden_dim, num_classes, dropout=dropout)

        self._last_gate: torch.Tensor | None = None
        self._last_keep_rate: float | None = None

    def forward(self,x: torch.Tensor,edge_indices: List[torch.Tensor],hard_gate: bool = False) -> torch.Tensor:
        if len(edge_indices) != self.max_hop:
            raise ValueError(
                f"H-LAMBDa expects {self.max_hop} edge_index tensors, got {len(edge_indices)}"
            )

        h_edge = self.edge_layer(x, edge_indices[0])

        gate_soft = self.gater(h_edge)
        if hard_gate:
            gate = (gate_soft > self.gate_threshold).float()
        else:
            gate = gate_soft

        h_fog = self.fog_layer(h_edge, edge_indices[1:], gate)

        logits = self.cloud_layer(h_edge, h_fog)

        self._last_gate = gate_soft.detach()
        self._last_keep_rate = float(gate_soft.mean().item())
        return logits

    def gate_sparsity_loss(self) -> torch.Tensor:
        if self._last_gate is None:
            return torch.tensor(0.0)
        return SuspicionGater.sparsity_loss(self._last_gate, self.target_keep)

    def last_keep_rate(self) -> float:
        return self._last_keep_rate or 0.0
