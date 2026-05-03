from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EdgeLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv = GCNConv(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index_1hop: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x, edge_index_1hop))


class FogLayer(nn.Module):
    def __init__(self, hidden_dim: int, max_hop: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        if max_hop < 2:
            raise ValueError("Fog layer needs max_hop >= 2")
        self.max_hop = max_hop
        self.dropout = dropout
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(max_hop - 1)])

    def forward(self,h_edge: torch.Tensor,khop_edge_indices: List[torch.Tensor],gate: torch.Tensor) -> torch.Tensor:
        if len(khop_edge_indices) != self.max_hop - 1:
            raise ValueError(
                f"Expected {self.max_hop - 1} k-hop edge sets, got {len(khop_edge_indices)}"
            )

        out = torch.zeros_like(h_edge)
        for i, conv in enumerate(self.convs):
            h_k = F.relu(conv(h_edge, khop_edge_indices[i]))
            h_k = F.dropout(h_k, p=self.dropout, training=self.training)
            out = out + gate[:, i : i + 1] * h_k
        return out


class CloudLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, h_edge: torch.Tensor, h_fog: torch.Tensor) -> torch.Tensor:
        h = torch.cat([h_edge, h_fog], dim=-1)
        h = F.relu(self.fuse(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)
