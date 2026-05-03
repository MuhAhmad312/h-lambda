
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


def khop_edge_index(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    if k == 1:
        return edge_index
    indices = edge_index
    values = torch.ones(indices.size(1), device=indices.device)
    a = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    ak = a
    for _ in range(k - 1):
        ak = torch.sparse.mm(ak, a).coalesce()
    return ak.indices()


class MDGCN(nn.Module):
    def __init__(self,in_dim: int,hidden_dim: int,num_classes: int,max_hop: int = 3,dropout: float = 0.3) -> None:
        super().__init__()
        self.max_hop = max_hop
        self.dropout = dropout
        self.conv1 = nn.ModuleList([GCNConv(in_dim, hidden_dim) for _ in range(max_hop)])
        self.conv2 = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(max_hop)])
        self.fuse = nn.Linear(hidden_dim * max_hop, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_indices: List[torch.Tensor]) -> torch.Tensor:
        if len(edge_indices) != self.max_hop:
            raise ValueError(
                f"MDGCN expects {self.max_hop} edge_index tensors, got {len(edge_indices)}"
            )

        branch_embeds = []
        for k in range(self.max_hop):
            ei = edge_indices[k]
            h = F.relu(self.conv1[k](x, ei))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2[k](h, ei)
            branch_embeds.append(h)

        h = torch.cat(branch_embeds, dim=-1)
        h = F.relu(self.fuse(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


def precompute_khop_indices(data, max_hop: int) -> List[torch.Tensor]:
    ei = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return [khop_edge_index(ei, data.num_nodes, k + 1) for k in range(max_hop)]
