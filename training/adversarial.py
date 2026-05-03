from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class AttackConfig:
    perturb_ratio: float = 0.05    # |delta| / |E|
    steps: int = 3
    add_fraction: float = 0.5      # of the budget, how much goes to additions
    seed: Optional[int] = None


def _random_edges(num_nodes: int, num_edges: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=generator)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=generator)
    # drop self-loops: the GCN normalization will re-add them as needed.
    keep = src != dst
    return torch.stack([src[keep], dst[keep]], dim=0)


def _drop_random_edges(edge_index: torch.Tensor, num_drop: int, generator: torch.Generator) -> torch.Tensor:
    if num_drop <= 0 or edge_index.size(1) == 0:
        return edge_index
    num_drop = min(num_drop, edge_index.size(1))
    perm = torch.randperm(edge_index.size(1), generator=generator, device=edge_index.device)
    keep_idx = perm[num_drop:]
    return edge_index[:, keep_idx]


def perturb_edge_index(edge_index: torch.Tensor,num_nodes: int,cfg: AttackConfig,generator: torch.Generator) -> torch.Tensor:
    budget = max(1, int(cfg.perturb_ratio * edge_index.size(1)))
    num_add = int(budget * cfg.add_fraction)
    num_drop = budget - num_add

    perturbed = _drop_random_edges(edge_index, num_drop, generator)
    new_edges = _random_edges(num_nodes, num_add, edge_index.device, generator)
    return torch.cat([perturbed, new_edges], dim=1)


@torch.no_grad()
def craft_adversarial_edges(model: torch.nn.Module,x: torch.Tensor,edge_indices: List[torch.Tensor],y: torch.Tensor,train_mask: torch.Tensor,cfg: AttackConfig,rebuild_khop) -> List[torch.Tensor]:
    device = x.device
    if cfg.seed is None:
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator(device=device).manual_seed(cfg.seed)

    base_edge_index = edge_indices[0]
    best_edges = base_edge_index
    best_loss = torch.tensor(float("-inf"), device=device)

    was_training = model.training
    model.eval()
    for _ in range(cfg.steps):
        candidate = perturb_edge_index(base_edge_index, x.size(0), cfg, gen)
        cand_khop = rebuild_khop(candidate)
        logits = model(x, cand_khop)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        if loss > best_loss:
            best_loss = loss
            best_edges = candidate
    if was_training:
        model.train()

    return rebuild_khop(best_edges)
