from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from ..models import MDGCN, HLAMBDa, precompute_khop_indices
from .adversarial import AttackConfig, craft_adversarial_edges


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 5e-3
    weight_decay: float = 5e-4
    hidden_dim: int = 64
    max_hop: int = 3
    dropout: float = 0.3
    adversarial: bool = False
    adv_weight: float = 0.5
    attack: AttackConfig = field(default_factory=AttackConfig)
    gate_loss_weight: float = 0.1
    target_keep: float = 0.15
    log_every: int = 5


def build_model(name: str, in_dim: int, num_classes: int, cfg: TrainConfig) -> nn.Module:
    if name == "md_gcn":
        return MDGCN(in_dim, cfg.hidden_dim, num_classes, max_hop=cfg.max_hop, dropout=cfg.dropout)
    if name == "h_lambda":
        return HLAMBDa(
            in_dim,
            cfg.hidden_dim,
            num_classes,
            max_hop=cfg.max_hop,
            target_keep=cfg.target_keep,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model: {name}")


def _class_weights(y: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y[mask], minlength=num_classes).float().clamp_min(1.0)
    w = counts.sum() / counts
    return (w / w.sum()) * num_classes


def train_model(model: nn.Module,data: Data,edge_indices: List[torch.Tensor],cfg: TrainConfig,device: torch.device,rebuild_khop: Optional[Callable[[torch.Tensor], List[torch.Tensor]]] = None):
    model = model.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]

    num_classes = int(y[y >= 0].max().item()) + 1
    weights = _class_weights(y, train_mask, num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        logits = model(x, edge_indices)
        loss_clean = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)

        loss = loss_clean
        if cfg.adversarial and rebuild_khop is not None:
            adv_edges = craft_adversarial_edges(
                model, x, edge_indices, y, train_mask, cfg.attack, rebuild_khop
            )
            adv_logits = model(x, adv_edges)
            loss_adv = F.cross_entropy(adv_logits[train_mask], y[train_mask], weight=weights)
            loss = (1 - cfg.adv_weight) * loss_clean + cfg.adv_weight * loss_adv

        if isinstance(model, HLAMBDa):
            loss = loss + cfg.gate_loss_weight * model.gate_sparsity_loss()

        loss.backward()
        opt.step()

        if epoch % cfg.log_every == 0 or epoch == 1:
            with torch.no_grad():
                model.eval()
                val_logits = model(x, edge_indices)
                val_pred = val_logits[val_mask].argmax(dim=-1)
                val_acc = (val_pred == y[val_mask]).float().mean().item()
            keep = float(model.last_keep_rate()) if isinstance(model, HLAMBDa) else 1.0
            history.append({"epoch": epoch, "loss": float(loss.item()), "val_acc": val_acc, "keep_rate": keep})
            print(
                f"epoch {epoch:3d} | loss {loss.item():.4f} | val_acc {val_acc:.4f}"
                + (f" | keep {keep:.2f}" if isinstance(model, HLAMBDa) else "")
            )

    return model, history
