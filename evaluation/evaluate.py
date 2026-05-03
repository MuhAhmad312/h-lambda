from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..models import HLAMBDa


@dataclass
class EvalResult:
    model: str
    accuracy: float
    f1_macro: float
    f1_illicit: float
    precision_illicit: float
    recall_illicit: float
    latency_ms_mean: float
    latency_ms_p95: float
    peak_cuda_mem_mb: float
    peak_cpu_mem_mb: float
    keep_rate: float
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _peak_cpu_mem_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    x: torch.Tensor,
    edge_indices: List[torch.Tensor],
    runs: int = 20,
    warmup: int = 3,
) -> Dict[str, float]:
    device = x.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    cpu_baseline = _peak_cpu_mem_mb()

    timings = []
    for i in range(runs + warmup):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x, edge_indices)
            end.record()
            torch.cuda.synchronize(device)
            t = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            _ = model(x, edge_indices)
            t = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            timings.append(t)

    timings_t = torch.tensor(timings)
    cuda_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    )
    return {
        "latency_ms_mean": float(timings_t.mean().item()),
        "latency_ms_p95": float(timings_t.quantile(0.95).item()),
        "peak_cuda_mem_mb": float(cuda_mb),
        "peak_cpu_mem_mb": max(0.0, _peak_cpu_mem_mb() - cpu_baseline),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    name: str,
    data,
    edge_indices: List[torch.Tensor],
    device: torch.device,
    runs: int = 20,
) -> EvalResult:
    model = model.to(device).eval()
    x = data.x.to(device)
    y = data.y.to(device)
    test_mask = data.test_mask.to(device)
    edge_indices = [ei.to(device) for ei in edge_indices]

    logits = model(x, edge_indices)
    pred = logits.argmax(dim=-1).cpu().numpy()
    truth = y.cpu().numpy()
    mask = test_mask.cpu().numpy()

    bench = benchmark_inference(model, x, edge_indices, runs=runs)

    keep = float(model.last_keep_rate()) if isinstance(model, HLAMBDa) else 1.0

    return EvalResult(
        model=name,
        accuracy=float(accuracy_score(truth[mask], pred[mask])),
        f1_macro=float(f1_score(truth[mask], pred[mask], average="macro", zero_division=0)),
        f1_illicit=float(f1_score(truth[mask], pred[mask], pos_label=1, zero_division=0)),
        precision_illicit=float(precision_score(truth[mask], pred[mask], pos_label=1, zero_division=0)),
        recall_illicit=float(recall_score(truth[mask], pred[mask], pos_label=1, zero_division=0)),
        latency_ms_mean=bench["latency_ms_mean"],
        latency_ms_p95=bench["latency_ms_p95"],
        peak_cuda_mem_mb=bench["peak_cuda_mem_mb"],
        peak_cpu_mem_mb=bench["peak_cpu_mem_mb"],
        keep_rate=keep,
    )


def save_results(results: List[EvalResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"saved {len(results)} results to {path}")
