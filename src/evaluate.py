#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities and plotting for EA-FA experiments.
- Perplexity/energy measurement for inference.
- Publication-quality PDF plots saved to .research/iteration1/images.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.cuda.amp import autocast
except Exception:
    import contextlib
    autocast = contextlib.nullcontext  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import PowerSampler


def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    steps: int,
    power: PowerSampler,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    model.eval()
    tokens_total = 0
    nll_total = 0.0
    n_batches = 0
    power.start()
    with torch.no_grad():
        it = iter(dataloader)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with autocast(enabled=True, dtype=amp_dtype):
                logits = model(x)
                log_probs = F.log_softmax(logits, dim=-1)
                nll = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), y.view(-1), reduction='mean')
            tokens_total += int(x.numel())
            nll_total += float(nll.item())
            n_batches += 1
    power.stop()
    ppl = math.exp(nll_total / max(1, n_batches))
    energy_j = power.energy_joules(subtract_idle=True)
    # naive tokens/sec and perf/W
    time_s = 0.0  # Not tracked here precisely; caller can time wall clock if needed
    return {
        "tokens": tokens_total,
        "time_s": time_s,
        "joules": energy_j,
        "ppl": ppl,
        "n_batches": n_batches,
    }


def bootstrap_ci(x: np.ndarray, alpha: float = 0.05, n_resamples: int = 5000) -> Tuple[float, float]:
    rng = np.random.default_rng(1234)
    x = np.asarray(x)
    if len(x) == 0:
        return (float('nan'), float('nan'))
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def plot_energy_bar(methods: List[str], j_per_1k_list: List[List[float]], save_path) -> None:
    means = [float(np.mean(v)) if len(v)>0 else float('nan') for v in j_per_1k_list]
    cis = [bootstrap_ci(np.array(v)) if len(v)>0 else (float('nan'), float('nan')) for v in j_per_1k_list]
    errs_lo = [m - c[0] for m, c in zip(means, cis)]
    errs_hi = [c[1] - m for m, c in zip(means, cis)]
    x = np.arange(len(methods))
    plt.rcParams.update({"font.size": 11, "figure.figsize": (5.0, 3.2), "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=[errs_lo, errs_hi], capsize=4, color=['#4C78A8' if 'EA-FA' in m else '#9ecae1' for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Joules per 1k tokens')
    ax.set_title('Energy-to-solution (lower is better)')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
