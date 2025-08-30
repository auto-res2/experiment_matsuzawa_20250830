# -*- coding: utf-8 -*-
"""
Preprocessing utilities: Time-Warp key-step discovery and ΔF computation.
- Provides generic ΔF heatmap computation and greedy key-step selection.
- Includes a toy pipeline to generate features for quick tests.
- Saves all figures as PDF under .research/iteration10/images.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_deltaF_heatmap(get_features_fn, n_steps: int) -> Tuple[np.ndarray, List[str]]:
    """Runs get_features_fn(step_index) for t=0..n_steps-1 and returns
    - deltaF: (n_layers, n_steps-1) matrix of L2 norms of per-step changes
    - layer_ids: order of layers
    """
    feats_prev: Dict[str, torch.Tensor] = {}
    layer_ids: List[str] = []
    deltas: Dict[str, List[float]] = {}
    for t in range(n_steps):
        feats = get_features_fn(t)
        if not layer_ids:
            layer_ids = sorted(list(feats.keys()))
            for k in layer_ids:
                deltas[k] = []
        if t > 0:
            for k in layer_ids:
                a = feats[k]
                b = feats_prev[k]
                delta = (a - b).float().pow(2).sum().sqrt().item()
                deltas[k].append(delta)
        feats_prev = {k: v.detach().cpu() for k, v in feats.items()}
    mat = np.stack([deltas[k] for k in layer_ids], axis=0)
    return mat, layer_ids


def greedy_timewarp_key_selection(deltaF: np.ndarray, K: int, stratify_bins: int = 4) -> List[int]:
    """Greedy selection of K key steps based on summed ΔF across layers with stratification.
    deltaF: (L, T-1) for steps 1..T-1. Return indices in 1..T-1 inclusive.
    """
    L, Tm1 = deltaF.shape
    scores = deltaF.sum(axis=0)
    bins = np.digitize(np.arange(Tm1), np.linspace(0, Tm1, stratify_bins+1)[1:-1])
    selected: List[int] = []
    used_bins = set()
    # First pass: encourage spread
    for _ in range(min(K, stratify_bins)):
        mask = np.ones_like(scores, dtype=bool)
        for b in used_bins:
            mask &= (bins != b)
        idx = np.argmax(np.where(mask, scores, -np.inf))
        if not np.isfinite(scores[idx]):
            break
        selected.append(int(idx+1))
        used_bins.add(int(bins[idx]))
    # Fill remaining with diversity penalty
    while len(selected) < K:
        penalty = np.zeros_like(scores)
        for s in selected:
            penalty += 0.1 / (np.abs(np.arange(Tm1) - (s-1)) + 1)
        idx = int(np.argmax(scores - penalty))
        cand = int(idx+1)
        if cand not in selected:
            selected.append(cand)
        else:
            order = np.argsort(-(scores - penalty))
            for j in order:
                cand = int(j+1)
                if cand not in selected:
                    selected.append(cand)
                    break
        if len(selected) >= K:
            break
    return sorted(selected)


def save_heatmap_pdf(deltaF: np.ndarray, key_steps: List[int], out_pdf: str, title: str = "ΔF Heatmap", xlabel: str = "Step", ylabel: str = "Layer"):
    ensure_dir(os.path.dirname(out_pdf))
    plt.figure(figsize=(7,4))
    sns.heatmap(deltaF, cmap='magma')
    for ks in key_steps:
        plt.axvline(ks + 0.5, color='cyan', linestyle='--', linewidth=1.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def timewarp_discover_toy(n_steps: int = 10, K: int = 3, image_shape: Tuple[int, int] = (64, 64), images_out_dir: str = ".research/iteration10/images") -> List[int]:
    """Runs a toy ΔF computation and returns selected key steps. Saves a PDF heatmap."""
    ensure_dir(images_out_dir)
    set_seed(123)

    class _Toy(nn.Module):
        def __init__(self, c: int = 16):
            super().__init__()
            self.conv1 = nn.Conv2d(3, c, 3, padding=1)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
            self.gate = nn.Linear(1, c)
        def forward(self, x, t):
            h = F.silu(self.conv1(x))
            g = self.gate(t.view(-1,1)).view(-1, h.shape[1], 1, 1)
            f1 = F.silu(h + g)
            f2 = F.silu(self.conv2(f1))
            return {"down_0": f1, "down_1": f2}

    B = 4
    H, W = image_shape
    x = torch.randn(B, 3, H, W)
    net = _Toy().eval()

    def _feat_fn(step_idx: int):
        t = torch.tensor([step_idx / max(1, n_steps-1)], dtype=torch.float32)
        with torch.no_grad():
            feats = net(x, t)
        return {k: v.detach().cpu() for k, v in feats.items()}

    deltaF, _ = compute_deltaF_heatmap(_feat_fn, n_steps=n_steps)
    key = greedy_timewarp_key_selection(deltaF, K=K, stratify_bins=4)
    pdf_path = os.path.join(images_out_dir, "deltaF_heatmap_timewarp_discovery_toy.pdf")
    save_heatmap_pdf(deltaF, key, pdf_path, title="ΔF Heatmap (Toy Time-Warp)")
    return key
