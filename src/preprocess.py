from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FiLM(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(time_dim, channels)
        self.to_shift = nn.Linear(time_dim, channels)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Ensure t_emb has batch dimension matching x; if a single time is provided, broadcast it
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        B = x.size(0)
        if t_emb.size(0) != B:
            if t_emb.size(0) == 1:
                t_emb = t_emb.expand(B, -1)
            else:
                raise ValueError(f"t_emb batch {t_emb.size(0)} does not match x batch {B}")
        s = self.to_scale(t_emb).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(t_emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
    def forward(self, t_scalar: torch.Tensor) -> torch.Tensor:
        return self.proj(t_scalar.view(-1, 1))


class SimpleDiffusionTeacher(nn.Module):
    """A tiny UNet-like epsilon predictor with multi-scale features and FiLM time conditioning.
    Returns epsilon and a list of features across 3 scales.
    """
    def __init__(self, in_channels: int = 4, base: int = 32, time_dim: int = 64):
        super().__init__()
        self.time = TimeEmbedding(time_dim)

        # Encoder-like
        self.conv1 = nn.Conv2d(in_channels, base, 3, padding=1)
        self.film1 = FiLM(base, time_dim)
        self.conv2 = nn.Conv2d(base, base, 3, padding=1)
        self.down1 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.film2 = FiLM(base*2, time_dim)
        self.conv3 = nn.Conv2d(base*2, base*2, 3, padding=1)
        self.down2 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.film3 = FiLM(base*4, time_dim)
        self.conv4 = nn.Conv2d(base*4, base*4, 3, padding=1)

        # Simple head to epsilon at highest resolution
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.out = nn.Conv2d(base, in_channels, 3, padding=1)

        self.feature_channels = [base, base*2, base*4]

    def forward(self, x: torch.Tensor, t_scalar: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        t_emb = self.time(t_scalar)
        h1 = F.silu(self.conv1(x))
        h1 = self.film1(h1, t_emb)
        h1 = F.silu(self.conv2(h1))  # [B, base, 32, 32]
        h2 = F.silu(self.down1(h1))
        h2 = self.film2(h2, t_emb)
        h2 = F.silu(self.conv3(h2))  # [B, 2*base, 16, 16]
        h3 = F.silu(self.down2(h2))
        h3 = self.film3(h3, t_emb)
        h3 = F.silu(self.conv4(h3))  # [B, 4*base, 8, 8]

        # Decode to epsilon
        u = F.silu(self.up1(h3))  # 16x16
        u = F.silu(self.up2(u))   # 32x32
        eps = self.out(u)         # [B,4,32,32]
        feats = [h1, h2, h3]
        return eps, feats


def compute_deltaF_over_time(teacher: SimpleDiffusionTeacher, T: int, batch: int, device: str) -> torch.Tensor:
    dev = torch.device(device)
    teacher = teacher.to(dev).eval()
    time_grid = torch.linspace(0.0, 1.0, steps=T, device=dev)

    # Record features per step as means over batch
    feats_seq: List[List[torch.Tensor]] = []  # [T][scales]
    with torch.no_grad():
        for t in range(T):
            latents = torch.randn(batch, 4, 32, 32, device=dev)
            _, feats = teacher(latents, time_grid[t])
            feats_seq.append([f.detach().cpu() for f in feats])

    # Compute ΔF aggregated across scales
    deltaF = torch.zeros(T-1)
    for s in range(len(teacher.feature_channels)):
        # stack over time: [T, B, C, H, W]
        xs = torch.stack([feats_seq[t][s] for t in range(T)], dim=0).float()
        # L2 change per step averaged over batch/spatial/channel, normalized by sqrt(C)
        C = xs.shape[2]
        d = (xs[1:] - xs[:-1]).pow(2).mean(dim=(1,2,3,4)).sqrt() / np.sqrt(C)
        deltaF[:d.shape[0]] += d
    return deltaF


def greedy_key_selection(deltaF: torch.Tensor, K: int, coverage: int = 2) -> List[int]:
    temp = deltaF.clone()
    Tm1 = temp.numel()
    chosen: List[int] = []
    for _ in range(K):
        idx = int(torch.argmax(temp).item())
        chosen.append(idx+1)  # 1-based
        low = max(0, idx - coverage)
        high = min(Tm1-1, idx + coverage)
        temp[low:high+1] *= 0.5
    return sorted(set(chosen))


def save_deltaF_plot(deltaF: torch.Tensor, keys_1based: List[int], out_pdf: str):
    if plt is None:
        print(f"Matplotlib not available; skipping plot {out_pdf}")
        return
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    sns.set(style='whitegrid')
    x = list(range(1, deltaF.numel()+1))
    plt.figure(figsize=(6,2))
    plt.imshow(deltaF.view(1,-1).cpu().numpy(), aspect='auto', cmap='viridis')
    for k in keys_1based:
        plt.axvline(k-1, color='w', linestyle='--', alpha=0.6)
    plt.yticks([])
    plt.xlabel('Step index (Δ between t and t-1)')
    plt.title('Aggregated ΔF over time')
    plt.tight_layout(); plt.savefig(out_pdf, bbox_inches='tight'); plt.close()
    print(f'Saved figure: {out_pdf}')
