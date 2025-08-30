from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Note: Inside src, always use relative imports
from .preprocess import SimpleDiffusionTeacher, set_seed


class LoRAPropBlock(nn.Module):
    """Low-Rank Propagation block: predicts feature at time t from cached feature at previous key step.
    Implements a simple rank-r adapter with a small time MLP producing an additive modulation.
    """
    def __init__(self, channels: int, rank: int = 8):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, rank), nn.SiLU(), nn.Linear(rank, rank)
        )

    def forward(self, feat: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        assert C == self.channels
        dt = dt.view(B, 1)
        time_vec = self.time_mlp(dt).view(B, self.rank, 1, 1)
        low = self.down(feat)
        low = low + time_vec
        up = self.up(low)
        return feat + up


class LightEpsHead(nn.Module):
    """Aggregates multi-scale features into an epsilon prediction."""
    def __init__(self, in_channels_list: List[int], out_channels: int = 4, hidden: int = 128):
        super().__init__()
        total_in = sum(in_channels_list)
        self.proj = nn.Sequential(
            nn.Conv2d(total_in, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(f, (32, 32)) for f in feats]
        x = torch.cat(pooled, dim=1)
        return self.proj(x)


class LoRAPropLightPath(nn.Module):
    """Holds LoRAPropBlocks per feature scale and a light epsilon head."""
    def __init__(self, layer_channels: List[int], rank: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([LoRAPropBlock(c, rank=rank) for c in layer_channels])
        self.head = LightEpsHead(layer_channels, out_channels=4)

    def forward(self, cached_feats: List[torch.Tensor], dt: torch.Tensor, zero_summary: bool = False) -> torch.Tensor:
        prop_feats = []
        for blk, feat in zip(self.blocks, cached_feats):
            feat_in = torch.zeros_like(feat) if zero_summary else feat
            prop = blk(feat_in, dt)
            prop_feats.append(prop)
        eps_hat = self.head(prop_feats)
        return eps_hat


@dataclass
class TrainConfig:
    seed: int = 123
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr: float = 5e-4
    batch_size: int = 8
    iters: int = 200
    rank: int = 8
    save_path: str = 'models/loraprop_lightpath.pt'


def train_lora_prop(keys_1based: List[int], T: int,
                    teacher: SimpleDiffusionTeacher,
                    cfg: TrainConfig) -> Tuple[LoRAPropLightPath, Dict[str, Any]]:
    """Train LoRA-Prop light path to predict teacher epsilon at non-key steps from cached key-step features."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    teacher = teacher.to(device).eval()

    layer_channels = teacher.feature_channels  # e.g., [32, 64, 128]
    light = LoRAPropLightPath(layer_channels=layer_channels, rank=cfg.rank).to(device)
    opt = optim.AdamW(light.parameters(), lr=cfg.lr)

    # Pre-create linear time grid [0,1] for T steps
    time_grid = torch.linspace(0.0, 1.0, steps=T, device=device)

    # Training loop
    loss_hist: List[float] = []
    non_key_steps = [t for t in range(1, T+1) if (t not in set(keys_1based))]
    if len(non_key_steps) == 0:
        raise RuntimeError('No non-key steps to train on; increase T or decrease K.')

    for it in range(cfg.iters):
        B = cfg.batch_size
        # Sample random timesteps t from non-key steps, ensure prev key exists
        t_idx = torch.tensor([non_key_steps[i % len(non_key_steps)] for i in range(B)], device=device)
        # Map to prev key
        prev_keys = []
        for ti in t_idx.tolist():
            candidates = [k for k in keys_1based if k < ti]
            if len(candidates) == 0:
                candidates = [keys_1based[0]]
            prev_keys.append(max(candidates))
        prev_keys = torch.tensor(prev_keys, device=device)

        # Random latents
        latents = torch.randn(B, 4, 32, 32, device=device)

        # Teacher forward at prev key and at t
        with torch.no_grad():
            feats_prev_list = []
            for b in range(B):
                eps_prev, feats_prev = teacher(latents[b:b+1], time_grid[prev_keys[b]-1])
                feats_prev_list.append(feats_prev)
            # Collate per-scale features into batched tensors
            # feats_prev_list: list length B, each is list of tensors [C,H,W] per scale
            cached_feats = []
            n_scales = len(layer_channels)
            for s in range(n_scales):
                cached_feats.append(torch.cat([feats_prev_list[b][s] for b in range(B)], dim=0))

            eps_targets = []
            for b in range(B):
                eps_t, _ = teacher(latents[b:b+1], time_grid[t_idx[b]-1])
                eps_targets.append(eps_t)
            eps_target = torch.cat(eps_targets, dim=0)

        # dt normalized by T
        dt = (t_idx.float() - prev_keys.float()) / float(T)

        eps_hat = light(cached_feats, dt, zero_summary=False)
        loss = F.mse_loss(eps_hat, eps_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_hist.append(float(loss.item()))
        if (it+1) % max(1, cfg.iters // 10) == 0:
            print(f"[train] iter {it+1}/{cfg.iters} | loss={loss.item():.6f}")

    # Save
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save({'state_dict': light.state_dict(), 'layer_channels': layer_channels, 'rank': cfg.rank,
                'loss_hist': loss_hist, 'T': T, 'keys': keys_1based}, cfg.save_path)
    print(f"Saved LoRA-Prop light path: {cfg.save_path}")

    return light, {'loss_hist': loss_hist}
