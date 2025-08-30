# -*- coding: utf-8 -*-
"""
Training utilities for LoRA-Prop (+ Time-Warp) with a lightweight, toy setup.
- Implements a tiny backbone that emits per-layer features across a synthetic time-grid.
- Trains rank-r LoRA adapters to extrapolate features from the nearest previous key-step.
- Saves trained adapters to the models directory and a training loss curve PDF to .research/iteration9/images.

Notes
- This file avoids heavy dependencies and can run on CPU/GPU. It is designed to be fast for a quick test.
- Real diffusion model integration (e.g., diffusers UNet) can be added behind feature flags in future iterations.
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Matplotlib only used for saving the training curve PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ----------------------------
# Core Modules: Time Emb + LoRA
# ----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, delta_t: torch.Tensor):  # delta_t in [0,1]
        half = self.dim // 2
        device = delta_t.device
        freqs = torch.exp(
            torch.linspace(math.log(1e-4), 0, steps=half, device=device)
        )
        angles = delta_t[:, None] * freqs[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class LoRAAdapter(nn.Module):
    """Rank-r adapter with time modulation for feature propagation.
    Assumes input features are (B, C, H, W). Two 1x1 convs implement low-rank.
    """
    def __init__(self, channels: int, rank: int = 8, time_dim: int = 128):
        super().__init__()
        self.rank = rank
        self.channels = channels
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.mod = nn.Linear(time_dim, rank)
        # Init near-zero to preserve base behavior
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, feat_prev: torch.Tensor, delta_t: torch.Tensor):
        B, C, H, W = feat_prev.shape
        t_emb = self.time_emb(delta_t)  # (B, time_dim)
        scale = self.mod(t_emb).view(B, self.rank, 1, 1)
        z = self.down(feat_prev)
        z = z * (1 + scale)
        out = self.up(z)
        return feat_prev + out  # residual prediction


class AdapterBank(nn.Module):
    """Container of per-layer adapters keyed by layer_id."""
    def __init__(self, layer_channels: Dict[str, int], rank: int = 8):
        super().__init__()
        self.adapters = nn.ModuleDict({k: LoRAAdapter(c, rank=rank) for k, c in layer_channels.items()})

    def forward(self, layer_id: str, feat_prev: torch.Tensor, delta_t: torch.Tensor):
        return self.adapters[layer_id](feat_prev, delta_t)


# ----------------------------
# Toy Backbone (for quick tests)
# ----------------------------
class ToyBackbone(nn.Module):
    """A tiny CNN that produces two per-layer feature maps for a given time t.
    This mimics a UNet's intermediate features without heavy models.
    """
    def __init__(self, c1: int = 32, c2: int = 64):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.enc = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.SiLU(),
        )
        self.down = nn.AvgPool2d(2)
        self.bottleneck = nn.Conv2d(c1, c2, 3, padding=1)
        self.time_gate1 = nn.Linear(1, c1)
        self.time_gate2 = nn.Linear(1, c2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # t is normalized in [0,1], shape (B,)
        B, _, H, W = x.shape
        h1 = self.enc(x)
        g1 = self.time_gate1(t.view(B,1)).view(B, -1, 1, 1)
        f1 = F.silu(h1 + g1)
        h2 = self.down(f1)
        h2 = self.bottleneck(h2)
        g2 = self.time_gate2(t.view(B,1)).view(B, -1, 1, 1)
        f2 = F.silu(h2 + g2)
        return {"down_0": f1, "down_1": f2}


# ----------------------------
# Dataset for Adapter Training
# ----------------------------
class ToyFeatureDataset(Dataset):
    """Produces (layer_id, feat_prev, feat_t, delta_t) tuples from synthetic sequences.
    - We synthesize sequences across seeds and images.
    - Key steps provided; non-key steps produce training pairs.
    """
    def __init__(
        self,
        n_images: int,
        image_shape: Tuple[int, int] = (64, 64),
        total_steps: int = 10,
        key_steps: List[int] = None,
        device: str = "cpu",
        seed: int = 123,
        c1: int = 32,
        c2: int = 64,
    ):
        super().__init__()
        set_seed(seed)
        self.device = torch.device(device)
        self.total_steps = total_steps
        self.key_steps = sorted(key_steps or [2, 5, 8])
        self.backbone = ToyBackbone(c1=c1, c2=c2).to(self.device)
        self.samples: List[Tuple[str, torch.Tensor, torch.Tensor, float]] = []
        H, W = image_shape

        for img_idx in range(n_images):
            x = torch.randn(1, 3, H, W, device=self.device)
            # Precompute all features for this image across time
            feats_by_t: Dict[int, Dict[str, torch.Tensor]] = {}
            for t_idx in range(total_steps):
                tnorm = torch.tensor([t_idx / max(1, total_steps-1)], device=self.device)
                with torch.no_grad():
                    feats_by_t[t_idx] = {k: v.detach().clone() for k, v in self.backbone(x, tnorm).items()}
            # Build training pairs for each non-key step
            for t_idx in range(total_steps):
                if t_idx in self.key_steps:
                    continue
                # nearest previous key step
                prev_keys = [ks for ks in self.key_steps if ks < t_idx]
                if len(prev_keys) == 0:
                    continue
                t_prev = max(prev_keys)
                delta_t = float((t_idx - t_prev) / max(1, total_steps-1))
                for layer_id in ["down_0", "down_1"]:
                    f_prev = feats_by_t[t_prev][layer_id]
                    f_t = feats_by_t[t_idx][layer_id]
                    self.samples.append((layer_id, f_prev.cpu(), f_t.cpu(), delta_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        layer_id, f_prev, f_t, delta_t = self.samples[idx]
        return layer_id, f_prev, f_t, torch.tensor([delta_t], dtype=torch.float32)


# ----------------------------
# Training Routine
# ----------------------------
@dataclass
class TrainConfig:
    rank: int = 8
    batch_size: int = 16
    epochs: int = 3
    lr: float = 1e-3
    n_images: int = 32
    total_steps: int = 10
    key_steps: Tuple[int, ...] = (2, 5, 8)
    device: str = "cpu"
    images_out_dir: str = ".research/iteration9/images"
    models_out_dir: str = "models"
    model_name: str = "loraprop_toy.pt"


def collate_fn(batch):
    # Group by layer_id for more efficient adapter calls
    layer_ids = [b[0] for b in batch]
    f_prev = torch.cat([b[1] for b in batch], dim=0)
    f_t = torch.cat([b[2] for b in batch], dim=0)
    delta_t = torch.cat([b[3] for b in batch], dim=0).view(-1)
    return layer_ids, f_prev, f_t, delta_t


def train_lora_adapters_toy(cfg: TrainConfig) -> str:
    """Trains LoRA adapters on synthetic feature sequences.
    Returns the path to the saved adapter checkpoint.
    """
    ensure_dir(cfg.images_out_dir)
    ensure_dir(cfg.models_out_dir)
    device = torch.device(cfg.device)

    # Create dataset and loaders
    dset = ToyFeatureDataset(
        n_images=cfg.n_images,
        image_shape=(64, 64),
        total_steps=cfg.total_steps,
        key_steps=list(cfg.key_steps),
        device=cfg.device,
        seed=123,
    )
    loader = DataLoader(dset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    # Determine channels by peeking at one sample
    _, f_prev, _, _ = next(iter(loader))
    C = f_prev.shape[1]
    # But we have two layers with different channels: replicate logic
    layer_channels = {"down_0": 32, "down_1": 64}
    adapters = AdapterBank(layer_channels, rank=cfg.rank).to(device)

    opt = torch.optim.AdamW(adapters.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    losses = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        adapters.train()
        running = 0.0
        n_batches = 0
        for layer_ids, f_prev, f_t, delta_t in loader:
            f_prev = f_prev.to(device)
            f_t = f_t.to(device)
            delta_t = delta_t.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                preds = []
                offset = 0
                # We have mixed layer_ids in the batch; process in a loop
                for i in range(len(layer_ids)):
                    lid = layer_ids[i]
                    pred_i = adapters(lid, f_prev[i:i+1], delta_t[i:i+1])
                    preds.append(pred_i)
                    offset += 1
                pred = torch.cat(preds, dim=0)
                loss = F.mse_loss(pred, f_t)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            n_batches += 1
        epoch_loss = running / max(1, n_batches)
        losses.append(epoch_loss)
        print(f"[Train] Epoch {epoch+1}/{cfg.epochs} - Loss: {epoch_loss:.6f}")

    t1 = time.time()
    print(f"[Train] Finished in {t1 - t0:.2f}s. Final loss: {losses[-1]:.6f}")

    # Save adapters
    ckpt_path = os.path.join(cfg.models_out_dir, cfg.model_name)
    torch.save({"state_dict": adapters.state_dict(), "rank": cfg.rank, "layer_channels": layer_channels}, ckpt_path)
    print(f"[Train] Saved adapters to {ckpt_path}")

    # Save training curve PDF
    plt.figure(figsize=(5,3))
    xs = list(range(1, len(losses)+1))
    plt.plot(xs, losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LoRA-Prop Toy Training Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_pdf = os.path.join(cfg.images_out_dir, "training_curve_loraprop_toy.pdf")
    plt.savefig(loss_pdf, bbox_inches='tight')
    plt.close()
    print(f"[Train] Saved training curve PDF to {loss_pdf}")

    return ckpt_path


def load_adapters_toy(ckpt_path: str, device: str = "cpu") -> AdapterBank:
    ckpt = torch.load(ckpt_path, map_location=device)
    bank = AdapterBank(ckpt["layer_channels"], rank=ckpt["rank"]).to(device)
    bank.load_state_dict(ckpt["state_dict"])
    bank.eval()
    return bank
