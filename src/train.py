import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional FLOPs estimation (safe import)
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

# -----------------------------
# Utilities
# -----------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prompt_to_seed(prompt: str) -> int:
    return abs(hash(prompt)) % (2**32)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    # timesteps in [0, 1]
    device = timesteps.device
    half_dim = dim // 2
    if half_dim == 0:
        return timesteps[:, None]
    emb = math.log(10000) / max(1, (half_dim - 1))
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # [B, dim]

# -----------------------------
# Toy data generator (prompts -> synthetic images)
# -----------------------------

class ToyPromptDataset(Dataset):
    def __init__(self, prompts: List[str], image_size: int = 64):
        self.prompts = prompts
        self.H = self.W = image_size

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        p = self.prompts[idx]
        seed = prompt_to_seed(p)
        torch.manual_seed(seed)
        np.random.seed(seed)
        img = self._synth_image_from_prompt(p)
        return {
            'prompt': p,
            'seed': int(seed),
            'image': img  # [3, H, W], in [-1,1]
        }

    def _synth_image_from_prompt(self, p: str) -> torch.Tensor:
        H, W = self.H, self.W
        canvas = torch.zeros(3, H, W)
        tokens = p.lower().split()
        # color palette from hashed tokens
        base_color = torch.tensor([
            ((abs(hash('r'+t)) % 1000) / 1000.0) for t in ['r','g','b']
        ])
        base_color = (base_color - 0.5) * 2.0
        canvas += base_color[:, None, None] * 0.1

        # patterns by tokens
        if any(tok in tokens for tok in ['panorama', 'wide', 'landscape', 'skyline']):
            for i in range(0, H, 4):
                canvas[:, i:i+2, :] += 0.15
        if any(tok in tokens for tok in ['portrait', 'face', 'person']):
            for k in range(3):
                cx, cy = (H//2 + (k-1)*6, W//2 + (k-1)*6)
                rr, cc = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                mask = ((rr - cx)**2 + (cc - cy)**2) < (H//4)**2
                color = torch.zeros(3)
                color[k] = 0.5
                canvas[:, mask] += color[:, None]
        if any(tok in tokens for tok in ['depth', 'canny', 'edge', 'control']):
            noise = torch.randn(3, H, W) * 0.05
            canvas += noise
            for i in range(0, W, 3):
                canvas[:, :, i] += 0.1
        if any(tok in tokens for tok in ['negative', 'lowres', 'noisy']):
            canvas += torch.randn_like(canvas) * 0.2
        if any(tok in tokens for tok in ['long', 'many', 'detailed', 'complex']):
            num_blobs = 8
            for _ in range(num_blobs):
                cx = np.random.randint(0, H)
                cy = np.random.randint(0, W)
                rad = np.random.randint(H//16, H//8)
                rr, cc = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                mask = ((rr - cx)**2 + (cc - cy)**2) < rad**2
                color = torch.rand(3) * 0.4
                canvas[:, mask] += color[:, None]
        canvas = canvas.clamp(-1, 1)
        return canvas

# -----------------------------
# Diffusion schedule utilities (DDIM-like minimal)
# -----------------------------


def make_beta_schedule(n_steps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor


def build_schedule(n_steps: int) -> DiffusionSchedule:
    betas = make_beta_schedule(n_steps)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alphas_bar=alphas_bar)


def q_sample(x0: torch.Tensor, t_idx: int, sched: DiffusionSchedule, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = sched.alphas_bar[t_idx]
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise


def ddim_step(x_t: torch.Tensor, eps_pred: torch.Tensor, t_idx: int, sched: DiffusionSchedule, eta: float = 0.0) -> torch.Tensor:
    a_t = sched.alphas[t_idx]
    a_bar_t = sched.alphas_bar[t_idx]
    if t_idx == 0:
        a_bar_prev = torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)
    else:
        a_bar_prev = sched.alphas_bar[t_idx-1]
    x0_pred = (x_t - torch.sqrt(1.0 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)
    dir_xt = torch.sqrt(1.0 - a_bar_prev) * eps_pred
    x_prev = torch.sqrt(a_bar_prev) * x0_pred + dir_xt
    return x_prev

# -----------------------------
# Tiny Teacher UNet (toy)
# -----------------------------

class ResBlock(nn.Module):
    def __init__(self, ch: int, t_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)
        self.t_proj = nn.Linear(t_emb_dim, ch*2)

    def forward(self, x, t_emb):
        scale, shift = self.t_proj(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = F.silu(self.gn1(self.conv1(x)))
        h = h * (1 + scale) + shift
        h = F.silu(self.gn2(self.conv2(h)))
        return x + h


class TinyUNetTeacher(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, t_emb_dim=64):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim), nn.SiLU(), nn.Linear(t_emb_dim, t_emb_dim)
        )
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.down1 = ResBlock(base_ch, t_emb_dim)
        self.ds1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)
        self.down2 = ResBlock(base_ch*2, t_emb_dim)
        self.mid = ResBlock(base_ch*2, t_emb_dim)
        self.us1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1)
        self.up1 = ResBlock(base_ch*2, t_emb_dim)
        self.out = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), nn.SiLU(), nn.Conv2d(base_ch, in_ch, 3, 1, 1)
        )
        self.enc_proj0 = nn.Conv2d(base_ch, 8, 1)
        self.enc_proj1 = nn.Conv2d(base_ch*2, 8, 1)
        self.last_enc_feats = None

    def forward(self, x_t: torch.Tensor, t_scalar: torch.Tensor, return_enc: bool = False):
        t_emb = sinusoidal_time_embedding(t_scalar, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        h0 = self.in_conv(x_t)
        h1 = self.down1(h0, t_emb)
        e0 = self.enc_proj0(h1)
        h2 = self.ds1(h1)
        h3 = self.down2(h2, t_emb)
        e1 = self.enc_proj1(h3)
        h4 = self.mid(h3, t_emb)
        u1 = self.us1(h4)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up1(u1, t_emb)
        u1 = torch.cat([u1, h0], dim=1)
        out_eps = self.out(u1)
        if return_enc:
            self.last_enc_feats = (e0.detach(), e1.detach())
            return out_eps, self.last_enc_feats
        return out_eps

# -----------------------------
# Surrogate predictor (depthwise separable)
# -----------------------------

class SurfBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c)
        self.pw = nn.Conv2d(c, c, kernel_size=1)
        self.gn = nn.GroupNorm(8, c)

    def forward(self, x):
        return F.silu(self.gn(self.pw(self.dw(x)) + x))


class SurfNet(nn.Module):
    def __init__(self, cin: int, out_ch: int = 3, c_hidden: int = 64, blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv2d(cin, c_hidden, 1)
        self.body = nn.Sequential(*[SurfBlock(c_hidden) for _ in range(blocks)])
        self.do = nn.Dropout2d(dropout)
        self.out = nn.Conv2d(c_hidden, out_ch, 1)

    def forward(self, x):
        x = self.do(self.inp(x))
        x = self.body(x)
        return self.out(x)

# -----------------------------
# SURF Sampler and key-step scheduling
# -----------------------------

@dataclass
class KeyStepSchedule:
    indices: List[int]


def heuristic_key_steps(T: int, K: int) -> KeyStepSchedule:
    if K >= T:
        return KeyStepSchedule(indices=list(range(T)))
    candidate = np.unique(np.clip((np.geomspace(1, T, num=K)*0.6).astype(int), 0, T-1))
    idxs = sorted(set([int(i) for i in candidate]))
    while len(idxs) < K:
        idxs.append(min(T-1, idxs[-1]+1))
    return KeyStepSchedule(indices=sorted(idxs[:K]))


class SurfSampler:
    def __init__(self, teacher: TinyUNetTeacher, sched: DiffusionSchedule, surf: nn.Module,
                 tau: float, key_steps: KeyStepSchedule, device: torch.device):
        self.teacher = teacher
        self.sched = sched
        self.surf = surf
        self.tau = float(tau)
        self.key_steps = set(key_steps.indices)
        self.device = device

    @torch.no_grad()
    def mc_var(self, inp: torch.Tensor, T: int = 4) -> torch.Tensor:
        self.surf.train()
        preds = []
        for _ in range(T):
            preds.append(self.surf(inp))
        self.surf.eval()
        preds = torch.stack(preds, dim=0)
        var = preds.var(dim=0)
        return var.mean(dim=(1,2,3))

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, steps: int, method: str, texts: Optional[List[str]] = None) -> Dict:
        B, C, H, W = x0.size()
        T = steps
        device = self.device
        x = torch.randn_like(x0)
        nfes = 0
        s = self.sched
        t_list = torch.linspace(1.0, 1.0/T, T, device=device)
        eps_prev = torch.zeros_like(x)
        last_cache = None
        surrogate_calls = 0
        teacher_calls = 0

        for i in range(T-1, -1, -1):
            t_scalar = t_list[i].expand(B)
            if method == 'plms50':
                eps, cache = self.teacher(x, t_scalar, return_enc=True)
                last_cache = cache
                eps_prev = eps
                nfes += 1
                teacher_calls += 1
            elif method == 'ddim8':
                keep = set(np.linspace(0, T-1, 8).round().astype(int).tolist())
                if i in keep:
                    eps, cache = self.teacher(x, t_scalar, return_enc=True)
                    last_cache = cache
                    eps_prev = eps
                    nfes += 1
                    teacher_calls += 1
                else:
                    eps = eps_prev
            elif method == 'pred_reuse':
                if i in self.key_steps or last_cache is None:
                    eps, cache = self.teacher(x, t_scalar, return_enc=True)
                    last_cache = cache
                    eps_prev = eps
                    nfes += 1
                    teacher_calls += 1
                else:
                    eps = eps_prev
            elif method == 'surf':
                if i in self.key_steps or last_cache is None:
                    eps, cache = self.teacher(x, t_scalar, return_enc=True)
                    last_cache = cache
                    eps_prev = eps
                    nfes += 1
                    teacher_calls += 1
                else:
                    e0, e1 = last_cache
                    t_emb = sinusoidal_time_embedding(t_scalar, 16)
                    t_map = t_emb[:, :, None, None].repeat(1, 1, H, W)
                    surf_inp = torch.cat([x, eps_prev, e0, e1, t_map], dim=1)
                    d_eps = self.surf(surf_inp)
                    eps = eps_prev + d_eps
                    surrogate_calls += 1
                    var = self.mc_var(surf_inp, T=4)
                    if (var > self.tau).any():
                        eps, cache = self.teacher(x, t_scalar, return_enc=True)
                        last_cache = cache
                        eps_prev = eps
                        nfes += 1
                        teacher_calls += 1
                    else:
                        eps_prev = eps
            else:
                raise ValueError(f"Unknown method {method}")

            x = ddim_step(x, eps, i, s, eta=0.0)

        x = x.clamp(-1, 1)
        return {
            'img': x.detach(),
            'NFE': nfes,
            'teacher_calls': teacher_calls,
            'surrogate_calls': surrogate_calls,
        }

# -----------------------------
# Training datasets
# -----------------------------

class ToyDiffusionTrainDataset(Dataset):
    def __init__(self, base_data: List[Dict], sched: DiffusionSchedule):
        self.base = base_data
        self.sched = sched

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        x0 = item['image']  # [3,H,W]
        T = len(self.sched.betas)
        t_idx = np.random.randint(0, T)
        x_t, noise = q_sample(x0, t_idx, self.sched)
        t_scalar = (t_idx + 1) / T
        return x_t, torch.tensor(t_scalar).float(), noise, x0


class SurfTrainDataset(Dataset):
    def __init__(self, base_data: List[Dict], teacher: TinyUNetTeacher, sched: DiffusionSchedule,
                 key_steps: KeyStepSchedule):
        self.base = base_data
        self.teacher = teacher
        self.sched = sched
        self.key = set(key_steps.indices)
        self.T = len(sched.betas)

    def __len__(self):
        return len(self.base)

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.base[idx]
        x0 = item['image']  # [3,H,W]
        all_idxs = list(range(self.T))
        non_key = [i for i in all_idxs if i not in self.key and i-1 >= 0]
        if not non_key:
            t_idx = np.random.randint(1, self.T)
        else:
            t_idx = random.choice(non_key)
        x_t, _ = q_sample(x0, t_idx, self.sched)
        x_t_b = x_t.unsqueeze(0)  # [1,3,H,W]
        t_prev = torch.tensor([(t_idx) / self.T]).float()
        eps_prev, cache_prev = self.teacher(x_t_b, t_prev, return_enc=True)
        t_curr = torch.tensor([(t_idx+1) / self.T]).float()
        eps_curr = self.teacher(x_t_b, t_curr)
        e0, e1 = cache_prev
        B, C, H, W = x_t_b.shape
        t_emb = sinusoidal_time_embedding(t_curr, 16)
        t_map = t_emb[:, :, None, None].repeat(1, 1, H, W)
        surf_inp = torch.cat([x_t_b, eps_prev, e0, e1, t_map], dim=1)
        d_eps_target = eps_curr - eps_prev
        return surf_inp.squeeze(0), d_eps_target.squeeze(0)

# -----------------------------
# Train loops and exports
# -----------------------------

def train_teacher(teacher: TinyUNetTeacher, sched: DiffusionSchedule, dataset: Dataset,
                  steps: int = 500, batch_size: int = 16, lr: float = 2e-3,
                  device: torch.device = torch.device('cpu')) -> List[float]:
    teacher.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(teacher.parameters(), lr=lr)
    it = iter(loader)
    losses = []
    for step in range(steps):
        try:
            x_t, t_scalar, noise, _ = next(it)
        except StopIteration:
            it = iter(loader)
            x_t, t_scalar, noise, _ = next(it)
        x_t = x_t.to(device)
        t_scalar = t_scalar.to(device)
        noise = noise.to(device)
        eps = teacher(x_t, t_scalar)
        loss = F.mse_loss(eps, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step+1) % max(1, (steps//5)) == 0:
            print(f"[Teacher] Step {step+1}/{steps}, loss={np.mean(losses[-50:]):.4f}")
    return losses


def train_surrogate(surf: SurfNet, dataset: SurfTrainDataset, steps: int = 500, batch_size: int = 16,
                    lr: float = 2e-3, device: torch.device = torch.device('cpu')) -> List[float]:
    surf.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(surf.parameters(), lr=lr)
    it = iter(loader)
    losses = []
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        pred = surf(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step+1) % max(1, (steps//5)) == 0:
            print(f"[Surrogate] Step {step+1}/{steps}, loss={np.mean(losses[-50:]):.4f}")
    return losses


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def load_checkpoint(model: nn.Module, path: str, map_location: Optional[str] = None):
    sd = torch.load(path, map_location=map_location or 'cpu')
    model.load_state_dict(sd)
    print(f"Loaded model from {path}")


def export_surrogate_torchscript(surf: SurfNet, example_inp: torch.Tensor, path: str) -> str:
    surf.eval()
    with torch.no_grad():
        traced = torch.jit.trace(surf, example_inp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.jit.save(traced, path)
    print(f"Saved TorchScript surrogate to {path}")
    return path

# -----------------------------
# FLOPs utilities (optional if thop installed)
# -----------------------------

def estimate_flops(model: nn.Module, input_shapes: Tuple[Tuple[int]], device: torch.device) -> float:
    if thop_profile is None:
        return float('nan')
    inputs = []
    for shape in input_shapes:
        inputs.append(torch.randn(*shape).to(device))
    model = model.to(device)
    flops, _ = thop_profile(model, inputs=tuple(inputs), verbose=False)
    return float(flops)
