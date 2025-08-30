import os
import math
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure headless plotting elsewhere; train does not plot except loss via helper.

# ----------------------------
# Utilities and reproducibility
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_device(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)


# ----------------------------
# Synthetic dataset (toy but structured)
# ----------------------------

class SyntheticPatternDataset(Dataset):
    def __init__(self, n: int = 1000, image_size: int = 32, n_classes: int = 3, seed: int = 42):
        set_seed(seed)
        self.n = n
        self.H = image_size
        self.W = image_size
        self.n_classes = n_classes
        self.samples = []
        patterns = ["stripes", "circles", "checkerboard"]
        for _ in range(n):
            cls = random.randint(0, n_classes-1)
            patt = patterns[cls]
            token_len = random.choice([5, 10, 20, 40])
            prompt = f"{patt} pattern with random colors and token length {token_len}"
            img = self._gen_image(patt)
            self.samples.append((img, cls, prompt, token_len))

    def _gen_image(self, patt: str) -> torch.Tensor:
        H, W = self.H, self.W
        img = np.zeros((3, H, W), dtype=np.float32)
        if patt == "stripes":
            stripe_w = random.randint(2, max(2, H//4))
            color = np.random.rand(3, 1, 1)
            for i in range(0, W, stripe_w*2):
                img[:, :, i:i+stripe_w] = color
        elif patt == "circles":
            cy, cx = random.randint(H//4, 3*H//4), random.randint(W//4, 3*W//4)
            r = random.randint(min(H,W)//8, min(H,W)//3)
            Y, X = np.ogrid[:H, :W]
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            color = np.random.rand(3)
            for c in range(3):
                img[c][mask] = color[c]
        else:  # checkerboard
            block = random.randint(2, max(2, H//4))
            color1 = np.random.rand(3, 1, 1)
            color2 = np.random.rand(3, 1, 1)
            for y in range(0, H, block):
                for x in range(0, W, block):
                    if ((y//block)+(x//block)) % 2 == 0:
                        img[:, y:y+block, x:x+block] = color1
                    else:
                        img[:, y:y+block, x:x+block] = color2
        img = np.clip(img, 0.0, 1.0)
        return torch.from_numpy(img)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, cls, prompt, token_len = self.samples[idx]
        return img, cls, prompt, token_len


# ----------------------------
# Diffusion scheduler (toy DDIM/DDPM-like)
# ----------------------------

class ToyScheduler:
    def __init__(self, n_steps: int = 40, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register(betas, alphas, alpha_bars)

    def register(self, betas, alphas, alpha_bars):
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.timesteps = torch.arange(self.n_steps-1, -1, -1, dtype=torch.long)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        # Ensure the indexing tensor and the indexed tensor are on the same device
        t_idx = t.long().to(x0.device)
        at = self.alpha_bars.to(x0.device)[t_idx].reshape(-1, 1, 1, 1)
        xt = torch.sqrt(at) * x0 + torch.sqrt(1 - at) * noise
        return xt, noise

    def ddim_step(self, x: torch.Tensor, t: int, eps_pred: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        device = x.device
        t_idx = t
        at = self.alpha_bars[t_idx].to(device)
        at_prev = self.alpha_bars[max(t_idx-1, 0)].to(device)
        eps = eps_pred
        x0 = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)
        dir_term = torch.sqrt(1 - at_prev) * eps
        x_prev = torch.sqrt(at_prev) * x0 + dir_term
        if eta > 0:
            sigma_t = eta * math.sqrt((1 - at_prev)/(1 - at)) * math.sqrt(1 - at/at_prev)
            x_prev = x_prev + sigma_t * torch.randn_like(x)
        return x_prev


# ----------------------------
# Tiny UNet-like model with feature summaries
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = ConvBlock(in_ch, out_ch)
        self.b2 = ConvBlock(out_ch, out_ch)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, 2, 1)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        skip = x
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch: Optional[int] = None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        if skip_ch is None:
            skip_ch = out_ch
        self.b1 = ConvBlock(out_ch + skip_ch, out_ch)
        self.b2 = ConvBlock(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.b1(x)
        x = self.b2(x)
        return x

class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act = nn.SiLU()
    def forward(self, t_emb):
        x = self.fc1(t_emb)
        x = self.act(x)
        x = self.fc2(x)
        return x

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device)
    )
    args = t.float().unsqueeze(-1) / freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class EpsHead(nn.Module):
    def __init__(self, in_ch, cond_dim, sum_dims: List[int]):
        super().__init__()
        self.cond_fc = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.sum_total = sum(sum_dims)
        self.summarizer_fc = nn.Sequential(
            nn.Linear(self.sum_total, 256), nn.SiLU(), nn.Linear(256, 256)
        )
        self.gamma = nn.Linear(256+128, in_ch)
        self.beta = nn.Linear(256+128, in_ch)
        self.conv1x1 = nn.Conv2d(in_ch, in_ch, 1)
        self.out = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x_in: torch.Tensor, summaries: List[torch.Tensor], cond: torch.Tensor):
        s = torch.cat(summaries, dim=1)
        s = self.summarizer_fc(s)
        c = self.cond_fc(cond)
        sc = torch.cat([s, c], dim=1)
        gamma = self.gamma(sc).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(sc).unsqueeze(-1).unsqueeze(-1)
        y = self.conv1x1(x_in)
        y = y * (1 + gamma) + beta
        y = F.silu(y)
        return self.out(y)

class TinyUNetSummarizer(nn.Module):
    def __init__(self, in_ch=3, base=32, time_dim=64, cond_dim=32, image_size=32):
        super().__init__()
        self.image_size = image_size
        self.in_conv = ConvBlock(in_ch, base)
        self.down1 = DownBlock(base, base)
        self.down2 = DownBlock(base, base*2)
        self.mid = ConvBlock(base*2, base*2)
        self.up2 = UpBlock(base*2, base, skip_ch=base*2)
        self.up1 = UpBlock(base, base)
        self.out_conv = ConvBlock(base, base)
        self.time_dim = time_dim
        self.time_mlp = TimeEmbed(time_dim)
        self.time_proj = nn.Linear(time_dim, base*2)
        self.cond_proj = nn.Linear(cond_dim, base)
        self.sum_dims = [base, base, base*2, base*2, base, base]
        self.eps_head = EpsHead(base, cond_dim, self.sum_dims)
        # Map internal feature channels to image channels for epsilon prediction
        self.eps_out = nn.Conv2d(base, in_ch, 1)

    def forward_heavy(self, x, t_emb, cond) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.in_conv(x)
        sum1 = F.adaptive_avg_pool2d(h, 1).flatten(1)
        d1, skip1 = self.down1(h)
        sum2 = F.adaptive_avg_pool2d(skip1, 1).flatten(1)
        d2, skip2 = self.down2(d1)
        sum3 = F.adaptive_avg_pool2d(d2, 1).flatten(1)
        m = self.mid(d2)
        sum4 = F.adaptive_avg_pool2d(m, 1).flatten(1)
        u2 = self.up2(m, skip2)
        sum5 = F.adaptive_avg_pool2d(u2, 1).flatten(1)
        u1 = self.up1(u2, skip1)
        u1 = self.out_conv(u1)
        sum6 = F.adaptive_avg_pool2d(u1, 1).flatten(1)
        summaries = [sum1, sum2, sum3, sum4, sum5, sum6]
        eps_feat = self.eps_head(u1, summaries, cond)
        eps = self.eps_out(eps_feat)
        return eps, summaries

    def forward(self, x, t: torch.Tensor, cond: torch.Tensor, summaries: Optional[List[torch.Tensor]] = None, light: bool = False):
        t_sin = sinusoidal_embedding(t, self.time_dim)
        t_feat = self.time_mlp(t_sin)
        _ = self.time_proj(t_feat)
        _ = self.cond_proj(cond)
        if not light:
            eps, sums = self.forward_heavy(x, t_feat, cond)
            return eps, sums
        else:
            assert summaries is not None, "Light forward requires summaries"
            # Lightweight path: create a small 1x1 conv on the fly
            conv = nn.Conv2d(x.size(1), 32, 1).to(x.device)
            u1 = F.silu(conv(x))
            eps_feat = self.eps_head(u1, summaries, cond)
            eps = self.eps_out(eps_feat)
            return eps, summaries


# ----------------------------
# LoRA-Prop adapters for summary vectors
# ----------------------------

class DeltaTFiLM(nn.Module):
    def __init__(self, hidden_dim: int, sum_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, sum_dim*2)
        )
    def forward(self, h: torch.Tensor, dt: torch.Tensor):
        out = self.net(dt.unsqueeze(1))
        return out

class LoRAAdapter(nn.Module):
    def __init__(self, sum_dim: int, r: int = 8, film_hidden: int = 32, alpha: float = 1.0):
        super().__init__()
        self.sum_dim = sum_dim
        self.r = r
        self.alpha = alpha
        self.A = nn.Linear(sum_dim, r, bias=False)
        self.B = nn.Linear(r, sum_dim, bias=False)
        self.film = DeltaTFiLM(film_hidden, sum_dim)
        self.register_buffer("int8_scales_A", torch.ones(r))
        self.register_buffer("int8_scales_B", torch.ones(sum_dim))
        self.int8 = False

    def enable_int8(self, enable=True):
        self.int8 = enable

    def forward(self, prev_sum: torch.Tensor, dt: torch.Tensor):
        if self.int8:
            A_w = (self.A.weight * self.int8_scales_A.view(-1,1)).to(prev_sum.dtype)
            B_w = (self.B.weight * self.int8_scales_B.view(-1,1)).to(prev_sum.dtype)
            h = F.linear(prev_sum, A_w)
            h = F.silu(h)
            delta = F.linear(h, B_w)
        else:
            h = self.A(prev_sum)
            h = F.silu(h)
            delta = self.B(h)
        film_out = self.film(h, dt)
        gamma, beta = film_out.chunk(2, dim=1)
        delta = delta * (1 + torch.tanh(gamma)) + torch.tanh(beta)
        return prev_sum + self.alpha * delta

class LoRAProp(nn.Module):
    def __init__(self, sum_dims: List[int], r: int = 8, film_hidden: int = 32, alpha: float = 1.0):
        super().__init__()
        self.adapters = nn.ModuleList([LoRAAdapter(d, r, film_hidden, alpha) for d in sum_dims])
    def enable_int8(self, enable=True):
        for a in self.adapters:
            a.enable_int8(enable)
    def forward(self, prev_summaries: List[torch.Tensor], dt: torch.Tensor) -> List[torch.Tensor]:
        return [ad(s, dt) for ad, s in zip(self.adapters, prev_summaries)]


# ----------------------------
# Helper: conditioning vector
# ----------------------------

def build_cond_vec(cls: torch.Tensor, tok_len: torch.Tensor, device) -> torch.Tensor:
    B = cls.shape[0]
    cls_onehot = F.one_hot(cls.to(torch.long), num_classes=3).float()
    tl = tok_len.float()
    bins = torch.stack([
        (tl <= 10).float(),
        ((tl > 10) & (tl <= 30)).float(),
        (tl > 30).float()
    ], dim=1)
    feat = torch.cat([cls_onehot, bins], dim=1).to(device)
    torch.manual_seed(123)
    proj = torch.randn(feat.size(1), 32, device=device) / math.sqrt(feat.size(1))
    return feat @ proj


# ----------------------------
# Key-step discovery (Time-Warp)
# ----------------------------

def discover_key_steps(model: TinyUNetSummarizer, scheduler: ToyScheduler, x_latent: torch.Tensor, cond: torch.Tensor, n_probe: int = 40, K: int = 3) -> List[int]:
    device = x_latent.device
    x = x_latent.clone()
    diffs = []
    summaries_hist = []
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps[:n_probe]):
            t_batch = torch.full((x.size(0),), float(t), device=device)
            eps, sums = model(x, t_batch, cond, light=False)
            summaries_hist.append([s.detach().float().cpu() for s in sums])
            x = scheduler.ddim_step(x, int(t), eps)
            if i > 0:
                prev = summaries_hist[i-1]
                cur = summaries_hist[i]
                score = 0.0
                for a, b in zip(prev, cur):
                    score += (a - b).pow(2).mean().item() ** 0.5
                diffs.append(score)
            else:
                diffs.append(0.0)
    diffs = np.array(diffs)
    candidate_indices = list(range(1, n_probe-1))
    picked = sorted(sorted(candidate_indices, key=lambda i: diffs[i], reverse=True)[:K])
    print(f"[KeyStep] selected key indices (len={K}): {picked}")
    return picked


# ----------------------------
# Training LoRA-Prop adapters
# ----------------------------

def train_lora_prop(model: TinyUNetSummarizer,
                    lora_prop: LoRAProp,
                    scheduler: ToyScheduler,
                    data_loader: DataLoader,
                    key_indices: List[int],
                    steps: int = 200,
                    lr: float = 1e-3,
                    device: Optional[torch.device] = None,
                    log_every: int = 50,
                    plot_outfile: Optional[str] = None) -> List[float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lora_prop.train()
    opt = torch.optim.AdamW(lora_prop.parameters(), lr=lr)
    losses = []

    probe_steps = scheduler.n_steps
    it = 0
    for epoch in range(10**9):
        for img, cls, prompt, tok_len in data_loader:
            it += 1
            img = img.to(device)
            cond = build_cond_vec(cls, tok_len, device)
            with torch.no_grad():
                j = random.randint(1, probe_steps-2)
                t_j = torch.full((img.size(0),), float(scheduler.timesteps[j]), device=device)
                x_t, _ = scheduler.add_noise(img, t_j.long())
                prev_keys = [k for k in key_indices if k <= j]
                if not prev_keys:
                    continue
                k = max(prev_keys)
                t_k = torch.full((img.size(0),), float(scheduler.timesteps[k]), device=device)
                _, sums_k = model(x_t, t_k, cond, light=False)
                _, sums_j = model(x_t, t_j, cond, light=False)
                dt = torch.full((img.size(0),), float(j-k) / (probe_steps-1), device=device)
            sums_hat = lora_prop(sums_k, dt)
            loss = 0.0
            for a, b in zip(sums_hat, sums_j):
                loss = loss + F.mse_loss(a, b)
            loss = loss / len(sums_hat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if it % log_every == 0:
                print(f"[LoRA-Train] step={it}, loss={loss.item():.6f}")
            if it >= steps:
                break
        if it >= steps:
            break

    if plot_outfile is not None:
        ensure_dir(os.path.dirname(plot_outfile))
        plt.figure(figsize=(5,3))
        plt.plot(losses, label="train loss")
        plt.xlabel("iteration")
        plt.ylabel("MSE")
        plt.title("LoRA-Prop Feature Regression Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_outfile, bbox_inches="tight")
        plt.close()
        print(f"[LoRA-Train] Saved training curve -> {plot_outfile}")

    return losses


# ----------------------------
# Saving / Loading helpers
# ----------------------------

def save_lora(lora: LoRAProp, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(lora.state_dict(), path)
    print(f"[Save] LoRA-Prop saved -> {path}")


def load_lora(lora: LoRAProp, path: str, map_location=None):
    sd = torch.load(path, map_location=map_location)
    lora.load_state_dict(sd)
    print(f"[Load] LoRA-Prop loaded <- {path}")
