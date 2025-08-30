#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities, model definitions, and EA-FA attention components.
- Implements Energy-Adaptive FlashAttention (EA-FA) analogs on T4-friendly SDPA.
- Provides NVML power sampler, EA-FA attention wrapper (RPA, MPTS, SKP analogs),
  TinyGPT model, dynamic batch-size search, and a single-epoch trainer.

Notes
- All imports within src use relative imports when referencing sibling modules.
- External dependencies are listed in requirements.txt.
- Designed to run on NVIDIA Tesla T4 (16 GB VRAM) and degrade gracefully
  if NVML/xFormers are not available.
"""
from __future__ import annotations
import math
import os
import time
import gc
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# Mixed-precision utilities
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    import contextlib
    autocast = contextlib.nullcontext  # type: ignore
    class _DummyScaler:
        def __init__(self): pass
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
    GradScaler = _DummyScaler  # type: ignore

# Optional NVML
try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# Optional xFormers
try:
    import xformers  # type: ignore
    from xformers.ops import memory_efficient_attention as xformers_mea  # type: ignore
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False


# ---------- Power sampling and DVFS ----------
class PowerSampler:
    """Background NVML power sampling and Joules integration."""
    def __init__(self, device_index: int = 0, sample_hz: float = 40.0) -> None:
        self.device_index = device_index
        self.sample_period = 1.0 / max(1.0, float(sample_hz))
        self.samples: List[Tuple[float, float]] = []  # (timestamp, watts)
        self._enabled = False
        self._thread = None
        self._stop = None
        self._nvml_ok = False
        self._handle = None
        self.idle_power_watts: Optional[float] = None
        if torch.cuda.is_available() and _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

    def _read_power_watts(self) -> Optional[float]:
        if not self._nvml_ok or self._handle is None:
            return None
        try:
            p_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return float(p_mw) / 1000.0
        except Exception:
            return None

    def start(self) -> None:
        if not self._nvml_ok:
            return
        import threading
        self.samples.clear()
        self._stop = threading.Event()
        def _run():
            while not self._stop.is_set():
                t0 = time.perf_counter()
                p = self._read_power_watts()
                if p is not None:
                    self.samples.append((t0, p))
                sleep_t = max(0.0, self.sample_period - (time.perf_counter() - t0))
                time.sleep(sleep_t)
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        self._enabled = True

    def stop(self) -> None:
        if not self._nvml_ok:
            return
        if self._thread is not None and self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        self._enabled = False

    def measure_idle(self, duration_s: float = 10.0) -> float:
        if not self._nvml_ok:
            self.idle_power_watts = 0.0
            return 0.0
        self.start()
        time.sleep(duration_s)
        self.stop()
        if not self.samples:
            self.idle_power_watts = 0.0
        else:
            self.idle_power_watts = float(np.mean([p for _, p in self.samples]))
        return float(self.idle_power_watts or 0.0)

    def energy_joules(self, subtract_idle: bool = True) -> float:
        if not self._nvml_ok or len(self.samples) < 2:
            return 0.0
        times = np.asarray([t for t, _ in self.samples], dtype=np.float64)
        powers = np.asarray([p for _, p in self.samples], dtype=np.float64)
        if subtract_idle and self.idle_power_watts is not None:
            powers = np.maximum(0.0, powers - float(self.idle_power_watts))
        dt = np.diff(times)
        p_mid = 0.5 * (powers[:-1] + powers[1:])
        return float(np.sum(p_mid * dt))


def try_set_power_limit_watts(limit_w: int, device_index: int = 0) -> bool:
    if not _NVML_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        min_lim, max_lim = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        min_w, max_w = int(min_lim / 1000), int(max_lim / 1000)
        set_w = int(np.clip(limit_w, min_w, max_w)) * 1000
        pynvml.nvmlDeviceSetPowerManagementLimit(h, set_w)
        return True
    except Exception:
        return False


# ---------- Attention backends ----------
class SDPAWrapper:
    """Unified interface to call different SDPA backends (PyTorch or xFormers)."""
    def __init__(self, method: str = "pytorch_mem_efficient") -> None:
        self.method = method
        if self.method.startswith("xformers") and not _XFORMERS_AVAILABLE:
            warnings.warn("xFormers not available; falling back to PyTorch mem_efficient")
            self.method = "pytorch_mem_efficient"

    def _pt_sdpa(self, q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False):
        enable_flash = False  # T4: use mem_efficient or math
        enable_mem = (self.method == "pytorch_mem_efficient")
        enable_math = (self.method == "pytorch_math")
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem,
        ):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    def _xformers(self, q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False):
        B, H, L, D = q.shape
        q_bshd = q.permute(0, 2, 1, 3).contiguous()
        k_bshd = k.permute(0, 2, 1, 3).contiguous()
        v_bshd = v.permute(0, 2, 1, 3).contiguous()
        out = xformers_mea(q_bshd, k_bshd, v_bshd, attn_bias=None, p=dropout_p, scale=None)
        return out.permute(0, 2, 1, 3).contiguous()

    def __call__(self, q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False):
        if self.method.startswith("xformers"):
            return self._xformers(q, k, v, attn_mask, dropout_p, is_causal)
        return self._pt_sdpa(q, k, v, attn_mask, dropout_p, is_causal)


class EAFAAttention:
    """Energy-Adaptive Attention wrapper implementing RPA, MPTS, and SKP analogs.

    - RPA: select backend and optionally set GPU power limit via NVML.
    - MPTS: route high-sharpness rows to FP32, others FP16.
    - SKP: norm-based KV tile skip via additive -inf mask.
    """
    def __init__(
        self,
        candidate_backends: List[str],
        power_limits_w: List[int] = [50, 60, 70],
        enable_rpa: bool = True,
        enable_mpts: bool = True,
        enable_skp: bool = True,
        entropy_tau: float = 1.5,
        skp_tile: int = 64,
        skp_eps: float = 1e-4,
        device_index: int = 0,
    ) -> None:
        self.candidate_backends = candidate_backends
        self.power_limits_w = power_limits_w
        self.enable_rpa = enable_rpa
        self.enable_mpts = enable_mpts
        self.enable_skp = enable_skp
        self.entropy_tau = entropy_tau
        self.skp_tile = skp_tile
        self.skp_eps = skp_eps
        self.device_index = device_index
        self._backend_cache: Dict[Tuple[int, int, bool, torch.dtype], str] = {}
        self._power_cache: Dict[Tuple[int, int, bool, torch.dtype], int] = {}

    def _select_backend_power(self, L: int, B: int, is_causal: bool, dtype: torch.dtype) -> Tuple[str, Optional[int]]:
        key = (L, B, is_causal, dtype)
        if key in self._backend_cache:
            return self._backend_cache[key], self._power_cache.get(key)
        chosen = "pytorch_mem_efficient"
        if L >= 1024 and _XFORMERS_AVAILABLE and any("xformers" in b for b in self.candidate_backends):
            chosen = "xformers"
        elif any(b == "pytorch_math" for b in self.candidate_backends):
            chosen = "pytorch_mem_efficient"
        power = None
        if self.enable_rpa and self.power_limits_w:
            power = int(np.median(self.power_limits_w))
        self._backend_cache[key] = chosen
        if power is not None:
            self._power_cache[key] = power
        return chosen, power

    def _apply_skp_mask(self, k: torch.Tensor, v: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_skp:
            return None
        with torch.no_grad():
            B, H, Lk, D = k.shape
            tile = self.skp_tile
            k2 = k.pow(2).mean(dim=-1)
            v2 = v.pow(2).mean(dim=-1)
            kv_energy = (k2 + v2).mean(dim=(0, 1))  # (Lk,)
            mask = torch.zeros((Lk,), device=k.device, dtype=torch.bool)
            for t in range(0, Lk, tile):
                seg = kv_energy[t:min(Lk, t+tile)]
                if float(seg.mean().item()) < self.skp_eps:
                    mask[t:min(Lk, t+tile)] = True
            if not bool(mask.any()):
                return None
            add_mask = torch.zeros((1, 1, 1, Lk), device=k.device, dtype=k.dtype)
            add_mask[..., mask] = float('-inf')
            return add_mask

    def _entropy_proxy(self, q: torch.Tensor, k: torch.Tensor, sample_stride: int = 8) -> torch.Tensor:
        B, H, Lq, D = q.shape
        Lk = k.shape[2]
        idx = torch.arange(0, Lk, sample_stride, device=q.device)
        k_sub = k[:, :, idx, :]
        scale = 1.0 / math.sqrt(D)
        logits = torch.einsum("bhld,bhmd->bhlm", q, k_sub) * scale
        return logits.max(dim=-1).values.detach()

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0,
                 is_causal: bool = True,
                 dtype_policy_fp16: torch.dtype = torch.float16) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, H, Lq, D = q.shape
        method, power = self._select_backend_power(Lq, B, is_causal, q.dtype)
        chosen_power = None
        if self.enable_rpa and power is not None:
            ok = try_set_power_limit_watts(power, device_index=self.device_index)
            if ok:
                chosen_power = power
        add_mask = self._apply_skp_mask(k, v)
        if add_mask is not None:
            attn_mask = (attn_mask + add_mask) if attn_mask is not None else add_mask
        backend = SDPAWrapper(method)
        out = torch.empty((B, H, Lq, D), device=q.device, dtype=q.dtype)
        with autocast(enabled=True, dtype=dtype_policy_fp16):
            out_lo = backend(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        out.copy_(out_lo)
        del out_lo
        routed_fp32_rows = 0
        total_rows = B * H * Lq
        if self.enable_mpts:
            sharp = self._entropy_proxy(q, k, sample_stride=8)
            hi_mask = (sharp > self.entropy_tau)
            routed_fp32_rows = int(hi_mask.sum().item())
            if routed_fp32_rows > 0:
                q_bh = q.reshape(B * H, Lq, D)
                k_bh = k.reshape(B * H, k.shape[2], D)
                v_bh = v.reshape(B * H, v.shape[2], D)
                out_bh = out.reshape(B * H, Lq, D)
                hi_mask_bh = hi_mask.reshape(B * H, Lq)
                backend_hi = SDPAWrapper("pytorch_math")
                for i in range(B * H):
                    sel = hi_mask_bh[i]
                    if int(sel.sum().item()) == 0:
                        continue
                    q_hi = q_bh[i, sel, :].unsqueeze(0).unsqueeze(1)  # (1,1,n_hi,D)
                    k_i = k_bh[i].unsqueeze(0).unsqueeze(1)  # (1,1,Lk,D)
                    v_i = v_bh[i].unsqueeze(0).unsqueeze(1)  # (1,1,Lk,D)
                    with torch.cuda.amp.autocast(enabled=False):
                        out_hi = backend_hi(q_hi, k_i, v_i, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
                    out_bh[i, sel, :] = out_hi.squeeze(0).squeeze(0).to(out_bh.dtype)
        diag = {
            "method": method,
            "power_W": chosen_power,
            "mpts_routed_fp32_rows": routed_fp32_rows,
            "total_rows": total_rows,
            "skp_active": add_mask is not None,
        }
        return out, diag


# ---------- Model ----------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(s + self.eps)
        return self.weight * x


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float, attn_impl: Optional[Callable] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.attn_impl = attn_impl
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        H = self.n_heads
        q = self.q_proj(self.norm1(x)).view(B, L, H, D // H).permute(0, 2, 1, 3).contiguous()
        k = self.k_proj(self.norm1(x)).view(B, L, H, D // H).permute(0, 2, 1, 3).contiguous()
        v = self.v_proj(self.norm1(x)).view(B, L, H, D // H).permute(0, 2, 1, 3).contiguous()
        is_causal = True
        if self.attn_impl is None:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        else:
            out, _ = self.attn_impl(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        x = x + self.dropout(self.o_proj(out))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, attn_impl: Optional[Callable] = None, max_seq_len: int = 4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout, attn_impl=attn_impl)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed[:, :L, :]
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.norm(x))
        return logits


# ---------- Training utilities ----------
@dataclass
class TrainConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    amp_dtype: torch.dtype = torch.float16
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    global_tokens_per_iter: int = 16384
    device_index: int = 0
    backends: List[str] = None  # type: ignore
    power_limits: List[int] = None  # type: ignore
    enable_rpa: bool = True
    enable_mpts: bool = True
    enable_skp: bool = True
    entropy_tau: float = 1.5
    skp_tile: int = 64
    skp_eps: float = 1e-4

    def __post_init__(self):
        if self.backends is None:
            self.backends = ["pytorch_mem_efficient", "pytorch_math", "xformers"]
        if self.power_limits is None:
            self.power_limits = [50, 60, 70]


def build_model(cfg: TrainConfig, device: torch.device) -> Tuple[nn.Module, EAFAAttention]:
    attn_impl = EAFAAttention(
        candidate_backends=cfg.backends,
        power_limits_w=cfg.power_limits,
        enable_rpa=cfg.enable_rpa,
        enable_mpts=cfg.enable_mpts,
        enable_skp=cfg.enable_skp,
        entropy_tau=cfg.entropy_tau,
        skp_tile=cfg.skp_tile,
        skp_eps=cfg.skp_eps,
        device_index=cfg.device_index,
    )
    model = TinyGPT(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        attn_impl=attn_impl,
    ).to(device)
    return model, attn_impl


def find_max_batch_size(
    model: nn.Module,
    seq_len: int,
    base_bs: int,
    max_tokens: int,
    device: torch.device,
    vocab_size: int,
    dtype: torch.dtype,
) -> Tuple[int, int]:
    torch.cuda.reset_peak_memory_stats(device)
    bs = max(1, base_bs)
    grad_accum = 1
    while seq_len * bs > max_tokens:
        bs = max(1, bs // 2)
    ok_bs = bs
    for trial in [bs, bs * 2, bs * 4]:
        try:
            x = torch.randint(0, vocab_size, (trial, seq_len), device=device)
            with autocast(enabled=True, dtype=dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
                loss.backward()
            ok_bs = trial
            model.zero_grad(set_to_none=True)
            del x, logits, loss
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                break
            else:
                raise
    grad_accum = max(1, max_tokens // max(1, seq_len * ok_bs))
    return ok_bs, grad_accum


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    dataloader,
    device: torch.device,
    steps: int,
    grad_accum: int,
    power: PowerSampler,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    model.train()
    tokens_total = 0
    loss_acc = 0.0
    power.start()
    t0 = time.time()
    it = iter(dataloader)
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                batch = next(it)
                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                with autocast(enabled=True, dtype=amp_dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                tokens_total += int(x.numel())
                loss_acc += float(loss.item())
            scaler.step(optimizer)
            scaler.update()
    except StopIteration:
        pass
    finally:
        power.stop()
    torch.cuda.synchronize()
    dur_s = max(1e-6, time.time() - t0)
    energy_j = power.energy_joules(subtract_idle=True)
    return {
        "tokens": tokens_total,
        "time_s": dur_s,
        "joules": energy_j,
        "loss_avg": loss_acc / max(1, steps * grad_accum),
    }
