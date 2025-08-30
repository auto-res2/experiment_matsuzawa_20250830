#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation and Energy-Adaptive SDPA experiments for NVIDIA T4 (16 GB).

Implements:
- NVML power sampling (25 Hz), idle subtraction, trapezoidal integration.
- EA-FA policy: runtime DVFS heuristic (RPA) + coarse mixed precision routing (MPTS) + backend selection.
- SDPA patch for GPT-2 attention, with xFormers optional.
- Dataset pipeline reading from a pre-tokenized token stream saved by preprocess.
- Robust logging, OOM-safe batch size search, VRAM monitoring.
- High-quality PDF plots saved to .research/iteration3/images.

Run context:
- Orchestrated from src.main (python -m src.main)
- All imports within src use relative paths.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional imports
try:
    import pynvml
    _HAVE_NVML = True
except Exception:
    _HAVE_NVML = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib is required for plotting.") from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for logging and analysis.") from e

try:
    from scipy import stats as spstats
except Exception as e:
    raise RuntimeError("scipy is required for statistical tests.") from e

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

try:
    import xformers
    import xformers.ops as xops
    _HAVE_XFORMERS = True
except Exception:
    _HAVE_XFORMERS = False

from .preprocess import TokenSequenceDataset

# ----------------------
# Globals
# ----------------------
DEFAULT_SEEDS = [11, 13]
BACKENDS = ["math", "mem_efficient", "xformers", "eafa"]
POWER_SAMPLING_HZ = 25.0
PLOT_STYLE = {
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


# ----------------------
# Power sampling via NVML
# ----------------------
@dataclass
class PowerSample:
    t: float
    watts: float


class PowerSampler:
    def __init__(self, device_index: int = 0, hz: float = POWER_SAMPLING_HZ):
        self.device_index = device_index
        self.hz = max(5.0, min(50.0, hz))
        self._device = None
        self._stop = False
        self._thread = None
        self.samples: List[PowerSample] = []
        self._init_nvml()

    def _init_nvml(self):
        if not _HAVE_NVML:
            logging.warning("pynvml not available; power sampling disabled.")
            return
        try:
            pynvml.nvmlInit()
            self._device = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as e:
            logging.warning("Failed to init NVML: %s", e)
            self._device = None

    def _read_power(self) -> Optional[float]:
        if self._device is None:
            return None
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self._device)
            return mw / 1000.0
        except Exception:
            return None

    def start(self):
        if self._device is None:
            return
        import threading, time
        self.samples.clear()
        self._stop = False
        def _worker():
            dt = 1.0 / self.hz
            while not self._stop:
                t = time.perf_counter()
                p = self._read_power()
                if p is not None:
                    self.samples.append(PowerSample(t, p))
                time.sleep(dt)
        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self):
        if self._device is None or self._thread is None:
            return
        self._stop = True
        self._thread.join(timeout=5.0)

    def energy_joules(self, idle_watts: float = 0.0) -> float:
        if not self.samples:
            return 0.0
        ts = np.array([s.t for s in self.samples])
        ps = np.array([s.watts for s in self.samples])
        ps = np.maximum(ps - idle_watts, 0.0)
        return float(np.trapz(ps, ts))

    def mean_power(self, idle_watts: float = 0.0) -> float:
        if not self.samples:
            return 0.0
        ps = np.array([s.watts for s in self.samples])
        ps = np.maximum(ps - idle_watts, 0.0)
        return float(ps.mean())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"t": [s.t for s in self.samples], "power_W": [s.watts for s in self.samples]})


def measure_idle_power(device_index: int, seconds: float = 10.0) -> float:
    if not _HAVE_NVML:
        return 0.0
    sampler = PowerSampler(device_index, hz=POWER_SAMPLING_HZ)
    sampler.start()
    dev = torch.device("cuda", device_index)
    x = torch.empty((1,), device=dev)
    torch.cuda.synchronize(device_index)
    import time as _time
    _time.sleep(seconds)
    sampler.stop()
    df = sampler.to_dataframe()
    return float(df["power_W"].mean()) if len(df) else 0.0


# ----------------------
# RPA DVFS controller (coarse heuristic)
# ----------------------
@dataclass
class DVFSState:
    requested_mem_mhz: Optional[int] = None
    requested_sm_mhz: Optional[int] = None
    granted: bool = False
    permission_denied: bool = False


class RPAController:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._device = None
        if _HAVE_NVML:
            try:
                pynvml.nvmlInit()
                self._device = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self._device = None
        self.state = DVFSState()

    def query_default_clocks(self) -> Tuple[Optional[int], Optional[int]]:
        if self._device is None:
            return (None, None)
        try:
            mem = pynvml.nvmlDeviceGetClockInfo(self._device, pynvml.NVML_CLOCK_MEM)
            sm = pynvml.nvmlDeviceGetClockInfo(self._device, pynvml.NVML_CLOCK_SM)
            return int(mem), int(sm)
        except Exception:
            return (None, None)

    def set_clocks(self, sm_mhz: int, mem_mhz: Optional[int] = None) -> DVFSState:
        st = DVFSState(requested_sm_mhz=sm_mhz, requested_mem_mhz=mem_mhz, granted=False)
        if self._device is None:
            st.permission_denied = True
            self.state = st
            return st
        try:
            if mem_mhz is None:
                current_mem = pynvml.nvmlDeviceGetClockInfo(self._device, pynvml.NVML_CLOCK_MEM)
                mem_mhz = int(current_mem)
            pynvml.nvmlDeviceSetApplicationsClocks(self._device, mem_mhz, sm_mhz)
            st.granted = True
        except Exception:
            st.permission_denied = True
        self.state = st
        return st

    def reset_clocks(self):
        if self._device is None:
            return
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(self._device)
        except Exception:
            pass

    def decide_and_apply(self, seq_len: int) -> DVFSState:
        try:
            _, sm = self.query_default_clocks()
            if sm is None:
                return self.state
            target = sm
            if seq_len <= 512:
                target = max(int(0.7 * sm), sm - 300)
            elif seq_len <= 1024:
                target = max(int(0.85 * sm), sm - 150)
            if target != sm:
                return self.set_clocks(sm_mhz=target)
            else:
                self.state = DVFSState(requested_sm_mhz=sm, requested_mem_mhz=None, granted=True)
                return self.state
        except Exception:
            return self.state


# ----------------------
# SDPA wrapper for GPT-2
# ----------------------
class SDPAWrapper(nn.Module):
    def __init__(self, attn: GPT2Attention, backend: str = "mem_efficient", eafa_ctrl: Optional["EAFAController"] = None):
        super().__init__()
        assert backend in BACKENDS
        self.c_attn = attn.c_attn
        self.c_proj = attn.c_proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.backend = backend
        self.eafa_ctrl = eafa_ctrl

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq, hidden = hidden_states.size()
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(hidden, dim=2)
        def shape(x):
            return x.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        q = shape(q)
        k = shape(k)
        v = shape(v)

        backend = self.backend
        autocast_dtype = torch.float16
        use_sdp_math = False
        use_sdp_mem = True
        use_xf = False

        if backend == "eafa" and self.eafa_ctrl is not None:
            decision = self.eafa_ctrl.choose_policy(bsz=bsz, seq=seq, n_heads=self.num_heads)
            backend = decision["backend"]
            autocast_dtype = decision["dtype"]
            use_sdp_math = decision.get("sdp_math", False)
            use_sdp_mem = decision.get("sdp_mem_eff", backend == "mem_efficient")
            use_xf = (backend == "xformers")
        elif backend == "math":
            use_sdp_math, use_sdp_mem, use_xf = True, False, False
        elif backend == "mem_efficient":
            use_sdp_math, use_sdp_mem, use_xf = False, True, False
        elif backend == "xformers":
            use_xf = True

        is_causal = True
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=use_sdp_math, enable_mem_efficient=use_sdp_mem):
            if use_xf:
                if not _HAVE_XFORMERS:
                    raise RuntimeError("xFormers requested but not installed.")
                qx = q.transpose(1, 2).reshape(bsz * self.num_heads, seq, self.head_dim)
                kx = k.transpose(1, 2).reshape(bsz * self.num_heads, seq, self.head_dim)
                vx = v.transpose(1, 2).reshape(bsz * self.num_heads, seq, self.head_dim)
                attn_bias = xops.LowerTriangularMask() if is_causal else None
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = xops.memory_efficient_attention(qx, kx, vx, attn_bias=attn_bias, p=0.0)
                out = out.view(bsz, self.num_heads, seq, self.head_dim)
            else:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(bsz, seq, hidden)
        out = self.c_proj(out)
        return out


def _get_parent(root: nn.Module, path: str) -> nn.Module:
    p = root
    parts = path.split(".")
    for s in parts[:-1]:
        p = getattr(p, s)
    return p


def patch_gpt2_attention(model: nn.Module, backend: str, eafa_ctrl: Optional["EAFAController"]) -> int:
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention):
            parent = _get_parent(model, name)
            child_name = name.split(".")[-1]
            setattr(parent, child_name, SDPAWrapper(module, backend=backend, eafa_ctrl=eafa_ctrl))
            replaced += 1
    logging.info("Patched %d GPT2Attention modules (backend=%s)", replaced, backend)
    return replaced


# ----------------------
# EA-FA policy (controller only, no custom kernels)
# ----------------------
class EAFAController:
    def __init__(self, device_index: int = 0, enable_rpa: bool = True, enable_mpts: bool = True):
        self.device_index = device_index
        self.enable_rpa = enable_rpa
        self.enable_mpts = enable_mpts
        self.rpa = RPAController(device_index=device_index)
        self.last_policy: Dict[str, Any] = {}

    def choose_policy(self, bsz: int, seq: int, n_heads: int) -> Dict[str, Any]:
        notes = []
        if self.enable_rpa:
            st = self.rpa.decide_and_apply(seq)
            if st.permission_denied:
                notes.append("DVFS-denied")
        else:
            notes.append("RPA-off")
        backend = "mem_efficient"
        if _HAVE_XFORMERS and seq >= 1024:
            backend = "xformers"
            notes.append("xformers")
        dtype = torch.float16
        if self.enable_mpts:
            env_force = os.environ.get("EAFAPrecision", "").lower()
            if env_force in ("fp32", "float32"):
                dtype = torch.float32
            elif env_force in ("fp16", "float16"):
                dtype = torch.float16
            else:
                dtype = torch.float32 if seq <= 512 else torch.float16
            notes.append(f"MPTS:{'fp32' if dtype==torch.float32 else 'fp16'}")
        self.last_policy = {
            "backend": backend,
            "dtype": dtype,
            "notes": ",".join(notes),
            "sdp_math": backend == "math",
            "sdp_mem_eff": backend == "mem_efficient",
        }
        return self.last_policy

    def reset(self):
        self.rpa.reset_clocks()


# ----------------------
# Data utilities
# ----------------------
@dataclass
class DataConfig:
    seq_len: int
    tokens_npy: str
    tokens_budget: int = 16000


def build_dataloader_from_tokens(tokens_npy: str, seq_len: int, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, int]:
    toks = np.load(tokens_npy)
    ds = TokenSequenceDataset(toks, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader, len(ds)


# ----------------------
# Model manager
# ----------------------
@dataclass
class ModelConfig:
    model_name: str = "gpt2"
    device_index: int = 0


class ModelManager:
    def __init__(self, cfg: ModelConfig):
        if not _HAVE_HF:
            raise RuntimeError("transformers is required.")
        self.cfg = cfg
        self.device = torch.device("cuda", cfg.device_index) if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA GPU required for this experiment.")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: Optional[nn.Module] = None

    def load_model(self, name: Optional[str] = None) -> nn.Module:
        model_name = name or self.cfg.model_name
        logging.info("Loading model %s ...", model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval().to(self.device)
        self.model = model
        return model

    def patch_attention(self, backend: str, eafa_ctrl: Optional[EAFAController]):
        assert self.model is not None
        patch_gpt2_attention(self.model, backend=backend, eafa_ctrl=eafa_ctrl)


# ----------------------
# Metrics & plotting
# ----------------------
@dataclass
class BatchMetrics:
    seed: int
    backend: str
    seq_len: int
    batch_id: int
    batch_size: int
    tokens: int
    time_s: float
    tokens_per_s: float
    energy_j: float
    energy_per_1k_tokens: float
    mean_power_w: float
    vram_peak_gb: float
    loss: float
    ppl: float


def compute_loss_and_ppl(model: nn.Module, input_ids: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        loss = float(out.loss.detach().cpu())
        ppl = float(math.exp(min(20.0, loss)))
        return loss, ppl


def verify_sdpa_invoked(model: nn.Module, device: torch.device) -> bool:
    try:
        x = torch.randint(0, 1000, (1, 32), device=device)
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            _ = model(input_ids=x, labels=x)
        txt = str(prof.key_averages())
        ok = ("scaled_dot_product_attention" in txt) or ("memory_efficient_attention" in txt)
        logging.info("Profiler SDPA present: %s", ok)
        return ok
    except Exception:
        return False


def dynamic_batch_size_search(model_mgr: ModelManager, tokens_npy: str, seq_len: int, target_tokens: int = 16000) -> int:
    device = model_mgr.device
    model = model_mgr.model
    assert model is not None
    torch.cuda.reset_peak_memory_stats(device)
    bs = max(1, target_tokens // seq_len)
    logging.info("Dynamic batch search start bs=%d (L=%d)", bs, seq_len)
    while bs >= 1:
        try:
            loader, _ = build_dataloader_from_tokens(tokens_npy, seq_len=seq_len, batch_size=bs)
            it = iter(loader)
            batch = next(it)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(input_ids=input_ids, labels=input_ids)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            logging.info("Trial bs=%d peak VRAM=%.2f GB", bs, peak)
            if peak < 15.5:
                return bs
            else:
                bs = max(1, int(bs * 0.9))
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                bs = max(1, int(bs * 0.9))
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            torch.cuda.empty_cache()
    return 1


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> Tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(1234)
    n = len(x)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = np.mean(x[idx])
    lo = float(np.quantile(samples, (1 - ci) / 2))
    hi = float(np.quantile(samples, 1 - (1 - ci) / 2))
    return lo, hi


def save_plots(df: pd.DataFrame, images_dir: Path, label: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)
    plots = [
        ("tokens_per_s", "Throughput (tokens/s)"),
        ("energy_per_1k_tokens", "Energy (J/1k tokens)"),
        ("mean_power_w", "Mean Power (W)"),
    ]
    for metric, title in plots:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        sub = df.groupby(["backend", "seq_len"])  # mean and CI across seeds/batches
        means = sub[metric].mean()
        cis = sub[metric].apply(lambda x: pd.Series(bootstrap_ci(x.to_numpy())))
        cis.columns = ["ci_lo", "ci_hi"]
        res = pd.concat([means, cis], axis=1).reset_index()
        backends = sorted(res["backend"].unique())
        seqs = sorted(res["seq_len"].unique())
        colors = plt.cm.Set2.colors
        width = 0.18
        for i, backend in enumerate(backends):
            vals, los, his = [], [], []
            for s in seqs:
                row = res[(res.backend == backend) & (res.seq_len == s)]
                if len(row) == 0:
                    vals.append(np.nan); los.append(0.0); his.append(0.0)
                else:
                    m = float(row[metric])
                    ci_lo = float(row["ci_lo"])
                    ci_hi = float(row["ci_hi"])
                    vals.append(m)
                    los.append(m - ci_lo)
                    his.append(ci_hi - m)
            xs = np.arange(len(seqs)) + i * width
            ax.bar(xs, vals, width=width, color=colors[i % len(colors)], label=backend)
            ax.errorbar(xs, vals, yerr=[los, his], fmt='none', ecolor='black', elinewidth=1, capsize=2)
        ax.set_xticks(np.arange(len(seqs)) + (len(backends) - 1) * width / 2)
        ax.set_xticklabels([str(s) for s in seqs])
        ax.set_ylabel(title)
        ax.set_xlabel("Sequence length L")
        ax.set_title(f"{title} by backend ({label})")
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = images_dir / f"{metric}_{label}.pdf"
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved plot %s", out_path)


# ----------------------
# Inference loop per backend
# ----------------------

def run_inference_epoch(
    model_mgr: ModelManager,
    tokens_npy: str,
    seq_len: int,
    backend: str,
    seed: int,
    max_batches: int,
    idle_power_w: float,
    eafa_ctrl: Optional[EAFAController],
    tokens_budget: int = 16000,
) -> Tuple[List[BatchMetrics], pd.DataFrame]:
    device = model_mgr.device
    model = model_mgr.model
    assert model is not None
    bs = dynamic_batch_size_search(model_mgr, tokens_npy, seq_len, target_tokens=tokens_budget)
    loader, _ = build_dataloader_from_tokens(tokens_npy, seq_len=seq_len, batch_size=bs)

    # Patch attention
    model_mgr.patch_attention(backend=backend, eafa_ctrl=eafa_ctrl if backend == "eafa" else None)

    sampler = PowerSampler(device_index=model_mgr.cfg.device_index, hz=POWER_SAMPLING_HZ)
    metrics: List[BatchMetrics] = []
    power_df_rows: List[pd.DataFrame] = []

    torch.cuda.reset_peak_memory_stats(device)
    batches_run = 0
    for batch_id, batch in enumerate(loader):
        if batches_run >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        tokens = int(input_ids.numel())
        torch.cuda.synchronize(device)
        sampler.start()
        t0 = time.perf_counter()
        loss, ppl = compute_loss_and_ppl(model, input_ids)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        sampler.stop()
        dt = t1 - t0
        tps = tokens / dt if dt > 0 else float("nan")
        e_j = sampler.energy_joules(idle_watts=idle_power_w)
        mean_p = sampler.mean_power(idle_watts=idle_power_w)
        peak_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        bm = BatchMetrics(
            seed=seed,
            backend=backend,
            seq_len=seq_len,
            batch_id=batch_id,
            batch_size=bs,
            tokens=tokens,
            time_s=dt,
            tokens_per_s=tps,
            energy_j=e_j,
            energy_per_1k_tokens=(e_j / (tokens / 1000.0)) if tokens > 0 else float("nan"),
            mean_power_w=mean_p,
            vram_peak_gb=peak_gb,
            loss=loss,
            ppl=ppl,
        )
        metrics.append(bm)
        df_trace = sampler.to_dataframe()
        if len(df_trace) > 0:
            df_trace["seed"] = seed
            df_trace["backend"] = backend
            df_trace["seq_len"] = seq_len
            df_trace["batch_id"] = batch_id
            power_df_rows.append(df_trace)
        batches_run += 1
        del input_ids
        torch.cuda.empty_cache()

    power_df = pd.concat(power_df_rows, ignore_index=True) if len(power_df_rows) else pd.DataFrame()
    # Reset DVFS if needed
    if eafa_ctrl is not None:
        eafa_ctrl.reset()
    return metrics, power_df


# ----------------------
# Orchestrator
# ----------------------
@dataclass
class RunConfig:
    seeds: List[int]
    seq_lens: List[int]
    backends: List[str]
    max_batches: int
    tokens_budget: int
    output_dir: str
    images_dir: str


@dataclass
class ExperimentConfig:
    model_cfg: ModelConfig
    run_cfg: RunConfig
    tokens_npy: str
    idle_seconds: float = 10.0


def run_experiments(cfg: ExperimentConfig) -> Dict[str, Any]:
    out_root = Path(cfg.run_cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    images_dir = Path(cfg.run_cfg.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    model_mgr = ModelManager(cfg.model_cfg)
    try:
        model_mgr.load_model(cfg.model_cfg.model_name)
    except RuntimeError as e:
        logging.error("Model load failed: %s", e)
        if cfg.model_cfg.model_name != "gpt2":
            logging.info("Falling back to 'gpt2'")
            model_mgr.load_model("gpt2")
        else:
            raise

    all_rows: List[Dict[str, Any]] = []
    all_traces: List[pd.DataFrame] = []

    # Idle baseline once per experiment (per seq_len we can reuse or re-measure; keep single for speed)
    idle_w = measure_idle_power(model_mgr.cfg.device_index, seconds=cfg.idle_seconds)
    logging.info("Measured idle power: %.2f W", idle_w)

    # Verify SDPA path in a pilot
    _ = verify_sdpa_invoked(model_mgr.model, device=model_mgr.device)

    # Filter available backends
    backends = []
    for b in cfg.run_cfg.backends:
        if b == "xformers" and not _HAVE_XFORMERS:
            logging.info("Skipping xFormers backend (not installed)")
            continue
        backends.append(b)

    for L in cfg.run_cfg.seq_lens:
        for seed in cfg.run_cfg.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            order = backends if (seed % 2 == 0) else list(reversed(backends))
            for backend in order:
                eafa_ctrl = EAFAController(device_index=model_mgr.cfg.device_index, enable_rpa=True, enable_mpts=True) if backend == "eafa" else None
                try:
                    metrics, power_df = run_inference_epoch(
                        model_mgr=model_mgr,
                        tokens_npy=cfg.tokens_npy,
                        seq_len=L,
                        backend=backend,
                        seed=seed,
                        max_batches=cfg.run_cfg.max_batches,
                        idle_power_w=idle_w,
                        eafa_ctrl=eafa_ctrl,
                        tokens_budget=cfg.run_cfg.tokens_budget,
                    )
                    for m in metrics:
                        all_rows.append(dataclasses.asdict(m))
                    if len(power_df):
                        all_traces.append(power_df)
                except RuntimeError as e:
                    logging.error("Run failed for backend=%s L=%d seed=%d: %s", backend, L, seed, e)
                finally:
                    torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_rows)
    if len(results_df) == 0:
        logging.warning("No results collected.")
        return {"results_csv": None}

    # Save artifacts
    results_csv = out_root / "metrics.csv"
    results_df.to_csv(results_csv, index=False)
    if len(all_traces):
        pd.concat(all_traces, ignore_index=True).to_csv(out_root / "power_traces.csv", index=False)

    # Plots
    save_plots(results_df, images_dir, label="overall")

    # Summary JSON
    summary = {
        "config": {
            "model_cfg": dataclasses.asdict(cfg.model_cfg),
            "run_cfg": dataclasses.asdict(cfg.run_cfg),
            "tokens_npy": cfg.tokens_npy,
        },
        "n_rows": len(results_df),
        "idle_watts": float(idle_w),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved results to %s and plots to %s", results_csv, images_dir)
    return {"results_csv": str(results_csv), "images_dir": str(images_dir)}


__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "RunConfig",
    "run_experiments",
]
