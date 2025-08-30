import os
import math
import time
import gc
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import (
    AttentionBackend,
    EnergyAwareAttention,
    replace_model_attentions_with_energy_aware,
    PowerSampler,
)
from .preprocess import ensure_dir

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


@dataclass
class EvalConfig:
    experiment_name: str = "eafa_t4"
    model_name: str = "gpt2"
    seq_len: int = 512
    eval_batch_size: int = 4
    warmup_batches: int = 5
    measure_batches: int = 20
    data_dir: str = "data"
    model_dir: str = "models"
    device: str = "cuda"
    gpu_index: int = 0
    enable_mpts: bool = True
    enable_skp: bool = True


def _auto_tokenizer(model_name: str):
    if not HF_AVAILABLE:
        raise RuntimeError("transformers not available. Please install requirements.")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok


def _make_loader(toks: torch.Tensor, bs: int):
    N = toks.size(0)
    for i in range(0, N, bs):
        if i + bs > N:
            break
        yield toks[i:i+bs]


def _binary_search_bs(model, seq_len: int, target_tokens: int, max_reserved_gb: float = 15.5) -> int:
    if not torch.cuda.is_available():
        return max(1, target_tokens // seq_len)
    tok = _auto_tokenizer("gpt2")
    B_max = max(1, target_tokens // seq_len)
    lo, hi, best = 1, B_max, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            with torch.no_grad():
                x = torch.full((mid, seq_len), tok.eos_token_id, dtype=torch.long, device="cuda")
                _ = model(input_ids=x)
                torch.cuda.synchronize()
            mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
            if mem_gb <= max_reserved_gb:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                hi = mid - 1
            else:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return best


def _eval_backend(model, tokenizer, data_tensor: torch.Tensor, backend_id: int, cfg: EvalConfig) -> Dict[str, Any]:
    # Switch all EA modules to desired backend
    for m in model.modules():
        if isinstance(m, EnergyAwareAttention):
            m.switch_backend(backend_id)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    loader = _make_loader(data_tensor, cfg.eval_batch_size)

    # Warmup
    with torch.no_grad():
        wcnt = 0
        for batch in loader:
            batch = batch.to(device)
            labels = batch.clone()
            labels[:, :-1] = batch[:, 1:]
            labels[:, -1] = tokenizer.eos_token_id
            _ = model(input_ids=batch, labels=labels)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            wcnt += 1
            if wcnt >= cfg.warmup_batches:
                break

    # Measure
    loader = _make_loader(data_tensor, cfg.eval_batch_size)
    n_batches = 0
    tot_loss = 0.0
    tot_tokens = 0
    tok_s_list: List[float] = []
    energy_list: List[float] = []

    for batch in loader:
        if n_batches >= cfg.measure_batches:
            break
        batch = batch.to(device)
        labels = batch.clone()
        labels[:, :-1] = batch[:, 1:]
        labels[:, -1] = tokenizer.eos_token_id

        ps = PowerSampler(gpu_index=cfg.gpu_index, interval_s=0.02)
        ps.start()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=batch, labels=labels)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        ps.stop()

        loss = float(out.loss.detach().cpu())
        nll = loss * (batch.numel() - batch.size(0))
        tot_loss += nll
        tot_tokens += (batch.numel() - batch.size(0))
        tok_s_list.append(float(batch.numel() / max(1e-6, (t1 - t0))))
        energy_list.append(ps.total_energy_j())

        n_batches += 1
        del out, batch, labels
        gc.collect()
        torch.cuda.empty_cache()

    mean_loss = tot_loss / max(1, tot_tokens)
    ppl = math.exp(min(30.0, max(-30.0, mean_loss)))
    return {
        "backend": AttentionBackend.NAMES[backend_id],
        "ppl": ppl,
        "mean_tokens_per_s": float(np.mean(tok_s_list)) if len(tok_s_list) else float('nan'),
        "mean_energy_j": float(np.nanmean(energy_list)) if len(energy_list) else float('nan'),
        "energy_j_list": energy_list,
        "tps_list": tok_s_list,
    }


def evaluate(cfg: EvalConfig, val_data_path: str, model_path: str) -> Dict[str, Any]:
    ensure_dir(".research/iteration8/images")

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    tokenizer = _auto_tokenizer(cfg.model_name)

    # Load model (fine-tuned if available)
    if os.path.isdir(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    else:
        # Fallback to base model
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    model.to(device)
    model.eval()

    # Replace attentions with energy-aware versions
    replaced = replace_model_attentions_with_energy_aware(model, cfg.model_name,
                                                          enable_mpts=cfg.enable_mpts,
                                                          enable_skp=cfg.enable_skp,
                                                          default_backend=AttentionBackend.SDPA_MATH)
    print(f"[eval] Replaced {replaced} attention modules with EnergyAwareAttention")

    # Data
    data_tensor = torch.load(val_data_path)  # [N, L]

    # Auto BS if cfg.eval_batch_size <= 0
    bs = cfg.eval_batch_size
    if bs <= 0:
        bs = _binary_search_bs(model, cfg.seq_len, target_tokens=16000)
    cfg.eval_batch_size = bs
    print(f"[eval] Using eval_batch_size={bs}")

    backends = [AttentionBackend.EA_FA_T4, AttentionBackend.SDPA_MATH]
    try:
        import xformers  # noqa: F401
        backends.insert(1, AttentionBackend.XFORMERS_MEA)
    except Exception:
        pass

    results: List[Dict[str, Any]] = []
    for b in backends:
        print(f"[eval] Running backend={AttentionBackend.NAMES[b]}")
        r = _eval_backend(model, tokenizer, data_tensor, b, cfg)
        results.append(r)
        print(f"[eval] {r['backend']}: PPL={r['ppl']:.3f} | tokens/s={r['mean_tokens_per_s']:.1f} | energy/batch={r['mean_energy_j']:.3f} J")

    # Save plots
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(data=df, x="backend", y="mean_energy_j", ax=ax, capsize=0.2, color="#4472c4")
    ax.set_ylabel("Energy per batch (J)")
    ax.set_xlabel("")
    ax.set_title("Energy per batch by backend")
    fig.tight_layout()
    fig.savefig(".research/iteration8/images/energy_per_batch.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(data=df, x="backend", y="mean_tokens_per_s", ax=ax, capsize=0.2, color="#70ad47")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xlabel("")
    ax.set_title("Throughput by backend")
    fig.tight_layout()
    fig.savefig(".research/iteration8/images/throughput.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Save CSV of results
    df.to_csv(".research/iteration8/images/eval_summary.csv", index=False)
    return {r['backend']: r for r in results}
