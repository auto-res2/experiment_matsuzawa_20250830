# -*- coding: utf-8 -*-
"""
Evaluation utilities for LoRA-Prop (+ Time-Warp) in a lightweight toy setup.
- Compares cache-only vs our LoRA-Prop adapters vs full backbone.
- Produces high-quality PDF plots saved under .research/iteration11/images.
- Includes throughput measurements on the toy backbone.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import ToyBackbone, AdapterBank, load_adapters_toy, ensure_dir, set_seed
from .preprocess import compute_deltaF_heatmap, greedy_timewarp_key_selection


@dataclass
class EvalConfig:
    total_steps: int = 10
    seeds: Tuple[int, ...] = (0, 1)
    image_shape: Tuple[int, int] = (64, 64)
    device: str = "cpu"
    images_out_dir: str = ".research/iteration11/images"
    models_ckpt_path: str = "models/loraprop_toy.pt"
    rank: int = 8


def _plot_pdf_line(x, y, title, xlabel, ylabel, out_pdf):
    plt.figure(figsize=(5,3))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def _plot_pdf_bar(labels, values, title, ylabel, out_pdf):
    plt.figure(figsize=(6,3.5))
    sns.barplot(x=labels, y=values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def _plot_pdf_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, out_pdf: str, key_steps: List[int]):
    plt.figure(figsize=(7,4))
    sns.heatmap(mat, cmap='magma')
    for ks in key_steps:
        plt.axvline(ks + 0.5, color='cyan', linestyle='--', linewidth=1.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def evaluate_toy(cfg: EvalConfig) -> Dict[str, float]:
    """Runs a small evaluation on the toy backbone to validate LoRA-Prop.
    Returns a dict of summary metrics (average MSE and throughput proxies).
    """
    ensure_dir(cfg.images_out_dir)
    device = torch.device(cfg.device)

    # 1) Key-step discovery (Dry ΔF heatmap on toy backbone)
    set_seed(123)
    B = 4
    H, W = cfg.image_shape
    toy = ToyBackbone().to(device).eval()
    x = torch.randn(B, 3, H, W, device=device)

    def get_feats(step_idx: int) -> Dict[str, torch.Tensor]:
        tnorm = torch.tensor([step_idx / max(1, cfg.total_steps-1)], device=device)
        with torch.no_grad():
            feats = toy(x, tnorm)
        # Move to CPU for ΔF computation
        return {k: v.detach().cpu() for k, v in feats.items()}

    deltaF, layer_ids = compute_deltaF_heatmap(get_feats, n_steps=cfg.total_steps)
    key_steps = greedy_timewarp_key_selection(deltaF, K=3, stratify_bins=4)
    heatmap_pdf = os.path.join(cfg.images_out_dir, "deltaF_heatmap_toy.pdf")
    _plot_pdf_heatmap(deltaF, "ΔF Heatmap (Toy)", "Step", "Layer", heatmap_pdf, key_steps)
    print(f"[Eval] Saved ΔF heatmap to {heatmap_pdf}; key steps: {key_steps}")

    # 2) Load adapters
    adapters = load_adapters_toy(cfg.models_ckpt_path, device=str(device))
    adapters.eval()

    # 3) Evaluate across seeds: compute MSE per step for cache-only vs ours
    mse_cache = []
    mse_ours = []
    runtimes = {"full": [], "cache": [], "ours": []}

    for seed in cfg.seeds:
        set_seed(seed)
        x = torch.randn(B, 3, H, W, device=device)
        with torch.no_grad():
            # Precompute full features per step (teacher)
            feats_by_t: Dict[int, Dict[str, torch.Tensor]] = {}
            t0 = time.time()
            for t_idx in range(cfg.total_steps):
                tnorm = torch.tensor([t_idx / max(1, cfg.total_steps-1)], device=device)
                feats_by_t[t_idx] = toy(x, tnorm)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            runtimes["full"].append(time.time() - t0)

            # Cache-only and ours predictions
            # For simplicity, reuse nearest previous key step per non-key step
            t0 = time.time()
            for t_idx in range(cfg.total_steps):
                if t_idx in key_steps:
                    continue
                prev_keys = [ks for ks in key_steps if ks < t_idx]
                if len(prev_keys) == 0:
                    continue
                t_prev = max(prev_keys)
                # cache-only: use f_prev as prediction
                errs = []
                for lid in ["down_0", "down_1"]:
                    f_prev = feats_by_t[t_prev][lid]
                    f_t = feats_by_t[t_idx][lid]
                    err = F.mse_loss(f_prev, f_t).item()
                    errs.append(err)
                mse_cache.append(float(np.mean(errs)))
            torch.cuda.synchronize() if device.type == 'cuda' else None
            runtimes["cache"].append(time.time() - t0)

            t0 = time.time()
            for t_idx in range(cfg.total_steps):
                if t_idx in key_steps:
                    continue
                prev_keys = [ks for ks in key_steps if ks < t_idx]
                if len(prev_keys) == 0:
                    continue
                t_prev = max(prev_keys)
                delta_t = float((t_idx - t_prev) / max(1, cfg.total_steps-1))
                delta_t_vec = torch.full((B,), delta_t, device=device)
                errs = []
                for lid in ["down_0", "down_1"]:
                    f_prev = feats_by_t[t_prev][lid]
                    pred = adapters(lid, f_prev, delta_t_vec)
                    f_t = feats_by_t[t_idx][lid]
                    err = F.mse_loss(pred, f_t).item()
                    errs.append(err)
                mse_ours.append(float(np.mean(errs)))
            torch.cuda.synchronize() if device.type == 'cuda' else None
            runtimes["ours"].append(time.time() - t0)

    # 4) Summaries and plots
    mse_cache_mean = float(np.mean(mse_cache)) if len(mse_cache) else float('nan')
    mse_ours_mean = float(np.mean(mse_ours)) if len(mse_ours) else float('nan')
    print(f"[Eval] Avg MSE - cache-only: {mse_cache_mean:.6f}, ours: {mse_ours_mean:.6f}")

    bar_pdf = os.path.join(cfg.images_out_dir, "mse_bar_cache_vs_ours_toy.pdf")
    _plot_pdf_bar(["cache-only", "LoRA-Prop"], [mse_cache_mean, mse_ours_mean], "Feature MSE (lower is better)", "MSE", bar_pdf)
    print(f"[Eval] Saved MSE bar plot to {bar_pdf}")

    thr_full = 1.0 / (np.mean(runtimes["full"]) + 1e-9)
    thr_cache = 1.0 / (np.mean(runtimes["cache"]) + 1e-9)
    thr_ours = 1.0 / (np.mean(runtimes["ours"]) + 1e-9)
    thr_pdf = os.path.join(cfg.images_out_dir, "throughput_bar_toy.pdf")
    _plot_pdf_bar(["full (teacher)", "cache-only", "LoRA-Prop"], [thr_full, thr_cache, thr_ours], "Throughput (1/sec, toy)", "it/s", thr_pdf)
    print(f"[Eval] Saved throughput plot to {thr_pdf}")

    return {
        "mse_cache_mean": mse_cache_mean,
        "mse_ours_mean": mse_ours_mean,
        "thr_full": float(thr_full),
        "thr_cache": float(thr_cache),
        "thr_ours": float(thr_ours),
        "key_steps": key_steps,
    }


def quick_test_eval(images_out_dir: str = ".research/iteration11/images"):
    """Minimal self-check to ensure evaluation code runs and saves PDFs."""
    ensure_dir(images_out_dir)
    cfg = EvalConfig(images_out_dir=images_out_dir)
    # Create a fake adapters checkpoint if it doesn't exist
    ckpt_path = cfg.models_ckpt_path
    if not os.path.exists(ckpt_path):
        from .train import TrainConfig, train_lora_adapters_toy
        print("[QuickEval] No adapters found. Training a tiny set for the test…")
        train_lora_adapters_toy(TrainConfig(images_out_dir=images_out_dir, models_out_dir=os.path.dirname(ckpt_path)))
    res = evaluate_toy(cfg)
    print(f"[QuickEval] Done. Summary: {res}")
