#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for EA-FA experiments.
Run from project root with: python -m src.main

Pipeline
1) Preprocess/build dataloaders
2) Build model (EA-FA or baseline)
3) Train (optional) and/or Evaluate (energy + PPL)
4) Save plots (PDF) to .research/iteration1/images

All standard output prints detailed results.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from .preprocess import set_seed, build_dataloaders
from .train import (
    TrainConfig,
    build_model,
    find_max_batch_size,
    train_one_epoch,
    PowerSampler,
)
from .evaluate import evaluate_perplexity, plot_energy_bar

# Directories
ROOT = Path(os.getcwd()).resolve()
IMAGES_DIR = ROOT / ".research" / "iteration1" / "images"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
for d in [IMAGES_DIR, MODEL_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


@dataclass
class RunConfig:
    exp_id: str = "iteration1"
    seeds: List[int] = None  # type: ignore
    modes: List[str] = None  # type: ignore  # ["inference", "training"]
    dataset_name: str = "synthetic"
    dataset_split: str = "wikitext-103-v1"
    tokenizer_name: str = "gpt2"
    seq_lens: List[int] = None  # type: ignore
    global_tokens_per_iter: int = 16384
    max_train_steps: int = 50
    measure_warmup_iters: int = 10
    measure_iters: int = 10
    vocab_size: int = 50257
    # EA-FA toggles
    backends: List[str] = None  # type: ignore
    power_limits: List[int] = None  # type: ignore
    enable_rpa: bool = True
    enable_mpts: bool = True
    enable_skp: bool = True
    entropy_tau: float = 1.5
    skp_tile: int = 64
    skp_eps: float = 1e-4
    # model
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    amp_dtype_name: str = "float16"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    device_index: int = 0

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42]
        if self.modes is None:
            self.modes = ["inference"]
        if self.seq_lens is None:
            self.seq_lens = [256, 512]
        if self.backends is None:
            self.backends = ["pytorch_mem_efficient", "pytorch_math", "xformers"]
        if self.power_limits is None:
            self.power_limits = [50, 60, 70]


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_run_config(cfg_path: Path | None) -> RunConfig:
    if cfg_path is None or not cfg_path.exists():
        return RunConfig()
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    rc = RunConfig(**raw)
    return rc


def build_train_cfg(run: RunConfig) -> TrainConfig:
    amp_dtype = DTYPE_MAP.get(run.amp_dtype_name, torch.float16)
    tcfg = TrainConfig(
        vocab_size=run.vocab_size,
        d_model=run.d_model,
        n_layers=run.n_layers,
        n_heads=run.n_heads,
        mlp_ratio=run.mlp_ratio,
        dropout=run.dropout,
        amp_dtype=amp_dtype,
        lr=run.lr,
        betas=run.betas,
        eps=run.eps,
        weight_decay=run.weight_decay,
        global_tokens_per_iter=run.global_tokens_per_iter,
        device_index=run.device_index,
        backends=run.backends,
        power_limits=run.power_limits,
        enable_rpa=run.enable_rpa,
        enable_mpts=run.enable_mpts,
        enable_skp=run.enable_skp,
        entropy_tau=run.entropy_tau,
        skp_tile=run.skp_tile,
        skp_eps=run.skp_eps,
    )
    return tcfg


def run_condition(seed: int, run: RunConfig, seq_len: int, method_label: str) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(f"cuda:{run.device_index}" if torch.cuda.is_available() else "cpu")
    # Build dataloader with base batch size, adjust later
    base_bs = max(1, run.global_tokens_per_iter // max(1, seq_len))
    dl, _ = build_dataloaders(run.dataset_name, run.dataset_split, run.tokenizer_name, seq_len, base_bs, num_workers=2)

    # Build model
    tcfg = build_train_cfg(run)
    model, attn_impl = build_model(tcfg, device)

    # Override for baseline (disable EA-FA and force backend)
    if method_label.startswith("baseline:"):
        backend_name = method_label.split(":", 1)[1]
        from .train import SDPAWrapper  # local import to avoid circular
        for blk in model.blocks:
            blk.attn_impl = lambda q, k, v, attn_mask=None, is_causal=True: (
                SDPAWrapper(backend_name)(q, k, v, attn_mask=attn_mask, is_causal=is_causal),
                {"method": backend_name, "power_W": None, "mpts_routed_fp32_rows": 0, "total_rows": int(q.shape[0]*q.shape[2]), "skp_active": False}
            )

    # Dynamic batch sizing
    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.amp_dtype in (torch.float16, torch.bfloat16))) if hasattr(torch.cuda, 'amp') else None
    if scaler is None:
        from .train import GradScaler as _GS
        scaler = _GS()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=tcfg.betas, eps=tcfg.eps, weight_decay=tcfg.weight_decay)

    try:
        bs_fit, ga = find_max_batch_size(model, seq_len, base_bs, run.global_tokens_per_iter, device, run.vocab_size, tcfg.amp_dtype)
    except Exception:
        bs_fit, ga = base_bs, 1
    # Rebuild dataloader with fitted batch size
    dl, _ = build_dataloaders(run.dataset_name, run.dataset_split, run.tokenizer_name, seq_len, bs_fit, num_workers=2)

    # Measure idle power once
    power = PowerSampler(device_index=run.device_index)
    if power.idle_power_watts is None:
        try:
            power.measure_idle(duration_s=5.0)
        except Exception:
            pass

    # Warm-up iterations
    it = iter(dl)
    for _ in range(max(0, run.measure_warmup_iters)):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True, dtype=tcfg.amp_dtype):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)
        del x, y, logits, loss
        torch.cuda.empty_cache()

    result: Dict[str, Any] = {"method": method_label, "seq_len": seq_len, "seed": seed}

    # Run modes
    if "training" in run.modes:
        train_metrics = train_one_epoch(model, optimizer, scaler, dl, device, steps=min(20, run.max_train_steps), grad_accum=ga, power=power, amp_dtype=tcfg.amp_dtype)
        tokens_k = train_metrics["tokens"] / 1000.0 if train_metrics["tokens"]>0 else 1.0
        result.update({
            "train_tokens": train_metrics["tokens"],
            "train_time_s": train_metrics["time_s"],
            "train_joules": train_metrics["joules"],
            "train_j_per_1k": train_metrics["joules"] / max(1e-9, tokens_k),
            "train_loss": train_metrics["loss_avg"],
        })
    if "inference" in run.modes:
        t_start = time.time()
        eval_metrics = evaluate_perplexity(model, dl, device, steps=run.measure_iters, power=power, amp_dtype=tcfg.amp_dtype)
        eval_time = time.time() - t_start
        tokens_k = eval_metrics["tokens"] / 1000.0 if eval_metrics["tokens"]>0 else 1.0
        result.update({
            "eval_tokens": eval_metrics["tokens"],
            "eval_time_s": eval_time,
            "eval_joules": eval_metrics["joules"],
            "eval_j_per_1k": eval_metrics["joules"] / max(1e-9, tokens_k),
            "eval_ppl": eval_metrics["ppl"],
        })

    # Save model checkpoint (optional minimal)
    ckpt_path = MODEL_DIR / f"{run.exp_id}_{method_label.replace(':','_')}_L{seq_len}_seed{seed}.pt"
    try:
        torch.save({"state_dict": model.state_dict(), "config": run.__dict__}, ckpt_path)
    except Exception:
        pass

    # Free GPU
    del model
    torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="EA-FA Experiments")
    parser.add_argument("--config", type=str, default=str((ROOT / "config" / "config.yaml")), help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Run a very quick smoke test")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    run = load_run_config(cfg_path if cfg_path.exists() else None)
    if args.smoke:
        run.seeds = [13]
        run.seq_lens = [128]
        run.measure_iters = 5
        run.measure_warmup_iters = 3
        run.max_train_steps = 5
        run.modes = ["inference"]

    print(f"Git commit: {get_git_commit()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    methods: List[str] = ["baseline:pytorch_mem_efficient", "EA-FA"]

    all_results: List[Dict[str, Any]] = []
    for seed in run.seeds:
        for L in run.seq_lens:
            for method in methods:
                print(f"Running seed={seed} L={L} method={method} modes={run.modes}")
                res = run_condition(seed, run, L, method)
                print({k: v for k, v in res.items() if k not in {}})
                all_results.append(res)

    # Aggregate and plot energy for inference
    ea_vals: List[float] = []
    base_vals: List[float] = []
    for r in all_results:
        if "eval_j_per_1k" not in r:
            continue
        if r["method"] == "EA-FA":
            ea_vals.append(float(r["eval_j_per_1k"]))
        elif r["method"].startswith("baseline:"):
            base_vals.append(float(r["eval_j_per_1k"]))

    pdf_path = IMAGES_DIR / "energy_eafa_vs_baseline.pdf"
    plot_energy_bar(["Baseline", "EA-FA"], [base_vals, ea_vals], pdf_path)
    print(f"Saved energy plot to {pdf_path}")

    # Save CSV summary under data/
    import csv
    csv_path = DATA_DIR / f"summary_{run.exp_id}.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = sorted({k for r in all_results for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
