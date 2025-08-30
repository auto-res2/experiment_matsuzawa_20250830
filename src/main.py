# -*- coding: utf-8 -*-
"""
Main entry for running the LoRA-Prop (+ Time-Warp) experimental pipeline.
- Orchestrates: preprocess (key-step discovery), train (LoRA adapters), and evaluate (toy experiments).
- All figures are saved as PDF under .research/iteration9/images.
- Run from project root: python -m src.main

This script is intentionally lightweight and performs a quick functional run.
"""

import os
import argparse
from typing import Any, Dict

import yaml

from .preprocess import timewarp_discover_toy
from .train import TrainConfig, train_lora_adapters_toy
from .evaluate import EvalConfig, evaluate_toy, quick_test_eval


DEFAULT_CONFIG_PATH = os.path.join("config", "experiment.yaml")


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        # default lightweight config
        return {
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            "total_steps": 10,
            "key_K": 3,
            "rank": 8,
            "train": {"epochs": 3, "batch_size": 16, "n_images": 32},
            "images_out_dir": ".research/iteration9/images",
            "models_out_dir": "models",
            "models_ckpt_name": "loraprop_toy.pt",
            "seeds": [0, 1]
        }
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LoRA-Prop (+ Time-Warp) Experimental Suite (Toy)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--quick", action="store_true", help="Run a quick self-test evaluation and exit")
    args = parser.parse_args()

    if args.quick:
        quick_test_eval()
        return

    cfg = load_config(args.config)
    images_out_dir = cfg.get("images_out_dir", ".research/iteration9/images")
    models_out_dir = cfg.get("models_out_dir", "models")
    device = cfg.get("device", "cpu")
    total_steps = int(cfg.get("total_steps", 10))
    K = int(cfg.get("key_K", 3))
    rank = int(cfg.get("rank", 8))
    seeds = tuple(cfg.get("seeds", [0, 1]))

    # 1) Preprocess: Time-Warp key-step discovery
    print("[Main] Discovering key-steps via Time-Warp (toy)…")
    key_steps = timewarp_discover_toy(n_steps=total_steps, K=K, images_out_dir=images_out_dir)
    print(f"[Main] Key steps selected: {key_steps}")

    # 2) Train: LoRA adapters on synthetic data
    print("[Main] Training LoRA-Prop adapters (toy)…")
    train_cfg = TrainConfig(
        rank=rank,
        batch_size=int(cfg.get("train", {}).get("batch_size", 16)),
        epochs=int(cfg.get("train", {}).get("epochs", 3)),
        lr=float(cfg.get("train", {}).get("lr", 1e-3)),
        n_images=int(cfg.get("train", {}).get("n_images", 32)),
        total_steps=total_steps,
        key_steps=tuple(key_steps),
        device=device,
        images_out_dir=images_out_dir,
        models_out_dir=models_out_dir,
        model_name=str(cfg.get("models_ckpt_name", "loraprop_toy.pt")),
    )
    ckpt_path = train_lora_adapters_toy(train_cfg)

    # 3) Evaluate: cache-only vs our adapters
    print("[Main] Evaluating adapters (toy)…")
    eval_cfg = EvalConfig(
        total_steps=total_steps,
        seeds=seeds,
        device=device,
        images_out_dir=images_out_dir,
        models_ckpt_path=ckpt_path,
        rank=rank,
    )
    summary = evaluate_toy(eval_cfg)
    print("[Main] Summary metrics:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
