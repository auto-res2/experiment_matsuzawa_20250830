from __future__ import annotations
import os
import sys
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

import torch

from .preprocess import (
    SimpleDiffusionTeacher,
    compute_deltaF_over_time,
    greedy_key_selection,
    save_deltaF_plot,
    set_seed,
)
from .train import train_lora_prop, TrainConfig
from .evaluate import eval_eps_prediction, onnx_export_light


@dataclass
class Config:
    seed: int
    device: str
    T_dry: int
    K_keys: int
    deltaF_batch: int
    train_iters: int
    train_lr: float
    train_batch: int
    rank: int
    results_dir: str
    images_dir: str
    model_save_path: str


def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return Config(
        seed=int(d.get('seed', 123)),
        device=d.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        T_dry=int(d.get('T_dry', 20)),
        K_keys=int(d.get('K_keys', 3)),
        deltaF_batch=int(d.get('deltaF_batch', 8)),
        train_iters=int(d.get('train_iters', 200)),
        train_lr=float(d.get('train_lr', 5e-4)),
        train_batch=int(d.get('train_batch', 8)),
        rank=int(d.get('rank', 8)),
        results_dir=d.get('results_dir', 'results'),
        images_dir=d.get('images_dir', '.research/iteration6/images'),
        model_save_path=d.get('model_save_path', 'models/loraprop_lightpath.pt'),
    )


def ensure_dirs(cfg: Config):
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)


def main():
    # Default config path
    config_path = os.environ.get('LORAPROP_CONFIG', 'config/config.yaml')
    cfg = load_config(config_path)
    ensure_dirs(cfg)

    print('=== LoRA-Prop (+ Time-Warp) synthetic experiment ===')
    print(f"Device: {cfg.device}")
    print(f"Config loaded from: {config_path}")

    set_seed(cfg.seed)

    # 1) Build teacher
    teacher = SimpleDiffusionTeacher(in_channels=4, base=32)

    # 2) Dry pass for ΔF and key-step discovery
    print('[Stage 1] Time-Warp (ΔF) dry pass ...')
    deltaF = compute_deltaF_over_time(teacher, T=cfg.T_dry, batch=cfg.deltaF_batch, device=cfg.device)
    keys_1based = greedy_key_selection(deltaF, K=cfg.K_keys, coverage=2)
    print(f"Selected key steps (1-based): {keys_1based}")
    # Save PDF figure
    deltaF_pdf = os.path.join(cfg.images_dir, 'deltaF_heatmap.pdf')
    save_deltaF_plot(deltaF, keys_1based, deltaF_pdf)

    # 3) Train LoRA-Prop light adapters
    print('[Stage 2] Training LoRA-Prop adapters (synthetic) ...')
    tcfg = TrainConfig(seed=cfg.seed, device=cfg.device, lr=cfg.train_lr, batch_size=cfg.train_batch,
                       iters=cfg.train_iters, rank=cfg.rank, save_path=cfg.model_save_path)
    light, train_info = train_lora_prop(keys_1based, T=cfg.T_dry, teacher=teacher, cfg=tcfg)

    # 4) Evaluate epsilon prediction quality across steps
    print('[Stage 3] Evaluation ...')
    eval_info = eval_eps_prediction(light, teacher, keys_1based, T=cfg.T_dry, batch=cfg.train_batch,
                                    device=cfg.device, save_dir_pdf=cfg.images_dir)

    # 5) ONNX export (optional)
    print('[Stage 4] ONNX export ...')
    onnx_path = os.path.join('models', 'loraprop_lightpath.onnx')
    _ = onnx_export_light(light, teacher, device=cfg.device, save_path=onnx_path)

    print('=== Finished. Artifacts ===')
    print(f"  - LoRA-Prop model: {cfg.model_save_path}")
    print(f"  - Figures (PDF): {cfg.images_dir}")
    print(f"  - ONNX (if exported): {onnx_path}")


if __name__ == '__main__':
    # Run the pipeline when invoked via: python -m src.main
    main()
