#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Energy-Adaptive SDPA experiments.

Run from project root:
  python -m src.main

This orchestrates:
- Preprocessing (token stream generation)
- Optional training (disabled by default)
- Evaluation & plotting (saves PDFs to .research/iteration3/images)

Config:
- A YAML config may be provided at config/experiment.yaml, or an alternate path via --config.
- Command-line flags can enable a smoke test with tiny settings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import torch

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required to read configuration.") from e

from .preprocess import PreprocessConfig, ensure_token_stream
from .train import TrainConfig, train_model
from .evaluate import (
    ExperimentConfig as EvalExperimentConfig,
    ModelConfig as EvalModelConfig,
    RunConfig as EvalRunConfig,
    run_experiments,
)


DEFAULT_CONFIG_PATH = "config/experiment.yaml"


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Energy-Adaptive SDPA experiment runner")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    p.add_argument("--smoke-test", action="store_true", help="Run a tiny smoke test configuration")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg_dict = load_config(args.config)

    # Directories
    output_dir = Path(cfg_dict.get("output_dir", ".research/iteration3"))
    images_dir = Path(cfg_dict.get("images_dir", ".research/iteration3/images"))
    setup_logging(output_dir)

    logging.info("CUDA available: %s", torch.cuda.is_available())

    # 1) Preprocess
    pp_cfg = PreprocessConfig(
        tokenizer_name=cfg_dict.get("tokenizer_name", "gpt2"),
        output_dir=cfg_dict.get("data_dir", "./data"),
        tokens_filename=cfg_dict.get("tokens_filename", "wikitext103_gpt2_tokens.npy"),
    )
    tokens_npy = ensure_token_stream(pp_cfg)

    # 2) Optional training (disabled by default)
    train_cfg = TrainConfig(
        model_name=cfg_dict.get("model_name", "gpt2"),
        device_index=cfg_dict.get("device_index", 0),
        do_train=bool(cfg_dict.get("do_train", False)),
        seq_len=int(cfg_dict.get("train_seq_len", 128)),
        batch_size=int(cfg_dict.get("train_batch_size", 4)),
        steps=int(cfg_dict.get("train_steps", 20)),
        lr=float(cfg_dict.get("lr", 5e-5)),
        warmup_steps=int(cfg_dict.get("warmup_steps", 5)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.01)),
        output_dir=cfg_dict.get("models_dir", "./models"),
        tokens_npy=tokens_npy,
    )
    trained_model_dir = train_model(train_cfg)

    # 3) Evaluation
    if args.smoke_test:
        seq_lens = [128]
        seeds = [11]
        backends = ["mem_efficient", "math", "eafa"]
        max_batches = 2
        idle_seconds = 5.0
        logging.info("Running smoke test: L=%s seeds=%s backends=%s batches=%d", seq_lens, seeds, backends, max_batches)
    else:
        seq_lens = cfg_dict.get("seq_lens", [512, 1024])
        seeds = cfg_dict.get("seeds", [11, 13])
        backends = cfg_dict.get("backends", ["mem_efficient", "math", "eafa"])  # xformers is optional
        max_batches = int(cfg_dict.get("max_batches", 20))
        idle_seconds = float(cfg_dict.get("idle_seconds", 10.0))

    eval_cfg = EvalExperimentConfig(
        model_cfg=EvalModelConfig(model_name=cfg_dict.get("model_name", "gpt2"), device_index=cfg_dict.get("device_index", 0)),
        run_cfg=EvalRunConfig(
            seeds=seeds,
            seq_lens=seq_lens,
            backends=backends,
            max_batches=max_batches,
            tokens_budget=int(cfg_dict.get("tokens_budget", 16000)),
            output_dir=str(output_dir),
            images_dir=str(images_dir),
        ),
        tokens_npy=tokens_npy,
        idle_seconds=idle_seconds,
    )

    artifacts = run_experiments(eval_cfg)
    logging.info("Artifacts: %s", json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
