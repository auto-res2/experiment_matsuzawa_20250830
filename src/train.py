#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script (optional) for the Energy-Adaptive SDPA experiments.

Notes:
- By default, we do not perform any training; we rely on pretrained GPT-2 weights.
- A tiny finetuning loop is provided for completeness and smoke testing. It is disabled unless do_train=True
  is set in the provided config.
- Training is kept extremely light to fit within the NVIDIA T4 16GB VRAM budget.

Run context:
- This module is orchestrated from src.main (python -m src.main)
- All imports within src must be relative.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

from .preprocess import TokenSequenceDataset


@dataclass
class TrainConfig:
    model_name: str = "gpt2"  # small to be safe on T4
    device_index: int = 0
    do_train: bool = False
    seq_len: int = 128
    batch_size: int = 4
    steps: int = 20
    lr: float = 5e-5
    warmup_steps: int = 5
    weight_decay: float = 0.01
    output_dir: str = "./models"
    tokens_npy: Optional[str] = None  # path to preprocessed token stream


def _simple_collate(batch):
    # batch: list of dicts with input_ids
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    return {"input_ids": input_ids}


def train_model(cfg: TrainConfig) -> Optional[str]:
    """Optionally finetune GPT-2 for a few steps; returns saved model directory or None.

    If cfg.do_train is False, this function logs and returns None.
    """
    if not cfg.do_train:
        logging.info("Training disabled (do_train=False). Using pretrained weights only.")
        return None

    if not _HAVE_HF:
        raise RuntimeError("Hugging Face transformers is required for training.")

    device = torch.device("cuda", cfg.device_index) if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for training.")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Loading model %s for (optional) finetuning...", cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.train()
    model.to(device)

    # Dataset from pre-tokenized stream
    if cfg.tokens_npy is None or not Path(cfg.tokens_npy).exists():
        raise FileNotFoundError("Preprocessed tokens file missing. Please run preprocess first.")
    toks = np.load(cfg.tokens_npy)
    dataset = TokenSequenceDataset(toks, seq_len=cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, collate_fn=_simple_collate)

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optim = AdamW(grouped, lr=cfg.lr)
    total_steps = cfg.steps
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    step = 0
    it = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = input_ids.clone()
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        sched.step()
        step += 1
        if step % 5 == 0:
            logging.info("[train] step %d/%d loss=%.4f", step, cfg.steps, float(loss.detach().cpu()))

    # Save model
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / f"{cfg.model_name}-tinyfinetune"
    model.save_pretrained(save_dir)
    logging.info("Saved finetuned model to %s", save_dir)
    return str(save_dir)


__all__ = ["TrainConfig", "train_model"]
