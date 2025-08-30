#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing utilities and reproducibility helpers.
- Synthetic dataset fallback to avoid large downloads in quick tests.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    # Ensure cuBLAS reproducibility is configured for deterministic algorithms
    # Must be set before CUDA ops that use cuBLAS.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


class PackedTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, seq_len: int, pad_token_id: int, stride: int | None = None) -> None:
        self.seq_len = int(seq_len)
        self.pad_token_id = int(pad_token_id)
        self.stride = int(stride) if stride is not None else int(seq_len)
        all_ids: List[int] = []
        for t in texts:
            ids = tokenizer.encode(t)
            if len(ids) == 0:
                continue
            all_ids.extend(ids + [tokenizer.eos_token_id])
        self.chunks: List[np.ndarray] = []
        i = 0
        while i < len(all_ids):
            window = all_ids[i: i + self.seq_len]
            if len(window) < self.seq_len:
                window = window + [self.pad_token_id] * (self.seq_len - len(window))
            self.chunks.append(np.asarray(window, dtype=np.int64))
            i += self.stride
        if len(self.chunks) == 0:
            self.chunks = [np.zeros(self.seq_len, dtype=np.int64)]
    def __len__(self) -> int:
        return len(self.chunks)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        arr = self.chunks[idx]
        x = torch.from_numpy(arr)
        y = torch.roll(x, shifts=-1, dims=0)
        return {"input_ids": x, "labels": y}


def build_dataloaders(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, Any]:
    # Synthetic short path
    if dataset_name.lower() == "synthetic":
        class DummyTok:
            eos_token_id = 50256
            pad_token_id = 50256
            def encode(self, s: str) -> List[int]:  # type: ignore
                rng = np.random.RandomState(abs(hash(s)) % (2**32))
                ln = rng.randint(10, 200)
                return rng.randint(0, 30000, size=ln).tolist()
        tok = DummyTok()
        texts = [f"synthetic sample {i}" for i in range(10000)]
        dataset = PackedTextDataset(texts, tok, seq_len=seq_len, pad_token_id=tok.pad_token_id)
        collate = lambda batch: {"input_ids": torch.stack([b["input_ids"] for b in batch], 0),
                                 "labels": torch.stack([b["labels"] for b in batch], 0)}
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)
        return dl, tok
    # HuggingFace path (optional)
    try:
        from datasets import load_dataset  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        ds = load_dataset(dataset_name, split=split)
        texts = [r["text"] for r in ds.select(range(min(5000, len(ds)))) if isinstance(r.get("text"), str)]
        dataset = PackedTextDataset(texts, tok, seq_len=seq_len, pad_token_id=tok.pad_token_id)
        collate = lambda batch: {"input_ids": torch.stack([b["input_ids"] for b in batch], 0),
                                 "labels": torch.stack([b["labels"] for b in batch], 0)}
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0), collate_fn=collate)
        return dl, tok
    except Exception as e:
        warnings = f"Falling back to synthetic dataset: {e}"
        print(warnings)
        class DummyTok:
            eos_token_id = 50256
            pad_token_id = 50256
            def encode(self, s: str) -> List[int]:  # type: ignore
                rng = np.random.RandomState(abs(hash(s)) % (2**32))
                ln = rng.randint(10, 200)
                return rng.randint(0, 30000, size=ln).tolist()
        tok = DummyTok()
        texts = [f"synthetic sample {i}" for i in range(10000)]
        dataset = PackedTextDataset(texts, tok, seq_len=seq_len, pad_token_id=tok.pad_token_id)
        collate = lambda batch: {"input_ids": torch.stack([b["input_ids"] for b in batch], 0),
                                 "labels": torch.stack([b["labels"] for b in batch], 0)}
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)
        return dl, tok
