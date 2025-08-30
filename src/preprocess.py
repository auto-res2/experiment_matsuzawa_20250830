#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing utilities for EA-FA experiments.

- Downloads WikiText-103 (test split) via Hugging Face datasets.
- Tokenizes with GPT-2 tokenizer and saves a flat token stream as a NumPy array.
- Provides a dataset to yield contiguous token chunks of length seq_len.

Run context:
- Orchestrated from src.main (python -m src.main)
- All imports within src are relative where applicable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

try:
    import datasets as hfdatasets
    _HAVE_DATASETS = True
except Exception:
    _HAVE_DATASETS = False


@dataclass
class PreprocessConfig:
    tokenizer_name: str = "gpt2"
    output_dir: str = "./data"
    tokens_filename: str = "wikitext103_gpt2_tokens.npy"


def ensure_token_stream(cfg: PreprocessConfig) -> str:
    """Create the flat token stream file if it does not exist. Returns path to .npy file."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.tokens_filename
    if out_path.exists():
        logging.info("Using existing token stream at %s", out_path)
        return str(out_path)

    if not _HAVE_HF or not _HAVE_DATASETS:
        raise RuntimeError("transformers and datasets are required for preprocessing.")

    logging.info("Loading WikiText-103 test split and tokenizing with %s...", cfg.tokenizer_name)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    ds = hfdatasets.load_dataset("wikitext", "wikitext-103-v1", split="test")
    text = "\n\n".join(ds["text"])  # concatenate for a long stream
    enc = tok(text, return_tensors=None, add_special_tokens=False)
    # enc["input_ids"] is a list of lists (ragged); flatten
    ragged = enc["input_ids"]
    flat = np.concatenate([np.array(x, dtype=np.int64) for x in ragged]).astype(np.int64)
    np.save(out_path, flat)
    logging.info("Saved token stream: %s (tokens=%d)", out_path, len(flat))
    return str(out_path)


class TokenSequenceDataset(Dataset):
    """Dataset yielding contiguous sequences of length seq_len from a 1D token array."""
    def __init__(self, token_ids: np.ndarray, seq_len: int):
        assert token_ids.ndim == 1
        self.seq_len = int(seq_len)
        n_full = (len(token_ids) // self.seq_len) * self.seq_len
        self.tokens = token_ids[:n_full].astype(np.int64)
        self.num_samples = len(self.tokens) // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        return {"input_ids": x}


__all__ = ["PreprocessConfig", "ensure_token_stream", "TokenSequenceDataset"]
