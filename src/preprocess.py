import os
from dataclasses import dataclass
from typing import Tuple

import torch

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class PreprocessConfig:
    experiment_name: str = "eafa_t4"
    model_name: str = "gpt2"
    seq_len: int = 512
    data_dir: str = "data"


def _auto_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok


def _tokenize_texts(tokenizer, texts, seq_len: int, stride: int = None) -> torch.Tensor:
    if stride is None:
        stride = seq_len
    ids = tokenizer("\n\n".join(texts), return_tensors=None, add_special_tokens=False)["input_ids"]
    ids = torch.tensor(ids, dtype=torch.long)
    blocks = []
    i = 0
    while i + seq_len <= ids.numel():
        blocks.append(ids[i:i+seq_len])
        i += stride
    if len(blocks) == 0:
        # pad a single block if text too short
        pad_id = tokenizer.pad_token_id
        blk = torch.full((seq_len,), pad_id, dtype=torch.long)
        blocks.append(blk)
    return torch.stack(blocks, dim=0)


def _synthetic_tokens(seq_len: int, n_blocks: int = 256, vocab_size: int = 50257) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(low=0, high=vocab_size, size=(n_blocks, seq_len), generator=g)


def preprocess(cfg: PreprocessConfig) -> Tuple[str, str]:
    ensure_dir(cfg.data_dir)
    train_out = os.path.join(cfg.data_dir, f"{cfg.experiment_name}_train_L{cfg.seq_len}.pt")
    val_out = os.path.join(cfg.data_dir, f"{cfg.experiment_name}_val_L{cfg.seq_len}.pt")

    if os.path.exists(train_out) and os.path.exists(val_out):
        return train_out, val_out

    if HF_AVAILABLE:
        try:
            ds_tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            tok = _auto_tokenizer(cfg.model_name)
            train_tokens = _tokenize_texts(tok, ds_tr["text"], seq_len=cfg.seq_len, stride=cfg.seq_len)
            val_tokens = _tokenize_texts(tok, ds_val["text"], seq_len=cfg.seq_len, stride=cfg.seq_len)
        except Exception:
            train_tokens = _synthetic_tokens(cfg.seq_len, n_blocks=512)
            val_tokens = _synthetic_tokens(cfg.seq_len, n_blocks=128)
    else:
        train_tokens = _synthetic_tokens(cfg.seq_len, n_blocks=512)
        val_tokens = _synthetic_tokens(cfg.seq_len, n_blocks=128)

    torch.save(train_tokens, train_out)
    torch.save(val_tokens, val_out)
    return train_out, val_out
