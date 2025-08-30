import argparse
import json
import os
from dataclasses import asdict

import torch

from .preprocess import preprocess, PreprocessConfig
from .train import train_model, TrainConfig
from .evaluate import evaluate, EvalConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Energy-Adaptive FlashAttention (T4) Experiment Runner")
    p.add_argument("--config", type=str, default=None, help="Path to a JSON/YAML config file.")
    p.add_argument("--smoke_test", action="store_true", help="Run a minimal smoke test (few steps/batches).")
    return p.parse_args()


def load_config(path: str) -> dict:
    if path is None:
        return {}
    with open(path, "r") as f:
        txt = f.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)


def main():
    args = parse_args()
    user_cfg = load_config(args.config) if args.config else {}

    # Defaults
    exp_name = user_cfg.get("experiment_name", "eafa_t4")
    model_name = user_cfg.get("model_name", "gpt2")
    seq_len = int(user_cfg.get("seq_len", 512))

    pp_cfg = PreprocessConfig(
        experiment_name=exp_name,
        model_name=model_name,
        seq_len=seq_len,
        data_dir=user_cfg.get("data_dir", "data"),
    )

    # Preprocess
    train_path, val_path = preprocess(pp_cfg)
    print(f"[main] Preprocessed data -> train: {train_path} | val: {val_path}")

    # Train
    tr_cfg = TrainConfig(
        experiment_name=exp_name,
        model_name=model_name,
        seq_len=seq_len,
        train_steps=int(user_cfg.get("train_steps", 50 if not args.smoke_test else 5)),
        warmup_steps=int(user_cfg.get("warmup_steps", 5 if not args.smoke_test else 1)),
        lr=float(user_cfg.get("lr", 5e-5)),
        weight_decay=float(user_cfg.get("weight_decay", 0.0)),
        train_batch_size=int(user_cfg.get("train_batch_size", 2 if not args.smoke_test else 1)),
        grad_accum_steps=int(user_cfg.get("grad_accum_steps", 1)),
        save_dir=user_cfg.get("model_dir", "models"),
        data_dir=user_cfg.get("data_dir", "data"),
        device=user_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        seed=int(user_cfg.get("seed", 42)),
    )
    model_path = train_model(tr_cfg, train_data_path=train_path)

    # Evaluate
    ev_cfg = EvalConfig(
        experiment_name=exp_name,
        model_name=model_name,
        seq_len=seq_len,
        eval_batch_size=int(user_cfg.get("eval_batch_size", 4 if not args.smoke_test else 2)),
        warmup_batches=int(user_cfg.get("eval_warmup_batches", 3 if not args.smoke_test else 1)),
        measure_batches=int(user_cfg.get("eval_measure_batches", 10 if not args.smoke_test else 2)),
        data_dir=user_cfg.get("data_dir", "data"),
        model_dir=user_cfg.get("model_dir", "models"),
        device=user_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        gpu_index=int(user_cfg.get("gpu_index", 0)),
        enable_mpts=bool(user_cfg.get("enable_mpts", True)),
        enable_skp=bool(user_cfg.get("enable_skp", True)),
    )
    results = evaluate(ev_cfg, val_data_path=val_path, model_path=model_path)

    print("[main] Evaluation summary:")
    for k, v in results.items():
        print(f"  - {k}: PPL={v['ppl']:.3f}, tokens/s={v['mean_tokens_per_s']:.1f}, energy/batch={v['mean_energy_j']:.3f} J")


if __name__ == "__main__":
    main()
