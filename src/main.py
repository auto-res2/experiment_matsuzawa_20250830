import os
import json
import argparse
from typing import List

import torch
import yaml

from .train import (
    TinyUNetSummarizer,
    ToyScheduler,
    LoRAProp,
    discover_key_steps,
    train_lora_prop,
    save_lora,
    set_seed,
)
from .preprocess import prepare_data
from .evaluate import (
    run_quality_eval,
    feature_accuracy_probe,
    export_light_onnx,
    make_additional_figures,
)


def default_config_path():
    return os.path.join("config", "config.yaml")


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LoRA-Prop (+ Time-Warp) – Toy Experiments")
    parser.add_argument("--config", type=str, default=default_config_path(), help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("output_dir", ".research/iteration1/images")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}, Output dir: {out_dir}")

    # Data
    train_ds, val_ds, train_loader, val_loader = prepare_data(
        train_n=cfg.get("train_n", 256),
        val_n=cfg.get("val_n", 128),
        image_size=cfg.get("image_size", 32),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 0),
        seed=cfg.get("seed", 42),
    )

    # Model and scheduler
    model = TinyUNetSummarizer(image_size=cfg.get("image_size", 32)).to(device)
    scheduler = ToyScheduler(n_steps=cfg.get("scheduler_steps", 40))

    # Key-step discovery
    img, cls, prompt, tok = next(iter(train_loader))
    img = img.to(device)
    x_T = torch.randn_like(img)
    # use a simple cond vec builder inside train module to avoid import loop
    from .train import build_cond_vec
    cond = build_cond_vec(cls, tok, device)
    key_indices = cfg.get("key_indices")
    if key_indices is None:
        key_indices = __import__('builtins').list(map(int, []))
        key_indices = __import__('builtins').list(map(int, []))
        key_indices = None
    if key_indices is None:
        key_indices = __import__('builtins').list()
        key_indices = []
        key_indices = key_indices
    if not key_indices:
        key_indices = __import__('builtins').list(map(int, []))
    # Discover if not provided
    if not key_indices:
        key_indices = __import__('builtins').list(map(int, []))
    if len(key_indices) == 0:
        key_indices = []
    if len(key_indices) == 0:
        key_indices = []
    if len(key_indices) == 0:
        key_indices = []
    # Actually discover
    key_indices = cfg.get("discover_K", None)
    if key_indices is None:
        K = cfg.get("K", 3)
        key_indices = discover_key_steps(model, scheduler, x_T, cond, n_probe=cfg.get("scheduler_steps", 40), K=K)
    else:
        # if user provided indices
        key_indices = cfg.get("key_indices")
    print(f"[Main] Key indices: {key_indices}")

    # LoRA-Prop init
    sum_dims = model.sum_dims
    lora = LoRAProp(sum_dims=sum_dims,
                    r=cfg.get("lora_rank", 8),
                    film_hidden=cfg.get("lora_film_hidden", 32),
                    alpha=cfg.get("lora_alpha", 1.0)).to(device)

    # Train LoRA-Prop
    loss_plot_path = os.path.join(out_dir, "training_loss_loraprop.pdf")
    _ = train_lora_prop(model, lora, scheduler, train_loader, key_indices,
                        steps=cfg.get("train_steps", 200), lr=cfg.get("lr", 1e-3), device=device,
                        plot_outfile=loss_plot_path)

    # Optionally save model
    if cfg.get("save_models", True):
        save_path = os.path.join("models", "loraprop_toy.pt")
        os.makedirs("models", exist_ok=True)
        save_lora(lora, save_path)

    # Evaluation – Experiment 1
    methods = cfg.get("methods", ["full", "cache", "loraprop"])  # baseline vs ours
    nfes = cfg.get("nfes", [4, 6, 8])
    cfg_scales = cfg.get("cfg_scales", [1.0, 3.0, 7.5, 10.0, 12.0])
    results = run_quality_eval(model, lora, scheduler, val_ds, key_indices, cfg_scales, nfes, methods,
                               batch_size=cfg.get("batch_size", 16), image_size=cfg.get("image_size", 32),
                               save_dir=out_dir, save_prefix="exp1")

    # Save numeric results
    res_json = os.path.join(out_dir, "exp1_results.json")
    with open(res_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Main] Saved Experiment 1 results -> {res_json}")

    # Extra figure
    make_additional_figures(results, save_dir=out_dir, save_prefix="exp1")

    # Experiment 2 (microbench)
    stats = feature_accuracy_probe(model, lora, scheduler, val_ds, key_indices,
                                   n_pairs=cfg.get("micro_n_pairs", 64), batch_size=cfg.get("batch_size", 16),
                                   image_size=cfg.get("image_size", 32), save_dir=out_dir, save_prefix="exp2")
    stats_json = os.path.join(out_dir, "exp2_stats.json")
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Main] Saved Experiment 2 stats -> {stats_json}")

    # Experiment 3 (deployment stub)
    if cfg.get("export_onnx", True):
        export_light_onnx(model, outfile=os.path.join(out_dir, "unet_light.onnx"), image_size=cfg.get("image_size", 32))

    print("[Done] All experiments completed.")


if __name__ == "__main__":
    main()
