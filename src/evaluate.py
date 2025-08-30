import os
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import (
    TinyUNetSummarizer,
    ToyScheduler,
    LoRAProp,
    build_cond_vec,
)

# ----------------------------
# Sampler and helpers
# ----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def classifier_free_guidance(eps_uncond, eps_cond, guidance_scale: float = 7.5):
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def build_time_grid(n_probe: int, total_steps: int, key_indices: List[int]) -> List[int]:
    base = np.linspace(0, n_probe-1, total_steps, dtype=int).tolist()
    merged = sorted(set(base).union(set(key_indices)))
    if len(merged) > total_steps:
        extras = [i for i in merged if i not in key_indices]
        while len(merged) > total_steps and extras:
            merged.remove(extras.pop(0))
    while len(merged) < total_steps:
        for k in key_indices:
            if len(merged) >= total_steps:
                break
            cand = min(k+1, n_probe-1)
            if cand not in merged:
                merged.append(cand)
        merged = sorted(set(merged))
    return sorted(merged)


def run_hybrid_sampling(model: TinyUNetSummarizer,
                        lora_prop: LoRAProp,
                        scheduler: ToyScheduler,
                        x_T: torch.Tensor,
                        cond_vec: torch.Tensor,
                        key_indices: List[int],
                        total_steps: int = 6,
                        method: str = "loraprop",
                        guidance_scale: float = 7.5,
                        verbose: bool = False) -> Dict[str, torch.Tensor]:
    device = x_T.device
    n_probe = scheduler.n_steps
    time_grid = build_time_grid(n_probe, total_steps, key_indices)
    key_set = set(key_indices)
    x = x_T
    prev_summaries = None
    eps_trace = []
    comp_stats = {"full_steps": 0, "light_steps": 0}
    for idx in time_grid:
        t_batch = torch.full((x.size(0),), float(scheduler.timesteps[idx]), device=device)
        # Ensure the very first step uses heavy path to initialize summaries
        if prev_summaries is None or idx in key_set or method == "full":
            eps_uc, sums_uc = model(x, t_batch, torch.zeros_like(cond_vec), light=False)
            eps_c, sums_c = model(x, t_batch, cond_vec, light=False)
            prev_summaries = sums_c
            comp_stats["full_steps"] += 1
        else:
            if method == "cache":
                sums_hat = prev_summaries
            elif method == "loraprop":
                eligible = [k for k in key_indices if k <= idx]
                prev_k = max(eligible) if eligible else idx
                dt = torch.full((x.size(0),), float(idx - prev_k), device=device)
                dt = dt / (n_probe-1)
                sums_hat = lora_prop(prev_summaries, dt)
            else:
                raise ValueError(f"Unknown method {method}")
            eps_uc, _ = model(x, t_batch, torch.zeros_like(cond_vec), summaries=sums_hat, light=True)
            eps_c, _ = model(x, t_batch, cond_vec, summaries=sums_hat, light=True)
            comp_stats["light_steps"] += 1
        eps = classifier_free_guidance(eps_uc, eps_c, guidance_scale)
        eps_trace.append(eps.detach())
        x = scheduler.ddim_step(x, int(scheduler.timesteps[idx]), eps)
        if verbose:
            print(f"[Hybrid] step {idx} (key={idx in key_set}), x.norm={x.norm().item():.3f}")
    return {"x_0": x, "eps_trace": eps_trace, "comp_stats": comp_stats, "time_grid": time_grid}


# ----------------------------
# FLOPs estimation (toy)
# ----------------------------

def conv2d_flops(h, w, cin, cout, k, groups=1):
    return 2 * h * w * cin * cout * (k*k) / groups


def estimate_model_flops(model: TinyUNetSummarizer, image_size: int = 32) -> Dict[str, float]:
    H = W = image_size
    base = 32
    total = conv2d_flops(H, W, 3, base, 3)
    total += conv2d_flops(H, W, base, base, 3) + conv2d_flops(H, W, base, base, 3) + conv2d_flops(H, W, base, base, 3)
    H //= 2
    total += conv2d_flops(H*2, W*2, base, base*2, 3) + conv2d_flops(H*2, W*2, base*2, base*2, 3) + conv2d_flops(H*2, W*2, base*2, base*2, 3)
    H //= 2
    total += conv2d_flops(H, W, base*2, base*2, 3)
    total += conv2d_flops(H, W, base*2, base, 2)
    total += conv2d_flops(H*2, W*2, base*3, base, 3) + conv2d_flops(H*2, W*2, base, base, 3)
    H *= 2
    total += conv2d_flops(H, W, base, base, 2)
    total += conv2d_flops(H, W, base*2, base, 3) + conv2d_flops(H, W, base, base, 3)
    total += conv2d_flops(H, W, base, base, 3)
    total += H*W*base*base*2
    return {"heavy_forward_flops": total}


def linear_flops(m, n):
    return 2 * m * n


def estimate_lora_flops(lora: LoRAProp, batch_size: int = 1) -> float:
    total = 0.0
    for ad in lora.adapters:
        d = ad.sum_dim
        r = ad.r
        total += batch_size * (linear_flops(d, r) + linear_flops(r, d))
    return total


# ----------------------------
# Metrics and plotting
# ----------------------------

def psnr_tensor(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def run_quality_eval(model,
                     lora,
                     scheduler,
                     dataset,
                     key_indices,
                     cfg_scales: List[float],
                     nfes: List[int],
                     methods: List[str],
                     batch_size: int = 8,
                     image_size: int = 32,
                     save_dir: str = ".research/iteration2/images",
                     save_prefix: str = "exp1"):
    ensure_dir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    flops_heavy = estimate_model_flops(model, image_size=image_size)["heavy_forward_flops"]
    for method in methods:
        for nfe in nfes:
            for gs in cfg_scales:
                t0 = time.time()
                cum_psnr = 0.0
                cum_mse = 0.0
                N = 0
                comp_full = 0
                comp_light = 0
                for img, cls, prompt, tok_len in loader:
                    B = img.size(0)
                    img = img.to(device)
                    cond = build_cond_vec(cls, tok_len, device)
                    x = torch.randn_like(img)
                    out = run_hybrid_sampling(model, lora, scheduler, x, cond, key_indices, total_steps=nfe, method=method, guidance_scale=gs)
                    x0 = out["x_0"].clamp(0,1)
                    psnr_val = psnr_tensor(x0, img)
                    mse = F.mse_loss(x0, img).item()
                    cum_psnr += psnr_val * B
                    cum_mse += mse * B
                    N += B
                    comp_full += out["comp_stats"]["full_steps"]
                    comp_light += out["comp_stats"]["light_steps"]
                t1 = time.time()
                light_flops = estimate_lora_flops(lora, batch_size=batch_size) + 1e7
                avg_full = comp_full / max(1, len(loader))
                avg_light = comp_light / max(1, len(loader))
                gflops_per_img = ((avg_full * flops_heavy) + (avg_light * light_flops)) / 1e9 / batch_size
                it_per_sec = (N / (t1 - t0 + 1e-9))
                results.append({
                    "method": method,
                    "nfe": nfe,
                    "guidance": gs,
                    "psnr": cum_psnr / N,
                    "mse": cum_mse / N,
                    "gflops": gflops_per_img,
                    "throughput_ips": it_per_sec
                })
                print(f"[Eval] method={method}, nfe={nfe}, gs={gs}: PSNR={results[-1]['psnr']:.3f}, MSE={results[-1]['mse']:.5f}, GFLOPs/img~{results[-1]['gflops']:.3f}, ips~{results[-1]['throughput_ips']:.2f}")
    # Plot Quality vs GFLOPs frontier (PSNR vs GFLOPs)
    plt.figure(figsize=(6,4))
    for method in methods:
        xs = [r["gflops"] for r in results if r["method"]==method]
        ys = [r["psnr"] for r in results if r["method"]==method]
        labels = [f"NFE={r['nfe']} gs={r['guidance']}" for r in results if r["method"]==method]
        plt.scatter(xs, ys, label=method)
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y), fontsize=7)
    plt.xlabel("GFLOPs / image (estimated)")
    plt.ylabel("PSNR (dB) ↑")
    plt.title("Quality vs Compute Frontier (Toy)")
    plt.legend()
    plt.tight_layout()
    fig_name = os.path.join(save_dir, f"{save_prefix}_quality_vs_gflops_loraprop.pdf")
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved plot -> {fig_name}")

    # Robustness vs guidance (line chart for best NFE per method)
    best_nfe = max(nfes)
    plt.figure(figsize=(6,4))
    for method in methods:
        xs = []
        ys = []
        for gs in sorted(set(cfg_scales)):
            vals = [r for r in results if r["method"]==method and r["nfe"]==best_nfe and r["guidance"]==gs]
            if len(vals)>0:
                xs.append(gs)
                ys.append(vals[0]["psnr"])
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Guidance scale")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Robustness vs Guidance (NFE={best_nfe})")
    plt.legend()
    plt.tight_layout()
    fig_name = os.path.join(save_dir, f"{save_prefix}_robustness_guidance.pdf")
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved plot -> {fig_name}")

    return results


def feature_accuracy_probe(model,
                           lora,
                           scheduler,
                           dataset,
                           key_indices,
                           n_pairs: int = 50,
                           batch_size: int = 8,
                           image_size: int = 32,
                           save_dir: str = ".research/iteration2/images",
                           save_prefix: str = "exp2"):
    ensure_dir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model.eval()
    mse_per_layer = []
    cos_per_layer = []

    with torch.no_grad():
        cnt = 0
        for _, (img, cls, prompt, tok_len) in enumerate(loader):
            if cnt >= n_pairs:
                break
            img = img.to(device)
            cond = build_cond_vec(cls, tok_len, device)
            probe_steps = scheduler.n_steps
            j = np.random.randint(1, probe_steps-1)
            prev_keys = [k for k in key_indices if k <= j]
            if not prev_keys:
                continue
            k = max(prev_keys)
            t_j = torch.full((img.size(0),), float(scheduler.timesteps[j]), device=device)
            t_k = torch.full((img.size(0),), float(scheduler.timesteps[k]), device=device)
            x_t, _ = scheduler.add_noise(img, t_j.long())
            _, sums_k = model(x_t, t_k, cond, light=False)
            _, sums_j = model(x_t, t_j, cond, light=False)
            dt = torch.full((img.size(0),), float(j-k) / (probe_steps-1), device=device)
            sums_hat = lora(sums_k, dt)
            for L, (a, b) in enumerate(zip(sums_hat, sums_j)):
                mse = F.mse_loss(a, b, reduction='mean').item()
                na = F.normalize(a, dim=1)
                nb = F.normalize(b, dim=1)
                cos = (na*nb).sum(dim=1).mean().item()
                mse_per_layer.append((L, mse))
                cos_per_layer.append((L, cos))
            cnt += img.size(0)

    layers = sorted(set([i for i,_ in mse_per_layer]))
    avg_mse = [np.mean([m for L,m in mse_per_layer if L==i]) for i in layers]
    avg_cos = [np.mean([c for L,c in cos_per_layer if L==i]) for i in layers]

    # Plots
    plt.figure(figsize=(5,3))
    plt.bar([str(i) for i in layers], avg_mse)
    plt.xlabel("Layer index")
    plt.ylabel("MSE")
    plt.title("Feature MSE per layer (LoRA-Prop)")
    plt.tight_layout()
    out1 = os.path.join(save_dir, f"{save_prefix}_feature_mse_layers.pdf")
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    print(f"[Microbench] Saved layer-wise feature MSE -> {out1}")

    plt.figure(figsize=(5,3))
    plt.bar([str(i) for i in layers], avg_cos)
    plt.xlabel("Layer index")
    plt.ylabel("Cosine sim")
    plt.title("Feature cosine similarity per layer")
    plt.tight_layout()
    out2 = os.path.join(save_dir, f"{save_prefix}_feature_cosine_layers.pdf")
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"[Microbench] Saved layer-wise cosine -> {out2}")

    # Estimated FLOPs reduction bar
    flops = estimate_model_flops(model, image_size=image_size)
    heavy = flops["heavy_forward_flops"]
    light = estimate_lora_flops(lora, batch_size=batch_size) + 1e7
    plt.figure(figsize=(5,3))
    plt.bar(["full", "non-key"], [heavy/1e9, light/1e9])
    plt.ylabel("GFLOPs per step")
    plt.title("Per-step compute: full vs non-key (estimated)")
    plt.tight_layout()
    out3 = os.path.join(save_dir, f"{save_prefix}_gflops_reduction.pdf")
    plt.savefig(out3, bbox_inches="tight")
    plt.close()
    print(f"[Microbench] Saved GFLOPs reduction plot -> {out3}")

    return {
        "avg_layer_mse": avg_mse,
        "avg_layer_cos": avg_cos,
    }


def export_light_onnx(model: TinyUNetSummarizer, outfile: str = "unet_light.onnx", image_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    cond = torch.randn(1, 32, device=device)
    base = 32
    s1 = torch.randn(1, base, device=device)
    s2 = torch.randn(1, base, device=device)
    s3 = torch.randn(1, base*2, device=device)
    s4 = torch.randn(1, base*2, device=device)
    s5 = torch.randn(1, base, device=device)
    s6 = torch.randn(1, base, device=device)

    class LightWrapper(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
        def forward(self, x_in, cond, s1, s2, s3, s4, s5, s6):
            return self.head(x_in, [s1, s2, s3, s4, s5, s6], cond)

    wrapper = LightWrapper(model.eps_head).to(device)
    try:
        import torch.onnx  # noqa
        torch.onnx.export(
            wrapper, (x, cond, s1, s2, s3, s4, s5, s6), outfile,
            input_names=["x_in", "cond", "s1","s2","s3","s4","s5","s6"],
            output_names=["eps"], opset_version=17, dynamic_axes={"x_in":{0:"B"}}
        )
        print(f"[ONNX] Exported light path -> {outfile}")
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")


def make_additional_figures(results_exp1: List[Dict], save_dir: str = ".research/iteration2/images", save_prefix: str = "exp1"):
    ensure_dir(save_dir)
    methods = sorted(set([r["method"] for r in results_exp1]))
    nfes = sorted(set([r["nfe"] for r in results_exp1]))
    mat = np.zeros((len(methods), len(nfes)))
    for i, m in enumerate(methods):
        for j, n in enumerate(nfes):
            vals = [r["psnr"] for r in results_exp1 if r["method"]==m and r["nfe"]==n]
            mat[i, j] = np.mean(vals) if len(vals)>0 else np.nan
    plt.figure(figsize=(5,3))
    sns.heatmap(mat, annot=True, xticklabels=nfes, yticklabels=methods, cmap="viridis")
    plt.xlabel("NFE")
    plt.ylabel("Method")
    plt.title("PSNR heatmap (avg over guidance)")
    plt.tight_layout()
    outp = os.path.join(save_dir, f"{save_prefix}_psnr_heatmap.pdf")
    plt.savefig(outp, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved heatmap -> {outp}")
