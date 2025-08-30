from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from .preprocess import SimpleDiffusionTeacher, set_seed
from .train import LoRAPropLightPath


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def eval_eps_prediction(light: LoRAPropLightPath,
                        teacher: SimpleDiffusionTeacher,
                        keys_1based: List[int],
                        T: int,
                        batch: int,
                        device: str,
                        save_dir_pdf: str) -> Dict[str, Any]:
    set_seed(321)
    dev = torch.device(device)
    _ensure_dir(save_dir_pdf)

    teacher = teacher.to(dev).eval()
    light = light.to(dev).eval()

    time_grid = torch.linspace(0.0, 1.0, steps=T, device=dev)

    # Evaluate MSE across steps for LoRA-Prop vs cache-only vs prev-eps
    methods = ['LoRA-Prop', 'Cache-Only', 'Prev-Eps']
    mse_per_step = {m: [] for m in methods}

    with torch.no_grad():
        for t in range(1, T+1):
            # Skip key steps in error aggregation (they would be computed by full UNet in real run)
            if t in set(keys_1based):
                # Put tiny zero to keep arrays lengths equal
                for m in methods:
                    mse_per_step[m].append(0.0)
                continue

            # Build batch of latents and corresponding prev keys
            latents = torch.randn(batch, 4, 32, 32, device=dev)
            prev_keys = []
            for _ in range(batch):
                prevs = [k for k in keys_1based if k < t]
                prev_keys.append(max(prevs) if len(prevs) > 0 else keys_1based[0])
            prev_keys = torch.tensor(prev_keys, device=dev)
            dt = (torch.full((batch,), float(t), device=dev) - prev_keys.float()) / float(T)

            eps_target_list = []
            feats_prev_all = []
            for b in range(batch):
                eps_t, _ = teacher(latents[b:b+1], time_grid[t-1])
                eps_target_list.append(eps_t)
                eps_prev, feats_prev = teacher(latents[b:b+1], time_grid[prev_keys[b]-1])
                feats_prev_all.append(feats_prev)
            eps_target = torch.cat(eps_target_list, dim=0)
            # Collate features per scale
            n_scales = len(teacher.feature_channels)
            cached_feats = []
            for s in range(n_scales):
                cached_feats.append(torch.cat([feats_prev_all[b][s] for b in range(batch)], dim=0))

            # LoRA-Prop
            eps_hat = light(cached_feats, dt, zero_summary=False)
            # Cache-only baseline (zero_summary=True)
            eps_cache = light(cached_feats, dt, zero_summary=True)
            # Prev-eps baseline (reuse eps at prev key)
            eps_prev = []
            for b in range(batch):
                eprev, _ = teacher(latents[b:b+1], time_grid[prev_keys[b]-1])
                eps_prev.append(eprev)
            eps_prev = torch.cat(eps_prev, dim=0)

            mse_per_step['LoRA-Prop'].append(float(F.mse_loss(eps_hat, eps_target).item()))
            mse_per_step['Cache-Only'].append(float(F.mse_loss(eps_cache, eps_target).item()))
            mse_per_step['Prev-Eps'].append(float(F.mse_loss(eps_prev, eps_target).item()))

    # Plot MSE vs timestep
    if plt is not None:
        sns.set(style='whitegrid')
        x = list(range(1, T+1))
        plt.figure(figsize=(6,4))
        for m in methods:
            plt.plot(x, mse_per_step[m], label=m)
        for k in keys_1based:
            plt.axvline(k, color='k', linestyle='--', alpha=0.2)
        plt.xlabel('Timestep (1-based)')
        plt.ylabel('Epsilon MSE')
        plt.title('Epsilon prediction error across timesteps')
        plt.legend()
        out_pdf = os.path.join(save_dir_pdf, 'epsilon_mse_vs_timestep.pdf')
        plt.tight_layout(); plt.savefig(out_pdf, bbox_inches='tight'); plt.close()
        print(f'Saved figure: {out_pdf}')

    # Aggregate
    agg = {m: float(np.mean(mse_per_step[m])) for m in methods}
    print("MSE summary:")
    for m in methods:
        print(f"  {m:10s}: {agg[m]:.6f}")

    return {
        'mse_per_step': mse_per_step,
        'mse_mean': agg,
        'keys': keys_1based,
        'T': T,
    }


def onnx_export_light(light: LoRAPropLightPath, teacher: SimpleDiffusionTeacher, device: str, save_path: str) -> Dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except Exception:
        print('ONNX not available; skipping export.')
        return {'exported': False}

    dev = torch.device(device)
    light = light.to(dev).eval()

    # Create dummy inputs (3 scales)
    B = 1
    feats = []
    Hs = [32, 16, 8]
    for c, H in zip(teacher.feature_channels, Hs):
        feats.append(torch.randn(B, c, H, H, device=dev))
    dt = torch.tensor([0.25], device=dev)

    # Wrap to handle list input
    class Wrapper(nn.Module):
        def __init__(self, m: LoRAPropLightPath):
            super().__init__()
            self.m = m
        def forward(self, f0, f1, f2, dt):
            return self.m([f0, f1, f2], dt)

    wrapper = Wrapper(light).to(dev)

    cpu_inputs = (feats[0].cpu(), feats[1].cpu(), feats[2].cpu(), dt.cpu())
    out_dir = os.path.dirname(save_path)
    os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(wrapper.cpu(), cpu_inputs, save_path,
                      input_names=['f0','f1','f2','dt'], output_names=['eps'], opset_version=17)
    print(f'Exported ONNX: {save_path}')

    # Parity check
    sess = ort.InferenceSession(save_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        'f0': cpu_inputs[0].numpy(),
        'f1': cpu_inputs[1].numpy(),
        'f2': cpu_inputs[2].numpy(),
        'dt': cpu_inputs[3].numpy(),
    })[0]
    with torch.no_grad():
        pt_out = wrapper.cpu()(*cpu_inputs).detach().cpu().numpy()
    mae = float(np.mean(np.abs(ort_out - pt_out)))
    print(f'ONNX parity MAE: {mae:.6e}')
    return {'exported': True, 'mae': mae, 'onnx_path': save_path}
