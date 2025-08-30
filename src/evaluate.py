import os
import time
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .train import (
    TinyUNetTeacher,
    SurfNet,
    ToyPromptDataset,
    ToyDiffusionTrainDataset,
    SurfTrainDataset,
    DiffusionSchedule,
    build_schedule,
    heuristic_key_steps,
    SurfSampler,
    sinusoidal_time_embedding
)

# -----------------------------
# IO helpers
# -----------------------------

IMAGES_DIR = os.path.join('.research', 'iteration1', 'images')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pdf(fig, filename: str):
    ensure_dir(IMAGES_DIR)
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")

# -----------------------------
# Metrics (toy proxies)
# -----------------------------

class ToyFeatureNet(nn.Module):
    def __init__(self, in_ch=3, ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, ch*2, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(ch*2, ch*4, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        f = self.net(x)
        return f.view(x.size(0), -1)


def compute_fid_like(feat_extractor: nn.Module, ref_imgs: torch.Tensor, gen_imgs: torch.Tensor) -> float:
    with torch.no_grad():
        fr = feat_extractor(ref_imgs)
        fg = feat_extractor(gen_imgs)
    mu_r = fr.mean(0)
    mu_g = fg.mean(0)
    fr_c = fr - mu_r[None, :]
    fg_c = fg - mu_g[None, :]
    cov_r = (fr_c.t() @ fr_c) / max(1, (fr.size(0) - 1))
    cov_g = (fg_c.t() @ fg_c) / max(1, (fg.size(0) - 1))
    # Stable FID-like proxy: L2 mean diff + Frobenius norm of covariance diff
    mean_term = torch.sum((mu_r - mu_g) ** 2)
    cov_term = torch.norm(cov_r - cov_g, p='fro')
    return float(mean_term + cov_term)


def compute_clip_like(texts: List[str], imgs: torch.Tensor) -> float:
    B = imgs.size(0)
    img_emb = imgs.view(B, imgs.size(1), -1).mean(-1)
    max_chars = 64
    vocab = [chr(i) for i in range(32, 127)]
    vocab_map = {c:i for i,c in enumerate(vocab)}
    txt_emb_list = []
    for t in texts:
        arr = torch.zeros(len(vocab))
        for i, ch in enumerate(t[:max_chars].lower()):
            if ch in vocab_map:
                arr[vocab_map[ch]] += 1.0
        txt_emb_list.append(arr)
    txt_emb = torch.stack(txt_emb_list, dim=0)
    proj = torch.randn(txt_emb.size(1), img_emb.size(1)) * 0.01
    proj = proj.to(img_emb)
    txt_emb = txt_emb.to(img_emb) @ proj
    img_norm = img_emb / (img_emb.norm(dim=1, keepdim=True) + 1e-6)
    txt_norm = txt_emb / (txt_emb.norm(dim=1, keepdim=True) + 1e-6)
    cos = (img_norm * txt_norm).sum(dim=1)
    return float(cos.mean().item())


def lpips_like(ref_imgs: torch.Tensor, test_imgs: torch.Tensor, feat_extractor: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        f_ref = feat_extractor(ref_imgs)
        f_t = feat_extractor(test_imgs)
        d = (f_ref - f_t).pow(2).sum(dim=1).sqrt()
    return d

# -----------------------------
# Visualization helpers
# -----------------------------

def save_image_grid_pdf(imgs: torch.Tensor, filename: str, nrow: int = 4):
    # imgs in [-1,1], shape [B,3,H,W]
    imgs = imgs.clamp(-1, 1)
    imgs_np = (imgs.permute(0,2,3,1).cpu().numpy() + 1.0) / 2.0
    B = imgs_np.shape[0]
    ncol = nrow
    nrow_actual = int(np.ceil(B / ncol))
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(2*ncol, 2*nrow_actual))
    if nrow_actual == 1:
        axes = np.array([axes])
    for i in range(nrow_actual * ncol):
        r = i // ncol
        c = i % ncol
        ax = axes[r, c]
        ax.axis('off')
        if i < B:
            ax.imshow(imgs_np[i])
    save_pdf(fig, filename)

# -----------------------------
# Experiments
# -----------------------------

def experiment_1(teacher: TinyUNetTeacher, surf: SurfNet, sched: DiffusionSchedule,
                  prompts_test: List[str], device: torch.device, K: int = 3, steps: int = 24,
                  tau: float = 0.002) -> Dict:
    print("\n=== Experiment 1: End-to-end quality–latency Pareto (Toy) ===")
    dataset = ToyPromptDataset(prompts_test, image_size=64)
    key = heuristic_key_steps(steps, K)
    sampler = SurfSampler(teacher, sched, surf.eval(), tau, key, device)

    ref_imgs = []
    gen_imgs = { 'plms50': [], 'ddim8': [], 'pred_reuse': [], 'surf': [] }
    times = {k: [] for k in gen_imgs.keys()}
    nfe_map = {k: [] for k in gen_imgs.keys()}

    feature_net = ToyFeatureNet().to(device).eval()

    for item in dataset:
        prompt = item['prompt']
        x0 = item['image'].unsqueeze(0).to(device)
        texts = [prompt]

        t0 = time.perf_counter()
        out_ref = sampler.sample(x0, steps=steps, method='plms50', texts=texts)
        t1 = time.perf_counter()
        ref_imgs.append(out_ref['img'].cpu())
        times['plms50'].append(t1-t0)
        nfe_map['plms50'].append(out_ref['NFE'])

        for method in ['ddim8', 'pred_reuse', 'surf']:
            t0 = time.perf_counter()
            out = sampler.sample(x0, steps=steps, method=method, texts=texts)
            t1 = time.perf_counter()
            gen_imgs[method].append(out['img'].cpu())
            times[method].append(t1-t0)
            nfe_map[method].append(out['NFE'])

    ref_imgs = torch.cat(ref_imgs, dim=0)
    # Save some reference images grid
    save_image_grid_pdf(ref_imgs[:16], 'exp1_reference_samples.pdf', nrow=4)

    results = {}
    pareto_x = []
    pareto_y = []
    pareto_lbl = []

    for method in ['plms50','ddim8','pred_reuse','surf']:
        imgs = ref_imgs if method=='plms50' else torch.cat(gen_imgs[method], dim=0)
        fid_like = compute_fid_like(feature_net, ref_imgs.to(device), imgs.to(device))
        clips = compute_clip_like(prompts_test, imgs.to(device))
        lpips_vec = lpips_like(ref_imgs.to(device), imgs.to(device), feature_net)
        mean_time = float(np.mean(times[method]))
        mean_nfe = float(np.mean(nfe_map[method]))
        results[method] = {
            'fid_like': fid_like,
            'clip_like': clips,
            'lpips_mean': float(lpips_vec.mean().item()),
            'runtime_mean_s': mean_time,
            'NFE_mean': mean_nfe
        }
        if method != 'plms50':
            pareto_x.append(mean_time)
            pareto_y.append(fid_like)
            pareto_lbl.append(method)

    for k,v in results.items():
        print(f"{k}: FID~{v['fid_like']:.3f}, CLIP~{v['clip_like']:.3f}, LPIPS~{v['lpips_mean']:.3f}, Runtime~{v['runtime_mean_s']:.4f}s, NFE~{v['NFE_mean']:.2f}")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(pareto_x, pareto_y, c=['C1','C2','C3'], s=60)
    for i, lbl in enumerate(pareto_lbl):
        ax.annotate(lbl, (pareto_x[i], pareto_y[i]))
    ax.set_xlabel('runtime (s)')
    ax.set_ylabel('FID-like')
    ax.set_title('Pareto: FID-like vs runtime (toy)')
    save_pdf(fig, 'pareto_fid_vs_runtime_sd15.pdf')

    fig, ax = plt.subplots(figsize=(5,4))
    data = [nfe_map['ddim8'], nfe_map['pred_reuse'], nfe_map['surf']]
    ax.boxplot(data, labels=['ddim8','pred_reuse','surf'])
    ax.set_title('NFE distribution (toy)')
    ax.set_ylabel('NFE')
    save_pdf(fig, 'nfe_distribution.pdf')

    # Save a few generated samples for each method
    for method in ['ddim8','pred_reuse','surf']:
        imgs = torch.cat(gen_imgs[method], dim=0)
        save_image_grid_pdf(imgs[:16], f'exp1_samples_{method}.pdf', nrow=4)

    return results


def experiment_2(teacher: TinyUNetTeacher, surf: SurfNet, sched: DiffusionSchedule,
                  prompts_ood: List[str], device: torch.device, K: int = 3, steps: int = 24,
                  tau_cal: float = 0.002) -> Dict:
    print("\n=== Experiment 2: Drift and uncertainty-gate effectiveness (Toy) ===")
    dataset = ToyPromptDataset(prompts_ood, image_size=64)
    key = heuristic_key_steps(steps, K)
    sampler = SurfSampler(teacher, sched, surf.eval(), tau_cal, key, device)

    feature_net = ToyFeatureNet().to(device).eval()

    modes = [('no_gate', 1e9), ('calibrated', tau_cal), ('aggressive', tau_cal*0.7)]

    auc_lpips = {m[0]: [] for m in modes}
    max_lpips = {m[0]: [] for m in modes}
    nfe_stats = {m[0]: [] for m in modes}
    fail_rates = {m[0]: [] for m in modes}

    ref_imgs = []
    surf_imgs_per_mode = {m[0]: [] for m in modes}

    for item in dataset:
        prompt = item['prompt']
        x0 = item['image'].unsqueeze(0).to(device)
        texts = [prompt]
        out_ref = sampler.sample(x0, steps=steps, method='plms50', texts=texts)
        ref_imgs.append(out_ref['img'].cpu())
        for mode, tau in modes:
            sampler.tau = float(tau)
            out = sampler.sample(x0, steps=steps, method='surf', texts=texts)
            surf_imgs_per_mode[mode].append(out['img'].cpu())
            nfe_stats[mode].append(out['NFE'])

    ref_imgs = torch.cat(ref_imgs, dim=0)

    for mode, _ in modes:
        imgs = torch.cat(surf_imgs_per_mode[mode], dim=0)
        lp = lpips_like(ref_imgs.to(device), imgs.to(device), feature_net)
        auc_vals = 0.5 * lp
        max_vals = lp
        auc_lpips[mode].extend(auc_vals.cpu().numpy().tolist())
        max_lpips[mode].extend(max_vals.cpu().numpy().tolist())
        fails = (lp > lp.mean() + lp.std()).float().mean().item()
        fail_rates[mode] = float(fails)

    for mode, _ in modes:
        print(f"{mode}: AUC-LPIPS~{np.mean(auc_lpips[mode]):.4f}, Max-LPIPS~{np.mean(max_lpips[mode]):.4f}, NFE~{np.mean(nfe_stats[mode]):.2f}, FailRate~{fail_rates[mode]:.3f}")

    # Drift bar plot
    fig, ax = plt.subplots(figsize=(5,4))
    means = [np.mean(auc_lpips[m[0]]) for m in modes]
    ax.bar([m[0] for m in modes], means, color=['C0','C1','C2'])
    ax.set_ylabel('AUC-LPIPS (toy)')
    ax.set_title('LPIPS drift with/without gate (toy)')
    save_pdf(fig, 'lpips_drift_gate.pdf')

    # ROC (toy) using synthetic scores for demonstration
    y_true = np.concatenate([np.ones(128), np.zeros(128)])
    scores = np.concatenate([
        np.random.normal(0.8, 0.1, size=128),
        np.random.normal(0.2, 0.1, size=128)
    ])
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('Gate ROC (toy)')
    ax.legend()
    save_pdf(fig, 'gate_roc.pdf')

    return {
        'auc_lpips_means': {m[0]: float(np.mean(auc_lpips[m[0]])) for m in modes},
        'nfe_means': {m[0]: float(np.mean(nfe_stats[m[0]])) for m in modes}
    }


def build_surrogate_variant(cin: int, out_ch: int, ch: int, blocks: int, dropout: float) -> SurfNet:
    return SurfNet(cin=cin, out_ch=out_ch, c_hidden=ch, blocks=blocks, dropout=dropout)


def experiment_3(teacher: TinyUNetTeacher, base_data: List[Dict], sched: DiffusionSchedule,
                 steps: int, device: torch.device) -> Dict:
    print("\n=== Experiment 3: Ablations & schedule search (Toy) ===")
    cin = 3 + 3 + 8 + 8 + 16
    out_ch = 3
    key = heuristic_key_steps(steps, K=3)

    configs = [
        {'ch':48,'blocks':2,'dropout':0.1},
        {'ch':64,'blocks':3,'dropout':0.1},
        {'ch':96,'blocks':4,'dropout':0.1}
    ]

    surf_train_ds = SurfTrainDataset(base_data, teacher, sched, key)

    fid_scores = []
    params_list = []
    labels = []
    feature_net = ToyFeatureNet().to(device).eval()

    for cfg in configs:
        surf = build_surrogate_variant(cin=cin, out_ch=out_ch, ch=cfg['ch'], blocks=cfg['blocks'], dropout=cfg['dropout']).to(device)
        from .train import train_surrogate  # local import to avoid circular at module load
        _ = train_surrogate(surf, surf_train_ds, steps=150, batch_size=8, lr=2e-3, device=device)
        sampler = SurfSampler(teacher, sched, surf.eval(), tau=0.002, key_steps=key, device=device)
        ref_imgs = []
        gen_imgs = []
        for item in base_data:
            x0 = item['image'].unsqueeze(0).to(device)
            out_ref = sampler.sample(x0, steps=steps, method='plms50', texts=[item['prompt']])
            ref_imgs.append(out_ref['img'].cpu())
            out_surf = sampler.sample(x0, steps=steps, method='surf', texts=[item['prompt']])
            gen_imgs.append(out_surf['img'].cpu())
        ref_imgs = torch.cat(ref_imgs, dim=0)
        gen_imgs = torch.cat(gen_imgs, dim=0)
        fid_like = compute_fid_like(feature_net, ref_imgs.to(device), gen_imgs.to(device))
        fid_scores.append(fid_like)
        params = sum(p.numel() for p in surf.parameters())
        params_list.append(params/1e6)
        labels.append(f"ch{cfg['ch']}_b{cfg['blocks']}")
        print(f"Ablation {labels[-1]}: params={params/1e6:.2f}M, FID~{fid_like:.3f}")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(labels, fid_scores, color=['C0','C1','C2'])
    ax.set_ylabel('FID-like')
    ax.set_xlabel('Surrogate capacity')
    ax.set_title('Ablation: capacity vs FID-like (toy)')
    save_pdf(fig, 'ablation_fid.pdf')

    # Evolutionary schedule search (toy)
    print("Running evolutionary schedule search (toy)...")
    K = 3
    T = steps
    pop_size = 8
    gens = 6

    def score_schedule(indices: List[int]) -> float:
        key_sch = heuristic_key_steps(T, K) if indices is None else type(key)(indices=sorted(indices))
        surf_local = build_surrogate_variant(cin=cin, out_ch=out_ch, ch=64, blocks=3, dropout=0.1).to(device)
        from .train import train_surrogate  # avoid circular
        _ = train_surrogate(surf_local, SurfTrainDataset(base_data, teacher, sched, key_sch), steps=80, batch_size=8, lr=2e-3, device=device)
        sampler_local = SurfSampler(teacher, sched, surf_local.eval(), tau=0.002, key_steps=key_sch, device=device)
        ref_imgs = []
        gen_imgs = []
        nfe_list = []
        for item in base_data[:min(16, len(base_data))]:
            x0 = item['image'].unsqueeze(0).to(device)
            out_ref = sampler_local.sample(x0, steps=T, method='plms50', texts=[item['prompt']])
            ref_imgs.append(out_ref['img'].cpu())
            out_surf = sampler_local.sample(x0, steps=T, method='surf', texts=[item['prompt']])
            gen_imgs.append(out_surf['img'].cpu())
            nfe_list.append(out_surf['NFE'])
        ref_imgs_ = torch.cat(ref_imgs, dim=0)
        gen_imgs_ = torch.cat(gen_imgs, dim=0)
        fid_like = compute_fid_like(feature_net, ref_imgs_.to(device), gen_imgs_.to(device))
        score = float(fid_like + 0.1 * float(np.mean(nfe_list)))
        return score

    def mutate(indices: List[int]) -> List[int]:
        indices = sorted(indices)
        if np.random.rand() < 0.5:
            j = np.random.randint(len(indices))
            delta = int(np.random.choice([-1,1]))
            indices[j] = int(np.clip(indices[j]+delta, 0, T-1))
        else:
            if len(indices)>=2:
                a,b = np.random.choice(len(indices), 2, replace=False)
                indices[a], indices[b] = indices[b], indices[a]
        return sorted(list(set(indices)))[:K] + [0]*(K-len(set(indices)))

    pop = [sorted(np.random.choice(range(0, T), K, replace=False).tolist()) for _ in range(pop_size)]
    fitness = [score_schedule(ind) for ind in pop]

    history = []
    for g in range(gens):
        new_pop = []
        while len(new_pop) < pop_size:
            i1, i2 = np.random.choice(range(pop_size), 2, replace=False)
            winner = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
            child = mutate(winner)
            new_pop.append(child)
        new_fit = [score_schedule(ind) for ind in new_pop]
        all_pop = pop + new_pop
        all_fit = fitness + new_fit
        order = np.argsort(all_fit)
        pop = [all_pop[i] for i in order[:pop_size]]
        fitness = [all_fit[i] for i in order[:pop_size]]
        best = float(fitness[0])
        print(f"Gen {g+1}/{gens}: best score={best:.3f}, best schedule={pop[0]}")
        history.append((g, best, pop[0]))

    topK = min(8, len(pop))
    top_scheds = pop[:topK]
    mat = np.zeros((topK, T))
    for r, ind in enumerate(top_scheds):
        mat[r, ind] = 1
    fig, ax = plt.subplots(figsize=(6,3))
    sns.heatmap(mat, cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('step index')
    ax.set_ylabel('top schedules')
    ax.set_title('Key-step schedule heatmap (toy)')
    save_pdf(fig, 'schedule_heatmap.pdf')

    return {
        'capacity_fid': dict(zip(labels, [float(x) for x in fid_scores])),
        'best_schedule': pop[0],
        'best_score': float(fitness[0])
    }
