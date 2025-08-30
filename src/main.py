import os
import json
from typing import Dict

import torch
import yaml

from .train import (
    seed_all,
    TinyUNetTeacher,
    SurfNet,
    DiffusionSchedule,
    build_schedule,
    ToyDiffusionTrainDataset,
    SurfTrainDataset,
    heuristic_key_steps,
    train_teacher,
    train_surrogate,
    save_checkpoint,
    export_surrogate_torchscript,
)
from .preprocess import prepare_data
from .evaluate import experiment_1, experiment_2, experiment_3


def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    cfg_path = os.path.join('config', 'experiment.yaml')
    cfg = load_config(cfg_path)

    seed = int(cfg.get('seed', 123))
    seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device', 'auto') != 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    prep = prepare_data(cfg)
    base_data = prep['base_data']
    train_prompts = prep['train_prompts']
    ood_prompts = prep['ood_prompts']

    # Build schedule
    steps = int(cfg.get('steps', 24))
    sched: DiffusionSchedule = build_schedule(steps)

    # Teacher
    teacher = TinyUNetTeacher(in_ch=3, base_ch=32, t_emb_dim=64).to(device)
    teacher_ds = ToyDiffusionTrainDataset(base_data, sched)

    t_steps = int(cfg.get('train', {}).get('teacher_steps', 200))
    t_bs = int(cfg.get('train', {}).get('teacher_batch', 8))
    lr = float(cfg.get('train', {}).get('lr', 2e-3))
    print("Training teacher...")
    _ = train_teacher(teacher, sched, teacher_ds, steps=t_steps, batch_size=t_bs, lr=lr, device=device)
    save_checkpoint(teacher, os.path.join('models', 'teacher_toy.pth'))

    # Surrogate
    K = int(cfg.get('K', 3))
    key = heuristic_key_steps(steps, K)
    # Inputs: x(3)+eps_prev(3)+e0(8)+e1(8)+t(16) = 38
    cin = 3 + 3 + 8 + 8 + 16
    surf = SurfNet(cin=cin, out_ch=3, c_hidden=int(cfg.get('surrogate', {}).get('hidden', 64)), blocks=int(cfg.get('surrogate', {}).get('blocks', 3)), dropout=float(cfg.get('surrogate', {}).get('dropout', 0.1))).to(device)
    surf_ds = SurfTrainDataset(base_data, teacher, sched, key)

    s_steps = int(cfg.get('train', {}).get('surrogate_steps', 150))
    s_bs = int(cfg.get('train', {}).get('surrogate_batch', 8))
    print("Training surrogate...")
    _ = train_surrogate(surf, surf_ds, steps=s_steps, batch_size=s_bs, lr=lr, device=device)
    os.makedirs('models', exist_ok=True)
    torch.save(surf.state_dict(), os.path.join('models', 'surfnet_toy.pth'))

    # Export TorchScript
    example_inp = torch.randn(1, cin, int(cfg.get('image_size', 64)), int(cfg.get('image_size', 64))).to(device)
    _ = export_surrogate_torchscript(surf, example_inp, path=os.path.join('models', 'surfnet_toy.ts'))

    # Experiments
    tau = float(cfg.get('tau', 0.002))

    results = {}
    if cfg.get('evaluation', {}).get('run_exp1', True):
        res1 = experiment_1(teacher, surf, sched, train_prompts[:5], device=device, K=K, steps=steps, tau=tau)
        results['exp1'] = res1
    if cfg.get('evaluation', {}).get('run_exp2', True):
        res2 = experiment_2(teacher, surf, sched, ood_prompts[:5], device=device, K=K, steps=steps, tau_cal=tau)
        results['exp2'] = res2
    if cfg.get('evaluation', {}).get('run_exp3', True):
        # Use a subset for speed
        res3 = experiment_3(teacher, base_data[:24], sched, steps=steps, device=device)
        results['exp3'] = res3

    # Save results JSON
    os.makedirs(os.path.join('.research', 'iteration1'), exist_ok=True)
    with open(os.path.join('.research', 'iteration1', 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nExperiment summaries saved to .research/iteration1/results.json")


if __name__ == '__main__':
    main()
