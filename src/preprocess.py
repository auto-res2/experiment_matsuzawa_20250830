import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .train import SyntheticPatternDataset, set_seed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_data(train_n: int = 256,
                 val_n: int = 128,
                 image_size: int = 32,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 seed: int = 42) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, DataLoader, DataLoader]:
    set_seed(seed)
    train_ds = SyntheticPatternDataset(n=train_n, image_size=image_size, seed=seed)
    val_ds = SyntheticPatternDataset(n=val_n, image_size=image_size, seed=seed+1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, train_loader, val_loader
