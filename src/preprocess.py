import os
import json
from typing import Dict, List, Tuple

from .train import ToyPromptDataset

DATA_DIR = 'data'


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_prompts(config: Dict) -> Tuple[List[str], List[str]]:
    base_prompts = config.get('base_prompts', [
        "A vivid landscape with mountains and river",
        "A portrait of a person with detailed face",
        "City skyline at night panorama",
        "Abstract art with many shapes complex",
        "Negative lowres noisy image",
    ])
    repeats = int(config.get('train_repeats', 8))
    train_prompts = base_prompts * repeats
    ood_prompts = config.get('ood_prompts', [
        "High detail panorama aspect ratio 32:9 city skyline at night",
        "long complex prompt with many many many tokens and entities that are rare and unusual",
        "canny edge control depth guidance mountain scene",
        "portrait face person depth control",
        "negative noisy lowres strange artifact"
    ])
    return train_prompts, ood_prompts


def prepare_data(config: Dict) -> Dict:
    ensure_dir(DATA_DIR)
    train_prompts, ood_prompts = build_prompts(config)
    image_size = int(config.get('image_size', 64))

    # Save prompts for reproducibility
    prompts_info = {
        'train_prompts': train_prompts,
        'ood_prompts': ood_prompts,
        'image_size': image_size
    }
    with open(os.path.join(DATA_DIR, 'prompts.json'), 'w') as f:
        json.dump(prompts_info, f, indent=2)

    # Build in-memory datasets (toy)
    train_ds = ToyPromptDataset(train_prompts, image_size=image_size)
    base_data = [train_ds[i] for i in range(len(train_ds))]

    return {
        'base_data': base_data,
        'train_prompts': train_prompts,
        'ood_prompts': ood_prompts
    }
