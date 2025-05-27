import random
import numpy as np
import torch
from torch.nn.modules import padding
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_dataloader(cfg: dict):
    ds_cfg = cfg['dataset']

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    )

    train_ds = getattr(datasets, ds_cfg['name'])(
        root='./data', train=True, download=True, transform=transform
    )
    test_ds = getattr(datasets, ds_cfg['name'])(
        root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )

    train_dataloader = DataLoader(train_ds, batch_size=ds_cfg['train_batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=ds_cfg['test_batch_size'], shuffle=False)

    return train_dataloader, test_dataloader
