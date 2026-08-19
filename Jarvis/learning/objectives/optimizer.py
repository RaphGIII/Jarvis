from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(module: nn.Module, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adam(module.parameters(), lr=learning_rate)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
