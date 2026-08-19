from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def numeric_observation_to_tensor(observation: Sequence[float] | Tensor) -> Tensor:
    if isinstance(observation, Tensor):
        return observation.float()
    return torch.tensor(list(observation), dtype=torch.float32)


def one_hot(index: int, size: int) -> Tensor:
    if index < 0 or index >= size:
        raise ValueError("index out of one-hot range")
    vector = torch.zeros(size, dtype=torch.float32)
    vector[index] = 1.0
    return vector


def cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    return F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T
