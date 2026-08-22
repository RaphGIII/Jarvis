from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from learning.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class PriorityConfig:
    alpha: float = DEFAULT_CONFIG.per_alpha
    beta: float = DEFAULT_CONFIG.per_beta
    epsilon: float = DEFAULT_CONFIG.per_epsilon


def compute_priority(error: float, config: PriorityConfig = PriorityConfig()) -> float:
    return float((abs(error) + config.epsilon) ** config.alpha)


def compute_learning_priority(
    td_error: float,
    prediction_error: float,
    prediction_error_weight: float = 0.5,
    config: PriorityConfig = PriorityConfig(),
) -> float:
    combined_error = abs(td_error) + prediction_error_weight * abs(prediction_error)
    return compute_priority(combined_error, config)


def sampling_probabilities(priorities: Tensor) -> Tensor:
    if priorities.numel() == 0:
        raise ValueError("priorities must not be empty")
    total = priorities.sum()
    if total <= 0:
        return torch.full_like(priorities, 1.0 / priorities.numel(), dtype=torch.float32)
    return priorities.float() / total.float()


def importance_sampling_weights(
    probabilities: Tensor,
    indices: Tensor,
    population_size: int,
    beta: float = DEFAULT_CONFIG.per_beta,
) -> Tensor:
    selected = probabilities[indices].clamp_min(1e-12)
    weights = (population_size * selected).pow(-beta)
    max_weight = weights.max().clamp_min(1e-12)
    return weights / max_weight
