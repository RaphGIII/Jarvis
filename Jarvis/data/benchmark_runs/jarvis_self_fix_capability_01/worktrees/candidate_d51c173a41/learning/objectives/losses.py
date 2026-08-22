from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from learning.config import DEFAULT_CONFIG


def mse_loss(predicted: Tensor, target: Tensor) -> Tensor:
    return torch.mean((predicted.float() - target.float()) ** 2)


def info_nce_loss(
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor | None = None,
    temperature: float = DEFAULT_CONFIG.info_nce_temperature,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    anchors = F.normalize(anchors.float(), dim=-1)
    positives = F.normalize(positives.float(), dim=-1)
    if negatives is None:
        candidates = positives
        logits = anchors @ candidates.T / temperature
        labels = torch.arange(anchors.shape[0], device=anchors.device)
        return F.cross_entropy(logits, labels)

    negatives = F.normalize(negatives.float(), dim=-1)
    positive_logits = torch.sum(anchors * positives, dim=-1, keepdim=True) / temperature
    negative_logits = anchors @ negatives.T / temperature
    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
    return F.cross_entropy(logits, labels)
