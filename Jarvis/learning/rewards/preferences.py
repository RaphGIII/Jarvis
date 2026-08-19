from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class PreferenceSample:
    prompt: Any
    chosen: Any
    rejected: Any
    metadata: dict[str, Any] | None = None


def preference_margin_loss(chosen_scores: Tensor, rejected_scores: Tensor, margin: float = 1.0) -> Tensor:
    return torch.relu(margin - (chosen_scores - rejected_scores)).mean()
