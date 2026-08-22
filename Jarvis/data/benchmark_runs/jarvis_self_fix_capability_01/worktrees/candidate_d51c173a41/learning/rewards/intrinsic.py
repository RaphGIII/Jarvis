from __future__ import annotations

import torch
from torch import Tensor


def novelty_reward(latent_state: Tensor, known_states: Tensor | None) -> float:
    """Distance to the nearest known latent state."""

    if known_states is None or known_states.numel() == 0:
        return 1.0
    latent = latent_state.reshape(1, -1).float()
    known = known_states.reshape(known_states.shape[0], -1).float()
    distances = torch.cdist(latent, known)
    return float(distances.min().item())


def curiosity_reward(predicted_next: Tensor, actual_next: Tensor) -> float:
    return float(torch.mean((predicted_next.float() - actual_next.float()) ** 2).item())


def learning_progress_reward(old_error: float, new_error: float) -> float:
    return float(max(0.0, old_error - new_error))
