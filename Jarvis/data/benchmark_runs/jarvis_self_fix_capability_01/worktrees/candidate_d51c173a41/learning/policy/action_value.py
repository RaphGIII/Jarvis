from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from learning.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class ActionValueConfig:
    state_dim: int = DEFAULT_CONFIG.latent_dim
    action_dim: int = 32
    hidden_dim: int = 128
    version: str = "action-value-0.2"


class ActionValueNetwork(nn.Module):
    """Concrete action-value model Q(z, a_embedding)."""

    def __init__(self, config: ActionValueConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionValueConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.state_dim + self.config.action_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.training_step = 0

    def forward(self, latent_state: Tensor, action_embedding: Tensor) -> Tensor:
        z = latent_state.float()
        a = action_embedding.float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)
