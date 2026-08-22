from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from learning.checkpointing import load_module_checkpoint, save_module_checkpoint
from learning.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class ValueConfig:
    state_dim: int = DEFAULT_CONFIG.latent_dim
    hidden_dim: int = 128
    version: str = "value-function-0.1"


class NeuralValueFunction(nn.Module):
    """Small trainable value function V_phi(z)."""

    def __init__(self, config: ValueConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.training_step = 0

    def forward(self, latent_state: Tensor) -> Tensor:
        z = latent_state.float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z).squeeze(-1)

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        return save_module_checkpoint(
            self,
            path,
            version=self.config.version,
            training_step=self.training_step,
            metrics=metrics,
            optimizer=optimizer,
            extra={"config": self.config.__dict__},
        )

    def load_checkpoint(self, path: str | Path, optimizer: torch.optim.Optimizer | None = None) -> dict:
        payload = load_module_checkpoint(self, path, optimizer=optimizer)
        self.training_step = int(payload.get("training_step", 0))
        return payload


@dataclass(frozen=True)
class QNetworkConfig:
    state_dim: int = DEFAULT_CONFIG.latent_dim
    num_actions: int = DEFAULT_CONFIG.action_dim
    hidden_dim: int = 128
    version: str = "q-network-0.1"


class QNetwork(nn.Module):
    """Small trainable action-value model Q_phi(z, a)."""

    def __init__(self, config: QNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or QNetworkConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.num_actions),
        )
        self.training_step = 0

    def forward(self, latent_state: Tensor) -> Tensor:
        z = latent_state.float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z)

    def q_value(self, latent_state: Tensor, action_index: int) -> Tensor:
        return self.forward(latent_state)[..., action_index]


def bellman_target(
    reward: Tensor,
    next_value: Tensor,
    done: Tensor,
    discount: float = DEFAULT_CONFIG.discount_factor,
) -> Tensor:
    return reward.float() + discount * next_value.float() * (1.0 - done.float())


def td_error(
    reward: Tensor,
    value: Tensor,
    next_value: Tensor,
    done: Tensor,
    discount: float = DEFAULT_CONFIG.discount_factor,
) -> Tensor:
    return bellman_target(reward, next_value, done, discount) - value.float()
