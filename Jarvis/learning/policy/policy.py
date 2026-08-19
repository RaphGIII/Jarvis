from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from learning.checkpointing import load_module_checkpoint, save_module_checkpoint
from learning.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class PolicyConfig:
    state_dim: int = DEFAULT_CONFIG.latent_dim
    num_actions: int = DEFAULT_CONFIG.action_dim
    hidden_dim: int = 128
    version: str = "neural-policy-0.1"


class NeuralPolicy(nn.Module):
    """Small trainable policy pi_theta(a | z)."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
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

    def action_distribution(self, latent_state: Tensor) -> Tensor:
        return F.softmax(self.forward(latent_state), dim=-1)

    def select_action(self, latent_state: Tensor) -> int:
        probabilities = self.action_distribution(latent_state)
        return int(torch.distributions.Categorical(probabilities).sample()[0].item())

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


def policy_gradient_loss(logits: Tensor, actions: Tensor, rewards: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.long().reshape(-1, 1)).squeeze(1)
    centered_rewards = rewards.float()
    if centered_rewards.numel() > 1:
        centered_rewards = centered_rewards - centered_rewards.mean()
    return -(selected_log_probs * centered_rewards).mean()
