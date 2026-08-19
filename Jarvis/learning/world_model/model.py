from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from learning.checkpointing import load_module_checkpoint, save_module_checkpoint
from learning.config import DEFAULT_CONFIG
from learning.objectives.losses import mse_loss


@dataclass(frozen=True)
class WorldModelConfig:
    latent_dim: int = DEFAULT_CONFIG.latent_dim
    action_dim: int = DEFAULT_CONFIG.action_dim
    hidden_dim: int = 128
    transition_loss_weight: float = DEFAULT_CONFIG.world_transition_weight
    reward_loss_weight: float = DEFAULT_CONFIG.world_reward_weight
    version: str = "world-model-0.1"


@dataclass
class WorldModelOutput:
    next_latent_pred: Tensor
    reward_pred: Tensor | None = None


class WorldModel(nn.Module):
    """Trainable model p(z_(t+1), r_t | z_t, a_t)."""

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        input_dim = self.config.latent_dim + self.config.action_dim
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.training_step = 0

    def forward(self, latent_state: Tensor, action_embedding: Tensor) -> WorldModelOutput:
        z = latent_state.float()
        a = action_embedding.float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        features = torch.cat([z, a], dim=-1)
        return WorldModelOutput(
            next_latent_pred=self.transition_net(features),
            reward_pred=self.reward_head(features).squeeze(-1),
        )

    def loss(
        self,
        latent_state: Tensor,
        action_embedding: Tensor,
        next_latent_actual: Tensor,
        reward_actual: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        output = self.forward(latent_state, action_embedding)
        transition_loss = mse_loss(output.next_latent_pred, next_latent_actual)
        reward_loss = torch.tensor(0.0, device=transition_loss.device)
        if reward_actual is not None and output.reward_pred is not None:
            reward_loss = mse_loss(output.reward_pred, reward_actual.float().reshape(-1))
        total = (
            self.config.transition_loss_weight * transition_loss
            + self.config.reward_loss_weight * reward_loss
        )
        return total, {
            "transition_loss": float(transition_loss.detach().cpu().item()),
            "reward_loss": float(reward_loss.detach().cpu().item()),
            "world_loss": float(total.detach().cpu().item()),
        }

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        latent_state: Tensor,
        action_embedding: Tensor,
        next_latent_actual: Tensor,
        reward_actual: Tensor | None = None,
    ) -> dict[str, float]:
        optimizer.zero_grad()
        loss, metrics = self.loss(latent_state, action_embedding, next_latent_actual, reward_actual)
        loss.backward()
        optimizer.step()
        self.training_step += 1
        metrics["training_step"] = float(self.training_step)
        return metrics

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
