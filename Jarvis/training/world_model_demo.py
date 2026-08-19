from __future__ import annotations

from dataclasses import dataclass

import torch

from learning.objectives.optimizer import make_optimizer, set_global_seeds
from learning.representations.embeddings import one_hot
from learning.world_model.model import WorldModel, WorldModelConfig


@dataclass(frozen=True)
class WorldModelDemoConfig:
    latent_dim: int = 5
    action_dim: int = 3
    samples: int = 192
    train_steps: int = 220
    learning_rate: float = 0.02
    seed: int = 11


def _synthetic_transitions(config: WorldModelDemoConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.randn(config.samples, config.latent_dim)
    action_indices = torch.randint(0, config.action_dim, (config.samples,))
    action_embeddings = torch.stack([one_hot(int(index), config.action_dim) for index in action_indices])
    effects = torch.tensor(
        [
            [0.3, 0.0, -0.1, 0.0, 0.2],
            [0.0, -0.4, 0.2, 0.1, 0.0],
            [-0.2, 0.2, 0.0, -0.3, 0.1],
        ],
        dtype=torch.float32,
    )[: config.action_dim, : config.latent_dim]
    next_states = states + effects[action_indices]
    rewards = next_states.sum(dim=1) * 0.1
    return states, action_embeddings, next_states, rewards


def run_world_model_demo(config: WorldModelDemoConfig | None = None) -> dict[str, float]:
    cfg = config or WorldModelDemoConfig()
    set_global_seeds(cfg.seed)
    states, actions, next_states, rewards = _synthetic_transitions(cfg)
    model = WorldModel(WorldModelConfig(latent_dim=cfg.latent_dim, action_dim=cfg.action_dim, hidden_dim=48))
    optimizer = make_optimizer(model, cfg.learning_rate)

    before_loss, before_metrics = model.loss(states, actions, next_states, rewards)
    for _ in range(cfg.train_steps):
        metrics = model.train_step(optimizer, states, actions, next_states, rewards)
    after_loss, after_metrics = model.loss(states, actions, next_states, rewards)
    return {
        "world_loss_before": float(before_loss.detach().item()),
        "world_loss_after": float(after_loss.detach().item()),
        "transition_loss_before": before_metrics["transition_loss"],
        "transition_loss_after": after_metrics["transition_loss"],
        "reward_loss_before": before_metrics["reward_loss"],
        "reward_loss_after": after_metrics["reward_loss"],
        "training_steps": float(metrics["training_step"]),
    }


if __name__ == "__main__":
    print(run_world_model_demo())
