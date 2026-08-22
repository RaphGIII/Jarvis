from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from learning.objectives.optimizer import make_optimizer, set_global_seeds
from learning.policy.policy import NeuralPolicy, PolicyConfig
from learning.policy.value import NeuralValueFunction, ValueConfig


@dataclass(frozen=True)
class PolicyLearningDemoConfig:
    state_dim: int = 4
    num_actions: int = 3
    samples: int = 192
    train_steps: int = 180
    learning_rate: float = 0.03
    seed: int = 7


def _synthetic_rewards(states: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            states[:, 0] - 0.25 * states[:, 2],
            states[:, 1] + 0.10 * states[:, 3],
            -0.5 * states[:, 0] - 0.5 * states[:, 1],
        ],
        dim=1,
    )


def _evaluate(policy: NeuralPolicy, states: torch.Tensor, rewards_by_action: torch.Tensor) -> float:
    with torch.no_grad():
        actions = torch.argmax(policy(states), dim=-1)
        return float(rewards_by_action.gather(1, actions.reshape(-1, 1)).mean().item())


def run_policy_learning_demo(config: PolicyLearningDemoConfig | None = None) -> dict[str, float]:
    cfg = config or PolicyLearningDemoConfig()
    set_global_seeds(cfg.seed)
    states = torch.randn(cfg.samples, cfg.state_dim)
    rewards_by_action = _synthetic_rewards(states)
    best_reward = rewards_by_action.max(dim=1).values

    policy = NeuralPolicy(PolicyConfig(state_dim=cfg.state_dim, num_actions=cfg.num_actions, hidden_dim=32))
    value = NeuralValueFunction(ValueConfig(state_dim=cfg.state_dim, hidden_dim=32))
    optimizer = make_optimizer(torch.nn.ModuleList([policy, value]), cfg.learning_rate)

    before_reward = _evaluate(policy, states, rewards_by_action)
    value_loss_start = None
    first_parameters = [parameter.detach().clone() for parameter in policy.parameters()]

    for _ in range(cfg.train_steps):
        logits = policy(states)
        probabilities = F.softmax(logits, dim=-1)
        expected_reward = (probabilities * rewards_by_action).sum(dim=1).mean()
        value_prediction = value(states)
        value_loss = F.mse_loss(value_prediction, best_reward.detach())
        loss = -expected_reward + 0.2 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy.training_step += 1
        value.training_step += 1
        if value_loss_start is None:
            value_loss_start = float(value_loss.detach().item())

    after_reward = _evaluate(policy, states, rewards_by_action)
    parameter_delta = sum(
        torch.sum(torch.abs(before - after.detach())).item()
        for before, after in zip(first_parameters, policy.parameters())
    )
    return {
        "policy_reward_before": before_reward,
        "policy_reward_after": after_reward,
        "value_loss_before": float(value_loss_start or 0.0),
        "value_loss_after": float(value_loss.detach().item()),
        "policy_parameter_l1_delta": float(parameter_delta),
    }


if __name__ == "__main__":
    print(run_policy_learning_demo())
