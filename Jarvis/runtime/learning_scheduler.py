from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from learning.config import DEFAULT_CONFIG
from learning.experience.replay_buffer import ReplayBuffer
from learning.objectives.optimizer import make_optimizer
from learning.policy.policy import NeuralPolicy
from learning.policy.value import NeuralValueFunction, bellman_target
from learning.world_model.model import WorldModel


@dataclass
class LearningSchedulerConfig:
    world_model_train_every_n_steps: int = DEFAULT_CONFIG.world_model_train_every_n_steps
    world_model_batch_size: int = DEFAULT_CONFIG.world_model_batch_size
    world_model_lr: float = DEFAULT_CONFIG.world_model_lr
    value_policy_train_every_n_steps: int = DEFAULT_CONFIG.value_policy_train_every_n_steps
    value_policy_batch_size: int = DEFAULT_CONFIG.value_policy_batch_size
    value_lr: float = DEFAULT_CONFIG.value_lr
    policy_lr: float = DEFAULT_CONFIG.policy_lr
    discount_factor: float = DEFAULT_CONFIG.discount_factor
    priority_prediction_error_weight: float = DEFAULT_CONFIG.priority_prediction_error_weight


@dataclass
class TrainingReport:
    did_update: bool = False
    world_loss: float | None = None
    transition_loss: float | None = None
    reward_loss: float | None = None
    value_loss: float | None = None
    policy_loss: float | None = None
    mean_td_error: float | None = None
    mean_prediction_error: float | None = None
    gradient_norm: float = 0.0
    replay_size: int = 0
    updated_modules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int | bool | str]:
        result: dict[str, float | int | bool | str] = {
            "did_update": self.did_update,
            "gradient_norm": self.gradient_norm,
            "replay_size": self.replay_size,
            "updated_modules": ",".join(self.updated_modules),
        }
        for key in [
            "world_loss",
            "transition_loss",
            "reward_loss",
            "value_loss",
            "policy_loss",
            "mean_td_error",
            "mean_prediction_error",
        ]:
            value = getattr(self, key)
            if value is not None:
                result[key] = float(value)
        return result


class LearningScheduler:
    """Medium-timescale trainer for world model, value and policy."""

    def __init__(
        self,
        world_model: WorldModel,
        policy: NeuralPolicy,
        value_function: NeuralValueFunction,
        config: LearningSchedulerConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.policy = policy
        self.value_function = value_function
        self.config = config or LearningSchedulerConfig()
        self.world_optimizer = make_optimizer(world_model, self.config.world_model_lr)
        self.policy_optimizer = make_optimizer(policy, self.config.policy_lr)
        self.value_optimizer = make_optimizer(value_function, self.config.value_lr)
        self.runtime_steps = 0

    def maybe_train(self, replay_buffer: ReplayBuffer, train_mode: bool = True) -> TrainingReport:
        self.runtime_steps += 1
        report = TrainingReport(replay_size=len(replay_buffer))
        if not train_mode:
            return report
        modules = []
        gradient_norm = 0.0

        if self._should_train_world(replay_buffer):
            world_metrics, world_grad_norm, world_errors, world_indices = self._train_world(replay_buffer)
            report.world_loss = world_metrics["world_loss"]
            report.transition_loss = world_metrics["transition_loss"]
            report.reward_loss = world_metrics["reward_loss"]
            report.mean_prediction_error = float(world_errors.mean().item())
            gradient_norm += world_grad_norm
            modules.append("WorldModel")
        else:
            world_errors = None
            world_indices = None

        if self._should_train_value_policy(replay_buffer):
            policy_metrics, policy_grad_norm, td_errors, indices = self._train_value_policy(replay_buffer)
            report.value_loss = policy_metrics["value_loss"]
            report.policy_loss = policy_metrics["policy_loss"]
            report.mean_td_error = float(td_errors.abs().mean().item())
            gradient_norm += policy_grad_norm
            modules.extend(["ValueFunction", "Policy"])
            prediction_errors = torch.zeros_like(td_errors)
            if world_errors is not None and world_indices == indices:
                prediction_errors = world_errors
            combined_errors = [
                abs(float(td)) + self.config.priority_prediction_error_weight * abs(float(prediction))
                for td, prediction in zip(td_errors.detach().cpu(), prediction_errors.detach().cpu())
            ]
            replay_buffer.update_priorities(indices, combined_errors)

        report.did_update = bool(modules)
        report.gradient_norm = float(gradient_norm)
        report.updated_modules = modules
        report.replay_size = len(replay_buffer)
        return report

    def _should_train_world(self, replay_buffer: ReplayBuffer) -> bool:
        return (
            len(replay_buffer) >= self.config.world_model_batch_size
            and self.runtime_steps % self.config.world_model_train_every_n_steps == 0
        )

    def _should_train_value_policy(self, replay_buffer: ReplayBuffer) -> bool:
        return (
            len(replay_buffer) >= self.config.value_policy_batch_size
            and self.runtime_steps % self.config.value_policy_train_every_n_steps == 0
        )

    def _train_world(self, replay_buffer: ReplayBuffer) -> tuple[dict[str, float], float, torch.Tensor, list[int]]:
        batch = replay_buffer.sample_priority(self.config.world_model_batch_size)
        latent, action_embeddings, next_latent, rewards, _, _ = self._batch_tensors(batch.transitions)
        self.world_optimizer.zero_grad()
        loss, metrics = self.world_model.loss(latent, action_embeddings, next_latent, rewards)
        loss.backward()
        grad_norm = self._gradient_norm(self.world_model.parameters())
        self.world_optimizer.step()
        self.world_model.training_step += 1
        with torch.no_grad():
            output = self.world_model(latent, action_embeddings)
            prediction_errors = torch.mean((output.next_latent_pred - next_latent) ** 2, dim=1)
        return metrics, grad_norm, prediction_errors, batch.indices

    def _train_value_policy(self, replay_buffer: ReplayBuffer) -> tuple[dict[str, float], float, torch.Tensor, list[int]]:
        batch = replay_buffer.sample_priority(self.config.value_policy_batch_size)
        latent, _, next_latent, rewards, dones, actions = self._batch_tensors(batch.transitions)

        with torch.no_grad():
            next_value = self.value_function(next_latent)
            target = bellman_target(rewards, next_value, dones, self.config.discount_factor)

        value_prediction = self.value_function(latent)
        td_errors = target - value_prediction
        value_loss = torch.mean(td_errors.pow(2))

        logits = self.policy(latent)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.long().reshape(-1, 1)).squeeze(1)
        policy_loss = -(selected_log_probs * td_errors.detach()).mean()

        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        total_loss = value_loss + policy_loss
        total_loss.backward()
        grad_norm = self._gradient_norm(self.value_function.parameters()) + self._gradient_norm(self.policy.parameters())
        self.value_optimizer.step()
        self.policy_optimizer.step()
        self.value_function.training_step += 1
        self.policy.training_step += 1

        return (
            {"value_loss": float(value_loss.detach().item()), "policy_loss": float(policy_loss.detach().item())},
            grad_norm,
            td_errors.detach(),
            batch.indices,
        )

    def _batch_tensors(self, transitions) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.stack([transition.latent_state.float() for transition in transitions if transition.latent_state is not None])
        next_latent = torch.stack([transition.next_latent_state.float() for transition in transitions if transition.next_latent_state is not None])
        action_embeddings = torch.tensor([transition.metadata["action_embedding"] for transition in transitions], dtype=torch.float32)
        rewards = torch.tensor([transition.reward for transition in transitions], dtype=torch.float32)
        dones = torch.tensor([1.0 if transition.done else 0.0 for transition in transitions], dtype=torch.float32)
        actions = torch.tensor([int(transition.action) for transition in transitions], dtype=torch.long)
        return latent, action_embeddings, next_latent, rewards, dones, actions

    @staticmethod
    def _gradient_norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().norm(2).item()) ** 2
        return total ** 0.5
