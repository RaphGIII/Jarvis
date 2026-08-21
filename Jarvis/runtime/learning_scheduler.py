from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from learning.config import DEFAULT_CONFIG
from learning.experience.replay_buffer import ReplayBuffer
from learning.objectives.optimizer import make_optimizer
from learning.policy.action_value import ActionValueNetwork, soft_update
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
    q_lr: float = DEFAULT_CONFIG.q_lr
    encoder_lr: float = DEFAULT_CONFIG.encoder_lr
    action_encoder_lr: float = DEFAULT_CONFIG.action_encoder_lr
    discount_factor: float = DEFAULT_CONFIG.discount_factor
    priority_prediction_error_weight: float = DEFAULT_CONFIG.priority_prediction_error_weight
    target_tau: float = DEFAULT_CONFIG.target_tau
    gradient_clip_norm: float = DEFAULT_CONFIG.gradient_clip_norm


@dataclass
class TrainingReport:
    did_update: bool = False
    world_loss: float | None = None
    transition_loss: float | None = None
    reward_loss: float | None = None
    value_loss: float | None = None
    q_loss: float | None = None
    policy_loss: float | None = None
    mean_td_error: float | None = None
    mean_q_error: float | None = None
    mean_prediction_error: float | None = None
    gradient_norm: float = 0.0
    world_gradient_norm: float = 0.0
    value_gradient_norm: float = 0.0
    q_gradient_norm: float = 0.0
    policy_gradient_norm: float = 0.0
    encoder_gradient_norm: float = 0.0
    action_encoder_gradient_norm: float = 0.0
    replay_size: int = 0
    updated_modules: list[str] = field(default_factory=list)
    q_return_rank_correlation: float | None = None
    value_return_correlation: float | None = None

    def to_dict(self) -> dict[str, float | int | bool | str]:
        result: dict[str, float | int | bool | str] = {
            "did_update": self.did_update,
            "gradient_norm": self.gradient_norm,
            "world_gradient_norm": self.world_gradient_norm,
            "value_gradient_norm": self.value_gradient_norm,
            "q_gradient_norm": self.q_gradient_norm,
            "policy_gradient_norm": self.policy_gradient_norm,
            "encoder_gradient_norm": self.encoder_gradient_norm,
            "action_encoder_gradient_norm": self.action_encoder_gradient_norm,
            "replay_size": self.replay_size,
            "updated_modules": ",".join(self.updated_modules),
        }
        for key in [
            "world_loss",
            "transition_loss",
            "reward_loss",
            "value_loss",
            "q_loss",
            "policy_loss",
            "mean_td_error",
            "mean_q_error",
            "mean_prediction_error",
            "q_return_rank_correlation",
            "value_return_correlation",
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
        *,
        observation_encoder=None,
        action_encoder=None,
        action_value: ActionValueNetwork | None = None,
        persistent_store=None,
    ) -> None:
        self.world_model = world_model
        self.policy = policy
        self.value_function = value_function
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder
        self.action_value = action_value
        self.persistent_store = persistent_store
        self.config = config or LearningSchedulerConfig()
        self.target_value = copy.deepcopy(value_function)
        self._freeze_target(self.target_value)
        self.target_action_value = copy.deepcopy(action_value) if action_value is not None else None
        self._freeze_target(self.target_action_value)
        self.world_optimizer = make_optimizer(world_model, self.config.world_model_lr)
        self.policy_optimizer = make_optimizer(policy, self.config.policy_lr)
        self.value_optimizer = make_optimizer(value_function, self.config.value_lr)
        self.q_optimizer = make_optimizer(action_value, self.config.q_lr) if action_value is not None else None
        self.encoder_optimizer = make_optimizer(observation_encoder, self.config.encoder_lr) if observation_encoder is not None else None
        self.action_encoder_optimizer = make_optimizer(action_encoder, self.config.action_encoder_lr) if action_encoder is not None else None
        self.runtime_steps = 0

    def maybe_train(self, replay_buffer: ReplayBuffer, train_mode: bool = True) -> TrainingReport:
        self.runtime_steps += 1
        report = TrainingReport(replay_size=len(replay_buffer))
        if not train_mode:
            return report
        if self._semantic_ready(replay_buffer):
            return self._train_semantic(replay_buffer)
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
        calibration = self.calibration_diagnostics(replay_buffer)
        report.q_return_rank_correlation = calibration.get("q_return_rank_correlation")
        report.value_return_correlation = calibration.get("value_return_correlation")
        return report

    def _semantic_ready(self, replay_buffer: ReplayBuffer) -> bool:
        if self.observation_encoder is None or self.action_encoder is None or self.action_value is None:
            return False
        if len(replay_buffer) < self.config.value_policy_batch_size:
            return False
        if self.runtime_steps % self.config.value_policy_train_every_n_steps != 0:
            return False
        try:
            recent = replay_buffer.sample_recent(1)[0]
        except (ValueError, IndexError):
            return False
        return all(
            key in recent.metadata
            for key in ["observation_features", "next_observation_features", "action_raw_features"]
        )

    def _train_semantic(self, replay_buffer: ReplayBuffer) -> TrainingReport:
        batch_size = min(self.config.value_policy_batch_size, len(replay_buffer))
        batch = replay_buffer.sample_priority(batch_size)
        (
            obs_semantic,
            obs_numeric,
            next_semantic,
            next_numeric,
            action_raw,
            rewards,
            dones,
            action_types,
        ) = self._semantic_batch_tensors(batch.transitions)

        z = self.observation_encoder(obs_semantic, obs_numeric)
        with torch.no_grad():
            next_z_target = self.observation_encoder(next_semantic, next_numeric)
            next_value = self.target_value(next_z_target)
            target = bellman_target(rewards, next_value, dones, self.config.discount_factor)
        action_embedding = self.action_encoder.forward_from_raw(action_raw)

        world_loss, world_metrics = self.world_model.loss(z, action_embedding, next_z_target, rewards)
        value_prediction = self.value_function(z)
        td_errors = target - value_prediction
        value_loss = torch.mean(td_errors.pow(2))
        q_prediction = self.action_value(z, action_embedding)
        q_errors = target - q_prediction
        q_loss = torch.mean(q_errors.pow(2))
        logits = self.policy(z)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action_types.reshape(-1, 1)).squeeze(1)
        policy_loss = -(selected_log_probs * td_errors.detach()).mean()
        total_loss = world_loss + value_loss + q_loss + policy_loss

        report = TrainingReport(replay_size=len(replay_buffer))
        if not torch.isfinite(total_loss):
            return report

        self._zero_semantic_optimizers()
        total_loss.backward()
        encoder_norm = self._clip(self.observation_encoder.parameters())
        action_encoder_norm = self._clip(self.action_encoder.parameters())
        world_norm = self._clip(self.world_model.parameters())
        value_norm = self._clip(self.value_function.parameters())
        q_norm = self._clip(self.action_value.parameters())
        policy_norm = self._clip(self.policy.parameters())
        total_norm = encoder_norm + action_encoder_norm + world_norm + value_norm + q_norm + policy_norm
        self._step_semantic_optimizers()
        soft_update(self.target_value, self.value_function, self.config.target_tau)
        soft_update(self.target_action_value, self.action_value, self.config.target_tau)

        self.world_model.training_step += 1
        self.value_function.training_step += 1
        self.action_value.training_step += 1
        self.policy.training_step += 1
        self.observation_encoder.training_step += 1
        self.action_encoder.training_step += 1

        with torch.no_grad():
            prediction = self.world_model(z.detach(), action_embedding.detach())
            prediction_errors = torch.mean((prediction.next_latent_pred - next_z_target) ** 2, dim=1)

        combined_errors = [
            abs(float(td)) + self.config.priority_prediction_error_weight * abs(float(prediction_error))
            for td, prediction_error in zip(td_errors.detach().cpu(), prediction_errors.detach().cpu())
        ]
        replay_buffer.update_priorities(batch.indices, combined_errors)
        if self.persistent_store is not None:
            for transition, td, prediction_error, priority in zip(batch.transitions, td_errors, prediction_errors, combined_errors):
                row_id = transition.metadata.get("persistent_row_id")
                if row_id is not None:
                    self.persistent_store.update_errors(
                        int(row_id),
                        float(td.detach().cpu().item()),
                        float(prediction_error.detach().cpu().item()),
                        float(priority),
                    )

        report.did_update = True
        report.world_loss = world_metrics["world_loss"]
        report.transition_loss = world_metrics["transition_loss"]
        report.reward_loss = world_metrics["reward_loss"]
        report.value_loss = float(value_loss.detach().item())
        report.q_loss = float(q_loss.detach().item())
        report.policy_loss = float(policy_loss.detach().item())
        report.mean_td_error = float(td_errors.detach().abs().mean().item())
        report.mean_q_error = float(q_errors.detach().abs().mean().item())
        report.mean_prediction_error = float(prediction_errors.detach().mean().item())
        report.gradient_norm = float(total_norm)
        report.world_gradient_norm = float(world_norm)
        report.value_gradient_norm = float(value_norm)
        report.q_gradient_norm = float(q_norm)
        report.policy_gradient_norm = float(policy_norm)
        report.encoder_gradient_norm = float(encoder_norm)
        report.action_encoder_gradient_norm = float(action_encoder_norm)
        calibration = self.calibration_diagnostics(replay_buffer)
        report.q_return_rank_correlation = calibration.get("q_return_rank_correlation")
        report.value_return_correlation = calibration.get("value_return_correlation")
        report.updated_modules = [
            "ObservationProjection",
            "ActionProjection",
            "WorldModel",
            "ValueFunction",
            "ActionValueNetwork",
            "Policy",
        ]
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
        if not torch.isfinite(loss):
            zeros = torch.zeros(len(batch.transitions), dtype=torch.float32)
            return {"world_loss": 0.0, "transition_loss": 0.0, "reward_loss": 0.0}, 0.0, zeros, batch.indices
        loss.backward()
        grad_norm = self._clip(self.world_model.parameters())
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
        if not torch.isfinite(total_loss):
            return {"value_loss": 0.0, "policy_loss": 0.0}, 0.0, torch.zeros_like(td_errors), batch.indices
        total_loss.backward()
        grad_norm = self._clip(self.value_function.parameters()) + self._clip(self.policy.parameters())
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

    def _semantic_batch_tensors(self, transitions) -> tuple[torch.Tensor, ...]:
        obs_semantic = torch.tensor([transition.metadata["observation_features"]["semantic"] for transition in transitions], dtype=torch.float32)
        obs_numeric = torch.tensor([transition.metadata["observation_features"]["numeric"] for transition in transitions], dtype=torch.float32)
        next_semantic = torch.tensor([transition.metadata["next_observation_features"]["semantic"] for transition in transitions], dtype=torch.float32)
        next_numeric = torch.tensor([transition.metadata["next_observation_features"]["numeric"] for transition in transitions], dtype=torch.float32)
        action_raw = torch.tensor([transition.metadata["action_raw_features"] for transition in transitions], dtype=torch.float32)
        rewards = torch.tensor([transition.reward for transition in transitions], dtype=torch.float32)
        dones = torch.tensor([1.0 if transition.done else 0.0 for transition in transitions], dtype=torch.float32)
        actions = torch.tensor([int(transition.action) for transition in transitions], dtype=torch.long)
        return obs_semantic, obs_numeric, next_semantic, next_numeric, action_raw, rewards, dones, actions

    def calibration_diagnostics(self, replay_buffer: ReplayBuffer, batch_size: int | None = None) -> dict[str, float]:
        if len(replay_buffer) < 2:
            return {"q_return_rank_correlation": 0.0, "value_return_correlation": 0.0}
        sample_size = min(batch_size or 64, len(replay_buffer))
        transitions = replay_buffer.sample_recent(sample_size)
        try:
            if self._semantic_ready_for_transitions(transitions):
                obs_semantic, obs_numeric, next_semantic, next_numeric, action_raw, rewards, dones, _ = self._semantic_batch_tensors(transitions)
                with torch.no_grad():
                    z = self.observation_encoder(obs_semantic, obs_numeric)
                    next_z = self.observation_encoder(next_semantic, next_numeric)
                    action_embedding = self.action_encoder.forward_from_raw(action_raw)
                    q_predictions = self.action_value(z, action_embedding).reshape(-1)
                    value_predictions = self.value_function(z).reshape(-1)
                    returns = bellman_target(rewards, self.target_value(next_z), dones, self.config.discount_factor).reshape(-1)
            else:
                latent, action_embeddings, next_latent, rewards, dones, _ = self._batch_tensors(transitions)
                with torch.no_grad():
                    q_predictions = (
                        self.action_value(latent, action_embeddings).reshape(-1)
                        if self.action_value is not None
                        else torch.zeros_like(rewards)
                    )
                    value_predictions = self.value_function(latent).reshape(-1)
                    returns = bellman_target(rewards, self.target_value(next_latent), dones, self.config.discount_factor).reshape(-1)
        except (KeyError, ValueError, RuntimeError):
            return {"q_return_rank_correlation": 0.0, "value_return_correlation": 0.0}
        return {
            "q_return_rank_correlation": self._correlation(q_predictions, returns),
            "value_return_correlation": self._correlation(value_predictions, returns),
        }

    def _semantic_ready_for_transitions(self, transitions) -> bool:
        return (
            self.observation_encoder is not None
            and self.action_encoder is not None
            and self.action_value is not None
            and all(
                all(key in transition.metadata for key in ["observation_features", "next_observation_features", "action_raw_features"])
                for transition in transitions
            )
        )

    def _zero_semantic_optimizers(self) -> None:
        for optimizer in [
            self.world_optimizer,
            self.value_optimizer,
            self.policy_optimizer,
            self.q_optimizer,
            self.encoder_optimizer,
            self.action_encoder_optimizer,
        ]:
            if optimizer is not None:
                optimizer.zero_grad()

    def _step_semantic_optimizers(self) -> None:
        for optimizer in [
            self.encoder_optimizer,
            self.action_encoder_optimizer,
            self.world_optimizer,
            self.value_optimizer,
            self.q_optimizer,
            self.policy_optimizer,
        ]:
            if optimizer is not None:
                optimizer.step()

    def _clip(self, parameters) -> float:
        params = [parameter for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip_norm).item())

    @staticmethod
    def _correlation(first: torch.Tensor, second: torch.Tensor) -> float:
        first = first.detach().float().reshape(-1)
        second = second.detach().float().reshape(-1)
        if first.numel() < 2 or second.numel() < 2:
            return 0.0
        first_centered = first - first.mean()
        second_centered = second - second.mean()
        denom = first_centered.norm() * second_centered.norm()
        if float(denom.item()) <= 1e-8:
            return 0.0
        return float(torch.clamp(torch.dot(first_centered, second_centered) / denom, -1.0, 1.0).item())

    @staticmethod
    def _gradient_norm(parameters) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().norm(2).item()) ** 2
        return total ** 0.5

    @staticmethod
    def _freeze_target(module) -> None:
        if module is None:
            return
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
