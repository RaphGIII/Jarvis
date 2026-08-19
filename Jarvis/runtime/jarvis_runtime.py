from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from environments.coding.actions import ActionCandidate, ActionEncoder, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.observation import CodingObservation, ObservationAdapter
from environments.coding.reward import CodingRewardEngine, CodingRewardResult
from environments.coding.task import CodingTask
from learning.config import DEFAULT_CONFIG
from learning.experience.persistent_store import PersistentExperienceStore
from learning.experience.prioritization import compute_learning_priority
from learning.experience.replay_buffer import ReplayBuffer
from learning.experience.transition import Transition
from learning.meta.self_model import SelfModel
from learning.objectives.optimizer import set_global_seeds
from learning.policy.policy import NeuralPolicy, PolicyConfig
from learning.policy.value import NeuralValueFunction, ValueConfig
from learning.representations.encoder import ObservationEncoder
from learning.rewards.intrinsic import novelty_reward
from learning.world_model.model import WorldModel, WorldModelConfig
from learning.world_model.uncertainty import UncertaintyEstimator
from runtime.action_generator import ActionGenerator, BrainProvider, HeuristicCodingActionGenerator
from runtime.checkpoints import RuntimeCheckpointManager
from runtime.events import RuntimeEvent
from runtime.learning_scheduler import LearningScheduler, LearningSchedulerConfig, TrainingReport
from runtime.runtime_state import RuntimeMode, RuntimeState


@dataclass(frozen=True)
class JarvisRuntimeConfig:
    latent_dim: int = DEFAULT_CONFIG.latent_dim
    observation_feature_dim: int = 24
    hidden_dim: int = 64
    replay_capacity: int = DEFAULT_CONFIG.replay_capacity
    data_dir: str = "data"
    seed: int = 13
    num_action_candidates: int = DEFAULT_CONFIG.num_action_candidates
    policy_score_weight: float = 1.2
    world_reward_weight: float = 0.7
    information_gain_weight: float = 0.2
    risk_weight: float = 0.8
    cost_weight: float = 0.15
    confidence_weight: float = 0.15
    train_exploration_epsilon: float = 0.15
    load_latest_checkpoints: bool = False


@dataclass
class ScoredAction:
    candidate: ActionCandidate
    score: float
    policy_score: float
    predicted_reward: float
    expected_information_gain: float
    risk: float
    uncertainty: float
    novelty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score,
            "policy_score": self.policy_score,
            "predicted_reward": self.predicted_reward,
            "expected_information_gain": self.expected_information_gain,
            "risk": self.risk,
            "uncertainty": self.uncertainty,
            "novelty": self.novelty,
        }


@dataclass
class RuntimeStepResult:
    observation: CodingObservation
    selected_action: ActionCandidate
    action_result: Any
    reward: CodingRewardResult
    transition: Transition
    training_report: TrainingReport
    done: bool
    success: bool
    scored_candidates: list[ScoredAction] = field(default_factory=list)


class JarvisRuntime:
    """Closed learning loop for controlled coding tasks."""

    def __init__(
        self,
        *,
        brain: BrainProvider | None = None,
        action_generator: ActionGenerator | None = None,
        config: JarvisRuntimeConfig | None = None,
        data_dir: str | Path | None = None,
        mode: RuntimeMode = RuntimeMode.TRAIN,
    ) -> None:
        self.config = config or JarvisRuntimeConfig()
        set_global_seeds(self.config.seed)
        self.brain = brain
        self.action_generator = action_generator or HeuristicCodingActionGenerator(self.config.num_action_candidates)
        self.observation_adapter = ObservationAdapter(self.config.observation_feature_dim)
        self.action_encoder = ActionEncoder(embedding_dim=len(ActionType))
        self.encoder = ObservationEncoder(
            input_dim=self.config.observation_feature_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.policy = NeuralPolicy(
            PolicyConfig(
                state_dim=self.config.latent_dim,
                num_actions=len(ActionType),
                hidden_dim=self.config.hidden_dim,
            )
        )
        self.value_function = NeuralValueFunction(
            ValueConfig(state_dim=self.config.latent_dim, hidden_dim=self.config.hidden_dim)
        )
        self.world_model = WorldModel(
            WorldModelConfig(
                latent_dim=self.config.latent_dim,
                action_dim=len(ActionType),
                hidden_dim=self.config.hidden_dim,
            )
        )
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        scheduler_config = LearningSchedulerConfig(
            world_model_batch_size=min(DEFAULT_CONFIG.world_model_batch_size, self.config.replay_capacity),
            value_policy_batch_size=min(DEFAULT_CONFIG.value_policy_batch_size, self.config.replay_capacity),
        )
        self.scheduler = LearningScheduler(self.world_model, self.policy, self.value_function, scheduler_config)
        root_data_dir = Path(data_dir or self.config.data_dir)
        self.checkpoint_manager = RuntimeCheckpointManager(root_data_dir / "checkpoints")
        self.experience_store = PersistentExperienceStore(root_data_dir / "experience.sqlite")
        self.reward_engine = CodingRewardEngine()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.self_model = SelfModel()
        self.state = RuntimeState(mode=mode)
        self.environment: CodingEnvironment | None = None
        self.events: list[RuntimeEvent] = []
        self.known_latents: list[torch.Tensor] = []
        self._rng = random.Random(self.config.seed)
        if self.config.load_latest_checkpoints:
            self.load_latest_checkpoints()

    def start_task(self, task: CodingTask, mode: RuntimeMode | None = None) -> CodingObservation:
        self.environment = CodingEnvironment(task)
        if mode is not None:
            self.state.mode = mode
        episode_id = f"{task.task_id}-{uuid.uuid4().hex[:8]}"
        self.state.reset_episode(task.task_id, episode_id)
        self.events.append(RuntimeEvent("episode_started", {"task_id": task.task_id, "episode_id": episode_id}))
        return self.environment.observe()

    def step(self, user_goal: str) -> RuntimeStepResult:
        if self.environment is None:
            raise RuntimeError("No CodingEnvironment is active. Call start_task first.")
        previous_observation = self.environment.observe()
        latent = self._encode_observation(previous_observation)
        candidates = self.action_generator.generate(user_goal, previous_observation)
        scored = self._score_candidates(latent, candidates, previous_observation)
        selected = self._select(scored)
        action_embedding = self.action_encoder.encode(selected.candidate)
        environment_step = self.environment.step(selected.candidate)
        next_latent = self._encode_observation(environment_step.observation)
        reward = self.reward_engine.compute(previous_observation, selected.candidate, environment_step)
        transition = self._make_transition(
            previous_observation,
            latent,
            selected,
            action_embedding,
            reward,
            environment_step.observation,
            next_latent,
            environment_step.done,
            environment_step.success,
            environment_step.objective_metrics,
        )
        combined_error = abs(float(transition.metadata.get("td_error", 0.0))) + DEFAULT_CONFIG.priority_prediction_error_weight * abs(
            float(transition.metadata.get("prediction_error", 0.0))
        )
        priority = compute_learning_priority(
            float(transition.metadata.get("td_error", 0.0)),
            float(transition.metadata.get("prediction_error", 0.0)),
            DEFAULT_CONFIG.priority_prediction_error_weight,
            self.replay_buffer.priority_config,
        )
        self.replay_buffer.add(transition, error=combined_error)
        row_id = self.experience_store.add_transition(
            task_id=self.state.task_id or "unknown",
            episode_id=self.state.episode_id or "unknown",
            step=self.state.step_count,
            transition=transition,
            action_payload=selected.candidate.to_dict(),
            reward_components=reward.components,
            priority=priority,
            model_versions=self._model_versions(),
        )
        transition.metadata["persistent_row_id"] = row_id
        self.state.trajectory.add(transition)
        self.state.step_count += 1
        self.state.total_reward += transition.reward
        self.known_latents.append(latent.detach())
        training_report = self.scheduler.maybe_train(
            self.replay_buffer,
            train_mode=self.state.mode == RuntimeMode.TRAIN,
        )
        self.state.latest_metrics = {
            **environment_step.objective_metrics,
            **training_report.to_dict(),
            "total_reward": self.state.total_reward,
        }
        if environment_step.done:
            self._update_self_model(environment_step.success)
            self.events.append(
                RuntimeEvent(
                    "episode_finished",
                    {
                        "task_id": self.state.task_id,
                        "episode_id": self.state.episode_id,
                        "success": environment_step.success,
                        "reward": self.state.total_reward,
                    },
                )
            )
        return RuntimeStepResult(
            observation=environment_step.observation,
            selected_action=selected.candidate,
            action_result=environment_step.action_result,
            reward=reward,
            transition=transition,
            training_report=training_report,
            done=environment_step.done,
            success=environment_step.success,
            scored_candidates=scored,
        )

    def run_episode(self, task: CodingTask, mode: RuntimeMode | None = None) -> dict[str, float | int | bool]:
        self.start_task(task, mode)
        done = False
        success = False
        while not done:
            result = self.step(task.description)
            done = result.done
            success = result.success
        return {
            "success": success,
            "reward": self.state.total_reward,
            "steps": self.state.step_count,
            "replay_size": len(self.replay_buffer),
            "persistent_experience": self.experience_store.count(),
        }

    def chat(self, message: str) -> str:
        if self.brain is None:
            return "Brain provider is not loaded. Use /train or /eval for CodingWorld."
        return self.brain.think(message)

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.state.mode.value,
            "task_id": self.state.task_id,
            "episode_id": self.state.episode_id,
            "step_count": self.state.step_count,
            "total_reward": self.state.total_reward,
            "replay_size": len(self.replay_buffer),
            "persistent_experience": self.experience_store.count(),
            "latest_metrics": self.state.latest_metrics,
        }

    def learning_summary(self) -> dict[str, Any]:
        return {
            "parameters_updated": {
                "WorldModel": self.world_model.training_step,
                "Policy": self.policy.training_step,
                "ValueFunction": self.value_function.training_step,
                "ObservationEncoder": "not updated online in v0.1",
            },
            "not_updated": ["Qwen foundation model"],
            "capabilities": {
                name: {
                    "attempts": estimate.attempts,
                    "success_rate": estimate.success_rate,
                    "mean_reward": estimate.mean_reward,
                    "trend": estimate.trend,
                    "uncertainty": estimate.uncertainty,
                }
                for name, estimate in self.self_model.capabilities.items()
            },
        }

    def save_checkpoints(self, metrics: dict[str, float] | None = None) -> dict[str, str]:
        paths = {
            "WorldModel": self.world_model.save_checkpoint(
                self.checkpoint_manager.directory / "world_model.pt",
                optimizer=self.scheduler.world_optimizer,
                metrics=metrics,
            ),
            "Policy": self.policy.save_checkpoint(
                self.checkpoint_manager.directory / "policy.pt",
                optimizer=self.scheduler.policy_optimizer,
                metrics=metrics,
            ),
            "ValueFunction": self.value_function.save_checkpoint(
                self.checkpoint_manager.directory / "value.pt",
                optimizer=self.scheduler.value_optimizer,
                metrics=metrics,
            ),
            "ObservationEncoder": self.checkpoint_manager.save_module(
                "observation_encoder",
                self.encoder,
                version="observation-encoder-0.1",
                training_step=0,
                metrics=metrics,
            ),
        }
        return {name: str(path) for name, path in paths.items()}

    def load_latest_checkpoints(self) -> dict[str, bool]:
        loaded = {
            "WorldModel": self.checkpoint_manager.load_latest_module(
                "world_model", self.world_model, optimizer=self.scheduler.world_optimizer
            )
            is not None,
            "Policy": self.checkpoint_manager.load_latest_module(
                "policy", self.policy, optimizer=self.scheduler.policy_optimizer
            )
            is not None,
            "ValueFunction": self.checkpoint_manager.load_latest_module(
                "value", self.value_function, optimizer=self.scheduler.value_optimizer
            )
            is not None,
            "ObservationEncoder": self.checkpoint_manager.load_latest_module("observation_encoder", self.encoder)
            is not None,
        }
        return loaded

    def _encode_observation(self, observation: CodingObservation) -> torch.Tensor:
        features = self.observation_adapter.encode(observation)
        with torch.no_grad():
            return self.encoder.encode(features).squeeze(0).detach()

    def _score_candidates(
        self,
        latent: torch.Tensor,
        candidates: list[ActionCandidate],
        observation: CodingObservation,
    ) -> list[ScoredAction]:
        if not candidates:
            raise ValueError("Action generator returned no candidates.")
        with torch.no_grad():
            policy_probs = self.policy.action_distribution(latent).squeeze(0)
        known = torch.stack(self.known_latents) if self.known_latents else None
        novelty = min(1.0, novelty_reward(latent, known))
        scored = []
        for candidate in candidates:
            action_embedding = self.action_encoder.encode(candidate)
            with torch.no_grad():
                prediction = self.world_model(latent, action_embedding)
                predicted_reward = float(prediction.reward_pred.reshape(-1)[0].item()) if prediction.reward_pred is not None else 0.0
            policy_score = float(policy_probs[candidate.action_index].item())
            risk = self._estimate_risk(candidate, observation)
            uncertainty = self.uncertainty_estimator.combine(
                model_uncertainty=1.0 / max(1.0, len(self.replay_buffer) ** 0.5),
                memory_uncertainty=1.0 if not self.known_latents else 0.2,
                disagreement=abs(predicted_reward) * 0.05,
                novelty=novelty,
            ).total
            expected_information_gain = 0.5 * uncertainty + 0.5 * novelty
            score = (
                self.config.policy_score_weight * policy_score
                + self.config.world_reward_weight * predicted_reward
                + self.config.information_gain_weight * expected_information_gain
                + self.config.confidence_weight * candidate.confidence
                - self.config.risk_weight * risk
                - self.config.cost_weight * candidate.estimated_cost
            )
            scored.append(
                ScoredAction(
                    candidate=candidate,
                    score=float(score),
                    policy_score=policy_score,
                    predicted_reward=predicted_reward,
                    expected_information_gain=float(expected_information_gain),
                    risk=float(risk),
                    uncertainty=float(uncertainty),
                    novelty=float(novelty),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)

    def _select(self, scored: list[ScoredAction]) -> ScoredAction:
        if self.state.mode == RuntimeMode.TRAIN and self._rng.random() < self.config.train_exploration_epsilon:
            return self._rng.choice(scored)
        return scored[0]

    def _estimate_risk(self, candidate: ActionCandidate, observation: CodingObservation) -> float:
        risk = 0.0
        if candidate.action_type in {ActionType.WRITE_FILE, ActionType.PATCH_FILE}:
            risk += 0.25
            if not observation.relevant_file_excerpts:
                risk += 0.25
            elif candidate.action_type == ActionType.PATCH_FILE:
                path = str(candidate.arguments.get("path", ""))
                old = str(candidate.arguments.get("old", ""))
                if path in observation.relevant_file_excerpts and old in observation.relevant_file_excerpts[path]:
                    risk -= 0.2
        if candidate.action_type == ActionType.FINISH and not (
            observation.test_state.get("ran", False)
            and observation.test_state.get("passed", 0) > 0
            and observation.test_state.get("failed", 0) == 0
        ):
            risk += 1.0
        return risk

    def _make_transition(
        self,
        observation: CodingObservation,
        latent: torch.Tensor,
        selected: ScoredAction,
        action_embedding: torch.Tensor,
        reward: CodingRewardResult,
        next_observation: CodingObservation,
        next_latent: torch.Tensor,
        done: bool,
        success: bool,
        objective_metrics: dict[str, float | int | str | bool],
    ) -> Transition:
        with torch.no_grad():
            predicted = self.world_model(latent, action_embedding)
            prediction_error = torch.mean((predicted.next_latent_pred.squeeze(0) - next_latent) ** 2).item()
            value = self.value_function(latent).reshape(-1)[0]
            next_value = self.value_function(next_latent).reshape(-1)[0]
            td_error = reward.total + DEFAULT_CONFIG.discount_factor * next_value.item() * (0.0 if done else 1.0) - value.item()
        return Transition(
            observation=observation.to_dict(),
            latent_state=latent,
            action=selected.candidate.action_index,
            reward=reward.total,
            next_observation=next_observation.to_dict(),
            next_latent_state=next_latent,
            done=done,
            uncertainty=selected.uncertainty,
            novelty=selected.novelty,
            success=success,
            metadata={
                "action": selected.candidate.to_dict(),
                "action_embedding": action_embedding.detach().cpu().tolist(),
                "reward_components": reward.components,
                "scoring": selected.to_dict(),
                "prediction_error": float(prediction_error),
                "td_error": float(td_error),
                "objective_metrics": objective_metrics,
                "mode": self.state.mode.value,
            },
        )

    def _update_self_model(self, success: bool) -> None:
        uncertainty = float(self.state.trajectory.transitions[-1].uncertainty) if self.state.trajectory.transitions else 1.0
        capabilities = [
            "coding",
            "debugging",
            "file_navigation",
            "test_reasoning",
            "code_editing",
            "error_diagnosis",
        ]
        for capability in capabilities:
            self.self_model.update_capability(capability, success, self.state.total_reward, uncertainty)

    def _model_versions(self) -> dict[str, str]:
        return {
            "WorldModel": self.world_model.config.version,
            "Policy": self.policy.config.version,
            "ValueFunction": self.value_function.config.version,
            "ObservationEncoder": "observation-encoder-0.1",
            "Qwen": "not-updated",
        }
