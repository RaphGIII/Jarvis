from __future__ import annotations

import random
import uuid
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.observation import CodingObservation, ObservationAdapter
from environments.coding.reward import CodingRewardEngine, CodingRewardResult
from environments.coding.sandbox_backend import SandboxBackend
from environments.coding.task import CodingTask
from learning.config import DEFAULT_CONFIG
from learning.experience.persistent_store import PersistentExperienceStore
from learning.experience.prioritization import compute_learning_priority
from learning.experience.replay_buffer import ReplayBuffer
from learning.experience.transition import Transition
from learning.meta.self_model import SelfModel
from learning.objectives.optimizer import set_global_seeds
from learning.objectives.optimizer import make_optimizer
from learning.policy.action_value import ActionValueConfig, ActionValueNetwork
from learning.policy.policy import NeuralPolicy, PolicyConfig
from learning.policy.value import NeuralValueFunction, ValueConfig
from learning.representations.action_encoding import SemanticActionEncoder
from learning.representations.semantic import (
    LightweightLocalEmbeddingProvider,
    ProjectionEncoder,
    SemanticObservationFeatures,
    SemanticTextEncoder,
)
from learning.rewards.intrinsic import novelty_reward
from learning.world_model.model import WorldModel, WorldModelConfig
from learning.world_model.uncertainty import UncertaintyEstimator
from runtime.action_generator import ActionGenerator, BrainProvider, HeuristicCodingActionGenerator, QwenActionGenerator, fallback_candidates
from runtime.checkpoints import RuntimeCheckpointManager
from runtime.events import RuntimeEvent
from runtime.learning_scheduler import LearningScheduler, LearningSchedulerConfig, TrainingReport
from runtime.profiling import PerformanceProfiler
from runtime.runtime_state import RuntimeMode, RuntimeState
from runtime.tensorboard import TensorBoardLogger


@dataclass(frozen=True)
class JarvisRuntimeConfig:
    latent_dim: int = DEFAULT_CONFIG.latent_dim
    observation_feature_dim: int = 24
    semantic_embedding_dim: int = DEFAULT_CONFIG.semantic_embedding_dim
    action_embedding_dim: int = DEFAULT_CONFIG.action_embedding_dim
    hidden_dim: int = 64
    replay_capacity: int = DEFAULT_CONFIG.replay_capacity
    data_dir: str = "data"
    seed: int = 13
    num_action_candidates: int = DEFAULT_CONFIG.num_action_candidates
    coding_generation_max_tokens: int = 450
    policy_score_weight: float = 1.2
    q_score_weight: float = DEFAULT_CONFIG.score_q_weight
    policy_log_score_weight: float = DEFAULT_CONFIG.score_policy_log_weight
    world_reward_weight: float = 0.7
    information_gain_weight: float = 0.2
    risk_weight: float = 0.8
    cost_weight: float = 0.15
    confidence_weight: float = 1.0
    feasibility_penalty: float = 1_000_000.0
    warmup_experiences: int = 50
    train_exploration_epsilon: float = 0.15
    epsilon_min: float = DEFAULT_CONFIG.epsilon_min
    epsilon_decay: float = DEFAULT_CONFIG.epsilon_decay
    load_latest_checkpoints: bool = False
    replay_warm_start_size: int = DEFAULT_CONFIG.replay_warm_start_size
    tensorboard_subdir: str = "tensorboard"
    trace_actions: bool = False
    eval_controller: str = "full"
    production_controller: str = "heuristic"
    learned_controller_mode: str = "active"
    value_score_weight: float = 0.4
    learned_component_clip: float = 2.0
    learned_gate_min_experiences: int = 50
    learned_gate_warmup_experiences: int = 250
    learned_gate_prediction_error_scale: float = 10.0
    learned_gate_recent_window: int = 50


@dataclass
class ScoredAction:
    candidate: ActionCandidate
    score: float
    policy_score: float
    q_value: float
    predicted_reward: float
    expected_information_gain: float
    risk: float
    uncertainty: float
    novelty: float
    feasible: bool = True
    feasibility_reason: str = ""
    learned_weight: float = 1.0
    heuristic_score: float = 0.0
    learned_score: float = 0.0
    q_score: float = 0.0
    value_score: float = 0.0
    world_score: float = 0.0
    final_score: float = 0.0
    controller_gate: float = 0.0
    controller_mode: str = "full"
    heuristic_winner: bool = False
    controller_changed_heuristic: bool = False
    shadow_learned_score: float = 0.0
    shadow_controller_gate: float = 0.0
    shadow_learned_winner: bool = False
    shadow_changed_heuristic: bool = False
    raw_score_components: dict[str, float] = field(default_factory=dict)
    normalized_score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score,
            "final_score": self.final_score,
            "policy_score": self.policy_score,
            "q_value": self.q_value,
            "predicted_reward": self.predicted_reward,
            "q_score": self.q_score,
            "value_score": self.value_score,
            "world_score": self.world_score,
            "expected_information_gain": self.expected_information_gain,
            "risk": self.risk,
            "uncertainty": self.uncertainty,
            "novelty": self.novelty,
            "feasible": self.feasible,
            "feasibility_reason": self.feasibility_reason,
            "learned_weight": self.learned_weight,
            "heuristic_score": self.heuristic_score,
            "learned_score": self.learned_score,
            "controller_gate": self.controller_gate,
            "controller_mode": self.controller_mode,
            "heuristic_winner": self.heuristic_winner,
            "controller_changed_heuristic": self.controller_changed_heuristic,
            "shadow_learned_score": self.shadow_learned_score,
            "shadow_controller_gate": self.shadow_controller_gate,
            "shadow_learned_winner": self.shadow_learned_winner,
            "shadow_changed_heuristic": self.shadow_changed_heuristic,
            "raw_score_components": self.raw_score_components,
            "normalized_score_components": self.normalized_score_components,
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
        semantic_text_encoder: SemanticTextEncoder | None = None,
        sandbox_backend: SandboxBackend | None = None,
        config: JarvisRuntimeConfig | None = None,
        data_dir: str | Path | None = None,
        mode: RuntimeMode = RuntimeMode.TRAIN,
    ) -> None:
        self.config = config or JarvisRuntimeConfig()
        set_global_seeds(self.config.seed)
        self.brain = brain
        self._freeze_brain_parameters(brain)
        self.sandbox_backend = sandbox_backend
        self.text_encoder = semantic_text_encoder or LightweightLocalEmbeddingProvider(self.config.semantic_embedding_dim)
        self.action_generator = action_generator or (
            QwenActionGenerator(brain, self.config.num_action_candidates, max_tokens=self.config.coding_generation_max_tokens)
            if brain is not None
            else HeuristicCodingActionGenerator(self.config.num_action_candidates)
        )
        self.observation_adapter = ObservationAdapter(self.config.observation_feature_dim)
        self.action_encoder = SemanticActionEncoder(
            self.text_encoder,
            action_embedding_dim=self.config.action_embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.encoder = ProjectionEncoder(
            semantic_dim=self.text_encoder.embedding_dim,
            numeric_dim=self.config.observation_feature_dim,
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
        self.action_value = ActionValueNetwork(
            ActionValueConfig(
                state_dim=self.config.latent_dim,
                action_dim=self.config.action_embedding_dim,
                hidden_dim=self.config.hidden_dim,
            )
        )
        self.world_model = WorldModel(
            WorldModelConfig(
                latent_dim=self.config.latent_dim,
                action_dim=self.config.action_embedding_dim,
                hidden_dim=self.config.hidden_dim,
            )
        )
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        scheduler_config = LearningSchedulerConfig(
            world_model_batch_size=min(DEFAULT_CONFIG.world_model_batch_size, self.config.replay_capacity),
            value_policy_batch_size=min(DEFAULT_CONFIG.value_policy_batch_size, self.config.replay_capacity),
        )
        self.scheduler = LearningScheduler(
            self.world_model,
            self.policy,
            self.value_function,
            scheduler_config,
            observation_encoder=self.encoder,
            action_encoder=self.action_encoder,
            action_value=self.action_value,
            persistent_store=None,
        )
        root_data_dir = Path(data_dir or self.config.data_dir)
        self.checkpoint_manager = RuntimeCheckpointManager(root_data_dir / "checkpoints")
        self.experience_store = PersistentExperienceStore(root_data_dir / "experience" / "experience.sqlite")
        self.scheduler.persistent_store = self.experience_store
        self.tensorboard = TensorBoardLogger(root_data_dir / self.config.tensorboard_subdir)
        self.reward_engine = CodingRewardEngine()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.self_model = SelfModel()
        self.state = RuntimeState(mode=mode)
        self.environment: CodingEnvironment | None = None
        self.events: list[RuntimeEvent] = []
        self.profiler = PerformanceProfiler()
        self.trace_label = "EPISODE"
        self.known_latents: list[torch.Tensor] = []
        self._rng = random.Random(self.config.seed)
        if self.config.load_latest_checkpoints:
            self.load_latest_checkpoints()
        self.warm_start_replay(self.config.replay_warm_start_size)

    def attach_brain(
        self,
        brain: BrainProvider,
        *,
        semantic_text_encoder: SemanticTextEncoder | None = None,
        num_action_candidates: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Attach one loaded Qwen brain to generation and semantic encoding.

        This method intentionally reuses the provided brain object for both
        candidate generation and hidden-state feature extraction. It does not
        load another foundation model and it freezes any exposed model
        parameters.
        """

        self.brain = brain
        self._freeze_brain_parameters(brain)
        model = getattr(brain, "model", None)
        if model is not None:
            model.eval()
        self.text_encoder = semantic_text_encoder or LightweightLocalEmbeddingProvider(self.config.semantic_embedding_dim)
        self.action_generator = QwenActionGenerator(
            brain,
            num_candidates=num_action_candidates or self.config.num_action_candidates,
            max_tokens=max_tokens or self.config.coding_generation_max_tokens,
        )
        self._rebuild_text_dependent_modules()

    def _rebuild_text_dependent_modules(self) -> None:
        self.action_encoder = SemanticActionEncoder(
            self.text_encoder,
            action_embedding_dim=self.config.action_embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.encoder = ProjectionEncoder(
            semantic_dim=self.text_encoder.embedding_dim,
            numeric_dim=self.config.observation_feature_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.scheduler.observation_encoder = self.encoder
        self.scheduler.action_encoder = self.action_encoder
        self.scheduler.encoder_optimizer = make_optimizer(self.encoder, self.scheduler.config.encoder_lr)
        self.scheduler.action_encoder_optimizer = make_optimizer(self.action_encoder, self.scheduler.config.action_encoder_lr)

    @staticmethod
    def _freeze_brain_parameters(brain: BrainProvider | None) -> None:
        model = getattr(brain, "model", None)
        if model is None:
            return
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def warm_start_replay(self, limit: int | None = None) -> int:
        transitions = self.experience_store.warm_start_transitions(limit or self.config.replay_warm_start_size)
        seen_rows: set[int] = set()
        loaded = 0
        for transition in transitions:
            row_id = transition.metadata.get("persistent_row_id")
            if row_id in seen_rows:
                continue
            action_raw = transition.metadata.get("action_raw_features")
            observation_features = transition.metadata.get("observation_features") or {}
            if action_raw is not None and len(action_raw) != self.action_encoder.raw_dim:
                continue
            if observation_features.get("semantic") is not None and len(observation_features["semantic"]) != self.text_encoder.embedding_dim:
                continue
            seen_rows.add(row_id)
            combined_error = abs(float(transition.metadata.get("td_error", 0.0))) + DEFAULT_CONFIG.priority_prediction_error_weight * abs(
                float(transition.metadata.get("prediction_error", 0.0))
            )
            self.replay_buffer.add(transition, error=combined_error)
            loaded += 1
        return loaded

    def start_task(self, task: CodingTask, mode: RuntimeMode | None = None) -> CodingObservation:
        if mode is not None:
            self.state.mode = mode
        self.environment = CodingEnvironment(
            task,
            backend=self.sandbox_backend,
            run_hidden_during_tests=self.state.mode == RuntimeMode.TRAIN,
            terminate_on_public_success=self.state.mode == RuntimeMode.TRAIN,
        )
        episode_id = f"{task.task_id}-{uuid.uuid4().hex[:8]}"
        self.state.reset_episode(task.task_id, episode_id)
        self.events.append(RuntimeEvent("episode_started", {"task_id": task.task_id, "episode_id": episode_id}))
        return self.environment.observe()

    def final_hidden_verification(self) -> dict[str, Any]:
        if self.environment is None:
            raise RuntimeError("No CodingEnvironment is active.")
        return self.environment.run_final_hidden_verifier()

    def step(self, user_goal: str) -> RuntimeStepResult:
        if self.environment is None:
            raise RuntimeError("No CodingEnvironment is active. Call start_task first.")
        with self.profiler.measure("total_step"):
            result = self._step_inner(user_goal)
        return result

    def _step_inner(self, user_goal: str) -> RuntimeStepResult:
        previous_observation = self.environment.observe()
        with self.profiler.measure("semantic_observation_encoding"):
            previous_features = self._observation_features(previous_observation)
        latent = self._encode_features(previous_features)
        with self.profiler.measure("brain_candidate_generation"):
            candidates = self.action_generator.generate(user_goal, previous_observation)
        self.profiler.increment("brain_requests", 0.0 if getattr(self.action_generator, "last_generation_metadata", {}).get("cache_hit") else 1.0)
        self.profiler.increment("candidate_cache_hits", 1.0 if getattr(self.action_generator, "last_generation_metadata", {}).get("cache_hit") else 0.0)
        self.profiler.increment("candidates_generated", len(candidates))
        generation_metadata = getattr(self.action_generator, "last_generation_metadata", {})
        for key in [
            "valid_candidates",
            "malformed_items",
            "schema_invalid_candidates",
            "parse_error_count",
            "zero_valid_qwen_candidates",
            "fallback_backfill_count",
            "duplicate_candidates",
            "generation_length_truncation_count",
            "structured_generation_requests",
            "structured_generation_failures",
            "candidate_regeneration_count",
            "candidate_regeneration_success_count",
        ]:
            if generation_metadata.get(key) is not None:
                self.profiler.increment(key, float(generation_metadata.get(key) or 0))
        generated_tokens = getattr(self.action_generator, "last_generation_metadata", {}).get("generated_tokens")
        if generated_tokens is not None:
            self.profiler.increment("generated_tokens", float(generated_tokens))
        with self.profiler.measure("policy_q_world_scoring"):
            scored = self._score_candidates(latent, candidates, previous_observation)
            scored = self._ensure_feasible_scored(latent, scored, previous_observation)
        selected = self._select(scored)
        assert selected.feasible, f"Invariant violation: selected infeasible action {selected.candidate.action_type.name}: {selected.feasibility_reason}"
        action_raw_features = self.action_encoder.raw_features(selected.candidate)
        with torch.no_grad():
            action_embedding = self.action_encoder.forward_from_raw(action_raw_features).squeeze(0).detach()
        with self.profiler.measure("docker_execution"):
            environment_step = self.environment.step(selected.candidate)
        with self.profiler.measure("semantic_observation_encoding"):
            next_features = self._observation_features(environment_step.observation)
        next_latent = self._encode_features(next_features)
        reward = self.reward_engine.compute(previous_observation, selected.candidate, environment_step)
        transition = self._make_transition(
            previous_observation,
            previous_features,
            latent,
            selected,
            action_raw_features,
            action_embedding,
            reward,
            environment_step.observation,
            next_features,
            next_latent,
            environment_step.done,
            environment_step.success,
            environment_step.objective_metrics,
            scored,
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
        if self.state.mode == RuntimeMode.TRAIN:
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
        with self.profiler.measure("optimizer_training_update"):
            training_report = self.scheduler.maybe_train(
                self.replay_buffer,
                train_mode=self.state.mode == RuntimeMode.TRAIN,
            )
        self.state.latest_metrics = {
            **environment_step.objective_metrics,
            **training_report.to_dict(),
            "total_reward": self.state.total_reward,
        }
        self._log_step_metrics(training_report, environment_step.success)
        self._trace_step(scored, selected, reward.total, environment_step)
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
        with self.profiler.measure("total_episode"):
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
        if hasattr(self.brain, "generate"):
            return self.brain.generate(message)
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
            "performance": self.profiler.summary(),
        }

    def learning_summary(self) -> dict[str, Any]:
        return {
            "parameters_updated": {
                "WorldModel": self.world_model.training_step,
                "Policy": self.policy.training_step,
                "ValueFunction": self.value_function.training_step,
                "ActionValueNetwork": self.action_value.training_step,
                "ObservationProjection": self.encoder.training_step,
                "ActionProjection": self.action_encoder.training_step,
            },
            "not_updated": ["Qwen foundation model"],
            "tensorboard": self.tensorboard.command,
            "performance": self.profiler.summary(),
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

    def save_checkpoints(self, metrics: dict[str, float] | None = None, category: str = "latest") -> dict[str, str]:
        paths = {
            "WorldModel": self.checkpoint_manager.save_module_snapshot(
                category,
                "world_model",
                self.world_model,
                version=self.world_model.config.version,
                training_step=self.world_model.training_step,
                optimizer=self.scheduler.world_optimizer,
                metrics=metrics,
            ),
            "Policy": self.checkpoint_manager.save_module_snapshot(
                category,
                "policy",
                self.policy,
                version=self.policy.config.version,
                training_step=self.policy.training_step,
                optimizer=self.scheduler.policy_optimizer,
                metrics=metrics,
            ),
            "ValueFunction": self.checkpoint_manager.save_module_snapshot(
                category,
                "value",
                self.value_function,
                version=self.value_function.config.version,
                training_step=self.value_function.training_step,
                optimizer=self.scheduler.value_optimizer,
                metrics=metrics,
            ),
            "ObservationEncoder": self.checkpoint_manager.save_module_snapshot(
                category,
                "observation_projection",
                self.encoder,
                version="observation-projection-0.2",
                training_step=self.encoder.training_step,
                metrics=metrics,
                optimizer=self.scheduler.encoder_optimizer,
            ),
            "ActionProjection": self.checkpoint_manager.save_module_snapshot(
                category,
                "action_projection",
                self.action_encoder,
                version="action-projection-0.2",
                training_step=self.action_encoder.training_step,
                metrics=metrics,
                optimizer=self.scheduler.action_encoder_optimizer,
            ),
            "ActionValueNetwork": self.checkpoint_manager.save_module_snapshot(
                category,
                "action_value",
                self.action_value,
                version=self.action_value.config.version,
                training_step=self.action_value.training_step,
                metrics=metrics,
                optimizer=self.scheduler.q_optimizer,
            ),
        }
        self.checkpoint_manager.save_category_metadata(category, metrics or {}, {"version": "runtime-snapshot"})
        return {name: str(path) for name, path in paths.items()}

    def save_training_resume_checkpoint(self, path: str | Path, *, train_completed: int) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "train_completed": int(train_completed),
            "models": {
                "world_model": self.world_model.state_dict(),
                "policy": self.policy.state_dict(),
                "value": self.value_function.state_dict(),
                "action_value": self.action_value.state_dict(),
                "observation_encoder": self.encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "target_value": self.scheduler.target_value.state_dict(),
                "target_action_value": self.scheduler.target_action_value.state_dict()
                if self.scheduler.target_action_value is not None
                else None,
            },
            "optimizers": {
                "world": self.scheduler.world_optimizer.state_dict(),
                "policy": self.scheduler.policy_optimizer.state_dict(),
                "value": self.scheduler.value_optimizer.state_dict(),
                "q": self.scheduler.q_optimizer.state_dict() if self.scheduler.q_optimizer is not None else None,
                "encoder": self.scheduler.encoder_optimizer.state_dict() if self.scheduler.encoder_optimizer is not None else None,
                "action_encoder": self.scheduler.action_encoder_optimizer.state_dict()
                if self.scheduler.action_encoder_optimizer is not None
                else None,
            },
            "training_steps": {
                "world_model": self.world_model.training_step,
                "policy": self.policy.training_step,
                "value": self.value_function.training_step,
                "action_value": self.action_value.training_step,
                "observation_encoder": self.encoder.training_step,
                "action_encoder": self.action_encoder.training_step,
                "scheduler_runtime_steps": self.scheduler.runtime_steps,
            },
            "replay": {
                "storage": list(self.replay_buffer._storage),
                "priorities": list(self.replay_buffer._priorities),
                "position": self.replay_buffer._position,
                "rng_state": self.replay_buffer._rng.getstate(),
            },
            "known_latents": [latent.detach().cpu() for latent in self.known_latents],
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "runtime": self._rng.getstate(),
            },
        }
        temp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temp)
        os.replace(temp, target)
        return target

    def load_training_resume_checkpoint(self, path: str | Path, *, expected_train_completed: int | None = None) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        train_completed = int(payload.get("train_completed", -1))
        if expected_train_completed is not None and train_completed != int(expected_train_completed):
            raise RuntimeError(
                f"Resume checkpoint train_completed={train_completed} does not match progress train_completed={expected_train_completed}."
            )
        models = payload["models"]
        self.world_model.load_state_dict(models["world_model"])
        self.policy.load_state_dict(models["policy"])
        self.value_function.load_state_dict(models["value"])
        self.action_value.load_state_dict(models["action_value"])
        self.encoder.load_state_dict(models["observation_encoder"])
        self.action_encoder.load_state_dict(models["action_encoder"])
        self.scheduler.target_value.load_state_dict(models["target_value"])
        if self.scheduler.target_action_value is not None and models.get("target_action_value") is not None:
            self.scheduler.target_action_value.load_state_dict(models["target_action_value"])
        optimizers = payload["optimizers"]
        self.scheduler.world_optimizer.load_state_dict(optimizers["world"])
        self.scheduler.policy_optimizer.load_state_dict(optimizers["policy"])
        self.scheduler.value_optimizer.load_state_dict(optimizers["value"])
        if self.scheduler.q_optimizer is not None and optimizers.get("q") is not None:
            self.scheduler.q_optimizer.load_state_dict(optimizers["q"])
        if self.scheduler.encoder_optimizer is not None and optimizers.get("encoder") is not None:
            self.scheduler.encoder_optimizer.load_state_dict(optimizers["encoder"])
        if self.scheduler.action_encoder_optimizer is not None and optimizers.get("action_encoder") is not None:
            self.scheduler.action_encoder_optimizer.load_state_dict(optimizers["action_encoder"])
        steps = payload["training_steps"]
        self.world_model.training_step = int(steps.get("world_model", 0))
        self.policy.training_step = int(steps.get("policy", 0))
        self.value_function.training_step = int(steps.get("value", 0))
        self.action_value.training_step = int(steps.get("action_value", 0))
        self.encoder.training_step = int(steps.get("observation_encoder", 0))
        self.action_encoder.training_step = int(steps.get("action_encoder", 0))
        self.scheduler.runtime_steps = int(steps.get("scheduler_runtime_steps", 0))
        replay = payload["replay"]
        self.replay_buffer._storage = list(replay.get("storage", []))
        self.replay_buffer._priorities = list(replay.get("priorities", []))
        self.replay_buffer._position = int(replay.get("position", 0))
        self.replay_buffer._rng.setstate(replay["rng_state"])
        self.known_latents = [latent.detach().clone() for latent in payload.get("known_latents", [])]
        rng = payload["rng"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        self._rng.setstate(rng["runtime"])
        return payload

    def load_latest_checkpoints(self) -> dict[str, bool]:
        payloads = {
            "WorldModel": self.checkpoint_manager.load_latest_module(
                "world_model", self.world_model, optimizer=self.scheduler.world_optimizer
            ),
            "Policy": self.checkpoint_manager.load_latest_module(
                "policy", self.policy, optimizer=self.scheduler.policy_optimizer
            ),
            "ValueFunction": self.checkpoint_manager.load_latest_module(
                "value", self.value_function, optimizer=self.scheduler.value_optimizer
            ),
            "ObservationProjection": self.checkpoint_manager.load_latest_module(
                "observation_projection", self.encoder, optimizer=self.scheduler.encoder_optimizer
            ),
            "ActionProjection": self.checkpoint_manager.load_latest_module(
                "action_projection", self.action_encoder, optimizer=self.scheduler.action_encoder_optimizer
            ),
            "ActionValueNetwork": self.checkpoint_manager.load_latest_module(
                "action_value", self.action_value, optimizer=self.scheduler.q_optimizer
            ),
        }
        modules = {
            "WorldModel": self.world_model,
            "Policy": self.policy,
            "ValueFunction": self.value_function,
            "ObservationProjection": self.encoder,
            "ActionProjection": self.action_encoder,
            "ActionValueNetwork": self.action_value,
        }
        for name, payload in payloads.items():
            if payload is not None and hasattr(modules[name], "training_step"):
                modules[name].training_step = int(payload.get("training_step", 0))
        loaded = {name: payload is not None for name, payload in payloads.items()}
        self.scheduler.target_value.load_state_dict(self.value_function.state_dict())
        if self.scheduler.target_action_value is not None:
            self.scheduler.target_action_value.load_state_dict(self.action_value.state_dict())
        return loaded

    def load_best_checkpoints(self) -> dict[str, bool]:
        payloads = {
            "WorldModel": self.checkpoint_manager.load_best_module(
                "world_model", self.world_model, optimizer=self.scheduler.world_optimizer
            ),
            "Policy": self.checkpoint_manager.load_best_module(
                "policy", self.policy, optimizer=self.scheduler.policy_optimizer
            ),
            "ValueFunction": self.checkpoint_manager.load_best_module(
                "value", self.value_function, optimizer=self.scheduler.value_optimizer
            ),
            "ObservationProjection": self.checkpoint_manager.load_best_module(
                "observation_projection", self.encoder, optimizer=self.scheduler.encoder_optimizer
            ),
            "ActionProjection": self.checkpoint_manager.load_best_module(
                "action_projection", self.action_encoder, optimizer=self.scheduler.action_encoder_optimizer
            ),
            "ActionValueNetwork": self.checkpoint_manager.load_best_module(
                "action_value", self.action_value, optimizer=self.scheduler.q_optimizer
            ),
        }
        modules = {
            "WorldModel": self.world_model,
            "Policy": self.policy,
            "ValueFunction": self.value_function,
            "ObservationProjection": self.encoder,
            "ActionProjection": self.action_encoder,
            "ActionValueNetwork": self.action_value,
        }
        for name, payload in payloads.items():
            if payload is not None and hasattr(modules[name], "training_step"):
                modules[name].training_step = int(payload.get("training_step", 0))
        loaded = {name: payload is not None for name, payload in payloads.items()}
        self.scheduler.target_value.load_state_dict(self.value_function.state_dict())
        if self.scheduler.target_action_value is not None:
            self.scheduler.target_action_value.load_state_dict(self.action_value.state_dict())
        return loaded

    def _log_step_metrics(self, training_report: TrainingReport, success: bool) -> None:
        step = max(1, self.scheduler.runtime_steps)
        self.tensorboard.log_scalar("train/replay_size", len(self.replay_buffer), step)
        if self.state.mode == RuntimeMode.TRAIN:
            self.tensorboard.log_scalar("train/episode_reward", self.state.total_reward, step)
            self.tensorboard.log_scalar("train/success_rate", 1.0 if success else 0.0, step)
            self.tensorboard.log_scalar("train/steps", self.state.step_count, step)
            for attr, tag in [
                ("world_loss", "train/world_loss"),
                ("value_loss", "train/value_loss"),
                ("q_loss", "train/q_loss"),
                ("policy_loss", "train/policy_loss"),
            ]:
                value = getattr(training_report, attr)
                if value is not None:
                    self.tensorboard.log_scalar(tag, value, step)
        else:
            self.tensorboard.log_scalar("eval/success_rate", 1.0 if success else 0.0, step)
            self.tensorboard.log_scalar("eval/mean_reward", self.state.total_reward, step)
            self.tensorboard.log_scalar("eval/mean_steps", self.state.step_count, step)

    def _observation_features(self, observation: CodingObservation) -> SemanticObservationFeatures:
        return self.observation_adapter.encode_semantic(observation, self.text_encoder)

    def _encode_features(self, features: SemanticObservationFeatures) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder.encode_features(features).squeeze(0).detach()

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
        entries: list[dict[str, Any]] = []
        legacy_learned_weight = self._learned_weight()
        with self.profiler.measure("semantic_action_encoding"):
            action_raw_batch = self.action_encoder.raw_features_batch(candidates)
        for candidate, action_raw in zip(candidates, action_raw_batch):
            feasible, feasibility_reason = self._validate_candidate_feasibility(candidate, observation)
            with torch.no_grad():
                action_embedding = self.action_encoder.forward_from_raw(action_raw).squeeze(0)
                prediction = self.world_model(latent, action_embedding)
                predicted_reward = float(prediction.reward_pred.reshape(-1)[0].item()) if prediction.reward_pred is not None else 0.0
                predicted_next_value = float(self.value_function(prediction.next_latent_pred.squeeze(0)).reshape(-1)[0].item())
                q_value = float(self.action_value(latent, action_embedding).reshape(-1)[0].item())
            policy_prior = float(policy_probs[candidate.action_index].item())
            policy_score = float(torch.log(torch.tensor(policy_prior + 1e-8)).item())
            risk = self._estimate_risk(candidate, observation)
            uncertainty = self.uncertainty_estimator.combine(
                model_uncertainty=1.0 / max(1.0, len(self.replay_buffer) ** 0.5),
                memory_uncertainty=1.0 if not self.known_latents else 0.2,
                disagreement=abs(predicted_reward) * 0.05,
                novelty=novelty,
            ).total
            expected_information_gain = 0.5 * uncertainty + 0.5 * novelty
            legacy_learned_score = (
                self.config.policy_score_weight * policy_prior
                + self.config.policy_log_score_weight * policy_score
                + self.config.q_score_weight * q_value
                + self.config.world_reward_weight * predicted_reward
                + self.config.information_gain_weight * expected_information_gain
            )
            heuristic_score = (
                + self.config.confidence_weight * candidate.confidence
                + self._cold_start_workflow_score(candidate, observation)
                - self._stagnation_penalty(candidate, observation)
                - self.config.risk_weight * risk
                - self.config.cost_weight * candidate.estimated_cost
            )
            entries.append(
                {
                    "candidate": candidate,
                    "policy_prior": policy_prior,
                    "policy_score": policy_score,
                    "q_value": q_value,
                    "predicted_reward": predicted_reward,
                    "predicted_next_value": predicted_next_value,
                    "expected_information_gain": float(expected_information_gain),
                    "risk": float(risk),
                    "uncertainty": float(uncertainty),
                    "novelty": float(novelty),
                    "feasible": feasible,
                    "feasibility_reason": feasibility_reason,
                    "heuristic_score": float(heuristic_score),
                    "legacy_learned_score": float(legacy_learned_score),
                }
            )

        normalized = self._normalized_controller_components(entries)
        shadow_mode = self.config.learned_controller_mode == "shadow"
        controller_mode = self.config.eval_controller if self.state.mode == RuntimeMode.EVAL else "training_legacy"
        if shadow_mode:
            controller_mode = self.config.production_controller
        active_components = self._active_controller_components(controller_mode)
        shadow_components = self._active_controller_components(self.config.eval_controller if self.config.eval_controller != "heuristic" else "full")
        gate = self._learned_controller_gate() if self.state.mode == RuntimeMode.EVAL and active_components else 0.0
        shadow_gate = self._learned_controller_gate() if shadow_mode and shadow_components else 0.0
        scored: list[ScoredAction] = []
        for entry, norm in zip(entries, normalized):
            if shadow_mode:
                shadow_learned_score = (
                    self.config.policy_log_score_weight * norm["policy"]
                    if "policy" in shadow_components
                    else 0.0
                )
                shadow_learned_score += self.config.q_score_weight * norm["q"] if "q" in shadow_components else 0.0
                shadow_learned_score += self.config.value_score_weight * norm["value"] if "value" in shadow_components else 0.0
                shadow_learned_score += self.config.world_reward_weight * norm["world"] if "world" in shadow_components else 0.0
                learned_score = shadow_learned_score
                score = entry["heuristic_score"]
                learned_weight = 0.0
                controller_gate = 0.0
            elif self.state.mode == RuntimeMode.EVAL:
                learned_score = (
                    self.config.policy_log_score_weight * norm["policy"]
                    if "policy" in active_components
                    else 0.0
                )
                learned_score += self.config.q_score_weight * norm["q"] if "q" in active_components else 0.0
                learned_score += self.config.value_score_weight * norm["value"] if "value" in active_components else 0.0
                learned_score += self.config.world_reward_weight * norm["world"] if "world" in active_components else 0.0
                score = entry["heuristic_score"] + gate * learned_score
                learned_weight = gate
                shadow_learned_score = 0.0
                controller_gate = gate
            else:
                learned_score = entry["legacy_learned_score"]
                score = entry["heuristic_score"] + legacy_learned_weight * learned_score
                learned_weight = legacy_learned_weight
                shadow_learned_score = 0.0
                controller_gate = 0.0
            if not entry["feasible"]:
                score -= self.config.feasibility_penalty
            scored.append(
                ScoredAction(
                    candidate=entry["candidate"],
                    score=float(score),
                    policy_score=entry["policy_score"],
                    q_value=entry["q_value"],
                    predicted_reward=entry["predicted_reward"],
                    expected_information_gain=entry["expected_information_gain"],
                    risk=entry["risk"],
                    uncertainty=entry["uncertainty"],
                    novelty=entry["novelty"],
                    feasible=entry["feasible"],
                    feasibility_reason=entry["feasibility_reason"],
                    learned_weight=float(learned_weight),
                    heuristic_score=entry["heuristic_score"],
                    learned_score=float(learned_score),
                    q_score=float(norm["q"]),
                    value_score=float(norm["value"]),
                    world_score=float(norm["world"]),
                    final_score=float(score),
                    controller_gate=float(controller_gate),
                    controller_mode=controller_mode,
                    shadow_learned_score=float(shadow_learned_score),
                    shadow_controller_gate=float(shadow_gate),
                    raw_score_components={
                        "policy": float(entry["policy_score"]),
                        "q": float(entry["q_value"]),
                        "value": float(entry["predicted_next_value"]),
                        "world": float(entry["predicted_reward"]),
                    },
                    normalized_score_components={key: float(value) for key, value in norm.items()},
                )
            )
        heuristic_winner = self._heuristic_winner(scored)
        selected_winner = max([item for item in scored if item.feasible] or scored, key=lambda item: item.score)
        shadow_winner = self._shadow_winner(scored) if shadow_mode else None
        for item in scored:
            item.heuristic_winner = item is heuristic_winner
            item.controller_changed_heuristic = selected_winner is not heuristic_winner
            item.shadow_learned_winner = item is shadow_winner
            item.shadow_changed_heuristic = shadow_winner is not None and shadow_winner is not heuristic_winner
        return sorted(scored, key=lambda item: item.score, reverse=True)

    def _normalized_controller_components(self, entries: list[dict[str, Any]]) -> list[dict[str, float]]:
        raw_by_name = {
            "policy": [entry["policy_score"] for entry in entries],
            "q": [entry["q_value"] for entry in entries],
            "value": [entry["predicted_next_value"] for entry in entries],
            "world": [entry["predicted_reward"] for entry in entries],
        }
        normalized_by_name = {name: self._normalize_component(values) for name, values in raw_by_name.items()}
        return [
            {name: normalized_by_name[name][index] for name in normalized_by_name}
            for index in range(len(entries))
        ]

    def _normalize_component(self, values: list[float]) -> list[float]:
        if len(values) < 2:
            return [0.0 for _ in values]
        tensor = torch.tensor(values, dtype=torch.float32)
        spread = float((tensor.max() - tensor.min()).item())
        if spread < 1e-8:
            return [0.0 for _ in values]
        normalized = (tensor - tensor.mean()) / tensor.std(unbiased=False).clamp_min(1e-6)
        clipped = torch.clamp(normalized, -self.config.learned_component_clip, self.config.learned_component_clip)
        return [float(value) for value in clipped.tolist()]

    def _active_controller_components(self, mode: str) -> set[str]:
        modes = {
            "heuristic": set(),
            "policy": {"policy"},
            "policy_q": {"policy", "q"},
            "policy_q_value": {"policy", "q", "value"},
            "full": {"policy", "q", "value", "world"},
            "training_legacy": {"policy", "q", "world"},
        }
        return modes.get(mode, modes["full"])

    def _learned_controller_gate(self) -> float:
        replay_size = len(self.replay_buffer)
        minimum = max(0, int(self.config.learned_gate_min_experiences))
        warmup = max(minimum + 1, int(self.config.learned_gate_warmup_experiences))
        if replay_size < minimum:
            return 0.0
        maturity = min(1.0, (replay_size - minimum) / max(1, warmup - minimum))
        errors = []
        if replay_size:
            try:
                recent = self.replay_buffer.sample_recent(min(replay_size, max(1, int(self.config.learned_gate_recent_window))))
            except (ValueError, IndexError):
                recent = []
            for transition in recent:
                errors.append(abs(float(transition.metadata.get("prediction_error", 0.0))))
        mean_error = sum(errors) / len(errors) if errors else 0.0
        scale = max(1e-6, float(self.config.learned_gate_prediction_error_scale))
        calibration = 1.0 / (1.0 + mean_error / scale)
        return max(0.0, min(1.0, maturity * calibration))

    @staticmethod
    def _heuristic_winner(scored: list[ScoredAction]) -> ScoredAction:
        pool = [item for item in scored if item.feasible] or scored
        return max(pool, key=lambda item: item.heuristic_score)

    @staticmethod
    def _shadow_winner(scored: list[ScoredAction]) -> ScoredAction:
        pool = [item for item in scored if item.feasible] or scored
        return max(pool, key=lambda item: item.heuristic_score + item.shadow_controller_gate * item.shadow_learned_score)

    def _ensure_feasible_scored(
        self,
        latent: torch.Tensor,
        scored: list[ScoredAction],
        observation: CodingObservation,
    ) -> list[ScoredAction]:
        if any(item.feasible for item in scored):
            return scored
        fallback = fallback_candidates(observation)
        fallback_scored = self._score_candidates(latent, fallback, observation)
        self.profiler.increment("post_feasibility_fallback_count", len(fallback_scored))
        if any(item.feasible for item in fallback_scored):
            return fallback_scored
        safe_scored = self._score_candidates(latent, [ActionCandidate(ActionType.LIST_FILES, reasoning_summary="Safe inspection fallback.")], observation)
        self.profiler.increment("post_feasibility_fallback_count", len(safe_scored))
        return safe_scored

    def _learned_weight(self) -> float:
        warmup = max(1, int(self.config.warmup_experiences))
        return max(0.0, min(1.0, len(self.replay_buffer) / warmup))

    def _validate_candidate_feasibility(self, candidate: ActionCandidate, observation: CodingObservation) -> tuple[bool, str]:
        action_type = candidate.action_type
        arguments = candidate.arguments
        if action_type in {ActionType.LIST_FILES, ActionType.RUN_TESTS, ActionType.INSPECT_ERROR}:
            return True, ""
        if action_type == ActionType.SEARCH_TEXT:
            return (True, "") if str(arguments.get("query", "")).strip() else (False, "query is empty")
        if action_type == ActionType.FINISH:
            public_passed = bool(
                observation.test_state.get("ran", False)
                and observation.test_state.get("passed", 0) > 0
                and observation.test_state.get("failed", 0) == 0
            )
            return (True, "") if public_passed else (False, "public tests are not passing")
        if action_type in {ActionType.READ_FILE, ActionType.RUN_PYTHON, ActionType.PATCH_FILE, ActionType.WRITE_FILE}:
            path = str(arguments.get("path", "")).strip()
            path_ok, path_reason = self._validate_relative_action_path(path)
            if not path_ok:
                return False, path_reason
            if not self._candidate_path_editable(path) and action_type in {ActionType.PATCH_FILE, ActionType.WRITE_FILE}:
                return False, "path is protected"
            path_exists = path in observation.workspace_tree
            if action_type in {ActionType.READ_FILE, ActionType.RUN_PYTHON, ActionType.PATCH_FILE} and not path_exists:
                return False, "path does not exist"
            if action_type == ActionType.PATCH_FILE:
                if path not in observation.relevant_file_excerpts:
                    return False, "read file before patching"
                old = str(arguments.get("old", ""))
                if not old:
                    return False, "old text is empty"
                if old not in observation.relevant_file_excerpts[path]:
                    return False, "old text unavailable"
            if action_type == ActionType.WRITE_FILE and path_exists and path not in observation.relevant_file_excerpts:
                return False, "read file before overwriting"
            return True, ""
        return False, "unsupported action type"

    def _validate_relative_action_path(self, path: str) -> tuple[bool, str]:
        if not path:
            return False, "path is empty"
        raw_path = Path(path)
        if raw_path.is_absolute():
            return False, "absolute path is not allowed"
        if ".." in raw_path.parts:
            return False, "parent traversal is not allowed"
        return True, ""

    def _candidate_path_editable(self, path: str) -> bool:
        if self.environment is None:
            return True
        normalized = Path(path).as_posix()
        name = Path(path).name
        return normalized not in self.environment.task.protected_paths and not name.startswith("test_")

    def _cold_start_workflow_score(self, candidate: ActionCandidate, observation: CodingObservation) -> float:
        score = 0.0
        tests_ran = bool(observation.test_state.get("ran", False))
        tests_passing = bool(observation.test_state.get("passed", 0) > 0 and observation.test_state.get("failed", 0) == 0)
        has_excerpts = bool(observation.relevant_file_excerpts)
        code_changed_since_last_test = self._code_changed_since_last_test()
        implementation_unread = any(
            path.endswith(".py")
            and not Path(path).name.startswith("test")
            and path not in observation.relevant_file_excerpts
            for path in observation.workspace_tree
        )
        if code_changed_since_last_test:
            if candidate.action_type == ActionType.RUN_TESTS:
                score += 2.5
            elif candidate.action_type in {
                ActionType.PATCH_FILE,
                ActionType.WRITE_FILE,
                ActionType.RUN_PYTHON,
                ActionType.INSPECT_ERROR,
                ActionType.LIST_FILES,
                ActionType.SEARCH_TEXT,
            }:
                score -= 0.75
        if candidate.action_type == ActionType.RUN_TESTS and not tests_ran:
            score += 0.35
        if candidate.action_type == ActionType.RUN_TESTS and tests_ran and implementation_unread:
            score -= 0.3
        if candidate.action_type == ActionType.LIST_FILES and observation.workspace_tree:
            score -= 0.2
        if candidate.action_type == ActionType.READ_FILE and implementation_unread:
            score += 0.75 if tests_ran else 0.45
        if candidate.action_type in {ActionType.PATCH_FILE, ActionType.WRITE_FILE} and has_excerpts:
            score += 0.15
        if candidate.action_type == ActionType.FINISH and tests_passing:
            score += 0.2
        return score

    def _code_changed_since_last_test(self) -> bool:
        if self.state.trajectory is None:
            return False
        for transition in reversed(self.state.trajectory.transitions):
            action_data = transition.metadata.get("action", {})
            action_type = str(action_data.get("action_type", "")).upper()
            if action_type == ActionType.RUN_TESTS.name:
                return False
            if action_type in {ActionType.PATCH_FILE.name, ActionType.WRITE_FILE.name}:
                metrics = transition.metadata.get("objective_metrics", {})
                if not metrics.get("invalid_action", False):
                    return True
        return False

    def _stagnation_penalty(self, candidate: ActionCandidate, observation: CodingObservation) -> float:
        if self.state.trajectory is None or not self.state.trajectory.transitions:
            return 0.0
        latest = self.state.trajectory.transitions[-1]
        action_data = latest.metadata.get("action", {})
        latest_type = str(action_data.get("action_type", "")).upper()
        if candidate.action_type == ActionType.RUN_TESTS:
            if self._code_changed_since_last_test():
                return 0.0
            return 0.8 if latest_type == ActionType.RUN_TESTS.name else 0.0
        if candidate.action_type == ActionType.INSPECT_ERROR and latest_type == ActionType.INSPECT_ERROR.name:
            if str(latest.next_observation.get("error_output", "")) == str(observation.error_output):
                return 0.6
        if candidate.action_type == ActionType.LIST_FILES and latest_type == ActionType.LIST_FILES.name:
            if latest.next_observation.get("workspace_tree") == observation.workspace_tree:
                return 0.5
        return 0.0

    def _select(self, scored: list[ScoredAction]) -> ScoredAction:
        feasible_scored = [item for item in scored if item.feasible]
        selection_pool = feasible_scored if feasible_scored else scored
        if self.state.mode == RuntimeMode.TRAIN:
            epsilon = max(
                self.config.epsilon_min,
                self.config.train_exploration_epsilon
                * torch.exp(torch.tensor(-self.config.epsilon_decay * max(0, self.scheduler.runtime_steps))).item(),
            )
            if self._rng.random() < epsilon:
                return self._rng.choice(selection_pool)
            scores = torch.tensor([item.score for item in selection_pool], dtype=torch.float32)
            probabilities = F.softmax(scores, dim=0)
            index = int(torch.multinomial(probabilities, 1).item())
            return selection_pool[index]
        return selection_pool[0]

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

    def _trace_step(
        self,
        scored: list[ScoredAction],
        selected: ScoredAction,
        reward: float,
        environment_step: Any,
    ) -> None:
        if not self.config.trace_actions:
            return
        step = self.state.step_count
        max_steps = self.environment.task.max_steps if self.environment is not None else step
        print(f"[{self.trace_label} step {step}/{max_steps}]", flush=True)
        for index, item in enumerate(scored, start=1):
            candidate = item.candidate
            path = str(candidate.arguments.get("path", ""))
            print(
                f"candidate {index}: {candidate.action_type.name}{(' ' + path) if path else ''}\n"
                f"    feasible={item.feasible} reason=\"{item.feasibility_reason}\" total_score={item.score:.4f} "
                f"learned_weight={item.learned_weight:.3f}",
                flush=True,
            )
            if candidate.action_type in {ActionType.PATCH_FILE, ActionType.WRITE_FILE}:
                old = self._truncate_trace_text(str(candidate.arguments.get("old", "")))
                new = self._truncate_trace_text(str(candidate.arguments.get("new", candidate.arguments.get("content", ""))))
                if old or new:
                    print(f"    old=\"{old}\" new=\"{new}\"", flush=True)
        ok = "OK" if getattr(environment_step.action_result, "ok", False) else "FAIL"
        action_result = environment_step.action_result
        print(f"selected: {selected.candidate.action_type.name}", flush=True)
        print(
            f"result: {ok} | reward={reward:.3f} | return_code={getattr(action_result, 'return_code', None)} "
            f"| kind={(getattr(action_result, 'data', {}) or {}).get('failure_kind', '')} "
            f"| message=\"{self._truncate_trace_text(getattr(action_result, 'message', ''))}\"",
            flush=True,
        )
        if not getattr(action_result, "ok", False):
            stdout = self._truncate_trace_text(getattr(action_result, "stdout", ""))
            stderr = self._truncate_trace_text(getattr(action_result, "stderr", ""))
            if stdout:
                print(f"public_stdout=\"{stdout}\"", flush=True)
            if stderr:
                print(f"public_stderr=\"{stderr}\"", flush=True)

    @staticmethod
    def _truncate_trace_text(text: str, limit: int = 120) -> str:
        sanitized = " ".join(text.replace("\r", "\n").split())
        if len(sanitized) > limit:
            return sanitized[: limit - 3] + "..."
        return sanitized

    def _make_transition(
        self,
        observation: CodingObservation,
        observation_features: SemanticObservationFeatures,
        latent: torch.Tensor,
        selected: ScoredAction,
        action_raw_features: torch.Tensor,
        action_embedding: torch.Tensor,
        reward: CodingRewardResult,
        next_observation: CodingObservation,
        next_observation_features: SemanticObservationFeatures,
        next_latent: torch.Tensor,
        done: bool,
        success: bool,
        objective_metrics: dict[str, float | int | str | bool],
        scored_candidates: list[ScoredAction],
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
                "action_type_index": selected.candidate.action_index,
                "observation_features": observation_features.to_metadata(),
                "next_observation_features": next_observation_features.to_metadata(),
                "semantic_text_cache_key": observation_features.cache_key,
                "next_semantic_text_cache_key": next_observation_features.cache_key,
                "action_raw_features": action_raw_features.detach().cpu().tolist(),
                "action_embedding": action_embedding.detach().cpu().tolist(),
                "candidate_scores": [item.to_dict() for item in scored_candidates],
                "action_generation": getattr(self.action_generator, "last_generation_metadata", {}),
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
            "ActionValueNetwork": self.action_value.config.version,
            "ObservationProjection": "observation-projection-0.2",
            "ActionProjection": "action-projection-0.2",
            "Qwen": "not-updated",
        }
