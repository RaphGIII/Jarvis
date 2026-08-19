from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from learning.experience.replay_buffer import ReplayBuffer
from learning.experience.transition import Transition
from learning.policy.exploration import CandidateAction, CandidateActionScorer
from learning.representations.embeddings import numeric_observation_to_tensor, one_hot
from learning.representations.encoder import ObservationEncoder
from learning.rewards.reward_model import MultiObjectiveRewardModel, RewardSignal
from learning.world_model.model import WorldModel


class HighLevelPlanner(Protocol):
    """LLM-facing boundary: propose candidates, do not train the LLM online."""

    def propose_actions(self, goal: str, observation: Any) -> list[CandidateAction]:
        ...


@dataclass
class LearningLoopTrace:
    observation: Any
    latent_shape: tuple[int, ...]
    action: Any
    reward: float
    prediction_error: float | None = None


@dataclass
class DevelopmentalLearningLoop:
    """observe -> encode -> predict/rank -> act -> store -> update."""

    encoder: ObservationEncoder
    replay_buffer: ReplayBuffer
    reward_model: MultiObjectiveRewardModel = field(default_factory=MultiObjectiveRewardModel)
    action_scorer: CandidateActionScorer = field(default_factory=CandidateActionScorer)
    world_model: WorldModel | None = None

    def choose_action(self, candidate_actions: list[CandidateAction]) -> CandidateAction:
        if not candidate_actions:
            raise ValueError("candidate_actions must not be empty")
        return self.action_scorer.rank(candidate_actions)[0]

    def record_experience(
        self,
        observation: Any,
        action: Any,
        reward_signal: RewardSignal,
        next_observation: Any,
        done: bool,
        *,
        success: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> LearningLoopTrace:
        obs_tensor = numeric_observation_to_tensor(observation)
        next_obs_tensor = numeric_observation_to_tensor(next_observation)
        latent = self.encoder.encode(obs_tensor).detach().squeeze(0)
        next_latent = self.encoder.encode(next_obs_tensor).detach().squeeze(0)
        reward = self.reward_model.score(reward_signal)
        transition = Transition(
            observation=observation,
            latent_state=latent,
            action=action,
            reward=reward,
            next_observation=next_observation,
            next_latent_state=next_latent,
            done=done,
            uncertainty=float(metadata.get("uncertainty", 0.0)) if metadata else 0.0,
            novelty=float(metadata.get("novelty", 0.0)) if metadata else 0.0,
            success=success,
            metadata=metadata or {},
        )
        self.replay_buffer.add(transition, error=transition.prediction_error)
        return LearningLoopTrace(
            observation=observation,
            latent_shape=tuple(latent.shape),
            action=action,
            reward=reward,
            prediction_error=transition.prediction_error,
        )

    def train_world_model_once(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        action_dim: int,
    ) -> dict[str, float]:
        if self.world_model is None:
            raise ValueError("world_model is required")
        batch = self.replay_buffer.sample(batch_size)
        if any(t.latent_state is None or t.next_latent_state is None for t in batch):
            raise ValueError("all sampled transitions need latent states")
        latent = torch.stack([t.latent_state for t in batch if t.latent_state is not None])
        next_latent = torch.stack([t.next_latent_state for t in batch if t.next_latent_state is not None])
        actions = torch.stack([one_hot(int(t.action), action_dim) for t in batch])
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        return self.world_model.train_step(optimizer, latent, actions, next_latent, rewards)
