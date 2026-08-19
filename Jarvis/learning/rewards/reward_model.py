from __future__ import annotations

from dataclasses import dataclass

from learning.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class RewardWeights:
    task: float = DEFAULT_CONFIG.reward_task_weight
    user: float = DEFAULT_CONFIG.reward_user_weight
    accuracy: float = DEFAULT_CONFIG.reward_accuracy_weight
    efficiency: float = DEFAULT_CONFIG.reward_efficiency_weight
    novelty: float = DEFAULT_CONFIG.reward_novelty_weight
    learning: float = DEFAULT_CONFIG.reward_learning_weight
    error: float = DEFAULT_CONFIG.reward_error_weight
    risk: float = DEFAULT_CONFIG.reward_risk_weight


@dataclass
class RewardSignal:
    task_success: float = 0.0
    user_feedback: float = 0.0
    correctness: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    learning_progress: float = 0.0
    error_penalty: float = 0.0
    risk_penalty: float = 0.0

    def total(self, weights: RewardWeights = RewardWeights()) -> float:
        return float(
            weights.task * self.task_success
            + weights.user * self.user_feedback
            + weights.accuracy * self.correctness
            + weights.efficiency * self.efficiency
            + weights.novelty * self.novelty
            + weights.learning * self.learning_progress
            - weights.error * self.error_penalty
            - weights.risk * self.risk_penalty
        )

    def components(self) -> dict[str, float]:
        return {
            "task_success": self.task_success,
            "user_feedback": self.user_feedback,
            "correctness": self.correctness,
            "efficiency": self.efficiency,
            "novelty": self.novelty,
            "learning_progress": self.learning_progress,
            "error_penalty": self.error_penalty,
            "risk_penalty": self.risk_penalty,
        }


class MultiObjectiveRewardModel:
    """Composes separate reward components without scattering hard rewards."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def score(self, signal: RewardSignal) -> float:
        return signal.total(self.weights)
