from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LearningStrategy:
    name: str
    exploration_rate: float = 0.1
    replay_ratio: float = 0.25
    learning_rate: float = 1e-3
    curriculum_difficulty: float = 0.5
    reflection_frequency: int = 10
    memory_retrieval_depth: int = 5
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyEvaluation:
    strategy: LearningStrategy
    mean_future_reward: float
    compute_cost: float
    objective_value: float


class MetaLearner:
    def __init__(self, compute_penalty: float = 0.05) -> None:
        self.compute_penalty = compute_penalty
        self.evaluations: list[StrategyEvaluation] = []

    def evaluate(self, strategy: LearningStrategy, mean_future_reward: float, compute_cost: float) -> StrategyEvaluation:
        objective = mean_future_reward - self.compute_penalty * compute_cost
        evaluation = StrategyEvaluation(strategy, mean_future_reward, compute_cost, objective)
        self.evaluations.append(evaluation)
        return evaluation

    def best_strategy(self) -> LearningStrategy | None:
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda item: item.objective_value).strategy
