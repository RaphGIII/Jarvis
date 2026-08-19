from __future__ import annotations

import math
from dataclasses import dataclass


def compute_forgetting(max_previous_score: float, current_score: float) -> float:
    return float(max(0.0, max_previous_score - current_score))


@dataclass(frozen=True)
class MemoryStrengthWeights:
    importance: float = 0.4
    retrieval: float = 0.2
    reward: float = 0.3
    age: float = 0.1


@dataclass
class MemoryStrength:
    importance: float
    retrieval_count: int
    reward: float
    age: float

    def score(self, weights: MemoryStrengthWeights = MemoryStrengthWeights()) -> float:
        return float(
            weights.importance * self.importance
            + weights.retrieval * math.log1p(self.retrieval_count)
            + weights.reward * self.reward
            - weights.age * self.age
        )


class ForgettingTracker:
    def __init__(self) -> None:
        self.max_scores: dict[str, float] = {}
        self.current_scores: dict[str, float] = {}

    def update(self, task_id: str, score: float) -> None:
        self.current_scores[task_id] = score
        self.max_scores[task_id] = max(score, self.max_scores.get(task_id, score))

    def forgetting(self, task_id: str) -> float:
        return compute_forgetting(self.max_scores.get(task_id, 0.0), self.current_scores.get(task_id, 0.0))
