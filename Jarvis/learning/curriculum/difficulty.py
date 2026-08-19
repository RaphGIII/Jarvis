from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskFeatures:
    normalized_steps: float = 0.0
    normalized_tools: float = 0.0
    uncertainty: float = 0.0
    novelty: float = 0.0


@dataclass(frozen=True)
class DifficultyWeights:
    steps: float = 0.35
    tools: float = 0.25
    uncertainty: float = 0.25
    novelty: float = 0.15


class DifficultyEstimator:
    def __init__(self, weights: DifficultyWeights | None = None) -> None:
        self.weights = weights or DifficultyWeights()

    def score(self, features: TaskFeatures) -> float:
        score = (
            self.weights.steps * features.normalized_steps
            + self.weights.tools * features.normalized_tools
            + self.weights.uncertainty * features.uncertainty
            + self.weights.novelty * features.novelty
        )
        return float(max(0.0, min(1.0, score)))
