from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeCandidate:
    candidate_id: str
    description: str
    patch_summary: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitnessResult:
    correctness: float = 0.0
    test_score: float = 0.0
    performance: float = 0.0
    generalization: float = 0.0
    complexity: float = 0.0
    risk: float = 0.0

    def effective_fitness(
        self,
        correctness_weight: float = 1.0,
        tests_weight: float = 1.0,
        performance_weight: float = 0.3,
        generalization_weight: float = 0.7,
        complexity_weight: float = 0.2,
        risk_weight: float = 1.0,
    ) -> float:
        return float(
            correctness_weight * self.correctness
            + tests_weight * self.test_score
            + performance_weight * self.performance
            + generalization_weight * self.generalization
            - complexity_weight * self.complexity
            - risk_weight * self.risk
        )


@dataclass
class Experiment:
    experiment_id: str
    candidates: list[CodeCandidate] = field(default_factory=list)
    fitness_results: dict[str, FitnessResult] = field(default_factory=dict)
    promoted_candidate_id: str | None = None

    def promote_best(self) -> str | None:
        if not self.fitness_results:
            return None
        self.promoted_candidate_id = max(
            self.fitness_results,
            key=lambda candidate_id: self.fitness_results[candidate_id].effective_fitness(),
        )
        return self.promoted_candidate_id
