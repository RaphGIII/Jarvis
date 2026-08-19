from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from learning.curriculum.difficulty import DifficultyEstimator, TaskFeatures


@dataclass
class TaskCandidate:
    task_id: str
    features: TaskFeatures
    predicted_success: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CurriculumManager:
    """Selects tasks near the zone of proximal development."""

    def __init__(
        self,
        difficulty_estimator: DifficultyEstimator | None = None,
        lower_success: float = 0.60,
        upper_success: float = 0.85,
    ) -> None:
        self.difficulty_estimator = difficulty_estimator or DifficultyEstimator()
        self.lower_success = lower_success
        self.upper_success = upper_success

    def select_next_task(self, candidates: list[TaskCandidate]) -> TaskCandidate | None:
        if not candidates:
            return None
        zpd = [
            candidate
            for candidate in candidates
            if self.lower_success < candidate.predicted_success < self.upper_success
        ]
        if zpd:
            return max(zpd, key=lambda candidate: self.difficulty_estimator.score(candidate.features))
        midpoint = (self.lower_success + self.upper_success) / 2.0
        return min(candidates, key=lambda candidate: abs(candidate.predicted_success - midpoint))
