from __future__ import annotations

from dataclasses import dataclass, field

from learning.continual.forgetting import ForgettingTracker


@dataclass
class ContinualLearner:
    """Tracks replay/consolidation statistics and forgetting metrics."""

    replay_ratio: float = 0.25
    consolidation_interval: int = 100
    stability_weight: float = 0.5
    plasticity_weight: float = 0.5
    forgetting_tracker: ForgettingTracker = field(default_factory=ForgettingTracker)

    def update_benchmark_score(self, task_id: str, score: float) -> None:
        self.forgetting_tracker.update(task_id, score)

    def forgetting(self, task_id: str) -> float:
        return self.forgetting_tracker.forgetting(task_id)

    @property
    def stability_plasticity_balance(self) -> float:
        total = self.stability_weight + self.plasticity_weight
        if total <= 0:
            return 0.5
        return self.stability_weight / total
