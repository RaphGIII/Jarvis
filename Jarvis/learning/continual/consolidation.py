from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from learning.experience.trajectory import Trajectory


@dataclass
class ConsolidationResult:
    semantic_candidates: list[dict[str, Any]] = field(default_factory=list)
    procedural_candidates: list[dict[str, Any]] = field(default_factory=list)
    archive_candidates: list[str] = field(default_factory=list)


class MemoryConsolidator:
    """Consolidates repeated episodic traces into semantic/procedural candidates."""

    def __init__(self, min_repetitions: int = 2) -> None:
        self.min_repetitions = min_repetitions

    def consolidate(self, trajectories: list[Trajectory]) -> ConsolidationResult:
        action_sequences = Counter(tuple(str(action) for action in trajectory.actions) for trajectory in trajectories)
        procedural = [
            {"action_sequence": list(sequence), "support": count}
            for sequence, count in action_sequences.items()
            if count >= self.min_repetitions and sequence
        ]
        success_rewards = [trajectory.total_reward for trajectory in trajectories if trajectory.final_success]
        semantic = []
        if success_rewards:
            semantic.append(
                {
                    "concept": "successful_task_reward_distribution",
                    "mean_reward": sum(success_rewards) / len(success_rewards),
                    "samples": len(success_rewards),
                }
            )
        return ConsolidationResult(semantic_candidates=semantic, procedural_candidates=procedural)
