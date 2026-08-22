from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from learning.experience.transition import Transition


@dataclass
class Trajectory:
    """A task-level sequence of transitions."""

    transitions: list[Transition] = field(default_factory=list)
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        self.transitions.extend(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        return float(sum(t.reward for t in self.transitions))

    @property
    def success_rate(self) -> float:
        if not self.transitions:
            return 0.0
        return sum(1 for t in self.transitions if t.success) / len(self.transitions)

    @property
    def final_success(self) -> bool:
        return bool(self.transitions and self.transitions[-1].success)

    @property
    def actions(self) -> list[Any]:
        return [transition.action for transition in self.transitions]

    def to_rl_records(self) -> list[dict[str, Any]]:
        return [
            {
                "state": transition.latent_state,
                "action": transition.action,
                "reward": transition.reward,
                "next_state": transition.next_latent_state,
                "done": transition.done,
            }
            for transition in self.transitions
        ]
