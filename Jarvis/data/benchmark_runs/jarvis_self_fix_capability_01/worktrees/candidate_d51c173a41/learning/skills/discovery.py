from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from learning.experience.trajectory import Trajectory
from learning.skills.option import Option


@dataclass
class DiscoveredSkill:
    name: str
    action_sequence: tuple[str, ...]
    support: int
    success_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_option(self) -> Option:
        return Option(
            name=self.name,
            initiation_set=[],
            policy_steps=list(self.action_sequence),
            termination_condition="all_steps_completed",
            metadata={"support": self.support, "success_rate": self.success_rate},
        )


class SkillDiscovery:
    """Heuristic discovery over repeated successful action subsequences."""

    def __init__(self, min_support: int = 2, sequence_length: int = 3) -> None:
        self.min_support = min_support
        self.sequence_length = sequence_length

    def discover(self, trajectories: list[Trajectory]) -> list[DiscoveredSkill]:
        counts: Counter[tuple[str, ...]] = Counter()
        successes: Counter[tuple[str, ...]] = Counter()
        for trajectory in trajectories:
            actions = tuple(str(action) for action in trajectory.actions)
            for start in range(0, max(0, len(actions) - self.sequence_length + 1)):
                sequence = actions[start : start + self.sequence_length]
                counts[sequence] += 1
                if trajectory.final_success:
                    successes[sequence] += 1

        discovered: list[DiscoveredSkill] = []
        for sequence, support in counts.items():
            if support >= self.min_support:
                success_rate = successes[sequence] / support
                discovered.append(
                    DiscoveredSkill(
                        name="skill_" + "_".join(sequence),
                        action_sequence=sequence,
                        support=support,
                        success_rate=success_rate,
                    )
                )
        return sorted(discovered, key=lambda skill: (skill.success_rate, skill.support), reverse=True)
