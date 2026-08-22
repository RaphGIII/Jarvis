from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Option:
    """Hierarchical RL option omega=(I_omega, pi_omega, beta_omega)."""

    name: str
    initiation_set: list[str]
    policy_steps: list[str]
    termination_condition: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_initiate(self, available_features: set[str]) -> bool:
        return set(self.initiation_set).issubset(available_features)

    def should_terminate(self, completed_steps: list[str]) -> bool:
        if self.termination_condition == "all_steps_completed":
            return len(completed_steps) >= len(self.policy_steps)
        return self.termination_condition in completed_steps
