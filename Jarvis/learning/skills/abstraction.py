from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillPattern:
    name: str
    preconditions: list[str]
    action_schema: list[str]
    expected_outcome: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, available_features: set[str]) -> bool:
        return set(self.preconditions).issubset(available_features)
