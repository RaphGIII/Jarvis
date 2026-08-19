from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch
from torch import Tensor


class ActionType(IntEnum):
    LIST_FILES = 0
    READ_FILE = 1
    SEARCH_TEXT = 2
    WRITE_FILE = 3
    PATCH_FILE = 4
    RUN_TESTS = 5
    RUN_PYTHON = 6
    INSPECT_ERROR = 7
    FINISH = 8


@dataclass
class ActionCandidate:
    action_type: ActionType
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning_summary: str = ""
    confidence: float = 0.5
    estimated_cost: float = 1.0

    @property
    def action_index(self) -> int:
        return int(self.action_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "arguments": self.arguments,
            "reasoning_summary": self.reasoning_summary,
            "confidence": self.confidence,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCandidate":
        raw_type = data.get("action_type", ActionType.LIST_FILES.name)
        if isinstance(raw_type, ActionType):
            action_type = raw_type
        elif isinstance(raw_type, int):
            action_type = ActionType(raw_type)
        else:
            action_type = ActionType[str(raw_type).strip().upper()]
        return cls(
            action_type=action_type,
            arguments=dict(data.get("arguments") or {}),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            confidence=float(data.get("confidence", 0.5)),
            estimated_cost=float(data.get("estimated_cost", 1.0)),
        )


class ActionEncoder:
    """Deterministic action embedding boundary, trainable later if needed."""

    def __init__(self, embedding_dim: int | None = None) -> None:
        self.num_actions = len(ActionType)
        self.embedding_dim = embedding_dim or self.num_actions
        if self.embedding_dim < self.num_actions:
            raise ValueError("embedding_dim must cover all action types")

    def encode(self, action_candidate: ActionCandidate) -> Tensor:
        vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
        vector[action_candidate.action_index] = 1.0
        if self.embedding_dim > self.num_actions:
            vector[self.num_actions] = max(0.0, min(1.0, action_candidate.confidence))
        if self.embedding_dim > self.num_actions + 1:
            vector[self.num_actions + 1] = min(1.0, max(0.0, action_candidate.estimated_cost / 10.0))
        return vector


def coerce_action(candidate: ActionCandidate | dict[str, Any]) -> ActionCandidate:
    if isinstance(candidate, ActionCandidate):
        return candidate
    return ActionCandidate.from_dict(candidate)
