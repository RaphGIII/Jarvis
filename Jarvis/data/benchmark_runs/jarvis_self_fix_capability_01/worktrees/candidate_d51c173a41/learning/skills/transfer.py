from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransferProposal:
    source_skill: str
    target_task: str
    shared_features: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
