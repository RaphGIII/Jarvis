from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class LatentState:
    vector: Tensor
    uncertainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.vector.shape[-1])


@dataclass
class BeliefState:
    """POMDP belief state b_t=f(theta)(o_1:t, a_1:t-1, memory)."""

    latent: LatentState
    history_length: int = 1
    memory_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, latent: LatentState, memory_refs: list[str] | None = None) -> "BeliefState":
        refs = list(self.memory_refs)
        if memory_refs:
            refs.extend(memory_refs)
        return BeliefState(
            latent=latent,
            history_length=self.history_length + 1,
            memory_refs=refs,
            metadata=dict(self.metadata),
        )
