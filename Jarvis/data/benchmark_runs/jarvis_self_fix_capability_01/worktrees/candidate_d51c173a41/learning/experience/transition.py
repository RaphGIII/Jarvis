from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class Transition:
    """Single experience tuple used by all learning mechanisms.

    It represents (o_t, z_t, a_t, r_t, o_(t+1), z_(t+1)).
    """

    observation: Any
    latent_state: Tensor | None
    action: Any
    reward: float
    next_observation: Any
    next_latent_state: Tensor | None
    done: bool
    uncertainty: float = 0.0
    novelty: float = 0.0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction_error(self) -> float:
        return float(self.metadata.get("prediction_error", 0.0))

    def with_metadata(self, **metadata: Any) -> "Transition":
        merged = dict(self.metadata)
        merged.update(metadata)
        return Transition(
            observation=self.observation,
            latent_state=self.latent_state,
            action=self.action,
            reward=self.reward,
            next_observation=self.next_observation,
            next_latent_state=self.next_latent_state,
            done=self.done,
            uncertainty=self.uncertainty,
            novelty=self.novelty,
            success=self.success,
            metadata=merged,
        )
