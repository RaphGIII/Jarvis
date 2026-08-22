from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor


@dataclass
class FailureExperience:
    z_before: Tensor | None
    action: Any
    predicted_outcome: Any
    actual_outcome: Any
    prediction_error: float
    reward: float
    failure_category: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FailurePredictor(Protocol):
    def probability_of_failure(self, latent_state: Tensor, action: Any) -> float:
        ...


class HeuristicFailurePredictor:
    """First non-parametric predictor until enough data exists for a model."""

    def __init__(self) -> None:
        self._counts: dict[str, tuple[int, int]] = {}

    def record(self, action: Any, failed: bool) -> None:
        key = str(action)
        failures, attempts = self._counts.get(key, (0, 0))
        self._counts[key] = (failures + int(failed), attempts + 1)

    def probability_of_failure(self, latent_state: Tensor, action: Any) -> float:
        failures, attempts = self._counts.get(str(action), (0, 0))
        if attempts == 0:
            return 0.5
        return failures / attempts
