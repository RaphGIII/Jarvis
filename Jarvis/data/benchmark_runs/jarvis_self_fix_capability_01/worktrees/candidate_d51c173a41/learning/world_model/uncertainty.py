from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UncertaintyWeights:
    model: float = 1.0
    memory: float = 0.7
    disagreement: float = 1.0
    novelty: float = 0.6


@dataclass
class UncertaintyEstimate:
    model_uncertainty: float = 0.0
    memory_uncertainty: float = 0.0
    disagreement: float = 0.0
    novelty: float = 0.0
    weights: UncertaintyWeights = field(default_factory=UncertaintyWeights)

    @property
    def total(self) -> float:
        numerator = (
            self.weights.model * self.model_uncertainty
            + self.weights.memory * self.memory_uncertainty
            + self.weights.disagreement * self.disagreement
            + self.weights.novelty * self.novelty
        )
        denominator = (
            self.weights.model
            + self.weights.memory
            + self.weights.disagreement
            + self.weights.novelty
        )
        return float(numerator / max(denominator, 1e-12))

    def action_mode(self, high: float = 0.7, medium: float = 0.35) -> str:
        if self.total >= high:
            return "seek_more_information"
        if self.total >= medium:
            return "act_carefully"
        return "act_directly"


class UncertaintyEstimator:
    def __init__(self, weights: UncertaintyWeights | None = None) -> None:
        self.weights = weights or UncertaintyWeights()

    def combine(
        self,
        model_uncertainty: float = 0.0,
        memory_uncertainty: float = 0.0,
        disagreement: float = 0.0,
        novelty: float = 0.0,
    ) -> UncertaintyEstimate:
        return UncertaintyEstimate(
            model_uncertainty=model_uncertainty,
            memory_uncertainty=memory_uncertainty,
            disagreement=disagreement,
            novelty=novelty,
            weights=self.weights,
        )
