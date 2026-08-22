from __future__ import annotations

from dataclasses import dataclass

from learning.meta.self_model import SelfModel


@dataclass
class MetacognitiveAssessment:
    weakest_capabilities: list[str]
    high_uncertainty_capabilities: list[str]
    improving_capabilities: list[str]


class MetacognitionEngine:
    def assess(self, self_model: SelfModel) -> MetacognitiveAssessment:
        estimates = list(self_model.capabilities.values())
        return MetacognitiveAssessment(
            weakest_capabilities=[item.capability for item in self_model.weakest_capabilities()],
            high_uncertainty_capabilities=[item.capability for item in estimates if item.uncertainty > 0.7],
            improving_capabilities=[item.capability for item in estimates if item.trend > 0.05],
        )
