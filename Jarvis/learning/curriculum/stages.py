from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DevelopmentalStage(IntEnum):
    NEWBORN = 0
    INFANT = 1
    CHILD = 2
    LEARNER = 3
    ADOLESCENT = 4
    ADVANCED = 5
    EXPERT = 6
    META_LEARNER = 7
    SELF_IMPROVING = 8


@dataclass
class StageEvaluation:
    stage: DevelopmentalStage
    score: float
    missing_capabilities: list[str]


class DevelopmentalStageEvaluator:
    """Evaluates stages by measurable competence rather than age/time."""

    requirements: dict[DevelopmentalStage, dict[str, float]] = {
        DevelopmentalStage.NEWBORN: {},
        DevelopmentalStage.INFANT: {"single_tool": 0.5},
        DevelopmentalStage.CHILD: {"short_sequences": 0.5},
        DevelopmentalStage.LEARNER: {"failure_detection": 0.6},
        DevelopmentalStage.ADOLESCENT: {"skill_abstraction": 0.6},
        DevelopmentalStage.ADVANCED: {"transfer": 0.65},
        DevelopmentalStage.EXPERT: {"complex_planning": 0.7},
        DevelopmentalStage.META_LEARNER: {"learning_strategy_analysis": 0.7},
        DevelopmentalStage.SELF_IMPROVING: {"controlled_self_improvement": 0.8},
    }

    def evaluate(self, capabilities: dict[str, float]) -> StageEvaluation:
        best = DevelopmentalStage.NEWBORN
        missing: list[str] = []
        for stage in DevelopmentalStage:
            required = self.requirements[stage]
            current_missing = [
                capability
                for capability, threshold in required.items()
                if capabilities.get(capability, 0.0) < threshold
            ]
            if current_missing:
                missing = current_missing
                break
            best = stage
        requirement = self.requirements[best]
        if not requirement:
            score = 0.0
        else:
            score = sum(min(1.0, capabilities.get(cap, 0.0) / threshold) for cap, threshold in requirement.items())
            score /= len(requirement)
        return StageEvaluation(stage=best, score=float(score), missing_capabilities=missing)
