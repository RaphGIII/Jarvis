"""The typed task vector: what a request is, before anyone asks a model.

One difficulty number hides the thing the router needs to know.  A request
can be trivial to reason about and dangerous to execute, or harmless and
deeply ambiguous; the two need different routes.  So a task is a vector of
named features in [0, 1], and each feature says where its value came from.

Most values are *objective*: ZEUS knows from its own registry whether a
capability is missing, from its graph how many subsystems are touched, from
the request whether the filesystem is written or the screen is needed.  A
semantic model estimates only the soft features -- implicit intent,
ambiguity, reasoning depth -- and the final value of every feature is the
maximum of the rule-based and the semantic estimate.  Nothing averages a
difficult task down.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

FEATURES: tuple[str, ...] = (
    "novelty",
    "reasoning_depth",
    "ambiguity",
    "context_dependency",
    "integration_breadth",
    "long_horizon",
    "multimodal",
    "external_dependencies",
    "action_risk",
    "failure_cost",
    "code_change",
    "privacy_class",
    "fresh_information_required",
)

#: Features the semantic model is allowed to raise.  The rest are ZEUS's own
#: facts and a model's opinion about them is ignored.
SOFT_FEATURES: frozenset[str] = frozenset({"novelty", "reasoning_depth", "ambiguity", "context_dependency", "long_horizon"})


class TaskClass(str, Enum):
    """The class the reliability model learns over."""

    #: Public-knowledge question, translation, summary, explanation.
    KNOWLEDGE = "knowledge"
    #: Interpreting the owner's meaning: intent, references, paraphrase.
    SEMANTIC = "semantic"
    #: Multi-step goals, ordering, deciding what to do.
    PLANNING = "planning"
    #: Combining several capabilities into a workflow.
    COMPOSITION = "composition"
    ENGINEERING_SMALL = "engineering.small"
    ENGINEERING_MEDIUM = "engineering.medium"
    ENGINEERING_LARGE = "engineering.large"


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if number < 0.0 else 1.0 if number > 1.0 else number


@dataclass(frozen=True)
class TaskVector:
    novelty: float = 0.0
    reasoning_depth: float = 0.0
    ambiguity: float = 0.0
    context_dependency: float = 0.0
    integration_breadth: float = 0.0
    long_horizon: float = 0.0
    multimodal: float = 0.0
    external_dependencies: float = 0.0
    action_risk: float = 0.0
    failure_cost: float = 0.0
    code_change: float = 0.0
    privacy_class: float = 0.0
    fresh_information_required: float = 0.0
    task_class: TaskClass = TaskClass.KNOWLEDGE
    #: feature -> "rule" | "semantic" | "max(rule,semantic)"
    provenance: dict[str, str] = field(default_factory=dict)
    #: Objective facts the rules used, kept for the record.
    facts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in FEATURES:
            object.__setattr__(self, name, _clamp(getattr(self, name)))

    # -- derived -----------------------------------------------------------

    @property
    def difficulty(self) -> float:
        """A summary for humans and thresholds; the router uses the vector."""

        cognitive = max(self.novelty, self.reasoning_depth, self.ambiguity, self.long_horizon)
        structural = max(self.integration_breadth, self.context_dependency, self.code_change)
        return round(max(cognitive, 0.8 * structural), 3)

    @property
    def required_reliability(self) -> float:
        """tau(x): how sure we must be before a route is acceptable.

        Cheap to be wrong about a knowledge question the owner will read and
        judge; expensive to be wrong about an action with a high failure cost.
        """

        base = 0.55
        base += 0.25 * max(self.failure_cost, self.action_risk)
        base += 0.05 * self.code_change
        return round(min(0.97, base), 3)

    def merged_with_semantic(self, soft: dict[str, Any]) -> "TaskVector":
        """D_final = max(D_rule_based, D_semantic_model), soft features only."""

        changes: dict[str, Any] = {}
        provenance = dict(self.provenance)
        for name, raw in (soft or {}).items():
            if name not in SOFT_FEATURES:
                continue
            value = _clamp(raw)
            current = getattr(self, name)
            if value > current:
                changes[name] = value
                provenance[name] = "semantic" if current == 0.0 else "max(rule,semantic)"
        if not changes:
            return self
        return TaskVector(**{**self.as_features(), **changes}, task_class=self.task_class, provenance=provenance, facts=dict(self.facts))

    def as_features(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in FEATURES}

    def to_dict(self) -> dict[str, Any]:
        return {**self.as_features(), "task_class": self.task_class.value, "difficulty": self.difficulty,
                "required_reliability": self.required_reliability, "provenance": dict(self.provenance),
                "facts": dict(self.facts)}


# ---------------------------------------------------------------------------
# Rule-based construction from ZEUS's own facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskFacts:
    """What ZEUS knows objectively about a request before any model runs."""

    text: str = ""
    #: Registry verdicts.
    capability_found: bool = False
    capability_confidence: float = 0.0
    capability_missing: bool = False
    capability_broken: bool = False
    #: Number of subsystems / capabilities the request touches (graph).
    subsystems: int = 0
    #: Objective needs.
    needs_screen: bool = False
    writes_filesystem: bool = False
    destructive: bool = False
    security_sensitive: bool = False
    needs_fresh_information: bool = False
    external_systems: int = 0
    #: Engineering facts.
    is_engineering: bool = False
    estimated_files_changed: int = 0
    new_subsystem: bool = False
    #: From the privacy router.
    privacy_ceiling: str = "PUBLIC"
    #: Conversation facts.
    refers_to_context: bool = False
    turns_in_conversation: int = 0
    is_question: bool = False


def _scale(count: int, full_at: int) -> float:
    return 0.0 if count <= 0 else min(1.0, count / float(full_at))


def rule_based(facts: TaskFacts) -> TaskVector:
    """Objective features from facts.  Deterministic and cheap."""

    provenance = {name: "rule" for name in FEATURES}
    text = str(facts.text or "")
    words = len(text.split())

    if facts.is_engineering:
        if facts.new_subsystem or facts.subsystems >= 4 or facts.estimated_files_changed >= 8:
            task_class = TaskClass.ENGINEERING_LARGE
        elif facts.subsystems >= 2 or facts.estimated_files_changed >= 3 or facts.capability_missing:
            task_class = TaskClass.ENGINEERING_MEDIUM
        else:
            task_class = TaskClass.ENGINEERING_SMALL
    elif facts.subsystems >= 2:
        task_class = TaskClass.COMPOSITION
    elif facts.capability_found or facts.capability_missing or facts.capability_broken or facts.refers_to_context:
        task_class = TaskClass.SEMANTIC
    elif facts.is_question or words <= 40:
        task_class = TaskClass.KNOWLEDGE
    else:
        task_class = TaskClass.PLANNING

    privacy = {"PUBLIC": 0.0, "PRIVATE": 0.6, "SECRET": 1.0}.get(str(facts.privacy_ceiling).upper(), 0.6)
    vector = TaskVector(
        novelty=1.0 if facts.new_subsystem else (0.6 if facts.capability_missing else 0.0),
        reasoning_depth=0.7 if task_class in {TaskClass.PLANNING, TaskClass.COMPOSITION} else (0.5 if facts.is_engineering else 0.0),
        ambiguity=0.0,
        context_dependency=0.6 if facts.refers_to_context else min(0.4, 0.05 * facts.turns_in_conversation),
        integration_breadth=_scale(facts.subsystems, 4),
        long_horizon=0.8 if facts.new_subsystem else _scale(facts.estimated_files_changed, 10),
        multimodal=1.0 if facts.needs_screen else 0.0,
        external_dependencies=_scale(facts.external_systems, 3),
        action_risk=1.0 if facts.destructive else (0.7 if facts.security_sensitive else (0.4 if facts.writes_filesystem else 0.0)),
        failure_cost=1.0 if facts.destructive else (0.6 if facts.writes_filesystem or facts.is_engineering else 0.1),
        code_change=1.0 if facts.is_engineering else (0.5 if facts.capability_missing or facts.capability_broken else 0.0),
        privacy_class=privacy,
        fresh_information_required=1.0 if facts.needs_fresh_information else 0.0,
        task_class=task_class,
        provenance=provenance,
        facts={
            "capability_found": facts.capability_found, "capability_confidence": facts.capability_confidence,
            "capability_missing": facts.capability_missing, "capability_broken": facts.capability_broken,
            "subsystems": facts.subsystems, "needs_screen": facts.needs_screen, "writes_filesystem": facts.writes_filesystem,
            "destructive": facts.destructive, "security_sensitive": facts.security_sensitive,
            "needs_fresh_information": facts.needs_fresh_information, "external_systems": facts.external_systems,
            "is_engineering": facts.is_engineering, "estimated_files_changed": facts.estimated_files_changed,
            "new_subsystem": facts.new_subsystem, "privacy_ceiling": facts.privacy_ceiling,
            "refers_to_context": facts.refers_to_context, "words": words,
        },
    )
    return vector


def feature_names() -> list[str]:
    return [f.name for f in fields(TaskVector) if f.name in FEATURES]
