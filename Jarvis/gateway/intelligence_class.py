"""Which class of intelligence a request needs -- decided before any provider is chosen.

The order of decisions is the whole point.  ZEUS first asks *what the task
requires*: nothing (a deterministic answer), the zero-cost pool, the smart
reasoner, the deep reasoner, or an engineer.  Only then does availability
routing happen inside that class.  There is no ladder: a request never runs a
cheap model to find out whether it was good enough and then a stronger one.
One request, one intended class, recorded before the first byte moves.

The classifier reads the task vector (rule-based facts, semantic soft
features) and the owner's own words.  It chooses the LOWEST class predicted
to be reliably sufficient -- not the cheapest, not automatically the
strongest -- with confidence margins around the boundaries: near the
zero-cost/smart boundary, when a wrong answer would waste meaningful owner
time, SMART is chosen directly.

The owner's mode outranks the classifier (§4): FREE pins ZERO_COST, SMART
and DEEP pin their class, BUILD pins engineering.  AUTO classifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gateway.modes import ChatMode
from gateway.task import TaskClass, TaskVector


class IntelligenceClass(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    ZERO_COST = "ZERO_COST"
    SMART = "SMART"
    DEEP = "DEEP"
    BUILD_STANDARD = "BUILD_STANDARD"
    BUILD_FRONTIER = "BUILD_FRONTIER"

    @property
    def roles(self) -> tuple[str, ...]:
        """The gateway roles that serve this class.  Provider names never appear here."""

        return {
            IntelligenceClass.DETERMINISTIC: (),
            IntelligenceClass.ZERO_COST: ("reasoning.free",),
            IntelligenceClass.SMART: ("reasoning.smart",),
            IntelligenceClass.DEEP: ("reasoning.deep",),
            IntelligenceClass.BUILD_STANDARD: ("engineer.standard", "engineer.codex"),
            IntelligenceClass.BUILD_FRONTIER: ("engineer.frontier", "engineer.frontier_alt"),
        }[self]

    @property
    def is_engineering(self) -> bool:
        return self in {IntelligenceClass.BUILD_STANDARD, IntelligenceClass.BUILD_FRONTIER}

    @property
    def is_paid(self) -> bool:
        return self in {IntelligenceClass.SMART, IntelligenceClass.DEEP, IntelligenceClass.BUILD_STANDARD, IntelligenceClass.BUILD_FRONTIER}


#: The reasoning classes in ascending order of capability and cost.
REASONING_ORDER: tuple[IntelligenceClass, ...] = (IntelligenceClass.ZERO_COST, IntelligenceClass.SMART, IntelligenceClass.DEEP)

#: Score boundaries between the reasoning classes, and the margin inside
#: which a boundary decision is "close".
ZERO_SMART_BOUNDARY = 0.35
SMART_DEEP_BOUNDARY = 0.65
BOUNDARY_MARGIN = 0.07


@dataclass
class ClassDecision:
    intelligence_class: IntelligenceClass
    #: The combined requirement score in [0, 1] (0 for owner-pinned classes).
    score: float = 0.0
    #: How far the score sits from the nearest boundary, 0..1 (1 = far).
    confidence: float = 1.0
    margin: float = 1.0
    reason: str = ""
    #: The individual signals, each 0..1, for the record.
    signals: dict[str, float] = field(default_factory=dict)
    #: "owner" (the mode pinned the class), "classifier", "engineering", "deterministic".
    source: str = "classifier"
    owner_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"intelligence_class": self.intelligence_class.value, "score": round(self.score, 3), "confidence": round(self.confidence, 3),
                "margin": round(self.margin, 3), "reason": self.reason, "signals": {k: round(v, 3) for k, v in self.signals.items()},
                "source": self.source, "owner_override": self.owner_override}


# ---------------------------------------------------------------------------
# Lexical signals: what the owner's words say about the work, language-agnostic stems
# ---------------------------------------------------------------------------

def _fold(text: str) -> str:
    lowered = (text or "").lower().replace("ß", "ss")
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(source, target)
    return lowered


_ANALYSIS = re.compile(r"\b(analysier\w*|vergleich\w*|leite?\s+\w*\s*ab|ableit\w*|beweis\w*|kausal\w*|methodisch\w*|kritisch\w*|bewert\w*"
                       r"|interpret\w*|diskutier\w*|evaluier\w*|begruend\w*|argumentier\w*|widerleg\w*|schwaech\w*|staerk\w*|abwaeg\w*"
                       r"|compare|derive|critique|evaluat\w*|analy[sz]\w*|assess\w*|interpret\w*|justif\w*|weigh)\b")
_DESIGN = re.compile(r"\b(entwirf\w*|entwerf\w*|design\w*|konzipier\w*|architektur\w*|architecture|strategie\w*|strategy|plane?\b|planung"
                     r"|schritt\s+fuer\s+schritt|step\s+by\s+step|roadmap|konzept\w*|optimier\w*|optimi[sz]\w*)\b")
_PRECISION = re.compile(r"\b(exakt\w*|praezis\w*|rigoros\w*|wahrscheinlichste\w*|quantitativ\w*|formal\w*|streng\w*|nachweis\w*|belege?\b"
                        r"|precise\w*|rigorous\w*|quantitative\w*|formal\w*|prove|most\s+likely|exact\w*)\b")
_STAKES = re.compile(r"\b(klinisch\w*|diagnos\w*|therapie\w*|dosis|dosierung|sicherheit\w*|rechtlich\w*|juristisch\w*|finanz\w*|steuer\w*"
                     r"|vertrag\w*|clinical|diagnos\w*|dosage|legal|safety|financial|contract\w*|notfall|emergency)\b")
_DELIVERABLE_LIGHT = re.compile(r"\b(ausfuehrlich\w*|detailliert\w*|genau\w*|gruendlich\w*|thorough\w*|detailed|in\s+detail)\b")
_DELIVERABLE_HEAVY = re.compile(r"\b(umfassend\w*|vollstaendig\w*|komplett\w*|dokument\w*|bericht\w*|gutachten|whitepaper|spezifikation\w*"
                                r"|comprehensive|complete\s+(?:report|document|specification)|report|specification)\b")
_NOVELTY = re.compile(r"\b(neu\w*|eigene\w*|erfinde\w*|innovativ\w*|noch\s+nie|novel|invent\w*|original\w*|from\s+scratch)\b")
_MULTISTEP = re.compile(r"\b(und\s+(?:dann|danach|anschliessend)|zuerst|erstens|zweitens|drittens|schliesslich|first|then|finally|afterwards)\b")
_CONSTRAINT_COUNT = re.compile(r"\b(\d{1,3})\s+(?:technischen\s+|harten\s+|weiteren\s+)?(?:constraints?|bedingungen|anforderungen|randbedingungen|kriterien"
                               r"|vorgaben|regeln|requirements|criteria|rules)\b")
_CONSTRAINT_WORDS = re.compile(r"\b(muss|muessen|darf\s+nicht|duerfen\s+nicht|mindestens|hoechstens|maximal|minimal|unter\s+der\s+bedingung|sofern"
                               r"|must|must\s+not|at\s+least|at\s+most|no\s+more\s+than|subject\s+to|given\s+that)\b")
_CONTEXT_REFS = re.compile(r"\b(diese[rsn]?\s+(?:drei|vier|fuenf|zwei|beiden|\d+)|die\s+(?:drei|vier|beiden|folgenden)|folgende[rsn]?|anhaengend\w*"
                           r"|beigefuegt\w*|these\s+(?:three|four|two|\d+)|the\s+following|attached)\b")
_TRIVIAL_QUESTION = re.compile(r"^\s*(was\s+(?:ist|bedeutet|heisst)|wer\s+(?:ist|war)|wie\s+heisst|what\s+(?:is|does)\s+\w+\s+mean|what\s+is|who\s+(?:is|was)|define)\b")


def _hits(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def signals_for(task: TaskVector, text: str) -> dict[str, float]:
    """Every factor the class depends on, each 0..1, from the vector and the words."""

    body = _fold(text)
    words = len(body.split())
    facts = task.facts or {}

    analysis = min(1.0, 0.45 * _hits(_ANALYSIS, body))
    design = min(1.0, 0.5 * _hits(_DESIGN, body))
    semantic = max(task.reasoning_depth, analysis, 0.6 * design)
    if _TRIVIAL_QUESTION.match(body) and words <= 12 and analysis == 0.0:
        semantic = min(semantic, 0.15)

    counted = _CONSTRAINT_COUNT.search(body)
    constraints = 0.0
    if counted:
        constraints = min(1.0, int(counted.group(1)) / 8.0)
    constraints = max(constraints, min(1.0, 0.2 * _hits(_CONSTRAINT_WORDS, body)))

    context = max(task.context_dependency, task.integration_breadth, min(1.0, 0.4 * _hits(_CONTEXT_REFS, body)), min(1.0, words / 400.0))
    reasoning = max(task.reasoning_depth, analysis * 0.9, design * 0.8, min(1.0, 0.35 * _hits(_MULTISTEP, body)))
    planning = max(task.long_horizon, design, 0.6 if task.task_class in {TaskClass.PLANNING, TaskClass.COMPOSITION} else 0.0)
    tools = max(task.integration_breadth, min(1.0, int(facts.get("subsystems", 0) or 0) / 3.0))
    novelty = max(task.novelty, min(1.0, 0.4 * _hits(_NOVELTY, body)))
    stakes = max(task.failure_cost, task.action_risk, min(1.0, 0.35 * _hits(_STAKES, body)))
    precision = min(1.0, 0.35 * _hits(_PRECISION, body))
    output = max(0.3 if _DELIVERABLE_LIGHT.search(body) else 0.0, 0.5 if _DELIVERABLE_HEAVY.search(body) else 0.0,
                 0.7 if design and _DESIGN.search(body) and "architektur" in body or "architecture" in body else 0.0)
    coding = max(task.code_change, 1.0 if facts.get("is_engineering") else 0.0)
    return {
        "semantic_complexity": semantic, "constraints": constraints, "ambiguity": task.ambiguity, "context_integration": context,
        "reasoning_depth": reasoning, "planning_depth": planning, "tool_coordination": tools, "novelty": novelty,
        "stakes": stakes, "precision": precision, "output_complexity": output, "coding": coding,
    }


def _score(signals: dict[str, float]) -> float:
    """0.6 × the strongest requirement + 0.4 × the mean of the three strongest.

    A single hard requirement (fifteen constraints) is enough to need deep
    reasoning; several moderate ones add up.  A weighted mean would let one
    hard requirement drown among easy ones.
    """

    considered = [v for k, v in signals.items() if k not in {"coding", "ambiguity"}]
    considered.sort(reverse=True)
    top = considered[0] if considered else 0.0
    mean3 = sum(considered[:3]) / max(1, min(3, len(considered)))
    return round(min(1.0, 0.6 * top + 0.4 * mean3), 4)


def _class_for_score(score: float) -> IntelligenceClass:
    if score < ZERO_SMART_BOUNDARY:
        return IntelligenceClass.ZERO_COST
    if score < SMART_DEEP_BOUNDARY:
        return IntelligenceClass.SMART
    return IntelligenceClass.DEEP


def _margin(score: float) -> float:
    return min(abs(score - ZERO_SMART_BOUNDARY), abs(score - SMART_DEEP_BOUNDARY))


def classify(task: TaskVector, mode: ChatMode | str, *, text: str = "", engineering_frontier: bool | None = None,
             deterministic: bool = False) -> ClassDecision:
    """The class this request needs, before any provider is considered.

    ``deterministic`` says a typed/registry path answers without a model.
    ``engineering_frontier`` is the engineering router's verdict when the
    request is engineering (None = not yet decided; standard is assumed).
    """

    mode = ChatMode.parse(mode)
    signals = signals_for(task, text)
    facts = task.facts or {}
    engineering = bool(facts.get("is_engineering")) or task.code_change >= 0.9 or task.task_class in {
        TaskClass.ENGINEERING_SMALL, TaskClass.ENGINEERING_MEDIUM, TaskClass.ENGINEERING_LARGE}

    if deterministic and not engineering:
        return ClassDecision(IntelligenceClass.DETERMINISTIC, reason="a typed path answers without a model", signals=signals,
                             source="deterministic")

    if mode is ChatMode.BUILD or engineering:
        frontier = bool(engineering_frontier) if engineering_frontier is not None else (
            task.task_class is TaskClass.ENGINEERING_LARGE or bool(facts.get("new_subsystem")))
        cls = IntelligenceClass.BUILD_FRONTIER if frontier else IntelligenceClass.BUILD_STANDARD
        why = "engineering: routed to the engineering router directly" + (" (owner: BUILD)" if mode is ChatMode.BUILD else "")
        return ClassDecision(cls, score=_score(signals), reason=why, signals=signals, source="engineering",
                             owner_override=mode is ChatMode.BUILD)

    pinned = {ChatMode.FREE: IntelligenceClass.ZERO_COST, ChatMode.SMART: IntelligenceClass.SMART, ChatMode.DEEP: IntelligenceClass.DEEP}
    if mode in pinned:
        return ClassDecision(pinned[mode], score=_score(signals), reason=f"owner mode {mode.value} pins the class", signals=signals,
                             source="owner", owner_override=True)

    score = _score(signals)
    cls = _class_for_score(score)
    margin = _margin(score)
    confidence = round(min(1.0, margin / 0.15), 3)
    reason = f"score {score:.2f}"
    words = len((text or "").split())
    if cls is IntelligenceClass.ZERO_COST and (ZERO_SMART_BOUNDARY - score) < BOUNDARY_MARGIN and (
            signals["stakes"] >= 0.3 or signals["output_complexity"] >= 0.5 or signals["precision"] >= 0.3 or words > 60):
        cls = IntelligenceClass.SMART
        reason += "; near the zero-cost boundary and a wrong answer would waste owner time: SMART directly"
    elif cls is IntelligenceClass.SMART and (SMART_DEEP_BOUNDARY - score) < BOUNDARY_MARGIN and (
            signals["constraints"] >= 0.6 or signals["planning_depth"] >= 0.7):
        cls = IntelligenceClass.DEEP
        reason += "; near the smart/deep boundary with hard constraints or deep planning: DEEP directly"
    else:
        strongest = max(signals.items(), key=lambda kv: kv[1] if kv[0] not in {"coding", "ambiguity"} else -1.0)
        reason += f"; strongest factor {strongest[0]} {strongest[1]:.2f}"
    return ClassDecision(cls, score=score, confidence=confidence, margin=round(margin, 4), reason=reason, signals=signals,
                         source="classifier")


def class_for_role(role: str) -> IntelligenceClass | None:
    """The class a pinned role belongs to, for requests the caller already routed (an engineer)."""

    for cls in IntelligenceClass:
        if role in cls.roles:
            return cls
    return None
