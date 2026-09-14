"""The typed EngineeringTaskVector: what kind of engineering a missing capability is.

Built from deterministic signals -- the EngineeringSpec (missing effects, the
partial plan, permissions, acceptance criteria, the owner's words) and the
ZEUS Catalog's impact information (dependents of the modules the change
touches).  Later, empirical success rates per class refine the router's
choice; the vector itself stays a description of the task.

The router chooses exactly ONE engineer before execution.  When the vector
carries frontier indicators the choice is ``engineer.frontier`` directly:
never "standard, see whether it fails, then frontier".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Iterable

FEATURES: tuple[str, ...] = (
    "change_size", "integration_breadth", "technical_novelty", "core_architecture_impact", "codebase_context_required",
    "verification_complexity", "external_system_count", "long_horizon", "risk", "specification_uncertainty",
)

#: Words that name an external system or a distinct subsystem the work must touch.
_SYSTEM_WORDS: dict[str, tuple[str, ...]] = {
    "screen": ("bildschirm", "screen", "beobacht", "observ", "fenster", "window"),
    "engine": ("stockfish", "engine", "analysier", "analyse", "analyz"),
    "storage": ("speicher", "datei", "file", "storage", "datenbank", "database", "persist"),
    "projects": ("projekt", "project"),
    "profile": ("profil", "profile", "langfristig", "long-term", "longterm", "dauerhaft", "historie"),
    "training": ("training", "trainings", "übung", "exercise", "personalisiert", "personalized"),
    "audio": ("mikrofon", "microphone", "audio", "sprach", "voice", "stimme"),
    "web": ("browser", "website", "web", "http", "api"),
    "calendar": ("kalender", "calendar", "termin"),
    "mail": ("mail", "e-mail", "email"),
    "ui": ("oberfläche", "ui", "anzeige", "dashboard", "ansicht"),
    "vision": ("kamera", "camera", "bild", "image", "ocr", "erkenn"),
}

_LONG_HORIZON = re.compile(r"\b(langfristig|long[- ]?term|dauerhaft|dauernd|über wochen|wochen|monate|months|weeks|immer wenn|"
                           r"whenever|nur wenn|beobachte|monitor|kontinuierlich|continuous|ongoing|historie|history|profil|profile)\w*",
                           re.I)
_NOVELTY = re.compile(r"\b(beobacht|observ|erkenn|recogni|klassifizier|classif|kategorisier|categori|lern|learn|modell|model|"
                      r"personalis|adaptiv|adaptive|vorhersag|predict)\w*", re.I)
_RISK = re.compile(r"\b(lösch|delete|remove|entfern|überschreib|overwrite|zahl|pay|kauf|buy|send|versend|mail|passw|secret|"
                   r"schlüssel|credential|system|registry|autostart)\w*", re.I)
_CORE_WORDS = re.compile(r"\b(zeus selbst|dich selbst|kern|core|architektur|architecture|router|gateway|lifecycle|supervisor|"
                         r"alle fähigkeiten|jede fähigkeit|grundsätzlich|überall)\w*", re.I)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale(count: float, full_at: float) -> float:
    return 0.0 if count <= 0 else _clamp(count / float(full_at))


@dataclass(frozen=True)
class EngineeringTaskVector:
    change_size: float = 0.0
    integration_breadth: float = 0.0
    technical_novelty: float = 0.0
    core_architecture_impact: float = 0.0
    codebase_context_required: float = 0.0
    verification_complexity: float = 0.0
    external_system_count: float = 0.0
    long_horizon: float = 0.0
    risk: float = 0.0
    specification_uncertainty: float = 0.0
    #: Where each number came from, for the audit trail.
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in FEATURES:
            object.__setattr__(self, name, _clamp(getattr(self, name)))

    @property
    def systems(self) -> list[str]:
        return list(self.signals.get("systems", []))

    def as_features(self) -> dict[str, float]:
        return {name: round(getattr(self, name), 3) for name in FEATURES}

    @property
    def task_class(self) -> str:
        """engineering.small / medium / large, from the vector alone."""

        if self.frontier_indicators():
            return "engineering.large"
        score = max(self.change_size, self.integration_breadth, self.core_architecture_impact, self.long_horizon)
        if score >= 0.7 or self.external_system_count >= 0.75:
            return "engineering.large"
        if score >= 0.35 or self.external_system_count >= 0.4 or self.technical_novelty >= 0.5:
            return "engineering.medium"
        return "engineering.small"

    def frontier_indicators(self) -> list[str]:
        """The hard indicators present.  Two or more mean frontier quality is required."""

        present: list[str] = []
        if self.core_architecture_impact >= 0.6:
            present.append("core architecture change")
        if self.change_size >= 0.7:
            present.append("major new subsystem")
        if self.integration_breadth >= 0.6:
            present.append("many independent subsystems integrated")
        if self.long_horizon >= 0.6:
            present.append("long autonomous implementation horizon / persistent state")
        if self.verification_complexity >= 0.6:
            present.append("high verification complexity")
        if self.external_system_count >= 0.75:
            present.append("multimodal + tools + persistence across several external systems")
        return present

    @property
    def frontier_required(self) -> bool:
        indicators = self.frontier_indicators()
        return len(indicators) >= 2 or self.core_architecture_impact >= 0.85

    def to_dict(self) -> dict[str, Any]:
        return {**self.as_features(), "task_class": self.task_class, "frontier_required": self.frontier_required,
                "frontier_indicators": self.frontier_indicators(), "signals": dict(self.signals)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineeringTaskVector":
        data = dict(data or {})
        return cls(**{f.name: data.get(f.name, 0.0) for f in fields(cls) if f.name in FEATURES}, signals=dict(data.get("signals") or {}))


def systems_named(text: str) -> list[str]:
    lower = str(text or "").lower()
    return [name for name, words in _SYSTEM_WORDS.items() if any(w in lower for w in words)]


def vector_from_spec(spec: Any, *, catalog_dependents: int = 0, impacted_core_modules: Iterable[str] = (),
                     goal_ambiguity: float | None = None) -> EngineeringTaskVector:
    """Deterministic signals from an EngineeringSpec (or a dict of its fields) and the Catalog's impact view."""

    get = (lambda key, default=None: spec.get(key, default)) if isinstance(spec, dict) else (lambda key, default=None: getattr(spec, key, default))
    owner_goal = str(get("owner_goal", "") or "")
    goal_spec = dict(get("goal_spec", {}) or {})
    missing = [str(m) for m in (get("missing_effects", []) or [])]
    partial = [str(c) for c in (get("closest_partial_plan", []) or [])]
    reusable = [str(c) for c in (get("reusable_capabilities", []) or [])]
    permissions = [str(p) for p in (get("required_permissions", []) or [])]
    acceptance = [str(a) for a in (get("acceptance_criteria", []) or [])]
    privacy = [str(p) for p in (get("privacy_constraints", []) or [])]
    text = owner_goal + " " + " ".join(goal_spec.get("secondary_goals", []) or []) + " " + str(goal_spec.get("primary_goal", "") or "")
    words = len(owner_goal.split())
    systems = systems_named(text)
    core_modules = [str(m) for m in impacted_core_modules]

    novelty_hits = len(set(m.group(0).lower() for m in _NOVELTY.finditer(text)))
    long_hits = len(set(m.group(0).lower() for m in _LONG_HORIZON.finditer(text)))
    risk_hits = len(set(m.group(0).lower() for m in _RISK.finditer(text)))
    core_hits = len(set(m.group(0).lower() for m in _CORE_WORDS.finditer(text)))
    clauses = len([c for c in re.split(r"[,;.]|\bund\b|\band\b", owner_goal) if c.strip()])

    change_size = max(_scale(len(missing), 3), _scale(words, 45), _scale(clauses, 6) * 0.9)
    integration_breadth = max(_scale(len(systems), 4), _scale(len(missing), 4) * 0.8, _scale(len(partial), 5) * 0.6)
    technical_novelty = max(_scale(novelty_hits, 3), 0.35 if missing and not reusable and not partial else (0.2 if missing else 0.0))
    core_architecture_impact = max(_scale(core_hits, 2), _scale(len(core_modules), 2), _scale(catalog_dependents, 12) * 0.8)
    codebase_context_required = max(_scale(catalog_dependents, 8), _scale(len(reusable) + len(core_modules), 4), 0.2 if partial else 0.0)
    verification_complexity = max(_scale(len(acceptance), 10), _scale(len(systems), 3) * 0.8,
                                  0.7 if "screen" in systems or "vision" in systems or "audio" in systems else 0.0)
    external_system_count = _scale(len(systems), 4)
    long_horizon = max(_scale(long_hits, 2), _scale(len(missing), 5) * 0.5)
    risk = max(_scale(risk_hits, 2), _scale(len(permissions), 3) * 0.7, 0.3 if "screen" in systems else 0.0,
               0.4 if any(p.startswith("the request is private") or p.startswith("the request is secret") for p in privacy) else 0.0)
    if goal_ambiguity is None:
        try:
            goal_ambiguity = float(goal_spec.get("ambiguity", 0.0) or 0.0)
        except (TypeError, ValueError):
            goal_ambiguity = 0.0
    specification_uncertainty = max(_clamp(goal_ambiguity), 0.5 if not missing else 0.0, 0.3 if not acceptance else 0.0)

    return EngineeringTaskVector(
        change_size=change_size, integration_breadth=integration_breadth, technical_novelty=technical_novelty,
        core_architecture_impact=core_architecture_impact, codebase_context_required=codebase_context_required,
        verification_complexity=verification_complexity, external_system_count=external_system_count, long_horizon=long_horizon,
        risk=risk, specification_uncertainty=specification_uncertainty,
        signals={"systems": systems, "missing_effects": len(missing), "partial_plan": len(partial), "reusable": len(reusable),
                 "words": words, "clauses": clauses, "novelty_words": novelty_hits, "long_horizon_words": long_hits,
                 "risk_words": risk_hits, "core_words": core_hits, "catalog_dependents": int(catalog_dependents),
                 "impacted_core_modules": core_modules, "permissions": len(permissions), "acceptance_criteria": len(acceptance)},
    )


def task_vector_for(vector: EngineeringTaskVector, text: str) -> Any:
    """The gateway's TaskVector for this engineering task: the same class, the same difficulty signals."""

    from dataclasses import replace

    from gateway.task import TaskClass, TaskFacts, rule_based

    task_class = vector.task_class
    facts = TaskFacts(text=text, is_engineering=True, capability_missing=True,
                      estimated_files_changed={"engineering.small": 1, "engineering.medium": 3, "engineering.large": 8}[task_class],
                      subsystems=max(1, len(vector.systems)) if task_class != "engineering.small" else 1,
                      new_subsystem=task_class == "engineering.large", needs_screen="screen" in vector.systems,
                      external_systems=len(vector.systems))
    task = rule_based(facts)
    try:
        task = replace(task, task_class=TaskClass(task_class))
    except (TypeError, ValueError):
        pass
    task.facts["engineering_vector"] = vector.as_features()
    task.facts["frontier_required"] = vector.frontier_required
    return task
