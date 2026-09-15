"""The authoritative semantic flow: the reasoning provider understands and proposes,
the deterministic planner grounds, validates and proves.

    owner utterance
      -> relevant world/context retrieval           (deterministic, cheap)
      -> reasoning provider  -> GoalSpec            (closed vocabulary only)
      -> compact capability cards (stage 1)         (a few hundred tokens)
      -> cheap deterministic path?  else
         reasoning provider -> PlanSpec              (full contracts, stage 2)
      -> deterministic validation of the PlanSpec   (requires/provides simulation)
         invalid -> the exact problem back to the provider, once
         still invalid -> state-space proof: a plan, or MISSING with evidence
      -> execute (core) -> verify effects -> GOAL_SATISFIED

The model is the intelligence.  The planner is not a second brain: it grounds
tokens, checks feasibility, finds trivial paths, and proves that an effect is
unreachable.  The legacy local model never produces a GoalSpec or a PlanSpec:
when only it is reachable the flow returns INTELLIGENCE_UNAVAILABLE.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from capabilities.contracts import SemanticContract, normalise_token
from capabilities.planner import CapabilityNode, CompositionPlanner, MissingCapabilityEvidence, Plan, PlanReport, WorldState, plans_for_goals
from gateway.estimate import estimate_tokens

PRIVACY_CLASSES = ("public", "private", "secret")

_WORD = re.compile(r"[a-zäöüß0-9]{3,}", re.I)
_STOP = frozenset({"the", "and", "der", "die", "das", "und", "ich", "mir", "mich", "mein", "meine", "meinen", "ist", "bin",
                   "hab", "habe", "schon", "wieder", "mal", "bitte", "jetzt", "noch", "nicht", "kein", "keine", "was", "wie",
                   "wer", "you", "for", "with", "von", "des", "dem", "den", "ein", "eine", "einen", "einem", "auf", "aus",
                   "sie", "wir", "uns", "nur", "auch", "dann", "kann", "soll", "will", "please", "again", "just"})


def words_of(text: str) -> set[str]:
    """The content words of a sentence: retrieval keys, never meaning."""

    return {w.lower() for w in _WORD.findall(str(text or "")) if w.lower() not in _STOP}


def card_vocabulary(capability_id: str, contract: SemanticContract, description: str = "", examples: Iterable[str] = (),
                    name: str = "") -> frozenset[str]:
    """Domain words of a capability: from id, name, goals, effects, events, projects, prose and examples."""

    out: set[str] = set()
    for token in (capability_id, name, contract.domain, *contract.goals, *contract.provides, *contract.events,
                  *contract.related_projects):
        out |= {w for w in re.split(r"[._\s\-]+", str(token).lower()) if len(w) >= 3}
    out |= words_of(description)
    for example in examples:
        out |= words_of(example)
    return frozenset(out)

#: The built-in primitives of the executor, as contracts, so a PlanSpec may
#: mix them with learned capabilities.  Tokens are generic effects.
BUILTIN_CONTRACTS: dict[str, dict[str, Any]] = {
    "file.write": {"produces": ["file_written"], "goals": ["write_file"], "risk_class": "reversible"},
    "file.read": {"produces": ["file_content"], "goals": ["read_file"]},
    "file.open": {"effects": ["program_window_open"], "goals": ["open_file"]},
    "project.create": {"produces": ["project_record"], "goals": ["create_project"], "risk_class": "reversible"},
    "music.play": {"effects": ["music_playing"], "goals": ["play_music"]},
    "music.pause": {"effects": ["music_paused"], "goals": ["pause_music"]},
    "music.resume": {"effects": ["music_playing"], "goals": ["resume_music"]},
    "knowledge.search": {"produces": ["knowledge_search_results"], "goals": ["find_knowledge"]},
    "knowledge.create": {"produces": ["knowledge_node"], "goals": ["store_knowledge"], "risk_class": "reversible"},
    "knowledge.link": {"produces": ["knowledge_relation"], "goals": ["link_knowledge"], "risk_class": "reversible"},
    "knowledge.read": {"produces": ["knowledge_node_content"], "goals": ["read_knowledge"]},
    "timer.start": {"effects": ["timer_running"], "goals": ["start_timer"]},
    "note.create": {"produces": ["note_file"], "goals": ["write_note"], "risk_class": "reversible"},
    "window.hide": {"effects": ["window_hidden"], "goals": ["hide_window"]},
    "say": {"effects": ["owner_informed"], "goals": ["inform_owner"]},
}

#: Retrieval vocabulary for the built-ins (German and English words the owner
#: uses for them).  Retrieval only: which cards a request may be about, never
#: what it means -- the provider decides that.
BUILTIN_WORDS: dict[str, tuple[str, ...]] = {
    "file.write": ("datei", "file", "schreib", "speicher", "write", "save"),
    "file.read": ("datei", "file", "lies", "read", "inhalt"),
    "file.open": ("öffne", "open", "datei", "ordner", "folder"),
    "project.create": ("projekt", "project", "anlegen", "erstelle", "create"),
    "music.play": ("musik", "music", "spiel", "play", "song", "lied", "spotify"),
    "music.pause": ("musik", "music", "pause", "stopp", "stop"),
    "music.resume": ("musik", "music", "weiter", "resume"),
    "knowledge.search": ("wissen", "knowledge", "such", "search", "finde"),
    "knowledge.create": ("wissen", "knowledge", "speicher", "merk", "befund", "erkenntnis", "notier", "store", "remember"),
    "knowledge.link": ("wissen", "knowledge", "verknüpf", "link", "verbinde"),
    "knowledge.read": ("wissen", "knowledge", "lies", "zeig", "read"),
    "timer.start": ("timer", "wecker", "minuten", "countdown"),
    "note.create": ("notiz", "note", "aufschreib"),
    "window.hide": ("fenster", "window", "versteck", "hide"),
    "say": ("sag", "say", "sprich"),
}


def _tokens(values: Any) -> list[str]:
    out: list[str] = []
    for value in values if isinstance(values, list) else []:
        token = normalise_token(value)
        if token and token not in out:
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Capability cards, two stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Card:
    capability_id: str
    contract: SemanticContract
    purpose: str
    examples: tuple[str, ...] = ()
    words: frozenset[str] = frozenset()
    builtin: bool = False
    inputs: dict[str, str] = field(default_factory=dict)
    health: str = "HEALTHY"

    def summary(self) -> str:
        """Stage 1: one compact line -- a few dozen tokens."""

        parts = [f"- {self.capability_id}: {self.purpose[:90]}"]
        if self.contract.goals:
            parts.append(f"  goals: {', '.join(self.contract.goals[:4])}")
        if self.contract.provides:
            parts.append(f"  produces: {', '.join(sorted(self.contract.provides)[:4])}")
        if self.contract.requires:
            parts.append(f"  needs: {', '.join(sorted(self.contract.requires)[:4])}")
        return "\n".join(parts)

    def detail(self) -> str:
        """Stage 2: the full contract and the input schema, for a selected card."""

        data = {"id": self.capability_id, "purpose": self.purpose, "contract": self.contract.to_dict(), "inputs": self.inputs,
                "health": self.health}
        if self.examples:
            data["examples"] = list(self.examples[:3])
        return json.dumps(data, ensure_ascii=False)

    def to_node(self) -> CapabilityNode:
        return CapabilityNode(self.capability_id, self.contract, self.health)


def builtin_cards(available_context: Iterable[str] = ("screen", "speaker", "microphone")) -> list[Card]:
    from service.composer import BUILTIN_PRIMITIVES

    available = set(available_context)
    cards: list[Card] = []
    for prim in BUILTIN_PRIMITIVES:
        if any(req not in available for req in prim.requires):
            continue
        contract = SemanticContract.from_dict(BUILTIN_CONTRACTS.get(prim.name, {"effects": [normalise_token(prim.name) + ".done"]}))
        words = frozenset(BUILTIN_WORDS.get(prim.name, ())) | frozenset(w for w in re.split(r"[.\s_]+", prim.name) if len(w) >= 3)
        cards.append(Card(prim.name, contract, prim.purpose, words=words, builtin=True, inputs=dict(prim.inputs)))
    return cards


def capability_cards(registry: Any) -> list[Card]:
    cards: list[Card] = []
    for manifest in registry.all():
        if not manifest.is_active() or manifest.is_broken():
            continue
        try:
            contract = manifest.semantic_contract()
        except Exception:  # noqa: BLE001
            continue
        purpose = str(manifest.description or manifest.name or "").split("\n")[0][:160]
        words = card_vocabulary(manifest.capability_id, contract, purpose, tuple(manifest.examples[:4]), manifest.name)
        schema = manifest.input_schema if isinstance(manifest.input_schema, dict) else {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        inputs = {k: str((v or {}).get("type", "value")) if isinstance(v, dict) else "value" for k, v in props.items()}
        cards.append(Card(manifest.capability_id, contract, purpose, tuple(manifest.examples[:4]), words,
                          inputs=inputs, health=manifest.health_state().value))
    return cards


# ---------------------------------------------------------------------------
# Context retrieval: only what the request is about
# ---------------------------------------------------------------------------

@dataclass
class ProjectSummary:
    title: str
    summary: str = ""

    @property
    def token(self) -> str:
        return f"project.{normalise_token(self.title)}"


@dataclass
class RetrievedContext:
    cards: list[Card]
    events: list[tuple[str, str]]  # (token, source)
    projects: list[ProjectSummary]
    reasons: dict[str, str]
    state: WorldState

    @property
    def empty(self) -> bool:
        return not self.cards

    def world_text(self) -> str:
        lines = [f"- {token} ({source})" for token, source in self.events] or ["- (kein aktuelles Ereignis)"]
        for project in self.projects:
            lines.append(f"- {project.token} (project: {project.title}" + (f" -- {project.summary[:120]}" if project.summary else "") + ")")
        return "\n".join(lines)

    def summaries_text(self) -> str:
        return "\n".join(card.summary() for card in self.cards) or "- (keine)"

    def to_dict(self) -> dict[str, Any]:
        return {"cards": [c.capability_id for c in self.cards], "events": [t for t, _ in self.events],
                "projects": [p.title for p in self.projects], "reasons": dict(self.reasons)}


def retrieve(text: str, state: WorldState, cards: Iterable[Card], projects: Iterable[ProjectSummary]) -> RetrievedContext:
    """Cards, events and projects the request is plausibly about -- nothing else.

    A card is relevant when one of its events is in the world state, one of
    its related projects is active, or the request shares a word with the
    card's own vocabulary.  A card that connects relevant cards -- it consumes
    what one produces or produces what one needs -- is relevant too, so a
    chain can be planned; the closure is deterministic and bounded by the
    registry.  Events are relevant when a relevant card declares them;
    projects when a relevant card relates to them or the request names them.
    Unrelated memory, projects and files never reach the model.
    """

    request_words = words_of(text)
    active_projects = {p.token: p for p in projects}
    chosen: list[Card] = []
    reasons: dict[str, str] = {}
    event_tokens = {token for token, source in state.sources.items() if source != "project"}
    for card in cards:
        why = ""
        if any(event in state.facts for event in card.contract.events):
            why = "event in the world state"
        elif any(f"project.{normalise_token(p)}" in active_projects for p in card.contract.related_projects):
            why = "related project is active"
        elif request_words & card.words:
            why = "the request names " + ", ".join(sorted(request_words & card.words)[:3])
        if why:
            chosen.append(card)
            reasons[card.capability_id] = why
    # Chain closure: connectors between relevant cards.
    rest = [c for c in cards if c.capability_id not in reasons]
    grew = True
    while grew and rest:
        grew = False
        produced = set().union(*(c.contract.provides for c in chosen)) if chosen else set()
        needed = set().union(*(c.contract.requires for c in chosen)) if chosen else set()
        for card in list(rest):
            if card.builtin:
                continue
            consumes = card.contract.requires & produced
            feeds = card.contract.provides & needed
            if consumes or feeds:
                link = sorted(consumes or feeds)[0]
                chosen.append(card)
                reasons[card.capability_id] = f"connects the chain via {link}"
                rest.remove(card)
                grew = True
    relevant_events: list[tuple[str, str]] = []
    if chosen:
        declared = set()
        for card in chosen:
            declared |= set(card.contract.events)
        for token in sorted(event_tokens):
            base = token.split(".", 1)[0]
            if token in declared or base in declared:
                relevant_events.append((token, state.sources.get(token, "event")))
    relevant_projects: list[ProjectSummary] = []
    for token, project in active_projects.items():
        related = any(f"project.{normalise_token(p)}" == token for card in chosen for p in card.contract.related_projects)
        named = bool(request_words & words_of(project.title))
        if related or named:
            relevant_projects.append(project)
    return RetrievedContext(cards=chosen, events=relevant_events, projects=relevant_projects, reasons=reasons, state=state)


# ---------------------------------------------------------------------------
# GoalSpec and PlanSpec
# ---------------------------------------------------------------------------

@dataclass
class GoalSpec:
    primary_goal: str = ""
    secondary_goals: list[str] = field(default_factory=list)
    referenced_entities: list[str] = field(default_factory=list)
    relevant_project: str = ""
    relevant_recent_events: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    privacy_class: str = "public"
    ambiguity: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    clarification_question: str = ""
    #: Tokens the model used that are not in the vocabulary: dropped, listed.
    rejected: list[str] = field(default_factory=list)

    @property
    def goals(self) -> list[str]:
        return [g for g in [self.primary_goal, *self.secondary_goals] if g]

    def to_dict(self) -> dict[str, Any]:
        return {"primary_goal": self.primary_goal, "secondary_goals": list(self.secondary_goals),
                "referenced_entities": list(self.referenced_entities), "relevant_project": self.relevant_project,
                "relevant_recent_events": list(self.relevant_recent_events), "constraints": list(self.constraints),
                "privacy_class": self.privacy_class, "ambiguity": round(self.ambiguity, 3), "confidence": round(self.confidence, 3),
                "reason": self.reason, "clarification_question": self.clarification_question, "rejected": list(self.rejected)}


@dataclass
class PlanStepSpec:
    capability_id: str
    intended_effect: str
    why: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    role: str = "required"

    def to_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, "intended_effect": self.intended_effect, "why": self.why,
                "arguments": dict(self.arguments), "role": self.role}


@dataclass
class PlanSpec:
    steps: list[PlanStepSpec] = field(default_factory=list)
    expected_final_goal: list[str] = field(default_factory=list)
    required_permissions: list[str] = field(default_factory=list)
    reason: str = ""
    #: The provider believes no plan exists; what it says is missing.
    missing: str = ""
    source: str = "provider"  # provider | deterministic | reproposal

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps], "expected_final_goal": list(self.expected_final_goal),
                "required_permissions": list(self.required_permissions), "reason": self.reason, "missing": self.missing,
                "source": self.source}


@dataclass
class ValidationProblem:
    step: int
    capability_id: str
    kind: str  # unknown_capability | unmet_requirement | invalid_effect | goal_not_reached | forbidden
    detail: str
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "capability_id": self.capability_id, "kind": self.kind, "detail": self.detail,
                "missing": list(self.missing)}


@dataclass
class PlanValidation:
    ok: bool
    problems: list[ValidationProblem]
    end_state: WorldState
    plan: Plan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": [p.to_dict() for p in self.problems], "end_facts": sorted(self.end_state.facts)}


def validate_plan(spec: PlanSpec, state: WorldState, cards: dict[str, Card], goal_facts: Iterable[str]) -> PlanValidation:
    """Simulate the PlanSpec over the contracts: every step applicable, every effect real, the goal reached."""

    problems: list[ValidationProblem] = []
    facts = set(state.facts)
    steps: list[Any] = []
    from capabilities.planner import PlanCost, PlanStep, step_cost

    for index, step in enumerate(spec.steps):
        card = cards.get(step.capability_id)
        if card is None:
            problems.append(ValidationProblem(index, step.capability_id, "unknown_capability",
                                              f"{step.capability_id} is not an available capability"))
            continue
        contract = card.contract
        unmet = sorted(contract.requires - facts)
        if unmet and step.role == "required":
            problems.append(ValidationProblem(index, step.capability_id, "unmet_requirement",
                                              f"{step.capability_id} needs {', '.join(unmet)} which no earlier step or the world provides",
                                              missing=unmet))
        effect = normalise_token(step.intended_effect)
        if effect and effect not in contract.provides:
            problems.append(ValidationProblem(index, step.capability_id, "invalid_effect",
                                              f"{step.capability_id} does not produce {effect}; it produces {', '.join(sorted(contract.provides)) or 'nothing declared'}"))
        facts |= contract.provides
        node = card.to_node()
        steps.append(PlanStep(card.capability_id, tuple(sorted(contract.requires)), tuple(sorted(contract.provides)), step_cost(node),
                              node.health, node.assumed_reliability, contract.risk_class))
    goal = {normalise_token(g) for g in goal_facts if normalise_token(g)}
    used = [cards[s.capability_id] for s in spec.steps if s.capability_id in cards]
    unreached = sorted(g for g in goal if g not in facts
                       and not any((g[5:] if g.startswith("goal.") else g) in c.contract.goals for c in used))
    if unreached:
        problems.append(ValidationProblem(len(spec.steps), "", "goal_not_reached",
                                          f"the plan does not reach {', '.join(unreached)}", missing=unreached))
    end = WorldState(facts=frozenset(facts))
    plan = None
    if not problems and steps:
        plan = Plan(steps=tuple(steps), goal=frozenset(goal), start=state, end=end, cost=sum((s.cost for s in steps), PlanCost()))
    return PlanValidation(ok=not problems, problems=problems, end_state=end, plan=plan)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class FlowMetrics:
    capability_summary_tokens: int = 0
    detailed_contract_tokens: int = 0
    world_context_tokens: int = 0
    total_model_context_tokens: int = 0
    provider_calls: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"capability_summary_tokens": self.capability_summary_tokens, "detailed_contract_tokens": self.detailed_contract_tokens,
                "world_context_tokens": self.world_context_tokens, "total_model_context_tokens": self.total_model_context_tokens,
                "provider_calls": self.provider_calls, "notes": list(self.notes)}


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------

GOAL_PROMPT = """Du bist die Verständnisschicht von ZEUS. Du führst nichts aus; du sagst, was der Owner meint.

Weltzustand (nur das Relevante):
{world}

Verfügbare Fähigkeiten (Kurzkarten):
{cards}

Ziel-Vokabular (NUR daraus wählen; passt nichts: "none"):
{vocabulary}

Regeln:
- primary_goal und secondary_goals: exakte Tokens aus dem Vokabular.
- relevant_project: exakt einer der Projekttitel aus dem Weltzustand oder "none".
- relevant_recent_events: exakte Ereignis-Tokens aus dem Weltzustand.
- referenced_entities: Dinge, die der Owner nennt (Dateien, Namen, Zeiten) -- wörtlich.
- Ein Fluch oder Kommentar ist ein Ziel, wenn ein Ereignis oder Projekt es trägt; sonst "none".
- ambiguity 0..1 (wie mehrdeutig), confidence 0..1. Bei echter Mehrdeutigkeit: clarification_question.
- privacy_class: public | private | secret (private, wenn persönliche Daten/Dateien/Gesundheit berührt werden).

Anfrage: {request}
JSON:"""

PLAN_PROMPT = """Du bist die Planungsschicht von ZEUS. Schlage einen Ablauf aus existierenden Fähigkeiten vor. Du führst nichts aus.

Ziel: {goal}
Zielfakten, die am Ende gelten müssen: {goal_facts}
Weltzustand: {world}
{constraints}
Verfügbare Fähigkeiten (vollständige Verträge; NUR diese ids):
{details}

Regeln:
- steps in Ausführungsreihenfolge; jede id EXAKT aus der Liste; arguments_json = JSON-Objekt als String mit den Eingaben der Fähigkeit.
- intended_effect: ein Token aus "produces"/"effects" der Fähigkeit.
- role: required oder optional (Nachschlagen ist optional).
- Fehlt eine Fähigkeit, um das Ziel zu erreichen: steps leer lassen und in "missing" sagen, welcher Effekt fehlt.
{feedback}
Anfrage des Owners: {request}
JSON:"""


@dataclass
class FlowResult:
    status: str  # PLAN | MISSING_CAPABILITY | CLARIFY | INTELLIGENCE_UNAVAILABLE | FREE_INTELLIGENCE_UNAVAILABLE | NONE
    goal: GoalSpec | None = None
    plan_spec: PlanSpec | None = None
    validation: PlanValidation | None = None
    plan: Plan | None = None
    report: PlanReport | None = None
    missing: MissingCapabilityEvidence | None = None
    context: RetrievedContext | None = None
    metrics: FlowMetrics = field(default_factory=FlowMetrics)
    reason: str = ""
    question: str = ""
    provider: str = ""
    model: str = ""
    route_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "goal": self.goal.to_dict() if self.goal else None,
                "plan_spec": self.plan_spec.to_dict() if self.plan_spec else None,
                "validation": self.validation.to_dict() if self.validation else None,
                "plan": self.plan.to_dict() if self.plan else None,
                "missing": self.missing.to_dict() if self.missing else None,
                "context": self.context.to_dict() if self.context else None, "metrics": self.metrics.to_dict(),
                "reason": self.reason, "question": self.question, "provider": self.provider, "model": self.model,
                "route_kind": self.route_kind}


#: The whole semantic decision path -- GoalSpec, PlanSpec, one correction
#: round -- must settle within this many seconds.  Each call is bounded by
#: the gateway's SEMANTIC_CALL_TIMEOUT_SECONDS as well; a free pool of two
#: models that both hang spends the deadline on the GoalSpec alone and the
#: flow then reports the typed unavailable status instead of trying more.
SEMANTIC_DEADLINE_SECONDS = 60.0


class IntelligenceFlow:
    def __init__(self, gateway: Any, cards: Iterable[Card], projects: Iterable[ProjectSummary] = (), *,
                 mode: Any = None, task_id: str = "", max_depth: int = 6,
                 reliability: Callable[[str], float | None] | None = None,
                 deadline_seconds: float = SEMANTIC_DEADLINE_SECONDS) -> None:
        self.gateway = gateway
        self.cards = list(cards)
        self.projects = list(projects)
        self.mode = mode
        self.task_id = task_id
        self.max_depth = max_depth
        #: Learned per-capability reliability (from real runs); None = the health assumption.
        self.reliability = reliability
        self.metrics = FlowMetrics()
        self.deadline_seconds = float(deadline_seconds)
        self._started = time.perf_counter()

    def remaining_seconds(self) -> float:
        return self.deadline_seconds - (time.perf_counter() - self._started)

    def unavailable_status(self) -> str:
        """FREE mode has its own typed answer: the free provider is not reachable and nothing paid or local replaces it."""

        from gateway.modes import ChatMode

        try:
            return "FREE_INTELLIGENCE_UNAVAILABLE" if ChatMode.parse(self.mode) is ChatMode.FREE else "INTELLIGENCE_UNAVAILABLE"
        except Exception:  # noqa: BLE001
            return "INTELLIGENCE_UNAVAILABLE"

    # -- vocabulary ------------------------------------------------------------

    @staticmethod
    def vocabulary(cards: Iterable[Card]) -> list[str]:
        out: set[str] = set()
        for card in cards:
            out |= set(card.contract.goals) | card.contract.provides
        return sorted(out)

    # -- provider calls ----------------------------------------------------------

    def _ask(self, prompt: str, schema: dict[str, Any], *, facts: Any, max_tokens: int, kind: str) -> tuple[str | None, Any, dict[str, Any]]:
        """One structured call.  Returns (text, reply, unavailable-reason dict)."""

        from gateway.gateway import SEMANTIC_CALL_TIMEOUT_SECONDS, GatewayError, GatewayRefused, GatewayRequest

        remaining = self.remaining_seconds()
        if remaining < 3.0:
            # Bounded, and said so: no further provider call is started once
            # the decision path has used its time.
            self.metrics.notes.append(f"semantic deadline of {self.deadline_seconds:.0f}s exhausted before the {kind} call")
            return None, None, {"reason": f"timeout: the semantic decision path used its {self.deadline_seconds:.0f}s before the {kind}",
                                "question": ""}
        self.metrics.total_model_context_tokens += estimate_tokens(prompt)
        self.metrics.provider_calls += 1
        request = GatewayRequest(prompt=prompt, mode=self.mode if self.mode is not None else "AUTO", facts=facts, schema=schema,
                                 max_output_tokens=max_tokens, temperature=0.0, task_id=self.task_id, overrides=False,
                                 allow_offline_fallback=False, timeout_seconds=min(SEMANTIC_CALL_TIMEOUT_SECONDS, remaining))
        try:
            reply = self.gateway.complete(request)
        except GatewayRefused as exc:
            return None, None, {"reason": f"no reasoning route: {exc.decision.reason[:200]}", "question": exc.decision.suggestion}
        except GatewayError as exc:
            classes = sorted({str(a.get("failure_class")) for a in getattr(exc, "attempts", [])} - {"ok", "None"})
            if getattr(exc, "typed_status", "") == "FREE_INTELLIGENCE_UNAVAILABLE":
                return None, None, {"reason": "free intelligence is temporarily unavailable" + (f" ({', '.join(classes)})" if classes else ""),
                                    "question": ""}
            return None, None, {"reason": f"reasoning provider {exc.status.value}: {exc}"[:300], "question": ""}
        if reply.decision.offline_fallback:
            return None, reply, {"reason": "only the offline fallback model is reachable; it does not produce a " + kind,
                                 "question": "Ein Cloud-Denkmodell ist nicht erreichbar (Schlüssel, Modus oder Budget)."}
        return reply.text, reply, {}

    # -- GoalSpec -----------------------------------------------------------------

    def understand(self, text: str, context: RetrievedContext) -> tuple[GoalSpec | None, dict[str, Any], Any]:
        from gateway.task import TaskFacts

        vocabulary = self.vocabulary(context.cards)
        world = context.world_text()
        cards_text = context.summaries_text()
        self.metrics.world_context_tokens += estimate_tokens(world)
        self.metrics.capability_summary_tokens += estimate_tokens(cards_text)
        prompt = GOAL_PROMPT.format(world=world, cards=cards_text, vocabulary=", ".join(vocabulary) or "(keins)", request=text.strip())
        project_titles = [p.title for p in context.projects]
        event_tokens = [t for t, _ in context.events]
        schema = {
            "type": "object",
            "properties": {
                "primary_goal": {"type": "string", "enum": vocabulary + ["none"]},
                "secondary_goals": {"type": "array", "items": {"type": "string", "enum": vocabulary or ["none"]}},
                "referenced_entities": {"type": "array", "items": {"type": "string"}},
                "relevant_project": {"type": "string", "enum": project_titles + ["none"]},
                "relevant_recent_events": {"type": "array", "items": {"type": "string", "enum": event_tokens or ["none"]}},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "privacy_class": {"type": "string", "enum": list(PRIVACY_CLASSES)},
                "ambiguity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
                "clarification_question": {"type": "string"},
            },
            "required": ["primary_goal", "confidence", "reason"],
        }
        facts = TaskFacts(text=text, refers_to_context=bool(context.events or context.projects), subsystems=len({c.contract.domain for c in context.cards if c.contract.domain}))
        raw, reply, unavailable = self._ask(prompt, schema, facts=facts, max_tokens=400, kind="GoalSpec")
        if raw is None:
            return None, unavailable, reply
        from brain.json_utils import lenient_json_loads

        try:
            data = lenient_json_loads(str(raw))
        except Exception:  # noqa: BLE001
            data = None
        if not isinstance(data, dict):
            return GoalSpec(reason="unparseable GoalSpec"), {}, reply
        vocab = set(vocabulary)
        rejected: list[str] = []

        def goal_token(value: Any) -> str:
            token = normalise_token(value)
            if token in vocab:
                return token
            if token and token != "none":
                rejected.append(token)
            return ""

        primary = goal_token(data.get("primary_goal"))
        secondary = [t for t in (goal_token(v) for v in (data.get("secondary_goals") or [])) if t and t != primary]
        project = str(data.get("relevant_project") or "")
        if project not in project_titles:
            project = ""
        events = [normalise_token(e) for e in (data.get("relevant_recent_events") or []) if normalise_token(e) in set(event_tokens)]
        privacy = str(data.get("privacy_class") or "public").lower()
        if privacy not in PRIVACY_CLASSES:
            privacy = "public"

        def number(key: str) -> float:
            try:
                return max(0.0, min(1.0, float(data.get(key) or 0.0)))
            except (TypeError, ValueError):
                return 0.0

        spec = GoalSpec(primary_goal=primary, secondary_goals=secondary,
                        referenced_entities=[str(e)[:120] for e in (data.get("referenced_entities") or []) if str(e).strip()][:8],
                        relevant_project=project, relevant_recent_events=events,
                        constraints=[str(c)[:160] for c in (data.get("constraints") or []) if str(c).strip()][:6],
                        privacy_class=privacy, ambiguity=number("ambiguity"), confidence=number("confidence"),
                        reason=str(data.get("reason") or "")[:300], clarification_question=str(data.get("clarification_question") or "")[:300],
                        rejected=rejected)
        return spec, {}, reply

    # -- grounding of a GoalSpec ---------------------------------------------------------

    def grounded(self, goal: str, text: str, context: RetrievedContext, spec: GoalSpec) -> tuple[bool, str]:
        supporters = [c for c in context.cards if goal in c.contract.goals or goal in c.contract.provides]
        if not supporters:
            return False, "no retrieved capability serves this goal"
        request_words = words_of(text)
        for card in supporters:
            if any(e in context.state.facts for e in card.contract.events):
                return True, f"an event in the world state supports {card.capability_id}"
            if spec.relevant_project and any(normalise_token(p) == normalise_token(spec.relevant_project) for p in card.contract.related_projects):
                return True, f"project {spec.relevant_project} relates to {card.capability_id}"
            if request_words & card.words:
                return True, f"the request names {', '.join(sorted(request_words & card.words)[:3])}"
        return False, "neither an event, a related project nor the request's words support this goal"

    # -- PlanSpec -------------------------------------------------------------------------

    def propose(self, text: str, spec: GoalSpec, context: RetrievedContext, goal_facts: list[str], *,
                feedback: list[ValidationProblem] | None = None) -> tuple[PlanSpec | None, dict[str, Any]]:
        from gateway.task import TaskFacts

        details = "\n".join(card.detail() for card in context.cards)
        self.metrics.detailed_contract_tokens += estimate_tokens(details)
        world = context.world_text()
        constraints = ""
        if spec.constraints:
            constraints = "Einschränkungen des Owners: " + "; ".join(spec.constraints)
        feedback_text = ""
        if feedback:
            feedback_text = ("DEIN VORHERIGER PLAN WAR UNGÜLTIG. Genau das war falsch:\n"
                             + "\n".join(f"- Schritt {p.step + 1} ({p.capability_id or 'Ziel'}): {p.detail}" for p in feedback)
                             + "\nSchlage einen gültigen Plan vor oder nenne den fehlenden Effekt in \"missing\".")
        prompt = PLAN_PROMPT.format(goal=", ".join(spec.goals), goal_facts=", ".join(goal_facts), world=world, constraints=constraints,
                                    details=details, feedback=feedback_text, request=text.strip())
        ids = [c.capability_id for c in context.cards]
        schema = {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "object", "properties": {
                    "capability_id": {"type": "string", "enum": ids or ["none"]},
                    "intended_effect": {"type": "string"},
                    "why": {"type": "string"},
                    "arguments_json": {"type": "string"},
                    "role": {"type": "string", "enum": ["required", "optional"]},
                }, "required": ["capability_id", "intended_effect"]}},
                "expected_final_goal": {"type": "array", "items": {"type": "string"}},
                "required_permissions": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "missing": {"type": "string"},
            },
            "required": ["steps", "reason"],
        }
        facts = TaskFacts(text=text, subsystems=len({c.contract.domain for c in context.cards if c.contract.domain}), refers_to_context=True)
        raw, reply, unavailable = self._ask(prompt, schema, facts=facts, max_tokens=900, kind="PlanSpec")
        if raw is None:
            return None, unavailable
        from brain.json_utils import lenient_json_loads

        try:
            data = lenient_json_loads(str(raw))
        except Exception:  # noqa: BLE001
            data = None
        if not isinstance(data, dict):
            return PlanSpec(reason="unparseable PlanSpec"), {}
        steps: list[PlanStepSpec] = []
        for row in data.get("steps") or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("capability_id") or "").strip()
            arguments: dict[str, Any] = {}
            raw_args = row.get("arguments_json") or row.get("arguments")
            if isinstance(raw_args, dict):
                arguments = dict(raw_args)
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    arguments = dict(parsed) if isinstance(parsed, dict) else {}
                except ValueError:
                    arguments = {}
            role = str(row.get("role") or "required").lower()
            steps.append(PlanStepSpec(cid, str(row.get("intended_effect") or ""), str(row.get("why") or "")[:200], arguments,
                                      role if role in {"required", "optional"} else "required"))
        return PlanSpec(steps=steps, expected_final_goal=_tokens(data.get("expected_final_goal")),
                        required_permissions=[str(p) for p in (data.get("required_permissions") or [])][:8],
                        reason=str(data.get("reason") or "")[:300], missing=str(data.get("missing") or "")[:300],
                        source="reproposal" if feedback else "provider"), {}

    # -- the run ------------------------------------------------------------------------------

    def run(self, text: str, state: WorldState) -> FlowResult:
        self._started = time.perf_counter()
        context = retrieve(text, state, self.cards, self.projects)
        result = FlowResult(status="NONE", context=context, metrics=self.metrics)
        if context.empty:
            result.reason = "no capability, event or project relates to the request"
            return result
        spec, unavailable, reply = self.understand(text, context)
        if reply is not None:
            result.provider, result.model = reply.provider, reply.model
        if spec is None:
            result.status, result.reason, result.question = self.unavailable_status(), unavailable.get("reason", ""), unavailable.get("question", "")
            return result
        result.goal = spec
        if not spec.primary_goal:
            if spec.clarification_question and spec.ambiguity >= 0.5:
                result.status, result.question, result.reason = "CLARIFY", spec.clarification_question, spec.reason
            else:
                result.reason = spec.reason or "no goal from the vocabulary applies"
            return result
        ok, why = self.grounded(spec.primary_goal, text, context, spec)
        if not ok:
            result.status, result.reason = "CLARIFY", why
            result.question = spec.clarification_question or f"Meinst du {spec.primary_goal}? Ich sehe dafür weder ein passendes Ereignis noch ein Projekt."
            return result
        if spec.ambiguity >= 0.8 or spec.confidence < 0.4:
            result.status, result.reason = "CLARIFY", f"ambiguity {spec.ambiguity:.2f}, confidence {spec.confidence:.2f}"
            result.question = spec.clarification_question or f"Meinst du {spec.primary_goal}?"
            return result

        cards = {c.capability_id: c for c in context.cards}
        nodes = [c.to_node() for c in context.cards]
        planner = CompositionPlanner(nodes, max_depth=self.max_depth, reliability=self.reliability)
        goal_tokens = spec.goals
        deterministic = plans_for_goals(planner, state, goal_tokens)
        result.report = deterministic
        goal_facts = sorted(deterministic.goal)

        # B. A trivial deterministic path -- exactly one plan of at most two
        # steps -- needs no proposal round.  Everything else is the provider's.
        needs_arguments = any(cards[s.capability_id].inputs for s in (deterministic.best.steps if deterministic.best else ())
                              if s.capability_id in cards)
        if deterministic.plans and len(deterministic.plans) == 1 and len(deterministic.best.steps) <= 2 and not needs_arguments:
            spec_plan = PlanSpec(steps=[PlanStepSpec(s.capability_id, s.provides[0] if s.provides else "", "deterministic path")
                                        for s in deterministic.best.steps], expected_final_goal=goal_facts, source="deterministic",
                                 reason="the only path; no inputs needed")
            validation = validate_plan(spec_plan, state, cards, goal_facts)
            result.plan_spec, result.validation, result.plan, result.status = spec_plan, validation, validation.plan, "PLAN"
            self.metrics.notes.append("deterministic path, no PlanSpec call")
            return result

        # The provider proposes; the planner validates; one round of exact feedback.
        proposal, unavailable = self.propose(text, spec, context, goal_facts)
        if proposal is None:
            result.status, result.reason, result.question = self.unavailable_status(), unavailable.get("reason", ""), unavailable.get("question", "")
            return result
        validation = validate_plan(proposal, state, cards, goal_facts) if proposal.steps else None
        if proposal.steps and validation is not None and not validation.ok:
            second, unavailable = self.propose(text, spec, context, goal_facts, feedback=validation.problems)
            if second is not None and second.steps:
                second_validation = validate_plan(second, state, cards, goal_facts)
                if second_validation.ok:
                    proposal, validation = second, second_validation
                else:
                    proposal, validation = second, second_validation
        result.plan_spec, result.validation = proposal, validation
        if validation is not None and validation.ok and validation.plan is not None:
            result.status, result.plan = "PLAN", validation.plan
            return result
        # C/D. The provider could not propose a valid plan: the planner proves
        # whether a path exists at all.
        if deterministic.plans:
            best = deterministic.best
            inputs = sorted({name for s in best.steps if s.capability_id in cards for name in cards[s.capability_id].inputs})
            if inputs:
                # A path exists but its steps need inputs only the provider
                # could have given: ask the owner, plainly, without the error.
                chain = " -> ".join(best.capability_ids)
                result.status, result.reason = "CLARIFY", "provider proposal invalid; the deterministic path needs inputs"
                result.question = f"Ich kann {chain} ausführen, brauche dafür aber konkrete Angaben ({', '.join(inputs[:4])}). Was genau soll ich damit tun?"
                self.metrics.notes.append("provider proposal invalid; deterministic path needs inputs")
                return result
            spec_plan = PlanSpec(steps=[PlanStepSpec(s.capability_id, s.provides[0] if s.provides else "", "deterministic path")
                                        for s in best.steps], expected_final_goal=goal_facts, source="deterministic",
                                 reason="the provider's proposal was invalid; the planner found a valid path")
            result.plan_spec, result.plan, result.status = spec_plan, best, "PLAN"
            result.validation = validate_plan(spec_plan, state, cards, goal_facts)
            self.metrics.notes.append("provider proposal invalid; deterministic path used")
            return result
        result.status, result.missing = "MISSING_CAPABILITY", deterministic.missing
        result.reason = proposal.missing or (deterministic.missing.describe() if deterministic.missing else "no path")
        return result

    def repropose(self, text: str, spec: GoalSpec, context: RetrievedContext, goal_facts: list[str], failed_capability: str,
                  failure: str, done: Iterable[str]) -> PlanSpec | None:
        """After a required step failed at run time: one new proposal that avoids it."""

        problems = [ValidationProblem(0, failed_capability, "unmet_requirement",
                                      f"{failed_capability} failed at run time: {failure[:160]}; do not use it again. Already done: {', '.join(done) or 'nothing'}")]
        proposal, _ = self.propose(text, spec, context, goal_facts, feedback=problems)
        if proposal is None or not proposal.steps or any(s.capability_id == failed_capability for s in proposal.steps):
            return None
        cards = {c.capability_id: c for c in context.cards}
        validation = validate_plan(proposal, context.state.with_facts(done, source="done"), cards, goal_facts)
        return proposal if validation.ok else None
