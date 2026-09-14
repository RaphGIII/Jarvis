"""From the owner's words and the world state to a semantic goal -- or an honest "cannot".

"fuck, schon wieder verloren", said forty seconds after a chess game ended
in a loss, inside the project *Schach Training*, means ``understand_recent_loss``.
The same sentence with no such context means nothing a capability can act on.
Both readings come from the same three ingredients:

1. the closed vocabulary of goals and effects the registry's contracts declare
   -- a model may choose from it, never add to it;
2. the world state: recent events, active projects, facts already produced;
3. grounding: a chosen goal is accepted only if the request's words, or the
   state, actually support the capabilities that serve it.  A goal the model
   likes but nothing supports becomes a clarification, not an action.

The model is the configured semantic provider behind the gateway.  When the
gateway can only offer the offline fallback (the legacy local model), or no
route at all, the result is a typed UNAVAILABLE -- the small model does not
get to decide a goal at low confidence and have it acted on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from capabilities.contracts import SemanticContract, normalise_token
from capabilities.planner import CapabilityNode, WorldState

STATUSES = ("DERIVED", "NONE", "CLARIFY", "UNAVAILABLE")

_WORD = re.compile(r"[a-zäöüß0-9]{3,}", re.I)
_STOP = frozenset({"the", "and", "der", "die", "das", "und", "ich", "mir", "mich", "mein", "meine", "meinen", "ist", "bin",
                   "hab", "habe", "schon", "wieder", "mal", "bitte", "jetzt", "noch", "nicht", "kein", "keine", "was", "wie",
                   "wer", "you", "for", "with", "von", "des", "dem", "den", "ein", "eine", "einen", "einem", "auf", "aus",
                   "sie", "wir", "uns", "nur", "auch", "dann", "kann", "soll", "will", "please", "again", "just"})


def words_of(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(str(text or "")) if w.lower() not in _STOP}


@dataclass(frozen=True)
class CapabilityCard:
    """What the deriver may know about a capability: id, contract, prose, examples."""

    capability_id: str
    contract: SemanticContract
    description: str = ""
    examples: tuple[str, ...] = ()
    name: str = ""

    def vocabulary(self) -> set[str]:
        """Domain words: from id, name, goals, effects, description and examples."""

        out: set[str] = set()
        for token in (self.capability_id, self.name, self.contract.domain, *self.contract.goals, *self.contract.provides,
                      *self.contract.events, *self.contract.related_projects):
            out |= {w for w in re.split(r"[._\s\-]+", str(token).lower()) if len(w) >= 3}
        out |= words_of(self.description)
        for example in self.examples:
            out |= words_of(example)
        return out


@dataclass
class GoalDerivation:
    status: str
    goals: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    question: str = ""
    grounding: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    offline: bool = False
    rejected: list[str] = field(default_factory=list)

    @property
    def derived(self) -> bool:
        return self.status == "DERIVED" and bool(self.goals)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "goals": list(self.goals), "confidence": round(self.confidence, 3), "reason": self.reason,
                "question": self.question, "grounding": dict(self.grounding), "provider": self.provider, "model": self.model,
                "offline": self.offline, "rejected": list(self.rejected)}


GOAL_SCHEMA_TEMPLATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goals": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "needs_clarification": {"type": "boolean"},
        "question": {"type": "string"},
    },
    "required": ["goals", "confidence", "reason"],
}

PROMPT = """Du bist die semantische Zielableitung von ZEUS. Du entscheidest NICHT, was ausgeführt wird; du benennst das Ziel.

Aus der Anfrage des Owners und dem Weltzustand wähle die semantischen Ziele, die der Owner meint.
Wähle NUR aus der Liste unten (exakte Tokens). Erfinde nichts. Passt nichts, gib eine leere Liste zurück.
Ein Fluch oder ein Kommentar ist ein Ziel, wenn der Kontext es trägt (ein gerade beendetes Ereignis, ein aktives Projekt);
ohne solchen Kontext ist derselbe Satz meist kein Ziel -- dann leere Liste oder needs_clarification.

Weltzustand:
{state}

Verfügbare Ziele (token: wofür):
{vocabulary}

Anfrage: {request}
Antworte mit einem JSON-Objekt: {{"goals": [...], "confidence": 0..1, "reason": "...", "needs_clarification": false, "question": ""}}
JSON:"""


class SemanticGoalDeriver:
    def __init__(self, gateway: Any, cards: Iterable[CapabilityCard], *, min_confidence: float = 0.5) -> None:
        self.gateway = gateway
        self.cards = list(cards)
        self.min_confidence = min_confidence

    # -- vocabulary ----------------------------------------------------------

    def vocabulary(self) -> dict[str, list[str]]:
        """goal/effect token -> capabilities that serve or produce it."""

        vocab: dict[str, list[str]] = {}
        for card in self.cards:
            for token in (*card.contract.goals, *card.contract.produces, *card.contract.effects):
                vocab.setdefault(token, [])
                if card.capability_id not in vocab[token]:
                    vocab[token].append(card.capability_id)
        return dict(sorted(vocab.items()))

    def _vocabulary_text(self) -> str:
        lines = []
        by_id = {card.capability_id: card for card in self.cards}
        for token, ids in self.vocabulary().items():
            hints = []
            for cid in ids[:3]:
                card = by_id.get(cid)
                if card is not None:
                    hints.append(f"{cid}: {(card.description or card.name)[:80]}")
            lines.append(f"- {token} ({'; '.join(hints)})")
        return "\n".join(lines) if lines else "- (keine)"

    # -- grounding -----------------------------------------------------------

    def grounded(self, goal: str, text: str, state: WorldState) -> tuple[bool, str]:
        """Does the request or the state support the capabilities behind this goal?"""

        request_words = words_of(text)
        supporters = [card for card in self.cards if goal in card.contract.goals or goal in card.contract.provides]
        if not supporters:
            return False, "no capability serves this goal"
        for card in supporters:
            for event in card.contract.events:
                if event in state.facts:
                    return True, f"event {event} in the world state supports {card.capability_id}"
            for project in card.contract.related_projects:
                if f"project.{normalise_token(project)}" in state.facts:
                    return True, f"project {project} is active and relates to {card.capability_id}"
            overlap = request_words & card.vocabulary()
            if overlap:
                return True, f"the request names {', '.join(sorted(overlap)[:4])} ({card.capability_id})"
        goal_words = set(re.split(r"[._]+", goal)) - _STOP
        if request_words & goal_words:
            return True, f"the request names {', '.join(sorted(request_words & goal_words))}"
        return False, "neither the request's words nor the world state support this goal"

    # -- derivation ------------------------------------------------------------

    def derive(self, text: str, state: WorldState, *, mode: Any = None, task_id: str = "") -> GoalDerivation:
        from gateway.gateway import GatewayError, GatewayRefused, GatewayRequest
        from gateway.task import TaskFacts

        vocab = self.vocabulary()
        if not vocab:
            return GoalDerivation(status="NONE", reason="no capability declares a goal or effect")
        state_lines = [f"- {token} ({source})" for token, source in sorted(state.sources.items())] or ["- (nichts Besonderes)"]
        prompt = PROMPT.format(state="\n".join(state_lines), vocabulary=self._vocabulary_text(), request=str(text).strip())
        schema = json.loads(json.dumps(GOAL_SCHEMA_TEMPLATE))
        schema["properties"]["goals"]["items"]["enum"] = list(vocab)
        facts = TaskFacts(text=text, refers_to_context=bool(state.facts), is_question=str(text).strip().endswith("?"))
        request = GatewayRequest(prompt=prompt, mode=mode if mode is not None else "AUTO", facts=facts, schema=schema,
                                 max_output_tokens=300, temperature=0.0, task_id=task_id, overrides=False)
        try:
            reply = self.gateway.complete(request)
        except GatewayRefused as exc:
            return GoalDerivation(status="UNAVAILABLE", reason=f"no semantic provider route: {exc.decision.reason[:200]}",
                                  question=exc.decision.suggestion)
        except GatewayError as exc:
            return GoalDerivation(status="UNAVAILABLE", reason=f"semantic provider {exc.status.value}: {exc}"[:300])
        if reply.decision.offline_fallback:
            # The legacy local model answered.  It may not decide a semantic
            # goal that leads to action; the owner is told what is missing.
            return GoalDerivation(status="UNAVAILABLE", provider=reply.provider, model=reply.model, offline=True,
                                  reason="only the offline fallback model is reachable; it does not decide semantic goals",
                                  question="Ein Cloud-Denkmodell ist nicht erreichbar (Schlüssel, Modus oder Budget). "
                                           "Sag mir direkt, was ich tun soll, oder konfiguriere einen Provider.")
        from brain.json_utils import lenient_json_loads

        try:
            data = lenient_json_loads(str(reply.text))
        except Exception:  # noqa: BLE001
            data = None
        if not isinstance(data, dict):
            return GoalDerivation(status="NONE", provider=reply.provider, model=reply.model, reason="unparseable answer")
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        raw_goals = data.get("goals") or []
        chosen: list[str] = []
        rejected: list[str] = []
        for item in raw_goals if isinstance(raw_goals, list) else []:
            token = normalise_token(item)
            if token in vocab and token not in chosen:
                chosen.append(token)
            elif token:
                rejected.append(token)  # invented: not in the registry's vocabulary
        reason = str(data.get("reason") or "")[:300]
        question = str(data.get("question") or "")[:300]
        base = dict(provider=reply.provider, model=reply.model, confidence=confidence, reason=reason, rejected=rejected)
        if bool(data.get("needs_clarification")) and not chosen:
            return GoalDerivation(status="CLARIFY", question=question or "Was genau meinst du?", **base)
        if not chosen:
            return GoalDerivation(status="NONE", **base)
        grounding: dict[str, Any] = {}
        accepted: list[str] = []
        for goal in chosen:
            ok, why = self.grounded(goal, text, state)
            grounding[goal] = {"grounded": ok, "why": why}
            if ok:
                accepted.append(goal)
        if not accepted:
            names = ", ".join(chosen)
            return GoalDerivation(status="CLARIFY", grounding=grounding,
                                  question=question or f"Meinst du {names}? Ich sehe dafür weder ein passendes Ereignis noch ein Projekt.",
                                  **base)
        if confidence < self.min_confidence:
            return GoalDerivation(status="CLARIFY", goals=accepted, grounding=grounding,
                                  question=question or f"Meinst du {', '.join(accepted)}?", **base)
        return GoalDerivation(status="DERIVED", goals=accepted, grounding=grounding, **base)


def cards_from_nodes(nodes: Iterable[CapabilityNode], descriptions: dict[str, str] | None = None,
                     examples: dict[str, Iterable[str]] | None = None, names: dict[str, str] | None = None) -> list[CapabilityCard]:
    descriptions = descriptions or {}
    examples = examples or {}
    names = names or {}
    return [CapabilityCard(node.capability_id, node.contract, descriptions.get(node.capability_id, ""),
                           tuple(examples.get(node.capability_id, ())), names.get(node.capability_id, ""))
            for node in nodes if node.usable]
