"""The Adaptive Owner Model: what ZEUS has learned about how this owner wants it to behave.

Feedback must not be decorative.  A 👎 with "zu kurz" on a technical
explanation makes future *technical explanations* longer — not every
answer.  A 👍 strengthens the current style in its context.  Everything
learned here is a bounded, decaying, reversible record the owner can read,
edit and delete; nothing here can touch the protected personality core, the
security policy or any platform invariant — this layer only ever produces
*style guidance* and *lesson records*.

Sources rank: an owner-authored rule outranks an explicit correction,
which outranks an explicit rating, which outranks repeated behaviour;
inference alone stays low-confidence.  One accidental thumb never becomes a
permanent preference: weights move in steps, are clamped, and decay unless
re-confirmed.

Domains: SPEECH, ANSWER_LENGTH, TECHNICAL_DEPTH, STYLE, ACTION_BEHAVIOR,
PRONUNCIATION, ENTITY_NAMES, PROJECT_WORKFLOW, UI_PREFERENCES, VERIFICATION,
WAKE.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

DOMAINS = ("SPEECH", "ANSWER_LENGTH", "TECHNICAL_DEPTH", "STYLE", "ACTION_BEHAVIOR", "PRONUNCIATION",
           "ENTITY_NAMES", "PROJECT_WORKFLOW", "UI_PREFERENCES", "VERIFICATION", "WAKE")

#: Source precedence (higher outranks lower when rules conflict).
SOURCE_PRIORITY = {"OWNER_RULE": 100, "EXPLICIT_CORRECTION": 80, "EXPLICIT_RATING": 60, "REPEATED_BEHAVIOR": 40, "INFERRED": 20}

#: The 👎 categories the interface offers, and the domain each one teaches.
FEEDBACK_CATEGORIES = {
    "TOO_SHORT": ("ANSWER_LENGTH", +1), "TOO_LONG": ("ANSWER_LENGTH", -1),
    "TOO_TECHNICAL": ("TECHNICAL_DEPTH", -1), "TOO_SIMPLE": ("TECHNICAL_DEPTH", +1),
    "BAD_STYLE": ("STYLE", 0), "WRONG_FACT": ("STYLE", 0), "MISUNDERSTOOD": ("SPEECH", 0),
    "WRONG_ACTION": ("ACTION_BEHAVIOR", 0), "INCOMPLETE": ("ANSWER_LENGTH", +1),
    "BAD_PRONUNCIATION": ("PRONUNCIATION", 0), "OTHER": ("STYLE", 0),
}

#: One rating moves a weight this far; three consistent ratings reach ~1.0.
RATING_STEP = 0.34
WEIGHT_LIMIT = 2.0
#: Confidence halves after this many days without confirmation.
DECAY_HALF_LIFE_DAYS = 60.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_days(iso: str) -> float:
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


# --------------------------------------------------------------------------
# context classification: which situation a piece of feedback is about
# --------------------------------------------------------------------------

_TECH = re.compile(r"\b(code|funktion|api|server|modell|algorithm\w*|technisch\w*|implementier\w*|debug\w*|fehler\w*|python|javascript|regex|datenbank|protokoll|biochem\w*|enzym\w*|molek\w*|physik\w*|anatom\w*|mechanis\w*|erklär\w*|explain|why|warum|wie funktioniert)\b", re.I)
_CONFIRM_BACKENDS = ("projects.store", "projects", "composer", "corrections", "ui", "policy", "planner", "tools", "music")


def classify_context(request: str = "", answer: str = "", backend: str = "", intent: str = "") -> dict[str, str]:
    """A small, deterministic context: the scope feedback attaches to."""

    if backend in _CONFIRM_BACKENDS or intent in {"action", "project", "music", "project_operation"}:
        kind = "action_confirmation"
    elif backend == "personality":
        kind = "small_talk"
    elif _TECH.search(request or "") or _TECH.search((answer or "")[:400]):
        kind = "technical_explanation"
    else:
        kind = "conversation"
    return {"kind": kind}


# --------------------------------------------------------------------------
# the learned record
# --------------------------------------------------------------------------

@dataclass
class AdaptiveRule:
    domain: str
    #: Which situations it applies to, e.g. {"kind": "technical_explanation"}.
    scope: dict[str, str] = field(default_factory=dict)
    #: -WEIGHT_LIMIT..+WEIGHT_LIMIT; the meaning depends on the domain
    #: (ANSWER_LENGTH: + = longer; TECHNICAL_DEPTH: + = deeper).
    weight: float = 0.0
    #: The owner-readable statement of the rule.
    text: str = ""
    source: str = "EXPLICIT_RATING"
    confidence: float = 0.3
    evidence: list = field(default_factory=list)
    rule_id: str = field(default_factory=lambda: f"ar_{uuid.uuid4().hex[:10]}")
    created_at: str = field(default_factory=_now)
    last_confirmed: str = field(default_factory=_now)
    enabled: bool = True
    reversible: bool = True
    version: int = 1

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY.get(self.source, 10)

    def effective_confidence(self) -> float:
        """Confidence after decay: unconfirmed learning fades, never hardens."""

        return self.confidence * math.pow(0.5, _age_days(self.last_confirmed) / DECAY_HALF_LIFE_DAYS)

    def matches(self, context: dict[str, str]) -> bool:
        return all(context.get(k) == v for k, v in self.scope.items())

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["priority"] = self.priority
        out["effective_confidence"] = round(self.effective_confidence(), 3)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveRule":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class AdaptiveOwnerModel:
    """Bounded, inspectable, deletable owner adaptation."""

    LIMIT = 200

    def __init__(self, path: str | Path, *, on_insight: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.rules: list[AdaptiveRule] = []
        self.feedback_log: list[dict[str, Any]] = []
        #: Verifier/resolver lessons: kind -> list of owner verdicts.
        self.lessons: dict[str, list[dict[str, Any]]] = {}
        self.on_insight = on_insight or (lambda _t, _p: None)
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.rules = [AdaptiveRule.from_dict(r) for r in data.get("rules", []) if isinstance(r, dict) and r.get("domain")]
        self.feedback_log = list(data.get("feedback", []))[-500:]
        self.lessons = {str(k): list(v)[-50:] for k, v in (data.get("lessons") or {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rules": [r.to_dict() for r in self.rules][-self.LIMIT:],
                   "feedback": self.feedback_log[-500:], "lessons": self.lessons, "saved_at": _now()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- feedback --------------------------------------------------------

    def record_response_feedback(self, *, rating: str, category: str = "", text: str = "", context: dict[str, str] | None = None,
                                 request: str = "", answer: str = "", request_id: str = "") -> dict[str, Any]:
        """One 👍/👎 (optionally with a category and words) becomes a scoped nudge."""

        context = dict(context or {})
        entry = {"kind": "response", "rating": rating, "category": category, "text": text[:300], "context": context,
                 "request": request[:200], "answer": (answer or "")[:200], "request_id": request_id, "at": _now()}
        with self._lock:
            self.feedback_log.append(entry)
            changed: list[AdaptiveRule] = []
            if rating == "down" and category in FEEDBACK_CATEGORIES:
                domain, direction = FEEDBACK_CATEGORIES[category]
                if direction != 0:
                    rule = self._nudge(domain, context, direction * RATING_STEP, source="EXPLICIT_RATING",
                                       evidence={"category": category, "text": text[:120], "request_id": request_id})
                    changed.append(rule)
            elif rating == "up":
                # confirmation: every enabled rule that shaped this context gains confidence
                for rule in self.rules:
                    if rule.enabled and rule.matches(context) and rule.domain in {"ANSWER_LENGTH", "TECHNICAL_DEPTH", "STYLE"}:
                        rule.confidence = min(1.0, rule.confidence + 0.08)
                        rule.last_confirmed = _now()
                        changed.append(rule)
            self.save()
        return {"ok": True, "learned": [r.to_dict() for r in changed]}

    def _nudge(self, domain: str, context: dict[str, str], step: float, *, source: str, evidence: dict[str, Any]) -> AdaptiveRule:
        scope = {"kind": context.get("kind", "conversation")}
        rule = next((r for r in self.rules if r.domain == domain and r.scope == scope and r.source != "OWNER_RULE"), None)
        if rule is None:
            rule = AdaptiveRule(domain=domain, scope=scope, source=source)
            self.rules.append(rule)
        before = rule.weight
        rule.weight = max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, rule.weight + step))
        rule.confidence = min(1.0, rule.effective_confidence() + (0.25 if source == "EXPLICIT_CORRECTION" else 0.18))
        rule.last_confirmed = _now()
        rule.evidence.append({**evidence, "at": _now(), "weight": round(rule.weight, 2)})
        del rule.evidence[:-12]
        rule.version += 1
        rule.text = self._describe(rule)
        if abs(before) < 0.5 <= abs(rule.weight):
            self.on_insight("adaptation", {"rule": rule.to_dict(), "summary": rule.text})
        return rule

    @staticmethod
    def _describe(rule: AdaptiveRule) -> str:
        kind = {"technical_explanation": "Bei technischen Erklärungen", "action_confirmation": "Bei Bestätigungen von Aktionen",
                "small_talk": "Im Smalltalk", "conversation": "Im Gespräch"}.get(rule.scope.get("kind", ""), "Allgemein")
        if rule.domain == "ANSWER_LENGTH":
            return f"{kind}: {'ausführlicher' if rule.weight > 0 else 'kürzer'} antworten"
        if rule.domain == "TECHNICAL_DEPTH":
            return f"{kind}: {'mehr' if rule.weight > 0 else 'weniger'} technische Tiefe"
        return f"{kind}: {rule.domain.lower()} anpassen"

    # -- owner-authored rules -------------------------------------------

    def add_owner_rule(self, text: str, *, domain: str = "STYLE", scope: dict[str, str] | None = None, weight: float = 0.0) -> AdaptiveRule:
        rule = AdaptiveRule(domain=domain if domain in DOMAINS else "STYLE", scope=dict(scope or {}), weight=max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, weight)),
                            text=text.strip(), source="OWNER_RULE", confidence=1.0)
        with self._lock:
            self.rules.append(rule)
            self.save()
        return rule

    def update_rule(self, rule_id: str, changes: dict[str, Any]) -> AdaptiveRule | None:
        with self._lock:
            rule = next((r for r in self.rules if r.rule_id == rule_id), None)
            if rule is None:
                return None
            for key in ("enabled", "text", "weight", "confidence"):
                if key in changes:
                    setattr(rule, key, type(getattr(rule, key))(changes[key]))
            if "scope" in changes and isinstance(changes["scope"], dict):
                rule.scope = {str(k): str(v) for k, v in changes["scope"].items()}
            rule.weight = max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, rule.weight))
            rule.version += 1
            self.save()
            return rule

    def delete_rule(self, rule_id: str) -> bool:
        with self._lock:
            before = len(self.rules)
            self.rules = [r for r in self.rules if r.rule_id != rule_id]
            if len(self.rules) != before:
                self.save()
                return True
            return False

    def list_rules(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rules if include_disabled or r.enabled]

    # -- what reaches the prompt ----------------------------------------

    def guidance(self, context: dict[str, str], *, min_confidence: float = 0.35, min_weight: float = 0.5) -> list[str]:
        """Prompt lines for this context: owner rules first, then confident learned nudges."""

        lines: list[str] = []
        applicable = sorted((r for r in self.rules if r.enabled and r.matches(context)), key=lambda r: -r.priority)
        for rule in applicable:
            if rule.source == "OWNER_RULE" and rule.text:
                lines.append(rule.text)
            elif abs(rule.weight) >= min_weight and rule.effective_confidence() >= min_confidence:
                if rule.domain == "ANSWER_LENGTH":
                    lines.append("Antworte hier " + ("deutlich ausführlicher und vollständiger als sonst üblich."
                                                     if rule.weight > 0 else "besonders knapp -- ein bis zwei Sätze."))
                elif rule.domain == "TECHNICAL_DEPTH":
                    lines.append("Gehe hier " + ("tiefer ins technische Detail." if rule.weight > 0 else "weniger ins technische Detail."))
                elif rule.text:
                    lines.append(rule.text)
        return lines[:5]

    # -- action/verifier lessons ----------------------------------------

    def record_action_feedback(self, *, kind: str, verdict: str, receipt_id: str = "", request: str = "", detail: str = "",
                               threshold: int = 3) -> dict[str, Any]:
        """"ZEUS said FAILED but it worked" (RESULT_WAS_SUCCESSFUL) and friends.

        Attached to the verifier *kind* (e.g. ``music.play``); after
        ``threshold`` consistent owner verdicts an INSIGHT is raised so the
        defect can become a SelfDev proposal -- production code is never
        rewritten from a thumbs-down.
        """

        with self._lock:
            rows = self.lessons.setdefault(kind, [])
            rows.append({"verdict": verdict, "receipt_id": receipt_id, "request": request[:200], "detail": detail[:300], "at": _now()})
            del rows[:-50]
            self.feedback_log.append({"kind": "action", "action_kind": kind, "verdict": verdict, "receipt_id": receipt_id, "at": _now()})
            same = [r for r in rows if r["verdict"] == verdict]
            insight = None
            if len(same) >= threshold and verdict in {"RESULT_WAS_SUCCESSFUL", "RESULT_WAS_WRONG"}:
                insight = (f"The {kind} verifier was overruled by the owner {len(same)} times "
                           f"({'reported failure but the result was correct' if verdict == 'RESULT_WAS_SUCCESSFUL' else 'reported success wrongly'}).")
                self.on_insight("verifier", {"kind": kind, "verdict": verdict, "count": len(same), "summary": insight,
                                             "evidence": same[-threshold:]})
            self.save()
        return {"ok": True, "count": len(same), "insight": insight}

    def stats(self) -> dict[str, Any]:
        return {"rules": len(self.rules), "enabled": sum(1 for r in self.rules if r.enabled),
                "feedback": len(self.feedback_log), "lessons": {k: len(v) for k, v in self.lessons.items()}}
