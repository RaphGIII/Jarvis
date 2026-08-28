"""Proactive thoughts: ZEUS noticing things, bounded and evidence-backed.

A thought is a typed observation (INSIGHT, CONNECTION, WARNING, OPPORTUNITY,
REMINDER, PROJECT_RISK, OPTIMIZATION, FOLLOW_UP, QUESTION, IDEA) with the
evidence it rests on, the project/context it concerns, a confidence, an
importance, why it matters and an optional suggested action.  Thoughts come
from *deterministic detectors* over what ZEUS already records -- missions,
corrections, capability health, project activity -- never from a model
inventing an event.  No model is woken for relevance detection; a deeper
investigation may use FAST_LOCAL later, and never BUILD_LOCAL by itself.

When ZEUS speaks up is a policy, not an impulse:

    LOW     -> stored (the inbox)
    MEDIUM  -> the inbox badge
    HIGH    -> said once, at the next natural moment (after an answer)
    URGENT  -> a notification now

Every thought has a key; the same key within its cooldown is one thought
with a higher count, not a second notification.  Dismissing a type three
times lowers that type's prominence; muting a type silences it.  The owner's
proactivity dial scales the thresholds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

TYPES = ("INSIGHT", "CONNECTION", "WARNING", "OPPORTUNITY", "REMINDER", "PROJECT_RISK", "OPTIMIZATION", "FOLLOW_UP", "QUESTION", "IDEA")
IMPORTANCE = ("LOW", "MEDIUM", "HIGH", "URGENT")
STATUSES = ("NEW", "IMPORTANT", "SAVED", "DISMISSED", "ACTED_ON")
#: The same key is one thought for this long.
COOLDOWN_HOURS = 24
#: Dismissals of one type before it is demoted a level.
DEMOTE_AFTER = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Thought:
    type: str
    title: str
    text: str
    why_it_matters: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)      # project_id, mission_id, capability_id, ...
    confidence: float = 0.6
    importance: str = "LOW"
    suggested_action: str = ""
    key: str = ""
    thought_id: str = ""
    status: str = "NEW"
    generated_at: str = field(default_factory=_now)
    updated_at: str = ""
    count: int = 1
    delivered_at: str = ""
    delivered_how: str = ""

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown thought type {self.type}")
        if self.importance not in IMPORTANCE:
            raise ValueError(f"unknown importance {self.importance}")
        if not self.key:
            self.key = hashlib.sha1(f"{self.type}|{self.title.lower()}".encode("utf-8")).hexdigest()[:12]
        if not self.thought_id:
            self.thought_id = "th_" + hashlib.sha1(f"{self.key}|{self.generated_at}".encode("utf-8")).hexdigest()[:10]
        if not self.evidence:
            raise ValueError("a thought needs evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Thought":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Detectors: pure functions over the records
# --------------------------------------------------------------------------

def _cause(detail: str) -> str:
    text = re.sub(r"\s+", " ", str(detail or "")).strip()
    text = re.sub(r"[0-9a-f]{8,}", "…", text)
    return text[:80].lower()


def repeated_failures(missions: Iterable[dict[str, Any]], *, language: str = "de") -> list[Thought]:
    """Two or more failed missions sharing a failure cause (or a family) -> one INSIGHT."""

    failed = [m for m in missions if str(m.get("state", "")).lower() == "failed"]
    by_cause: dict[str, list[dict[str, Any]]] = {}
    for m in failed:
        cause = _cause(m.get("failure") or m.get("reason") or ((m.get("blockers") or [""])[0] if m.get("blockers") else ""))
        if cause:
            by_cause.setdefault(cause, []).append(m)
    out = []
    for cause, group in by_cause.items():
        if len(group) < 2:
            continue
        titles = ", ".join(sorted({str(m.get("title") or m.get("goal", ""))[:40] for m in group}))
        de = language.startswith("de")
        out.append(Thought(
            type="INSIGHT",
            title=(f"{len(group)} Missionen scheiterten an derselben Ursache" if de else f"{len(group)} missions failed for the same cause"),
            text=(f"Die Missionen {titles} endeten mit derselben Ursache: „{cause}“. Das ist eher ein grundsätzlicher Fehler als ein Einzelfall."
                  if de else f"Missions {titles} ended with the same cause: '{cause}'. That looks systemic rather than incidental."),
            why_it_matters=("Eine gemeinsame Ursache wird durch eine Reparatur behoben, nicht durch weitere Versuche." if de
                            else "One shared cause is fixed by one repair, not by more attempts."),
            evidence=[{"kind": "mission", "ref": m.get("id", ""), "summary": f"{m.get('title', '')[:60]} — {m.get('failure') or m.get('reason') or ''}"[:160]} for m in group],
            context={"mission_ids": [m.get("id", "") for m in group]},
            confidence=0.7, importance="HIGH" if len(group) >= 3 else "MEDIUM",
            suggested_action=("Ursache grundsätzlich reparieren (eine Mission statt vieler Versuche)." if de else "Repair the cause once."),
            key="failures|" + cause[:40],
        ))
    return out


def blocked_family(missions: Iterable[dict[str, Any]], *, language: str = "de") -> list[Thought]:
    """Two blocked missions with the same blocker text -> PROJECT_RISK."""

    blocked = [m for m in missions if str(m.get("state", "")).lower() == "blocked" and m.get("blockers")]
    by_blocker: dict[str, list[dict[str, Any]]] = {}
    for m in blocked:
        by_blocker.setdefault(_cause(m["blockers"][0]), []).append(m)
    out = []
    for blocker, group in by_blocker.items():
        if len(group) < 2:
            continue
        de = language.startswith("de")
        out.append(Thought(
            type="PROJECT_RISK",
            title=(f"{len(group)} Missionen hängen am selben Blocker" if de else f"{len(group)} missions share one blocker"),
            text=(f"Blocker: „{blocker}“. Wahrscheinlich löst ein Fix alle." if de else f"Blocker: '{blocker}'. One fix probably frees all of them."),
            why_it_matters="Blockierte Arbeit summiert sich; ein gemeinsamer Blocker ist der billigste Hebel." if de else "Blocked work accumulates; a shared blocker is the cheapest lever.",
            evidence=[{"kind": "mission", "ref": m.get("id", ""), "summary": str(m.get("title", ""))[:80]} for m in group],
            context={"mission_ids": [m.get("id", "") for m in group]}, confidence=0.65, importance="MEDIUM",
            suggested_action="Den Blocker als eigene Mission angehen." if de else "Take the blocker on as its own mission.",
            key="blocked|" + blocker[:40],
        ))
    return out


def project_inactivity(projects: Iterable[dict[str, Any]], *, now: datetime | None = None, days: int = 3, language: str = "de") -> list[Thought]:
    """An owner project that is not finished and has not moved for ``days`` -> REMINDER (LOW)."""

    now = now or datetime.now(timezone.utc)
    out = []
    for p in projects:
        if p.get("origin", "owner") != "owner":
            continue
        state = str(p.get("state", "")).lower()
        if state in {"completed", "complete", "abandoned", "archived", "draft"}:
            continue
        try:
            updated = datetime.fromisoformat(str(p.get("updated_at", "")).replace("Z", "+00:00"))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        idle = now - updated
        if idle < timedelta(days=days):
            continue
        de = language.startswith("de")
        title = str(p.get("title") or p.get("goal", ""))[:50]
        out.append(Thought(
            type="REMINDER",
            title=(f"Projekt „{title}“ seit {idle.days} Tagen unbewegt" if de else f"Project '{title}' idle for {idle.days} days"),
            text=(f"Letzte Aktivität {updated.date().isoformat()}; Zustand {state}." if de else f"Last activity {updated.date().isoformat()}; state {state}."),
            why_it_matters="Ein angefangenes Projekt ohne Bewegung verliert Kontext." if de else "A started project without movement loses context.",
            evidence=[{"kind": "project", "ref": p.get("id", ""), "summary": f"updated_at {p.get('updated_at', '')}"}],
            context={"project_id": p.get("id", "")}, confidence=0.9, importance="LOW",
            key=f"idle|{p.get('id', '')}",
        ))
    return out


def repeated_corrections(corrections: Iterable[dict[str, Any]], *, language: str = "de") -> list[Thought]:
    """Three or more corrections about the same domain/entity -> OPTIMIZATION."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for c in corrections:
        if not c.get("active", True):
            continue
        entities = c.get("entities") or {}
        subject = str(entities.get("provider") or entities.get("name") or entities.get("query") or c.get("classification", ""))
        domain = str(c.get("scope", ""))
        groups.setdefault(f"{c.get('classification', '')}|{subject.lower()}", []).append(c)
    out = []
    for key, group in groups.items():
        if len(group) < 3:
            continue
        de = language.startswith("de")
        subject = key.split("|", 1)[1] or key.split("|")[0]
        out.append(Thought(
            type="OPTIMIZATION",
            title=(f"{len(group)} Korrekturen zum selben Thema ({subject})" if de else f"{len(group)} corrections on the same subject ({subject})"),
            text=("Du korrigierst mich wiederholt in derselben Sache. Ich sollte daraus eine allgemeinere Regel machen statt Einzelfälle zu merken."
                  if de else "You keep correcting the same thing. A general rule would serve better than remembered instances."),
            why_it_matters="Wiederholte Korrekturen sind ein Regeldefekt, kein Datenproblem." if de else "Repeated corrections are a rule defect, not a data problem.",
            evidence=[{"kind": "correction", "ref": c.get("correction_id", ""), "summary": str(c.get("what_was_wrong", ""))[:100]} for c in group],
            context={"corrections": [c.get("correction_id", "") for c in group]}, confidence=0.75, importance="MEDIUM",
            suggested_action="Eine Resolver-Regel daraus ableiten." if de else "Derive a resolver rule from them.",
            key="corrections|" + key[:50],
        ))
    return out


def capability_degradation(manifests: Iterable[dict[str, Any]], *, language: str = "de") -> list[Thought]:
    out = []
    for m in manifests:
        health = m.get("health") or {}
        state = str(health.get("state", "unverified"))
        if state not in {"failing", "degraded"}:
            continue
        de = language.startswith("de")
        cid = str(m.get("capability_id", ""))
        out.append(Thought(
            type="WARNING",
            title=(f"Fähigkeit {cid} ist {'ausgefallen' if state == 'failing' else 'angeschlagen'}" if de else f"Capability {cid} is {state}"),
            text=(f"Letzter Fehler: {health.get('last_error', '')[:120] or 'unbekannt'} ({health.get('consecutive_failures', 0)} in Folge)."
                  if de else f"Last error: {health.get('last_error', '')[:120] or 'unknown'} ({health.get('consecutive_failures', 0)} in a row)."),
            why_it_matters="Der Planer wählt sie weiterhin, wenn auch nachrangig; ein Auftrag scheitert daran real." if de
                            else "The planner still offers it, demoted; a real request can fail on it.",
            evidence=[{"kind": "capability", "ref": cid, "summary": f"health {state}; last_error_at {health.get('last_error_at', '')}"}],
            context={"capability_id": cid}, confidence=0.9, importance="HIGH" if state == "failing" else "MEDIUM",
            suggested_action="Reparieren (Capability Center → Repair)." if de else "Repair it (Capability Center → Repair).",
            key=f"capability|{cid}|{state}",
        ))
    return out


def shared_dependency(projects: Iterable[dict[str, Any]], missions: Iterable[dict[str, Any]], *, language: str = "de") -> list[Thought]:
    """Two owner projects whose missions name the same capability/subsystem -> CONNECTION."""

    words: dict[str, set[str]] = {}
    owner = [p for p in projects if p.get("origin", "owner") == "owner"]
    for p in owner:
        text = f"{p.get('title', '')} {p.get('goal', '')}".lower()
        for w in re.findall(r"\b(audio|voice|spotify|screen|knowledge|wake|listener|gateway|chess|stockfish|graph|mission|capability)\b", text):
            words.setdefault(w, set()).add(str(p.get("id", "")))
    out = []
    titles = {str(p.get("id", "")): str(p.get("title") or p.get("goal", ""))[:40] for p in owner}
    for word, ids in words.items():
        if len(ids) < 2:
            continue
        de = language.startswith("de")
        names = ", ".join(titles[i] for i in sorted(ids))
        out.append(Thought(
            type="CONNECTION",
            title=(f"{names} teilen „{word}“" if de else f"{names} share '{word}'"),
            text=(f"Die Projekte {names} berühren beide {word}. Eine gemeinsame Abstraktion wäre sinnvoll." if de
                  else f"Projects {names} both touch {word}. A shared abstraction would make sense."),
            why_it_matters="Doppelte Lösungen driften auseinander." if de else "Two solutions of one thing drift apart.",
            evidence=[{"kind": "project", "ref": i, "summary": titles[i]} for i in sorted(ids)],
            context={"project_ids": sorted(ids)}, confidence=0.5, importance="LOW",
            key=f"shared|{word}|{'+'.join(sorted(ids))}",
        ))
    return out


# --------------------------------------------------------------------------
# Store + policy
# --------------------------------------------------------------------------

class ThoughtStore:
    """One JSON file; dedupe by key with a cooldown; per-type dismissal memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.thoughts: dict[str, Thought] = {}
        self.dismissed_by_type: dict[str, int] = {}
        self.muted_types: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in data.get("thoughts", []):
            try:
                t = Thought.from_dict(row)
                self.thoughts[t.thought_id] = t
            except Exception:  # noqa: BLE001
                continue
        self.dismissed_by_type = dict(data.get("dismissed_by_type") or {})
        self.muted_types = set(data.get("muted_types") or [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"thoughts": [t.to_dict() for t in self.thoughts.values()], "dismissed_by_type": self.dismissed_by_type,
                   "muted_types": sorted(self.muted_types)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def effective_importance(self, thought: Thought) -> str:
        level = IMPORTANCE.index(thought.importance)
        demotions = self.dismissed_by_type.get(thought.type, 0) // DEMOTE_AFTER
        return IMPORTANCE[max(0, level - demotions)]

    def offer(self, thought: Thought, *, now: float | None = None) -> tuple[Thought | None, str]:
        """Take a detector's thought: (stored thought, outcome) where outcome is new | refreshed | muted | cooling."""

        if thought.type in self.muted_types:
            return None, "muted"
        with self._lock:
            existing = next((t for t in self.thoughts.values() if t.key == thought.key), None)
            if existing is not None:
                if existing.status == "DISMISSED":
                    return None, "dismissed"
                try:
                    age = datetime.now(timezone.utc) - datetime.fromisoformat(existing.updated_at or existing.generated_at)
                except ValueError:
                    age = timedelta(0)
                existing.count += 1
                existing.evidence = thought.evidence
                existing.text = thought.text
                existing.updated_at = _now()
                if age < timedelta(hours=COOLDOWN_HOURS):
                    self.save()
                    return existing, "cooling"
                existing.status = "NEW" if existing.status not in {"SAVED", "ACTED_ON"} else existing.status
                existing.delivered_at = ""
                self.save()
                return existing, "refreshed"
            thought.importance = self.effective_importance(thought)
            thought.updated_at = thought.generated_at
            if thought.importance in {"HIGH", "URGENT"}:
                thought.status = "IMPORTANT"
            self.thoughts[thought.thought_id] = thought
            self.save()
            return thought, "new"

    def get(self, thought_id: str) -> Thought | None:
        return self.thoughts.get(thought_id)

    def set_status(self, thought_id: str, status: str) -> Thought | None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status}")
        with self._lock:
            t = self.thoughts.get(thought_id)
            if t is None:
                return None
            if status == "DISMISSED" and t.status != "DISMISSED":
                self.dismissed_by_type[t.type] = self.dismissed_by_type.get(t.type, 0) + 1
            t.status = status
            t.updated_at = _now()
            self.save()
            return t

    def mute(self, thought_type: str, muted: bool = True) -> None:
        if thought_type not in TYPES:
            raise ValueError(f"unknown thought type {thought_type}")
        with self._lock:
            (self.muted_types.add if muted else self.muted_types.discard)(thought_type)
            self.save()

    def mark_delivered(self, thought_id: str, how: str) -> None:
        with self._lock:
            t = self.thoughts.get(thought_id)
            if t is not None:
                t.delivered_at, t.delivered_how = _now(), how
                self.save()

    def undelivered(self, *, min_importance: str = "HIGH") -> list[Thought]:
        floor = IMPORTANCE.index(min_importance)
        return sorted((t for t in self.thoughts.values() if not t.delivered_at and t.status in {"NEW", "IMPORTANT"}
                       and IMPORTANCE.index(t.importance) >= floor), key=lambda t: (-IMPORTANCE.index(t.importance), t.generated_at))

    def list(self, status: str = "") -> list[Thought]:
        rows = [t for t in self.thoughts.values() if not status or t.status == status]
        return sorted(rows, key=lambda t: (t.status == "DISMISSED", -IMPORTANCE.index(t.importance), t.updated_at or t.generated_at), reverse=False)

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for t in self.thoughts.values():
            out[t.status] = out.get(t.status, 0) + 1
        return out


def speak_policy(proactivity: int) -> dict[str, str]:
    """From the owner's proactivity dial: which importance is spoken, which interrupts."""

    if proactivity < 20:
        return {"speak_at": "URGENT", "interrupt_at": "URGENT"}
    if proactivity < 60:
        return {"speak_at": "HIGH", "interrupt_at": "URGENT"}
    return {"speak_at": "MEDIUM", "interrupt_at": "HIGH"}


class ThoughtEngine:
    """Runs the detectors over what the core knows, on cheap triggers and an idle timer."""

    TRIGGERS = ("mission_finished", "correction", "capability_health", "idle", "manual", "knowledge")
    MIN_INTERVAL_SECONDS = 60.0

    def __init__(self, store: ThoughtStore, *, facts: Callable[[], dict[str, Any]], language: Callable[[], str] = lambda: "de",
                 proactivity: Callable[[], int] = lambda: 50, emit: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.store = store
        self.facts = facts
        self.language = language
        self.proactivity = proactivity
        self.emit = emit or (lambda kind, payload: None)
        self._last_run = 0.0
        self.runs: list[dict[str, Any]] = []

    def tick(self, trigger: str = "manual", *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force and now - self._last_run < self.MIN_INTERVAL_SECONDS:
            return {"ran": False, "reason": "cooldown", "trigger": trigger}
        self._last_run = now
        facts = self.facts()
        language = self.language() or "de"
        found: list[Thought] = []
        for detector in (
            lambda: repeated_failures(facts.get("missions", []), language=language),
            lambda: blocked_family(facts.get("missions", []), language=language),
            lambda: project_inactivity(facts.get("projects", []), language=language),
            lambda: repeated_corrections(facts.get("corrections", []), language=language),
            lambda: capability_degradation(facts.get("capabilities", []), language=language),
            lambda: shared_dependency(facts.get("projects", []), facts.get("missions", []), language=language),
        ):
            try:
                found.extend(detector())
            except Exception as exc:  # noqa: BLE001 - one detector never stops the others
                self.emit("diagnostic", {"thoughts": f"detector failed: {exc}"})
        outcomes = {"new": 0, "refreshed": 0, "cooling": 0, "muted": 0, "dismissed": 0}
        new: list[Thought] = []
        for thought in found:
            stored, outcome = self.store.offer(thought)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome in {"new", "refreshed"} and stored is not None:
                new.append(stored)
        policy = speak_policy(int(self.proactivity() or 50))
        for thought in new:
            if IMPORTANCE.index(thought.importance) >= IMPORTANCE.index(policy["interrupt_at"]):
                self.emit("notification", {"kind": "thought", "text": thought.title, "thought_id": thought.thought_id, "importance": thought.importance})
                self.store.mark_delivered(thought.thought_id, "notification")
        record = {"ran": True, "trigger": trigger, "at": _now(), "found": len(found), **outcomes, "policy": policy}
        self.runs.append(record)
        del self.runs[:-20]
        if new:
            self.emit("diagnostic", {"thoughts": f"{len(new)} new/refreshed thought(s) after {trigger}", "ids": [t.thought_id for t in new]})
        return record

    def next_to_say(self) -> Thought | None:
        """A thought worth saying at the next natural moment, per the owner's dial; None when nothing qualifies."""

        policy = speak_policy(int(self.proactivity() or 50))
        pending = self.store.undelivered(min_importance=policy["speak_at"])
        return pending[0] if pending else None
