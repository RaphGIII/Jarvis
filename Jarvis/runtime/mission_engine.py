"""The durable Mission Engine: work larger than one model context.

A model's context is temporary; a mission's state is authoritative.  Every
mission is one JSON file that says what was asked, how it was read, what
constrains it, what would count as done, what has been established (evidence),
what is still open (hypotheses, tasks), what was tried and failed, where it is
(phase) and what happens next.  A new process resumes from that file -- never
from a 500k-token transcript.

Phases (any mission, whatever its kind):

    UNDERSTAND → INVESTIGATE → RESEARCH → DECOMPOSE → PLAN → EXECUTE →
    OBSERVE → VERIFY → DIAGNOSE → REPLAN → ESCALATE → BLOCKED → COMPLETE
    (+ FAILED, CANCELLED, PAUSED)

Transitions are checked: a mission cannot go from PLAN to COMPLETE without
passing VERIFY, and nothing marks COMPLETE without at least one piece of
evidence of kind EXECUTION_RECEIPT or VERIFIED_FACT (see :mod:`runtime.evidence`).

The engine does not know how to *do* anything.  Kinds register a handler
(``selfdev``, ``capability``, ``research``, ``project``, ``complex``) that is
asked for the next step; the engine keeps the record, enforces the phases,
persists after every step, and answers pause / resume / cancel / inspect.
Existing mission systems (SelfDev, capability acquisition) are projected into
the same listing so Mission Control shows one truth.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from runtime.evidence import Evidence, EvidenceKind


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


PHASES = ("UNDERSTAND", "INVESTIGATE", "RESEARCH", "DECOMPOSE", "PLAN", "EXECUTE", "OBSERVE", "VERIFY", "DIAGNOSE",
          "REPLAN", "ESCALATE", "BLOCKED", "COMPLETE", "FAILED", "CANCELLED", "PAUSED")
TERMINAL = {"COMPLETE", "FAILED", "CANCELLED"}

#: Where a phase may go next.  Loose enough for real work (DIAGNOSE can go back
#: to INVESTIGATE), strict where it matters: COMPLETE only after VERIFY.
TRANSITIONS: dict[str, set[str]] = {
    "UNDERSTAND": {"INVESTIGATE", "RESEARCH", "DECOMPOSE", "PLAN", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "INVESTIGATE": {"RESEARCH", "DECOMPOSE", "PLAN", "EXECUTE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "RESEARCH": {"INVESTIGATE", "DECOMPOSE", "PLAN", "VERIFY", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "DECOMPOSE": {"PLAN", "EXECUTE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "PLAN": {"EXECUTE", "RESEARCH", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "EXECUTE": {"OBSERVE", "VERIFY", "DIAGNOSE", "ESCALATE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "OBSERVE": {"VERIFY", "DIAGNOSE", "EXECUTE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "VERIFY": {"COMPLETE", "DIAGNOSE", "EXECUTE", "REPLAN", "ESCALATE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "DIAGNOSE": {"REPLAN", "INVESTIGATE", "RESEARCH", "EXECUTE", "ESCALATE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "REPLAN": {"PLAN", "EXECUTE", "ESCALATE", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "ESCALATE": {"EXECUTE", "VERIFY", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"},
    "BLOCKED": {"UNDERSTAND", "INVESTIGATE", "PLAN", "EXECUTE", "ESCALATE", "FAILED", "CANCELLED", "PAUSED"},
    "PAUSED": set(PHASES) - {"PAUSED"},
    "COMPLETE": set(), "FAILED": set(), "CANCELLED": set(),
}


@dataclass
class Task:
    task_id: str
    title: str
    status: str = "todo"  # todo | active | done | failed | blocked | skipped
    depends_on: list[str] = field(default_factory=list)
    result: str = ""
    attempts: int = 0
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Mission:
    goal: str
    kind: str = "complex"  # selfdev | capability | research | project | complex
    mission_id: str = field(default_factory=lambda: f"m_{uuid.uuid4().hex[:10]}")
    interpretation: str = ""
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)  # {text, status: open|confirmed|refuted}
    tasks: list[dict[str, Any]] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    failed_approaches: list[dict[str, Any]] = field(default_factory=list)
    phase: str = "UNDERSTAND"
    previous_phase: str = ""
    next_action: str = ""
    blockers: list[str] = field(default_factory=list)
    owner_input_required: str = ""
    outcome: str = ""  # complete | failed | cancelled
    reason: str = ""
    scope: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    history: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    pause_requested: bool = False
    links: dict[str, Any] = field(default_factory=dict)  # project_id, selfdev mission, capability ...

    @property
    def finished(self) -> bool:
        return self.phase in TERMINAL

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["finished"] = self.finished
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mission":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class MissionEngineStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def path_for(self, mission_id: str) -> Path:
        return self.root / f"{mission_id}.json"

    def save(self, mission: Mission) -> Path:
        mission.updated_at = _now()
        path = self.path_for(mission.mission_id)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            # Flags set from another thread survive an in-memory save.
            if path.is_file() and not (mission.cancel_requested and mission.pause_requested):
                try:
                    disk = json.loads(path.read_text(encoding="utf-8"))
                    mission.cancel_requested = mission.cancel_requested or bool(disk.get("cancel_requested"))
                    mission.pause_requested = mission.pause_requested or bool(disk.get("pause_requested"))
                except (OSError, ValueError):
                    pass
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(mission.to_dict(), indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        return path

    def load(self, mission_id: str) -> Mission | None:
        path = self.path_for(mission_id)
        if not path.is_file():
            return None
        try:
            return Mission.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def list(self) -> list[Mission]:
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                out.append(Mission.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue
        return out

    def active(self) -> list[Mission]:
        return [m for m in self.list() if not m.finished and m.phase != "PAUSED"]


class MissionPaused(RuntimeError):
    pass


class MissionCancelled(RuntimeError):
    pass


class MissionEngine:
    """Keeps the record, enforces the phases, persists every step."""

    def __init__(self, store: MissionEngineStore, *, emit: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.store = store
        self.emit = emit or (lambda kind, payload: None)
        self.handlers: dict[str, Callable[["MissionEngine", Mission], None]] = {}

    def register(self, kind: str, handler: Callable[["MissionEngine", Mission], None]) -> None:
        self.handlers[kind] = handler

    # -- creation ------------------------------------------------------

    def create(self, goal: str, *, kind: str = "complex", interpretation: str = "", constraints: Iterable[str] = (),
               acceptance: Iterable[str] = (), scope: str = "", links: dict[str, Any] | None = None) -> Mission:
        mission = Mission(goal=goal.strip(), kind=kind, interpretation=interpretation, constraints=list(constraints),
                          acceptance_criteria=list(acceptance), scope=scope, links=dict(links or {}))
        self._event(mission, "created", f"{kind}: {goal[:120]}")
        self.store.save(mission)
        self.emit("progress", {"kind": "mission", "mission_id": mission.mission_id, "phase": mission.phase,
                               "summary": f"mission created: {goal[:100]}", "request": goal[:200]})
        return mission

    # -- phases --------------------------------------------------------

    def transition(self, mission: Mission, phase: str, detail: str = "") -> Mission:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase}")
        allowed = TRANSITIONS.get(mission.phase, set())
        if phase != mission.phase and phase not in allowed:
            raise ValueError(f"{mission.phase} -> {phase} is not a valid transition")
        if phase == "COMPLETE" and not self.has_proof(mission):
            raise ValueError("COMPLETE needs an execution receipt or a verified fact as evidence")
        self._check_flags(mission)
        mission.previous_phase, mission.phase = mission.phase, phase
        if phase == "COMPLETE":
            mission.outcome = "complete"
        elif phase == "FAILED":
            mission.outcome = "failed"
        elif phase == "CANCELLED":
            mission.outcome = "cancelled"
        self._event(mission, phase, detail)
        self.store.save(mission)
        self.emit("progress", {"kind": "mission", "mission_id": mission.mission_id, "phase": phase,
                               "summary": f"mission {phase}: {detail}"[:300], "request": mission.goal[:200]})
        return mission

    def _check_flags(self, mission: Mission) -> None:
        latest = self.store.load(mission.mission_id)
        if latest is not None:
            mission.cancel_requested = mission.cancel_requested or latest.cancel_requested
            mission.pause_requested = mission.pause_requested or latest.pause_requested
        if mission.cancel_requested and not mission.finished:
            raise MissionCancelled(mission.mission_id)
        if mission.pause_requested and not mission.finished:
            raise MissionPaused(mission.mission_id)

    def has_proof(self, mission: Mission) -> bool:
        return any(e.get("kind") in {EvidenceKind.EXECUTION_RECEIPT.value, EvidenceKind.VERIFIED_FACT.value} for e in mission.evidence)

    # -- state --------------------------------------------------------

    def add_evidence(self, mission: Mission, evidence: Evidence) -> Evidence:
        mission.evidence.append(evidence.to_dict())
        self._event(mission, "evidence", f"{evidence.kind.value}: {evidence.claim[:100]}")
        self.store.save(mission)
        return evidence

    def add_hypothesis(self, mission: Mission, text: str) -> dict[str, Any]:
        row = {"id": f"h{len(mission.hypotheses) + 1}", "text": text, "status": "open", "at": _now()}
        mission.hypotheses.append(row)
        self.store.save(mission)
        return row

    def settle_hypothesis(self, mission: Mission, hypothesis_id: str, status: str, evidence_id: str = "") -> None:
        for row in mission.hypotheses:
            if row["id"] == hypothesis_id:
                row["status"] = status
                row["evidence_id"] = evidence_id
        self.store.save(mission)

    def add_task(self, mission: Mission, title: str, *, depends_on: Iterable[str] = ()) -> Task:
        task = Task(task_id=f"t{len(mission.tasks) + 1}", title=title, depends_on=list(depends_on))
        mission.tasks.append(task.to_dict())
        self.store.save(mission)
        return task

    def ready_tasks(self, mission: Mission) -> list[dict[str, Any]]:
        done = {t["task_id"] for t in mission.tasks if t["status"] == "done"}
        return [t for t in mission.tasks if t["status"] == "todo" and all(d in done for d in t.get("depends_on", []))]

    def blocked_tasks(self, mission: Mission) -> list[tuple[dict[str, Any], list[str]]]:
        """Tasks that cannot start, with the unfinished task ids that block them."""

        done = {t["task_id"] for t in mission.tasks if t["status"] == "done"}
        out = []
        for t in mission.tasks:
            if t["status"] == "todo":
                missing = [d for d in t.get("depends_on", []) if d not in done]
                if missing:
                    out.append((t, missing))
        return out

    def update_task(self, mission: Mission, task_id: str, *, status: str, result: str = "", evidence_id: str = "") -> None:
        for t in mission.tasks:
            if t["task_id"] == task_id:
                t["status"] = status
                t["result"] = result[:600]
                if status == "active":
                    t["attempts"] = int(t.get("attempts", 0)) + 1
                if evidence_id:
                    t.setdefault("evidence_ids", []).append(evidence_id)
                if status == "done" and task_id not in mission.completed:
                    mission.completed.append(task_id)
        self._event(mission, "task", f"{task_id} {status}: {result[:80]}")
        self.store.save(mission)

    def fail_approach(self, mission: Mission, approach: str, why: str) -> None:
        mission.failed_approaches.append({"approach": approach[:300], "why": why[:400], "at": _now()})
        self._event(mission, "failed_approach", f"{approach[:80]}: {why[:80]}")
        self.store.save(mission)

    def block(self, mission: Mission, blocker: str, *, owner_input: str = "") -> Mission:
        mission.blockers.append(blocker[:300])
        mission.owner_input_required = owner_input[:300]
        return self.transition(mission, "BLOCKED", blocker)

    def set_next(self, mission: Mission, action: str) -> None:
        mission.next_action = action[:300]
        self.store.save(mission)

    # -- control ------------------------------------------------------

    def request_cancel(self, mission_id: str) -> dict[str, Any]:
        mission = self.store.load(mission_id)
        if mission is None:
            return {"ok": False, "error": f"no mission {mission_id}"}
        if mission.finished:
            return {"ok": False, "error": f"mission is already {mission.phase}"}
        mission.cancel_requested = True
        self.store.save(mission)
        return {"ok": True, "mission_id": mission_id}

    def request_pause(self, mission_id: str) -> dict[str, Any]:
        mission = self.store.load(mission_id)
        if mission is None or mission.finished:
            return {"ok": False, "error": "no such active mission"}
        mission.pause_requested = True
        self.store.save(mission)
        return {"ok": True, "mission_id": mission_id}

    def resume(self, mission_id: str) -> dict[str, Any]:
        mission = self.store.load(mission_id)
        if mission is None or mission.finished:
            return {"ok": False, "error": "no such mission, or it is finished"}
        mission.pause_requested = False
        mission.cancel_requested = False
        if mission.phase == "PAUSED":
            mission.phase = mission.previous_phase or "UNDERSTAND"
            self._event(mission, "resumed", f"back to {mission.phase}")
        self.store.save(mission)
        return {"ok": True, "mission_id": mission_id, "phase": mission.phase}

    def settle(self, mission: Mission, exc: BaseException) -> Mission:
        """What a handler's exception means for the record."""

        if isinstance(exc, MissionCancelled):
            mission.previous_phase, mission.phase, mission.outcome = mission.phase, "CANCELLED", "cancelled"
            mission.reason = "cancelled by the owner"
        elif isinstance(exc, MissionPaused):
            mission.previous_phase, mission.phase = mission.phase, "PAUSED"
            mission.reason = "paused by the owner"
        else:
            mission.previous_phase, mission.phase, mission.outcome = mission.phase, "FAILED", "failed"
            mission.reason = f"{type(exc).__name__}: {exc}"[:600]
        self._event(mission, mission.phase, mission.reason)
        self.store.save(mission)
        self.emit("progress", {"kind": "mission", "mission_id": mission.mission_id, "phase": mission.phase,
                               "summary": f"mission {mission.phase}: {mission.reason}"[:300]})
        return mission

    def run(self, mission: Mission) -> Mission:
        """Hand the mission to its kind's handler; the record survives whatever happens."""

        handler = self.handlers.get(mission.kind)
        if handler is None:
            mission.reason = f"no handler for missions of kind {mission.kind!r}"
            return self.transition(mission, "BLOCKED", mission.reason)
        try:
            handler(self, mission)
        except (MissionCancelled, MissionPaused, Exception) as exc:  # noqa: BLE001
            return self.settle(mission, exc)
        return mission

    # -- startup ------------------------------------------------------

    def resumable(self) -> list[Mission]:
        """Missions a fresh process should pick up: not finished, not paused."""

        return [m for m in self.store.list() if not m.finished and m.phase != "PAUSED"]

    def mark_interrupted(self) -> list[Mission]:
        """At startup: a mission left mid-phase by a dead process is noted, kept resumable."""

        out = []
        for m in self.resumable():
            m.history.append({"at": _now(), "event": "interrupted", "detail": f"the process ended during {m.phase}"})
            m.next_action = m.next_action or f"resume from {m.phase}"
            self.store.save(m)
            out.append(m)
        return out

    def _event(self, mission: Mission, event: str, detail: str) -> None:
        mission.history.append({"at": _now(), "event": event, "detail": detail[:400]})
        if len(mission.history) > 400:
            mission.history = mission.history[-400:]

    # -- summary for the owner ------------------------------------------

    @staticmethod
    def brief(mission: Mission) -> dict[str, Any]:
        """The compact, durable summary a new context reads instead of a transcript."""

        return {
            "mission_id": mission.mission_id, "kind": mission.kind, "goal": mission.goal, "interpretation": mission.interpretation,
            "phase": mission.phase, "next_action": mission.next_action, "constraints": mission.constraints,
            "acceptance_criteria": mission.acceptance_criteria,
            "established": [e.get("claim") for e in mission.evidence if e.get("kind") in {"execution_receipt", "verified_fact", "tool_observation"}][-12:],
            "open_hypotheses": [h["text"] for h in mission.hypotheses if h.get("status") == "open"][:8],
            "tasks": {"done": len(mission.completed), "total": len(mission.tasks),
                      "next": [t["title"] for t in mission.tasks if t["status"] == "todo"][:5]},
            "failed_approaches": [f["approach"] for f in mission.failed_approaches][-6:],
            "blockers": mission.blockers[-4:], "owner_input_required": mission.owner_input_required,
        }
