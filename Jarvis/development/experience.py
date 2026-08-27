"""Verified development experience: what a self-development mission learned.

After a mission ends -- promoted, failed or rolled back -- a compact,
structured entry is kept: the subsystem, the goal, the files that mattered,
the search path that found them, what failed and why, what finally worked,
which tests and verifier decided it, how long each stage took, whether the
expert was needed.  Before the next mission, the entries that match its
goal are retrieved and turned into a few lines the coder reads first.

Compact on purpose: injecting an old trajectory would cost the context
window that prompt-size measurements already showed to dominate the wall
clock.  A dozen lines that say "the header lives in ui/index.html and
ui/app.js; the uptime widget was added by editing the header markup and a
poll in app.js; three anchor failures came from guessing line text" are what
shorten the next investigation.

Whether it helps is measured, not assumed: ``compare`` reports investigate
time, model calls, expert use and total time for similar tasks over time.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STOP = {"the", "and", "der", "die", "das", "und", "in", "im", "zu", "zum", "zur", "von", "mit", "ein", "eine", "einen", "den", "dem",
        "des", "an", "auf", "bei", "für", "fuer", "als", "auch", "soll", "sollen", "dass", "wenn", "ich", "du", "dein", "deine", "deinen",
        "deiner", "zeus", "your", "you", "of", "to", "a", "it", "is", "be", "this", "that", "my", "me", "bitte", "selbst", "dann"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Zäöüß_]{3,}", (text or "").lower()) if w not in STOP}


@dataclass
class Experience:
    goal: str
    subsystem: str  # ui | code | supervisor | capability | voice ...
    outcome: str    # promoted | failed | rolled_back | cancelled
    relevant_files: list[str] = field(default_factory=list)
    search_path: list[str] = field(default_factory=list)      # what the investigation ranked, in order
    failed_hypotheses: list[str] = field(default_factory=list)
    strategy: str = ""                                          # what finally worked, one paragraph
    tests: list[str] = field(default_factory=list)
    verifier: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    model_calls: int = 0
    local_attempts: int = 0
    expert_used: bool = False
    expert_provider: str = ""
    mission_id: str = ""
    revision: str = ""
    experience_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    at: str = field(default_factory=_now)
    #: how often it was retrieved for a later mission
    used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary_lines(self) -> list[str]:
        lines = [f"- [{self.outcome}, {self.subsystem}] {self.goal[:100]}"]
        if self.relevant_files:
            lines.append(f"  files that mattered: {', '.join(self.relevant_files[:6])}")
        if self.strategy:
            lines.append(f"  what worked: {self.strategy[:220]}")
        if self.failed_hypotheses:
            lines.append(f"  what failed: {'; '.join(h[:80] for h in self.failed_hypotheses[:3])}")
        if self.tests:
            lines.append(f"  tests that decided it: {', '.join(self.tests[:4])}")
        return lines


class ExperienceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _rows(self) -> list[Experience]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Experience.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                continue
        return out

    def _write(self, rows: Iterable[Experience]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, experience: Experience) -> Experience:
        with self._lock:
            rows = self._rows()
            rows.append(experience)
            self._write(rows)
        return experience

    def list(self) -> list[Experience]:
        return self._rows()

    def relevant(self, goal: str, *, subsystem: str = "", limit: int = 3) -> list[Experience]:
        """The most similar earlier missions, verified outcomes first."""

        wanted = terms(goal)
        scored: list[tuple[float, Experience]] = []
        for row in self._rows():
            overlap = len(wanted & terms(row.goal)) + 0.5 * len(wanted & set(w for f in row.relevant_files for w in terms(f.replace("/", " "))))
            if subsystem and row.subsystem == subsystem:
                overlap += 1.0
            if row.outcome == "promoted":
                overlap += 0.5
            if overlap >= 1.5:
                scored.append((overlap, row))
        scored.sort(key=lambda item: (-item[0], item[1].at))
        chosen = [row for _, row in scored[:limit]]
        if chosen:
            with self._lock:
                rows = self._rows()
                ids = {c.experience_id for c in chosen}
                for row in rows:
                    if row.experience_id in ids:
                        row.used += 1
                self._write(rows)
        return chosen

    def guidance(self, goal: str, *, subsystem: str = "", limit: int = 3) -> str:
        """The compact lines a coder reads before it starts.  Empty when nothing matches."""

        rows = self.relevant(goal, subsystem=subsystem, limit=limit)
        if not rows:
            return ""
        lines = ["Verified experience from earlier self-development on this repository:"]
        for row in rows:
            lines.extend(row.summary_lines())
        return "\n".join(lines)

    def compare(self, goal: str, *, subsystem: str = "") -> dict[str, Any]:
        """Did similar tasks get cheaper?  Numbers per mission, in time order."""

        wanted = terms(goal)
        rows = [r for r in self._rows() if (subsystem and r.subsystem == subsystem) or len(wanted & terms(r.goal)) >= 2]
        rows.sort(key=lambda r: r.at)
        series = [{"mission_id": r.mission_id, "at": r.at, "outcome": r.outcome, "investigate_s": r.timings.get("investigate", 0.0),
                   "build_s": r.timings.get("build", 0.0), "total_s": round(sum(r.timings.values()), 1), "model_calls": r.model_calls,
                   "local_attempts": r.local_attempts, "expert": r.expert_used, "files": len(r.relevant_files)} for r in rows]
        trend = {}
        if len(series) >= 2:
            first, last = series[0], series[-1]
            for key in ("total_s", "build_s", "model_calls"):
                trend[key] = {"first": first[key], "last": last[key], "change": round(last[key] - first[key], 1)}
        return {"missions": series, "trend": trend}


def from_selfdev_mission(mission: Any, *, revision: str = "") -> Experience:
    """Distil a finished SelfDev mission into an experience entry."""

    events = getattr(mission, "events", []) or []
    failed = []
    for e in events:
        err = str(e.get("error", "") or "")
        if e.get("phase") == "BUILD" and err:
            failed.append(err[:160])
        if e.get("phase") == "VERIFY" and "FAILED" in str(e.get("detail", "")):
            failed.append(str(e.get("detail", ""))[:160])
    strategy = ""
    expert = getattr(mission, "expert", {}) or {}
    if getattr(mission, "outcome", "") == "promoted":
        who = f"the expert ({expert.get('provider', 'expert')})" if getattr(mission, "escalated", False) else "BUILD_LOCAL"
        strategy = f"{who} changed {', '.join((getattr(mission, 'changed_files', []) or [])[:5])}; verified by {getattr(mission, 'verification', {}).get('detail', '')[:120]}"
    verification = getattr(mission, "verification", {}) or {}
    return Experience(
        goal=str(getattr(mission, "request", ""))[:300], subsystem=str(getattr(mission, "area", "") or "code"),
        outcome=str(getattr(mission, "outcome", "") or "failed"),
        relevant_files=list(getattr(mission, "changed_files", []) or [])[:12] or list((getattr(mission, "investigation", {}) or {}).get("files", []))[:8],
        search_path=list((getattr(mission, "investigation", {}) or {}).get("files", []))[:8],
        failed_hypotheses=failed[:6], strategy=strategy, tests=list(verification.get("tests", []) or [])[:6],
        verifier=str(verification.get("detail", ""))[:200], timings=dict(getattr(mission, "timings", {}) or {}),
        model_calls=int(getattr(mission, "model_calls", 0) or 0), local_attempts=int(getattr(mission, "local_attempts", 0) or 0),
        expert_used=bool(getattr(mission, "escalated", False)), expert_provider=str(expert.get("provider", "")),
        mission_id=str(getattr(mission, "mission_id", "")), revision=revision or str(getattr(mission, "expected_revision", ""))[:12],
    )
