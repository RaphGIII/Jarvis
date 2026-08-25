"""Long missions, written down as they go, so a restart is not a restart.

A capability acquisition runs for one to two hours.  Until now, anything that
interrupted it -- a crash, a reboot, an expert quota running out, ZEUS being
restarted to pick up a fix -- threw the whole thing away, and the next attempt
began by re-deriving evidence that had already been paid for.  That is not
merely slow; it is actively misleading, because the performance ledger then
records a fresh set of local failures for work the model was never asked to
redo, and the escalation controller reasons from those counts.

So a mission records what it has established.  The unit is *evidence*, not
progress: which attempts ran, what they concluded, which hypotheses are still
open, and what the next action is.  Resuming means not repeating an attempt
whose answer is already known -- it never means assuming an unfinished attempt
succeeded.

Deliberately conservative about what may be resumed.  A checkpoint is only
honoured when the goal, the capability and the defect all match, and when it is
recent enough that the machine is probably still the machine it describes.
Anything else starts clean, because a wrong resume is worse than a slow start:
it would skip work on the strength of evidence gathered about a different
question.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: After this, a checkpoint describes a machine that may no longer exist --
#: models pulled, packages installed, the capability edited by something else.
#: Six hours is long enough to survive a night and short enough that the world
#: has probably not moved underneath it.
MAX_AGE_SECONDS = 6 * 60 * 60


@dataclass
class Attempt:
    """One thing tried, and what it established."""

    tier: str
    hypothesis: str = ""
    evidence: str = ""
    succeeded: bool = False
    seconds: float = 0.0
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "hypothesis": self.hypothesis[:400],
            "evidence": self.evidence[:600],
            "succeeded": self.succeeded,
            "seconds": round(self.seconds, 1),
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attempt":
        return cls(
            tier=str(data.get("tier", "")),
            hypothesis=str(data.get("hypothesis", "")),
            evidence=str(data.get("evidence", "")),
            succeeded=bool(data.get("succeeded", False)),
            seconds=float(data.get("seconds", 0.0) or 0.0),
            at=str(data.get("at", "")),
        )


@dataclass
class MissionCheckpoint:
    """Everything a resumed mission is allowed to take on trust."""

    capability_id: str
    goal_fingerprint: str
    defect: str = ""
    phase: str = "start"
    attempts: list[Attempt] = field(default_factory=list)
    escalated: bool = False
    acquired: bool = False
    open_hypotheses: list[str] = field(default_factory=list)
    next_action: str = ""
    mission_id: str = field(default_factory=lambda: f"msn_{uuid.uuid4().hex[:10]}")
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def local_attempts(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.tier == "build_local")

    @property
    def age_seconds(self) -> float:
        return time.time() - self.updated_at

    @property
    def stale(self) -> bool:
        return self.age_seconds > MAX_AGE_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "capability_id": self.capability_id,
            "goal_fingerprint": self.goal_fingerprint,
            "defect": self.defect[:600],
            "phase": self.phase,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "escalated": self.escalated,
            "acquired": self.acquired,
            "open_hypotheses": self.open_hypotheses[:12],
            "next_action": self.next_action[:400],
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "local_attempts": self.local_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionCheckpoint":
        return cls(
            capability_id=str(data.get("capability_id", "")),
            goal_fingerprint=str(data.get("goal_fingerprint", "")),
            defect=str(data.get("defect", "")),
            phase=str(data.get("phase", "start")),
            attempts=[Attempt.from_dict(item) for item in (data.get("attempts") or [])],
            escalated=bool(data.get("escalated", False)),
            acquired=bool(data.get("acquired", False)),
            open_hypotheses=[str(item) for item in (data.get("open_hypotheses") or [])],
            next_action=str(data.get("next_action", "")),
            mission_id=str(data.get("mission_id") or f"msn_{uuid.uuid4().hex[:10]}"),
            started_at=float(data.get("started_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )

    def describe(self) -> str:
        head = f"resuming {self.mission_id}: {self.local_attempts} local attempt(s) already made"
        if self.escalated:
            head += ", already escalated once"
        return head


def fingerprint(goal: str) -> str:
    """Identify the *question*, not its wording.

    Whitespace and the recalled-lesson preamble change between runs without
    changing what is being asked, and a fingerprint that moved every time would
    make every checkpoint unresumable -- which looks exactly like the feature
    not working.
    """

    import hashlib
    import re

    core = re.sub(r"\s+", " ", (goal or "")).strip().lower()
    # A recalled lesson is prepended to the goal, so strip it back off before
    # deciding whether this is the same question as last time.
    marker = "repair an existing"
    if marker in core:
        core = core[core.index(marker):]
    return hashlib.sha256(core.encode("utf-8")).hexdigest()[:16]


class MissionStore:
    """Checkpoints on disk, one per capability."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def path_for(self, capability_id: str) -> Path:
        safe = (capability_id or "unknown").replace(".", "_").replace("/", "_")
        return self.root / f"{safe}.json"

    def save(self, checkpoint: MissionCheckpoint) -> Path:
        checkpoint.updated_at = time.time()
        path = self.path_for(checkpoint.capability_id)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(checkpoint.to_dict(), indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        return path

    def load(self, capability_id: str) -> MissionCheckpoint | None:
        path = self.path_for(capability_id)
        if not path.is_file():
            return None
        try:
            return MissionCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def resumable(self, capability_id: str, goal: str, defect: str = "") -> MissionCheckpoint | None:
        """A checkpoint that may be trusted for *this* question, or ``None``.

        Every condition here exists because resuming wrongly is worse than not
        resuming: it would skip work on the strength of evidence gathered about
        something else.
        """

        checkpoint = self.load(capability_id)
        if checkpoint is None:
            return None
        if checkpoint.acquired:
            return None
        if checkpoint.stale:
            return None
        if checkpoint.goal_fingerprint != fingerprint(goal):
            return None
        if (checkpoint.defect or "")[:200] != (defect or "")[:200]:
            return None
        return checkpoint

    def clear(self, capability_id: str) -> None:
        path = self.path_for(capability_id)
        try:
            path.unlink()
        except OSError:
            pass

    def all(self) -> list[MissionCheckpoint]:
        if not self.root.is_dir():
            return []
        found = []
        for path in sorted(self.root.glob("*.json")):
            try:
                found.append(MissionCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue
        return found
