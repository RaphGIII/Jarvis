"""The known-good pointer and the deployment receipts.

A promotion is a claim until a fresh process has started from it and passed a
health check.  Only the supervisor can make that observation, so only the
supervisor writes the pointer.  The application reads it (to show which
revision is trusted) and never writes it.

Both files live under ``data/jarvis/supervisor``, which git ignores: a
``git reset`` back to the known-good revision must not be able to erase the
record of why it happened.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KnownGood:
    revision: str = ""
    verified_at: str = ""
    #: What the health check saw when this revision was accepted.
    health: dict[str, Any] = field(default_factory=dict)
    #: The previous known-good, so a rollback can go back two steps if the
    #: current one turns out to be bad later.
    previous: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentReceipt:
    """One attempt to run a revision, and what became of it."""

    kind: str  # start | restart | promotion | rollback | hold
    revision: str
    outcome: str  # healthy | unhealthy | rolled_back | rollback_failed | held | crashed
    reason: str = ""
    known_good_before: str = ""
    known_good_after: str = ""
    promotion_id: str = ""
    health: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnownGoodStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.pointer = self.state_dir / "known_good.json"
        self.receipts = self.state_dir / "deployments.jsonl"

    def load(self) -> KnownGood:
        if not self.pointer.is_file():
            return KnownGood()
        try:
            data = json.loads(self.pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return KnownGood()
        return KnownGood(
            revision=str(data.get("revision", "")),
            verified_at=str(data.get("verified_at", "")),
            health=dict(data.get("health") or {}),
            previous=str(data.get("previous", "")),
        )

    def mark(self, revision: str, health: dict[str, Any]) -> KnownGood:
        current = self.load()
        updated = KnownGood(
            revision=revision,
            verified_at=_now(),
            health=dict(health),
            previous=current.revision if current.revision != revision else current.previous,
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.pointer.with_suffix(".tmp")
        tmp.write_text(json.dumps(updated.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.pointer)
        return updated

    def record(self, receipt: DeploymentReceipt) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.receipts.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.receipts.is_file():
            return []
        lines = [line for line in self.receipts.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out
