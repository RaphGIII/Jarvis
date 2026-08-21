from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AcquisitionTrajectory:
    goal: str
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, stage: str, payload: dict[str, Any]) -> None:
        self.events.append(
            {
                "stage": stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {"trajectory_id": self.trajectory_id, "goal": self.goal, "events": self.events}


class AcquisitionTrajectoryStore:
    """Append-only JSONL store for capability-acquisition trajectories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trajectory: AcquisitionTrajectory, outcome: dict[str, Any]) -> None:
        record = {
            **trajectory.to_dict(),
            "outcome": outcome,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
