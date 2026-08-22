from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TrainingRunRecord:
    experiment_id: str
    model_version: str
    dataset_version: str
    hyperparameters: dict[str, Any]
    start_metrics: dict[str, float]
    end_metrics: dict[str, float]
    loss_curve: list[float]
    validation_metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExperimentTracker:
    """Local JSONL tracker; can be swapped for SQLite later."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: TrainingRunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def read_all(self) -> list[TrainingRunRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(TrainingRunRecord(**json.loads(line)))
        return records
