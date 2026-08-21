from __future__ import annotations

import base64
import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BenchmarkProgressStore:
    """Small JSON progress file for long v0.3 benchmark runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = self.load()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "stage": "created",
                "baseline_completed": 0,
                "train_completed": 0,
                "validation_completed": 0,
                "final_holdout_completed": 0,
                "metrics": {},
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, **updates: Any) -> dict[str, Any]:
        self.state.update(updates)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["python_random_state"] = self.encode_random_state(random.getstate())
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)
        return self.state

    def metric(self, name: str, value: Any) -> None:
        metrics = dict(self.state.get("metrics") or {})
        metrics[name] = value
        self.save(metrics=metrics)

    @staticmethod
    def encode_random_state(state: object) -> str:
        return base64.b64encode(pickle.dumps(state)).decode("ascii")

    @staticmethod
    def decode_random_state(payload: str) -> object:
        return pickle.loads(base64.b64decode(payload.encode("ascii")))
