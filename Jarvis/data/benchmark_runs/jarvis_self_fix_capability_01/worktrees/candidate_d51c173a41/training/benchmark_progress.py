from __future__ import annotations

import base64
import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


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
                "committed_rng": None,
                "live_rng": None,
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, *, runtime: Any | None = None, commit_rng: bool = False, capture_live_rng: bool = True, **updates: Any) -> dict[str, Any]:
        self.state.update(updates)
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        snapshot = self.rng_snapshot(runtime)
        if capture_live_rng:
            self.state["live_rng"] = snapshot
            self.state["python_random_state"] = snapshot["python"]
            self.state["torch_rng_state"] = snapshot["torch"]
            self.state["torch_cuda_rng_state"] = snapshot["torch_cuda"]
        if commit_rng:
            self.state["committed_rng"] = snapshot
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)
        return self.state

    def metric(self, name: str, value: Any) -> None:
        metrics = dict(self.state.get("metrics") or {})
        metrics[name] = value
        self.save(metrics=metrics)

    def rng_snapshot(self, runtime: Any | None = None) -> dict[str, str | None]:
        return {
            "python": self.encode_random_state(random.getstate()),
            "torch": self.encode_random_state(torch.get_rng_state()),
            "torch_cuda": self.encode_random_state(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
            "runtime": self.encode_random_state(runtime._rng.getstate()) if runtime is not None and hasattr(runtime, "_rng") else None,
            "replay": self.encode_random_state(runtime.replay_buffer._rng.getstate())
            if runtime is not None and hasattr(runtime, "replay_buffer")
            else None,
        }

    @classmethod
    def restore_rng_snapshot(cls, snapshot: dict[str, str | None] | None, runtime: Any | None = None) -> None:
        if not snapshot:
            return
        if snapshot.get("python"):
            random.setstate(cls.decode_random_state(snapshot["python"]))
        if snapshot.get("torch"):
            torch.set_rng_state(cls.decode_random_state(snapshot["torch"]))
        if snapshot.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cls.decode_random_state(snapshot["torch_cuda"]))
        if runtime is not None and snapshot.get("runtime"):
            runtime._rng.setstate(cls.decode_random_state(snapshot["runtime"]))
        if runtime is not None and snapshot.get("replay") and hasattr(runtime, "replay_buffer"):
            runtime.replay_buffer._rng.setstate(cls.decode_random_state(snapshot["replay"]))

    @staticmethod
    def encode_random_state(state: object) -> str:
        return base64.b64encode(pickle.dumps(state)).decode("ascii")

    @staticmethod
    def decode_random_state(payload: str) -> object:
        return pickle.loads(base64.b64decode(payload.encode("ascii")))
