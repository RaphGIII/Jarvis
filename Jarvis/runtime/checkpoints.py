from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch import nn

from learning.checkpointing import load_module_checkpoint, save_module_checkpoint


class RuntimeCheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "latest").mkdir(exist_ok=True)
        (self.directory / "best").mkdir(exist_ok=True)

    def save_module(
        self,
        name: str,
        module: nn.Module,
        *,
        version: str,
        training_step: int,
        metrics: dict[str, float] | None = None,
        optimizer=None,
    ) -> Path:
        return save_module_checkpoint(
            module,
            self.directory / f"{name}.pt",
            version=version,
            training_step=training_step,
            metrics=metrics,
            optimizer=optimizer,
        )

    def load_latest_module(self, name: str, module: nn.Module, optimizer=None) -> dict[str, Any] | None:
        candidates = sorted(self.directory.glob(f"{name}*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return load_module_checkpoint(module, candidates[0], optimizer=optimizer)

    def save_category_metadata(self, category: str, metrics: dict[str, float], extra: dict[str, Any] | None = None) -> Path:
        target_dir = self.directory / category
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "category": category,
            "metrics": metrics,
            "extra": extra or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = target_dir / "checkpoint_metadata.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def should_promote(self, candidate: dict[str, float], best: dict[str, float] | None) -> bool:
        if best is None:
            return True
        candidate_regression = candidate.get("regression_rate", 1.0)
        best_regression = best.get("regression_rate", 1.0)
        if candidate_regression > best_regression + 1e-9:
            return False
        candidate_success = candidate.get("success_rate", 0.0)
        best_success = best.get("success_rate", 0.0)
        if candidate_success > best_success + 1e-9:
            return True
        if candidate_success < best_success - 1e-9:
            return False
        if candidate_regression < best_regression - 1e-9:
            return True
        candidate_steps = candidate.get("mean_steps_to_solution", float("inf"))
        best_steps = best.get("mean_steps_to_solution", float("inf"))
        if candidate_steps < best_steps - 1e-9:
            return True
        if candidate_steps > best_steps + 1e-9:
            return False
        if candidate.get("mean_reward", float("-inf")) > best.get("mean_reward", float("-inf")):
            return True
        return False

    def best_metrics(self) -> dict[str, float] | None:
        path = self.directory / "best" / "checkpoint_metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("metrics")
