from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
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

    def save_module_snapshot(
        self,
        category: str,
        name: str,
        module: nn.Module,
        *,
        version: str,
        training_step: int,
        metrics: dict[str, float] | None = None,
        optimizer=None,
    ) -> Path:
        target = self.directory / category / f"{name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_state": module.state_dict(),
            "version": version,
            "training_step": training_step,
            "metrics": metrics or {},
            "extra": {"category": category},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        temp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temp)
        os.replace(temp, target)
        return target

    def load_latest_module(self, name: str, module: nn.Module, optimizer=None) -> dict[str, Any] | None:
        payload = self.load_category_module("latest", name, module, optimizer=optimizer)
        if payload is not None:
            return payload
        candidates = sorted(self.directory.glob(f"{name}*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return load_module_checkpoint(module, candidates[0], optimizer=optimizer)

    def load_best_module(self, name: str, module: nn.Module, optimizer=None) -> dict[str, Any] | None:
        return self.load_category_module("best", name, module, optimizer=optimizer)

    def load_category_module(self, category: str, name: str, module: nn.Module, optimizer=None) -> dict[str, Any] | None:
        path = self.directory / category / f"{name}.pt"
        if not path.exists():
            return None
        return load_module_checkpoint(module, path, optimizer=optimizer)

    def save_category_metadata(self, category: str, metrics: dict[str, float], extra: dict[str, Any] | None = None) -> Path:
        target_dir = self.directory / category
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "category": category,
            "metrics": metrics,
            "extra": extra or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = target_dir / "metadata.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
        return path

    def should_promote(self, candidate: dict[str, float], best: dict[str, float] | None) -> bool:
        if str(candidate.get("split", "")).lower() == "holdout":
            return False
        if best is None:
            return True
        candidate_success = candidate.get("success_rate", 0.0)
        best_success = best.get("success_rate", 0.0)
        if candidate_success > best_success + 1e-9:
            return True
        if candidate_success < best_success - 1e-9:
            return False
        candidate_regression = candidate.get("regression_rate", 1.0)
        best_regression = best.get("regression_rate", 1.0)
        if candidate_regression < best_regression - 1e-9:
            return True
        if candidate_regression > best_regression + 1e-9:
            return False
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
        path = self.directory / "best" / "metadata.json"
        if not path.exists():
            path = self.directory / "best" / "checkpoint_metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("metrics")
