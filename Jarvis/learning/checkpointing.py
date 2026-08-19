from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def next_available_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    version = 1
    while True:
        candidate = parent / f"{stem}.v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def save_module_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    version: str,
    training_step: int,
    metrics: dict[str, float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    target = next_available_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": module.state_dict(),
        "version": version,
        "training_step": training_step,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, target)
    return target


def load_module_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    module.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
