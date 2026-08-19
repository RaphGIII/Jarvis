from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from learning.checkpointing import load_module_checkpoint, save_module_checkpoint


class RuntimeCheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

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
