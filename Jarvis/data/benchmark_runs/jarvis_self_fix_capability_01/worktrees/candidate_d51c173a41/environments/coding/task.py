from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodingTask:
    description: str
    workspace: Path
    test_command: list[str] = field(default_factory=lambda: [sys.executable, "-m", "pytest", "-q"])
    hidden_workspace: Path | None = None
    hidden_test_command: list[str] | None = None
    protected_paths: set[str] = field(default_factory=set)
    task_id: str = "coding-task"
    max_steps: int = 12
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        if self.hidden_workspace is not None:
            self.hidden_workspace = Path(self.hidden_workspace)
        self.protected_paths = {Path(path).as_posix() for path in self.protected_paths}

    @property
    def workspace_name(self) -> str:
        return self.workspace.name
