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
    task_id: str = "coding-task"
    max_steps: int = 12
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)

    @property
    def workspace_name(self) -> str:
        return self.workspace.name
