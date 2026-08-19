from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ActionResult:
    ok: bool
    message: str
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentStep:
    observation: Any
    action_result: ActionResult
    success: bool
    done: bool
    objective_metrics: dict[str, float | int | str | bool]


class Environment(Protocol):
    def observe(self) -> Any:
        ...

    def step(self, action: Any) -> EnvironmentStep:
        ...
