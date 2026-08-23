from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from development.repository_engineer import (
    ModelRequestBudget,
    RepositoryCandidateResult,
    RepositoryEngineer,
    SelfDeveloperCheckpoint,
    SelfImprovementGoal,
    SelfImprovementMemory,
)


def _json_string_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)

    value = json.loads(raw)

    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{name} must be a JSON array of strings")

    return value


def _json_command(name: str) -> list[str] | None:
    raw = os.getenv(name)

    if not raw:
        return None

    value = json.loads(raw)

    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(
            f"{name} must be a JSON command array, "
            'for example ["python", "-m", "pytest", "-q"]'
        )

    return value


@dataclass
class BuildExecutionResult:
    preflight: dict[str, Any]
    candidate: RepositoryCandidateResult
    benchmark_root: str

    @property
    def success(self) -> bool:
        return bool(self.candidate.success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "preflight": self.preflight,
            "benchmark_root": self.benchmark_root,
            "candidate": self.candidate.to_dict(),
        }


class BuildExecutor:
    """
    Thin interactive adapter around the existing RepositoryEngineer.

    It deliberately contains no planning, patch generation, review or
    promotion logic of its own. Those responsibilities remain inside
    RepositoryEngineer.

    The brain it is handed is whatever tier the caller selected; by default
    that is BUILD_LOCAL, so a repository change needs no remote compute. The
    class was called BuildRemoteExecutor when remote inference was the only
    way to do this work; the old name remains as an alias.
    """

    def __init__(
        self,
        *,
        repository_path: str | Path,
        brain: Any,
        run_root: str | Path | None = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        self.brain = brain

        base = (
            Path(run_root).resolve()
            if run_root is not None
            else Path(tempfile.gettempdir()) / "jarvis_build_runs"
        )

        self.run_root = base
        self.run_root.mkdir(parents=True, exist_ok=True)

    def execute(self, objective: str) -> BuildExecutionResult:
        objective = objective.strip()

        if not objective:
            raise ValueError("BUILD_REMOTE objective must not be empty")

        if not self.repository_path.exists():
            raise ValueError(
                f"Repository does not exist: {self.repository_path}"
            )

        run_id = uuid.uuid4().hex[:12]
        benchmark_root = self.run_root / run_id
        benchmark_root.mkdir(parents=True, exist_ok=True)

        worktree_root = (
            Path(tempfile.gettempdir())
            / "jarvis_selfdev"
            / run_id
        )

        targeted = _json_command(
            "JARVIS_BUILD_TARGETED_TEST_COMMAND"
        )

        full = _json_command(
            "JARVIS_BUILD_FULL_TEST_COMMAND"
        )

        # Deterministic evidence is required by default.
        # This can explicitly be disabled for a controlled experiment
        # by setting JARVIS_BUILD_FULL_TEST_COMMAND=[].
        if full is None:
            full = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
            ]

        targeted_commands = [targeted] if targeted else []
        full_commands = [full] if full else []

        allowed_paths = _json_string_list(
            "JARVIS_BUILD_ALLOWED_PATHS",
            ["."],
        )

        protected_paths = _json_string_list(
            "JARVIS_BUILD_PROTECTED_PATHS",
            [],
        )

        goal = SelfImprovementGoal(
            objective=objective,
            success_criteria=[
                "Implement the requested repository change.",
                "Preserve unrelated existing behavior.",
                "Do not modify protected paths.",
                "Produce a reviewable isolated candidate.",
            ],
            allowed_paths=allowed_paths,
            protected_paths=protected_paths,
            tests=targeted_commands,
            full_tests=full_commands,
        )

        context_window_raw = os.getenv(
            "JARVIS_BUILD_CONTEXT_WINDOW"
        )

        context_window = (
            int(context_window_raw)
            if context_window_raw
            else None
        )

        max_cycles = int(
            os.getenv("JARVIS_BUILD_MAX_CYCLES", "3")
        )

        timeout_seconds = float(
            os.getenv("JARVIS_BUILD_COMMAND_TIMEOUT", "120")
        )

        checkpoint = SelfDeveloperCheckpoint(
            benchmark_root
        )

        engineer = RepositoryEngineer(
            brain=self.brain,
            worktree_root=worktree_root,
            memory=SelfImprovementMemory(
                benchmark_root
                / "self_development_trajectories.jsonl"
            ),
            timeout_seconds=timeout_seconds,
            max_cycles=max_cycles,
            context_budget=ModelRequestBudget.from_env(
                context_window
            ),
            checkpoint=checkpoint,
            resume_command="",
        )

        preflight = engineer.preflight()

        candidate = engineer.improve(
            self.repository_path,
            goal,
            targeted_commands,
            full_test_commands=full_commands,
            benchmark_commands=[],
            max_cycles=max_cycles,
        )

        return BuildExecutionResult(
            preflight=preflight,
            candidate=candidate,
            benchmark_root=str(benchmark_root),
        )


#: The name this class had when the heavy tier was necessarily remote.
BuildRemoteExecutor = BuildExecutor
