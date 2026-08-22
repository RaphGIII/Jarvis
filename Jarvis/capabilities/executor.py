from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capabilities.models import CapabilityManifest
from environments.coding.sandbox_backend import SandboxBackend


@dataclass
class CapabilityExecutionResult:
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    return_code: int | None = None


class CapabilityExecutor:
    """Executes installed capabilities in a fresh sandbox workspace."""

    def __init__(self, execution_root: str | Path, backend: SandboxBackend) -> None:
        self.execution_root = Path(execution_root)
        self.execution_root.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def execute(self, manifest: CapabilityManifest, payload: dict[str, Any]) -> CapabilityExecutionResult:
        source = Path(manifest.source_location)
        if not source.exists():
            return CapabilityExecutionResult(False, error="Installed capability source does not exist.")
        workspace = self.execution_root / f"{manifest.capability_id.replace('.', '_')}_{uuid.uuid4().hex[:10]}"
        shutil.copytree(source, workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (workspace / "request.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        (workspace / "runner.py").write_text(_runner_source(manifest.entrypoint), encoding="utf-8")
        completed = self.backend.run(
            ["python", "runner.py"],
            cwd=workspace,
            timeout_seconds=10.0,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        if completed.returncode != 0:
            return CapabilityExecutionResult(False, error=(completed.stderr or completed.stdout)[-2000:], return_code=completed.returncode)
        try:
            output = json.loads((workspace / "output.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return CapabilityExecutionResult(False, error=f"Capability did not produce valid output: {type(exc).__name__}", return_code=completed.returncode)
        return CapabilityExecutionResult(True, output=output, return_code=completed.returncode)


def _runner_source(entrypoint: str) -> str:
    module = Path(entrypoint).stem
    return f'''import importlib
import json
import pathlib

payload = json.loads(pathlib.Path("request.json").read_text(encoding="utf-8"))
module = importlib.import_module("{module}")
result = module.run(payload)
pathlib.Path("output.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
'''
