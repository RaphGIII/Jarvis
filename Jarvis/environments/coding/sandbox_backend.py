from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SandboxPolicy:
    network_disabled: bool = True
    non_root: bool = True
    cpu_limit: float = 1.0
    memory_limit_mb: int = 512
    pid_limit: int = 128
    timeout_seconds: float = 5.0
    privileged: bool = False


class SandboxBackend(Protocol):
    policy: SandboxPolicy

    def run(self, command: list[str], cwd: Path, timeout_seconds: float, env: dict[str, str]) -> subprocess.CompletedProcess:
        ...


class DisabledSandboxBackend:
    """Safe default: do not execute generated code without an explicit sandbox."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run(self, command: list[str], cwd: Path, timeout_seconds: float, env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            command,
            returncode=126,
            stdout="",
            stderr="Unsafe code execution disabled: no secure sandbox backend configured.",
        )


class LocalTestSandboxBackend:
    """Explicit test/demo backend. Production code must inject Docker or remain disabled."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run(self, command: list[str], cwd: Path, timeout_seconds: float, env: dict[str, str]) -> subprocess.CompletedProcess:
        clean_env = {
            key: value
            for key, value in env.items()
            if not key.upper().startswith(("SECRET", "TOKEN", "API_KEY"))
        }
        clean_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=min(timeout_seconds, self.policy.timeout_seconds),
            shell=False,
            env=clean_env,
        )


class DockerSandboxBackend:
    """Container backend declaration for production deployment wiring."""

    def __init__(self, image: str = "python:3.12-slim", policy: SandboxPolicy | None = None) -> None:
        self.image = image
        self.policy = policy or SandboxPolicy()

    @staticmethod
    def is_available() -> bool:
        try:
            completed = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=2)
            return completed.returncode == 0
        except Exception:
            return False

    def run(self, command: list[str], cwd: Path, timeout_seconds: float, env: dict[str, str]) -> subprocess.CompletedProcess:
        if not self.is_available():
            return subprocess.CompletedProcess(
                command,
                returncode=126,
                stdout="",
                stderr="Docker sandbox unavailable; unsafe host fallback is disabled.",
            )
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            str(self.policy.cpu_limit),
            "--memory",
            f"{self.policy.memory_limit_mb}m",
            "--pids-limit",
            str(self.policy.pid_limit),
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{cwd.resolve()}:/workspace:rw",
            "-w",
            "/workspace",
            self.image,
            *command,
        ]
        return subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=min(timeout_seconds, self.policy.timeout_seconds),
            shell=False,
            env={key: value for key, value in os.environ.items() if key in {"PATH", "SYSTEMROOT", "COMSPEC"}},
        )
