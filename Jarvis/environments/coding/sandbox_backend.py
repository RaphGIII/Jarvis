from __future__ import annotations

import os
import subprocess
import sys
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

    def run(
        self,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        env: dict[str, str],
        verifier_workspace: Path | None = None,
    ) -> subprocess.CompletedProcess:
        ...


class DisabledSandboxBackend:
    """Safe default: do not execute generated code without an explicit sandbox."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run(
        self,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        env: dict[str, str],
        verifier_workspace: Path | None = None,
    ) -> subprocess.CompletedProcess:
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

    def run(
        self,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        env: dict[str, str],
        verifier_workspace: Path | None = None,
    ) -> subprocess.CompletedProcess:
        clean_env = {
            key: value
            for key, value in env.items()
            if not key.upper().startswith(("SECRET", "TOKEN", "API_KEY"))
        }
        clean_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        # Preserve PATH (and the venv marker) so the host Python interpreter
        # actually running this backend can be located by name (e.g. "python").
        # Without PATH, subprocess falls back to os.defpath, which does not
        # include venv bin directories and breaks command resolution.
        clean_env.setdefault("PATH", os.environ.get("PATH", os.defpath))
        if "VIRTUAL_ENV" in os.environ:
            clean_env.setdefault("VIRTUAL_ENV", os.environ["VIRTUAL_ENV"])
        clean_env["JARVIS_WORKSPACE"] = str(cwd.resolve())
        if verifier_workspace is not None:
            clean_env["PYTHONPATH"] = str(cwd.resolve())
        return subprocess.run(
            command,
            cwd=verifier_workspace or cwd,
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
            completed = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return completed.returncode == 0
        except Exception:
            return False

    def run(
        self,
        command: list[str],
        cwd: Path,
        timeout_seconds: float,
        env: dict[str, str],
        verifier_workspace: Path | None = None,
    ) -> subprocess.CompletedProcess:
        if not self.is_available():
            return subprocess.CompletedProcess(
                command,
                returncode=126,
                stdout="",
                stderr="Docker sandbox unavailable; unsafe host fallback is disabled.",
            )
        workspace = cwd.resolve()
        verifier = verifier_workspace.resolve() if verifier_workspace is not None else None
        normalized_command = self._normalize_command(command, workspace, verifier)
        container_cwd = "/verifier" if verifier is not None else "/workspace"
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
            "--cap-drop",
            "ALL",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONPATH=/workspace",
            "-e",
            "JARVIS_WORKSPACE=/workspace",
            "-v",
            f"{workspace}:/workspace:rw",
            "-w",
            container_cwd,
        ]
        if self.policy.non_root:
            docker_command.extend(["--user", "65532:65532"])
        if verifier is not None:
            docker_command.extend(["-v", f"{verifier}:/verifier:ro"])
        docker_command.extend([self.image, *normalized_command])
        return subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=min(timeout_seconds, self.policy.timeout_seconds),
            shell=False,
            env={key: value for key, value in os.environ.items() if key in {"PATH", "SYSTEMROOT", "COMSPEC"}},
        )

    def _normalize_command(self, command: list[str], workspace: Path, verifier: Path | None) -> list[str]:
        normalized: list[str] = []
        for index, token in enumerate(command):
            if index == 0 and self._is_host_python(token):
                normalized.append("python")
                continue
            normalized.append(self._containerize_path(token, workspace, verifier))
        return normalized

    @staticmethod
    def _is_host_python(token: str) -> bool:
        lowered = token.lower().replace("\\", "/")
        executable = str(Path(sys.executable)).lower().replace("\\", "/")
        return (
            lowered == executable
            or lowered.endswith("/python.exe")
            or lowered.endswith("/python")
            or lowered in {"python", "python.exe", "py", "py.exe"}
        )

    @staticmethod
    def _containerize_path(token: str, workspace: Path, verifier: Path | None) -> str:
        try:
            path = Path(token)
        except (OSError, ValueError):
            return token
        if not path.is_absolute():
            return token
        resolved = path.resolve(strict=False)
        if resolved == workspace:
            return "/workspace"
        if resolved.is_relative_to(workspace):
            return "/workspace/" + resolved.relative_to(workspace).as_posix()
        if verifier is not None:
            if resolved == verifier:
                return "/verifier"
            if resolved.is_relative_to(verifier):
                return "/verifier/" + resolved.relative_to(verifier).as_posix()
        return token
