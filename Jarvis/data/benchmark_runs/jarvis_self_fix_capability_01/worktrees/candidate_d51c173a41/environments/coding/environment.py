from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from environments.base import ActionResult, EnvironmentStep
from environments.coding.actions import ActionCandidate, ActionType, coerce_action
from environments.coding.observation import CodingObservation
from environments.coding.sandbox_backend import DisabledSandboxBackend, SandboxBackend
from environments.coding.task import CodingTask


class SandboxViolation(ValueError):
    pass


class CodingEnvironment:
    """A controlled local coding environment scoped to one sandbox workspace."""

    def __init__(
        self,
        task: CodingTask,
        timeout_seconds: float = 5.0,
        backend: SandboxBackend | None = None,
        *,
        run_hidden_during_tests: bool = False,
        terminate_on_public_success: bool = True,
    ) -> None:
        self.task = task
        self.workspace = task.workspace.resolve()
        self.timeout_seconds = timeout_seconds
        self.backend = backend or DisabledSandboxBackend()
        self.run_hidden_during_tests = run_hidden_during_tests
        self.terminate_on_public_success = terminate_on_public_success
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.step_number = 0
        self.latest_action: dict[str, Any] | None = None
        self.latest_result = ""
        self.latest_error = ""
        self.open_files: dict[str, str] = {}
        self.done = False
        self.success = False
        self.test_state: dict[str, Any] = {
            "ran": False,
            "passed": 0,
            "failed": 0,
            "total": 0,
            "last_return_code": None,
        }
        self.hidden_state: dict[str, Any] = {
            "ran": False,
            "passed": 0,
            "failed": 0,
            "runs": 0,
            "last_return_code": None,
        }

    def observe(self) -> CodingObservation:
        return CodingObservation(
            task_description=self.task.description,
            workspace_tree=self._workspace_tree(),
            relevant_file_excerpts=dict(self.open_files),
            latest_action=self.latest_action,
            latest_action_result=self.latest_result,
            test_state=dict(self.test_state),
            error_output=self.latest_error,
            step_number=self.step_number,
            remaining_budget=max(0, self.task.max_steps - self.step_number),
        )

    def step(self, action: ActionCandidate | dict[str, Any]) -> EnvironmentStep:
        if self.done:
            result = ActionResult(False, "Environment is already done.")
            return EnvironmentStep(self.observe(), result, self.success, True, self._metrics(invalid_action=True))

        candidate = coerce_action(action)
        self.step_number += 1
        self.latest_action = candidate.to_dict()

        try:
            result = self._execute(candidate)
        except SandboxViolation as exc:
            result = ActionResult(False, str(exc), stderr=str(exc), data={"invalid_action": True})
        except Exception as exc:
            result = ActionResult(False, f"{type(exc).__name__}: {exc}", stderr=str(exc), data={"invalid_action": True})

        self.latest_result = result.message
        self.latest_error = result.stderr or (result.stdout if not result.ok else "")
        self.success = self._tests_passed()
        max_steps_hit = self.step_number >= self.task.max_steps
        success_terminal = self.success and self.terminate_on_public_success
        self.done = bool(success_terminal or max_steps_hit or candidate.action_type == ActionType.FINISH)
        invalid_action = bool(result.data.get("invalid_action", False)) if result.data else False
        return EnvironmentStep(
            observation=self.observe(),
            action_result=result,
            success=self.success,
            done=self.done,
            objective_metrics=self._metrics(invalid_action=invalid_action, max_steps_hit=max_steps_hit),
        )

    def _execute(self, candidate: ActionCandidate) -> ActionResult:
        action_type = candidate.action_type
        arguments = candidate.arguments
        if action_type == ActionType.LIST_FILES:
            tree = "\n".join(self._workspace_tree())
            return ActionResult(True, "Listed files.", stdout=tree, data={"files": self._workspace_tree()})
        if action_type == ActionType.READ_FILE:
            relative_path = str(arguments.get("path", ""))
            path = self._safe_path(relative_path, must_exist=True)
            text = path.read_text(encoding="utf-8")
            self.open_files[relative_path] = self._excerpt(text)
            return ActionResult(True, f"Read {relative_path}.", stdout=self.open_files[relative_path], data={"path": relative_path})
        if action_type == ActionType.SEARCH_TEXT:
            query = str(arguments.get("query", ""))
            if not query:
                return ActionResult(False, "SEARCH_TEXT requires query.", data={"invalid_action": True})
            matches = self._search_text(query)
            return ActionResult(True, f"Found {len(matches)} matches.", stdout="\n".join(matches), data={"matches": matches})
        if action_type == ActionType.WRITE_FILE:
            relative_path = str(arguments.get("path", ""))
            self._assert_editable(relative_path)
            content = str(arguments.get("content", ""))
            path = self._safe_path(relative_path, allow_create=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self.open_files[relative_path] = self._excerpt(content)
            return ActionResult(True, f"Wrote {relative_path}.", data={"path": relative_path})
        if action_type == ActionType.PATCH_FILE:
            return self._patch_file(arguments)
        if action_type == ActionType.RUN_TESTS:
            return self._run_tests()
        if action_type == ActionType.RUN_PYTHON:
            script = str(arguments.get("path", ""))
            path = self._safe_path(script, must_exist=True)
            return self._run_process([sys.executable, script], label=f"Ran Python {script}.")
        if action_type == ActionType.INSPECT_ERROR:
            return ActionResult(True, "Inspected latest error.", stdout=self.latest_error, data={"error": self.latest_error})
        if action_type == ActionType.FINISH:
            ok = self._tests_passed()
            return ActionResult(ok, "Finished successfully." if ok else "Finished before tests passed.")
        return ActionResult(False, f"Unsupported action: {action_type}", data={"invalid_action": True})

    def _patch_file(self, arguments: dict[str, Any]) -> ActionResult:
        relative_path = str(arguments.get("path", ""))
        self._assert_editable(relative_path)
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        if not old:
            return ActionResult(False, "PATCH_FILE requires old text.", data={"invalid_action": True})
        path = self._safe_path(relative_path, must_exist=True)
        text = path.read_text(encoding="utf-8")
        if old not in text:
            return ActionResult(False, "Patch text not found.", stderr="Patch text not found.", data={"invalid_action": True})
        patched = text.replace(old, new, 1)
        path.write_text(patched, encoding="utf-8")
        self.open_files[relative_path] = self._excerpt(patched)
        return ActionResult(True, f"Patched {relative_path}.", data={"path": relative_path})

    def _run_tests(self) -> ActionResult:
        result = self._run_process(self.task.test_command, label="Ran tests.")
        self.test_state = self._parse_test_state(result.stdout, result.stderr, result.return_code)
        if self.run_hidden_during_tests and self._public_tests_passed() and self.task.hidden_test_command is not None:
            hidden = self.run_final_hidden_verifier()
            hidden_ok = bool(hidden["success"])
            if not hidden_ok:
                return ActionResult(
                    False,
                    "Private verifier did not pass.",
                    stderr="",
                    return_code=int(hidden.get("return_code") or 1),
                    data={"private_verifier_failed": True},
                )
        return result

    def run_final_hidden_verifier(self) -> dict[str, Any]:
        """Run private verifier once for external evaluation without exposing output."""

        if self.task.hidden_test_command is None:
            return {"ran": False, "success": self._public_tests_passed(), "return_code": None, "runs": self.hidden_state["runs"]}
        if self.hidden_state["ran"]:
            return {
                "ran": True,
                "success": bool(self.hidden_state["passed"]),
                "return_code": self.hidden_state["last_return_code"],
                "runs": self.hidden_state["runs"],
            }
        hidden = self._run_process(
            self.task.hidden_test_command,
            label="Ran final verifier.",
            cwd=self.workspace,
            verifier_workspace=self.task.hidden_workspace,
            expose_output=False,
        )
        hidden_ok = hidden.return_code == 0
        self.hidden_state.update(
            {
                "ran": True,
                "passed": int(hidden_ok),
                "failed": int(not hidden_ok),
                "runs": int(self.hidden_state["runs"]) + 1,
                "last_return_code": hidden.return_code,
            }
        )
        return {"ran": True, "success": hidden_ok, "return_code": hidden.return_code, "runs": self.hidden_state["runs"]}

    def _run_process(
        self,
        command: list[str],
        label: str,
        cwd: Path | None = None,
        verifier_workspace: Path | None = None,
        expose_output: bool = True,
    ) -> ActionResult:
        completed = self.backend.run(
            command,
            cwd=cwd or self.workspace,
            timeout_seconds=self.timeout_seconds,
            env=self._subprocess_env(),
            verifier_workspace=verifier_workspace,
        )
        ok = completed.returncode == 0
        failure_kind = self._classify_process_failure(completed.stdout, completed.stderr, completed.returncode)
        return ActionResult(
            ok,
            label if ok else f"{label} Return code {completed.returncode}.",
            stdout=completed.stdout[-4000:] if expose_output else "",
            stderr=completed.stderr[-4000:] if expose_output else ("Hidden verifier failed." if not ok else ""),
            return_code=completed.returncode,
            data={"failure_kind": failure_kind} if not ok else {},
        )

    @staticmethod
    def _classify_process_failure(stdout: str, stderr: str, return_code: int | None) -> str:
        output = f"{stdout}\n{stderr}".lower()
        infrastructure_markers = [
            "docker sandbox unavailable",
            "cannot connect to the docker daemon",
            "error response from daemon",
            "container",
            "image pull",
            "network timed out",
            "timed out",
            "temporary failure",
            "resource temporarily unavailable",
            "no such file or directory: 'docker'",
        ]
        if return_code in {125, 126, 127} or any(marker in output for marker in infrastructure_markers):
            return "infrastructure"
        return "test_failure"

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return env

    def _assert_editable(self, relative_path: str) -> None:
        normalized = Path(relative_path).as_posix()
        name = Path(relative_path).name
        if normalized in self.task.protected_paths or name.startswith("test_"):
            raise SandboxViolation("Protected test/evaluator files cannot be modified.")

    def _safe_path(
        self,
        relative_path: str,
        *,
        must_exist: bool = False,
        allow_create: bool = False,
    ) -> Path:
        if not relative_path:
            raise SandboxViolation("Path argument is required.")
        raw_path = Path(relative_path)
        if raw_path.is_absolute():
            raise SandboxViolation("Absolute paths are not allowed in CodingWorld.")
        if ".." in raw_path.parts:
            raise SandboxViolation("Parent path traversal is not allowed in CodingWorld.")
        candidate = (self.workspace / raw_path).resolve(strict=False)
        if not candidate.is_relative_to(self.workspace):
            raise SandboxViolation("Path escapes CodingWorld workspace.")
        if must_exist and not candidate.exists():
            raise SandboxViolation(f"Path does not exist: {relative_path}")
        if candidate.exists() and candidate.is_symlink():
            raise SandboxViolation("Symlink access is not allowed in CodingWorld.")
        if allow_create and not candidate.parent.resolve(strict=False).is_relative_to(self.workspace):
            raise SandboxViolation("Create path escapes CodingWorld workspace.")
        return candidate

    def _workspace_tree(self) -> list[str]:
        paths: list[str] = []
        for path in sorted(self.workspace.rglob("*")):
            if path.is_symlink() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(self.workspace).as_posix()
            if path.is_file():
                paths.append(relative)
        return paths[:80]

    def _search_text(self, query: str) -> list[str]:
        matches: list[str] = []
        for relative in self._workspace_tree():
            path = self._safe_path(relative, must_exist=True)
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(f"{relative}:{line_number}: {line[:200]}")
        return matches[:50]

    def _parse_test_state(self, stdout: str, stderr: str, return_code: int | None) -> dict[str, Any]:
        output = f"{stdout}\n{stderr}".lower()
        passed = self._first_int(r"(\d+)\s+passed", output)
        failed = self._first_int(r"(\d+)\s+failed", output)
        errors = self._first_int(r"(\d+)\s+errors?", output)
        if passed == 0 and failed == 0 and errors == 0:
            if return_code == 0:
                passed = 1
            else:
                failed = 1
        total = passed + failed + errors
        return {
            "ran": True,
            "passed": passed,
            "failed": failed + errors,
            "total": max(total, 1),
            "last_return_code": return_code,
        }

    @staticmethod
    def _first_int(pattern: str, text: str) -> int:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _excerpt(text: str, limit: int = 2000) -> str:
        return text[:limit]

    def _tests_passed(self) -> bool:
        if self.run_hidden_during_tests and self.task.hidden_test_command is not None and self.hidden_state["ran"]:
            return bool(self.hidden_state["passed"])
        return self._public_tests_passed()

    def _public_tests_passed(self) -> bool:
        return bool(self.test_state.get("ran", False) and self.test_state.get("passed", 0) > 0 and self.test_state.get("failed", 0) == 0)

    def _metrics(self, invalid_action: bool = False, max_steps_hit: bool = False) -> dict[str, float | int | str | bool]:
        return {
            "tests_passed": int(self.test_state.get("passed", 0)),
            "tests_failed": int(self.test_state.get("failed", 0)),
            "tests_total": int(self.test_state.get("total", 0)),
            "invalid_action": bool(invalid_action),
            "max_steps_hit": bool(max_steps_hit),
            "step_number": self.step_number,
            "success": self.success,
        }
