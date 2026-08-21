from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from capabilities.models import SkillSpecification
from development.memory import DevelopmentExperience, DevelopmentMemory, classify_failure
from environments.coding.sandbox_backend import SandboxBackend


class DevelopmentState(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    IMPLEMENT = "IMPLEMENT"
    TEST = "TEST"
    DIAGNOSE = "DIAGNOSE"
    REVISE = "REVISE"
    VERIFY = "VERIFY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class DevelopmentFile:
    path: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "content": self.content}


@dataclass
class SoftwareTestResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    failure_classes: list[str] = field(default_factory=list)

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectRequest:
    goal: str
    specification: SkillSpecification
    workspace: Path
    test_command: list[str]
    protected_paths: set[str] = field(default_factory=set)
    dependency_restrictions: list[str] = field(default_factory=lambda: ["Python standard library only."])
    permissions: list[str] = field(default_factory=list)
    max_repair_cycles: int = 4
    context_file_limit: int = 12
    context_char_limit: int = 24_000


@dataclass
class DevelopmentResult:
    success: bool
    final_state: DevelopmentState
    summary: str = ""
    plan: str = ""
    files: list[DevelopmentFile] = field(default_factory=list)
    public_test_result: SoftwareTestResult | None = None
    repair_cycles: int = 0
    llm_calls: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "final_state": self.final_state.value,
            "summary": self.summary,
            "plan": self.plan,
            "files": [item.to_dict() for item in self.files],
            "public_test_result": self.public_test_result.to_dict() if self.public_test_result else None,
            "repair_cycles": self.repair_cycles,
            "llm_calls": self.llm_calls,
            "token_usage": self.token_usage,
            "trajectory": self.trajectory,
            "failures": self.failures,
            "repairs": self.repairs,
            "error": self.error,
        }


def implementation_bundle_schema() -> dict[str, Any]:
    file_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "plan": {"type": "string"},
            "files": {"type": "array", "items": file_schema},
        },
        "required": ["summary", "files"],
    }


def repair_bundle_schema() -> dict[str, Any]:
    schema = implementation_bundle_schema()
    schema["properties"]["diagnosis"] = {"type": "string"}
    schema["required"] = ["diagnosis", "files"]
    return schema


class AutonomousSoftwareEngineer:
    """Deterministic orchestration for greenfield software creation."""

    def __init__(
        self,
        *,
        brain: Any,
        backend: SandboxBackend,
        memory: DevelopmentMemory,
        trace: bool = False,
    ) -> None:
        self.brain = brain
        self.backend = backend
        self.memory = memory
        self.trace = trace

    def build(self, request: ProjectRequest) -> DevelopmentResult:
        result = DevelopmentResult(False, DevelopmentState.UNDERSTAND)
        request.workspace.mkdir(parents=True, exist_ok=True)
        self._record(result, DevelopmentState.UNDERSTAND, {"goal": request.goal, "specification": request.specification.to_dict()})
        prior = self.memory.retrieve(request.goal, request.specification.to_dict(), limit=3)
        self._record(result, DevelopmentState.PLAN, {"retrieved_experience_count": len(prior)})

        bundle = self._request_implementation(request, prior, result)
        if bundle is None:
            result.final_state = DevelopmentState.FAILED
            result.error = "Implementation generation returned no valid files."
            self._persist_memory(request, result, hidden_success=None)
            return result
        result.summary = str(bundle.get("summary", ""))
        result.plan = str(bundle.get("plan", ""))
        initial_files = self._files_from_bundle(bundle)
        if not self._materialize(request, initial_files, result):
            self._persist_memory(request, result, hidden_success=None)
            return result
        result.files = self._current_files(request)

        test_result = self._run_tests(request)
        result.public_test_result = test_result
        result.failures.append(test_result.to_dict()) if not test_result.success else None
        self._record(result, DevelopmentState.TEST, test_result.to_dict())
        if test_result.success:
            result.success = True
            result.final_state = DevelopmentState.COMPLETE
            self._record(result, DevelopmentState.COMPLETE, {"public_tests": True})
            self._persist_memory(request, result, hidden_success=None)
            return result

        for cycle in range(1, request.max_repair_cycles + 1):
            result.repair_cycles = cycle
            self._record(result, DevelopmentState.DIAGNOSE, {"cycle": cycle, "failure_classes": test_result.failure_classes})
            repair = self._request_repair(request, test_result, result, prior)
            if repair is None:
                result.error = "Repair generation returned no valid files."
                break
            result.repairs.append({"cycle": cycle, "diagnosis": repair.get("diagnosis", ""), "files": repair.get("files", [])})
            repair_files = self._files_from_bundle(repair)
            self._record(result, DevelopmentState.REVISE, {"cycle": cycle, "diagnosis": repair.get("diagnosis", ""), "files": [item.path for item in repair_files]})
            if not self._materialize(request, repair_files, result):
                break
            result.files = self._current_files(request)
            test_result = self._run_tests(request)
            result.public_test_result = test_result
            if not test_result.success:
                result.failures.append(test_result.to_dict())
            self._record(result, DevelopmentState.TEST, {"cycle": cycle, **test_result.to_dict()})
            if test_result.success:
                result.success = True
                result.final_state = DevelopmentState.COMPLETE
                self._record(result, DevelopmentState.COMPLETE, {"public_tests": True, "repair_cycle": cycle})
                self._persist_memory(request, result, hidden_success=None)
                return result

        result.success = False
        result.final_state = DevelopmentState.FAILED
        result.error = result.error or "Repair budget exhausted before public tests passed."
        self._record(result, DevelopmentState.FAILED, {"error": result.error})
        self._persist_memory(request, result, hidden_success=None)
        return result

    def _request_implementation(
        self,
        request: ProjectRequest,
        prior: list[DevelopmentExperience],
        result: DevelopmentResult,
    ) -> dict[str, Any] | None:
        prompt = self._implementation_prompt(request, prior)
        raw = self._generate_structured(prompt, implementation_bundle_schema(), max_tokens=2200, temperature=0.2)
        result.llm_calls += 1
        self._accumulate_tokens(result)
        bundle = _parse_json_object(raw)
        self._record(result, DevelopmentState.IMPLEMENT, {"raw_response_hash": _stable_hash(raw), "valid": _valid_bundle(bundle)})
        return bundle if _valid_bundle(bundle) else None

    def _request_repair(
        self,
        request: ProjectRequest,
        failure: SoftwareTestResult,
        result: DevelopmentResult,
        prior: list[DevelopmentExperience],
    ) -> dict[str, Any] | None:
        prompt = self._repair_prompt(request, failure, result, prior)
        raw = self._generate_structured(prompt, repair_bundle_schema(), max_tokens=2600, temperature=0.2)
        result.llm_calls += 1
        self._accumulate_tokens(result)
        bundle = _parse_json_object(raw)
        self._record(result, DevelopmentState.REVISE, {"raw_response_hash": _stable_hash(raw), "valid": _valid_bundle(bundle)})
        return bundle if _valid_bundle(bundle) else None

    def _generate_structured(self, prompt: str, schema: dict[str, Any], *, max_tokens: int, temperature: float) -> str:
        if hasattr(self.brain, "generate_structured"):
            try:
                return self.brain.generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
            except (AttributeError, NotImplementedError, RuntimeError, ValueError, TypeError):
                pass
        if hasattr(self.brain, "generate_coding"):
            return self.brain.generate_coding(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
        if hasattr(self.brain, "generate"):
            return self.brain.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
        return self.brain.think(prompt, max_tokens=max_tokens)

    def _materialize(self, request: ProjectRequest, files: list[DevelopmentFile], result: DevelopmentResult) -> bool:
        self._record(result, DevelopmentState.IMPLEMENT, {"materializing_files": [item.path for item in files]})
        for item in files:
            try:
                path = self._safe_output_path(request, item.path)
            except ValueError as exc:
                result.final_state = DevelopmentState.FAILED
                result.error = str(exc)
                self._record(result, DevelopmentState.FAILED, {"error": str(exc)})
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item.content, encoding="utf-8")
        return True

    def _safe_output_path(self, request: ProjectRequest, relative_path: str) -> Path:
        raw = Path(relative_path)
        normalized = raw.as_posix()
        if not normalized or raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"Unsafe generated path: {relative_path}")
        if normalized in request.protected_paths or raw.name.startswith("test_"):
            raise ValueError(f"Generated implementation attempted to modify protected path: {relative_path}")
        candidate = (request.workspace / raw).resolve(strict=False)
        root = request.workspace.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Generated path escapes workspace: {relative_path}")
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"Generated path targets symlink: {relative_path}")
        return candidate

    def _run_tests(self, request: ProjectRequest) -> SoftwareTestResult:
        self._trace("[TEST] running public tests")
        completed = self.backend.run(
            request.test_command,
            cwd=request.workspace,
            timeout_seconds=20.0,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = f"{completed.stdout}\n{completed.stderr}"
        return SoftwareTestResult(
            success=completed.returncode == 0,
            stdout=completed.stdout[-6000:],
            stderr=completed.stderr[-6000:],
            return_code=completed.returncode,
            failure_classes=classify_failure(output),
        )

    def _implementation_prompt(self, request: ProjectRequest, prior: list[DevelopmentExperience]) -> str:
        return (
            "Return JSON only. You are an autonomous software engineer creating a complete local Python project.\n"
            "Generate a complete implementation bundle in one response. Use full-file contents.\n"
            "Do not modify public tests, hidden tests, skill_spec.json, or security/promotion files.\n"
            "The public API contract is: main.py must expose def run(payload: dict) -> dict.\n"
            f"Original goal: {request.goal}\n"
            f"Specification:\n{json.dumps(request.specification.to_dict(), indent=2, sort_keys=True)}\n"
            f"Dependency restrictions: {json.dumps(request.dependency_restrictions)}\n"
            f"Permissions: {json.dumps(request.permissions)}\n"
            f"Current project state:\n{self._project_context(request)}\n"
            f"Relevant prior successful development patterns:\n{self._memory_context(prior)}\n"
            "Response schema: {\"summary\":\"...\",\"plan\":\"...\",\"files\":[{\"path\":\"main.py\",\"content\":\"...\"}]}"
        )

    def _repair_prompt(
        self,
        request: ProjectRequest,
        failure: SoftwareTestResult,
        result: DevelopmentResult,
        prior: list[DevelopmentExperience],
    ) -> str:
        return (
            "Return JSON only. Repair the current project using full-file replacements for changed files.\n"
            "Do not modify public tests, hidden tests, skill_spec.json, or security/promotion files.\n"
            f"Original goal: {request.goal}\n"
            f"Specification:\n{json.dumps(request.specification.to_dict(), indent=2, sort_keys=True)}\n"
            f"Current project state:\n{self._project_context(request)}\n"
            f"Exact public test stdout:\n{failure.stdout}\n"
            f"Exact public test stderr:\n{failure.stderr}\n"
            f"Failure classes: {json.dumps(failure.failure_classes)}\n"
            f"Previous repair history:\n{json.dumps(result.repairs, indent=2, sort_keys=True)}\n"
            f"Relevant prior repair patterns:\n{self._memory_context(prior)}\n"
            "Response schema: {\"diagnosis\":\"...\",\"files\":[{\"path\":\"main.py\",\"content\":\"complete corrected file\"}]}"
        )

    def _project_context(self, request: ProjectRequest) -> str:
        files = []
        total_chars = 0
        for path in sorted(request.workspace.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(request.workspace).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            excerpt = text[:4000]
            if total_chars + len(excerpt) > request.context_char_limit:
                break
            files.append(f"{relative}:\n{excerpt}")
            total_chars += len(excerpt)
            if len(files) >= request.context_file_limit:
                break
        return "\n\n".join(files) if files else "(empty project)"

    @staticmethod
    def _memory_context(prior: list[DevelopmentExperience]) -> str:
        if not prior:
            return "[]"
        compact = [
            {
                "goal": exp.goal,
                "plan": exp.plan,
                "failure_classes": exp.failure_classes,
                "repair_cycles": exp.repair_cycles,
                "successful_files": [item.get("path", "") for item in exp.final_code[:5]],
            }
            for exp in prior
        ]
        return json.dumps(compact, indent=2, sort_keys=True)

    def _current_files(self, request: ProjectRequest) -> list[DevelopmentFile]:
        files = []
        for path in sorted(request.workspace.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(request.workspace).as_posix()
            if relative in request.protected_paths:
                continue
            try:
                files.append(DevelopmentFile(relative, path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                continue
        return files

    def _files_from_bundle(self, bundle: dict[str, Any]) -> list[DevelopmentFile]:
        return [
            DevelopmentFile(str(item.get("path", "")), str(item.get("content", "")))
            for item in list(bundle.get("files") or [])
            if isinstance(item, dict)
        ]

    def _persist_memory(self, request: ProjectRequest, result: DevelopmentResult, hidden_success: bool | None) -> None:
        fingerprint = self.memory.fingerprint(request.goal, request.specification.to_dict())
        failure_classes = []
        for failure in result.failures:
            failure_classes.extend(failure.get("failure_classes") or [])
        self.memory.record(
            DevelopmentExperience(
                fingerprint=fingerprint,
                goal=request.goal,
                spec=request.specification.to_dict(),
                plan=result.plan,
                implementation=[item.to_dict() for item in result.files],
                failures=result.failures,
                repairs=result.repairs,
                final_code=[item.to_dict() for item in result.files] if result.success else [],
                public_success=result.success,
                hidden_success=hidden_success,
                token_usage=result.token_usage,
                repair_cycles=result.repair_cycles,
                failure_classes=sorted(set(failure_classes)),
            )
        )

    def _accumulate_tokens(self, result: DevelopmentResult) -> None:
        metadata = getattr(self.brain, "last_metadata", {}) or {}
        for key in ["generated_tokens", "total_tokens"]:
            value = metadata.get(key)
            if value is not None:
                result.token_usage[key] = int(result.token_usage.get(key, 0)) + int(value)

    def _record(self, result: DevelopmentResult, state: DevelopmentState, payload: dict[str, Any]) -> None:
        result.final_state = state
        result.trajectory.append({"state": state.value, "payload": payload})
        self._trace(f"[{state.value}]")

    def _trace(self, message: str) -> None:
        if self.trace:
            print(message, flush=True)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _valid_bundle(bundle: dict[str, Any]) -> bool:
    files = bundle.get("files")
    return isinstance(files, list) and any(isinstance(item, dict) and item.get("path") and "content" in item for item in files)


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
