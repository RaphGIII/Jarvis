from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from brain.json_utils import lenient_json_loads
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
    max_review_cycles: int = 2
    max_blind_repair_cycles: int = 2
    structured_regeneration_attempts: int = 2
    internal_tests_workspace: Path | None = None
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
    internal_test_result: SoftwareTestResult | None = None
    internal_test_suite: dict[str, Any] = field(default_factory=dict)
    contract: dict[str, Any] = field(default_factory=dict)
    internal_verification_success: bool = False
    reviewer_approved: bool = False
    review_findings: list[dict[str, Any]] = field(default_factory=list)
    review_cycles: int = 0
    repair_cycles: int = 0
    blind_repair_cycles: int = 0
    blind_hidden_repair_success: bool = False
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
            "internal_test_result": self.internal_test_result.to_dict() if self.internal_test_result else None,
            "internal_test_suite": self.internal_test_suite,
            "contract": self.contract,
            "internal_verification_success": self.internal_verification_success,
            "reviewer_approved": self.reviewer_approved,
            "review_findings": self.review_findings,
            "review_cycles": self.review_cycles,
            "repair_cycles": self.repair_cycles,
            "blind_repair_cycles": self.blind_repair_cycles,
            "blind_hidden_repair_success": self.blind_hidden_repair_success,
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


def review_schema() -> dict[str, Any]:
    test_schema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "name": {"type": "string"},
            "input": {"type": "object"},
            "expected": {"type": "object"},
            "raises": {"type": "boolean"},
        },
        "required": ["name", "input"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "approved": {"type": "boolean"},
            "contract_violations": {"type": "array", "items": {"type": "string"}},
            "risk_cases": {"type": "array", "items": {"type": "string"}},
            "recommended_tests": {"type": "array", "items": test_schema},
            "repair_required": {"type": "boolean"},
        },
        "required": ["approved", "contract_violations", "risk_cases", "recommended_tests", "repair_required"],
    }


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
        from development.qa import DeterministicReviewer, InternalTestEngineer, compile_contract

        result = DevelopmentResult(False, DevelopmentState.UNDERSTAND)
        request.workspace.mkdir(parents=True, exist_ok=True)
        contract = compile_contract(request.specification)
        result.contract = contract.to_dict()
        internal_tests_dir = request.internal_tests_workspace or request.workspace.parent / f"{request.workspace.name}_internal_qa"
        internal_suite = InternalTestEngineer(self.brain).create_suite(request.specification, contract, internal_tests_dir)
        result.internal_test_suite = internal_suite.to_dict()
        reviewer = DeterministicReviewer()
        self._record(
            result,
            DevelopmentState.UNDERSTAND,
            {
                "goal": request.goal,
                "specification": request.specification.to_dict(),
                "contract": result.contract,
                "internal_tests": result.internal_test_suite,
            },
        )
        prior = self.memory.retrieve(request.goal, request.specification.to_dict(), limit=3)
        self._record(result, DevelopmentState.PLAN, {"retrieved_experience_count": len(prior)})

        bundle = self._request_implementation(request, prior, result)
        if bundle is None:
            result.final_state = DevelopmentState.FAILED
            result.error = "Implementation generation returned no valid files."
            return result
        result.summary = str(bundle.get("summary", ""))
        result.plan = str(bundle.get("plan", ""))
        initial_files = self._files_from_bundle(bundle)
        if not self._materialize(request, initial_files, result):
            return result
        result.files = self._current_files(request)

        verification_ok = self._verify_project(request, result, internal_suite, reviewer, contract)
        if verification_ok:
            result.success = True
            result.final_state = DevelopmentState.COMPLETE
            self._record(
                result,
                DevelopmentState.COMPLETE,
                {
                    "public_tests": True,
                    "internal_verification": result.internal_verification_success,
                    "reviewer_approved": result.reviewer_approved,
                },
            )
            return result

        for cycle in range(1, request.max_repair_cycles + 1):
            result.repair_cycles = cycle
            failure_classes = self._latest_failure_classes(result)
            self._record(result, DevelopmentState.DIAGNOSE, {"cycle": cycle, "failure_classes": failure_classes})
            repair = self._request_repair(request, result, prior)
            if repair is None:
                result.error = "Repair generation returned no valid files."
                break
            result.repairs.append({"cycle": cycle, "diagnosis": repair.get("diagnosis", ""), "files": repair.get("files", [])})
            repair_files = self._files_from_bundle(repair)
            self._record(result, DevelopmentState.REVISE, {"cycle": cycle, "diagnosis": repair.get("diagnosis", ""), "files": [item.path for item in repair_files]})
            if not self._materialize(request, repair_files, result):
                break
            result.files = self._current_files(request)
            verification_ok = self._verify_project(request, result, internal_suite, reviewer, contract, cycle=cycle)
            if verification_ok:
                result.success = True
                result.final_state = DevelopmentState.COMPLETE
                self._record(
                    result,
                    DevelopmentState.COMPLETE,
                    {
                        "public_tests": True,
                        "internal_verification": result.internal_verification_success,
                        "reviewer_approved": result.reviewer_approved,
                        "repair_cycle": cycle,
                    },
                )
                return result

        result.success = False
        result.final_state = DevelopmentState.FAILED
        result.error = result.error or "Repair budget exhausted before public/internal/reviewer verification passed."
        self._record(result, DevelopmentState.FAILED, {"error": result.error})
        return result

    def _request_implementation(
        self,
        request: ProjectRequest,
        prior: list[DevelopmentExperience],
        result: DevelopmentResult,
    ) -> dict[str, Any] | None:
        prompt = self._implementation_prompt(request, prior)
        # The initial bundle must materialize every file the specification
        # requires (e.g. a mandated helper module), not just main.py -- a
        # weak model otherwise inlines everything into main.py and silently
        # violates a multi-file design requirement, which then only surfaces
        # much later (or not at all) at test/promotion time. main.py itself
        # must always be required too: after being told to add a missing
        # helper module, a weak model can overcorrect and return *only* the
        # helper file, dropping the entrypoint entirely.
        required_files = ["main.py"] + [
            path for path in request.specification.proposed_file_structure if path != "main.py"
        ]
        return self._request_valid_bundle(
            prompt,
            implementation_bundle_schema(),
            result,
            max_tokens=2200,
            temperature=0.2,
            state=DevelopmentState.IMPLEMENT,
            attempts=1 + max(0, request.structured_regeneration_attempts),
            required_files=required_files,
        )

    def _request_repair(
        self,
        request: ProjectRequest,
        result: DevelopmentResult,
        prior: list[DevelopmentExperience],
        *,
        blind_external_failure: bool = False,
    ) -> dict[str, Any] | None:
        prompt = self._repair_prompt(request, result, prior, blind_external_failure=blind_external_failure)
        return self._request_valid_bundle(
            prompt,
            repair_bundle_schema(),
            result,
            max_tokens=2800,
            temperature=0.2,
            state=DevelopmentState.REVISE,
            attempts=1 + max(0, request.structured_regeneration_attempts),
        )

    def _request_valid_bundle(
        self,
        prompt: str,
        schema: dict[str, Any],
        result: DevelopmentResult,
        *,
        max_tokens: int,
        temperature: float,
        state: DevelopmentState,
        attempts: int,
        required_files: list[str] | None = None,
    ) -> dict[str, Any] | None:
        last_raw = ""
        last_bundle: dict[str, Any] | None = None
        # A weak local model asked to add a missing required file can
        # overcorrect and return *only* that file on the next attempt,
        # forgetting files it correctly produced earlier (e.g. main.py on
        # attempt 1, aggregator.py on attempt 2, each individually valid but
        # never together). Accumulate real, non-placeholder files by path
        # across attempts so a later attempt's file doesn't erase an earlier
        # attempt's still-good file.
        accumulated_files: dict[str, dict[str, Any]] = {}
        for attempt in range(1, max(1, attempts) + 1):
            raw = self._generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature)
            last_raw = raw
            result.llm_calls += 1
            self._accumulate_tokens(result)
            bundle = _parse_json_object(raw)
            valid = _valid_bundle(bundle)
            if valid:
                last_bundle = bundle
                for item in bundle.get("files", []):
                    if isinstance(item, dict) and item.get("path") and not _is_placeholder_text(str(item.get("content", ""))):
                        accumulated_files[item["path"]] = item
            missing_files = _missing_required_files(bundle, required_files) if valid else list(required_files or [])
            merged_missing = [path for path in (required_files or []) if path not in accumulated_files]
            self._record(
                result,
                state,
                {
                    "raw_response_hash": _stable_hash(raw),
                    "valid": valid,
                    "attempt": attempt,
                    "missing_files": missing_files,
                    "merged_missing_files": merged_missing,
                },
            )
            if valid and not missing_files:
                return bundle
            if valid and not merged_missing and last_bundle is not None:
                merged = dict(last_bundle)
                merged["files"] = list(accumulated_files.values())
                return merged
            prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON for the required schema. "
                "Regenerate JSON only with complete files."
            )
            if missing_files:
                prompt += (
                    f" The response MUST include real, non-empty implementations for these required files: "
                    f"{', '.join(missing_files)}."
                )
        self._record(result, state, {"raw_response_hash": _stable_hash(last_raw), "valid": False, "regeneration_exhausted": True})
        return None

    def _verify_project(
        self,
        request: ProjectRequest,
        result: DevelopmentResult,
        internal_suite: Any,
        reviewer: Any,
        contract: Any,
        *,
        cycle: int | None = None,
    ) -> bool:
        from development.qa import run_internal_tests

        public_result = self._run_tests(request)
        result.public_test_result = public_result
        if not public_result.success:
            result.failures.append({"phase": "public", **public_result.to_dict()})
        self._record(result, DevelopmentState.TEST, {"phase": "public", "cycle": cycle, **public_result.to_dict()})
        if not public_result.success:
            result.internal_verification_success = False
            result.reviewer_approved = False
            return False

        internal_result = run_internal_tests(self.backend, request.workspace, internal_suite)
        result.internal_test_result = internal_result
        result.internal_verification_success = bool(internal_result.success)
        if not internal_result.success:
            result.failures.append({"phase": "internal", **internal_result.to_dict()})
        self._record(result, DevelopmentState.TEST, {"phase": "internal", "cycle": cycle, **internal_result.to_dict()})
        if not internal_result.success:
            result.reviewer_approved = False
            return False

        for review_cycle in range(1, request.max_review_cycles + 1):
            result.review_cycles += 1
            finding = self._request_review(
                request,
                result,
                reviewer,
                contract,
                internal_suite,
                public_result,
                internal_result,
            )
            result.review_findings.append(finding.to_dict())
            self._record(result, DevelopmentState.VERIFY, {"phase": "review", "cycle": review_cycle, **finding.to_dict()})
            if finding.recommended_tests:
                from development.qa import InternalTestEngineer

                internal_suite = InternalTestEngineer(self.brain).append_review_tests(internal_suite, finding.recommended_tests)
                result.internal_test_suite = internal_suite.to_dict()
                internal_result = run_internal_tests(self.backend, request.workspace, internal_suite)
                result.internal_test_result = internal_result
                result.internal_verification_success = bool(internal_result.success)
                if not internal_result.success:
                    result.failures.append({"phase": "review_recommended_internal", **internal_result.to_dict()})
                    self._record(
                        result,
                        DevelopmentState.TEST,
                        {"phase": "review_recommended_internal", "cycle": cycle, **internal_result.to_dict()},
                    )
                    result.reviewer_approved = False
                    return False
                continue
            result.reviewer_approved = bool(finding.approved and not finding.repair_required)
            return result.reviewer_approved

        result.reviewer_approved = False
        return False

    def _request_review(
        self,
        request: ProjectRequest,
        result: DevelopmentResult,
        reviewer: Any,
        contract: Any,
        internal_suite: Any,
        public_result: SoftwareTestResult,
        internal_result: SoftwareTestResult,
    ) -> Any:
        prompt = self._review_prompt(request, result, contract, internal_suite, public_result, internal_result)
        schema = review_schema()
        if hasattr(self.brain, "generate_structured"):
            try:
                raw = self.brain.generate_structured(prompt, schema, max_tokens=1600, temperature=0.1, top_p=0.9)
                result.llm_calls += 1
                self._accumulate_tokens(result)
                finding = _review_from_payload(_parse_json_object(raw))
                if finding is not None:
                    return finding
            except (AttributeError, NotImplementedError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return reviewer.review(
            spec=request.specification,
            contract=contract,
            implementation_root=request.workspace,
            public_result=public_result,
            internal_result=internal_result,
            internal_suite=internal_suite,
        )

    def blind_generalization_repair(self, request: ProjectRequest, result: DevelopmentResult) -> DevelopmentResult:
        from development.qa import DeterministicReviewer, InternalTestEngineer, compile_contract

        contract = compile_contract(request.specification)
        internal_tests_dir = request.internal_tests_workspace or request.workspace.parent / f"{request.workspace.name}_internal_qa"
        internal_suite = InternalTestEngineer(self.brain).create_suite(request.specification, contract, internal_tests_dir)
        reviewer = DeterministicReviewer()
        prior = self.memory.retrieve(request.goal, request.specification.to_dict(), limit=3)
        for cycle in range(1, request.max_blind_repair_cycles + 1):
            result.blind_repair_cycles = cycle
            self._record(
                result,
                DevelopmentState.DIAGNOSE,
                {
                    "phase": "blind_hidden_repair",
                    "cycle": cycle,
                    "message": "external acceptance verification failed",
                },
            )
            repair = self._request_repair(request, result, prior, blind_external_failure=True)
            if repair is None:
                result.error = "Blind repair generation returned no valid files."
                return result
            result.repairs.append({"cycle": result.repair_cycles + cycle, "blind": True, "diagnosis": repair.get("diagnosis", ""), "files": repair.get("files", [])})
            repair_files = self._files_from_bundle(repair)
            self._record(result, DevelopmentState.REVISE, {"phase": "blind_hidden_repair", "cycle": cycle, "files": [item.path for item in repair_files]})
            if not self._materialize(request, repair_files, result):
                return result
            result.files = self._current_files(request)
            if self._verify_project(request, result, internal_suite, reviewer, contract, cycle=result.repair_cycles + cycle):
                result.success = True
                result.final_state = DevelopmentState.COMPLETE
                result.blind_hidden_repair_success = True
                return result
        result.blind_hidden_repair_success = False
        return result

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
            "Return JSON only. Role: Implementer. You are implementing an Architect-approved local Python capability.\n"
            "Generate a complete implementation bundle in one response. Use full-file contents.\n"
            "Do not modify public tests, hidden tests, skill_spec.json, or security/promotion files.\n"
            "The public API contract is: main.py must expose def run(payload: dict) -> dict.\n"
            "The specification's public_tests define the exact payload keys `run` will receive and the exact "
            "keys/values it must return; implement run() to match those keys literally, even if they differ "
            "from words used in the goal text.\n"
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
        result: DevelopmentResult,
        prior: list[DevelopmentExperience],
        *,
        blind_external_failure: bool = False,
    ) -> str:
        blind_text = (
            "External acceptance verification failed. Hidden verifier code, inputs, outputs, and traceback are secret and unavailable.\n"
            "Search for unhandled contract edge cases using only the original goal, specification, public/internal tests, and reviewer findings.\n"
            if blind_external_failure
            else ""
        )
        return (
            "Return JSON only. Role: Repairer. Repair the current project using full-file replacements for changed files.\n"
            "Do not modify public tests, hidden tests, skill_spec.json, or security/promotion files.\n"
            "The specification's public_tests define the exact payload keys `run` will receive and the exact "
            "keys/values it must return; the failing tests below show the literal keys expected. Match them exactly.\n"
            f"{blind_text}"
            f"Original goal: {request.goal}\n"
            f"Specification:\n{json.dumps(request.specification.to_dict(), indent=2, sort_keys=True)}\n"
            f"Current project state:\n{self._project_context(request)}\n"
            "Exact public test and internal test results follow. Use them as objective failure evidence.\n"
            f"Verification failure summary:\n{json.dumps(self._verification_failure_summary(result), indent=2, sort_keys=True)}\n"
            f"Previous repair history:\n{json.dumps(result.repairs, indent=2, sort_keys=True)}\n"
            f"Relevant prior repair patterns:\n{self._memory_context(prior)}\n"
            "Response schema: {\"diagnosis\":\"...\",\"files\":[{\"path\":\"main.py\",\"content\":\"...\"}]}"
        )

    def _review_prompt(
        self,
        request: ProjectRequest,
        result: DevelopmentResult,
        contract: Any,
        internal_suite: Any,
        public_result: SoftwareTestResult,
        internal_result: SoftwareTestResult,
    ) -> str:
        return (
            "Return JSON only. Role: Reviewer. You cannot modify files. Independently review the implementation.\n"
            "If plausible behavior is missing, recommend additional black-box tests. Do not use or infer hidden verifier contents.\n"
            f"Original goal: {request.goal}\n"
            f"Executable contract:\n{json.dumps(contract.to_dict(), indent=2, sort_keys=True)}\n"
            f"Specification:\n{json.dumps(request.specification.to_dict(), indent=2, sort_keys=True)}\n"
            f"Implementation:\n{self._project_context(request)}\n"
            f"Public test result:\n{json.dumps(public_result.to_dict(), indent=2, sort_keys=True)}\n"
            f"Internal test suite:\n{json.dumps(internal_suite.to_dict(), indent=2, sort_keys=True)}\n"
            f"Internal test result:\n{json.dumps(internal_result.to_dict(), indent=2, sort_keys=True)}\n"
            "Response schema: {\"approved\":true,\"contract_violations\":[],\"risk_cases\":[],"
            "\"recommended_tests\":[{\"name\":\"...\",\"input\":{},\"expected\":{},\"raises\":false}],\"repair_required\":false}"
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

    def record_lifecycle_memory(
        self,
        request: ProjectRequest,
        result: DevelopmentResult,
        *,
        public_success: bool,
        internal_verification_success: bool,
        reviewer_approved: bool,
        hidden_success: bool | None,
        promotion_success: bool,
        execution_success: bool,
        second_call_success: bool,
    ) -> None:
        fingerprint = self.memory.fingerprint(request.goal, request.specification.to_dict())
        failure_classes = []
        for failure in result.failures:
            failure_classes.extend(failure.get("failure_classes") or [])
        final_success = bool(
            public_success
            and internal_verification_success
            and reviewer_approved
            and hidden_success
            and promotion_success
            and execution_success
            and second_call_success
        )
        self.memory.record(
            DevelopmentExperience(
                fingerprint=fingerprint,
                goal=request.goal,
                spec=request.specification.to_dict(),
                plan=result.plan,
                implementation=[item.to_dict() for item in result.files],
                failures=result.failures,
                repairs=result.repairs,
                final_code=[item.to_dict() for item in result.files] if final_success else [],
                public_success=public_success,
                internal_verification_success=internal_verification_success,
                reviewer_approved=reviewer_approved,
                hidden_success=hidden_success,
                promotion_success=promotion_success,
                execution_success=execution_success,
                second_call_success=second_call_success,
                token_usage=result.token_usage,
                repair_cycles=result.repair_cycles,
                blind_repair_cycles=result.blind_repair_cycles,
                failure_classes=sorted(set(failure_classes)),
            )
        )

    def _verification_failure_summary(self, result: DevelopmentResult) -> dict[str, Any]:
        return {
            "public": result.public_test_result.to_dict() if result.public_test_result else None,
            "internal": result.internal_test_result.to_dict() if result.internal_test_result else None,
            "internal_verification_success": result.internal_verification_success,
            "reviewer_approved": result.reviewer_approved,
            "review_findings": result.review_findings[-3:],
            "recent_failures": result.failures[-3:],
        }

    def _latest_failure_classes(self, result: DevelopmentResult) -> list[str]:
        classes: list[str] = []
        for failure in result.failures[-3:]:
            classes.extend(failure.get("failure_classes") or [])
        if result.review_findings and not result.reviewer_approved:
            classes.append("review_contract_risk")
        return sorted(set(classes))

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
        data = lenient_json_loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _valid_bundle(bundle: dict[str, Any]) -> bool:
    files = bundle.get("files")
    if not isinstance(files, list):
        return False
    real_files = [item for item in files if isinstance(item, dict) and item.get("path") and "content" in item]
    if not real_files:
        return False
    # A weak local model can literally return the "..." placeholder token from
    # the schema example as a file's actual `content` instead of generating
    # real code (the flip side of the earlier placeholder-echo bug: giving the
    # model an unambiguous placeholder stops it from parroting descriptive
    # phrases, but doesn't stop it from emitting the placeholder itself as if
    # it were a valid response). Any file whose content is just the
    # placeholder makes the whole bundle invalid so the caller retries/repairs
    # instead of writing a broken file.
    return all(not _is_placeholder_text(str(item.get("content", ""))) for item in real_files)


def _missing_required_files(bundle: dict[str, Any], required_files: list[str] | None) -> list[str]:
    """Files the specification mandates (e.g. a required helper module) that
    the bundle failed to provide with real, non-placeholder content."""
    if not required_files:
        return []
    present = {
        item.get("path")
        for item in bundle.get("files", [])
        if isinstance(item, dict) and item.get("path") and not _is_placeholder_text(str(item.get("content", "")))
    }
    return [path for path in required_files if path not in present]


_PLACEHOLDER_TOKENS = {"...", "…", "<...>", "n/a", "none", ""}


def _is_placeholder_text(value: str) -> bool:
    return value.strip().strip(".").lower() in _PLACEHOLDER_TOKENS or set(value.strip()) <= {".", "…"}


def _is_placeholder_payload(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    return all(isinstance(v, str) and _is_placeholder_text(v) for v in payload.values())


def _review_from_payload(payload: dict[str, Any]) -> Any | None:
    """Parse a brain-generated reviewer response into a ReviewFinding.

    Weak local models frequently echo the literal "..." placeholder tokens
    from the response-schema example back as real findings, and can produce
    self-contradictory output (approved=True but repair_required=True).
    Both are treated as malformed so the caller falls back to the
    deterministic AST-based reviewer instead of silently trusting
    fabricated/contradictory review output (fail closed).
    """
    if not isinstance(payload.get("approved"), bool):
        return None
    approved = bool(payload.get("approved", False))
    repair_required = bool(payload.get("repair_required", False))
    if approved and repair_required:
        return None

    from development.qa import InternalTestCase, ReviewFinding

    violations = [str(item) for item in payload.get("contract_violations", []) if not _is_placeholder_text(str(item))]
    risks = [str(item) for item in payload.get("risk_cases", []) if not _is_placeholder_text(str(item))]

    tests = []
    for item in payload.get("recommended_tests", []):
        if not isinstance(item, dict) or not isinstance(item.get("input"), dict):
            continue
        name = str(item.get("name") or "review_case")
        input_payload = dict(item.get("input") or {})
        if _is_placeholder_text(name) or _is_placeholder_payload(input_payload):
            continue
        expected = item.get("expected")
        tests.append(
            InternalTestCase(
                name=name,
                payload=input_payload,
                expected=expected if isinstance(expected, dict) else None,
                raises=bool(item.get("raises", False)),
            )
        )
    return ReviewFinding(
        approved=approved and not violations and not risks,
        contract_violations=violations,
        risk_cases=risks,
        recommended_tests=tests,
        repair_required=repair_required or bool(violations) or bool(risks),
    )


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
