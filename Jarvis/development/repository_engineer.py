from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import subprocess
import uuid
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    from brain.providers import ProviderError, StructuredGenerationUnsupported
except Exception:  # pragma: no cover - keeps repository tooling importable in minimal test envs
    ProviderError = RuntimeError  # type: ignore
    StructuredGenerationUnsupported = NotImplementedError  # type: ignore


class RepositoryStage:
    PREFLIGHT_COMPLETE = "PREFLIGHT_COMPLETE"
    WORKTREE_CREATED = "WORKTREE_CREATED"
    BENCHMARK_BEFORE_COMPLETE = "BENCHMARK_BEFORE_COMPLETE"
    PLAN_COMPLETE = "PLAN_COMPLETE"
    PATCH_CYCLE = "PATCH_CYCLE"
    TARGETED_TESTS = "TARGETED_TESTS"
    REVIEW_STAGE = "REVIEW_STAGE"
    FULL_TESTS = "FULL_TESTS"
    BENCHMARK_AFTER = "BENCHMARK_AFTER"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    IMPLEMENT = "IMPLEMENT"
    TEST_TARGETED = "TEST_TARGETED"
    DIAGNOSE = "DIAGNOSE"
    REPAIR = "REPAIR"
    REVIEW = "REVIEW"
    TEST_FULL = "TEST_FULL"
    BENCHMARK = "BENCHMARK"
    EVALUATE = "EVALUATE"
    CANDIDATE_READY = "SELF_DEVELOPMENT_CANDIDATE_READY"
    REJECTED = "SELF_DEVELOPMENT_CANDIDATE_REJECTED"
    PAUSED = "SELF_DEVELOPMENT_PAUSED"


class ProtectionState:
    PRISTINE = "PRISTINE"
    MODIFIED = "MODIFIED"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class SelfImprovementGoal:
    objective: str
    success_criteria: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=lambda: ["."])
    protected_paths: list[str] = field(default_factory=list)
    tests: list[list[str]] = field(default_factory=list)
    full_tests: list[list[str]] = field(default_factory=list)
    benchmark: list[str] | None = None
    benchmarks: list[list[str]] = field(default_factory=list)
    require_benchmark_improvement: bool = False
    metric_name: str | None = None
    metric_minimums: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.benchmark and not data["benchmarks"]:
            data["benchmarks"] = [self.benchmark]
        return data

    def benchmark_commands(self) -> list[list[str]]:
        commands = list(self.benchmarks)
        if self.benchmark is not None:
            commands.append(list(self.benchmark))
        return commands


@dataclass
class RepositoryCommandResult:
    command: list[str]
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    cwd: str = ""
    executable: str = ""

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepositoryReview:
    approved: bool
    blocking_findings: list[str] = field(default_factory=list)
    optional_findings: list[str] = field(default_factory=list)
    recommended_tests: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepositoryCandidateResult:
    status: str
    worktree: str
    diff: str = ""
    tests: list[RepositoryCommandResult] = field(default_factory=list)
    full_tests: list[RepositoryCommandResult] = field(default_factory=list)
    benchmarks_before: list[RepositoryCommandResult] = field(default_factory=list)
    benchmarks_after: list[RepositoryCommandResult] = field(default_factory=list)
    review: RepositoryReview | None = None
    protected_pristine: bool = True
    protection_state: str = ProtectionState.NOT_EVALUATED
    changed_files: list[str] = field(default_factory=list)
    diff_path: str = ""
    result_path: str = ""
    rationale: str = ""
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cycles: int = 0
    error: str = ""
    failure_kind: str = ""
    resume_command: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"SELF_DEVELOPMENT_CANDIDATE_READY", "SELF_IMPROVEMENT_CANDIDATE_READY"}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tests"] = [item.to_dict() for item in self.tests]
        data["full_tests"] = [item.to_dict() for item in self.full_tests]
        data["benchmarks_before"] = [item.to_dict() for item in self.benchmarks_before]
        data["benchmarks_after"] = [item.to_dict() for item in self.benchmarks_after]
        data["review"] = self.review.to_dict() if self.review else None
        data["success"] = self.success
        return data


class SelfImprovementMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload: dict[str, Any]) -> None:
        payload = {**payload, "saved_at": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def retrieve(self, goal: str, *, limit: int = 3) -> list[dict[str, Any]]:
        goal_terms = _terms(goal)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self.load_all():
            target = " ".join(
                [
                    str(record.get("goal", {}).get("objective", "")),
                    str(record.get("outcome", {}).get("rationale", "")),
                    str(record.get("outcome", {}).get("status", "")),
                ]
            )
            overlap = len(goal_terms & _terms(target)) / max(1, len(goal_terms))
            success_bonus = 0.3 if record.get("outcome", {}).get("success") else 0.0
            score = overlap + success_bonus
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].get("saved_at", "")), reverse=True)
        return [record for _, record in scored[:limit]]


@dataclass
class ModelRequestBudget:
    context_window: int = 8192
    safety_margin: int = 256
    chars_per_token: float = 4.0
    stage_desired_outputs: dict[str, int] = field(
        default_factory=lambda: {
            RepositoryStage.UNDERSTAND: 512,
            RepositoryStage.PLAN: 768,
            RepositoryStage.IMPLEMENT: 1800,
            RepositoryStage.REPAIR: 1800,
            RepositoryStage.REVIEW: 512,
        }
    )
    observations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls, context_window: int | None = None) -> "ModelRequestBudget":
        return cls(context_window=int(context_window or os.getenv("JARVIS_BRAIN_CONTEXT_WINDOW", "8192")))

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token) + 1)

    def prepare(self, stage: str, prompt: str, desired_output_tokens: int | None = None) -> tuple[str, int, dict[str, Any]]:
        desired = int(desired_output_tokens or self.stage_desired_outputs.get(stage, 768))
        prompt_tokens = self.estimate_tokens(prompt)
        allowed_output = max(64, min(desired, self.context_window - prompt_tokens - self.safety_margin))
        compacted = False
        if prompt_tokens + allowed_output + self.safety_margin > self.context_window:
            allowed_input_tokens = max(256, self.context_window - desired - self.safety_margin)
            prompt = _compact_text(prompt, int(allowed_input_tokens * self.chars_per_token))
            compacted = True
            prompt_tokens = self.estimate_tokens(prompt)
            allowed_output = max(64, min(desired, self.context_window - prompt_tokens - self.safety_margin))
        if prompt_tokens + allowed_output + self.safety_margin > self.context_window:
            raise ValueError(
                f"CONTEXT_OVERFLOW stage={stage} input_tokens={prompt_tokens} output_tokens={allowed_output} context_window={self.context_window}"
            )
        record = {
            "stage": stage,
            "estimated_input_tokens": prompt_tokens,
            "desired_output_tokens": desired,
            "allowed_output_tokens": allowed_output,
            "context_window": self.context_window,
            "safety_margin": self.safety_margin,
            "compacted": compacted,
        }
        self.observations.append(record)
        return prompt, allowed_output, record


class RepositoryContextManager:
    def __init__(self, *, char_budget: int = 18000) -> None:
        self.char_budget = char_budget
        self.tree: list[str] = []
        self.inspected_files: dict[str, str] = {}
        self.file_hashes: dict[str, str] = {}
        self.searches: list[dict[str, Any]] = []
        self.test_files: list[str] = []
        self.notes: list[dict[str, Any]] = []
        self.recent_failures: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self.tree,
            "inspected_files": self.inspected_files,
            "file_hashes": self.file_hashes,
            "searches": self.searches,
            "test_files": self.test_files,
            "notes": self.notes,
            "recent_failures": self.recent_failures,
        }

    def add_tree(self, paths: list[str]) -> None:
        seen = set(self.tree)
        for path in paths:
            if path not in seen:
                self.tree.append(path)
                seen.add(path)
        self.tree = self.tree[:240]

    def add_file(self, path: str, content: str) -> None:
        self.inspected_files[path] = content[:8000]
        self.file_hashes[path] = _stable_hash(content)
        self._enforce_budget()

    def add_search(self, observation: dict[str, Any]) -> None:
        key = json.dumps(observation, sort_keys=True)
        if all(json.dumps(item, sort_keys=True) != key for item in self.searches):
            self.searches.append(observation)
        self.searches = self.searches[-24:]

    def add_tests(self, paths: list[str]) -> None:
        for path in paths:
            if path not in self.test_files:
                self.test_files.append(path)
        self.test_files = self.test_files[:80]

    def add_note(self, observation: dict[str, Any]) -> None:
        self.notes.append(observation)
        self.notes = self.notes[-12:]

    def add_failures(self, failures: list[RepositoryCommandResult]) -> None:
        self.recent_failures = [item.to_dict() for item in failures][-6:]

    def selected_file_context(self, plan: dict[str, Any] | None = None, *, include_files: bool = False) -> dict[str, str]:
        if not include_files:
            return {key: f"{len(value)} chars" for key, value in self.inspected_files.items()}
        wanted = [str(item) for item in (plan or {}).get("files_to_change", []) if item]
        selected: dict[str, str] = {}
        for path in wanted:
            if path in self.inspected_files:
                selected[path] = self.inspected_files[path]
        for path, content in self.inspected_files.items():
            if path not in selected and len(selected) < 8:
                selected[path] = content
        return selected

    def _enforce_budget(self) -> None:
        while sum(len(value) for value in self.inspected_files.values()) > self.char_budget and len(self.inspected_files) > 1:
            key = next(iter(self.inspected_files))
            self.inspected_files.pop(key, None)
            self.file_hashes.pop(key, None)


class SelfDeveloperCheckpoint:
    def __init__(self, run_dir: str | Path | None = None, *, run_id: str | None = None) -> None:
        if run_dir is None:
            run_dir = Path(tempfile.gettempdir()) / "jarvis_selfdev_runs" / (run_id or uuid.uuid4().hex[:12])
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "self_developer_checkpoint.json"
        self.state: dict[str, Any] = self.load()

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "events": []}

    def save(self, stage: str, payload: dict[str, Any]) -> None:
        self.state["last_stage"] = stage
        self.state[stage] = _sanitize_for_log(payload)
        self.state.setdefault("events", []).append({"stage": stage, "timestamp": datetime.now(timezone.utc).isoformat()})
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


class RepositoryEngineer:
    """Iterative autonomous repository candidate builder with worktree isolation."""

    def __init__(
        self,
        *,
        brain: Any,
        worktree_root: str | Path | None = None,
        memory: SelfImprovementMemory | None = None,
        timeout_seconds: float = 30.0,
        max_cycles: int = 5,
        max_investigation_rounds: int = 4,
        structured_regeneration_attempts: int = 2,
        context_budget: ModelRequestBudget | None = None,
        checkpoint: SelfDeveloperCheckpoint | None = None,
        resume_command: str = "",
    ) -> None:
        self.brain = brain
        self.run_id = uuid.uuid4().hex[:12]
        self.worktree_root = Path(worktree_root or (Path(tempfile.gettempdir()) / "jarvis_selfdev" / self.run_id)).resolve()
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.memory = memory or SelfImprovementMemory(self.worktree_root / "self_improvement_trajectories.jsonl")
        self.timeout_seconds = timeout_seconds
        self.max_cycles = max_cycles
        self.max_investigation_rounds = max_investigation_rounds
        self.structured_regeneration_attempts = structured_regeneration_attempts
        self.context_budget = context_budget or ModelRequestBudget.from_env()
        self.checkpoint = checkpoint
        self.resume_command = resume_command

    def preflight(self) -> dict[str, Any]:
        payload = {
            "provider": getattr(self.brain, "provider_name", type(self.brain).__name__),
            "model": getattr(self.brain, "model_name", ""),
            "context_window": self.context_budget.context_window,
            "structured_generation": "UNKNOWN",
        }
        if not hasattr(self.brain, "health_check") and not hasattr(self.brain, "capabilities"):
            payload["health"] = {"ok": True, "mock_provider": True}
            payload["structured_generation"] = "SKIPPED_FOR_TEST_MOCK"
            if self.checkpoint:
                self.checkpoint.save(RepositoryStage.PREFLIGHT_COMPLETE, payload)
            return payload
        if hasattr(self.brain, "health_check"):
            health = self.brain.health_check()
            payload["health"] = health
            if not health.get("ok", False):
                raise RuntimeError(f"provider preflight failed: {health.get('error')}")
        schema = {"type": "object", "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        raw = self._generate_json("Return {\"ok\": true}.", schema, stage="PREFLIGHT", desired_output_tokens=64, temperature=0.0)
        if raw.get("ok") is not True:
            raise RuntimeError("structured-generation preflight failed")
        payload["structured_generation"] = "OK"
        if self.checkpoint:
            self.checkpoint.save(RepositoryStage.PREFLIGHT_COMPLETE, payload)
        return payload

    def improve(
        self,
        repository_path: str | Path,
        goal: SelfImprovementGoal,
        acceptance_commands: list[list[str]] | None = None,
        *,
        full_test_commands: list[list[str]] | None = None,
        benchmark_commands: list[list[str]] | None = None,
        max_cycles: int | None = None,
    ) -> RepositoryCandidateResult:
        source = Path(repository_path).resolve()
        if not source.exists() or not source.is_dir():
            raise ValueError(f"Repository path does not exist: {source}")
        targeted_commands = list(acceptance_commands or goal.tests)
        full_commands = list(full_test_commands if full_test_commands is not None else goal.full_tests)
        bench_commands = list(benchmark_commands if benchmark_commands is not None else goal.benchmark_commands())
        protected_hashes = self._hash_protected(source, goal.protected_paths)
        worktree: Path | None = None
        trajectory: dict[str, Any] = {
            "goal": goal.to_dict(),
            "repository": str(source),
            "worktree": "",
            "events": [],
        }
        try:
            saved_worktree = (self.checkpoint.state.get(RepositoryStage.WORKTREE_CREATED, {}) if self.checkpoint else {}).get("worktree")
            worktree = Path(saved_worktree).resolve() if saved_worktree and Path(saved_worktree).exists() else self._create_worktree(source)
            trajectory["worktree"] = str(worktree)
            result = RepositoryCandidateResult(RepositoryStage.REJECTED, str(worktree))
            if self.checkpoint:
                self.checkpoint.save(RepositoryStage.WORKTREE_CREATED, {"worktree": str(worktree), "source": str(source), "goal_fingerprint": _stable_hash(json.dumps(goal.to_dict(), sort_keys=True))})
            saved_before = (self.checkpoint.state.get(RepositoryStage.BENCHMARK_BEFORE_COMPLETE, {}) if self.checkpoint else {}).get("results")
            before_benchmarks = [RepositoryCommandResult(**item) for item in saved_before] if saved_before else self._run_commands(worktree, bench_commands, stage=RepositoryStage.BENCHMARK)
            if self.checkpoint and not saved_before:
                self.checkpoint.save(RepositoryStage.BENCHMARK_BEFORE_COMPLETE, {"results": [item.to_dict() for item in before_benchmarks]})
            self._record(trajectory, RepositoryStage.BENCHMARK, {"when": "before", "results": [item.to_dict() for item in before_benchmarks]})
            context = self._investigate(worktree, goal, trajectory)
            prior = self.memory.retrieve(goal.objective, limit=3)
            saved_plan = (self.checkpoint.state.get(RepositoryStage.PLAN_COMPLETE, {}) if self.checkpoint else {}).get("plan")
            plan = saved_plan if isinstance(saved_plan, dict) else self._request_plan(goal, context, prior, trajectory)
            self._ensure_plan_files_read(worktree, context, plan)
            if self.checkpoint and not saved_plan:
                self.checkpoint.save(RepositoryStage.PLAN_COMPLETE, {"plan": plan, "context": _context_for_prompt(context)})
            last_failures: list[RepositoryCommandResult] = []
            last_targeted: list[RepositoryCommandResult] = []
            last_full: list[RepositoryCommandResult] = []
            review = RepositoryReview(False, ["not reviewed"])
            max_dev_cycles = max_cycles or self.max_cycles
            for cycle in range(1, max_dev_cycles + 1):
                proposal = (
                    self._request_patch(goal, context, plan, prior, trajectory)
                    if cycle == 1
                    else self._request_repair(goal, context, plan, last_failures, review, trajectory)
                )
                self._apply_proposal(worktree, source, goal, proposal, context)
                if not self._current_diff(worktree).strip():
                    last_failures = [RepositoryCommandResult(["proposal"], False, stderr="NO_EFFECTIVE_CHANGE", return_code=1)]
                    self._record(trajectory, RepositoryStage.DIAGNOSE, {"cycle": cycle, "reason": "NO_EFFECTIVE_CHANGE"})
                    continue
                if self.checkpoint:
                    self.checkpoint.save(f"PATCH_CYCLE_{cycle}", {"proposal": _redact_large(proposal), "diff": self._current_diff(worktree)[-12000:]})
                self._record(trajectory, RepositoryStage.IMPLEMENT if cycle == 1 else RepositoryStage.REPAIR, _redact_large(proposal) | {"cycle": cycle})
                targeted = self._run_commands(worktree, targeted_commands, stage=RepositoryStage.TEST_TARGETED)
                last_targeted = targeted
                if self.checkpoint:
                    self.checkpoint.save(f"TARGETED_TESTS_{cycle}", {"results": [item.to_dict() for item in targeted]})
                self._record(trajectory, RepositoryStage.TEST_TARGETED, {"cycle": cycle, "results": [item.to_dict() for item in targeted]})
                if targeted and not all(item.success for item in targeted):
                    last_failures = targeted
                    context.setdefault("recent_failures", [item.to_dict() for item in targeted])
                    self._record(trajectory, RepositoryStage.DIAGNOSE, self._diagnosis_payload(worktree, last_failures))
                    continue
                diff = self._current_diff(worktree)
                review = self._request_review(goal, context, diff, targeted, trajectory)
                self._record(trajectory, RepositoryStage.REVIEW, review.to_dict() | {"cycle": cycle})
                if self.checkpoint:
                    self.checkpoint.save(f"REVIEW_{cycle}", review.to_dict())
                if review.recommended_tests:
                    extra = self._run_commands(worktree, review.recommended_tests, stage=RepositoryStage.TEST_TARGETED)
                    targeted.extend(extra)
                    last_targeted = targeted
                    self._record(trajectory, RepositoryStage.TEST_TARGETED, {"cycle": cycle, "review_recommended": [item.to_dict() for item in extra]})
                    if extra and not all(item.success for item in extra):
                        last_failures = extra
                        continue
                if review.blocking_findings:
                    last_failures = [RepositoryCommandResult(["review"], False, stderr="\n".join(review.blocking_findings), return_code=1)]
                    continue
                full = self._run_commands(worktree, full_commands, stage=RepositoryStage.TEST_FULL)
                last_full = full
                if self.checkpoint:
                    self.checkpoint.save(f"FULL_TESTS_{cycle}", {"results": [item.to_dict() for item in full]})
                self._record(trajectory, RepositoryStage.TEST_FULL, {"cycle": cycle, "results": [item.to_dict() for item in full]})
                if full and not all(item.success for item in full):
                    last_failures = full
                    continue
                after_benchmarks = self._run_commands(worktree, bench_commands, stage=RepositoryStage.BENCHMARK)
                if self.checkpoint:
                    self.checkpoint.save(f"BENCHMARK_AFTER_{cycle}", {"results": [item.to_dict() for item in after_benchmarks]})
                self._record(trajectory, RepositoryStage.BENCHMARK, {"when": "after", "cycle": cycle, "results": [item.to_dict() for item in after_benchmarks]})
                benchmark_ok = self._benchmarks_ok(goal, before_benchmarks, after_benchmarks)
                protection_state = self._protected_state(source, worktree, protected_hashes)
                protected_pristine = protection_state == ProtectionState.PRISTINE
                diff = self._current_diff(worktree)
                changed_files = self._changed_files(worktree)
                ready = bool(diff.strip() and protected_pristine and benchmark_ok and (not targeted or all(item.success for item in targeted)) and (not full or all(item.success for item in full)))
                status = RepositoryStage.CANDIDATE_READY if ready else RepositoryStage.REJECTED
                result = RepositoryCandidateResult(
                    status=status,
                    worktree=str(worktree),
                    diff=diff,
                    tests=targeted,
                    full_tests=full,
                    benchmarks_before=before_benchmarks,
                    benchmarks_after=after_benchmarks,
                    review=review,
                    protected_pristine=protected_pristine,
                    protection_state=protection_state,
                    changed_files=changed_files,
                    rationale=str(proposal.get("analysis", "")),
                    cycles=cycle,
                    error="" if ready else "candidate lacked deterministic acceptance evidence",
                )
                self._write_result_artifacts(result, trajectory)
                self._record(trajectory, RepositoryStage.EVALUATE, result.to_dict())
                if self.checkpoint:
                    self.checkpoint.save(RepositoryStage.EVALUATION_COMPLETE, result.to_dict())
                self.memory.record({"trajectory_id": result.trajectory_id, **trajectory, "outcome": result.to_dict()})
                return result
            result = self._rejected(worktree, trajectory, last_failures, before_benchmarks, last_targeted, last_full, review, max_dev_cycles)
            return result
        except Exception as exc:
            provider_failure = _provider_failure_payload(exc)
            status = RepositoryStage.PAUSED if provider_failure else RepositoryStage.REJECTED
            result = RepositoryCandidateResult(
                status=status,
                worktree=str(worktree or ""),
                protected_pristine=True if worktree is not None else False,
                protection_state=ProtectionState.NOT_EVALUATED,
                error=str(exc),
                failure_kind=provider_failure.get("kind", "") if provider_failure else "",
                resume_command=self.resume_command,
            )
            if worktree is not None:
                self._write_result_artifacts(result, trajectory)
            self._record(trajectory, "FAILED", result.to_dict())
            self.memory.record({"trajectory_id": result.trajectory_id, **trajectory, "outcome": result.to_dict()})
            return result

    def project_tree(self, repository_path: str | Path, path: str | None = None, *, limit: int = 240) -> list[str]:
        root = Path(repository_path).resolve()
        scope = self._safe_subtree_path(root, path) if path else root
        files = []
        for candidate in sorted(scope.rglob("*")):
            if len(files) >= limit:
                break
            if candidate.is_file() and _visible_repo_file(candidate, root):
                files.append(candidate.relative_to(root).as_posix())
        return files

    def read_file(self, repository_path: str | Path, relative_path: str, *, max_chars: int = 8000) -> str:
        root = Path(repository_path).resolve()
        path = self._safe_read_path(root, relative_path)
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def read_file_range(self, repository_path: str | Path, relative_path: str, start: int, end: int) -> str:
        root = Path(repository_path).resolve()
        path = self._safe_read_path(root, relative_path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        low = max(1, int(start))
        high = min(len(lines), int(end))
        return "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(low, high + 1))

    def search_text(self, repository_path: str | Path, query: str, path: str | None = None, *, limit: int = 40) -> list[str]:
        root = Path(repository_path).resolve()
        matches = []
        for relative in self.project_tree(root, path=path, limit=1000):
            file_path = root / relative
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    matches.append(f"{relative}:{line_number}: {line[:240]}")
                    break
            if len(matches) >= limit:
                break
        return matches

    def find_symbol_or_import(self, repository_path: str | Path, name: str, path: str | None = None, *, limit: int = 40) -> list[str]:
        pattern = re.compile(rf"\b(class|def|from|import)\b.*\b{re.escape(name)}\b|\b{re.escape(name)}\b")
        root = Path(repository_path).resolve()
        matches = []
        for relative in self.project_tree(root, path=path, limit=1000):
            if not relative.endswith(".py"):
                continue
            text = (root / relative).read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{relative}:{line_number}: {line[:240]}")
                    break
            if len(matches) >= limit:
                break
        return matches

    def inspect_tests(self, repository_path: str | Path, path: str | None = None, *, limit: int = 80) -> list[str]:
        return [item for item in self.project_tree(repository_path, path=path, limit=1000) if item.endswith(".py") and ("test" in Path(item).name.lower())][:limit]

    def _create_worktree(self, repository_path: Path) -> Path:
        worktree = self.worktree_root / f"candidate_{uuid.uuid4().hex[:10]}"
        worktree.parent.mkdir(parents=True, exist_ok=True)
        self._assert_external_candidate_path(repository_path, worktree)
        if (repository_path / ".git").exists() and self._git_root(repository_path) == repository_path.resolve():
            completed = self._run_git_raw(repository_path, ["worktree", "add", "--detach", str(worktree), "HEAD"], check=False)
            if completed.returncode == 0:
                return worktree.resolve()
            if worktree.exists():
                shutil.rmtree(worktree, onerror=_make_writable_and_retry)
        shutil.copytree(repository_path, worktree, ignore=_fallback_copy_ignore)
        self._run_git_raw(worktree, ["init"], check=True)
        self._run_git_raw(worktree, ["config", "user.email", "jarvis@example.invalid"], check=True)
        self._run_git_raw(worktree, ["config", "user.name", "Jarvis RepositoryEngineer"], check=True)
        self._run_git_raw(worktree, ["add", "."], check=True)
        self._run_git_raw(worktree, ["commit", "-m", "baseline"], check=True)
        return worktree.resolve()

    def _assert_external_candidate_path(self, source: Path, destination: Path) -> None:
        source_root = source.resolve()
        destination_root = destination.resolve(strict=False)
        if destination_root == source_root:
            raise ValueError("candidate worktree destination must not equal the source repository")
        if destination_root.is_relative_to(source_root):
            raise ValueError("candidate worktree destination must live outside the source repository")
        if source_root.is_relative_to(destination_root):
            raise ValueError("candidate worktree destination must not contain the source repository")

    def _investigate(self, worktree: Path, goal: SelfImprovementGoal, trajectory: dict[str, Any]) -> dict[str, Any]:
        manager = RepositoryContextManager()
        manager.add_tree(self.project_tree(worktree))
        manager.add_tests(self.inspect_tests(worktree))
        context = manager.to_dict()
        fallback_terms = [term for term in sorted(_terms(goal.objective)) if len(term) > 4][:6]
        fallback_requests = [{"tool": "search_text", "query": term} for term in fallback_terms]
        fallback_requests.extend({"tool": "read_file", "path": path} for path in self._likely_files(context, goal)[:8])
        for round_index in range(1, self.max_investigation_rounds + 1):
            requests = self._request_investigation(goal, context, round_index)
            if not requests:
                requests = fallback_requests[: max(1, 4 - round_index)]
            if not requests:
                break
            observations = [self._run_repository_tool(worktree, request, goal) for request in requests[:8]]
            for observation in observations:
                if observation.get("tool") == "read_file":
                    manager.add_file(observation["path"], observation.get("content", ""))
                elif observation.get("tool") == "tree":
                    manager.add_tree(observation.get("results", []))
                elif observation.get("tool") in {"search_text", "find_symbol_import"}:
                    manager.add_search(observation)
                elif observation.get("tool") == "inspect_tests":
                    manager.add_tests(observation.get("results", []))
                else:
                    manager.add_note(observation)
            context = manager.to_dict()
            self._record(trajectory, RepositoryStage.UNDERSTAND, {"round": round_index, "requests": requests, "observations": _compact_observations(observations)})
            if self.checkpoint:
                self.checkpoint.save(f"INVESTIGATION_ROUND_{round_index}", {"context": _context_for_prompt(context), "requests": requests})
            if context["inspected_files"] and round_index >= 2:
                break
        if not context["inspected_files"]:
            for path in self._likely_files(context, goal)[:3]:
                try:
                    content = self.read_file(worktree, path)
                    manager.add_file(path, content)
                except Exception:
                    continue
            context = manager.to_dict()
        if not context["inspected_files"]:
            raise ValueError("investigation quality gate failed: no relevant source file was inspected")
        return context

    def _request_investigation(self, goal: SelfImprovementGoal, context: dict[str, Any], round_index: int) -> list[dict[str, Any]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "requests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "tool": {"type": "string"},
                            "path": {"type": "string"},
                            "query": {"type": "string"},
                            "symbol": {"type": "string"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                        },
                        "required": ["tool"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["requests"],
        }
        prompt = (
            "Return JSON only. Role: Repository Architect. Decide which repository information is needed next.\n"
            "Allowed tools: tree, search_text, read_file, read_file_range, find_symbol_import, inspect_tests, git_diff.\n"
            "Do not request secrets, environment variables, or files outside the repository.\n"
            f"Round: {round_index}\nGoal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Known context:\n{json.dumps(_context_for_prompt(context), indent=2, sort_keys=True)}"
        )
        payload = self._generate_json(prompt, schema, stage=RepositoryStage.UNDERSTAND, desired_output_tokens=512, temperature=0.1)
        requests = payload.get("requests") if isinstance(payload, dict) else []
        if not isinstance(requests, list):
            return []
        return [item for item in requests if isinstance(item, dict)]

    def _run_repository_tool(self, worktree: Path, request: dict[str, Any], goal: SelfImprovementGoal) -> dict[str, Any]:
        tool = str(request.get("tool", "")).strip()
        try:
            if tool == "tree":
                path = str(request.get("path") or "")
                return {"tool": tool, "path": path, "results": self.project_tree(worktree, path=path or None)}
            if tool == "search_text":
                query = str(request.get("query", ""))
                path = str(request.get("path") or "")
                return {"tool": tool, "query": query, "path": path, "results": self.search_text(worktree, query, path=path or None)}
            if tool == "read_file":
                path = str(request.get("path", ""))
                return {"tool": tool, "path": path, "content": self.read_file(worktree, path)}
            if tool == "read_file_range":
                path = str(request.get("path", ""))
                return {"tool": tool, "path": path, "content": self.read_file_range(worktree, path, int(request.get("start", 1)), int(request.get("end", 120)))}
            if tool == "find_symbol_import":
                symbol = str(request.get("symbol") or request.get("query") or "")
                path = str(request.get("path") or "")
                return {"tool": tool, "symbol": symbol, "path": path, "results": self.find_symbol_or_import(worktree, symbol, path=path or None)}
            if tool == "inspect_tests":
                path = str(request.get("path") or "")
                return {"tool": tool, "path": path, "results": self.inspect_tests(worktree, path=path or None)}
            if tool == "git_diff":
                return {"tool": tool, "diff": self._current_diff(worktree)}
            return {"tool": tool, "error": "unsupported repository tool"}
        except Exception as exc:
            return {"tool": tool, "error": str(exc)}

    def _request_plan(self, goal: SelfImprovementGoal, context: dict[str, Any], prior: list[dict[str, Any]], trajectory: dict[str, Any]) -> dict[str, Any]:
        schema = {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "analysis": {"type": "string"},
                "plan": {"type": "string"},
                "files_to_change": {"type": "array", "items": {"type": "string"}},
                "tests_to_run": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["analysis", "plan"],
        }
        prompt = (
            "Return JSON only. Role: Repository Architect. Make a concise implementation plan.\n"
            f"Goal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Repository context:\n{json.dumps(_context_for_prompt(context), indent=2, sort_keys=True)}\n"
            f"Relevant prior self-development memory:\n{json.dumps(_prior_for_prompt(prior), indent=2, sort_keys=True)}"
        )
        plan = self._generate_json(prompt, schema, stage=RepositoryStage.PLAN, desired_output_tokens=768, temperature=0.1)
        self._record(trajectory, RepositoryStage.PLAN, plan if isinstance(plan, dict) else {"plan": ""})
        return plan if isinstance(plan, dict) else {"analysis": "", "plan": ""}

    def _request_patch(self, goal: SelfImprovementGoal, context: dict[str, Any], plan: dict[str, Any], prior: list[dict[str, Any]], trajectory: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Return JSON only. Role: Repository Implementer. Produce a repository improvement candidate.\n"
            "Use complete file replacements. Do not alter protected paths. Final readiness requires deterministic tests.\n"
            f"SelfImprovementGoal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Plan:\n{json.dumps(plan, indent=2, sort_keys=True)}\n"
            f"Repository context:\n{json.dumps(_context_for_prompt(context, include_files=True, plan=plan), indent=2, sort_keys=True)}\n"
            f"Relevant prior memory:\n{json.dumps(_prior_for_prompt(prior), indent=2, sort_keys=True)}"
        )
        return self._generate_patch_bundle(prompt)

    def _request_repair(
        self,
        goal: SelfImprovementGoal,
        context: dict[str, Any],
        plan: dict[str, Any],
        failures: list[RepositoryCommandResult],
        review: RepositoryReview,
        trajectory: dict[str, Any],
    ) -> dict[str, Any]:
        diff = self._current_diff(Path(trajectory["worktree"]))
        prompt = (
            "Return JSON only. Role: Repository Repairer. Diagnose failing deterministic evidence and return corrected complete files.\n"
            "Do not weaken tests, benchmarks, hidden verifiers, safety policy, or protected paths.\n"
            f"Goal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Plan:\n{json.dumps(plan, indent=2, sort_keys=True)}\n"
            f"Current diff:\n{diff[-12000:]}\n"
            f"Failures:\n{json.dumps([item.to_dict() for item in failures], indent=2, sort_keys=True)}\n"
            f"Reviewer findings:\n{json.dumps(review.to_dict(), indent=2, sort_keys=True)}\n"
            f"Relevant files:\n{json.dumps(_context_for_prompt(context, include_files=True, plan=plan), indent=2, sort_keys=True)}"
        )
        return self._generate_patch_bundle(prompt)

    def _generate_patch_bundle(self, prompt: str) -> dict[str, Any]:
        schema = _patch_schema()
        stage = RepositoryStage.REPAIR if "Repository Repairer" in prompt else RepositoryStage.IMPLEMENT
        payload = self._generate_json(prompt, schema, stage=stage, desired_output_tokens=1800, temperature=0.2)
        if not isinstance(payload.get("files"), list):
            raise ValueError("repository proposal did not contain files")
        payload.setdefault("new_files", [])
        payload.setdefault("deleted_files", [])
        return payload

    def _request_review(self, goal: SelfImprovementGoal, context: dict[str, Any], diff: str, targeted: list[RepositoryCommandResult], trajectory: dict[str, Any]) -> RepositoryReview:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "approved": {"type": "boolean"},
                "blocking_findings": {"type": "array", "items": {"type": "string"}},
                "optional_findings": {"type": "array", "items": {"type": "string"}},
                "recommended_tests": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            },
            "required": ["approved", "blocking_findings", "optional_findings", "recommended_tests"],
        }
        prompt = (
            "Return JSON only. Role: Repository Reviewer. You cannot edit files.\n"
            "Separate BLOCKING CONTRACT VIOLATIONS from optional improvements. Do not require endless polish.\n"
            f"Goal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Diff:\n{diff[-16000:]}\n"
            f"Targeted test results:\n{json.dumps([item.to_dict() for item in targeted], indent=2, sort_keys=True)}\n"
            f"Relevant context:\n{json.dumps(_context_for_prompt(context), indent=2, sort_keys=True)}"
        )
        payload = self._generate_json(prompt, schema, stage=RepositoryStage.REVIEW, desired_output_tokens=512, temperature=0.1)
        if not isinstance(payload, dict) or not isinstance(payload.get("approved"), bool):
            return self._deterministic_review_fallback(diff, targeted, "malformed reviewer output")
        blocking = payload.get("blocking_findings", [])
        optional = payload.get("optional_findings", [])
        recommended = payload.get("recommended_tests", [])
        if not isinstance(blocking, list) or not isinstance(optional, list) or not isinstance(recommended, list):
            return self._deterministic_review_fallback(diff, targeted, "invalid reviewer fields")
        return RepositoryReview(
            approved=bool(payload.get("approved", False)),
            blocking_findings=[str(item) for item in blocking],
            optional_findings=[str(item) for item in optional],
            recommended_tests=[list(map(str, item)) for item in recommended if isinstance(item, list)],
        )

    def _deterministic_review_fallback(self, diff: str, targeted: list[RepositoryCommandResult], reason: str) -> RepositoryReview:
        if diff.strip() and (not targeted or all(item.success for item in targeted)):
            return RepositoryReview(approved=True, optional_findings=[f"REVIEW_FALLBACK_USED: {reason}"])
        return RepositoryReview(approved=False, blocking_findings=[f"REVIEW_UNAVAILABLE: {reason}"])

    def _generate_json(self, prompt: str, schema: dict[str, Any], *, stage: str, desired_output_tokens: int, temperature: float) -> dict[str, Any]:
        last_raw = ""
        for attempt in range(1, self.structured_regeneration_attempts + 2):
            budgeted_prompt, max_tokens, budget_record = self.context_budget.prepare(stage, prompt, desired_output_tokens)
            raw = self._generate_structured(budgeted_prompt, schema, max_tokens=max_tokens, temperature=temperature)
            last_raw = raw
            try:
                data = json.loads(_extract_json(raw))
                if isinstance(data, dict):
                    data.setdefault("_budget", budget_record)
                    return data
            except json.JSONDecodeError:
                pass
            prompt = prompt + "\n\nPrevious response was malformed. Regenerate JSON only for the schema."
        raise ValueError(f"structured repository generation failed: {_stable_hash(last_raw)}")

    def _generate_structured(self, prompt: str, schema: dict[str, Any], *, max_tokens: int, temperature: float) -> str:
        if hasattr(self.brain, "generate_structured"):
            try:
                return self.brain.generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
            except (AttributeError, NotImplementedError, StructuredGenerationUnsupported):
                pass
        if hasattr(self.brain, "generate_coding"):
            return self.brain.generate_coding(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
        return self.brain.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)

    def _ensure_plan_files_read(self, worktree: Path, context: dict[str, Any], plan: dict[str, Any]) -> None:
        for relative in [str(item) for item in plan.get("files_to_change", []) if item]:
            if relative in context.get("inspected_files", {}):
                continue
            path = self._safe_read_path(worktree, relative)
            content = path.read_text(encoding="utf-8", errors="replace")
            context.setdefault("inspected_files", {})[relative] = content[:8000]
            context.setdefault("file_hashes", {})[relative] = _stable_hash(content)

    def _apply_proposal(self, worktree: Path, source: Path, goal: SelfImprovementGoal, proposal: dict[str, Any], context: dict[str, Any]) -> None:
        for item in [*proposal.get("files", []), *proposal.get("new_files", [])]:
            if not isinstance(item, dict):
                continue
            path = self._safe_path(worktree, source, str(item.get("path", "")), goal)
            relative = str(item.get("path", "")).replace("\\", "/")
            if path.exists():
                current = path.read_text(encoding="utf-8", errors="replace")
                if relative not in context.get("inspected_files", {}):
                    context.setdefault("inspected_files", {})[relative] = current[:8000]
                    context.setdefault("file_hashes", {})[relative] = _stable_hash(current)
                expected_hash = context.get("file_hashes", {}).get(relative)
                if expected_hash and expected_hash != _stable_hash(current):
                    raise ValueError(f"implementation quality gate failed: stale file context for {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(item.get("content", "")), encoding="utf-8")
            context.setdefault("inspected_files", {})[relative] = str(item.get("content", ""))[:8000]
            context.setdefault("file_hashes", {})[relative] = _stable_hash(str(item.get("content", "")))
        for relative in proposal.get("deleted_files", []):
            path = self._safe_path(worktree, source, str(relative), goal)
            if path.exists():
                path.unlink()
        self._clear_python_caches(worktree)

    def _run_commands(self, worktree: Path, commands: list[list[str]], *, stage: str) -> list[RepositoryCommandResult]:
        results = []
        for command in commands:
            if not command:
                continue
            completed = subprocess.run(
                command,
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                env=_safe_command_env(),
            )
            output = f"{completed.stdout}\n{completed.stderr}"
            results.append(
                RepositoryCommandResult(
                    command=list(command),
                    success=completed.returncode == 0,
                    stdout=completed.stdout[-8000:],
                    stderr=completed.stderr[-8000:],
                    return_code=completed.returncode,
                    metrics=_extract_metrics(output),
                    cwd=str(worktree),
                    executable=command[0],
                )
            )
            self._clear_python_caches(worktree)
        return results

    def _clear_python_caches(self, worktree: Path) -> None:
        for cache_dir in worktree.rglob("__pycache__"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, onerror=_make_writable_and_retry)
        for pyc_file in worktree.rglob("*.pyc"):
            if pyc_file.is_file():
                try:
                    pyc_file.unlink()
                except OSError:
                    _make_writable_and_retry(os.unlink, str(pyc_file), None)

    def _current_diff(self, worktree: Path) -> str:
        self._git(worktree, ["add", "-N", "."], check=False)
        return self._git(worktree, ["diff", "--", "."], check=False).stdout

    def _changed_files(self, worktree: Path) -> list[str]:
        completed = self._git(worktree, ["diff", "--name-only", "--", "."], check=False)
        return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]

    def _benchmarks_ok(self, goal: SelfImprovementGoal, before: list[RepositoryCommandResult], after: list[RepositoryCommandResult]) -> bool:
        if after and not all(item.success for item in after):
            return False
        after_metrics = _merge_metrics(after)
        for metric, minimum in goal.metric_minimums.items():
            if after_metrics.get(metric, float("-inf")) < float(minimum):
                return False
        if not goal.require_benchmark_improvement:
            return True
        before_metrics = _merge_metrics(before)
        if goal.metric_name:
            return after_metrics.get(goal.metric_name, float("-inf")) > before_metrics.get(goal.metric_name, float("-inf"))
        common = set(before_metrics) & set(after_metrics)
        return any(after_metrics[key] > before_metrics[key] for key in common)

    def _write_result_artifacts(self, result: RepositoryCandidateResult, trajectory: dict[str, Any]) -> None:
        root = Path(result.worktree)
        diff_path = root / "SELF_DEVELOPMENT_DIFF.patch"
        result_path = root / "SELF_DEVELOPMENT_RESULT.json"
        if result.diff:
            diff_path.write_text(result.diff, encoding="utf-8")
            result.diff_path = str(diff_path)
        payload = {"trajectory": trajectory, "outcome": result.to_dict(), "inspect_command": f"git -C {result.worktree} diff"}
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        result.result_path = str(result_path)

    def _rejected(
        self,
        worktree: Path,
        trajectory: dict[str, Any],
        failures: list[RepositoryCommandResult],
        before: list[RepositoryCommandResult],
        targeted: list[RepositoryCommandResult] | None = None,
        full: list[RepositoryCommandResult] | None = None,
        review: RepositoryReview | None = None,
        cycles: int | None = None,
    ) -> RepositoryCandidateResult:
        result = RepositoryCandidateResult(
            status=RepositoryStage.REJECTED,
            worktree=str(worktree),
            diff=self._current_diff(worktree),
            tests=targeted if targeted is not None else failures,
            full_tests=full or [],
            benchmarks_before=before,
            review=review,
            protected_pristine=False,
            protection_state=ProtectionState.NOT_EVALUATED,
            changed_files=self._changed_files(worktree),
            error="development cycles exhausted before deterministic acceptance",
            cycles=cycles or self.max_cycles,
        )
        self._write_result_artifacts(result, trajectory)
        self._record(trajectory, RepositoryStage.EVALUATE, result.to_dict())
        self.memory.record({"trajectory_id": result.trajectory_id, **trajectory, "outcome": result.to_dict()})
        return result

    def _safe_path(self, worktree: Path, source: Path, relative_path: str, goal: SelfImprovementGoal) -> Path:
        raw = Path(relative_path)
        normalized = raw.as_posix()
        if not normalized or raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"unsafe repository path: {relative_path}")
        if _path_matches(normalized, goal.protected_paths):
            raise ValueError(f"protected repository path: {relative_path}")
        if goal.allowed_paths and not _path_matches(normalized, goal.allowed_paths):
            raise ValueError(f"path not allowed for this goal: {relative_path}")
        candidate = (worktree / raw).resolve(strict=False)
        root = worktree.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"path escapes worktree: {relative_path}")
        live_candidate = (source / raw).resolve(strict=False)
        if live_candidate.exists() and live_candidate.is_symlink():
            raise ValueError(f"refuse to edit symlinked live path: {relative_path}")
        return candidate

    def _safe_read_path(self, root: Path, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"unsafe repository read path: {relative_path}")
        path = (root / raw).resolve(strict=False)
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"read path escapes repository: {relative_path}")
        if not path.exists() or not path.is_file():
            raise ValueError(f"repository file does not exist: {relative_path}")
        return path

    def _safe_subtree_path(self, root: Path, relative_path: str | None) -> Path:
        if not relative_path:
            return root
        raw = Path(relative_path)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"unsafe repository subtree path: {relative_path}")
        path = (root / raw).resolve(strict=False)
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"subtree path escapes repository: {relative_path}")
        if not path.exists():
            return path
        if path.is_file():
            return path.parent
        return path

    def _hash_protected(self, source: Path, protected_paths: list[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for relative in protected_paths:
            raw = Path(relative)
            if raw.is_absolute() or ".." in raw.parts:
                raise ValueError(f"unsafe protected path: {relative}")
            path = (source / raw).resolve(strict=False)
            if not path.is_relative_to(source.resolve()):
                raise ValueError(f"protected path escapes repository: {relative}")
            if path.exists() and path.is_file():
                hashes[raw.as_posix()] = sha256(path.read_bytes()).hexdigest()
            elif path.exists() and path.is_dir():
                root_key = raw.as_posix().rstrip("/") + "/"
                entries: list[str] = []
                for child in sorted(path.rglob("*")):
                    if child.is_file() and _visible_repo_file(child, source):
                        rel = child.relative_to(source).as_posix()
                        entries.append(rel)
                        hashes[rel] = sha256(child.read_bytes()).hexdigest()
                hashes[root_key] = "DIR:" + _stable_hash("\n".join(entries))
        return hashes

    def _protected_pristine(self, source: Path, worktree: Path, protected_hashes: dict[str, str]) -> bool:
        return self._protected_state(source, worktree, protected_hashes) == ProtectionState.PRISTINE

    def _protected_state(self, source: Path, worktree: Path, protected_hashes: dict[str, str]) -> str:
        for relative, expected in protected_hashes.items():
            if expected.startswith("DIR:"):
                continue
            source_path = (source / relative).resolve(strict=False)
            candidate_path = (worktree / relative).resolve(strict=False)
            if not candidate_path.exists() or sha256(candidate_path.read_bytes()).hexdigest() != expected:
                return ProtectionState.MODIFIED
            if source_path.exists() and sha256(source_path.read_bytes()).hexdigest() != expected:
                return ProtectionState.MODIFIED
        protected_roots = set()
        for relative in protected_hashes:
            parts = Path(relative).parts
            if parts:
                protected_roots.add(parts[0])
        for root_name in protected_roots:
            source_root = source / root_name
            candidate_root = worktree / root_name
            if candidate_root.exists() and source_root.exists() and source_root.is_dir():
                expected_files = {
                    path
                    for path, value in protected_hashes.items()
                    if not value.startswith("DIR:") and (path == root_name or path.startswith(root_name + "/"))
                }
                actual_files = {
                    path.relative_to(worktree).as_posix()
                    for path in candidate_root.rglob("*")
                    if path.is_file() and _visible_repo_file(path, worktree)
                }
                if actual_files != expected_files:
                    return ProtectionState.MODIFIED
        return ProtectionState.PRISTINE

    def _git(self, worktree: Path, args: list[str], *, check: bool) -> subprocess.CompletedProcess:
        return self._run_git_raw(worktree, args, check=check)

    def _run_git_raw(self, cwd: Path, args: list[str], *, check: bool) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["GIT_CEILING_DIRECTORIES"] = str(cwd.resolve().parent.parent)
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=self.timeout_seconds, check=check, env=env)

    def _git_root(self, cwd: Path) -> Path | None:
        completed = self._run_git_raw(cwd, ["rev-parse", "--show-toplevel"], check=False)
        if completed.returncode != 0:
            return None
        return Path(completed.stdout.strip()).resolve()

    def _likely_files(self, context: dict[str, Any], goal: SelfImprovementGoal) -> list[str]:
        terms = _terms(goal.objective)
        scored: list[tuple[int, str]] = []
        for path in context.get("tree", []):
            lower = path.lower()
            score = sum(1 for term in terms if term in lower)
            if path.endswith(".py") and score:
                scored.append((score, path))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [path for _, path in scored] or [path for path in context.get("tree", []) if str(path).endswith(".py")][:6]

    def _diagnosis_payload(self, worktree: Path, failures: list[RepositoryCommandResult]) -> dict[str, Any]:
        return {"failures": [item.to_dict() for item in failures], "diff": self._current_diff(worktree)[-12000:]}

    def _record(self, trajectory: dict[str, Any], stage: str, payload: dict[str, Any]) -> None:
        trajectory.setdefault("events", []).append(
            {"stage": stage, "timestamp": datetime.now(timezone.utc).isoformat(), "payload": _sanitize_for_log(payload)}
        )


def _patch_schema() -> dict[str, Any]:
    file_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "analysis": {"type": "string"},
            "files": {"type": "array", "items": file_schema},
            "new_files": {"type": "array", "items": file_schema},
            "deleted_files": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["analysis", "files"],
    }


def _visible_repo_file(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    if any(_ignored_repo_part(part) or part.startswith(".pytest_tmp") for part in parts):
        return False
    normalized = path.relative_to(root).as_posix()
    if normalized.startswith("data/benchmark_runs/"):
        return False
    if normalized.startswith("skills/_staging/"):
        return False
    if path.suffix.lower() in {".pyc", ".pt", ".sqlite", ".png", ".jpg", ".jpeg", ".gif", ".bin"}:
        return False
    return True


def _fallback_copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if _ignored_repo_part(name) or name.startswith(".pytest_tmp") or name.endswith(".pyc"):
            ignored.add(name)
    directory_path = Path(directory)
    if directory_path.name == "data":
        ignored.update(name for name in names if name == "benchmark_runs")
    if directory_path.name == "skills":
        ignored.update(name for name in names if name == "_staging")
    return ignored


def _ignored_repo_part(part: str) -> bool:
    return part in {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".runtime",
        ".cache",
        "_staging",
        "node_modules",
        "build",
        "dist",
        ".jarvis_worktrees",
        "worktrees",
        "jarvis_selfdev",
    }


def _path_matches(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    normalized = path.strip("/").replace("\\", "/")
    for pattern in patterns:
        clean = pattern.strip("/").replace("\\", "/")
        if clean in {"", "."}:
            return True
        if normalized == clean or normalized.startswith(clean + "/"):
            return True
    return False


def _extract_json(text: str) -> str:
    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text


def _redact_large(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    for key in ["files", "new_files"]:
        redacted[key] = [
            {"path": item.get("path"), "content_hash": _stable_hash(str(item.get("content", "")))}
            for item in payload.get(key, [])
            if isinstance(item, dict)
        ]
    return redacted


def _context_for_prompt(context: dict[str, Any], *, include_files: bool = False, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "tree_excerpt": list(context.get("tree", []))[:160],
        "searches": context.get("searches", [])[-12:],
        "test_files": context.get("test_files", [])[:80],
        "notes": context.get("notes", [])[-8:],
    }
    inspected = context.get("inspected_files", {})
    if include_files:
        wanted = [str(item) for item in (plan or {}).get("files_to_change", []) if item]
        selected = {path: inspected[path] for path in wanted if path in inspected}
        for key, value in inspected.items():
            if key not in selected and len(selected) < 8:
                selected[key] = value
        payload["inspected_files"] = selected
    else:
        payload["inspected_files"] = {key: f"{len(value)} chars" for key, value in list(inspected.items())[:20]}
    if context.get("recent_failures"):
        payload["recent_failures"] = context.get("recent_failures", [])[-6:]
    return payload


def _compact_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in observations:
        reduced = dict(item)
        if "content" in reduced:
            reduced["content_hash"] = _stable_hash(str(reduced.pop("content")))
        compact.append(reduced)
    return compact


def _prior_for_prompt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "goal": item.get("goal", {}).get("objective", ""),
            "status": item.get("outcome", {}).get("status", ""),
            "changed_files": item.get("outcome", {}).get("changed_files", []),
            "error": item.get("outcome", {}).get("error", ""),
        }
        for item in records
    ]


def _extract_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        for match in re.finditer(r"([A-Z][A-Z0-9_]+|[a-z][a-z0-9_]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)", line):
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
    return metrics


def _merge_metrics(results: list[RepositoryCommandResult]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for result in results:
        merged.update(result.metrics)
    return merged


def _compact_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n[...SELFDEVELOPER_CONTEXT_COMPACTED...]\n\n" + text[-tail:]


def _provider_failure_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ProviderError):
        return {
            "kind": exc.kind,
            "status": exc.status,
            "message": exc.message,
            "attempt": exc.attempt,
            "model": exc.model,
        }
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, ProviderError):
        return _provider_failure_payload(cause)
    return {}


def _safe_command_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "JARVIS_BRAIN_PROVIDER",
        "JARVIS_BRAIN_BASE_URL",
        "JARVIS_BRAIN_MODEL",
        "JARVIS_BRAIN_API_KEY",
        "JARVIS_BRAIN_TIMEOUT",
        "JARVIS_BRAIN_TEMPERATURE",
        "JARVIS_BRAIN_TOP_P",
        "JARVIS_BRAIN_MAX_TOKENS",
        "JARVIS_BRAIN_RETRIES",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _make_writable_and_retry(function: Any, path: str, _exc_info: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        pass


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("<redacted>" if "api_key" in key.lower() or "token" in key.lower() or "secret" in key.lower() else _sanitize_for_log(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    if isinstance(value, str):
        for key in ["JARVIS_BRAIN_API_KEY", "API_KEY", "TOKEN", "SECRET"]:
            secret = os.getenv(key)
            if secret:
                value = value.replace(secret, "<redacted>")
        return value
    return value


def _terms(text: str) -> set[str]:
    return {term for term in re.split(r"[^a-z0-9]+", text.lower()) if len(term) > 2}


def _stable_hash(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()
