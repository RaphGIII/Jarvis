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

from brain.json_utils import lenient_json_loads
from runtime.deadline import CallTimeout, Deadline, DeadlineExceeded, call_with_timeout
from runtime.heartbeat import Heartbeat
from development.edit_engine import (
    EditBudget,
    EditEngine,
    EditError,
    EditResult,
    PathPolicy,
    edit_schema,
    parse_bundle,
)

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


#: Edit failures that represent a violated boundary rather than a near-miss.
#: Re-prompting the model on these would just invite a second attempt at the
#: same boundary, so they end the run.
_FATAL_EDIT_KINDS = frozenset(
    {"protected_path", "path_not_allowed", "path_escape", "unsafe_path", "symlink_target"}
)


def _correction_hint(error: EditError) -> str:
    """Turn an :class:`EditError` kind into advice the model can act on."""

    hints = {
        "no_unique_match": (
            "Your search text does not appear in the file. Pick a DIFFERENT anchor "
            "that you can see verbatim in CURRENT CODE below."
        ),
        "ambiguous_search": (
            "Your search text appears more than once. Add one or two neighbouring "
            "lines so the anchor becomes unique."
        ),
        "stale_context": (
            "The file changed since you last saw it. Use only the CURRENT CODE below."
        ),
        "no_effective_edit": (
            "Your replacement was identical to what is already there. Make a real change."
        ),
        "missing_target": (
            "That file does not exist. Either target an existing file, or add it via new_files."
        ),
        "create_over_existing": (
            "That file already exists. Edit it with search/replace instead of new_files."
        ),
        "rewrite_too_large": (
            "Do not rewrite the whole file. Emit small search/replace edits instead."
        ),
        "rewrite_truncates_file": (
            "You replaced a whole file with a fragment, which would delete the rest of it. "
            "Emit a search/replace edit instead: put the EXACT existing lines you want to change "
            "in 'search', and the new version of just those lines in 'replace'."
        ),
        "duplicate_definition": (
            "Your replacement repeated a 'def' or 'class' line that was already part of the search "
            "anchor, so the file would contain it twice. Put that line in the search OR the replace, "
            "not both."
        ),
        "syntax_error": (
            "Your replacement text broke the file's Python syntax. Look at the indentation: "
            "the 'replace' text must line up with the code around the anchor exactly as it appears "
            "in CURRENT CODE."
        ),
        "empty_search": (
            "Every edit needs a non-empty 'search' anchor copied verbatim from CURRENT CODE, "
            "plus the new version of those same lines in 'replace'."
        ),
        "invalid_edit": (
            "Each entry of files[] needs a path plus either search+replace, or content."
        ),
    }
    return hints.get(error.kind, "Emit a corrected minimal patch.")


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

    def add_file(self, path: str, content: str, *, complete: bool = True) -> None:
        """Record a file the model has seen.

        The hash is the staleness baseline, so it may only be recorded when the
        *whole* file was read.  Hashing a truncated read is a silent trap: the
        stored digest can never equal the digest of the real file, so every
        subsequent edit to that file is rejected as stale and the run cannot
        make progress.  Observed exactly that way once a source file grew past
        the 8000-character read limit.
        """

        self.inspected_files[path] = content[:8000]
        if complete:
            self.file_hashes[path] = _stable_hash(content)
        else:
            self.file_hashes.pop(path, None)
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
        max_seconds: float | None = None,
        model_call_timeout_seconds: float = 600.0,
        heartbeat: Heartbeat | None = None,
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
        #: Wall-clock bound for the whole run.  There was none, so a single
        #: wedged generation could hold a self-development run open forever.
        self.max_seconds = max_seconds
        self.deadline = Deadline.of(max_seconds, name="self-development")
        self.model_call_timeout_seconds = float(model_call_timeout_seconds)
        self.heartbeat = heartbeat or Heartbeat(None)
        #: The trajectory of the run currently in progress.  Helper methods deep
        #: in the patch path need somewhere to record what they observed without
        #: threading the dict through every signature.
        self._active_trajectory: dict[str, Any] = {"events": []}

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

        # Fresh clock per run: an engineer reused across runs must not have the
        # first run's elapsed time counted against the second.
        self.deadline = Deadline.of(self.max_seconds, name="self-development")
        self.heartbeat.set_budget(self.max_seconds)
        self.heartbeat.beat("preflight", goal.objective[:120], progress=True)

        worktree_state = self.checkpoint.state.get(RepositoryStage.WORKTREE_CREATED, {}) if self.checkpoint else {}
        saved_worktree = worktree_state.get("worktree")
        resuming = bool(saved_worktree and Path(saved_worktree).exists())
        if resuming:
            saved_goal = worktree_state.get("goal")
            current_fingerprint = _stable_hash(json.dumps(goal.to_dict(), sort_keys=True))
            if saved_goal is not None and worktree_state.get("goal_fingerprint") != current_fingerprint:
                # A resumed run must keep operating under the ORIGINAL goal
                # (allowed_paths/protected_paths/tests/etc.), not whatever the
                # caller happens to pass on the --resume invocation. Silently
                # accepting a different goal here would let a resumed run lose
                # its permission boundaries (e.g. protected_paths) if the
                # operator forgets to repeat every CLI flag identically.
                goal = SelfImprovementGoal(**saved_goal)
                acceptance_commands = None
                full_test_commands = None
                benchmark_commands = None

        # The owner's protected paths apply to every goal that targets ZEUS
        # itself, whatever the caller declared. A goal cannot opt out of them;
        # it can only add to them.
        if (source / "zeus_supervisor").is_dir() or (source / "owner").is_dir():
            from owner.protected import PROTECTED_PATHS

            merged = list(goal.protected_paths)
            merged.extend(p for p in PROTECTED_PATHS if p not in merged)
            goal = SelfImprovementGoal(**{**goal.to_dict(), "protected_paths": merged})

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
        self._active_trajectory = trajectory
        try:
            worktree = Path(saved_worktree).resolve() if resuming else self._create_worktree(source)
            trajectory["worktree"] = str(worktree)
            result = RepositoryCandidateResult(RepositoryStage.REJECTED, str(worktree))
            if self.checkpoint and not resuming:
                self.checkpoint.save(
                    RepositoryStage.WORKTREE_CREATED,
                    {
                        "worktree": str(worktree),
                        "source": str(source),
                        "goal": goal.to_dict(),
                        "goal_fingerprint": _stable_hash(json.dumps(goal.to_dict(), sort_keys=True)),
                    },
                )
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
                if self.deadline.expired:
                    # Out of time is a distinct outcome from out of ideas, and
                    # it is resumable: the checkpoint and the worktree are on
                    # disk, so a later run picks up where this one stopped.
                    self._record(trajectory, RepositoryStage.EVALUATE, {"cycle": cycle, "reason": "TIME_LIMIT"})
                    result = self._rejected(
                        worktree, trajectory, last_failures, before_benchmarks, last_targeted, last_full, review, cycle - 1
                    )
                    result.status = RepositoryStage.PAUSED
                    result.failure_kind = "time_limit"
                    result.error = (
                        f"time budget of {self.max_seconds:.0f}s exhausted after {self.deadline.elapsed:.0f}s"
                    )
                    result.resume_command = self.resume_command
                    self.heartbeat.finish("time_limit")
                    return result

                self.heartbeat.beat(f"cycle_{cycle}", goal.objective[:120])
                try:
                    proposal = (
                        self._request_patch(goal, context, plan, prior, trajectory)
                        if cycle == 1
                        else self._request_repair(goal, context, plan, last_failures, review, trajectory)
                    )
                    self._apply_proposal(worktree, source, goal, proposal, context)
                except (CallTimeout, DeadlineExceeded) as exc:
                    # The model did not answer in time. That is evidence, not a
                    # reason to stop -- the next cycle re-plans with it recorded.
                    last_failures = [
                        RepositoryCommandResult(["model"], False, stderr=str(exc), return_code=1)
                    ]
                    self._record(
                        trajectory,
                        RepositoryStage.DIAGNOSE,
                        {"cycle": cycle, "reason": "MODEL_TIMEOUT", "detail": str(exc)},
                    )
                    continue
                except EditError as exc:
                    # A patch that could not be applied is an ordinary event in
                    # autonomous development, not the end of the mission: the
                    # next cycle gets the failure as diagnostic evidence and
                    # tries again.  Policy violations are the exception -- they
                    # must terminate the run rather than be retried.
                    if exc.kind in _FATAL_EDIT_KINDS:
                        raise
                    last_failures = [
                        RepositoryCommandResult(
                            ["patch"], False, stderr=f"{exc.kind}: {exc.detail}", return_code=1
                        )
                    ]
                    self._record(
                        trajectory,
                        RepositoryStage.DIAGNOSE,
                        {"cycle": cycle, "reason": "PATCH_NOT_APPLIED", "kind": exc.kind, "detail": exc.detail},
                    )
                    continue
                except ValueError as exc:
                    if _provider_failure_payload(exc):
                        raise
                    last_failures = [
                        RepositoryCommandResult(["patch"], False, stderr=str(exc), return_code=1)
                    ]
                    self._record(
                        trajectory,
                        RepositoryStage.DIAGNOSE,
                        {"cycle": cycle, "reason": "PATCH_GENERATION_FAILED", "detail": str(exc)},
                    )
                    continue

                changed_now = self._assert_changed_files_allowed(
                    worktree,
                    goal,
                )
                if not changed_now:
                    last_failures = [RepositoryCommandResult(["proposal"], False, stderr="NO_EFFECTIVE_CHANGE", return_code=1)]
                    self._record(trajectory, RepositoryStage.DIAGNOSE, {"cycle": cycle, "reason": "NO_EFFECTIVE_CHANGE"})
                    continue
                if self.checkpoint:
                    self.checkpoint.save(f"PATCH_CYCLE_{cycle}", {"proposal": _redact_large(proposal), "diff": self._current_diff(worktree)[-12000:]})
                self._record(trajectory, RepositoryStage.IMPLEMENT if cycle == 1 else RepositoryStage.REPAIR, _redact_large(proposal) | {"cycle": cycle})
                targeted = self._run_commands(worktree, targeted_commands, stage=RepositoryStage.TEST_TARGETED)
                self._assert_changed_files_allowed(worktree, goal)
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
                    self._assert_changed_files_allowed(worktree, goal)
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
                self._assert_changed_files_allowed(worktree, goal)
                last_full = full
                if self.checkpoint:
                    self.checkpoint.save(f"FULL_TESTS_{cycle}", {"results": [item.to_dict() for item in full]})
                self._record(trajectory, RepositoryStage.TEST_FULL, {"cycle": cycle, "results": [item.to_dict() for item in full]})
                if full and not all(item.success for item in full):
                    last_failures = full
                    continue
                after_benchmarks = self._run_commands(worktree, bench_commands, stage=RepositoryStage.BENCHMARK)
                self._assert_changed_files_allowed(worktree, goal)
                if self.checkpoint:
                    self.checkpoint.save(f"BENCHMARK_AFTER_{cycle}", {"results": [item.to_dict() for item in after_benchmarks]})
                self._record(trajectory, RepositoryStage.BENCHMARK, {"when": "after", "cycle": cycle, "results": [item.to_dict() for item in after_benchmarks]})
                benchmark_ok = self._benchmarks_ok(goal, before_benchmarks, after_benchmarks)
                protection_state = self._protected_state(source, worktree, protected_hashes)
                protected_pristine = protection_state == ProtectionState.PRISTINE
                diff = self._current_diff(worktree)
                changed_files = self._assert_changed_files_allowed(
                    worktree,
                    goal,
                )
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
                self.heartbeat.finish(result.status)
                return result
            result = self._rejected(worktree, trajectory, last_failures, before_benchmarks, last_targeted, last_full, review, max_dev_cycles)
            self.heartbeat.finish(result.status)
            return result
        except Exception as exc:
            provider_failure = _provider_failure_payload(exc)
            # Running out of time, and a model that stopped answering, are both
            # PAUSED rather than REJECTED: the worktree and checkpoint are on
            # disk and a later run can carry on. Rejecting them would throw away
            # recoverable work and misreport why it stopped.
            if isinstance(exc, (DeadlineExceeded, CallTimeout)) and self.deadline.expired:
                # Both can be true at once -- the calls timed out *and* the
                # budget ran out. The budget is the outer constraint, so it is
                # the more useful thing to report: "give it longer", not "the
                # model is slow".
                status, failure_kind = RepositoryStage.PAUSED, "time_limit"
            elif isinstance(exc, DeadlineExceeded):
                status, failure_kind = RepositoryStage.PAUSED, "time_limit"
            elif isinstance(exc, CallTimeout):
                status, failure_kind = RepositoryStage.PAUSED, "model_timeout"
            elif provider_failure:
                status, failure_kind = RepositoryStage.PAUSED, provider_failure.get("kind", "")
            else:
                status, failure_kind = RepositoryStage.REJECTED, ""
            self.heartbeat.finish(failure_kind or "rejected")
            result = RepositoryCandidateResult(
                status=status,
                worktree=str(worktree or ""),
                protected_pristine=True if worktree is not None else False,
                protection_state=ProtectionState.NOT_EVALUATED,
                error=str(exc),
                failure_kind=failure_kind,
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
        return self.read_file_complete(repository_path, relative_path, max_chars=max_chars)[0]

    def read_file_complete(
        self, repository_path: str | Path, relative_path: str, *, max_chars: int = 8000
    ) -> tuple[str, bool]:
        """Read a file, and say whether the caller got all of it.

        Callers that record a staleness hash need to know: a digest taken over
        a truncated read can never match the real file again.
        """

        root = Path(repository_path).resolve()
        path = self._safe_read_path(root, relative_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars], len(text) <= max_chars

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
                    manager.add_file(
                        observation["path"],
                        observation.get("content", ""),
                        complete=bool(observation.get("complete", True)),
                    )
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
                    content, complete = self.read_file_complete(worktree, path)
                    manager.add_file(path, content, complete=complete)
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
                content, complete = self.read_file_complete(worktree, path)
                return {"tool": tool, "path": path, "content": content, "complete": complete}
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
        worktree = Path(trajectory["worktree"])
        focused = _focused_edit_context(worktree, goal, context, max_chars=self._focus_budget())

        plan_text = json.dumps(plan, indent=2, sort_keys=True)
        if len(plan_text) > 2200:
            plan_text = plan_text[:2200] + "\n...[truncated]"

        prior_text = json.dumps(_prior_for_prompt(prior), indent=2, sort_keys=True)
        if len(prior_text) > 900:
            prior_text = prior_text[:900] + "\n...[truncated]"

        prompt = (
            "Return JSON only. Role: Repository Implementer.\n"
            "Produce the smallest safe edit that satisfies the goal.\n"
            "For EXISTING files use exact search/replace edits only.\n"
            "The search string should normally be only 1-3 lines and MUST come from the FOCUSED CURRENT CODE below.\n"
            "Do not reconstruct or rewrite an existing file. Change only the smallest necessary token or lines.\n"
            "Prefer one edit. Maximum 8 edits. Never modify protected paths.\n"
            "If a requested literal is new, anchor the edit on nearby literals that already exist in the current code.\n"
            f"Goal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Plan:\n{plan_text}\n"
            f"FOCUSED CURRENT CODE:\n{focused}\n"
            f"Relevant prior memory:\n{prior_text}\n"
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
        worktree = Path(trajectory["worktree"])
        diff = self._current_diff(worktree)
        focused = _focused_edit_context(worktree, goal, context, max_chars=self._focus_budget())

        failure_text = json.dumps(
            [item.to_dict() for item in failures],
            indent=2,
            sort_keys=True,
        )
        if len(failure_text) > 3500:
            failure_text = failure_text[-3500:]

        review_text = json.dumps(review.to_dict(), indent=2, sort_keys=True)
        if len(review_text) > 2200:
            review_text = review_text[-2200:]

        prompt = (
            "Return JSON only. Role: Repository Repairer.\n"
            "Repair the current candidate with the smallest possible search/replace edit.\n"
            "Use CURRENT CODE below as ground truth. Never guess an old version of a line.\n"
            "Search strings should normally be 1-3 lines copied from CURRENT CODE.\n"
            "Do not rewrite complete existing files.\n"
            "Do not weaken tests, hidden verifiers, benchmarks, safety policy, or protected paths.\n"
            f"Goal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Current diff:\n{diff[-5000:]}\n"
            f"Failures:\n{failure_text}\n"
            f"Reviewer findings:\n{review_text}\n"
            f"FOCUSED CURRENT CODE:\n{focused}\n"
        )
        return self._generate_patch_bundle(prompt)



    def _generate_patch_bundle(self, prompt: str) -> dict[str, Any]:
        schema = _patch_schema()
        stage = RepositoryStage.REPAIR if "Repository Repairer" in prompt else RepositoryStage.IMPLEMENT
        payload = self._generate_json(
            prompt,
            schema,
            stage=stage,
            desired_output_tokens=750,
            temperature=0.0,
        )
        if not isinstance(payload.get("files"), list):
            raise ValueError("repository proposal did not contain edits")
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
        timeouts = 0
        for attempt in range(1, self.structured_regeneration_attempts + 2):
            budgeted_prompt, max_tokens, budget_record = self.context_budget.prepare(stage, prompt, desired_output_tokens)
            try:
                raw = self._generate_structured(budgeted_prompt, schema, max_tokens=max_tokens, temperature=temperature)
            except CallTimeout as exc:
                # A call that never answers is a failed attempt, exactly like a
                # malformed one, and belongs to this retry loop. Left to escape,
                # a single slow generation during investigation ended the whole
                # run, while a malformed one at the same stage was merely
                # retried -- an indefensible difference.
                timeouts += 1
                last_raw = str(exc)
                self.heartbeat.beat(stage, f"model call timed out ({timeouts})")
                continue
            last_raw = raw
            try:
                data = lenient_json_loads(_extract_json(raw))
                if isinstance(data, dict):
                    data.setdefault("_budget", budget_record)
                    return data
            except json.JSONDecodeError:
                pass
            prompt = prompt + "\n\nPrevious response was malformed. Regenerate JSON only for the schema."
        if timeouts:
            # Distinguish "the model said something unusable" from "the model
            # said nothing at all"; only the second is a stall.
            raise CallTimeout(f"{stage} generation", self.model_call_timeout_seconds)
        raise ValueError(f"structured repository generation failed: {_stable_hash(last_raw)}")

    def _generate_structured(self, prompt: str, schema: dict[str, Any], *, max_tokens: int, temperature: float) -> str:
        """Call the model under a hard time bound.

        Self-development had no wall-clock bound of any kind: a single wedged
        generation could hold the whole run indefinitely, and no checkpoint
        would ever be written to show it. The bound is enforced here rather than
        left to the HTTP layer, whose timeouts are per-read -- with a
        non-streaming local model, "still thinking" and "never coming back" look
        the same on the socket.
        """

        self.deadline.require()
        timeout = self.deadline.clamp(self.model_call_timeout_seconds)

        def invoke() -> str:
            if hasattr(self.brain, "generate_structured"):
                try:
                    return self.brain.generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
                except (AttributeError, NotImplementedError, StructuredGenerationUnsupported):
                    pass
            if hasattr(self.brain, "generate_coding"):
                return self.brain.generate_coding(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
            return self.brain.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.9)

        return call_with_timeout(
            invoke,
            timeout,
            what="model call",
            on_abandon=lambda: self.heartbeat.beat("model_timeout", f"abandoned a call after {timeout:.0f}s"),
        )

    def _ensure_plan_files_read(self, worktree: Path, context: dict[str, Any], plan: dict[str, Any]) -> None:
        for relative in [str(item) for item in plan.get("files_to_change", []) if item]:
            if relative in context.get("inspected_files", {}):
                continue
            path = self._safe_read_path(worktree, relative)
            content = path.read_text(encoding="utf-8", errors="replace")
            context.setdefault("inspected_files", {})[relative] = content[:8000]
            context.setdefault("file_hashes", {})[relative] = _stable_hash(content)

    def _focus_budget(self) -> int:
        """How much source to show the model when it composes an edit.

        Scaled to the context the model actually has. The 5000-character window
        this replaces was sized for an 8192-token context and stayed put after
        the tuner measured 24576 usable on this machine -- so the model was
        asked to copy an exact anchor out of a keyhole view of a 430-line file
        and missed thirteen times running. Measuring the machine and then
        prompting as though it had not been measured is the worst of both.

        Roughly 40% of the window: the rest is goal, plan, failures and the
        model's own output.
        """

        window = max(4096, int(self.context_budget.context_window))
        chars = int(window * self.context_budget.chars_per_token * 0.4)
        return max(5000, min(chars, 48000))

    def _edit_engine(
        self,
        worktree: Path,
        source: Path,
        goal: SelfImprovementGoal,
        context: dict[str, Any],
    ) -> EditEngine:
        policy = PathPolicy(
            worktree,
            allowed_paths=goal.allowed_paths,
            protected_paths=goal.protected_paths,
            live_root=source,
        )
        return EditEngine(
            policy,
            budget=EditBudget.from_env(os.getenv),
            expected_hashes=dict(context.get("file_hashes", {})),
        )

    def _apply_proposal(
        self,
        worktree: Path,
        source: Path,
        goal: SelfImprovementGoal,
        proposal: dict[str, Any],
        context: dict[str, Any],
    ) -> EditResult:
        """Apply a model-proposed change bundle, correcting it if it misses.

        The deterministic work lives in :mod:`development.edit_engine`; what
        this method adds is the *recovery* loop.  A local model routinely gets
        an anchor slightly wrong on the first try, and treating that as a failed
        mission would make the whole system useless.  So a recoverable
        :class:`EditError` -- a stale, ambiguous or unmatched anchor -- is fed
        back to the model together with the current file content, up to
        ``JARVIS_BUILD_PATCH_CORRECTIONS`` times.

        Non-recoverable errors (a protected path, an escape attempt, a blown
        budget) are re-raised immediately: those are policy decisions, and
        asking the model to try again would only be asking it to attack the
        boundary a second time.
        """

        corrections = max(0, int(os.getenv("JARVIS_BUILD_PATCH_CORRECTIONS", "3")))
        current_bundle = proposal
        attempt = 0

        while True:
            attempt += 1
            engine = self._edit_engine(worktree, source, goal, context)
            try:
                plan = parse_bundle(current_bundle)
                result = engine.apply(plan)
            except EditError as exc:
                if not exc.recoverable or attempt > corrections:
                    raise
                self._record(
                    self._active_trajectory,
                    RepositoryStage.DIAGNOSE,
                    {"patch_correction": attempt, "kind": exc.kind, "detail": exc.detail},
                )
                try:
                    current_bundle = self._request_patch_correction(
                        worktree, goal, context, current_bundle, exc
                    )
                except EditError:
                    raise
                except Exception:
                    # If we cannot even ask for a correction (no brain wired,
                    # provider down, malformed response), the useful thing to
                    # report is the edit problem itself, not the failure of the
                    # attempt to fix it.
                    raise exc from None
                continue

            if current_bundle is not proposal:
                # Surface the bundle that actually landed, so trajectory logs
                # and checkpoints describe reality rather than the first guess.
                proposal.clear()
                proposal.update(current_bundle)

            self._sync_context_after_edit(worktree, context, result)
            self._clear_python_caches(worktree)
            return result

    def _request_patch_correction(
        self,
        worktree: Path,
        goal: SelfImprovementGoal,
        context: dict[str, Any],
        bundle: dict[str, Any],
        error: EditError,
    ) -> dict[str, Any]:
        focused = _focused_edit_context(worktree, goal, context, max_chars=self._focus_budget())
        previous = json.dumps(bundle, indent=2, sort_keys=True)
        if len(previous) > 3000:
            previous = previous[:3000] + "\n...[truncated]"

        prompt = (
            "Return JSON only. Role: Repository Patch Corrector.\n"
            "Your previous patch was NOT written to disk because it could not be applied safely.\n"
            "Nothing has changed on disk. Produce a corrected patch.\n"
            f"Failure kind: {error.kind}\n"
            f"Failure detail: {error.detail}\n"
            f"{_correction_hint(error)}\n"
            "Copy every search string CHARACTER-FOR-CHARACTER from CURRENT CODE below.\n"
            "CURRENT CODE is the absolute ground truth. Never guess an older version of a line.\n"
            "Do not rewrite whole files. Do not touch protected paths.\n"
            f"Goal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Previous rejected patch:\n{previous}\n"
            f"CURRENT CODE:\n{focused}\n"
        )
        return self._generate_patch_bundle(prompt)

    def _sync_context_after_edit(
        self, worktree: Path, context: dict[str, Any], result: EditResult
    ) -> None:
        """Re-read edited files so the next prompt sees post-edit reality."""

        inspected = context.setdefault("inspected_files", {})
        hashes = context.setdefault("file_hashes", {})

        for applied in result.applied:
            if applied.deleted:
                inspected.pop(applied.path, None)
                hashes.pop(applied.path, None)
                continue
            path = worktree / applied.path
            try:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            inspected[applied.path] = text[:8000]
            hashes[applied.path] = _stable_hash(text)

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
                env=_safe_command_env(worktree),
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
        changed: set[str] = set()

        commands = [
            ["diff", "--name-only", "--", "."],
            ["diff", "--cached", "--name-only", "--", "."],
            ["ls-files", "--others", "--exclude-standard", "--", "."],
        ]

        for args in commands:
            completed = self._git(worktree, args, check=False)

            for line in completed.stdout.splitlines():
                relative = line.strip().replace("\\", "/")

                if relative:
                    changed.add(relative)

        return sorted(changed)

    def _assert_changed_files_allowed(
        self,
        worktree: Path,
        goal: SelfImprovementGoal,
    ) -> list[str]:
        """
        Independent fail-closed write-scope verification.

        _safe_path() protects normal proposal writes. This second gate
        verifies the actual Git-visible result so future writer bugs,
        newly created files, or other side effects cannot silently
        escape the declared write scope.
        """
        changed = self._changed_files(worktree)
        violations: list[str] = []

        for relative in changed:
            if _path_matches(relative, goal.protected_paths):
                violations.append(f"{relative} (protected)")
                continue

            if (
                goal.allowed_paths
                and not _path_matches(relative, goal.allowed_paths)
            ):
                violations.append(
                    f"{relative} (outside allowed_paths)"
                )

        if violations:
            raise ValueError(
                "write scope violation: "
                + ", ".join(sorted(violations))
            )

        return changed

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
        """Persist the diff and the full trajectory next to -- not inside -- the worktree.

        These files are Jarvis's own output, not part of the candidate.  Written
        inside the worktree they would show up in ``git diff`` as candidate
        content, and on a resumed run the leftovers from the previous attempt
        would trip the write-scope gate before the first patch is even applied.
        """

        worktree = Path(result.worktree)
        root = worktree.parent / f"{worktree.name}_artifacts"
        root.mkdir(parents=True, exist_ok=True)
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
                hashes[raw.as_posix()] = _protection_digest(path)
            elif path.exists() and path.is_dir():
                root_key = raw.as_posix().rstrip("/") + "/"
                entries: list[str] = []
                for child in sorted(path.rglob("*")):
                    if child.is_file() and _visible_repo_file(child, source):
                        rel = child.relative_to(source).as_posix()
                        entries.append(rel)
                        hashes[rel] = _protection_digest(child)
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
            if (
                not candidate_path.exists()
                or _protection_digest(candidate_path) != expected
            ):
                return ProtectionState.MODIFIED

            if (
                source_path.exists()
                and _protection_digest(source_path) != expected
            ):
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


def _focused_edit_context(worktree, goal, context, *, max_chars=5000):
    import re

    root = Path(worktree).resolve()
    candidates = []

    def add_candidate(relative):
        relative = str(relative).replace("\\", "/").strip()
        if not relative:
            return

        candidate = (root / relative).resolve()

        try:
            candidate.relative_to(root)
        except ValueError:
            return

        if candidate.is_file():
            pair = (relative, candidate)
            if pair not in candidates:
                candidates.append(pair)

    for relative in getattr(goal, "allowed_paths", []) or []:
        add_candidate(relative)

    for relative in context.get("inspected_files", {}).keys():
        add_candidate(relative)

    goal_text = json.dumps(goal.to_dict(), sort_keys=True)

    tokens = re.findall(
        r"/[A-Za-z0-9_.-]+|[A-Za-z_][A-Za-z0-9_.-]{2,}",
        goal_text,
    )

    stopwords = {
        "implementiere", "unterst?tzung", "zusaetzlichen", "zus?tzlichen",
        "befehl", "veraendere", "ver?ndere", "ausschliesslich",
        "ausschlie?lich", "halte", "minimal", "andere", "funktionalitaet",
        "funktionalit?t", "repository", "change", "changes", "implement",
        "support", "only", "file", "files", "goal", "allowed_paths",
        "protected_paths", "objective",
    }

    keywords = []
    for token in tokens:
        lowered = token.lower()
        if lowered in stopwords:
            continue
        if lowered not in keywords:
            keywords.append(lowered)

    # Deterministic navigation first: software finds the code, the model edits
    # it.  Lexical scoring cannot separate `if word in {"/quit", "/bye"}` from a
    # help string listing the same words, and when it guesses wrong the model
    # never sees the line it was asked to change -- the observed cause of four
    # consecutive failed self-patch runs.  The AST can separate them for free.
    indexed = _indexed_regions(root, keywords, candidates, max_chars)
    if indexed:
        return indexed

    chunks = []
    remaining = max_chars

    for relative, path in candidates[:12]:
        if remaining <= 0:
            break

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = text.splitlines()

        if not lines:
            continue

        # Show the whole file whenever it fits in what is left of the budget.
        #
        # This threshold used to be a hard-coded 3200 characters and never moved
        # when the budget grew to 39k. The consequence was not merely wasteful,
        # it was wrong: range selection ranks lines by keyword overlap, and the
        # help text at the top of the CLI mentions "/quit /exit /bye leave" more
        # densely than the code that implements them. Asked twice to add an exit
        # word, the model twice edited the documentation, because that is the
        # only part of the file it was shown.
        if len(text) <= max(3200, remaining):
            selected_ranges = [(0, len(lines))]
        else:
            scored = []

            for index, line in enumerate(lines):
                lowered = line.lower()
                score = 0

                for keyword in keywords:
                    if keyword in lowered:
                        score += 4 if keyword.startswith("/") else 1

                if score:
                    scored.append((score, index))

            scored.sort(key=lambda item: (-item[0], item[1]))

            hit_indices = [index for _, index in scored[:6]]

            if not hit_indices:
                hit_indices = [0, max(0, len(lines) - 1)]

            ranges = []
            for index in sorted(hit_indices):
                start = max(0, index - 9)
                end = min(len(lines), index + 10)

                if ranges and start <= ranges[-1][1] + 2:
                    ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
                else:
                    ranges.append((start, end))

            selected_ranges = ranges

        for start, end in selected_ranges:
            if remaining <= 0:
                break

            # Deliberately NOT line-numbered.  The prompt asks the model to
            # copy search anchors character-for-character out of this block, and
            # a numeric gutter is copied along with everything else -- producing
            # anchors that can never match the real file.  The line range lives
            # in the header instead, where it is informative but uncopyable.
            body = "\n".join(lines[index] for index in range(start, end))

            block = (
                f"\n--- {relative} lines {start + 1}-{end} ---\n"
                f"{body}\n"
                f"--- end {relative} ---\n"
            )

            if len(block) > remaining:
                block = block[:remaining]

            chunks.append(block)
            remaining -= len(block)

    if not chunks:
        return "(no focused editable code found)"

    return "".join(chunks)


def _indexed_regions(root, keywords, candidates, max_chars):
    """Select source regions by syntactic role, and say why each was chosen.

    Returns "" when the index has nothing to offer -- no Python candidates, no
    hits -- so the caller falls back to plain lexical selection rather than
    handing the model an empty block.
    """

    python_paths = [relative for relative, path in candidates if str(path).endswith(".py")]
    if not python_paths:
        return ""

    try:
        from development.code_index import CodeIndex, Role
    except ImportError:  # pragma: no cover - defensive
        return ""

    index = CodeIndex(root)

    # Literal terms -- "/goodbye", "--verbose", "SOME_KEY" -- are what a goal is
    # usually phrased in, and they are exactly what plain search over-matches.
    # Path components ("jarvis", "cli") are excluded: they match every line in
    # the file and so distinguish nothing, which is the definition of noise.
    path_words = {part.lower() for relative in python_paths for part in re.split(r"[/._]", relative) if part}
    literals = [
        word
        for word in keywords
        if (word.startswith("/") or "_" in word or len(word) > 5) and word.lower() not in path_words
    ]

    # Command-like tokens first: "/quit" is the kind of term a goal hinges on,
    # while "implementiere" is filler that happens to be long.  Then keep
    # scanning until three terms have actually RESOLVED -- taking the first
    # three off the list reported nothing at all when the three leading terms
    # had no hits and the term that mattered sat fourth.
    literals.sort(key=lambda word: (not word.startswith("/"), -len(word)))

    report_lines = []
    for term in literals[:12]:
        if len(report_lines) >= 3:
            break
        term = term.rstrip(".,;:!?)")
        occurrences = index.find_literal(term, paths=python_paths, limit=12)
        if not occurrences:
            continue
        executable = [item for item in occurrences if item.role.executable]
        prose = [item for item in occurrences if not item.role.executable]
        if executable and prose:
            # The one sentence that would have prevented four failed runs.
            report_lines.append(
                f"'{term}' appears as EXECUTABLE CODE at "
                + ", ".join(f"{item.path}:{item.line}" for item in executable[:4])
                + " and as NON-EXECUTABLE TEXT (help/docs/comments) at "
                + ", ".join(f"{item.path}:{item.line}" for item in prose[:4])
                + ". Change the executable code; editing the text will not work."
            )

    header = ""
    if report_lines:
        header = "REPOSITORY INDEX (deterministic, trust this over your own search):\n" + "\n".join(
            f"  {line}" for line in report_lines
        ) + "\n"

    regions = index.regions_for_terms(
        keywords,
        paths=python_paths,
        budget_chars=max(1000, max_chars - len(header)),
        max_regions=6,
    )
    if not regions:
        return ""

    # If the best region is real code, drop pure-prose regions entirely: showing
    # the help text alongside the branch is what invited the model to edit it.
    if regions[0].role >= Role.FUNCTION_CODE:
        executable_regions = [region for region in regions if region.role.executable]
        if executable_regions:
            regions = executable_regions

    return header + "".join(region.render() for region in regions)


def _patch_schema() -> dict[str, Any]:
    """Change-bundle schema for guided generation.

    Defined once in :mod:`development.edit_engine` so the schema the model
    is constrained to and the parser that consumes its output can never
    drift apart.

    Whole-file rewrites are excluded. Repository work means changing files that
    already exist and are usually far longer than a small model can reproduce
    faithfully; leaving the field in the schema let it answer a one-line request
    with a three-line "rewrite" of a 189-line module, over and over. New files
    still go through new_files, which is unaffected.
    """

    return edit_schema(allow_rewrite=False)


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


def _protection_digest(path: Path) -> str:
    """
    Hash protected repository content deterministically across Git
    worktrees on Windows.

    Text files treat CRLF and LF as equivalent because Git may materialize
    different working-tree line endings without changing repository
    content. Binary files remain byte-exact.
    """
    data = path.read_bytes()

    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n")

    return sha256(data).hexdigest()


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


def _safe_command_env(cwd: Path | None = None) -> dict[str, str]:
    """A scrubbed environment for candidate test runs.

    Scrubbed, not starved.  An earlier version passed through only PATH and a
    few Windows basics, which meant a Python installed with user-site packages
    could not find them: ``python -m pytest`` answered "No module named pytest"
    and every candidate was rejected for a reason that had nothing to do with
    the candidate.  ``APPDATA`` is where Windows keeps the per-user site
    directory, so it has to be here.

    Letting user site-packages back in brings its own hazard, and this machine
    demonstrated it immediately: something installed there ships a top-level
    package called ``tests``, which shadowed a candidate's own ``tests`` package
    and made its suite unimportable.  ``PYTHONPATH`` therefore pins the
    candidate's own directory to the front of ``sys.path``, so a candidate's
    modules always win over whatever happens to be installed on the host.  When
    testing a candidate, that is the only defensible precedence.

    What stays out is anything credential-shaped.  A subprocess a model chose to
    run has no business seeing an API key.
    """

    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        # Needed to resolve the per-user site-packages directory.
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "LANG",
        "LC_ALL",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "JARVIS_BRAIN_PROVIDER",
        "JARVIS_BRAIN_BASE_URL",
        "JARVIS_BRAIN_MODEL",
        "JARVIS_BRAIN_TIMEOUT",
        "JARVIS_BRAIN_TEMPERATURE",
        "JARVIS_BRAIN_TOP_P",
        "JARVIS_BRAIN_MAX_TOKENS",
        "JARVIS_BRAIN_RETRIES",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if cwd is not None:
        env["PYTHONPATH"] = str(Path(cwd).resolve())
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
