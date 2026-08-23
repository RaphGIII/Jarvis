"""The autonomous control loop.

    GOAL -> INVESTIGATE -> DECOMPOSE -> PLAN -> EXECUTE -> OBSERVE
         -> VERIFY -> DIAGNOSE -> REPLAN -> ... -> ACCEPT

The loop's whole reason for existing is captured by one rule: **a failed attempt
is not a failed task, and a failed task is not a failed project.**  A small local
model gets things wrong constantly -- a mistyped anchor, a wrong import, a test
that fails for a reason it did not anticipate.  Each of those is evidence to act
on, not a reason to stop.  The loop therefore only terminates for the four
reasons the mission allows:

* the acceptance criteria are objectively satisfied (:attr:`StopReason.ACCEPTED`),
* the user stopped it (:attr:`StopReason.CANCELLED`),
* a configured budget ran out (STEP_LIMIT / TIME_LIMIT / FAILURE_LIMIT),
* something genuinely external is blocking it (:attr:`StopReason.BLOCKED`).

Three design choices are worth stating outright, because each is load-bearing:

*Deterministic verification only.*  A project is accepted when its acceptance
commands exit zero, never because the model said the work looked done.  This is
what makes "success" mean something.

*The model proposes, tools dispose.*  Every effect on the world goes through
:class:`~tools.registry.ToolRegistry`, so permissions, timeouts and the audit
trail apply uniformly and a prompt-injected instruction cannot reach past them.

*Everything durable lives in the project.*  After each step the project is saved,
so killing the process mid-run loses at most the current step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from brain.json_utils import lenient_json_loads
from projects.models import (
    Phase,
    Project,
    ProjectState,
    ResourceLimits,
    StepRecord,
    StopReason,
    Task,
    TaskStatus,
)
from projects.store import ProjectStore
from tools.registry import ToolCall, ToolContext, ToolRegistry, ToolResult


@dataclass
class SessionResult:
    """What one call to :meth:`ProjectEngine.run` achieved."""

    project: Project
    stop_reason: StopReason
    steps: int = 0
    seconds: float = 0.0
    accepted: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project.id,
            "stop_reason": self.stop_reason.value,
            "steps": self.steps,
            "seconds": round(self.seconds, 1),
            "accepted": self.accepted,
            "message": self.message,
            "progress": self.project.progress(),
        }


@dataclass
class EngineHooks:
    """Optional callbacks, so a UI can watch a run without the engine knowing about UIs."""

    on_step: Callable[[Project, StepRecord], None] | None = None
    on_phase: Callable[[Project, Phase], None] | None = None
    #: Return True to stop the loop at the next step boundary.
    should_cancel: Callable[[], bool] | None = None


class ProjectEngine:
    """Drives a :class:`Project` toward its acceptance criteria."""

    def __init__(
        self,
        *,
        brain: Any,
        store: ProjectStore,
        tools: ToolRegistry,
        hooks: EngineHooks | None = None,
        verifier: Callable[[Project, ToolContext], list[tuple[str, bool, str]]] | None = None,
        max_tool_calls_per_step: int = 4,
    ) -> None:
        self.brain = brain
        self.store = store
        self.tools = tools
        self.hooks = hooks or EngineHooks()
        self.max_tool_calls_per_step = max_tool_calls_per_step
        self._verify = verifier or self._default_verifier

    # ------------------------------------------------------------------
    # Project creation
    # ------------------------------------------------------------------

    def create_project(
        self,
        goal: str,
        *,
        kind: str = "software",
        title: str = "",
        workspace: str | None = None,
        repository: str = "",
        limits: ResourceLimits | None = None,
        acceptance: list[tuple[str, list[str]]] | None = None,
        constraints: list[str] | None = None,
    ) -> Project:
        project = Project(goal=goal.strip(), kind=kind, title=title or _title_from(goal))
        project.repository = repository
        project.limits = limits or ResourceLimits()
        project.constraints = list(constraints or [])
        project.add_requirement(goal.strip(), source="user")
        for text, check in acceptance or []:
            project.add_acceptance(text, check=check)
        if workspace:
            project.workspace = str(workspace)
        self.store.workspace_for(project)
        self.store.save(project)
        return project

    def add_requirement(self, project: Project, text: str) -> Project:
        """Accept a new requirement mid-project.

        A project that already believed itself finished reopens: the user has
        moved the goalposts, which is the normal way an incrementally specified
        system grows, not an error.
        """

        project.add_requirement(text, source="user")
        if project.state in {ProjectState.COMPLETED, ProjectState.PAUSED, ProjectState.BLOCKED}:
            project.state = ProjectState.PLANNING
        self.store.save(project)
        return project

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self, project: Project, *, context: ToolContext | None = None, max_steps: int | None = None) -> SessionResult:
        """Work on the project until it is accepted, blocked, or out of budget."""

        started = time.perf_counter()
        limits = project.limits
        budget = min(max_steps or limits.max_steps, limits.max_steps)
        context = context or self.tool_context(project)
        consecutive_failures = 0
        steps = 0

        if project.state is ProjectState.DRAFT:
            project.state = ProjectState.INVESTIGATING

        stop = StopReason.STEP_LIMIT
        message = ""

        try:
            while steps < budget:
                if self.hooks.should_cancel and self.hooks.should_cancel():
                    stop, message = StopReason.CANCELLED, "stopped by the user"
                    break
                elapsed = time.perf_counter() - started
                if elapsed >= limits.max_seconds:
                    stop, message = StopReason.TIME_LIMIT, f"time budget of {limits.max_seconds:.0f}s exhausted"
                    break
                if consecutive_failures >= limits.max_consecutive_failures:
                    stop, message = (
                        StopReason.FAILURE_LIMIT,
                        f"{consecutive_failures} consecutive steps without progress",
                    )
                    break
                if project.user_blockers():
                    stop, message = StopReason.BLOCKED, "; ".join(item.text for item in project.user_blockers())
                    break

                phase = self._next_phase(project)
                if self.hooks.on_phase:
                    self.hooks.on_phase(project, phase)

                step = self._run_phase(phase, project, context)
                steps += 1
                project.record_step(step)
                project.seconds_spent += step.duration_seconds
                # Productivity, not success: see StepRecord.productive.
                consecutive_failures = 0 if step.productive else consecutive_failures + 1

                self.store.save(project)
                if self.hooks.on_step:
                    self.hooks.on_step(project, step)

                if project.state is ProjectState.COMPLETED:
                    stop, message = StopReason.ACCEPTED, "all objective acceptance criteria pass"
                    break
                if project.state is ProjectState.BLOCKED and project.user_blockers():
                    stop, message = StopReason.BLOCKED, "; ".join(item.text for item in project.user_blockers())
                    break
            else:
                message = f"step budget of {budget} exhausted"
        except Exception as exc:  # a loop crash must still leave a saved, resumable project
            stop = StopReason.ERROR
            message = f"{type(exc).__name__}: {exc}"
            project.add_blocker(f"engine error: {message}", needs_user=True)

        seconds = time.perf_counter() - started
        accepted = project.state is ProjectState.COMPLETED

        if not accepted and not project.state.terminal:
            project.state = ProjectState.BLOCKED if stop is StopReason.BLOCKED else ProjectState.PAUSED

        project.last_stop_reason = stop.value
        self.store.save(project)

        return SessionResult(
            project=project, stop_reason=stop, steps=steps, seconds=seconds, accepted=accepted, message=message
        )

    def tool_context(self, project: Project) -> ToolContext:
        workspace = self.store.workspace_for(project)
        readable = [workspace]
        if project.repository:
            readable.append(project.repository)
        return ToolContext(
            workspace=workspace,
            readable_roots=[__import__("pathlib").Path(item) for item in readable],
            timeout_seconds=project.limits.step_timeout_seconds,
            protected_paths=list(project.metadata.get("protected_paths") or []),
            allowed_paths=list(project.metadata.get("allowed_paths") or []),
        )

    # ------------------------------------------------------------------
    # Phase selection
    # ------------------------------------------------------------------

    def _next_phase(self, project: Project) -> Phase:
        """Choose the next phase from the project's own state.

        Deliberately deterministic.  Letting the model pick its own next phase
        sounds flexible but in practice lets a weak model loop forever between
        planning and re-planning without ever executing anything.
        """

        if not project.findings and not project.tasks:
            return Phase.INVESTIGATE
        if not project.acceptance:
            return Phase.DECOMPOSE
        if not project.tasks:
            return Phase.DECOMPOSE

        # Everything actionable is finished: check the criteria for real.
        if not project.ready_tasks():
            if any(task.open for task in project.tasks):
                # Only blocked or exhausted work remains.
                return Phase.REPLAN
            return Phase.VERIFY

        last = project.steps[-1] if project.steps else None
        if last and last.phase is Phase.EXECUTE:
            return Phase.VERIFY if last.success else Phase.DIAGNOSE
        if last and last.phase is Phase.DIAGNOSE:
            return Phase.EXECUTE
        if last and last.phase is Phase.VERIFY and not last.success:
            return Phase.DIAGNOSE
        return Phase.EXECUTE

    def _run_phase(self, phase: Phase, project: Project, context: ToolContext) -> StepRecord:
        started = time.perf_counter()
        handler = {
            Phase.INVESTIGATE: self._phase_investigate,
            Phase.DECOMPOSE: self._phase_decompose,
            Phase.PLAN: self._phase_decompose,
            Phase.EXECUTE: self._phase_execute,
            Phase.VERIFY: self._phase_verify,
            Phase.DIAGNOSE: self._phase_diagnose,
            Phase.REPLAN: self._phase_replan,
        }[phase]
        try:
            step = handler(project, context)
        except Exception as exc:
            step = StepRecord(phase=phase, summary=f"{type(exc).__name__}: {exc}", success=False, productive=False)
        step.phase = phase
        if not step.success and step.phase is not Phase.VERIFY:
            # Invariant: a step that failed did not advance the work.  VERIFY is
            # the one exception -- it reports failure while still having newly
            # proved a criterion that was failing before, which is real progress.
            step.productive = False
        step.duration_seconds = time.perf_counter() - started
        return step

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _phase_investigate(self, project: Project, context: ToolContext) -> StepRecord:
        """Look at the world before planning against it."""

        project.state = ProjectState.INVESTIGATING
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "maxItems": self.max_tool_calls_per_step,
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                        "required": ["name"],
                    },
                },
                "findings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tool_calls"],
        }
        prompt = (
            "Return JSON only. You are investigating before writing any code.\n"
            "Choose up to "
            f"{self.max_tool_calls_per_step} tool calls that tell you what you still need to know.\n"
            "Prefer listing files and reading the ones that matter. Do not guess at file contents.\n\n"
            f"{self._project_brief(project)}\n"
            f"Available tools:\n{self.tools.render_for_prompt(tags=['investigate', 'environment', 'research'])}\n"
        )
        payload = self._ask_json(prompt, schema, max_tokens=700)
        calls = payload.get("tool_calls") or []

        if not calls:
            # A model that proposes nothing still needs the loop to make
            # progress, so fall back to the single most useful observation.
            calls = [{"name": "list_files", "arguments": {}}]

        results = self._invoke_all(calls, context)
        for text in payload.get("findings") or []:
            project.add_finding(str(text), source="model", confidence=0.5)
        for result in results:
            if result.ok:
                project.add_finding(
                    f"{result.name}({_brief_args(result.arguments)}) -> {_brief_output(result.output)}",
                    source="observation",
                    reference=result.name,
                    confidence=0.95,
                )
        summary = f"investigated with {len(results)} tool call(s); {len(project.findings)} findings"
        return StepRecord(
            phase=Phase.INVESTIGATE,
            summary=summary,
            success=any(result.ok for result in results),
            tool_calls=[result.to_dict() for result in results],
        )

    def _phase_decompose(self, project: Project, context: ToolContext) -> StepRecord:
        """Turn the goal plus what we now know into tasks and checkable criteria."""

        project.state = ProjectState.PLANNING
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "acceptance": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}, "check": {"type": "array", "items": {"type": "string"}}},
                        "required": ["text"],
                    },
                },
                "tasks": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
                        "required": ["title"],
                    },
                },
                "decisions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tasks"],
        }
        prompt = (
            "Return JSON only. Break this goal into a small number of concrete tasks.\n"
            "Each task must be something one focused change can accomplish.\n"
            "Also give acceptance criteria. Every criterion SHOULD carry a `check`: an executable\n"
            'command array such as ["python", "-m", "pytest", "-q", "tests/test_x.py"].\n'
            "A criterion without a runnable check can never be proved, so it will not count.\n\n"
            f"{self._project_brief(project)}\n"
        )
        payload = self._ask_json(prompt, schema, max_tokens=1100)

        added_tasks = 0
        existing_titles = {task.title.strip().lower() for task in project.tasks}
        for item in payload.get("tasks") or []:
            title = str(item.get("title", "")).strip()
            if not title or title.lower() in existing_titles:
                continue
            project.add_task(title, detail=str(item.get("detail", "")), kind=project.kind)
            existing_titles.add(title.lower())
            added_tasks += 1

        added_criteria = 0
        existing_criteria = {item.text.strip().lower() for item in project.acceptance}
        for item in payload.get("acceptance") or []:
            text = str(item.get("text", "")).strip()
            if not text or text.lower() in existing_criteria:
                continue
            check = [str(part) for part in (item.get("check") or []) if str(part).strip()]
            project.add_acceptance(text, check=check)
            existing_criteria.add(text.lower())
            added_criteria += 1

        for text in payload.get("decisions") or []:
            project.add_decision(str(text), rationale="chosen during decomposition")

        if not project.tasks:
            # Never leave the loop with nothing to execute: fall back to the
            # goal itself as a single task rather than spinning on planning.
            project.add_task(project.goal[:120], detail=project.goal, kind=project.kind)
            added_tasks = 1

        if not project.objective_criteria():
            project.add_blocker(
                "no acceptance criterion has a runnable check, so success cannot be proved", needs_user=False
            )

        return StepRecord(
            phase=Phase.DECOMPOSE,
            summary=f"added {added_tasks} task(s) and {added_criteria} acceptance criterion(a)",
            success=added_tasks > 0 or added_criteria > 0,
            detail={"tasks": added_tasks, "acceptance": added_criteria},
        )

    def _phase_execute(self, project: Project, context: ToolContext) -> StepRecord:
        """Attempt the next ready task with real tools."""

        project.state = ProjectState.EXECUTING
        task = project.next_task()
        if task is None:
            return StepRecord(phase=Phase.EXECUTE, summary="no ready task", success=False)

        task.status = TaskStatus.IN_PROGRESS
        task.attempts += 1

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reasoning": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "maxItems": self.max_tool_calls_per_step,
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                        "required": ["name"],
                    },
                },
            },
            "required": ["tool_calls"],
        }
        prompt = (
            "Return JSON only. Carry out ONE task using the tools below.\n"
            "Emit the tool calls that actually perform the work -- writing files, running tests.\n"
            "Read a file before editing it. Copy search anchors character-for-character.\n\n"
            f"TASK: {task.title}\n{task.detail}\n\n"
            + (f"PREVIOUS ATTEMPT FAILED WITH:\n{task.last_error[:1500]}\n\n" if task.last_error else "")
            + f"{self._project_brief(project)}\n"
            f"Available tools:\n{self.tools.render_for_prompt()}\n"
        )
        payload = self._ask_json(prompt, schema, max_tokens=1600)
        calls = payload.get("tool_calls") or []
        if not calls:
            task.last_error = "the model proposed no tool calls"
            task.status = TaskStatus.PENDING
            self._maybe_abandon(project, task)
            return StepRecord(phase=Phase.EXECUTE, summary=task.last_error, success=False, task_id=task.id)

        results = self._invoke_all(calls, context)
        effective = [result for result in results if result.ok and result.name not in _READ_ONLY_TOOLS]
        failures = [result for result in results if not result.ok]

        for result in results:
            if result.ok and result.name in {"apply_edits", "write_file"}:
                for path in _changed_paths(result):
                    project.add_artifact(path, kind="file", description=f"changed while: {task.title}")

        if failures and not effective:
            task.status = TaskStatus.PENDING
            task.last_error = "; ".join(f"{item.name}: {item.error}" for item in failures)[:2000]
            project.add_experiment(
                hypothesis=str(payload.get("reasoning", ""))[:400] or task.title,
                method=", ".join(item.name for item in results),
                outcome=task.last_error,
                succeeded=False,
                lesson=_lesson_from(failures),
                task_id=task.id,
            )
            self._maybe_abandon(project, task)
            return StepRecord(
                phase=Phase.EXECUTE,
                summary=f"task attempt {task.attempts} failed: {task.last_error[:200]}",
                success=False,
                task_id=task.id,
                tool_calls=[item.to_dict() for item in results],
            )

        # Work happened.  Whether it was the *right* work is for VERIFY to say;
        # marking the task done here and letting verification reopen it keeps
        # the loop moving instead of stalling on self-assessment.
        task.status = TaskStatus.DONE
        task.completed_at = _now()
        task.last_error = ""
        task.evidence = [f"{item.name}: {_brief_output(item.output)}" for item in effective][:5]
        project.add_experiment(
            hypothesis=str(payload.get("reasoning", ""))[:400] or task.title,
            method=", ".join(item.name for item in results),
            outcome="tools applied successfully",
            succeeded=True,
            task_id=task.id,
        )
        return StepRecord(
            phase=Phase.EXECUTE,
            summary=f"completed task: {task.title}",
            success=True,
            task_id=task.id,
            tool_calls=[item.to_dict() for item in results],
        )

    def _phase_verify(self, project: Project, context: ToolContext) -> StepRecord:
        """Run the acceptance checks.  This is the only source of 'done'."""

        project.state = ProjectState.VERIFYING
        previously_satisfied = sum(1 for item in project.objective_criteria() if item.satisfied)
        outcomes = self._verify(project, context)

        if not outcomes:
            project.add_blocker("no runnable acceptance check exists, so completion cannot be proved", needs_user=False)
            return StepRecord(phase=Phase.VERIFY, summary="nothing to verify", success=False)

        passed = [text for text, ok, _ in outcomes if ok]
        failed = [(text, evidence) for text, ok, evidence in outcomes if not ok]
        newly_satisfied = len(passed) > previously_satisfied

        if not failed and project.acceptance_satisfied():
            project.state = ProjectState.COMPLETED
            for blocker in project.active_blockers():
                project.resolve_blocker(blocker.id, "acceptance criteria passed")
            return StepRecord(
                phase=Phase.VERIFY,
                summary=f"all {len(passed)} acceptance check(s) pass",
                success=True,
                detail={"passed": passed},
            )

        return StepRecord(
            phase=Phase.VERIFY,
            summary=f"{len(passed)} passed, {len(failed)} failed",
            success=False,
            productive=newly_satisfied,
            detail={"passed": passed, "failed": [text for text, _ in failed], "evidence": [ev for _, ev in failed][:3]},
        )

    def _phase_diagnose(self, project: Project, context: ToolContext) -> StepRecord:
        """Read the evidence from the failure and reopen the right task.

        Diagnosis is where recovery actually happens: it converts a failing test
        or a rejected patch into a concrete next action, which is what stops a
        single mistake from ending the project.
        """

        evidence = self._recent_failure_evidence(project)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "diagnosis": {"type": "string"},
                "fix": {"type": "string"},
                "new_task": {"type": "string"},
                "blocked_on_user": {"type": "boolean"},
                "blocker": {"type": "string"},
            },
            "required": ["diagnosis", "fix"],
        }
        prompt = (
            "Return JSON only. Something failed. Work out WHY from the evidence, then say what to change.\n"
            "Base the diagnosis on the evidence text, not on what you expected to happen.\n"
            "Set blocked_on_user only for something no amount of code can fix "
            "(a missing credential, a hardware decision) -- never for a failing test.\n\n"
            f"EVIDENCE:\n{evidence[:4000]}\n\n"
            f"{self._project_brief(project)}\n"
        )
        payload = self._ask_json(prompt, schema, max_tokens=700)

        diagnosis = str(payload.get("diagnosis", "")).strip()
        fix = str(payload.get("fix", "")).strip()
        if diagnosis:
            project.add_finding(f"diagnosis: {diagnosis}", source="diagnosis", confidence=0.6)

        if payload.get("blocked_on_user") and str(payload.get("blocker", "")).strip():
            project.add_blocker(str(payload["blocker"]).strip(), needs_user=True)
            project.state = ProjectState.BLOCKED
            return StepRecord(phase=Phase.DIAGNOSE, summary=f"blocked: {payload['blocker']}", success=False)

        # Prefer reopening the task that failed over inventing a new one, so the
        # attempt counter keeps its meaning and the project cannot grow an
        # unbounded tail of near-duplicate tasks.
        reopened = self._reopen_failed_task(project, fix)
        if reopened is None and self._repair_budget_left(project):
            title = str(payload.get("new_task", "")).strip() or (fix[:110] if fix else "repair the failing behaviour")
            reopened = project.add_task(
                title, detail=f"{_AUTO_REPAIR}\n{diagnosis}\n\nProposed fix: {fix}", kind=project.kind
            )
        if reopened is None:
            # Every avenue this diagnosis can open has already been attempted.
            # Say so plainly rather than spawning another near-identical task.
            project.add_finding(
                "repair budget exhausted: every task derived from diagnosis has been attempted",
                source="engine",
                confidence=1.0,
            )
            return StepRecord(
                phase=Phase.DIAGNOSE,
                summary="no further repair avenue is available",
                success=False,
                productive=False,
                detail={"diagnosis": diagnosis, "fix": fix},
            )

        return StepRecord(
            phase=Phase.DIAGNOSE,
            summary=f"diagnosed: {diagnosis[:160]}",
            success=bool(diagnosis),
            # Understanding a failure is not the same as fixing it.  Marking a
            # diagnosis productive would let the loop alternate execute-fail /
            # diagnose-succeed indefinitely without ever advancing.
            productive=False,
            task_id=reopened.id,
            detail={"diagnosis": diagnosis, "fix": fix},
        )

    def _phase_replan(self, project: Project, context: ToolContext) -> StepRecord:
        """Nothing is runnable: unstick the plan or admit what is missing."""

        stuck = [task for task in project.tasks if task.open and task.exhausted]
        for task in stuck:
            task.status = TaskStatus.ABANDONED

        blocked = [task for task in project.tasks if task.open]
        if blocked:
            # Dependencies that can never be satisfied would deadlock the loop.
            done = {task.id for task in project.tasks if task.status is TaskStatus.DONE}
            for task in blocked:
                unmet = [dep for dep in task.depends_on if dep not in done]
                if unmet:
                    task.depends_on = [dep for dep in task.depends_on if dep in done]
                    project.add_finding(
                        f"dropped unsatisfiable dependencies from task {task.title!r}", source="engine", confidence=1.0
                    )
            return StepRecord(phase=Phase.REPLAN, summary=f"unblocked {len(blocked)} task(s)", success=True)

        if stuck:
            lessons = "; ".join(item.lesson for item in project.experiments if item.lesson)[:600]
            project.add_finding(
                f"abandoned {len(stuck)} task(s) after exhausting their attempts. Lessons: {lessons}",
                source="engine",
                confidence=1.0,
            )
            return StepRecord(phase=Phase.REPLAN, summary=f"abandoned {len(stuck)} exhausted task(s)", success=True)

        return StepRecord(phase=Phase.REPLAN, summary="nothing left to replan", success=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_abandon(self, project: Project, task: Task) -> None:
        if task.exhausted:
            task.status = TaskStatus.ABANDONED
            project.add_finding(
                f"task {task.title!r} abandoned after {task.attempts} attempts: {task.last_error[:300]}",
                source="engine",
                confidence=1.0,
            )

    def _repair_budget_left(self, project: Project) -> bool:
        """Cap how many repair tasks a diagnosis may invent.

        Without a cap, an unsolvable task is abandoned once its attempts run
        out, the next diagnosis invents a fresh task with a fresh attempt
        counter, and the two spin until the step budget is gone -- looking busy
        while proving nothing.  Observed directly while building this loop.
        """

        auto_created = sum(1 for task in project.tasks if _AUTO_REPAIR in task.detail)
        return auto_created < max(2, project.limits.max_task_attempts)

    def _reopen_failed_task(self, project: Project, fix: str) -> Task | None:
        candidates = [task for task in project.tasks if task.status in {TaskStatus.DONE, TaskStatus.FAILED}]
        if not candidates:
            return None
        task = candidates[-1]
        if task.exhausted:
            return None
        task.status = TaskStatus.PENDING
        if fix:
            task.detail = f"{task.detail}\n\nRepair: {fix}".strip()
        return task

    def _recent_failure_evidence(self, project: Project) -> str:
        parts: list[str] = []
        for step in reversed(project.steps[-6:]):
            if step.success:
                continue
            parts.append(f"[{step.phase.value}] {step.summary}")
            for key in ("evidence", "failed"):
                if step.detail.get(key):
                    parts.append(json.dumps(step.detail[key], default=str)[:1500])
            for call in step.tool_calls[:3]:
                if not call.get("ok"):
                    parts.append(f"  {call.get('name')} failed: {str(call.get('error'))[:800]}")
                else:
                    parts.append(f"  {call.get('name')} -> {_brief_output(call.get('output'))}")
        return "\n".join(parts) or "no failure evidence recorded"

    def _default_verifier(self, project: Project, context: ToolContext) -> list[tuple[str, bool, str]]:
        """Run each objective criterion's command and record the evidence."""

        outcomes: list[tuple[str, bool, str]] = []
        for criterion in project.objective_criteria():
            result = self.tools.invoke(
                ToolCall(name="run_command", arguments={"command": list(criterion.check)}), context
            )
            output = result.output if isinstance(result.output, dict) else {}
            ok = bool(result.ok and output.get("success"))
            evidence = (
                f"$ {' '.join(criterion.check)}\n"
                f"exit={output.get('returncode', 'n/a')}\n"
                f"{str(output.get('stdout', ''))[-1500:]}\n{str(output.get('stderr', ''))[-1500:]}"
                if result.ok
                else f"$ {' '.join(criterion.check)}\n{result.error}"
            )
            criterion.satisfied = ok
            criterion.last_evidence = evidence[-3000:]
            criterion.last_checked_at = _now()
            outcomes.append((criterion.text, ok, evidence))
        return outcomes

    def _invoke_all(self, calls: list[Any], context: ToolContext) -> list[ToolResult]:
        results: list[ToolResult] = []
        for payload in calls[: self.max_tool_calls_per_step]:
            if not isinstance(payload, dict):
                continue
            results.append(self.tools.invoke_payload(payload, context))
        return results

    def _project_brief(self, project: Project) -> str:
        """The compact project state a prompt needs.

        Bounded on purpose: a small model given a full project history performs
        worse than one given the ten most relevant lines, and the whole point of
        keeping durable state is that it does not all need to be in the prompt.
        """

        lines = [
            f"GOAL: {project.goal}",
            f"KIND: {project.kind}",
            f"WORKSPACE: {project.workspace}",
        ]
        if project.constraints:
            lines.append("CONSTRAINTS:\n" + "\n".join(f"  - {item}" for item in project.constraints[:6]))
        if project.requirements:
            lines.append("REQUIREMENTS:\n" + "\n".join(f"  - {item.text}" for item in project.requirements[-8:]))
        if project.acceptance:
            lines.append(
                "ACCEPTANCE:\n"
                + "\n".join(
                    f"  - [{'x' if item.satisfied else ' '}] {item.text}"
                    + (f"  (check: {' '.join(item.check)})" if item.check else "  (NO RUNNABLE CHECK)")
                    for item in project.acceptance[:8]
                )
            )
        if project.tasks:
            lines.append(
                "TASKS:\n"
                + "\n".join(f"  - [{task.status.value}] {task.title}" for task in project.tasks[-10:])
            )
        if project.findings:
            lines.append("KNOWN FACTS:\n" + "\n".join(f"  - {item.text[:220]}" for item in project.findings[-8:]))
        failed = [item for item in project.experiments if not item.succeeded]
        if failed:
            lines.append(
                "ALREADY TRIED AND FAILED (do not repeat):\n"
                + "\n".join(f"  - {item.hypothesis[:120]} -> {item.outcome[:160]}" for item in failed[-5:])
            )
        if project.decisions:
            lines.append("DECISIONS:\n" + "\n".join(f"  - {item.text[:160]}" for item in project.decisions[-5:]))
        return "\n".join(lines)

    def _ask_json(self, prompt: str, schema: dict[str, Any], *, max_tokens: int, attempts: int = 3) -> dict[str, Any]:
        """Ask for schema-shaped JSON, retrying with feedback on malformed output.

        Returns ``{}`` rather than raising when every attempt fails: an
        unparseable response is a bad step, and the loop is built to survive bad
        steps.  Raising here would turn a recoverable model error into a dead
        project.
        """

        current = prompt
        for _ in range(max(1, attempts)):
            try:
                raw = self._generate(current, schema, max_tokens=max_tokens)
                data = lenient_json_loads(_extract_json(raw))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            except Exception:
                return {}
            current = prompt + "\n\nYour previous reply was not valid JSON. Reply with JSON only, matching the schema."
        return {}

    def _generate(self, prompt: str, schema: dict[str, Any], *, max_tokens: int) -> str:
        if hasattr(self.brain, "generate_structured"):
            try:
                return self.brain.generate_structured(prompt, schema, max_tokens=max_tokens, temperature=0.1)
            except NotImplementedError:
                pass
        if hasattr(self.brain, "generate_coding"):
            return self.brain.generate_coding(prompt, max_tokens=max_tokens, temperature=0.1)
        return self.brain.generate(prompt, max_tokens=max_tokens, temperature=0.1)


#: Marker in a task's detail identifying it as invented by a diagnosis rather
#: than by decomposition, so :meth:`ProjectEngine._repair_budget_left` can bound them.
_AUTO_REPAIR = "[auto-repair]"

#: Tools that observe without changing anything, so a step consisting only of
#: these has not actually advanced the task.
_READ_ONLY_TOOLS = frozenset(
    {"list_files", "read_file", "search_text", "find_definition", "which", "git", "git_diff", "web_search", "fetch_url", "check_syntax"}
)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _title_from(goal: str) -> str:
    text = " ".join(goal.split())
    return text[:70] + ("..." if len(text) > 70 else "")


def _extract_json(text: str) -> str:
    import re

    match = re.search(r"(\{.*\})", str(text).strip(), flags=re.DOTALL)
    return match.group(1) if match else str(text)


def _brief_args(arguments: dict[str, Any]) -> str:
    rendered = json.dumps(arguments, default=str)
    return rendered if len(rendered) <= 120 else rendered[:117] + "..."


def _brief_output(output: Any) -> str:
    if isinstance(output, str):
        return output[:300]
    rendered = json.dumps(output, default=str)
    return rendered if len(rendered) <= 300 else rendered[:297] + "..."


def _changed_paths(result: ToolResult) -> list[str]:
    output = result.output if isinstance(result.output, dict) else {}
    return [str(item.get("path")) for item in output.get("applied") or [] if isinstance(item, dict) and item.get("path")]


def _lesson_from(failures: list[ToolResult]) -> str:
    """Turn a tool failure into advice worth remembering next attempt."""

    lessons = {
        "no_unique_match": "the search anchor must be copied verbatim from the current file",
        "ambiguous_search": "the search anchor must be unique; include neighbouring lines",
        "stale_context": "re-read a file before editing it",
        "protected_path": "that path is protected and must not be edited",
        "command_denied": "that executable is not permitted in this run",
        "timeout": "that command takes too long; make it smaller or faster",
        "unknown_tool": "use only the tools listed in the prompt",
        "invalid_arguments": "check the tool's required arguments",
        "create_over_existing": "the file already exists; edit it instead of creating it",
    }
    for failure in failures:
        if failure.error_kind in lessons:
            return lessons[failure.error_kind]
    return failures[0].error[:200] if failures else ""
