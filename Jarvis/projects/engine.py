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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
from runtime.deadline import CallTimeout, Deadline, DeadlineExceeded, call_with_timeout
from runtime.heartbeat import Heartbeat
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
        verify_interval: int = 3,
        heartbeat: Heartbeat | None = None,
    ) -> None:
        self.brain = brain
        self.store = store
        self.tools = tools
        self.hooks = hooks or EngineHooks()
        self.max_tool_calls_per_step = max_tool_calls_per_step
        #: Replaced per run with the project's own budget.  Unbounded until then
        #: so a caller driving phases directly behaves as it always did.
        self.deadline = Deadline.none()
        #: Ceiling for a single model call, before the mission budget clamps it.
        self.step_timeout_seconds = 600.0
        #: Why the last generation produced nothing, surfaced to DIAGNOSE.
        self.last_model_error = ""
        #: Proves the process is breathing even mid-call.  A no-op without a
        #: path, so nothing here depends on a heartbeat existing.
        self.heartbeat = heartbeat or Heartbeat(None)
        #: How many EXECUTE steps may pass before checking in with a real
        #: verification run.  See :meth:`_next_phase` for why this is not 1.
        self.verify_interval = max(1, verify_interval)
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

        # The budget is now carried *into* the work rather than checked around
        # it.  Checked only here at the top, a single step that blocked forever
        # would never reach the check again, and the loop would sit at
        # "elapsed < max_seconds" indefinitely.
        self.deadline = Deadline.of(limits.max_seconds, name=f"project {project.id}")
        self.step_timeout_seconds = float(limits.step_timeout_seconds)
        self.heartbeat.set_budget(limits.max_seconds)
        self.heartbeat.beat("starting", project.title, progress=True)

        if project.state is ProjectState.DRAFT:
            project.state = ProjectState.INVESTIGATING

        stop = StopReason.STEP_LIMIT
        message = ""

        try:
            # A resumed project may have been changed underneath its recorded
            # verification: by an expert working in the same workspace after an
            # escalation, by a person, or by another run.  Planning from stale
            # evidence is not a small error -- seen live, an entire failure
            # budget went on diagnosing a defect that had already been fixed,
            # because every DIAGNOSE read the last recorded failure and nothing
            # had re-run the checks since.
            #
            # So a resumed project re-establishes what is actually true before
            # it plans anything, which is what a careful engineer does on
            # sitting back down: run the tests first, then decide what is broken.
            if steps < budget and self._evidence_predates_the_workspace(project, context):
                step = self._run_phase(Phase.VERIFY, project, context)
                steps += 1
                project.record_step(step)
                project.seconds_spent += step.duration_seconds
                self.store.save(project)
                if self.hooks.on_step:
                    self.hooks.on_step(project, step)
                if project.state is ProjectState.COMPLETED:
                    stop, message = StopReason.ACCEPTED, "all objective acceptance criteria pass"
                    budget = 0  # nothing left to do; fall past the loop

            while steps < budget:
                if self.hooks.should_cancel and self.hooks.should_cancel():
                    stop, message = StopReason.CANCELLED, "stopped by the user"
                    break
                if self.deadline.expired:
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

                self.heartbeat.beat(phase.value, project.title)
                step = self._run_phase(phase, project, context)
                steps += 1
                project.record_step(step)
                project.seconds_spent += step.duration_seconds
                self.heartbeat.beat(phase.value, step.summary, progress=True, last_step_ok=step.success)
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
                # `while...else` also runs when the opening verification
                # accepted the project and zeroed the budget, and calling that
                # "budget exhausted" would misreport a pass as a stall.
                if stop is StopReason.STEP_LIMIT:
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

    @classmethod
    def _protected_reason(cls, project: Project, workspace: str) -> str:
        """Why a proven file is closed, and which file to change instead.

        A refusal that names only what is forbidden leaves the model to guess
        the alternative, and it guesses the same wrong thing repeatedly: three
        runs were spent trying to edit an accepted module in different ways.
        Naming the file that *is* in play turns the refusal into a direction.
        """

        in_play = sorted(
            Path(item).name
            for item in cls._files_named_by(project, workspace, satisfied=False)
        )
        base = (
            " -- this file is covered by an acceptance check that currently PASSES, so changing "
            "it cannot fix a failing check and may break a passing one"
        )
        if not in_play:
            return base + "."
        return f"{base}. Change the code that calls it instead: {', '.join(in_play)}."

    @classmethod
    def _files_named_by(cls, project: Project, workspace: str, *, satisfied: bool) -> list[str]:
        """Workspace files named by the command of a satisfied/unsatisfied criterion."""

        try:
            stems = {path.stem: path for path in Path(workspace).glob("*.py")}
        except OSError:
            return []
        found: set[str] = set()
        for criterion in project.objective_criteria():
            if criterion.satisfied is not satisfied:
                continue
            words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(criterion.check)))
            found.update(stems.keys() & words)
        return [stems[stem].name for stem in sorted(found)]

    @staticmethod
    def _proven_files(project: Project, workspace: str) -> list[str]:
        """Workspace files that a passing check depends on and no failing one does.

        When a call and a definition disagree, there are two ways to make them
        agree, and only one of them is safe.  Seen live: `pipeline.py` called
        `position(path, x, y, square)` while `position(path)` took one argument,
        and the loop set about editing `position.py` -- the accepted, verified
        module -- rather than the four-day-old line that called it wrongly.
        Changing the proven side cannot fix the failing check and can only break
        a passing one.

        The rule is mechanical rather than a judgement: a file is proven if a
        *satisfied* criterion's command names it, and it is back in play the
        moment an *unsatisfied* criterion names it.  So a later requirement that
        genuinely needs to change an accepted module unprotects it by having a
        failing check that mentions it, and nothing has to be decided in advance.

        This is enforced rather than advised, because a weak model reads "avoid
        editing this" as a suggestion and edits it anyway.
        """

        criteria = project.objective_criteria()
        if not criteria:
            return []
        try:
            stems = {path.stem: path for path in Path(workspace).glob("*.py")}
        except OSError:
            return []
        if not stems:
            return []

        proven: set[str] = set()
        in_play: set[str] = set()
        for criterion in criteria:
            words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(criterion.check)))
            (proven if criterion.satisfied else in_play).update(stems.keys() & words)
        # Workspace-relative, because PathPolicy matches repo-relative strings:
        # an absolute path never equals "position.py" and the protection is
        # silently inert. That is exactly how this shipped once -- the unit test
        # asserted Path(item).name, which passes either way, so it checked the
        # shape of the answer instead of whether anything was protected.
        return [stems[stem].name for stem in sorted(proven - in_play)]

    @staticmethod
    def _work_in_hand(project: Project) -> str:
        """What the project is currently trying to do, as plain text.

        Used to steer which files a prompt looks at.  The newest requirement and
        the open tasks say what matters now; the last error only says where the
        model happened to slip.
        """

        parts: list[str] = []
        if project.requirements:
            parts.append(project.requirements[-1].text)
        parts.extend(task.title for task in project.tasks if task.open)
        for item in project.objective_criteria():
            if not item.satisfied:
                parts.append(item.text)
        return " ".join(parts)

    def _evidence_predates_the_workspace(self, project: Project, context: ToolContext) -> bool:
        """Whether the workspace changed after the acceptance checks last ran.

        Recorded verification is a claim about a set of files at a moment.  If
        anything wrote to those files afterwards -- an expert after an
        escalation, a person, another run -- the claim is about a workspace
        that no longer exists, and the loop must not plan from it.

        Only a criterion that has actually been checked counts: a project whose
        checks have never run has nothing stale to re-establish, and verifying
        on that path would just spend a step confirming the obvious.
        """

        checked = [item.last_checked_at for item in project.objective_criteria() if item.last_checked_at]
        if not checked:
            return False

        try:
            newest = max(
                (path.stat().st_mtime for path in Path(context.workspace).rglob("*.py")),
                default=0.0,
            )
        except OSError:
            return False
        if not newest:
            return False

        try:
            oldest_check = min(datetime.fromisoformat(item) for item in checked)
        except ValueError:
            # An unparseable timestamp is not evidence that anything is current.
            return True
        if oldest_check.tzinfo is None:
            oldest_check = oldest_check.replace(tzinfo=timezone.utc)

        return datetime.fromtimestamp(newest, tz=timezone.utc) > oldest_check

    def tool_context(self, project: Project) -> ToolContext:
        workspace = self.store.workspace_for(project)
        readable = [workspace]
        if project.repository:
            readable.append(project.repository)
        return ToolContext(
            workspace=workspace,
            readable_roots=[Path(item) for item in readable],
            timeout_seconds=project.limits.step_timeout_seconds,
            protected_paths=list(project.metadata.get("protected_paths") or [])
            + self._proven_files(project, workspace),
            protected_reason=self._protected_reason(project, workspace),
            allowed_paths=list(project.metadata.get("allowed_paths") or []),
        )

    # ------------------------------------------------------------------
    # Phase selection
    # ------------------------------------------------------------------

    def _next_phase(self, project: Project) -> Phase:
        """Choose the next phase from the project's own state.

        Deliberately deterministic.  Letting the model pick its own next phase
        sounds flexible, but in practice a weak model loops between planning and
        re-planning without ever executing anything.

        The ordering here was corrected after a live run against the local 7B
        model: the first version verified after *every* task, so with six tasks
        still pending it failed, diagnosed "no tests were found" -- correctly,
        because the tests had not been written yet -- and re-ran early tasks.
        Two thirds of the step budget went on diagnosing a plan that had simply
        not finished running.  Verification is now something the loop does when
        the plan is done, plus occasionally in case the work is already
        complete, and a verification failure with work still queued means
        "carry on", not "something is wrong".
        """

        if not project.findings and not project.tasks:
            return Phase.INVESTIGATE
        if not project.acceptance or not project.tasks:
            return Phase.DECOMPOSE

        ready = project.ready_tasks()
        last = project.steps[-1] if project.steps else None

        if last is not None:
            # A tool that failed needs to be understood before trying again.
            if last.phase is Phase.EXECUTE and not last.success:
                return Phase.DIAGNOSE
            if last.phase is Phase.DIAGNOSE:
                return Phase.EXECUTE if project.ready_tasks() else Phase.REPLAN
            if last.phase is Phase.VERIFY and not last.success:
                # Only investigate the failure once there is nothing left to build.
                return Phase.EXECUTE if ready else Phase.DIAGNOSE

        if not ready:
            # Only blocked or exhausted work remains, or none at all.
            return Phase.REPLAN if any(task.open for task in project.tasks) else Phase.VERIFY

        # There is work queued.  Check in occasionally so an early finish is
        # noticed, but do not pay for a full verification after every task.
        if self._executes_since_verify(project) >= self.verify_interval:
            return Phase.VERIFY
        return Phase.EXECUTE

    def _executes_since_verify(self, project: Project) -> int:
        count = 0
        for step in reversed(project.steps):
            if step.phase is Phase.VERIFY:
                break
            if step.phase is Phase.EXECUTE:
                count += 1
        return count

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
        if step.productive is None:
            # No explicit opinion: a step that failed did not advance the work.
            # Phases that know better (VERIFY proving a new criterion, EXECUTE
            # routing around an impossible task) say so themselves.
            step.productive = step.success
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
            "Plan ONLY work the tools below can actually perform. Do not plan steps like\n"
            "initialising a git repository or setting up scaffolding that is not required.\n"
            "Put source files directly in the workspace root; do not nest a directory named\n"
            "after the project.\n"
            "Also give acceptance criteria. Every criterion SHOULD carry a `check`: an executable\n"
            'command array such as ["python", "-m", "pytest", "-q"].\n'
            "A criterion without a runnable check can never be proved, so it will not count.\n\n"
            f"{self._project_brief(project)}\n"
            # Planning blind to the toolset produces plans that cannot be run;
            # the planner needs the same capability list the executor gets.
            f"Tools available to carry out each task:\n{self.tools.render_for_prompt()}\n"
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
        existing_criteria = {_criterion_key(item.text) for item in project.acceptance}
        # Two criteria that run the same command prove the same thing, so the
        # second is pure cost: it doubles every verification run and clutters
        # the brief.  Observed live -- the model proposed "The tests pass." next
        # to a caller-supplied "the tests pass".
        existing_checks = {tuple(item.check) for item in project.acceptance if item.check}
        for item in payload.get("acceptance") or []:
            text = str(item.get("text", "")).strip()
            check = [str(part) for part in (item.get("check") or []) if str(part).strip()]
            if not text or _criterion_key(text) in existing_criteria:
                continue
            if check and tuple(check) in existing_checks:
                continue
            project.add_acceptance(text, check=check)
            existing_criteria.add(_criterion_key(text))
            if check:
                existing_checks.add(tuple(check))
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
            "Read a file before editing it. Copy search anchors character-for-character.\n"
            "Put source files directly in the workspace root unless the task says otherwise; "
            "do NOT create a nested directory named after the project.\n"
            # Observed live, in two different subsystems: the model investigates
            # with a tool, then writes that tool's name into the source file it
            # is producing. Tools and library functions look identical in a
            # transcript -- both are names that were called and returned useful
            # data -- so the distinction has to be stated rather than assumed.
            # The chess project spent its entire failure budget on
            # `image_region(path, x, y, w, h)` called from inside position.py.
            "The tools below exist ONLY while you are working. They are NOT importable and NOT "
            "callable from the source files you write. Anything your code needs at runtime must "
            "come from the Python standard library or an installed package. If a tool told you "
            "something useful, put the ANSWER in your code or compute it there.\n\n"
            f"TASK: {task.title}\n{task.detail}\n\n"
            + (f"THIS TASK'S PREVIOUS ATTEMPT FAILED WITH:\n{task.last_error[:1500]}\n\n" if task.last_error else "")
            # The single most important context: what the acceptance check
            # actually printed.  Without it the model rewrites the same stub
            # over and over, because it never sees why the stub is wrong.
            + self._failing_check_evidence(project)
            + self._missing_module_notice(project, context)
            # The file the task and the last failure are about goes first, and
            # in full: that is the text an anchor has to be copied from.
            + self._workspace_snapshot(context, focus=f"{task.title} {task.detail} {task.last_error}")
            + self._anchor_trouble_notice(task)
            + f"{self._project_brief(project)}\n"
            f"Available tools:\n{self.tools.render_for_prompt(exclude=self._withdrawn_tools(task))}\n"
        )
        self.last_model_error = ""
        payload = self._ask_json(prompt, schema, max_tokens=1600)
        calls = payload.get("tool_calls") or []
        if not calls:
            # Say *why* nothing came back. "The model proposed no tool calls"
            # and "the model never answered" call for completely different
            # responses, and DIAGNOSE can only tell them apart if the step says
            # which one happened.
            task.last_error = self.last_model_error or "the model proposed no tool calls"
            task.status = TaskStatus.PENDING
            self._maybe_abandon(project, task)
            return StepRecord(
                phase=Phase.EXECUTE,
                summary=task.last_error,
                success=False,
                task_id=task.id,
                detail={"model_error": self.last_model_error} if self.last_model_error else {},
            )

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

            # A task whose every tool call was refused on policy grounds cannot
            # ever succeed -- retrying it just spends the failure budget on a
            # foregone conclusion.  Observed live: a plan step "initialise a git
            # repository" was denied three times before the run gave up, while
            # six perfectly workable tasks sat untouched behind it.
            if all(not item.retryable for item in failures):
                task.status = TaskStatus.ABANDONED
                project.add_finding(
                    f"task {task.title!r} is not possible with the permitted tools: {task.last_error[:220]}",
                    source="engine",
                    confidence=1.0,
                )
                remaining = bool(project.ready_tasks())
                return StepRecord(
                    phase=Phase.EXECUTE,
                    summary=f"abandoned impossible task: {task.title}",
                    success=False,
                    # Routing around a task that can never work does advance the
                    # plan, so long as there is other work to move on to.
                    productive=remaining,
                    task_id=task.id,
                    tool_calls=[item.to_dict() for item in results],
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
        exhausted = self._exhausted_diagnoses(project)
        prompt = (
            "Return JSON only. Something failed. Work out WHY from the evidence, then say what to change.\n"
            "Base the diagnosis on the evidence text, not on what you expected to happen.\n"
            "Quote the specific error from the evidence in your diagnosis.\n"
            "Set blocked_on_user only for something no amount of code can fix "
            "(a missing credential, a hardware decision) -- never for a failing test.\n\n"
            + exhausted
            + f"EVIDENCE:\n{evidence[:4000]}\n\n"
            + self._failing_check_evidence(project)
            + self._missing_module_notice(project, context)
            # Lead the focus with the work in hand, not the last error. Focusing
            # on the failure alone is self-reinforcing: a model that wrongly
            # edited board.py produces an error naming board.py, which sorts
            # board.py to the front of the snapshot and spends the character
            # budget on it, which makes the next wrong edit likelier. Seen live,
            # five attempts running, while the actual task was to create a file
            # that did not exist yet and therefore appeared nowhere in the error.
            + self._workspace_snapshot(context, focus=f"{self._work_in_hand(project)} {evidence}")
            + f"{self._project_brief(project)}\n"
        )
        payload = self._ask_json(prompt, schema, max_tokens=700)

        diagnosis = str(payload.get("diagnosis", "")).strip()
        fix = str(payload.get("fix", "")).strip()
        repeated = self._remember_diagnosis(project, diagnosis)
        if repeated:
            # The same wrong answer arriving again is itself evidence: it says
            # the visible symptom has been explained to exhaustion and the cause
            # lies somewhere the current evidence does not show.
            project.add_finding(
                f"diagnosis {diagnosis[:120]!r} has now been produced {repeated} times without fixing "
                "anything, so it is not the cause; look at the implementation rather than the symptom",
                source="engine",
                confidence=1.0,
            )
        if diagnosis:
            project.add_finding(f"diagnosis: {diagnosis}", source="diagnosis", confidence=0.6)

        if payload.get("blocked_on_user") and str(payload.get("blocker", "")).strip():
            project.add_blocker(str(payload["blocker"]).strip(), needs_user=True)
            project.state = ProjectState.BLOCKED
            return StepRecord(phase=Phase.DIAGNOSE, summary=f"blocked: {payload['blocker']}", success=False)

        # If something is already queued and retryable, the diagnosis is all
        # that was needed: EXECUTE will retry it with this evidence attached.
        # Creating a task here as well produced a duplicate for every failed
        # tool call, and the duplicates then competed for the attempt budget.
        already_queued = project.next_task()
        if already_queued is not None:
            if fix:
                already_queued.detail = f"{already_queued.detail}\n\nRepair: {fix}".strip()
            return StepRecord(
                phase=Phase.DIAGNOSE,
                summary=f"diagnosed: {diagnosis[:160]}",
                success=bool(diagnosis),
                productive=False,
                task_id=already_queued.id,
                detail={"diagnosis": diagnosis, "fix": fix, "retrying": already_queued.title},
            )

        # Prefer reopening the task that failed over inventing a new one, so the
        # attempt counter keeps its meaning and the project cannot grow an
        # unbounded tail of near-duplicate tasks.
        reopened = self._reopen_failed_task(project, fix)
        if reopened is None and self._repair_budget_left(project):
            title = str(payload.get("new_task", "")).strip() or (fix[:110] if fix else "repair the failing behaviour")
            # A repeated diagnosis produces a repeated task title.  Creating it
            # again just buys another three attempts at something already shown
            # not to work; seen live as four identically-titled "install pytest"
            # tasks consuming an entire step budget.
            if any(task.title.strip().lower() == title.strip().lower() for task in project.tasks):
                project.add_finding(
                    f"diagnosis keeps proposing {title!r}, which has already been attempted; a different approach is needed",
                    source="engine",
                    confidence=1.0,
                )
            else:
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

        # Counted since the last VERIFIED PROGRESS, not over the project's whole
        # life. A long-lived project accumulates requirements across sessions --
        # which is the point of persistence -- and a lifetime count means the
        # third requirement starts with a budget the first one already spent.
        # Observed on the chess project: 4 of 5 criteria passing, and the
        # remaining one refused every repair with "no further repair avenue".
        #
        # A satisfied acceptance check is proof the loop is productive, so it is
        # the right thing to reset against: spinning still exhausts the budget
        # because spinning produces no passing checks.
        since = self._last_progress_at(project)
        auto_created = sum(
            1 for task in project.tasks
            if _AUTO_REPAIR in task.detail and (not since or task.created_at > since)
        )
        return auto_created < max(2, project.limits.max_task_attempts)

    @staticmethod
    def _last_progress_at(project: Project) -> str:
        """When the loop last demonstrably achieved something.

        The most recent successful VERIFY, which is the only unambiguous
        evidence of progress the project keeps.
        """

        for step in reversed(project.steps):
            if step.phase is not Phase.VERIFY:
                continue
            # `success` means EVERY criterion passed, which is too strict a
            # definition of progress: going from four passing to five is
            # plainly progress, and the loop that achieved it should not then
            # be told it has no repair budget left. VERIFY already records
            # `productive` for exactly that case.
            if step.success or step.productive:
                return getattr(step, "at", "") or ""
        return ""

    def _reopen_failed_task(self, project: Project, fix: str) -> Task | None:
        """Reopen the task most likely to be responsible for the failure.

        "Most likely" means the one most recently executed, taken from the step
        trajectory rather than from list order.  An earlier version used
        ``tasks[-1]``, which on a live run kept reopening "create a project
        directory" while the actual defect sat in the implementation task.
        """

        recent_ids = [step.task_id for step in reversed(project.steps) if step.phase is Phase.EXECUTE and step.task_id]
        by_id = {task.id: task for task in project.tasks}

        for task_id in recent_ids:
            task = by_id.get(task_id)
            if task is None or task.open:
                continue
            if task.status not in {TaskStatus.DONE, TaskStatus.FAILED}:
                continue
            if task.exhausted:
                # Out of attempts -- unless the task was marked DONE and the
                # behaviour it claimed is still failing.  A task is DONE
                # because the executor said so, not because the acceptance
                # criterion it targets went green, so "DONE with a red
                # criterion" is a task that was never finished.  Give it a
                # fresh run-length budget rather than treating a stale
                # lifetime count as a verdict.
                if task.status is not TaskStatus.DONE or not task.reopenable:
                    continue
                task.reopenings += 1
                task.attempts = 0
                task.last_error = ""
            task.status = TaskStatus.PENDING
            if fix:
                task.detail = f"{task.detail}\n\nRepair: {fix}".strip()
            return task
        return None

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
            if result.ok:
                stdout = str(output.get("stdout", ""))
                stderr = str(output.get("stderr", ""))
                returncode = output.get("returncode", "n/a")
                evidence = (
                    f"$ {' '.join(criterion.check)}\n"
                    f"exit={returncode}"
                    + _explain_exit(criterion.check, returncode, f"{stdout}\n{stderr}")
                    + f"\n{stdout[-1500:]}\n{stderr[-1500:]}"
                )
            else:
                evidence = f"$ {' '.join(criterion.check)}\n{result.error}"
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

    #: Edit failures that mean "I know what to change but cannot express it as
    #: an anchor".  Repeating them is the single most common way a local model
    #: burns a step budget.
    _ANCHOR_TROUBLE = (
        "no_unique_match",
        "no safe unique match",
        "ambiguous_search",
        "must match exactly once",
        # Repeatedly sending an anchorless edit is the same predicament wearing
        # a different error: the model knows the content it wants and cannot
        # express it as an anchor.
        "needs a 'search' anchor",
        # And so is an edit that lands but does not parse. Six consecutive
        # "unparseable" failures on a twenty-line module ended one live
        # capability run; the model could describe the fix perfectly each time
        # and could not splice it in. Sending the whole small file is the way
        # out, and it will still meet the shrink guard if it sends a fragment.
        "unparseable",
        # The model sent an edit identical to what is already there, which means
        # its idea of the file and the file itself have diverged. Sending the
        # whole small file resynchronises them.
        "no effective edit",
    )

    def _anchor_failed_before(self, task: Task) -> bool:
        return bool(task.last_error) and any(marker in task.last_error for marker in self._ANCHOR_TROUBLE)

    def _withdrawn_tools(self, task: Task) -> set[str]:
        """Tools to stop offering for this attempt.

        A model that cannot land an anchor keeps trying to land an anchor, and
        telling it to use ``write_file`` instead does not work -- a live run
        showed the advice arriving in the error message and being ignored eight
        times running.  Taking ``apply_edits`` off the menu does work, for the
        same reason that removing the rewrite verb from the repository schema
        worked: a model cannot choose what it is not offered.
        """

        return {"apply_edits"} if self._anchor_failed_before(task) else set()

    def _anchor_trouble_notice(self, task: Task) -> str:
        if not self._anchor_failed_before(task):
            return ""
        return (
            "NOTE: your last attempt could not match its search anchor. apply_edits is not available "
            "for this attempt. Use write_file and send the complete corrected contents of the file.\n\n"
        )

    #: How many distinct past diagnoses to quote back to the model.  Enough to
    #: rule out a wrong answer, few enough to leave room for the evidence.
    _DIAGNOSIS_MEMORY = 6

    @staticmethod
    def _diagnosis_fingerprint(text: str) -> str:
        """Collapse a diagnosis to what makes it the same answer twice.

        Not an exact match: a model re-stating the same wrong cause rarely
        reproduces its own wording byte-for-byte, and a comparison that strict
        would never fire.
        """

        import re as _re

        words = _re.findall(r"[a-z0-9_]+", text.lower())
        return " ".join(words[:24])

    def _remember_diagnosis(self, project: Project, diagnosis: str) -> int:
        """Record a diagnosis; return how many times this one has now been made."""

        if not diagnosis.strip():
            return 0
        fingerprint = self._diagnosis_fingerprint(diagnosis)
        if not fingerprint:
            return 0
        history = project.metadata.setdefault("diagnosis_history", [])
        seen = sum(1 for item in history if item.get("fingerprint") == fingerprint)
        history.append({"fingerprint": fingerprint, "text": diagnosis[:400]})
        del history[: max(0, len(history) - 2 * self._DIAGNOSIS_MEMORY)]
        return seen + 1 if seen else 0

    def _exhausted_diagnoses(self, project: Project) -> str:
        """Quote back the explanations that have already been tried and failed.

        This is what makes a retry a genuinely different request rather than the
        identical prompt sent again.  A small model asked the same question
        gives the same answer -- six times, in the run that motivated this -- so
        the prompt has to change, and the honest change is to rule out the
        answers already known to be useless.
        """

        history = project.metadata.get("diagnosis_history") or []
        if not history:
            return ""
        seen: list[str] = []
        for item in reversed(history):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if text not in seen:
                seen.append(text)
            if len(seen) >= self._DIAGNOSIS_MEMORY:
                break
        if not seen:
            return ""
        lines = "\n".join(f"  {index}. {text[:220]}" for index, text in enumerate(seen, start=1))
        return (
            "DIAGNOSES ALREADY MADE THAT DID NOT FIX ANYTHING -- do not repeat these, "
            "the cause is something else:\n"
            f"{lines}\n"
            "Look at a different part of the evidence this time. If the tests keep failing the same way, "
            "suspect the code under test rather than the test.\n\n"
        )

    def _missing_module_notice(self, project: Project, context: ToolContext) -> str:
        """Say plainly when a failing check needs a file the project has not written.

        Observed live: `import engine` failed, the model read "No module named
        'engine'" as a packaging problem, and spent its whole repair budget on a
        task to pip-install python-chess. The module was one the requirement had
        asked it to WRITE.

        The distinction is mechanical -- either engine.py is in the workspace or
        it is not -- so it should not be left to inference from an error message
        that genuinely reads both ways.
        """

        import re as _re
        from pathlib import Path as _Path

        failing = [
            item for item in project.objective_criteria()
            if not item.satisfied and item.last_evidence
        ]
        if not failing:
            return ""

        workspace = _Path(context.workspace)
        notices: list[str] = []
        seen: set[str] = set()
        for item in failing:
            for name in _re.findall(r"No module named '([A-Za-z_][A-Za-z0-9_]*)'", item.last_evidence):
                if name in seen:
                    continue
                seen.add(name)
                if (workspace / f"{name}.py").is_file():
                    continue
                notices.append(
                    f"{name}.py DOES NOT EXIST in the workspace. "
                    f"\"No module named '{name}'\" here means the file has not been written yet -- "
                    "it is NOT a missing package and installing anything will not help. "
                    f"Write {name}.py."
                )
        if not notices:
            return ""
        body = "\n".join(f"  {line}" for line in notices)
        return f"WHAT THE FAILURE ACTUALLY MEANS:\n{body}\n\n"

    def _failing_check_evidence(self, project: Project) -> str:
        """The output of the acceptance checks that are currently failing.

        A local model cannot repair a defect it has never been shown.  Feeding
        the real stdout/stderr back is what turns "write the file again" into
        "the function returns None, so return the dict instead".
        """

        failing = [
            item for item in project.objective_criteria() if not item.satisfied and item.last_evidence
        ]
        if not failing:
            return ""
        blocks = [
            f"--- {item.text} ---\n{item.last_evidence[-1800:]}" for item in failing[:2]
        ]
        return "OUTPUT OF THE FAILING ACCEPTANCE CHECK (this is what you must make pass):\n" + "\n".join(blocks) + "\n\n"

    def _workspace_snapshot(
        self, context: ToolContext, *, max_files: int = 12, max_chars: int = 6000, focus: str = ""
    ) -> str:
        """The files that exist now, with the small ones inlined.

        Shown unnumbered and verbatim so a search anchor copied out of it can
        actually match -- the same reasoning as in the repository engineer.

        Files named in ``focus`` come first and get the character budget before
        anything else.  Alphabetical order put ``main.py`` behind other files
        and truncated exactly the content the model needed to copy an anchor
        from, which is a strange way to fail.
        """

        try:
            from tools.builtin import iter_files, relative_to

            paths = [path for path in iter_files(context.workspace, limit=200) if path.suffix in {".py", ".txt", ".md", ".toml", ".cfg", ".json"}]
        except Exception:
            return ""
        if not paths:
            return "WORKSPACE IS EMPTY.\n\n"

        if focus:
            mentioned = focus.lower()
            paths.sort(key=lambda path: 0 if path.name.lower() in mentioned else 1)

        lines = ["CURRENT WORKSPACE FILES:"]
        lines.extend(f"  {relative_to(context.workspace, path)}" for path in paths[:40])

        remaining = max_chars
        bodies: list[str] = []
        for path in paths[:max_files]:
            if remaining <= 0:
                break
            try:
                text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
            except OSError:
                continue
            name = relative_to(context.workspace, path)
            block = f"\n--- {name} ---\n{text}\n--- end {name} ---\n"
            if len(block) > remaining:
                block = block[:remaining] + "\n...[truncated]\n"
            bodies.append(block)
            remaining -= len(block)

        return "\n".join(lines) + "\n\nCURRENT FILE CONTENTS:" + "".join(bodies) + "\n\n"

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
            # Not truncated. Constraints are short, deliberately chosen, and each
            # one usually exists because a run failed without it -- silently
            # dropping the seventh is how carefully-written guidance stops
            # reaching the model.
            lines.append("CONSTRAINTS:\n" + "\n".join(f"  - {item}" for item in project.constraints))
        if project.requirements:
            lines.append("REQUIREMENTS:\n" + "\n".join(f"  - {item.text}" for item in project.requirements[-8:]))
        if project.acceptance:
            lines.append(
                "ACCEPTANCE:\n"
                + "\n".join(
                    f"  - [{'x' if item.satisfied else ' '}] {item.text}"
                    + (f"  (check: {' '.join(item.check)})" if item.check else "  (NO RUNNABLE CHECK)")
                    # The NEWEST eight. Every other list here takes the tail, and
                    # taking the head instead means that on a project which
                    # accumulates requirements -- the case this engine exists for
                    # -- the criteria the model is currently trying to satisfy are
                    # the first ones dropped.
                    for item in project.acceptance[-8:]
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
            except CallTimeout as exc:
                # A model that does not answer is a failed step, not a reason to
                # wait. Recorded so DIAGNOSE sees why the step produced nothing.
                self.last_model_error = str(exc)
                return {}
            except DeadlineExceeded as exc:
                self.last_model_error = str(exc)
                return {}
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            except Exception as exc:
                self.last_model_error = f"{type(exc).__name__}: {exc}"
                return {}
            current = prompt + "\n\nYour previous reply was not valid JSON. Reply with JSON only, matching the schema."
        return {}

    def model_call_timeout(self) -> float:
        """How long one generation may take, never longer than the mission has.

        A 900-second provider timeout is sensible on its own and absurd when the
        mission has forty seconds left. Clamping here is what stops one call
        from spending a budget everything else respected.
        """

        return self.deadline.clamp(self.step_timeout_seconds)

    def _generate(self, prompt: str, schema: dict[str, Any], *, max_tokens: int) -> str:
        """Call the model under a hard time bound.

        The bound is enforced here rather than left to the HTTP layer, whose
        timeouts are per-read. With ``stream: false`` a local model sends
        nothing until it has finished, so from the socket's point of view a slow
        generation and a wedged server are indistinguishable -- and only one of
        them should be waited out.
        """

        self.deadline.require()
        timeout = self.model_call_timeout()

        def invoke() -> str:
            if hasattr(self.brain, "generate_structured"):
                try:
                    return self.brain.generate_structured(prompt, schema, max_tokens=max_tokens, temperature=0.1)
                except NotImplementedError:
                    pass
            if hasattr(self.brain, "generate_coding"):
                return self.brain.generate_coding(prompt, max_tokens=max_tokens, temperature=0.1)
            return self.brain.generate(prompt, max_tokens=max_tokens, temperature=0.1)

        return call_with_timeout(
            invoke,
            timeout,
            what="model call",
            on_abandon=lambda: self.heartbeat.beat("model_timeout", f"abandoned a call after {timeout:.0f}s"),
        )


#: Marker in a task's detail identifying it as invented by a diagnosis rather
#: than by decomposition, so :meth:`ProjectEngine._repair_budget_left` can bound them.
_AUTO_REPAIR = "[auto-repair]"

#: Tools that observe without changing anything, so a step consisting only of
#: these has not actually advanced the task.
#:
#: ``run_tests`` belongs here despite being a subprocess: running the tests is
#: how you find out whether the work is done, not a way of doing it.  Counting
#: it as progress let a live run mark "implement run()" as complete four times
#: over without the function ever being written, because each attempt ran the
#: tests and that looked like effective work.
_READ_ONLY_TOOLS = frozenset(
    {
        "list_files",
        "read_file",
        "search_text",
        "find_definition",
        "find_program",
        "git",
        "git_diff",
        "web_search",
        "fetch_url",
        "check_syntax",
        "run_tests",
    }
)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


#: pytest's documented exit codes.  A bare "exit=5" tells a small model
#: nothing, and on a live run one concluded from it that pytest was not
#: installed -- then spent twenty steps trying to install it, while the actual
#: problem was that no test file had been written yet.  Spelling the meaning out
#: costs one line and removes the whole failure mode.
_PYTEST_EXIT_MEANINGS = {
    0: "all tests passed",
    1: "tests ran and some FAILED -- read the assertion output below and fix the code under test",
    2: "the test run was interrupted",
    3: "an internal pytest error occurred",
    4: "pytest was used incorrectly (bad arguments)",
    5: (
        "pytest ran successfully but COLLECTED NO TESTS. pytest is installed and working. "
        "You must CREATE a test file named test_*.py containing functions named test_*"
    ),
}


def _explain_exit(command: list[str], returncode: Any, output: str) -> str:
    """Turn an exit code into something a small model can act on."""

    try:
        code = int(returncode)
    except (TypeError, ValueError):
        return ""

    rendered = " ".join(str(part) for part in command).lower()
    if "pytest" in rendered and code in _PYTEST_EXIT_MEANINGS:
        return f"  ({_PYTEST_EXIT_MEANINGS[code]})"
    if code == 0:
        return "  (success)"
    lowered = output.lower()
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "  (a module is missing -- install it with install_packages, or fix the import)"
    if "syntaxerror" in lowered:
        return "  (a source file does not parse -- fix the syntax error named below)"
    return ""


def _criterion_key(text: str) -> str:
    """Normalise a criterion so trivial restatements collapse together."""

    import re

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


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
