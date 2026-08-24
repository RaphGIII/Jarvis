"""The durable representation of a piece of work.

A complex request is not an LLM call; it is a :class:`Project` that outlives the
conversation, the process and the machine's uptime.  Everything the autonomous
loop learns has to land in one of the structures here, because anything held
only in a prompt is lost the moment the process exits -- and an agent that
forgets what it already tried cannot make progress over hundreds of steps.

The vocabulary is deliberately domain-neutral.  A project has requirements,
acceptance criteria, tasks, findings, decisions, experiments and blockers.
Nothing here says "software": the same structures describe a research project or
a capability acquisition, which is what keeps the control loop general.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class ProjectState(str, Enum):
    """Where a project is in its lifecycle."""

    DRAFT = "DRAFT"
    INVESTIGATING = "INVESTIGATING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    #: Waiting on something only a human can supply.
    BLOCKED = "BLOCKED"
    #: Stopped by a resource limit; resumable.
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"

    @property
    def terminal(self) -> bool:
        return self in {ProjectState.COMPLETED, ProjectState.ABANDONED}


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    #: Attempted repeatedly without success; kept for the record, not retried.
    ABANDONED = "ABANDONED"
    SKIPPED = "SKIPPED"


class Phase(str, Enum):
    """One turn of the autonomous control loop."""

    INVESTIGATE = "INVESTIGATE"
    DECOMPOSE = "DECOMPOSE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    VERIFY = "VERIFY"
    DIAGNOSE = "DIAGNOSE"
    REPLAN = "REPLAN"
    ACCEPT = "ACCEPT"


class StopReason(str, Enum):
    """Why the loop stopped.  Only ACCEPTED means the goal was met."""

    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    STEP_LIMIT = "STEP_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    FAILURE_LIMIT = "FAILURE_LIMIT"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class AcceptanceCriterion:
    """An objective, checkable condition for the project being done.

    ``check`` is a deterministic command (``["python", "-m", "pytest", "-q"]``).
    A criterion without one can still be recorded, but it can never be *proved*,
    so :meth:`Project.acceptance_satisfied` refuses to count it.  That refusal is
    what stops the system from declaring victory on its own say-so.
    """

    text: str
    id: str = field(default_factory=lambda: _new_id("ac"))
    check: list[str] = field(default_factory=list)
    satisfied: bool = False
    last_evidence: str = ""
    last_checked_at: str = ""

    @property
    def objective(self) -> bool:
        return bool(self.check)


@dataclass
class Requirement:
    """Something the user asked for, accumulated across conversations."""

    text: str
    id: str = field(default_factory=lambda: _new_id("req"))
    source: str = "user"
    added_at: str = field(default_factory=_now)
    satisfied: bool = False


@dataclass
class Task:
    """A unit of work small enough for one execution attempt."""

    title: str
    id: str = field(default_factory=lambda: _new_id("task"))
    detail: str = ""
    status: TaskStatus = TaskStatus.PENDING
    #: Task ids that must be DONE first.
    depends_on: list[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    #: How many times this task was completed and then reopened because the
    #: behaviour it claimed to deliver was still failing.  Bounded so a task
    #: cannot oscillate between DONE and reopened for the whole step budget.
    reopenings: int = 0
    max_reopenings: int = 2
    created_at: str = field(default_factory=_now)
    completed_at: str = ""
    last_error: str = ""
    evidence: list[str] = field(default_factory=list)
    kind: str = "software"

    @property
    def open(self) -> bool:
        return self.status in {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    @property
    def reopenable(self) -> bool:
        """Whether a finished task may be given a fresh attempt budget.

        ``attempts`` is a *run-length* budget: three failures in a row means
        stop trying this way.  A completion in between breaks that run, so a
        task marked DONE whose behaviour later proves still broken is not out
        of budget -- it was never done, and the attempts were spent under a
        false belief.

        Found live: on a long persistent project every task eventually reached
        three attempts, and because reopening skips exhausted tasks, the
        project permanently lost the ability to repair anything.  The loop
        reported "no further repair avenue is available" while the real reason
        was a lifetime counter answering a question about right now.
        """

        return self.reopenings < self.max_reopenings


@dataclass
class Finding:
    """Something discovered to be true, with where it came from.

    ``source`` separates a fact read out of a file or a documentation page from
    a model's guess.  Keeping that distinction is what lets a later step trust
    the first and re-check the second.
    """

    text: str
    id: str = field(default_factory=lambda: _new_id("find"))
    source: str = "observation"
    reference: str = ""
    confidence: float = 0.7
    added_at: str = field(default_factory=_now)


@dataclass
class Decision:
    """An architecture or approach decision, and what it ruled out."""

    text: str
    id: str = field(default_factory=lambda: _new_id("dec"))
    rationale: str = ""
    alternatives: list[str] = field(default_factory=list)
    added_at: str = field(default_factory=_now)


@dataclass
class Experiment:
    """A recorded attempt: hypothesis, what happened, what it taught.

    Failed experiments are as valuable as successful ones -- they are the only
    thing stopping the loop from re-trying an approach it already disproved.
    """

    hypothesis: str
    id: str = field(default_factory=lambda: _new_id("exp"))
    method: str = ""
    outcome: str = ""
    succeeded: bool = False
    lesson: str = ""
    task_id: str = ""
    added_at: str = field(default_factory=_now)


@dataclass
class Blocker:
    """Something the project cannot get past on its own."""

    text: str
    id: str = field(default_factory=lambda: _new_id("blk"))
    #: True when only a human can clear it (a credential, a hardware decision).
    needs_user: bool = False
    resolved: bool = False
    resolution: str = ""
    added_at: str = field(default_factory=_now)


@dataclass
class Artifact:
    """Something the project produced."""

    path: str
    id: str = field(default_factory=lambda: _new_id("art"))
    kind: str = "file"
    description: str = ""
    added_at: str = field(default_factory=_now)


@dataclass
class StepRecord:
    """One turn of the loop, for the trajectory and later dataset export."""

    phase: Phase
    summary: str = ""
    id: str = field(default_factory=lambda: _new_id("step"))
    index: int = 0
    task_id: str = ""
    success: bool = True
    #: Whether this step advanced the *work*, as opposed to merely completing.
    #: A diagnosis can succeed brilliantly and move nothing, and abandoning an
    #: impossible task fails while genuinely advancing the plan -- so the loop's
    #: stuck-detector counts productivity rather than success.  ``None`` means
    #: "no opinion": the engine then treats productivity as equal to success.
    productive: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    at: str = field(default_factory=_now)


@dataclass
class ResourceLimits:
    """The budget for one autonomous session.

    Difficult work is allowed to take hundreds of steps; what is not allowed is
    for it to take them *silently and forever*.  Every limit here produces a
    PAUSED project that can be resumed, never a lost one.
    """

    max_steps: int = 120
    max_seconds: float = 3600.0
    #: Consecutive failed steps before the loop concedes it is stuck.
    max_consecutive_failures: int = 8
    max_task_attempts: int = 3
    #: Seconds for any single tool or command.
    step_timeout_seconds: float = 600.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Project:
    """The durable aggregate.  Everything the loop knows lives here."""

    goal: str
    id: str = field(default_factory=lambda: _new_id("proj"))
    title: str = ""
    state: ProjectState = ProjectState.DRAFT
    kind: str = "software"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    #: Where the work happens.  For self-development this is a candidate
    #: worktree; for a new project, an isolated workspace directory.
    workspace: str = ""
    repository: str = ""

    requirements: list[Requirement] = field(default_factory=list)
    acceptance: list[AcceptanceCriterion] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)

    limits: ResourceLimits = field(default_factory=ResourceLimits)
    #: Steps and seconds already spent, accumulated across sessions.
    steps_spent: int = 0
    seconds_spent: float = 0.0
    last_stop_reason: str = ""
    #: Free-form per-project state (capability ids, model overrides, ...).
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------

    def open_tasks(self) -> list[Task]:
        return [task for task in self.tasks if task.open]

    def ready_tasks(self) -> list[Task]:
        """Open tasks whose dependencies are satisfied and which have attempts left."""

        done = {task.id for task in self.tasks if task.status is TaskStatus.DONE}
        return [
            task
            for task in self.tasks
            if task.open and not task.exhausted and all(dependency in done for dependency in task.depends_on)
        ]

    def next_task(self) -> Task | None:
        ready = self.ready_tasks()
        if not ready:
            return None
        # Prefer a task already under way, so a half-finished piece of work is
        # completed rather than abandoned for a fresh one.
        in_progress = [task for task in ready if task.status is TaskStatus.IN_PROGRESS]
        return (in_progress or ready)[0]

    def task(self, task_id: str) -> Task | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    def active_blockers(self) -> list[Blocker]:
        return [blocker for blocker in self.blockers if not blocker.resolved]

    def user_blockers(self) -> list[Blocker]:
        return [blocker for blocker in self.active_blockers() if blocker.needs_user]

    def objective_criteria(self) -> list[AcceptanceCriterion]:
        return [item for item in self.acceptance if item.objective]

    def acceptance_satisfied(self) -> bool:
        """True only when every objectively checkable criterion has passed.

        A project with no objective criteria is never automatically accepted:
        without a deterministic check there is no evidence, and declaring
        success without evidence is precisely the failure mode this system is
        built to avoid.
        """

        objective = self.objective_criteria()
        if not objective:
            return False
        return all(item.satisfied for item in objective)

    def progress(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for task in self.tasks:
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
        objective = self.objective_criteria()
        return {
            "state": self.state.value,
            "tasks": by_status,
            "tasks_total": len(self.tasks),
            "acceptance_total": len(self.acceptance),
            "acceptance_objective": len(objective),
            "acceptance_satisfied": sum(1 for item in objective if item.satisfied),
            "steps_spent": self.steps_spent,
            "seconds_spent": round(self.seconds_spent, 1),
            "blockers": len(self.active_blockers()),
            "experiments": len(self.experiments),
        }

    # -- mutations -------------------------------------------------------

    def touch(self) -> None:
        self.updated_at = _now()

    def add_requirement(self, text: str, *, source: str = "user") -> Requirement:
        """Add a requirement, ignoring an exact restatement of a known one."""

        normalised = text.strip()
        for existing in self.requirements:
            if existing.text.strip().lower() == normalised.lower():
                return existing
        requirement = Requirement(text=normalised, source=source)
        self.requirements.append(requirement)
        self.touch()
        return requirement

    def add_acceptance(self, text: str, *, check: list[str] | None = None) -> AcceptanceCriterion:
        criterion = AcceptanceCriterion(text=text.strip(), check=list(check or []))
        self.acceptance.append(criterion)
        self.touch()
        return criterion

    def add_task(self, title: str, *, detail: str = "", depends_on: list[str] | None = None, kind: str = "software") -> Task:
        task = Task(title=title.strip(), detail=detail, depends_on=list(depends_on or []), kind=kind)
        self.tasks.append(task)
        self.touch()
        return task

    def add_finding(self, text: str, *, source: str = "observation", reference: str = "", confidence: float = 0.7) -> Finding:
        finding = Finding(text=text.strip(), source=source, reference=reference, confidence=confidence)
        self.findings.append(finding)
        self.touch()
        return finding

    def add_decision(self, text: str, *, rationale: str = "", alternatives: list[str] | None = None) -> Decision:
        decision = Decision(text=text.strip(), rationale=rationale, alternatives=list(alternatives or []))
        self.decisions.append(decision)
        self.touch()
        return decision

    def add_experiment(self, hypothesis: str, **kwargs: Any) -> Experiment:
        experiment = Experiment(hypothesis=hypothesis.strip(), **kwargs)
        self.experiments.append(experiment)
        self.touch()
        return experiment

    def add_blocker(self, text: str, *, needs_user: bool = False) -> Blocker:
        for existing in self.active_blockers():
            if existing.text.strip().lower() == text.strip().lower():
                return existing
        blocker = Blocker(text=text.strip(), needs_user=needs_user)
        self.blockers.append(blocker)
        self.touch()
        return blocker

    def resolve_blocker(self, blocker_id: str, resolution: str = "") -> None:
        for blocker in self.blockers:
            if blocker.id == blocker_id:
                blocker.resolved = True
                blocker.resolution = resolution
                self.touch()

    def add_artifact(self, path: str, *, kind: str = "file", description: str = "") -> Artifact:
        for existing in self.artifacts:
            if existing.path == path:
                return existing
        artifact = Artifact(path=path, kind=kind, description=description)
        self.artifacts.append(artifact)
        self.touch()
        return artifact

    def add_reference(self, url: str, *, title: str = "", summary: str = "") -> dict[str, Any]:
        for existing in self.references:
            if existing.get("url") == url:
                return existing
        reference = {"url": url, "title": title, "summary": summary, "retrieved_at": _now()}
        self.references.append(reference)
        self.touch()
        return reference

    def record_step(self, record: StepRecord) -> StepRecord:
        record.index = len(self.steps) + 1
        self.steps.append(record)
        self.steps_spent += 1
        self.touch()
        return record

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["limits"] = self.limits.to_dict()
        for index, task in enumerate(self.tasks):
            data["tasks"][index]["status"] = task.status.value
        for index, step in enumerate(self.steps):
            data["steps"][index]["phase"] = step.phase.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        project = cls(goal=str(data.get("goal", "")))
        project.id = str(data.get("id") or project.id)
        project.title = str(data.get("title", ""))
        project.state = _enum(ProjectState, data.get("state"), ProjectState.DRAFT)
        project.kind = str(data.get("kind", "software"))
        project.created_at = str(data.get("created_at") or project.created_at)
        project.updated_at = str(data.get("updated_at") or project.updated_at)
        project.workspace = str(data.get("workspace", ""))
        project.repository = str(data.get("repository", ""))
        project.constraints = [str(item) for item in data.get("constraints") or []]
        project.references = [dict(item) for item in data.get("references") or [] if isinstance(item, dict)]

        project.requirements = [Requirement(**_only(Requirement, item)) for item in data.get("requirements") or []]
        project.acceptance = [AcceptanceCriterion(**_only(AcceptanceCriterion, item)) for item in data.get("acceptance") or []]
        project.findings = [Finding(**_only(Finding, item)) for item in data.get("findings") or []]
        project.decisions = [Decision(**_only(Decision, item)) for item in data.get("decisions") or []]
        project.experiments = [Experiment(**_only(Experiment, item)) for item in data.get("experiments") or []]
        project.blockers = [Blocker(**_only(Blocker, item)) for item in data.get("blockers") or []]
        project.artifacts = [Artifact(**_only(Artifact, item)) for item in data.get("artifacts") or []]

        project.tasks = []
        for item in data.get("tasks") or []:
            fields = _only(Task, item)
            fields["status"] = _enum(TaskStatus, item.get("status"), TaskStatus.PENDING)
            project.tasks.append(Task(**fields))

        project.steps = []
        for item in data.get("steps") or []:
            fields = _only(StepRecord, item)
            fields["phase"] = _enum(Phase, item.get("phase"), Phase.OBSERVE)
            project.steps.append(StepRecord(**fields))

        limits = data.get("limits") or {}
        project.limits = ResourceLimits(**_only(ResourceLimits, limits)) if limits else ResourceLimits()
        project.steps_spent = int(data.get("steps_spent", 0))
        project.seconds_spent = float(data.get("seconds_spent", 0.0))
        project.last_stop_reason = str(data.get("last_stop_reason", ""))
        project.metadata = dict(data.get("metadata") or {})
        return project


def _only(cls: Any, payload: Any) -> dict[str, Any]:
    """Keep only the keys a dataclass accepts.

    Records written by an older schema version load cleanly instead of raising
    on a field that has since been renamed -- a project must survive its own
    software being upgraded.
    """

    if not isinstance(payload, dict):
        return {}
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    return {key: value for key, value in payload.items() if key in allowed}


def _enum(cls: Any, value: Any, default: Any) -> Any:
    try:
        return cls(str(value))
    except (ValueError, TypeError):
        return default
