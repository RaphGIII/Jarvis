"""When ZEUS is asked for something it cannot do yet.

A capability gap is not an error.  "Play Lose Yourself on Spotify" is a
perfectly reasonable request that happens to arrive before the capability that
serves it exists, and the right response is to go and build it -- not to refuse,
and emphatically not to do something else that superficially resembles it.

The rule this module exists to enforce is that the *substitute* is never
acceptable.  A music request that cannot reach Spotify does not quietly become
a YouTube search, a generated tone, a local file, or a browser window.  It
either becomes a real Spotify capability or it becomes an honest failure, and
there is no third branch.

The escalation ladder is evidence-driven rather than time-driven:

1. BUILD_LOCAL attempts the implementation against the real machine.
2. Every attempt, pass or fail, is counted in the performance ledger.
3. :class:`~experts.escalation.EscalationController` decides from that count --
   not from anyone judging the model to look stuck.
4. Only then is an expert consulted, and only through
   :class:`~experts.gateway.ExpertGateway`, which checks the cost policy before
   a provider is chosen.
5. The expert's report is not the verdict.  Its acceptance commands are re-run
   here, and a capability is promoted only on verification this process
   performed itself.

That last point is the one that has bitten this project before, in two
directions: an expert that says it succeeded without having executed anything,
and a local capability recorded as acquired while every branch returned "Dry
run:" and nothing happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from service.events import EventType


@dataclass
class AcquisitionStep:
    """One thing that happened during a mission, for the record."""

    stage: str
    detail: str
    ok: bool = True
    seconds: float = 0.0
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "detail": self.detail[:400], "ok": self.ok,
                "seconds": round(self.seconds, 1), "at": self.at}


@dataclass
class AcquisitionResult:
    """What the mission achieved, and what it cost."""

    goal: str
    acquired: bool = False
    capability_id: str = ""
    escalated: bool = False
    expert_used: str = ""
    local_attempts: int = 0
    seconds: float = 0.0
    reason: str = ""
    steps: list[AcquisitionStep] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "acquired": self.acquired,
            "capability_id": self.capability_id,
            "escalated": self.escalated,
            "expert_used": self.expert_used,
            "local_attempts": self.local_attempts,
            "seconds": round(self.seconds, 1),
            "reason": self.reason,
            "steps": [step.to_dict() for step in self.steps],
            "verification": self.verification,
        }


def repair_goal(specification: str, defect: str) -> str:
    """A brief for fixing working code, with the defect first.

    Ordering is the whole point.  The defect used to be appended *after* the
    full capability specification, and the planner read a build brief and
    planned a build: handed a 794-line implementation that failed one check, it
    decomposed the work into "implement the run function", "implement search",
    "implement playback control", "implement current state reporting", grew the
    file by fifty lines re-implementing what already worked, and never touched
    the one wrong constant it had been sent to change.

    A model plans from what it reads first.  So a repair brief opens with the
    defect and says in its first line that this is not a rebuild; the
    specification follows as reference material for a file that already
    satisfies it.
    """

    return (
        "REPAIR an existing, working implementation. This is NOT a rebuild.\n\n"
        f"THE DEFECT TO FIX:\n{defect}\n\n"
        "main.py already exists in the workspace and already implements this capability. "
        "It passes every check except the one implied above. Your entire job is to make "
        "that one check pass.\n\n"
        "  - Find the specific code responsible for the defect and change it.\n"
        "  - Do NOT re-implement actions that already work.\n"
        "  - Do NOT add features, restructure, reformat or rename anything.\n"
        "  - The smaller the change, the more likely it is to be correct.\n"
        "  - If an anchor will not match, read the region first and anchor on a line you "
        "have just read. Do not retype the file from memory.\n\n"
        "For reference only -- the contract the existing file already satisfies:\n\n"
        f"{specification}"
    )


class AcquisitionMission:
    """Builds a missing capability, escalating only on counted evidence."""

    #: How many BUILD_LOCAL runs to attempt before the controller is even asked.
    #: The controller has its own minimum; this stops a single unlucky run from
    #: reaching for an expert, and stops an endless local loop from never doing so.
    MIN_LOCAL_ATTEMPTS = 1
    MAX_LOCAL_ATTEMPTS = 3

    def __init__(
        self,
        *,
        service: Any,
        kernel: Any,
        emit: Callable[[EventType, dict[str, Any]], None] | None = None,
        gateway: Any = None,
        ledger: Any = None,
        controller: Any = None,
        memory: Any = None,
    ) -> None:
        self.service = service
        self.kernel = kernel
        self._emit = emit or (lambda *_args, **_kwargs: None)
        self._gateway = gateway
        self._ledger = ledger
        self._controller = controller
        self._memory = memory

    # -- wiring ----------------------------------------------------------

    @property
    def ledger(self) -> Any:
        if self._ledger is None:
            from experts.escalation import PerformanceLedger

            self._ledger = PerformanceLedger(Path(self.kernel.state_root) / "performance.jsonl")
        return self._ledger

    @property
    def controller(self) -> Any:
        if self._controller is None:
            from experts.escalation import EscalationController

            self._controller = EscalationController(self.ledger)
        return self._controller

    @property
    def memory(self) -> Any:
        """Lessons from work that was independently verified.

        Recall is gated on :attr:`experts.memory.Lesson.verified`, so what a
        provider *said* it did never becomes something a later run is taught.
        An unverified lesson is a rumour, and putting a rumour in a prompt as
        though it were knowledge is how one bad answer becomes several.
        """

        if self._memory is None:
            from experts.memory import ExpertMemory

            self._memory = ExpertMemory(Path(self.kernel.state_root) / "expert_lessons.jsonl")
        return self._memory

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            from experts.claude_code import ClaudeCodeExpert
            from experts.gateway import ExpertGateway
            from runtime.cost_policy import CostPolicy

            # The policy is loaded and passed explicitly rather than left to a
            # default, because "which channels may be spent on" is exactly the
            # decision that must not be implicit.
            self._gateway = ExpertGateway([ClaudeCodeExpert()], policy=CostPolicy.load())
        return self._gateway

    # -- reporting -------------------------------------------------------

    def _step(self, result: AcquisitionResult, stage: str, detail: str, *, ok: bool = True,
              seconds: float = 0.0) -> None:
        from datetime import datetime, timezone

        step = AcquisitionStep(stage=stage, detail=detail, ok=ok, seconds=seconds,
                               at=datetime.now(timezone.utc).isoformat())
        result.steps.append(step)
        # Published so the mission is visible in Activity while it runs, rather
        # than being a silence followed by an outcome.
        self._emit(EventType.PROGRESS, {"summary": f"acquisition/{stage}: {detail[:180]}",
                                        "stage": stage, "ok": ok})

    # -- the mission -----------------------------------------------------

    def run(
        self,
        goal: str,
        *,
        capability_id: str = "",
        keywords: list[str] | None = None,
        extra_checks: list[Any] | None = None,
        max_steps: int = 60,
        max_seconds: float = 1800.0,
        expert_constraints: list[str] | None = None,
        expert_acceptance: list[tuple[str, list[str]]] | None = None,
        repair: str = "",
    ) -> AcquisitionResult:
        """Attempt locally, count the evidence, escalate only if it says to.

        ``repair`` names a defect in an already-registered capability.  It is
        disabled first, so the resolver stops handing out something known to be
        broken and the rebuild seeds from the installed source rather than from
        a blank skeleton -- a repair improves what exists.
        """

        started = time.perf_counter()
        result = AcquisitionResult(goal=goal)
        from experts.escalation import Attempt, classify_goal

        task_class = classify_goal(goal)
        if repair and capability_id:
            self._retire(capability_id, repair, result)
            goal = repair_goal(goal, repair)
            result.goal = goal
        self._step(result, "start",
                   ("repairing " + capability_id) if repair else "missing capability"
                   + f"; task class {task_class!r}")

        # What worked last time, before spending anything finding out again.
        remembered = self._recall(goal, task_class, result)
        if remembered:
            goal = remembered + goal
            result.goal = goal

        outcome = None
        for attempt in range(self.MAX_LOCAL_ATTEMPTS):
            result.local_attempts = attempt + 1
            attempt_started = time.perf_counter()
            self._step(result, "build_local", f"attempt {attempt + 1} of {self.MAX_LOCAL_ATTEMPTS}")
            try:
                outcome = self.service.ensure(
                    goal,
                    max_steps=max_steps,
                    keywords=list(keywords or []),
                    max_seconds=max_seconds,
                    extra_checks=list(extra_checks or []),
                    # Named rather than derived from the goal text. The id the
                    # build uses has to be the id it registers under, or a
                    # rebuild looks up something that does not exist and starts
                    # from a blank skeleton instead of the working version.
                    capability_id=capability_id,
                )
            except Exception as exc:
                outcome = None
                self._step(result, "build_local", f"raised: {type(exc).__name__}: {exc}", ok=False,
                           seconds=time.perf_counter() - attempt_started)
            else:
                self._step(
                    result, "build_local",
                    f"{outcome.status}: {outcome.reason[:200] or 'verified'}",
                    ok=bool(outcome.usable),
                    seconds=time.perf_counter() - attempt_started,
                )
            self.ledger.record(
                Attempt(task_class=task_class, tier="build_local",
                        succeeded=bool(outcome and outcome.usable),
                        seconds=time.perf_counter() - attempt_started)
            )
            if outcome is not None and outcome.usable:
                result.acquired = True
                result.capability_id = outcome.capability_id
                result.verification = dict(outcome.verification or {})
                result.seconds = time.perf_counter() - started
                self._step(result, "acquired", f"{outcome.capability_id} verified locally")
                self._remember(goal, task_class, result, provider="build_local")
                return result

            if attempt + 1 >= self.MIN_LOCAL_ATTEMPTS:
                decision = self._should_escalate(task_class, result)
                if decision is not None and decision.escalate:
                    self._step(result, "escalation",
                               f"{decision.reason} | {'; '.join(decision.evidence[:3])}")
                    escalated = self._escalate(
                        goal, result,
                        capability_id=capability_id,
                        keywords=keywords,
                        extra_checks=extra_checks,
                        constraints=expert_constraints,
                        acceptance=expert_acceptance,
                        task_class=task_class,
                        repair_mode=bool(repair),
                    )
                    result.seconds = time.perf_counter() - started
                    return escalated

        result.reason = (
            outcome.reason if outcome is not None and outcome.reason
            else "local attempts did not produce a verified capability"
        )
        result.seconds = time.perf_counter() - started
        self._step(result, "failed", result.reason, ok=False)
        return result

    def _retire(self, capability_id: str, defect: str, result: AcquisitionResult) -> None:
        """Stop handing out a capability that is known to be broken.

        Disabling rather than deleting: the installed source is what the
        rebuild seeds from, and the record of what was registered and why it
        stopped being trusted is worth keeping.
        """

        try:
            self.service.registry.disable(capability_id, reason=defect[:300])
        except Exception as exc:
            self._step(result, "retire", f"could not disable {capability_id}: {exc}", ok=False)
            return
        self._step(result, "retire", f"disabled {capability_id}: {defect[:160]}")

    def _should_escalate(self, task_class: str, result: AcquisitionResult) -> Any:
        from experts.escalation import EscalationSignals

        try:
            return self.controller.decide(
                EscalationSignals(
                    task_class=task_class,
                    local_failures=result.local_attempts,
                    distinct_diagnoses=len({step.detail[:60] for step in result.steps
                                            if step.stage == "build_local" and not step.ok}),
                    abandoned_tasks=result.local_attempts,
                    files_in_scope=1,
                )
            )
        except Exception as exc:
            self._step(result, "escalation", f"controller unavailable: {exc}", ok=False)
            return None

    # -- escalation ------------------------------------------------------

    def _escalate(
        self,
        goal: str,
        result: AcquisitionResult,
        *,
        capability_id: str,
        keywords: list[str] | None,
        extra_checks: list[Any] | None,
        constraints: list[str] | None,
        acceptance: list[tuple[str, list[str]]] | None,
        task_class: str,
        repair_mode: bool = False,
    ) -> AcquisitionResult:
        """Ask an expert, then verify its work here before believing any of it."""

        from experts.contracts import ExpertJob
        from experts.escalation import Attempt

        result.escalated = True
        workspace = self._workspace_for(goal)
        if workspace is None:
            result.reason = "no workspace was available to escalate into"
            self._step(result, "escalation", result.reason, ok=False)
            return result

        try:
            status = self.gateway.status()
        except Exception as exc:
            status = {"error": str(exc)}
        self._step(result, "escalation", f"gateway status: {status}")

        # A repair is not a rewrite, and the expert may be unable to run
        # anything to find out. Observed: asked to fix one wrong constant in a
        # 794-line module, an expert that said plainly its permissions refused
        # to execute python grew the file to 979 lines, introduced a Windows
        # path mangled by string escaping, and left the constant untouched.
        # Editing blind is a reason to change less, not more.
        repair_rules = [
            "This is a REPAIR of working code, not a rewrite. Make the smallest change that "
            "fixes the stated defect and nothing else.",
            "Do not restructure, reformat, rename, or 'improve' anything you were not asked "
            "to fix. Every unrelated edit is a new way for this to fail.",
            "You may be unable to execute anything here. If so, say so, and be more "
            "conservative rather than less: an unverified broad change is worse than an "
            "unverified narrow one.",
            "Windows paths must use raw strings, forward slashes or pathlib. A backslash in "
            "an ordinary string is an escape and the path will not be what it looks like.",
        ] if repair_mode else []

        job = ExpertJob(
            goal=goal,
            workspace=Path(workspace),
            constraints=list(constraints or []) + repair_rules,
            acceptance=list(acceptance or []),
            previous_failures=[step.detail for step in result.steps
                               if step.stage == "build_local" and not step.ok],
            expected_artifacts=["main.py"],
            allowed_paths=["main.py", "test_capability.py"],
            max_seconds=1500.0,
        )

        started = time.perf_counter()
        try:
            expert = self.gateway.submit(job)
        except Exception as exc:
            result.reason = f"the expert gateway failed: {type(exc).__name__}: {exc}"
            self._step(result, "escalation", result.reason, ok=False)
            return result

        elapsed = time.perf_counter() - started
        result.expert_used = getattr(expert, "provider", "")
        self._step(
            result, "expert",
            f"{getattr(expert.status, 'value', expert.status)}: {expert.summary[:200]}",
            ok=bool(getattr(expert, "verified", False)),
            seconds=elapsed,
        )
        self.ledger.record(Attempt(task_class=task_class, tier="expert",
                                   succeeded=bool(getattr(expert, "verified", False)),
                                   seconds=elapsed))

        # The expert's own report is informational. What decides the outcome is
        # this process re-running the capability's own gates over the workspace
        # the expert edited.
        self._step(result, "verify", "re-running the capability checks here")
        verification = self._verify_workspace(workspace, extra_checks)
        result.verification = verification
        if not verification.get("ok"):
            result.reason = f"the expert's work did not verify: {verification.get('detail', '')[:200]}"
            self._step(result, "verify", result.reason, ok=False)
            return result

        installed = self._install(goal, workspace, capability_id=capability_id,
                                  keywords=keywords, verification=verification)
        if installed is None:
            result.reason = "verified, but the capability could not be registered"
            self._step(result, "promote", result.reason, ok=False)
            return result

        result.acquired = True
        result.capability_id = installed
        self._step(result, "promote", f"registered {installed} after independent verification")
        self._remember(goal, task_class, result, provider=result.expert_used or "expert")
        return result

    # -- what gets carried forward ---------------------------------------

    def _recall(self, goal: str, task_class: str, result: AcquisitionResult) -> str:
        """Verified lessons relevant to this goal, as prompt text."""

        try:
            context = self.memory.context_for(goal, task_class=task_class)
        except Exception as exc:
            self._step(result, "recall", f"memory unavailable: {exc}", ok=False)
            return ""
        if context:
            self._step(result, "recall",
                       f"{context.count('A previous task of this kind')} verified lesson(s) recalled")
        return context

    def _remember(self, goal: str, task_class: str, result: AcquisitionResult,
                  *, provider: str) -> None:
        """Write down what worked, with the verification that proved it.

        Only ever called after this process has re-run the checks itself, and
        the verification recorded is that re-run -- not the provider's account
        of it. ``Lesson.verified`` gates recall, so a lesson without real
        evidence can be written but can never be taught.
        """

        from experts.memory import Lesson

        checks = [
            {"criterion": str(check.get("name", "")), "passed": bool(check.get("ok"))}
            for check in (result.verification.get("checks") or [])
        ]
        if not checks:
            self._step(result, "remember", "nothing verified to remember", ok=False)
            return
        try:
            self.memory.record(
                Lesson(
                    task_class=task_class,
                    goal=goal[:1500],
                    failed_approaches=[step.detail for step in result.steps
                                       if step.stage == "build_local" and not step.ok][:6],
                    successful_approach=next(
                        (step.detail for step in reversed(result.steps)
                         if step.stage in {"expert", "build_local"} and step.ok), ""),
                    verification=checks,
                    provider=provider,
                    pattern=f"capability {result.capability_id} satisfied: "
                            + ", ".join(item["criterion"] for item in checks if item["passed"]),
                )
            )
            self._step(result, "remember", f"lesson recorded ({len(checks)} verified checks)")
        except Exception as exc:
            self._step(result, "remember", f"could not record: {exc}", ok=False)

    # -- helpers that reach into the capability service -------------------

    def _workspace_for(self, goal: str) -> Path | None:
        """The workspace of the *latest* acquisition attempt for this goal.

        Sorted by recency here rather than trusting the search order.  Three
        attempts leave three near-identical projects, and ``ProjectStore.find``
        ranks by keyword overlap with a bonus for non-terminal state -- which
        happens to tie-break on ``updated_at`` today, but would hand the expert
        attempt one's abandoned workspace the moment those scores diverged.
        The expert should be fixing the most recent attempt, not an older one.
        """

        try:
            projects = [
                project for project in self.kernel.projects.find(goal, limit=6)
                if getattr(project, "kind", "") == "capability"
            ]
        except Exception:
            return None
        projects.sort(key=lambda project: getattr(project, "updated_at", ""), reverse=True)
        for project in projects:
            try:
                return Path(self.kernel.projects.workspace_for(project))
            except Exception:
                continue
        return None

    def _verify_workspace(self, workspace: Path, extra_checks: list[Any] | None) -> dict[str, Any]:
        try:
            return self.service._verify(Path(workspace), extra_checks=list(extra_checks or []))
        except Exception as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "checks": []}

    def _install(self, goal: str, workspace: Path, *, capability_id: str,
                 keywords: list[str] | None, verification: dict[str, Any]) -> str | None:
        """Register the verified workspace, carrying the verification with it.

        ``keywords`` is not decoration.  It is what the knowledge graph indexes,
        and it is the only reason a later "spiel was von den Beatles" finds a
        capability whose id says *spotify* -- lexical matching cannot bridge
        music to Spotify on its own.
        """

        try:
            manifest = self.service._install(
                capability_id or self.service.suggest_id(goal),
                goal,
                Path(workspace),
                verification,
                keywords=list(keywords or []),
            )
        except Exception as exc:
            self._emit(EventType.ERROR, {"error": f"registration failed: {type(exc).__name__}: {exc}"})
            return None
        return str(getattr(manifest, "capability_id", "")) or None
