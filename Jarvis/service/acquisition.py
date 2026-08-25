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
    ) -> None:
        self.service = service
        self.kernel = kernel
        self._emit = emit or (lambda *_args, **_kwargs: None)
        self._gateway = gateway
        self._ledger = ledger
        self._controller = controller

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
    ) -> AcquisitionResult:
        """Attempt locally, count the evidence, escalate only if it says to."""

        started = time.perf_counter()
        result = AcquisitionResult(goal=goal)
        from experts.escalation import Attempt, classify_goal

        task_class = classify_goal(goal)
        self._step(result, "start", f"missing capability; task class {task_class!r}")

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

        job = ExpertJob(
            goal=goal,
            workspace=Path(workspace),
            constraints=list(constraints or []),
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
        return result

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
