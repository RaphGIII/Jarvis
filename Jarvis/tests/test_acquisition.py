"""A capability gap is a mission, and escalation is a decision made from counts.

Two properties matter here and neither is about music.

*Escalation is evidence-driven.*  The controller decides from the performance
ledger -- how many local attempts, how many succeeded -- not from elapsed time,
not from a hunch, and not from a hard-coded "try twice then give up to the
expert".  This project has a ledger and a controller that were, until now,
imported by nothing but their own tests; the mission is what wires them to
something real.

*The expert's report is not the verdict.*  ``ExpertResult.summary`` is prose
from a provider that may not have executed anything at all -- in two earlier
escalations in this repository the expert stated plainly that its own attempts
to run code were refused by its permission layer, and the only actual execution
was the re-run performed afterwards.  So the mission re-runs the capability's
own gates over the workspace and promotes on that, or does not promote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from service.acquisition import AcquisitionMission


class StubOutcome:
    def __init__(self, usable, capability_id="", reason="", status="failed"):
        self.usable = usable
        self.capability_id = capability_id
        self.reason = reason
        self.status = status
        self.verification = {"ok": usable, "checks": []}


class StubService:
    """A capability service that succeeds after a configured number of tries."""

    def __init__(self, succeed_on=None, verify_ok=True):
        self.succeed_on = succeed_on
        self.calls = 0
        self.verify_ok = verify_ok
        self.installed = []
        self.verified = 0

    def ensure(self, goal, **kwargs):
        self.calls += 1
        if self.succeed_on is not None and self.calls >= self.succeed_on:
            return StubOutcome(True, "music.provider.spotify", status="acquired")
        return StubOutcome(False, reason=f"attempt {self.calls} did not verify")

    def _verify(self, workspace, *, extra_checks=None):
        self.verified += 1
        return {"ok": self.verify_ok, "detail": "ok" if self.verify_ok else "checks failed",
                "checks": []}

    def _install(self, capability_id, goal, workspace, verification, *, keywords=None):
        self.installed.append((capability_id, tuple(keywords or ())))
        return type("M", (), {"capability_id": capability_id})()

    @staticmethod
    def suggest_id(goal):
        return "local.suggested"


class StubProject:
    kind = "capability"
    id = "proj_1"


class StubStore:
    def __init__(self, workspace):
        self._workspace = workspace

    def find(self, goal, limit=3):
        return [StubProject()]

    def workspace_for(self, project):
        return self._workspace


class StubKernel:
    def __init__(self, tmp_path):
        self.state_root = tmp_path
        self.projects = StubStore(tmp_path / "workspace")


class StubDecision:
    def __init__(self, escalate):
        self.escalate = escalate
        self.reason = "local pass rate 0%" if escalate else "keep trying locally"
        self.evidence = ["0 of 3 passed"]


class StubController:
    def __init__(self, escalate=True):
        self._escalate = escalate
        self.decisions = 0

    def decide(self, signals):
        self.decisions += 1
        self.signals = signals
        return StubDecision(self._escalate)


class StubLedger:
    def __init__(self):
        self.attempts = []

    def record(self, attempt):
        self.attempts.append(attempt)


class StubExpertResult:
    def __init__(self, verified=True, summary="I built it and it works perfectly."):
        self.provider = "claude_code"
        self.summary = summary
        self.verified = verified
        self.status = type("S", (), {"value": "COMPLETED"})()


class StubGateway:
    def __init__(self, result=None, raises=None):
        self.result = result or StubExpertResult()
        self.raises = raises
        self.jobs = []

    def status(self):
        return {"expert_available": True}

    def submit(self, job):
        self.jobs.append(job)
        if self.raises:
            raise self.raises
        return self.result


def mission(tmp_path, *, service, controller=None, gateway=None, ledger=None):
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    return AcquisitionMission(
        service=service,
        kernel=StubKernel(tmp_path),
        gateway=gateway or StubGateway(),
        ledger=ledger or StubLedger(),
        controller=controller or StubController(escalate=False),
    )


# --------------------------------------------------------------------------
# Local first
# --------------------------------------------------------------------------

def test_a_capability_built_locally_is_never_escalated(tmp_path):
    service = StubService(succeed_on=1)
    run = mission(tmp_path, service=service)

    result = run.run("build a provider")

    assert result.acquired is True
    assert result.escalated is False
    assert result.capability_id == "music.provider.spotify"
    assert service.calls == 1


def test_every_local_attempt_is_counted_whether_it_passed_or_not(tmp_path):
    """The ledger is what the controller reasons from, so an uncounted attempt
    is an attempt that cannot influence the decision."""

    ledger = StubLedger()
    run = mission(tmp_path, service=StubService(succeed_on=None), ledger=ledger,
                  controller=StubController(escalate=False))

    run.run("build a provider")

    assert len(ledger.attempts) == AcquisitionMission.MAX_LOCAL_ATTEMPTS
    assert all(attempt.tier == "build_local" for attempt in ledger.attempts)
    assert not any(attempt.succeeded for attempt in ledger.attempts)


def test_the_controller_decides_escalation_not_a_counter(tmp_path):
    controller = StubController(escalate=False)
    run = mission(tmp_path, service=StubService(succeed_on=None), controller=controller)

    result = run.run("build a provider")

    assert controller.decisions >= 1
    assert result.escalated is False
    assert result.acquired is False


def test_the_signals_handed_to_the_controller_describe_the_real_attempts(tmp_path):
    controller = StubController(escalate=False)
    run = mission(tmp_path, service=StubService(succeed_on=None), controller=controller)

    run.run("build a provider")

    assert controller.signals.local_failures == AcquisitionMission.MAX_LOCAL_ATTEMPTS
    assert controller.signals.files_in_scope == 1


# --------------------------------------------------------------------------
# Escalation, and what it is not allowed to conclude
# --------------------------------------------------------------------------

def test_escalation_happens_only_when_the_controller_says_so(tmp_path):
    gateway = StubGateway()
    run = mission(tmp_path, service=StubService(succeed_on=None),
                  controller=StubController(escalate=True), gateway=gateway)

    result = run.run("build a provider")

    assert result.escalated is True
    assert len(gateway.jobs) == 1
    assert result.expert_used == "claude_code"


def test_the_experts_own_claim_never_promotes_anything(tmp_path):
    """The failure this guards against, observed twice in this repository: an
    expert reports success having executed nothing at all."""

    service = StubService(succeed_on=None, verify_ok=False)
    gateway = StubGateway(StubExpertResult(verified=True, summary="Done, fully working."))
    run = mission(tmp_path, service=service, controller=StubController(escalate=True),
                  gateway=gateway)

    result = run.run("build a provider")

    assert result.acquired is False
    assert service.verified == 1, "the mission must re-run the checks itself"
    assert service.installed == [], "nothing may be registered on the expert's word"
    assert "did not verify" in result.reason


def test_a_verified_expert_result_is_promoted_with_its_keywords(tmp_path):
    service = StubService(succeed_on=None, verify_ok=True)
    run = mission(tmp_path, service=service, controller=StubController(escalate=True),
                  gateway=StubGateway())

    result = run.run("build a provider", capability_id="music.provider.spotify",
                     keywords=["musik", "song"])

    assert result.acquired is True
    assert result.capability_id == "music.provider.spotify"
    assert service.installed == [("music.provider.spotify", ("musik", "song"))]


def test_the_expert_attempt_is_counted_too(tmp_path):
    """Otherwise the ledger says escalation always works, having recorded only
    the escalations that did."""

    ledger = StubLedger()
    run = mission(tmp_path, service=StubService(succeed_on=None), ledger=ledger,
                  controller=StubController(escalate=True), gateway=StubGateway())

    run.run("build a provider")

    tiers = [attempt.tier for attempt in ledger.attempts]
    assert "expert" in tiers


def test_a_gateway_that_fails_is_an_honest_failure_not_a_crash(tmp_path):
    run = mission(tmp_path, service=StubService(succeed_on=None),
                  controller=StubController(escalate=True),
                  gateway=StubGateway(raises=RuntimeError("no provider configured")))

    result = run.run("build a provider")

    assert result.acquired is False
    assert "no provider configured" in result.reason


def test_the_mission_records_every_stage_for_the_activity_view(tmp_path):
    events = []
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    run = AcquisitionMission(
        service=StubService(succeed_on=None),
        kernel=StubKernel(tmp_path),
        emit=lambda kind, payload: events.append((kind, payload)),
        gateway=StubGateway(),
        ledger=StubLedger(),
        controller=StubController(escalate=True),
    )

    result = run.run("build a provider")

    stages = [step.stage for step in result.steps]
    assert stages[0] == "start"
    assert "build_local" in stages
    assert "escalation" in stages
    assert "expert" in stages
    assert "verify" in stages
    assert events, "the mission must be visible while it runs, not only afterwards"


def test_a_service_that_raises_does_not_end_the_mission(tmp_path):
    class Exploding(StubService):
        def ensure(self, goal, **kwargs):
            self.calls += 1
            raise RuntimeError("the engine fell over")

    service = Exploding()
    run = mission(tmp_path, service=service, controller=StubController(escalate=False))

    result = run.run("build a provider")

    assert result.acquired is False
    assert service.calls == AcquisitionMission.MAX_LOCAL_ATTEMPTS
    assert any("fell over" in step.detail for step in result.steps)
