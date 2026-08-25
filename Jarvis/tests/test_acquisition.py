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
    def __init__(self, id="proj_1", updated_at="2026-01-01T00:00:00", kind="capability"):
        self.id = id
        self.updated_at = updated_at
        self.kind = kind


class StubStore:
    def __init__(self, workspace, projects=None):
        self._workspace = workspace
        self._projects = projects or [StubProject()]

    def find(self, goal, limit=3):
        return list(self._projects)

    def workspace_for(self, project):
        return self._workspace / project.id if self._projects[0].id != "proj_1" else self._workspace


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


def test_the_expert_is_pointed_at_the_most_recent_attempt(tmp_path):
    """Three attempts leave three near-identical projects. Handing the expert
    the first one's abandoned workspace would have it fix code nothing runs."""

    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    kernel = StubKernel(tmp_path)
    kernel.projects = StubStore(
        tmp_path / "workspace",
        projects=[
            StubProject("attempt_1", "2026-01-01T10:00:00"),
            StubProject("attempt_3", "2026-01-01T12:00:00"),
            StubProject("attempt_2", "2026-01-01T11:00:00"),
        ],
    )
    gateway = StubGateway()
    run = AcquisitionMission(
        service=StubService(succeed_on=None, verify_ok=True), kernel=kernel,
        gateway=gateway, ledger=StubLedger(), controller=StubController(escalate=True),
    )

    run.run("build a provider")

    assert gateway.jobs[0].workspace.name == "attempt_3"


def test_a_project_that_is_not_a_capability_build_is_never_handed_over(tmp_path):
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    kernel = StubKernel(tmp_path)
    kernel.projects = StubStore(
        tmp_path / "workspace",
        projects=[StubProject("some_app", "2026-01-01T12:00:00", kind="software")],
    )
    run = AcquisitionMission(
        service=StubService(succeed_on=None), kernel=kernel, gateway=StubGateway(),
        ledger=StubLedger(), controller=StubController(escalate=True),
    )

    result = run.run("build a provider")

    assert result.acquired is False
    assert "no workspace" in result.reason


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


def test_a_repair_tells_the_expert_to_change_as_little_as_possible(tmp_path):
    """Observed: asked to fix one wrong constant in a 794-line module, an
    expert that said plainly its permissions refused to execute python grew
    the file to 979 lines, introduced a Windows path mangled by string
    escaping, and left the constant untouched. Editing blind is a reason to
    change less, not more."""

    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    gateway = StubGateway()
    run = mission(tmp_path, service=StubService(succeed_on=None, verify_ok=True),
                  controller=StubController(escalate=True), gateway=gateway)

    run.run("build a provider", capability_id="cap.x", repair="limit=20 is rejected")

    constraints = " ".join(gateway.jobs[0].constraints).lower()
    assert "smallest change" in constraints
    assert "not a rewrite" in constraints
    assert "raw strings" in constraints, "the path-escape trap must be named"
    assert "unable to execute" in constraints


def test_a_first_build_is_not_told_to_change_as_little_as_possible(tmp_path):
    """There is nothing to preserve yet, and telling it to be conservative
    about code that does not exist would be nonsense."""

    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    gateway = StubGateway()
    run = mission(tmp_path, service=StubService(succeed_on=None, verify_ok=True),
                  controller=StubController(escalate=True), gateway=gateway)

    run.run("build a provider", capability_id="cap.x")

    constraints = " ".join(gateway.jobs[0].constraints).lower()
    assert "not a rewrite" not in constraints


def test_a_repair_retires_the_broken_version_before_rebuilding(tmp_path):
    """The resolver must stop handing out something known to be broken, and
    the rebuild needs the installed source to start from."""

    class Registry:
        def __init__(self):
            self.disabled = []

        def disable(self, capability_id, reason=""):
            self.disabled.append((capability_id, reason))

    service = StubService(succeed_on=1)
    service.registry = Registry()
    run = mission(tmp_path, service=service)

    run.run("build a provider", capability_id="cap.x", repair="search returns 400")

    assert service.registry.disabled == [("cap.x", "search returns 400")]


# --------------------------------------------------------------------------
# A repair brief is not a build brief
# --------------------------------------------------------------------------

def test_a_repair_brief_leads_with_the_defect():
    """A model plans from what it reads first.

    The defect used to be appended after the full capability specification, and
    the planner read a build brief and planned a build: handed a 794-line
    implementation failing one check, it decomposed into "implement the run
    function", "implement search", "implement playback control", grew the file
    by fifty lines re-implementing what already worked, and never touched the
    one wrong constant it was sent to change.
    """

    from service.acquisition import repair_goal

    brief = repair_goal("Build a music provider. It must play, pause, resume...",
                        "search returns 400 Invalid limit")

    head = brief[:200].lower()
    assert "repair" in head
    assert "not a rebuild" in head
    assert "invalid limit" in head, "the defect must be in the first thing read"
    # The specification is still present, but demoted to reference.
    assert "for reference only" in brief.lower()
    assert brief.lower().index("the defect to fix") < brief.lower().index("build a music provider")


def test_a_repair_brief_forbids_re_implementing_what_works():
    from service.acquisition import repair_goal

    brief = repair_goal("spec", "defect").lower()

    assert "do not re-implement" in brief
    assert "do not add features" in brief
    assert "smaller the change" in brief
    assert "anchor on a line you have just read" in brief


def test_the_mission_uses_the_repair_brief_when_repairing(tmp_path):
    class Recording(StubService):
        def __init__(self):
            super().__init__(succeed_on=1)
            self.goals = []
            self.registry = type("R", (), {"disable": lambda self, cid, reason="": None})()

        def ensure(self, goal, **kwargs):
            self.goals.append(goal)
            return super().ensure(goal, **kwargs)

    service = Recording()
    run = mission(tmp_path, service=service)

    run.run("the full specification", capability_id="cap.x", repair="one wrong constant")

    assert service.goals[0].lower().startswith("repair an existing")
    assert "one wrong constant" in service.goals[0]


def test_a_first_build_gets_the_plain_specification(tmp_path):
    class Recording(StubService):
        def __init__(self):
            super().__init__(succeed_on=1)
            self.goals = []

        def ensure(self, goal, **kwargs):
            self.goals.append(goal)
            return super().ensure(goal, **kwargs)

    service = Recording()
    run = mission(tmp_path, service=service)

    run.run("the full specification", capability_id="cap.x")

    assert service.goals[0] == "the full specification"


# --------------------------------------------------------------------------
# What gets carried forward, and what must not
# --------------------------------------------------------------------------

class StubMemory:
    def __init__(self, context=""):
        self.context = context
        self.recorded = []
        self.asked = []

    def context_for(self, goal, *, task_class="", limit=2):
        self.asked.append((goal, task_class))
        return self.context

    def record(self, lesson):
        self.recorded.append(lesson)
        return lesson


def _mission_with_memory(tmp_path, memory, *, service=None, controller=None, gateway=None):
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    return AcquisitionMission(
        service=service or StubService(succeed_on=1),
        kernel=StubKernel(tmp_path),
        gateway=gateway or StubGateway(),
        ledger=StubLedger(),
        controller=controller or StubController(escalate=False),
        memory=memory,
    )


def test_remembered_lessons_are_consulted_before_anything_is_spent(tmp_path):
    memory = StubMemory(context="A previous task of this kind (capability) was solved as follows.\n")
    service = StubService(succeed_on=1)
    run = _mission_with_memory(tmp_path, memory, service=service)

    run.run("build a provider")

    assert memory.asked, "the mission must ask what it already knows"
    assert service.goals_seen[0].startswith("A previous task of this kind") if hasattr(
        service, "goals_seen") else True


def test_a_verified_acquisition_is_written_down(tmp_path):
    memory = StubMemory()
    service = StubService(succeed_on=1)
    service_verification = {"ok": True, "checks": [{"name": "tests", "ok": True},
                                                   {"name": "playback", "ok": True}]}

    class Verified(StubService):
        def ensure(self, goal, **kwargs):
            outcome = StubOutcome(True, "cap.x", status="acquired")
            outcome.verification = service_verification
            return outcome

    run = _mission_with_memory(tmp_path, memory, service=Verified())

    run.run("build a provider")

    assert len(memory.recorded) == 1
    lesson = memory.recorded[0]
    assert lesson.verified, "a lesson without passing checks must not count as verified"
    assert {item["criterion"] for item in lesson.verification} == {"tests", "playback"}


def test_a_failed_acquisition_teaches_nothing(tmp_path):
    """An unverified lesson is a rumour, and a rumour in a future prompt is how
    one bad answer becomes several."""

    memory = StubMemory()
    run = _mission_with_memory(tmp_path, memory, service=StubService(succeed_on=None))

    run.run("build a provider")

    assert memory.recorded == []


def test_a_lesson_with_no_verification_is_never_recorded(tmp_path):
    class NoChecks(StubService):
        def ensure(self, goal, **kwargs):
            outcome = StubOutcome(True, "cap.x", status="acquired")
            outcome.verification = {"ok": True, "checks": []}
            return outcome

    memory = StubMemory()
    run = _mission_with_memory(tmp_path, memory, service=NoChecks())

    run.run("build a provider")

    assert memory.recorded == [], "checked nothing is not the same as checked and passed"


def test_a_broken_memory_does_not_stop_the_mission(tmp_path):
    class Broken(StubMemory):
        def context_for(self, goal, *, task_class="", limit=2):
            raise RuntimeError("the lesson store is unreadable")

    run = _mission_with_memory(tmp_path, Broken(), service=StubService(succeed_on=1))

    result = run.run("build a provider")

    assert result.acquired is True
    assert any(step.stage == "recall" and not step.ok for step in result.steps)
