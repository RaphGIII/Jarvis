"""What happens when something never comes back.

Written after an evidence run went quiet and could not be distinguished, from
the outside, from a wedged one. It had in fact finished -- but the investigation
showed the system had no way to tell those apart and no bound that would have
stopped a genuine hang:

* the project loop compared elapsed time against its budget only *between*
  steps, so a step that blocked forever never reached the check again;
* the repository engineer had no wall-clock bound at all;
* ``urlopen(timeout=...)`` bounds each read, not the request, and a
  non-streaming local model sends nothing until it is finished -- so "still
  thinking" and "never coming back" are identical on the socket.

Every test here uses a model that genuinely blocks, so the bounds are exercised
rather than described.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import pytest

from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal
from projects.engine import ProjectEngine
from projects.models import Phase, ResourceLimits, StopReason
from projects.store import ProjectStore
from runtime.deadline import CallTimeout, Deadline, DeadlineExceeded, call_with_timeout
from runtime.heartbeat import Heartbeat, check_liveness, read_heartbeat
from tools.builtin import builtin_tools
from tools.registry import RiskLevel, ToolPolicy, ToolRegistry


class WedgedBrain:
    """A model that accepts a request and never answers.

    The failure mode that matters: not an error, not a refusal, just silence.
    """

    def __init__(self, *, block_seconds: float = 30.0):
        self.block_seconds = block_seconds
        self.calls = 0
        self.released = threading.Event()

    def generate_structured(self, prompt, schema, **kwargs):
        self.calls += 1
        self.released.wait(self.block_seconds)
        return json.dumps({"tool_calls": []})


class SlowThenFineBrain:
    """Wedges once, then behaves. Proves a timeout is survivable, not terminal."""

    def __init__(self, *, block_seconds: float = 30.0, payload=None):
        self.block_seconds = block_seconds
        self.payload = payload or {"tool_calls": []}
        self.calls = 0

    def generate_structured(self, prompt, schema, **kwargs):
        self.calls += 1
        if self.calls == 1:
            time.sleep(self.block_seconds)
        properties = schema.get("properties") or {}
        if "diagnosis" in properties:
            return json.dumps({"diagnosis": "the model timed out", "fix": "retry"})
        if "tasks" in properties:
            return json.dumps({"tasks": [{"title": "do it"}], "acceptance": []})
        if "findings" in properties:
            return json.dumps({"tool_calls": [{"name": "list_files", "arguments": {}}]})
        return json.dumps(self.payload)


@pytest.fixture
def tools():
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.MODERATE))
    registry.register_many(builtin_tools())
    return registry


# ====================================================================== deadline


def test_a_deadline_reports_what_is_left():
    deadline = Deadline.of(10.0, name="x")
    assert 9.0 < deadline.remaining <= 10.0
    assert not deadline.expired
    deadline.require()  # does not raise


def test_an_expired_deadline_refuses_new_work():
    deadline = Deadline(budget=0.01, name="x")
    time.sleep(0.05)
    assert deadline.expired
    with pytest.raises(DeadlineExceeded):
        deadline.require()


def test_a_deadline_clamps_a_call_timeout_to_what_remains():
    """A 900s provider timeout is absurd when the mission has 40s left."""

    deadline = Deadline.of(5.0)
    assert deadline.clamp(900.0) <= 5.0
    assert deadline.clamp(1.0) == 1.0


def test_no_budget_means_no_clamping():
    assert Deadline.none().clamp(900.0) == 900.0
    assert not Deadline.none().expired


def test_a_child_budget_can_be_shorter_but_never_longer():
    parent = Deadline.of(10.0)
    assert parent.child(3.0).budget == 3.0
    assert parent.child(1000.0).budget <= 10.0


# ================================================================ call_with_timeout


def test_a_blocking_call_is_abandoned_rather_than_waited_out():
    started = time.monotonic()
    with pytest.raises(CallTimeout):
        call_with_timeout(lambda: time.sleep(30), timeout=0.5, what="wedged call")
    assert time.monotonic() - started < 5.0, "the caller must not wait for the call it abandoned"


def test_a_fast_call_returns_its_value():
    assert call_with_timeout(lambda: 42, timeout=5.0) == 42


def test_an_exception_still_propagates_to_the_caller():
    def explode():
        raise ValueError("from the worker thread")

    with pytest.raises(ValueError, match="from the worker thread"):
        call_with_timeout(explode, timeout=5.0)


def test_an_unbounded_timeout_runs_inline_without_a_thread():
    names = []
    call_with_timeout(lambda: names.append(threading.current_thread().name), timeout=float("inf"))
    assert names == [threading.current_thread().name]


def test_abandonment_is_reported_so_it_can_be_recorded():
    seen = []
    with pytest.raises(CallTimeout):
        call_with_timeout(lambda: time.sleep(30), timeout=0.3, on_abandon=lambda: seen.append(True))
    assert seen == [True]


# ===================================================================== heartbeat


def test_a_heartbeat_records_liveness_and_progress_separately(tmp_path):
    beat = Heartbeat(tmp_path / "hb.json", run="test")
    beat.beat("investigating", "looking around")
    first = read_heartbeat(tmp_path / "hb.json")

    time.sleep(0.01)
    beat.beat("investigating", "still looking")  # alive, but no progress
    second = read_heartbeat(tmp_path / "hb.json")

    assert second["updated_at"] > first["updated_at"], "liveness must advance"
    assert second["progress_at"] == first["progress_at"], "progress must not"

    beat.beat("executing", "did something", progress=True)
    third = read_heartbeat(tmp_path / "hb.json")
    assert third["progress_at"] > first["progress_at"]
    assert third["steps"] == 1


def test_the_ticker_keeps_the_file_fresh_during_a_long_call(tmp_path):
    """The whole point: a slow step must not look like a dead process."""

    path = tmp_path / "hb.json"
    with Heartbeat(path, run="test", interval=1.0):
        before = read_heartbeat(path)["updated_at"]
        time.sleep(2.5)  # a "long model call" during which nothing else happens
        after = read_heartbeat(path)["updated_at"]

    assert after > before, "a live run must keep breathing while it waits on the model"


def test_a_live_run_is_classified_alive(tmp_path):
    path = tmp_path / "hb.json"
    beat = Heartbeat(path, run="test")
    beat.beat("executing", "working", progress=True)
    assert check_liveness(path).state == "alive"


def test_a_finished_run_is_not_reported_as_stalled(tmp_path):
    """The exact misreading that prompted this work."""

    path = tmp_path / "hb.json"
    beat = Heartbeat(path, run="test")
    beat.beat("executing", "working", progress=True)
    beat.finish("did not pass")

    liveness = check_liveness(path, dead_after=0.0, stalled_after=0.0)
    assert liveness.state == "finished"
    assert not liveness.needs_attention
    assert "did not pass" in liveness.detail


def test_a_process_that_vanished_is_classified_dead(tmp_path):
    path = tmp_path / "hb.json"
    beat = Heartbeat(path, run="test")
    beat.beat("executing", "working", progress=True)

    payload = read_heartbeat(path)
    payload["pid"] = 999999  # a pid that cannot exist
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert check_liveness(path).state == "dead"


def test_alive_but_not_progressing_is_classified_stalled(tmp_path):
    path = tmp_path / "hb.json"
    beat = Heartbeat(path, run="test")
    beat.beat("executing", "working", progress=True)
    time.sleep(0.05)
    beat.beat("executing", "still working")  # liveness only

    liveness = check_liveness(path, dead_after=600.0, stalled_after=0.01)
    assert liveness.state == "stalled"
    assert liveness.needs_attention


def test_a_missing_heartbeat_is_unknown_not_a_verdict(tmp_path):
    assert check_liveness(tmp_path / "nothing.json").state == "unknown"


def test_a_heartbeat_that_cannot_be_written_does_not_break_the_run(tmp_path):
    """Reporting on a run must never be able to kill it."""

    beat = Heartbeat(tmp_path / "hb.json", run="test")
    beat.path = tmp_path / "does" / "not" / "exist" / "hb.json"
    beat.beat("executing", "working")  # must not raise


# ================================================================= project engine


def test_a_wedged_model_does_not_hold_the_project_loop_open(tmp_path, tools):
    """The headline guarantee: a silent model costs a step, not the mission."""

    store = ProjectStore(tmp_path / "projects")
    brain = WedgedBrain(block_seconds=30.0)
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project(
        "something that will never be answered",
        limits=ResourceLimits(max_steps=3, max_seconds=20, step_timeout_seconds=1.0),
    )

    started = time.monotonic()
    result = engine.run(project)
    elapsed = time.monotonic() - started

    assert elapsed < 20.0, f"the loop waited {elapsed:.0f}s on a model that never answered"
    assert result.stop_reason in {StopReason.STEP_LIMIT, StopReason.FAILURE_LIMIT, StopReason.TIME_LIMIT}
    brain.released.set()


def test_the_mission_budget_bounds_a_single_call(tmp_path, tools):
    """A call may never be given more time than the mission has left."""

    store = ProjectStore(tmp_path / "projects")
    engine = ProjectEngine(brain=WedgedBrain(), store=store, tools=tools)
    engine.deadline = Deadline.of(3.0)
    engine.step_timeout_seconds = 900.0

    assert engine.model_call_timeout() <= 3.0


def test_a_timed_out_call_is_recorded_as_the_reason_the_step_failed(tmp_path, tools):
    """DIAGNOSE cannot act on "nothing happened"; it needs to know why."""

    store = ProjectStore(tmp_path / "projects")
    brain = WedgedBrain(block_seconds=30.0)
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project(
        "goes nowhere", limits=ResourceLimits(max_steps=4, max_seconds=20, step_timeout_seconds=1.0)
    )
    project.add_acceptance("ok", check=[sys.executable, "-c", "raise SystemExit(1)"])
    project.add_task("do the thing")
    project.add_finding("seeded so the loop reaches EXECUTE")
    store.save(project)

    engine.run(project)

    execute_steps = [step for step in project.steps if step.phase is Phase.EXECUTE]
    assert execute_steps, "the loop must have attempted the task"
    assert any("did not return within" in step.summary for step in execute_steps), [
        step.summary for step in execute_steps
    ]
    brain.released.set()


def test_a_timeout_is_survivable_rather_than_terminal(tmp_path, tools):
    """One slow call must not end a run that would otherwise succeed."""

    store = ProjectStore(tmp_path / "projects")
    brain = SlowThenFineBrain(
        block_seconds=3.0,
        payload={"tool_calls": [{"name": "write_file", "arguments": {"path": "made.txt", "content": "ok\n"}}]},
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project(
        "make a file",
        limits=ResourceLimits(max_steps=14, max_seconds=120, step_timeout_seconds=1.0, max_consecutive_failures=8),
    )
    project.add_acceptance("the file exists", check=[sys.executable, "-c", "open('made.txt')"])
    store.save(project)

    result = engine.run(project)

    assert brain.calls > 1, "the run must have continued past the timed-out call"
    assert result.accepted, result.message


def test_a_run_writes_a_heartbeat_a_supervisor_can_read(tmp_path, tools):
    store = ProjectStore(tmp_path / "projects")
    path = tmp_path / "hb.json"
    brain = SlowThenFineBrain(block_seconds=0.0, payload={"tool_calls": []})
    engine = ProjectEngine(brain=brain, store=store, tools=tools, heartbeat=Heartbeat(path, run="test"))
    project = engine.create_project("anything", limits=ResourceLimits(max_steps=2, max_seconds=30))

    engine.run(project)

    payload = read_heartbeat(path)
    assert payload is not None
    assert payload["steps"] >= 1
    assert payload["budget_seconds"] == 30.0


# =============================================================== self-development


class WedgedEngineerBrain:
    def __init__(self, *, block_seconds: float = 30.0):
        self.block_seconds = block_seconds
        self.calls = 0

    def generate_structured(self, prompt, schema, **kwargs):
        self.calls += 1
        time.sleep(self.block_seconds)
        return json.dumps({})


def _repo(root):
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    for args in (["init"], ["config", "user.email", "j@example.invalid"], ["config", "user.name", "J"], ["add", "."]):
        subprocess.run(["git", *args], cwd=str(root), capture_output=True, timeout=60)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", "b"], cwd=str(root), capture_output=True, timeout=60)
    return root


def test_self_development_cannot_run_forever(tmp_path):
    """It previously had no wall-clock bound of any kind."""

    repo = _repo(tmp_path / "repo")
    engineer = RepositoryEngineer(
        brain=WedgedEngineerBrain(block_seconds=30.0),
        worktree_root=tmp_path / "wt",
        max_cycles=3,
        max_seconds=8.0,
        model_call_timeout_seconds=1.0,
    )

    started = time.monotonic()
    result = engineer.improve(repo, SelfImprovementGoal(objective="anything", allowed_paths=["calc.py"]))
    elapsed = time.monotonic() - started

    assert elapsed < 60.0, f"self-development ran for {elapsed:.0f}s against an 8s budget"
    assert not result.success


def test_a_timed_out_run_is_paused_and_resumable(tmp_path):
    """Out of time is a different outcome from out of ideas, and recoverable."""

    repo = _repo(tmp_path / "repo")
    engineer = RepositoryEngineer(
        brain=WedgedEngineerBrain(block_seconds=10.0),
        worktree_root=tmp_path / "wt",
        max_cycles=4,
        max_seconds=3.0,
        model_call_timeout_seconds=0.5,
        resume_command="python -m jarvis.self_develop --resume <dir>",
    )

    result = engineer.improve(repo, SelfImprovementGoal(objective="anything", allowed_paths=["calc.py"]))

    assert result.failure_kind == "time_limit"
    assert result.status.endswith("PAUSED")
    assert result.resume_command
    assert (tmp_path / "wt").exists(), "the worktree must survive for inspection and resumption"


def test_a_model_timeout_is_diagnosed_rather_than_propagated(tmp_path):
    """A timed-out generation is evidence for the next cycle, not an exception."""

    repo = _repo(tmp_path / "repo")
    brain = WedgedEngineerBrain(block_seconds=5.0)
    engineer = RepositoryEngineer(
        brain=brain,
        worktree_root=tmp_path / "wt",
        max_cycles=2,
        max_seconds=60.0,
        model_call_timeout_seconds=0.5,
    )

    result = engineer.improve(repo, SelfImprovementGoal(objective="anything", allowed_paths=["calc.py"]))

    assert not result.success
    assert result.failure_kind != "time_limit", "the run had time; only individual calls timed out"
    assert brain.calls >= 2, "it must have tried again after the first call timed out"


# ==================================================================== the doctor


def test_the_doctor_finds_and_classifies_a_run(tmp_path):
    """The tool that would have answered the question in one command."""

    from jarvis.doctor import find_heartbeats, render

    run = tmp_path / "run"
    beat = Heartbeat(run / "heartbeat.json", run="evidence")
    beat.beat("scenario_A", "working", progress=True)

    found = find_heartbeats([tmp_path])
    assert found == [run / "heartbeat.json"]

    liveness = check_liveness(found[0])
    assert liveness.state == "alive"
    assert "scenario_A" in render(found[0], liveness)


def test_the_doctor_reports_a_finished_run_as_finished(tmp_path):
    """The exact case that was misread as a hang."""

    from jarvis.doctor import find_heartbeats

    run = tmp_path / "run"
    beat = Heartbeat(run / "heartbeat.json", run="evidence")
    beat.beat("scenario_F", "working", progress=True)
    beat.finish("did not pass")

    liveness = check_liveness(find_heartbeats([tmp_path])[0], dead_after=0.0, stalled_after=0.0)
    assert liveness.state == "finished"
    assert not liveness.needs_attention


def test_the_doctor_prefers_the_newest_run(tmp_path):
    from jarvis.doctor import find_heartbeats

    Heartbeat(tmp_path / "old" / "heartbeat.json", run="old").beat("x", "x")
    time.sleep(0.05)
    Heartbeat(tmp_path / "new" / "heartbeat.json", run="new").beat("x", "x")

    assert find_heartbeats([tmp_path])[0].parent.name == "new"


# ======================================================== evidence wall clock


def test_an_evidence_run_has_a_hard_total_budget():
    """Previously the only bound was a shell `timeout` wrapper.

    That is not part of the program: run the command any other way and the
    bound simply is not there.
    """

    from jarvis.record_evidence import build_parser

    defaults = build_parser().parse_args([])
    assert defaults.total_seconds > 0
    assert defaults.scenario_seconds > 0
    assert defaults.scenario_seconds <= defaults.total_seconds


def test_a_scenario_budget_can_never_exceed_the_total():
    deadline = Deadline.of(10.0, name="evidence run")
    assert deadline.child(1800.0, name="scenario").budget <= 10.0


def test_capability_acquisition_accepts_a_time_budget(tmp_path, tools):
    """Acquisition is a long job, but never an unbounded one."""

    from capabilities.registry import CapabilityRegistry
    from capabilities.service import CapabilityService
    from knowledge.graph import KnowledgeGraph

    store = ProjectStore(tmp_path / "projects")
    service = CapabilityService(
        registry=CapabilityRegistry(tmp_path / "registry.json"),
        engine=ProjectEngine(brain=WedgedBrain(block_seconds=30.0), store=store, tools=tools),
        graph=KnowledgeGraph(tmp_path / "palace.sqlite"),
        root=tmp_path / "installed",
    )

    started = time.monotonic()
    outcome = service.ensure("something unanswerable", max_steps=5, max_seconds=6.0)
    elapsed = time.monotonic() - started

    assert not outcome.acquired
    assert elapsed < 60.0, f"acquisition ran {elapsed:.0f}s against a 6s budget"
