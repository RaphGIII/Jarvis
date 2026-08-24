from __future__ import annotations

import json
import sys

import pytest

from projects.engine import EngineHooks, ProjectEngine
from projects.models import Phase, Project, ProjectState, ResourceLimits, StopReason, TaskStatus
from projects.store import ProjectStore
from tools.builtin import builtin_tools
from tools.registry import RiskLevel, ToolPolicy, ToolRegistry


class ScriptedBrain:
    """Answers by prompt shape, so a test can script an entire run.

    Keyed on the schema rather than on prompt text, because that is what the
    engine actually varies between phases.
    """

    def __init__(self, *, investigate=None, decompose=None, execute=None, diagnose=None):
        self.investigate = investigate or {"tool_calls": [{"name": "list_files", "arguments": {}}]}
        self.decompose = decompose or {"tasks": [], "acceptance": []}
        self.execute = execute if execute is not None else []
        self.diagnose = diagnose or {"diagnosis": "the implementation is wrong", "fix": "correct it"}
        self.calls = {"investigate": 0, "decompose": 0, "execute": 0, "diagnose": 0}
        self.prompts: list[str] = []

    def generate_structured(self, prompt, schema, *, max_tokens=1000, temperature=0.1, top_p=0.9):
        self.prompts.append(prompt)
        properties = schema.get("properties") or {}
        if "diagnosis" in properties:
            self.calls["diagnose"] += 1
            return json.dumps(self.diagnose)
        if "tasks" in properties:
            self.calls["decompose"] += 1
            return json.dumps(self.decompose)
        if "findings" in properties:
            self.calls["investigate"] += 1
            return json.dumps(self.investigate)
        index = self.calls["execute"]
        self.calls["execute"] += 1
        if isinstance(self.execute, list):
            payload = self.execute[min(index, len(self.execute) - 1)] if self.execute else {"tool_calls": []}
        else:
            payload = self.execute
        return json.dumps(payload)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "projects")


@pytest.fixture
def tools(tmp_path):
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.MODERATE))
    registry.register_many(builtin_tools())
    return registry


def make_engine(store, tools, brain, **kwargs):
    return ProjectEngine(brain=brain, store=store, tools=tools, **kwargs)


def _writes(path, content):
    return {"tool_calls": [{"name": "write_file", "arguments": {"path": path, "content": content}}]}


# ------------------------------------------------------------------ persistence


def test_project_survives_a_restart(store, tools):
    """The whole point of a durable project: a new process picks it up."""

    engine = make_engine(store, tools, ScriptedBrain())
    project = engine.create_project("Build a thing", acceptance=[("it works", ["python", "-c", "pass"])])
    project.add_finding("the config lives in settings.py")
    project.add_decision("use sqlite", rationale="no server needed")
    project.add_task("write the module")
    store.save(project)

    reopened = ProjectStore(store.root).load(project.id)
    assert reopened.goal == "Build a thing"
    assert [item.text for item in reopened.findings] == ["the config lives in settings.py"]
    assert [item.text for item in reopened.decisions] == ["use sqlite"]
    assert [task.title for task in reopened.tasks] == ["write the module"]
    assert reopened.acceptance[0].check == ["python", "-c", "pass"]


def test_store_survives_a_corrupt_document(store, tools):
    engine = make_engine(store, tools, ScriptedBrain())
    good = engine.create_project("good project")
    (store.root / "broken.json").write_text("{not json", encoding="utf-8")
    assert [item.id for item in store.list_projects()] == [good.id]


def test_save_is_atomic_so_a_kill_cannot_truncate(store, tools):
    engine = make_engine(store, tools, ScriptedBrain())
    project = engine.create_project("atomic")
    for index in range(20):
        project.add_finding(f"finding {index}")
        store.save(project)
    assert len(json.loads(store.path_for(project.id).read_text(encoding="utf-8"))["findings"]) == 20
    assert not list(store.root.glob("*.tmp"))


def test_find_ranks_live_projects_above_finished_ones(store, tools):
    engine = make_engine(store, tools, ScriptedBrain())
    old = engine.create_project("chess board recognition overlay")
    old.state = ProjectState.COMPLETED
    store.save(old)
    live = engine.create_project("chess board recognition overlay")
    assert store.find("chess overlay")[0].id == live.id


# ------------------------------------------------------------------ acceptance


def test_completion_requires_a_runnable_check(store, tools):
    """A criterion nobody can run must never count as satisfied."""

    project = Project(goal="x")
    project.add_acceptance("it feels right")  # no check
    assert not project.acceptance_satisfied()

    project.add_acceptance("tests pass", check=["python", "-c", "pass"])
    assert not project.acceptance_satisfied()
    project.acceptance[1].satisfied = True
    assert project.acceptance_satisfied(), "only the objective criterion should be required"


def test_run_accepts_only_when_the_check_actually_passes(store, tools, tmp_path):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "create marker", "detail": "write marker.txt"}],
            "acceptance": [{"text": "marker exists", "check": [sys.executable, "-c", "open('marker.txt')"]}],
        },
        execute=[_writes("marker.txt", "present\n")],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("create a marker file")
    result = engine.run(project, max_steps=12)

    assert result.accepted
    assert result.stop_reason is StopReason.ACCEPTED
    assert project.state is ProjectState.COMPLETED
    assert (store.workspace_for(project) / "marker.txt").exists()


def test_a_project_whose_check_fails_is_never_accepted(store, tools):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "do nothing useful"}],
            "acceptance": [{"text": "impossible", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[_writes("something.txt", "x\n")],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("fail honestly", limits=ResourceLimits(max_steps=8, max_consecutive_failures=3))
    result = engine.run(project)

    assert not result.accepted
    assert project.state is not ProjectState.COMPLETED
    assert any(not item.satisfied for item in project.objective_criteria())


def test_decomposition_without_a_runnable_check_records_a_blocker(store, tools):
    brain = ScriptedBrain(decompose={"tasks": [{"title": "t"}], "acceptance": [{"text": "looks good"}]})
    engine = make_engine(store, tools, brain)
    project = engine.create_project("vague goal", limits=ResourceLimits(max_steps=4))
    engine.run(project)
    assert any("runnable check" in blocker.text for blocker in project.blockers)


# ------------------------------------------------------------------ recovery


def test_a_failed_tool_call_does_not_end_the_project(store, tools):
    """Mission requirement B, at the project level."""

    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "edit the file"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "pass"]}],
        },
        execute=[
            # A search anchor that cannot match: the classic weak-model failure.
            {"tool_calls": [{"name": "apply_edits", "arguments": {"files": [{"path": "nope.py", "search": "absent", "replace": "x"}]}}]},
            _writes("recovered.txt", "ok\n"),
        ],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("recover from a bad edit")
    result = engine.run(project, max_steps=14)

    assert brain.calls["execute"] >= 2, "the loop must retry after a failed tool call"
    assert result.accepted
    assert any(step.phase is Phase.DIAGNOSE for step in project.steps)


def test_the_loop_stops_after_too_many_consecutive_failures(store, tools):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "impossible"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[{"tool_calls": [{"name": "read_file", "arguments": {"path": "missing.py"}}]}],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("never succeeds", limits=ResourceLimits(max_steps=60, max_consecutive_failures=4))
    result = engine.run(project)

    assert result.stop_reason in {StopReason.FAILURE_LIMIT, StopReason.STEP_LIMIT}
    assert result.steps < 60, "the loop must give up before burning the whole step budget"
    assert project.state is ProjectState.PAUSED, "a budget stop must leave the project resumable"


def test_a_task_is_abandoned_after_its_attempts_run_out(store, tools):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "doomed"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[{"tool_calls": [{"name": "read_file", "arguments": {"path": "missing.py"}}]}],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("doomed work", limits=ResourceLimits(max_steps=25, max_consecutive_failures=25))
    engine.run(project)
    assert any(task.status is TaskStatus.ABANDONED for task in project.tasks)


def test_failed_experiments_are_fed_back_into_the_prompt(store, tools):
    """An agent that forgets its failures repeats them."""

    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "edit"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[{"tool_calls": [{"name": "apply_edits", "arguments": {"files": [{"path": "a.py", "search": "gone", "replace": "x"}]}}]}],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("learn from failure", limits=ResourceLimits(max_steps=12, max_consecutive_failures=12))
    engine.run(project)

    assert any(not item.succeeded for item in project.experiments)
    assert any("ALREADY TRIED AND FAILED" in prompt for prompt in brain.prompts)


def test_a_malformed_model_reply_costs_a_step_not_the_project(store, tools):
    class BrokenBrain(ScriptedBrain):
        def __init__(self):
            super().__init__()
            self.replies = 0

        def generate_structured(self, prompt, schema, *, max_tokens=1000, temperature=0.1, top_p=0.9):
            self.replies += 1
            if self.replies <= 3:
                return "I am not JSON at all."
            return super().generate_structured(prompt, schema, max_tokens=max_tokens)

    engine = make_engine(store, tools, BrokenBrain())
    project = engine.create_project("survive garbage", limits=ResourceLimits(max_steps=6))
    result = engine.run(project)
    assert result.stop_reason is not StopReason.ERROR
    assert project.steps, "the loop must keep stepping through malformed replies"


def test_a_provider_that_always_raises_does_not_lose_the_project(store, tools):
    class DeadBrain:
        def generate_structured(self, *args, **kwargs):
            raise ConnectionError("ollama is down")

    engine = make_engine(store, tools, DeadBrain())
    project = engine.create_project("provider is down", limits=ResourceLimits(max_steps=5))
    result = engine.run(project)
    assert not result.accepted
    assert store.load(project.id).id == project.id, "the project must still be on disk"


# ------------------------------------------------------------------ blocking


def test_a_genuine_external_blocker_stops_the_run(store, tools):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "call the paid API"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[{"tool_calls": [{"name": "read_file", "arguments": {"path": "missing.py"}}]}],
        diagnose={
            "diagnosis": "there is no API credential",
            "fix": "the user must supply one",
            "blocked_on_user": True,
            "blocker": "an API key is required and cannot be derived",
        },
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("needs a credential", limits=ResourceLimits(max_steps=20))
    result = engine.run(project)

    assert result.stop_reason is StopReason.BLOCKED
    assert project.state is ProjectState.BLOCKED
    assert project.user_blockers()


def test_a_failing_test_is_not_treated_as_an_external_blocker(store, tools):
    """The model must not be able to escape work by declaring itself blocked."""

    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "fix it"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[{"tool_calls": [{"name": "read_file", "arguments": {"path": "missing.py"}}]}],
        diagnose={"diagnosis": "a test fails", "fix": "correct the logic", "blocked_on_user": False},
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("keep going", limits=ResourceLimits(max_steps=8, max_consecutive_failures=8))
    result = engine.run(project)
    assert result.stop_reason is not StopReason.BLOCKED


# ------------------------------------------------------------------ control


def test_the_user_can_cancel_between_steps(store, tools):
    stop = {"now": False}

    def should_cancel():
        return stop["now"]

    brain = ScriptedBrain(
        decompose={"tasks": [{"title": "t"}], "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}]},
        execute=[_writes("x.txt", "x")],
    )

    def on_step(project, step):
        stop["now"] = True

    engine = make_engine(store, tools, brain, hooks=EngineHooks(on_step=on_step, should_cancel=should_cancel))
    project = engine.create_project("cancellable", limits=ResourceLimits(max_steps=50))
    result = engine.run(project)
    assert result.stop_reason is StopReason.CANCELLED
    assert result.steps == 1


def test_a_paused_project_resumes_where_it_left_off(store, tools):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "create marker"}],
            "acceptance": [{"text": "marker", "check": [sys.executable, "-c", "open('marker.txt')"]}],
        },
        execute=[_writes("marker.txt", "ok\n")],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("resumable")

    first = engine.run(project, max_steps=1)
    assert not first.accepted
    steps_after_first = project.steps_spent

    reopened = ProjectStore(store.root).load(project.id)
    assert reopened.steps_spent == steps_after_first
    second = ProjectEngine(brain=brain, store=store, tools=tools).run(reopened, max_steps=12)
    assert second.accepted
    assert reopened.steps_spent > steps_after_first, "spend accumulates across sessions"


def test_step_and_time_budgets_are_honoured(store, tools):
    brain = ScriptedBrain(
        decompose={"tasks": [{"title": "t"}], "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}]},
        execute=[_writes("x.txt", "x")],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("budgeted", limits=ResourceLimits(max_steps=3, max_consecutive_failures=99))
    result = engine.run(project)
    assert result.steps <= 3
    assert result.stop_reason is StopReason.STEP_LIMIT


# ------------------------------------------------------------------ evolution


def test_a_new_requirement_reopens_a_completed_project(store, tools):
    """How an incrementally specified system actually grows."""

    engine = make_engine(store, tools, ScriptedBrain())
    project = engine.create_project("v1")
    project.state = ProjectState.COMPLETED
    store.save(project)

    engine.add_requirement(project, "also show an overlay")
    assert project.state is ProjectState.PLANNING
    assert [item.text for item in project.requirements][-1] == "also show an overlay"
    assert ProjectStore(store.root).load(project.id).state is ProjectState.PLANNING


def test_requirements_accumulate_without_duplicates(store, tools):
    engine = make_engine(store, tools, ScriptedBrain())
    project = engine.create_project("capture the screen")
    engine.add_requirement(project, "recognise the board")
    engine.add_requirement(project, "Recognise the board")  # same thing, different case
    engine.add_requirement(project, "connect to stockfish")
    assert len(project.requirements) == 3  # goal + two distinct additions


def test_replan_drops_dependencies_that_can_never_be_met(store, tools):
    engine = make_engine(store, tools, ScriptedBrain())
    project = engine.create_project("deadlocked")
    project.add_acceptance("ok", check=[sys.executable, "-c", "pass"])
    task = project.add_task("blocked task")
    task.depends_on = ["task_does_not_exist"]
    project.add_finding("seeded so investigation is skipped")
    store.save(project)

    assert project.next_task() is None, "the task is deadlocked to begin with"
    engine.run(project, max_steps=2)
    assert project.next_task() is not None or project.state is ProjectState.COMPLETED


def test_workspace_is_created_under_the_store(store, tools):
    engine = make_engine(store, tools, ScriptedBrain())
    project = engine.create_project("workspace check")
    workspace = store.workspace_for(project)
    assert workspace.exists()
    assert workspace.is_relative_to(store.root)


def test_artifacts_are_recorded_when_files_are_written(store, tools):
    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "write it"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "open('out.txt')"]}],
        },
        execute=[_writes("out.txt", "hello\n")],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("artifact tracking")
    engine.run(project, max_steps=10)
    assert any(item.path == "out.txt" for item in project.artifacts)


# --------------------------------------- withdrawing a tool the model misuses

def test_apply_edits_is_withdrawn_after_the_anchor_keeps_missing(store, tools):
    """Telling a model to switch tools does not work; not offering one does.

    A live run showed the "use write_file instead" advice arriving in the error
    message and being ignored eight times running, each attempt failing the
    same way on a twenty-line file.
    """

    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "fix the module"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[
            {"tool_calls": [{"name": "apply_edits", "arguments": {"files": [{"path": "a.py", "search": "absent anchor", "replace": "x"}]}}]},
        ],
    )
    engine = make_engine(store, tools, brain)
    project = engine.create_project("fix it", limits=ResourceLimits(max_steps=8, max_consecutive_failures=8))
    (store.workspace_for(project) / "a.py").write_bytes(b"value = 1\n")

    engine.run(project)

    later = [prompt for prompt in brain.prompts if "apply_edits is not available" in prompt]
    assert later, "after an anchor miss the model must be told the tool is gone"
    assert "apply_edits(" not in later[0].split("Available tools:")[1]
    assert "write_file(" in later[0].split("Available tools:")[1]


def test_apply_edits_is_offered_on_a_first_attempt(store, tools):
    """The withdrawal is a response to failure, not a permanent restriction."""

    brain = ScriptedBrain(
        decompose={
            "tasks": [{"title": "edit"}],
            "acceptance": [{"text": "ok", "check": [sys.executable, "-c", "raise SystemExit(1)"]}],
        },
        execute=[_writes("a.py", "value = 1\n")],
    )
    engine = make_engine(store, tools, brain)
    engine.run(engine.create_project("edit", limits=ResourceLimits(max_steps=3)))

    execute_prompts = [prompt for prompt in brain.prompts if "Available tools:" in prompt and "TASK:" in prompt]
    assert execute_prompts
    assert "apply_edits(" in execute_prompts[0].split("Available tools:")[1]
