from __future__ import annotations

import json
import sys

import pytest

from capabilities.registry import CapabilityRegistry
from capabilities.service import CapabilityService
from knowledge.graph import KnowledgeGraph
from projects.engine import ProjectEngine
from projects.store import ProjectStore
from tools.builtin import builtin_tools
from tools.registry import RiskLevel, ToolPolicy, ToolRegistry


class ScriptedBrain:
    """Plays a fixed sequence of tool calls, so acquisition is deterministic."""

    def __init__(self, *, implementation: str, tests: str, investigate_tool: str = "which"):
        self.implementation = implementation
        self.tests = tests
        self.investigate_tool = investigate_tool
        self.executes = 0
        self.prompts: list[str] = []

    def generate_structured(self, prompt, schema, *, max_tokens=1000, temperature=0.1, top_p=0.9):
        self.prompts.append(prompt)
        properties = schema.get("properties") or {}
        if "diagnosis" in properties:
            return json.dumps({"diagnosis": "the implementation was incomplete", "fix": "finish it"})
        if "tasks" in properties:
            return json.dumps(
                {
                    "tasks": [{"title": "implement main.run"}, {"title": "write the tests"}],
                    "acceptance": [],
                }
            )
        if "findings" in properties:
            return json.dumps({"tool_calls": [{"name": self.investigate_tool, "arguments": {"name": "python"}}]})

        self.executes += 1
        # Chosen by what the prompt asks for, not by a call counter. A counter
        # meant that if the first write was refused, main.py was never offered
        # again -- which is not how a real model behaves, and hid a genuine bug
        # behind a test-double artefact.
        title = prompt.split("TASK:", 1)[-1].splitlines()[0] if "TASK:" in prompt else ""
        wants_tests = "test" in title.lower()
        if wants_tests:
            return json.dumps(
                {"tool_calls": [{"name": "write_file", "arguments": {"path": "test_capability.py", "content": self.tests}}]}
            )
        return json.dumps(
            {"tool_calls": [{"name": "write_file", "arguments": {"path": "main.py", "content": self.implementation}}]}
        )


GOOD_IMPLEMENTATION = '''from __future__ import annotations

import shutil
from typing import Any


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Pretend to play an audio file; report what it would do when dry-running."""

    dry_run = bool(payload.get("dry_run"))
    path = payload.get("path")
    player = shutil.which("python")

    if not path:
        return {"ok": False, "error": "no path was given", "dry_run": dry_run}
    if player is None:
        return {"ok": False, "error": "no player is installed", "dry_run": dry_run}
    if dry_run:
        return {"ok": True, "dry_run": True, "player": player, "would_run": [player, str(path)]}
    return {"ok": True, "dry_run": False, "player": player, "played": str(path)}
'''

GOOD_TESTS = '''import main


def test_dry_run_reports_the_player_without_playing():
    result = main.run({"path": "song.wav", "dry_run": True})
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["player"]
    assert "played" not in result


def test_missing_path_fails_cleanly_without_raising():
    result = main.run({"dry_run": True})
    assert result["ok"] is False
    assert "error" in result
'''

TRIVIAL_TESTS = '''import main


def test_it_returns_something():
    assert main.run({"dry_run": True}) is not None
'''

BROKEN_IMPLEMENTATION = '''def run(payload):
    raise RuntimeError("this capability does not work")
'''


@pytest.fixture
def service(tmp_path):
    def build(brain):
        store = ProjectStore(tmp_path / "projects")
        tools = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.MODERATE))
        tools.register_many(builtin_tools())
        engine = ProjectEngine(brain=brain, store=store, tools=tools)
        graph = KnowledgeGraph(tmp_path / "palace.sqlite")
        registry = CapabilityRegistry(tmp_path / "registry.json")
        return CapabilityService(
            registry=registry, engine=engine, graph=graph, root=tmp_path / "installed", execution_timeout=60
        )

    return build


# ------------------------------------------------------------------ acquiring


def test_a_missing_capability_is_acquired_verified_and_registered(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome = instance.ensure("play an audio file through the speakers", max_steps=25)

    assert outcome.acquired, outcome.reason
    assert outcome.status == "acquired"
    assert instance.has(outcome.capability_id)
    assert outcome.verification["ok"]


def test_an_unusable_implementation_is_never_registered(service):
    """The gate that stops "it compiled" from counting as "it works"."""

    instance = service(ScriptedBrain(implementation=BROKEN_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome = instance.ensure("do something impossible", max_steps=12)

    assert not outcome.acquired
    assert outcome.status == "failed"
    assert instance.list() == []


def test_trivial_tests_do_not_count_as_verification(service):
    """A test that asserts nothing about behaviour proves nothing."""

    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=TRIVIAL_TESTS))
    outcome = instance.ensure("play an audio file", max_steps=25)

    assert not outcome.acquired
    substantive = next(
        item for item in outcome.verification["checks"] if item["name"] == "tests_are_substantive"
    )
    assert not substantive["ok"]


def test_acquisition_briefs_the_model_on_the_traps_that_actually_bit(service):
    """Each of these constraints exists because a live run failed without it."""

    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    brain = instance.engine.brain
    instance.ensure("play an audio file", max_steps=25)
    briefing = "\n".join(brain.prompts)

    assert "actually installed" in briefing
    # A hand-rolled PATH walk missed powershell.exe, so the capability
    # concluded the machine had no player at all.
    assert "shutil.which" in briefing
    # Tests hard-coding "mpg123" made a correct implementation unpassable.
    assert "NOT assert that any particular external program is installed" in briefing
    # payload['dry_run'] raised KeyError instead of failing cleanly.
    assert ".get()" in briefing


def test_the_workspace_is_seeded_with_a_working_skeleton(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    instance.ensure("play an audio file", max_steps=25)
    project = instance.engine.store.list_projects()[0]
    assert "run(payload" in (instance.engine.store.workspace_for(project) / "main.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ reuse


def test_an_existing_capability_is_reused_without_rebuilding(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    first = instance.ensure("play an audio file through the speakers", max_steps=25)
    assert first.acquired

    brain = instance.engine.brain
    calls_before = brain.executes
    second = instance.ensure("play an audio file through the speakers")

    assert second.status == "available"
    assert not second.acquired
    assert second.capability_id == first.capability_id
    assert brain.executes == calls_before, "reuse must not rebuild anything"


def test_a_capability_is_found_by_the_words_a_user_would_use(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    acquired = instance.ensure(
        "play an audio file through the speakers", max_steps=25, keywords=["music", "song", "sound"]
    )
    assert acquired.acquired

    found = instance.resolve("play some music")
    assert found is not None and found.capability_id == acquired.capability_id


# ------------------------------------------------------------------ execution


def test_an_acquired_capability_can_actually_be_run(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome = instance.ensure("play an audio file", max_steps=25)

    execution = instance.execute(outcome.capability_id, {"path": "song.wav", "dry_run": True})

    assert execution.ok, execution.error
    assert execution.output["player"]
    assert execution.output["would_run"][1] == "song.wav"


def test_execution_reports_a_clean_failure_as_data(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome = instance.ensure("play an audio file", max_steps=25)

    execution = instance.execute(outcome.capability_id, {"dry_run": True})  # no path

    assert not execution.ok
    assert "no path" in execution.error


def test_a_capability_that_raises_cannot_take_jarvis_down(service, tmp_path):
    """Model-authored code runs in a subprocess for exactly this reason."""

    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome = instance.ensure("play an audio file", max_steps=25)

    installed = tmp_path / "installed"
    main = next(installed.rglob("main.py"))
    main.write_text("def run(payload):\n    raise RuntimeError('boom')\n", encoding="utf-8")

    execution = instance.execute(outcome.capability_id, {"dry_run": True})
    assert not execution.ok
    assert "boom" in execution.error


def test_executing_an_unknown_capability_is_an_error_not_a_crash(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    execution = instance.execute("local.does.not.exist")
    assert not execution.ok
    assert "no active capability" in execution.error


def test_use_acquires_then_runs_in_one_call(service):
    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome, execution = instance.use("play an audio file", {"path": "a.wav", "dry_run": True}, max_steps=25)
    assert outcome.usable
    assert execution is not None and execution.ok


# ------------------------------------------------------------------ restart


def test_a_capability_survives_a_restart_and_still_runs(service, tmp_path):
    """Mission requirement F: usable after restart, with nothing rebuilt."""

    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    acquired = instance.ensure("play an audio file through the speakers", max_steps=25)
    assert acquired.acquired

    class ExplodingBrain:
        def generate_structured(self, *args, **kwargs):
            raise AssertionError("a restarted Jarvis must not need the model to reuse a capability")

    fresh = CapabilityService(
        registry=CapabilityRegistry(tmp_path / "registry.json"),
        engine=ProjectEngine(
            brain=ExplodingBrain(), store=ProjectStore(tmp_path / "projects"), tools=ToolRegistry()
        ),
        graph=KnowledgeGraph(tmp_path / "palace.sqlite"),
        root=tmp_path / "installed",
        execution_timeout=60,
    )

    resolved = fresh.resolve("play an audio file through the speakers")
    assert resolved is not None and resolved.capability_id == acquired.capability_id

    execution = fresh.execute(resolved.capability_id, {"path": "after_restart.wav", "dry_run": True})
    assert execution.ok, execution.error
    assert execution.output["would_run"][1] == "after_restart.wav"


# ------------------------------------------------------------------ naming


@pytest.mark.parametrize(
    "goal, expected_prefix",
    [
        ("play an audio file through the speakers", "local.play.audio"),
        ("Build a reusable capability that can resize images", "local.resize.images"),
    ],
)
def test_identifiers_are_readable_and_derived_from_the_goal(goal, expected_prefix):
    assert CapabilityService.suggest_id(goal).startswith(expected_prefix)


def test_identifier_generation_never_produces_an_empty_name():
    assert CapabilityService.suggest_id("the and for").startswith("local.")


# ------------------------------------------------------------------ discovery

SCHEMA_IMPLEMENTATION = '''from __future__ import annotations

from typing import Any

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "audio_path": {"type": "string", "description": "path to the file"},
        "dry_run": {"type": "boolean"},
    },
    "required": ["audio_path"],
}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    audio_path = payload.get("audio_path")
    if not audio_path:
        return {"ok": False, "error": "audio_path was not provided"}
    return {"ok": True, "dry_run": bool(payload.get("dry_run")), "would_play": audio_path}
'''

SCHEMA_TESTS = '''import main


def test_declared_key_is_accepted():
    result = main.run({"audio_path": "a.wav", "dry_run": True})
    assert result["ok"] is True
    assert result["would_play"] == "a.wav"


def test_missing_key_fails_cleanly():
    result = main.run({"dry_run": True})
    assert result["ok"] is False
'''


def test_a_capability_publishes_the_payload_keys_it_accepts(service):
    """A key nobody can discover is a key nobody will ever send.

    A live acquisition produced an audio player expecting `audio_path` while the
    caller passed `path`, so a correctly-built capability was unusable.
    """

    instance = service(ScriptedBrain(implementation=SCHEMA_IMPLEMENTATION, tests=SCHEMA_TESTS))
    outcome = instance.ensure("play an audio file", max_steps=25)
    assert outcome.acquired, outcome.reason

    schema = outcome.manifest.input_schema
    assert "audio_path" in schema["properties"]
    assert schema["required"] == ["audio_path"]

    execution = instance.execute(outcome.capability_id, {"audio_path": "song.wav", "dry_run": True})
    assert execution.ok and execution.output["would_play"] == "song.wav"


def test_a_capability_without_a_declared_schema_still_installs(service):
    """The declaration is how discovery works, not a condition of existing."""

    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    outcome = instance.ensure("play an audio file", max_steps=25)
    assert outcome.acquired
    assert "dry_run" in outcome.manifest.input_schema["properties"]


def test_constraints_reach_the_model_in_full(service):
    """Constraints are load-bearing; truncating the list silently loses them."""

    instance = service(ScriptedBrain(implementation=GOOD_IMPLEMENTATION, tests=GOOD_TESTS))
    instance.ensure("play an audio file", max_steps=25)
    briefing = "\n".join(instance.engine.brain.prompts)
    project = instance.engine.store.list_projects()[0]
    for constraint in project.constraints:
        assert constraint in briefing


# ------------------------------------------------------- the scaffold trap

class LazyBrain(ScriptedBrain):
    """A model that changes nothing, leaving the seeded skeleton in place."""

    def generate_structured(self, prompt, schema, *, max_tokens=1000, temperature=0.1, top_p=0.9):
        self.prompts.append(prompt)
        properties = schema.get("properties") or {}
        if "diagnosis" in properties:
            return json.dumps({"diagnosis": "nothing to do", "fix": "nothing"})
        if "tasks" in properties:
            return json.dumps({"tasks": [{"title": "do nothing"}], "acceptance": []})
        if "findings" in properties:
            return json.dumps({"tool_calls": [{"name": "list_files", "arguments": {}}]})
        self.executes += 1
        return json.dumps({"tool_calls": [{"name": "read_file", "arguments": {"path": "main.py"}}]})


def test_the_seeded_skeleton_cannot_certify_itself(service):
    """Observed live: an untouched scaffold registered as a verified capability.

    run() returned a dict, and the placeholder test asserted only that "ok" was
    present, so both the contract check and the test check passed while the
    capability did nothing whatsoever.
    """

    instance = service(LazyBrain(implementation="", tests=""))
    outcome = instance.ensure("play an audio file", max_steps=8)

    assert not outcome.acquired
    assert instance.list() == []
    implemented = next(item for item in outcome.verification["checks"] if item["name"] == "implemented")
    assert not implemented["ok"]


def test_the_skeleton_marker_reaches_both_gates(service):
    """The loop must also see it, so it keeps working instead of stopping."""

    from capabilities.service import NOT_IMPLEMENTED

    instance = service(LazyBrain(implementation="", tests=""))
    instance.ensure("play an audio file", max_steps=8)
    project = instance.engine.store.list_projects()[0]

    contract = next(item for item in project.acceptance if "implemented" in item.text)
    assert NOT_IMPLEMENTED in " ".join(contract.check)
    assert not contract.satisfied
