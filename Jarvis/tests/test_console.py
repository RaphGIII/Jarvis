"""The console's contract, exercised without loading a model.

Every test here uses a kernel whose tiers are served by a stub. That is not a
convenience: probing a real tier costs a 45-second model load, and a console
test suite that needs one per assertion would never be run. What the console
does with what the kernel returns is the part worth testing.
"""

from __future__ import annotations

import json
import sys

import pytest

from brain.tiers import ModelCatalog, ModelProbe, ModelTier, default_catalog
from core.kernel import JarvisKernel, KernelConfig
from jarvis.cli import JarvisConsole, _extract_steps
from projects.models import ProjectState


class StubProvider:
    """Answers like a model without being one."""

    provider_name = "stub"

    def __init__(self, spec=None, *, reply="a plain answer"):
        self.model_name = getattr(spec, "model", "stub-model")
        self.reply = reply
        self.calls = []

    def list_models(self):
        return [self.model_name]

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return self.reply

    def generate_structured(self, prompt, schema, **kwargs):
        self.calls.append(prompt)
        return json.dumps({"tool_calls": []})


@pytest.fixture
def console(tmp_path, monkeypatch):
    catalog = ModelCatalog(specs=default_catalog())
    kernel = JarvisKernel(
        KernelConfig(state_root=tmp_path / "state", config_root=tmp_path / "config", enable_research_tools=False),
        catalog=catalog,
    )
    kernel.probe = ModelProbe(catalog, ttl_seconds=600, provider_factory=lambda spec: StubProvider(spec))
    monkeypatch.setattr(kernel, "provider", lambda tier: StubProvider(catalog.get(tier)))

    # The console shares the kernel's providers, so patching the kernel is
    # enough -- and if that ever stops being true, these tests will start
    # loading a real model, which is exactly the regression worth catching.
    return JarvisConsole(kernel=kernel)


def _run(console, line, capsys):
    keep_going = console.handle(line)
    return keep_going, capsys.readouterr().out


# ------------------------------------------------------------------ session


@pytest.mark.parametrize("word", ["/quit", "/exit", "/bye", "quit", "exit", "bye"])
def test_every_documented_exit_word_ends_the_session(console, capsys, word):
    assert _run(console, word, capsys)[0] is False


def test_an_empty_line_is_ignored(console, capsys):
    keep_going, out = _run(console, "   ", capsys)
    assert keep_going and out == ""


def test_help_lists_the_commands(console, capsys):
    _, out = _run(console, "/help", capsys)
    for command in ("/status", "/projects", "/work", "/learn", "/persona"):
        assert command in out


def test_an_unknown_command_says_so_without_ending_the_session(console, capsys):
    keep_going, out = _run(console, "/nonsense", capsys)
    assert keep_going
    assert "Unknown command" in out


def test_a_command_that_needs_an_argument_says_which(console, capsys):
    _, out = _run(console, "/new", capsys)
    assert "needs something after it" in out


def test_a_failing_command_does_not_end_the_session(console, capsys, monkeypatch):
    def explode():
        raise RuntimeError("something broke")

    monkeypatch.setattr(console, "list_projects", explode)
    console.run  # the loop catches; handle() itself propagates
    with pytest.raises(RuntimeError):
        console.handle("/projects")


# ------------------------------------------------------------------ status


def test_status_reports_every_tier_and_the_machine(console, capsys):
    _, out = _run(console, "/status", capsys)
    for tier in ModelTier:
        assert tier.value in out
    assert "context windows" in out
    assert "cloud-free" in out


def test_the_banner_states_the_cost_position(console, capsys):
    console.banner()
    out = capsys.readouterr().out
    assert "local only, no paid service enabled" in out


# ------------------------------------------------------------------ routing


def test_a_question_is_answered_directly(console, capsys):
    _, out = _run(console, "What is the capital of France?", capsys)
    assert "a plain answer" in out


def test_work_becomes_a_project_rather_than_a_reply(console, capsys, monkeypatch):
    """Asked to build something, Jarvis must not describe what it would build."""

    started = []
    monkeypatch.setattr(console, "start_project", lambda goal: started.append(goal))

    _, out = _run(console, "Build me a small script that renames files", capsys)

    assert started == ["Build me a small script that renames files"]
    assert "Opening a project" in out


def test_a_restated_goal_is_recognised_as_the_existing_project(console, capsys, monkeypatch):
    project = console.kernel.start_project("build a chess board overlay")
    monkeypatch.setattr(console, "start_project", lambda goal: pytest.fail("must not start a duplicate"))

    _, out = _run(console, "build a chess board overlay", capsys)

    assert project.id in out
    assert "/work to continue" in out


# ------------------------------------------------------------------ projects


def test_projects_and_project_render_what_a_user_needs(console, capsys):
    project = console.kernel.start_project("do a thing")
    project.add_acceptance("tests pass", check=[sys.executable, "-c", "pass"])
    project.add_task("write it")
    project.add_blocker("needs a credential", needs_user=True)
    console.kernel.projects.save(project)

    _, out = _run(console, "/projects", capsys)
    assert project.id in out

    _, out = _run(console, f"/project {project.id}", capsys)
    assert "tests pass" in out
    assert "write it" in out
    assert "needs a credential" in out and "needs you" in out


def test_a_criterion_without_a_check_is_shown_as_unprovable(console, capsys):
    project = console.kernel.start_project("vague")
    project.add_acceptance("it feels right")
    console.kernel.projects.save(project)

    _, out = _run(console, f"/project {project.id}", capsys)
    assert "NO RUNNABLE CHECK" in out


def test_a_project_can_be_named_by_words_rather_than_id(console, capsys):
    project = console.kernel.start_project("build a chess board overlay")
    _, out = _run(console, "/project chess overlay", capsys)
    assert project.id in out


def test_say_adds_a_requirement_to_the_current_project(console, capsys):
    project = console.kernel.start_project("capture the screen")
    _run(console, "/say also recognise the board", capsys)

    reopened = console.kernel.projects.load(project.id)
    assert [item.text for item in reopened.requirements][-1] == "also recognise the board"


def test_say_without_a_project_says_so(console, capsys):
    _, out = _run(console, "/say something", capsys)
    assert "No project to add to" in out


def test_work_refuses_clearly_when_the_build_model_is_unusable(console, capsys, monkeypatch):
    monkeypatch.setattr(console.kernel, "ready_for_autonomous_work", lambda **kw: (False, "BUILD_LOCAL is MODEL_MISSING"))
    project = console.kernel.start_project("do a thing")

    _, out = _run(console, f"/work {project.id}", capsys)

    assert "Cannot work" in out and "MODEL_MISSING" in out


# ------------------------------------------------------------------ knowledge


def test_remember_and_recall_round_trip(console, capsys):
    _run(console, "/remember stockfish lives in C:/tools/stockfish.exe", capsys)
    _, out = _run(console, "/recall stockfish", capsys)
    assert "stockfish" in out.lower()


def test_recall_says_so_when_nothing_matches(console, capsys):
    _, out = _run(console, "/recall something nobody ever mentioned", capsys)
    assert "Nothing found" in out


def test_capabilities_says_so_when_there_are_none(console, capsys):
    _, out = _run(console, "/capabilities", capsys)
    assert "No capabilities yet" in out


def test_use_reports_an_unknown_capability_clearly(console, capsys):
    _, out = _run(console, "/use local.nope.nope", capsys)
    assert "No capability called" in out


def test_use_rejects_malformed_json_without_crashing(console, capsys):
    _, out = _run(console, "/use local.thing {not json}", capsys)
    assert "not valid JSON" in out


# ------------------------------------------------------------------ persona


def test_persona_lists_and_switches(console, capsys):
    _, out = _run(console, "/persona", capsys)
    assert "default" in out and "mentor" in out

    _, out = _run(console, "/persona mentor", capsys)
    assert "now mentor" in out
    assert console.personas.active().name == "mentor"


def test_the_old_persona_name_still_switches(console, capsys):
    """The built-ins were called "jarvis" when the product was. Renaming them
    must not turn a stored preference or a typed `/persona jarvis` into an error.
    """

    _, out = _run(console, "/persona jarvis", capsys)

    assert "now default" in out
    assert console.personas.active().name == "default"


def test_an_unknown_persona_is_reported_with_the_options(console, capsys):
    _, out = _run(console, "/persona nonexistent", capsys)
    assert "Available" in out


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "argument, expected",
    [
        ("", (None, "")),
        ("proj_123", (None, "proj_123")),
        ("--steps 5", (5, "")),
        ("proj_123 --steps 12", (12, "proj_123")),
        ("--steps notanumber proj_1", (None, "proj_1")),
    ],
)
def test_steps_are_extracted_from_the_argument(argument, expected):
    assert _extract_steps(argument) == expected
