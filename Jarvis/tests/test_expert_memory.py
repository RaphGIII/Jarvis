"""Buying a lesson once.

Escalation spends subscription quota that cannot be replaced when it runs out,
so an expert solving something is an expensive event. The expensive part is not
the tokens -- it is that the approach existed nowhere before and will exist
nowhere afterwards unless something writes it down.

Two properties carry the weight here. Only VERIFIED lessons are ever recalled,
because an unverified one is what the expert claimed and teaching a claim to a
future run spreads the mistake. And what failed locally is recorded alongside
what worked, because "threading deadlocked here" saves a future attempt more
reliably than the working code does.
"""

from __future__ import annotations

import json

import pytest

from experts.memory import ExpertMemory, Lesson, lesson_from_escalation


def verified(**kwargs) -> Lesson:
    kwargs.setdefault("task_class", "debug")
    kwargs.setdefault("goal", "fix the flaky websocket reconnect")
    kwargs.setdefault("verification", [{"criterion": "tests pass", "passed": True}])
    return Lesson(**kwargs)


@pytest.fixture()
def memory(tmp_path):
    return ExpertMemory(tmp_path / "lessons.jsonl")


# --------------------------------------------------------------------------
# Only what was actually verified
# --------------------------------------------------------------------------

def test_a_lesson_with_passing_checks_is_verified():
    assert verified().verified


def test_a_lesson_with_a_failing_check_is_not():
    assert not verified(verification=[{"criterion": "tests pass", "passed": False}]).verified


def test_a_lesson_with_no_checks_at_all_is_not():
    """The expert said it worked and nothing confirmed it. That is a rumour."""

    assert not verified(verification=[]).verified


def test_only_verified_lessons_are_recalled(memory):
    memory.record(verified(goal="fix the flaky websocket reconnect", verification=[]))

    assert memory.recall("fix the flaky websocket reconnect") == []


def test_a_verified_lesson_is_recalled(memory):
    memory.record(verified(goal="fix the flaky websocket reconnect"))

    assert memory.recall("the websocket reconnect is flaky again")


# --------------------------------------------------------------------------
# What failed matters as much as what worked
# --------------------------------------------------------------------------

def test_failed_approaches_reach_the_context(memory):
    memory.record(
        verified(
            goal="make the worker shut down cleanly",
            failed_approaches=["tried threading.Event, deadlocked on join"],
            successful_approach="use a sentinel on the queue",
        )
    )

    context = memory.context_for("the worker will not shut down cleanly")

    assert "deadlocked on join" in context
    assert "do not repeat" in context.lower()


def test_the_successful_approach_reaches_the_context(memory):
    memory.record(verified(goal="parse the config", successful_approach="tomllib, not a regex"))

    assert "tomllib" in memory.context_for("parse the config file")


def test_files_and_patterns_reach_the_context(memory):
    memory.record(
        verified(goal="add a retry", files=["net/client.py"], pattern="wrap in a bounded loop")
    )

    context = memory.context_for("add a retry to the client")

    assert "net/client.py" in context
    assert "bounded loop" in context


def test_no_lesson_produces_no_heading(memory):
    """A prompt saying "previous lessons:" with nothing under it invites
    the model to invent some."""

    assert memory.context_for("something never seen before") == ""


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def test_an_unrelated_lesson_is_not_recalled(memory):
    memory.record(verified(goal="fix the websocket reconnect"))

    assert memory.recall("render the chess board from an image") == []


def test_the_same_task_class_is_preferred(memory):
    memory.record(verified(task_class="build", goal="build a parser for the log format"))
    memory.record(verified(task_class="debug", goal="build a parser for the log format"))

    recalled = memory.recall("build a parser for the log format", task_class="debug", limit=1)

    assert recalled[0].task_class == "debug"


def test_a_lesson_from_another_class_can_still_help(memory):
    """A debugging lesson can help a build task that fails the same way."""

    memory.record(verified(task_class="debug", goal="fix the unicode decode on windows"))

    assert memory.recall("unicode decode fails on windows", task_class="build")


def test_recall_is_bounded(memory):
    for index in range(10):
        memory.record(verified(goal=f"fix the websocket reconnect variant {index}"))

    assert len(memory.recall("fix the websocket reconnect", limit=2)) == 2


def test_common_words_do_not_create_false_matches(memory):
    memory.record(verified(goal="please can you fix the thing"))

    assert memory.recall("please can you add the other thing") == []


# --------------------------------------------------------------------------
# Building a lesson from a real escalation
# --------------------------------------------------------------------------

class FakeResult:
    provider = "claude_code"
    summary = "Rewrote the shutdown path to use a queue sentinel."
    files_changed = ["worker.py"]
    commands_run = ["claude -p ..."]
    test_evidence = [("the worker shuts down", True, "exit=0")]


def test_a_lesson_is_built_from_the_expert_result():
    lesson = lesson_from_escalation(
        goal="make the worker shut down cleanly",
        task_class="debug",
        result=FakeResult(),
        failed_approaches=["threading.Event deadlocked"],
        pattern="sentinel on the queue",
    )

    assert lesson.verified
    assert lesson.files == ["worker.py"]
    assert lesson.failed_approaches == ["threading.Event deadlocked"]


def test_verification_comes_from_what_jarvis_reran_not_the_summary():
    """The provider's account of its work is context, never proof."""

    class Lying:
        provider = "x"
        summary = "All tests pass, everything is perfect."
        files_changed: list = []
        commands_run: list = []
        test_evidence = [("the worker shuts down", False, "exit=1")]

    lesson = lesson_from_escalation(goal="g", task_class="debug", result=Lying())

    assert not lesson.verified


def test_a_result_with_no_evidence_yields_an_unverified_lesson():
    class Silent:
        provider = "x"
        summary = "done"
        files_changed: list = []
        commands_run: list = []
        test_evidence: list = []

    assert not lesson_from_escalation(goal="g", task_class="build", result=Silent()).verified


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_lessons_survive_a_restart(tmp_path):
    first = ExpertMemory(tmp_path / "lessons.jsonl")
    first.record(verified(goal="fix the websocket reconnect"))

    reloaded = ExpertMemory(tmp_path / "lessons.jsonl")

    assert reloaded.recall("fix the websocket reconnect")


def test_a_corrupt_line_does_not_destroy_the_rest(tmp_path):
    path = tmp_path / "lessons.jsonl"
    memory = ExpertMemory(path)
    memory.record(verified(goal="fix the websocket reconnect"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    assert ExpertMemory(path).recall("fix the websocket reconnect")


def test_the_store_is_bounded(tmp_path):
    memory = ExpertMemory(tmp_path / "lessons.jsonl", keep=5)

    for index in range(50):
        memory.record(verified(goal=f"goal {index}"))

    assert len(memory.all()) == 5


def test_the_summary_counts_verified_separately(memory):
    memory.record(verified(goal="a real one"))
    memory.record(verified(goal="an unproven one", verification=[]))

    summary = memory.summary()

    assert summary["lessons"] == 2
    assert summary["verified"] == 1


# --------------------------------------------------------------------------
# The Codex adapter's safety, which holds whether or not it is installed
# --------------------------------------------------------------------------

def test_the_codex_adapter_strips_metered_credentials(monkeypatch):
    from experts.codex import CodexExpert

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-inherited")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "also-not")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = CodexExpert(executable="/nonexistent")._environment()

    assert "OPENAI_API_KEY" not in env
    assert "AZURE_OPENAI_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"


def test_the_codex_adapter_declares_the_subscription_channel():
    from runtime.cost_policy import SpendChannel

    from experts.codex import CodexExpert

    assert CodexExpert.channel is SpendChannel.SUBSCRIPTION_CLI


def test_an_absent_codex_cli_is_reported_honestly():
    from experts.codex import CodexExpert

    availability = CodexExpert(executable="").availability()

    assert not availability.available
    assert "not installed" in availability.detail


def test_an_unverified_adapter_does_not_claim_to_be_ready():
    """It was written against documented behaviour, not observed behaviour."""

    from experts.codex import CodexExpert

    assert CodexExpert.verified_on_this_machine is False


def test_an_explicitly_empty_executable_is_not_auto_detected():
    """The bug the Claude adapter had: `or` fell through to PATH."""

    from experts.codex import CodexExpert

    assert CodexExpert(executable="").executable == ""
