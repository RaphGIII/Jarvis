"""When to stop trying locally, decided from counts rather than opinions.

The brief's instruction -- "do not rely only on the LLM saying this is hard" --
is the whole design constraint. A model's estimate of its own difficulty comes
from the same weights that are about to fail, and small models are
systematically overconfident on exactly the tasks they cannot do. So every
signal here is something that can be counted or measured, and the tests below
check both that real evidence escalates and that plausible-sounding
non-evidence does not.
"""

from __future__ import annotations

import pytest

from experts.escalation import (
    Attempt,
    EscalationController,
    EscalationDecision,
    EscalationSignals,
    PerformanceLedger,
    classify_goal,
)


@pytest.fixture()
def ledger(tmp_path):
    return PerformanceLedger(tmp_path / "performance.jsonl")


@pytest.fixture()
def controller(ledger):
    return EscalationController(ledger)


def fill(ledger, task_class, tier, *, passed, failed):
    for _ in range(passed):
        ledger.record(Attempt(task_class=task_class, tier=tier, succeeded=True))
    for _ in range(failed):
        ledger.record(Attempt(task_class=task_class, tier=tier, succeeded=False))


# --------------------------------------------------------------------------
# The bias is towards trying locally
# --------------------------------------------------------------------------

def test_nothing_having_failed_does_not_escalate(controller):
    decision = controller.decide(EscalationSignals())

    assert not decision.escalate
    assert "nothing has failed" in decision.reason


def test_one_failure_is_normal_operation_not_a_crisis(controller):
    """The loop is designed to fail and repair; one failure means it is working."""

    decision = controller.decide(EscalationSignals(local_failures=1, distinct_diagnoses=1))

    assert not decision.escalate


def test_two_failures_with_new_diagnoses_each_time_keep_going(controller):
    """Different diagnoses mean the loop is still learning from its evidence."""

    decision = controller.decide(EscalationSignals(local_failures=2, distinct_diagnoses=2))

    assert not decision.escalate


def test_a_wide_change_alone_does_not_escalate(controller):
    """Scope is a risk factor, not a failure. Try it first."""

    decision = controller.decide(EscalationSignals(files_in_scope=20))

    assert not decision.escalate


def test_escalation_can_be_switched_off_entirely(ledger):
    controller = EscalationController(ledger, enabled=False)

    decision = controller.decide(EscalationSignals(local_failures=99, user_requested=True))

    assert not decision.escalate
    assert "disabled" in decision.reason


# --------------------------------------------------------------------------
# What does escalate
# --------------------------------------------------------------------------

def test_the_user_asking_settles_it(controller):
    decision = controller.decide(EscalationSignals(user_requested=True))

    assert decision.escalate
    assert "user" in decision.reason


def test_repeating_the_same_diagnosis_escalates(controller):
    """Three failures explained the same way once: the loop has stopped learning."""

    decision = controller.decide(EscalationSignals(local_failures=4, distinct_diagnoses=1))

    assert decision.escalate
    assert "repeating the same diagnosis" in decision.reason


def test_exhausted_tasks_escalate(controller):
    decision = controller.decide(
        EscalationSignals(local_failures=3, distinct_diagnoses=3, abandoned_tasks=1)
    )

    assert decision.escalate
    assert "exhausted" in decision.reason


def test_a_wide_change_that_has_already_failed_escalates(controller):
    decision = controller.decide(EscalationSignals(files_in_scope=12, local_failures=1))

    assert decision.escalate
    assert "wide change" in decision.reason


def test_burning_the_time_budget_while_failing_escalates(controller):
    decision = controller.decide(
        EscalationSignals(local_failures=2, distinct_diagnoses=2, seconds_spent=700, seconds_budget=1000)
    )

    assert decision.escalate
    assert "time budget" in decision.reason


def test_every_decision_carries_its_evidence(controller):
    decision = controller.decide(EscalationSignals(local_failures=4, distinct_diagnoses=1))

    assert decision.evidence
    assert any("failed verification" in item for item in decision.evidence)


# --------------------------------------------------------------------------
# Measured history, not impressions
# --------------------------------------------------------------------------

def test_a_measured_poor_local_rate_escalates_after_one_failure(controller, ledger):
    """1-in-5 lifetime is exactly scenario A's record. Do not re-derive it."""

    fill(ledger, "self_development", "build_local", passed=1, failed=4)

    decision = controller.decide(
        EscalationSignals(task_class="self_development", local_failures=1, distinct_diagnoses=1)
    )

    assert decision.escalate
    assert "20%" in decision.reason
    assert any("5 attempts" in item for item in decision.evidence)


def test_a_good_local_rate_does_not_escalate_on_one_failure(controller, ledger):
    fill(ledger, "build", "build_local", passed=9, failed=1)

    decision = controller.decide(
        EscalationSignals(task_class="build", local_failures=1, distinct_diagnoses=1)
    )

    assert not decision.escalate


def test_a_poor_rate_with_too_little_history_is_not_acted_on(controller, ledger):
    """Two failures is not a rate, it is two failures."""

    fill(ledger, "vision", "build_local", passed=0, failed=2)

    decision = controller.decide(
        EscalationSignals(task_class="vision", local_failures=1, distinct_diagnoses=1)
    )

    assert not decision.escalate


def test_an_unknown_task_class_is_not_treated_as_hopeless(controller):
    """No history means unknown, not known-bad; otherwise every new kind of
    work escalates on its first attempt."""

    decision = controller.decide(
        EscalationSignals(task_class="never-seen-before", local_failures=1, distinct_diagnoses=1)
    )

    assert not decision.escalate


def test_a_poor_history_still_needs_something_to_have_failed_here(controller, ledger):
    fill(ledger, "self_development", "build_local", passed=1, failed=9)

    decision = controller.decide(EscalationSignals(task_class="self_development"))

    assert not decision.escalate, "escalating before trying would waste quota on a task that might work"


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------

def test_an_unseen_combination_has_no_rate(ledger):
    rate, samples = ledger.success_rate("anything", "build_local")

    assert rate is None and samples == 0


def test_the_rate_is_computed_from_what_was_recorded(ledger):
    fill(ledger, "build", "build_local", passed=3, failed=1)

    rate, samples = ledger.success_rate("build", "build_local")

    assert rate == 0.75 and samples == 4


def test_tiers_are_counted_separately(ledger):
    fill(ledger, "build", "build_local", passed=0, failed=4)
    fill(ledger, "build", "expert", passed=4, failed=0)

    assert ledger.success_rate("build", "build_local")[0] == 0.0
    assert ledger.success_rate("build", "expert")[0] == 1.0


def test_history_survives_a_restart(tmp_path):
    path = tmp_path / "performance.jsonl"
    first = PerformanceLedger(path)
    fill(first, "build", "build_local", passed=2, failed=1)

    reloaded = PerformanceLedger(path)

    assert reloaded.success_rate("build", "build_local") == (pytest.approx(2 / 3), 3)


def test_a_corrupt_line_does_not_destroy_the_history(tmp_path):
    path = tmp_path / "performance.jsonl"
    ledger = PerformanceLedger(path)
    fill(ledger, "build", "build_local", passed=1, failed=0)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    reloaded = PerformanceLedger(path)

    assert reloaded.success_rate("build", "build_local")[1] == 1


def test_the_ledger_is_bounded(tmp_path):
    ledger = PerformanceLedger(tmp_path / "performance.jsonl", keep=10)

    fill(ledger, "build", "build_local", passed=50, failed=50)

    assert len(ledger.attempts()) == 10


def test_recording_through_the_controller_reaches_the_ledger(controller, ledger):
    controller.record("build", "build_local", True, seconds=12.5)

    assert ledger.success_rate("build", "build_local") == (1.0, 1)


def test_a_controller_without_a_ledger_still_works():
    controller = EscalationController(None)

    controller.record("build", "build_local", True)
    decision = controller.decide(EscalationSignals(local_failures=4, distinct_diagnoses=1))

    assert decision.escalate


def test_the_summary_groups_by_class_and_tier(ledger):
    fill(ledger, "build", "build_local", passed=1, failed=1)

    summary = ledger.summary()

    assert summary["build/build_local"] == {"attempts": 2, "passed": 1, "rate": 0.5}


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "goal,expected",
    [
        ("Add a /goodbye command to this repository", "self_development"),
        ("Learn to play music on this computer", "capability"),
        ("Change the eye animation in the UI", "ui"),
        ("Recognise the chess board in this image", "vision"),
        ("Fix the failing test in the parser", "debug"),
        ("Build a small CLI tool for me", "build"),
        ("Something entirely unlike the others", "general"),
    ],
)
def test_goals_are_classified_coarsely(goal, expected):
    assert classify_goal(goal) == expected


def test_german_goals_are_classified_too():
    assert classify_goal("Baue mir ein kleines Programm") == "build"
    assert classify_goal("Repariere den Fehler im Parser") == "debug"


def test_an_empty_goal_is_general():
    assert classify_goal("") == "general"
