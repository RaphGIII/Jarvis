"""A stochastic model repeats itself; the loop must not repeat with it.

From the capability trajectory of 2026-08-24T10:10:52 -- twenty-one steps,
failed:

    9  DIAGNOSE  The test `test_play_audio_file` is failing because ...
    12 DIAGNOSE  The test `test_play_audio_file` is failing because ...
    15 DIAGNOSE  The test `test_play_audio_file` is failing because ...
    17 DIAGNOSE  The test `test_play_audio_file` is failing because ...
    19 DIAGNOSE  The test `test_play_audio_file` is failing because ...
    21 DIAGNOSE  The test `test_play_audio_file` is failing because ...

Six identical diagnoses, each blaming the test's assertion when the real defect
was that run() still returned the placeholder.  A third of the step budget went
on re-deriving the same wrong answer, because the prompt was byte-identical
each time and a model asked the same question gives the same answer.

The fix is not to ask harder.  It is to make the retry a genuinely different
request by ruling out what has already been shown not to work.
"""

from __future__ import annotations

import pytest

from projects.engine import ProjectEngine


class _Engine(ProjectEngine):
    """Only the diagnosis bookkeeping is under test; nothing else is needed."""

    def __init__(self) -> None:  # noqa: D107 - deliberately does not call super()
        pass


class _Project:
    def __init__(self) -> None:
        self.metadata: dict = {}


@pytest.fixture()
def engine() -> _Engine:
    return _Engine()


@pytest.fixture()
def project() -> _Project:
    return _Project()


# --------------------------------------------------------------------------
# Recognising the same answer twice
# --------------------------------------------------------------------------

def test_a_first_diagnosis_is_not_a_repeat(engine, project):
    assert engine._remember_diagnosis(project, "the import path is wrong") == 0


def test_the_same_diagnosis_twice_is_reported_as_a_repeat(engine, project):
    engine._remember_diagnosis(project, "the import path is wrong")

    assert engine._remember_diagnosis(project, "the import path is wrong") == 2


def test_repeat_counting_keeps_climbing(engine, project):
    counts = [engine._remember_diagnosis(project, "same cause") for _ in range(4)]

    assert counts == [0, 2, 3, 4]


def test_rewording_the_same_answer_still_counts_as_a_repeat(engine, project):
    """Byte equality would never fire: a model rarely repeats itself exactly."""

    engine._remember_diagnosis(project, "The test test_play is failing because the assertion is wrong")

    repeat = engine._remember_diagnosis(
        project, "the test test_play is FAILING because   the assertion is wrong!!"
    )

    assert repeat == 2


def test_a_genuinely_different_diagnosis_is_not_a_repeat(engine, project):
    engine._remember_diagnosis(project, "the assertion in the test is wrong")

    assert engine._remember_diagnosis(project, "run() still returns the placeholder dict") == 0


def test_an_empty_diagnosis_is_not_recorded(engine, project):
    assert engine._remember_diagnosis(project, "   ") == 0
    assert not project.metadata.get("diagnosis_history")


def test_history_does_not_grow_without_bound(engine, project):
    for index in range(200):
        engine._remember_diagnosis(project, f"cause number {index}")

    assert len(project.metadata["diagnosis_history"]) <= 2 * engine._DIAGNOSIS_MEMORY


# --------------------------------------------------------------------------
# Turning that into a different prompt
# --------------------------------------------------------------------------

def test_nothing_is_quoted_back_on_the_first_diagnosis(engine, project):
    assert engine._exhausted_diagnoses(project) == ""


def test_previous_diagnoses_are_quoted_back_and_ruled_out(engine, project):
    engine._remember_diagnosis(project, "the assertion in the test is wrong")

    text = engine._exhausted_diagnoses(project)

    assert "DID NOT FIX ANYTHING" in text
    assert "the assertion in the test is wrong" in text
    assert "do not repeat" in text.lower()


def test_the_prompt_points_away_from_the_symptom(engine, project):
    """The specific wrong turn taken six times: blaming the test, not the code."""

    engine._remember_diagnosis(project, "the assertion is wrong")

    assert "suspect the code under test rather than the test" in engine._exhausted_diagnoses(project)


def test_each_distinct_diagnosis_is_quoted_once_not_once_per_occurrence(engine, project):
    for _ in range(5):
        engine._remember_diagnosis(project, "the assertion in the test is wrong")

    text = engine._exhausted_diagnoses(project)

    assert text.count("the assertion in the test is wrong") == 1


def test_the_quoted_list_is_bounded(engine, project):
    for index in range(50):
        engine._remember_diagnosis(project, f"distinct cause number {index}")

    text = engine._exhausted_diagnoses(project)

    assert text.count("\n  ") <= engine._DIAGNOSIS_MEMORY


def test_the_retry_prompt_actually_differs_from_the_first(engine, project):
    """The whole point: the second request must not be the first one again."""

    first = engine._exhausted_diagnoses(project)
    engine._remember_diagnosis(project, "the assertion in the test is wrong")
    second = engine._exhausted_diagnoses(project)

    assert first != second
    assert len(second) > len(first)


# --------------------------------------------------------------------------
# The loop must be graded by the bar that decides acceptance
# --------------------------------------------------------------------------

def test_the_loop_and_the_verifier_use_literally_the_same_checks():
    """A criterion that can veto acceptance but never appears in the loop's
    evidence is a hidden rubric, and the loop converges on failing it.

    Both live F failures ended contract=ok, implemented=FAILED, because
    `implemented` also inspected the test file while the loop's contract check
    did not -- so the loop had no way to see the check that decided its fate.
    Keeping two lists in step by hand is what failed; they are now one list.
    """

    import inspect

    from capabilities.service import CapabilityService, capability_checks

    names = {check.name for check in capability_checks()}
    assert names == {"tests", "contract", "implemented"}

    builder = inspect.getsource(CapabilityService._start_project)
    verifier = inspect.getsource(CapabilityService._verify)

    assert "capability_checks()" in builder, "the loop's acceptance must come from the shared list"
    assert "capability_checks()" in verifier, "verification must come from the same shared list"


def test_every_check_is_a_runnable_command_not_a_model_opinion():
    from capabilities.service import capability_checks

    for check in capability_checks():
        assert check.command, f"{check.name} has no command"
        assert check.text.strip(), f"{check.name} has no human-readable criterion"


def test_the_seeded_skeleton_fails_the_implemented_check(tmp_path):
    """Run the real criterion against the real skeleton it starts from."""

    import subprocess

    from capabilities.service import _TEMPLATE_MAIN, _TEMPLATE_TEST, capability_checks

    (tmp_path / "main.py").write_text(_TEMPLATE_MAIN, encoding="utf-8")
    (tmp_path / "test_capability.py").write_text(_TEMPLATE_TEST, encoding="utf-8")
    check = next(item for item in capability_checks() if item.name == "implemented")

    completed = subprocess.run(list(check.command), cwd=tmp_path, capture_output=True, text=True)

    assert completed.returncode != 0, "the skeleton must not certify itself"
    assert "placeholder marker" in completed.stderr


def test_a_test_file_that_never_calls_run_fails_the_implemented_check(tmp_path):
    import subprocess

    from capabilities.service import capability_checks

    (tmp_path / "main.py").write_text(
        "def run(payload):\n    return {'ok': True}\n", encoding="utf-8"
    )
    (tmp_path / "test_capability.py").write_text(
        "def test_nothing():\n    assert True\n    assert 1 == 1\n", encoding="utf-8"
    )
    check = next(item for item in capability_checks() if item.name == "implemented")

    completed = subprocess.run(list(check.command), cwd=tmp_path, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "never call main.run" in completed.stderr


def test_a_real_implementation_passes_the_implemented_check(tmp_path):
    import subprocess

    from capabilities.service import capability_checks

    (tmp_path / "main.py").write_text(
        "def run(payload):\n    return {'ok': True, 'error': ''}\n", encoding="utf-8"
    )
    (tmp_path / "test_capability.py").write_text(
        "import main\n"
        "\n"
        "\n"
        "def test_it():\n"
        "    result = main.run({'dry_run': True})\n"
        "    assert result['ok']\n"
        "    assert 'error' in result\n",
        encoding="utf-8",
    )
    check = next(item for item in capability_checks() if item.name == "implemented")

    completed = subprocess.run(list(check.command), cwd=tmp_path, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "SUBSTANCE_OK" in completed.stdout
