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
    assert names == {"tests", "contract", "implemented", "static"}

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


# --------------------------------------------------------------------------
# A long-lived project must not run out of repairs
# --------------------------------------------------------------------------

def test_the_repair_budget_resets_on_verified_progress():
    """Persistence is the point; a lifetime budget defeats it.

    The chess project accumulated requirements across four sessions. By the
    third, every diagnosis was refused with "no further repair avenue is
    available" -- the budget counted auto-repair tasks over the whole project
    life, so requirements one and two had spent what requirement three needed.

    A satisfied acceptance check is proof the loop is productive, so it resets
    the count. Spinning still exhausts it, because spinning produces no
    passing checks.
    """

    from projects.engine import _AUTO_REPAIR
    from projects.models import Phase, Project, StepRecord, Task

    engine = _Engine()
    project = Project(goal="a long project")

    # Three repairs spent on an earlier requirement...
    for index in range(3):
        project.tasks.append(Task(title=f"repair {index}", detail=f"{_AUTO_REPAIR} earlier work"))
        project.tasks[-1].created_at = "2020-01-01T00:00:00+00:00"

    assert not engine._repair_budget_left(project), "the budget is spent before progress"

    # ...then something actually passed.
    project.steps.append(
        StepRecord(phase=Phase.VERIFY, summary="all checks pass", success=True,
                   at="2020-06-01T00:00:00+00:00")
    )

    assert engine._repair_budget_left(project), "verified progress must restore the budget"


def test_spinning_still_exhausts_the_budget():
    """The property the lifetime count was protecting, kept."""

    from projects.engine import _AUTO_REPAIR
    from projects.models import Phase, Project, StepRecord, Task

    engine = _Engine()
    project = Project(goal="a stuck project")
    project.steps.append(
        StepRecord(phase=Phase.VERIFY, summary="passed once", success=True,
                   at="2020-01-01T00:00:00+00:00")
    )
    for index in range(4):
        task = Task(title=f"repair {index}", detail=f"{_AUTO_REPAIR} going nowhere")
        task.created_at = "2020-06-01T00:00:00+00:00"   # after the last pass
        project.tasks.append(task)

    assert not engine._repair_budget_left(project)


def test_a_project_with_no_verified_progress_yet_still_has_a_budget():
    from projects.models import Project

    assert _Engine()._repair_budget_left(Project(goal="brand new"))


# --------------------------------------------------------------------------
# "No module named X" reads two ways; only one of them is true here
# --------------------------------------------------------------------------

def _failing_project(evidence: str):
    from projects.models import Project

    project = Project(goal="build something")
    criterion = project.add_acceptance("it works", check=["python", "-c", "import engine"])
    criterion.satisfied = False
    criterion.last_evidence = evidence
    return project


def test_a_module_the_project_must_write_is_named_as_such(tmp_path):
    """Observed live: the model read "No module named 'engine'" as a packaging
    problem and spent its entire repair budget on `pip install python-chess`.
    The module was one the requirement had asked it to write."""

    from tools.registry import ToolContext

    project = _failing_project("ModuleNotFoundError: No module named 'engine'")
    notice = _Engine()._missing_module_notice(project, ToolContext(workspace=tmp_path))

    assert "engine.py DOES NOT EXIST" in notice
    assert "NOT a missing package" in notice


def test_nothing_is_said_once_the_file_exists(tmp_path):
    from tools.registry import ToolContext

    (tmp_path / "engine.py").write_text("def analyse(fen):\n    return {}\n", encoding="utf-8")
    project = _failing_project("ModuleNotFoundError: No module named 'engine'")

    assert _Engine()._missing_module_notice(project, ToolContext(workspace=tmp_path)) == ""


def test_a_genuinely_absent_package_is_not_mislabelled(tmp_path):
    """numpy.py is not something the project was asked to write, but the notice
    only fires for names with no file, so it would claim numpy must be written.
    It says the same thing either way -- write it or install it, the file is
    what is missing -- and the workspace check is what keeps it honest."""

    from tools.registry import ToolContext

    project = _failing_project("ModuleNotFoundError: No module named 'engine'")
    notice = _Engine()._missing_module_notice(project, ToolContext(workspace=tmp_path))

    assert "engine" in notice
    assert "numpy" not in notice


def test_a_failure_with_no_missing_module_says_nothing(tmp_path):
    from tools.registry import ToolContext

    project = _failing_project("AssertionError: expected 3, got 4")

    assert _Engine()._missing_module_notice(project, ToolContext(workspace=tmp_path)) == ""


def test_each_missing_module_is_mentioned_once(tmp_path):
    from tools.registry import ToolContext

    project = _failing_project(
        "No module named 'engine'\nlater: No module named 'engine'"
    )
    notice = _Engine()._missing_module_notice(project, ToolContext(workspace=tmp_path))

    assert notice.count("engine.py DOES NOT EXIST") == 1


def test_partial_progress_also_restores_the_repair_budget():
    """Four criteria passing to five is progress, even though VERIFY reports
    failure because the fifth is still red.

    The chess project hit this: every VERIFY read "4 passed, 2 failed", so no
    step was ever `success`, so the budget never reset and the requirement was
    refused every repair -- while genuinely making progress.
    """

    from projects.engine import _AUTO_REPAIR
    from projects.models import Phase, Project, StepRecord, Task

    engine = _Engine()
    project = Project(goal="partially working")
    for index in range(3):
        task = Task(title=f"repair {index}", detail=f"{_AUTO_REPAIR} earlier")
        task.created_at = "2020-01-01T00:00:00+00:00"
        project.tasks.append(task)

    assert not engine._repair_budget_left(project)

    project.steps.append(
        StepRecord(phase=Phase.VERIFY, summary="4 passed, 2 failed", success=False,
                   productive=True, at="2020-06-01T00:00:00+00:00")
    )

    assert engine._repair_budget_left(project)


def test_an_unproductive_verify_does_not_restore_it():
    from projects.engine import _AUTO_REPAIR
    from projects.models import Phase, Project, StepRecord, Task

    engine = _Engine()
    project = Project(goal="stuck")
    for index in range(3):
        task = Task(title=f"repair {index}", detail=f"{_AUTO_REPAIR} going nowhere")
        task.created_at = "2020-06-01T00:00:00+00:00"
        project.tasks.append(task)
    project.steps.append(
        StepRecord(phase=Phase.VERIFY, summary="4 passed, 2 failed", success=False,
                   productive=False, at="2020-01-01T00:00:00+00:00")
    )

    assert not engine._repair_budget_left(project)


# --------------------------------------------------------------------------
# A finished task whose criterion is still red
# --------------------------------------------------------------------------

def _executed(project, task):
    """Record that this task was the most recently executed one."""

    from projects.models import Phase, StepRecord

    project.steps.append(StepRecord(phase=Phase.EXECUTE, summary="ran", task_id=task.id))


def test_a_done_task_out_of_attempts_can_still_be_reopened():
    """The bug that froze chess requirement 3.

    Every task reached three attempts, reopening skips exhausted tasks, and the
    project permanently lost the ability to repair anything -- while reporting
    "no further repair avenue is available".
    """

    from projects.models import Project, Task, TaskStatus

    engine = _Engine()
    project = Project(goal="engine.py")
    task = Task(title="implement analyse()", status=TaskStatus.DONE, attempts=3)
    project.tasks.append(task)
    _executed(project, task)

    reopened = engine._reopen_failed_task(project, "the Stockfish path is mangled")

    assert reopened is task
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0, "a completion breaks the run of failures"
    assert "Stockfish path" in task.detail


def test_reopening_is_bounded():
    """Otherwise a task oscillates between DONE and reopened forever."""

    from projects.models import Project, Task, TaskStatus

    engine = _Engine()
    project = Project(goal="engine.py")
    task = Task(title="implement analyse()", status=TaskStatus.DONE, attempts=3)
    project.tasks.append(task)
    _executed(project, task)

    for _ in range(task.max_reopenings):
        assert engine._reopen_failed_task(project, "still wrong") is task
        task.status = TaskStatus.DONE
        task.attempts = 3

    assert engine._reopen_failed_task(project, "still wrong") is None


def test_a_failed_task_out_of_attempts_is_still_finished():
    """Nothing claimed it worked, so three failures in a row really is three."""

    from projects.models import Project, Task, TaskStatus

    engine = _Engine()
    project = Project(goal="engine.py")
    task = Task(title="implement analyse()", status=TaskStatus.FAILED, attempts=3)
    project.tasks.append(task)
    _executed(project, task)

    assert engine._reopen_failed_task(project, "try again") is None


def test_a_done_task_with_attempts_left_keeps_its_count():
    """Reopening is not a free reset -- only exhaustion buys a fresh budget."""

    from projects.models import Project, Task, TaskStatus

    engine = _Engine()
    project = Project(goal="engine.py")
    task = Task(title="implement analyse()", status=TaskStatus.DONE, attempts=1)
    project.tasks.append(task)
    _executed(project, task)

    assert engine._reopen_failed_task(project, "fix") is task
    assert task.attempts == 1
    assert task.reopenings == 0


def test_an_open_task_is_never_reopened():
    from projects.models import Project, Task, TaskStatus

    engine = _Engine()
    project = Project(goal="engine.py")
    task = Task(title="in flight", status=TaskStatus.IN_PROGRESS, attempts=3)
    project.tasks.append(task)
    _executed(project, task)

    assert engine._reopen_failed_task(project, "fix") is None


# --------------------------------------------------------------------------
# Evidence that predates the workspace
# --------------------------------------------------------------------------

def _context(workspace):
    from tools.registry import ToolContext

    return ToolContext(workspace=str(workspace), readable_roots=[workspace])


def _project_checked_at(when, *, text="analyse() works"):
    from projects.models import AcceptanceCriterion, Project

    project = Project(goal="chess")
    project.acceptance.append(
        AcceptanceCriterion(text=text, check=["python", "-c", "pass"], last_checked_at=when)
    )
    return project


def test_a_file_written_after_the_check_makes_the_evidence_stale(tmp_path):
    """The bug that burned a whole failure budget.

    An expert fixed engine.py during an escalation. Nothing wrote that back
    into the project, so on resume every DIAGNOSE addressed a defect that had
    already been repaired, and the run died on FAILURE_LIMIT without ever
    verifying.
    """

    engine = _Engine()
    project = _project_checked_at("2020-01-01T00:00:00+00:00")
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")

    assert engine._evidence_predates_the_workspace(project, _context(tmp_path))


def test_an_untouched_workspace_leaves_the_evidence_current(tmp_path):
    import os
    import time

    engine = _Engine()
    target = tmp_path / "engine.py"
    target.write_text("x = 1\n", encoding="utf-8")
    os.utime(target, (time.time() - 3600, time.time() - 3600))
    project = _project_checked_at("2999-01-01T00:00:00+00:00")

    assert not engine._evidence_predates_the_workspace(project, _context(tmp_path))


def test_a_project_that_has_never_been_checked_has_nothing_to_re_establish(tmp_path):
    """Otherwise every fresh project spends a step confirming the obvious."""

    engine = _Engine()
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")
    project = _project_checked_at("")

    assert not engine._evidence_predates_the_workspace(project, _context(tmp_path))


def test_an_empty_workspace_is_not_stale(tmp_path):
    engine = _Engine()
    project = _project_checked_at("2020-01-01T00:00:00+00:00")

    assert not engine._evidence_predates_the_workspace(project, _context(tmp_path))


def test_an_unreadable_timestamp_counts_as_stale(tmp_path):
    """A timestamp that cannot be parsed is not evidence that anything is current."""

    engine = _Engine()
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")
    project = _project_checked_at("not a date")

    assert engine._evidence_predates_the_workspace(project, _context(tmp_path))


def test_the_oldest_check_decides(tmp_path):
    """One current criterion must not vouch for a stale one."""

    from projects.models import AcceptanceCriterion

    engine = _Engine()
    project = _project_checked_at("2999-01-01T00:00:00+00:00", text="fresh")
    project.acceptance.append(
        AcceptanceCriterion(
            text="stale", check=["python", "-c", "pass"],
            last_checked_at="2020-01-01T00:00:00+00:00",
        )
    )
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")

    assert engine._evidence_predates_the_workspace(project, _context(tmp_path))
