"""What the engineer is told, and what the owner is told back.

Live mission ``5eaac370d3`` (2026-09-09).  The owner asked:

    "Zeus bringe dir die Fähigkeit bei, dass du immer im fullscreen läufst und
     keine fenster ansicht und ich mit f11 das togglen kann"

Codex was correctly chosen as the engineer, worked for 759 seconds in an
isolated worktree, produced a real, tested change -- to ``service/routing.py``
and ``tests/test_routing.py``, teaching the ROUTER to recognise the sentence.
No fullscreen. The mission then failed with:

    "no verified candidate: ZEUS still imports and its kernel still assembles:
     ok; targeted tests: ... ok; the change has the shape the request needs:
     ok; no existing definition was removed or hijacked: o[k]"

-- four passing checks, truncated before the one that failed. Three separate
defects produced that, and one test each is below:

1.  ``_investigate`` scored every matching term equally, so "bringe",
    "fähigkeit", "keine", "kann" -- the words used to ask for ANY change --
    outvoted "fullscreen" and "f11", and ``tests/test_routing.py`` came second
    in a list the brief introduced as "read them first".
2.  Nothing looked at whether the request was already built. It was: mission
    ``d1309425e9`` had implemented and promoted fullscreen and F11 the day
    before.
3.  The reason a candidate is rejected was assembled from every check, passing
    ones first, and cut at 300 characters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from service.selfdev import SelfDevMission, SelfDevRunner, _is_ordinary_word

REQUEST = ("Zeus bringe dir die Fähigkeit bei, dass du immer im fullscreen läufst "
           "und keine fenster ansicht und ich mit f11 das togglen kann")


def _repo(tmp_path: Path) -> Path:
    """A miniature ZEUS: the fullscreen feature already built, plus noise.

    The noise is the point. ``service/routing.py`` and ``tests/test_routing.py``
    are written so that they match the request's *asking* words and nothing
    else, exactly as the real ones did.
    """

    repo = tmp_path / "repo"
    (repo / "jarvis").mkdir(parents=True)
    (repo / "service").mkdir()
    (repo / "tests").mkdir()
    (repo / "ui").mkdir()

    (repo / "jarvis" / "window.py").write_text(
        'DEFAULT_WINDOW_MODE = "fullscreen"\n\n\n'
        'def normalize_window_mode(value):\n'
        '    if value in {"window", "windowed", "fenster"}:\n'
        '        return "windowed"\n'
        '    if value == "fullscreen":\n'
        '        return "fullscreen"\n'
        '    return DEFAULT_WINDOW_MODE\n',
        encoding="utf-8")
    (repo / "service" / "desktop.py").write_text(
        'def toggle_fullscreen(state):\n'
        '    """F11 switches between fullscreen and a framed window."""\n'
        '    return "windowed" if state == "fullscreen" else "fullscreen"\n',
        encoding="utf-8")
    (repo / "service" / "routing.py").write_text(
        '# Requests worded "bringe dir bei", "kann", "keine": the owner asking\n'
        '# for a capability. Nothing here knows anything about windows.\n'
        'LEARN = ("bringe", "bring", "lerne")\n'
        'ASKING = ("kann", "keine", "immer", "ansicht")\n\n\n'
        'def classify(text):\n'
        '    if any(word in text for word in LEARN):\n'
        '        return "learn"\n'
        '    if any(word in text for word in ASKING):\n'
        '        return "ask"\n'
        '    return "other"\n',
        encoding="utf-8")
    (repo / "tests" / "test_routing.py").write_text(
        'from service.routing import classify\n\n\n'
        'def test_bringe_kann_keine_immer_ansicht_are_learn_or_ask():\n'
        '    assert classify("bringe dir bei") == "learn"\n'
        '    assert classify("keine ansicht kann immer") == "ask"\n',
        encoding="utf-8")
    (repo / "ui" / "app.js").write_text(
        'document.addEventListener("keydown", (e) => {\n'
        '  if (e.key === "F11") { api("/api/window", { action: "toggle_fullscreen" }); }\n'
        '});\n',
        encoding="utf-8")
    return repo


class _Runner(SelfDevRunner):
    """Only the two deterministic phases, with the phases recorded."""

    def __init__(self, repo: Path, experience: Any) -> None:
        self.repository = repo
        self.experience = experience
        self.phases: list[tuple[str, str]] = []

    def _phase(self, mission: SelfDevMission, phase: str, detail: str = "") -> None:
        self.phases.append((phase, detail))
        mission.phase = phase


class _NoExperience:
    def relevant(self, *a: Any, **k: Any) -> list[Any]:
        return []

    def guidance(self, *a: Any, **k: Any) -> str:
        return ""


class _PriorFullscreenMission:
    """The 2026-09-08 mission that already implemented this, as it is stored."""

    class _Entry:
        goal = "Zeus, ich möchte, dass du ab jetzt standardmäßig randlos im Vollbild startest. F11 soll togglen."
        outcome = "promoted"
        relevant_files = ["jarvis/window.py", "service/desktop.py"]

    def relevant(self, *a: Any, **k: Any) -> list[Any]:
        return [self._Entry()]

    def guidance(self, *a: Any, **k: Any) -> str:
        return ""


# ==========================================================================
# 1. the words used to ASK for a change are not the subject of the change
# ==========================================================================


def test_the_vocabulary_of_asking_is_not_searched_for() -> None:
    for word in ("bringe", "fähigkeit", "faehigkeit", "kann", "keine", "immer", "ansicht", "läufst"):
        assert _is_ordinary_word(word), word
    for word in ("fullscreen", "f11", "togglen", "fenster", "uptime", "spotify"):
        assert not _is_ordinary_word(word), word


def test_the_implementation_ranks_above_the_test_that_merely_matches_the_wording(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = _Runner(repo, _NoExperience())
    mission = SelfDevMission(request=REQUEST)
    mission.area = "code"

    runner._investigate(mission)

    files = mission.investigation["files"]
    assert files, "the investigation must find something"
    assert files[0] in {"jarvis/window.py", "service/desktop.py"}, files
    assert "tests/test_routing.py" not in files, "a test file is never an implementation candidate"
    assert "service/routing.py" not in files, (
        "the router only matched the words the owner uses to ask for anything", files)


def test_test_files_are_offered_as_context_and_labelled_as_such(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = _Runner(repo, _NoExperience())
    mission = SelfDevMission(request="Zeus, ändere das fullscreen verhalten")
    mission.area = "code"

    runner._investigate(mission)

    assert "tests" in mission.investigation
    for name in mission.investigation.get("tests", []):
        assert name.startswith("tests/")


# ==========================================================================
# 2. what is already built is read before anything is written
# ==========================================================================


def test_the_survey_finds_the_feature_that_already_exists(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = _Runner(repo, _PriorFullscreenMission())
    mission = SelfDevMission(request=REQUEST)
    mission.area = "code"

    runner._investigate(mission)   # runs _survey at the end

    lines = mission.existing["lines"]
    assert lines, "fullscreen and F11 are both in this repository already"
    joined = "\n".join(lines)
    assert "jarvis/window.py" in joined
    assert "ui/app.js" in joined, "the F11 handler is not Python and must still be found"
    assert mission.existing["prior_promoted"], "the earlier promoted mission on this subject is evidence"
    assert any(phase == "SURVEY" for phase, _ in runner.phases)


def test_the_survey_searches_for_subjects_and_not_for_german(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = _Runner(repo, _NoExperience())
    mission = SelfDevMission(request=REQUEST)
    mission.area = "code"

    runner._investigate(mission)

    terms = [t.lower() for t in mission.existing["terms"]]
    assert "fullscreen" in terms and "f11" in terms
    # "Fähigkeit" split into "higkeit" and "läufst" into "ufst" when umlauts
    # were excluded from the pattern instead of rejected after it.
    assert not any(t in {"higkeit", "ufst", "bringe", "kann"} for t in terms), terms


def test_the_brief_tells_the_engineer_to_complete_rather_than_duplicate(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = _Runner(repo, _PriorFullscreenMission())
    mission = SelfDevMission(request=REQUEST)
    mission.area = "code"
    runner._investigate(mission)

    brief = runner._goal_text(mission)

    assert "ALREADY IN THE CODE" in brief
    assert "jarvis/window.py" in brief
    assert "do not re-implement what already works" in brief.lower()
    assert "complete only that" in brief
    # The criterion the candidate is actually judged on, stated to the party
    # being judged.
    assert "HOW THIS IS JUDGED" in brief
    assert "ROUTE better" in brief and "Implement the behaviour." in brief


def test_a_request_about_something_new_gets_no_already_exists_paragraph(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    runner = _Runner(repo, _NoExperience())
    mission = SelfDevMission(request="Zeus, lerne den Umgang mit Kaffeemaschinen ueber Bluetooth")
    mission.area = "code"
    runner._investigate(mission)

    assert "ALREADY IN THE CODE" not in runner._goal_text(mission)


# ==========================================================================
# 3. a rejection says which check failed
# ==========================================================================


def _rejected() -> SelfDevMission:
    mission = SelfDevMission(request=REQUEST)
    mission.changed_files = ["service/routing.py", "tests/test_routing.py"]
    mission.engineering = {"engineer": "CODEX", "build_local_invocations": 0}
    mission.verification = {
        "ok": False,
        "tests": ["tests/test_routing.py", "tests/test_console.py"],
        "checks": [
            {"criterion": "ZEUS still imports and its kernel still assembles", "ok": True, "output": "JARVIS_HEALTH_OK"},
            {"criterion": "targeted tests: tests/test_routing.py, tests/test_console.py", "ok": True, "output": "216 passed"},
            {"criterion": "the change has the shape the request needs", "ok": True, "output": "2 logic file(s)"},
            {"criterion": "no existing definition was removed or hijacked", "ok": True, "output": "every existing definition is still defined"},
            {"criterion": "a reader of the diff sees the request implemented (model inference)", "ok": False,
             "output": "partly: The request is recognized and tested in the code, but the actual fullscreen "
                       "functionality or F11 toggle is not implemented in the system."},
        ],
        "failure_reason": ("a reader of the diff sees the request implemented (model inference): partly: The request "
                           "is recognized and tested in the code, but the actual fullscreen functionality or F11 "
                           "toggle is not implemented in the system."),
        "passed": ["ZEUS still imports and its kernel still assembles"],
        "failed": ["a reader of the diff sees the request implemented (model inference)"],
    }
    return mission


def test_the_rejection_reason_is_the_check_that_failed() -> None:
    mission = _rejected()
    reason = mission.verification["failure_reason"]
    assert "fullscreen" in reason and "not implemented" in reason
    assert "JARVIS_HEALTH_OK" not in reason
    assert not reason.startswith("ZEUS still imports")


def test_mission_control_shows_the_state_and_not_a_list_of_passing_tests() -> None:
    board = _rejected().control_summary()

    assert board["ENGINEER"] == "CODEX"
    assert board["BUILD_LOCAL_INVOCATIONS"] == 0
    assert board["FILES_CHANGED"] == ["service/routing.py", "tests/test_routing.py"]
    assert board["TESTS"] == ["tests/test_routing.py", "tests/test_console.py"]
    assert "fullscreen" in board["FAILURE_REASON"]
    assert board["CHECKS_FAILED"] and "not implemented" in board["CHECKS_FAILED"][0]
    assert len(board["CHECKS_PASSED"]) == 4
    assert board["AWAITING_OWNER"] is False
    assert set(board) >= {"GOAL", "ROUTE", "ENGINEER", "PHASE", "FILES_CHANGED", "TESTS",
                          "FAILURE_REASON", "CANDIDATE", "AWAITING_OWNER", "RESULT"}


def test_a_verified_candidate_waiting_for_the_password_says_so() -> None:
    mission = SelfDevMission(request=REQUEST)
    mission.phase = "AWAITING_AUTHORIZATION"
    mission.outcome = "verified_awaiting_authorization"
    mission.worktree = r"D:\zeus-candidates\5eaac370d3"
    mission.verification = {"ok": True, "checks": [], "tests": []}

    board = mission.control_summary()
    assert board["AWAITING_OWNER"] is True
    assert board["CANDIDATE"].endswith("5eaac370d3")
    assert board["RESULT"] == "verified_awaiting_authorization"
    assert board["FAILURE_REASON"] == ""
