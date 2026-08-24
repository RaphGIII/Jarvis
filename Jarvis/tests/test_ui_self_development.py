"""Jarvis editing its own interface, and being stopped when it breaks it.

A broken UI is the quietest failure in this system. A syntax error in app.js
does not stop the server, does not fail a test, and shows up only as a blank
page in a browser nobody has open. Everything here exists so that promotion has
something that can say no.

The breakage cases are the ones a model actually produces when it rewrites a
file: truncation, a deleted element, a renamed function, an unbalanced brace.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from development.ui_developer import UI_PATHS, UIChangeResult, UIDeveloper
from jarvis.verify_ui import defines, delimiters_balanced, verify


@pytest.fixture()
def ui_copy(tmp_path):
    """A throwaway copy of the real interface, so tests can break it."""

    source = Path(__file__).resolve().parent.parent / "ui"
    destination = tmp_path / "ui"
    shutil.copytree(source, destination)
    return destination


# --------------------------------------------------------------------------
# The health check refuses what it should
# --------------------------------------------------------------------------

def test_the_real_interface_is_healthy(ui_copy):
    """A check that cries wolf is worse than no check."""

    assert verify(ui_copy, serve=False).ok


def test_a_truncated_script_is_caught(ui_copy):
    target = ui_copy / "app.js"
    target.write_text(target.read_text(encoding="utf-8")[:4000], encoding="utf-8")

    assert not verify(ui_copy, serve=False).ok


def test_a_deleted_element_id_is_caught(ui_copy):
    """Remove #eye and the client throws on load, with no other symptom."""

    target = ui_copy / "index.html"
    target.write_text(target.read_text(encoding="utf-8").replace('id="eye"', 'id="iris"'), encoding="utf-8")

    report = verify(ui_copy, serve=False)

    assert not report.ok
    assert any("parts" in check["name"] for check in report.checks if not check["ok"])


def test_a_missing_script_is_caught(ui_copy):
    (ui_copy / "eye.js").unlink()

    assert not verify(ui_copy, serve=False).ok


def test_an_emptied_script_is_caught(ui_copy):
    (ui_copy / "app.js").write_text("", encoding="utf-8")

    assert not verify(ui_copy, serve=False).ok


def test_losing_the_token_placeholder_is_caught(ui_copy):
    """The page would load and then fail to authenticate: "Jarvis is down"."""

    target = ui_copy / "index.html"
    target.write_text(target.read_text(encoding="utf-8").replace("__JARVIS_TOKEN__", '""'), encoding="utf-8")

    assert not verify(ui_copy, serve=False).ok


def test_an_unbalanced_brace_is_caught(ui_copy):
    target = ui_copy / "app.js"
    target.write_text(target.read_text(encoding="utf-8") + "\nfunction broken() {\n", encoding="utf-8")

    assert not verify(ui_copy, serve=False).ok


def test_a_renamed_entry_point_is_caught(ui_copy):
    """window.startJarvis = startJarvis still MENTIONS the name after the
    definition is gone, so a substring check misses this."""

    target = ui_copy / "app.js"
    target.write_text(
        target.read_text(encoding="utf-8").replace("function startJarvis", "function boot"),
        encoding="utf-8",
    )

    assert not verify(ui_copy, serve=False).ok


def test_the_live_serving_check_passes_on_the_real_interface(ui_copy):
    """The only check that exercises the path the browser actually takes."""

    report = verify(ui_copy, serve=True)

    assert report.ok
    assert any("is served" in check["name"] for check in report.checks)


# --------------------------------------------------------------------------
# Definition versus mention
# --------------------------------------------------------------------------

def test_a_definition_counts():
    assert defines("function startJarvis() {}", "startJarvis")
    assert defines("class JarvisEye { }", "JarvisEye")
    assert defines("const startJarvis = () => {}", "startJarvis")


def test_a_mention_does_not():
    assert not defines("window.startJarvis = startJarvis;", "startJarvis")


def test_a_longer_name_containing_it_does_not():
    assert not defines("function startJarvisLater() {}", "startJarvis")


# --------------------------------------------------------------------------
# The delimiter scanner
# --------------------------------------------------------------------------

def test_balanced_source_passes():
    assert delimiters_balanced("function a() { return [1, 2]; }")[0]


def test_an_unclosed_brace_is_reported_with_its_line():
    ok, detail = delimiters_balanced("function a() {\n  if (x) {\n")

    assert not ok
    assert "line" in detail


def test_braces_inside_strings_are_ignored():
    assert delimiters_balanced('const s = "a { b [ c (";')[0]


def test_braces_inside_comments_are_ignored():
    assert delimiters_balanced("// a { b\n/* c ( d */\nfunction a() {}")[0]


def test_template_literals_are_ignored():
    assert delimiters_balanced("const s = `a { b`;\nfunction a() {}")[0]


def test_an_escaped_quote_does_not_end_the_string():
    assert delimiters_balanced('const s = "he said \\" { ";\nfunction a() {}')[0]


def test_a_stray_closing_brace_is_caught():
    ok, detail = delimiters_balanced("function a() {}\n}")

    assert not ok
    assert "unexpected" in detail


# --------------------------------------------------------------------------
# The change pipeline
# --------------------------------------------------------------------------

class FakeCandidate:
    def __init__(self, status, worktree="", changed_files=None, error=""):
        self.status = status
        self.worktree = worktree
        self.changed_files = changed_files or []
        self.error = error


class FakeEngineer:
    def __init__(self, candidate):
        self.candidate = candidate
        self.goal = None
        self.acceptance = None

    def improve(self, repository, goal, acceptance_commands=None, **kwargs):
        self.goal = goal
        self.acceptance = acceptance_commands
        return self.candidate


class FakePromoter:
    def __init__(self, success=True, outcome="promoted"):
        self.success = success
        self.outcome = outcome
        self.moved: list[str] = []

    def promote(self, candidate, *, changed_files, health_check, commit_message=""):
        self.moved = list(changed_files)

        class Record:
            success = self.success
            def to_dict(_self):
                return {"outcome": self.outcome, "error": "" if self.success else "health check failed"}

        return Record()


def test_the_health_check_is_the_acceptance_criterion(tmp_path, ui_copy):
    """The loop must see itself failing it, not learn afterwards."""

    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent), ["ui/app.js"]))
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=FakePromoter())

    developer.change("make the eye greener")

    assert engineer.acceptance
    assert "jarvis.verify_ui" in " ".join(engineer.acceptance[0])


def test_only_ui_paths_are_offered_to_the_model(tmp_path, ui_copy):
    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent), ["ui/app.js"]))
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=FakePromoter())

    developer.change("make the eye greener")

    assert set(engineer.goal.allowed_paths) == set(UI_PATHS)


def test_a_broken_candidate_is_never_promoted(tmp_path, ui_copy):
    """The live interface must not see it at all."""

    (ui_copy / "app.js").write_text("function broken() {", encoding="utf-8")
    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent), ["ui/app.js"]))
    promoter = FakePromoter()
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=promoter)

    result = developer.change("break everything")

    assert not result.ok
    assert result.status == "unhealthy_candidate"
    assert promoter.moved == [], "promotion must not have been attempted"


def test_a_healthy_candidate_is_promoted(tmp_path, ui_copy):
    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent), ["ui/app.js"]))
    promoter = FakePromoter()
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=promoter)

    result = developer.change("make the eye greener")

    assert result.ok
    assert promoter.moved == ["ui/app.js"]


def test_a_candidate_that_touched_other_files_only_moves_the_ui_ones(tmp_path, ui_copy):
    """A UI change does not get to smuggle an engine edit through promotion."""

    engineer = FakeEngineer(
        FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent),
                      ["ui/app.js", "projects/engine.py"])
    )
    promoter = FakePromoter()
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=promoter)

    developer.change("make the eye greener")

    assert promoter.moved == ["ui/app.js"]


def test_a_rejected_candidate_stops_before_verification(tmp_path):
    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_REJECTED", error="tests failed"))
    promoter = FakePromoter()
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=promoter)

    result = developer.change("something")

    assert not result.ok
    assert promoter.moved == []


def test_a_rollback_is_reported_as_such(tmp_path, ui_copy):
    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent), ["ui/app.js"]))
    promoter = FakePromoter(success=False, outcome="rolled_back")
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=promoter)

    result = developer.change("something risky")

    assert not result.ok
    assert result.rolled_back


def test_preview_mode_leaves_the_live_interface_alone(tmp_path, ui_copy):
    engineer = FakeEngineer(FakeCandidate("SELF_DEVELOPMENT_CANDIDATE_READY", str(ui_copy.parent), ["ui/app.js"]))
    promoter = FakePromoter()
    developer = UIDeveloper(tmp_path, engineer=engineer, promoter=promoter)

    result = developer.change("try something", promote=False)

    assert result.ok
    assert result.status == "verified_not_promoted"
    assert promoter.moved == []


def test_a_development_exception_is_reported_not_raised(tmp_path):
    class Exploding:
        def improve(self, *a, **k):
            raise RuntimeError("the model went away")

    result = UIDeveloper(tmp_path, engineer=Exploding()).change("anything")

    assert not result.ok
    assert "went away" in result.detail


def test_the_health_check_demands_proof_it_ran(tmp_path):
    """Exit zero is not enough if something swallowed the command."""

    check = UIDeveloper(tmp_path).health_check()

    assert check.expect_output == "UI_OK"


def test_the_result_serialises_for_the_ui():
    payload = UIChangeResult(ok=True, status="promoted", changed_files=["ui/app.js"]).to_dict()

    assert payload["ok"] is True
    assert payload["changed_files"] == ["ui/app.js"]
