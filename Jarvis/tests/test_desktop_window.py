"""Starting ZEUS puts the interface in its own window, not in the browser.

The owner's complaint was concrete: double-clicking ZEUS.exe dropped the
interface into their normal browser, as one more tab among their own, sharing
that profile and closable by accident. What is checked here is the whole path
that decision travels:

*   ``config/supervisor.json`` -- the supervisor no longer opens a browser at
    all, which is the half of the change that cannot be seen from the core.
*   ``jarvis.serve.interface_plan`` -- and in particular that the supervisor's
    ``--no-browser`` still means "an interface, just not a browser one".
*   ``jarvis.window`` -- the command line that makes a Chromium engine an
    application window with a profile of its own, and the fallback that keeps a
    machine without one from getting no interface at all.

Nothing here starts a browser; the argument vector is the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis import window
from jarvis.serve import build_parser, interface_plan, show_interface

REPOSITORY = Path(__file__).resolve().parent.parent


def plan(*argv: str, environ: dict[str, str] | None = None) -> tuple[str, bool]:
    return interface_plan(build_parser().parse_args(list(argv)), environ or {})


# --------------------------------------------------------------------------
# What ZEUS.exe does
# --------------------------------------------------------------------------

def test_the_supervisor_no_longer_opens_a_browser() -> None:
    """The half of the change that lives in configuration rather than code."""

    from zeus_supervisor.config import SupervisorConfig

    assert SupervisorConfig.load(REPOSITORY).open_browser is False


def test_the_supervisor_config_is_valid_json_the_loader_understands() -> None:
    data = json.loads((REPOSITORY / "config" / "supervisor.json").read_text(encoding="utf-8"))

    assert data["open_browser"] is False


def test_the_supervisors_launch_flags_still_produce_an_interface() -> None:
    """``--no-browser`` is what the supervisor passes the core. It has to mean
    "not a browser", not "nothing at all", or ZEUS.exe would show no UI."""

    assert plan("--no-browser") == ("window", False)


def test_the_default_is_a_window_with_a_browser_to_fall_back_on() -> None:
    assert plan() == ("window", True)


# --------------------------------------------------------------------------
# Saying no
# --------------------------------------------------------------------------

def test_the_browser_can_still_be_asked_for() -> None:
    assert plan("--browser") == ("browser", False)


def test_declining_the_window_falls_back_to_the_browser() -> None:
    assert plan("--no-window") == ("browser", False)


def test_declining_both_opens_nothing() -> None:
    assert plan("--no-window", "--no-browser") == ("none", False)
    assert plan("--browser", "--no-browser") == ("none", False)


def test_the_environment_can_decide_for_callers_that_cannot_pass_flags() -> None:
    assert plan(environ={"ZEUS_UI": "browser"}) == ("browser", False)
    assert plan(environ={"ZEUS_UI": "window"}) == ("window", False)
    assert plan(environ={"ZEUS_UI": "none"}) == ("none", False)


def test_an_explicit_flag_outranks_the_environment() -> None:
    assert plan("--no-window", environ={"ZEUS_UI": "window"}) == ("none", False)


def test_nothing_is_opened_and_nothing_is_said_when_nothing_was_asked_for() -> None:
    assert show_interface("http://127.0.0.1:8420/", ("none", False)) == ""


# --------------------------------------------------------------------------
# The window itself
# --------------------------------------------------------------------------

def test_the_command_is_an_application_window_on_its_own_profile(tmp_path: Path) -> None:
    command = window.window_command("msedge.exe", "http://127.0.0.1:8420/?token=x", profile_dir=tmp_path)

    assert command[0] == "msedge.exe"
    # --app is what removes the tab strip and the address bar.
    assert "--app=http://127.0.0.1:8420/?token=x" in command
    # --user-data-dir is what keeps it out of the owner's own browsing.
    assert f"--user-data-dir={tmp_path}" in command
    assert any(argument.startswith("--window-size=") for argument in command)
    assert "--no-first-run" in command


def test_the_profile_follows_the_state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_STATE_ROOT", str(tmp_path))

    assert window.default_profile_dir() == tmp_path / "window"


def test_an_explicit_engine_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "my-chromium.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("ZEUS_WINDOW_BROWSER", str(fake))

    assert window.find_engine() == str(fake)


def test_an_engine_that_does_not_exist_is_not_silently_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEUS_WINDOW_BROWSER", "no-such-browser-anywhere.exe")

    assert window.find_engine() == ""


def test_opening_launches_the_engine_and_reports_the_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[list[str]] = []

    class FakeProcess:
        pid = 4242

        def __init__(self, command, **_: object) -> None:
            launched.append(list(command))

    monkeypatch.setattr(window, "find_engine", lambda: "msedge.exe")
    monkeypatch.setattr(window.subprocess, "Popen", FakeProcess)

    launch = window.open_window("http://127.0.0.1:8420/", profile_dir=tmp_path / "profile")

    assert launch.ok and launch.mode == "window" and launch.pid == 4242
    assert launched and "--app=http://127.0.0.1:8420/" in launched[0]
    assert (tmp_path / "profile").is_dir(), "the profile directory is created before use"


def test_a_machine_without_an_engine_still_gets_an_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(window, "find_engine", lambda: "")
    monkeypatch.setattr(window.webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)

    launch = window.open_window("http://127.0.0.1:8420/")

    assert launch.ok and launch.mode == "browser"
    assert opened == ["http://127.0.0.1:8420/"]


def test_the_fallback_can_be_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the supervisor's --no-browser buys: no browser, ever."""

    def explode(*_: object, **__: object) -> bool:
        raise AssertionError("the browser must not be opened when the fallback is refused")

    monkeypatch.setattr(window, "find_engine", lambda: "")
    monkeypatch.setattr(window.webbrowser, "open", explode)

    launch = window.open_window("http://127.0.0.1:8420/", fallback=False)

    assert not launch.ok and launch.mode == "none"
    assert "Chromium" in launch.detail


def test_an_engine_that_will_not_start_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object):
        raise OSError("access denied")

    monkeypatch.setattr(window, "find_engine", lambda: "msedge.exe")
    monkeypatch.setattr(window.subprocess, "Popen", refuse)
    monkeypatch.setattr(window.webbrowser, "open", lambda *a, **k: True)

    launch = window.open_window("http://127.0.0.1:8420/")

    assert launch.mode == "browser"
    assert "access denied" in launch.detail
