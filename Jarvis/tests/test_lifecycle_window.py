"""Closing the window, opening it again, and stopping ZEUS for real.

The owner described three states the system could get into and not name:

*   The window closed with X and no way back to it.  ZEUS was still running --
    correctly -- but starting ``ZEUS.exe`` again did nothing the owner could
    see, because the supervisor holds one mutex per machine and the second
    launch has nothing to do but exit.
*   A second window, and a second set of processes, after every self-update,
    because the window was opened fresh on every start of the core.
*   No way to say "stop everything".  ``/api/shutdown`` stops this process; the
    speech worker it started and the listener the supervisor started were left
    holding the microphone and the GPU.

What is checked here is each of those, without a browser and without a
microphone: the session file that makes a window recognisable, the two signals
that ask for it back, and the order the pieces come down in.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from jarvis import window
from jarvis.serve import port_is_taken
from service import desktop as desktop_module
from service.desktop import DesktopWindow
from service.lifecycle import Lifecycle
from service.voice import VoiceService

REPOSITORY = Path(__file__).resolve().parent.parent


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    directory = tmp_path / "window"
    directory.mkdir()
    return directory


@pytest.fixture(autouse=True)
def unsupervised(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may reach the real machine's supervisor.

    Left set, ``ZEUS_SUPERVISED`` would send a shutdown request down a real
    control channel, and ``ZEUS_SUPERVISOR_DIR`` would point the log watcher at
    the running system's own log. Tests that want either set them themselves.
    """

    monkeypatch.delenv("ZEUS_SUPERVISED", raising=False)
    monkeypatch.delenv("ZEUS_SUPERVISOR_DIR", raising=False)


def launch(pid: int = 4242, mode: str = "window", engine: str = "msedge.exe") -> window.WindowLaunch:
    return window.WindowLaunch(True, mode, engine, pid, "application window")


# --------------------------------------------------------------------------
# The session file: what makes a window recognisable at all
# --------------------------------------------------------------------------

def test_the_session_lives_beside_the_profile_not_inside_it(profile: Path) -> None:
    """Everything under the profile directory belongs to Chromium."""

    path = window.session_path(profile)

    assert path.parent == profile.parent
    assert path.name == "window-session.json"


def test_a_recorded_window_can_be_read_back(profile: Path) -> None:
    window.write_session(launch(pid=99), "http://127.0.0.1:8420/?token=x", profile)

    session = window.read_session(profile)

    assert session["pid"] == 99
    assert session["url"] == "http://127.0.0.1:8420/?token=x"
    assert session["mode"] == "window"


def test_no_session_and_an_unreadable_one_both_mean_no_window(profile: Path) -> None:
    assert window.read_session(profile) == {}

    window.session_path(profile).write_text("{not json", encoding="utf-8")
    assert window.read_session(profile) == {}
    assert window.window_is_open(profile) is False


def test_clearing_forgets_the_window_and_tolerates_there_being_none(profile: Path) -> None:
    window.write_session(launch(), "http://x/", profile)
    window.clear_session(profile)

    assert window.read_session(profile) == {}
    window.clear_session(profile)  # must not raise the second time


def test_liveness_is_answered_about_real_processes(profile: Path) -> None:
    assert window.process_alive(os.getpid()) is True
    assert window.process_alive(0) is False
    assert window.process_alive(-1) is False
    assert window.process_alive("not a pid") is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# One window, reused -- the fix for "a second window after every update"
# --------------------------------------------------------------------------

def test_an_open_window_is_reused_rather_than_duplicated(profile: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://127.0.0.1:8420/?token=x"
    window.write_session(launch(pid=4242), url, profile)
    monkeypatch.setattr(window, "process_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(window, "raise_window", lambda pid: True)

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("a second window must not be opened while one is open")

    monkeypatch.setattr(window, "open_window", refuse)

    result = window.ensure_window(url, profile_dir=profile)

    assert result.ok and result.mode == "window" and result.pid == 4242
    assert "already open" in result.detail


def test_a_window_that_died_is_replaced(profile: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://127.0.0.1:8420/?token=x"
    window.write_session(launch(pid=4242), url, profile)
    monkeypatch.setattr(window, "process_alive", lambda _: False)
    monkeypatch.setattr(window, "open_window", lambda *_a, **_k: launch(pid=777))

    result = window.ensure_window(url, profile_dir=profile)

    assert result.pid == 777
    assert window.read_session(profile)["pid"] == 777, "the new window is the one recorded"


def test_a_window_on_a_url_this_core_no_longer_serves_is_closed_first(
    profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live window pointing at a dead core shows a page that cannot
    reconnect, which is worse than no window at all."""

    killed: list[int] = []
    window.write_session(launch(pid=4242), "http://127.0.0.1:8420/?token=old", profile)
    monkeypatch.setattr(window, "process_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(window, "_terminate_tree", lambda pid: killed.append(pid) or True)
    monkeypatch.setattr(window, "open_window", lambda *_a, **_k: launch(pid=777))

    result = window.ensure_window("http://127.0.0.1:8420/?token=new", profile_dir=profile)

    assert killed == [4242]
    assert result.pid == 777


def test_closing_the_window_takes_the_process_tree_and_forgets_it(
    profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed: list[int] = []
    window.write_session(launch(pid=4242), "http://x/", profile)
    monkeypatch.setattr(window, "process_alive", lambda _: True)
    monkeypatch.setattr(window, "_terminate_tree", lambda pid: killed.append(pid) or True)

    result = window.close_window(profile_dir=profile)

    assert result == {"closed": True, "pid": 4242}
    assert killed == [4242]
    assert window.read_session(profile) == {}


def test_closing_when_nothing_is_open_is_not_an_error(profile: Path) -> None:
    assert window.close_window(profile_dir=profile) == {"closed": False, "pid": 0}


# --------------------------------------------------------------------------
# Asking for the window back
# --------------------------------------------------------------------------

def test_a_beacon_asks_for_the_window_once(tmp_path: Path, profile: Path) -> None:
    desk = DesktopWindow("http://x/", state_root=tmp_path, profile_dir=profile)

    desk.request_show("a second ZEUS was started")

    assert desk.pending_request() == "a beacon was written"
    assert desk.pending_request() == "", "the beacon is consumed, not re-fired forever"


def test_the_module_level_beacon_reaches_the_same_file(tmp_path: Path, profile: Path) -> None:
    """What a second core writes on its way out, from a process that has no
    DesktopWindow of its own."""

    desk = DesktopWindow("http://x/", state_root=tmp_path, profile_dir=profile)

    written = desktop_module.request_show(tmp_path, reason="a second ZEUS was started")

    assert written == desk.beacon
    assert desk.pending_request() == "a beacon was written"


def test_a_second_zeus_exe_is_noticed_through_the_supervisors_log(
    tmp_path: Path, profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor_dir = tmp_path / "supervisor"
    (supervisor_dir / "logs").mkdir(parents=True)
    log = supervisor_dir / "logs" / "supervisor.log"
    log.write_text("older lines that were here before this core started\n", encoding="utf-8")
    monkeypatch.setenv("ZEUS_SUPERVISOR_DIR", str(supervisor_dir))

    desk = DesktopWindow("http://x/", state_root=tmp_path, profile_dir=profile)
    desk._log_offset = log.stat().st_size

    assert desk.pending_request() == "", "what was already in the log is not a request"

    with log.open("a", encoding="utf-8") as handle:
        handle.write("2026-08-27 [start] launching core\n")
    assert desk.pending_request() == "", "ordinary supervisor chatter is not a request"

    with log.open("a", encoding="utf-8") as handle:
        handle.write("2026-08-27 another ZEUS supervisor is already running; opening its interface instead\n")
    assert desk.pending_request() == "ZEUS was started again"
    assert desk.pending_request() == "", "the same line does not fire twice"


def test_a_rotated_log_does_not_replay_itself(
    tmp_path: Path, profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor_dir = tmp_path / "supervisor"
    (supervisor_dir / "logs").mkdir(parents=True)
    log = supervisor_dir / "logs" / "supervisor.log"
    log.write_text("x" * 5000, encoding="utf-8")
    monkeypatch.setenv("ZEUS_SUPERVISOR_DIR", str(supervisor_dir))

    desk = DesktopWindow("http://x/", state_root=tmp_path, profile_dir=profile)
    desk._log_offset = log.stat().st_size
    log.write_text("a fresh, shorter log\n", encoding="utf-8")

    assert desk.pending_request() == ""
    assert desk._log_offset == log.stat().st_size


def test_the_phrase_this_depends_on_is_still_in_the_supervisor() -> None:
    """The one coupling in this feature that is not a function call.

    A second ``ZEUS.exe`` cannot reach the running core -- it has no token and
    exits before it could get one -- so the only thing it leaves behind is a
    line in the supervisor's log. The supervisor is owner-protected and cannot
    be changed to help; if that line ever does change, this fails loudly
    instead of the window quietly never coming back.
    """

    source = (REPOSITORY / "zeus_supervisor" / "supervisor.py").read_text(encoding="utf-8")

    assert "another ZEUS supervisor is already running" in source
    assert desktop_module.RELAUNCH_MARKER in "another ZEUS supervisor is already running"


def test_showing_the_window_goes_through_ensure_not_open(
    tmp_path: Path, profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(window, "ensure_window", lambda url, **_k: calls.append(url) or launch())

    desk = DesktopWindow("http://x/", state_root=tmp_path, profile_dir=profile)
    result = desk.show("a beacon was written")

    assert calls == ["http://x/"]
    assert result["ok"] and result["reason"] == "a beacon was written"
    assert desk.shows == 1


# --------------------------------------------------------------------------
# Hiding the interface is not stopping the program
# --------------------------------------------------------------------------

class FakeDesktop:
    def __init__(self) -> None:
        self.did: list[str] = []

    def show(self, reason: str = "") -> dict[str, Any]:
        self.did.append(f"show:{reason}")
        return {"ok": True, "mode": "window"}

    def hide(self, reason: str = "") -> dict[str, Any]:
        self.did.append(f"hide:{reason}")
        return {"ok": True, "closed": True}

    def state(self) -> dict[str, Any]:
        return {"open": True, "pid": 4242}


class FakeVoice:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> dict[str, Any]:
        self.closed = True
        return {"ok": True, "detail": "speech worker stopped"}


class FakeSnapshot:
    @staticmethod
    def to_dict() -> dict[str, Any]:
        return {"state": "idle"}


class FakeCore:
    """Only what :class:`Lifecycle` actually reaches for."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self._voice: Any = FakeVoice()
        self.state = type("State", (), {"snapshot": FakeSnapshot()})()

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self.events.append((args, kwargs))


def test_hiding_the_window_leaves_the_core_alone() -> None:
    core = FakeCore()
    lifecycle = Lifecycle(core)
    lifecycle.desktop = FakeDesktop()

    result = lifecycle.window("hide", reason="the owner closed the window")

    assert result["ok"] and result["closed"]
    assert lifecycle.exit_code == 0 and not lifecycle.exit_event.is_set()
    assert core._voice.closed is False, "the speech worker keeps running"


def test_showing_and_reading_the_window_are_the_other_two_actions() -> None:
    lifecycle = Lifecycle(FakeCore())
    desk = FakeDesktop()
    lifecycle.desktop = desk

    assert lifecycle.window("show")["ok"]
    assert lifecycle.window("state") == {"ok": True, "open": True, "pid": 4242}
    assert lifecycle.window("teleport")["ok"] is False
    assert desk.did == ["show:requested through the API"]


def test_a_core_with_no_window_says_so_rather_than_failing() -> None:
    lifecycle = Lifecycle(FakeCore())

    assert lifecycle.window("show")["ok"] is False
    assert lifecycle.health()["window"] == {"present": False, "open": False}


def test_health_names_the_state_that_had_no_name() -> None:
    """A healthy core with no window is exactly what X leaves behind."""

    lifecycle = Lifecycle(FakeCore())
    lifecycle.desktop = FakeDesktop()

    assert lifecycle.health()["window"]["open"] is True


# --------------------------------------------------------------------------
# ZEUS vollständig beenden
# --------------------------------------------------------------------------

def test_quitting_takes_down_the_children_before_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    swept: list[bool] = []
    monkeypatch.setattr(desktop_module, "stop_speech_processes", lambda **_k: swept.append(True) or {"killed": []})

    core = FakeCore()
    lifecycle = Lifecycle(core)
    desk = FakeDesktop()
    lifecycle.desktop = desk

    result = lifecycle.request_quit("the owner asked ZEUS to quit", requested_by="ui")

    assert result["quit"] and result["stopping"]
    assert desk.did == ["hide:the owner asked ZEUS to quit"], "the window goes first"
    assert core._voice.closed is True, "the speech worker is stopped, not orphaned"
    assert swept == [True], "and anything neither of us owns is swept up"
    # The exit code is the one the supervisor reads as "stay down".
    from zeus_supervisor import EXIT_SHUTDOWN_REQUESTED

    assert lifecycle.exit_code == EXIT_SHUTDOWN_REQUESTED


def test_one_child_failing_does_not_leave_the_others_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop_module, "stop_speech_processes", lambda **_k: {"killed": []})

    class Explodes(FakeDesktop):
        def hide(self, reason: str = "") -> dict[str, Any]:
            raise RuntimeError("the window would not close")

    core = FakeCore()
    lifecycle = Lifecycle(core)
    lifecycle.desktop = Explodes()

    report = lifecycle.stop_children("quit")

    assert report["window"]["ok"] is False
    assert core._voice.closed is True, "the worker is stopped even though the window was not"


def test_a_core_that_never_spoke_has_no_worker_to_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop_module, "stop_speech_processes", lambda **_k: {"killed": []})

    class Silent(FakeCore):
        def __init__(self) -> None:
            super().__init__()
            self._voice = None  # type: ignore[assignment]

    lifecycle = Lifecycle(Silent())
    lifecycle.desktop = FakeDesktop()

    assert lifecycle.stop_children("quit")["speech"] == {"ok": True, "detail": "never started"}


def test_the_speech_worker_is_closed_and_forgotten() -> None:
    """Nothing used to close it, so every restart orphaned one."""

    class Engine:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    service = VoiceService(bus=None)  # type: ignore[arg-type]
    assert service.close() == {"ok": True, "detail": "the speech engine was never started"}

    engine = Engine()
    service._engine = engine
    assert service.close()["ok"] is True
    assert engine.closed is True
    assert service._engine is None


# --------------------------------------------------------------------------
# One core per port
# --------------------------------------------------------------------------

def test_an_occupied_port_is_seen_before_anything_binds_to_it() -> None:
    """On Windows SO_REUSEADDR means "share the port", not "reuse a dead one",
    so a second core would bind happily and quietly steal the connections."""

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        assert port_is_taken("127.0.0.1", port) is True
    finally:
        listener.close()

    assert port_is_taken("127.0.0.1", port) is False


def test_port_zero_cannot_collide() -> None:
    assert port_is_taken("127.0.0.1", 0) is False


# --------------------------------------------------------------------------
# The routes the interface calls
# --------------------------------------------------------------------------

def test_the_api_exposes_hiding_quitting_and_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.http import JarvisHTTPServer

    monkeypatch.setattr(desktop_module, "stop_speech_processes", lambda **_k: {"killed": []})
    monkeypatch.setattr(desktop_module, "count_speech_processes",
                        lambda: {"core": 1, "listener": 1, "worker": 1})

    core = FakeCore()
    core.lifecycle = Lifecycle(core)  # type: ignore[attr-defined]
    core.lifecycle.desktop = FakeDesktop()  # type: ignore[attr-defined]
    server = JarvisHTTPServer(core, port=0, token="t")

    status, hidden = server.handle_api("/api/window", {"action": "hide", "reason": "X"})
    assert status == 200 and hidden["closed"] is True

    status, counts = server.handle_api("/api/processes", {})
    assert status == 200 and counts["ok"] is True and counts["duplicates"] == []

    status, quit_result = server.handle_api("/api/quit", {"reason": "done"})
    assert status == 200 and quit_result["quit"] is True


def test_duplicate_processes_are_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop_module, "count_speech_processes",
                        lambda: {"core": 1, "listener": 2, "worker": 1})

    lifecycle = Lifecycle(FakeCore())

    report = lifecycle.process_counts()

    assert report["ok"] is False
    assert report["duplicates"] == ["listener"]


# --------------------------------------------------------------------------
# The command line a shortcut would call
# --------------------------------------------------------------------------

def test_the_running_url_is_taken_from_the_supervisors_own_status_file(
    tmp_path: Path, profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor_dir = tmp_path / "supervisor"
    supervisor_dir.mkdir(parents=True)
    (supervisor_dir / "status.json").write_text(
        json.dumps({"url": "http://127.0.0.1:8420/?token=live"}), encoding="utf-8"
    )
    monkeypatch.setenv("ZEUS_SUPERVISOR_DIR", str(supervisor_dir))

    assert window.running_url(profile) == "http://127.0.0.1:8420/?token=live"


def test_without_a_supervisor_the_window_remembers_its_own_url(
    profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZEUS_SUPERVISOR_DIR", raising=False)
    window.write_session(launch(), "http://127.0.0.1:8420/?token=solo", profile)

    assert window.running_url(profile) == "http://127.0.0.1:8420/?token=solo"


def test_show_reuses_the_window_and_needs_no_url(profile: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEUS_SUPERVISOR_DIR", raising=False)
    window.write_session(launch(pid=4242), "http://127.0.0.1:8420/?token=x", profile)
    monkeypatch.setattr(window, "process_alive", lambda _: True)
    monkeypatch.setattr(window, "raise_window", lambda _: True)
    monkeypatch.setattr(window, "open_window", lambda *_a, **_k: pytest.fail("must reuse, not reopen"))

    assert window.main(["--show", "--profile-dir", str(profile)]) == 0


def test_status_and_close_are_answerable_from_the_command_line(
    profile: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    window.write_session(launch(pid=4242), "http://x/", profile)
    monkeypatch.setattr(window, "process_alive", lambda _: True)
    monkeypatch.setattr(window, "_terminate_tree", lambda _: True)

    assert window.main(["--status", "--profile-dir", str(profile)]) == 0
    assert json.loads(capsys.readouterr().out)["open"] is True

    assert window.main(["--close", "--profile-dir", str(profile)]) == 0
    assert window.read_session(profile) == {}
