"""The desktop/supervisor lifecycle, without a real window or a real supervisor.

Win32 is replaced by fakes; what is tested is the *decisions*: which window is
reused, what a second launch does, what a full quit ends, what a restart keeps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from service import desktop as desktop_mod
from service.desktop import DesktopWindow, FoundWindow
from service.lifecycle import Lifecycle


class FakeWin:
    """A tiny window manager: windows appear on launch, hide/show/focus toggle visibility."""

    def __init__(self) -> None:
        self.windows: dict[int, FoundWindow] = {}
        self.next_hwnd = 100
        self.focused: list[int] = []
        self.launches = 0
        self.killed: list[int] = []

    def find(self, title, include_hidden=True, **_):
        return [w for w in self.windows.values() if w.title == title and (include_hidden or w.visible)]

    def focus(self, hwnd):
        w = self.windows.get(hwnd)
        if w is None:
            return False
        self.windows[hwnd] = FoundWindow(w.hwnd, w.pid, w.title, True, False)
        self.focused.append(hwnd)
        return True

    def hide(self, hwnd):
        w = self.windows.get(hwnd)
        if w is None:
            return False
        self.windows[hwnd] = FoundWindow(w.hwnd, w.pid, w.title, False, False)
        return True

    def launch(self, title):
        self.launches += 1
        hwnd, self.next_hwnd = self.next_hwnd, self.next_hwnd + 1
        self.windows[hwnd] = FoundWindow(hwnd, 5000 + hwnd, title, True, False)
        return {"ok": True}

    def kill(self, pid, tree=False):
        for hwnd, w in list(self.windows.items()):
            if w.pid == pid:
                del self.windows[hwnd]
                self.killed.append(pid)
                return True
        return False


@pytest.fixture
def win(monkeypatch, tmp_path):
    fake = FakeWin()
    monkeypatch.setattr(desktop_mod, "find_windows", fake.find)
    monkeypatch.setattr(desktop_mod, "focus", fake.focus)
    monkeypatch.setattr(desktop_mod, "hide_window", fake.hide)
    monkeypatch.setattr(desktop_mod, "apply_identity", lambda hwnd, **k: {"app_id": True, "icon": True})
    monkeypatch.setattr(desktop_mod, "find_engine", lambda: r"C:\fake\msedge.exe")
    monkeypatch.setattr(desktop_mod, "find_windows_of", lambda pids, **k: [])
    monkeypatch.setattr(desktop_mod.DesktopWindow, "profile_pids", lambda self: set())
    import service.processes as processes

    monkeypatch.setattr(processes, "kill", fake.kill)
    return fake


def _window(tmp_path, win, title="ZEUS"):
    dw = DesktopWindow(url="http://127.0.0.1:8420/", title=title, state_root=tmp_path / "state", profile_dir=tmp_path / "profile")
    dw._launch = lambda: win.launch(title)  # type: ignore[method-assign]
    dw._wait_for_window = lambda timeout: dw.find(include_hidden=False)  # type: ignore[method-assign]
    return dw


def test_show_launches_once_then_reuses_the_window(tmp_path, win):
    dw = _window(tmp_path, win)
    first = dw.show(reason="startup")
    assert first["action"] == "launched" and win.launches == 1 and first["ok"]
    assert dw.identity == {"app_id": True, "icon": True}
    second = dw.show(reason="second ZEUS.exe")
    assert second["action"] == "focused" and win.launches == 1
    assert json.loads(dw.session_path.read_text())["action"] == "focused"


def test_hide_keeps_the_process_and_show_restores_it(tmp_path, win):
    dw = _window(tmp_path, win)
    dw.show()
    assert dw.hide()["action"] == "hidden"
    assert dw.status()["exists"] and not dw.status()["visible"]
    shown = dw.show(reason="beacon")
    assert shown["action"] == "restored" and win.launches == 1


def test_closing_with_x_leaves_the_core_and_show_relaunches(tmp_path, win):
    dw = _window(tmp_path, win)
    dw.show()
    win.windows.clear()  # the owner pressed X: the engine process is gone
    assert not dw.status()["exists"]
    shown = dw.show(reason="second ZEUS.exe")
    assert shown["action"] == "launched" and win.launches == 2


def test_a_restart_finds_the_surviving_window_instead_of_adding_one(tmp_path, win):
    old = _window(tmp_path, win)
    old.show()
    # A new core process: fresh object, same title, same machine.
    new = _window(tmp_path, win)
    shown = new.show(reason="startup")
    assert shown["action"] == "focused" and win.launches == 1 and len(win.windows) == 1


def test_duplicates_are_closed_down_to_one(tmp_path, win):
    dw = _window(tmp_path, win)
    dw.show()
    win.launch("ZEUS")  # a leftover from somewhere
    assert len(win.windows) == 2
    dw.show()
    assert len(win.windows) == 1


def test_beacon_watcher_shows_the_window(tmp_path, win):
    dw = _window(tmp_path, win)
    dw.show()
    dw.hide()
    dw.start_watcher(interval=0.05)
    try:
        dw.beacon_path.parent.mkdir(parents=True, exist_ok=True)
        dw.beacon_path.write_text("1")
        deadline = time.time() + 2
        while time.time() < deadline and not dw.status()["visible"]:
            time.sleep(0.05)
        assert dw.status()["visible"] and not dw.beacon_path.exists()
    finally:
        dw.stop_watcher()


# --------------------------------------------------------------------------
# Lifecycle: quit, leave, readiness
# --------------------------------------------------------------------------

class FakeCore:
    def __init__(self, tmp_path):
        self.kernel = SimpleNamespace(state_root=tmp_path / "state")
        self.identity = SimpleNamespace(product_name="ZEUS")
        self.events = []
        self.state = SimpleNamespace(snapshot=SimpleNamespace(to_dict=lambda: {}))
        self._voice = SimpleNamespace(_engine=SimpleNamespace(close=lambda: self.events.append("engine closed")))
        self.transcript = []

    def emit(self, kind, payload, **_):
        self.events.append((getattr(kind, "value", kind), payload))


def test_full_quit_ends_window_speech_and_asks_for_shutdown(tmp_path, win, monkeypatch):
    core = FakeCore(tmp_path)
    life = Lifecycle(core)
    monkeypatch.setattr(life, "save_conversation", lambda reason: None)
    life.desktop = _window(tmp_path, win)
    life.desktop.show()
    import service.processes as processes

    monkeypatch.setattr(processes, "kill_role", lambda role, keep=(): [42] if role == "listener" else [])
    report = life.request_quit("test")
    assert report["quit"] and report["stopping"] and life.exit_code == 0
    assert report["window"]["action"] == "closed" and win.windows == {}
    assert "engine closed" in core.events and report["speech"]["closed"]
    assert report["listeners_killed"] == [42]


def test_a_restart_leaves_the_window_but_a_shutdown_closes_it(tmp_path, win, monkeypatch):
    core = FakeCore(tmp_path)
    life = Lifecycle(core)
    life.desktop = _window(tmp_path, win)
    life.desktop.show()
    life.leave(final=False)
    assert len(win.windows) == 1 and "engine closed" in core.events
    life.leave(final=True)
    assert win.windows == {}


def test_readiness_levels_are_separate(tmp_path):
    core = FakeCore(tmp_path)
    life = Lifecycle(core)
    assert life.readiness() == {"UI_READY": False, "CORE_READY": False, "AI_READY": False, "VOICE_READY": False, "FULL_READY": False}
    life.mark("http", True, "http://127.0.0.1:8420/")
    assert life.readiness()["UI_READY"] and not life.readiness()["AI_READY"] and not life.ready
    life.mark("fast_local", True, "OK")
    assert life.ready and life.readiness()["AI_READY"] and not life.readiness()["FULL_READY"]
    life.mark("voice", True); life.mark("recogniser", True)
    assert life.readiness()["FULL_READY"]


# --------------------------------------------------------------------------
# Supervisor: a second launch signals, never starts
# --------------------------------------------------------------------------

def test_second_supervisor_signals_the_running_one_or_leaves_a_beacon(tmp_path, monkeypatch):
    from zeus_supervisor.supervisor import Supervisor

    sup = Supervisor.__new__(Supervisor)
    sup.state_dir = tmp_path / "sup"
    sup.state_dir.mkdir()
    sup.config = SimpleNamespace(open_browser=False, host="127.0.0.1", port=8420)
    logs = []
    sup.log = logs.append
    calls = []
    sup._api = lambda path, payload=None, timeout=5: (calls.append(path), {"action": "focused", "seconds": 0.03})[1]
    assert sup._signal_running_instance() == 0 and calls == ["/api/window/show"]

    def down(path, payload=None, timeout=5):
        raise OSError("connection refused")

    sup._api = down
    assert sup._signal_running_instance() == 0
    assert (sup.state_dir / "control" / "window-show").is_file()


# --------------------------------------------------------------------------
# Process counting: the venv launcher pair is one process
# --------------------------------------------------------------------------

def test_venv_launcher_pairs_and_shells_are_not_counted(monkeypatch):
    import service.processes as processes

    rows = [
        processes.ProcessInfo(100, 1, "python.exe", r"C:\Python314\python.exe -m zeus_supervisor run"),
        processes.ProcessInfo(101, 100, "python.exe", r"C:\Python314\python.exe -m jarvis.serve --port 8420"),
        processes.ProcessInfo(102, 100, "python.exe", r"D:\repo\.venv-speech\Scripts\python.exe -m speech.listener --url x"),
        processes.ProcessInfo(103, 102, "python.exe", r"D:\repo\.venv-speech\Scripts\python.exe -m speech.listener --url x"),
        processes.ProcessInfo(104, 101, "python.exe", r"D:\repo\.venv-speech\Scripts\python.exe -m speech.worker"),
        processes.ProcessInfo(105, 104, "python.exe", r"D:\repo\.venv-speech\Scripts\python.exe -m speech.worker"),
        processes.ProcessInfo(106, 7, "bash.exe", "bash -c 'python -m zeus_supervisor run'"),
        processes.ProcessInfo(107, 7, "powershell.exe", "powershell -Command \"... -like '*speech.listener*' ...\""),
    ]
    monkeypatch.setattr(processes, "list_processes", lambda patterns: rows)
    assert processes.counts() == {"core": 1, "listener": 1, "worker": 1, "supervisor": 1}
    assert [p.pid for p in processes.zeus_processes()["listener"]] == [102]


def test_hidden_helper_windows_of_the_engine_are_never_closed(tmp_path, win, monkeypatch):
    """Chromium keeps hidden, untitled windows in the same process; closing one closes the engine."""

    dw = _window(tmp_path, win)
    dw.show()
    keep = dw.hwnd
    helper = FoundWindow(9999, win.windows[keep].pid, "", False, False)
    monkeypatch.setattr(desktop_mod, "find_windows_of", lambda pids, **k: [helper, win.windows[keep]])
    closed = []
    monkeypatch.setattr(desktop_mod, "_win32", lambda: None)
    import service.processes as processes

    monkeypatch.setattr(processes, "kill", lambda pid, tree=False: closed.append(pid) or True)
    assert dw._ensure_single(keep=keep) == []
    assert closed == [] and len(win.windows) == 1
