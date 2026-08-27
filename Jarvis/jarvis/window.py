"""The interface in its own desktop window, not in the owner's browser.

Starting ZEUS used to hand the interface to whatever browser the machine
happens to prefer: a new tab among the owner's other tabs, sharing that
profile's session, extensions and history, and disappearing the moment the tab
was closed by accident.  The owner asked for an application instead -- one
window, its own taskbar button, no address bar, nothing of the browser around
it.

There is no dependency budget for that: the brief forbids adding packages, and
the two stdlib options are a Tk window that cannot render HTML and
``webbrowser``, which is precisely the thing being replaced.  What every
Windows machine *does* already have is a Chromium engine (Edge ships with the
OS), and Chromium's ``--app=`` mode is exactly an application window -- no
tabs, no toolbar, its own entry in the taskbar, the page ``<title>`` as the
window title.  Pointed at ``--user-data-dir`` of our own it is also a separate
profile and a separate process from the owner's browsing, so closing ZEUS never
closes their tabs and vice versa.

The fallback matters as much as the window.  A machine with no Chromium engine
must still get an interface, so this degrades to ``webbrowser`` rather than to
nothing: the worst outcome here is the old behaviour, never a blank screen.

One window, reused.  Opening a fresh one on every start of the core was the
safe choice while nothing recorded the old one: a wrong answer would have left
the owner staring at nothing, and an extra window is only a nuisance.  It
stopped being only a nuisance once ZEUS started restarting itself -- every
promotion added another Chromium window and another set of processes to a
desktop the owner never closed.  So the window is now recorded in a session
file (:func:`session_path`) and :func:`ensure_window` reuses it, but the old
bias is kept where it belongs: *any* doubt about the recorded window -- a dead
pid, an unreadable file, a URL that no longer matches -- opens a new one.  The
failure mode is still "one window too many", never "no window".

That session file is also what makes a second ``ZEUS.exe`` cheap.  The
supervisor refuses to start twice (one mutex per machine), so the second launch
has nothing to do except put the interface back in front of the owner:
:func:`ensure_window` finds the live window and raises it, in the time it takes
to read one small JSON file.

Nothing in this module imports the service, so it can be exercised -- and its
command line inspected -- without starting ZEUS.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Chromium builds keep their per-profile state in a directory of their own.
#: Ours lives beside the rest of the state so that a backup of ``data/jarvis``
#: carries the window's zoom level and geometry with it.
PROFILE_DIRNAME = "window"

#: Where the running window is recorded, beside its profile rather than inside
#: it: everything under the profile directory belongs to Chromium, and a file
#: of ours in there would be one more thing a profile reset could take with it.
SESSION_SUFFIX = "-session.json"

#: Default geometry.  Large enough for the eye, the log and a side panel
#: without being a maximised window the owner has to shrink every time.
DEFAULT_SIZE = (1280, 860)

#: Chromium engines on Windows, relative to a program-files or local-app-data
#: root.  Edge first: it is present on every supported Windows installation,
#: so it is the one answer that is almost always right.
_WINDOWS_ENGINES = (
    r"Microsoft\Edge\Application\msedge.exe",
    r"Google\Chrome\Application\chrome.exe",
    r"Chromium\Application\chrome.exe",
    r"BraveSoftware\Brave-Browser\Application\brave.exe",
    r"Vivaldi\Application\vivaldi.exe",
)

#: Names to try on PATH when none of the usual install roots has one.
_WINDOWS_ON_PATH = ("msedge", "chrome", "chromium", "brave", "vivaldi")

_MACOS_ENGINES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)

_POSIX_ENGINES = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "microsoft-edge-stable", "brave-browser",
)


def default_profile_dir() -> Path:
    """Where the window keeps its profile.  Honours ``JARVIS_STATE_ROOT``."""

    root = os.getenv("JARVIS_STATE_ROOT", "").strip()
    base = Path(root) if root else Path(__file__).resolve().parent.parent / "data" / "jarvis"
    return base / PROFILE_DIRNAME


def _windows_roots() -> list[str]:
    return [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]


def find_engine() -> str:
    """The Chromium executable to host the window, or ``""`` if there is none.

    ``ZEUS_WINDOW_BROWSER`` overrides the search.  An override that does not
    resolve returns nothing rather than quietly falling back to a different
    browser: being pointed at the wrong engine is worth noticing.
    """

    override = os.getenv("ZEUS_WINDOW_BROWSER", "").strip().strip('"')
    if override:
        if Path(override).is_file():
            return override
        return shutil.which(override) or ""

    if sys.platform == "win32":
        for relative in _WINDOWS_ENGINES:
            for root in _windows_roots():
                if root and (Path(root) / relative).is_file():
                    return str(Path(root) / relative)
        for name in _WINDOWS_ON_PATH:
            found = shutil.which(name)
            if found:
                return found
        return ""

    if sys.platform == "darwin":
        for path in _MACOS_ENGINES:
            if Path(path).is_file():
                return path

    for name in _POSIX_ENGINES:
        found = shutil.which(name)
        if found:
            return found
    return ""


def window_command(
    engine: str,
    url: str,
    *,
    profile_dir: str | Path,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> list[str]:
    """The command line that turns a Chromium engine into an application window.

    ``--app`` is what removes the browser: no tabs, no address bar, its own
    taskbar button.  ``--user-data-dir`` is what keeps it out of the owner's
    browsing: a separate profile, so their session, extensions and open tabs
    are untouched and closing one never closes the other.
    """

    width, height = size
    return [
        str(engine),
        f"--app={url}",
        f"--user-data-dir={Path(profile_dir)}",
        f"--window-size={int(width)},{int(height)}",
        # A first-run wizard or a "make me your default browser" bar in front
        # of the interface would defeat the point of an application window.
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-mode",
    ]


@dataclass
class WindowLaunch:
    """What actually happened, so the caller can say it out loud."""

    ok: bool = False
    #: ``window`` (its own desktop window), ``browser`` (the fallback) or
    #: ``none`` (nothing was opened).
    mode: str = "none"
    engine: str = ""
    pid: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "mode": self.mode, "engine": self.engine,
            "pid": self.pid, "detail": self.detail,
        }

    def describe(self) -> str:
        if self.mode == "window":
            return f"in its own window ({Path(self.engine).name})"
        if self.mode == "browser":
            return f"in the default browser -- {self.detail}"
        return self.detail or "not opened"


def open_window(
    url: str,
    *,
    profile_dir: str | Path | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    fallback: bool = True,
) -> WindowLaunch:
    """Show ``url`` in a desktop window, falling back to the browser.

    Never raises: a failure to open a window must not take the service down
    with it, because the service is perfectly usable from another device once
    it is running.
    """

    engine = find_engine()
    if engine:
        directory = Path(profile_dir) if profile_dir is not None else default_profile_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            command = window_command(engine, url, profile_dir=directory, size=size)
            process = subprocess.Popen(  # noqa: S603 - a located executable, fixed arguments
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return WindowLaunch(True, "window", engine, process.pid, "application window")
        except (OSError, ValueError) as exc:
            detail = f"{Path(engine).name} would not start: {exc}"
    else:
        detail = "no Chromium engine (Edge, Chrome, Brave, Vivaldi) was found on this machine"

    if not fallback:
        return WindowLaunch(False, "none", engine, 0, detail)
    try:
        opened = bool(webbrowser.open(url))
    except Exception as exc:  # noqa: BLE001 - any browser failure is just "no UI here"
        return WindowLaunch(False, "none", engine, 0, f"{detail}; the browser also failed: {exc}")
    if not opened:
        return WindowLaunch(False, "none", engine, 0, f"{detail}; no browser could be opened either")
    return WindowLaunch(True, "browser", engine, 0, detail)


# --------------------------------------------------------------------------
# The window session
#
# One small JSON file, written when a window is opened and consulted before the
# next one would be.  It records what is needed to answer "is the window the
# owner had still there?" and nothing else.
# --------------------------------------------------------------------------

def session_path(profile_dir: str | Path | None = None) -> Path:
    directory = Path(profile_dir) if profile_dir is not None else default_profile_dir()
    return directory.with_name(directory.name + SESSION_SUFFIX)


def read_session(profile_dir: str | Path | None = None) -> dict[str, Any]:
    """The recorded window, or ``{}``.  An unreadable file is no window."""

    try:
        data = json.loads(session_path(profile_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_session(launch: "WindowLaunch", url: str, profile_dir: str | Path | None = None) -> None:
    path = session_path(profile_dir)
    payload = {
        "pid": int(launch.pid), "mode": launch.mode, "engine": launch.engine,
        "url": url, "opened_at": time.time(), "owner_pid": os.getpid(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def clear_session(profile_dir: str | Path | None = None) -> None:
    try:
        session_path(profile_dir).unlink()
    except OSError:
        pass


def process_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process, without importing anything.

    Pids are recycled, so this can in principle say yes about a stranger.  The
    consequence is bounded -- a window that is not reopened when it should have
    been -- and the caller checks the recorded URL as well, which a recycled pid
    would not have written.
    """

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes

        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # Declared rather than left to ctypes' defaults: a HANDLE is pointer
        # sized and the default int return truncates it on 64-bit.
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def raise_window(pid: int) -> bool:
    """Bring the window owned by ``pid`` to the front.  Best effort.

    Windows only, and only cosmetic: a window that is open but behind the
    owner's other applications has technically satisfied "the interface is
    there", and practically has not.  Windows refuses foreground changes from a
    process that does not hold it, so a False here is ordinary rather than an
    error -- the window is still open, and still on the taskbar.
    """

    if sys.platform != "win32":
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        SW_RESTORE = 9
        found = False

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd: Any, _: Any) -> bool:
            nonlocal found
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value != pid or not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            found = True
            return False  # the first top-level window of the process is the one

        user32.EnumWindows(visit, 0)
        return found
    except Exception:  # noqa: BLE001 - decoration, never a reason to fail a launch
        return False


def window_is_open(profile_dir: str | Path | None = None) -> bool:
    session = read_session(profile_dir)
    return session.get("mode") == "window" and process_alive(int(session.get("pid") or 0))


def ensure_window(
    url: str,
    *,
    profile_dir: str | Path | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    fallback: bool = True,
) -> WindowLaunch:
    """The window, showing ``url`` -- the one that is already open if there is one.

    This is what every caller should use.  :func:`open_window` always starts a
    new process; this one starts a process only when there is no window to put
    in front of the owner, which is what keeps a restart from leaving a trail of
    Chromium windows behind it.
    """

    directory = Path(profile_dir) if profile_dir is not None else default_profile_dir()
    session = read_session(directory)
    pid = int(session.get("pid") or 0)
    live = session.get("mode") == "window" and process_alive(pid)

    if live and str(session.get("url", "")) == url:
        raised = raise_window(pid)
        return WindowLaunch(
            True, "window", str(session.get("engine", "")), pid,
            "raised the window that was already open" if raised else "the window is already open",
        )
    if live:
        # A window on a URL this core no longer serves is worse than no window:
        # it shows a page that cannot reconnect. Close it and open the real one.
        close_window(profile_dir=directory)

    launch = open_window(url, profile_dir=directory, size=size, fallback=fallback)
    write_session(launch, url, directory)
    return launch


def close_window(*, profile_dir: str | Path | None = None) -> dict[str, Any]:
    """Close the recorded window and forget it.  The core is untouched.

    Chromium spawns a renderer and a handful of utility processes per window,
    so on Windows the whole tree goes rather than the one process we launched;
    otherwise the browser process leaves its children behind.
    """

    directory = Path(profile_dir) if profile_dir is not None else default_profile_dir()
    session = read_session(directory)
    pid = int(session.get("pid") or 0)
    closed = False
    if pid and process_alive(pid):
        closed = _terminate_tree(pid)
    clear_session(directory)
    return {"closed": closed, "pid": pid}


def _terminate_tree(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            completed = subprocess.run(  # noqa: S603 - a fixed command, a numeric argument
                ["taskkill", "/T", "/F", "/PID", str(int(pid))],
                capture_output=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return completed.returncode == 0
        except (OSError, subprocess.SubprocessError, ValueError):
            return False
    try:
        os.kill(int(pid), 15)
        return True
    except (OSError, ValueError):
        return False


def running_url(profile_dir: str | Path | None = None) -> str:
    """The URL of the ZEUS that is up, for a caller that was not told one.

    The supervisor's status file is preferred because it is written by the
    process that owns the port; the window's own session file is the fallback,
    which covers an unsupervised ``python -m jarvis.serve``.  Read as JSON, not
    imported: this module stays independent of both the supervisor and the
    service.
    """

    directory = Path(profile_dir) if profile_dir is not None else default_profile_dir()
    supervisor_dir = os.getenv("ZEUS_SUPERVISOR_DIR", "").strip()
    status = Path(supervisor_dir) if supervisor_dir else directory.parent / "supervisor"
    try:
        data = json.loads((status / "status.json").read_text(encoding="utf-8"))
        url = str(data.get("url", "")) if isinstance(data, dict) else ""
        if url:
            return url
    except (OSError, ValueError):
        pass
    return str(read_session(directory).get("url", ""))


def main(argv: list[str] | None = None) -> int:
    """``python -m jarvis.window <url>`` -- open a window at a URL.

    Useful on its own: it is how a second window onto a running ZEUS is
    opened, and how the engine search is checked on a new machine.

    ``--show`` is the one worth knowing about: with no URL and no arguments it
    puts the window of the running ZEUS back in front of the owner, reusing it
    when it is still open.  That is what a desktop shortcut should call, and it
    costs a JSON read rather than a process start.
    """

    parser = argparse.ArgumentParser(
        prog="python -m jarvis.window",
        description="Open a URL in its own desktop window instead of a browser tab.",
    )
    parser.add_argument("url", nargs="?", default="", help="the URL to show")
    parser.add_argument("--profile-dir", default="", help="override the window's profile directory")
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--no-fallback", action="store_true", help="fail rather than use the browser")
    parser.add_argument("--which", action="store_true", help="print the engine that would be used and stop")
    parser.add_argument("--show", action="store_true",
                        help="reuse the window that is already open; find the URL if none is given")
    parser.add_argument("--close", action="store_true", help="close the window; the core keeps running")
    parser.add_argument("--status", action="store_true", help="print what is recorded about the window")
    args = parser.parse_args(argv)
    profile = args.profile_dir or None

    if args.which:
        engine = find_engine()
        print(engine or "no Chromium engine found")
        return 0 if engine else 1

    if args.status:
        session = dict(read_session(profile))
        session["open"] = window_is_open(profile)
        print(json.dumps(session, indent=2, sort_keys=True))
        return 0

    if args.close:
        result = close_window(profile_dir=profile)
        print("closed the window" if result["closed"] else "no window was open")
        return 0

    url = args.url or (running_url(profile) if args.show else "")
    if not url:
        parser.error("a URL is required" if not args.show else "no running ZEUS was found to show")

    opener = ensure_window if args.show else open_window
    launch = opener(url, profile_dir=profile, size=(args.width, args.height), fallback=not args.no_fallback)
    print(launch.describe())
    return 0 if launch.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
