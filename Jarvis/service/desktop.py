"""The desktop window, from the core's side: one window, outliving nothing.

The owner's complaint was about states the system did not have a name for.
Closing the window with X left ZEUS running with no way back to it, and
starting ``ZEUS.exe`` again did nothing visible at all -- the supervisor holds
one mutex per machine, so the second launch correctly refuses to start a second
ZEUS and then, correctly and uselessly, exits.  Meanwhile every self-update
added one more Chromium window to the desktop, because the window was opened
fresh on every start of the core.

So the window is given an owner, and the owner is the core:

*It is a view, not the program.*  Closing it kills a browser process.  Nothing
in the core is attached to that process, so the conversation model, the event
bus and the wake-word listener carry on exactly as they were.  This module
makes that a decision rather than an accident by never watching for the window
to die and never reacting when it does.

*There is one of it.*  :func:`jarvis.window.ensure_window` reuses the recorded
window when it is still open, so a restart -- a promotion, a crash the
supervisor recovered from -- lands in the window the owner already has, whose
event stream reconnects on its own.

*It can be asked back.*  Two signals, because the thing that needs to ask is a
second process that cannot call into this one:

``<state>/window-show``
    A beacon file.  Anything that can write a file can bring the window back:
    a desktop shortcut, a scheduled task, ``python -m jarvis.window --show``,
    or this core's own duplicate-instance guard.

the supervisor's log
    A second ``ZEUS.exe`` cannot reach the core -- it does not know the token
    and exits before it could learn one -- but it does append one line to
    ``supervisor.log`` on its way out, saying that another supervisor is
    already running.  Watching for that line is what makes double-clicking the
    icon put the window back.  The coupling is to a *phrase in another
    package's log*, which is exactly as fragile as it sounds, so it is pinned
    by a test that reads the supervisor's source: if that line ever changes,
    the test says so instead of the feature quietly dying.

Polling, not a filesystem watcher: two ``stat`` calls every quarter second is
free, and ``ReadDirectoryChangesW`` would be a Windows-only code path with its
own failure modes for a feature whose whole job is not to fail.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

#: Write this file to bring the window back.  Deleted as soon as it is acted on.
BEACON_NAME = "window-show"

#: What a second ``ZEUS.exe`` leaves in the supervisor log.  Pinned by
#: ``tests/test_lifecycle_window.py`` against the supervisor's own source.
RELAUNCH_MARKER = "already running"

#: How often the beacon and the log are checked.  The owner's requirement is a
#: window within two seconds of a second launch; this spends a quarter of that
#: on noticing and leaves the rest to Chromium.
POLL_SECONDS = 0.25


class DesktopWindow:
    """The one window this core shows, and the signals that ask for it back."""

    def __init__(
        self,
        url: str,
        *,
        state_root: str | Path | None = None,
        profile_dir: str | Path | None = None,
        poll_seconds: float = POLL_SECONDS,
        log: Any = None,
    ) -> None:
        from jarvis import window as window_module

        self.window = window_module
        self.url = url
        self.profile_dir = Path(profile_dir) if profile_dir is not None else window_module.default_profile_dir()
        self.state_root = Path(state_root) if state_root is not None else self.profile_dir.parent
        self.poll_seconds = float(poll_seconds)
        self._log = log
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log_offset = -1
        self.shows = 0
        self.last_detail = ""

    # -- paths ---------------------------------------------------------

    @property
    def beacon(self) -> Path:
        return self.state_root / BEACON_NAME

    @property
    def supervisor_log(self) -> Path:
        raw = os.environ.get("ZEUS_SUPERVISOR_DIR", "").strip()
        directory = Path(raw) if raw else self.state_root / "supervisor"
        return directory / "logs" / "supervisor.log"

    # -- what the owner sees -------------------------------------------

    def show(self, reason: str = "") -> dict[str, Any]:
        """Put the window in front of the owner, reusing the open one."""

        with self._lock:
            launch = self.window.ensure_window(self.url, profile_dir=self.profile_dir)
            self.shows += 1
            self.last_detail = launch.detail
        if reason:
            self.note(f"showing the window ({reason}): {launch.detail or launch.mode}")
        return {"ok": launch.ok, "reason": reason, **launch.to_dict()}

    def hide(self, reason: str = "") -> dict[str, Any]:
        """Close the window.  The core, the listener and the worker stay up."""

        with self._lock:
            result = self.window.close_window(profile_dir=self.profile_dir)
        self.note(f"closed the window ({reason or 'requested'}); the core keeps running")
        return {"ok": True, "reason": reason, **result}

    def state(self) -> dict[str, Any]:
        session = self.window.read_session(self.profile_dir)
        return {
            "open": self.window.window_is_open(self.profile_dir),
            "url": self.url,
            "pid": int(session.get("pid") or 0),
            "engine": str(session.get("engine", "")),
            "mode": str(session.get("mode", "")),
            "shows": self.shows,
            "detail": self.last_detail,
            "watching": self._thread is not None and self._thread.is_alive(),
            "beacon": str(self.beacon),
        }

    # -- the watcher ---------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Anything already in the log happened before this core existed; only
        # what arrives from now on is a request.
        self._log_offset = self._log_size()
        self._clear_beacon()
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="jarvis-window")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                reason = self.pending_request()
            except Exception:  # noqa: BLE001 - a watcher that raises is a watcher that stopped
                continue
            if reason:
                try:
                    self.show(reason)
                except Exception as exc:  # noqa: BLE001
                    self.note(f"could not show the window: {type(exc).__name__}: {exc}")

    def pending_request(self) -> str:
        """Why the window should be shown right now, or ``""``.

        Split out from the loop so the two signals can be exercised without a
        thread and without a browser.
        """

        if self.beacon.exists():
            self._clear_beacon()
            return "a beacon was written"
        if self._supervisor_log_says_relaunch():
            return "ZEUS was started again"
        return ""

    def request_show(self, reason: str = "") -> Path:
        """Write the beacon.  For a process that cannot call :meth:`show`."""

        beacon = self.beacon
        try:
            beacon.parent.mkdir(parents=True, exist_ok=True)
            beacon.write_text(reason or "show", encoding="utf-8")
        except OSError:
            pass
        return beacon

    def _clear_beacon(self) -> None:
        try:
            self.beacon.unlink()
        except OSError:
            pass

    def _log_size(self) -> int:
        try:
            return self.supervisor_log.stat().st_size
        except OSError:
            return 0

    def _supervisor_log_says_relaunch(self) -> bool:
        size = self._log_size()
        if self._log_offset < 0:
            self._log_offset = size
            return False
        if size <= self._log_offset:
            # Rotated or truncated: start again from where it is now rather
            # than re-reading a file that is no longer the one we were reading.
            self._log_offset = size
            return False
        try:
            with self.supervisor_log.open("rb") as handle:
                handle.seek(self._log_offset)
                added = handle.read(64_000).decode("utf-8", errors="replace")
        except OSError:
            return False
        self._log_offset = size
        return RELAUNCH_MARKER in added

    # -- plumbing ------------------------------------------------------

    def note(self, message: str) -> None:
        if self._log is not None:
            try:
                self._log(message)
            except Exception:  # noqa: BLE001
                pass


def request_show(state_root: str | Path | None = None, reason: str = "") -> Path:
    """Ask the running core for its window, from a process that is not it.

    Used by the duplicate-instance guard in :mod:`jarvis.serve`: a second core
    that finds the port already served has nothing useful to do except hand the
    owner back the window belonging to the one that is up.
    """

    from jarvis import window as window_module

    root = Path(state_root) if state_root is not None else window_module.default_profile_dir().parent
    beacon = root / BEACON_NAME
    try:
        beacon.parent.mkdir(parents=True, exist_ok=True)
        beacon.write_text(reason or "show", encoding="utf-8")
    except OSError:
        pass
    return beacon


def stop_speech_processes(*, exclude: set[int] | None = None) -> dict[str, Any]:
    """Kill any ``speech.listener`` or ``speech.worker`` still holding hardware.

    The supervisor stops the listener it started, and the core's speech engine
    stops the worker it started, so this is the sweep for the ones neither of
    them owns: a worker orphaned by a core that was killed rather than asked to
    stop, a listener left by a supervisor that did not get to run its cleanup.
    Those processes hold the microphone and a whisper model on the GPU, and a
    second copy of either is exactly the duplicate the owner asked to never see.

    Windows-only, because that is where the machine is and because there is no
    dependency budget for a cross-platform process library.  Returns what it
    did rather than raising: this runs on the way out, and failing to kill a
    stray must not stop the shutdown it is part of.
    """

    import subprocess
    import sys

    result: dict[str, Any] = {"killed": [], "checked": False}
    if sys.platform != "win32":
        return result

    skip = {os.getpid()} | (exclude or set())
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*speech.listener*' -or $_.CommandLine -like '*speech.worker*' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        # Popen rather than run, so the query's own pid is known: the shell
        # command contains the very strings it searches for, so it matches
        # itself and would otherwise be in its own kill list.
        query = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            text=True, creationflags=no_window,
        )
        skip.add(query.pid)
        stdout, _ = query.communicate(timeout=30)
    except (OSError, subprocess.SubprocessError):
        return result

    result["checked"] = True
    for token in (stdout or "").split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid in skip:
            continue
        try:
            killed = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True, timeout=20, creationflags=no_window,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if killed.returncode == 0:
            result["killed"].append(pid)
    return result  # noqa: RET504


def count_speech_processes() -> dict[str, int]:
    """How many listeners and workers are running -- the owner's real check.

    Exposed on ``/api/window`` so "are there duplicate processes?" is a
    question the interface can answer, rather than one that needs a terminal
    and a PowerShell incantation.
    """

    import subprocess
    import sys

    counts = {"listener": 0, "worker": 0, "core": 0}
    if sys.platform != "win32":
        return counts
    script = (
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | "
        "ForEach-Object { $_.CommandLine }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return counts
    for line in completed.stdout.splitlines():
        if "speech.listener" in line:
            counts["listener"] += 1
        elif "speech.worker" in line:
            counts["worker"] += 1
        elif "jarvis.serve" in line:
            counts["core"] += 1
    return counts
