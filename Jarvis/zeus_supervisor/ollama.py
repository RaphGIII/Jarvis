"""Ollama as a managed background service: start it once, prove it answers, never wait on it.

``ollama serve`` is a long-running server, not a command.  The supervisor
therefore never ``wait()``s or ``communicate()``s with it: it spawns it
detached (its own process group, no window, no inherited pipes), keeps the
handle only to read the pid and notice an early death, and learns readiness
from the one source that means anything -- the HTTP API answering
``/api/version`` -- inside a bounded time.

Everything here is idempotent and storm-proof by construction:

* an API that already answers is RUNNING and nothing is spawned;
* an ``ollama`` process that exists but does not answer yet (a slow start,
  the tray app's own server, a previous supervisor's child) is STARTING and
  is polled, not duplicated;
* one spawn at a time (a lock), a cool-down between spawns and a budget per
  window, so a crashing Ollama is retried a few times and then reported
  FAILED with the reason instead of being restarted for ever;
* a missing binary is MISSING; a spawn that raised, a child that exited or an
  API that never came up inside the timeout is FAILED; anything else that is
  simply not there is UNAVAILABLE.

The live defect this replaces: the frozen supervisor crashed in preflight on
a cached check (``AttributeError: _ollama_exe``) whenever Ollama was not
already running, and its crash handler then sat in a modal message box --
which looked like a hang.  The lifecycle below has no path that blocks
longer than its timeout, and no path that raises out.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable
from urllib.parse import urlparse


class OllamaState(str, Enum):
    RUNNING = "RUNNING"          # the API answers
    STARTING = "STARTING"        # a server process exists; the API does not answer yet
    UNAVAILABLE = "UNAVAILABLE"  # nothing is running and nothing was (or could be) started
    FAILED = "FAILED"            # a start was attempted and did not lead to a running server
    MISSING = "MISSING"          # no ollama binary on this machine


@dataclass
class OllamaStatus:
    state: OllamaState
    reason: str = ""
    version: str = ""
    pid: int = 0
    started_by_supervisor: bool = False
    spawns: int = 0
    url: str = ""
    exe: str = ""
    models_dir: str = ""
    checked_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.state is OllamaState.RUNNING

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["state"] = self.state.value
        out["ok"] = self.ok
        return out


def _http_version(url: str, timeout: float = 3.0) -> str:
    request = urllib.request.Request(url.rstrip("/") + "/api/version", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("version", "?"))


def _port_open(url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _server_process_exists() -> bool:
    """Whether an ``ollama`` server process is present on this machine (cheap, no PowerShell)."""

    try:
        if sys.platform == "win32":
            completed = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/NH", "/FO", "CSV"],
                                       capture_output=True, text=True, timeout=10,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return "ollama.exe" in completed.stdout.lower()
        completed = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True, timeout=10)
        return completed.returncode == 0 and bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _detached_flags() -> int:
    if sys.platform != "win32":
        return 0
    return (getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0))


def _default_spawner(command: list[str], env: dict[str, str]) -> Any:
    kwargs: dict[str, Any] = dict(env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if sys.platform == "win32":
        kwargs["creationflags"] = _detached_flags()
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


class OllamaService:
    """Owns the lifecycle of the local Ollama server for one supervisor."""

    def __init__(
        self,
        *,
        url: str,
        exe_finder: Callable[[], str],
        models_dir: str = "",
        start_timeout: float = 45.0,
        spawn_cooldown: float = 30.0,
        max_spawns: int = 3,
        spawn_window: float = 600.0,
        probe: Callable[[str, float], str] = _http_version,
        port_probe: Callable[[str], bool] = _port_open,
        process_probe: Callable[[], bool] = _server_process_exists,
        spawner: Callable[[list[str], dict[str, str]], Any] = _default_spawner,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.url = url
        self.exe_finder = exe_finder
        self.models_dir = models_dir
        self.start_timeout = float(start_timeout)
        self.spawn_cooldown = float(spawn_cooldown)
        self.max_spawns = int(max_spawns)
        self.spawn_window = float(spawn_window)
        self._probe = probe
        self._port_probe = port_probe
        self._process_probe = process_probe
        self._spawner = spawner
        self._clock = clock
        self._sleep = sleep
        self._log = log or (lambda _m: None)
        self._lock = threading.Lock()
        self._process: Any = None
        self._spawn_times: list[float] = []
        self._last_failure = ""
        self.last = OllamaStatus(OllamaState.UNAVAILABLE, "not checked yet", url=url)

    # -- observation (never starts anything) -----------------------------

    def version(self, timeout: float = 3.0) -> str:
        try:
            return self._probe(self.url, timeout)
        except Exception:  # noqa: BLE001 - not answering is the information
            return ""

    def own_process_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _spawns_in_window(self) -> list[float]:
        cutoff = self._clock() - self.spawn_window
        self._spawn_times = [t for t in self._spawn_times if t >= cutoff]
        return self._spawn_times

    def status(self) -> OllamaStatus:
        """What is true right now.  Cheap; no side effects; never raises."""

        exe = ""
        try:
            exe = str(self.exe_finder() or "")
        except Exception:  # noqa: BLE001
            exe = ""
        base = dict(url=self.url, exe=exe, models_dir=self.models_dir, spawns=len(self._spawns_in_window()),
                    started_by_supervisor=self._process is not None, pid=getattr(self._process, "pid", 0) or 0,
                    checked_at=time.time())
        version = self.version()
        if version:
            self.last = OllamaStatus(OllamaState.RUNNING, "the API answers", version=version, **base)
        elif self.own_process_alive():
            self.last = OllamaStatus(OllamaState.STARTING, f"ollama serve (pid {self._process.pid}) is up; the API on {self.url} does not answer yet", **base)
        elif self._port_probe(self.url) or self._process_probe():
            self.last = OllamaStatus(OllamaState.STARTING, f"an ollama server process exists but the API on {self.url} does not answer", **base)
        elif not exe:
            self.last = OllamaStatus(OllamaState.MISSING, "ollama is not installed (no binary found)", **base)
        elif self._last_failure:
            self.last = OllamaStatus(OllamaState.FAILED, self._last_failure, **base)
        else:
            self.last = OllamaStatus(OllamaState.UNAVAILABLE, f"nothing answers on {self.url} and no server process exists", **base)
        return self.last

    # -- the one action ----------------------------------------------------

    def ensure(self, *, timeout: float | None = None) -> OllamaStatus:
        """Make Ollama answer if it can be made to, within ``timeout`` seconds.

        Idempotent: a running server is left alone; a starting one is
        waited for; only a genuinely absent one is spawned, at most once per
        cool-down and ``max_spawns`` per window.  Returns the resulting
        status; never raises; never blocks longer than the timeout plus one
        probe.
        """

        budget = self.start_timeout if timeout is None else float(timeout)
        started = self._clock()
        with self._lock:
            current = self.status()
            if current.ok:
                return current
            if current.state is OllamaState.MISSING:
                return current
            spawned_now = False
            if current.state is not OllamaState.STARTING:
                spawned_now = self._spawn(current.exe)
                if spawned_now is None:
                    # spawn refused or failed; self._last_failure explains
                    return self.status()
            # poll readiness, bounded
            deadline = started + budget
            while self._clock() < deadline:
                version = self.version()
                if version:
                    status = self.status()
                    self._last_failure = ""
                    self._log(f"ollama {status.state.value}: version {version}"
                              + (f", started by the supervisor (pid {status.pid})" if spawned_now else ", already running"))
                    return status
                if self._process is not None and self._process.poll() is not None:
                    code = self._process.returncode
                    self._process = None
                    self._last_failure = f"ollama serve exited with code {code} before its API answered"
                    self._log(self._last_failure)
                    return self.status()
                self._sleep(0.5)
            waited = self._clock() - started
            self._last_failure = (f"ollama serve did not answer on {self.url} within {waited:.0f}s"
                                  + (f" (pid {self._process.pid} is still running)" if self.own_process_alive() else ""))
            self._log(self._last_failure)
            status = self.status()
            if status.state is OllamaState.STARTING:
                # still coming up, but the boot could not wait any longer: that is a failure of this attempt
                status = OllamaStatus(OllamaState.FAILED, self._last_failure, **{k: v for k, v in status.to_dict().items()
                                                                                 if k not in {"state", "reason", "ok"}})
                self.last = status
            return status

    def _spawn(self, exe: str) -> bool | None:
        """Start ``ollama serve`` detached.  True = spawned, None = refused/failed (reason recorded)."""

        if not exe:
            self._last_failure = "ollama is not installed (no binary found)"
            return None
        recent = self._spawns_in_window()
        if len(recent) >= self.max_spawns:
            self._last_failure = (f"ollama serve was started {len(recent)} times in the last {self.spawn_window / 60:.0f} min and never stayed up; "
                                  "not starting it again -- run `ollama serve` in a terminal and read its error")
            return None
        if recent and self._clock() - recent[-1] < self.spawn_cooldown:
            self._last_failure = f"ollama serve was started {self._clock() - recent[-1]:.0f}s ago; waiting before another attempt"
            return None
        env = dict(os.environ)
        if self.models_dir:
            env["OLLAMA_MODELS"] = self.models_dir
        try:
            self._process = self._spawner([exe, "serve"], env)
        except Exception as exc:  # noqa: BLE001 - OSError, or a broken launcher
            self._process = None
            self._last_failure = f"could not start ollama serve: {type(exc).__name__}: {exc}"
            self._log(self._last_failure)
            return None
        self._spawn_times.append(self._clock())
        self._log(f"started ollama serve (pid {getattr(self._process, 'pid', '?')}, models from {self.models_dir or 'its default store'})")
        return True

    def to_dict(self) -> dict[str, Any]:
        return self.last.to_dict()
