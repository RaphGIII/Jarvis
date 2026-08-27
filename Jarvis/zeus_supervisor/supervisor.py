"""The loop: start ZEUS, prove it is up, keep it up, undo what breaks it.

    boot
      preflight ------- fails -> HOLD with the diagnosis on the status page
      launch core
      wait for READY -- fails -> if HEAD != known-good: roll back and retry
                                 else: count a failure; HOLD after three
      READY ---------- HEAD becomes known-good (only here, only ever here)
      watch the child
        exit 75 -> a promotion asks to be tried: launch again, verify as above
        exit 0  -> the owner asked for a shutdown: stop everything
        crash   -> restart with backoff; a boot loop ends in a rollback or a HOLD

Every transition writes a deployment receipt and the status file the
application shows the owner.  Nothing here is inferred from a port being open:
READY is the core's own health report, which requires a real generation.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import EXIT_RESTART_REQUESTED, EXIT_SHUTDOWN_REQUESTED, __version__
from .config import SupervisorConfig
from .control import ControlChannel, ControlRequest
from .instance import InstanceLock
from .known_good import DeploymentReceipt, KnownGoodStore
from .preflight import Preflight, PreflightReport, _no_window, git
from .status_page import StatusPage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Supervisor:
    def __init__(self, config: SupervisorConfig, *, log=None) -> None:
        self.config = config
        self.state_dir = config.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.state_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._logfile = (self.logs_dir / "supervisor.log").open("a", encoding="utf-8")
        self._log_lines: list[str] = []
        self._external_log = log
        self.known_good = KnownGoodStore(self.state_dir)
        self.control = ControlChannel(self.state_dir)
        self.lock = InstanceLock(self.state_dir)
        self.status_page = StatusPage(config.host, config.port, self._status_snapshot)
        self.token = self._load_token()
        self.core: subprocess.Popen | None = None
        self.listener: subprocess.Popen | None = None
        self.phase = "starting"
        self.detail = ""
        self.remedy = ""
        self.revision = ""
        self.failures: list[float] = []
        self.preflight_report: PreflightReport | None = None
        self.last_health: dict[str, Any] = {}
        self._stop = threading.Event()
        self._browser_opened = False

    # -- plumbing ------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"{_now()} {message}"
        self._log_lines.append(line)
        del self._log_lines[:-200]
        try:
            self._logfile.write(line + "\n")
            self._logfile.flush()
        except OSError:
            pass
        if self._external_log:
            self._external_log(message)

    def _set(self, phase: str, detail: str = "", remedy: str = "") -> None:
        self.phase, self.detail, self.remedy = phase, detail, remedy
        self.log(f"[{phase}] {detail}" + (f" -- {remedy}" if remedy else ""))
        self._write_status()

    def _status_snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "detail": self.detail,
            "remedy": self.remedy,
            "revision": self.revision,
            "known_good": self.known_good.load().to_dict(),
            "log": list(self._log_lines[-20:]),
        }

    def _write_status(self) -> None:
        kg = self.known_good.load()
        self.control.write_status({
            "supervisor_version": __version__,
            "phase": self.phase,
            "detail": self.detail,
            "remedy": self.remedy,
            "revision": self.revision,
            "known_good": kg.revision,
            "known_good_verified_at": kg.verified_at,
            "core_pid": self.core.pid if self.core and self.core.poll() is None else 0,
            "listener_pid": self.listener.pid if self.listener and self.listener.poll() is None else 0,
            "failures_in_window": len(self._recent_failures()),
            "frozen": bool(getattr(sys, "frozen", False)),
            "url": self.url,
        })

    def _load_token(self) -> str:
        path = self.state_dir / "token"
        try:
            token = path.read_text(encoding="utf-8").strip()
            if len(token) >= 16:
                return token
        except OSError:
            pass
        token = secrets.token_urlsafe(24)
        path.write_text(token, encoding="utf-8")
        return token

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/?token={self.token}"

    def _api(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
        url = f"http://{self.config.host}:{self.config.port}{path}"
        data = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers={"X-Jarvis-Token": self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _head(self) -> str:
        completed = git(self.config.repository, "rev-parse", "HEAD")
        return completed.stdout.strip() if completed.returncode == 0 else ""

    # -- lifecycle -----------------------------------------------------

    def run(self) -> int:
        if not self.lock.acquire():
            return self._signal_running_instance()
        try:
            self._install_signal_handlers()
            return self._main_loop()
        finally:
            self.shutdown_children()
            self.status_page.stop()
            self.lock.release()

    def _signal_running_instance(self) -> int:
        """A second ZEUS.exe: bring the running one's window back, do not start another.

        The running core is asked over its own API (the token is in the state
        directory both share).  While it is still booting and the API is not
        there yet, a beacon file is left where the core's window watcher
        looks the moment it is up.  Either way exactly one runtime exists.
        """

        started = time.monotonic()
        try:
            answer = self._api("/api/window/show", {"reason": "second ZEUS.exe invocation"}, timeout=8)
            self.log(f"another ZEUS is running; its window was {answer.get('action', 'shown')} in "
                     f"{answer.get('seconds', '?')}s ({time.monotonic() - started:.2f}s end to end)")
            return 0
        except Exception as exc:  # noqa: BLE001 - not up yet, or not ours
            self.log(f"another ZEUS supervisor is running and its API did not answer ({exc}); leaving a beacon")
        try:
            beacon = self.state_dir / "control" / "window-show"
            beacon.parent.mkdir(parents=True, exist_ok=True)
            beacon.write_text(str(time.time()), encoding="utf-8")
        except OSError as exc:
            self.log(f"could not leave the beacon: {exc}")
            if self.config.open_browser:
                webbrowser.open(self.url)
            return 3
        return 0

    def _relaunch(self, request: ControlRequest) -> None:
        exe = Path(request.exe) if request.exe else Path(sys.executable if getattr(sys, "frozen", False) else "")
        watchdog = [
            self.config.python, "-m", "zeus_supervisor.relaunch",
            "--wait-pid", str(os.getpid()), "--exe", str(exe), "--previous", request.previous,
            "--state", str(self.state_dir), "--port", str(self.config.port), "--timeout", str(int(self.config.ready_timeout)),
            "--promotion", request.promotion_id,
        ]
        flags = _no_window()
        if sys.platform == "win32":
            flags |= subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        try:
            subprocess.Popen(watchdog, cwd=str(self.config.repository), env=self._child_env(), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
            self._set("relaunching", f"handing over to {exe} ({request.reason})")
            self.known_good.record(DeploymentReceipt(
                kind="relaunch", revision=self.revision, outcome="handed_over", reason=request.reason,
                known_good_before=self.known_good.load().revision,
            ))
        except OSError as exc:
            self._set("error", f"could not start the relaunch watchdog: {exc}", "Start ZEUS.exe by hand")

    def _sweep_processes(self, patterns: tuple[str, ...], *, keep: tuple[int, ...] = ()) -> int:
        """End every process whose command line matches, except ``keep``."""

        if sys.platform != "win32":
            return 0
        clauses = " -or ".join(f"$_.CommandLine -like '*{p}*'" for p in patterns)
        script = (f"Get-CimInstance Win32_Process | Where-Object {{ ({clauses}) -and $_.Name -ne 'powershell.exe' }} "
                  f"| ForEach-Object {{ $_.ProcessId }}")
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                                       timeout=30, creationflags=_no_window())
        except (OSError, subprocess.SubprocessError):
            return 0
        killed = 0
        for line in completed.stdout.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid and pid not in keep and pid != os.getpid():
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, creationflags=_no_window())
                killed += 1
        return killed

    def _install_signal_handlers(self) -> None:
        def handler(*_: Any) -> None:
            self._stop.set()

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def _open_window_early(self) -> None:
        """The window at T0, onto the status page; the core takes it over.

        The status page refreshes itself every three seconds and is replaced
        by the interface the moment the core binds the port, and the core's
        window owner finds this window by its title instead of opening a
        second one.  The owner therefore sees ZEUS within about a second of
        the launch, with the model still loading behind it.
        """

        if self.config.open_browser or os.environ.get("ZEUS_UI", "").strip().lower() in {"none", "off", "0", "false", "headless", "browser"}:
            return
        try:
            repo = str(self.config.repository)
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from jarvis.window import open_window

            launch = open_window(self.url, fallback=False)
            self.log(f"window opened at launch: {launch.describe()}" if launch.ok else f"window not opened at launch: {launch.detail}")
        except Exception as exc:  # noqa: BLE001 - the window is never a reason not to boot
            self.log(f"window not opened at launch: {exc}")

    def _main_loop(self) -> int:
        self.status_page.start()
        self._open_window_early()
        self._set("preflight", "checking Python, repository, Ollama and the models")
        if self.config.open_browser and not self._browser_opened:
            # Opened now, at the status page, so the owner sees progress
            # rather than a blank tab; the page hands over to the real UI
            # by itself.
            self._browser_opened = True
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

        from .preflight import PreflightCache

        # No generation here: the core's READY is a real generation in the
        # process that will answer, and waiting for a second one first only
        # delayed the window.  Stable checks come from the fingerprint cache.
        report = Preflight(self.config, log=self.log, cache=PreflightCache(self.state_dir / "preflight_cache.json")).run(
            generation=bool(self.config.preflight_generation)
        )
        self.preflight_report = report
        self.revision = report.revision
        if not report.ok:
            blocker = report.blocker
            self._set("error", f"{blocker.name}: {blocker.detail}", blocker.remedy)
            self._hold()
            return 2

        pending: ControlRequest | None = None
        while not self._stop.is_set():
            outcome = self._start_and_verify(pending)
            pending = None
            if outcome == "held":
                self._hold()
                return 2
            if outcome == "stopped":
                return 0
            # outcome == "healthy": watch the child until it exits
            code = self._watch()
            if self._stop.is_set():
                self._set("stopping", "shutting down")
                return 0
            request = self.control.take()
            if code == EXIT_SHUTDOWN_REQUESTED and (request is None or request.action == "shutdown"):
                self._set("stopped", "ZEUS asked to be shut down" + (f": {request.reason}" if request else ""))
                return 0
            if request is not None and request.action == "shutdown":
                self._set("stopped", f"shutdown requested: {request.reason}")
                return 0
            if request is not None and request.action == "relaunch":
                # A promoted executable: this supervisor steps aside and a
                # watchdog starts the new one once the instance lock is free,
                # restoring the previous release if it never gets READY.
                self._relaunch(request)
                return 0
            if code == EXIT_RESTART_REQUESTED or (request is not None and request.action == "restart"):
                pending = request or ControlRequest(action="restart", reason="core exited with the restart code")
                self._set("restarting", f"restart requested: {pending.reason}")
                continue
            # Anything else is a crash.
            self.failures.append(time.monotonic())
            self.known_good.record(DeploymentReceipt(
                kind="restart", revision=self.revision, outcome="crashed", reason=f"exit code {code}",
                known_good_before=self.known_good.load().revision,
            ))
            self._set("restarting", f"ZEUS exited with code {code}; restarting")
            time.sleep(min(2.0 * len(self._recent_failures()), 10.0))
        return 0

    def _recent_failures(self) -> list[float]:
        cutoff = time.monotonic() - self.config.failure_window
        self.failures = [t for t in self.failures if t >= cutoff]
        return self.failures

    def _start_and_verify(self, request: ControlRequest | None) -> str:
        """Launch the core and wait for READY; roll back or hold on failure.

        Returns ``healthy``, ``held`` or ``stopped``.
        """

        while not self._stop.is_set():
            if len(self._recent_failures()) >= self.config.max_failures:
                self._set("held", f"{self.config.max_failures} failed starts in {self.config.failure_window:.0f}s; not restarting again",
                          "Read data/jarvis/supervisor/logs/core.log, fix the cause, then start ZEUS again")
                self.known_good.record(DeploymentReceipt(
                    kind="hold", revision=self.revision, outcome="held", reason="boot loop",
                    known_good_before=self.known_good.load().revision,
                ))
                return "held"

            self.revision = self._head()
            before = self.known_good.load()
            started = time.monotonic()
            self._set("starting", f"starting ZEUS at {self.revision[:12]}")
            self.status_page.stop()
            try:
                self._launch_core()
            except OSError as exc:
                self.failures.append(time.monotonic())
                self._set("error", f"could not launch ZEUS: {exc}")
                self.status_page.start()
                continue

            health = self._wait_ready()
            self.last_health = health
            elapsed = time.monotonic() - started
            kind = "promotion" if request and request.promotion_id else ("restart" if request else "start")

            if health.get("ready"):
                after = self.known_good.mark(self.revision, health)
                self.known_good.record(DeploymentReceipt(
                    kind=kind, revision=self.revision, outcome="healthy", reason=request.reason if request else "boot",
                    known_good_before=before.revision, known_good_after=after.revision,
                    promotion_id=request.promotion_id if request else "", health=health, duration_seconds=round(elapsed, 1),
                ))
                self.failures.clear()
                self._set("ready", f"ZEUS is ready at {self.revision[:12]} ({elapsed:.0f}s)")
                self._launch_listener()
                return "healthy"

            # Not ready. Stop the child, decide between rollback and retry.
            reason = str(health.get("reason") or health.get("error") or "no READY within the timeout")
            self.log(f"health check failed: {reason}")
            self.shutdown_children(core_only=True)
            self.status_page.start()
            self.failures.append(time.monotonic())

            if before.revision and before.revision != self.revision:
                self._set("rolling back", f"{self.revision[:12]} failed its health check ({reason}); returning to {before.revision[:12]}")
                ok, detail = self._rollback(before.revision)
                self.known_good.record(DeploymentReceipt(
                    kind="rollback", revision=self.revision, outcome="rolled_back" if ok else "rollback_failed",
                    reason=f"{reason}; {detail}", known_good_before=before.revision, known_good_after=before.revision,
                    promotion_id=request.promotion_id if request else "", health=health, duration_seconds=round(elapsed, 1),
                ))
                if not ok:
                    self._set("held", f"rollback to {before.revision[:12]} failed: {detail}",
                              "The repository needs a human: git status, then git reset --hard <known good>")
                    return "held"
                # The rollback is one deliberate move, not a failure to count.
                self.failures.pop()
                request = ControlRequest(action="restart", reason=f"rollback after failed promotion: {reason}")
                continue

            self.known_good.record(DeploymentReceipt(
                kind=kind, revision=self.revision, outcome="unhealthy", reason=reason,
                known_good_before=before.revision, health=health, duration_seconds=round(elapsed, 1),
            ))
            self._set("restarting", f"ZEUS did not become ready ({reason}); retrying", str(health.get("remedy", "")))
            time.sleep(3.0)
        return "stopped"

    # -- children ------------------------------------------------------

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["ZEUS_SUPERVISED"] = "1"
        env["ZEUS_SUPERVISOR_DIR"] = str(self.state_dir)
        env["ZEUS_SUPERVISOR_PID"] = str(os.getpid())
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.config.ollama_models_dir:
            env["OLLAMA_MODELS"] = self.config.ollama_models_dir
        return env

    def _launch_core(self) -> None:
        log_path = self.logs_dir / "core.log"
        self._rotate(log_path)
        token_file = self.state_dir / "token"
        command = [
            self.config.python, "-m", "jarvis.serve",
            "--host", self.config.host, "--port", str(self.config.port),
            "--token-file", str(token_file), "--no-browser",
        ]
        handle = log_path.open("ab")
        self.core = subprocess.Popen(
            command, cwd=str(self.config.repository), env=self._child_env(),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=_no_window() | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        self.log(f"launched core pid {self.core.pid}: {' '.join(command)}")

    def _kill_stale_listeners(self) -> int:
        """Listeners from an earlier supervisor hold the microphone; two of
        them answer every wake twice.  Only ours may run."""

        if sys.platform != "win32":
            return 0
        mine = self.listener.pid if self.listener is not None and self.listener.poll() is None else 0
        script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*speech.listener*' "
                  "-and $_.Name -ne 'powershell.exe' } | ForEach-Object { $_.ProcessId }")
        try:
            completed = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                                       timeout=30, creationflags=_no_window())
        except (OSError, subprocess.SubprocessError):
            return 0
        killed = 0
        for line in completed.stdout.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid and pid != mine:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, creationflags=_no_window())
                killed += 1
        if killed:
            self.log(f"killed {killed} stale listener process(es)")
        return killed

    def _launch_listener(self) -> None:
        if not self.config.voice:
            return
        python = self.config.speech_python
        if python is None:
            return
        if self.listener is not None and self.listener.poll() is None:
            return
        self._kill_stale_listeners()
        log_path = self.logs_dir / "listener.log"
        self._rotate(log_path)
        command = [str(python), "-m", "speech.listener", "--url", f"http://{self.config.host}:{self.config.port}",
                   "--token", self.token]
        try:
            self.listener = subprocess.Popen(
                command, cwd=str(self.config.repository), env=self._child_env(),
                stdout=log_path.open("ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=_no_window() | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            self.log(f"launched listener pid {self.listener.pid}")
        except OSError as exc:
            self.log(f"listener did not start: {exc}")

    @staticmethod
    def _rotate(path: Path, keep: int = 3) -> None:
        if not path.is_file() or path.stat().st_size < 2_000_000:
            return
        for index in range(keep, 0, -1):
            older = path.with_suffix(f".{index}.log")
            newer = path if index == 1 else path.with_suffix(f".{index - 1}.log")
            if newer.is_file():
                try:
                    older.unlink(missing_ok=True)
                    newer.rename(older)
                except OSError:
                    pass

    def _wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.ready_timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.core is not None and self.core.poll() is not None:
                return {"ready": False, "reason": f"process exited with code {self.core.returncode} before READY",
                        "remedy": "see data/jarvis/supervisor/logs/core.log", **last}
            try:
                last = self._api("/api/health", timeout=5)
            except Exception:
                last = {}
            if last.get("ready"):
                if last.get("revision") and self.revision and last["revision"] != self.revision:
                    return {"ready": False, "reason": f"core reports revision {last['revision'][:12]} but {self.revision[:12]} was launched", **last}
                return last
            stage = last.get("detail") or "waiting for the core"
            if self.detail != stage:
                self._set("starting", str(stage))
            time.sleep(1.0)
        return {"ready": False, "reason": f"not READY after {self.config.ready_timeout:.0f}s", **last}

    def _watch(self) -> int:
        """Block until the core exits; keep the listener alive meanwhile."""

        assert self.core is not None
        last_listener_start = time.monotonic()
        while not self._stop.is_set():
            code = self.core.poll()
            if code is not None:
                self.log(f"core exited with code {code}")
                return int(code)
            if self.listener is not None and self.listener.poll() is not None and time.monotonic() - last_listener_start > 15:
                self.log(f"listener exited with code {self.listener.returncode}; restarting it")
                last_listener_start = time.monotonic()
                self._launch_listener()
            time.sleep(0.5)
        return EXIT_SHUTDOWN_REQUESTED

    def shutdown_children(self, *, core_only: bool = False) -> None:
        if self.core is not None and self.core.poll() is None:
            try:
                self._api("/api/shutdown", {"reason": "supervisor stopping"}, timeout=5)
            except Exception:
                pass
            try:
                self.core.wait(timeout=self.config.stop_timeout)
            except subprocess.TimeoutExpired:
                self.log("core did not stop in time; terminating")
                self.core.kill()
                try:
                    self.core.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if core_only:
            return
        if self.listener is not None and self.listener.poll() is None:
            self.listener.kill()
            try:
                self.listener.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # Nothing of ours survives a full stop: listeners and speech workers
        # from this or any earlier runtime, and the window.
        swept = self._sweep_processes(("speech.listener", "speech.worker"))
        if swept:
            self.log(f"swept {swept} speech process(es) on shutdown")

    # -- rollback ------------------------------------------------------

    def _rollback(self, known_good: str) -> tuple[bool, str]:
        """Return the repository to the known-good revision, keeping history.

        A revert keeps the bad commit in the log with a commit that undoes it,
        which is what a deployment receipt should point at.  Uncommitted work
        is stashed, never discarded: the owner may be mid-edit with other
        tools.  Only if the revert cannot apply does this fall back to a
        reset, and the abandoned commit is named in the receipt so it remains
        recoverable from the reflog.
        """

        repo = self.config.repository
        notes: list[str] = []
        status = git(repo, "status", "--porcelain")
        if status.stdout.strip():
            # Runtime state under data/ is never source and includes this
            # supervisor's own open log, so it is kept out of the stash.
            stash = git(repo, "stash", "push", "-u", "-m", f"zeus-supervisor rollback {_now()}",
                        "--", ".", ":(exclude)data")
            notes.append("stashed uncommitted changes" if stash.returncode == 0 else f"stash failed: {stash.stderr.strip()[:200]}")
        revert = git(repo, "-c", "commit.gpgsign=false", "revert", "--no-edit", f"{known_good}..HEAD", timeout=120)
        if revert.returncode == 0:
            notes.append(f"reverted to the tree of {known_good[:12]}")
        else:
            git(repo, "revert", "--abort")
            abandoned = self._head()
            reset = git(repo, "reset", "--hard", known_good)
            if reset.returncode != 0:
                return False, "; ".join(notes + [f"revert and reset both failed: {reset.stderr.strip()[:200]}"])
            notes.append(f"revert did not apply; reset --hard to {known_good[:12]} (abandoned {abandoned[:12]}, in the reflog)")
        diff = git(repo, "diff", "--stat", known_good, "--")
        if diff.stdout.strip():
            return False, "; ".join(notes + ["tree still differs from known-good"])
        return True, "; ".join(notes)

    # -- hold ----------------------------------------------------------

    def _hold(self) -> None:
        """Stay up serving the diagnosis until told to stop."""

        self.status_page.start()
        self.log("holding; the status page shows the diagnosis")
        while not self._stop.is_set():
            time.sleep(1.0)
