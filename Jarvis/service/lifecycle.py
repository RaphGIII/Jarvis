"""Readiness, restart and shutdown: the core's side of the supervisor contract.

Three things the supervisor needs from the application, kept together because
they are one conversation:

*Am I ready?*  Not "did a port open" -- the conversation model has produced
real text in this process.  :meth:`Lifecycle.health` is what ``/api/health``
returns and what the supervisor waits for before it marks a revision known-good.

*Restart me / shut me down.*  A promotion that wants to be tried, or an owner
asking ZEUS to stop, ends with the process exiting on purpose.  The request is
written to the control channel first, then the conversation is saved, then the
process exits with the agreed code, so the supervisor learns what happened
even though the process that knew is gone.

*Pick up where I left off.*  The transcript is saved on a planned exit and
restored on the next start if it is recent, so "Zeus, update yourself" does
not end with a blank screen and no memory of having been asked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from service.events import EventType

#: A saved conversation older than this is history, not a session to resume.
RESUME_MAX_AGE_SECONDS = 30 * 60


class Lifecycle:
    def __init__(self, core: Any) -> None:
        self.core = core
        self.started = time.time()
        self.exit_event = threading.Event()
        self.exit_code = 0
        self.exit_reason = ""
        self._stages: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._revision = ""
        self.resumed: dict[str, Any] = {}
        #: The managed desktop window, once ``attach_desktop`` ran (serve.py).
        self.desktop: Any = None

    # -- readiness -----------------------------------------------------

    def mark(self, stage: str, ok: bool | None, detail: str = "") -> None:
        with self._lock:
            self._stages[stage] = {"ok": ok, "detail": detail, "at": time.time()}

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._stages.items()}

    @property
    def ready(self) -> bool:
        stage = self._stages.get("fast_local")
        return bool(stage and stage.get("ok"))

    def health(self) -> dict[str, Any]:
        """READY means a real answer came out of the conversation model here."""

        stages = self.stages
        fast = stages.get("fast_local")
        if fast is None:
            detail = "loading the conversation model"
        elif fast.get("ok"):
            detail = "ready"
        else:
            detail = f"conversation model unavailable: {fast.get('detail', '')}"
        voice = stages.get("voice", {}).get("ok")
        recogniser = stages.get("recogniser", {}).get("ok")
        return {
            "ready": self.ready,
            "detail": detail,
            "revision": self.revision(),
            "supervised": self.supervised,
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - self.started, 1),
            "stages": stages,
            "voice": voice,
            "recogniser": recogniser,
            "state": self.core.state.snapshot.to_dict(),
            "resumed": self.resumed,
            "window": self.desktop.status() if self.desktop is not None else None,
            "readiness": self.readiness(),
        }

    def readiness(self) -> dict[str, Any]:
        """The separate readiness levels, each earned by its own evidence."""

        stages = self.stages
        ui = bool(stages.get("http", {}).get("ok"))
        ai = bool(stages.get("fast_local", {}).get("ok"))
        voice = bool(stages.get("voice", {}).get("ok")) and bool(stages.get("recogniser", {}).get("ok"))
        # INTERACTIVE = the owner can genuinely use the product: core answering
        # and the conversation model warm.  Voice keeps warming behind the main
        # UI with an honest indicator — waiting the extra ~40 s of Whisper load
        # before showing anything made the whole product feel broken.
        return {"UI_READY": ui, "CORE_READY": ui, "AI_READY": ai, "VOICE_READY": voice,
                "INTERACTIVE_READY": ui and ai,
                "FULL_READY": ui and ai and voice}

    # -- the desktop window ------------------------------------------

    def attach_desktop(self, url: str) -> Any:
        """Own the window.  Returns the DesktopWindow, or None where there is no engine."""

        try:
            from service.desktop import DesktopWindow
        except Exception:  # noqa: BLE001
            return None
        root = Path(__file__).resolve().parents[1]
        icon = root / "ui" / "zeus.ico"
        title = str(getattr(self.core.identity, "product_name", "") or "ZEUS")
        desktop = DesktopWindow(
            url=url, title=title, state_root=Path(self.core.kernel.state_root), icon=icon if icon.is_file() else None,
            emit=lambda kind, payload: self.core.emit(EventType(kind), payload),
        )
        if not desktop.status().get("available"):
            return None
        self.desktop = desktop
        return desktop

    def window(self, action: str = "status", *, reason: str = "") -> dict[str, Any]:
        """``show`` / ``hide`` / ``close`` / ``status`` for the desktop window."""

        if self.desktop is None:
            return {"ok": False, "error": "no desktop window is managed by this core", "action": action}
        if action == "show":
            return self.desktop.show(reason=reason or "api")
        if action == "hide":
            return self.desktop.hide(reason=reason or "api")
        if action == "minimize":
            return self.desktop.minimize(reason=reason or "api")
        if action == "close":
            return self.desktop.close(reason=reason or "api")
        return {"ok": True, "action": "status", **self.desktop.status()}

    def process_counts(self) -> dict[str, Any]:
        """Real counts from the process table: core, listener, worker, supervisor, window."""

        from service.processes import zeus_processes

        by_role = zeus_processes()
        out: dict[str, Any] = {role: [p.to_dict() for p in rows] for role, rows in by_role.items()}
        out["counts"] = {role: len(rows) for role, rows in by_role.items()}
        out["counts"]["window"] = int(self.desktop.status().get("windows", 0)) if self.desktop is not None else 0
        out["pid"] = os.getpid()
        return out

    def request_quit(self, reason: str = "owner asked ZEUS to quit completely", *, requested_by: str = "core") -> dict[str, Any]:
        """"ZEUS vollständig beenden": window, speech, listener, core, supervisor.

        Everything this process can end is ended here, before the exit code
        tells the supervisor to end the rest and itself.
        """

        report: dict[str, Any] = {"reason": reason}
        try:
            report["window"] = self.desktop.close(reason=reason) if self.desktop is not None else {"ok": True, "action": "absent"}
        except Exception as exc:  # noqa: BLE001
            report["window"] = {"ok": False, "error": str(exc)}
        report["speech"] = self._close_speech()
        try:
            from service.processes import kill_role

            report["listeners_killed"] = kill_role("listener")
            report["workers_killed"] = kill_role("worker")
        except Exception as exc:  # noqa: BLE001
            report["processes_error"] = str(exc)
        report.update(self.request_shutdown(reason, requested_by=requested_by))
        report["quit"] = True
        return report

    def _close_speech(self) -> dict[str, Any]:
        voice = getattr(self.core, "_voice", None)
        engine = getattr(voice, "_engine", None) if voice is not None else None
        if engine is None:
            return {"ok": True, "closed": False}
        try:
            engine.close()
            return {"ok": True, "closed": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def leave(self, *, final: bool) -> dict[str, Any]:
        """Called by serve.py on the way out.  ``final`` is a shutdown (the
        window and the worker go); a restart keeps the window for the next
        core to find."""

        report: dict[str, Any] = {"final": final}
        if self.desktop is not None:
            try:
                self.desktop.stop_watcher()
                if final:
                    report["window"] = self.desktop.close(reason=self.exit_reason or "shutdown")
            except Exception as exc:  # noqa: BLE001
                report["window_error"] = str(exc)
        report["speech"] = self._close_speech()
        return report

    def revision(self) -> str:
        if self._revision:
            return self._revision
        try:
            root = Path(self.core.kernel.state_root).resolve().parents[1]
        except Exception:
            root = Path(__file__).resolve().parent.parent
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
            )
            self._revision = completed.stdout.strip() if completed.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            self._revision = ""
        return self._revision

    # -- supervisor contract ------------------------------------------

    @property
    def supervised(self) -> bool:
        return os.environ.get("ZEUS_SUPERVISED", "") == "1"

    def _control(self) -> Any:
        from zeus_supervisor.control import ControlChannel

        raw = os.environ.get("ZEUS_SUPERVISOR_DIR", "").strip()
        state_dir = Path(raw) if raw else Path(self.core.kernel.state_root) / "supervisor"
        return ControlChannel(state_dir)

    def supervisor_status(self) -> dict[str, Any]:
        try:
            control = self._control()
            from zeus_supervisor.known_good import KnownGoodStore

            store = KnownGoodStore(control.state_dir)
            return {
                "supervised": self.supervised,
                "status": control.read_status(),
                "known_good": store.load().to_dict(),
                "deployments": store.history(limit=20),
            }
        except Exception as exc:
            return {"supervised": self.supervised, "error": f"{type(exc).__name__}: {exc}"}

    def request_restart(self, reason: str, *, expected_revision: str = "", promotion_id: str = "",
                        requested_by: str = "core") -> dict[str, Any]:
        from zeus_supervisor import EXIT_RESTART_REQUESTED

        if not self.supervised:
            return {"ok": False, "error": "not running under the supervisor; restart it by hand",
                    "supervised": False}
        self._control().request("restart", reason=reason, expected_revision=expected_revision,
                                promotion_id=promotion_id, requested_by=requested_by)
        self.core.emit(EventType.NOTIFICATION, {"text": f"restarting: {reason}", "kind": "restart"})
        self._plan_exit(EXIT_RESTART_REQUESTED, reason)
        return {"ok": True, "restarting": True, "reason": reason}

    def request_relaunch(self, reason: str, *, exe: str, previous: str = "", promotion_id: str = "",
                         requested_by: str = "core") -> dict[str, Any]:
        """Hand over to a promoted ZEUS.exe: the supervisor exits, a watchdog starts the new one."""

        from zeus_supervisor import EXIT_SHUTDOWN_REQUESTED

        if not self.supervised:
            return {"ok": False, "error": "not running under the supervisor; start the new ZEUS.exe by hand", "supervised": False}
        self._control().request("relaunch", reason=reason, exe=exe, previous=previous, promotion_id=promotion_id,
                                requested_by=requested_by)
        self.core.emit(EventType.NOTIFICATION, {"text": f"relaunching into the promoted release: {reason}", "kind": "relaunch"})
        self._plan_exit(EXIT_SHUTDOWN_REQUESTED, reason)
        return {"ok": True, "relaunching": True, "exe": exe, "reason": reason}

    def request_shutdown(self, reason: str, *, requested_by: str = "core") -> dict[str, Any]:
        from zeus_supervisor import EXIT_SHUTDOWN_REQUESTED

        if self.supervised:
            self._control().request("shutdown", reason=reason, requested_by=requested_by)
        self.core.emit(EventType.NOTIFICATION, {"text": f"shutting down: {reason}", "kind": "shutdown"})
        self._plan_exit(EXIT_SHUTDOWN_REQUESTED, reason)
        return {"ok": True, "stopping": True, "reason": reason}

    def _plan_exit(self, code: int, reason: str) -> None:
        self.exit_code = code
        self.exit_reason = reason
        self.save_conversation(reason, archive=code == 0)
        # Give the HTTP response time to leave before the process does.
        threading.Timer(0.5, self.exit_event.set).start()

    # -- conversation persistence -------------------------------------

    def _resume_path(self) -> Path:
        return Path(self.core.kernel.state_root) / "conversation_resume.json"

    def save_conversation(self, reason: str, *, archive: bool = False) -> None:
        # On a FINAL shutdown the archive keeps a durable copy (the resume
        # file below would otherwise be the only trace, and it is consumed).
        # A planned restart restores the live transcript instead — archiving
        # there too would duplicate the same conversation on every restart.
        if archive:
            try:
                self.core.conversations.archive([turn.to_dict() for turn in self.core.history],
                                                language=self.core.language, reason=reason)
            except Exception:  # noqa: BLE001
                pass
        try:
            payload = {
                "saved_at": time.time(),
                "reason": reason,
                "language": self.core.language,
                "turns": [turn.to_dict() for turn in self.core.history][-40:],
                "revision": self.revision(),
            }
            path = self._resume_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass

    def restore_conversation(self) -> dict[str, Any]:
        """Load a recent saved transcript.  Returns what was restored."""

        from service.core import ConversationTurn

        path = self._resume_path()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        age = time.time() - float(data.get("saved_at", 0))
        try:
            path.unlink()
        except OSError:
            pass
        if age > RESUME_MAX_AGE_SECONDS:
            return {}
        turns = [
            ConversationTurn(role=str(t.get("role", "")), text=str(t.get("text", "")), at=str(t.get("at", "")),
                             backend=str(t.get("backend", "")))
            for t in data.get("turns", []) if isinstance(t, dict)
        ]
        with self.core._lock:
            self.core._history.extend(turns)
        self.core.language = str(data.get("language", "")) or self.core.language
        self.resumed = {
            "turns": len(turns), "reason": str(data.get("reason", "")), "age_seconds": round(age, 1),
            "previous_revision": str(data.get("revision", "")),
        }
        return self.resumed
