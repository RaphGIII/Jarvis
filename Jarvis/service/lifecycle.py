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
        }

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
        self.save_conversation(reason)
        # Give the HTTP response time to leave before the process does.
        threading.Timer(0.5, self.exit_event.set).start()

    # -- conversation persistence -------------------------------------

    def _resume_path(self) -> Path:
        return Path(self.core.kernel.state_root) / "conversation_resume.json"

    def save_conversation(self, reason: str) -> None:
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
