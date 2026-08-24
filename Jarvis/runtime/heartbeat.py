"""A liveness signal, so a long run can be told apart from a dead one.

The problem this solves showed up the first time an autonomous run went quiet:
from the outside, "the 7B model is three minutes into a large prompt" and "the
process is wedged and will never return" look exactly the same.  Checkpoints do
not help, because they are written *between* stages and a stall happens *inside*
one.

A heartbeat is a tiny file whose ``updated_at`` advances every few seconds while
the run is alive, including during a single long model call.  That turns the
question from "has anything happened lately?" -- which is about progress and can
legitimately be no for minutes -- into "is this process still breathing?", which
has a real answer.

The two are kept separate on purpose:

``stage`` / ``progress_at``
    Advance when the run genuinely moves forward.  A stale one means no progress.

``updated_at``
    Advances on a timer regardless.  A stale one means the process is gone or
    truly wedged.

Reading is deliberately cheap and lock-free, so a supervisor, another shell, or
:mod:`jarvis.doctor` can check on a run without disturbing it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HeartbeatState:
    """What a heartbeat file says."""

    pid: int = 0
    run: str = ""
    stage: str = "starting"
    detail: str = ""
    started_at: str = ""
    #: Bumped by the ticker; proves the process is alive.
    updated_at: str = ""
    #: Bumped only by real progress; proves the run is getting somewhere.
    progress_at: str = ""
    steps: int = 0
    budget_seconds: float | None = None
    elapsed_seconds: float = 0.0
    finished: bool = False
    outcome: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Heartbeat:
    """Writes a liveness file, optionally on a background timer."""

    def __init__(self, path: str | Path | None, *, run: str = "", interval: float = 5.0) -> None:
        self.path = Path(path) if path else None
        self.interval = max(1.0, float(interval))
        self.started = time.monotonic()
        self.state = HeartbeatState(pid=os.getpid(), run=run, started_at=_now(), updated_at=_now(), progress_at=_now())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    # -- writing ---------------------------------------------------------

    def _write(self) -> None:
        if self.path is None:
            return
        self.state.elapsed_seconds = round(time.monotonic() - self.started, 1)
        payload = json.dumps(self.state.to_dict(), indent=2, sort_keys=True, default=str)
        try:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            # A heartbeat that cannot be written must never break the run it is
            # reporting on.
            pass

    def beat(self, stage: str | None = None, detail: str = "", *, progress: bool = False, **extra: Any) -> None:
        """Record that the run is alive, and optionally that it moved forward."""

        with self._lock:
            if stage is not None:
                self.state.stage = stage
            if detail:
                self.state.detail = detail[:400]
            self.state.updated_at = _now()
            if progress:
                self.state.progress_at = _now()
                self.state.steps += 1
            if extra:
                self.state.extra.update(extra)
            self._write()

    def set_budget(self, seconds: float | None) -> None:
        with self._lock:
            self.state.budget_seconds = None if seconds is None else round(float(seconds), 1)
            self._write()

    def finish(self, outcome: str) -> None:
        with self._lock:
            self.state.finished = True
            self.state.outcome = outcome
            self.state.updated_at = _now()
            self._write()
        self.stop()

    # -- the ticker ------------------------------------------------------

    def start(self) -> "Heartbeat":
        """Begin bumping ``updated_at`` on a timer.

        This is the part that makes a long model call distinguishable from a
        wedged one: without it the file goes stale for exactly as long as the
        model takes to answer, and staleness stops meaning anything.
        """

        if self._thread is not None or self.path is None:
            return self
        self._stop.clear()

        def tick() -> None:
            while not self._stop.wait(self.interval):
                with self._lock:
                    self.state.updated_at = _now()
                    self._write()

        self._thread = threading.Thread(target=tick, name="jarvis-heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


@dataclass
class Liveness:
    """A supervisor's verdict about a run it did not start."""

    #: "alive", "stalled", "dead", "finished", "unknown"
    state: str
    detail: str = ""
    seconds_since_beat: float | None = None
    seconds_since_progress: float | None = None
    heartbeat: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_attention(self) -> bool:
        return self.state in {"stalled", "dead"}


def read_heartbeat(path: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _age(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - when).total_seconds())


def check_liveness(
    path: str | Path,
    *,
    dead_after: float = 60.0,
    stalled_after: float = 900.0,
) -> Liveness:
    """Classify a run from its heartbeat file.

    ``dead_after`` is measured against the ticker, so it can be short -- a
    healthy run touches the file every few seconds whatever it is doing.
    ``stalled_after`` is measured against real progress, so it must be generous:
    a single legitimate step against a local 7B model can take minutes.
    """

    payload = read_heartbeat(path)
    if payload is None:
        return Liveness("unknown", f"no heartbeat at {path}")

    since_beat = _age(payload.get("updated_at"))
    since_progress = _age(payload.get("progress_at"))

    if payload.get("finished"):
        return Liveness(
            "finished",
            str(payload.get("outcome", "")),
            since_beat,
            since_progress,
            payload,
        )

    pid = int(payload.get("pid") or 0)
    if pid and not _process_alive(pid):
        return Liveness("dead", f"process {pid} is gone and the run never finished", since_beat, since_progress, payload)

    if since_beat is not None and since_beat > dead_after:
        return Liveness(
            "dead",
            f"no heartbeat for {since_beat:.0f}s (limit {dead_after:.0f}s)",
            since_beat,
            since_progress,
            payload,
        )

    if since_progress is not None and since_progress > stalled_after:
        return Liveness(
            "stalled",
            f"alive but no progress for {since_progress:.0f}s (limit {stalled_after:.0f}s) "
            f"in stage {payload.get('stage', '?')}",
            since_beat,
            since_progress,
            payload,
        )

    return Liveness("alive", f"in stage {payload.get('stage', '?')}", since_beat, since_progress, payload)


def _process_alive(pid: int) -> bool:
    """Whether a pid is still running, without signalling it."""

    if os.name == "nt":
        import subprocess

        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                # tasklist emits the console codepage, not the locale default;
                # decoding it as cp1252 raises on any non-ASCII byte and the
                # liveness check dies in a reader thread.
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return True  # cannot tell; assume alive rather than kill a live run
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
