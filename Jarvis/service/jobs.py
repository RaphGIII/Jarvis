"""WorkItems: every long-running action is a visible, durable job.

The live failure this fixes: the owner asked for an image, saw NOTHING for
minutes, and the result appeared out of nowhere later.  Execution and UI
state were disconnected — work happened, but no first-class object
represented it.

A Job is that object.  It has a lifecycle
(QUEUED → UNDERSTANDING/PLANNING → WAITING_FOR_RESOURCE → EXECUTING →
VERIFYING → FINALIZING → COMPLETED/FAILED/CANCELLED), a phase line for the
owner ("Modell lädt", "Generiere", "Speichere"), optional numeric progress,
and a result.  Every change is emitted as an EventType.JOB event, so the eye
and the Work Center always mirror reality; finished jobs are appended to a
JSONL history so results survive reloads and attach back to conversations.

This deliberately complements the Mission Engine rather than replacing it:
missions are the durable *record* of goals; jobs are the live *now* — what
is running at this moment, how far it is, and what it produced.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ACTIVE_STATES = ("QUEUED", "UNDERSTANDING", "PLANNING", "WAITING_FOR_RESOURCE",
                 "EXECUTING", "VERIFYING", "FINALIZING")
DONE_STATES = ("COMPLETED", "FAILED", "CANCELLED")


@dataclass
class Job:
    job_id: str
    title: str
    kind: str                      # image | research | acquisition | selfdev | web | index | pdf | ...
    state: str = "QUEUED"
    phase: str = ""                # the owner-readable current step
    progress: float | None = None  # 0..1 when measurable, None = phase animation
    detail: str = ""
    background: bool = True
    cancellable: bool = False
    scope: str = ""
    request_id: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    timings: dict[str, float] = field(default_factory=dict)  # phase -> seconds since create
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "title": self.title, "kind": self.kind, "state": self.state,
                "phase": self.phase, "progress": self.progress, "detail": self.detail,
                "background": self.background, "cancellable": self.cancellable, "scope": self.scope,
                "request_id": self.request_id, "created_at": self.created_at,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "seconds": round((self.finished_at or time.time()) - self.created_at, 1),
                "result": self.result, "error": self.error, "timings": self.timings}


class JobBoard:
    """The live registry of work, with an append-only history on disk."""

    def __init__(self, history_path: str | Path | None = None,
                 emit: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.history_path = Path(history_path) if history_path else None
        self._emit = emit or (lambda payload: None)
        self._jobs: dict[str, Job] = {}
        self._history_cache: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def _history_tail(self, limit: int = 12) -> list[dict[str, Any]]:
        """Finished jobs from before this process: results survive restarts."""

        if self._history_cache is None:
            rows: list[dict[str, Any]] = []
            if self.history_path is not None and self.history_path.is_file():
                try:
                    for line in self.history_path.read_text(encoding="utf-8").splitlines()[-60:]:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            continue
                except OSError:
                    pass
            self._history_cache = rows
        return self._history_cache[-limit:]

    # -- lifecycle -------------------------------------------------------

    def create(self, title: str, *, kind: str, scope: str = "", request_id: str = "",
               background: bool = True, cancellable: bool = False, phase: str = "") -> Job:
        job = Job(job_id="job_" + uuid.uuid4().hex[:10], title=str(title)[:120], kind=kind,
                  scope=scope, request_id=request_id, background=background, cancellable=cancellable,
                  phase=phase)
        with self._lock:
            self._jobs[job.job_id] = job
        self._announce(job, "created")
        return job

    def _get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._get(job_id)
        return job.to_dict() if job else None

    def phase(self, job_id: str, phase: str, *, state: str = "EXECUTING",
              progress: float | None = None, detail: str = "") -> None:
        job = self._get(job_id)
        if job is None or not job.active:
            return
        if not job.started_at and state == "EXECUTING":
            job.started_at = time.time()
        job.state = state
        job.phase = phase
        job.progress = progress
        if detail:
            job.detail = detail[:300]
        job.timings[phase or state] = round(time.time() - job.created_at, 2)
        self._announce(job, "phase")

    def complete(self, job_id: str, result: dict[str, Any] | None = None, *, phase: str = "") -> None:
        job = self._get(job_id)
        if job is None or job.state in DONE_STATES:
            return
        job.state = "COMPLETED"
        job.phase = phase or "fertig"
        job.progress = 1.0
        job.finished_at = time.time()
        job.result = dict(result or {})
        job.timings["completed"] = round(job.finished_at - job.created_at, 2)
        self._announce(job, "completed")
        self._persist(job)

    def fail(self, job_id: str, error: str, *, detail: str = "") -> None:
        job = self._get(job_id)
        if job is None or job.state in DONE_STATES:
            return
        job.state = "FAILED"
        job.error = str(error)[:400]
        if detail:
            job.detail = detail[:300]
        job.finished_at = time.time()
        self._announce(job, "failed")
        self._persist(job)

    def cancel(self, job_id: str) -> bool:
        job = self._get(job_id)
        if job is None or job.state in DONE_STATES:
            return False
        job.cancel_event.set()
        if not job.cancellable:
            return False  # the runner may still honour the event; state waits for it
        job.state = "CANCELLED"
        job.finished_at = time.time()
        self._announce(job, "cancelled")
        self._persist(job)
        return True

    def cancelled(self, job_id: str) -> bool:
        job = self._get(job_id)
        return bool(job and job.cancel_event.is_set())

    # -- queries ---------------------------------------------------------

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.active]
        return [j.to_dict() for j in sorted(jobs, key=lambda j: j.created_at)]

    def recent(self, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock:
            done = [j.to_dict() for j in self._jobs.values() if j.state in DONE_STATES]
        seen = {j["job_id"] for j in done}
        done += [row for row in self._history_tail(limit) if row.get("job_id") not in seen]
        done.sort(key=lambda j: j.get("finished_at") or 0, reverse=True)
        return done[:limit]

    def snapshot(self) -> dict[str, Any]:
        return {"active": self.active(), "recent": self.recent()}

    # -- plumbing --------------------------------------------------------

    def _announce(self, job: Job, event: str) -> None:
        try:
            self._emit({"event": event, **job.to_dict()})
        except Exception:  # noqa: BLE001 - reporting must never break the work
            pass

    def _persist(self, job: Job) -> None:
        if self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass
