"""Background indexing: every document ZEUS reads is a persisted job with real progress.

A job is a row in the study index (``index_jobs``), so it survives the owner leaving the
Studium page, a reload of the interface and a restart of ZEUS.  One worker thread takes
jobs in order and runs the document pipeline:

    queued -> parsing -> rendering (slides) -> ocr (pages without text) -> chunking
           -> indexing (searchable from here) -> embedding (semantic search) -> ready
                                                                      \\-> failed

Every phase reports what it really did -- pages, slides, OCR pages, chunks, embedded
chunks, the page being read -- and the percentage is computed from those counters, never
from a timer.  Work is never done twice: an unchanged file is skipped by its fingerprint,
OCR is cached per rendered page, embeddings are cached per passage text, and a job that
was interrupted resumes from those caches.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from study.index import JOB_COUNTERS

ACTIVE = ("queued", "running")
DONE = ("ready", "failed", "cancelled", "skipped")
PHASES = ("queued", "parsing", "rendering", "ocr", "chunking", "indexing", "embedding", "ready", "failed")


class Cancelled(Exception):
    pass


def job_fraction(job: dict[str, Any], *, embedding_planned: bool = False) -> float:
    """How far a job is, 0..1, from its counters: each kind of work weighs what it costs, only work the document has counts."""

    if job.get("state") in DONE:
        return 1.0
    parts: list[tuple[float, float]] = []   # (weight, done fraction)

    def part(weight: float, done: int, total: int) -> None:
        if total:
            parts.append((weight, min(1.0, done / total)))

    phase = job.get("phase") or "queued"
    order = {name: i for i, name in enumerate(PHASES)}
    reached = order.get(phase, 0)
    pages = int(job.get("pages_total") or 0)
    part(1.0 if pages else 0.0, int(job.get("pages_completed") or 0), pages)
    if not pages:
        parts.append((0.5, 1.0 if reached > order["parsing"] else 0.0))
    part(3.0, int(job.get("slides_completed") or 0), int(job.get("slides_total") or 0))
    part(6.0, int(job.get("ocr_pages_completed") or 0), int(job.get("ocr_pages_total") or 0))
    chunks = int(job.get("chunks_total") or 0)
    parts.append((0.5, 1.0 if reached > order["indexing"] else (int(job.get("chunks_completed") or 0) / chunks if chunks else 0.0)))
    embedding_total = int(job.get("embedding_chunks_total") or 0)
    if embedding_total:
        part(3.0, int(job.get("embedding_chunks_completed") or 0), embedding_total)
    elif embedding_planned:
        parts.append((3.0, 0.0))  # semantic indexing will follow: its share is reserved from the start
    if job.get("kind") == "embed" and not int(job.get("embedding_chunks_total") or 0):
        return 0.0
    total = sum(w for w, _ in parts)
    return round(sum(w * f for w, f in parts) / total, 4) if total else 0.0


class JobProgress:
    """What the pipeline tells its job: counters and phase.  Persists and announces at a calm rate."""

    def __init__(self, indexer: "Indexer", job: dict[str, Any]) -> None:
        self.indexer = indexer
        self.job = job
        self._saved = 0.0
        self._sent = 0.0

    def phase(self, name: str, **counters: Any) -> None:
        self.job["phase"] = name
        self.update(force=True, **counters)

    def update(self, *, force: bool = False, **counters: Any) -> None:
        for key, value in counters.items():
            if key in JOB_COUNTERS or key in {"document_id", "source_type", "name"}:
                self.job[key] = value
        now = time.time()
        self.job["updated_at"] = now
        if force or now - self._saved >= 0.5:
            self.indexer.index.save_job(self.job)
            self._saved = now
        if force or now - self._sent >= 0.25:
            self.indexer.announce(self.job)
            self._sent = now
        if self.indexer.cancel_requested(self.job["job_id"]):
            raise Cancelled()

    def should_stop(self) -> bool:
        return self.indexer.cancel_requested(self.job["job_id"]) or self.indexer.stopping


class Indexer:
    def __init__(self, service: Any, *, emit: Callable[[str, dict[str, Any]], None]) -> None:
        self.service = service
        self.index = service.index
        self.emit = emit
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self.stopping = False
        self.current: str = ""
        self._high: dict[str, float] = {}

    # -- queue ---------------------------------------------------------------------------

    def new_job(self, *, kind: str, name: str, path: str = "", provider: str = "upload", source_id: str = "", root: str = "",
                fingerprint: str = "", document_id: str = "", source_type: str = "") -> dict[str, Any]:
        now = time.time()
        job = {"job_id": uuid.uuid4().hex[:12], "kind": kind, "document_id": document_id, "name": name, "path": path, "provider": provider,
               "source_id": source_id, "root": root, "fingerprint": fingerprint, "state": "queued", "phase": "queued", "reason": "", "error": "",
               "source_type": source_type, **{key: 0 for key in JOB_COUNTERS}, "created_at": now, "started_at": None, "updated_at": now,
               "finished_at": None, "attempts": 0}
        return job

    def enqueue(self, job: dict[str, Any]) -> dict[str, Any]:
        """Queue a job -- unless the same work is already queued or running (same fingerprint, or same path for a file)."""

        with self._lock:
            for other in self.index.jobs(states=ACTIVE):
                same_file = job.get("path") and other.get("path") and os.path.normcase(other["path"]) == os.path.normcase(job["path"])
                same_content = job.get("fingerprint") and other.get("fingerprint") == job["fingerprint"] and other.get("kind") == job["kind"]
                same_doc = job["kind"] == "embed" and other.get("document_id") == job.get("document_id") and job.get("document_id")
                if (same_file and other.get("kind") == job["kind"]) or same_content or same_doc:
                    return other
            self.index.save_job(job)
        self.announce(job)
        self._wake.set()
        return job

    def finish(self, job: dict[str, Any], state: str, *, phase: str | None = None, reason: str = "", error: str = "") -> dict[str, Any]:
        job.update(state=state, phase=phase or ("ready" if state in {"ready", "skipped"} else state), reason=reason, error=error,
                   finished_at=time.time(), updated_at=time.time())
        self.index.save_job(job)
        self.announce(job)
        hook = getattr(self.service, "job_finished", None)
        if hook is not None:
            try:
                hook(job)
            except Exception:  # noqa: BLE001
                pass
        return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.index.job(job_id)
        if job is None:
            return {"ok": False, "error": "Auftrag unbekannt"}
        if job["state"] == "queued":
            self.finish(job, "cancelled", reason="cancelled")
        elif job["state"] == "running":
            self._cancel.add(job_id)
        return {"ok": True, "job": self.index.job(job_id)}

    def cancel_requested(self, job_id: str) -> bool:
        return job_id in self._cancel

    def fraction(self, job: dict[str, Any]) -> float:
        """The shown progress of a job never goes back: work discovered later (OCR pages found while reading) widens the
        remaining part instead of shrinking what the owner already saw."""

        value = job_fraction(job, embedding_planned=job.get("kind") == "import" and self.embedding_planned())
        best = max(value, self._high.get(job["job_id"], 0.0))
        self._high[job["job_id"]] = best
        if len(self._high) > 2000:
            self._high.clear()
        return round(best, 4)

    def embedding_planned(self) -> bool:
        can = getattr(self.service, "_can_embed", None)
        return bool(can()) if can else False

    def announce(self, job: dict[str, Any]) -> None:
        payload = dict(job)
        payload["fraction"] = self.fraction(job)
        self.emit("study.progress", payload)

    # -- worker -------------------------------------------------------------------------

    def start(self) -> None:
        """Resume what a restart interrupted, then work in the background until ZEUS stops."""

        if self._thread is not None and self._thread.is_alive():
            return
        for job in self.index.jobs(states=("running",)):
            job.update(state="queued", attempts=int(job.get("attempts") or 0) + 1, updated_at=time.time())
            self.index.save_job(job)
        self.index.prune_jobs(older_than=time.time() - 7 * 86400)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="study-indexer")
        self._thread.start()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        _lower_thread_priority()
        while not self.stopping:
            self._wake.wait(timeout=30)
            self._wake.clear()
            try:
                self.service.backfill_embedding_jobs()
                while self.run_next():
                    if self.stopping:
                        return
            except Exception:  # noqa: BLE001 - the worker never dies; the job that failed says why
                time.sleep(2)

    def run_next(self) -> bool:
        queued = self.index.jobs(states=("queued",), limit=1)
        if not queued:
            return False
        self.run(queued[0])
        return True

    def run_pending(self, *, limit: int = 1000) -> int:
        """Synchronously work the queue (tests, and the interface's "wait" imports)."""

        done = 0
        while done < limit and self.run_next():
            done += 1
        return done

    def run(self, job: dict[str, Any]) -> dict[str, Any]:
        job.update(state="running", started_at=job.get("started_at") or time.time(), attempts=int(job.get("attempts") or 0))
        self.current = job["job_id"]
        progress = JobProgress(self, job)
        progress.phase("parsing" if job["kind"] != "embed" else "embedding")
        try:
            result = self.service.run_job(job, progress)
        except Cancelled:
            self._cancel.discard(job["job_id"])
            return self.finish(job, "cancelled", reason="cancelled")
        except Exception as exc:  # noqa: BLE001
            return self.finish(job, "failed", phase="failed", reason="indexing", error=f"Fehler beim Einlesen: {type(exc).__name__}: {exc}"[:300])
        finally:
            self.current = ""
        state = result.get("state") or ("ready" if result.get("ok") else "failed")
        return self.finish(job, state, phase="failed" if state == "failed" else None, reason=result.get("reason", ""), error=result.get("error", ""))

    # -- what the interface shows ----------------------------------------------------------

    def summary(self, *, recent_seconds: float = 20.0) -> dict[str, Any]:
        """The indexing line under the Studium search: the active batch, its real percentage, what is being read now."""

        now = time.time()
        active = self.index.jobs(states=ACTIVE)
        # the batch is everything finished since its oldest open job was added: finished documents stay counted while the rest
        # runs, so "3 von 7" and the percentage only ever move forward; once all is done the batch lingers a moment, then folds
        started = min((float(j.get("created_at") or now) for j in active), default=now - recent_seconds)
        recent = self.index.jobs(since=min(started, now - recent_seconds), states=DONE)
        batch = active + [j for j in recent if j["job_id"] not in {a["job_id"] for a in active}]
        failed = [j for j in self.index.jobs(since=now - 3600, states=("failed",))]
        running = next((j for j in active if j["state"] == "running"), None)
        documents = [j for j in batch if (j["kind"] != "embed" or not any(b["document_id"] == j["document_id"] and b["kind"] != "embed" for b in batch))
                     and not (j["state"] == "skipped" and j.get("reason") == "duplicate")]  # "war schon da" is not a document being indexed
        fractions = [self.fraction(j) for j in documents]
        percent = round(100 * sum(fractions) / len(fractions)) if fractions else 100
        pages_total = sum(int(j.get("pages_total") or 0) + int(j.get("slides_total") or 0) for j in documents)
        pages_done = sum(int(j.get("pages_completed") or 0) + int(j.get("slides_completed") or 0) for j in documents)
        return {"ok": True, "active": bool(active), "documents_total": len(documents),
                "documents_completed": sum(1 for j in documents if j["state"] in DONE), "percent": percent,
                "pages_total": pages_total, "pages_completed": pages_done,
                "current": self._describe(running) if running else None,
                "jobs": [self._describe(j) for j in batch][-50:], "failed": [self._describe(j) for j in failed][-10:]}

    def _describe(self, job: dict[str, Any]) -> dict[str, Any]:
        data = dict(job)
        data["fraction"] = self.fraction(job)
        return data


def _lower_thread_priority() -> None:
    """Indexing is background work: on Windows the worker thread runs below normal priority, so talking to ZEUS stays quick."""

    if os.name != "nt":
        return
    try:
        import ctypes

        THREAD_PRIORITY_BELOW_NORMAL = -1
        ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:  # noqa: BLE001
        pass


def fingerprint_file(path: str | Path) -> str:
    from study.service import _hash_file

    return _hash_file(Path(path))
