"""The Studium service: import, connect, index, find, open.

The one object the core and the HTTP routes talk to.  It owns the study index (its own
SQLite file, its own namespace), the page cache, the connected sources, the semantic
embeddings, and the path from a question to a page: search -> result with location ->
page image with every match marked -> the original file.

Importing is a persisted job of the background indexer (study/indexer.py) with real progress:

    queued -> parsing -> ocr -> chunking -> indexing -> rendering (slides) -> embedding -> ready
                                                                                    or failed (unsupported | missing | too_large |
                                                                                        parse_failed | encrypted | goodnotes_native)

Removing a document from Studium never deletes the owner's file unless explicitly asked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from study import goodnotes, ocr, parsers, retrieval, taxonomy
from study.highlight import arrange
from study.index import StudyIndex

MAX_FILE_BYTES = 400 * 1024 * 1024
MAX_SCAN_FILES = 3000
EMBED_BATCH = 8
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "$recycle.bin", "system volume information"}
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    stem = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", Path(name).name).strip(" .")
    return stem or "Material"


def _parse_failure(exc: Exception) -> tuple[str, str]:
    """The calm owner sentence for a file that could not be read, and its reason code."""

    name = type(exc).__name__
    message = str(exc).lower()
    if "decrypt" in message or "encrypt" in message or "password" in message or name in {"FileNotDecryptedError", "DependencyError"}:
        return "encrypted", "Diese PDF ist mit einem Passwort geschützt. Entferne den Schutz und füge sie erneut hinzu."
    if name in {"PdfReadError", "PdfStreamError", "EmptyFileError", "BadZipFile", "ParseError"}:
        return "parse_failed", "Diese Datei ist beschädigt oder unvollständig und konnte nicht gelesen werden."
    return "parse_failed", f"Konnte nicht gelesen werden: {name}"


def _call_ocr(doc: Any, path: Path, engine: Any, cache_dir: Path, on_progress: Callable[..., None], should_stop: Callable[[], bool]) -> int:
    """OCR for the pages without text, with page progress when the OCR code supports it."""

    import inspect

    parameters = inspect.signature(parsers.ocr_pdf_pages).parameters
    kwargs: dict[str, Any] = {"engine": engine, "cache_dir": cache_dir}
    if "ocr_progress" in parameters:
        kwargs["ocr_progress"] = lambda done, total, *rest: on_progress(done, total, *(rest[:1]))
    if "should_stop" in parameters:
        kwargs["should_stop"] = should_stop
    return parsers.ocr_pdf_pages(doc, path, **kwargs)


class StudyService:
    def __init__(self, data_dir: str | Path, store_dir: str | Path, *, engine: Any = None, library_root: str | Path | None = None,
                 expander: Callable[[str], list[str]] | None = None, emit: Callable[[str, dict[str, Any]], None] | None = None,
                 embeddings: Any = None, slide_renderer: Any = None, background: bool = False) -> None:
        """``embeddings``: a provider (study.embeddings), ``None`` for the configured one, ``False`` for none.
        ``slide_renderer``: a module-like object with ``available()`` and ``render_slides()``, ``None`` for study.slides, ``False`` for none.
        ``background``: the product -- a background indexer works the persisted job queue and connected folders are watched;
        tests run jobs themselves (``indexer.run_pending()``, ``embed_pending()``)."""

        from study.indexer import Indexer

        self.data_dir = Path(data_dir)
        self.store_dir = Path(store_dir)
        self.library_root = Path(library_root) if library_root else None
        self.index = StudyIndex(self.data_dir / "index.sqlite")
        self.engine = engine
        self.expander = expander
        self.emit = emit or (lambda kind, payload: None)
        self._scan_lock = threading.Lock()
        self.scanning: dict[str, Any] = {}
        self._embeddings_arg = embeddings
        self._embeddings: Any = None
        self._matrix: Any = None
        self._matrix_key: tuple[str, int] | None = None
        self._matrix_lock = threading.Lock()
        if slide_renderer is None:
            from study import slides as slide_renderer  # noqa: PLW0127 - the default renderer
        self.slides = slide_renderer or None
        self._render_lock = threading.Lock()
        self._render_gate = threading.Lock()        # one page render at a time; queued ones may be dropped when superseded
        self._render_meta_lock = threading.Lock()
        self._render_latest: dict[str, int] = {}
        self._renders = 0
        self.rendering: dict[str, str] = {}
        self.background = background
        self.indexer = Indexer(self, emit=self.emit)
        self._watch_thread: threading.Thread | None = None
        if background:
            self.indexer.start()
            self.start_watching()

    # -- importing --------------------------------------------------------------------

    def import_path(self, path: str | Path, *, provider: str = "upload", source_id: str = "", copy: bool = False,
                    root: str | Path | None = None, wait: bool = True) -> dict[str, Any]:
        """Import one file as an index job: checked now, read now (``wait``) or by the background indexer."""

        source = Path(path)
        name = source.name
        kind = parsers.source_type_for(source)
        job = self.indexer.new_job(kind="import", name=name, path=str(source), provider=provider, source_id=source_id,
                                   root=str(root or ""), source_type=kind)
        refused = self._refusal(source, kind)
        if refused is not None:
            self.index.save_job(job)
            self.indexer.finish(job, "failed", phase="failed", reason=refused["reason"], error=refused["error"])
            return {**refused, "ok": False, "path": str(source), "job_id": job["job_id"], "import_id": job["job_id"]}
        content_hash = _hash_file(source)
        job["fingerprint"] = content_hash
        duplicate = self.index.find_by_hash(content_hash)
        if duplicate and provider in {"upload", "zeus_files"}:
            self.index.save_job(job)
            self.indexer.finish(job, "skipped", reason="duplicate")
            return {"ok": True, "duplicate": True, "document": duplicate, "chunks": None, "job_id": job["job_id"], "import_id": job["job_id"]}
        stored = source
        if copy:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            stored = self.store_dir / _safe_name(source.name)
            counter = 2
            while stored.exists() and _hash_file(stored) != content_hash:
                stored = self.store_dir / f"{Path(_safe_name(source.name)).stem} ({counter}){source.suffix}"
                counter += 1
            if not stored.exists():
                shutil.copy2(source, stored)
        job["path"] = str(stored)
        job["root"] = str(root or "")
        job["original_path"] = str(source)
        if not wait:
            queued = self.indexer.enqueue(job)
            return {"ok": True, "queued": True, "job_id": queued["job_id"], "import_id": queued["job_id"], "name": name}
        self.index.save_job(job)
        job["_inline"] = True
        finished = self.indexer.run(job)
        outcome = dict(job.get("_outcome") or {})
        outcome.setdefault("ok", finished["state"] in {"ready", "skipped"})
        if finished["state"] == "failed":
            outcome.update(ok=False, reason=finished.get("reason"), error=finished.get("error"))
        outcome.update(job_id=job["job_id"], import_id=job["job_id"])
        return outcome

    def _refusal(self, source: Path, kind: str) -> dict[str, Any] | None:
        if source.suffix.lower() == goodnotes.NATIVE_SUFFIX:
            return {"reason": "goodnotes_native", "error": ("Diese GoodNotes-Datei kann ZEUS derzeit nicht direkt lesen. Exportiere sie als PDF "
                                                             "oder verbinde deinen GoodNotes-Backup-Ordner.")}
        if not kind:
            return {"reason": "unsupported", "error": f"Dieses Format kann Studium nicht lesen: {source.suffix or source.name}"}
        if not source.is_file():
            return {"reason": "missing", "error": "Datei nicht gefunden"}
        if source.stat().st_size > MAX_FILE_BYTES:
            return {"reason": "too_large", "error": "Datei ist zu groß"}
        return None

    def run_job(self, job: dict[str, Any], progress: Any) -> dict[str, Any]:
        """One job of the background indexer (study.indexer): import a file, or embed an indexed document."""

        if job["kind"] == "embed":
            report = self._embed_document(job["document_id"], progress)
            return {"ok": not report.get("error"), "state": "failed" if report.get("error") else "ready",
                    "reason": "embedding" if report.get("error") else "", "error": report.get("error", "")}
        outcome = self._run_import(job, progress)
        job["_outcome"] = outcome
        return outcome

    def _run_import(self, job: dict[str, Any], progress: Any) -> dict[str, Any]:
        stored = Path(job["path"])
        source = Path(job.get("original_path") or job["path"])
        provider = job.get("provider") or "upload"
        kind = parsers.source_type_for(stored)
        refused = self._refusal(stored, kind)
        if refused is not None:
            return {**refused, "ok": False, "state": "failed"}
        content_hash = job.get("fingerprint") or _hash_file(stored)
        job["fingerprint"] = content_hash
        doc_id = hashlib.sha1((f"path:{os.path.normcase(str(stored.resolve()))}" if provider not in {"upload", "zeus_files", "goodnotes"}
                               else f"hash:{content_hash}").encode("utf-8")).hexdigest()[:16]
        planned = self._planned_units(stored, kind)
        if planned.get("reason"):
            return {"ok": False, "state": "failed", **planned}
        progress.phase("parsing", document_id=doc_id, source_type=kind, pages_total=planned.get("pages", 0), slides_total=planned.get("slides", 0))
        if self.index.document(doc_id) is None:
            # visible at once (list, star view) while it is being read; replaced by the real record when indexing finishes
            from study.model import NormalizedDocument

            stat = stored.stat()
            self.index.upsert({"id": doc_id, "filename": source.name, "title": parsers._title_from_name(source), "stored_path": str(stored),
                               "original_path": str(source), "provider": provider, "source_id": job.get("source_id") or "", "content_hash": content_hash,
                               "size_bytes": stat.st_size, "created_at": stat.st_ctime, "updated_at": stat.st_mtime, "status": "processing",
                               "flags": {"placeholder": True, "planned_units": planned.get("pages") or planned.get("slides") or 0}},
                              NormalizedDocument(source_type=kind, title=parsers._title_from_name(source)))
        try:
            if kind == "image":
                # a photo or a handwritten page: its word boxes are kept, so the viewer marks the match on the image itself
                progress.phase("ocr", ocr_pages_total=1, ocr_pages_completed=0)
                doc = parsers.parse_image(stored, cache_dir=self.data_dir / "cache" / "ocr" / doc_id, should_stop=progress.should_stop)
            else:
                doc = parsers.parse(stored, engine=self.engine if kind == "pdf" else None)
        except Exception as exc:  # noqa: BLE001 - a broken file is reported, never fatal
            reason, error = _parse_failure(exc)
            return {"ok": False, "state": "failed", "reason": reason, "error": error}
        if kind == "pdf":
            progress.update(pages_completed=len(doc.units), pages_total=len(doc.units))
        ocr_pages = 0
        cache_dir = self.data_dir / "cache" / "ocr" / doc_id
        if kind == "pdf" and any(not u.has_text for u in doc.units) and self.engine is not None and ocr.available():
            pending = sum(1 for u in doc.units if not u.has_text)
            progress.phase("ocr", ocr_pages_total=pending, ocr_pages_completed=0)

            def ocr_progress(done: int, total: int, page: int | None = None) -> None:
                progress.update(ocr_pages_completed=done, ocr_pages_total=total, current_page=int(page or 0))

            ocr_pages = _call_ocr(doc, stored, self.engine, cache_dir, ocr_progress, progress.should_stop)
            progress.update(force=True, ocr_pages_completed=pending)
        elif kind == "image":
            progress.update(force=True, ocr_pages_total=1, ocr_pages_completed=1)
        flags: dict[str, Any] = {}
        # a GoodNotes export (PDF by its producer, page images by name and folder): notes, likely handwritten
        try:
            detected = goodnotes.detect(source, metadata=doc.metadata)
        except Exception:  # noqa: BLE001 - recognising the origin is a bonus
            detected = None
        if detected is None and kind == "pdf" and goodnotes.is_goodnotes_pdf(doc.metadata):
            detected = {"origin": "goodnotes", "kind": "pdf"}
        if detected and detected.get("kind") != "native":
            flags["goodnotes"] = True
            flags["origin"] = f"goodnotes_{detected.get('kind') or 'export'}"
            if detected.get("notebook"):
                flags["notebook"] = detected["notebook"]
            if detected.get("page"):
                flags["notebook_page"] = detected["page"]
            provider = provider if provider not in {"upload"} else "goodnotes"
        empty = [u.number for u in doc.units if not u.has_text]
        if empty:
            flags["pages_without_text"] = len(empty)
        if ocr_pages:
            flags["ocr_pages"] = ocr_pages
        for key in ("text_engine", "ocr_languages", "ocr_engine", "ocr_mean_confidence", "has_handwriting"):
            if doc.metadata.get(key) not in (None, "", False):
                flags[key] = doc.metadata[key]
        flags["figures"] = sum(len(getattr(u, "figures", None) or []) for u in doc.units)
        progress.phase("chunking")
        from study.index import chunk_unit

        chunks_total = sum(len(chunk_unit(u)) for u in doc.units)
        progress.update(force=True, chunks_total=chunks_total)
        progress.phase("indexing")
        sample = "\n".join(u.text for u in doc.units[:6])[:8000]
        categories = taxonomy.infer(source, title=doc.title, sample=sample, root=job.get("root") or None)
        stat = stored.stat()
        meta = {"id": doc_id, "filename": source.name, "title": doc.title, "stored_path": str(stored), "original_path": str(source),
                "provider": provider, "source_id": job.get("source_id") or "", "content_hash": content_hash, "size_bytes": stat.st_size,
                "created_at": stat.st_ctime, "updated_at": stat.st_mtime, "flags": {**flags, "mtime": stat.st_mtime, "warnings": doc.warnings[:10]},
                "status": "indexed" if doc.text_units else "no_text", **categories}
        chunks = self.index.upsert(meta, doc)
        progress.update(force=True, chunks_completed=chunks, chunks_total=chunks)
        record = self.index.document(doc_id)
        self.emit("study.indexed", {"document_id": doc_id, "title": record["title"] if record else doc.title, "chunks": chunks, "provider": provider})
        if kind == "pptx" and self.slides is not None and not job.get("_inline") and self.background and self.slides.available():
            # only the product's background worker draws slides (PowerPoint is started at most once per presentation, then cached)
            progress.phase("rendering", slides_total=len(doc.units), slides_completed=0)
            rendered = self.render_slides_now(doc_id)
            if rendered.get("ok"):
                progress.update(force=True, slides_completed=len(doc.units))
        elif kind == "pptx" and self.slides is not None and self.background:
            self.schedule_slides(doc_id)
        if self._can_embed():
            if not job.get("_inline"):
                self._embed_document(doc_id, progress)
            elif self.background:
                # an import someone is waiting for (a chat attachment) is searchable now; meaning follows as its own job
                self.indexer.enqueue(self.indexer.new_job(kind="embed", name=source.name, document_id=doc_id))
        return {"ok": True, "state": "ready", "duplicate": False, "document": self.index.document(doc_id), "chunks": chunks,
                "warnings": doc.warnings[:10], "empty": not doc.text_units, "ocr_pages": ocr_pages}

    @staticmethod
    def _planned_units(path: Path, kind: str) -> dict[str, Any]:
        """Pages/slides known before reading (for honest progress), and the refusals a file shows at once (encrypted PDF)."""

        try:
            if kind == "pdf":
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                if reader.is_encrypted:
                    try:
                        if not reader.decrypt(""):
                            raise ValueError
                    except Exception:  # noqa: BLE001
                        return {"reason": "encrypted", "error": "Diese PDF ist mit einem Passwort geschützt. Entferne den Schutz und füge sie erneut hinzu."}
                return {"pages": len(reader.pages)}
            if kind == "pptx":
                import zipfile

                with zipfile.ZipFile(path) as archive:
                    return {"slides": sum(1 for n in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n))}
        except Exception as exc:  # noqa: BLE001 - the parser reports it properly
            if kind == "pdf":
                reason, error = _parse_failure(exc)
                return {"reason": reason, "error": error}
        return {}

    def import_bytes(self, name: str, data: bytes, *, provider: str = "upload", wait: bool = True) -> dict[str, Any]:
        """Store an uploaded file in the library and import it -- now, or as a background job."""

        refusal = None
        if not data:
            refusal = {"reason": "empty", "error": "leere Datei"}
        elif Path(name).suffix.lower() == goodnotes.NATIVE_SUFFIX or not parsers.source_type_for(name):
            refusal = self._refusal(Path(name), parsers.source_type_for(name))
        if refusal is not None:
            for earlier in self.index.jobs(since=time.time() - 3600, states=("failed",)):
                if earlier.get("name") == name and earlier.get("reason") == refusal["reason"]:
                    # the same file refused again: one calm line, not one per attempt
                    earlier.update(finished_at=time.time(), updated_at=time.time())
                    self.index.save_job(earlier)
                    return {"ok": False, **refusal, "job_id": earlier["job_id"], "import_id": earlier["job_id"]}
            job = self.indexer.new_job(kind="import", name=name, provider=provider)
            self.index.save_job(job)
            self.indexer.finish(job, "failed", phase="failed", reason=refusal["reason"], error=refusal["error"])
            return {"ok": False, **refusal, "job_id": job["job_id"], "import_id": job["job_id"]}
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self.index.find_by_hash(digest)
        if duplicate is not None:
            # the same bytes are already in Studium: nothing is written into the owner's library a second time
            job = self.indexer.new_job(kind="import", name=name, provider=provider, fingerprint=digest, document_id=duplicate["id"])
            self.index.save_job(job)
            self.indexer.finish(job, "skipped", reason="duplicate")
            return {"ok": True, "duplicate": True, "document": duplicate, "chunks": None, "job_id": job["job_id"], "import_id": job["job_id"]}
        for other in self.index.jobs(states=("queued", "running")):
            if other.get("fingerprint") == digest:
                return {"ok": True, "queued": True, "job_id": other["job_id"], "import_id": other["job_id"], "name": name, "already": True}
        self.store_dir.mkdir(parents=True, exist_ok=True)
        target = self.store_dir / _safe_name(name)
        counter = 2
        while target.exists() and _hash_file(target) != digest:
            target = self.store_dir / f"{Path(_safe_name(name)).stem} ({counter}){Path(name).suffix}"
            counter += 1
        written = False
        if not target.exists():
            target.write_bytes(data)
            written = True
        result = self.import_path(target, provider=provider, copy=False, wait=wait)
        if wait and not result.get("ok") and written:
            # a file Studium could not read does not stay behind in the owner's library
            try:
                target.unlink()
            except OSError:
                pass
        return result

    def job_finished(self, job: dict[str, Any]) -> None:
        """Called by the indexer for every finished job: an upload Studium could not read does not stay behind in the owner's library."""

        if job.get("state") in {"failed", "cancelled"} and job.get("kind") == "import" and job.get("document_id"):
            placeholder = self.index.document(job["document_id"])
            if placeholder is not None and placeholder.get("status") == "processing":
                self.index.delete(job["document_id"])  # the file could not be read: no empty entry stays in Studium
        if job.get("state") != "failed" or job.get("kind") != "import" or job.get("provider") != "upload" or not job.get("path"):
            return
        path = Path(job["path"])
        try:
            path.resolve().relative_to(self.store_dir.resolve())
        except (ValueError, OSError):
            return  # never a file outside ZEUS's own upload folder
        if any(os.path.normcase(d.get("stored_path") or "") == os.path.normcase(str(path)) for d in self.index.documents(limit=100000)):
            return
        try:
            path.unlink()
        except OSError:
            pass

    def indexing(self) -> dict[str, Any]:
        """What the background indexer is doing now (the thin progress line under the Studium search)."""

        return self.indexer.summary()

    def index_states(self) -> dict[str, str]:
        """Per document: queued | processing | failed; documents without an open job are ready (the star view's quiet rings)."""

        states: dict[str, str] = {}
        for job in self.index.jobs(since=time.time() - 3600, states=("failed",)):
            if job.get("document_id"):
                states[job["document_id"]] = "failed"
        for job in self.index.jobs(states=("queued", "running")):
            if job.get("document_id"):
                states[job["document_id"]] = "processing" if job["state"] == "running" else "queued"
        return states

    # -- semantic embeddings -------------------------------------------------------------

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            if self._embeddings_arg is False:
                from study.embeddings import NoEmbeddings

                self._embeddings = NoEmbeddings("abgeschaltet")
            elif self._embeddings_arg is None:
                from study.embeddings import create_provider

                self._embeddings = create_provider()
            else:
                self._embeddings = self._embeddings_arg
        return self._embeddings

    def _can_embed(self) -> bool:
        provider = self.embeddings
        return getattr(provider, "id", "none") != "none"

    def _embed_document(self, doc_id: str, progress: Any = None) -> dict[str, Any]:
        """Vectors for every chunk of one document -- from the passage cache where the same text was embedded before."""

        from study.embeddings import passage_text

        provider = self.embeddings
        report: dict[str, Any] = {"documents": 0, "chunks": 0, "cached": 0, "error": ""}
        chunks = self.index.chunks_of(doc_id)
        if not chunks or not self._can_embed():
            return report
        texts = [passage_text(c["text"], title=c.get("title") or "", heading=c.get("heading") or "") for c in chunks]
        hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        cached = self.index.cached_vectors(provider.id, hashes)
        if progress is not None:
            progress.phase("embedding", embedding_chunks_total=len(chunks), embedding_chunks_completed=0)
        done = 0
        pending: list[int] = []
        for position, text_hash in enumerate(hashes):
            if text_hash in cached:
                continue
            pending.append(position)
        reuse = [i for i in range(len(chunks)) if hashes[i] in cached]
        if reuse:
            self.index.store_vectors(doc_id, provider.id, [chunks[i]["rowid"] for i in reuse], [cached[hashes[i]] for i in reuse], complete=not pending)
            done = len(reuse)
            report["cached"] = done
            if progress is not None:
                progress.update(force=True, embedding_chunks_completed=done)
        if pending and not provider.available():
            report["error"] = (provider.status() or {}).get("reason") or "keine Embeddings verfügbar"
            return report
        for start in range(0, len(pending), EMBED_BATCH):
            batch = pending[start:start + EMBED_BATCH]
            try:
                vectors = provider.embed_passages([texts[i] for i in batch])
            except Exception as exc:  # noqa: BLE001 - semantic search is an addition; keyword search keeps working
                report["error"] = f"{type(exc).__name__}: {exc}"
                return report
            self.index.cache_vectors(provider.id, [(hashes[i], vector) for i, vector in zip(batch, vectors)])
            done += len(batch)
            self.index.store_vectors(doc_id, provider.id, [chunks[i]["rowid"] for i in batch], vectors, complete=done >= len(chunks))
            if progress is not None:
                progress.update(embedding_chunks_completed=done)
        report["documents"] = 1
        report["chunks"] = done
        return report

    def embed_pending(self, *, limit_documents: int | None = None) -> dict[str, Any]:
        """Embed every indexed document that has no complete vector set for the current provider (synchronously)."""

        provider = self.embeddings
        report = {"provider": getattr(provider, "id", "none"), "documents": 0, "chunks": 0, "error": ""}
        if not self._can_embed() or not provider.available():
            report["error"] = (provider.status() or {}).get("reason") or "keine Embeddings verfügbar"
            return report
        pending = self.index.unembedded(provider.id)
        if limit_documents is not None:
            pending = pending[:limit_documents]
        for doc in pending:
            one = self._embed_document(doc["id"])
            if one.get("error"):
                report["error"] = one["error"]
                return report
            report["documents"] += one["documents"]
            report["chunks"] += one["chunks"]
        return report

    def backfill_embedding_jobs(self) -> int:
        """Documents indexed without vectors for the current provider (a new model, an earlier failure) get an embedding job."""

        if not self._can_embed():
            return 0
        queued = 0
        for doc in self.index.unembedded(self.embeddings.id):
            job = self.indexer.new_job(kind="embed", name=doc.get("filename") or doc.get("title") or "", document_id=doc["id"])
            if self.indexer.enqueue(job) is job:
                queued += 1
        return queued

    def schedule_embedding(self) -> None:
        """Kept for callers of the old embedder: the background indexer picks up documents without vectors."""

        self.indexer.wake()

    def _vector_matrix(self) -> Any:
        provider = self.embeddings
        key = (provider.id, self.index.version)
        with self._matrix_lock:
            if self._matrix is None or self._matrix_key != key:
                from study.embeddings import VectorMatrix

                rowids, vectors = self.index.vectors(provider.id)
                self._matrix, self._matrix_key = VectorMatrix(rowids, vectors), key
            return self._matrix

    def _semantic_hits(self, question: str, *, document_id: str = "") -> tuple[list[tuple[int, float]], float | None, Any]:
        provider = self.embeddings
        try:
            if isinstance(getattr(provider, "id", None), str) and provider.id == "none":
                return [], None, None
            matrix = self._vector_matrix()
            if not len(matrix) or not provider.available():
                return [], None, None
            vector = provider.embed_query(question)
        except Exception:  # noqa: BLE001 - keyword search answers alone
            return [], None, None
        import numpy as np

        similarities = matrix.vectors @ vector.astype(np.float32)
        median = float(np.median(similarities))
        allowed = None
        if document_id:
            allowed = {c["rowid"] for c in self.index.chunks_of(document_id)}
            if not allowed:
                return [], median, vector
        return matrix.top(vector, retrieval.SEMANTIC_TOP_K, allowed=allowed), median, vector

    def _semantic_focus(self, result: dict[str, Any], vector: Any) -> None:
        """A place found by meaning gets the sentence that carries the meaning as its focus: marked on the page, shown in the excerpt."""

        hit = result["best"]
        location = hit["location"]
        unit = self.index.unit(result["document"]["id"], int(location.get("unit") or 1))
        if not unit:
            return
        start, end = int(location.get("char_start") or 0), int(location.get("char_end") or 0) or len(unit["text"])
        passage = unit["text"][start:end]
        sentences = []
        for match in _SENTENCE.finditer(passage):
            sentence = match.group(0).strip()
            if 25 <= len(sentence) <= 400:
                sentences.append((start + match.start() + (len(match.group(0)) - len(match.group(0).lstrip())), sentence))
        if not sentences:
            return
        sentences = sentences[:14]
        try:
            import numpy as np

            vectors = self.embeddings.embed_passages([s for _, s in sentences])
            best = int(np.argmax(vectors @ vector.astype(np.float32)))
        except Exception:  # noqa: BLE001
            return
        at, sentence = sentences[best]
        hit["focus"] = sentence
        hit["phrases"] = [sentence]
        hit["at"] = at
        hit["excerpt"] = sentence
        hit["highlights"] = [[0, len(sentence)]]

    # -- connected sources ------------------------------------------------------------

    def connect_folder(self, path: str | Path, *, provider: str = "folder", label: str = "") -> dict[str, Any]:
        folder = Path(path)
        if not folder.is_dir():
            return {"ok": False, "error": "Ordner nicht gefunden"}
        source_id = hashlib.sha1(os.path.normcase(str(folder.resolve())).encode("utf-8")).hexdigest()[:12]
        source = {"id": source_id, "kind": "folder", "path": str(folder.resolve()), "provider": provider, "label": label or folder.name,
                  "added_at": time.time()}
        known = {s["id"]: s for s in self.index.sources()}
        if source_id in known:
            source["added_at"] = known[source_id]["added_at"]
        self.index.upsert_source(source)
        return {"ok": True, "source": source}

    def scan_source(self, source_id: str, *, background: bool | None = None) -> dict[str, Any]:
        """Walk a connected folder: new and changed files are imported (as background jobs in the product), unchanged ones skipped."""

        background = self.background if background is None else background
        source = next((s for s in self.index.sources() if s["id"] == source_id), None)
        if source is None:
            return {"ok": False, "error": "Quelle unbekannt"}
        folder = Path(source["path"])
        if not folder.is_dir():
            return {"ok": False, "error": "Ordner nicht erreichbar"}
        report = {"ok": True, "source_id": source_id, "found": 0, "indexed": 0, "queued": 0, "unchanged": 0, "failed": 0, "missing": 0, "moved": 0,
                  "native_goodnotes": 0, "errors": []}
        documents = [d for d in self.index.documents(limit=100000) if d.get("source_id") == source_id]
        known = {os.path.normcase(d["stored_path"]): d for d in documents}
        seen: set[str] = set()
        wait = not background
        with self._scan_lock:
            self.scanning[source_id] = report
            for current, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS and not d.startswith(".")]
                for name in files:
                    target = Path(current) / name
                    if name.lower().endswith(goodnotes.NATIVE_SUFFIX):
                        report["native_goodnotes"] += 1
                        continue
                    if not parsers.source_type_for(target) or name.startswith("~$"):
                        continue
                    report["found"] += 1
                    if report["found"] > MAX_SCAN_FILES:
                        break
                    key = os.path.normcase(str(target.resolve()))
                    seen.add(key)
                    existing = known.get(key)
                    try:
                        stat = target.stat()
                    except OSError:
                        continue
                    flags = (existing or {}).get("flags") or {}
                    if existing and abs(float(flags.get("mtime") or 0) - stat.st_mtime) < 1 and int(existing.get("size_bytes") or -1) == stat.st_size:
                        report["unchanged"] += 1
                        if flags.get("missing"):
                            self.index.set_flags(existing["id"], {"missing": False})
                        continue
                    result = self.import_path(target, provider=source.get("provider") or "folder", source_id=source_id, root=folder, wait=wait)
                    if result.get("queued"):
                        report["queued"] += 1
                    elif result.get("ok"):
                        report["indexed"] += 1
                    else:
                        report["failed"] += 1
                        report["errors"].append(f"{name}: {result.get('error')}")
            # a file that is gone: moved (the same content indexed again at its new place) or missing -- never deleted from disk
            for key, doc in known.items():
                if key in seen:
                    continue
                twin = next((d for d in self.index.documents(limit=100000)
                             if d["id"] != doc["id"] and d.get("content_hash") == doc.get("content_hash") and os.path.normcase(d["stored_path"]) in seen), None)
                if twin is not None:
                    self.index.delete(doc["id"])
                    report["moved"] += 1
                elif not Path(doc["stored_path"]).exists():
                    self.index.set_flags(doc["id"], {"missing": True})
                    report["missing"] += 1
            source["last_scan"] = time.time()
            source["files"] = report["found"]
            self.index.upsert_source(source)
            self.scanning.pop(source_id, None)
        report["errors"] = report["errors"][:20]
        return report

    def start_watching(self, *, interval: float = 120.0) -> None:
        """Connected folders (GoodNotes backups, synced course folders) are re-checked quietly; only new or changed files become jobs."""

        if self._watch_thread is not None and self._watch_thread.is_alive():
            return

        def loop() -> None:
            while True:
                time.sleep(interval)
                for source in self.index.sources():
                    try:
                        if Path(source["path"]).is_dir():
                            self.scan_source(source["id"], background=True)
                    except Exception:  # noqa: BLE001 - a flaky sync folder never stops the watcher
                        continue

        self._watch_thread = threading.Thread(target=loop, daemon=True, name="study-watch")
        self._watch_thread.start()

    def import_zeus_files(self) -> dict[str, Any]:
        """Everything in the ZEUS library (D:\\ZEUS_Wissen) Studium can read, connected as a source of its own."""

        if self.library_root is None or not self.library_root.is_dir():
            return {"ok": False, "error": "Keine ZEUS-Bibliothek gefunden"}
        connected = self.connect_folder(self.library_root, provider="zeus_files", label="ZEUS-Bibliothek")
        if not connected.get("ok"):
            return connected
        return self.scan_source(connected["source"]["id"])

    def remove_source(self, source_id: str, *, forget_documents: bool = False) -> dict[str, Any]:
        removed = 0
        if forget_documents:
            for doc in self.index.documents(limit=100000):
                if doc.get("source_id") == source_id:
                    removed += int(self.index.delete(doc["id"]))
        return {"ok": self.index.remove_source(source_id), "documents_removed": removed}

    # -- finding -----------------------------------------------------------------------

    def expansions_for(self, topic: str) -> list[str]:
        key = retrieval.fold(topic)
        if not key or self.expander is None:
            return []
        cached = self.index.expansion(key)
        if cached is not None:
            return cached
        try:
            terms = [str(t).strip() for t in (self.expander(topic) or []) if str(t).strip()][:8]
        except Exception:  # noqa: BLE001 - expansion is a bonus, never a requirement
            return []
        self.index.remember_expansion(key, terms)
        return terms

    def search(self, query: str, *, limit: int = 8, document_id: str = "", filters: dict[str, Any] | None = None,
               expand: bool = True, semantic: bool = True) -> dict[str, Any]:
        read = retrieval.read_query(query)
        # the command around a topic ("Zeig mir die Seite zum ...") is noise for meaning; a question is its own meaning
        question = read.topic if read.topic and read.topic != query.strip(" ?.!") and len(read.topic) < len(query) else query
        hits, median, vector = self._semantic_hits(question, document_id=document_id) if semantic else ([], None, None)
        matrix = self._matrix if vector is not None else None
        similarity_for = (lambda rowids: matrix.similarity_of(rowids, vector)) if matrix is not None else None
        direct = retrieval.search(self.index, query, limit=limit, document_id=document_id, filters=filters, semantic=hits, semantic_median=median,
                                  similarity_for=similarity_for)
        results = direct.get("results") or []
        if not (results and results[0]["best"].get("exact")) and expand and self.expander is not None and not results:
            # Nothing found by words or meaning: ask for the other names the topic goes by (cached per topic), then search again.
            expansions = self.expansions_for(direct.get("topic") or query)
            if expansions:
                widened = retrieval.search(self.index, query, limit=limit, expansions=expansions, document_id=document_id, filters=filters,
                                           semantic=hits, semantic_median=median, similarity_for=similarity_for)
                if len(widened.get("results") or []) > len(results):
                    direct, results = widened, widened.get("results") or []
        if vector is not None:
            for result in results[:3]:
                if result["best"].get("match") == "semantic":
                    self._semantic_focus(result, vector)
        direct["semantic"] = bool(hits)
        return direct

    # -- opening -------------------------------------------------------------------------

    def documents(self, *, limit: int = 200, order: str = "recent") -> list[dict[str, Any]]:
        return self.index.documents(limit=limit, order=order)

    def document(self, doc_id: str) -> dict[str, Any]:
        doc = self.index.document(doc_id)
        if doc is None:
            return {"ok": False, "error": "Dokument nicht gefunden", "reason": "missing_document"}
        stored = Path(doc["stored_path"])
        answer = {"ok": True, "document": doc, "units": self.index.units(doc_id), "viewer": self.viewer_kind(doc), "file_present": stored.is_file()}
        if doc["source_type"] == "pptx":
            answer["slides"] = self.slide_state(doc)
        return answer

    def viewer_kind(self, doc: dict[str, Any]) -> str:
        if doc["source_type"] == "pdf":
            return "pdf_pages" if self._engine_ok() else "pdf_native"
        if doc["source_type"] == "image":
            return "image"
        if doc["source_type"] == "pptx":
            return "slides"
        return "sections"

    def _engine_ok(self) -> bool:
        return self.engine is not None and bool(getattr(self.engine, "available", False))

    def unit(self, doc_id: str, number: int) -> dict[str, Any]:
        doc = self.index.document(doc_id)
        unit = self.index.unit(doc_id, number)
        if doc is None or unit is None:
            return {"ok": False, "error": "Stelle nicht gefunden"}
        self.index.touch_opened(doc_id)
        return {"ok": True, "document": doc, "unit": unit, "units": doc["units"],
                "figures": [{"number": f.get("number"), "kind": f.get("kind"), "box": f.get("box"), "caption": f.get("caption")}
                            for f in self.index.figures(doc_id, number)]}

    def sections(self, doc_id: str, *, max_chars: int = 2_000_000) -> dict[str, Any]:
        """The whole structured document (Word, Markdown, text): every section with its blocks, for continuous reading and the outline."""

        doc = self.index.document(doc_id)
        if doc is None:
            return {"ok": False, "error": "Dokument nicht gefunden"}
        units = []
        used = 0
        for meta in self.index.units(doc_id):
            unit = self.index.unit(doc_id, meta["number"]) or {}
            used += len(unit.get("text") or "")
            if used > max_chars:
                break
            units.append({"number": unit.get("number"), "kind": unit.get("kind"), "title": unit.get("title"), "blocks": unit.get("blocks") or []})
        self.index.touch_opened(doc_id)
        return {"ok": True, "document": doc, "units": units, "complete": len(units) == int(doc.get("units") or 0)}

    def page_image(self, doc_id: str, page: int, *, width: int = 1400) -> Path | None:
        return self.render_page(doc_id, page, width=width).get("path")

    #: page bitmaps on disk, bounded; the oldest are dropped first
    PAGE_CACHE_BYTES = 600 * 1024 * 1024

    def render_page(self, doc_id: str, page: int, *, width: int = 1400, clip: list[float] | None = None, session: str = "", seq: int = 0,
                    priority: int = 0) -> dict[str, Any]:
        """A page (or a region of it) as PNG: from the cache, or rendered -- unless the viewer that asked has already moved on.

        ``session``/``seq`` identify the viewer and its view generation (it rises with every page change or zoom; every request
        of one view shares it): a queued render of an older generation is dropped instead of occupying the renderer
        (flipping from page 20 to 80 to 21 renders 21, not all three).
        ``priority`` 0 is the visible page, 1 a neighbour fetched ahead."""

        doc = self.index.document(doc_id)
        if doc is None:
            return {"path": None}
        if doc["source_type"] == "pptx":
            return {"path": self.slide_image(doc, int(page))}
        if doc["source_type"] != "pdf" or not self._engine_ok():
            return {"path": None}
        stored = Path(doc["stored_path"])
        if not stored.is_file():
            return {"path": None, "missing": True}
        if clip:
            clip = [round(max(0.0, min(1.0, float(v))), 4) for v in clip[:4]]
            width = max(400, min(12000, int(width) // 200 * 200))
            name = f"{int(page)}_{width}_{'-'.join(str(v) for v in clip)}"
        else:
            width = max(300, min(3200, int(width) // 100 * 100))
            name = f"{int(page)}_{width}"
        mtime = int(stored.stat().st_mtime)
        target = self.data_dir / "cache" / "pages" / doc_id / f"{name}_{mtime}.png"
        if target.is_file():
            return {"path": target, "cached": True}
        if session:
            with self._render_meta_lock:
                latest = max(self._render_latest.get(session, 0), int(seq))
                self._render_latest[session] = latest
        with self._render_gate:
            if session:
                latest = self._render_latest.get(session, 0)
                if (priority == 0 and int(seq) < latest) or (priority > 0 and int(seq) < latest - 2):
                    return {"path": None, "superseded": True}
            if target.is_file():
                return {"path": target, "cached": True}
            target.parent.mkdir(parents=True, exist_ok=True)
            answer = self.engine.render(stored, int(page), target, width=width, clip=clip)
        self._renders += 1
        if self._renders % 40 == 0:
            threading.Thread(target=self._evict_page_cache, daemon=True, name="study-page-cache").start()
        return {"path": target if answer.get("ok") and target.is_file() else None}

    def _evict_page_cache(self) -> None:
        root = self.data_dir / "cache" / "pages"
        try:
            files = [(p.stat().st_mtime, p.stat().st_size, p) for p in root.rglob("*.png")]
        except OSError:
            return
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files):
            if total <= self.PAGE_CACHE_BYTES:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue

    # -- slides --------------------------------------------------------------------------

    def _slide_dir(self, doc: dict[str, Any]) -> Path:
        stored = Path(doc["stored_path"])
        mtime = int(stored.stat().st_mtime) if stored.is_file() else 0
        return self.data_dir / "cache" / "slides" / doc["id"] / str(mtime)

    def slide_state(self, doc: dict[str, Any]) -> dict[str, Any]:
        """ready | rendering | deferred | failed | unavailable -- and why."""

        folder = self._slide_dir(doc)
        count = int(doc.get("units") or 0)
        if count and all((folder / f"slide_{n}.png").is_file() for n in range(1, count + 1)):
            return {"state": "ready"}
        if self.slides is None or not self.slides.available():
            return {"state": "unavailable", "note": "Ohne PowerPoint zeigt Studium Folien als Text."}
        current = self.rendering.get(doc["id"], "")
        if current in {"rendering", "queued"}:
            return {"state": "rendering"}
        if current.startswith("deferred"):
            return {"state": "deferred", "note": "PowerPoint ist gerade geöffnet – die Folienbilder entstehen, sobald es geschlossen ist."}
        if current.startswith("failed"):
            return {"state": "failed", "note": current.partition(":")[2].strip() or "Die Folien konnten nicht als Bild erzeugt werden."}
        return {"state": "pending"}

    def slide_image(self, doc: dict[str, Any], number: int) -> Path | None:
        target = self._slide_dir(doc) / f"slide_{number}.png"
        if target.is_file():
            return target
        self.schedule_slides(doc["id"])
        return None

    def render_slides_now(self, doc_id: str) -> dict[str, Any]:
        doc = self.index.document(doc_id)
        if doc is None or doc["source_type"] != "pptx":
            return {"ok": False, "error": "keine Präsentation"}
        if self.slides is None or not self.slides.available():
            return {"ok": False, "error": "PowerPoint ist nicht verfügbar", "unavailable": True}
        stored = Path(doc["stored_path"])
        if not stored.is_file():
            return {"ok": False, "error": "Originaldatei fehlt", "reason": "missing"}
        folder = self._slide_dir(doc)
        with self._render_lock:
            count = int(doc.get("units") or 0)
            if count and all((folder / f"slide_{n}.png").is_file() for n in range(1, count + 1)):
                return {"ok": True, "cached": True}
            self.rendering[doc_id] = "rendering"
            folder.mkdir(parents=True, exist_ok=True)
            answer = self.slides.render_slides(stored, folder, width=1600)
            if answer.get("ok"):
                self.rendering.pop(doc_id, None)
            elif answer.get("deferred"):
                self.rendering[doc_id] = f"deferred:{time.time()}"
            else:
                self.rendering[doc_id] = f"failed: {answer.get('error') or ''}"
        return answer

    def schedule_slides(self, doc_id: str) -> None:
        state = self.rendering.get(doc_id, "")
        if self.slides is None or state in {"rendering", "queued"} or state.startswith("failed"):
            return
        if state.startswith("deferred") and time.time() - float(state.partition(":")[2] or 0) < 60:
            return  # PowerPoint was open a moment ago: ask again in a minute, not on every slide request
        self.rendering[doc_id] = "queued"
        threading.Thread(target=lambda: self.render_slides_now(doc_id), daemon=True, name=f"study-slides-{doc_id}").start()

    # -- highlights ------------------------------------------------------------------------

    def locate(self, doc_id: str, page: int, phrases: list[str], *, at: int | None = None) -> dict[str, Any]:
        """Every match of the phrases on a page or slide, as fractions of it (x, y, w, h), one primary -- for the viewer's highlights.

        The first phrase is the focus; ``at`` is the character offset the search result pointed at, which picks the
        primary occurrence when the focus stands on the page more than once."""

        doc = self.index.document(doc_id)
        if doc is None:
            return {"ok": False, "error": "Dokument nicht gefunden"}
        phrases = [str(p) for p in phrases if str(p or "").strip()][:8]
        unit = self.index.unit(doc_id, int(page)) or {}
        matches: list[dict[str, Any]] = []
        if doc["source_type"] == "pptx":
            matches = self._slide_matches(unit, phrases)
        elif (doc["source_type"] == "pdf" and unit.get("ocr")) or doc["source_type"] == "image":
            matches = self._ocr_matches(doc_id, int(page), phrases)
        elif doc["source_type"] == "pdf":
            if not self._engine_ok():
                return {"ok": False, "error": "Seitenmarkierung braucht den PDF-Renderer", "unavailable": True}
            answer = self.engine.locate(doc["stored_path"], int(page), phrases)
            if not answer.get("ok"):
                return {"ok": False, "error": answer.get("error", "nicht gefunden")}
            width, height = (answer.get("page_size") or [1, 1])
            for match in answer.get("matches") or [{"phrase": phrases[0] if phrases else "", "rank": 0, "start": -1, "rects": answer.get("rects") or []}]:
                matches.append({**match, "rects": [[r[0] / width, r[1] / height, r[2] / width, r[3] / height] for r in match.get("rects") or []]})
        else:
            return {"ok": False, "error": "nur für Seiten und Folien"}
        arranged = arrange(matches, at=at)
        return {"ok": True, "matches": arranged, "rects": [r for m in arranged for r in m["rects"]],
                "matched": list(dict.fromkeys(m["phrase"] for m in arranged))}

    def _ocr_matches(self, doc_id: str, page: int, phrases: list[str]) -> list[dict[str, Any]]:
        path = self.data_dir / "cache" / "ocr" / doc_id / f"ocr_{page}.json"
        try:
            words = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        matches = []
        for rank, found in enumerate(ocr.phrase_boxes(words, phrases)):
            for occurrence in found.get("occurrences") or []:
                matches.append({"phrase": found["phrase"], "rank": rank, "start": occurrence["start"], "rects": occurrence["rects"]})
        return matches

    @staticmethod
    def _slide_matches(unit: dict[str, Any], phrases: list[str]) -> list[dict[str, Any]]:
        """On a slide the shape is the place: the box of every shape whose text holds the phrase."""

        matches = []
        for rank, phrase in enumerate(phrases):
            needle = retrieval.fold(phrase)
            if len(needle) < 2:
                continue
            for block in unit.get("blocks") or []:
                box = block.get("box")
                folded, _index = retrieval.fold_with_index(block.get("text") or "")
                position = folded.find(needle)
                if box and position >= 0:
                    matches.append({"phrase": phrase, "rank": rank, "start": int(block.get("char_start") or 0) + position, "rects": [box]})
        return matches

    def original(self, doc_id: str) -> Path | None:
        doc = self.index.document(doc_id)
        if doc is None:
            return None
        path = Path(doc["stored_path"])
        return path if path.is_file() else None

    def set_categories(self, doc_id: str, changes: dict[str, str]) -> dict[str, Any]:
        doc = self.index.set_categories(doc_id, changes)
        return {"ok": doc is not None, "document": doc}

    def delete(self, doc_id: str, *, delete_file: bool = False) -> dict[str, Any]:
        """Remove a document from Studium (index, caches).  The file itself stays -- unless the owner explicitly asks
        (``delete_file``), and even then only a file inside ZEUS's own Studium folder, never a file in a connected source."""

        doc = self.index.document(doc_id)
        if doc is None:
            return {"ok": False, "error": "Dokument nicht gefunden"}
        for job in self.index.jobs(states=("queued", "running")):
            if job.get("document_id") == doc_id:
                self.indexer.cancel(job["job_id"])
        self.index.delete(doc_id)
        stored = Path(doc["stored_path"])
        removed_file = False
        refused = ""
        if delete_file:
            try:
                stored.resolve().relative_to(self.store_dir.resolve())
                if doc.get("source_id"):
                    refused = "Die Datei gehört zu einem verbundenen Ordner und bleibt dort."
                elif stored.is_file():
                    stored.unlink()
                    removed_file = True
            except ValueError:
                refused = "Die Datei liegt außerhalb des Studium-Ordners und bleibt unangetastet."
            except OSError as exc:
                refused = f"Die Datei ließ sich nicht löschen: {type(exc).__name__}"
        for cache in ("pages", "ocr", "slides"):
            shutil.rmtree(self.data_dir / "cache" / cache / doc_id, ignore_errors=True)
        self.rendering.pop(doc_id, None)
        return {"ok": True, "removed_file": removed_file, "file_kept": not removed_file, "file_note": refused}

    # -- context for questions -----------------------------------------------------------

    def context(self, doc_id: str, unit_number: int, *, selection: str = "", max_chars: int = 9000) -> dict[str, Any]:
        """Exactly the material a question about "this page" / "this section" may use: the unit, or only the selection."""

        doc = self.index.document(doc_id)
        unit = self.index.unit(doc_id, unit_number)
        if doc is None or unit is None:
            return {"ok": False, "error": "Stelle nicht gefunden"}
        from study.model import Location

        location = Location(page=unit_number if unit["kind"] == "page" else None, slide=unit_number if unit["kind"] == "slide" else None,
                            heading_path=[h for h in (unit.get("title") or "").split(" › ") if h] if unit["kind"] == "section" else [],
                            unit=unit_number)
        text = selection.strip() or unit["text"]
        return {"ok": True, "document": {"id": doc_id, "title": doc["title"], "filename": doc["filename"], "source_type": doc["source_type"]},
                "location": location.to_dict(), "scope": "selection" if selection.strip() else unit["kind"], "text": text[:max_chars],
                "truncated": len(text) > max_chars, "has_text": bool(text.strip())}

    def status(self) -> dict[str, Any]:
        provider = self.embeddings
        embeddings = dict(provider.status() or {})
        if getattr(provider, "id", "none") != "none":
            embeddings.update(self.index.embedding_stats(provider.id))
        indexing = self.indexer.summary()
        embeddings["working"] = indexing.get("current") or {}
        return {"ok": True, "stats": self.index.stats(), "sources": self.index.sources(), "scanning": dict(self.scanning),
                "pdf_renderer": self._engine_ok(), "ocr": ocr.status(), "goodnotes": {"direct_api": False},
                "embeddings": embeddings, "slides": {"available": bool(self.slides is not None and self.slides.available())},
                "indexing": indexing, "imports": indexing.get("jobs") or [],
                "formats": sorted(set(parsers.EXTENSIONS))}

    def galaxy(self) -> dict[str, Any]:
        """The "Sterne" view of the library: one point per document, clustered by subject (study.galaxy)."""

        from study import galaxy

        documents = self.index.documents(limit=100000)
        vectors_by_doc: dict[str, Any] | None = None
        provider = self.embeddings
        if isinstance(getattr(provider, "id", None), str) and provider.id != "none":
            try:
                rowids, matrix = self.index.vectors(provider.id)
                if rowids:
                    with self.index._lock:  # noqa: SLF001 - one read of the chunk -> document map, no public accessor for it
                        document_of = {int(r[0]): str(r[1]) for r in self.index._db.execute(  # noqa: SLF001
                            "SELECT chunk, document_id FROM vectors WHERE model=?", (provider.id,))}
                    vectors_by_doc = galaxy.centroids(rowids, matrix, document_of)
            except Exception:  # noqa: BLE001 - the sky is drawn without semantic neighbours
                vectors_by_doc = None
        states = None
        index_states = getattr(self, "index_states", None)
        if callable(index_states):
            try:
                states = index_states()
            except Exception:  # noqa: BLE001 - unknown states read as ready
                states = None
        return galaxy.build(documents, vectors_by_doc=vectors_by_doc, index_states=states if isinstance(states, dict) else None)
