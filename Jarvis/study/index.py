"""The study index: documents, units and chunks in SQLite, searchable by two FTS5 indexes.

- ``chunks_word``    unicode61 tokens, diacritics folded, BM25 ranked (exact names, prefixes)
- ``chunks_tri``     trigram tokens (substrings: German compounds, hyphenation, partial words)

A chunk never loses where it came from: page, slide, heading path, paragraph and the
character span inside its unit.  Units keep their full text, size and image count, so
the viewer can show a page and a text panel without re-reading the file.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from study.model import NormalizedDocument, Unit

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, filename TEXT, title TEXT, source_type TEXT, stored_path TEXT, original_path TEXT,
  provider TEXT, source_id TEXT, course TEXT, subject TEXT, semester TEXT, module TEXT, topic TEXT,
  confirmed TEXT DEFAULT '[]', units INTEGER, text_units INTEGER, status TEXT, flags TEXT DEFAULT '{}',
  content_hash TEXT, size_bytes INTEGER, created_at REAL, updated_at REAL, indexed_at REAL, opened_at REAL
);
CREATE TABLE IF NOT EXISTS units (
  document_id TEXT, number INTEGER, kind TEXT, title TEXT, text TEXT, width REAL, height REAL, images INTEGER,
  has_text INTEGER, ocr INTEGER, blocks TEXT, PRIMARY KEY (document_id, number)
);
CREATE TABLE IF NOT EXISTS chunks (
  rowid INTEGER PRIMARY KEY, document_id TEXT, ordinal INTEGER, unit INTEGER, page INTEGER, slide INTEGER,
  heading TEXT, paragraph INTEGER, char_start INTEGER, char_end INTEGER, text TEXT, figure INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_word USING fts5(text, heading, title, tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5(text, heading, title, tokenize='trigram remove_diacritics 1');
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, kind TEXT, path TEXT, provider TEXT, label TEXT, added_at REAL, last_scan REAL, files INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS expansions (key TEXT PRIMARY KEY, terms TEXT, at REAL);
CREATE TABLE IF NOT EXISTS vectors (chunk INTEGER, model TEXT, document_id TEXT, dim INTEGER, vec BLOB, PRIMARY KEY (chunk, model));
CREATE INDEX IF NOT EXISTS vectors_doc ON vectors(document_id, model);
CREATE TABLE IF NOT EXISTS embedded (document_id TEXT, model TEXT, chunks INTEGER, at REAL, PRIMARY KEY (document_id, model));
CREATE TABLE IF NOT EXISTS vector_cache (model TEXT, text_hash TEXT, dim INTEGER, vec BLOB, at REAL, PRIMARY KEY (model, text_hash));
CREATE TABLE IF NOT EXISTS index_jobs (
  job_id TEXT PRIMARY KEY, kind TEXT, document_id TEXT, name TEXT, path TEXT, provider TEXT, source_id TEXT, root TEXT, fingerprint TEXT,
  state TEXT, phase TEXT, reason TEXT, error TEXT, source_type TEXT,
  pages_total INTEGER DEFAULT 0, pages_completed INTEGER DEFAULT 0, slides_total INTEGER DEFAULT 0, slides_completed INTEGER DEFAULT 0,
  ocr_pages_total INTEGER DEFAULT 0, ocr_pages_completed INTEGER DEFAULT 0, chunks_total INTEGER DEFAULT 0, chunks_completed INTEGER DEFAULT 0,
  embedding_chunks_total INTEGER DEFAULT 0, embedding_chunks_completed INTEGER DEFAULT 0, current_page INTEGER DEFAULT 0,
  created_at REAL, started_at REAL, updated_at REAL, finished_at REAL, attempts INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS index_jobs_state ON index_jobs(state, created_at);
CREATE TABLE IF NOT EXISTS figures (
  document_id TEXT, unit INTEGER, number INTEGER, kind TEXT, box TEXT, caption TEXT, text TEXT, PRIMARY KEY (document_id, unit, number)
);
"""

JOB_COUNTERS = ("pages_total", "pages_completed", "slides_total", "slides_completed", "ocr_pages_total", "ocr_pages_completed", "chunks_total",
                "chunks_completed", "embedding_chunks_total", "embedding_chunks_completed", "current_page")
JOB_FIELDS = ("kind", "document_id", "name", "path", "provider", "source_id", "root", "fingerprint", "state", "phase", "reason", "error",
              "source_type", *JOB_COUNTERS, "created_at", "started_at", "updated_at", "finished_at", "attempts")

CHUNK_TARGET = 700
CHUNK_OVERLAP = 120
_FIGURE = re.compile(r"(?i)\b(abb\.|abbildung|grafik|schema|schaubild|diagramm|fig\.|figure|tabelle)\b")


def _windows(text: str, *, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """Character spans of about ``target`` characters, cut at a sentence or line end where one is near."""

    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + target)
        if end < length:
            window = text[start:end + 160]
            cut = max(window.rfind(". ", target - 220), window.rfind("\n", target - 220), window.rfind("? ", target - 220))
            if cut > 200:
                end = start + cut + 1
        spans.append((start, end))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
        while start < length and not text[start - 1].isspace() and start < end:
            start += 1
    return spans


def chunk_unit(unit: Unit) -> list[dict[str, Any]]:
    """The searchable pieces of a unit, each with its exact location."""

    chunks: list[dict[str, Any]] = []
    if not unit.text.strip():
        return chunks
    page = unit.number if unit.kind == "page" else None
    slide = unit.number if unit.kind == "slide" else None
    if unit.kind in {"page", "slide"}:
        for start, end in _windows(unit.text):
            piece = unit.text[start:end].strip()
            if piece:
                chunks.append({"unit": unit.number, "page": page, "slide": slide, "heading": unit.title,
                               "paragraph": None, "char_start": start, "char_end": end, "text": piece})
        return chunks
    # sections: consecutive blocks grouped up to the target size, located at their first paragraph
    group: list[Any] = []

    def flush() -> None:
        if not group:
            return
        start, end = group[0].char_start, group[-1].char_end
        piece = unit.text[start:end].strip()
        if piece:
            first = next((b for b in group if b.paragraph is not None), group[0])
            chunks.append({"unit": unit.number, "page": None, "slide": None, "heading": " › ".join(first.heading_path),
                           "paragraph": first.paragraph, "char_start": start, "char_end": end, "text": piece})
        group.clear()

    size = 0
    for block in unit.blocks:
        if group and size + len(block.text) > CHUNK_TARGET + 200:
            flush()
            size = 0
        group.append(block)
        size += len(block.text)
    flush()
    return chunks


def _figure_chunks(unit: Unit) -> list[dict[str, Any]]:
    """A figure is findable by its caption and the text around it, located at its region (study.figures)."""

    if not getattr(unit, "figures", None):
        return []
    try:
        from study.figures import figure_chunks
    except ImportError:
        return []
    try:
        # a scanned or photographed page is one big image: the page itself is the place, not a "figure" framing all of it
        return [c for c in figure_chunks(unit) if c.get("kind") != "scan"]
    except Exception:  # noqa: BLE001 - a figure never breaks indexing the page
        return []


def _ocr_quality(unit: Unit, start: int, end: int) -> tuple[float | None, bool]:
    """Mean OCR confidence of the recognised blocks inside a chunk's span (length-weighted), and whether they are handwriting."""

    if not unit.ocr:
        return None, False
    weight = 0.0
    total = 0.0
    handwriting = False
    for block in unit.blocks:
        overlap = min(end, block.char_end) - max(start, block.char_start)
        confidence = getattr(block, "confidence", None)
        if overlap <= 0:
            continue
        handwriting = handwriting or block.kind == "handwriting"
        if confidence is not None:
            weight += overlap
            total += overlap * float(confidence)
    return (round(total / weight, 3) if weight else None), handwriting


class StudyIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self.version = 0   # bumps whenever chunks or vectors change: in-memory vector matrices reload on a new version
        with self._lock:
            self._db.executescript(SCHEMA)
            # older index files: chunk columns added later (figure number, OCR confidence, handwriting) -- no rebuild needed
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(chunks)")}
            for name, decl in (("figure_no", "INTEGER DEFAULT 0"), ("ocr_confidence", "REAL"), ("handwriting", "INTEGER DEFAULT 0")):
                if name not in columns:
                    self._db.execute(f"ALTER TABLE chunks ADD COLUMN {name} {decl}")
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writing --------------------------------------------------------------------

    def upsert(self, meta: dict[str, Any], doc: NormalizedDocument) -> int:
        """Replace the document's units and chunks; returns the number of chunks indexed."""

        doc_id = meta["id"]
        now = time.time()
        with self._lock:
            db = self._db
            existing = db.execute("SELECT created_at, confirmed, course, subject, semester, module, topic, opened_at FROM documents WHERE id=?",
                                  (doc_id,)).fetchone()
            confirmed = json.loads(existing["confirmed"]) if existing else []
            categories = {key: meta.get(key, "") for key in ("course", "subject", "semester", "module", "topic")}
            if existing:
                for key in confirmed:
                    categories[key] = existing[key]
            self._delete_chunks(doc_id)
            db.execute("DELETE FROM units WHERE document_id=?", (doc_id,))
            db.execute(
                "INSERT OR REPLACE INTO documents (id, filename, title, source_type, stored_path, original_path, provider, source_id, course, subject,"
                " semester, module, topic, confirmed, units, text_units, status, flags, content_hash, size_bytes, created_at, updated_at, indexed_at, opened_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, meta.get("filename", ""), meta.get("title") or doc.title, doc.source_type, meta.get("stored_path", ""),
                 meta.get("original_path", ""), meta.get("provider", "upload"), meta.get("source_id", ""), categories["course"], categories["subject"],
                 categories["semester"], categories["module"], categories["topic"], json.dumps(confirmed), len(doc.units), doc.text_units,
                 meta.get("status", "indexed"), json.dumps(meta.get("flags") or {}, ensure_ascii=False), meta.get("content_hash", ""),
                 int(meta.get("size_bytes") or 0), existing["created_at"] if existing else float(meta.get("created_at") or now),
                 float(meta.get("updated_at") or now), now, existing["opened_at"] if existing else None))
            title = meta.get("title") or doc.title
            count = 0
            for unit in doc.units:
                blocks = [{"text": b.text[:4000], "kind": b.kind, "heading_path": b.heading_path, "paragraph": b.paragraph,
                           "char_start": b.char_start, "char_end": b.char_end, "box": b.box,
                           "confidence": b.confidence, "role": b.role} for b in unit.blocks]
                db.execute("INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (doc_id, unit.number, unit.kind, unit.title, unit.text, unit.width, unit.height, unit.images,
                            int(unit.has_text), int(unit.ocr), json.dumps(blocks, ensure_ascii=False)))
                figures = list(getattr(unit, "figures", None) or [])
                for number, figure_data in enumerate(figures, start=1):
                    db.execute("INSERT OR REPLACE INTO figures VALUES (?,?,?,?,?,?,?)",
                               (doc_id, unit.number, int(figure_data.get("index") or number), str(figure_data.get("kind") or "figure"),
                                json.dumps(figure_data.get("box")), str(figure_data.get("caption") or ""), str(figure_data.get("text") or "")[:2000]))
                for chunk in chunk_unit(unit) + _figure_chunks(unit):
                    figure_no = int(chunk.get("figure_index") or 0)
                    figure = int(bool(figure_no) or unit.images > 0 or bool(_FIGURE.search(chunk["text"])))
                    confidence, handwriting = _ocr_quality(unit, int(chunk["char_start"]), int(chunk["char_end"]))
                    cursor = db.execute(
                        "INSERT INTO chunks (document_id, ordinal, unit, page, slide, heading, paragraph, char_start, char_end, text, figure,"
                        " figure_no, ocr_confidence, handwriting) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (doc_id, count, chunk["unit"], chunk["page"], chunk["slide"], chunk["heading"], chunk["paragraph"],
                         chunk["char_start"], chunk["char_end"], chunk["text"], figure, figure_no, confidence, int(handwriting)))
                    rowid = cursor.lastrowid
                    db.execute("INSERT INTO chunks_word (rowid, text, heading, title) VALUES (?,?,?,?)", (rowid, chunk["text"], chunk["heading"], title))
                    db.execute("INSERT INTO chunks_tri (rowid, text, heading, title) VALUES (?,?,?,?)", (rowid, chunk["text"], chunk["heading"], title))
                    count += 1
            db.commit()
            return count

    def _delete_chunks(self, doc_id: str) -> None:
        rows = [r["rowid"] for r in self._db.execute("SELECT rowid FROM chunks WHERE document_id=?", (doc_id,))]
        for rowid in rows:
            self._db.execute("DELETE FROM chunks_word WHERE rowid=?", (rowid,))
            self._db.execute("DELETE FROM chunks_tri WHERE rowid=?", (rowid,))
        self._db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        self._db.execute("DELETE FROM figures WHERE document_id=?", (doc_id,))
        # chunk rowids are new after every re-index: their vectors are stale by construction
        self._db.execute("DELETE FROM vectors WHERE document_id=?", (doc_id,))
        self._db.execute("DELETE FROM embedded WHERE document_id=?", (doc_id,))
        self.version += 1

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            self._delete_chunks(doc_id)
            self._db.execute("DELETE FROM units WHERE document_id=?", (doc_id,))
            gone = self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0
            self._db.commit()
            return gone

    def set_categories(self, doc_id: str, changes: dict[str, str]) -> dict[str, Any] | None:
        allowed = {k: str(v).strip()[:120] for k, v in changes.items() if k in {"course", "subject", "semester", "module", "topic", "title"}}
        with self._lock:
            row = self._db.execute("SELECT confirmed FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                return None
            confirmed = set(json.loads(row["confirmed"] or "[]"))
            for key, value in allowed.items():
                self._db.execute(f"UPDATE documents SET {key}=? WHERE id=?", (value, doc_id))  # noqa: S608 - key is whitelisted
                if key != "title":
                    confirmed.add(key)
            self._db.execute("UPDATE documents SET confirmed=? WHERE id=?", (json.dumps(sorted(confirmed)), doc_id))
            self._db.commit()
        return self.document(doc_id)

    def set_flags(self, doc_id: str, changes: dict[str, Any]) -> None:
        with self._lock:
            row = self._db.execute("SELECT flags FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                return
            flags = json.loads(row["flags"] or "{}")
            flags.update(changes)
            self._db.execute("UPDATE documents SET flags=? WHERE id=?", (json.dumps(flags, ensure_ascii=False), doc_id))
            self._db.commit()

    def touch_opened(self, doc_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE documents SET opened_at=? WHERE id=?", (time.time(), doc_id))
            self._db.commit()

    # -- reading --------------------------------------------------------------------

    @staticmethod
    def _doc_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["confirmed"] = json.loads(data.get("confirmed") or "[]")
        data["flags"] = json.loads(data.get("flags") or "{}")
        return data

    def document(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._doc_row(row) if row else None

    def documents(self, *, limit: int = 500, order: str = "recent") -> list[dict[str, Any]]:
        order_sql = {"recent": "COALESCE(opened_at, 0) DESC, updated_at DESC", "title": "title COLLATE NOCASE"}.get(order, "updated_at DESC")
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM documents ORDER BY {order_sql} LIMIT ?", (int(limit),)).fetchall()  # noqa: S608
        return [self._doc_row(r) for r in rows]

    def find_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
        return self._doc_row(row) if row else None

    def unit(self, doc_id: str, number: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM units WHERE document_id=? AND number=?", (doc_id, int(number))).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["blocks"] = json.loads(data.get("blocks") or "[]")
        data["has_text"] = bool(data["has_text"])
        data["ocr"] = bool(data["ocr"])
        return data

    def units(self, doc_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT number, kind, title, has_text, images, width, height, length(text) AS chars FROM units"
                                    " WHERE document_id=? ORDER BY number", (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    def chunks(self, rowids: list[int]) -> dict[int, dict[str, Any]]:
        if not rowids:
            return {}
        marks = ",".join("?" for _ in rowids)
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM chunks WHERE rowid IN ({marks})", rowids).fetchall()  # noqa: S608
        return {r["rowid"]: dict(r) for r in rows}

    def match_word(self, query: str, limit: int = 120) -> list[tuple[int, float]]:
        if not query:
            return []
        with self._lock:
            try:
                rows = self._db.execute("SELECT rowid, bm25(chunks_word, 1.0, 2.2, 1.6) AS s FROM chunks_word WHERE chunks_word MATCH ? ORDER BY s LIMIT ?",
                                        (query, int(limit))).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(int(r["rowid"]), float(r["s"])) for r in rows]

    def match_trigram(self, query: str, limit: int = 120) -> list[tuple[int, float]]:
        if not query:
            return []
        with self._lock:
            try:
                rows = self._db.execute("SELECT rowid, bm25(chunks_tri, 1.0, 2.0, 1.4) AS s FROM chunks_tri WHERE chunks_tri MATCH ? ORDER BY s LIMIT ?",
                                        (query, int(limit))).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(int(r["rowid"]), float(r["s"])) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            docs = self._db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(units),0) AS u FROM documents").fetchone()
            chunks = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            by_type = {r["source_type"]: r["n"] for r in self._db.execute("SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type")}
        return {"documents": docs["n"], "units": docs["u"], "chunks": chunks["n"], "by_type": by_type}

    # -- vectors ------------------------------------------------------------------------

    def unembedded(self, model: str) -> list[dict[str, Any]]:
        """Documents with chunks but no complete vector set for ``model``, most recently used first."""

        with self._lock:
            rows = self._db.execute(
                "SELECT d.id, d.title, d.filename, (SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.id) AS chunks FROM documents d"
                " WHERE NOT EXISTS (SELECT 1 FROM embedded e WHERE e.document_id=d.id AND e.model=?)"
                " ORDER BY COALESCE(d.opened_at, 0) DESC, d.indexed_at DESC", (model,)).fetchall()
        return [dict(r) for r in rows if r["chunks"]]

    def recognised_chunks(self, *, limit: int = 4000, document_id: str = "") -> list[tuple[int, str]]:
        """Chunks whose text came from OCR or handwriting recognition (most recently indexed first)."""

        sql = ("SELECT c.rowid, c.text FROM chunks c JOIN documents d ON d.id=c.document_id"
               " WHERE (c.ocr_confidence IS NOT NULL OR c.handwriting=1)")
        args: list[Any] = []
        if document_id:
            sql += " AND c.document_id=?"
            args.append(document_id)
        sql += " ORDER BY d.indexed_at DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            return [(int(r["rowid"]), r["text"]) for r in self._db.execute(sql, args)]

    def chunks_of(self, doc_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT c.rowid, c.text, c.heading, d.title FROM chunks c JOIN documents d ON d.id=c.document_id"
                                    " WHERE c.document_id=? ORDER BY c.ordinal", (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    def store_vectors(self, doc_id: str, model: str, rowids: list[int], vectors: Any, *, complete: bool) -> int:
        """Vectors for chunks of one document.  Rows whose chunk vanished meanwhile (a re-index) are dropped, never stored stale."""

        import numpy as np

        array = np.asarray(vectors, dtype=np.float32)
        stored = 0
        with self._lock:
            alive = {r["rowid"] for r in self._db.execute("SELECT rowid FROM chunks WHERE document_id=?", (doc_id,))}
            for rowid, vector in zip(rowids, array):
                if rowid not in alive:
                    continue
                self._db.execute("INSERT OR REPLACE INTO vectors (chunk, model, document_id, dim, vec) VALUES (?,?,?,?,?)",
                                 (int(rowid), model, doc_id, int(array.shape[1]), vector.tobytes()))
                stored += 1
            if complete and alive:
                # complete means: every chunk the document has now carries a vector for this model -- not just this batch
                have = {r["chunk"] for r in self._db.execute("SELECT chunk FROM vectors WHERE document_id=? AND model=?", (doc_id, model))}
                if alive <= have:
                    self._db.execute("INSERT OR REPLACE INTO embedded (document_id, model, chunks, at) VALUES (?,?,?,?)", (doc_id, model, len(alive), time.time()))
            self._db.commit()
            self.version += 1
        return stored

    def vectors(self, model: str) -> tuple[list[int], Any]:
        import numpy as np

        with self._lock:
            rows = self._db.execute("SELECT chunk, dim, vec FROM vectors WHERE model=? ORDER BY chunk", (model,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        dim = int(rows[0]["dim"])
        matrix = np.frombuffer(b"".join(r["vec"] for r in rows if r["dim"] == dim), dtype=np.float32).reshape(-1, dim)
        return [int(r["chunk"]) for r in rows if r["dim"] == dim], matrix

    def embedding_stats(self, model: str) -> dict[str, Any]:
        with self._lock:
            docs = self._db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            done = self._db.execute("SELECT COUNT(*) AS n FROM embedded WHERE model=?", (model,)).fetchone()["n"]
            vectors = self._db.execute("SELECT COUNT(*) AS n FROM vectors WHERE model=?", (model,)).fetchone()["n"]
        return {"documents": docs, "embedded_documents": done, "vectors": vectors}

    def cached_vectors(self, model: str, hashes: list[str]) -> dict[str, Any]:
        """Vectors already computed for exactly these passage texts (by hash): a re-import or a resumed job embeds nothing twice."""

        import numpy as np

        if not hashes:
            return {}
        found: dict[str, Any] = {}
        with self._lock:
            for start in range(0, len(hashes), 400):
                part = hashes[start:start + 400]
                marks = ",".join("?" for _ in part)
                for row in self._db.execute(f"SELECT text_hash, dim, vec FROM vector_cache WHERE model=? AND text_hash IN ({marks})",  # noqa: S608
                                            [model, *part]):
                    found[row["text_hash"]] = np.frombuffer(row["vec"], dtype=np.float32)
        return found

    def cache_vectors(self, model: str, items: list[tuple[str, Any]]) -> None:
        import numpy as np

        with self._lock:
            now = time.time()
            for text_hash, vector in items:
                array = np.asarray(vector, dtype=np.float32)
                self._db.execute("INSERT OR REPLACE INTO vector_cache (model, text_hash, dim, vec, at) VALUES (?,?,?,?,?)",
                                 (model, text_hash, int(array.shape[0]), array.tobytes(), now))
            self._db.commit()

    # -- index jobs -----------------------------------------------------------------------

    def save_job(self, job: dict[str, Any]) -> None:
        values = [job.get(key) for key in JOB_FIELDS]
        with self._lock:
            self._db.execute(f"INSERT OR REPLACE INTO index_jobs (job_id, {', '.join(JOB_FIELDS)}) VALUES (?, {', '.join('?' for _ in JOB_FIELDS)})",  # noqa: S608
                             [job["job_id"], *values])
            self._db.commit()

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM index_jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def jobs(self, *, states: tuple[str, ...] | None = None, since: float = 0.0, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM index_jobs WHERE 1=1"
        args: list[Any] = []
        if states:
            sql += f" AND state IN ({','.join('?' for _ in states)})"
            args += list(states)
        if since:
            sql += " AND COALESCE(finished_at, updated_at, created_at) >= ?"
            args.append(since)
        sql += " ORDER BY created_at LIMIT ?"
        args.append(int(limit))
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args)]

    def prune_jobs(self, *, older_than: float) -> None:
        with self._lock:
            self._db.execute("DELETE FROM index_jobs WHERE state IN ('ready','failed','cancelled','skipped') AND COALESCE(finished_at, 0) < ?", (older_than,))
            self._db.commit()

    # -- figures --------------------------------------------------------------------------

    def figures(self, doc_id: str, unit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if unit is None:
                rows = self._db.execute("SELECT * FROM figures WHERE document_id=? ORDER BY unit, number", (doc_id,)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM figures WHERE document_id=? AND unit=? ORDER BY number", (doc_id, int(unit))).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["box"] = json.loads(data.get("box") or "null")
            out.append(data)
        return out

    # -- sources and expansions ----------------------------------------------------------

    def upsert_source(self, source: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO sources (id, kind, path, provider, label, added_at, last_scan, files) VALUES (?,?,?,?,?,?,?,?)",
                             (source["id"], source.get("kind", "folder"), source.get("path", ""), source.get("provider", "folder"),
                              source.get("label", ""), float(source.get("added_at") or time.time()), source.get("last_scan"), int(source.get("files") or 0)))
            self._db.commit()

    def sources(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM sources ORDER BY added_at")]

    def remove_source(self, source_id: str) -> bool:
        with self._lock:
            gone = self._db.execute("DELETE FROM sources WHERE id=?", (source_id,)).rowcount > 0
            self._db.commit()
            return gone

    def expansion(self, key: str) -> list[str] | None:
        with self._lock:
            row = self._db.execute("SELECT terms FROM expansions WHERE key=?", (key,)).fetchone()
        return json.loads(row["terms"]) if row else None

    def remember_expansion(self, key: str, terms: list[str]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO expansions VALUES (?,?,?)", (key, json.dumps(terms, ensure_ascii=False), time.time()))
            self._db.commit()
