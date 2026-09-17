"""The PDF page worker: renders pages and finds text rectangles with Qt's PDF engine (PySide6.QtPdf).

Runs as its own process (``python -m study.pdf_worker``) so Qt never lives inside the
server process.  Protocol: one JSON object per line on stdin, one JSON answer per line
on stdout.

    {"op": "info",   "path": P}                                   -> pages, sizes (points), title/producer/creator
    {"op": "text",   "path": P}                                   -> text of every page
    {"op": "render", "path": P, "page": N, "width": PX, "out": F} -> PNG written to F
    {"op": "locate", "path": P, "page": N, "phrases": [...]}      -> rectangles (points) of each phrase found

Pages are 1-based on the wire.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import OrderedDict
from typing import Any


def _normalized(text: str) -> tuple[str, list[int]]:
    """Lower-cased text with whitespace runs collapsed and hyphen-breaks joined, plus a map back to raw indices."""

    out: list[str] = []
    index: list[int] = []
    previous_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if previous_space or not out:
                continue
            out.append(" ")
            index.append(i)
            previous_space = True
            continue
        previous_space = False
        out.append(ch.lower())
        index.append(i)
    return "".join(out), index


def _phrase_key(phrase: str) -> str:
    return re.sub(r"\s+", " ", phrase.strip().lower())


def _occurrences(normalized: str, key: str, *, limit: int = 40) -> list[tuple[int, int]]:
    """Every (start, end) of ``key`` in the normalized page text; a word split by a line-end hyphen ("Mecha- nismus") still counts."""

    spans: list[tuple[int, int]] = []
    at = normalized.find(key)
    while at >= 0 and len(spans) < limit:
        spans.append((at, at + len(key)))
        at = normalized.find(key, at + len(key))
    if spans:
        return spans
    pieces = [re.escape(ch) if not ch.isspace() else r"\s" for ch in key]
    pattern = re.compile(r"(?:-\s|­\s?)?".join(pieces))
    return [(m.start(), m.end()) for m in pattern.finditer(normalized)][:limit]


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtPdf import QPdfDocument

    app = QGuiApplication([])  # noqa: F841 - QtPdf rendering needs the application object
    documents: OrderedDict[str, Any] = OrderedDict()

    def document(path: str) -> Any:
        # The file is read into memory and loaded from there: the worker never holds a handle on the owner's file,
        # so a PDF can be deleted, replaced or synced (OneDrive, GoodNotes backups) while it is open in Studium.
        stamp = os.stat(path).st_mtime_ns
        cached = documents.get(path)
        if cached is not None and cached[2] == stamp:
            documents.move_to_end(path)
            return cached[0]
        if cached is not None:
            documents.pop(path)[0].close()
        with open(path, "rb") as handle:
            data = QByteArray(handle.read())
        buffer = QBuffer()
        buffer.setData(data)
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        doc = QPdfDocument(None)
        doc.load(buffer)
        if doc.status() != QPdfDocument.Status.Ready:
            raise RuntimeError(f"cannot open PDF ({doc.error()})")
        documents[path] = (doc, buffer, stamp)
        while len(documents) > 2:  # bounded: two open documents in memory at most
            _, old = documents.popitem(last=False)
            old[0].close()
        return doc

    def page_text(doc: Any, number: int) -> str:
        return doc.getAllText(number - 1).text() or ""

    def handle(request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "ping":
            return {"ok": True}
        path = str(request.get("path") or "")
        doc = document(path)
        if op == "info":
            sizes = []
            for i in range(doc.pageCount()):
                size = doc.pagePointSize(i)
                sizes.append([round(size.width(), 2), round(size.height(), 2)])
            meta = {}
            for key, field in (("title", QPdfDocument.MetaDataField.Title), ("producer", QPdfDocument.MetaDataField.Producer),
                               ("creator", QPdfDocument.MetaDataField.Creator), ("author", QPdfDocument.MetaDataField.Author)):
                try:
                    meta[key] = str(doc.metaData(field) or "")
                except Exception:  # noqa: BLE001 - metadata is optional
                    meta[key] = ""
            return {"ok": True, "pages": doc.pageCount(), "sizes": sizes, "meta": meta}
        if op == "text":
            return {"ok": True, "pages": [page_text(doc, n) for n in range(1, doc.pageCount() + 1)]}
        if op == "render":
            # the whole page at ``width`` pixels -- or only ``clip`` ([x, y, w, h] as page fractions) of the page rendered at ``width``,
            # so zooming in re-renders just the visible region sharply instead of a huge page bitmap
            number = int(request.get("page") or 1)
            size = doc.pagePointSize(number - 1)
            clip = request.get("clip")
            width = max(200, min(12000 if clip else 3200, int(request.get("width") or 1400)))
            height = int(width * size.height() / max(1.0, size.width()))
            out = str(request.get("out") or "")
            if clip:
                from PySide6.QtCore import QRect
                from PySide6.QtPdf import QPdfDocumentRenderOptions

                x, y, w, h = (max(0.0, min(1.0, float(v))) for v in clip)
                rect = QRect(int(x * width), int(y * height), max(1, int(w * width)), max(1, int(h * height)))
                if rect.width() * rect.height() > 3200 * 3200:
                    raise RuntimeError("clip too large")
                options = QPdfDocumentRenderOptions()
                options.setScaledSize(QSize(width, height))
                options.setScaledClipRect(rect)
                image = doc.render(number - 1, QSize(rect.width(), rect.height()), options)
            else:
                image = doc.render(number - 1, QSize(width, height))
            if not out or not image.save(out, "PNG"):
                raise RuntimeError("render failed")
            return {"ok": True, "out": out, "width": width, "height": height, "clip": clip}
        if op == "locate":
            # every occurrence of every phrase, each with the text runs it covers; the caller merges and ranks them
            number = int(request.get("page") or 1)
            raw = page_text(doc, number)
            folded, index = _normalized(raw)
            rects: list[list[float]] = []
            matched: list[str] = []
            matches: list[dict[str, Any]] = []
            for rank, phrase in enumerate(request.get("phrases") or []):
                key = _phrase_key(str(phrase))
                if len(key) < 2:
                    continue
                found_any = False
                for start, end in _occurrences(folded, key):
                    raw_start, raw_end = index[start], index[end - 1] + 1
                    selection = doc.getSelectionAtIndex(number - 1, raw_start, raw_end - raw_start)
                    boxes = []
                    for polygon in selection.bounds():
                        box = polygon.boundingRect()
                        boxes.append([round(box.x(), 2), round(box.y(), 2), round(box.width(), 2), round(box.height(), 2)])
                    if boxes:
                        matches.append({"phrase": str(phrase), "rank": rank, "start": start, "rects": boxes})
                        if not found_any:
                            rects.extend(boxes)
                        found_any = True
                if found_any:
                    matched.append(str(phrase))
            size = doc.pagePointSize(number - 1)
            return {"ok": True, "rects": rects, "matched": matched, "matches": matches, "page_size": [round(size.width(), 2), round(size.height(), 2)]}
        return {"ok": False, "error": f"unknown op {op!r}"}

    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line.decode("utf-8"))
            answer = handle(request)
        except Exception as exc:  # noqa: BLE001 - one bad document never ends the worker
            answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out.write((json.dumps(answer, ensure_ascii=True) + "\n").encode("ascii"))
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
