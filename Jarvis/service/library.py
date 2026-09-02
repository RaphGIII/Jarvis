"""The knowledge library: the owner's study shelf as REAL files.

The Wissen view shows two kinds of things: the knowledge graph (ZEUS's own
memory) and this library — folders, notes, PDFs and imported documents that
live as ordinary files under one root the owner can open in Explorer.
Nothing here is a visual simulation: creating a folder creates a folder,
creating a note writes a Markdown file, importing copies the file in.

Rules:

* **one root** — everything the library touches stays under ``root``
  (default ``D:\\ZEUS_Wissen``); paths are resolved and checked, so a crafted
  ``..`` cannot climb out;
* **plain formats** — notes are ``.md`` with a one-line header; summaries are
  ``.md`` beside their source; no proprietary store, Explorer-friendly;
* **bounded** — the tree listing caps entries per folder and total depth so a
  runaway folder cannot freeze the view.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("D:/ZEUS_Wissen") if os.name == "nt" else Path.home() / "ZEUS_Wissen"
MAX_DEPTH = 6
MAX_ENTRIES = 300

#: Folders the library starts with, so the shelf is not an empty void.
STARTER_FOLDERS = ("Studium", "Notizen", "PDFs", "Zusammenfassungen")

_SAFE = re.compile(r"[^\w\s\-.,()äöüÄÖÜß+&]", re.UNICODE)


def _safe_name(name: str) -> str:
    cleaned = _SAFE.sub("", str(name or "")).strip().strip(".")
    return cleaned[:120] or "unbenannt"


class Library:
    def __init__(self, root: str | Path | None = None) -> None:
        env = os.environ.get("ZEUS_LIBRARY_ROOT", "").strip()
        self.root = Path(root or env or DEFAULT_ROOT)

    # -- plumbing --------------------------------------------------------

    def _ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in STARTER_FOLDERS:
            (self.root / name).mkdir(exist_ok=True)
        return self.root

    def _resolve(self, relative: str) -> Path | None:
        """A path inside the root, or None when it would escape."""

        target = (self.root / str(relative or "").strip("/\\")).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError:
            return None
        return target

    # -- reading ---------------------------------------------------------

    def tree(self) -> dict[str, Any]:
        """The whole shelf, bounded: folders with their files, depth-first."""

        root = self._ensure_root()
        entries: list[dict[str, Any]] = []
        count = 0

        def walk(folder: Path, depth: int) -> None:
            nonlocal count
            if depth > MAX_DEPTH or count >= MAX_ENTRIES:
                return
            try:
                children = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return
            for child in children:
                if count >= MAX_ENTRIES:
                    return
                if child.name.startswith((".", "$", "~")):
                    continue
                rel = str(child.relative_to(root)).replace("\\", "/")
                try:
                    stat = child.stat()
                except OSError:
                    continue
                entries.append({
                    "path": rel, "name": child.name, "type": "dir" if child.is_dir() else "file",
                    "ext": child.suffix.lower().lstrip("."), "size": 0 if child.is_dir() else stat.st_size,
                    "modified_at": stat.st_mtime, "depth": depth,
                })
                count += 1
                if child.is_dir():
                    walk(child, depth + 1)

        walk(root, 0)
        return {"ok": True, "root": str(root), "entries": entries,
                "truncated": count >= MAX_ENTRIES}

    def read_note(self, relative: str) -> dict[str, Any]:
        target = self._resolve(relative)
        if target is None or not target.is_file():
            return {"ok": False, "error": f"keine Datei: {relative}"}
        if target.suffix.lower() not in {".md", ".txt"}:
            return {"ok": False, "error": "nur .md/.txt werden hier gelesen"}
        try:
            return {"ok": True, "path": str(target), "text": target.read_text(encoding="utf-8", errors="replace")[:40000]}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

    # -- writing (all real, all under the root) --------------------------

    def create_folder(self, relative: str) -> dict[str, Any]:
        self._ensure_root()
        parts = [p for p in str(relative or "").replace("\\", "/").split("/") if p.strip()]
        safe = "/".join(_safe_name(p) for p in parts)
        target = self._resolve(safe)
        if target is None or not safe:
            return {"ok": False, "error": f"ungültiger Ordnername: {relative}"}
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(target), "relative": safe, "exists": target.is_dir()}

    def write_note(self, folder: str, title: str, text: str) -> dict[str, Any]:
        self._ensure_root()
        parent = self._resolve(folder) if folder else self.root / "Notizen"
        if parent is None:
            return {"ok": False, "error": f"ungültiger Ordner: {folder}"}
        parent.mkdir(parents=True, exist_ok=True)
        name = _safe_name(title) or "notiz"
        target = parent / f"{name}.md"
        n = 2
        while target.exists():
            target = parent / f"{name} ({n}).md"
            n += 1
        body = f"# {title.strip() or name}\n\n{str(text or '').strip()}\n"
        try:
            target.write_text(body, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        read_back = target.is_file() and target.stat().st_size > 0
        return {"ok": read_back, "path": str(target), "relative": str(target.relative_to(self.root)).replace("\\", "/"),
                "read_back": read_back}

    def import_file(self, source: str, folder: str = "") -> dict[str, Any]:
        """Copy a real file into the shelf (PDFs land in PDFs/ by default)."""

        self._ensure_root()
        src = Path(str(source or "").strip('" '))
        if not src.is_file():
            return {"ok": False, "error": f"Quelle nicht gefunden: {source}"}
        if folder:
            parent = self._resolve(folder)
            if parent is None:
                return {"ok": False, "error": f"ungültiger Ordner: {folder}"}
        else:
            parent = self.root / ("PDFs" if src.suffix.lower() == ".pdf" else "Notizen")
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / src.name
        n = 2
        while target.exists():
            target = parent / f"{src.stem} ({n}){src.suffix}"
            n += 1
        try:
            shutil.copy2(src, target)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": target.is_file(), "path": str(target),
                "relative": str(target.relative_to(self.root)).replace("\\", "/"),
                "bytes": target.stat().st_size if target.is_file() else 0}

    def move(self, relative: str, into: str) -> dict[str, Any]:
        """Move a file/folder to another folder, both inside the root."""

        src = self._resolve(relative)
        dst_parent = self._resolve(into)
        if src is None or not src.exists():
            return {"ok": False, "error": f"nicht gefunden: {relative}"}
        if dst_parent is None:
            return {"ok": False, "error": f"ungültiges Ziel: {into}"}
        dst_parent.mkdir(parents=True, exist_ok=True)
        target = dst_parent / src.name
        if target.exists():
            return {"ok": False, "error": f"existiert bereits: {target.name}"}
        try:
            shutil.move(str(src), str(target))
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(target), "relative": str(target.relative_to(self.root)).replace("\\", "/")}

    def save_summary(self, source_relative: str, text: str) -> dict[str, Any]:
        """A summary Markdown beside the source's name under Zusammenfassungen/."""

        src = self._resolve(source_relative)
        stem = src.stem if src is not None else _safe_name(source_relative)
        parent = self.root / "Zusammenfassungen"
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / f"{stem} – Zusammenfassung.md"
        try:
            target.write_text(str(text or ""), encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": target.is_file(), "path": str(target),
                "relative": str(target.relative_to(self.root)).replace("\\", "/")}
