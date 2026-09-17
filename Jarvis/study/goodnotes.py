"""GoodNotes: one source adapter that says exactly what it can read and what it cannot.

What exists: GoodNotes offers no public API or SDK for reading notebooks, and its native
``.goodnotes`` file is an undocumented package -- reading the handwriting out of it would
be guessing.  What GoodNotes does offer is export (PDF, and single pages as images) and
automatic backup of notebooks as PDF into a cloud folder (Google Drive, Dropbox,
OneDrive).  On Windows such a folder is synced to disk.

So :class:`GoodNotesSourceAdapter`:

1. ``detect``: recognises GoodNotes PDF exports (producer/creator metadata, or a GoodNotes
   folder), image exports (EXIF Software/Make, PNG text chunks, XMP, or a GoodNotes
   folder) and native ``.goodnotes`` files,
2. ``inspect_native``: opens a native file *read-only and bounded* (member count,
   uncompressed size and file size limits; nothing extracted, nothing executed) and reports
   what the package holds -- always ``supported: False``, because the page content
   (ink strokes) is not in a documented format.  Standard PDFs/images embedded as
   attachments (backgrounds the owner imported, not the handwriting) are listed as
   ``readable_attachments``,
3. ``discover_folders``: finds GoodNotes backup/export folders under the synced cloud roots
   (OneDrive, Dropbox, Google Drive, iCloud Drive), depth-limited, never following
   links or junctions,
4. ``export_help``: tells the owner, in German, how to export or auto-back-up as PDF.

Image exports: GoodNotes' page-image file names are not documented and differ between
versions and share targets, so a file name alone never makes an image a GoodNotes export;
only metadata or a GoodNotes folder does.  A trailing page number in the name ("... 3",
"Seite 3", "page_3") is used as the page hint, nothing more.

Handwriting in an exported PDF has no text layer; OCR reads it when an engine is
installed, and the page itself is always opened exactly.

The module-level functions (``is_goodnotes_pdf``, ``cloud_roots``, ``discover``,
``native_files``, ``status``, ``notebook_name``) are thin wrappers kept for existing callers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

NATIVE_SUFFIX = ".goodnotes"
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

NATIVE_MESSAGE = ("Diese GoodNotes-Datei kann ZEUS derzeit nicht direkt lesen. "
                  "Exportiere sie als PDF oder verbinde deinen GoodNotes-Backup-Ordner.")

_PAGE_HINT = re.compile(r"(?i)(?:^|[\s_\-])(?:seite|page|s\.|p\.?)?\s*[_\-]?\s*(\d{1,3})$")  # 1-3 digits: "Klausur 2024" is a year, not page 2024


def _mentions_goodnotes(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return "goodnotes" in re.sub(r"[\s_\-]+", "", str(value)).lower()


def _in_goodnotes_folder(path: Path) -> str:
    """The name of the nearest enclosing folder whose name says GoodNotes, or ''."""

    for parent in path.parents:
        if parent.name and _mentions_goodnotes(parent.name):
            return parent.name
    return ""


def _iso(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


class GoodNotesSourceAdapter:
    """Everything GoodNotes-specific: detection, native inspection, folder discovery, export help."""

    origin = "goodnotes_export"

    # native package limits: a crafted archive can never make ZEUS read gigabytes or walk millions of names
    max_file_bytes = 2 * 1024 ** 3
    max_members = 20_000
    max_uncompressed_bytes = 4 * 1024 ** 3
    max_ratio = 200             # a member expanding more than this is treated as a bomb
    sniff_members = 400         # members whose first bytes are read to recognise PDFs/images
    max_image_probe_bytes = 200 * 1024 ** 2

    # -------------------------------------------------------------------------- detect

    def is_goodnotes_pdf(self, metadata: dict[str, Any] | None) -> bool:
        metadata = metadata or {}
        return _mentions_goodnotes(metadata.get("producer")) or _mentions_goodnotes(metadata.get("creator"))

    @staticmethod
    def notebook_name(path: str | Path) -> str:
        return Path(path).stem

    @staticmethod
    def _page_hint(stem: str) -> int | None:
        match = _PAGE_HINT.search(stem.strip())
        if not match:
            return None
        number = int(match.group(1))
        return number if 0 < number < 10000 else None

    def _pdf_metadata(self, path: Path) -> dict[str, Any]:
        try:
            from pypdf import PdfReader

            info = PdfReader(str(path), strict=False).metadata or {}
            return {"producer": str(info.get("/Producer") or ""), "creator": str(info.get("/Creator") or "")}
        except Exception:  # noqa: BLE001 - unreadable PDF: no metadata evidence
            return {}

    def image_metadata(self, path: Path) -> dict[str, str]:
        """Software/Make/Model (EXIF), PNG text chunks and XMP of an image, without decoding pixels."""

        out: dict[str, str] = {}
        try:
            if path.stat().st_size > self.max_image_probe_bytes:
                return out
            from PIL import Image

            with Image.open(path) as image:
                for key, value in (image.info or {}).items():
                    if isinstance(value, (str, bytes)) and key not in {"icc_profile", "exif"}:
                        text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
                        out[str(key)] = text[:2000]
                try:
                    exif = image.getexif()
                    for tag, name in ((305, "Software"), (271, "Make"), (272, "Model"), (315, "Artist"), (270, "ImageDescription")):
                        value = exif.get(tag)
                        if value:
                            out["exif:" + name] = str(value)[:500]
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001 - not an image Pillow can open
            return out
        return out

    def detect(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """``{"origin", "kind": pdf|image|native, "notebook", "page", "confidence", "evidence"}`` or None."""

        target = Path(path)
        suffix = target.suffix.lower()
        folder = _in_goodnotes_folder(target)
        if suffix == NATIVE_SUFFIX:
            return {"origin": self.origin, "kind": "native", "notebook": target.stem, "page": None, "confidence": 1.0,
                    "evidence": ["suffix"]}
        if suffix in PDF_SUFFIXES:
            meta = metadata if metadata is not None else (self._pdf_metadata(target) if target.is_file() else {})
            evidence = []
            if self.is_goodnotes_pdf(meta):
                evidence.append("metadata")
            if folder:
                evidence.append("folder")
            if not evidence:
                return None
            confidence = 0.99 if len(evidence) == 2 else (0.95 if evidence[0] == "metadata" else 0.7)
            return {"origin": self.origin, "kind": "pdf", "notebook": target.stem, "page": None, "confidence": confidence,
                    "evidence": evidence}
        if suffix in IMAGE_SUFFIXES:
            meta = metadata if metadata is not None else (self.image_metadata(target) if target.is_file() else {})
            evidence = []
            if any(_mentions_goodnotes(v) for v in meta.values()):
                evidence.append("metadata")
            if folder:
                evidence.append("folder")
            if not evidence:
                return None
            confidence = 0.95 if len(evidence) == 2 else (0.9 if evidence[0] == "metadata" else 0.6)
            page = self._page_hint(target.stem)
            notebook = target.stem
            if page is not None:
                notebook = _PAGE_HINT.sub("", target.stem).strip(" _-") or target.stem
            parent = target.parent.name
            if folder and parent and parent != folder and not _mentions_goodnotes(parent):
                notebook = parent  # pages exported into a folder named after the notebook
            return {"origin": self.origin, "kind": "image", "notebook": notebook, "page": page, "confidence": confidence,
                    "evidence": evidence}
        return None

    # ------------------------------------------------------------------------ native

    @staticmethod
    def _sniff(head: bytes) -> str:
        if head.startswith(b"%PDF-"):
            return "pdf"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if head.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "webp"
        if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
            return "heic"
        return ""

    @staticmethod
    def _group(name: str) -> str:
        lower = name.lower().strip("/")
        first = lower.split("/", 1)[0]
        if first.startswith("attachment"):
            return "attachments"
        if first.startswith("thumbnail") or "thumbnail" in lower:
            return "thumbnails"
        if first in {"notes", "note", "pages", "page"}:
            return "notes"
        if first.startswith("search") or "searchindex" in lower:
            return "search"
        if "/" not in lower and (lower.endswith(".pb") or lower.startswith("index") or lower.endswith(".plist") or lower.endswith(".json")):
            return "index"
        return "other"

    def inspect_native(self, path: str | Path, *, max_members: int | None = None, max_uncompressed_bytes: int | None = None,
                       max_file_bytes: int | None = None) -> dict[str, Any]:
        """What a native ``.goodnotes`` package contains.  Read-only, bounded, never ``supported``."""

        target = Path(path)
        max_members = self.max_members if max_members is None else max_members
        max_uncompressed = self.max_uncompressed_bytes if max_uncompressed_bytes is None else max_uncompressed_bytes
        max_file = self.max_file_bytes if max_file_bytes is None else max_file_bytes
        report: dict[str, Any] = {
            "path": str(target), "notebook": target.stem, "supported": False, "container": "unknown", "size": None,
            "members": 0, "uncompressed_bytes": 0, "groups": {}, "readable_attachments": [], "thumbnails": [],
            "handwriting_readable": False, "limits_exceeded": None, "error": None, "message": NATIVE_MESSAGE,
            "reason": "GoodNotes hat kein öffentliches Dateiformat: die Handschrift (Striche) liegt in einem undokumentierten Format.",
            "help": self.export_help(),
        }
        try:
            info = target.stat()
        except OSError:
            report["container"] = "missing"
            report["error"] = "not_found"
            return report
        if stat.S_ISDIR(info.st_mode):
            report["container"] = "directory"
            names: list[tuple[str, int]] = []
            total = 0
            for current, dirs, files in os.walk(target, followlinks=False):
                dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(current, d))]
                for name in files:
                    full = Path(current) / name
                    try:
                        size = full.stat().st_size
                    except OSError:
                        continue
                    names.append((full.relative_to(target).as_posix(), size))
                    total += size
                    if len(names) > max_members:
                        report["limits_exceeded"] = "members"
                        break
                if report["limits_exceeded"]:
                    break
            report["members"], report["uncompressed_bytes"] = len(names), total
            self._fill_groups(report, names)
            for name, size in names[: self.sniff_members]:
                try:
                    with open(target / name, "rb") as handle:
                        kind = self._sniff(handle.read(16))
                except OSError:
                    continue
                self._note_readable(report, name, size, kind)
            return report
        report["size"] = info.st_size
        if info.st_size > max_file:
            report["limits_exceeded"] = "file_size"
            return report
        try:
            with open(target, "rb") as handle:
                head = handle.read(8)
        except OSError as exc:
            report["error"] = type(exc).__name__
            return report
        if not head.startswith(b"PK"):
            report["container"] = "unknown"
            return report
        try:
            archive = zipfile.ZipFile(target)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            report["error"] = type(exc).__name__
            return report
        with archive:
            report["container"] = "zip"
            infos = archive.infolist()
            report["members"] = len(infos)
            if len(infos) > max_members:
                report["limits_exceeded"] = "members"
                return report
            total = sum(max(0, i.file_size) for i in infos)
            report["uncompressed_bytes"] = total
            if total > max_uncompressed:
                report["limits_exceeded"] = "uncompressed_size"
                return report
            entries = [(i.filename, i.file_size) for i in infos if not i.is_dir()]
            self._fill_groups(report, entries)
            sniffed = 0
            for item in infos:
                if item.is_dir() or item.flag_bits & 0x1:  # encrypted members are not opened
                    continue
                if item.compress_size and item.file_size / max(1, item.compress_size) > self.max_ratio and item.file_size > 1024 ** 2:
                    report.setdefault("suspicious_members", []).append(item.filename)
                    continue
                if sniffed >= self.sniff_members:
                    break
                sniffed += 1
                try:
                    with archive.open(item) as member:
                        kind = self._sniff(member.read(16))
                except Exception:  # noqa: BLE001 - unsupported compression, damaged member
                    continue
                self._note_readable(report, item.filename, item.file_size, kind)
        return report

    def _fill_groups(self, report: dict[str, Any], entries: Iterable[tuple[str, int]]) -> None:
        groups: dict[str, dict[str, Any]] = {}
        for name, size in entries:
            group = groups.setdefault(self._group(name), {"count": 0, "bytes": 0, "examples": []})
            group["count"] += 1
            group["bytes"] += max(0, int(size))
            if len(group["examples"]) < 5:
                group["examples"].append(name)
        report["groups"] = groups

    def _note_readable(self, report: dict[str, Any], name: str, size: int, kind: str) -> None:
        if not kind:
            return
        entry = {"member": name, "type": kind, "size": int(size), "group": self._group(name)}
        if entry["group"] == "thumbnails":
            report["thumbnails"].append(entry)
        else:
            report["readable_attachments"].append(entry)

    def extract_attachment(self, path: str | Path, member: str, *, max_bytes: int = 512 * 1024 ** 2) -> Path | None:
        """Copy one readable attachment of a zip package into a fresh temp dir (never next to the source).

        The caller owns (and removes) the returned file's folder.  Only PDFs and images are
        copied, size-capped, under a generated name -- member paths never reach the filesystem.
        """

        try:
            with zipfile.ZipFile(path) as archive:
                item = archive.getinfo(member)
                if item.is_dir() or item.flag_bits & 0x1 or item.file_size > max_bytes:
                    return None
                with archive.open(item) as source:
                    head = source.read(16)
                    kind = self._sniff(head)
                    if not kind:
                        return None
                    folder = Path(tempfile.mkdtemp(prefix="zeus_goodnotes_"))
                    out = folder / f"attachment.{ 'jpg' if kind == 'jpeg' else kind}"
                    written = len(head)
                    with open(out, "wb") as sink:
                        sink.write(head)
                        while True:
                            chunk = source.read(1024 ** 2)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > max_bytes:
                                sink.close()
                                shutil.rmtree(folder, ignore_errors=True)
                                return None
                            sink.write(chunk)
                    return out
        except (KeyError, zipfile.BadZipFile, OSError, ValueError, RuntimeError):
            return None

    # --------------------------------------------------------------------- discovery

    def cloud_roots(self, *, include_drives: bool = True) -> list[dict[str, Any]]:
        """Local roots of synced cloud storage: ``{"path": Path, "provider"}``, existing folders only, deduplicated."""

        env = os.environ
        home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
        candidates: list[tuple[Path, str]] = []
        for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            if env.get(key):
                candidates.append((Path(env[key]), "onedrive"))
        candidates.append((home / "OneDrive", "onedrive"))
        try:
            for child in home.iterdir():
                if child.name.lower().startswith("onedrive - "):
                    candidates.append((child, "onedrive"))
        except OSError:
            pass
        candidates.append((home / "Dropbox", "dropbox"))
        for base in (env.get("LOCALAPPDATA"), env.get("APPDATA")):
            if not base:
                continue
            info = Path(base) / "Dropbox" / "info.json"
            try:
                data = json.loads(info.read_text(encoding="utf-8"))
                for account in data.values() if isinstance(data, dict) else []:
                    if isinstance(account, dict) and account.get("path"):
                        candidates.append((Path(account["path"]), "dropbox"))
            except (OSError, ValueError):
                pass
        for name in ("Google Drive", "My Drive", "GoogleDrive", "Meine Ablage"):
            candidates.append((home / name, "google_drive"))
        if include_drives and os.name == "nt":
            for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
                for name in ("My Drive", "Meine Ablage"):
                    candidates.append((Path(f"{letter}:/{name}"), "google_drive"))
        candidates.append((home / "iCloudDrive", "icloud"))
        candidates.append((home / "iCloud Drive", "icloud"))
        seen: set[str] = set()
        roots: list[dict[str, Any]] = []
        for path, provider in candidates:
            try:
                if not path.is_dir():
                    continue
                key = os.path.normcase(str(path.resolve()))
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            roots.append({"path": path, "provider": provider})
        return roots

    @staticmethod
    def _is_link(entry: os.DirEntry) -> bool:
        try:
            if entry.is_symlink():
                return True
            isjunction = getattr(os.path, "isjunction", None)
            if isjunction is not None and isjunction(entry.path):
                return True
            # only link-like reparse points: OneDrive/iCloud placeholders are reparse points too and must be walked
            tag = getattr(entry.stat(follow_symlinks=False), "st_reparse_tag", 0)
            return tag in {getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C), getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)}
        except OSError:
            return True

    def _walk_dirs(self, root: Path, max_depth: int, budget: list[int]) -> Iterable[tuple[Path, int]]:
        """Directories under ``root`` (root included) up to ``max_depth``, no links/junctions, each real folder once."""

        visited: set[tuple[int, int]] = set()
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                info = current.stat()
                key = (info.st_dev, info.st_ino)
            except OSError:
                continue
            if key in visited and key != (0, 0):
                continue
            visited.add(key)
            budget[0] -= 1
            if budget[0] < 0:
                return
            yield current, depth
            if depth >= max_depth:
                continue
            try:
                with os.scandir(current) as entries:
                    children = [Path(e.path) for e in entries
                                if not e.name.startswith((".", "$")) and e.name.lower() not in {"node_modules", "appdata"}
                                and e.is_dir(follow_symlinks=False) and not self._is_link(e)]
            except OSError:
                continue
            stack.extend((child, depth + 1) for child in sorted(children, reverse=True))

    def _count(self, folder: Path, *, max_depth: int = 6, max_dirs: int = 5000) -> dict[str, Any]:
        counts = {"pdfs": 0, "images": 0, "native_files": 0, "last_modified": None}
        latest = 0.0
        budget = [max_dirs]
        for current, _depth in self._walk_dirs(folder, max_depth, budget):
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            suffix = Path(entry.name).suffix.lower()
                            if suffix in PDF_SUFFIXES:
                                counts["pdfs"] += 1
                            elif suffix in IMAGE_SUFFIXES:
                                counts["images"] += 1
                            elif suffix == NATIVE_SUFFIX:
                                counts["native_files"] += 1
                            else:
                                continue
                            latest = max(latest, entry.stat(follow_symlinks=False).st_mtime)
                        except OSError:
                            continue
            except OSError:
                continue
        counts["last_modified"] = _iso(latest)
        return counts

    def discover_folders(self, *, roots: list[Path | dict[str, Any]] | None = None, max_depth: int = 4, max_dirs: int = 20000,
                         include_drives: bool = True) -> list[dict[str, Any]]:
        """GoodNotes folders under the cloud roots: path, provider, pdf/image/native counts, last modified."""

        if roots is None:
            resolved = self.cloud_roots(include_drives=include_drives)
        else:
            resolved = [r if isinstance(r, dict) else {"path": Path(r), "provider": "folder"} for r in roots]
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for root in resolved:
            base = Path(root["path"])
            budget = [max_dirs]
            skip_under: list[Path] = []
            for current, _depth in self._walk_dirs(base, max_depth, budget):
                if any(parent == current or parent in current.parents for parent in skip_under):
                    continue
                if not _mentions_goodnotes(current.name):
                    continue
                skip_under.append(current)
                key = os.path.normcase(str(current))
                if key in seen:
                    continue
                seen.add(key)
                found.append({"path": str(current), "provider": root.get("provider", "folder"), "root": str(base), **self._count(current)})
        return found

    @staticmethod
    def native_files(folder: str | Path, *, limit: int = 500) -> list[str]:
        out: list[str] = []
        for current, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(NATIVE_SUFFIX):
                    out.append(str(Path(current) / name))
                    if len(out) >= limit:
                        return out
        return out

    # -------------------------------------------------------------------------- help

    @staticmethod
    def export_help() -> dict[str, Any]:
        return {
            "title": "GoodNotes-Notizen für ZEUS bereitstellen",
            "summary": "ZEUS liest GoodNotes über PDF: einmal exportiert oder automatisch als PDF in einen synchronisierten Ordner gesichert.",
            "export_pdf": [
                "Öffne in GoodNotes das Notizbuch.",
                "Tippe auf „Teilen“ bzw. das Export-Symbol und wähle „Exportieren“ (bei einzelnen Seiten „Diese Seite exportieren“).",
                "Wähle das Format PDF und speichere die Datei in einem Ordner, den dein PC synchronisiert (OneDrive, Dropbox, Google Drive, iCloud Drive).",
                "Füge die PDF in ZEUS hinzu oder verbinde den Ordner.",
            ],
            "auto_backup": [
                "Öffne in GoodNotes die Einstellungen und dort „Automatisches Backup“.",
                "Schalte das Backup ein und wähle Google Drive, Dropbox oder OneDrive als Ziel.",
                "Wähle als Dateiformat PDF (nicht das GoodNotes-Format).",
                "Verbinde in ZEUS den Backup-Ordner auf deinem PC – neue Seiten erscheinen dann automatisch.",
            ],
            "notes": [
                "Handschrift wird nur mit Texterkennung (OCR) durchsuchbar; die Seite selbst öffnet ZEUS immer exakt.",
                "Native .goodnotes-Dateien kann ZEUS derzeit nicht direkt lesen: GoodNotes veröffentlicht dieses Format nicht.",
            ],
        }

    def status(self) -> dict[str, Any]:
        return {
            "direct_api": False,
            "supported": ["GoodNotes-PDF-Export (Datei hinzufügen)", "automatisches Backup als PDF in einen synchronisierten Ordner (Ordner verbinden)",
                          "GoodNotes-Bildexport (PNG/JPEG/WebP), erkannt über Metadaten oder den GoodNotes-Ordner"],
            "limits": ["Kein offizieller Zugriff auf .goodnotes-Notizbücher", "Handschrift ist nur mit OCR durchsuchbar"],
            "discovered": self.discover_folders(),
            "help": self.export_help(),
        }


ADAPTER = GoodNotesSourceAdapter()


# ------------------------------------------------------------------ module-level wrappers

def is_goodnotes_pdf(metadata: dict[str, Any]) -> bool:
    return ADAPTER.is_goodnotes_pdf(metadata)


def notebook_name(path: str | Path) -> str:
    return ADAPTER.notebook_name(path)


def detect(path: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return ADAPTER.detect(path, metadata=metadata)


def inspect_native(path: str | Path, **limits: Any) -> dict[str, Any]:
    return ADAPTER.inspect_native(path, **limits)


def cloud_roots() -> list[Path]:
    return [root["path"] for root in ADAPTER.cloud_roots()]


def discover_folders(**options: Any) -> list[dict[str, Any]]:
    return ADAPTER.discover_folders(**options)


def discover(*, max_depth: int = 3) -> list[dict[str, Any]]:
    """Folders under the synced cloud roots whose name says GoodNotes, with how many PDFs and native files they hold."""

    return ADAPTER.discover_folders(max_depth=max_depth)


def native_files(folder: str | Path, *, limit: int = 500) -> list[str]:
    return ADAPTER.native_files(folder, limit=limit)


def export_help() -> dict[str, Any]:
    return ADAPTER.export_help()


def status() -> dict[str, Any]:
    return ADAPTER.status()
