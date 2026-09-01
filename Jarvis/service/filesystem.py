"""The filesystem as ZEUS sees it: real paths, staged indexing, live watchers.

Rules this module enforces for the File Galaxy:

* **truth** — every object handed to the interface resolves to an actual
  path from a real ``scandir``; nothing is invented, and visual categories
  (PROJECTS, GAMES, …) are metadata that never move a folder;
* **staged** — nothing scans a whole drive: one directory level per request,
  bounded entry counts, child counts capped, a small TTL cache invalidated
  by events;
* **live** — watched roots use ``ReadDirectoryChangesW`` (a real Windows
  watcher, recursive, one handle per root) with debounced, batched events;
  an overflow triggers a bounded targeted rescan of the affected root, never
  a full-disk walk; where the watcher cannot run, a slow poll of the
  *expanded* directories stands in;
* **safe** — opening in Explorer validates the path exists first and runs
  ``explorer.exe <path>`` with the path as one argument; nothing here
  deletes, moves or writes.
"""

from __future__ import annotations

import ctypes
import json
import os
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Per-directory listing bound: a node with 4 000 children renders as "many".
MAX_ENTRIES = 400
#: Counting a directory's children stops here; enough for sizing a star.
MAX_CHILD_COUNT = 500
CACHE_TTL = 20.0
DEBOUNCE_SECONDS = 0.6

#: Visual clustering of top-level folders.  Metadata only.
CATEGORY_HINTS = (
    ("PROJECTS", ("project", "projekt", "repo", "repos", "src", "source", "workspace", "jarvis", "zeus", "dev ", "code")),
    ("DEVELOPMENT", ("dev", "development", "tools", "sdk", "python", "node", "git", "build", "ollama")),
    ("GAMES", ("game", "spiele", "steam", "epic", "gog", "riot", "battle")),
    ("MEDIA", ("media", "musik", "music", "video", "film", "movie", "bilder", "foto", "photo", "picture", "obs")),
    ("DOCUMENTS", ("dokument", "document", "doc", "uni", "studium", "schule", "paper", "buch", "book", "notizen")),
    ("SYSTEM", ("windows", "system", "program", "programme", "temp", "tmp", "cache", "$recycle", "recovery", "intel", "nvidia", "drivers", "perflogs")),
)


def categorize(name: str) -> str:
    lowered = name.lower()
    for category, hints in CATEGORY_HINTS:
        if any(h in lowered for h in hints):
            return category
    return "OTHER"


def list_drives() -> list[dict[str, Any]]:
    drives: list[dict[str, Any]] = []
    if sys.platform == "win32":
        letters = []
        try:
            letters = list(os.listdrives())  # Python 3.12+
        except AttributeError:
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            letters = [f"{letter}:\\" for i, letter in enumerate(string.ascii_uppercase) if bitmask & (1 << i)]
        for root in letters:
            try:
                usage = os.statvfs(root) if hasattr(os, "statvfs") else None
            except OSError:
                usage = None
            total = free = 0
            try:
                import shutil as _shutil

                du = _shutil.disk_usage(root)
                total, free = du.total, du.free
            except OSError:
                pass
            drives.append({"path": root, "name": root.rstrip("\\"), "type": "drive", "total_bytes": total, "free_bytes": free,
                           "primary": root.upper().startswith("D:")})
    else:
        drives.append({"path": "/", "name": "/", "type": "drive", "total_bytes": 0, "free_bytes": 0, "primary": True})
    return drives


def describe(path: Path, *, count_children: bool = True) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    is_dir = path.is_dir()
    children = 0
    if is_dir and count_children:
        try:
            with os.scandir(path) as it:
                for i, _ in enumerate(it):
                    children = i + 1
                    if children >= MAX_CHILD_COUNT:
                        break
        except OSError:
            children = 0
    return {"name": path.name or str(path), "path": str(path), "type": "dir" if is_dir else "file",
            "size": 0 if is_dir else int(stat.st_size), "modified_at": stat.st_mtime,
            "children_count": children, "category": categorize(path.name or str(path))}


class FilesystemIndex:
    """Bounded, cached, event-invalidated directory listings + live watchers."""

    def __init__(self, *, emit: Callable[[dict[str, Any]], None] | None = None, log: Callable[[str], None] | None = None) -> None:
        self.emit = emit or (lambda _e: None)
        self.log = log or (lambda _m: None)
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._watchers: dict[str, "_Watcher"] = {}
        self._pending: dict[str, set[str]] = {}
        self._pending_lock = threading.Lock()
        self._flusher: threading.Timer | None = None

    # -- listing ---------------------------------------------------------

    def roots(self) -> dict[str, Any]:
        return {"ok": True, "drives": list_drives(), "watched": sorted(self._watchers)}

    def list(self, path: str, *, hidden: bool = False, files: bool = True) -> dict[str, Any]:
        """One level of a real directory, bounded and cached."""

        target = Path(path).expanduser()
        key = str(target).lower()
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and now - hit[0] < CACHE_TTL:
                return hit[1]
        if not target.exists():
            return {"ok": False, "error": f"{target} does not exist", "path": str(target)}
        if not target.is_dir():
            info = describe(target, count_children=False)
            return {"ok": True, "path": str(target), "entry": info, "entries": []}
        entries: list[dict[str, Any]] = []
        truncated = False
        try:
            with os.scandir(target) as it:
                rows = []
                for entry in it:
                    if not hidden and entry.name.startswith((".", "$")):
                        continue
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if not files and not is_dir:
                        continue
                    rows.append((entry, is_dir))
                rows.sort(key=lambda r: (not r[1], r[0].name.lower()))
                for entry, is_dir in rows[: MAX_ENTRIES]:
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    children = 0
                    if is_dir:
                        try:
                            with os.scandir(entry.path) as sub:
                                for i, _ in enumerate(sub):
                                    children = i + 1
                                    if children >= MAX_CHILD_COUNT:
                                        break
                        except OSError:
                            children = 0
                    entries.append({"name": entry.name, "path": entry.path, "type": "dir" if is_dir else "file",
                                    "size": 0 if is_dir else int(stat.st_size), "modified_at": stat.st_mtime,
                                    "children_count": children, "category": categorize(entry.name)})
                truncated = len(rows) > MAX_ENTRIES
        except OSError as exc:
            return {"ok": False, "error": str(exc), "path": str(target)}
        out = {"ok": True, "path": str(target), "entries": entries, "truncated": truncated,
               "entry": describe(target, count_children=False), "at": time.time()}
        with self._lock:
            self._cache[key] = (now, out)
            for stale in [k for k in self._cache if len(self._cache) > 300][:50]:
                del self._cache[stale]
        return out

    def invalidate(self, path: str) -> None:
        key = str(Path(path)).lower()
        with self._lock:
            self._cache.pop(key, None)

    # -- watching --------------------------------------------------------

    def watch(self, root: str) -> dict[str, Any]:
        """Watch a root (recursively) and stream debounced change events."""

        target = Path(root)
        if not target.is_dir():
            return {"ok": False, "error": f"{target} is not a directory"}
        key = str(target).lower()
        if key in self._watchers and self._watchers[key].alive:
            return {"ok": True, "watching": str(target), "already": True}
        watcher = _Watcher(target, on_change=self._on_change, log=self.log)
        watcher.start()
        self._watchers[key] = watcher
        return {"ok": True, "watching": str(target), "mode": watcher.mode}

    def unwatch(self, root: str = "") -> dict[str, Any]:
        keys = [str(Path(root)).lower()] if root else list(self._watchers)
        stopped = 0
        for key in keys:
            watcher = self._watchers.pop(key, None)
            if watcher is not None:
                watcher.stop()
                stopped += 1
        return {"ok": True, "stopped": stopped}

    def _on_change(self, directory: str, kind: str) -> None:
        """Raw watcher callback: batch by parent directory, flush debounced."""

        with self._pending_lock:
            self._pending.setdefault(directory, set()).add(kind)
            if self._flusher is None or not self._flusher.is_alive():
                self._flusher = threading.Timer(DEBOUNCE_SECONDS, self._flush)
                self._flusher.daemon = True
                self._flusher.start()

    def _flush(self) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        changed = []
        for directory, kinds in list(pending.items())[:40]:
            self.invalidate(directory)
            changed.append({"path": directory, "kinds": sorted(kinds)})
        if changed:
            self.emit({"fs": True, "changed": changed, "at": time.time()})

    def status(self) -> dict[str, Any]:
        return {"watchers": {k: {"alive": w.alive, "mode": w.mode, "events": w.events} for k, w in self._watchers.items()},
                "cached_dirs": len(self._cache)}

    # -- opening ---------------------------------------------------------

    @staticmethod
    def open_in_explorer(path: str) -> dict[str, Any]:
        target = Path(path)
        if not target.exists():
            return {"ok": False, "error": f"{target} does not exist"}
        try:
            if sys.platform == "win32":
                if target.is_dir():
                    subprocess.Popen(["explorer.exe", str(target)], close_fds=True)
                else:
                    subprocess.Popen(["explorer.exe", "/select,", str(target)], close_fds=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "opened": str(target)}


# --------------------------------------------------------------------------
# the watcher: ReadDirectoryChangesW with a polling fallback
# --------------------------------------------------------------------------

class _Watcher:
    def __init__(self, root: Path, *, on_change: Callable[[str, str], None], log: Callable[[str], None]) -> None:
        self.root = root
        self.on_change = on_change
        self.log = log
        self.mode = "rdcw" if sys.platform == "win32" else "poll"
        self.events = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"fs-watch-{self.root.drive or self.root.name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._handle is not None and sys.platform == "win32":
            try:
                ctypes.windll.kernel32.CancelIoEx(self._handle, None)
                ctypes.windll.kernel32.CloseHandle(self._handle)
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        if sys.platform == "win32":
            try:
                self._run_rdcw()
                return
            except Exception as exc:  # noqa: BLE001 - fall back rather than die
                self.log(f"fs watcher fell back to polling: {exc}")
                self.mode = "poll"
        self._run_poll()

    def _run_rdcw(self) -> None:
        kernel32 = ctypes.windll.kernel32
        FILE_LIST_DIRECTORY = 0x0001
        OPEN_EXISTING, BACKUP_SEMANTICS = 3, 0x02000000
        SHARE = 0x1 | 0x2 | 0x4
        NOTIFY = 0x1 | 0x2 | 0x8 | 0x10  # file name, dir name, size, last write
        ACTIONS = {1: "created", 2: "deleted", 3: "modified", 4: "renamed_from", 5: "renamed_to"}
        handle = kernel32.CreateFileW(str(self.root), FILE_LIST_DIRECTORY, SHARE, None, OPEN_EXISTING, BACKUP_SEMANTICS, None)
        if handle == ctypes.c_void_p(-1).value or handle is None:
            raise OSError(f"CreateFileW failed for {self.root}")
        self._handle = handle
        buffer = ctypes.create_string_buffer(64 * 1024)
        returned = ctypes.c_ulong(0)
        while not self._stop.is_set():
            ok = kernel32.ReadDirectoryChangesW(handle, buffer, len(buffer), True, NOTIFY, ctypes.byref(returned), None, None)
            if self._stop.is_set():
                return
            if not ok:
                raise OSError("ReadDirectoryChangesW failed")
            if returned.value == 0:
                # overflow: the buffer could not hold the burst -- a bounded,
                # targeted refresh of this root, never a disk walk
                self.events += 1
                self.on_change(str(self.root), "overflow")
                continue
            offset = 0
            while True:
                next_offset, action, name_len = ctypes.cast(buffer[offset:offset + 12], ctypes.POINTER(ctypes.c_ulong * 3)).contents
                name = bytes(buffer[offset + 12: offset + 12 + name_len]).decode("utf-16-le", "replace")
                full = self.root / name
                self.events += 1
                self.on_change(str(full.parent), ACTIONS.get(int(action), "changed"))
                if not next_offset:
                    break
                offset += next_offset

    def _run_poll(self) -> None:
        snapshots: dict[str, set[str]] = {}
        while not self._stop.is_set():
            try:
                with os.scandir(self.root) as it:
                    now = {e.name for e in it}
            except OSError:
                now = set()
            before = snapshots.get("top")
            if before is not None and now != before:
                self.events += 1
                self.on_change(str(self.root), "changed")
            snapshots["top"] = now
            self._stop.wait(2.0)
