"""Deterministic application launching: "Öffne Spotify" without a model.

The previous path for "öffne <app>" fell through to the composer, which asks
a local model to plan — seconds of latency for something Windows already
knows.  This module indexes the installed apps once via ``Get-StartApps``
(which covers BOTH classic Start-Menu apps and Store/UWP apps such as
Spotify), caches the index on disk so a fresh core answers without paying
the PowerShell start, resolves an app name deterministically, and launches
through ``explorer.exe shell:appsFolder\\<AppID>`` — the one path that works
for desktop and Store apps alike.

Verification is real: after launching, the process list is polled briefly
for a matching process, so the receipt says "Spotify läuft" only when
Windows says so — with the honest note that some launcher-style apps run
under a different process name than their tile.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

MEMORY_TTL_SECONDS = 600.0
DISK_TTL_SECONDS = 24 * 3600.0
VERIFY_SECONDS = 4.0

#: Spoken/typed aliases that differ from the app's tile name.
ALIASES = {
    "browser": "microsoft edge", "edge": "microsoft edge",
    "chrome": "google chrome",
    "taschenrechner": "rechner", "calculator": "rechner",
    "notizblock": "editor", "notepad": "editor",
    "dateiexplorer": "explorer", "datei explorer": "explorer", "file explorer": "explorer",
    "eingabeaufforderung": "terminal", "konsole": "terminal", "cmd": "terminal",
}


def _fold(text: str) -> str:
    lowered = str(text or "").lower().replace("ß", "ss")
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(a, b)
    return "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)).strip()


class AppLauncher:
    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self._index: dict[str, str] = {}   # folded name -> AppID
        self._names: dict[str, str] = {}   # folded name -> display name
        self._indexed_at = 0.0

    # -- the index -------------------------------------------------------

    def _from_disk(self) -> bool:
        if self.cache_path is None or not self.cache_path.is_file():
            return False
        try:
            if time.time() - self.cache_path.stat().st_mtime > DISK_TTL_SECONDS:
                return False
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict) or not data.get("apps"):
            return False
        self._index = {k: v[0] for k, v in data["apps"].items()}
        self._names = {k: v[1] for k, v in data["apps"].items()}
        return True

    def _to_disk(self) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            apps = {k: [self._index[k], self._names.get(k, k)] for k in self._index}
            self.cache_path.write_text(json.dumps({"apps": apps, "at": time.time()}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _scan(self) -> None:
        """Get-StartApps once; ~1s of PowerShell, then cached."""

        try:
            # bytes + explicit UTF-8: console codepages mangle app names with
            # non-ASCII characters (the tasklist-is-OEM lesson, again)
            raw = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }"],
                capture_output=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            out = (raw or b"").decode("utf-8", "replace")
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        index: dict[str, str] = {}
        names: dict[str, str] = {}
        for line in out.splitlines():
            name, _, app_id = line.partition("|")
            name, app_id = name.strip(), app_id.strip()
            if not name or not app_id:
                continue
            folded = _fold(name)
            if any(bad in folded for bad in ("uninstall", "deinstall")):
                continue
            index.setdefault(folded, app_id)
            names.setdefault(folded, name)
        if index:
            self._index, self._names = index, names
            self._to_disk()

    def index(self, *, force: bool = False) -> dict[str, str]:
        now = time.monotonic()
        if not force and self._index and now - self._indexed_at < MEMORY_TTL_SECONDS:
            return self._index
        if force or not self._index:
            if force or not self._from_disk():
                self._scan()
        self._indexed_at = now
        return self._index

    def resolve(self, query: str) -> tuple[str, str, list[str]]:
        """(display name, AppID or '', candidate display names)."""

        wanted = _fold(query)
        wanted = _fold(ALIASES.get(wanted, wanted))
        index = self.index()
        if wanted in index:
            return self._names.get(wanted, wanted), index[wanted], [self._names.get(wanted, wanted)]
        starts = [n for n in index if n.startswith(wanted)]
        contains = [n for n in index if wanted in n] if not starts else []
        candidates = sorted(starts or contains, key=len)
        if candidates:
            best = candidates[0]
            return self._names.get(best, best), index[best], [self._names.get(c, c) for c in candidates[:5]]
        return query, "", []

    # -- launching -------------------------------------------------------

    @staticmethod
    def _match_keys(name: str, app_id: str) -> set[str]:
        """Folded strings a matching process name may contain (or be contained in).

        The tile name is localized ("Rechner") while the process is not
        (CalculatorApp.exe), so the AppID's alphanumeric runs are keys too.
        """

        import re as _re

        keys = {_fold(name)}
        for token in _re.findall(r"[A-Za-z]{5,}", app_id or ""):
            keys.add(_fold(token))
        return {k for k in keys if len(k) >= 4}

    @staticmethod
    def _running(keys: set[str]) -> str:
        """A running process whose image name matches one of ``keys``, via raw Win32.

        Not psutil: its first ``process_iter`` on this machine took 142
        seconds across ~1400 processes.  EnumProcesses + a limited-rights
        image-name query does the same scan in milliseconds.
        """

        if sys.platform != "win32" or not keys:
            return ""
        import ctypes
        from ctypes import wintypes

        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        pids = (wintypes.DWORD * 4096)()
        returned = wintypes.DWORD(0)
        if not psapi.EnumProcesses(ctypes.byref(pids), ctypes.sizeof(pids), ctypes.byref(returned)):
            return ""
        buffer = ctypes.create_unicode_buffer(1024)
        for i in range(returned.value // ctypes.sizeof(wintypes.DWORD)):
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pids[i])
            if not handle:
                continue
            size = wintypes.DWORD(len(buffer))
            ok = kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
            kernel32.CloseHandle(handle)
            if not ok:
                continue
            pname = _fold(Path(buffer.value).stem)
            for suffix in ("app", "desktop", "uwp"):
                if pname.endswith(suffix) and len(pname) > len(suffix) + 3:
                    pname = pname[: -len(suffix)]
            if pname and len(pname) >= 4 and any(pname in k or k in pname for k in keys):
                return Path(buffer.value).name
        return ""

    def launch(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        if sys.platform != "win32":
            return {"ok": False, "error": "app launching is Windows-only here"}
        name, app_id, candidates = self.resolve(query)
        if not app_id:
            return {"ok": False, "error": f"keine App namens „{query}“ gefunden",
                    "candidates": candidates, "query": query}
        keys = self._match_keys(name, app_id)
        already = self._running(keys)
        try:
            # shell:appsFolder resolves AUMIDs for desktop AND Store apps alike
            subprocess.Popen(["explorer.exe", f"shell:appsFolder\\{app_id}"], close_fds=True)
        except OSError as exc:
            return {"ok": False, "error": str(exc), "query": query, "app": name}
        process = already
        deadline = time.monotonic() + VERIFY_SECONDS
        while not process and time.monotonic() < deadline:
            time.sleep(0.25)
            process = self._running(keys)
        return {"ok": True, "app": name, "app_id": app_id, "already_running": bool(already),
                "process": process, "process_verified": bool(process),
                "seconds": round(time.perf_counter() - started, 2), "query": query}
