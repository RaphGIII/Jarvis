"""Desktop observation, the honest foundation: WHAT the owner uses, never a
screenshot.

Opt-in and owner-controlled: sampling is OFF until the owner enables it in
the Owner view, the flag persists in a plain JSON file, and everything the
observer ever records is one JSONL line per sample — foreground process
name, a truncated window title, a timestamp.  No pixels, no keys, no
network.  The file is the audit: the owner can open it in an editor and see
exactly what ZEUS saw.

Pattern detection is deliberately simple and inspectable: minutes per
application per day, plus the most common app→app switches.  Out of that
come SUGGESTIONS ("Du nutzt X oft — soll 'Öffne X' eine Schnellaktion
sein?"), never actions: this module cannot launch, click or type.
"""

from __future__ import annotations

import ctypes
import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SAMPLE_SECONDS = 5.0
MAX_TITLE = 80
MAX_FILE_BYTES = 5_000_000  # ~ a few weeks; then the oldest half is dropped


def _foreground() -> tuple[str, str]:
    """(process image name, window title) of the foreground window, or ('','')."""

    try:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", ""
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 256)
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        exe = ""
        if handle:
            buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buffer))
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                exe = Path(buffer.value).name
            kernel32.CloseHandle(handle)
        return exe, title.value[:MAX_TITLE]
    except Exception:  # noqa: BLE001 - observation must never crash the core
        return "", ""


class DesktopObserver:
    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir) / "observer"
        self.config_path = self.dir / "config.json"
        self.samples_path = self.dir / "samples.jsonl"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.enabled = self._load_enabled()
        if self.enabled:
            self._start()

    # -- owner control ---------------------------------------------------

    def _load_enabled(self) -> bool:
        try:
            return bool(json.loads(self.config_path.read_text(encoding="utf-8")).get("enabled"))
        except (OSError, ValueError):
            return False

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        self.enabled = bool(enabled)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps({"enabled": self.enabled, "changed_at": datetime.now().isoformat()}), encoding="utf-8")
        if self.enabled:
            self._start()
        else:
            self._stop.set()
        return self.status()

    def status(self) -> dict[str, Any]:
        samples = 0
        try:
            samples = sum(1 for _ in self.samples_path.open(encoding="utf-8"))
        except OSError:
            pass
        return {"ok": True, "enabled": self.enabled, "running": bool(self._thread and self._thread.is_alive() and not self._stop.is_set()),
                "samples": samples, "log": str(self.samples_path),
                "records": "nur Prozessname + Fenstertitel (gekürzt) + Zeit — keine Screenshots, keine Tasten, kein Netz"}

    # -- sampling --------------------------------------------------------

    def _start(self) -> None:
        if self._thread and self._thread.is_alive() and not self._stop.is_set():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="desktop-observer")
        self._thread.start()

    def _run(self) -> None:
        last = ("", "")
        while not self._stop.is_set():
            exe, title = _foreground()
            if exe and (exe, title) != last:
                last = (exe, title)
                row = {"at": time.time(), "exe": exe, "title": title}
                try:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    with self._lock, self.samples_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self._rotate()
                except OSError:
                    pass
            self._stop.wait(SAMPLE_SECONDS)

    def _rotate(self) -> None:
        try:
            if self.samples_path.stat().st_size <= MAX_FILE_BYTES:
                return
            lines = self.samples_path.read_text(encoding="utf-8").splitlines()
            self.samples_path.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")
        except OSError:
            pass

    # -- patterns and suggestions ---------------------------------------

    def _rows(self, *, since_hours: float = 72.0) -> list[dict[str, Any]]:
        cutoff = time.time() - since_hours * 3600
        rows = []
        try:
            with self.samples_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("at", 0) >= cutoff:
                        rows.append(row)
        except OSError:
            pass
        return rows

    def patterns(self, *, since_hours: float = 72.0) -> dict[str, Any]:
        rows = self._rows(since_hours=since_hours)
        # each sample row marks a focus CHANGE; time in an app = until the next change
        usage: Counter[str] = Counter()
        switches: Counter[tuple[str, str]] = Counter()
        for i, row in enumerate(rows):
            exe = str(row.get("exe", ""))
            if not exe:
                continue
            nxt = rows[i + 1]["at"] if i + 1 < len(rows) else row["at"] + SAMPLE_SECONDS
            usage[exe] += min(3600.0, max(SAMPLE_SECONDS, nxt - row["at"]))
            if i + 1 < len(rows):
                pair = (exe, str(rows[i + 1].get("exe", "")))
                if pair[0] != pair[1] and all(pair):
                    switches[pair] += 1
        own = {"zeus.exe", "python.exe", "msedge.exe"}
        suggestions = []
        for exe, seconds in usage.most_common(8):
            if exe.lower() in own or seconds < 600:
                continue
            app = exe.rsplit(".", 1)[0]
            suggestions.append({"kind": "quick_open", "app": app,
                                "evidence": f"{round(seconds / 60)} Minuten im Vordergrund in den letzten {int(since_hours)}h",
                                "text": f"Du nutzt {app} oft ({round(seconds / 60)} min). „Öffne {app}“ funktioniert bereits als Schnellbefehl."})
        return {"ok": True, "enabled": self.enabled, "window_hours": since_hours, "samples": len(rows),
                "top_apps": [{"exe": e, "minutes": round(s / 60)} for e, s in usage.most_common(10)],
                "top_switches": [{"from": a, "to": b, "count": c} for (a, b), c in switches.most_common(8)],
                "suggestions": suggestions[:5]}
