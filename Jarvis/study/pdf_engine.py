"""The server-side client of the PDF page worker, with a pypdf fallback for text.

``PdfEngine.available`` is True when the Qt worker starts (PySide6 with QtPdf present).
Without it, text still comes from pypdf and the viewer falls back to the browser's
own PDF viewer; page images and exact highlight rectangles need the worker.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent


class PdfEngine:
    def __init__(self, *, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self._process: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.Lock()
        self._unavailable_reason = ""

    # -- the worker ---------------------------------------------------------------

    def _start(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        if self._unavailable_reason:
            return False
        try:
            import importlib.util

            if importlib.util.find_spec("PySide6") is None:
                self._unavailable_reason = "PySide6 (QtPdf) is not installed"
                return False
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen([sys.executable, "-m", "study.pdf_worker"], cwd=str(_ROOT), stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            self._unavailable_reason = f"worker could not start: {exc}"
            return False
        self._lines = queue.Queue()
        stdout = self._process.stdout

        def pump() -> None:
            assert stdout is not None
            for line in stdout:
                self._lines.put(line)
            self._lines.put(b"")

        threading.Thread(target=pump, daemon=True, name="study-pdf-worker").start()
        answer = self._call_locked({"op": "ping"}, timeout=40.0)
        if not answer.get("ok"):
            self._unavailable_reason = f"worker did not answer: {answer.get('error', '')}"
            self.close()
            return False
        return True

    def _call_locked(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            return {"ok": False, "error": "worker not running"}
        try:
            process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            process.stdin.flush()
            line = self._lines.get(timeout=timeout or self.timeout)
        except (OSError, queue.Empty) as exc:
            self.close()
            return {"ok": False, "error": f"worker failed: {type(exc).__name__}"}
        if not line:
            self.close()
            return {"ok": False, "error": "worker exited"}
        try:
            return json.loads(line.decode("ascii"))
        except ValueError:
            return {"ok": False, "error": "unreadable worker answer"}

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._start():
                return {"ok": False, "error": self._unavailable_reason or "PDF engine unavailable", "unavailable": True}
            return self._call_locked(request)

    @property
    def available(self) -> bool:
        with self._lock:
            return self._start()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass

    # -- operations ---------------------------------------------------------------

    def info(self, path: str | Path) -> dict[str, Any]:
        return self.call({"op": "info", "path": str(path)})

    def texts(self, path: str | Path) -> list[str] | None:
        answer = self.call({"op": "text", "path": str(path)})
        return list(answer.get("pages") or []) if answer.get("ok") else None

    def render(self, path: str | Path, page: int, out: str | Path, *, width: int = 1400, clip: list[float] | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {"op": "render", "path": str(path), "page": int(page), "width": int(width), "out": str(out)}
        if clip:
            request["clip"] = [round(float(v), 4) for v in clip]
        return self.call(request)

    def locate(self, path: str | Path, page: int, phrases: list[str]) -> dict[str, Any]:
        return self.call({"op": "locate", "path": str(path), "page": int(page), "phrases": [str(p) for p in phrases if str(p).strip()]})


_engine: PdfEngine | None = None


def engine() -> PdfEngine:
    global _engine
    if _engine is None:
        _engine = PdfEngine()
    return _engine
