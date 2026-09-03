"""image.generate: local, free, and now FAST after the first image.

v1 spawned a one-shot process per image and paid 200–370 s of model load per
generation — the owner watched nothing happen for minutes.  v2 keeps a
PERSISTENT worker (imagegen/worker.py in the image venv): the model loads
once into CPU RAM, each job moves it to the GPU only for the generation
window (~15–25 s total), then the VRAM is freed again.  Phases stream back
so the Job system can show "Modell lädt / Generiere / Speichere" honestly.

Naming and destination honour the owner's creation defaults
(service/defaults.py) unless the request names a path explicitly.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_PYTHON = Path(os.environ.get("ZEUS_IMAGE_PYTHON", r"D:\JarvisLocal\venv-image\Scripts\python.exe"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("ZEUS_IMAGE_DIR", r"D:\ZEUS_Wissen\Bilder"))
#: measured live: fp16 SD-Turbo peaked at 3575 MiB on a 640x512 run.
NEEDED_VRAM_MIB = 3800
FIRST_LOAD_TIMEOUT = 900.0
PHASE_TIMEOUT = 600.0

PHASE_LABELS = {"loading_model": "Modell lädt", "to_gpu": "Modell → GPU", "generating": "Generiere",
                "saving": "Speichere", "to_cpu": "GPU wird freigegeben"}


def _slug(prompt: str) -> str:
    folded = re.sub(r"[^a-z0-9äöüß]+", "-", prompt.lower()).strip("-")
    return (folded[:48] or "bild").rstrip("-")


class ImageGenerator:
    """Client for the persistent worker; one generation at a time."""

    def __init__(self, *, python: Path | None = None, output_dir: Path | None = None,
                 repo_root: Path | None = None) -> None:
        self.python = Path(python or DEFAULT_PYTHON)
        self.default_output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
        self.busy = False
        self.model_loaded = False
        self._proc: subprocess.Popen | None = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()

    # -- worker lifecycle ------------------------------------------------

    def available(self) -> dict[str, Any]:
        if not self.python.is_file():
            return {"ok": False, "error": f"kein Bild-Venv unter {self.python} — Installation fehlt"}
        return {"ok": True, "python": str(self.python), "output_dir": str(self.default_output_dir),
                "worker": bool(self._proc and self._proc.poll() is None), "model_loaded": self.model_loaded}

    def _reader(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout or ():
            self._lines.put(line)

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self.model_loaded = False
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            [str(self.python), "-u", "-m", "imagegen.worker"],
            cwd=str(self.repo_root), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._reader, args=(self._proc,), daemon=True, name="image-worker-reader").start()

    def stop_worker(self) -> None:
        proc, self._proc = self._proc, None
        self.model_loaded = False
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()

    # -- generation ------------------------------------------------------

    def resolve_output(self, prompt: str, *, output_path: str = "",
                       output_dir: str = "", name_template: str = "") -> Path:
        if output_path:
            return Path(output_path)
        directory = Path(output_dir) if output_dir else self.default_output_dir
        template = name_template or "{date}_{time}_{slug}"
        name = (template.replace("{date}", time.strftime("%Y%m%d"))
                        .replace("{time}", time.strftime("%H%M%S"))
                        .replace("{slug}", _slug(prompt)))
        if not name.lower().endswith(".png"):
            name += ".png"
        return directory / name

    def generate(self, prompt: str, *, negative: str = "", size: str = "512x512",
                 steps: int = 2, seed: int = -1, output_path: str = "",
                 output_dir: str = "", name_template: str = "",
                 on_phase: Callable[[str, float], None] | None = None,
                 cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "leerer Prompt"}
        ready = self.available()
        if not ready.get("ok"):
            return ready
        with self._lock:
            if cancel_check and cancel_check():
                return {"ok": False, "error": "abgebrochen, bevor es losging", "cancelled": True}
            self.busy = True
            started = time.monotonic()
            try:
                self._ensure_worker()
                out = self.resolve_output(prompt, output_path=output_path,
                                          output_dir=output_dir, name_template=name_template)
                rid = f"img{int(time.time())}"
                request = {"op": "generate", "id": rid, "prompt": prompt, "negative": negative,
                           "size": size, "steps": int(steps), "seed": int(seed), "out": str(out)}
                self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
                deadline = time.monotonic() + (PHASE_TIMEOUT if self.model_loaded else FIRST_LOAD_TIMEOUT)
                while time.monotonic() < deadline:
                    try:
                        line = self._lines.get(timeout=2.0)
                    except queue.Empty:
                        if self._proc.poll() is not None:
                            return {"ok": False, "error": "der Bild-Worker ist abgestürzt", "worker_died": True}
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    kind = event.get("event")
                    if kind == "phase" and event.get("id") == rid:
                        phase = str(event.get("phase", ""))
                        if phase != "loading_model":
                            self.model_loaded = True
                        if on_phase:
                            try:
                                on_phase(phase, float(event.get("at", 0.0)))
                            except Exception:  # noqa: BLE001
                                pass
                        deadline = time.monotonic() + PHASE_TIMEOUT
                    elif kind == "result" and event.get("id") == rid:
                        event["total_seconds"] = round(time.monotonic() - started, 1)
                        self.model_loaded = self.model_loaded or bool(event.get("ok"))
                        # verification is the file, not the exit line
                        if event.get("ok") and not (out.is_file() and out.stat().st_size > 0):
                            return {"ok": False, "error": "der Worker meldete Erfolg, aber die Datei fehlt", "file": str(out)}
                        return event
                return {"ok": False, "error": f"der Bild-Worker hat nicht innerhalb des Zeitfensters geantwortet"}
            except OSError as exc:
                return {"ok": False, "error": str(exc)}
            finally:
                self.busy = False


#: "Erzeuge mir ein Bild von einem Adler über den Alpen." → the prompt part.
_IMAGE_REQ = re.compile(r"\b(erzeug\w*|generier\w*|male?|zeichne\w*|erstell\w*|mach\w*|create|generate|draw|paint)\b.{0,40}?\b(bild|foto|grafik|zeichnung|image|picture)\b", re.I | re.S)
_PROMPT_PART = re.compile(r"\b(?:bild|foto|grafik|zeichnung|image|picture)\s+(?:von|ueber|über|mit|aus|of|showing)\s+(?P<p>.+?)\s*[.!?]?$", re.I | re.S)


def parse_image_request(text: str) -> str | None:
    """The image prompt inside an owner sentence, or None if not an image ask."""

    body = str(text or "").strip()
    if not _IMAGE_REQ.search(body):
        return None
    m = _PROMPT_PART.search(body)
    if m:
        return m.group("p").strip()
    stripped = re.sub(r"\b(zeus|jarvis|erzeug\w*|generier\w*|male?|zeichne\w*|erstell\w*|mach\w*|mir|mal|bitte|ein(?:e[ns]?)?|neues?|bild|foto|grafik|image|picture|von|create|generate|draw|paint|a|an|of)\b", " ", body, flags=re.I)
    stripped = re.sub(r"\s+", " ", stripped).strip(" .,!?-")
    return stripped or "ein stimmungsvolles Bild"
