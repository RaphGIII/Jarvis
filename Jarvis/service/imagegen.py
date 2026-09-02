"""image.generate: local, free, offline once the model is cached.

The heavy lifting happens in a SEPARATE process from a dedicated venv
(D:\\JarvisLocal\\venv-image, torch cu118 + diffusers, SD-Turbo fp16): one
generation = one process = the VRAM is provably free afterwards.  The chat
model (FAST_LOCAL via Ollama) shares the single GTX 1070; when free VRAM is
too tight for the ~2.6 GB the pipeline needs, the generator asks Ollama to
unload the chat model first and says so — the next chat turn pays one model
reload instead of the interface freezing mid-generation (§ resource rule:
never evict uncontrolled, never freeze, always say what is happening).

Output lands in the owner's library (D:\\ZEUS_Wissen\\Bilder), so the Wissen
galaxy shows every generated image as a real file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_PYTHON = Path(os.environ.get("ZEUS_IMAGE_PYTHON", r"D:\JarvisLocal\venv-image\Scripts\python.exe"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("ZEUS_IMAGE_DIR", r"D:\ZEUS_Wissen\Bilder"))
#: measured live: fp16 SD-Turbo peaked at 3359 MiB on the first 512x512 run.
NEEDED_VRAM_MIB = 3800
TIMEOUT_SECONDS = 900.0


def _slug(prompt: str) -> str:
    folded = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return (folded[:48] or "bild").rstrip("-")


class ImageGenerator:
    def __init__(self, *, python: Path | None = None, output_dir: Path | None = None,
                 repo_root: Path | None = None) -> None:
        self.python = Path(python or DEFAULT_PYTHON)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
        self.busy = False

    def available(self) -> dict[str, Any]:
        if not self.python.is_file():
            return {"ok": False, "error": f"kein Bild-Venv unter {self.python} — Installation fehlt"}
        return {"ok": True, "python": str(self.python), "output_dir": str(self.output_dir)}

    def generate(self, prompt: str, *, negative: str = "", size: str = "512x512",
                 steps: int = 2, seed: int = -1, output_path: str = "") -> dict[str, Any]:
        """Run one real generation; the returned dict mirrors the worker's JSON."""

        prompt = str(prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "leerer Prompt"}
        ready = self.available()
        if not ready.get("ok"):
            return ready
        if self.busy:
            return {"ok": False, "error": "es läuft bereits eine Bildgenerierung"}
        out = Path(output_path) if output_path else self.output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{_slug(prompt)}.png"
        command = [str(self.python), "-m", "imagegen.generate", "--prompt", prompt,
                   "--out", str(out), "--size", size, "--steps", str(int(steps)), "--seed", str(int(seed))]
        if negative:
            command += ["--negative", negative]
        self.busy = True
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       cwd=str(self.repo_root), timeout=TIMEOUT_SECONDS,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Bildgenerierung hat {TIMEOUT_SECONDS:.0f}s überschritten"}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            self.busy = False
        last_line = (completed.stdout or "").strip().splitlines()[-1:] or [""]
        try:
            result = json.loads(last_line[0])
        except ValueError:
            result = {"ok": False, "error": (completed.stderr or completed.stdout or "keine Ausgabe")[-400:]}
        result.setdefault("total_seconds", round(time.perf_counter() - started, 1))
        # verification is the file, not the exit code
        if result.get("ok") and not (out.is_file() and out.stat().st_size > 0):
            result = {"ok": False, "error": "der Worker meldete Erfolg, aber die Datei fehlt", "file": str(out)}
        return result


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
    # no "von …": strip the command words, keep the rest as prompt
    stripped = re.sub(r"\b(zeus|jarvis|erzeug\w*|generier\w*|male?|zeichne\w*|erstell\w*|mach\w*|mir|mal|bitte|ein(?:e[ns]?)?|neues?|bild|foto|grafik|image|picture|von|create|generate|draw|paint|a|an|of)\b", " ", body, flags=re.I)
    stripped = re.sub(r"\s+", " ", stripped).strip(" .,!?-")
    return stripped or "ein stimmungsvolles Bild"
