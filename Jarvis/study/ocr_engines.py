"""OCR engines behind one call: ``read_page(image) -> text, words, blocks`` for printed text and handwriting.

Two recognisers, both local:

* ``PrintedTextOCR`` -- Tesseract (see ``study.ocr`` for tessdata and language choice), run as a low-priority child
  process that is killed when the caller cancels or the page budget runs out; ``tsv`` output gives the block /
  paragraph / line / word hierarchy with per-word confidences.
* ``HandwritingOCR`` -- TrOCR fine-tuned on German handwriting (``fhswf/TrOCR_german_handwritten``; English lines
  can use ``microsoft/trocr-base-handwritten``).  TrOCR reads one text LINE at a time, so the page is segmented
  first (ink mask without ruled lines, word blobs, blobs grouped to lines, long lines split at gaps).  The model
  lives in ``study.handwriting_worker`` -- a separate BELOW_NORMAL process with a bounded thread count.

``read_page`` decides per line in "auto" mode: Tesseract reads the page; lines it read with low confidence, and ink
it did not read at all, go to the handwriting recogniser; per line the better reading wins and a plausible loser is
kept as a ``variant`` (extra searchable text, never mixed into the primary text).

Contract (the same as ``study.ocr.image_words``): ``text`` is the page text (blocks separated by a blank line, lines by
a newline, words by one space -- already stable under the parsers' ``_clean``), and every ``words[i]`` carries
``start``/``end`` offsets into exactly that text plus a ``box`` [x, y, w, h] as fractions of the ORIGINAL image, so
``study.ocr.phrase_boxes`` works unchanged.

Everything is synchronous, CPU-bounded and cancellable (``should_stop``); it is meant for a background indexing
thread, never for a request handler.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from study import ocr

ENGINE_VERSION = "zeus-ocr-2"
MODELS_ROOT = Path(r"D:\JarvisLocal\ocr_models")
GERMAN_MODEL = "trocr_german_handwritten"      # fhswf/TrOCR_german_handwritten (afl-3.0)
ENGLISH_MODEL = "trocr_base_handwritten"       # microsoft/trocr-base-handwritten (MIT)
ROLES = ("title", "body", "annotation", "label", "diagram_label")

_ROOT = Path(__file__).resolve().parent.parent
_BELOW_NORMAL = 0x00004000

ShouldStop = Callable[[], bool] | None


# ------------------------------------------------------------------------------- data

@dataclass
class OcrWord:
    text: str
    confidence: float
    box: list[float]                  # fractions of the original image
    start: int = 0
    end: int = 0
    handwriting: bool = False


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: list[float]                  # fractions of the original image
    words: list[OcrWord] = field(default_factory=list)
    handwriting: bool = False
    engine: str = "tesseract"
    px: list[float] = field(default_factory=list, repr=False)   # [x, y, w, h] on the prepared image
    group: tuple[int, int] = (0, 0)   # Tesseract (block, paragraph); (-1, n) for segmented lines


@dataclass
class OcrBlock:
    text: str
    confidence: float
    box: list[float]
    lines: list[OcrLine] = field(default_factory=list)
    order: int = 0
    role: str = "body"
    handwriting: bool = False
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "box": self.box, "order": self.order, "role": self.role,
                "handwriting": self.handwriting, "char_start": self.char_start, "char_end": self.char_end,
                "lines": [{"text": line.text, "confidence": line.confidence, "box": line.box, "handwriting": line.handwriting,
                           "engine": line.engine} for line in self.lines]}


def _normal(text: Any) -> str:
    return ocr._normal_word(text)


def _union(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _round_box(box: list[float]) -> list[float]:
    return [round(float(v), 5) for v in box]


def _overlap_area(a: list[float], b: list[float]) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


# ------------------------------------------------------------------------------- printed text (Tesseract)

def _run_child(cmd: list[str], *, timeout: float, should_stop: ShouldStop, env: dict[str, str] | None = None) -> bytes | None:
    """Run a child at BELOW_NORMAL priority; None when cancelled, timed out or failed."""

    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | _BELOW_NORMAL) if os.name == "nt" else 0
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                   creationflags=flags, env=env)
    except OSError:
        return None
    chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: chunks.append(process.stdout.read() if process.stdout else b""), daemon=True)
    reader.start()
    deadline = time.monotonic() + max(1.0, timeout)
    while reader.is_alive():
        reader.join(0.2)
        if (should_stop is not None and should_stop()) or time.monotonic() > deadline:
            try:
                process.kill()
            except OSError:
                pass
            reader.join(2.0)
            return None
    process.wait()
    return chunks[0] if chunks and process.returncode == 0 else None


def parse_tsv(raw: str) -> dict[str, list[Any]]:
    """Tesseract ``tsv`` output as the ``image_to_data`` DICT layout."""

    keys = ("level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text")
    data: dict[str, list[Any]] = {key: [] for key in keys}
    for index, line in enumerate(raw.splitlines()):
        if index == 0 and line.startswith("level"):
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            continue
        if len(parts) == 11:
            parts.append("")
        try:
            numbers = [int(p) for p in parts[:10]]
            conf = float(parts[10])
        except ValueError:
            continue
        for key, value in zip(keys[:10], numbers):
            data[key].append(value)
        data["conf"].append(conf)
        data["text"].append("\t".join(parts[11:]))
    return data


class PrintedTextOCR:
    """Tesseract with word confidences, as a cancellable low-priority child process."""

    name = "tesseract"

    def __init__(self, *, threads: int = 2) -> None:
        self.threads = max(1, threads)

    def available(self) -> bool:
        return bool(ocr.tesseract_path())

    def reason(self) -> str:
        return "" if self.available() else "Tesseract ist nicht installiert."

    def data(self, gray: np.ndarray, *, languages: str, psm: int = 3, timeout: float = 120.0,
             should_stop: ShouldStop = None) -> dict[str, list[Any]] | None:
        exe = ocr.tesseract_path()
        if not exe:
            return None
        from PIL import Image

        fd, name = tempfile.mkstemp(prefix="zeus-ocr-", suffix=".png")
        os.close(fd)
        try:
            Image.fromarray(gray).save(name)
            env = {**os.environ, "OMP_THREAD_LIMIT": str(self.threads)}
            base = [exe, name, "stdout", "--psm", str(psm)]
            active, program = ocr.active_tessdata_dir(), ocr.program_tessdata_dir()
            if active is not None and not (program is not None and ocr._same_path(active, program)):
                base += ["--tessdata-dir", str(active)]
            tsv = ["-c", "tessedit_create_tsv=1"]
            attempts = [base + (["-l", languages] if languages else []) + tsv]
            if languages:
                attempts.append([exe, name, "stdout", "--psm", str(psm)] + tsv)
            for cmd in attempts:
                raw = _run_child(cmd, timeout=timeout, should_stop=should_stop, env=env)
                if should_stop is not None and should_stop():
                    return None
                if raw is not None:
                    return parse_tsv(raw.decode("utf-8", errors="replace"))
            return None
        finally:
            try:
                os.remove(name)
            except OSError:
                pass

    def read(self, prepared: Any, *, languages: str, timeout: float = 120.0, should_stop: ShouldStop = None) -> list[OcrLine] | None:
        data = self.data(prepared.image, languages=languages, timeout=timeout, should_stop=should_stop)
        if data is None:
            return None
        return lines_from_data(data, prepared)


def lines_from_data(data: dict[str, list[Any]], prepared: Any) -> list[OcrLine]:
    """Tesseract words grouped to lines (in Tesseract's reading order), boxes mapped to the original image."""

    count = len(data.get("text") or [])
    lines: dict[tuple[int, int, int], OcrLine] = {}
    order: list[tuple[int, int, int]] = []
    for i in range(count):
        word = _normal(data["text"][i])
        conf = float(data.get("conf", [0] * count)[i])
        if not word or conf < 0:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        px = [float(data["left"][i]), float(data["top"][i]), float(data["width"][i]), float(data["height"][i])]
        entry = OcrWord(text=word, confidence=round(max(0.0, min(1.0, conf / 100.0)), 4), box=prepared.to_original_box(px))
        entry._px = px  # type: ignore[attr-defined]
        if key not in lines:
            lines[key] = OcrLine(text="", confidence=0.0, box=[], group=key[:2])
            order.append(key)
        lines[key].words.append(entry)
    result = []
    for key in order:
        line = lines[key]
        line.text = " ".join(w.text for w in line.words)
        weight = sum(len(w.text) for w in line.words) or 1
        line.confidence = round(sum(w.confidence * len(w.text) for w in line.words) / weight, 4)
        line.px = _union([w._px for w in line.words])  # type: ignore[attr-defined]
        line.box = prepared.to_original_box(line.px)
        result.append(line)
    return result


# ------------------------------------------------------------------------------- line segmentation

def segment_lines(gray: np.ndarray, *, text_height: float = 0.0) -> tuple[list[list[float]], list[list[float]]]:
    """(text line boxes, diagram boxes) in pixels of ``gray``: blobs of ink grouped into lines, long lines split at gaps."""

    import cv2

    from study import preprocess

    mask = preprocess.remove_rule_lines(preprocess.ink_mask(gray))
    h, w = mask.shape[:2]
    th = text_height or preprocess.text_height(gray) or 24.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    diagrams: list[list[float]] = []
    clean = np.zeros_like(mask)
    labels_keep = []
    for i in range(1, count):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < 6 or (ch < th * 0.2 and cw < th * 0.2):
            continue
        if (ch > th * 4.5 and cw > th * 4.5) or ch > th * 6:
            diagrams.append([float(x), float(y), float(cw), float(ch)])
            continue
        labels_keep.append(i)
    if labels_keep:
        _, label_img, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean[np.isin(label_img, labels_keep)] = 255
    kx = max(3, int(th * 0.9)) | 1
    ky = max(1, int(th * 0.12))
    blobs_mask = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(blobs_mask, connectivity=8)
    blobs = []
    for i in range(1, count):
        x, y, bw, bh, _area = (int(v) for v in stats[i])
        # undo the dilation margins
        x0, x1 = x + kx // 2, x + bw - kx // 2
        y0, y1 = y + ky // 2, y + bh - ky // 2
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        blobs.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
    blobs.sort(key=lambda b: (b[1] + b[3] / 2, b[0]))
    lines: list[list[list[float]]] = []
    for blob in blobs:
        cy = blob[1] + blob[3] / 2
        best, best_score = None, 0.0
        for line in lines:
            box = _union(line)
            vertical = min(box[1] + box[3], blob[1] + blob[3]) - max(box[1], blob[1])
            ratio = vertical / max(1.0, min(blob[3], box[3]))
            line_cy = sum(b[1] + b[3] / 2 for b in line) / len(line)
            gap = max(box[0] - (blob[0] + blob[2]), blob[0] - (box[0] + box[2]), 0.0)
            if ratio >= 0.45 and abs(cy - line_cy) < max(th * 0.8, 0.45 * min(blob[3], box[3])) and gap < th * 5.0:
                if ratio > best_score:
                    best, best_score = line, ratio
        if best is None:
            lines.append([blob])
        else:
            best.append(blob)
    boxes: list[list[float]] = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        boxes.extend(_split_long(line, th))
    boxes = _merge_row_fragments([b for b in boxes if b[3] >= th * 0.5 and b[2] >= th * 1.2], th)
    boxes.sort(key=lambda b: (round((b[1] + b[3] / 2) / max(th, 1.0)), b[0]))
    return boxes, diagrams


def _same_row(a: list[float], b: list[float]) -> bool:
    vertical = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return vertical >= 0.5 * min(a[3], b[3])


def _merge_row_fragments(boxes: list[list[float]], th: float, max_aspect: float = 26.0) -> list[list[float]]:
    """Pieces of one written line (wide word gaps in handwriting) joined while the crop stays readable for TrOCR."""

    pending = sorted(boxes, key=lambda b: b[0])
    merged: list[list[float]] = []
    for box in pending:
        for index, other in enumerate(merged):
            gap = box[0] - (other[0] + other[2])
            union = _union([other, box])
            if _same_row(other, box) and gap < th * 12 and union[2] / max(union[3], th) <= max_aspect:
                merged[index] = union
                break
        else:
            merged.append(list(box))
    return merged


def _split_long(blobs: list[list[float]], th: float, max_aspect: float = 26.0) -> list[list[float]]:
    box = _union(blobs)
    if len(blobs) < 2 or box[2] / max(box[3], th) <= max_aspect:
        return [box]
    middle = box[0] + box[2] / 2
    best_index, best_score = 1, -1.0
    for i in range(1, len(blobs)):
        gap = blobs[i][0] - (blobs[i - 1][0] + blobs[i - 1][2])
        distance = abs(blobs[i][0] - middle) / max(box[2], 1.0)
        score = gap / max(th, 1.0) - 2.0 * distance
        if score > best_score:
            best_index, best_score = i, score
    return _split_long(blobs[:best_index], th, max_aspect) + _split_long(blobs[best_index:], th, max_aspect)


# ------------------------------------------------------------------------------- handwriting (TrOCR worker)

def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def _model_files_present(folder: Path) -> bool:
    try:
        weights = (folder / "model.safetensors").is_file() or (folder / "pytorch_model.bin").is_file()
        tokenizer = (folder / "tokenizer.json").is_file() or (folder / "vocab.json").is_file()
        return weights and tokenizer and (folder / "config.json").is_file() and (folder / "preprocessor_config.json").is_file()
    except OSError:
        return False


class HandwritingOCR:
    """TrOCR line recognition in a low-priority worker process; lazy, bounded, cancellable."""

    name = "trocr"

    def __init__(self, *, model_dir: str | Path | None = None, english_dir: str | Path | None = None, threads: int | None = None,
                 batch: int = 4, idle_seconds: float = 600.0, start_timeout: float = 300.0,
                 recognizer: Callable[[list[np.ndarray], str], list[tuple[str, float]]] | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else _env_path("ZEUS_HANDWRITING_MODEL_DIR", MODELS_ROOT / GERMAN_MODEL)
        self.english_dir = Path(english_dir) if english_dir else _env_path("ZEUS_HANDWRITING_MODEL_DIR_EN", MODELS_ROOT / ENGLISH_MODEL)
        self.threads = threads or int(os.environ.get("ZEUS_OCR_THREADS", "3") or 3)
        self.batch = max(1, batch)
        self.idle_seconds = idle_seconds
        self.start_timeout = start_timeout
        self._recognizer = recognizer
        self._process: subprocess.Popen[bytes] | None = None
        self._process_dir: Path | None = None
        self._lines: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.RLock()
        self._failure = ""
        self.seconds_per_line = 5.0     # moving estimate, used to stop before the budget runs out

    # -- availability
    def _missing_packages(self) -> list[str]:
        return [name for name in ("torch", "transformers", "PIL") if importlib.util.find_spec(name) is None]

    def model_for(self, language: str) -> Path:
        if language.split("+")[0] == "eng" and _model_files_present(self.english_dir):
            return self.english_dir
        return self.model_dir

    def reason(self) -> str:
        """"" when usable, else a calm German sentence saying what is missing."""

        if self._recognizer is not None:
            return ""
        if not _model_files_present(self.model_dir):
            return (f"Handschriftmodell fehlt: {self.model_dir} (TrOCR für deutsche Handschrift, "
                    "fhswf/TrOCR_german_handwritten). Handschriftliche Seiten werden nur mit Tesseract gelesen.")
        missing = self._missing_packages()
        if missing:
            return ("Python-Pakete für die Handschrifterkennung fehlen: " + ", ".join(missing)
                    + ". Handschriftliche Seiten werden nur mit Tesseract gelesen.")
        if self._failure:
            return f"Handschrifterkennung ist ausgefallen: {self._failure}"
        return ""

    def available(self) -> bool:
        return not self.reason()

    def status(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        return {"available": self.available(), "engine": "trocr", "model": self.model_dir.name, "model_dir": str(self.model_dir),
                "english_model": self.english_dir.name if _model_files_present(self.english_dir) else "",
                "running": running, "threads": self.threads, "reason": self.reason()}

    def cache_tag(self) -> str:
        if not self.available():
            return "hw=none"
        if self._recognizer is not None:
            return "hw=custom"
        return f"hw={self.model_dir.name}"

    # -- worker
    def _wait(self, timeout: float, should_stop: ShouldStop) -> bytes | None:
        deadline = time.monotonic() + max(0.5, timeout)
        while True:
            try:
                return self._lines.get(timeout=0.25)
            except queue.Empty:
                pass
            if should_stop is not None and should_stop():
                return None
            if time.monotonic() > deadline:
                return None
            if self._process is None or self._process.poll() is not None:
                try:
                    return self._lines.get_nowait()
                except queue.Empty:
                    return b""

    def _start(self, folder: Path, should_stop: ShouldStop) -> bool:
        if self._process is not None and self._process.poll() is None and self._process_dir == folder:
            return True
        self.close()
        flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | _BELOW_NORMAL) if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen([sys.executable, "-m", "study.handwriting_worker", str(folder), str(self.threads),
                                              str(int(self.idle_seconds))], cwd=str(_ROOT), stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            self._failure = f"Worker startet nicht ({type(exc).__name__})"
            return False
        self._process_dir = folder
        self._lines = queue.Queue()
        stdout, lines = self._process.stdout, self._lines

        def pump() -> None:
            if stdout is None:
                return
            for line in stdout:
                lines.put(line)
            lines.put(b"")

        threading.Thread(target=pump, daemon=True, name="study-handwriting-worker").start()
        answer = self._call({"op": "ping"}, timeout=self.start_timeout, should_stop=should_stop)
        if not answer.get("ok"):
            if answer.get("error") != "cancelled":
                self._failure = f"Worker antwortet nicht ({answer.get('error', '')})"
            self.close()
            return False
        return True

    def _call(self, request: dict[str, Any], *, timeout: float, should_stop: ShouldStop) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            return {"ok": False, "error": "worker not running"}
        try:
            process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            process.stdin.flush()
        except OSError as exc:
            self.close()
            return {"ok": False, "error": type(exc).__name__}
        line = self._wait(timeout, should_stop)
        if line is None:
            # cancelled or over budget in the middle of a batch: the worker is busy, so it goes
            self.close()
            return {"ok": False, "error": "cancelled" if should_stop is not None and should_stop() else "timeout"}
        if not line:
            self.close()
            return {"ok": False, "error": "worker exited"}
        try:
            return json.loads(line.decode("ascii"))
        except ValueError:
            return {"ok": False, "error": "unreadable answer"}

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass

    # -- recognition
    def recognize(self, crops: list[np.ndarray], *, language: str = "deu", deadline: float | None = None,
                  should_stop: ShouldStop = None) -> list[tuple[str, float] | None]:
        """One (text, confidence) per crop, None for crops not read (budget, cancellation, failure)."""

        results: list[tuple[str, float] | None] = [None] * len(crops)
        if not crops or not self.available():
            return results
        if self._recognizer is not None:
            for start in range(0, len(crops), self.batch):
                if (should_stop is not None and should_stop()) or (deadline is not None and time.monotonic() > deadline):
                    break
                part = self._recognizer(crops[start:start + self.batch], language)
                for offset, item in enumerate(part):
                    results[start + offset] = (str(item[0]), float(item[1]))
            return results
        from PIL import Image

        folder = self.model_for(language)
        with self._lock:
            starting = time.monotonic()
            if not self._start(folder, should_stop):
                return results
            if deadline is not None:
                deadline += time.monotonic() - starting  # loading the model once is not this page's reading time
            for start in range(0, len(crops), self.batch):
                chunk = crops[start:start + self.batch]
                now = time.monotonic()
                if should_stop is not None and should_stop():
                    break
                if deadline is not None and now + self.seconds_per_line * len(chunk) * 0.8 > deadline:
                    break
                encoded = []
                widest = 0.0
                for crop in chunk:
                    buffer = io.BytesIO()
                    Image.fromarray(crop).save(buffer, format="PNG")
                    encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
                    widest = max(widest, crop.shape[1] / max(1, crop.shape[0]))
                max_tokens = int(min(160, max(24, widest * 4.5)))
                timeout = (deadline - now + 20.0) if deadline is not None else 600.0
                answer = self._call({"op": "recognize", "images": encoded, "max_tokens": max_tokens}, timeout=timeout,
                                    should_stop=should_stop)
                if not answer.get("ok"):
                    break
                elapsed = time.monotonic() - now
                self.seconds_per_line = 0.7 * self.seconds_per_line + 0.3 * (elapsed / max(1, len(chunk)))
                for offset, item in enumerate(answer.get("lines") or []):
                    results[start + offset] = (str(item.get("text") or ""), float(item.get("confidence") or 0.0))
        return results


_handwriting: HandwritingOCR | None = None
_printed: PrintedTextOCR | None = None
_singleton_lock = threading.Lock()


def handwriting_engine() -> HandwritingOCR:
    global _handwriting
    with _singleton_lock:
        if _handwriting is None:
            _handwriting = HandwritingOCR()
        return _handwriting


def printed_engine() -> PrintedTextOCR:
    global _printed
    with _singleton_lock:
        if _printed is None:
            _printed = PrintedTextOCR()
        return _printed


def any_available() -> bool:
    return ocr.available() or handwriting_engine().available()


def cache_tag(languages: str = "", hint: str = "auto") -> str:
    """Part of a cache key: results made by an older engine, other languages or without the handwriting model differ."""

    return f"{ENGINE_VERSION}|lang={languages}|hint={hint}|tess={'1' if ocr.available() else '0'}|{handwriting_engine().cache_tag()}"


# ------------------------------------------------------------------------------- decisions

_WORDLIKE = re.compile(r"^[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]*[A-Za-zÄÖÜäöüß]$|^[A-Za-zÄÖÜäöüß]$")


def plausibility(text: str) -> float:
    """0..1: how much a reading looks like words rather than symbol soup (tokens with letters and vowels)."""

    tokens = [t.strip(".,;:!?()[]\"'«»„“”") for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    good = 0.0
    for token in tokens:
        letters = sum(ch.isalpha() for ch in token)
        if _WORDLIKE.match(token) and (len(token) <= 3 or re.search(r"[aeiouyäöüAEIOUYÄÖÜ]", token)):
            good += 1.0
        elif re.fullmatch(r"[A-Za-zÄÖÜäöüßµ]{1,4}(?:/[A-Za-zÄÖÜäöüßµ\d]{1,4})+", token):
            good += 1.0  # units: ml/min, mmol/l
        elif re.fullmatch(r"[\d.,%/\-+→←↑↓=<>]+", token) or re.fullmatch(r"[A-Za-z]{1,3}\d+[A-Za-zÄÖÜäöüß\-]*", token):
            good += 0.8
        elif letters >= max(2, len(token) * 0.6):
            good += 0.5
    return round(good / len(tokens), 3)


def _score(text: str, confidence: float) -> float:
    return confidence * (0.4 + 0.6 * plausibility(text))


def _fold(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


# ------------------------------------------------------------------------------- page assembly

def _line_from_segment(text: str, confidence: float, px: list[float], prepared: Any, index: int) -> OcrLine:
    words: list[OcrWord] = []
    tokens = [_normal(t) for t in text.split()]
    tokens = [t for t in tokens if t]
    joined = " ".join(tokens)
    total = max(1, len(joined))
    cursor = 0
    x, y, w, h = px
    for token in tokens:
        x0 = x + w * cursor / total
        x1 = x + w * (cursor + len(token)) / total
        words.append(OcrWord(text=token, confidence=round(confidence, 4), box=prepared.to_original_box([x0, y, x1 - x0, h]),
                             handwriting=True))
        cursor += len(token) + 1
    return OcrLine(text=joined, confidence=round(confidence, 4), box=prepared.to_original_box(px), words=words, handwriting=True,
                   engine="trocr", px=list(px), group=(-1, index))


def _join_row_lines(lines: list[OcrLine], th: float) -> list[OcrLine]:
    """Handwritten pieces of one row (split for the recogniser) become one text line again, left to right."""

    joined: list[OcrLine] = []
    for line in sorted(lines, key=lambda l: l.px[0]):
        target = next((j for j in joined if _same_row(j.px, line.px) and 0 <= line.px[0] - (j.px[0] + j.px[2]) + th < th * 13), None)
        if target is None:
            joined.append(line)
            continue
        weight = len(target.text) + len(line.text) or 1
        target.confidence = round((target.confidence * len(target.text) + line.confidence * len(line.text)) / weight, 4)
        target.text = f"{target.text} {line.text}"
        target.words.extend(line.words)
        target.px = _union([target.px, line.px])
        target.box = _round_box(_union([target.box, line.box]))
    return joined


def _group_geometric(lines: list[OcrLine]) -> list[list[OcrLine]]:
    if not lines:
        return []
    heights = sorted(line.px[3] for line in lines)
    median = heights[len(heights) // 2] or 1.0
    ordered = sorted(lines, key=lambda l: (l.px[1], l.px[0]))
    groups: list[list[OcrLine]] = []
    for line in ordered:
        placed = False
        for group in reversed(groups[-4:]):
            last = group[-1]
            box = _union([g.px for g in group])
            gap = line.px[1] - (last.px[1] + last.px[3])
            horizontal = min(box[0] + box[2], line.px[0] + line.px[2]) - max(box[0], line.px[0])
            if -median * 0.5 <= gap <= median * 1.3 and horizontal > 0 and line.handwriting == last.handwriting:
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])
    return groups


def _reading_order(groups: list[list[OcrLine]]) -> list[list[OcrLine]]:
    if not groups:
        return groups
    heights = sorted(line.px[3] for group in groups for line in group)
    median = heights[len(heights) // 2] or 1.0
    return sorted(groups, key=lambda g: (round(_union([l.px for l in g])[1] / (median * 1.5)), _union([l.px for l in g])[0]))


def _assign_roles(blocks: list[OcrBlock], groups: list[list[OcrLine]], size: tuple[int, int], diagrams: list[list[float]]) -> None:
    width, height = size
    all_heights = sorted(line.px[3] for group in groups for line in group)
    median = all_heights[len(all_heights) // 2] if all_heights else 1.0
    printed_lines = sum(1 for group in groups for line in group if not line.handwriting)
    total_lines = sum(len(group) for group in groups) or 1
    boxes = [_union([l.px for l in g]) for g in groups]
    for index, (block, group) in enumerate(zip(blocks, groups)):
        box = boxes[index]
        line_height = sorted(l.px[3] for l in group)[len(group) // 2]
        words = len(block.text.split())
        role = "body"
        if index < 3 and box[1] < height * 0.25 and line_height >= median * 1.35 and len(group) <= 2 and words <= 10:
            role = "title"
        elif len(group) == 1 and words <= 4 and len(block.text) <= 32:
            distances = []
            for other_index, other in enumerate(boxes):
                if other_index == index:
                    continue
                dx = max(other[0] - (box[0] + box[2]), box[0] - (other[0] + other[2]), 0.0)
                dy = max(other[1] - (box[1] + box[3]), box[1] - (other[1] + other[3]), 0.0)
                distances.append(max(dx, dy))
            isolated = not distances or min(distances) > median * 1.5
            near_diagram = any(_overlap_area([box[0] - median * 3, box[1] - median * 3, box[2] + median * 6, box[3] + median * 6], d) > 0
                               for d in diagrams)
            if near_diagram:
                role = "diagram_label"
            elif isolated:
                role = "label"
        if role == "body":
            margin = box[0] + box[2] < width * 0.18 or box[0] > width * 0.82
            if (block.handwriting and printed_lines > total_lines * 0.5) or (margin and box[2] < width * 0.25):
                role = "annotation"
        block.role = role


def _crop(gray: np.ndarray, px: list[float], th: float) -> np.ndarray:
    import cv2

    h, w = gray.shape[:2]
    pad_x, pad_y = th * 0.6, th * 0.35
    x0, y0 = int(max(0, px[0] - pad_x)), int(max(0, px[1] - pad_y))
    x1, y1 = int(min(w, px[0] + px[2] + pad_x)), int(min(h, px[1] + px[3] + pad_y))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.full((8, 8), 255, np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)


def read_page(image_path: str | Path | np.ndarray, *, hint: str = "auto", languages: str | None = None, sample: str = "",
              should_stop: ShouldStop = None, budget_seconds: float = 150.0, printed: PrintedTextOCR | None = None,
              handwriting: HandwritingOCR | None = None, prepare_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read one page image.  ``hint``: "auto", "printed", "handwriting", or "goodnotes" (auto, leaning to handwriting).

    Returns ``{"text", "words", "blocks", "variants", "engine", "languages", "handwriting", "mean_confidence", "seconds",
    "warnings", "preprocess", "version"}``; ``text`` is "" when nothing could be read.  Cancelled pages return what was
    read so far with ``"cancelled": True`` -- callers must not cache such a result."""

    started = time.monotonic()
    deadline = started + max(5.0, budget_seconds)
    printed = printed or printed_engine()
    handwriting = handwriting or handwriting_engine()
    hint = hint if hint in {"auto", "printed", "handwriting", "goodnotes"} else "auto"
    lang = languages if languages is not None else (ocr.choose_languages(sample) if printed.available() else "deu")
    result: dict[str, Any] = {"text": "", "words": [], "blocks": [], "variants": [], "engine": "", "languages": lang or "",
                              "handwriting": False, "mean_confidence": None, "seconds": 0.0, "warnings": [], "preprocess": [],
                              "version": ENGINE_VERSION, "cancelled": False}
    warnings: list[str] = result["warnings"]

    def stopped() -> bool:
        return bool(should_stop is not None and should_stop())

    from study import preprocess

    try:
        prepared = preprocess.prepare(image_path, **(prepare_options or {}))
    except Exception as exc:  # noqa: BLE001 - an unreadable image is a warning, not a crash
        warnings.append(f"Bild nicht lesbar ({type(exc).__name__})")
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
    result["preprocess"] = list(prepared.steps)
    th = prepared.text_height or 24.0

    # 1. printed reading
    printed_lines: list[OcrLine] = []
    if hint != "handwriting" or not handwriting.available():
        if printed.available():
            read = printed.read(prepared, languages=lang, timeout=max(5.0, deadline - time.monotonic()), should_stop=should_stop)
            if read is None:
                if stopped():
                    result["cancelled"] = True
                else:
                    warnings.append("Tesseract hat die Seite nicht gelesen.")
            else:
                printed_lines = read
        if hint == "handwriting" and not handwriting.available():
            warnings.append(handwriting.reason())

    # 2. handwriting where the printed reading is weak or missing
    hw_lines: list[OcrLine] = []
    variants: list[dict[str, Any]] = []
    replaced: set[int] = set()
    page_conf = _mean([l.confidence for l in printed_lines], [len(l.text) for l in printed_lines])
    want_hw = hint in {"handwriting", "goodnotes"} or not printed_lines or (page_conf is not None and page_conf < 0.88) \
        or any(l.confidence < 0.6 for l in printed_lines)
    if hint == "printed":
        want_hw = False
    if want_hw and not result["cancelled"] and handwriting.available():
        segments, diagrams = segment_lines(prepared.image, text_height=th)
        # measured on the evaluation set (D:/JarvisLocal/ocr_eval): Tesseract lines at >= 0.6 confidence were better than
        # TrOCR almost always; below that, and wherever Tesseract read nothing, TrOCR usually wins
        low_threshold = 0.85 if hint in {"handwriting", "goodnotes"} else 0.60
        candidates: list[tuple[float, int, list[float], list[int]]] = []
        for index, px in enumerate(segments):
            area = max(1.0, px[2] * px[3])
            members = []
            covered = 0.0
            for line_index, line in enumerate(printed_lines):
                overlap = _overlap_area(px, line.px)
                if overlap > 0.5 * max(1.0, line.px[2] * line.px[3]) or (overlap > 0.3 * area):
                    members.append(line_index)
                    covered += overlap
            conf = _mean([printed_lines[i].confidence for i in members], [len(printed_lines[i].text) for i in members])
            if hint == "handwriting" or not members or covered < 0.25 * area or (conf is not None and conf < low_threshold):
                candidates.append((conf if conf is not None else 0.0, index, px, members))
        # lowest printed confidence first: the budget goes where Tesseract is worst
        candidates.sort(key=lambda c: (c[0], c[2][1]))
        taken_members: set[int] = set()
        filtered = []
        for cand in candidates:
            if any(m in taken_members for m in cand[3]):
                # a Tesseract line spanning two segments: compare it once, against the first segment
                cand = (cand[0], cand[1], cand[2], [m for m in cand[3] if m not in taken_members])
            taken_members.update(cand[3])
            filtered.append(cand)
        crops = [_crop(prepared.image, cand[2], th) for cand in filtered]
        readings = handwriting.recognize(crops, language=lang or "deu", deadline=deadline, should_stop=should_stop) if crops else []
        unread = sum(1 for r in readings if r is None)
        if stopped():
            result["cancelled"] = True
        elif unread:
            warnings.append(f"Zeitbudget der Handschrifterkennung erreicht: {unread} von {len(crops)} Zeilen nur mit Tesseract gelesen.")
        for cand, reading in zip(filtered, readings):
            if reading is None:
                continue
            conf_p, index, px, members = cand
            text_h = " ".join(_normal(t) for t in reading[0].split() if _normal(t))
            conf_h = float(reading[1])
            text_p = " ".join(printed_lines[m].text for m in sorted(members, key=lambda m: printed_lines[m].px[0]))
            if not text_h:
                continue
            line = _line_from_segment(text_h, conf_h, px, prepared, index)
            score_h, score_p = _score(text_h, conf_h), (_score(text_p, conf_p) if members else 0.0)
            # TrOCR token probabilities run high even on misreadings: it must beat Tesseract clearly unless the source is known
            # to be handwritten
            margin_factor, margin = (1.0, 0.0) if hint in {"handwriting", "goodnotes"} else (1.2, 0.05)
            if not members:
                # ink Tesseract did not read at all: keep only something word-like (not a lone dash from a stroke)
                if conf_h >= 0.35 and plausibility(text_h) >= 0.3 and re.search(r"[A-Za-zÄÖÜäöüß]{2}|\d", text_h):
                    hw_lines.append(line)
                continue
            if score_h > score_p * margin_factor + margin:
                hw_lines.append(line)
                replaced.update(members)
                if text_p and _fold(text_p) != _fold(text_h) and score_p >= 0.3:
                    variants.append({"text": text_p, "engine": "tesseract", "confidence": round(conf_p, 4), "box": line.box})
            elif _fold(text_p) != _fold(text_h) and score_h >= 0.3:
                variants.append({"text": text_h, "engine": "trocr", "confidence": round(conf_h, 4), "box": line.box})
        result["_diagrams"] = diagrams
    elif want_hw and not handwriting.available() and hint != "printed" and (hint == "goodnotes" or (page_conf is not None and page_conf < 0.7)):
        reason = handwriting.reason()
        if reason and reason not in warnings:
            warnings.append(reason)

    # 3. assemble blocks, order, roles, text and word offsets
    kept_printed = [l for i, l in enumerate(printed_lines) if i not in replaced]
    diagrams = result.pop("_diagrams", [])
    if hw_lines:
        hw_lines = _join_row_lines(hw_lines, th)
        groups = _reading_order(_group_geometric(kept_printed + hw_lines))
    else:
        by_group: dict[tuple[int, int], list[OcrLine]] = {}
        for line in kept_printed:
            by_group.setdefault(line.group, []).append(line)
        groups = list(by_group.values())
    blocks: list[OcrBlock] = []
    words: list[dict[str, Any]] = []
    parts: list[str] = []
    length = 0
    for order, group in enumerate(groups):
        block_start = length + (2 if parts else 0)
        if parts:
            parts.append("\n\n")
            length += 2
        for line_index, line in enumerate(group):
            if line_index:
                parts.append("\n")
                length += 1
            for word_index, word in enumerate(line.words):
                if word_index:
                    parts.append(" ")
                    length += 1
                word.start, word.end = length, length + len(word.text)
                parts.append(word.text)
                length += len(word.text)
                words.append({"text": word.text, "start": word.start, "end": word.end, "box": _round_box(word.box),
                              "confidence": word.confidence, "handwriting": word.handwriting, "block": order})
        block_text = "\n".join(line.text for line in group)
        hand_chars = sum(len(l.text) for l in group if l.handwriting)
        blocks.append(OcrBlock(text=block_text, confidence=_mean([l.confidence for l in group], [len(l.text) for l in group]) or 0.0,
                               box=_round_box(_union([l.box for l in group])), lines=group, order=order,
                               handwriting=hand_chars * 2 >= max(1, sum(len(l.text) for l in group)),
                               char_start=block_start, char_end=block_start + len(block_text)))
    _assign_roles(blocks, groups, prepared.size, diagrams)
    text = "".join(parts)
    all_lines = [l for g in groups for l in g]
    hand = sum(len(l.text) for l in all_lines if l.handwriting)
    total = sum(len(l.text) for l in all_lines)
    result.update({
        "text": text, "words": words, "blocks": [b.to_dict() for b in blocks], "variants": variants,
        "engine": "+".join(name for name, used in (("tesseract", any(not l.handwriting for l in all_lines) or (printed_lines and not hw_lines)),
                                                   ("trocr", bool(hw_lines))) if used),
        "handwriting": bool(total and hand * 10 >= total * 3),
        "mean_confidence": _mean([l.confidence for l in all_lines], [len(l.text) for l in all_lines]),
        "seconds": round(time.monotonic() - started, 2),
    })
    if hw_lines and handwriting.model_for(lang or "deu") is not None:
        result["handwriting_model"] = handwriting.model_for(lang or "deu").name if handwriting._recognizer is None else "custom"
    result["warnings"] = [w for w in warnings if w]
    return result


def _mean(values: list[float], weights: list[int]) -> float | None:
    total = sum(weights)
    if not values or not total:
        return None
    return round(sum(v * w for v, w in zip(values, weights)) / total, 4)


def status() -> dict[str, Any]:
    return {"printed": {"available": ocr.available(), "engine": "tesseract"}, "handwriting": handwriting_engine().status(),
            "version": ENGINE_VERSION}


# ------------------------------------------------------------------------------- caching + index blocks

def page_hash(image_path: str | Path, tag: str) -> str:
    """sha1 of the image bytes plus the engine tag: the same page read by the same engine is never read twice."""

    import hashlib

    digest = hashlib.sha1()
    with open(image_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(b"|" + tag.encode("utf-8"))
    return digest.hexdigest()


def read_page_cached(image_path: str | Path, *, cache_root: Path | None, hint: str = "auto", languages: str | None = None,
                     sample: str = "", should_stop: ShouldStop = None, budget_seconds: float = 150.0) -> tuple[dict[str, Any], bool]:
    """(result, cache hit).  ``cache_root / "by_hash" / <sha1>.json`` is shared by all documents; cancelled readings are
    never stored, so an interrupted import resumes where it stopped."""

    target: Path | None = None
    if cache_root is not None:
        try:
            target = Path(cache_root) / "by_hash" / f"{page_hash(image_path, cache_tag(languages or '', hint))}.json"
            if target.is_file():
                cached = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and "text" in cached and "words" in cached:
                    return cached, True
        except (OSError, ValueError):
            target = target if target is not None else None
    result = read_page(image_path, hint=hint, languages=languages, sample=sample, should_stop=should_stop, budget_seconds=budget_seconds)
    if target is not None and not result.get("cancelled"):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
            temp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, target)
        except OSError:
            pass
    return result, False


def unit_blocks(result: dict[str, Any], text: str) -> list[Any]:
    """``study.model.Block`` objects for a unit whose text is ``text`` (the cleaned page text of ``result``)."""

    from study.model import Block

    blocks: list[Any] = []
    same = str(result.get("text") or "") == text
    cursor = 0
    for item in result.get("blocks") or []:
        block_text = str(item.get("text") or "")
        if not block_text:
            continue
        if same:
            start = int(item.get("char_start", 0))
        else:
            start = text.find(block_text, cursor)
            if start < 0:
                continue
        end = start + len(block_text)
        cursor = end
        blocks.append(Block(text=block_text, kind="handwriting" if item.get("handwriting") else "ocr_block", char_start=start, char_end=end,
                            box=list(item.get("box") or []) or None, confidence=item.get("confidence"), role=str(item.get("role") or "body")))
    if not blocks and text:
        blocks.append(Block(text=text, kind="page_text", char_start=0, char_end=len(text), confidence=result.get("mean_confidence")))
    return blocks


def variant_text(result: dict[str, Any]) -> str:
    """The alternative readings as one extra searchable passage ("" when there are none)."""

    seen: set[str] = set()
    parts = []
    for variant in result.get("variants") or []:
        value = _normal(variant.get("text"))
        if value and value.lower() not in seen:
            seen.add(value.lower())
            parts.append(value)
    return "\n".join(parts)


__all__ = ["OcrWord", "OcrLine", "OcrBlock", "PrintedTextOCR", "HandwritingOCR", "read_page", "read_page_cached", "segment_lines",
           "plausibility", "handwriting_engine", "printed_engine", "any_available", "cache_tag", "status", "parse_tsv",
           "lines_from_data", "page_hash", "unit_blocks", "variant_text", "ENGINE_VERSION", "ROLES"]
