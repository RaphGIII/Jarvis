"""OCR for scanned pages and handwriting -- used only when a page carries no text of its own.

The engine is Tesseract through pytesseract when the Tesseract program is installed.  On a
machine without it, ``available()`` is False and the page is indexed without text: it stays
openable by title and page, and says so, instead of pretending to be searchable.

Languages: the program's own tessdata folder is admin-only and usually holds just ``eng`` and
``osd``.  German therefore lives in a user tessdata folder (``ZEUS_TESSDATA_DIR``, else
``D:\\JarvisLocal\\tessdata``) that ``ensure_language`` fills without admin rights; every call
into Tesseract then passes ``--tessdata-dir`` so the engine reads from there.  The owner studies
in German, so German comes first unless a sample of the document is clearly English.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

_CANDIDATES = (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")
_LOCAL_ROOT = Path(r"D:\JarvisLocal")
_DOWNLOAD_URL = "https://github.com/tesseract-ocr/tessdata/raw/main/{lang}.traineddata"
_MIN_MODEL_BYTES = 1_000_000
_BASE_LANGS = ("eng", "osd")

_lang_cache: dict[tuple[str, str], frozenset[str]] = {}
_lang_lock = threading.Lock()


# ------------------------------------------------------------------------------- engine + data

def tesseract_path() -> str:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return ""


def available() -> bool:
    if not tesseract_path():
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def program_tessdata_dir() -> Path | None:
    exe = tesseract_path()
    if not exe:
        return None
    folder = Path(exe).resolve().parent / "tessdata"
    return folder if folder.is_dir() else None


def tessdata_dir() -> Path | None:
    """Where languages live and get installed: ZEUS_TESSDATA_DIR, else D:\\JarvisLocal\\tessdata, else the program's folder."""

    env = os.environ.get("ZEUS_TESSDATA_DIR", "").strip()
    if env:
        return Path(env)
    if _LOCAL_ROOT.is_dir():
        return _LOCAL_ROOT / "tessdata"
    return program_tessdata_dir()


def _has_models(folder: Path | None) -> bool:
    try:
        return bool(folder and folder.is_dir() and any(folder.glob("*.traineddata")))
    except OSError:
        return False


def active_tessdata_dir() -> Path | None:
    """The folder Tesseract actually reads: the user folder once it holds models, the program's folder until then."""

    target = tessdata_dir()
    if os.environ.get("ZEUS_TESSDATA_DIR", "").strip():
        return target
    if _has_models(target):
        return target
    return program_tessdata_dir()


def _user_dir_flag() -> str:
    """``--tessdata-dir <dir>`` when reading from a folder other than the program's own, else ""."""

    active = active_tessdata_dir()
    program = program_tessdata_dir()
    if active is None:
        return ""
    if program is not None and _same_path(active, program):
        return ""
    return f"--tessdata-dir {_config_path(active)}"


def _same_path(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _config_path(folder: Path) -> str:
    # pytesseract splits ``config`` with shlex(posix=False) on Windows, which keeps quotes as literal
    # characters that then reach Tesseract; forward slashes and no quotes survive when there is no space.
    # A folder with a space goes through its 8.3 short name on Windows (which has none).
    text = str(folder)
    if " " in text and os.name == "nt":
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(text, buffer, len(buffer)) and " " not in buffer.value:
                text = buffer.value
        except Exception:  # noqa: BLE001 - keep the long name
            pass
    text = text.replace("\\", "/")
    return f'"{text}"' if " " in text else text


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def languages() -> set[str]:
    """The languages Tesseract can load right now (from the active tessdata folder); cached until an install."""

    exe = tesseract_path()
    if not exe:
        return set()
    active = active_tessdata_dir()
    key = (exe, str(active or ""))
    with _lang_lock:
        cached = _lang_cache.get(key)
    if cached is not None:
        return set(cached)
    cmd = [exe, "--list-langs"]
    if active is not None:
        cmd += ["--tessdata-dir", str(active)]
    found: set[str] = set()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=20, creationflags=_no_window())
        output = (proc.stdout or b"") + b"\n" + (proc.stderr or b"")
        for raw in output.decode("utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.lower().startswith("list of available") or " " in line or ":" in line:
                continue
            if re.fullmatch(r"[A-Za-z0-9_\-]+", line):
                found.add(line)
    except (OSError, subprocess.SubprocessError):
        return set()
    with _lang_lock:
        _lang_cache[key] = frozenset(found)
    return found


def invalidate_languages() -> None:
    with _lang_lock:
        _lang_cache.clear()


# ------------------------------------------------------------------------------- language choice

_DE_WORDS = frozenset(
    "der die das und ist nicht ein eine einer eines einem den dem des mit von zu auf für im in sich auch "
    "als wird werden wie bei oder aus nach über durch sind war hat haben kann können wenn dass daß welche "
    "welcher welches noch nur zum zur vom unter zwischen diese dieser dieses wir ihr sie es man sowie "
    "beim bzw z.b. mehr sehr wo warum weil jedoch ohne gegen".split())
_EN_WORDS = frozenset(
    "the and is are of to in that it with for on as was were be by this which from or an at not have has "
    "can will what when where why how these those their there they we you but if into than then also "
    "between about does do".split())


def detect_language(text: str) -> str:
    """"de", "en" or "" (too little to tell): stop words plus umlauts, no dependency."""

    if not text:
        return ""
    lowered = text.lower()
    tokens = re.findall(r"[a-zäöüß]+", lowered)
    de = sum(1 for t in tokens if t in _DE_WORDS)
    en = sum(1 for t in tokens if t in _EN_WORDS)
    de += min(5, sum(lowered.count(ch) for ch in "äöüß"))
    if de + en < 2:
        return ""
    if de >= 2 and de > en * 1.2:
        return "de"
    if en >= 2 and en > de * 1.2:
        return "en"
    return ""


def choose_languages(sample: str = "") -> str:
    """The Tesseract ``lang`` string: German unless the sample is clearly English; only installed languages.

    One model, not a combination: measured on a handwritten German page, ``deu`` read "direkter Weg fördert",
    "Prüfungsfrage" and "über" correctly while ``deu+eng`` let the English model win ("divekter Weg fordert",
    "Prifungsfrage", "uber").  The German model reads Latin and English technical terms well enough."""

    installed = languages()
    has_de, has_en = "deu" in installed, "eng" in installed
    if detect_language(sample) == "en" and has_en:
        return "eng"
    if has_de:
        return "deu"
    if has_en:
        return "eng"
    others = sorted(lang for lang in installed if lang != "osd")
    return others[0] if others else ""


# ------------------------------------------------------------------------------- reading images

def _prepare(languages_: str | None, sample: str) -> tuple[str, str]:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = tesseract_path()
    lang = languages_ if languages_ is not None else choose_languages(sample)
    return lang, _user_dir_flag()


def image_text(path: str | Path, *, languages: str | None = None, sample: str = "") -> str:
    """Text read from an image file; "" when OCR is unavailable or reads nothing."""

    if not available():
        return ""
    import pytesseract
    from PIL import Image

    lang, config = _prepare(languages, sample)
    try:
        with Image.open(path) as image:
            return str(pytesseract.image_to_string(image, lang=lang or None, config=config) or "").strip()
    except Exception:  # noqa: BLE001 - a language pack or an image the engine refuses
        try:
            with Image.open(path) as image:
                return str(pytesseract.image_to_string(image) or "").strip()
        except Exception:  # noqa: BLE001
            return ""


_WS = re.compile(r"[ \t\u00a0\n]+")


def _normal_word(raw: Any) -> str:
    # the same normalisation the parsers' _clean applies, so the stored unit text equals this text
    text = str(raw or "").replace("\r", " ").replace("\u00ad", "").replace("\x00", "")
    return _WS.sub(" ", text).strip()


def words_from_data(data: dict[str, list[Any]], width: float, height: float) -> tuple[str, list[dict[str, Any]]]:
    """Text + word boxes from ``image_to_data`` output (Output.DICT)."""

    parts: list[str] = []
    words: list[dict[str, Any]] = []
    length = 0
    previous: tuple[Any, Any, Any] | None = None
    count = len(data.get("text") or [])
    for i in range(count):
        word = _normal_word(data["text"][i])
        if not word:
            continue
        block, par, line = data.get("block_num", [0] * count)[i], data.get("par_num", [0] * count)[i], data.get("line_num", [0] * count)[i]
        if previous is None:
            sep = ""
        elif (block, par) != previous[:2]:
            sep = "\n\n"
        elif line != previous[2]:
            sep = "\n"
        else:
            sep = " "
        previous = (block, par, line)
        parts.append(sep)
        length += len(sep)
        start = length
        parts.append(word)
        length += len(word)
        left, top = float(data["left"][i]), float(data["top"][i])
        w, h = float(data["width"][i]), float(data["height"][i])
        box = [round(left / width, 5), round(top / height, 5), round(w / width, 5), round(h / height, 5)] if width and height else [0.0, 0.0, 0.0, 0.0]
        words.append({"text": word, "start": start, "end": length, "box": box})
    return "".join(parts), words


def image_words(path: str | Path, *, languages: str | None = None, sample: str = "") -> dict[str, Any]:
    """``{"text", "words": [{"text", "start", "end", "box": [x, y, w, h] as fractions}], "languages"}``."""

    empty: dict[str, Any] = {"text": "", "words": [], "languages": ""}
    if not available():
        return empty
    import pytesseract
    from PIL import Image

    lang, config = _prepare(languages, sample)
    attempts = [(lang, config), ("", "")]
    for attempt_lang, attempt_config in attempts:
        try:
            with Image.open(path) as image:
                width, height = float(image.width), float(image.height)
                data = pytesseract.image_to_data(image, lang=attempt_lang or None, config=attempt_config,
                                                 output_type=pytesseract.Output.DICT)
        except Exception:  # noqa: BLE001 - try the engine's defaults once more
            continue
        text, words = words_from_data(data, width, height)
        return {"text": text, "words": words, "languages": attempt_lang or "eng"}
    return empty


def align_words(ocr_words: dict[str, Any], text: str) -> dict[str, Any]:
    """The same words with offsets recomputed against ``text`` (when a later cleanup changed the text)."""

    if ocr_words.get("text") == text:
        return ocr_words
    aligned: list[dict[str, Any]] = []
    cursor = 0
    for word in ocr_words.get("words") or []:
        found = text.find(word["text"], cursor)
        if found < 0:
            continue
        aligned.append({**word, "start": found, "end": found + len(word["text"])})
        cursor = found + len(word["text"])
    return {**ocr_words, "text": text, "words": aligned}


# ------------------------------------------------------------------------------- highlighting

def _phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    phrase = unicodedata.normalize("NFC", phrase).strip()
    if not phrase:
        return None
    pieces: list[str] = []
    joint = r"(?:-[ \t]*\n\s*)?"  # a word hyphenated across a line break
    previous_space = False
    for index, ch in enumerate(phrase):
        if ch.isspace():
            if not previous_space:
                pieces.append(r"\s+")
            previous_space = True
            continue
        if index and not previous_space and ch != "-" and phrase[index - 1] != "-":
            pieces.append(joint)
        pieces.append(r"-\s*" if ch == "-" else re.escape(ch))
        previous_space = False
    return re.compile("".join(pieces), re.IGNORECASE)


def phrase_boxes(ocr_words: dict[str, Any], phrases: list[str]) -> list[dict[str, Any]]:
    """Every occurrence of every phrase as rectangles, one per text line the occurrence spans."""

    text = unicodedata.normalize("NFC", str(ocr_words.get("text") or ""))
    words = list(ocr_words.get("words") or [])
    results: list[dict[str, Any]] = []
    for phrase in phrases:
        pattern = _phrase_pattern(phrase)
        rects: list[list[float]] = []
        occurrences: list[dict[str, Any]] = []
        if pattern is not None:
            for match in pattern.finditer(text):
                start, end = match.span()
                first = len(rects)
                by_line: dict[int, list[float]] = {}
                order: list[int] = []
                for word in words:
                    if word["end"] <= start or word["start"] >= end:
                        continue
                    line = text.count("\n", 0, word["start"])
                    x, y, w, h = (float(v) for v in word["box"])
                    if line not in by_line:
                        by_line[line] = [x, y, x + w, y + h]
                        order.append(line)
                    else:
                        r = by_line[line]
                        r[0], r[1], r[2], r[3] = min(r[0], x), min(r[1], y), max(r[2], x + w), max(r[3], y + h)
                for line in order:
                    x0, y0, x1, y1 = by_line[line]
                    rects.append([round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)])
                if len(rects) > first:
                    occurrences.append({"start": start, "rects": rects[first:]})
        results.append({"phrase": phrase, "rects": rects, "occurrences": occurrences})
    return results


# ------------------------------------------------------------------------------- installing

def ensure_language(lang: str = "deu", *, download: bool = True) -> dict[str, Any]:
    """Install ``lang`` into the user tessdata folder (no admin rights): eng/osd copied, the model downloaded."""

    result: dict[str, Any] = {"ok": False, "language": lang, "path": "", "tessdata_dir": "", "error": ""}
    if not re.fullmatch(r"[A-Za-z0-9_]+", lang or ""):
        result["error"] = "invalid language name"
        return result
    target = tessdata_dir()
    if target is None:
        result["error"] = "no tessdata folder (Tesseract not installed)"
        return result
    result["tessdata_dir"] = str(target)
    model = target / f"{lang}.traineddata"
    result["path"] = str(model)
    try:
        target.mkdir(parents=True, exist_ok=True)
        program = program_tessdata_dir()
        if program is not None and not _same_path(program, target):
            for base in _BASE_LANGS:
                source, dest = program / f"{base}.traineddata", target / f"{base}.traineddata"
                if source.is_file() and not dest.is_file():
                    shutil.copy2(source, dest)
        if not (model.is_file() and model.stat().st_size > _MIN_MODEL_BYTES):
            if not download:
                result["error"] = f"{lang}.traineddata fehlt in {target}"
                return result
            fd, temp_name = tempfile.mkstemp(prefix=f".{lang}.", suffix=".part", dir=str(target))
            try:
                with os.fdopen(fd, "wb") as handle:
                    request = urllib.request.Request(_DOWNLOAD_URL.format(lang=lang), headers={"User-Agent": "ZEUS-study-ocr"})
                    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
                        shutil.copyfileobj(response, handle, length=1 << 20)
                size = os.path.getsize(temp_name)
                if size <= _MIN_MODEL_BYTES:
                    raise OSError(f"download too small ({size} bytes)")
                os.replace(temp_name, model)
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never raised
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    invalidate_languages()
    if lang not in languages():
        result["error"] = f"Tesseract lädt {lang} aus {active_tessdata_dir()} nicht"
        return result
    result["ok"] = True
    return result


def status() -> dict[str, object]:
    ok = available()
    langs = sorted(languages()) if ok else []
    german = "deu" in langs
    folder = tessdata_dir()
    if not ok:
        note = "Keine OCR-Engine installiert: handschriftliche oder gescannte Seiten sind öffnenbar, aber nicht nach Inhalt durchsuchbar."
    elif not german:
        where = str(folder) if folder is not None else r"D:\JarvisLocal\tessdata"
        note = (f"Deutsches Sprachpaket für die Texterkennung fehlt: deu.traineddata in {where} ablegen "
                "(zusammen mit eng.traineddata und osd.traineddata, ohne Adminrechte möglich). "
                "Bis dahin werden gescannte Seiten nur englisch gelesen und Umlaute gehen verloren.")
    else:
        note = ""
    return {"available": ok, "engine": "tesseract" if tesseract_path() else "",
            "tessdata_dir": str(active_tessdata_dir() or folder or ""), "languages": langs, "german": german, "note": note,
            "handwriting": handwriting_status()}


def handwriting_status() -> dict[str, object]:
    """The handwriting engine: ``{"available", "model", "reason", ...}`` -- never starts the model."""

    try:
        from study import ocr_engines

        return ocr_engines.handwriting_engine().status()
    except Exception as exc:  # noqa: BLE001 - status never raises
        return {"available": False, "engine": "trocr", "model": "", "reason": f"Handschrifterkennung nicht ladbar ({type(exc).__name__})."}


def read_page(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Printed + handwriting OCR of one page image with blocks, roles and confidences (see ``study.ocr_engines.read_page``)."""

    from study import ocr_engines

    return ocr_engines.read_page(path, **kwargs)
