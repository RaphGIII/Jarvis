"""German OCR for the study library: tessdata resolution, language choice, word boxes and highlight rectangles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from study import ocr, parsers
from study.model import NormalizedDocument, Unit


# ------------------------------------------------------------------ language choice

@pytest.fixture
def installed(monkeypatch):
    def set_langs(*langs: str) -> None:
        monkeypatch.setattr(ocr, "languages", lambda: set(langs))
    return set_langs


GERMAN = "Der Sympathikus steigert die Herzfrequenz, und über die Beta-Rezeptoren wird das Herz schneller."
ENGLISH = "The heart is a muscle that pumps blood through the body, and the lungs are where it gets oxygen."


def test_detect_language():
    assert ocr.detect_language(GERMAN) == "de"
    assert ocr.detect_language(ENGLISH) == "en"
    assert ocr.detect_language("") == ""
    assert ocr.detect_language("Herz 42") == ""


def test_german_first_for_german_or_unknown_samples(installed):
    installed("eng", "deu", "osd")
    assert ocr.choose_languages(GERMAN) == "deu"
    assert ocr.choose_languages("") == "deu"
    assert ocr.choose_languages("x 17 y") == "deu"


def test_english_first_when_the_sample_is_clearly_english(installed):
    installed("eng", "deu", "osd")
    assert ocr.choose_languages(ENGLISH) == "eng"


def test_only_english_when_german_is_missing(installed):
    installed("eng", "osd")
    assert ocr.choose_languages(GERMAN) == "eng"
    assert ocr.choose_languages(ENGLISH) == "eng"


def test_never_an_uninstalled_language(installed):
    for langs in [("eng",), ("deu",), ("osd", "fra"), ("eng", "deu"), ()]:
        installed(*langs)
        for sample in (GERMAN, ENGLISH, ""):
            chosen = ocr.choose_languages(sample)
            assert all(part in langs for part in chosen.split("+") if part), (langs, chosen)
            assert "osd" not in chosen.split("+")


# ------------------------------------------------------------------ tessdata + status

def test_tessdata_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEUS_TESSDATA_DIR", str(tmp_path / "td"))
    assert ocr.tessdata_dir() == tmp_path / "td"
    assert ocr.active_tessdata_dir() == tmp_path / "td"


def test_user_dir_is_passed_to_tesseract(monkeypatch, tmp_path):
    folder = tmp_path / "tessdata"
    folder.mkdir()
    monkeypatch.setenv("ZEUS_TESSDATA_DIR", str(folder))
    monkeypatch.setattr(ocr, "program_tessdata_dir", lambda: tmp_path / "program")
    flag = ocr._user_dir_flag()
    assert flag.startswith("--tessdata-dir ")
    assert '"' not in flag or " " in str(folder)


def test_without_env_the_local_folder_is_the_install_target(monkeypatch):
    monkeypatch.delenv("ZEUS_TESSDATA_DIR", raising=False)
    target = ocr.tessdata_dir()
    if ocr._LOCAL_ROOT.is_dir():
        assert target == ocr._LOCAL_ROOT / "tessdata"
    else:
        assert target == ocr.program_tessdata_dir()


def test_status_names_the_missing_german_pack(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEUS_TESSDATA_DIR", str(tmp_path / "tessdata"))
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "tesseract_path", lambda: r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    monkeypatch.setattr(ocr, "languages", lambda: {"eng", "osd"})
    status = ocr.status()
    assert status["german"] is False and status["languages"] == ["eng", "osd"]
    assert "deu.traineddata" in status["note"] and str(tmp_path / "tessdata") in status["note"]
    assert "Sprachpaket für die Texterkennung fehlt" in status["note"]
    monkeypatch.setattr(ocr, "languages", lambda: {"eng", "osd", "deu"})
    status = ocr.status()
    assert status["german"] is True and status["note"] == ""


def test_ensure_language_without_download_reports_the_gap(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEUS_TESSDATA_DIR", str(tmp_path / "tessdata"))
    monkeypatch.setattr(ocr, "program_tessdata_dir", lambda: None)
    result = ocr.ensure_language("deu", download=False)
    assert result["ok"] is False and "deu.traineddata" in result["error"]
    assert result["tessdata_dir"] == str(tmp_path / "tessdata")


# ------------------------------------------------------------------ word boxes + highlights

def _data(rows):
    keys = ("block_num", "par_num", "line_num", "left", "top", "width", "height", "text")
    return {key: [row[i] for row in rows] for i, key in enumerate(keys)}


def test_words_from_data_builds_the_indexed_text():
    data = _data([
        (1, 1, 1, 0, 0, 0, 0, ""),
        (1, 1, 1, 10, 10, 90, 20, "direkter"),
        (1, 1, 1, 110, 10, 50, 20, "Weg"),
        (1, 1, 2, 10, 40, 80, 20, "fördert"),
        (2, 1, 1, 10, 100, 120, 20, "Prüfungs\u00adfrage:"),
    ])
    text, words = ocr.words_from_data(data, 1000, 500)
    assert text == "direkter Weg\nfördert\n\nPrüfungsfrage:"
    assert parsers._clean(text) == text
    for word in words:
        assert text[word["start"]:word["end"]] == word["text"]
    assert words[0]["box"] == [0.01, 0.02, 0.09, 0.04]


def _synthetic():
    rows = [  # (text, line, x) on a 1000x1000 page, each line 20 high
        [("Der", 0, 10), ("direkte", 0, 60), ("Weg", 0, 150), ("fördert", 0, 210)],
        [("Bewegung.", 1, 10), ("Dopamin", 1, 120), ("fördert", 1, 220), ("über", 1, 300)],
        [("D1-Rezep-", 2, 10)],
        [("toren", 3, 10), ("den", 3, 80), ("direkten", 3, 130), ("Weg", 3, 230)],
    ]
    parts, words = [], []
    cursor = 0
    for line in rows:
        if parts:
            parts.append("\n")
            cursor += 1
        for index, (token, number, x) in enumerate(line):
            if index:
                parts.append(" ")
                cursor += 1
            parts.append(token)
            words.append({"text": token, "start": cursor, "end": cursor + len(token),
                          "box": [x / 1000, (10 + number * 30) / 1000, len(token) * 10 / 1000, 20 / 1000]})
            cursor += len(token)
    return {"text": "".join(parts), "words": words, "languages": "deu+eng"}


def test_phrase_boxes_every_occurrence_with_umlauts():
    result = ocr.phrase_boxes(_synthetic(), ["FÖRDERT"])
    assert result[0]["phrase"] == "FÖRDERT"
    rects = result[0]["rects"]
    assert len(rects) == 2
    assert rects[0] == [0.21, 0.01, 0.07, 0.02] and rects[1] == [0.22, 0.04, 0.07, 0.02]
    # umlauts are never folded: "fordert" is not "fördert"
    assert ocr.phrase_boxes(_synthetic(), ["fordert"])[0]["rects"] == []


def test_phrase_boxes_merge_per_line_across_line_breaks_and_hyphens():
    words = _synthetic()
    result = ocr.phrase_boxes(words, ["Weg fördert Bewegung", "D1-Rezeptoren", "direkte Weg", "Dopamin fördert über"])
    by_phrase = {item["phrase"]: item["rects"] for item in result}
    # one rectangle per line segment, not one per word
    assert by_phrase["Weg fördert Bewegung"] == [[0.15, 0.01, 0.13, 0.02], [0.01, 0.04, 0.09, 0.02]]
    # "D1-Rezep-" + line break + "toren"
    assert by_phrase["D1-Rezeptoren"] == [[0.01, 0.07, 0.09, 0.02], [0.01, 0.1, 0.05, 0.02]]
    assert by_phrase["Dopamin fördert über"] == [[0.12, 0.04, 0.22, 0.02]]
    # "direkte Weg" occurs once ("direkten Weg" is a different word)
    assert by_phrase["direkte Weg"] == [[0.06, 0.01, 0.12, 0.02]]


def test_align_words_follows_a_changed_text():
    words = {"text": "a  Herz", "words": [{"text": "Herz", "start": 3, "end": 7, "box": [0, 0, 0, 0]}], "languages": "deu"}
    aligned = ocr.align_words(words, "a Herz")
    assert aligned["words"][0]["start"] == 2 and aligned["text"] == "a Herz"


def test_ocr_pdf_pages_stores_boxes_matching_the_unit_text(monkeypatch, tmp_path):
    text = "direkter Weg fördert\nBewegung"
    fake = {"text": text, "languages": "deu",
            "words": [{"text": "direkter", "start": 0, "end": 8, "box": [0.1, 0.1, 0.1, 0.05]},
                      {"text": "Weg", "start": 9, "end": 12, "box": [0.25, 0.1, 0.05, 0.05]},
                      {"text": "fördert", "start": 13, "end": 20, "box": [0.32, 0.1, 0.1, 0.05]},
                      {"text": "Bewegung", "start": 21, "end": 29, "box": [0.1, 0.2, 0.12, 0.05]}]}
    seen = {}
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "languages", lambda: {"eng", "deu", "osd"})

    # the parser reads pages through the engine layer (printed + handwriting) since study.ocr_engines
    from study import ocr_engines

    def read_page(path, *, languages=None, sample="", **_):
        seen["languages"] = languages
        return fake

    monkeypatch.setattr(ocr_engines, "read_page", read_page)

    class Engine:
        def render(self, path, number, out, width):
            Path(out).write_bytes(b"png")
            return {"ok": True}

    doc = NormalizedDocument(source_type="pdf", title="Scan")
    doc.units.append(Unit(number=1, kind="page", text="Das Herz ist ein Muskel und die Pumpe des Kreislaufs.", has_text=True))
    doc.units.append(Unit(number=2, kind="page", text="", has_text=False))
    done = parsers.ocr_pdf_pages(doc, tmp_path / "x.pdf", engine=Engine(), cache_dir=tmp_path / "cache")
    assert done == 1 and seen["languages"] == "deu"
    assert doc.metadata["ocr_languages"] == "deu"
    stored = json.loads((tmp_path / "cache" / "ocr_2.json").read_text(encoding="utf-8"))
    assert stored["text"] == doc.units[1].text
    for word in stored["words"]:
        assert doc.units[1].text[word["start"]:word["end"]] == word["text"]
    assert ocr.phrase_boxes(stored, ["fördert Bewegung"])[0]["rects"] == [[0.32, 0.1, 0.1, 0.05], [0.1, 0.2, 0.12, 0.05]]


# ------------------------------------------------------------------ the real engine

def test_real_german_ocr_reads_umlauts(tmp_path):
    if not ocr.available():
        pytest.skip("Tesseract/pytesseract nicht installiert")
    if "deu" not in ocr.languages():
        pytest.skip("deu.traineddata nicht installiert (ocr.ensure_language('deu') einmal ausführen)")
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path(r"C:\Windows\Fonts\segoeui.ttf")
    if not font_path.is_file():
        pytest.skip("Schrift segoeui.ttf fehlt")
    font = ImageFont.truetype(str(font_path), 40)
    image = Image.new("RGB", (1600, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), "direkter Weg fördert Bewegung", fill="black", font=font)
    draw.text((40, 140), "Prüfungsfrage: Welche Wirkung hat Dopamin über D1-Rezeptoren?", fill="black", font=font)
    path = tmp_path / "scan.png"
    image.save(path)

    words = ocr.image_words(path)
    assert "deu" in words["languages"]
    assert "fördert" in words["text"] and "über" in words["text"]
    assert "fördert" in ocr.image_text(path)
    for word in words["words"]:
        assert words["text"][word["start"]:word["end"]] == word["text"]
    rects = ocr.phrase_boxes(words, ["fördert"])[0]["rects"]
    assert len(rects) == 1 and all(0 <= v <= 1 for v in rects[0])
