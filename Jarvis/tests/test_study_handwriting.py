"""Study OCR architecture: preprocessing geometry, engine abstraction, handwriting decisions, caching and cancellation.

Nothing here downloads anything.  Tests that need Tesseract or the local TrOCR model skip with a reason when missing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from study import ocr, ocr_engines, parsers, preprocess
from study.model import NormalizedDocument, Unit

FONT = Path(r"C:\Windows\Fonts\segoeui.ttf")
HAND_FONT = Path(r"C:\Windows\Fonts\Inkfree.ttf")
NO_GEOMETRY = {"orientation": False, "perspective": False, "deskew_page": False, "crop": False, "scale": False}


def _font(path: Path, size: int) -> ImageFont.ImageFont:
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _page(lines: list[str], *, size=(1600, 1000), font=FONT, font_size=40, top=80, step=90, title_size=None) -> np.ndarray:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        fs = title_size if (title_size and index == 0) else font_size
        draw.text((90, top + index * step), line, fill="black", font=_font(font, fs))
    return np.array(image)


def _red_box(rgb: np.ndarray) -> list[float]:
    red = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 90)
    ys, xs = np.nonzero(red)
    assert len(xs), "marker not found"
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def _close(a, b, tol):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# ------------------------------------------------------------------ preprocessing geometry

def test_boxes_map_back_through_rotation_crop_and_scaling():
    page = _page(["Frank-Starling Mechanismus der Vorlast", "direkter Weg fördert Bewegung", "Dopamin über D1-Rezeptoren"] * 3,
                 size=(900, 1200), font_size=20, step=40)
    page = cv2.copyMakeBorder(page, 200, 150, 180, 220, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.rectangle(page, (600, 900), (660, 940), (230, 20, 20), -1)
    rotated, _ = preprocess.rotate(page, 3.0)
    original = rotated.copy()
    oh, ow = rotated.shape[:2]
    marker = _red_box(rotated)
    expected = [marker[0] / ow, marker[1] / oh, marker[2] / ow, marker[3] / oh]

    prepared = preprocess.prepare(rotated, orientation=False, perspective=False)
    assert np.array_equal(rotated, original), "the caller's image is never modified"
    assert any(step.startswith("deskew") for step in prepared.steps)
    assert any(step.startswith("crop") for step in prepared.steps)
    assert any(step.startswith("scale") for step in prepared.steps), prepared.steps
    assert prepared.original_size == (ow, oh)
    back = prepared.to_original_box(_red_box(prepared.rgb))
    assert _close(back[:2], expected[:2], 0.012), (back, expected)
    assert _close(back[2:], expected[2:], 0.02), (back, expected)
    # and forth again: points are exact, a rotated box only grows to its bounding rectangle
    point = np.array([[marker[0], marker[1]]])
    moved = (prepared.matrix @ np.array([marker[0], marker[1], 1.0]))[:2]
    assert np.allclose(prepared.to_original_points(moved), point, atol=1e-6)
    forth = prepared.to_original_box(prepared.from_original_box(expected))
    assert forth[0] <= expected[0] + 1e-4 and forth[1] <= expected[1] + 1e-4 and forth[2] >= expected[2] - 1e-4


def test_deskew_corrects_a_rotated_page():
    page = _page([f"Zeile {i}: Basalganglienschleife direkter Weg fördert Bewegung" for i in range(9)])
    for angle in (4.0, -2.5):
        rotated, _ = preprocess.rotate(page, angle)
        estimate = preprocess.estimate_skew(preprocess.to_gray(rotated))
        assert abs(estimate + angle) <= 0.4, (angle, estimate)
        corrected, _, applied = preprocess.deskew(rotated)
        assert abs(preprocess.estimate_skew(preprocess.to_gray(corrected))) <= 0.4
        assert applied == estimate
    assert preprocess.estimate_skew(preprocess.to_gray(page)) == 0.0


def test_perspective_only_when_the_page_quadrilateral_is_confident():
    page = _page([f"Seite mit Notizen Nummer {i} über das Herz" for i in range(8)])
    cv2.rectangle(page, (1200, 700), (1260, 740), (230, 20, 20), -1)
    quad, confidence = preprocess.find_page_quad(page)
    assert quad is None or confidence < 0.6
    flat = preprocess.prepare(page, orientation=False)
    assert not flat.perspective and not any(s.startswith("perspective") for s in flat.steps)

    canvas = np.full((1400, 2000, 3), 60, np.uint8)
    h, w = page.shape[:2]
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                         np.float32([[250, 150], [1750, 250], [1850, 1250], [150, 1150]]))
    cv2.warpPerspective(page, matrix, (2000, 1400), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    quad, confidence = preprocess.find_page_quad(canvas)
    assert quad is not None and confidence >= 0.6
    photo = preprocess.prepare(canvas, orientation=False)
    assert photo.perspective
    marker = _red_box(canvas)
    expected = [marker[0] / 2000, marker[1] / 1400, marker[2] / 2000, marker[3] / 1400]
    back = photo.to_original_box(_red_box(photo.rgb))
    assert _close(back[:2], expected[:2], 0.012), (back, expected)

    # a textureless dark image or a noise image is never warped
    noise = np.random.default_rng(1).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    assert preprocess.find_page_quad(noise)[1] < 0.6


def test_quarter_turn_matrices_are_exact():
    image = np.zeros((30, 50), np.uint8)
    image[5, 40] = 255
    for degrees in (90, 180, 270):
        turned, matrix = preprocess.rotate90(image, degrees)
        y, x = (int(v[0]) for v in np.nonzero(turned))
        mapped = matrix @ np.array([40, 5, 1.0])
        assert (round(mapped[0]), round(mapped[1])) == (x, y)


def test_rule_lines_are_erased_but_ink_stays():
    page = _page(["Nachlast erhöht den Sauerstoffverbrauch"], font_size=44)
    for y in range(60, 1000, 90):
        cv2.line(page, (40, y), (1560, y), (170, 190, 225), 2)
    gray = preprocess.flatten_illumination(preprocess.to_gray(page))
    erased, rules = preprocess.erase_rule_lines(gray)
    assert rules >= 8
    assert erased[240, 800] == 255 and (erased < 100).sum() > 0.8 * (gray < 100).sum()


# ------------------------------------------------------------------ engines with fakes

class FakePrinted(ocr_engines.PrintedTextOCR):
    """Returns prepared-pixel words; with NO_GEOMETRY prepared pixels are original pixels."""

    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.calls = 0

    def available(self):
        return True

    def data(self, gray, *, languages, psm=3, timeout=120.0, should_stop=None):
        self.calls += 1
        keys = ("block_num", "par_num", "line_num", "left", "top", "width", "height", "conf", "text")
        return {key: [row[i] for row in self.rows] for i, key in enumerate(keys)}


def _row_words(block, par, line, x, y, text, conf, *, char_w=20, h=40):
    rows = []
    for word in text.split():
        rows.append((block, par, line, x, y, len(word) * char_w, h, conf, word))
        x += len(word) * char_w + char_w
    return rows


def _unavailable_handwriting(tmp_path):
    return ocr_engines.HandwritingOCR(model_dir=tmp_path / "kein_modell", english_dir=tmp_path / "kein_modell_en")


def test_blocks_roles_order_and_confidence_from_the_printed_engine(tmp_path):
    page = _page(["Herzmechanik", "Die Vorlast dehnt das Myokard.", "Die Nachlast ist der Widerstand."], title_size=70)
    rows = (_row_words(1, 1, 1, 90, 80, "Herzmechanik", 96, char_w=35, h=70)
            + _row_words(2, 1, 1, 90, 170, "Die Vorlast dehnt das Myokard.", 91)
            + _row_words(2, 1, 2, 90, 260, "Die Nachlast ist der Widerstand.", 89)
            + _row_words(3, 1, 1, 1350, 850, "Abb. 3", 90))
    result = ocr_engines.read_page(page, hint="printed", printed=FakePrinted(rows), handwriting=_unavailable_handwriting(tmp_path),
                                   prepare_options=NO_GEOMETRY)
    assert result["engine"] == "tesseract" and result["handwriting"] is False
    assert result["text"] == "Herzmechanik\n\nDie Vorlast dehnt das Myokard.\nDie Nachlast ist der Widerstand.\n\nAbb. 3"
    blocks = result["blocks"]
    assert [b["order"] for b in blocks] == [0, 1, 2]
    assert blocks[0]["role"] == "title" and blocks[1]["role"] == "body" and blocks[2]["role"] == "label"
    for block in blocks:
        assert 0 < block["confidence"] <= 1 and block["role"] in ocr_engines.ROLES
        assert result["text"][block["char_start"]:block["char_end"]] == block["text"]
        assert all(0 <= v <= 1 for v in block["box"])
    assert blocks[0]["box"][:2] == [round(90 / 1600, 5), round(80 / 1000, 5)]
    assert 0.85 < result["mean_confidence"] < 0.97
    # the words contract of study.ocr.image_words: offsets into exactly this text, phrase_boxes works unchanged
    assert parsers._clean(result["text"]) == result["text"]
    for word in result["words"]:
        assert result["text"][word["start"]:word["end"]] == word["text"] and 0 <= word["confidence"] <= 1
    rects = ocr.phrase_boxes(result, ["Myokard. Die Nachlast"])[0]["rects"]
    assert len(rects) == 2 and all(0 <= v <= 1 for r in rects for v in r)


def test_unavailable_handwriting_engine_gives_a_calm_german_reason_and_printed_fallback(tmp_path):
    engine = _unavailable_handwriting(tmp_path)
    assert engine.available() is False
    assert "Handschriftmodell fehlt" in engine.reason() and "Tesseract" in engine.reason()
    status = engine.status()
    assert status["available"] is False and status["reason"] == engine.reason()
    assert engine.recognize([np.zeros((10, 10, 3), np.uint8)]) == [None]
    page = _page(["direkter Weg fördert Bewegung"])
    rows = _row_words(1, 1, 1, 90, 80, "direkter Weg fördert Bewegung", 45)
    result = ocr_engines.read_page(page, hint="handwriting", printed=FakePrinted(rows), handwriting=engine, prepare_options=NO_GEOMETRY)
    assert result["engine"] == "tesseract" and "fördert" in result["text"]
    assert any("Handschriftmodell fehlt" in w for w in result["warnings"])
    # ocr.status reports the handwriting engine without starting anything
    assert "handwriting" in ocr.status() and "reason" in ocr.status()["handwriting"]


def test_missing_python_packages_are_named(tmp_path, monkeypatch):
    folder = tmp_path / "model"
    folder.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "tokenizer.json"):
        (folder / name).write_text("{}", encoding="utf-8")
    engine = ocr_engines.HandwritingOCR(model_dir=folder)
    monkeypatch.setattr(engine, "_missing_packages", lambda: ["transformers"])
    assert engine.available() is False and "Python-Pakete" in engine.reason() and "transformers" in engine.reason()


def test_auto_replaces_weak_printed_lines_with_handwriting_and_keeps_the_loser_as_variant(tmp_path):
    lines = ["Frank-Starling Vorlast Sarkomerlänge", "direkter Weg fördert Bewegung"]
    page = _page(lines, font=HAND_FONT, font_size=48, top=100, step=140)
    rows = (_row_words(1, 1, 1, 90, 100, "Frcmk-Sterling Vorlost Sarkomerlange", 52, char_w=24, h=50)
            + _row_words(1, 1, 2, 90, 240, "dinekter Weg fordert Bewegung", 58, char_w=24, h=50))
    seen = []

    def recognizer(crops, language):
        seen.append((len(crops), language))
        answers = []
        for crop in crops:
            assert crop.ndim == 3 and crop.shape[2] == 3
            answers.append((lines[len(answers) + sum(n for n, _ in seen[:-1])], 0.93))
        return answers

    handwriting = ocr_engines.HandwritingOCR(recognizer=recognizer)
    result = ocr_engines.read_page(page, hint="auto", printed=FakePrinted(rows), handwriting=handwriting, prepare_options=NO_GEOMETRY)
    assert seen and seen[0][1] == "deu" or seen[0][1] == ocr.choose_languages("")
    assert "trocr" in result["engine"] and result["handwriting"] is True
    assert "Sarkomerlänge" in result["text"] and "fördert" in result["text"]
    assert "Frcmk" not in result["text"]
    assert any("fordert" in v["text"] for v in result["variants"]), result["variants"]
    assert all(b["handwriting"] for b in result["blocks"])
    for word in result["words"]:
        assert result["text"][word["start"]:word["end"]] == word["text"] and word["handwriting"]
    rect = ocr.phrase_boxes(result, ["fördert"])[0]["rects"]
    assert len(rect) == 1 and 0.05 < rect[0][1] < 0.35


def test_auto_keeps_confident_printed_text_without_calling_the_handwriting_engine(tmp_path):
    page = _page(["Das Herz ist ein Muskel"])
    rows = _row_words(1, 1, 1, 90, 80, "Das Herz ist ein Muskel", 95)
    calls = []
    handwriting = ocr_engines.HandwritingOCR(recognizer=lambda crops, lang: calls.append(1) or [("x", 0.9)] * len(crops))
    result = ocr_engines.read_page(page, printed=FakePrinted(rows), handwriting=handwriting, prepare_options=NO_GEOMETRY)
    assert result["text"] == "Das Herz ist ein Muskel" and calls == [] and result["engine"] == "tesseract"


def test_budget_and_cancellation_stop_the_handwriting_reader(tmp_path):
    page = _page([f"Zeile {i} mit Handschrift" for i in range(6)], font=HAND_FONT, font_size=40, step=120)
    handwriting = ocr_engines.HandwritingOCR(recognizer=lambda crops, lang: [("Zeile mit Handschrift", 0.9)] * len(crops), batch=1)
    stop_after = {"n": 0}

    def should_stop():
        stop_after["n"] += 1
        return stop_after["n"] > 3

    result = ocr_engines.read_page(page, hint="handwriting", printed=FakePrinted([]), handwriting=handwriting,
                                   prepare_options=NO_GEOMETRY, should_stop=should_stop)
    assert result["cancelled"] is True


def test_plausibility_prefers_words_over_symbol_soup():
    assert ocr_engines.plausibility("direkter Weg fördert Bewegung") > 0.9
    assert ocr_engines.plausibility("D1-Rezeptoren 120 ml/min") > 0.7
    assert ocr_engines.plausibility("|| ~~ #§ Xq") < 0.4


def test_segment_lines_finds_lines_and_ignores_ruled_paper():
    page = _page([f"Notiz {i}: Aldosteron fördert Natrium" for i in range(5)], font=HAND_FONT, font_size=44, top=100, step=130)
    for y in range(90, 1000, 65):
        cv2.line(page, (30, y), (1570, y), (170, 190, 225), 2)
    boxes, diagrams = ocr_engines.segment_lines(preprocess.to_gray(page))
    assert len(boxes) == 5, boxes
    assert diagrams == []
    ys = [b[1] for b in boxes]
    assert ys == sorted(ys)


def test_parse_tsv_reads_tesseract_output():
    raw = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "1\t1\t0\t0\t0\t0\t0\t0\t100\t50\t-1\t\n"
           "5\t1\t1\t1\t1\t1\t10\t10\t40\t20\t91.5\tfördert\n")
    data = ocr_engines.parse_tsv(raw)
    assert data["text"] == ["", "fördert"] and data["conf"] == [-1.0, 91.5] and data["left"] == [0, 10]


# ------------------------------------------------------------------ caching, progress, cancellation in the parser

def _fake_result(text="direkter Weg fördert Bewegung"):
    return {"text": text, "languages": "deu", "engine": "tesseract", "handwriting": False, "mean_confidence": 0.9,
            "words": [{"text": "direkter", "start": 0, "end": 8, "box": [0.1, 0.1, 0.1, 0.05]}],
            "blocks": [{"text": text, "char_start": 0, "char_end": len(text), "box": [0.1, 0.1, 0.5, 0.05], "confidence": 0.9,
                        "role": "body", "handwriting": False, "order": 0}],
            "variants": [], "warnings": [], "cancelled": False}


class RenderEngine:
    def __init__(self):
        self.content: dict[int, bytes] = {}

    def render(self, path, number, out, width):
        Path(out).write_bytes(self.content.get(number, f"page-{number}".encode()))
        return {"ok": True}


def _scan_doc(pages=2):
    doc = NormalizedDocument(source_type="pdf", title="Scan")
    for n in range(1, pages + 1):
        doc.units.append(Unit(number=n, kind="page", text="", has_text=False))
    return doc


@pytest.fixture
def counted(monkeypatch):
    calls = []

    def read_page(path, **kwargs):
        calls.append(Path(path).read_bytes())
        return _fake_result()

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "languages", lambda: {"eng", "deu", "osd"})
    monkeypatch.setattr(ocr_engines, "read_page", read_page)
    return calls


def test_ocr_pages_are_reused_by_image_hash_and_a_changed_page_is_read_again(tmp_path, counted):
    engine = RenderEngine()
    progress = []
    done = parsers.ocr_pdf_pages(_scan_doc(), tmp_path / "a.pdf", engine=engine, cache_dir=tmp_path / "ocr" / "doc1",
                                 ocr_progress=lambda d, t: progress.append((d, t)))
    assert done == 2 and len(counted) == 2 and progress == [(1, 2), (2, 2)]
    assert len(list((tmp_path / "ocr" / "by_hash").glob("*.json"))) == 2

    # a restart / another document with the same page images: no OCR call, progress still complete
    doc = _scan_doc()
    progress.clear()
    done = parsers.ocr_pdf_pages(doc, tmp_path / "a.pdf", engine=engine, cache_dir=tmp_path / "ocr" / "doc2",
                                 ocr_progress=lambda d, t: progress.append((d, t)))
    assert done == 2 and len(counted) == 2 and progress == [(1, 2), (2, 2)]
    stored = json.loads((tmp_path / "ocr" / "doc2" / "ocr_1.json").read_text(encoding="utf-8"))
    assert stored["text"] == doc.units[0].text and "blocks" in stored and stored["engine"] == "tesseract"
    assert doc.units[0].ocr and doc.units[0].blocks[0].kind == "ocr_block" and doc.units[0].blocks[0].confidence == 0.9
    assert doc.metadata["ocr_engine"] == "tesseract" and doc.metadata["has_handwriting"] is False
    assert doc.metadata["ocr_mean_confidence"] == 0.9

    # page 2 changed: only page 2 is read again
    engine.content[2] = b"page-2-edited"
    parsers.ocr_pdf_pages(_scan_doc(), tmp_path / "a.pdf", engine=engine, cache_dir=tmp_path / "ocr" / "doc1")
    assert len(counted) == 3 and counted[-1] == b"page-2-edited"

    # an explicit shared cache root
    parsers.ocr_pdf_pages(_scan_doc(), tmp_path / "a.pdf", engine=engine, cache_dir=tmp_path / "x" / "doc", cache_root=tmp_path / "shared")
    assert len(counted) == 5 and (tmp_path / "shared" / "by_hash").is_dir()


def test_the_cache_tag_invalidates_results_of_another_engine(tmp_path, counted, monkeypatch):
    engine = RenderEngine()
    parsers.ocr_pdf_pages(_scan_doc(1), tmp_path / "a.pdf", engine=engine, cache_dir=tmp_path / "ocr" / "d")
    monkeypatch.setattr(ocr_engines, "ENGINE_VERSION", "zeus-ocr-next")
    parsers.ocr_pdf_pages(_scan_doc(1), tmp_path / "a.pdf", engine=engine, cache_dir=tmp_path / "ocr" / "d")
    assert len(counted) == 2


def test_ocr_pages_stop_between_pages_and_never_cache_a_cancelled_page(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "languages", lambda: {"deu", "osd"})
    state = {"stop": False, "calls": 0}

    def read_page(path, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            state["stop"] = True
            return {**_fake_result(), "cancelled": True}
        return _fake_result()

    monkeypatch.setattr(ocr_engines, "read_page", read_page)
    progress = []
    doc = _scan_doc(4)
    done = parsers.ocr_pdf_pages(doc, tmp_path / "a.pdf", engine=RenderEngine(), cache_dir=tmp_path / "ocr" / "d",
                                 ocr_progress=lambda d, t: progress.append((d, t)), should_stop=lambda: state["stop"])
    assert done == 1 and progress == [(1, 4)] and state["calls"] == 2
    assert doc.units[0].has_text and not doc.units[1].has_text
    assert len(list((tmp_path / "ocr" / "by_hash").glob("*.json"))) == 1


def test_parse_image_uses_the_engines_and_records_metadata(tmp_path, counted):
    path = tmp_path / "notiz.webp"
    Image.fromarray(_page(["direkter Weg fördert Bewegung"])).save(path, format="WEBP")
    progress = []
    doc = parsers.parse(path) if False else parsers.parse_image(path, cache_dir=tmp_path / "ocr" / "img", ocr_progress=lambda d, t: progress.append((d, t)))
    assert parsers.source_type_for(path) == "image" and parsers.source_type_for("a.JPEG") == "image"
    unit = doc.units[0]
    assert unit.ocr and unit.has_text and unit.text == "direkter Weg fördert Bewegung" and unit.width == 1600
    assert unit.blocks[0].role == "body" and unit.blocks[0].box == [0.1, 0.1, 0.5, 0.05]
    assert doc.metadata["ocr_engine"] == "tesseract" and doc.metadata["has_handwriting"] is False
    assert progress == [(1, 1)] and (tmp_path / "ocr" / "img" / "ocr_1.json").is_file()
    parsers.parse_image(path, cache_dir=tmp_path / "ocr" / "img2")
    assert len(counted) == 1, "the same image is not read twice"


# ------------------------------------------------------------------ real engines (skip when not installed)

def test_real_tesseract_read_page_contract(tmp_path, monkeypatch):
    if not ocr.available() or "deu" not in ocr.languages():
        pytest.skip("Tesseract mit deu.traineddata nicht installiert")
    if not FONT.is_file():
        pytest.skip("Schrift segoeui.ttf fehlt")
    monkeypatch.setattr(ocr_engines, "_handwriting", _unavailable_handwriting(tmp_path))
    path = tmp_path / "scan.png"
    rotated, _ = preprocess.rotate(_page(["Basalganglienschleife: direkter Weg fördert Bewegung",
                                           "Prüfungsfrage: Wirkung von Dopamin über D1-Rezeptoren"]), 2.0)
    Image.fromarray(rotated).save(path)
    result = ocr_engines.read_page(path, hint="printed")
    assert "fördert" in result["text"] and "Prüfungsfrage" in result["text"]
    assert result["engine"] == "tesseract" and result["mean_confidence"] > 0.7
    assert any(s.startswith("deskew") for s in result["preprocess"])
    for word in result["words"]:
        assert result["text"][word["start"]:word["end"]] == word["text"]
    rects = ocr.phrase_boxes(result, ["fördert"])[0]["rects"]
    assert len(rects) == 1
    x, y, w, h = rects[0]
    # "fördert" sits in the right half of the first line of the rotated original
    assert 0.3 < x < 0.8 and y < 0.35 and w < 0.2


def test_real_handwriting_model_reads_a_handwriting_font_page_in_auto_mode(tmp_path):
    engine = ocr_engines.handwriting_engine()
    if not engine.available():
        pytest.skip(engine.reason())
    if not ocr.available():
        pytest.skip("Tesseract nicht installiert")
    font = Path(r"C:\Windows\Fonts\LHANDW.TTF")
    if not font.is_file():
        pytest.skip("Handschrift-Schrift LHANDW.TTF fehlt")
    path = tmp_path / "notiz.jpg"
    image = Image.fromarray(_page(["Basalganglienschleife: direkter Weg", "Morbus Parkinson: Verlust dopaminerger Neurone"],
                                  size=(1400, 360), font=font, font_size=46, top=60, step=150))
    image.save(path, quality=60)
    started = time.monotonic()
    try:
        result = ocr_engines.read_page(path, hint="goodnotes", budget_seconds=240)
    finally:
        engine.close()
    assert "trocr" in result["engine"], result
    assert result["handwriting"] is True
    folded = result["text"].lower() + " " + " ".join(v["text"].lower() for v in result["variants"])
    assert "parkinson" in folded or "direkter" in folded, result["text"]
    assert time.monotonic() - started < 300
