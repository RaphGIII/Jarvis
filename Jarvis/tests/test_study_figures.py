"""Figures, diagrams, tables and captions are located objects: page + region + caption."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from study import figures
from study.model import Unit
from study.parsers import parse_pdf, parse_pptx
from study_fixtures import make_pdf

A4 = (595.0, 842.0)


def _pdf(pages: list[dict], *, size: tuple[float, float] = A4) -> bytes:
    """A tiny PDF builder: each page {"content": bytes, "xobjects": {name: spec}, "rotate": int, "form_matrix": bytes}.

    ``spec`` is "IMG" (the shared 2x2 image) or ``(form content, inner image name)`` for a Form XObject drawing it.
    """

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    image = add(b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4 >>\nstream\n"
                b"\x00\xff\xff\x00\nendstream")
    pages_id = add(b"")
    page_ids = []
    for page in pages:
        entries = []
        for name, spec in (page.get("xobjects") or {}).items():
            if spec == "IMG":
                ref = image
            else:
                content, inner = spec  # a Form XObject drawing the image under ``inner``
                matrix = page.get("form_matrix", b"[1 0 0 1 0 0]")
                ref = add(b"<< /Type /XObject /Subtype /Form /BBox [0 0 595 842] /Matrix " + matrix
                          + b" /Resources << /XObject << /" + inner.encode() + b" " + str(image).encode() + b" 0 R >> >> /Length "
                          + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
            entries.append(b"/" + name.encode() + b" " + str(ref).encode() + b" 0 R")
        content = page["content"]
        stream = add(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        resources = b"<< /Font << /F1 " + str(font).encode() + b" 0 R >> /XObject << " + b" ".join(entries) + b" >> >>"
        rotate = f" /Rotate {page['rotate']}".encode() if page.get("rotate") else b""
        page_ids.append(add(b"<< /Type /Page /Parent " + str(pages_id).encode() + b" 0 R /MediaBox [0 0 "
                            + f"{size[0]:g} {size[1]:g}".encode() + b"]" + rotate + b" /Resources " + resources
                            + b" /Contents " + str(stream).encode() + b" 0 R >>"))
    objects[pages_id - 1] = (b"<< /Type /Pages /Kids [" + b" ".join(str(i).encode() + b" 0 R" for i in page_ids) + b"] /Count "
                             + str(len(page_ids)).encode() + b" >>")
    catalog = add(b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root " + str(catalog).encode() + b" 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n"
    return bytes(out)


def _text(x: float, y: float, line: str, size: int = 11) -> bytes:
    return f"BT /F1 {size} Tf {x:g} {y:g} Td ({line}) Tj ET\n".encode("cp1252")


def _write(tmp_path: Path, data: bytes, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _close(box, expected, tol=0.01):
    return all(abs(a - b) <= tol for a, b in zip(box, expected))


# ---------------------------------------------------------------------------- placement

def test_the_fixture_image_is_found_on_its_page_at_the_drawn_position(tmp_path):
    path = _write(tmp_path, make_pdf([["Einleitung"], ["Stoffwechsel", "Text"], ["Ende"]], image_on={2}))
    found = figures.pdf_figures(path)
    assert list(found) == [2]
    (figure,) = found[2]
    # 100x100 pt at (72, 200) on 595x842, y from the top: 1 - 300/842
    assert _close(figure["box"], [72 / 595, 1 - 300 / 842, 100 / 595, 100 / 842], tol=0.002)
    assert figure["kind"] == "figure" and figure["index"] == 1


def test_nested_cm_transforms_compose(tmp_path):
    content = b"q 2 0 0 2 0 0 cm q 1 0 0 1 50 100 cm q 150 0 0 100 0 0 cm /Im1 Do Q Q Q\n"
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {"Im1": "IMG"}}]))
    (figure,) = figures.pdf_figures(path)[1]
    # scale 2 then translate (50,100) in scaled space -> origin (100, 200), size 300x200
    assert _close(figure["box"], [100 / 595, 1 - 400 / 842, 300 / 595, 200 / 842], tol=0.002)


def test_q_restores_the_transform_after_a_drawing(tmp_path):
    content = b"q 10 0 0 10 500 800 cm Q q 120 0 0 120 300 300 cm /Im1 Do Q\n"
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {"Im1": "IMG"}}]))
    (figure,) = figures.pdf_figures(path)[1]
    assert _close(figure["box"], [300 / 595, 1 - 420 / 842, 120 / 595, 120 / 842], tol=0.002)


def test_an_image_inside_a_form_xobject_is_located_through_the_form_matrix(tmp_path):
    page = {"content": b"q 1 0 0 1 100 50 cm /Fm1 Do Q\n",
            "xobjects": {"Fm1": (b"q 200 0 0 150 0 0 cm /Pic Do Q", "Pic")},
            "form_matrix": b"[1 0 0 1 20 30]"}
    path = _write(tmp_path, _pdf([page]))
    (figure,) = figures.pdf_figures(path)[1]
    # form matrix (20,30) inside page translation (100,50): image at (120, 80), 200x150
    assert _close(figure["box"], [120 / 595, 1 - 230 / 842, 200 / 595, 150 / 842], tol=0.002)


def test_tiny_icons_are_ignored_and_a_full_page_image_is_a_scan(tmp_path):
    icons = b"q 16 0 0 16 20 20 cm /Im1 Do Q q 12 0 0 12 560 810 cm /Im1 Do Q\n"
    scan = b"q 595 0 0 842 0 0 cm /Im1 Do Q\n"
    path = _write(tmp_path, _pdf([{"content": icons, "xobjects": {"Im1": "IMG"}}, {"content": scan, "xobjects": {"Im1": "IMG"}}]))
    found = figures.pdf_figures(path)
    assert 1 not in found
    assert [f["kind"] for f in found[2]] == ["scan"]
    assert figures.image_count(found[2]) == 0


def test_a_rotated_page_reports_the_region_as_the_viewer_shows_it(tmp_path):
    content = b"q 100 0 0 100 0 742 cm /Im1 Do Q\n"  # top-left corner of the unrotated page
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {"Im1": "IMG"}, "rotate": 90}]))
    (figure,) = figures.pdf_figures(path)[1]
    # rotated 90 degrees clockwise the top-left corner is on the top-right
    assert figure["box"][0] + figure["box"][2] == pytest.approx(1.0, abs=0.002)
    assert figure["box"][1] == pytest.approx(0.0, abs=0.002)


def test_a_broken_content_stream_never_breaks_parsing(tmp_path):
    content = b"q 100 0 0 100 72 200 cm /Im1 Do Q\nq 1 0 0 cm /Nope Do (unterminated"
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {"Im1": "IMG"}}]))
    found = figures.pdf_figures(path)  # must not raise
    assert isinstance(found, dict)
    assert figures.pdf_figures(tmp_path / "missing.pdf") == {}
    garbage = _write(tmp_path, b"%PDF-1.4 nonsense", "garbage.pdf")
    assert figures.pdf_figures(garbage) == {}


# ------------------------------------------------------------------------------ captions

def test_a_caption_below_an_image_is_attached_to_it(tmp_path):
    content = (_text(72, 780, "Kapitel 5: Stoffwechsel")
               + b"q 300 0 0 200 72 400 cm /Im1 Do Q\n"
               + _text(72, 380, "Abbildung 5.2: Der Citratzyklus im Mitochondrium")
               + b"q 200 0 0 150 72 120 cm /Im1 Do Q\n"
               + _text(72, 280, "Tabelle 3: Enzyme und Cofaktoren"))
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {"Im1": "IMG"}}]))
    first, second = figures.pdf_figures(path)[1]
    assert first["caption"].startswith("Abbildung 5.2") and first["kind"] == "figure"
    assert _close(first["box"], [72 / 595, 1 - 600 / 842, 300 / 595, 200 / 842], tol=0.002)
    assert "Citratzyklus" in first["text"]
    # the table caption sits above the second image
    assert second["caption"].startswith("Tabelle 3") and second["kind"] == "table"
    assert [first["index"], second["index"]] == [1, 2]


def test_a_caption_without_an_image_is_a_figure_located_at_its_line(tmp_path):
    content = _text(72, 780, "Glykolyse") + _text(72, 500, "Schema: Ablauf der Glykolyse") + _text(72, 300, "siehe Abbildung 3 im Skript")
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {}}]))
    (figure,) = figures.pdf_figures(path)[1]
    assert figure["kind"] == "diagram" and figure["source"] == "caption"
    assert figure["caption"] == "Schema: Ablauf der Glykolyse"
    assert figure["box"] is not None and abs(figure["box"][1] - (1 - 511 / 842)) < 0.02


def test_one_caption_and_one_picture_belong_together_however_far_apart(tmp_path):
    path = _write(tmp_path, make_pdf([["Stoffwechsel", "Abb. 4: Der Citratzyklus im Mitochondrium"]], image_on={1}))
    (figure,) = figures.pdf_figures(path)[1]
    assert figure["caption"] == "Abb. 4: Der Citratzyklus im Mitochondrium"
    assert figure["source"] == "image"


def test_caption_lines_need_a_number_or_colon_at_the_line_start():
    text = "Abb. 3: Niere\nsiehe Abbildung 4\nFigure 2.1 Heart\nSchema der Atmung ohne Nummer\nTab. 7 Werte\nGrafik:"
    assert figures.caption_lines(text) == ["Abb. 3: Niere", "Figure 2.1 Heart", "Tab. 7 Werte", "Grafik:"]
    assert figures.caption_kind("Tab. 7 Werte") == "table"
    assert figures.caption_kind("Diagramm 2: Verlauf") == "diagram"


def test_an_optional_locate_places_captions_the_text_layer_could_not(tmp_path):
    regions = [{"kind": "figure", "box": [0.1, 0.5, 0.4, 0.2]}]
    captions = [{"text": "Abb. 1: Nephron", "box": None}]
    located = figures.attach_captions(regions, captions)
    assert located[0]["caption"] == "Abb. 1: Nephron"

    calls = []

    def locate(page, phrases):
        calls.append((page, phrases))
        return {"page_size": [595, 842], "matches": [{"phrase": phrases[0], "rects": [[72, 421, 200, 12]]}]}

    class FakePage:
        def get(self, _key, default=None):
            return default

        def get_contents(self):
            return None

        def extract_text(self, **_):
            return ""

    result = figures.page_figures(FakePage(), 3, text="Einleitung\nAbb. 1: Nephron", locate=locate)
    assert calls == [(3, ["Abb. 1: Nephron"])]
    assert result[0]["box"] == [round(72 / 595, 5), 0.5, round(200 / 595, 5), round(12 / 842, 5)]


# ------------------------------------------------------------------------------- parsers

def test_parse_pdf_fills_unit_figures_and_counts_only_real_pictures(tmp_path):
    content = (_text(72, 780, "Stoffwechsel der Zelle") + b"q 16 0 0 16 20 20 cm /Im1 Do Q\n"
               + b"q 300 0 0 200 72 400 cm /Im1 Do Q\n" + _text(72, 380, "Abb. 2: Citratzyklus"))
    path = _write(tmp_path, _pdf([{"content": content, "xobjects": {"Im1": "IMG"}},
                                  {"content": _text(72, 780, "Nur Text auf dieser Seite"), "xobjects": {"Im1": "IMG"}}]))
    doc = parse_pdf(path)
    first, second = doc.units
    assert first.images == 1 and len(first.figures) == 1
    assert first.figures[0]["caption"] == "Abb. 2: Citratzyklus" and first.figures[0]["index"] == 1
    # an image declared in the resources but never painted is not a figure
    assert second.images == 0 and second.figures == []


def test_parse_pdf_keeps_the_fixture_image_count(tmp_path):
    path = _write(tmp_path, make_pdf([["Seite eins"], ["Seite zwei mit Bild"]], image_on={2}))
    doc = parse_pdf(path)
    assert [u.images for u in doc.units] == [0, 1]
    assert doc.units[1].figures[0]["box"] is not None


def _pptx_with_pictures(path: Path) -> Path:
    ns = ('xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')

    def xfrm(tag, x, y, w, h, extra=""):
        return f'<{tag}><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/>{extra}</a:xfrm></{tag}>'

    title = ('<p:sp><p:nvSpPr><p:cNvPr id="1" name="t"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>'
             + xfrm("p:spPr", 0, 0, 9144000, 1000000) + '<p:txBody><a:p><a:r><a:t>Citratzyklus</a:t></a:r></a:p></p:txBody></p:sp>')
    pic = ('<p:pic><p:nvPicPr><p:cNvPr id="2" name="Bild"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
           + xfrm("p:spPr", 914400, 1371600, 3657600, 2743200) + '</p:pic>')
    caption = ('<p:sp><p:nvSpPr><p:cNvPr id="3" name="c"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>' + xfrm("p:spPr", 914400, 4200000, 3657600, 400000)
               + '<p:txBody><a:p><a:r><a:t>Abb. 1: Der Zyklus im Überblick</a:t></a:r></a:p></p:txBody></p:sp>')
    logo = ('<p:pic><p:nvPicPr><p:cNvPr id="4" name="Logo"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
            + xfrm("p:spPr", 8800000, 6600000, 200000, 150000) + '</p:pic>')
    group_pic = ('<p:pic><p:nvPicPr><p:cNvPr id="6" name="G"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>' + xfrm("p:spPr", 0, 0, 1000, 1000) + '</p:pic>')
    group = ('<p:grpSp><p:nvGrpSpPr><p:cNvPr id="5" name="grp"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
             '<p:grpSpPr><a:xfrm><a:off x="5000000" y="1371600"/><a:ext cx="3000000" cy="2000000"/>'
             '<a:chOff x="0" y="0"/><a:chExt cx="1000" cy="1000"/></a:xfrm></p:grpSpPr>' + group_pic + '</p:grpSp>')
    slide = f'<?xml version="1.0"?><p:sld {ns}><p:cSld><p:spTree>{title}{pic}{caption}{logo}{group}</p:spTree></p:cSld></p:sld>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", f'<?xml version="1.0"?><p:presentation {ns}><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>'
                                                 '<p:sldSz cx="9144000" cy="6858000"/></p:presentation>')
        archive.writestr("ppt/_rels/presentation.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                                            '<Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/></Relationships>')
        archive.writestr("ppt/slides/slide1.xml", slide)
    return path


def test_slide_pictures_become_figures_with_the_nearest_text_as_caption(tmp_path):
    from study.slides import slide_geometry

    path = _pptx_with_pictures(tmp_path / "zyklus.pptx")
    found = figures.slide_figures(slide_geometry(path)[1])
    assert len(found) == 2  # the picture and the group picture; the logo is an icon
    main = next(f for f in found if f["caption"])
    assert main["caption"] == "Abb. 1: Der Zyklus im Überblick"
    assert _close(main["box"], [0.1, 0.2, 0.4, 0.4], tol=0.001)
    grouped = next(f for f in found if f is not main)
    assert _close(grouped["box"], [5000000 / 9144000, 0.2, 3000000 / 9144000, 2000000 / 6858000], tol=0.001)
    assert sorted(f["index"] for f in found) == [1, 2]

    doc = parse_pptx(path)
    assert doc.units[0].images == 2 and len(doc.units[0].figures) == 2


def test_figure_chunks_are_shaped_like_index_chunks(tmp_path):
    from study.index import chunk_unit

    path = _write(tmp_path, make_pdf([["Stoffwechsel", "Abb. 4: Der Citratzyklus im Mitochondrium"]], image_on={1}))
    unit = parse_pdf(path).units[0]
    chunks = figures.figure_chunks(unit)
    assert len(chunks) == 1
    chunk = chunks[0]
    base_keys = set(chunk_unit(unit)[0])
    assert base_keys <= set(chunk)
    assert chunk["figure_index"] == 1 and chunk["page"] == 1 and chunk["slide"] is None
    assert chunk["box"] == unit.figures[0]["box"]
    assert unit.text[chunk["char_start"]:chunk["char_end"]] == "Abb. 4: Der Citratzyklus im Mitochondrium"
    assert figures.figure_chunks(Unit(number=1, kind="section", text="x")) == []


def test_a_scanned_page_is_not_indexed_as_a_figure_of_itself():
    """Seen live: a handwritten GoodNotes page (one full-page image) opened with the whole page framed as "Abbildung 1"."""

    from study.index import _figure_chunks
    from study.model import Unit

    scan = Unit(number=1, kind="page", text="Basalganglienschleife: direkter Weg", ocr=True)
    scan.figures = [{"kind": "scan", "box": [0, 0, 1, 1], "caption": "", "text": scan.text, "index": 1}]
    assert _figure_chunks(scan) == []
    drawing = Unit(number=2, kind="page", text="Abb. 3: Aktionspotential")
    drawing.figures = [{"kind": "figure", "box": [0.1, 0.2, 0.5, 0.3], "caption": "Abb. 3: Aktionspotential", "text": "", "index": 1}]
    assert len(_figure_chunks(drawing)) == 1
