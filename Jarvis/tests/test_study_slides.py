"""PowerPoint slides: exact shape geometry from the file, notes without chrome, real slide images from PowerPoint."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import zipfile
from pathlib import Path

import pytest

from study import slides
from study.parsers import parse

NS = ('xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
CX, CY = 12192000, 6858000


def _xfrm(x: int, y: int, w: int, h: int) -> str:
    return f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'


def _sp(text_lines: list[str], *, ph: str = "", xfrm: str = "") -> str:
    nv = f"<p:nvPr>{ph}</p:nvPr>"
    body = "".join(f"<a:p><a:r><a:t>{line}</a:t></a:r></a:p>" for line in text_lines)
    return f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="s"/><p:cNvSpPr/>{nv}</p:nvSpPr><p:spPr>{xfrm}</p:spPr><p:txBody><a:bodyPr/>{body}</p:txBody></p:sp>'


def _tree(shapes: str) -> str:
    return f"<p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/>{shapes}</p:spTree></p:cSld>"


def build_pptx(path: Path) -> Path:
    """Slide 1: explicit xfrm + a group; slide 2: title and body inherit from the layout and the master; notes with a slide number."""

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ppt/presentation.xml", f'<?xml version="1.0"?><p:presentation {NS}><p:sldIdLst><p:sldId id="256" r:id="rId1"/>'
                                           f'<p:sldId id="257" r:id="rId2"/></p:sldIdLst><p:sldSz cx="{CX}" cy="{CY}"/></p:presentation>')
        z.writestr("ppt/_rels/presentation.xml.rels", f'<?xml version="1.0"?><Relationships xmlns="{RELS}">'
                   f'<Relationship Id="rId1" Type="{REL_TYPE}slide" Target="slides/slide1.xml"/>'
                   f'<Relationship Id="rId2" Type="{REL_TYPE}slide" Target="slides/slide2.xml"/></Relationships>')
        group = ('<p:grpSp><p:nvGrpSpPr/><p:grpSpPr><a:xfrm><a:off x="6096000" y="3429000"/><a:ext cx="6096000" cy="3429000"/>'
                 '<a:chOff x="0" y="0"/><a:chExt cx="12192000" cy="6858000"/></a:xfrm></p:grpSpPr>'
                 + _sp(["Gruppentext"], xfrm=_xfrm(0, 0, 6096000, 3429000)) + "</p:grpSp>")
        z.writestr("ppt/slides/slide1.xml", f'<?xml version="1.0"?><p:sld {NS}>'
                   + _tree(_sp(["Motorik"], ph='<p:ph type="title"/>', xfrm=_xfrm(0, 0, CX, CY // 4))
                           + _sp(["Freier Text"], xfrm=_xfrm(CX // 2, CY // 2, CX // 4, CY // 4)) + group) + "</p:sld>")
        z.writestr("ppt/slides/slide2.xml", f'<?xml version="1.0"?><p:sld {NS}>'
                   + _tree(_sp(["Basalganglien-Schleife"], ph='<p:ph type="title"/>')
                           + _sp(["direkter Weg", "indirekter Weg"], ph='<p:ph type="body" idx="1"/>')) + "</p:sld>")
        z.writestr("ppt/slides/_rels/slide2.xml.rels", f'<?xml version="1.0"?><Relationships xmlns="{RELS}">'
                   f'<Relationship Id="rId1" Type="{REL_TYPE}slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                   f'<Relationship Id="rId2" Type="{REL_TYPE}notesSlide" Target="../notesSlides/notesSlide2.xml"/></Relationships>')
        # the layout places the title; the body has no xfrm there and comes from the master
        z.writestr("ppt/slideLayouts/slideLayout1.xml", f'<?xml version="1.0"?><p:sldLayout {NS}>'
                   + _tree(_sp([], ph='<p:ph type="title"/>', xfrm=_xfrm(609600, 304800, 10972800, 1143000))
                           + _sp([], ph='<p:ph type="body" idx="1"/>')) + "</p:sldLayout>")
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", f'<?xml version="1.0"?><Relationships xmlns="{RELS}">'
                   f'<Relationship Id="rId1" Type="{REL_TYPE}slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')
        z.writestr("ppt/slideMasters/slideMaster1.xml", f'<?xml version="1.0"?><p:sldMaster {NS}>'
                   + _tree(_sp([], ph='<p:ph type="title"/>', xfrm=_xfrm(0, 0, 100, 100))
                           + _sp([], ph='<p:ph type="body" idx="1"/>', xfrm=_xfrm(838200, 1714500, 10515600, 4572000))) + "</p:sldMaster>")
        z.writestr("ppt/notesSlides/notesSlide2.xml", f'<?xml version="1.0"?><p:notes {NS}>'
                   + _tree('<p:sp><p:nvSpPr><p:cNvPr id="2" name="img"/><p:cNvSpPr/><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>'
                           + _sp(["Prüfungsrelevant: Wirkung von Dopamin."], ph='<p:ph type="body" idx="1"/>')
                           + _sp(["2"], ph='<p:ph type="sldNum" sz="quarter" idx="5"/>')) + "</p:notes>")
    return path


def test_explicit_xfrm_and_group_transform(tmp_path):
    geometry = slides.slide_geometry(build_pptx(tmp_path / "Neuro.pptx"))
    assert geometry[1]["size"] == [CX, CY]
    shapes = {s["text"]: s for s in geometry[1]["shapes"]}
    assert shapes["Motorik"]["kind"] == "title" and shapes["Motorik"]["box"] == [0.0, 0.0, 1.0, 0.25]
    assert shapes["Freier Text"]["kind"] == "text" and shapes["Freier Text"]["box"] == [0.5, 0.5, 0.25, 0.25]
    # a child filling the top-left quarter of the group's child space lands in the top-left quarter of the group's box
    assert shapes["Gruppentext"]["box"] == [0.5, 0.5, 0.25, 0.25]


def test_placeholder_geometry_is_inherited_from_layout_then_master(tmp_path):
    geometry = slides.slide_geometry(build_pptx(tmp_path / "Neuro.pptx"))
    title, body = geometry[2]["shapes"]
    assert title["kind"] == "title" and title["box"] == [0.05, 0.04444, 0.9, 0.16667], "the layout's title xfrm, not the master's"
    assert body["kind"] == "body" and body["paragraphs"] == ["direkter Weg", "indirekter Weg"]
    assert body["box"] == [0.06875, 0.25, 0.8625, 0.66667], "the layout has no xfrm for the body: the master's"


def test_notes_come_from_the_body_placeholder_without_the_slide_number(tmp_path):
    doc = parse(build_pptx(tmp_path / "Neuro.pptx"))
    notes = [b for b in doc.units[1].blocks if b.kind == "notes"]
    assert [b.text for b in notes] == ["Prüfungsrelevant: Wirkung von Dopamin."]
    assert not doc.units[1].text.rstrip().endswith("2")


def test_blocks_carry_boxes_and_slides_their_size(tmp_path):
    doc = parse(build_pptx(tmp_path / "Neuro.pptx"))
    assert doc.metadata["slide_size"] == [CX, CY]
    slide = doc.units[1]
    assert (slide.width, slide.height) == (960.0, 540.0)
    boxes = {b.text: b.box for b in slide.blocks}
    assert boxes["Basalganglien-Schleife"] == [0.05, 0.04444, 0.9, 0.16667]
    assert boxes["direkter Weg"] == boxes["indirekter Weg"] == [0.06875, 0.25, 0.8625, 0.66667]
    assert boxes["Prüfungsrelevant: Wirkung von Dopamin."] is None


def test_odd_pptx_still_parses_without_boxes(tmp_path):
    path = tmp_path / "Kaputt.pptx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ppt/presentation.xml", f'<?xml version="1.0"?><p:presentation {NS}><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>'
                                           '<p:sldSz cx="abc"/></p:presentation>')
        z.writestr("ppt/_rels/presentation.xml.rels", f'<?xml version="1.0"?><Relationships xmlns="{RELS}">'
                   f'<Relationship Id="rId1" Type="{REL_TYPE}slide" Target="slides/slide1.xml"/></Relationships>')
        z.writestr("ppt/slides/slide1.xml", f'<?xml version="1.0"?><p:sld {NS}><p:cSld><p:spTree>'
                   + _sp(["Titel"], ph='<p:ph type="title"/>', xfrm='<a:xfrm><a:off x="1"/></a:xfrm>') + "</p:spTree></p:cSld></p:sld>")
        z.writestr("ppt/slides/_rels/slide1.xml.rels", "<not xml")
    doc = parse(path)
    assert doc.units[0].title == "Titel" and doc.units[0].blocks[0].box is None
    assert "slide_size" not in doc.metadata
    assert slides.slide_geometry(path)[1]["shapes"][0]["box"] is None


def test_render_defers_when_powerpoint_is_already_open(tmp_path, monkeypatch):
    source = build_pptx(tmp_path / "Neuro.pptx")
    monkeypatch.setattr(slides, "available", lambda: True)
    monkeypatch.setattr(slides, "powerpoint_pids", lambda: {4242})

    def forbidden(*args, **kwargs):
        raise AssertionError("nothing may be launched or killed while the owner's PowerPoint is open")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(slides, "_kill_pid", forbidden)
    result = slides.render_slides(source, tmp_path / "out")
    assert result["ok"] is False and result["deferred"] is True
    assert "PowerPoint ist gerade geöffnet" in result["error"]
    assert not (tmp_path / "out").exists()


def test_render_timeout_kills_only_the_powerpoint_it_started(tmp_path, monkeypatch):
    source = build_pptx(tmp_path / "Neuro.pptx")
    monkeypatch.setattr(slides, "available", lambda: True)
    snapshots = iter([set(), {111}])
    killed: list[int] = []

    class Hanging:
        def __init__(self, *args, **kwargs):
            self.killed = False

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("powershell", timeout)

        def kill(self):
            killed.append(-1)

    monkeypatch.setattr(slides, "powerpoint_pids", lambda: next(snapshots, {111}))
    monkeypatch.setattr(subprocess, "Popen", Hanging)
    monkeypatch.setattr(slides, "_kill_pid", killed.append)
    # the hung render shows no window (the machine's real window list is not this test's business)
    monkeypatch.setattr(slides, "visible_window_pids", lambda: set())
    result = slides.render_slides(source, tmp_path / "out", timeout=1.0)
    assert result["ok"] is False and not result["deferred"] and "nicht innerhalb" in result["error"]
    assert killed == [-1, 111], "the hung PowerShell and the PowerPoint that was not running before; nothing else"


REAL_PPTX = Path(os.environ.get("ZEUS_OFFICE_RENDER_PPTX", r"C:\Users\rapha\AppData\Local\Temp\claude\D--Jarvis-recovery-20260823-repo"
                                r"\c4fca376-5496-4cec-8ed7-4e6553b570ac\scratchpad\material_v2\Neuroanatomie_Folien.pptx"))


@pytest.mark.skipif(not (os.environ.get("ZEUS_OFFICE_RENDER_TEST") == "1" and slides.available()),
                    reason="launches PowerPoint: only with ZEUS_OFFICE_RENDER_TEST=1 on a machine with PowerPoint")
def test_real_powerpoint_render(tmp_path):
    source = tmp_path / "Folien.pptx"
    if REAL_PPTX.is_file():
        shutil.copyfile(REAL_PPTX, source)
    else:
        build_pptx(source)
    expected = len(slides.slide_geometry(source))
    result = slides.render_slides(source, tmp_path / "out", width=1600)
    assert result["ok"], result
    assert len(result["slides"]) == expected
    cx, cy = slides.slide_geometry(source)[1]["size"]
    for file in result["slides"]:
        header = Path(file).read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", header[16:24])
        assert width == 1600 and abs(height - 1600 * cy / cx) <= 2
    assert not slides.powerpoint_pids(), "the render leaves no PowerPoint behind"


def test_a_powerpoint_with_a_visible_window_is_never_ended(monkeypatch):
    """Windows joins a presentation the owner opens during a render into the render's process: that process shows a window."""

    killed: list[int] = []
    monkeypatch.setattr(slides, "_kill_pid", killed.append)
    slides._kill_ours(set(), lambda: {111, 222}, visible=lambda: {222})
    assert killed == [111]
    killed.clear()
    slides._kill_ours(set(), lambda: {111}, visible=lambda: {-1})
    assert killed == [], "when windows cannot be read, nothing is ended"
