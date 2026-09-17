"""Parsers: one per format, all producing a :class:`NormalizedDocument`.

Adding a format is one function here and one entry in ``PARSERS``.  Office formats are
read as the zip + XML they are (no third-party library): DOCX gives headings and
paragraphs, PPTX gives slides in presentation order with titles and speaker notes.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

from study import ocr
from study.model import Block, NormalizedDocument, Unit

EXTENSIONS = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".md": "markdown", ".markdown": "markdown", ".txt": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".heic": "image", ".tif": "image", ".tiff": "image",
}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def source_type_for(path: str | Path) -> str:
    return EXTENSIONS.get(Path(path).suffix.lower(), "")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "").replace("\x00", "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_name(path: Path) -> str:
    return re.sub(r"[_]+", " ", path.stem).strip() or path.name


# ------------------------------------------------------------------------------------ PDF

def parse_pdf(path: Path, *, engine: Any = None) -> NormalizedDocument:
    """Pages with their text (Qt's engine when available, else pypdf), page sizes, image counts and metadata."""

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    meta_raw = reader.metadata or {}
    metadata = {"producer": str(meta_raw.get("/Producer") or ""), "creator": str(meta_raw.get("/Creator") or ""),
                "title": str(meta_raw.get("/Title") or ""), "author": str(meta_raw.get("/Author") or "")}
    texts: list[str] | None = None
    if engine is not None:
        texts = engine.texts(path)
    # the file name is the owner's name for the document; PDF title metadata is often a tool's leftover ("Microsoft Word - x.docx")
    doc = NormalizedDocument(source_type="pdf", title=_title_from_name(path), metadata=metadata)
    if texts is None:
        doc.metadata["text_engine"] = "pypdf"
    else:
        doc.metadata["text_engine"] = "qtpdf"
    from study import figures

    locate = None
    if engine is not None and callable(getattr(engine, "locate", None)):
        def locate(page_number: int, phrases: list[str]) -> dict[str, Any]:
            return engine.locate(path, page_number, phrases)
    for index, page in enumerate(reader.pages):
        number = index + 1
        raw = texts[index] if texts is not None and index < len(texts) else ""
        if texts is None:
            try:
                raw = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - one broken page never loses the document
                doc.warnings.append(f"page {number}: {type(exc).__name__}")
                raw = ""
        text = _clean(raw)
        try:
            box = page.mediabox
            width, height = float(box.width), float(box.height)
        except Exception:  # noqa: BLE001
            width = height = 0.0
        images = 0
        try:
            resources = page.get("/Resources") or {}
            xobjects = resources.get("/XObject") if hasattr(resources, "get") else None
            if xobjects is not None:
                xobjects = xobjects.get_object()
                images = sum(1 for key in xobjects if str(xobjects[key].get_object().get("/Subtype")) == "/Image")
        except Exception:  # noqa: BLE001
            images = 0
        unit = Unit(number=number, kind="page", text=text, width=width, height=height, images=images, has_text=len(text) >= 12)
        # located figures: where images are painted, their captions; images then counts the pictures really shown
        # (declared-but-unused XObjects and icons no longer make a page a "figure page")
        try:
            unit.figures = figures.page_figures(page, number, reader=reader, text=text, locate=locate)
            unit.images = figures.image_count(unit.figures)
        except Exception as exc:  # noqa: BLE001 - figures are a bonus: the resource count stays
            doc.warnings.append(f"page {number}: figures {type(exc).__name__}")
        unit.title = text.split("\n", 1)[0][:120] if text else ""
        if text:
            unit.blocks.append(Block(text=text, kind="page_text", char_start=0, char_end=len(text)))
        doc.units.append(unit)
    return doc


def ocr_pdf_pages(doc: NormalizedDocument, path: Path, *, engine: Any, cache_dir: Path, cache_root: Path | None = None,
                  ocr_progress: Callable[[int, int], None] | None = None, should_stop: Callable[[], bool] | None = None) -> int:
    """Pages without a text layer (scans, handwriting): rendered and read by OCR (printed + handwriting engines).

    Each page image is read once per engine version: results are cached by the hash of the rendered PNG under
    ``cache_root / "by_hash"`` (default: the shared parent of ``cache_dir``), so a restarted or partly changed import
    only reads new pages.  ``ocr_progress(done, total)`` follows every page (cache hits included); ``should_stop()``
    ends the loop between pages and cancels a running page."""

    from study import ocr_engines

    if engine is None or not ocr_engines.any_available():
        return 0
    import json

    pending = [unit for unit in doc.units if not unit.has_text]
    if not pending:
        return 0
    done = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_root if cache_root is not None else cache_dir.parent
    # the document's own text says which language its scanned pages are in; unknown -> German first
    sample = "\n".join(unit.text for unit in doc.units if unit.has_text and unit.text)[:6000]
    languages = ocr.choose_languages(sample) if ocr.available() else "deu"
    hint = "auto"
    try:
        from study import goodnotes

        if goodnotes.is_goodnotes_pdf(doc.metadata):
            hint = "goodnotes"  # handwritten notes app: lean towards the handwriting reader
    except Exception:  # noqa: BLE001 - the hint is a bonus
        hint = "auto"
    used: set[str] = set()
    engines: set[str] = set()
    confidences: list[tuple[float, int]] = []
    handwriting = False
    for position, unit in enumerate(pending, start=1):
        if should_stop is not None and should_stop():
            break
        out = cache_dir / f"ocr_{unit.number}.png"
        rendered = engine.render(path, unit.number, out, width=1800)
        if rendered.get("ok"):
            result, _hit = ocr_engines.read_page_cached(out, cache_root=root, hint=hint, languages=languages, should_stop=should_stop)
            if result.get("cancelled"):
                break
            text = _clean(result.get("text") or "")
            if len(text) >= 12:
                # offsets in the word boxes must point into exactly the text the index stores
                aligned = ocr.align_words(result, text)
                stored = {**aligned, "languages": result.get("languages") or languages}
                if stored.get("languages"):
                    used.add(str(stored["languages"]))
                try:
                    (cache_dir / f"ocr_{unit.number}.json").write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
                except OSError as exc:
                    doc.warnings.append(f"page {unit.number}: OCR boxes not cached ({type(exc).__name__})")
                blocks = ocr_engines.unit_blocks(result, text)
                extra = _clean(ocr_engines.variant_text(result))
                if extra:
                    # alternative readings stay searchable after the page text, never inside it
                    start = len(text) + 2
                    blocks.append(Block(text=extra, kind="ocr_variant", char_start=start, char_end=start + len(extra), role="variant"))
                    text = text + "\n\n" + extra
                unit.text, unit.has_text, unit.ocr, unit.blocks = text, True, True, blocks
                handwriting = handwriting or bool(result.get("handwriting"))
                if result.get("engine"):
                    engines.update(str(result["engine"]).split("+"))
                if result.get("mean_confidence") is not None:
                    confidences.append((float(result["mean_confidence"]), len(text)))
                done += 1
            for warning in result.get("warnings") or []:
                note = f"page {unit.number}: {warning}"
                if note not in doc.warnings and len(doc.warnings) < 50:
                    doc.warnings.append(note)
        if ocr_progress is not None:
            try:
                ocr_progress(position, len(pending))
            except Exception:  # noqa: BLE001 - progress reporting never stops the reading
                pass
    doc.metadata["ocr_languages"] = ", ".join(sorted(used)) or languages
    doc.metadata["has_handwriting"] = handwriting
    weight = sum(w for _, w in confidences)
    doc.metadata["ocr_mean_confidence"] = round(sum(c * w for c, w in confidences) / weight, 4) if weight else None
    doc.metadata["ocr_engine"] = "+".join(name for name in ("tesseract", "trocr") if name in engines)
    return done


# ----------------------------------------------------------------------------------- DOCX

def _docx_heading_level(paragraph: ET.Element, styles: dict[str, int]) -> int:
    props = paragraph.find(f"{_W}pPr")
    if props is None:
        return 0
    outline = props.find(f"{_W}outlineLvl")
    if outline is not None:
        try:
            return int(outline.get(f"{_W}val", "9")) + 1
        except ValueError:
            pass
    style = props.find(f"{_W}pStyle")
    if style is None:
        return 0
    style_id = style.get(f"{_W}val", "")
    if style_id in styles:
        return styles[style_id]
    match = re.match(r"(?i)^(heading|berschrift|überschrift)\s*(\d)$", style_id.replace("Ü", "Ü"))
    if match:
        return int(match.group(2))
    match = re.match(r"(?i).*?(\d)$", style_id)
    if match and re.match(r"(?i)^(heading|.*berschrift)", style_id):
        return int(match.group(1))
    return 0


def _docx_styles(archive: zipfile.ZipFile) -> dict[str, int]:
    """styleId -> heading level, from styles.xml (localized style names like "Überschrift 1" included)."""

    levels: dict[str, int] = {}
    try:
        root = ET.fromstring(archive.read("word/styles.xml"))
    except (KeyError, ET.ParseError):
        return levels
    for style in root.findall(f"{_W}style"):
        style_id = style.get(f"{_W}styleId", "")
        name_node = style.find(f"{_W}name")
        name = (name_node.get(f"{_W}val", "") if name_node is not None else "").lower()
        match = re.match(r"^(heading|überschrift|berschrift)\s*(\d)$", name)
        if match:
            levels[style_id] = int(match.group(2))
        elif name == "title":
            levels[style_id] = 1
    return levels


def parse_docx(path: Path, **_: Any) -> NormalizedDocument:
    """Sections by heading, paragraphs numbered through the document.  No page numbers: Word has none without layout."""

    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        styles = _docx_styles(archive)
        core_title = ""
        try:
            core = ET.fromstring(archive.read("docProps/core.xml"))
            node = core.find("{http://purl.org/dc/elements/1.1/}title")
            core_title = (node.text or "").strip() if node is not None else ""
        except (KeyError, ET.ParseError):
            pass
    body = root.find(f"{_W}body")
    doc = NormalizedDocument(source_type="docx", title=core_title or _title_from_name(path))
    heading_path: list[str] = []
    levels: list[int] = []
    section: Unit | None = None
    paragraph_no = 0

    def new_section(title: str) -> Unit:
        unit = Unit(number=len(doc.units) + 1, kind="section", title=title)
        doc.units.append(unit)
        return unit

    def add(unit: Unit, text: str, kind: str) -> None:
        nonlocal paragraph_no
        paragraph_no += 1
        start = len(unit.text) + (2 if unit.text else 0)
        unit.text = (unit.text + "\n\n" + text) if unit.text else text
        unit.blocks.append(Block(text=text, kind=kind, heading_path=list(heading_path), paragraph=paragraph_no,
                                 char_start=start, char_end=start + len(text)))

    for node in list(body) if body is not None else []:
        if node.tag == f"{_W}p":
            text = _clean("".join(t.text or "" for t in node.iter(f"{_W}t")))
            if not text:
                continue
            level = _docx_heading_level(node, styles)
            if level:
                while levels and levels[-1] >= level:
                    levels.pop()
                    heading_path.pop()
                levels.append(level)
                heading_path.append(text[:160])
                section = new_section(" › ".join(heading_path))
                add(section, text, "heading")
                continue
            if section is None:
                section = new_section(doc.title)
            add(section, text, "paragraph")
        elif node.tag == f"{_W}tbl":
            rows = []
            for row in node.iter(f"{_W}tr"):
                cells = [_clean("".join(t.text or "" for t in cell.iter(f"{_W}t"))) for cell in row.iter(f"{_W}tc")]
                rows.append(" | ".join(cells))
            text = "\n".join(r for r in rows if r.strip())
            if text:
                if section is None:
                    section = new_section(doc.title)
                add(section, text, "table")
    doc.metadata["paragraphs"] = paragraph_no
    doc.metadata["page_numbers"] = "none: Word documents have no stable page numbers without a layout engine"
    return doc


# ----------------------------------------------------------------------------------- PPTX

def parse_pptx(path: Path, **_: Any) -> NormalizedDocument:
    """Slides in presentation order, each with its title, body text and speaker notes."""

    doc = NormalizedDocument(source_type="pptx", title=_title_from_name(path))
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        order: list[str] = []
        try:
            presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
            rels = ET.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
            targets = {rel.get("Id"): rel.get("Target", "") for rel in rels.findall(f"{_REL}Relationship")}
            for slide_id in presentation.iter(f"{_P}sldId"):
                target = targets.get(slide_id.get(f"{_R}id"), "")
                if target:
                    order.append("ppt/" + target.lstrip("/").removeprefix("ppt/"))
        except (KeyError, ET.ParseError):
            pass
        if not order:
            order = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        from study import slides as slide_geo

        size: tuple[int, int] | None = None
        inheritance: Any = None
        try:
            size = slide_geo.slide_size(archive)
            inheritance = slide_geo._Inheritance(archive)
        except Exception:  # geometry is a bonus: odd XML must never stop the text
            size = None
        if size:
            doc.metadata["slide_size"] = [size[0], size[1]]
        # speaker notes live in the notes body placeholder; slide number, date, header, footer and slide image are chrome
        notes_chrome = {"sldNum", "dt", "hdr", "ftr", "sldImg"}
        for number, slide_name in enumerate(order, start=1):
            try:
                slide = ET.fromstring(archive.read(slide_name))
            except (KeyError, ET.ParseError):
                continue
            boxes: dict[int, list[float] | None] = {}
            slide_shapes: list[dict[str, Any]] = []
            try:
                for element, shape_info in slide_geo.shape_entries(archive, slide_name, slide, size, inheritance):
                    boxes[id(element)] = shape_info.get("box")
                    slide_shapes.append(shape_info)
            except Exception:
                boxes = {}
                slide_shapes = []
            title = ""
            title_box: list[float] | None = None
            paragraphs: list[tuple[str, list[float] | None]] = []
            for shape in slide.iter(f"{_P}sp"):
                is_title = any(ph.get("type") in {"title", "ctrTitle"} for ph in shape.iter(f"{_P}ph"))
                box = boxes.get(id(shape))
                for para in shape.iter(f"{_A}p"):
                    text = _clean("".join(t.text or "" for t in para.iter(f"{_A}t")))
                    if not text:
                        continue
                    if is_title and not title:
                        title, title_box = text, box
                    else:
                        paragraphs.append((text, box))
            images = sum(1 for _ in slide.iter(f"{_P}pic"))
            unit = Unit(number=number, kind="slide", title=title, images=images)
            if slide_shapes:
                try:
                    from study import figures as slide_figs

                    unit.figures = slide_figs.slide_figures({"size": list(size) if size else None, "shapes": slide_shapes})
                    unit.images = slide_figs.image_count(unit.figures)
                except Exception:  # figures are a bonus: the picture count stays
                    unit.figures = []
            if size:
                unit.width, unit.height = size[0] / 12700, size[1] / 12700
            notes_name = ""
            try:
                notes_name = slide_geo._related(archive, slide_name, "notesSlide")
            except Exception:
                notes_name = ""
            if notes_name not in names:
                notes_name = slide_name.replace("slides/slide", "notesSlides/notesSlide")
            notes = ""
            if notes_name in names:
                try:
                    notes_root = ET.fromstring(archive.read(notes_name))
                    note_parts = []
                    for shape in notes_root.iter(f"{_P}sp"):
                        ph = next(iter(shape.iter(f"{_P}ph")), None)
                        if ph is not None and ph.get("type") in notes_chrome:
                            continue
                        note_parts.extend(t.text or "" for t in shape.iter(f"{_A}t"))
                    notes = _clean(" ".join(note_parts))
                    notes = re.sub(r"^\d+\s*$", "", notes)
                except ET.ParseError:
                    notes = ""
            text = ""
            entries = ([("slide_title", title, title_box)] if title else []) + [("paragraph", p, b) for p, b in paragraphs] \
                + ([("notes", notes, None)] if notes else [])
            for kind, part, part_box in entries:
                start = len(text) + (2 if text else 0)
                text = (text + "\n\n" + part) if text else part
                unit.blocks.append(Block(text=part, kind=kind, char_start=start, char_end=start + len(part), box=part_box))
            unit.text = text
            unit.has_text = bool(text.strip())
            doc.units.append(unit)
    return doc


# -------------------------------------------------------------------------- Markdown / text

def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_markdown(path: Path, **_: Any) -> NormalizedDocument:
    text = _read_text(path).replace("\r\n", "\n")
    doc = NormalizedDocument(source_type="markdown", title=_title_from_name(path))
    heading_path: list[str] = []
    levels: list[int] = []
    unit: Unit | None = None
    paragraph_no = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal unit, paragraph_no
        body = _clean("\n".join(buffer))
        buffer.clear()
        if not body:
            return
        if unit is None:
            unit = Unit(number=len(doc.units) + 1, kind="section", title=doc.title)
            doc.units.append(unit)
        paragraph_no += 1
        start = len(unit.text) + (2 if unit.text else 0)
        unit.text = (unit.text + "\n\n" + body) if unit.text else body
        unit.blocks.append(Block(text=body, kind="paragraph", heading_path=list(heading_path), paragraph=paragraph_no,
                                 char_start=start, char_end=start + len(body)))

    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        match = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            level = len(match.group(1))
            if level == 1 and not doc.units and not heading_path:
                doc.title = match.group(2).strip()
            while levels and levels[-1] >= level:
                levels.pop()
                heading_path.pop()
            levels.append(level)
            heading_path.append(match.group(2).strip()[:160])
            unit = Unit(number=len(doc.units) + 1, kind="section", title=" › ".join(heading_path))
            doc.units.append(unit)
            paragraph_no += 1
            heading = match.group(2).strip()
            unit.text = heading
            unit.blocks.append(Block(text=heading, kind="heading", heading_path=list(heading_path), paragraph=paragraph_no,
                                     char_start=0, char_end=len(heading)))
            continue
        if not line.strip() and not in_fence:
            flush()
            continue
        buffer.append(line)
    flush()
    return doc


def parse_text(path: Path, **_: Any) -> NormalizedDocument:
    doc = parse_markdown(path)
    doc.source_type = "text"
    return doc


# ---------------------------------------------------------------------------------- images

def parse_image(path: Path, *, cache_dir: Path | None = None, cache_root: Path | None = None,
                ocr_progress: Callable[[int, int], None] | None = None, should_stop: Callable[[], bool] | None = None,
                **_: Any) -> NormalizedDocument:
    """A photo of notes or a scan (PNG, JPEG, WebP, ...): one unit, read by the printed + handwriting OCR engines.

    Boxes are fractions of the image as a viewer shows it (EXIF orientation applied).  With ``cache_dir`` the word boxes
    go to ``cache_dir / "ocr_1.json"`` for highlighting; readings are reused by image hash under ``cache_root / "by_hash"``
    (default: the parent of ``cache_dir``)."""

    from study import ocr_engines

    doc = NormalizedDocument(source_type="image", title=_title_from_name(path))
    width = height = 0.0
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = float(image.width), float(image.height)
            orientation = image.getexif().get(0x0112, 1)
            if orientation in (5, 6, 7, 8):  # a phone photo stored sideways: the viewer shows it turned
                width, height = height, width
    except Exception:  # noqa: BLE001 - HEIC and friends without a codec
        doc.warnings.append("image size unknown")
    text = ""
    blocks: list[Block] = []
    available = ocr_engines.any_available()
    if available and not (should_stop is not None and should_stop()):
        languages = ocr.choose_languages("") if ocr.available() else "deu"  # a lone photo has no sample: German first
        root = cache_root if cache_root is not None else (cache_dir.parent if cache_dir is not None else None)
        result, _hit = ocr_engines.read_page_cached(path, cache_root=root, hint="auto", languages=languages, should_stop=should_stop)
        doc.metadata["ocr_languages"] = result.get("languages") or languages
        if not result.get("cancelled"):
            text = _clean(result.get("text") or "")
            for warning in result.get("warnings") or []:
                doc.warnings.append(warning)
        if len(text) >= 12:
            if cache_dir is not None:
                import json

                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    stored = {**ocr.align_words(result, text), "languages": result.get("languages") or languages}
                    (cache_dir / "ocr_1.json").write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
                except OSError as exc:
                    doc.warnings.append(f"OCR boxes not cached ({type(exc).__name__})")
            blocks = ocr_engines.unit_blocks(result, text)
            extra = _clean(ocr_engines.variant_text(result))
            if extra:
                start = len(text) + 2
                blocks.append(Block(text=extra, kind="ocr_variant", char_start=start, char_end=start + len(extra), role="variant"))
                text = text + "\n\n" + extra
        doc.metadata["has_handwriting"] = bool(result.get("handwriting")) and bool(text)
        doc.metadata["ocr_mean_confidence"] = result.get("mean_confidence") if text else None
        doc.metadata["ocr_engine"] = str(result.get("engine") or "") if text else ""
        if ocr_progress is not None and not result.get("cancelled"):
            try:
                ocr_progress(1, 1)
            except Exception:  # noqa: BLE001
                pass
    unit = Unit(number=1, kind="page", text=text, width=width, height=height, images=1, has_text=len(text) >= 12, ocr=bool(text))
    if text:
        unit.blocks.extend(blocks or [Block(text=text, kind="page_text", char_start=0, char_end=len(text))])
    doc.units.append(unit)
    if not text:
        doc.warnings.append("no text: " + ("OCR read nothing" if available else "no OCR engine installed"))
    return doc


PARSERS: dict[str, Callable[..., NormalizedDocument]] = {
    "pdf": parse_pdf, "docx": parse_docx, "pptx": parse_pptx, "markdown": parse_markdown, "text": parse_text, "image": parse_image,
}


def parse(path: str | Path, *, engine: Any = None) -> NormalizedDocument:
    target = Path(path)
    kind = source_type_for(target)
    if kind not in PARSERS:
        raise ValueError(f"unsupported study material: {target.suffix or target.name}")
    return PARSERS[kind](target, engine=engine)
