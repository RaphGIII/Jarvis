"""Figures, diagrams, tables and captions as located objects: a unit, a region, a caption.

"Zeig mir die Abbildung zum Citratzyklus" needs more than a page that mentions a figure:
it needs *where* the figure is, so the viewer can open the page with that region marked.

PDF
    ``page_figures`` walks the page's content stream with a graphics-state stack
    (``q``/``Q``/``cm``) and records where every image is painted (``Do`` on an image
    XObject, inline images), descending into Form XObjects (their ``/Matrix`` and own
    resources, nested, with a cycle and depth guard).  An image paints the unit square
    through the current transform, so its region is the bounding box of the four
    transformed corners.  Regions are fractions of the (crop) page ``[x, y, w, h]`` with
    ``y`` measured from the TOP, page rotation applied -- the viewer's convention.

    Tiny images (logos, bullets, icons) are ignored; an image that covers most of the page
    is a scan, not a figure (``kind: "scan"``).  Adjacent image tiles (one picture cut into
    strips, as some producers do) are merged into one region.

    Captions are lines that *start* with Abb./Abbildung/Fig./Figure/Tabelle/Tab./Schema/
    Diagramm/Grafik followed by a number or a colon.  Their line boxes come from pypdf's
    text visitor (or, optionally, from an injected ``locate`` callable such as the PDF
    worker's ``locate`` op).  A caption is attached to the nearest image region by vertical
    gap (below or above).  A caption without an image region is still a figure object
    (vector-drawn diagrams and text tables have no image) located at its caption line.

Limits, stated honestly: vector-drawn diagrams (paths, no image) are only found through
their caption; a table without a caption is not found at all; the caption box from the
text visitor is an estimate of the line (font size times character count), not a glyph
exact rectangle; captions phrased as prose ("siehe Abbildung 3") are deliberately not
captions.

Slides
    ``slide_figures`` turns the picture shapes of ``study.slides.slide_geometry`` (group
    pictures included -- the geometry walk already resolves groups) into figures; the
    caption is the nearest text shape.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable

Box = list[float]  # [x, y, w, h] fractions, y from the top

MIN_FIGURE_AREA = 0.01      # below ~1% of the page: an icon.  (A 100x100 pt image on A4 is 2.0%: a figure.)
MIN_FIGURE_SIDE = 0.04      # thinner than 4% of the page in either direction: a rule, a bullet, a strip
SCAN_AREA = 0.90            # above 90% of the page: the page itself is a picture
CAPTION_GAP = 0.25          # a caption further than a quarter page from a region is not its caption
MAX_OPERATIONS = 400_000    # content-stream operations walked per page, forms included
MAX_FORM_DEPTH = 8
TILE_TOLERANCE = 0.004

CAPTION = re.compile(
    r"^\s*(?P<word>Abbildung|Abb\.|Fig\.|Figure|Tabelle|Tab\.|Schema|Diagramm|Grafik)"
    r"\s*(?:(?P<num>[A-Z]?\d+(?:[.\-–]\d+)*[a-z]?)\s*[:.–\-]?|:)(?P<rest>.*)$",
    re.IGNORECASE,
)

_KIND_BY_WORD = {"tabelle": "table", "tab.": "table", "schema": "diagram", "diagramm": "diagram", "grafik": "diagram"}


def caption_kind(caption: str) -> str:
    match = CAPTION.match(caption or "")
    if not match:
        return "figure"
    return _KIND_BY_WORD.get(match.group("word").lower(), "figure")


def caption_lines(text: str) -> list[str]:
    """Lines of ``text`` that are captions, in reading order."""

    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line and len(line) <= 400 and CAPTION.match(line):
            out.append(line)
    return out


# ----------------------------------------------------------------------------- geometry

Matrix = tuple[float, float, float, float, float, float]
_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mul(m: Matrix, n: Matrix) -> Matrix:
    """m x n in PDF row-vector convention: apply m first, then n."""

    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)


def _matrix(values: Iterable[Any]) -> Matrix | None:
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(nums) != 6 or not all(math.isfinite(v) for v in nums):
        return None
    return (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])


def _unit_square(ctm: Matrix) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) in user space of the unit square painted through ``ctm``."""

    a, b, c, d, e, f = ctm
    xs = [e, a + e, c + e, a + c + e]
    ys = [f, b + f, d + f, b + d + f]
    return min(xs), min(ys), max(xs), max(ys)


class _Page:
    """Converts user-space rectangles of one page into viewer fractions."""

    def __init__(self, page: Any) -> None:
        box = None
        for name in ("cropbox", "mediabox"):
            try:
                box = getattr(page, name)
                left, bottom, right, top = float(box.left), float(box.bottom), float(box.right), float(box.top)
                break
            except Exception:  # noqa: BLE001
                box = None
        if box is None:
            left, bottom, right, top = 0.0, 0.0, 612.0, 792.0
        self.left, self.bottom = min(left, right), min(bottom, top)
        self.width = max(1e-6, abs(right - left))
        self.height = max(1e-6, abs(top - bottom))
        try:
            rotate = int(page.get("/Rotate") or 0) % 360
        except Exception:  # noqa: BLE001
            rotate = 0
        self.rotate = rotate if rotate in (0, 90, 180, 270) else 0

    def fraction(self, x0: float, y0: float, x1: float, y1: float) -> Box | None:
        u0, u1 = (x0 - self.left) / self.width, (x1 - self.left) / self.width
        v0, v1 = 1.0 - (y1 - self.bottom) / self.height, 1.0 - (y0 - self.bottom) / self.height
        corners = [(u0, v0), (u1, v0), (u0, v1), (u1, v1)]
        if self.rotate == 90:
            corners = [(1.0 - v, u) for u, v in corners]
        elif self.rotate == 180:
            corners = [(1.0 - u, 1.0 - v) for u, v in corners]
        elif self.rotate == 270:
            corners = [(v, 1.0 - u) for u, v in corners]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        fx0, fy0 = max(0.0, min(xs)), max(0.0, min(ys))
        fx1, fy1 = min(1.0, max(xs)), min(1.0, max(ys))
        if fx1 <= fx0 or fy1 <= fy0:
            return None
        return [round(fx0, 5), round(fy0, 5), round(fx1 - fx0, 5), round(fy1 - fy0, 5)]


def _resolve(obj: Any) -> Any:
    try:
        return obj.get_object() if hasattr(obj, "get_object") else obj
    except Exception:  # noqa: BLE001
        return None


def _xobjects(resources: Any) -> Any:
    resources = _resolve(resources)
    if resources is None or not hasattr(resources, "get"):
        return None
    xobjects = _resolve(resources.get("/XObject"))
    return xobjects if hasattr(xobjects, "get") else None


def image_placements(page: Any, reader: Any = None) -> list[dict[str, Any]]:
    """Every image painted on ``page``: ``{"rect": (x0, y0, x1, y1) user space, "name", "inline"}``.

    Never raises: a broken content stream yields what was found before the break.
    """

    from pypdf.generic import ContentStream

    found: list[dict[str, Any]] = []
    budget = [MAX_OPERATIONS]

    def walk(content: Any, resources: Any, ctm: Matrix, depth: int, active: tuple[int, ...]) -> None:
        try:
            operations = content.operations
        except Exception:  # noqa: BLE001 - unparseable stream: nothing from here
            return
        xobjects = _xobjects(resources)
        stack: list[Matrix] = []
        for operands, operator in operations:
            budget[0] -= 1
            if budget[0] < 0:
                return
            try:
                if operator == b"q":
                    stack.append(ctm)
                elif operator == b"Q":
                    if stack:
                        ctm = stack.pop()
                elif operator == b"cm":
                    m = _matrix(operands)
                    if m is not None:
                        ctm = _mul(m, ctm)
                elif operator == b"INLINE IMAGE":
                    found.append({"rect": _unit_square(ctm), "name": "", "inline": True})
                elif operator == b"Do" and operands and xobjects is not None:
                    name = operands[0]
                    ref = xobjects.get(name)
                    xobj = _resolve(ref)
                    if xobj is None or not hasattr(xobj, "get"):
                        continue
                    subtype = str(xobj.get("/Subtype") or "")
                    if subtype == "/Image":
                        found.append({"rect": _unit_square(ctm), "name": str(name), "inline": False})
                    elif subtype == "/Form" and depth < MAX_FORM_DEPTH:
                        key = getattr(ref, "idnum", None) or id(xobj)
                        if key in active:
                            continue
                        form_matrix = _matrix(xobj.get("/Matrix") or _IDENTITY) or _IDENTITY
                        form_resources = xobj.get("/Resources") or resources
                        try:
                            form_content = ContentStream(xobj, reader)
                        except Exception:  # noqa: BLE001
                            continue
                        walk(form_content, form_resources, _mul(form_matrix, ctm), depth + 1, active + (key,))
            except Exception:  # noqa: BLE001 - one odd operator never ends the walk
                continue

    try:
        resources = page.get("/Resources")
        has_xobjects = bool(_xobjects(resources))
        contents = page.get_contents()
        if contents is None:
            return found
        if not has_xobjects:
            try:
                raw = contents.get_data()
            except Exception:  # noqa: BLE001
                raw = b""
            if b"BI" not in raw:
                return found  # nothing that could paint an image: skip the slow parse
        walk(contents, resources, _IDENTITY, 0, ())
    except Exception:  # noqa: BLE001
        pass
    return found


def _area(box: Box) -> float:
    return box[2] * box[3]


def _touch(a: Box, b: Box, tol: float = TILE_TOLERANCE) -> bool:
    return not (a[0] + a[2] + tol < b[0] or b[0] + b[2] + tol < a[0] or a[1] + a[3] + tol < b[1] or b[1] + b[3] + tol < a[1])


def _union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)]


def _merge_tiles(boxes: list[Box]) -> list[Box]:
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _touch(boxes[i], boxes[j]):
                    boxes[i] = _union(boxes[i], boxes[j])
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def image_regions(page: Any, reader: Any = None) -> list[dict[str, Any]]:
    """Image regions of a page as fractions: ``{"kind": "figure"|"scan", "box"}``, icons dropped, tiles merged."""

    geometry = _Page(page)
    scans: list[Box] = []
    pictures: list[Box] = []
    for placement in image_placements(page, reader):
        box = geometry.fraction(*placement["rect"])
        if box is None:
            continue
        if _area(box) >= SCAN_AREA:
            scans.append(box)
        else:
            pictures.append(box)
    regions = [{"kind": "scan", "box": box} for box in _merge_tiles(scans)]
    for box in _merge_tiles(pictures):
        if _area(box) >= SCAN_AREA:
            regions.append({"kind": "scan", "box": box})
        elif _area(box) >= MIN_FIGURE_AREA and min(box[2], box[3]) >= MIN_FIGURE_SIDE:
            regions.append({"kind": "figure", "box": box})
    return regions


# ---------------------------------------------------------------------------- text lines

def text_lines(page: Any) -> list[dict[str, Any]]:
    """The page's text as lines with estimated boxes (fractions, y from the top), via pypdf's text visitor."""

    geometry = _Page(page)
    runs: list[tuple[float, float, float, str]] = []  # x, y, size, text (user space)

    def visitor(text: str, cm: Any, tm: Any, _font: Any, font_size: Any) -> None:
        if not text or not text.strip():
            return
        try:
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            scale = math.hypot(tm[2] * cm[0] + tm[3] * cm[2], tm[2] * cm[1] + tm[3] * cm[3]) or 1.0
            size = float(font_size or 10.0) * (scale if scale > 0 else 1.0)
            if size <= 0 or size > 200:
                size = 10.0
        except Exception:  # noqa: BLE001
            return
        for piece in text.split("\n"):
            if piece.strip():
                runs.append((float(x), float(y), size, piece))

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:  # noqa: BLE001 - whatever was visited before the break is kept
        pass
    lines: list[dict[str, Any]] = []
    for x, y, size, piece in sorted(runs, key=lambda r: (-r[1], r[0])):
        line = next((ln for ln in lines if abs(ln["y"] - y) <= max(1.0, 0.4 * size)), None)
        width = 0.5 * size * len(piece)
        if line is None:
            lines.append({"y": y, "size": size, "x0": x, "x1": x + width, "parts": [(x, piece)]})
        else:
            line["x0"], line["x1"] = min(line["x0"], x), max(line["x1"], x + width)
            line["size"] = max(line["size"], size)
            line["parts"].append((x, piece))
    out = []
    for line in lines:
        text = re.sub(r"\s+", " ", " ".join(p for _, p in sorted(line["parts"], key=lambda p: p[0]))).strip()
        box = geometry.fraction(line["x0"], line["y"] - 0.25 * line["size"], line["x1"], line["y"] + 0.85 * line["size"])
        if text and box is not None:
            out.append({"text": text, "box": box})
    out.sort(key=lambda ln: (ln["box"][1], ln["box"][0]))
    return out


# ------------------------------------------------------------------------------ captions

def _vertical_gap(a: Box, b: Box) -> float:
    """0 when the boxes overlap vertically, else the empty space between them."""

    if a[1] + a[3] < b[1]:
        return b[1] - (a[1] + a[3])
    if b[1] + b[3] < a[1]:
        return a[1] - (b[1] + b[3])
    return 0.0


def _horizontal_overlap(a: Box, b: Box) -> bool:
    return not (a[0] + a[2] < b[0] or b[0] + b[2] < a[0])


def _nearby(region: Box, lines: list[dict[str, Any]], exclude: set[str], limit: int = 3) -> str:
    near = []
    for line in lines:
        if line["text"] in exclude or line.get("box") is None:
            continue
        gap = _vertical_gap(region, line["box"])
        if gap <= 0.06 and _horizontal_overlap(region, line["box"]):
            near.append((gap, line["text"]))
    near.sort(key=lambda item: item[0])
    return " ".join(text for _, text in near[:limit])[:400]


def attach_captions(regions: list[dict[str, Any]], captions: list[dict[str, Any]], lines: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Figures from image regions and caption lines (``{"text", "box" | None}``), numbered top to bottom."""

    lines = lines or []
    figures: list[dict[str, Any]] = []
    candidates = [r for r in regions if r["kind"] != "scan"]
    free_regions = list(range(len(candidates)))
    attached: dict[int, dict[str, Any]] = {}
    pairs = []
    for ci, caption in enumerate(captions):
        if caption.get("box") is None:
            continue
        for ri in free_regions:
            gap = _vertical_gap(candidates[ri]["box"], caption["box"])
            penalty = 0.0 if _horizontal_overlap(candidates[ri]["box"], caption["box"]) else 0.05
            pairs.append((gap + penalty, gap, ci, ri))
    pairs.sort()
    used_captions: set[int] = set()
    for _score, gap, ci, ri in pairs:
        if ci in used_captions or ri in attached or gap > CAPTION_GAP:
            continue
        attached[ri] = captions[ci]
        used_captions.add(ci)
    rest_captions = [ci for ci in range(len(captions)) if ci not in used_captions]
    rest_regions = [ri for ri in range(len(candidates)) if ri not in attached]
    # one caption and one picture left on the page: they belong together, however far apart
    # (also when the caption position is unknown).  Several of each without positions: reading order.
    if rest_captions and len(rest_captions) == len(rest_regions) and (
            len(rest_captions) == 1 or all(captions[ci].get("box") is None for ci in rest_captions)):
        ordered = sorted(rest_regions, key=lambda ri: (candidates[ri]["box"][1], candidates[ri]["box"][0]))
        for ci, ri in zip(rest_captions, ordered):
            attached[ri] = captions[ci]
            used_captions.add(ci)
        rest_captions = []
    for ri, region in enumerate(candidates):
        caption = attached.get(ri)
        caption_text = caption["text"] if caption else ""
        figures.append({"kind": caption_kind(caption_text) if caption else "figure", "box": region["box"], "caption": caption_text,
                        "source": "image", "caption_box": caption.get("box") if caption else None,
                        "context": _nearby(region["box"], lines, {caption_text})})
    for ci in rest_captions:
        caption = captions[ci]
        figures.append({"kind": caption_kind(caption["text"]), "box": caption.get("box"), "caption": caption["text"],
                        "source": "caption", "caption_box": caption.get("box"), "context": ""})
    for region in regions:
        if region["kind"] == "scan":
            figures.append({"kind": "scan", "box": region["box"], "caption": "", "source": "image", "caption_box": None, "context": ""})
    figures.sort(key=lambda f: (f["box"] is None, f["box"][1] if f["box"] else 0.0, f["box"][0] if f["box"] else 0.0))
    for number, figure in enumerate(figures, start=1):
        figure["index"] = number
    return figures


def figure_text(figure: dict[str, Any], unit_text: str) -> str:
    """The searchable text of a figure: its caption, the lines around it, a little text after the caption."""

    caption = (figure.get("caption") or "").strip()
    parts = [caption] if caption else []
    context = (figure.get("context") or "").strip()
    if context:
        parts.append(context)
    text = unit_text or ""
    if caption and text:
        at = _find(text, caption)
        if at >= 0:
            after = text[at + len(caption): at + len(caption) + 240].strip()
            if after:
                parts.append(after)
    if not parts and figure.get("kind") == "scan":
        parts.append(text[:240].strip())
    joined = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
    return joined[:600]


def _find(text: str, phrase: str) -> int:
    at = text.find(phrase)
    if at >= 0:
        return at
    folded = re.sub(r"\s+", " ", phrase).strip()
    match = re.search(r"\s+".join(re.escape(w) for w in folded.split(" ")), text)
    return match.start() if match else -1


Locate = Callable[[int, list[str]], dict[str, Any]]


def page_figures(page: Any, number: int, *, reader: Any = None, text: str | None = None, locate: Locate | None = None) -> list[dict[str, Any]]:
    """Figures of one PDF page.  ``text``: the page text when already extracted (gates the positional text pass).

    ``locate(page_number, phrases)``: optional, the PDF worker's ``locate`` answer
    (``{"matches": [{"phrase", "rects": [[x, y, w, h] points from the top]}], "page_size": [w, h]}``)
    used for captions the text visitor could not place.
    """

    try:
        regions = image_regions(page, reader)
    except Exception:  # noqa: BLE001
        regions = []
    wanted = caption_lines(text) if text is not None else None
    lines: list[dict[str, Any]] = []
    if wanted is None or wanted or regions:
        try:
            lines = text_lines(page)
        except Exception:  # noqa: BLE001
            lines = []
    captions = [{"text": ln["text"], "box": ln["box"]} for ln in lines if CAPTION.match(ln["text"])]
    if wanted:
        placed = {re.sub(r"\s+", " ", c["text"]) for c in captions}
        for line in wanted:
            if re.sub(r"\s+", " ", line) not in placed and not any(line.startswith(c["text"]) or c["text"].startswith(line) for c in captions):
                captions.append({"text": line, "box": None})
    if locate is not None and any(c["box"] is None for c in captions):
        try:
            answer = locate(number, [c["text"] for c in captions if c["box"] is None]) or {}
            size = answer.get("page_size") or [0, 0]
            for match in answer.get("matches") or []:
                rects = match.get("rects") or []
                target = next((c for c in captions if c["box"] is None and c["text"] == match.get("phrase")), None)
                if target is None or not rects or not size[0] or not size[1]:
                    continue
                x0 = min(r[0] for r in rects)
                y0 = min(r[1] for r in rects)
                x1 = max(r[0] + r[2] for r in rects)
                y1 = max(r[1] + r[3] for r in rects)
                target["box"] = [round(max(0.0, x0 / size[0]), 5), round(max(0.0, y0 / size[1]), 5),
                                 round(min(1.0, (x1 - x0) / size[0]), 5), round(min(1.0, (y1 - y0) / size[1]), 5)]
        except Exception:  # noqa: BLE001 - the worker is optional
            pass
    figures = attach_captions(regions, captions, lines)
    unit_text = text if text is not None else "\n".join(ln["text"] for ln in lines)
    for figure in figures:
        figure["text"] = figure_text(figure, unit_text)
    return figures


def pdf_figures(pdf_path: str | Path, *, locate: Locate | None = None) -> dict[int, list[dict[str, Any]]]:
    """Per 1-based page: the figures of a PDF.  Pages without figures are left out.  Never raises."""

    out: dict[int, list[dict[str, Any]]] = {}
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path), strict=False)
        pages = list(reader.pages)
    except Exception:  # noqa: BLE001
        return out
    for index, page in enumerate(pages):
        try:
            figures = page_figures(page, index + 1, reader=reader, locate=locate)
        except Exception:  # noqa: BLE001
            continue
        if figures:
            out[index + 1] = figures
    return out


def image_count(figures: list[dict[str, Any]]) -> int:
    """How many real pictures (not scans, not caption-only objects) a unit shows."""

    return sum(1 for f in figures if f.get("source") == "image" and f.get("kind") != "scan")


# -------------------------------------------------------------------------------- slides

def slide_figures(geometry: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Figures of one slide from ``slide_geometry(...)[n]`` (``{"size", "shapes"}``): pictures, captioned by the nearest text shape."""

    shapes = (geometry or {}).get("shapes") or []
    texts = [s for s in shapes if s.get("kind") in {"text", "body", "title"} and s.get("text") and s.get("box")]
    regions = []
    for shape in shapes:
        if shape.get("kind") != "picture":
            continue
        box = shape.get("box")
        if not box:
            regions.append({"kind": "figure", "box": None})
            continue
        area = box[2] * box[3]
        if area >= SCAN_AREA:
            regions.append({"kind": "scan", "box": list(box)})
        elif area >= MIN_FIGURE_AREA * 0.5 and min(box[2], box[3]) >= MIN_FIGURE_SIDE * 0.5:
            regions.append({"kind": "figure", "box": list(box)})
    figures: list[dict[str, Any]] = []
    used: set[int] = set()
    for region in regions:
        if region["kind"] == "scan":
            figures.append({"kind": "scan", "box": region["box"], "caption": "", "source": "image", "caption_box": None, "context": ""})
            continue
        best = None
        if region["box"] is not None:
            scored = []
            for ti, shape in enumerate(texts):
                if ti in used:
                    continue
                gap = _vertical_gap(region["box"], shape["box"])
                is_caption = bool(CAPTION.match(shape["text"]))
                if shape["kind"] == "title" and not is_caption:
                    continue
                if not is_caption and gap > 0.2:
                    continue
                penalty = (0.0 if _horizontal_overlap(region["box"], shape["box"]) else 0.1) - (0.5 if is_caption else 0.0)
                scored.append((gap + penalty, ti))
            if scored:
                best = min(scored)[1]
        caption = ""
        caption_box = None
        if best is not None:
            used.add(best)
            first = texts[best]["text"].split("\n")
            caption = next((p for p in first if CAPTION.match(p)), first[0])[:300]
            caption_box = texts[best]["box"]
        figures.append({"kind": caption_kind(caption) if caption else "figure", "box": region["box"], "caption": caption,
                        "source": "image", "caption_box": caption_box, "context": ""})
    figures.sort(key=lambda f: (f["box"] is None, f["box"][1] if f["box"] else 0.0, f["box"][0] if f["box"] else 0.0))
    for number, figure in enumerate(figures, start=1):
        figure["index"] = number
    return figures


# ------------------------------------------------------------------------------- chunks

def figure_chunks(unit: Any) -> list[dict[str, Any]]:
    """One index chunk per figure of a unit, shaped like ``study.index.chunk_unit`` plus ``figure_index``, ``box``, ``kind``."""

    chunks: list[dict[str, Any]] = []
    page = unit.number if unit.kind == "page" else None
    slide = unit.number if unit.kind == "slide" else None
    for figure in getattr(unit, "figures", None) or []:
        text = (figure.get("text") or figure_text(figure, unit.text)).strip()
        if not text:
            continue
        caption = figure.get("caption") or ""
        start = _find(unit.text, caption) if caption else -1
        char_start, char_end = (start, start + len(caption)) if start >= 0 else (0, 0)
        chunks.append({"unit": unit.number, "page": page, "slide": slide, "heading": unit.title, "paragraph": None,
                       "char_start": char_start, "char_end": char_end, "text": text,
                       "figure_index": figure.get("index"), "box": figure.get("box"), "kind": figure.get("kind", "figure")})
    return chunks
