"""The normalized representation every parser produces, whatever the file format.

A document is a list of units.  A unit is what the owner navigates to: a PDF page,
a slide, a section of a Word or Markdown document.  A unit holds blocks -- the
smallest pieces of text with a stable location (a paragraph, a heading, a page's
text run).  Locations are exact where the format knows them (PDF page, slide
number) and structural where it does not (Word: heading path + paragraph index;
Word has no stable page numbers without a layout engine, so none are invented).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SOURCE_TYPES = ("pdf", "docx", "pptx", "markdown", "text", "image")


@dataclass
class Location:
    """Where a piece of text lives.  Only the fields the format really knows are set."""

    page: int | None = None           # 1-based PDF page
    slide: int | None = None          # 1-based slide
    heading_path: list[str] = field(default_factory=list)
    paragraph: int | None = None      # 1-based paragraph index within the document
    unit: int = 1                     # 1-based unit (page / slide / section) the viewer opens
    char_start: int = 0               # offsets into the unit's text
    char_end: int = 0

    def label(self) -> str:
        """The owner-facing German label: "Seite 38", "Folie 4", "Herzmechanik › Vorlast · Absatz 3"."""

        if self.page is not None:
            return f"Seite {self.page}"
        if self.slide is not None:
            return f"Folie {self.slide}"
        parts = []
        if self.heading_path:
            parts.append(" › ".join(self.heading_path))
        if self.paragraph is not None:
            parts.append(f"Absatz {self.paragraph}")
        return " · ".join(parts) or f"Abschnitt {self.unit}"

    def to_dict(self) -> dict[str, Any]:
        return {"page": self.page, "slide": self.slide, "heading_path": list(self.heading_path), "paragraph": self.paragraph,
                "unit": self.unit, "char_start": self.char_start, "char_end": self.char_end, "label": self.label()}


@dataclass
class Block:
    """A paragraph, heading or text run with its location inside its unit."""

    text: str
    kind: str = "paragraph"           # paragraph | heading | page_text | slide_title | notes | table
    heading_path: list[str] = field(default_factory=list)
    paragraph: int | None = None
    char_start: int = 0
    char_end: int = 0
    box: list[float] | None = None    # slides: [x, y, w, h] as fractions of the slide, when the file places the shape
    confidence: float | None = None   # OCR blocks: mean recognition confidence 0..1; None for text the file carries
    role: str = ""                    # OCR blocks: title | body | annotation | label | diagram_label


@dataclass
class Unit:
    """What the viewer opens: a page, a slide, a section."""

    number: int                       # 1-based
    kind: str                         # page | slide | section
    text: str = ""
    title: str = ""
    blocks: list[Block] = field(default_factory=list)
    width: float = 0.0                # page size in points (PDF), when known
    height: float = 0.0
    images: int = 0                   # embedded images: "zeig mir die Abbildung"
    has_text: bool = True
    ocr: bool = False                 # text came from OCR, not from the file
    # located figures (study.figures): {"kind": figure|table|diagram|scan, "box": [x, y, w, h] fractions,
    # y from the top, or None; "caption", "text", "index": 1-based within the unit}
    figures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    source_type: str
    title: str
    units: list[Unit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def text_units(self) -> int:
        return sum(1 for unit in self.units if unit.has_text and unit.text.strip())
