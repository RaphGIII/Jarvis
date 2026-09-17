"""Real study documents for the tests, written byte by byte: PDF (with text, optional image), DOCX, PPTX, Markdown."""

from __future__ import annotations

import zipfile
from pathlib import Path


def make_pdf(pages: list[list[str]], *, producer: str = "ZEUS test", image_on: set[int] | None = None) -> bytes:
    """A PDF whose pages carry the given lines as real text (Helvetica, WinAnsi), pages 1-based in ``image_on`` get an image."""

    image_on = image_on or set()
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    image = add(b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4 >>\nstream\n"
                b"\x00\xff\xff\x00\nendstream")
    pages_id = len(objects) + 1
    objects.append(b"")  # placeholder for the pages tree
    page_ids: list[int] = []
    for number, lines in enumerate(pages, start=1):
        content = b"BT /F1 12 Tf 72 770 Td 16 TL\n"
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("cp1252", errors="replace")
            content += b"(" + safe + b") Tj T*\n"
        content += b"ET\n"
        if number in image_on:
            content += b"q 100 0 0 100 72 200 cm /Im1 Do Q\n"
        stream = add(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream")
        resources = b"<< /Font << /F1 " + str(font).encode() + b" 0 R >>"
        if number in image_on:
            resources += b" /XObject << /Im1 " + str(image).encode() + b" 0 R >>"
        resources += b" >>"
        page_ids.append(add(b"<< /Type /Page /Parent " + str(pages_id).encode() + b" 0 R /MediaBox [0 0 595 842] /Resources " + resources
                            + b" /Contents " + str(stream).encode() + b" 0 R >>"))
    objects[pages_id - 1] = (b"<< /Type /Pages /Kids [" + b" ".join(str(i).encode() + b" 0 R" for i in page_ids) + b"] /Count "
                             + str(len(page_ids)).encode() + b" >>")
    catalog = add(b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>")
    info = add(b"<< /Producer (" + producer.encode("latin-1") + b") /Title (Test) >>")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root " + str(catalog).encode() + b" 0 R /Info "
            + str(info).encode() + b" 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n")
    return bytes(out)


_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def make_docx(path: Path, blocks: list[tuple[str, str]]) -> Path:
    """blocks: ("h1"|"h2"|"p", text)."""

    body = []
    for kind, text in blocks:
        style = {"h1": "Heading1", "h2": "Heading2"}.get(kind)
        props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        safe = text.replace("&", "&amp;").replace("<", "&lt;")
        body.append(f"<w:p>{props}<w:r><w:t xml:space=\"preserve\">{safe}</w:t></w:r></w:p>")
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {_W}><w:body>{"".join(body)}</w:body></w:document>'
    styles = (f'<?xml version="1.0" encoding="UTF-8"?><w:styles {_W}>'
              '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style></w:styles>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
    return path


def make_pptx(path: Path, slides: list[tuple[str, list[str], str]]) -> Path:
    """slides: (title, bullet lines, speaker notes)."""

    ns = ('xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
    with zipfile.ZipFile(path, "w") as archive:
        ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(len(slides)))
        archive.writestr("ppt/presentation.xml", f'<?xml version="1.0"?><p:presentation {ns}><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>')
        rels = "".join(f'<Relationship Id="rId{i + 1}" Type="slide" Target="slides/slide{i + 1}.xml"/>' for i in range(len(slides)))
        archive.writestr("ppt/_rels/presentation.xml.rels",
                         f'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>')
        for i, (title, lines, notes) in enumerate(slides, start=1):
            title_shape = f'<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>'
            body = "".join(f"<a:p><a:r><a:t>{line}</a:t></a:r></a:p>" for line in lines)
            archive.writestr(f"ppt/slides/slide{i}.xml", f'<?xml version="1.0"?><p:sld {ns}><p:cSld><p:spTree>{title_shape}'
                             f'<p:sp><p:txBody>{body}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
            if notes:
                archive.writestr(f"ppt/notesSlides/notesSlide{i}.xml", f'<?xml version="1.0"?><p:notes {ns}><p:cSld><p:spTree><p:sp><p:txBody>'
                                 f'<a:p><a:r><a:t>{notes}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>')
    return path


HEART_PAGES = [
    ["Physiologie des Herzens", "Kapitel 3: Herzmechanik"],
    ["Druck-Volumen-Beziehung des Ventrikels", "Die Arbeitsdiagramme zeigen Systole und Diastole."],
    ["Der Frank-Starling-Mechanismus", "Eine erhöhte Vorlast dehnt die Sarkomere des Ventrikels.", "Dadurch steigt das Schlagvolumen."],
    ["Nachlast und Kontraktilität", "Sympathikus steigert die Inotropie."],
]
