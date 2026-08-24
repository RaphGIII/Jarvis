"""Turning the user's files into a knowledge graph they can actually navigate.

The temptation with ingestion is to store everything and let retrieval sort it
out.  That produces a graph with thousands of nodes and no structure -- a list
with extra steps -- and it is why so many "second brain" tools end up as search
boxes over a pile.

So three decisions shape this module:

*Documents become several nodes, not one.*  A 40-page design document has a
dozen distinct ideas in it, and a single node called "design.md" can only ever
be retrieved as a whole.  Markdown headings are the author's own structure and
are far better section boundaries than a fixed character count, so they are used
where they exist.

*Links that already exist are preserved as edges.*  ``[[wikilinks]]`` and
markdown links are the user telling you how their notes relate.  Discarding them
and re-deriving relationships with an embedding model would be throwing away
ground truth to replace it with a guess.

*Re-ingesting a file updates it rather than duplicating it.*  Nodes carry the
source path as provenance, so a folder can be re-scanned as it changes without
the graph growing a new copy of everything each time.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from knowledge.graph import EdgeType, KnowledgeGraph, Node, NodeType

#: Extensions read directly as text.
TEXT_SUFFIXES = frozenset(
    {
        ".md", ".markdown", ".txt", ".rst", ".org",
        ".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".sh", ".ps1",
        ".html", ".css", ".xml", ".csv",
    }
)

#: Handled by an external extractor.
BINARY_SUFFIXES = frozenset({".pdf"})

#: Never walked into.
SKIP_DIRECTORIES = frozenset(
    {
        ".git", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", ".idea",
        ".vscode", "site-packages", ".cache",
    }
)

#: Files above this are summarised by their head rather than stored whole.
MAX_FILE_BYTES = 2_000_000
#: Target size of one section node.  Big enough to hold an idea, small enough
#: that retrieving it does not flood a 24k context window.
SECTION_TARGET_CHARS = 1400
SECTION_MAX_CHARS = 3000


@dataclass
class IngestReport:
    files_seen: int = 0
    files_ingested: int = 0
    nodes_created: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "files_ingested": self.files_ingested,
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "skipped": self.skipped[:50],
            "errors": self.errors[:50],
        }


@dataclass
class Section:
    """One retrievable piece of a document."""

    title: str
    body: str
    #: Heading depth, or 0 for a document with no headings.
    level: int = 0


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_text(path: Path, *, venv_python: Path | None = None) -> tuple[str, str]:
    """Return ``(text, note)``.  ``note`` explains any degradation."""

    suffix = path.suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return _read_pdf(path, venv_python=venv_python)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", f"unreadable: {exc}"
    if b"\x00" in data[:2048]:
        return "", "looks like a binary file"
    truncated = ""
    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES]
        truncated = f"truncated at {MAX_FILE_BYTES} bytes"
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n"), truncated


def _read_pdf(path: Path, *, venv_python: Path | None = None) -> tuple[str, str]:
    """Extract PDF text, using whichever interpreter has pypdf available.

    Tried in-process first, then the extras virtualenv.  A PDF that cannot be
    read says so rather than silently producing an empty note, because an empty
    node is worse than an absent one: it looks like the document was ingested.
    """

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages), ""
    except ImportError:
        pass
    except Exception as exc:
        return "", f"pdf could not be parsed: {exc}"

    interpreter = venv_python or _extras_python()
    if interpreter is None:
        return "", "no PDF extractor available (install pypdf)"

    script = (
        "import sys;from pypdf import PdfReader;"
        "r=PdfReader(sys.argv[1]);"
        "sys.stdout.write('\\n\\n'.join((p.extract_text() or '') for p in r.pages))"
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", script, str(path)],
            capture_output=True, text=True, timeout=180, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"pdf extraction failed: {exc}"
    if completed.returncode != 0:
        return "", f"pdf extraction failed: {(completed.stderr or '')[-200:]}"
    return completed.stdout, ""


def _extras_python() -> Path | None:
    root = Path(__file__).resolve().parent.parent / ".venv-speech"
    for candidate in (root / "Scripts" / "python.exe", root / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def split_sections(text: str, *, title: str = "") -> list[Section]:
    """Break a document into retrievable pieces.

    Markdown headings win where they exist: they are the author's own view of
    where one idea ends and the next begins, which no character count can
    reconstruct.  Everything else falls back to paragraph-aware chunking.
    """

    text = (text or "").strip()
    if not text:
        return []

    headings = list(_HEADING.finditer(text))
    if headings:
        sections: list[Section] = []
        preamble = text[: headings[0].start()].strip()
        if preamble:
            sections.append(Section(title=title or "Introduction", body=preamble, level=0))
        for index, match in enumerate(headings):
            start = match.end()
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            heading = match.group(2).strip()
            level = len(match.group(1))
            # A section that is still enormous is chunked again; a heading is a
            # good boundary but not a guarantee of a reasonable size.
            for piece in _chunk(body):
                sections.append(Section(title=heading, body=piece, level=level))
        if sections:
            return sections

    return [Section(title=title or "Document", body=piece) for piece in _chunk(text)]


def _chunk(text: str) -> list[str]:
    """Paragraph-aware chunking, so a chunk never starts mid-sentence."""

    if len(text) <= SECTION_MAX_CHARS:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > SECTION_MAX_CHARS and current:
            chunks.append(current)
            current = paragraph
        elif len(candidate) >= SECTION_TARGET_CHARS:
            chunks.append(candidate)
            current = ""
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    # A single paragraph longer than the maximum still has to be cut somewhere.
    result: list[str] = []
    for chunk in chunks:
        while len(chunk) > SECTION_MAX_CHARS * 2:
            result.append(chunk[:SECTION_MAX_CHARS])
            chunk = chunk[SECTION_MAX_CHARS:]
        result.append(chunk)
    return [item for item in result if item.strip()]


_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_MDLINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def extract_links(text: str) -> list[str]:
    """Link targets the author wrote, in the order they appear.

    These are ground truth about how the user's notes relate.  Discarding them
    and re-deriving relationships from embeddings would replace knowledge with
    a guess.
    """

    found: list[str] = []
    for match in _WIKILINK.finditer(text or ""):
        target = match.group(1).strip()
        if target and target not in found:
            found.append(target)
    for match in _MDLINK.finditer(text or ""):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        stem = Path(target).stem
        if stem and stem not in found:
            found.append(stem)
    return found


# --------------------------------------------------------------------------
# Ingesting
# --------------------------------------------------------------------------

class Ingester:
    """Puts files into the graph, and keeps them there consistently."""

    def __init__(self, graph: KnowledgeGraph, *, project_id: str = "") -> None:
        self.graph = graph
        self.project_id = project_id

    # -- files ---------------------------------------------------------

    def ingest_file(self, path: str | Path, *, report: IngestReport | None = None) -> IngestReport:
        report = report or IngestReport()
        target = Path(path).expanduser().resolve()
        report.files_seen += 1

        if not target.is_file():
            report.errors.append({"path": str(target), "error": "not a file"})
            return report

        suffix = target.suffix.lower()
        if suffix not in TEXT_SUFFIXES and suffix not in BINARY_SUFFIXES:
            report.skipped.append({"path": str(target), "reason": f"unsupported type {suffix or '(none)'}"})
            return report

        text, note = read_text(target)
        if not text.strip():
            report.skipped.append({"path": str(target), "reason": note or "empty"})
            return report

        provenance = str(target)
        existing = {node.id: node for node in self._nodes_for(provenance)}

        document = self._upsert(
            node_type=NodeType.FILE,
            title=target.name,
            body=(note + "\n\n" if note else "") + text[:600],
            provenance=provenance,
            key="document",
            existing=existing,
            report=report,
            metadata={"path": provenance, "suffix": suffix, "bytes": target.stat().st_size},
        )

        sections = split_sections(text, title=target.stem)
        for index, section in enumerate(sections):
            node = self._upsert(
                node_type=NodeType.NOTE,
                title=f"{target.stem} — {section.title}" if section.title else f"{target.stem} #{index + 1}",
                body=section.body,
                provenance=provenance,
                key=f"section:{index}",
                existing=existing,
                report=report,
                metadata={"path": provenance, "section": index, "level": section.level},
            )
            if self.graph.link(document.id, node.id, EdgeType.PART_OF):
                report.edges_created += 1
            for target_title in extract_links(section.body):
                if self._link_by_title(node, target_title):
                    report.edges_created += 1

        # Anything previously ingested from this file but no longer present --
        # a deleted section -- is removed, so the graph tracks the file rather
        # than accumulating its history.
        for stale in existing.values():
            self.graph.delete_node(stale.id)

        report.files_ingested += 1
        return report

    def ingest_folder(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        max_files: int = 500,
        report: IngestReport | None = None,
    ) -> IngestReport:
        report = report or IngestReport()
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            report.errors.append({"path": str(root), "error": "not a directory"})
            return report

        for target in self._walk(root, recursive=recursive):
            if report.files_ingested >= max_files:
                report.skipped.append({"path": str(root), "reason": f"stopped at {max_files} files"})
                break
            self.ingest_file(target, report=report)
        return report

    def ingest_text(
        self,
        title: str,
        text: str,
        *,
        source: str = "user",
        node_type: NodeType = NodeType.NOTE,
        report: IngestReport | None = None,
    ) -> Node:
        """Store something the user typed or said, not a file."""

        report = report or IngestReport()
        node = self.graph.remember(node_type, title, text, provenance=source)
        report.nodes_created += 1
        for target_title in extract_links(text):
            if self._link_by_title(node, target_title):
                report.edges_created += 1
        return node

    # -- internals -----------------------------------------------------

    def _walk(self, root: Path, *, recursive: bool) -> Iterable[Path]:
        import os

        if not recursive:
            yield from sorted(item for item in root.iterdir() if item.is_file())
            return
        for current, directories, filenames in os.walk(root):
            directories[:] = sorted(name for name in directories if name not in SKIP_DIRECTORIES and not name.startswith("."))
            for name in sorted(filenames):
                yield Path(current) / name

    def _nodes_for(self, provenance: str) -> list[Node]:
        return [node for node in self.graph.nodes(limit=5000) if node.provenance == provenance]

    def _upsert(
        self,
        *,
        node_type: NodeType,
        title: str,
        body: str,
        provenance: str,
        key: str,
        existing: dict[str, Node],
        report: IngestReport,
        metadata: dict[str, Any],
    ) -> Node:
        """Create or update the node for one piece of one file.

        Identity is (provenance, key) rather than the title, so editing a
        heading updates the node instead of orphaning it and creating a
        duplicate -- which is how a re-scanned folder doubles in size.
        """

        match = None
        for node in existing.values():
            if node.metadata.get("ingest_key") == key:
                match = node
                break

        payload = dict(metadata)
        payload["ingest_key"] = key
        payload["content_hash"] = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

        if match is not None:
            existing.pop(match.id, None)
            if match.metadata.get("content_hash") == payload["content_hash"] and match.title == title:
                return match
            match.title = title
            match.body = body
            match.metadata = payload
            self.graph.add_node(match)
            report.nodes_updated += 1
            return match

        node = self.graph.remember(
            node_type, title, body, provenance=provenance, metadata=payload,
            tags=[node_type.value],
        )
        report.nodes_created += 1
        return node

    def _link_by_title(self, source: Node, title: str) -> bool:
        """Connect to an existing node with this title, if there is one.

        Deliberately does not create the missing target.  A link to a note that
        does not exist yet is a common and harmless state in a personal wiki,
        and inventing an empty node for every dangling reference fills the
        graph with placeholders nobody wrote.
        """

        for node_type in (NodeType.NOTE, NodeType.CONCEPT, NodeType.FILE, NodeType.PROJECT):
            found = self.graph.find_by_title(node_type, title)
            if found is not None and found.id != source.id:
                return bool(self.graph.link(source.id, found.id, EdgeType.MENTIONS))
        return False
