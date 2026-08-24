"""Getting the user's files into the graph without turning it into a pile.

The failure mode being designed against is the one every "second brain" tool
eventually has: thousands of nodes, no structure, and a search box over a heap.
Three properties prevent it, and each has tests here -- documents become several
retrievable pieces, the links the author already wrote survive as edges, and
re-scanning a folder updates it instead of duplicating it.
"""

from __future__ import annotations

import pytest

from knowledge.graph import EdgeType, KnowledgeGraph, NodeType
from knowledge.ingest import (
    SECTION_MAX_CHARS,
    Ingester,
    IngestReport,
    extract_links,
    read_text,
    split_sections,
)


@pytest.fixture()
def graph(tmp_path):
    instance = KnowledgeGraph(tmp_path / "graph.db")
    yield instance
    instance.close()


@pytest.fixture()
def ingester(graph):
    return Ingester(graph)


NOTE = """# Project Jarvis

The personal AI system, running locally.

## Speech

Whisper for hearing, Piper for speaking. See [[Ollama]] for the runtime.

## Plans

Build the device gateway next. Related: [notes](gateway.md).
"""


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def test_headings_become_separate_sections():
    """A single node for a 40-page document can only be retrieved as a whole."""

    sections = split_sections(NOTE)

    assert [s.title for s in sections] == ["Project Jarvis", "Speech", "Plans"]


def test_a_preamble_before_the_first_heading_is_kept():
    sections = split_sections("Some intro text.\n\n# First\n\nBody", title="doc")

    assert sections[0].title == "doc"
    assert "intro" in sections[0].body


def test_heading_depth_is_recorded():
    sections = split_sections("# Top\n\nx\n\n### Deep\n\ny")

    assert [s.level for s in sections] == [1, 3]


def test_a_document_without_headings_still_splits_on_paragraphs():
    text = "\n\n".join(f"Paragraph number {index} with some content in it." * 12 for index in range(8))

    sections = split_sections(text, title="plain")

    assert len(sections) > 1
    assert all(len(s.body) <= SECTION_MAX_CHARS * 2 for s in sections)


def test_a_short_document_stays_one_section():
    assert len(split_sections("Just a sentence.")) == 1


def test_an_empty_document_yields_nothing():
    assert split_sections("   ") == []


def test_a_huge_single_paragraph_is_still_cut():
    """No punctuation and no blank lines must not defeat chunking."""

    sections = split_sections("word " * 5000)

    assert len(sections) > 1


def test_a_giant_section_under_one_heading_is_chunked_again():
    text = "# Everything\n\n" + "\n\n".join("paragraph " * 60 for _ in range(20))

    sections = split_sections(text)

    assert len(sections) > 1
    assert all(s.title == "Everything" for s in sections)


# --------------------------------------------------------------------------
# Links the author wrote
# --------------------------------------------------------------------------

def test_wikilinks_are_extracted():
    assert extract_links("See [[Ollama]] and [[Device Gateway]].") == ["Ollama", "Device Gateway"]


def test_a_wikilink_alias_uses_the_target():
    assert extract_links("[[Ollama|the runtime]]") == ["Ollama"]


def test_relative_markdown_links_are_extracted_by_stem():
    assert extract_links("[notes](gateway.md)") == ["gateway"]


def test_external_links_are_not_treated_as_graph_edges():
    """A URL is a citation, not a node in the user's own notes."""

    assert extract_links("[docs](https://example.com) and [x](#anchor)") == []


def test_duplicate_links_appear_once():
    assert extract_links("[[A]] [[A]] [[A]]") == ["A"]


def test_no_links_is_an_empty_list():
    assert extract_links("plain text") == []


# --------------------------------------------------------------------------
# Ingesting a file
# --------------------------------------------------------------------------

def test_a_markdown_file_becomes_a_document_and_its_sections(tmp_path, ingester, graph):
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")

    report = ingester.ingest_file(tmp_path / "notes.md")

    assert report.files_ingested == 1
    titles = {node.title for node in graph.nodes(limit=50)}
    assert "notes.md" in titles
    assert any("Speech" in title for title in titles)


def test_sections_are_linked_to_their_document(tmp_path, ingester, graph):
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")

    document = graph.find_by_title(NodeType.FILE, "notes.md")
    edges = graph.edges_from(document.id, type=EdgeType.PART_OF)

    assert len(edges) == 3


def test_a_wikilink_to_an_existing_note_becomes_an_edge(tmp_path, ingester, graph):
    """Ground truth from the user, not a guess from an embedding."""

    ollama = graph.note("Ollama", "the runtime")
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")

    ingester.ingest_file(tmp_path / "notes.md")

    assert graph.edges_to(ollama.id, type=EdgeType.MENTIONS)


def test_a_dangling_wikilink_does_not_invent_a_node(tmp_path, ingester, graph):
    """A link to a note that does not exist yet is normal in a personal wiki."""

    (tmp_path / "notes.md").write_text("# T\n\nSee [[Nothing Here Yet]].", encoding="utf-8")

    ingester.ingest_file(tmp_path / "notes.md")

    assert graph.find_by_title(NodeType.NOTE, "Nothing Here Yet") is None


def test_source_code_is_ingested(tmp_path, ingester, graph):
    (tmp_path / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    report = ingester.ingest_file(tmp_path / "app.py")

    assert report.files_ingested == 1


def test_an_unsupported_file_type_is_skipped_with_a_reason(tmp_path, ingester):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    report = ingester.ingest_file(tmp_path / "image.png")

    assert report.files_ingested == 0
    assert "unsupported" in report.skipped[0]["reason"]


def test_a_binary_file_with_a_text_extension_is_refused(tmp_path, ingester):
    (tmp_path / "data.txt").write_bytes(b"text\x00\x00binary")

    report = ingester.ingest_file(tmp_path / "data.txt")

    assert report.files_ingested == 0
    assert "binary" in report.skipped[0]["reason"]


def test_an_empty_file_is_skipped(tmp_path, ingester):
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    assert ingester.ingest_file(tmp_path / "empty.md").files_ingested == 0


def test_a_missing_file_is_an_error_not_a_crash(tmp_path, ingester):
    report = ingester.ingest_file(tmp_path / "nope.md")

    assert report.errors


# --------------------------------------------------------------------------
# Re-ingesting: the property that keeps the graph usable
# --------------------------------------------------------------------------

def test_ingesting_the_same_file_twice_does_not_duplicate_it(tmp_path, ingester, graph):
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")
    before = len(graph.nodes(limit=100))

    ingester.ingest_file(tmp_path / "notes.md")

    assert len(graph.nodes(limit=100)) == before


def test_an_unchanged_file_creates_no_updates(tmp_path, ingester):
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")

    report = ingester.ingest_file(tmp_path / "notes.md")

    assert report.nodes_created == 0
    assert report.nodes_updated == 0


def test_editing_a_section_updates_the_node_rather_than_orphaning_it(tmp_path, ingester, graph):
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")

    edited = NOTE.replace("Whisper for hearing", "Whisper for hearing, updated")
    (tmp_path / "notes.md").write_text(edited, encoding="utf-8")
    report = ingester.ingest_file(tmp_path / "notes.md")

    assert report.nodes_updated >= 1
    assert report.nodes_created == 0


def test_renaming_a_heading_updates_rather_than_duplicates(tmp_path, ingester, graph):
    """Identity is (file, position), not the title -- or a renamed heading
    leaves an orphan behind and adds a new node beside it."""

    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")
    before = len(graph.nodes(limit=100))

    (tmp_path / "notes.md").write_text(NOTE.replace("## Speech", "## Audio"), encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")

    assert len(graph.nodes(limit=100)) == before
    assert any("Audio" in node.title for node in graph.nodes(limit=100))


def test_a_removed_section_is_removed_from_the_graph(tmp_path, ingester, graph):
    (tmp_path / "notes.md").write_text(NOTE, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")

    shortened = NOTE.split("## Plans")[0]
    (tmp_path / "notes.md").write_text(shortened, encoding="utf-8")
    ingester.ingest_file(tmp_path / "notes.md")

    assert not any("Plans" in node.title for node in graph.nodes(limit=100))


# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------

def test_a_folder_is_walked(tmp_path, ingester):
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\n\nbody", encoding="utf-8")

    report = ingester.ingest_folder(tmp_path)

    assert report.files_ingested == 2


def test_noise_directories_are_never_walked(tmp_path, ingester):
    (tmp_path / "real.md").write_text("# R\n\nbody", encoding="utf-8")
    for name in ("node_modules", "__pycache__", ".git"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "junk.md").write_text("# J\n\nbody", encoding="utf-8")

    report = ingester.ingest_folder(tmp_path)

    assert report.files_ingested == 1


def test_folder_ingestion_respects_a_file_limit(tmp_path, ingester):
    for index in range(20):
        (tmp_path / f"note{index}.md").write_text(f"# N{index}\n\nbody", encoding="utf-8")

    report = ingester.ingest_folder(tmp_path, max_files=5)

    assert report.files_ingested == 5
    assert any("stopped at" in item["reason"] for item in report.skipped)


def test_a_non_recursive_scan_stays_at_the_top(tmp_path, ingester):
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\n\nbody", encoding="utf-8")

    report = ingester.ingest_folder(tmp_path, recursive=False)

    assert report.files_ingested == 1


def test_a_missing_folder_is_an_error(tmp_path, ingester):
    assert ingester.ingest_folder(tmp_path / "nope").errors


# --------------------------------------------------------------------------
# Typed or spoken input
# --------------------------------------------------------------------------

def test_text_can_be_ingested_without_a_file(ingester, graph):
    node = ingester.ingest_text("A thought", "Something worth keeping.")

    assert graph.get(node.id) is not None
    assert node.provenance == "user"


def test_links_in_typed_text_are_connected(ingester, graph):
    graph.note("Ollama", "runtime")

    node = ingester.ingest_text("A thought", "Reminds me of [[Ollama]].")

    assert graph.edges_from(node.id, type=EdgeType.MENTIONS)


# --------------------------------------------------------------------------
# PDFs
# --------------------------------------------------------------------------

def test_an_unreadable_pdf_reports_rather_than_creating_an_empty_node(tmp_path):
    """An empty node is worse than no node: it looks like it worked."""

    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")

    text, note = read_text(bad)

    assert text.strip() == ""
    assert note, "a failure must be explained"


def test_report_serialises_for_the_ui():
    payload = IngestReport(files_seen=3, files_ingested=2, nodes_created=5).to_dict()

    assert payload["files_ingested"] == 2
    assert "skipped" in payload and "errors" in payload
