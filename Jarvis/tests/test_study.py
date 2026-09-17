"""Studium: ingestion keeps every location, retrieval finds the page, the chat opens it, the viewer's question stays on it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from service.events import EventType
from study.index import StudyIndex
from study.intents import parse_study_operation
from study.parsers import parse
from study.pdf_engine import engine as pdf_engine
from study.retrieval import read_query
from study.service import StudyService
from study_fixtures import HEART_PAGES, make_docx, make_pdf, make_pptx
from test_actionability import Provider, make


@pytest.fixture
def material(tmp_path):
    """Three formats about the heart and the basal ganglia, and a note about aldosterone."""

    src = tmp_path / "src" / "Medizin" / "Physiologie" / "3. Semester"
    src.mkdir(parents=True)
    (src / "Physiologie Herz.pdf").write_bytes(make_pdf(HEART_PAGES + [["Stoffwechsel", "Abb. 4: Der Citratzyklus im Mitochondrium"]], image_on={5},
                                                        producer="Goodnotes 6"))
    make_docx(src / "Vorlesung Herz.docx", [("h1", "Herzmechanik"), ("p", "Einleitung zur Herzmechanik."), ("h2", "Vorlast"),
                                            ("p", "Nach dem Frank-Starling-Mechanismus erhöht mehr Vorlast das Schlagvolumen."),
                                            ("h1", "Motorik"), ("p", "Die Basalganglien-Schleife moduliert die Willkürmotorik.")])
    make_pptx(src / "Neuro.pptx", [("Motorik", ["Kortex", "Rückenmark"], ""),
                                   ("Basalganglienschleife", ["direkter Weg", "indirekter Weg"], "Prüfungsrelevant: Dopamin")])
    (src / "Niere.md").write_text("# Niere\n\nWirkung von Aldosteron: Natriumrückresorption im Sammelrohr.\n\n## Renin\n\nRenin spaltet Angiotensinogen.\n",
                                  encoding="utf-8")
    return src


def service(tmp_path, *, engine=None, expander=None) -> StudyService:
    return StudyService(tmp_path / "study", tmp_path / "store", engine=engine, expander=expander, library_root=tmp_path / "library")


def imported(tmp_path, material, **kw) -> StudyService:
    svc = service(tmp_path, **kw)
    for path in sorted(material.iterdir()):
        result = svc.import_path(path, copy=True, root=tmp_path / "src")
        assert result["ok"], result
    return svc


def by_name(svc: StudyService, filename: str) -> dict:
    return next(d for d in svc.documents() if d["filename"] == filename)


# ---------------------------------------------------------------------------
# ingestion and metadata
# ---------------------------------------------------------------------------

def test_a_pdf_keeps_its_pages_images_and_goodnotes_origin(tmp_path, material):
    svc = imported(tmp_path, material)
    doc = by_name(svc, "Physiologie Herz.pdf")
    assert doc["source_type"] == "pdf" and doc["units"] == 5 and doc["title"] == "Physiologie Herz"
    assert doc["flags"]["goodnotes"] is True and doc["provider"] == "goodnotes", "a GoodNotes export is recognised by its producer"
    page3 = svc.index.unit(doc["id"], 3)
    assert page3["kind"] == "page" and "Frank-Starling-Mechanismus" in page3["text"] and page3["width"] == 595 and page3["height"] == 842
    assert svc.index.unit(doc["id"], 5)["images"] == 1 and svc.index.unit(doc["id"], 2)["images"] == 0
    assert Path(doc["stored_path"]).is_file() and Path(doc["stored_path"]).parent == tmp_path / "store"
    assert doc["content_hash"] and doc["created_at"] and doc["updated_at"] and doc["indexed_at"]


def test_word_has_sections_and_paragraphs_but_no_invented_page_numbers(tmp_path, material):
    svc = imported(tmp_path, material)
    doc = by_name(svc, "Vorlesung Herz.docx")
    units = svc.index.units(doc["id"])
    assert [u["kind"] for u in units] == ["section", "section", "section"]
    assert [u["title"] for u in units] == ["Herzmechanik", "Herzmechanik › Vorlast", "Motorik"]
    vorlast = svc.index.unit(doc["id"], 2)
    paragraph = next(b for b in vorlast["blocks"] if "Frank-Starling" in b["text"])
    assert paragraph["heading_path"] == ["Herzmechanik", "Vorlast"] and paragraph["paragraph"] == 4
    hit = svc.search("Frank-Starling-Mechanismus", expand=False)["results"]
    word = next(r for r in hit if r["document"]["id"] == doc["id"])["best"]["location"]
    assert word["page"] is None and word["heading_path"] == ["Herzmechanik", "Vorlast"] and word["paragraph"] == 4
    assert word["label"] == "Herzmechanik › Vorlast · Absatz 4"


def test_slides_come_in_presentation_order_with_titles_and_notes(tmp_path, material):
    svc = imported(tmp_path, material)
    doc = by_name(svc, "Neuro.pptx")
    slide = svc.index.unit(doc["id"], 2)
    assert slide["kind"] == "slide" and slide["title"] == "Basalganglienschleife"
    assert [b["kind"] for b in slide["blocks"]] == ["slide_title", "paragraph", "paragraph", "notes"] and "Dopamin" in slide["text"]


def test_categories_are_inferred_and_an_owner_correction_survives_reindexing(tmp_path, material):
    svc = imported(tmp_path, material)
    doc = by_name(svc, "Physiologie Herz.pdf")
    assert (doc["course"], doc["subject"], doc["semester"]) == ("Medizin", "Physiologie", "3. Semester")
    assert doc["topic"] == "Herz"
    svc.set_categories(doc["id"], {"subject": "Kardiologie", "module": "Kreislauf"})
    svc.index.upsert({**doc, "subject": "Physiologie", "module": ""}, parse(doc["stored_path"]))  # the same document indexed again
    corrected = svc.index.document(doc["id"])
    assert corrected["subject"] == "Kardiologie" and corrected["module"] == "Kreislauf", "re-indexing keeps what the owner corrected"
    assert set(corrected["confirmed"]) >= {"subject", "module"}


def test_a_duplicate_is_recognised_and_removing_material_removes_its_copy(tmp_path, material):
    svc = imported(tmp_path, material)
    data = (material / "Niere.md").read_bytes()
    again = svc.import_bytes("Niere-Kopie.md", data)
    assert again["ok"] and again["duplicate"] is True
    doc = by_name(svc, "Niere.md")
    stored = Path(doc["stored_path"])
    # removing from Studium keeps the file; deleting the file is a separate, explicit choice
    kept = svc.delete(doc["id"])
    assert kept["ok"] and kept["file_kept"] and stored.exists()
    again = svc.import_path(stored, provider="upload")
    assert again["ok"]
    gone = svc.delete(again["document"]["id"], delete_file=True)
    assert gone["ok"] and gone["removed_file"] and not stored.exists()
    assert all(d["id"] != doc["id"] for d in svc.documents())
    assert svc.search("Aldosteron", expand=False)["results"] == []


def test_a_connected_folder_is_indexed_once_and_rescanned_incrementally(tmp_path, material):
    svc = service(tmp_path)
    (material / "Notizbuch.goodnotes").write_bytes(b"not readable")
    source = svc.connect_folder(tmp_path / "src", provider="goodnotes")["source"]
    first = svc.scan_source(source["id"])
    assert first["indexed"] == 4 and first["native_goodnotes"] == 1 and first["failed"] == 0
    second = svc.scan_source(source["id"])
    assert second["indexed"] == 0 and second["unchanged"] == 4, "unchanged files are not read again"
    note = material / "Niere.md"
    note.write_text(note.read_text(encoding="utf-8") + "\n## ADH\n\nADH baut Aquaporine ein.\n", encoding="utf-8")
    later = time.time() + 5
    os.utime(note, (later, later))
    third = svc.scan_source(source["id"])
    assert third["indexed"] == 1 and svc.search("Aquaporine", expand=False)["results"], "a changed file is read again"
    assert all(Path(d["stored_path"]).parent != tmp_path / "store" for d in svc.documents()), "connected files are indexed in place, not copied"


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def test_the_topic_is_read_out_of_the_command():
    assert read_query("Zeig mir die Seite zum Frank-Starling-Mechanismus.").topic == "Frank-Starling-Mechanismus"
    assert read_query("Wo habe ich etwas zum Plexus brachialis?").topic == "Plexus brachialis"
    assert read_query("Was habe ich zur Wirkung von Aldosteron notiert?").topic == "Wirkung von Aldosteron"
    assert read_query("Öffne genau die Seite mit der Abbildung zum Citratzyklus").topic == "Citratzyklus"
    figure = read_query("Zeig mir genau die Seite mit der Abbildung zum Citratzyklus.")
    assert figure.figure and figure.page
    assert read_query("Öffne das Skript von gestern").since_days == 2


def test_asking_for_the_page_finds_the_exact_page(tmp_path, material):
    svc = imported(tmp_path, material)
    found = svc.search("Zeig mir die Seite zum Frank-Starling-Mechanismus.", expand=False)
    best = found["results"][0]
    assert best["document"]["filename"] == "Physiologie Herz.pdf" and best["best"]["location"]["page"] == 3
    assert best["best"]["location"]["label"] == "Seite 3" and best["best"]["exact"] and best["best"]["focus"] == "Frank-Starling-Mechanismus"
    excerpt, spans = best["best"]["excerpt"], best["best"]["highlights"]
    assert any(excerpt[a:b] == "Frank-Starling-Mechanismus" for a, b in spans)
    assert {r["document"]["filename"] for r in found["results"]} >= {"Physiologie Herz.pdf", "Vorlesung Herz.docx"}


def test_a_compound_is_found_however_it_is_hyphenated_across_all_material(tmp_path, material):
    svc = imported(tmp_path, material)
    found = svc.search("Wo habe ich etwas zur Basalganglienschleife?", expand=False)["results"]
    names = [r["document"]["filename"] for r in found]
    assert set(names[:2]) == {"Neuro.pptx", "Vorlesung Herz.docx"}
    slide = next(r for r in found if r["document"]["filename"] == "Neuro.pptx")["best"]
    word = next(r for r in found if r["document"]["filename"] == "Vorlesung Herz.docx")["best"]
    assert slide["location"]["slide"] == 2 and word["focus"] == "Basalganglien-Schleife", "the written spelling is what the viewer marks"


def test_a_figure_question_prefers_the_page_with_the_figure(tmp_path, material):
    svc = imported(tmp_path, material)
    (material / "Zusammenfassung.md").write_text("# Stoffwechsel\n\nDer Citratzyklus liefert NADH.\n", encoding="utf-8")
    svc.import_path(material / "Zusammenfassung.md", copy=True)
    found = svc.search("Öffne genau die Seite mit der Abbildung zum Citratzyklus.", expand=False)["results"]
    assert found[0]["document"]["filename"] == "Physiologie Herz.pdf" and found[0]["best"]["location"]["page"] == 5


def test_results_are_ranked_by_relevance_not_by_file_type(tmp_path, material):
    svc = imported(tmp_path, material)
    found = svc.search("Renin Angiotensinogen", expand=False)["results"]
    assert found[0]["document"]["filename"] == "Niere.md" and found[0]["relevance"] == 1.0
    assert all(r["relevance"] <= 1.0 for r in found)


def test_expansion_is_asked_only_when_the_exact_words_do_not_find_the_topic(tmp_path, material):
    calls: list[str] = []

    def expander(topic):
        calls.append(topic)
        return ["Starling-Gesetz", "Vorlast"]

    svc = imported(tmp_path, material, expander=expander)
    svc.search("Frank-Starling-Mechanismus")
    assert calls == [], "an exact hit needs no synonyms"
    widened = svc.search("Wo steht etwas zur Vorlastabhängigkeit?")
    assert len(calls) == 1, "a topic the exact words do not find asks for its other names"
    assert widened["results"], "the related terms find the material"
    svc.search("Wo steht etwas zur Vorlastabhängigkeit?")
    assert len(calls) == 1, "expansions are cached per topic"


# ---------------------------------------------------------------------------
# the page itself: highlight rectangles, page images, context
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qt_engine():
    engine = pdf_engine()
    if not engine.available:
        pytest.skip("the PDF page engine (PySide6 QtPdf) is not available")
    return engine


def test_the_passage_is_located_on_the_page_and_the_page_renders(tmp_path, material, qt_engine):
    svc = imported(tmp_path, material, engine=qt_engine)
    doc = by_name(svc, "Physiologie Herz.pdf")
    assert doc["flags"]["text_engine"] == "qtpdf"
    marked = svc.locate(doc["id"], 3, ["Frank-Starling-Mechanismus"])
    assert marked["ok"] and marked["matched"] == ["Frank-Starling-Mechanismus"] and len(marked["rects"]) == 1
    x, y, w, h = marked["rects"][0]
    assert 0 < x < 0.5 and 0 < y < 0.2 and 0.1 < w < 0.6 and 0 < h < 0.05, "the first line of the page, as fractions of the page"
    assert svc.locate(doc["id"], 2, ["Frank-Starling-Mechanismus"])["rects"] == [], "never marked on a page that does not contain it"
    image = svc.page_image(doc["id"], 3, width=800)
    assert image is not None and image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert svc.page_image(doc["id"], 3, width=800) == image, "rendered once, then cached"


def test_pages_render_sharp_in_regions_and_skip_what_the_viewer_moved_past(tmp_path, material, qt_engine):
    """Zoomed in, only a region is rendered at the high scale; a render older than the viewer's newest view is dropped."""

    from PIL import Image

    svc = imported(tmp_path, material, engine=qt_engine)
    doc = by_name(svc, "Physiologie Herz.pdf")
    full = svc.render_page(doc["id"], 3, width=1000, session="v1", seq=1)
    assert full["path"] is not None and not full.get("cached")
    assert svc.render_page(doc["id"], 3, width=1000, session="v1", seq=1)["cached"], "the same page again comes from the cache"
    tile = svc.render_page(doc["id"], 3, width=4000, clip=[0.25, 0.0, 0.25, 0.2], session="v1", seq=1)
    with Image.open(tile["path"]) as region:
        assert region.size == (1000, int(4000 * 842 / 595 * 0.2)) or abs(region.size[1] - 4000 * 842 / 595 * 0.2) <= 2, region.size
    svc.render_page(doc["id"], 4, width=1000, session="v1", seq=5)
    skipped = svc.render_page(doc["id"], 2, width=1000, session="v1", seq=3)
    assert skipped.get("superseded") and skipped["path"] is None, "the viewer is already two pages further"
    assert svc.render_page(doc["id"], 2, width=1000, session="v1", seq=4, priority=1)["path"] is not None, "a neighbour fetched ahead still renders"
    assert svc.render_page(doc["id"], 2, width=1200, session="other", seq=1)["path"] is not None, "another viewer is not affected"
    assert svc.render_page(doc["id"], 3, width=900000)["path"].name.startswith("3_3200_"), "never a giant bitmap"


def test_the_context_of_a_place_is_that_place_or_only_the_selection(tmp_path, material):
    svc = imported(tmp_path, material)
    doc = by_name(svc, "Physiologie Herz.pdf")
    page = svc.context(doc["id"], 3)
    assert page["scope"] == "page" and "Frank-Starling" in page["text"] and "Nachlast" not in page["text"] and page["location"]["label"] == "Seite 3"
    selection = svc.context(doc["id"], 3, selection="Dadurch steigt das Schlagvolumen.")
    assert selection["scope"] == "selection" and selection["text"] == "Dadurch steigt das Schlagvolumen."


def test_study_material_stays_out_of_personal_memory_and_the_knowledge_graph(tmp_path, material):
    core, _ = make(tmp_path)
    before = core.knowledge_stats().get("nodes", 0)
    for path in sorted(material.iterdir()):
        assert core.study.import_path(path, copy=True)["ok"]
    assert core.knowledge_stats().get("nodes", 0) == before, "study knowledge has its own namespace"
    assert core.study.index.path.parent == Path(core.kernel.state_root) / "study"


# ---------------------------------------------------------------------------
# commands in the conversation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence, operation, target", [
    ("Zeig mir die Seite über den Frank-Starling-Mechanismus.", "study.locate", "Frank-Starling-Mechanismus"),
    ("Wo habe ich etwas zum Plexus brachialis?", "study.search", "Plexus brachialis"),
    ("Öffne meine Notizen zur Basalganglienschleife.", "study.locate", "Basalganglienschleife"),
    ("Zeig mir genau die Seite mit der Abbildung zum Citratzyklus.", "study.locate", "Citratzyklus"),
    ("Was habe ich zur Wirkung von Aldosteron notiert?", "study.answer", "Wirkung von Aldosteron"),
    ("Öffne das Skript von gestern.", "study.open_recent", "skript"),
    ("Wo stand nochmal die Henderson-Hasselbalch-Gleichung?", "study.search", "Henderson-Hasselbalch-Gleichung"),
    ("Welche Seite war das?", "study.which_page", ""),
    ("Zeig mir die Abbildung.", "study.figure", ""),
    ("Fass diese Seite zusammen.", "study.summarize", ""),
    ("Prüf mich aus diesem Kapitel.", "study.quiz", ""),
    ("Mach daraus 20 schwere MC-Fragen.", "study.quiz", ""),
    ("Erklär mir nur den markierten Abschnitt.", "study.explain", ""),
    ("Vergleiche das mit meinen anderen Notizen.", "study.compare", ""),
    ("Vergleiche meine Vorlesungsnotizen mit dem Skript.", "study.compare", ""),
    ("Was fehlt meinen Notizen im Vergleich zum Skript?", "study.missing", ""),
    ("Welche Themen fehlen mir für diese Vorlesung?", "study.missing", ""),
])
def test_study_sentences_become_typed_actions(sentence, operation, target):
    action = parse_study_operation(sentence)
    assert action is not None and action.operation == operation and action.target == target and action.object_type == "study"


@pytest.mark.parametrize("sentence", ["Zeig mir die Seite von Amazon", "Öffne die Datei notiz.txt", "Wie spät ist es?", "Zeig mir meine Projekte",
                                      "Öffne Spotify", "Erklär mir den Citratzyklus", "Öffne die Webseite zum Citratzyklus"])
def test_other_sentences_are_not_study_commands(sentence):
    assert parse_study_operation(sentence) is None


def test_mc_questions_carry_their_count_and_difficulty():
    action = parse_study_operation("Mach daraus 20 schwere MC-Fragen.")
    assert action.arguments["count"] == 20 and action.arguments["difficulty"] == "schwer" and action.arguments["multiple_choice"]


def drain(core, text, *, meta=None, wait=20.0, until=EventType.MESSAGE):
    with core.bus.subscribe(replay=False) as sub:
        core.send_message(text, meta=meta)
        deadline = time.time() + wait
        events = []
        while time.time() < deadline:
            events.extend(sub.drain())
            if any(e.type is until for e in events):
                time.sleep(0.2)
                events.extend(sub.drain())
                break
            time.sleep(0.05)
    return events


class Recording(Provider):
    def __init__(self, answer="Laut deinem Skript (Physiologie Herz.pdf · Seite 3) steigt das Schlagvolumen."):
        super().__init__()
        self.systems: list[str] = []
        self.answer = answer

    def generate_stream(self, prompt, **kw):
        self.systems.append(kw.get("system", ""))
        self.prompts.append(prompt)
        yield self.answer


def test_zeig_mir_die_seite_opens_the_viewer_at_the_page_without_a_model(tmp_path, material):
    provider = Recording()
    core, _ = make(tmp_path, provider=provider)
    for path in sorted(material.iterdir()):
        core.study.import_path(path, copy=True)
    events = drain(core, "Zeig mir die Seite zum Frank-Starling-Mechanismus.")
    opened = [e.payload for e in events if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "open_view"]
    assert opened and opened[-1]["view"] == "study"
    doc = by_name(core.study, "Physiologie Herz.pdf")
    params = opened[-1]["params"]
    assert {k: params[k] for k in ("doc", "unit", "focus")} == {"doc": doc["id"], "unit": "3", "focus": "Frank-Starling-Mechanismus"}
    unit_text = core.study.index.unit(doc["id"], 3)["text"]
    assert unit_text[int(params["at"]):].startswith("Frank-Starling-Mechanismus"), "the viewer is told where on the page the hit is"
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "Physiologie Herz.pdf · Seite 3."
    sources = message["meta"]["study_sources"]
    assert sources[0]["document_id"] == doc["id"] and sources[0]["location"]["page"] == 3
    assert provider.prompts == [], "finding a page asks no model"
    later = drain(core, "Welche Seite war das?")
    assert next(e.payload for e in later if e.type is EventType.MESSAGE)["text"] == "Das war Physiologie Herz.pdf, Seite 3."


def test_without_material_study_words_are_left_to_the_conversation(tmp_path):
    provider = Recording("Hier ist keine Studienaktion.")
    core, _ = make(tmp_path, provider=provider)
    events = drain(core, "Wo stand nochmal die Adresse vom Bäcker?")
    assert not any(e.type is EventType.TOOL and str(e.payload.get("summary", "")).startswith("study:") for e in events)


def test_an_answer_from_the_notes_uses_the_material_and_cites_it(tmp_path, material):
    provider = Recording()
    core, _ = make(tmp_path, provider=provider)
    for path in sorted(material.iterdir()):
        core.study.import_path(path, copy=True)
    events = drain(core, "Was habe ich zur Wirkung von Aldosteron notiert?")
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    system = "\n".join(provider.systems)
    assert "MATERIAL AUS DEN UNTERLAGEN DES BESITZERS" in system and "Natriumrückresorption" in system and "Niere.md" in system
    assert message["meta"]["study_sources"][0]["filename"] == "Niere.md"


def test_ask_zeus_from_the_viewer_uses_only_the_open_page(tmp_path, material):
    provider = Recording("Dieser Abschnitt beschreibt die Vorlast.")
    core, _ = make(tmp_path, provider=provider)
    for path in sorted(material.iterdir()):
        core.study.import_path(path, copy=True)
    doc = by_name(core.study, "Physiologie Herz.pdf")
    events = drain(core, "Erklär mir nur diesen Abschnitt.", meta={"source": "text", "study_context": {"document_id": doc["id"], "unit": 3, "selection": ""}})
    system = "\n".join(provider.systems)
    assert "Eine erhöhte Vorlast dehnt die Sarkomere" in system and "Sympathikus steigert die Inotropie" not in system, "page 4 is not the question's source"
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["meta"]["study_sources"][0]["location"]["label"] == "Seite 3"
    provider.systems.clear()
    drain(core, "Was bedeutet das?", meta={"source": "text", "study_context": {"document_id": doc["id"], "unit": 3, "selection": "Dadurch steigt das Schlagvolumen."}})
    system = "\n".join(provider.systems)
    assert "Dadurch steigt das Schlagvolumen." in system and "Eine erhöhte Vorlast dehnt" not in system, "only the selection"


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

def test_study_routes_upload_search_unit_and_page_image(tmp_path, material):
    from service.http import JarvisHTTPServer

    core, _ = make(tmp_path)
    server = JarvisHTTPServer(core, port=0, token="tok")
    server.start()
    base = f"http://{server.host}:{server.port}"

    def post(path, payload=None, raw=None, content_type="application/json"):
        data = raw if raw is not None else json.dumps(payload or {}).encode()
        request = urllib.request.Request(base + path, data=data, method="POST", headers={"X-Jarvis-Token": "tok", "Content-Type": content_type})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode())

    try:
        uploaded = post("/api/study/upload?name=Physiologie%20Herz.pdf", raw=(material / "Physiologie Herz.pdf").read_bytes(),
                        content_type="application/octet-stream")
        assert uploaded["ok"] and uploaded["document"]["units"] == 5
        doc_id = uploaded["document"]["id"]
        found = post("/api/study/search", {"query": "Zeig mir die Seite zum Frank-Starling-Mechanismus"})
        assert found["results"][0]["best"]["location"]["page"] == 3
        unit = post("/api/study/unit", {"id": doc_id, "unit": 3})
        assert unit["ok"] and unit["unit"]["number"] == 3
        listed = post("/api/study/documents", {})
        assert listed["documents"][0]["id"] == doc_id and listed["documents"][0]["opened_at"], "opening a unit marks the document as used"
        status = post("/api/study/status", {})
        assert status["stats"]["documents"] == 1 and "ocr" in status
        refused = urllib.request.Request(f"{base}/api/study/original?id={doc_id}")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(refused, timeout=10)
        with urllib.request.urlopen(f"{base}/api/study/original?id={doc_id}&token=tok", timeout=30) as response:
            assert response.headers["Content-Type"] == "application/pdf" and response.read()[:5] == b"%PDF-"
        if core.study.engine is not None and core.study.engine.available:
            with urllib.request.urlopen(f"{base}/api/study/page.png?id={doc_id}&page=3&w=700&token=tok", timeout=60) as response:
                assert response.headers["Content-Type"] == "image/png" and response.read()[:4] == b"\x89PNG"
        assert post("/api/media/control", {"action": "shuffle"})["ok"] is False
    finally:
        server.stop()


def test_the_presence_ribbon_and_its_controller(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, str(root / "ui" / "tests" / "presence.test.mjs")], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0 and "ALL OK" in completed.stdout, completed.stdout[-2000:] + completed.stderr[-500:]


def test_the_index_file_is_its_own_store(tmp_path):
    index = StudyIndex(tmp_path / "x" / "index.sqlite")
    assert index.stats() == {"documents": 0, "units": 0, "chunks": 0, "by_type": {}}
    index.close()
