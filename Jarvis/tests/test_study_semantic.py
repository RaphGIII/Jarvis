"""Studium polish: hybrid retrieval (words + meaning + metadata), every match highlighted, Word sections, slides, import states."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from service.study_actions import compact, view_params
from study import embeddings as emb
from study.highlight import arrange, merge_line_rects, overlap
from study.retrieval import read_query
from study.service import StudyService
from study_fixtures import HEART_PAGES, make_docx, make_pdf, make_pptx

# ---------------------------------------------------------------------------------------------------------------
# a deterministic embedding provider: concepts, not words -- "zurückströmen" and "Vorlast" mean the same thing here
# ---------------------------------------------------------------------------------------------------------------

CONCEPTS = [
    ("filling", ("vorlast", "füllung", "zurückström", "rückstrom", "vordehnung", "dehnt", "frank-starling", "mehr blut")),
    ("output", ("schlagvolumen", "auswurf", "pumpt", "wirft")),
    ("pressure", ("blutdruck", "barorezeptor", "druck")),
    ("energy", ("citratzyklus", "atp", "mitochondri", "energie")),
    ("movement", ("basalganglien", "bewegung", "motorik", "striatum")),
    ("kidney", ("niere", "aldosteron", "natrium", "salz")),
]


class ConceptEmbeddings:
    """Vectors from concept counts over a shared base, so cosine similarities sit in the compressed band real models produce."""

    local = True

    def __init__(self, ident: str = "fake:concepts") -> None:
        self.id = ident
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        import re
        import zlib

        lower = text.lower()
        buckets = 24
        v = np.zeros(1 + len(CONCEPTS) + buckets, dtype=np.float32)
        v[0] = 1.0
        for i, (_name, words) in enumerate(CONCEPTS, start=1):
            v[i] = 1.5 if any(w in lower for w in words) else 0.0
        for token in re.findall(r"\w{3,}", lower):  # the words themselves: different texts are never identical vectors
            v[1 + len(CONCEPTS) + zlib.crc32(token.encode("utf-8")) % buckets] += 0.3
        return v / np.linalg.norm(v)

    def available(self) -> bool:
        return True

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def status(self) -> dict:
        return {"provider": "fake", "id": self.id, "available": True, "local": True}

    def close(self) -> None:
        return None


@pytest.fixture
def library(tmp_path):
    src = tmp_path / "src" / "Medizin" / "Physiologie"
    src.mkdir(parents=True)
    pages = HEART_PAGES + [["Kreislauf", "Barorezeptoren im Aortenbogen melden den Blutdruck."],
                           ["Wiederholung", "Der Frank-Starling-Mechanismus: mehr Vorlast, mehr Schlagvolumen.",
                            "Merke: Frank-Starling-Mechanismus gilt für beide Ventrikel."]]
    (src / "Physiologie Herz.pdf").write_bytes(make_pdf(pages))
    make_docx(src / "Vorlesung Herz.docx", [("h1", "Herzmechanik"), ("p", "Einleitung zur Herzmechanik."), ("h2", "Vorlast"),
                                            ("p", "Die Vorlast ist die enddiastolische Füllung des Ventrikels."),
                                            ("p", "Nach dem Frank-Starling-Mechanismus erhöht mehr Vorlast das Schlagvolumen."),
                                            ("h1", "Niere"), ("p", "Aldosteron fördert die Natriumrückresorption.")])
    make_pptx(src / "Neuro.pptx", [("Motorik", ["Kortex", "Rückenmark"], ""),
                                   ("Basalganglienschleife", ["direkter Weg", "indirekter Weg"], "Prüfungsrelevant")])
    (src / "Stoffwechsel.md").write_text("# Stoffwechsel\n\nDer Citratzyklus liefert Reduktionsäquivalente für die ATP-Bildung.\n", encoding="utf-8")
    return src


def make_service(tmp_path, library, *, embeddings=None, **kw) -> StudyService:
    events: list[tuple[str, dict]] = []
    svc = StudyService(tmp_path / "study", tmp_path / "store", embeddings=embeddings if embeddings is not None else False,
                       slide_renderer=kw.pop("slide_renderer", False), emit=lambda k, p: events.append((k, p)), **kw)
    svc.events = events  # type: ignore[attr-defined]
    for path in sorted(library.iterdir()):
        assert svc.import_path(path, copy=True, root=library.parent)["ok"]
    return svc


def by_name(svc, filename):
    return next(d for d in svc.documents() if d["filename"] == filename)


# ---------------------------------------------------------------------------------------------------------------
# hybrid retrieval
# ---------------------------------------------------------------------------------------------------------------

def test_embeddings_are_stored_per_provider_and_dropped_on_reindex(tmp_path, library):
    provider = ConceptEmbeddings()
    svc = make_service(tmp_path, library, embeddings=provider)
    report = svc.embed_pending()
    assert report["documents"] == 4 and report["chunks"] > 0 and not report["error"]
    stats = svc.index.embedding_stats(provider.id)
    assert stats["embedded_documents"] == 4 and stats["vectors"] == report["chunks"]
    assert svc.embed_pending()["documents"] == 0, "nothing is embedded twice"
    # another provider id: its own vectors, never mixed with the old ones
    svc._embeddings = ConceptEmbeddings("fake:other")
    assert len(svc.index.unembedded("fake:other")) == 4
    # a re-index gives chunks new ids: their old vectors are gone, the document is pending again
    pdf = by_name(svc, "Physiologie Herz.pdf")
    folder_doc = svc.import_path(Path(pdf["stored_path"]), provider="folder", source_id="s1")["document"]
    svc._embeddings = provider
    svc.embed_pending()
    assert folder_doc["id"] not in {d["id"] for d in svc.index.unembedded(provider.id)}
    before = len(svc.index.vectors(provider.id)[0])
    svc.import_path(Path(pdf["stored_path"]), provider="folder", source_id="s1")  # the same file again: same document, new chunks
    assert folder_doc["id"] in {d["id"] for d in svc.index.unembedded(provider.id)}
    assert len(svc.index.vectors(provider.id)[0]) < before, "the re-indexed document's old vectors are gone"
    rowids, vectors = svc.index.vectors(provider.id)
    alive = {r["rowid"] for d in svc.documents() for r in svc.index.chunks_of(d["id"])}
    assert set(rowids) <= alive and vectors.shape[0] == len(rowids)


def test_a_paraphrase_without_the_key_term_finds_the_place_by_meaning(tmp_path, library):
    svc = make_service(tmp_path, library, embeddings=ConceptEmbeddings())
    svc.embed_pending()
    keyword_only = svc.search("Warum pumpt der Ventrikel mehr, wenn mehr Blut zurückströmt?", semantic=False)
    hybrid = svc.search("Warum pumpt der Ventrikel mehr, wenn mehr Blut zurückströmt?")
    assert hybrid["semantic"] is True
    top = hybrid["results"][0]
    assert top["document"]["filename"] in {"Physiologie Herz.pdf", "Vorlesung Herz.docx"}
    where = top["best"]["location"]
    assert where["page"] in {3, 6} or where["heading_path"][-1:] == ["Vorlast"], where
    ranked_keyword = [r["document"]["filename"] for r in keyword_only["results"]]
    assert "Stoffwechsel.md" not in [r["document"]["filename"] for r in hybrid["results"][:2]]
    assert ranked_keyword != [r["document"]["filename"] for r in hybrid["results"]] or top["best"]["similarity"] is not None


def test_exact_phrase_and_file_name_still_rank_first(tmp_path, library):
    svc = make_service(tmp_path, library, embeddings=ConceptEmbeddings())
    svc.embed_pending()
    result = svc.search("Zeig mir die Seite zum Frank-Starling-Mechanismus")
    best = result["results"][0]["best"]
    assert best["exact"] is True and best["match"] == "phrase"
    assert result["results"][0]["document"]["filename"] == "Physiologie Herz.pdf"
    named = svc.search("Stoffwechsel")
    assert named["results"][0]["document"]["filename"] == "Stoffwechsel.md", "a file named after the topic ranks first"


def test_nothing_is_invented_for_an_off_topic_question(tmp_path, library):
    svc = make_service(tmp_path, library, embeddings=ConceptEmbeddings())
    svc.embed_pending()
    assert svc.search("Rezept für Apfelkuchen")["results"] == []
    assert svc.search("Quantenverschränkung")["results"] == []


def test_a_place_found_only_by_meaning_is_marked_as_such_with_its_sentence(tmp_path, library):
    svc = make_service(tmp_path, library, embeddings=ConceptEmbeddings())
    svc.embed_pending()
    result = svc.search("Salzhaushalt")  # no document says "Salzhaushalt"
    assert result["results"], result
    best = result["results"][0]["best"]
    assert best["match"] == "semantic" and "Aldosteron" in best["focus"] and best["phrases"] == [best["focus"]]


def test_metadata_filters_and_type_hints(tmp_path, library):
    svc = make_service(tmp_path, library, embeddings=ConceptEmbeddings())
    svc.embed_pending()
    only_slides = svc.search("Motorik", filters={"source_type": ["pptx"]})
    assert {r["document"]["source_type"] for r in only_slides["results"]} == {"pptx"}
    assert svc.search("Frank-Starling-Mechanismus", filters={"subject": "Anatomie"})["results"] == []
    assert svc.search("Frank-Starling-Mechanismus", filters={"subject": "Physiologie"})["results"]
    assert read_query("Wo steht in meinen Folien etwas zur Basalganglienschleife?").types == {"pptx"}
    assert read_query("Öffne meine Notizen zur Basalganglienschleife").notes is True
    assert read_query("Zeig mir die Seite über Vorlast").terms == ["vorlast"], "umlaut stop words are folded like the text"


def test_embeddings_never_leave_the_machine_without_permission():
    remote = emb.create_provider({"provider": "ollama", "model": "bge-m3", "url": "https://embeddings.example.com"})
    assert isinstance(remote, emb.NoEmbeddings) and "allow_external" in remote.reason
    local = emb.create_provider({"provider": "ollama", "model": "bge-m3", "url": "http://127.0.0.1:11434"})
    assert isinstance(local, emb.OllamaEmbeddings) and local.local
    allowed = emb.create_provider({"provider": "ollama", "model": "bge-m3", "url": "https://embeddings.example.com", "allow_external": True})
    assert isinstance(allowed, emb.OllamaEmbeddings) and not allowed.local
    assert isinstance(emb.create_provider({"provider": "none"}), emb.NoEmbeddings)
    onnx = emb.create_provider({"provider": "onnx", "model": "m", "model_dir": "Z:/nowhere"})
    assert onnx.local and not onnx.available() and "Modell fehlt" in onnx.status()["reason"]


def test_the_configured_default_is_a_local_model():
    config = emb.load_config()
    assert config["provider"] == "onnx" and config["allow_external"] is False and "e5" in config["model"]


# ---------------------------------------------------------------------------------------------------------------
# every match on a page
# ---------------------------------------------------------------------------------------------------------------

def test_runs_on_a_line_merge_and_duplicates_are_not_drawn_twice():
    runs = [[200, 67, 36, 12], [236, 68, 13, 11], [249, 67, 143, 15]]  # "Frank" "-" "Starling-Mechanismus" as QtPdf reports them
    assert merge_line_rects(runs) == [[200.0, 67.0, 192.0, 15.0]]
    assert len(merge_line_rects([[10, 10, 50, 10], [10, 40, 50, 10]])) == 2, "two lines stay two rectangles"
    matches = [{"phrase": "Frank-Starling-Mechanismus", "rank": 0, "start": 10, "rects": runs},
               {"phrase": "Frank-Starling-Mechanismus", "rank": 0, "start": 400, "rects": [[72, 300, 190, 14]]},
               {"phrase": "Vorlast", "rank": 1, "start": 60, "rects": [[300, 67, 50, 15]]},
               {"phrase": "Starling", "rank": 2, "start": 16, "rects": [[240, 67, 60, 15]]}]
    arranged = arrange(matches, at=390)
    assert [m["primary"] for m in arranged].count(True) == 1
    primary = next(m for m in arranged if m["primary"])
    assert primary["start"] == 400, "the occurrence the result pointed at is primary"
    assert all(m["phrase"] != "Starling" for m in arranged), "a word inside an already marked phrase is not drawn again"
    assert overlap([0, 0, 10, 10], [2, 2, 4, 4]) == 1.0


@pytest.mark.skipif(not __import__("importlib").util.find_spec("PySide6"), reason="PySide6 (QtPdf) not installed")
def test_every_occurrence_on_a_pdf_page_is_located_with_one_primary(tmp_path, library):
    from study.pdf_engine import PdfEngine

    engine = PdfEngine()
    try:
        svc = make_service(tmp_path, library, engine=engine)
        pdf = by_name(svc, "Physiologie Herz.pdf")
        found = svc.search("Frank-Starling-Mechanismus")
        hit = found["results"][0]["best"]
        page = hit["location"]["page"]
        answer = svc.locate(pdf["id"], page, hit["phrases"], at=hit["at"])
        assert answer["ok"] and answer["matches"], answer
        occurrences = [m for m in answer["matches"] if m["phrase"] == hit["focus"]]
        unit_text = svc.index.unit(pdf["id"], page)["text"]
        assert len(occurrences) == unit_text.count("Frank-Starling-Mechanismus") >= 1
        assert sum(m["primary"] for m in answer["matches"]) == 1
        for match in answer["matches"]:
            for x, y, w, h in match["rects"]:
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        repeated = svc.locate(pdf["id"], 6, ["Frank-Starling-Mechanismus"])
        assert len(repeated["matches"]) == 2, "both mentions on the repetition page are marked"
    finally:
        engine.close()


def test_scanned_pages_are_marked_from_their_ocr_word_boxes(tmp_path, library):
    svc = make_service(tmp_path, library)
    pdf = by_name(svc, "Physiologie Herz.pdf")
    unit = svc.index.unit(pdf["id"], 1)
    svc.index._db.execute("UPDATE units SET ocr=1 WHERE document_id=? AND number=1", (pdf["id"],))
    text = "Basalganglienschleife fördert Bewegung\nBasalganglienschleife"
    words = {"text": text, "words": [{"text": "Basalganglienschleife", "start": 0, "end": 21, "box": [0.1, 0.1, 0.3, 0.05]},
                                     {"text": "fördert", "start": 22, "end": 29, "box": [0.42, 0.1, 0.1, 0.05]},
                                     {"text": "Bewegung", "start": 30, "end": 38, "box": [0.54, 0.1, 0.12, 0.05]},
                                     {"text": "Basalganglienschleife", "start": 39, "end": 60, "box": [0.1, 0.2, 0.3, 0.05]}]}
    cache = tmp_path / "study" / "cache" / "ocr" / pdf["id"]
    cache.mkdir(parents=True)
    (cache / "ocr_1.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    answer = svc.locate(pdf["id"], 1, ["Basalganglienschleife", "fördert"], at=39)
    assert unit is not None and answer["ok"]
    focus = [m for m in answer["matches"] if m["phrase"] == "Basalganglienschleife"]
    assert len(focus) == 2 and next(m for m in answer["matches"] if m["primary"])["start"] == 39
    assert any(m["phrase"] == "fördert" for m in answer["matches"])


# ---------------------------------------------------------------------------------------------------------------
# Word: sections, headings, paragraphs -- never page numbers
# ---------------------------------------------------------------------------------------------------------------

def test_word_results_open_the_exact_section_and_paragraph(tmp_path, library):
    svc = make_service(tmp_path, library)
    found = svc.search("Frank-Starling-Mechanismus", filters={"source_type": "docx"})
    result = found["results"][0]
    location = result["best"]["location"]
    assert location["page"] is None and location["heading_path"] == ["Herzmechanik", "Vorlast"] and location["paragraph"] == 5
    params = view_params(compact(result))
    assert params["para"] == "5" and params["unit"] == str(location["unit"]) and params["focus"] == "Frank-Starling-Mechanismus"
    sections = svc.sections(result["document"]["id"])
    assert sections["ok"] and sections["complete"]
    titles = [u["title"] for u in sections["units"]]
    assert titles == ["Herzmechanik", "Herzmechanik › Vorlast", "Niere"]
    paragraphs = [b["paragraph"] for u in sections["units"] for b in u["blocks"]]
    assert paragraphs == sorted(paragraphs) and 5 in paragraphs, "paragraphs are numbered through the whole document"
    vorlast = sections["units"][1]
    assert any(b["paragraph"] == 5 and "Frank-Starling" in b["text"] for b in vorlast["blocks"])
    assert svc.document(result["document"]["id"])["viewer"] == "sections"


# ---------------------------------------------------------------------------------------------------------------
# PowerPoint: the exact slide, as an image when PowerPoint can draw it, the shape marked
# ---------------------------------------------------------------------------------------------------------------

class FakeSlides:
    def __init__(self, *, deferred: bool = False) -> None:
        self.deferred = deferred
        self.calls = 0

    def available(self) -> bool:
        return True

    def render_slides(self, path, out_dir, *, width=1600, timeout=120.0):
        self.calls += 1
        if self.deferred:
            return {"ok": False, "deferred": True, "error": "PowerPoint ist gerade geöffnet"}
        out = Path(out_dir)
        count = len([n for n in zipfile.ZipFile(path).namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")])
        files = []
        for n in range(1, count + 1):
            (out / f"slide_{n}.png").write_bytes(b"\x89PNG fake")
            files.append(str(out / f"slide_{n}.png"))
        return {"ok": True, "slides": files}


def test_slides_open_at_the_slide_with_an_image_and_the_shape_marked(tmp_path, library):
    renderer = FakeSlides()
    svc = make_service(tmp_path, library, slide_renderer=renderer)
    found = svc.search("Basalganglienschleife")
    result = next(r for r in found["results"] if r["document"]["source_type"] == "pptx")
    assert result["best"]["location"]["slide"] == 2 and result["best"]["location"]["label"] == "Folie 2"
    params = view_params(compact(result))
    assert params["unit"] == "2" and "para" not in params
    doc = result["document"]
    info = svc.document(doc["id"])
    assert info["viewer"] == "slides" and info["slides"]["state"] == "pending"
    assert svc.render_slides_now(doc["id"])["ok"] and renderer.calls == 1
    assert svc.document(doc["id"])["slides"]["state"] == "ready"
    image = svc.page_image(doc["id"], 2)
    assert image is not None and image.name == "slide_2.png"
    assert svc.render_slides_now(doc["id"]).get("cached") and renderer.calls == 1, "rendered once, then cached"
    # the fixture's shapes carry no positions: nothing is invented, no rectangle drawn
    assert svc.locate(doc["id"], 2, ["Basalganglienschleife"])["matches"] == []
    unit = svc.index.unit(doc["id"], 2)
    blocks = unit["blocks"]
    blocks[0]["box"] = [0.07, 0.05, 0.86, 0.19]
    svc.index._db.execute("UPDATE units SET blocks=? WHERE document_id=? AND number=2", (json.dumps(blocks), doc["id"]))
    marked = svc.locate(doc["id"], 2, ["Basalganglienschleife"])
    assert marked["matches"] and marked["matches"][0]["rects"] == [[0.07, 0.05, 0.86, 0.19]] and marked["matches"][0]["primary"]


def test_slides_wait_while_powerpoint_is_open_and_fall_back_without_it(tmp_path, library):
    svc = make_service(tmp_path, library, slide_renderer=FakeSlides(deferred=True))
    doc = by_name(svc, "Neuro.pptx")
    assert svc.render_slides_now(doc["id"]).get("deferred")
    assert svc.document(doc["id"])["slides"]["state"] == "deferred"
    plain = make_service(tmp_path / "plain", library)
    assert plain.document(by_name(plain, "Neuro.pptx")["id"])["slides"]["state"] == "unavailable"


# ---------------------------------------------------------------------------------------------------------------
# import states
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# the real local model, when it is installed (the product's configuration)
# ---------------------------------------------------------------------------------------------------------------

def _real_model() -> emb.OnnxEmbeddings | None:
    config = emb.load_config()
    provider = emb.create_provider(config)
    if not isinstance(provider, emb.OnnxEmbeddings) or not provider._files_present():
        return None
    return provider


@pytest.mark.skipif(_real_model() is None, reason="local embedding model not installed")
def test_the_real_local_model_ranks_a_german_paraphrase(tmp_path, library):
    provider = _real_model()
    try:
        svc = make_service(tmp_path, library, embeddings=provider)
        svc.embed_pending()
        found = svc.search("Hormon für die Natriumaufnahme in der Niere")
        best = found["results"][0]
        assert best["document"]["filename"] == "Vorlesung Herz.docx" and best["best"]["location"]["heading_path"] == ["Niere"]
        vector = provider.embed_query("Frank-Starling-Mechanismus")
        assert vector.shape[0] >= 384 and abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3
    finally:
        provider.close()


def test_the_interface_logic_holds_drawer_marks_and_quiet_missions():
    """ui/tests/study_ui.test.mjs: the narrow-window drawer, phrase marking (umlauts, hyphenation, primary), stale missions."""

    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, str(root / "ui" / "tests" / "study_ui.test.mjs")], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0 and "ALL OK" in completed.stdout, completed.stdout[-2000:] + completed.stderr[-500:]


def test_the_pdf_rendering_plan_holds_preview_sharp_tiles_and_cancellation():
    """ui/tests/study_pdf.test.mjs: preview then sharp, tiles only when zoomed in, bounded bitmaps, superseded and aborted renders."""

    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, str(root / "ui" / "tests" / "study_pdf.test.mjs")], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0 and "all passed" in completed.stdout, completed.stdout[-2000:] + completed.stderr[-500:]


def test_the_indexing_line_speaks_real_progress_and_calm_failures():
    """ui/tests/study_indexing.test.mjs: which document of how many, the OCR page, calm failure words, the job list order."""

    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, str(root / "ui" / "tests" / "study_indexing.test.mjs")], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0 and "ALL OK" in completed.stdout, completed.stdout[-2000:] + completed.stderr[-500:]


def test_a_misheard_word_still_marks_the_written_phrase(tmp_path, library):
    """Speech recognition heard "Frank Stalin Mechanismus": the page is found and its real spelling is what gets marked."""

    svc = make_service(tmp_path, library)
    found = svc.search("Zeig mir die Seite zum Frank Stalin Mechanismus.")
    best = found["results"][0]["best"]
    assert found["results"][0]["document"]["filename"] == "Physiologie Herz.pdf"
    assert best["focus"] == "Frank-Starling-Mechanismus" and best["phrases"][0] == "Frank-Starling-Mechanismus"
    two_words = svc.search("Vorlast Kontraktilität")
    assert all(r["best"]["focus"] in {"Vorlast", "Kontraktilität"} for r in two_words["results"]), "two words are never taken for a phrase"


def test_asking_for_notes_or_slides_opens_that_kind_of_material(tmp_path, library):
    """"Öffne meine Notizen zur X" opens the owner's notes, "in meinen Folien" the slides -- even when another kind says the same."""

    (library / "Neuro-Notizen.md").write_text("# Motorik\n\nBasalganglienschleife: direkter Weg fördert, indirekter Weg hemmt.\n", encoding="utf-8")
    svc = make_service(tmp_path, library, embeddings=ConceptEmbeddings())
    svc.embed_pending()
    notes = svc.search("Öffne meine Notizen zur Basalganglienschleife")
    assert notes["results"][0]["document"]["filename"] == "Neuro-Notizen.md", [r["document"]["filename"] for r in notes["results"]]
    slides = svc.search("Wo steht in meinen Folien etwas zur Basalganglienschleife?")
    assert slides["results"][0]["document"]["filename"] == "Neuro.pptx"
    plain = svc.search("Basalganglienschleife")
    assert {r["document"]["filename"] for r in plain["results"]} >= {"Neuro-Notizen.md", "Neuro.pptx"}, "without a hint both kinds are found"


def test_uploads_never_leave_stray_files_in_the_library(tmp_path, library):
    """A duplicate is not written a second time, and a file Studium could not read is removed again."""

    svc = make_service(tmp_path, library)
    store = tmp_path / "store"
    before = sorted(p.name for p in store.iterdir())
    same = (store / "Stoffwechsel.md").read_bytes()
    assert svc.import_bytes("Stoffwechsel Kopie.md", same)["duplicate"] is True
    assert svc.import_bytes("Kaputt.docx", b"not a zip")["reason"] == "parse_failed"
    assert sorted(p.name for p in store.iterdir()) == before


@pytest.mark.skipif(not __import__("importlib").util.find_spec("PySide6"), reason="PySide6 (QtPdf) not installed")
def test_an_open_pdf_does_not_lock_the_owners_file(tmp_path):
    """The page worker reads PDFs into memory: an open document can be deleted or replaced (OneDrive, GoodNotes backups)."""

    from study.pdf_engine import PdfEngine

    engine = PdfEngine()
    try:
        path = tmp_path / "Skript.pdf"
        path.write_bytes(make_pdf(HEART_PAGES))
        assert len(engine.texts(path) or []) == 4
        assert engine.render(path, 1, tmp_path / "p1.png", width=400)["ok"]
        path.write_bytes(make_pdf(HEART_PAGES[:2]))  # replaced while open
        assert len(engine.texts(path) or []) == 2, "a changed file is read again"
        path.unlink()  # raises PermissionError on Windows when the worker holds a handle
        assert not path.exists()
    finally:
        engine.close()
