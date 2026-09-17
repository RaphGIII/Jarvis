"""Studium from the chat: one local pipeline, a remembered place, and commands about that place -- no model involved.

"Zeig mir die Seite zum Frank-Starling-Mechanismus" opens the page; then "Seite danach", "Seite davor",
"Welche Seite war das?", "Mach größer", "Zeig die ganze Seite", "Was steht darunter?", "Zurück zu der Stelle",
"Zeig mir noch einen Treffer" act on that place.  Without an open place those sentences are not study commands.
"""

from __future__ import annotations

import pytest

from service.events import EventType
from study.intents import parse_study_operation
from study_fixtures import HEART_PAGES, make_docx, make_pdf
from test_study import Recording, by_name, drain
from test_actionability import make


@pytest.fixture
def notes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pages = HEART_PAGES + [["Wiederholung", "Der Frank-Starling-Mechanismus gilt für beide Ventrikel."]]
    (src / "Physiologie Herz.pdf").write_bytes(make_pdf(pages))
    make_docx(src / "Vorlesung Niere.docx", [("h1", "Niere"), ("p", "Aldosteron fördert die Natriumrückresorption im Sammelrohr.")])
    return src


def core_with(tmp_path, notes):
    provider = Recording()
    core, _ = make(tmp_path, provider=provider)
    for path in sorted(notes.iterdir()):
        assert core.study.import_path(path, copy=True)["ok"]
    return core, provider


def message(events):
    return next(e.payload for e in events if e.type is EventType.MESSAGE)


def opened(events):
    return [e.payload for e in events if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "open_view"]


@pytest.mark.parametrize("text, operation, arguments", [
    ("Seite davor", "study.page", {"step": -1}),
    ("Seite danach", "study.page", {"step": 1}),
    ("Geh zu Seite 4", "study.page", {"number": 4}),
    ("Zurück zu der Stelle", "study.return", None),
    ("Mach größer", "study.view", {"zoom": 0.25}),
    ("Mach kleiner", "study.view", {"zoom": -0.25}),
    ("Zeig die ganze Seite", "study.view", {"fit": True}),
    ("Was steht darunter?", "study.below", {}),
    ("Was habe ich daneben geschrieben?", "study.beside", {}),
    ("Zeig mir noch einen Treffer", "study.next_hit", {}),
])
def test_commands_about_the_open_place_are_study_navigation_only_with_a_place(text, operation, arguments):
    with_place = parse_study_operation(text, has_material=True, has_context=True)
    assert with_place is not None and with_place.operation == operation, (text, with_place)
    if arguments is not None:
        assert dict(with_place.arguments) == arguments
    assert parse_study_operation(text, has_material=True, has_context=False) is None, f"{text!r} with nothing open is not a study command"


def test_scope_subject_and_exclusions_decide_without_a_model():
    locate = parse_study_operation("Zeig mir meine handschriftlichen Notizen zur Vorlast", has_material=True)
    assert locate.operation == "study.locate" and locate.arguments["handwriting"] is True
    assert parse_study_operation("Öffne die Datei notiz.txt", has_material=True) is None, "a named file belongs to the file tools"
    assert parse_study_operation("Zeig mir das Wetter", has_material=True, probe=lambda topic: True) is None
    assert parse_study_operation("Zeig mir Frank-Starling", has_material=True, probe=lambda topic: False) is None, "not in the library: not Studium"
    probed = parse_study_operation("Zeig mir Frank-Starling", has_material=True, probe=lambda topic: "starling" in topic.lower())
    assert probed is not None and probed.operation == "study.locate"


def test_a_found_place_can_be_navigated_zoomed_read_on_and_returned_to(tmp_path, notes):
    core, provider = core_with(tmp_path, notes)
    doc = by_name(core.study, "Physiologie Herz.pdf")
    first = drain(core, "Zeig mir die Seite zum Frank-Starling-Mechanismus.")
    assert opened(first)[-1]["params"]["unit"] == "3", "exact phrase on page 3 dominates the repetition page"

    after = drain(core, "Seite danach")
    assert opened(after)[-1]["params"] == {"doc": doc["id"], "unit": "4"}
    before = drain(core, "Seite davor")
    assert opened(before)[-1]["params"]["unit"] == "3"
    assert message(drain(core, "Welche Seite war das?"))["text"] == "Das war Physiologie Herz.pdf, Seite 3."

    bigger = drain(core, "Mach größer")
    view = [e.payload for e in bigger if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "study_view"]
    assert view and view[-1]["command"] == {"zoom": 0.25} and view[-1]["params"]["doc"] == doc["id"]
    assert view[-1]["level"] == 1.25 and view[-1]["params"]["focus"] == "Frank-Starling-Mechanismus", "enough to reopen the place at that size"
    whole = drain(core, "Zeig die ganze Seite")
    assert [e.payload["command"] for e in whole if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "study_view"] == [{"fit": True}]

    below = message(drain(core, "Was steht darunter?"))["text"]
    assert "Vorlast dehnt die Sarkomere" in below, below

    drain(core, "Seite danach")
    back = drain(core, "Zurück zu der Stelle")
    assert opened(back)[-1]["params"]["unit"] == "3" and opened(back)[-1]["params"]["focus"] == "Frank-Starling-Mechanismus"

    other = drain(core, "Zeig mir noch einen Treffer")
    assert opened(other)[-1]["params"]["unit"] == "5", "the next place for the same topic"
    assert provider.prompts == [], "every step was local"


def test_the_open_place_survives_a_restart(tmp_path, notes):
    core, _ = core_with(tmp_path, notes)
    drain(core, "Zeig mir die Seite zum Frank-Starling-Mechanismus.")
    from service.study_actions import StudyActions

    again = StudyActions(core)
    assert again.has_context() and again.context()["unit"] == 3 and again.context()["document_id"] == by_name(core.study, "Physiologie Herz.pdf")["id"]


def test_the_viewer_reports_its_place_and_the_chat_follows_it(tmp_path, notes):
    core, _ = core_with(tmp_path, notes)
    doc = by_name(core.study, "Physiologie Herz.pdf")
    answer = core.study_actions.set_focus(doc["id"], 2, zoom=1.5, mode="page", highlights=["Systole"], focus="Systole")
    assert answer["ok"] and answer["focus"]["zoom"] == 1.5 and answer["focus"]["highlights"] == ["Systole"]
    assert opened(drain(core, "Seite danach"))[-1]["params"]["unit"] == "3"


# ---------------------------------------------------------------------------
# dominance, uncertain recognition
# ---------------------------------------------------------------------------

def test_dominance_opens_only_a_clear_winner():
    from service.study_actions import dominant

    place = lambda doc, relevance, match="terms": {"document_id": doc, "relevance": relevance, "match": match}  # noqa: E731
    assert dominant([place("a", 1.0)]) and dominant([place("a", 1.0), place("a", 0.99)]), "one document: open it"
    assert dominant([place("a", 1.0, "phrase"), place("b", 0.98)]), "the exact phrase beats the loose words"
    assert dominant([place("a", 1.0), place("b", 0.7)])
    assert not dominant([place("a", 1.0), place("b", 0.9)]), "two documents about equally good: ask"
    assert not dominant([])


def test_two_equal_places_are_offered_not_guessed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    make_docx(src / "Kreislauf A.docx", [("h1", "Kreislauf"), ("p", "Barorezeptoren melden den Blutdruck an den Hirnstamm.")])
    make_docx(src / "Kreislauf B.docx", [("h1", "Blutdruck"), ("p", "Barorezeptoren melden den Blutdruck an den Hirnstamm.")])  # not a duplicate
    provider = Recording()
    core, _ = make(tmp_path, provider=provider)
    for path in sorted(src.iterdir()):
        assert core.study.import_path(path, copy=True).get("duplicate") is not True
    events = drain(core, "Zeig mir die Stelle zu den Barorezeptoren.")
    text = message(events)["text"]
    assert not opened(events) and "Welche soll ich öffnen?" in text, text
    assert provider.prompts == []


def _recognised_page(svc, doc_id, filename, text, *, confidence, kind="handwriting"):
    from study.model import Block, NormalizedDocument, Unit

    doc = NormalizedDocument(source_type="image", title=filename.rsplit(".", 1)[0])
    unit = Unit(number=1, kind="page", text=text, ocr=True, has_text=True)
    unit.blocks.append(Block(text=text, kind=kind, char_start=0, char_end=len(text), confidence=confidence, role="body"))
    doc.units.append(unit)
    svc.index.upsert({"id": doc_id, "filename": filename, "title": doc.title, "stored_path": str(svc.store_dir / filename), "original_path": "",
                      "provider": "upload", "source_id": "", "content_hash": doc_id, "size_bytes": 1, "created_at": 1.0, "updated_at": 1.0,
                      "status": "ready", "flags": {}}, doc)


def test_uncertain_handwriting_is_a_possible_hit_and_says_so(tmp_path):
    from service.study_actions import place_word
    from test_study import service

    svc = service(tmp_path)
    _recognised_page(svc, "hand0000000000001", "Mitschrift.jpg", "Frank Starlinq Mechanismus: mehr Vorlast mehr Schlagvolumen", confidence=0.42)
    found = svc.search("Frank-Starling-Mechanismus", semantic=False)
    assert found["results"], "a misread word is still found"
    best = found["results"][0]["best"]
    assert best["handwriting"] is True and best["certainty"] == "possible", best
    source = {**best, "document_id": "hand0000000000001"}
    assert place_word(source).lower().startswith("möglicher"), place_word(source)

    _recognised_page(svc, "hand0000000000002", "Sauber.jpg", "Frank-Starling-Mechanismus: mehr Vorlast, mehr Schlagvolumen", confidence=0.95)
    ranked = svc.search("Frank-Starling-Mechanismus", semantic=False)["results"]
    assert ranked[0]["document"]["filename"] == "Sauber.jpg", "a confident reading ranks above an uncertain one"
    assert ranked[0]["best"]["certainty"] == "sure"


def test_misread_handwriting_is_found_but_never_called_sure(tmp_path):
    """Seen live on a handwritten photo: "Aldosteron" read as "Aidoteron", "Nierenphysiologie" as "Nieren: musiogie"."""

    from test_study import service

    svc = service(tmp_path)
    _recognised_page(svc, "hand0000000000003", "Foto.jpg", "Nieren: musiogie\n\nAidoteron: Natriummekresorption", confidence=0.82)
    results = svc.search("Aldosteron", semantic=False)["results"]
    assert results and results[0]["document"]["filename"] == "Foto.jpg", "a near spelling is found"
    assert results[0]["best"]["certainty"] == "possible", results[0]["best"]
    for hit in svc.search("Nierenphysiologie", semantic=False)["results"]:
        assert hit["best"]["certainty"] == "possible", "a stem of a compound is not the word"
    assert svc.search("Natriummekresorption", semantic=False)["results"][0]["best"]["certainty"] == "sure", "letter for letter, confidently read"
