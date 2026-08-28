"""What reaches the TTS provider: normalisation, the lexicon, owner corrections.

Text-level checks only.  Whether the result *sounds* right is the owner's
call; nothing here claims acoustic quality.
"""

from __future__ import annotations

import json

import pytest

from speech.pronounce import ACCEPTANCE_SET, Lexicon, Pronouncer, normalize, spell


def test_the_acceptance_set_changes_where_the_lexicon_says_so():
    p = Pronouncer(Lexicon())
    rendered = {s: p.render(s, language="de").spoken for s in ACCEPTANCE_SET}
    assert rendered["GitHub"] == "Git-Hab" and rendered["Spotify"] == "Spottifai" and rendered["GPU"] == "Ge-Pe-U"
    assert rendered["Knowledge Graph"] == "Nolledsch Graf" and rendered["Stockfish"] == "Stockfisch"
    assert rendered["Desoxyribonukleinsäure"] == "Desoxy-ribo-nuklein-säure"
    # German words that espeak-de already reads well are left alone
    assert rendered["Mitochondrien"] == "Mitochondrien" and rendered["Physiologie"] == "Physiologie"


def test_the_displayed_text_is_never_changed():
    p = Pronouncer(Lexicon())
    r = p.render("ZEUS verwendet die GPU.", language="de")
    assert r.displayed == "ZEUS verwendet die GPU." and r.spoken == "Zeus verwendet die Ge-Pe-U." and r.changed


def test_acronyms_units_numbers_and_urls_are_normalised():
    assert normalize("Die API antwortet in 250ms.", language="de") == "Die A-Pe-I antwortet in 250 Millisekunden."
    assert normalize("12 GB frei, 80% belegt", language="de") == "12 Gigabyte frei, 80 Prozent belegt"
    assert normalize("siehe https://lichess.org/analysis", language="de") == "siehe lichess Punkt org Schrägstrich analysis"
    assert normalize("The API needs 4 GB.", language="en") == "The A-P-I needs 4 gigabytes."
    assert normalize("JSON und HTML bleiben Wörter", language="de") == "JSON und HTML bleiben Wörter"
    assert spell("CPU", "de") == "Ze-Pe-U"


def test_english_terms_inside_german_sentences_use_the_german_respelling():
    p = Pronouncer(Lexicon())
    assert p.render("Das Update von GitHub läuft über die Cloud.", language="de").spoken == "Das Apdeit von Git-Hab läuft über die Klaud."


def test_owner_entries_persist_and_outrank_the_seed(tmp_path):
    path = tmp_path / "voice" / "lexicon.json"
    lex = Lexicon(path)
    lex.set("Zeus", "Zojs", language="de", note="owner")
    assert json.loads(path.read_text(encoding="utf-8"))["entries"][0]["surface"] == "Zeus"

    again = Lexicon(path)
    assert Pronouncer(again).render("Zeus ist bereit", language="de").spoken == "Zojs ist bereit"
    assert again.lookup("Zeus", "de").source == "owner"
    assert again.remove("Zeus", language="de")
    assert Pronouncer(Lexicon(path)).render("Zeus ist bereit", language="de").spoken == "Zeus ist bereit"


def test_whole_words_only_and_longest_first():
    p = Pronouncer(Lexicon())
    assert p.render("Knowledge Graph", language="de").spoken == "Nolledsch Graf"
    assert p.render("Knowledgebase", language="de").spoken == "Knowledgebase", "no partial-word replacement"


def test_the_language_selects_the_lexicon():
    p = Pronouncer(Lexicon())
    assert p.render("Zeus", language="en").spoken == "Zoos"
    assert p.render("Zeus", language="de").spoken == "Zeus"


def test_the_service_speaks_the_rendered_form_and_shows_the_original(tmp_path):
    from service.events import EventBus, EventType
    from service.voice import VoiceService, VoiceSettings
    from speech.contracts import Audio

    spoken: list[str] = []

    class Engine:
        def status(self): return {"available": True, "voices": []}
        def synthesize(self, text, *, voice="", language=""):
            spoken.append(text)
            return Audio(samples=bytes(4410), sample_rate=22050)

    bus = EventBus()
    service = VoiceService(bus, engine_factory=Engine, settings=VoiceSettings(enabled=True, language="de"), settings_path=tmp_path / "voice" / "settings.json")
    with bus.subscribe(replay=False) as sub:
        service.speak_stream(["ZEUS verwendet die GPU."])
        events = [e.payload for e in sub.drain() if e.type is EventType.SPEECH]

    assert spoken == ["Zeus verwendet die Ge-Pe-U."]
    assert events and events[0]["text"] == "ZEUS verwendet die GPU." and events[0]["spoken"] == "Zeus verwendet die Ge-Pe-U."
    assert service.lexicon.owner_path == tmp_path / "voice" / "lexicon.json"
