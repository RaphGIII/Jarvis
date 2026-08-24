"""Where to cut a stream of tokens so Jarvis can start speaking early.

Two failure modes are being defended against, and they pull opposite ways:
waiting for the whole answer (seconds of silence after every question) and
cutting too eagerly (clipped, robotic delivery). The tests below pin the
asymmetry that resolves it -- an early first phrase, longer ones after -- and
the places where a naive full-stop split would stumble audibly.
"""

from __future__ import annotations

import pytest

from speech.chunker import PhraseChunker, split_for_speech


def stream(chunker, text, *, size=3):
    """Feed text the way a model produces it: in small, arbitrary pieces."""

    found = []
    for index in range(0, len(text), size):
        found += chunker.feed(text[index : index + size])
    return found


# --------------------------------------------------------------------------
# The first phrase is the one the user waits for
# --------------------------------------------------------------------------

def test_the_first_phrase_is_emitted_early():
    chunker = PhraseChunker()

    phrases = stream(chunker, "Guten Morgen. Ich habe die Ergebnisse der letzten Nacht geprüft und alles ist durchgelaufen.")

    assert phrases, "nothing was emitted before the answer finished"
    assert phrases[0].text == "Guten Morgen."
    assert phrases[0].first


def test_later_phrases_are_allowed_to_be_longer():
    """Only the opening phrase is on the latency path."""

    chunker = PhraseChunker(first_min_chars=2, min_chars=60)
    text = "Ja. Kurz. Noch kurz. Das hier ist ein deutlich längerer Satz, der genug Substanz hat."

    phrases = stream(chunker, text) + chunker.flush()

    # The opening phrase goes out as soon as a sentence completes, however
    # short: it is the only one whose delay the user hears.
    assert phrases[0].text == "Ja."
    # After that the rule inverts. "Kurz." and "Noch kurz." are each far below
    # min_chars, so they accumulate instead of becoming clipped little
    # synthesis jobs of their own.
    assert phrases[1].text == "Kurz. Noch kurz. Das hier ist ein deutlich längerer Satz, der genug Substanz hat."
    assert len(phrases) == 2


def test_nothing_is_emitted_until_a_boundary_is_reached():
    chunker = PhraseChunker()

    assert chunker.feed("Ich denke gerade") == []


def test_flush_emits_the_remainder():
    chunker = PhraseChunker()
    chunker.feed("Kein Satzzeichen am Ende")

    phrases = chunker.flush()

    assert [p.text for p in phrases] == ["Kein Satzzeichen am Ende"]
    assert phrases[0].first


def test_flush_on_an_empty_buffer_emits_nothing():
    assert PhraseChunker().flush() == []


def test_flush_after_everything_was_already_emitted():
    chunker = PhraseChunker(first_min_chars=1)
    chunker.feed("Fertig. ")

    assert chunker.flush() == []


# --------------------------------------------------------------------------
# Places a naive split would stumble
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Die Zahl ist 3.14 und damit erledigt sich die Frage nach dem Umfang des Kreises.",
        "Die Datei heisst config.json und liegt im Projektverzeichnis auf dieser Maschine.",
        "Besuche example.com für die vollständige Dokumentation dieses Projekts hier.",
    ],
)
def test_a_full_stop_inside_a_token_is_not_a_sentence_end(text):
    """3.14, config.json, example.com -- cutting here is audible."""

    parts = split_for_speech(text)

    assert len(parts) == 1, f"split into {parts}"


@pytest.mark.parametrize(
    "abbreviation",
    ["z.B.", "d.h.", "usw.", "bzw.", "ca.", "Dr.", "Mr.", "etc.", "vgl."],
)
def test_common_abbreviations_do_not_end_a_sentence(abbreviation):
    text = f"Wir nutzen lokale Modelle, {abbreviation} für die Umwandlung von Sprache in Text auf dem Rechner."

    parts = split_for_speech(text)

    assert len(parts) == 1, f"{abbreviation} split into {parts}"


def test_an_initial_is_not_a_sentence_end():
    parts = split_for_speech("Das Papier stammt von J. Smith und beschreibt genau dieses Verfahren im Detail.")

    assert len(parts) == 1, parts


def test_an_ellipsis_is_one_boundary_not_three():
    parts = split_for_speech("Einen Moment... Ich sehe gerade nach, was dort tatsächlich passiert ist.")

    assert len(parts) == 2
    assert parts[0] == "Einen Moment..."


def test_combined_terminators_stay_together():
    parts = split_for_speech("Wirklich?! Das hätte ich so nun wirklich nicht erwartet an dieser Stelle.")

    assert parts[0] == "Wirklich?!"


def test_a_closing_quote_travels_with_its_sentence():
    parts = split_for_speech('Er sagte "das genügt." Danach war das Thema für alle Beteiligten erledigt.')

    assert parts[0].endswith('"'), parts[0]


def test_a_trailing_terminator_waits_for_the_next_token():
    """Mid-stream, "3." could be a decimal; only the next token settles it."""

    chunker = PhraseChunker(first_min_chars=1)

    assert chunker.feed("Das Ergebnis ist 3") == []
    assert chunker.feed(".") == [], "a full stop at the very end is still ambiguous"
    assert chunker.feed("14 und fertig") == []


# --------------------------------------------------------------------------
# A model that never punctuates must not be able to withhold speech
# --------------------------------------------------------------------------

def test_a_very_long_sentence_breaks_at_a_clause():
    chunker = PhraseChunker(first_min_chars=10, min_chars=40, max_chars=100)
    text = (
        "Ich habe die Konfiguration geprüft, danach die Modelle geladen, "
        "anschließend die Tests gestartet, und alles lief durch"
    )

    phrases = stream(chunker, text)

    assert phrases, "a long clause-separated run must not block speech"
    assert phrases[0].reason in {"clause", "sentence"}


def test_text_with_no_punctuation_at_all_still_gets_spoken():
    chunker = PhraseChunker(first_min_chars=10, min_chars=40, max_chars=100, hard_max_chars=150)
    text = "wort " * 60

    phrases = stream(chunker, text)

    assert phrases, "no punctuation must not mean no speech"
    assert all(len(p.text) <= 160 for p in phrases)


def test_a_hard_break_lands_on_a_word_boundary():
    chunker = PhraseChunker(first_min_chars=5, min_chars=20, max_chars=60, hard_max_chars=80)

    phrases = stream(chunker, "abc " * 40)

    assert phrases
    assert not phrases[0].text.endswith("ab"), "cut mid-word"


# --------------------------------------------------------------------------
# Streaming behaves the same as whole text
# --------------------------------------------------------------------------

@pytest.mark.parametrize("size", [1, 2, 5, 17, 200])
def test_the_result_does_not_depend_on_token_size(size):
    """A chunker whose output depends on how the model happened to batch is wrong."""

    text = "Erstens. Zweitens kommt ein etwas längerer Satz mit mehr Inhalt. Und drittens noch einer."
    chunker = PhraseChunker()

    phrases = stream(chunker, text, size=size) + chunker.flush()

    assert [p.text for p in phrases] == split_for_speech(text)


def test_no_content_is_lost():
    text = "Eins. Zwei drei vier. Fünf sechs sieben acht neun zehn elf zwölf dreizehn."

    joined = " ".join(split_for_speech(text))

    assert joined.replace(" ", "") == text.replace(" ", "")


def test_no_phrase_is_empty_or_whitespace():
    text = "Ja.   Nein.    Vielleicht auch nicht, wer weiss das schon so genau heute."

    for part in split_for_speech(text):
        assert part.strip() == part
        assert part


def test_exactly_one_phrase_is_marked_first():
    text = "Eins. Zwei drei vier fünf sechs sieben. Acht neun zehn elf zwölf dreizehn vierzehn."

    chunker = PhraseChunker()
    phrases = chunker.feed(text) + chunker.flush()

    assert sum(1 for p in phrases if p.first) == 1


def test_reset_starts_a_new_answer():
    chunker = PhraseChunker()
    chunker.feed("Erste Antwort. ")
    chunker.reset()

    assert chunker.pending == ""
    phrases = chunker.feed("Zweite Antwort ist hier und lang genug für einen Satz.") + chunker.flush()
    assert phrases[0].first


def test_english_works_too():
    parts = split_for_speech("Good morning. I checked the results and everything completed successfully.")

    assert parts[0] == "Good morning."


def test_a_single_short_sentence_is_still_spoken():
    assert split_for_speech("Ja.") == ["Ja."]
