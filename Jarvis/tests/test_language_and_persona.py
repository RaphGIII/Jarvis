"""Language detection, and the persona surviving a change of backend.

Detection matters before any model runs: whisper decodes better when told the
language, and a TTS voice has to be picked before there is anything to say.
Asking an LLM would put a generation on the critical path of every utterance to
learn something a lookup settles in microseconds.

The hard case is short input. "ok", "ja", "stop" carry almost no signal, and a
confident wrong answer means the recogniser gets the wrong hint and the voice
changes mid-conversation -- which sounds far worse than occasionally answering
in the wrong language.
"""

from __future__ import annotations

import time

import pytest

from persona.language import LanguageGuess, detect, language_name, stable_language
from service.core import JarvisCore


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Wie geht es mit meinem Projekt weiter und was ist der nächste Schritt?", "de"),
        ("Kannst du mir bitte sagen, was gestern nicht funktioniert hat?", "de"),
        ("What is the current status of the project and what should we do next?", "en"),
        ("Can you please tell me why the tests are failing on this machine?", "en"),
        ("Peux-tu me dire ce qui ne fonctionne pas avec le projet aujourd'hui?", "fr"),
        ("¿Puedes decirme qué está pasando con el proyecto hoy por favor?", "es"),
    ],
)
def test_ordinary_sentences_are_identified(text, expected):
    guess = detect(text)

    assert guess.language == expected, guess.detail
    assert guess.confident, f"{guess.confidence} is too low for a full sentence"


def test_umlauts_are_strong_evidence_for_german():
    assert detect("Größe und Prüfung").language == "de"


def test_a_borrowed_accent_does_not_flip_a_whole_english_sentence():
    """One "café" must not outvote a sentence of English stopwords."""

    guess = detect("I went to the café and then I came back to the office to work")

    assert guess.language == "en", guess.detail


# --------------------------------------------------------------------------
# Short input: the case that breaks naive detectors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["ok", "ja", "stop", "hm", "42", "..."])
def test_a_one_word_message_is_never_confident(text):
    assert not detect(text).confident, f"{text!r} should not be trusted"


def test_empty_input_reports_no_confidence():
    guess = detect("")

    assert guess.confidence == 0.0
    assert not guess.confident


def test_a_default_is_returned_for_unrecognisable_input():
    assert detect("xyzzy plugh", default="de").language == "de"


# --------------------------------------------------------------------------
# Stickiness
# --------------------------------------------------------------------------

def test_a_short_reply_does_not_switch_the_conversation():
    """The failure this prevents: German conversation, user says "ok", voice flips."""

    assert stable_language("ok", current="de") == "de"


def test_a_full_sentence_in_another_language_does_switch():
    assert stable_language(
        "Actually, could you explain what went wrong with the last build?", current="de"
    ) == "en"


def test_with_no_history_the_default_is_used():
    assert stable_language("hm", current="", default="en") == "en"


def test_a_confident_detection_wins_over_the_default():
    assert stable_language(
        "Wie lange dauert das noch und was fehlt dafür genau?", current="", default="en"
    ) == "de"


def test_language_names_are_human_readable():
    assert language_name("de") == "German"
    assert language_name("en-GB") == "English"
    assert language_name("zz") == "zz"


# --------------------------------------------------------------------------
# The core keeps one identity across backends
# --------------------------------------------------------------------------

class StubKernel:
    def __init__(self, state_root):
        self.state_root = state_root
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        class P:
            def generate_stream(self, prompt, **_):
                yield "Alles klar."

        return P()


@pytest.fixture()
def core(tmp_path):
    return JarvisCore(kernel=StubKernel(tmp_path))


def test_the_prompt_comes_from_the_persona_store(core):
    prompt = core._compose_prompt("hallo")

    from core.identity import current

    assert f"You are {current().assistant_name}" in prompt
    # The invariant rules live in the store and must reach the prompt.
    assert "Never claim" in prompt or "never claim" in prompt


def test_the_invariant_rules_are_appended_after_the_persona(core):
    """A verbose character must not be able to crowd them out."""

    from core.identity import current

    prompt = core._compose_prompt("hallo")
    persona_text = prompt.index(f"You are {current().assistant_name}")

    assert prompt.index("Prefer saying you do not know") > persona_text


def test_a_broken_persona_file_does_not_silence_jarvis(tmp_path):
    (tmp_path / "personas.json").write_text("{not json", encoding="utf-8")
    instance = JarvisCore(kernel=StubKernel(tmp_path))

    from core.identity import current

    prompt = instance._compose_prompt("hallo")

    assert f"You are {current().assistant_name}" in prompt


def test_the_detected_language_reaches_the_prompt(core):
    core.send_message("Wie geht es mit dem Projekt weiter und was fehlt noch dafür?")
    _settle(core)

    assert core.language == "de"
    assert "German" in core._compose_prompt("und weiter?")


def test_the_conversation_language_is_reported_in_status(core):
    core.send_message("What is the status of the project and what happens next here?")
    _settle(core)

    assert core.status()["language"] == "en"


def test_language_can_be_pinned_and_released(core):
    core.set_language("fr")
    assert core.language == "fr"

    # A pinned language is not overridden by a single confident message...
    core.set_language("")
    assert core.status()["language"] == "auto"


def test_switching_persona_changes_the_prompt(core):
    before = core._compose_prompt("hallo")
    result = core.set_persona("terse")
    after = core._compose_prompt("hallo")

    assert result["ok"], result
    assert before != after
    assert "few words" in after or "minimal" in after


def test_an_unknown_persona_is_refused_rather_than_silently_ignored(core):
    result = core.set_persona("nonexistent-persona")

    assert result["ok"] is False


def test_personas_can_be_listed(core):
    listing = core.list_personas()

    assert listing["active"]["name"]
    assert any(item["name"] == "default" for item in listing["personas"])


def test_the_backend_is_never_the_identity(core):
    """Whatever answered, the user is talking to Jarvis."""

    core.send_message("hallo")
    _settle(core)

    reply = core.history[-1]
    assert reply.backend == "stub"
    assert "stub" not in reply.text.lower()


def _settle(core, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(turn.role == "assistant" for turn in core.history):
            return
        time.sleep(0.05)
