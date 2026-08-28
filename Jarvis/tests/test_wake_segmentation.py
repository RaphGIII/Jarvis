"""The wake word is metadata, never user text (acceptance A–F of the segmentation brief)."""

from __future__ import annotations

import pytest

from service.core import JarvisCore
from service.events import EventType
from service.voice import VoiceService
from speech.contracts import Audio, Transcript
from speech.wake_segment import sounds_like, strip_wake_word


def test_a_continuous_speech_loses_only_the_wake_word():
    seg = strip_wake_word("Zeus, wie geht es dir?", wake_word="Zeus")
    assert seg.text == "Wie geht es dir?" and seg.removed.startswith("Zeus")


def test_e_a_misheard_wake_word_is_removed_inside_a_wake_session():
    for heard in ("Solls, wie geht es dir?", "Seus wie geht es dir?", "Zoiß, wie geht es dir?", "Hey Zeus, wie geht es dir?"):
        assert strip_wake_word(heard, wake_word="Zeus").text == "Wie geht es dir?", heard


def test_f_no_global_replacement_outside_a_wake_session():
    for heard in ("Solls, wie geht es dir?", "Jesus ist eine historische Figur", "Servus, wie geht es dir?"):
        seg = strip_wake_word(heard, wake_word="Zeus", wake_session=False)
        assert seg.text == heard and not seg.removed


def test_ordinary_words_are_never_taken_for_the_wake_word():
    assert not sounds_like("Servus", "Zeus")      # too long, different shape
    assert not sounds_like("Jesus", "Zeus")       # starts with j
    assert not sounds_like("Wie", "Zeus")
    assert not sounds_like("Sonne", "Zeus")       # ends in e
    assert sounds_like("Solls", "Zeus") and sounds_like("Zeus", "Zeus")
    # inside a session a sentence that merely starts with an ordinary word is untouched
    assert strip_wake_word("Servus, wie geht es dir?", wake_word="Zeus").text == "Servus, wie geht es dir?"
    assert strip_wake_word("Jesus ist eine historische Figur", wake_word="Zeus").text == "Jesus ist eine historische Figur"


def test_word_timestamps_keep_a_late_look_alike():
    words = [{"word": "Solls", "start": 1.8, "end": 2.1}]
    seg = strip_wake_word("Solls das so sein?", wake_word="Zeus", words=words)
    assert seg.text == "Solls das so sein?" and "outside the wake tail" in seg.reason


def test_only_the_wake_word_leaves_nothing_to_route():
    assert strip_wake_word("Zeus.", wake_word="Zeus").text == ""


class Kernel:
    def __init__(self):
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        class P:
            def generate_stream(self, prompt, **_):
                yield "ok"
        return P()


def make(heard, confidence=0.9):
    core = JarvisCore(kernel=Kernel())

    class Engine:
        def status(self): return {"available": True, "voices": []}
        def transcribe(self, audio, *, language=""): return Transcript(text=heard, confidence=confidence)

    core._voice = VoiceService(core.bus, engine_factory=Engine)
    return core, Audio(samples=bytes(16000), sample_rate=16000).to_wav()


def test_router_receives_the_command_and_the_turn_carries_wake_metadata():
    core, wav = make("Solls, wie geht es dir?")
    with core.bus.subscribe(replay=False) as sub:
        result = core.hear(wav, wake=0.93, session="vs9", answer=False)
        events = sub.drain()
    assert result["ok"] and result["text"] == "Wie geht es dir?"
    wake_rows = [e.payload for e in events if e.type is EventType.DIAGNOSTIC and e.payload.get("wake")]
    assert wake_rows and wake_rows[0]["score"] == 0.93 and wake_rows[0]["command"] == "Wie geht es dir?"


def test_b_pause_then_command_gives_the_same_result():
    core, wav = make("Wie geht es dir?")
    assert core.hear(wav, wake=0.9, session="vs10", answer=False)["text"] == "Wie geht es dir?"


def test_c_and_d_no_wake_session_means_no_request():
    for heard in ("Servus, wie geht es dir?", "Jesus ist eine historische Figur"):
        core, wav = make(heard)
        result = core.hear(wav, answer=False)
        assert result["ok"] is False and result["ignored"]
        assert core.history == []


def test_the_conversation_memory_holds_the_command_not_the_wake_word():
    core, wav = make("Zeus, wie geht es dir?")
    core.hear(wav, wake=0.95, session="vs11", answer=True)
    assert core.history[0].text == "Wie geht es dir?"
    assert core.history[0].meta["wake_word"] == "Zeus" and core.history[0].meta["wake_score"] == 0.95
    assert core.history[0].to_dict()["meta"]["session"] == "vs11"
