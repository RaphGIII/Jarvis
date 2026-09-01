"""Ghost-speech acceptance: silence, noise, clicks, hallucinations, echo, replay.

Expected for every invalid case: 0 new USER messages, 0 actions, 0 receipts,
0 missions.  A visible owner sentence must correspond to exactly one current,
accepted utterance.  The fake recogniser *always* returns a plausible
sentence -- that is the point: text coming back is not evidence.
"""

from __future__ import annotations

import time

import pytest
from _audio import clicks_wav, noise_wav, silence_wav, speech_wav

from service.core import JarvisCore
from service.events import EventType
from service.voice import VoiceService
from speech.contracts import Audio, Transcript
from speech.utterance import AcceptanceGate, AudioEvidence, UtteranceEvidence, UtteranceLedger, similarity


class Kernel:
    def __init__(self):
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        class P:
            def generate_stream(self, prompt, **_):
                yield "ok"
        return P()


def make(heard="Erstelle ein neues Projekt namens Biochemie.", quality=None, language="de"):
    core = JarvisCore(kernel=Kernel())

    class Engine:
        calls = 0

        def status(self): return {"available": True, "voices": []}

        def transcribe(self, audio, *, language="", hotwords=""):
            Engine.calls += 1
            return Transcript(text=heard, language=language or "de", confidence=0.9, quality=dict(quality or {}))

    core._voice = VoiceService(core.bus, engine_factory=Engine)
    return core, Engine


def user_messages(core, events):
    return [e for e in events if e.type is EventType.USER_MESSAGE]


# --------------------------------------------------------------------------
# audio evidence
# --------------------------------------------------------------------------

def test_audio_evidence_separates_speech_from_silence_noise_and_clicks():
    speech = AudioEvidence.from_pcm(Audio.from_wav(speech_wav()).samples)
    silence = AudioEvidence.from_pcm(Audio.from_wav(silence_wav()).samples)
    fan = AudioEvidence.from_pcm(Audio.from_wav(noise_wav()).samples)
    keys = AudioEvidence.from_pcm(Audio.from_wav(clicks_wav()).samples)

    assert speech.speech_seconds >= 0.5 and speech.rms > 500
    assert silence.rms < 1 and silence.speech_seconds == 0
    assert fan.speech_seconds < 0.1, fan.to_dict()
    assert keys.speech_seconds < 0.25, keys.to_dict()
    assert speech.fingerprint != silence.fingerprint


# --------------------------------------------------------------------------
# the gate, on evidence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["silence", "fan", "keyboard"])
def test_silence_noise_and_keyboard_create_nothing_even_when_whisper_returns_a_sentence(label):
    wav = {"silence": silence_wav, "fan": noise_wav, "keyboard": clicks_wav}[label]()
    core, engine = make(heard="Ich habe eine Frage zu meinem Projekt.")
    with core.bus.subscribe(replay=False) as sub:
        result = core.hear(wav, wake=0.9, session=f"vs-{label}")
        events = sub.drain()

    assert result["ok"] is False and result["ignored"], result
    assert core.history == [] and core._session_receipts == []
    assert user_messages(core, events) == []
    assert engine.calls == 0, "silence must be refused before the recogniser is asked"
    traces = [e.payload for e in events if e.type is EventType.DIAGNOSTIC and e.payload.get("voice_trace")]
    assert traces and traces[0]["verdict"]["accepted"] is False


def test_real_speech_is_accepted_with_provenance():
    core, _ = make(heard="Erstelle ein neues Projekt namens Biochemie.", quality={"no_speech_probability": 0.05, "avg_logprob": -0.3, "compression_ratio": 1.2})
    with core.bus.subscribe(replay=False) as sub:
        result = core.hear(speech_wav(1.6), wake=0.91, session="vs-ok", answer=False)
        events = sub.drain()

    assert result["ok"] is True and result["speech_confidence"] > 0.6
    transcripts = [e.payload for e in events if e.type is EventType.TRANSCRIPT]
    assert transcripts and transcripts[0]["accepted"] is True and transcripts[0]["utterance_id"] == result["utterance_id"]


def test_whisper_doubt_is_read_no_speech_probability_rejects():
    core, _ = make(heard="Untertitelung des ZDF für funk.", quality={"no_speech_probability": 0.92, "avg_logprob": -0.9})
    result = core.hear(speech_wav(), wake=0.9, session="vs-nsp")
    assert result["ok"] is False and "no-speech" in result["reason"]
    assert core.history == []


def test_implausible_decoding_rejects():
    core, _ = make(heard="Vielen Dank fürs Zuschauen und bis zum nächsten Mal.", quality={"no_speech_probability": 0.2, "avg_logprob": -1.6})
    result = core.hear(speech_wav(), wake=0.9, session="vs-lp")
    assert result["ok"] is False and "implausible" in result["reason"]


def test_a_repetition_loop_rejects():
    core, _ = make(heard="ja ja ja ja ja ja ja ja", quality={"compression_ratio": 3.1})
    result = core.hear(speech_wav(), wake=0.9, session="vs-rep")
    assert result["ok"] is False and "repetition" in result["reason"]


def test_too_many_words_for_the_speech_cannot_be_real():
    core, _ = make(heard="Das ist ein sehr langer Satz mit sehr vielen Wörtern die niemand in einer halben Sekunde sagen kann wirklich nicht")
    result = core.hear(speech_wav(0.5), wake=0.9, session="vs-rate")
    assert result["ok"] is False and "cannot fit" in result["reason"]


# --------------------------------------------------------------------------
# duplicates, replay, stale sessions
# --------------------------------------------------------------------------

def test_the_same_utterance_id_is_executed_at_most_once():
    core, _ = make()
    wav = speech_wav()
    first = core.hear(wav, wake=0.9, session="vs1", answer=False, evidence={"utterance": "vs1-u1"})
    second = core.hear(speech_wav(1.3, seed=9), wake=0.9, session="vs1", answer=False, evidence={"utterance": "vs1-u1"})
    assert first["ok"] is True
    assert second["ok"] is False and "replay" in second["reason"]


def test_identical_audio_posted_twice_is_a_replay():
    core, _ = make()
    wav = speech_wav()
    assert core.hear(wav, wake=0.9, session="vsA", answer=False)["ok"] is True
    again = core.hear(wav, wake=0.9, session="vsB", answer=False)
    assert again["ok"] is False and "replay" in again["reason"]


def test_the_same_sentence_heard_again_within_the_window_is_a_duplicate():
    core, _ = make()
    assert core.hear(speech_wav(1.2, seed=1), wake=0.9, session="vs1", answer=False)["ok"] is True
    again = core.hear(speech_wav(1.2, seed=2), wake=0.9, session="vs2", answer=False)
    assert again["ok"] is False and "duplicate" in again["reason"]


def test_send_message_is_idempotent_by_request_id():
    core, _ = make()
    first = core.send_message("Hallo", request_id="req-1")
    second = core.send_message("Hallo", request_id="req-1")
    assert first["ok"] is True and second["ok"] is False and second["duplicate"] is True
    assert len([t for t in core.history if t.role == "user"]) == 1


def test_a_message_without_provenance_never_enters_the_conversation():
    core, _ = make()
    result = core.send_message("Erstelle ein Projekt", meta={"source": "whisper_hallucination"})
    assert result["ok"] is False and "provenance" in result["error"]
    assert core.history == []


# --------------------------------------------------------------------------
# ZEUS hearing itself
# --------------------------------------------------------------------------

def test_zeus_own_speech_captured_by_the_microphone_is_not_a_request():
    core, _ = make(heard="Erledigt. Projekt Biochemie ist angelegt.")
    core.voice.note_spoken("Erledigt. Projekt „Biochemie“ ist angelegt – mit drei Aufgaben.", 3.0)
    result = core.hear(speech_wav(), wake=0.9, session="vs-echo", evidence={"interrupted": "speech"})
    assert result["ok"] is False and "self-echo" in result["reason"]
    assert core.history == []


def test_barge_in_with_owner_speech_during_playback_still_works():
    core, _ = make(heard="Stopp, spiel etwas anderes.")
    core.voice.note_spoken("Hier ist ein langer Satz über das Wetter von morgen.", 3.0)
    result = core.hear(speech_wav(), wake=0.9, session="vs-barge", answer=False, evidence={"interrupted": "speech"})
    assert result["ok"] is True, result


def test_similarity_is_containment_aware():
    assert similarity("Projekt Biochemie ist angelegt", "Erledigt. Projekt Biochemie ist angelegt – mit drei Aufgaben.") >= 0.85
    assert similarity("Wie geht es dir", "Systeme laufen. Was steht an?") < 0.4


# --------------------------------------------------------------------------
# thoughts are never the owner's words
# --------------------------------------------------------------------------

def test_a_thought_is_delivered_as_zeus_never_as_a_user_message():
    core, _ = make()
    with core.bus.subscribe(replay=False) as sub:
        core._deliver("Eine Sache noch: Test.", scope="", backend="thoughts", meta={"source": "zeus_thought", "thought_id": "t1"})
        events = sub.drain()
    assert user_messages(core, events) == []
    messages = [e.payload for e in events if e.type is EventType.MESSAGE]
    assert messages and messages[0]["meta"]["source"] == "zeus_thought"
    assert core.history[-1].role == "assistant"


def test_the_ledger_forgets_outside_its_window():
    clock = [100.0]
    ledger = UtteranceLedger(window_seconds=10, clock=lambda: clock[0])
    ev = UtteranceEvidence(utterance_id="u1", audio=AudioEvidence(fingerprint="abc", envelope="def", frames=50))
    ledger.accept(ev)
    assert ledger.seen(ev)
    clock[0] += 11
    assert ledger.seen(ev) == ""
