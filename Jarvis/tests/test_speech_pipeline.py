"""The streaming speech pipeline, with fake engines so the timing is testable.

The properties being pinned are the ones that make voice feel like conversation
rather than like a form submission: audio starts while the model is still
generating, and Jarvis shuts up the instant it is spoken to.
"""

from __future__ import annotations

import threading
import time

import pytest

from speech.chunker import PhraseChunker
from speech.contracts import Audio, Transcript, Voice, VoiceRegistry
from speech.pipeline import NullSink, SpeechMetrics, StreamingSpeaker


def fake_synthesize(text: str) -> Audio:
    """One byte of PCM per character, so audio length tracks text length."""

    return Audio(samples=b"\x00\x00" * max(1, len(text)), sample_rate=1000)


def slow_tokens(pieces, delay=0.02):
    for piece in pieces:
        time.sleep(delay)
        yield piece


# --------------------------------------------------------------------------
# Speaking before the answer is finished
# --------------------------------------------------------------------------

def test_audio_starts_before_the_token_stream_ends():
    """The whole point of the pipeline."""

    audio_times = []
    generation_done = []

    def tokens():
        yield "Guten Morgen. "
        for _ in range(20):
            time.sleep(0.02)
            yield "noch ein Wort "
        generation_done.append(time.perf_counter())

    speaker = StreamingSpeaker(
        fake_synthesize,
        sink=NullSink(),
        on_audio=lambda audio, phrase: audio_times.append(time.perf_counter()),
    )
    speaker.speak_stream(tokens())

    assert audio_times, "no audio was produced"
    assert generation_done, "the generator never completed"
    assert audio_times[0] < generation_done[0], (
        "the first audio must exist before generation finished"
    )


def test_the_first_phrase_is_the_opening_sentence():
    spoken = []
    speaker = StreamingSpeaker(fake_synthesize, sink=NullSink(), on_phrase=spoken.append)

    speaker.speak_stream(["Guten Morgen. ", "Der Rest des Satzes kommt später und ist länger."])

    assert spoken[0].text == "Guten Morgen."
    assert spoken[0].first


def test_metrics_record_time_to_first_audio():
    speaker = StreamingSpeaker(fake_synthesize, sink=NullSink())

    metrics = speaker.speak_stream(slow_tokens(["Hallo. ", "Und noch ein längerer Satz dazu hier."]))

    data = metrics.to_dict()
    assert data["time_to_first_audio"] is not None
    assert data["time_to_first_audio"] <= data["total_seconds"]
    assert data["phrases"] >= 1


def test_everything_gets_spoken_in_order():
    sink = NullSink()
    spoken = []
    speaker = StreamingSpeaker(fake_synthesize, sink=sink, on_phrase=spoken.append)

    speaker.speak_stream(["Eins. ", "Zwei drei vier fünf sechs sieben acht. ", "Neun zehn elf zwölf dreizehn."])

    assert len(sink.played) == len(spoken)
    assert spoken[0].text == "Eins."


def test_a_single_string_is_still_chunked_and_spoken():
    sink = NullSink()
    speaker = StreamingSpeaker(fake_synthesize, sink=sink)

    speaker.speak("Ein vollständiger Satz. Und noch ein zweiter Satz mit etwas mehr Inhalt darin.")

    assert len(sink.played) >= 2


def test_nothing_is_spoken_for_empty_output():
    sink = NullSink()
    speaker = StreamingSpeaker(fake_synthesize, sink=sink)

    speaker.speak_stream([])

    assert sink.played == []


# --------------------------------------------------------------------------
# Barge-in
# --------------------------------------------------------------------------

def test_interrupting_stops_playback_promptly():
    """A voice assistant that cannot be told to stop is unusable."""

    sink = NullSink(realtime=True)
    speaker = StreamingSpeaker(fake_synthesize, sink=sink)

    # 4000 samples at 1000 Hz = 4 seconds of "audio" per phrase.
    text = "Ein sehr langer Satz der lange dauert und viele Zeichen hat damit er lange spielt. " * 3

    def interrupt_soon():
        time.sleep(0.25)
        speaker.interrupt()

    threading.Thread(target=interrupt_soon, daemon=True).start()
    started = time.perf_counter()
    speaker.speak_stream([text])
    elapsed = time.perf_counter() - started

    assert speaker.interrupted
    assert elapsed < 3.0, f"took {elapsed:.1f}s to stop; playback was not interruptible"


def test_interruption_abandons_phrases_that_have_not_been_spoken():
    """Interrupting mid-answer must drop the rest, not merely stop the current phrase."""

    sink = NullSink()
    speaker = StreamingSpeaker(fake_synthesize, sink=sink)

    def tokens():
        yield "Erster Satz. "
        # By now the first phrase is on its way to the sink.
        time.sleep(0.15)
        speaker.interrupt()
        for _ in range(30):
            yield "weiterer Satz der nicht mehr gesprochen werden darf. "

    speaker.speak_stream(tokens())

    assert speaker.interrupted
    assert len(sink.played) <= 1, f"kept speaking after interruption: {len(sink.played)} phrases"


def test_a_new_utterance_clears_a_previous_interruption():
    sink = NullSink()
    speaker = StreamingSpeaker(fake_synthesize, sink=sink)
    speaker.interrupt()
    speaker.speak_stream(["ignoriert"])

    speaker.speak_stream(["Jetzt wieder normal sprechen bitte, das ist ein neuer Satz."])

    assert sink.played, "the next utterance must not inherit the interruption"
    assert not speaker.interrupted


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_a_phrase_that_will_not_synthesize_does_not_silence_the_rest():
    calls = []

    def flaky(text: str) -> Audio:
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("engine hiccup")
        return fake_synthesize(text)

    sink = NullSink()
    speaker = StreamingSpeaker(flaky, sink=sink)

    speaker.speak_stream(["Erster Satz. ", "Zweiter Satz der auch gesprochen werden soll hier."])

    assert len(calls) >= 2
    assert sink.played, "the remaining phrases must still be spoken"


def test_a_sink_that_raises_does_not_kill_the_pipeline():
    class BrokenSink(NullSink):
        def play(self, audio, *, should_stop):
            raise RuntimeError("no sound card")

    speaker = StreamingSpeaker(fake_synthesize, sink=BrokenSink())

    metrics = speaker.speak_stream(["Ein Satz. ", "Noch ein Satz mit genügend Inhalt darin."])

    assert metrics.phrases >= 1


def test_synthesis_runs_ahead_of_playback():
    """While phrase one plays, phrase two should already be generated."""

    synthesized: list[float] = []
    speaker = StreamingSpeaker(
        lambda text: (synthesized.append(time.perf_counter()), fake_synthesize(text))[1],
        sink=NullSink(realtime=True),
    )

    text = "Erster Satz hier mit Inhalt. Zweiter Satz hier mit Inhalt. Dritter Satz hier mit Inhalt."
    speaker.speak_stream([text])

    assert len(synthesized) >= 2
    # All synthesis finishes well inside the total playback time.
    assert synthesized[-1] - synthesized[0] < 1.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_metrics_are_empty_before_anything_happens():
    assert SpeechMetrics().to_dict()["time_to_first_audio"] is None


def test_the_realtime_factor_is_reported():
    speaker = StreamingSpeaker(fake_synthesize, sink=NullSink())

    metrics = speaker.speak_stream(["Ein Satz mit genügend Zeichen um Audio zu erzeugen."])

    assert metrics.to_dict()["realtime_factor"] is not None


# --------------------------------------------------------------------------
# Audio container
# --------------------------------------------------------------------------

def test_audio_round_trips_through_wav():
    original = Audio(samples=b"\x01\x02" * 100, sample_rate=16000)

    restored = Audio.from_wav(original.to_wav())

    assert restored.samples == original.samples
    assert restored.sample_rate == 16000


def test_audio_duration_is_computed_from_the_samples():
    assert Audio(samples=b"\x00\x00" * 16000, sample_rate=16000).seconds == pytest.approx(1.0)


def test_an_empty_transcript_knows_it_is_empty():
    assert Transcript(text="   ").empty
    assert not Transcript(text="hallo").empty


# --------------------------------------------------------------------------
# Voice registry
# --------------------------------------------------------------------------

def test_a_voice_dropped_into_the_directory_is_discovered(tmp_path):
    """Installing a voice must be a file copy, not a code change."""

    (tmp_path / "de_DE-thorsten-medium.onnx").write_bytes(b"fake model")
    (tmp_path / "de_DE-thorsten-medium.onnx.json").write_text("{}", encoding="utf-8")
    registry = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)

    found = registry.discover()

    assert [voice.id for voice in found] == ["de_DE-thorsten-medium"]
    assert registry.get("de_DE-thorsten-medium").language == "de-DE"


def test_a_model_without_its_config_is_not_registered(tmp_path):
    (tmp_path / "orphan.onnx").write_bytes(b"fake")
    registry = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)

    assert registry.discover() == []


def test_voices_survive_a_restart(tmp_path):
    registry = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)
    registry.add(Voice(id="v1", name="One", language="de-DE", model_path=str(tmp_path / "v1.onnx")))

    reloaded = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)

    assert reloaded.get("v1") is not None


def test_a_male_voice_is_preferred_for_the_persona(tmp_path):
    for name, gender in (("f", "female"), ("m", "male")):
        path = tmp_path / f"{name}.onnx"
        path.write_bytes(b"x")
        VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path).add(
            Voice(id=name, name=name, language="de-DE", gender=gender, model_path=str(path))
        )
    registry = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)

    assert registry.for_language("de").gender == "male"


def test_a_language_with_no_voice_returns_nothing(tmp_path):
    registry = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)

    assert registry.for_language("ja") is None


def test_licence_and_source_are_recorded(tmp_path):
    """A registry that forgets provenance cannot answer redistribution questions."""

    registry = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)
    registry.add(
        Voice(id="v", name="V", language="de-DE", licence="CC BY 4.0", source="rhasspy/piper-voices")
    )

    reloaded = VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path)
    assert reloaded.get("v").licence == "CC BY 4.0"


def test_a_corrupt_registry_file_does_not_crash(tmp_path):
    (tmp_path / "voices.json").write_text("{not json", encoding="utf-8")

    assert VoiceRegistry(tmp_path / "voices.json", voices_dir=tmp_path).all() == []
