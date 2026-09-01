"""Voice over the service boundary: audio in over HTTP, audio out by reference.

The design under test is that capture and playback live in a *client* -- the
browser now, a small HDMI box later -- while recognition, thinking and synthesis
stay in the core. These tests use a fake engine so the boundary itself is what
is being checked, not whisper's accuracy.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from service.core import JarvisCore
from service.events import EventBus, EventType
from service.http import JarvisHTTPServer
from service.voice import AudioStore, VoiceService, VoiceSettings
from speech.contracts import Audio, Transcript


class FakeEngine:
    """Deterministic stand-in for whisper + piper."""

    def __init__(self, *, heard="hallo jarvis", fail=False):
        self.heard = heard
        self.fail = fail
        self.synthesized: list[str] = []

    def status(self):
        return {"available": True, "voices": [{"id": "test-voice"}]}

    def transcribe(self, audio, *, language=""):
        if self.fail:
            raise RuntimeError("engine exploded")
        return Transcript(text=self.heard, language=language or "de", confidence=0.9)

    def synthesize(self, text, *, voice="", language=""):
        self.synthesized.append(text)
        return Audio(samples=b"\x00\x00" * max(1, len(text)), sample_rate=1000)


class StubKernel:
    def __init__(self, chunks=("Guten ", "Tag. ", "Alles bereit hier soweit ich sehe.")):
        self.chunks = chunks
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        chunks = self.chunks

        class P:
            def generate_stream(self, prompt, **_):
                yield from chunks

        return P()


def wav_bytes(seconds=1.2, rate=16000):
    # Speech-like samples: the acceptance gate measures the audio, and a
    # silent buffer is refused before the (fake) recogniser is even asked.
    from _audio import speech_wav

    return speech_wav(seconds)


# --------------------------------------------------------------------------
# Audio store
# --------------------------------------------------------------------------

def test_audio_is_retrievable_by_id():
    store = AudioStore()
    key = store.put(Audio(samples=b"\x01\x02"))

    assert store.get(key).samples == b"\x01\x02"


def test_the_store_is_bounded():
    """A voice session must not become a memory leak measured in megabytes."""

    store = AudioStore(limit=4)
    keys = [store.put(Audio(samples=b"\x00" * 100)) for _ in range(20)]

    assert len(store) == 4
    assert store.get(keys[0]) is None, "the oldest audio must have been evicted"
    assert store.get(keys[-1]) is not None


def test_an_unknown_id_returns_nothing():
    assert AudioStore().get("nope") is None


# --------------------------------------------------------------------------
# Hearing
# --------------------------------------------------------------------------

def test_a_posted_utterance_becomes_a_transcript_event():
    """The TRANSCRIPT event carries the gate's verdict: it is published by
    hear(), after acceptance, never by the recogniser boundary itself --
    a rejected hallucination must not reach the owner's screen as text."""

    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine(heard="wie spät ist es"))

    with bus.subscribe(replay=False) as subscription:
        transcript = service.transcribe(wav_bytes())
        events = subscription.drain()
    assert transcript.text == "wie spät ist es"
    assert not any(e.type is EventType.TRANSCRIPT for e in events), "no transcript before the verdict"

    core = JarvisCore(kernel=StubKernel())
    core._voice = VoiceService(core.bus, engine_factory=lambda: FakeEngine(heard="wie spät ist es"))
    with core.bus.subscribe(replay=False) as subscription:
        core.hear(wav_bytes(), origin="ui", answer=False)
        published = [e.payload for e in subscription.drain() if e.type is EventType.TRANSCRIPT]
    assert published and published[0]["accepted"] is True and published[0]["text"].lower().startswith("wie spät ist es")


def test_transcribing_announces_the_state_first():
    """The eye should show LISTENING/TRANSCRIBING while it happens."""

    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine())

    with bus.subscribe(replay=False) as subscription:
        service.transcribe(wav_bytes())
        states = [e.payload.get("state") for e in subscription.drain() if e.type is EventType.STATE]

    assert "transcribing" in states


def test_unreadable_audio_is_reported_not_raised():
    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine())

    with bus.subscribe(replay=False) as subscription:
        transcript = service.transcribe(b"this is not a wav file")
        errors = [e for e in subscription.drain() if e.type is EventType.ERROR]

    assert transcript.empty
    assert errors


def test_an_engine_failure_degrades_to_silence_not_a_crash():
    """Speech failing must never take Jarvis down with it."""

    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine(fail=True))

    transcript = service.transcribe(wav_bytes())

    assert transcript.empty


# --------------------------------------------------------------------------
# Speaking
# --------------------------------------------------------------------------

def test_each_phrase_is_published_as_its_own_audio_reference():
    bus = EventBus()
    engine = FakeEngine()
    service = VoiceService(bus, engine_factory=lambda: engine)

    with bus.subscribe(replay=False) as subscription:
        service.speak_stream(["Guten Tag. ", "Hier ist noch ein zweiter Satz mit genug Inhalt."])
        speech = [e.payload for e in subscription.drain() if e.type is EventType.SPEECH]

    assert len(speech) >= 2
    assert speech[0]["text"] == "Guten Tag."
    assert speech[0]["first"] is True
    assert speech[0]["url"].startswith("/api/voice/audio/")


def test_audio_is_referenced_not_embedded():
    """Megabytes of PCM must not travel through the event stream."""

    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine())

    with bus.subscribe(replay=False) as subscription:
        service.speak_stream(["Ein Satz hier."])
        speech = [e.payload for e in subscription.drain() if e.type is EventType.SPEECH]

    payload = json.dumps(speech[0])
    assert len(payload) < 400, "the event should carry a URL, not the samples"


def test_the_audio_is_actually_in_the_store():
    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine())

    with bus.subscribe(replay=False) as subscription:
        service.speak_stream(["Ein vollständiger Satz."])
        speech = [e.payload for e in subscription.drain() if e.type is EventType.SPEECH]

    assert service.store.get(speech[0]["audio_id"]) is not None


def test_interrupting_publishes_a_stop_so_the_client_falls_silent():
    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine())

    with bus.subscribe(replay=False) as subscription:
        service.interrupt()
        speech = [e.payload for e in subscription.drain() if e.type is EventType.SPEECH]

    assert speech and speech[0].get("stop") is True


def test_speaking_reports_latency_metrics():
    bus = EventBus()
    service = VoiceService(bus, engine_factory=lambda: FakeEngine())

    metrics = service.speak_stream(["Ein Satz. ", "Und noch einer mit deutlich mehr Inhalt darin."])

    assert metrics["phrases"] >= 1
    assert metrics["time_to_first_audio"] is not None


# --------------------------------------------------------------------------
# Voice mode
# --------------------------------------------------------------------------

def test_replies_are_not_spoken_before_voice_is_used():
    """Dictating once must not make every later typed message talk back."""

    core = JarvisCore(kernel=StubKernel())
    engine = FakeEngine()
    core._voice = VoiceService(core.bus, engine_factory=lambda: engine)

    core.send_message("hallo")
    _wait_for_reply(core)

    assert engine.synthesized == []


def test_speaking_to_jarvis_enters_voice_mode_and_it_answers_aloud():
    core = JarvisCore(kernel=StubKernel())
    engine = FakeEngine(heard="wie geht es weiter")
    core._voice = VoiceService(core.bus, engine_factory=lambda: engine)

    result = core.hear(wav_bytes(), origin="ui")
    _wait_for_reply(core)

    assert result["ok"] and result["text"] == "Wie geht es weiter?" and result["raw_text"] == "wie geht es weiter"
    assert core._voice.settings.enabled
    assert engine.synthesized, "the reply should have been spoken"


def test_silence_is_reported_rather_than_answered():
    core = JarvisCore(kernel=StubKernel())
    core._voice = VoiceService(core.bus, engine_factory=lambda: FakeEngine(heard="   "))

    result = core.hear(wav_bytes())

    assert result["ok"] is False
    assert core.history == [], "an empty transcript must not become a question"


def test_speak_replies_can_be_turned_off_while_staying_in_voice_mode():
    core = JarvisCore(kernel=StubKernel())
    engine = FakeEngine()
    core._voice = VoiceService(
        core.bus, engine_factory=lambda: engine, settings=VoiceSettings(enabled=True, speak_replies=False)
    )

    core.send_message("hallo")
    _wait_for_reply(core)

    assert engine.synthesized == []


def test_the_displayed_text_and_the_spoken_text_come_from_one_generation():
    """Two generations would cost double and could drift apart audibly."""

    core = JarvisCore(kernel=StubKernel())
    engine = FakeEngine()
    core._voice = VoiceService(core.bus, engine_factory=lambda: engine, settings=VoiceSettings(enabled=True))

    core.send_message("hallo")
    _wait_for_reply(core)

    spoken = " ".join(engine.synthesized).replace(" ", "")
    displayed = core.history[-1].text.replace(" ", "")
    assert spoken == displayed


def test_stopping_interrupts_the_speaker_too():
    core = JarvisCore(kernel=StubKernel())
    core._voice = VoiceService(core.bus, engine_factory=lambda: FakeEngine(), settings=VoiceSettings(enabled=True))

    with core.bus.subscribe(replay=False) as subscription:
        core.stop_current()
        speech = [e.payload for e in subscription.drain() if e.type is EventType.SPEECH]

    assert any(item.get("stop") for item in speech)


def _wait_for_reply(core, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(turn.role == "assistant" for turn in core.history):
            time.sleep(0.15)  # let the speech pipeline drain
            return
        time.sleep(0.05)


# --------------------------------------------------------------------------
# Over HTTP
# --------------------------------------------------------------------------

@pytest.fixture()
def server():
    core = JarvisCore(kernel=StubKernel())
    core._voice = VoiceService(core.bus, engine_factory=lambda: FakeEngine(heard="hallo jarvis"))
    instance = JarvisHTTPServer(core, port=0, token="tok")
    instance.start()
    yield instance
    instance.stop()


def post_audio(server, data, path="/api/voice/utterance"):
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/octet-stream", "X-Jarvis-Token": "tok"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode())


def test_an_utterance_can_be_posted_as_raw_bytes(server):
    result = post_audio(server, wav_bytes(), path="/api/voice/utterance?origin=ui")

    assert result["ok"] is True
    assert result["text"] == "Hallo jarvis." and result["raw_text"] == "hallo jarvis"


def test_posting_audio_requires_the_token(server):
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}/api/voice/utterance",
        data=wav_bytes(),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)

    assert caught.value.code == 401


def test_synthesized_audio_can_be_fetched_back(server):
    server.core.voice.settings.enabled = True
    key = server.core.voice.store.put(Audio(samples=b"\x00\x00" * 100, sample_rate=8000))

    request = urllib.request.Request(
        f"http://{server.host}:{server.port}/api/voice/audio/{key}.wav",
        headers={"X-Jarvis-Token": "tok"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type")

    assert content_type == "audio/wav"
    assert body.startswith(b"RIFF")


def test_expired_audio_reports_404_rather_than_failing_oddly(server):
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}/api/voice/audio/deadbeef.wav",
        headers={"X-Jarvis-Token": "tok"},
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)

    assert caught.value.code == 404


def test_voice_settings_can_be_read_and_changed(server):
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}/api/voice",
        data=json.dumps({"language": "de", "speak_replies": False}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Jarvis-Token": "tok"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode())

    assert payload["settings"]["language"] == "de"
    assert payload["settings"]["speak_replies"] is False
