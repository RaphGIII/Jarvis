"""Voice as a service: audio arrives over the wire, audio goes back over the wire.

The obvious implementation would open the microphone in the Jarvis process.
This one deliberately does not, because the topology the brief is aiming at puts
capture and playback somewhere else entirely -- a browser today, a small
HDMI-connected box by the television later, a phone after that -- while the
model, the memory and the projects stay on one machine.

So the boundary is: a *client* captures an utterance and posts it; the core
transcribes, thinks, and publishes speech back as events the client plays.  The
browser is then just the first device client rather than a special case, and
nothing has to be rebuilt when the real one arrives.

Synthesised audio is held in a small bounded store and referenced by id.  The
alternative -- base64 inside the event -- would put megabytes of PCM through the
same SSE stream that carries state changes, where one long answer would delay
every subsequent event behind it.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from service.events import EventBus, EventType
from service.state import JarvisState
from speech.chunker import Phrase, PhraseChunker
from speech.contracts import Audio, Transcript


@dataclass
class VoiceSettings:
    enabled: bool = False
    language: str = ""
    voice_id: str = ""
    #: Speak answers aloud.  Separate from `enabled`: a user may want dictation
    #: without Jarvis talking back.
    speak_replies: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "language": self.language,
            "voice_id": self.voice_id,
            "speak_replies": self.speak_replies,
        }


class AudioStore:
    """Recently synthesised audio, addressable by id.

    Bounded and FIFO.  Audio is only useful for the few seconds between being
    generated and being played; keeping it beyond that would turn a voice
    session into a memory leak measured in megabytes per minute.
    """

    def __init__(self, *, limit: int = 64) -> None:
        self._items: OrderedDict[str, Audio] = OrderedDict()
        self._lock = threading.Lock()
        self.limit = limit

    def put(self, audio: Audio) -> str:
        key = uuid.uuid4().hex[:16]
        with self._lock:
            self._items[key] = audio
            while len(self._items) > self.limit:
                self._items.popitem(last=False)
        return key

    def get(self, key: str) -> Audio | None:
        with self._lock:
            return self._items.get(key)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class VoiceService:
    """Transcription in, synthesis out, both as events on the bus."""

    def __init__(
        self,
        bus: EventBus,
        *,
        engine_factory: Callable[[], Any] | None = None,
        settings: VoiceSettings | None = None,
    ) -> None:
        self.bus = bus
        self.settings = settings or VoiceSettings()
        self.store = AudioStore()
        self._engine: Any = None
        self._engine_factory = engine_factory
        self._lock = threading.Lock()
        self._speaker: Any = None

    # -- engine ----------------------------------------------------------

    @property
    def engine(self) -> Any:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    if self._engine_factory is not None:
                        self._engine = self._engine_factory()
                    else:
                        from speech.engine import SpeechEngine

                        self._engine = SpeechEngine()
        return self._engine

    def status(self) -> dict[str, Any]:
        try:
            engine_status = self.engine.status()
        except Exception as exc:
            engine_status = {"available": False, "detail": str(exc), "voices": []}
        return {"settings": self.settings.to_dict(), "engine": engine_status, "cached_audio": len(self.store)}

    # -- hearing ---------------------------------------------------------

    def transcribe(self, wav: bytes, *, language: str = "") -> Transcript:
        """Turn a posted utterance into text, announcing progress as it goes."""

        self.bus.publish(EventType.STATE, {"state": JarvisState.TRANSCRIBING.value, "detail": "listening"})
        try:
            audio = Audio.from_wav(wav)
        except Exception as exc:
            self.bus.publish(EventType.ERROR, {"error": f"unreadable audio: {exc}"})
            return Transcript(text="")

        try:
            transcript = self.engine.transcribe(audio, language=language or self.settings.language)
        except Exception as exc:
            self.bus.publish(EventType.ERROR, {"error": f"transcription failed: {exc}"})
            return Transcript(text="")

        self.bus.publish(EventType.TRANSCRIPT, transcript.to_dict())
        return transcript

    # -- speaking --------------------------------------------------------

    def speak_stream(self, tokens: Iterable[str], *, scope: str = "") -> dict[str, Any]:
        """Speak a token stream, publishing each phrase's audio as it is ready.

        Returns the latency metrics, so the diagnostics view can show what the
        user actually experienced rather than what the design intends.
        """

        from speech.pipeline import StreamingSpeaker

        def synthesize(text: str) -> Audio:
            return self.engine.synthesize(
                text, voice=self.settings.voice_id, language=self.settings.language
            )

        def publish(audio: Audio, phrase: Phrase) -> None:
            key = self.store.put(audio)
            self.bus.publish(
                EventType.SPEECH,
                {
                    "audio_id": key,
                    "url": f"/api/voice/audio/{key}.wav",
                    "text": phrase.text,
                    "first": phrase.first,
                    "seconds": round(audio.seconds, 2),
                },
                scope=scope,
            )

        # The server does not play audio; the client does. Metrics therefore
        # measure up to "audio available", which is the honest thing to claim
        # from here -- the last hop is the device's.
        from speech.pipeline import NullSink

        speaker = StreamingSpeaker(synthesize, sink=NullSink(), on_audio=publish)
        with self._lock:
            self._speaker = speaker

        self.bus.publish(EventType.STATE, {"state": JarvisState.SPEAKING.value, "detail": ""})
        metrics = speaker.speak_stream(tokens)
        with self._lock:
            self._speaker = None
        return metrics.to_dict()

    def interrupt(self) -> None:
        """Barge-in: stop synthesising and tell the client to stop playing."""

        with self._lock:
            speaker = self._speaker
        if speaker is not None:
            speaker.interrupt()
        self.bus.publish(EventType.SPEECH, {"stop": True})

    # -- shutdown ----------------------------------------------------------

    def close(self) -> dict[str, Any]:
        """Stop the speech worker.

        The worker is a separate interpreter holding whisper and piper on the
        GPU, spoken to over pipes.  Nothing used to close it, so every exit --
        including the planned restart of a self-update -- orphaned one, and the
        next start added another.  Closing it here means a restart costs one
        model load rather than one leaked process.
        """

        engine = self._engine
        if engine is None:
            return {"ok": True, "detail": "the speech engine was never started"}
        try:
            engine.close()
        except Exception as exc:  # noqa: BLE001 - a shutdown must not fail on the way out
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._engine = None
        return {"ok": True, "detail": "speech worker stopped"}
