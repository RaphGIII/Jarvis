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

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable

from service.events import EventBus, EventType
from service.state import JarvisState
from speech.chunker import Phrase, PhraseChunker
from speech.contracts import Audio, Transcript


@dataclass
class VoiceSettings:
    """Everything the owner can set about voice, with one meaning each.

    ``wake_sensitivity`` is the wake-word *threshold*: the detector's score
    (0..1) must reach it on two consecutive frames.  Lower = more sensitive.
    ``None`` means "use the trained model's recommendation".  It is read by
    the listener and by the Voice Studio test through the same
    :func:`speech.wake_zeus.resolve_threshold`, so both see one number.

    ``volume`` is the playback volume of ZEUS's own speech (0..1).  It is
    applied by the client that plays audio and by nothing else: it never
    touches microphone samples, wake features, scores or thresholds --
    :func:`wake_inputs` is the whole set of settings the wake path may read,
    and a test pins that ``volume`` is not in it.
    """

    enabled: bool = False
    language: str = ""
    voice_id: str = ""
    #: Speak answers aloud.  Separate from `enabled`: a user may want dictation
    #: without Jarvis talking back.
    speak_replies: bool = True
    #: Device names the clients use; free text, informational to the core.
    microphone: str = ""
    output: str = ""
    #: The piper voice for synthesis (empty = engine default).
    voice: str = ""
    wake_sensitivity: float | None = None
    volume: float = 1.0
    #: Utterance gate (see JarvisCore.hear): a transcript below this
    #: confidence, or shorter than this many words, is not a request.
    min_utterance_confidence: float = 0.35
    min_utterance_words: int = 2
    #: The same text heard again within this many seconds is one utterance.
    duplicate_window_seconds: float = 12.0

    #: Settings the wake-word path is allowed to read.  Deliberately a list,
    #: so the exclusion of ``volume`` is a fact a test can assert.
    WAKE_INPUTS = ("wake_sensitivity",)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def apply(self, changes: dict[str, Any]) -> dict[str, str]:
        """Set known fields with type coercion; returns {field: reason} for what was refused."""

        refused: dict[str, str] = {}
        for key, value in changes.items():
            if key not in {f.name for f in fields(self)}:
                refused[key] = "unknown setting"
                continue
            try:
                setattr(self, key, self._coerce(key, value))
            except (TypeError, ValueError) as exc:
                refused[key] = str(exc)
        return refused

    @staticmethod
    def _coerce(key: str, value: Any) -> Any:
        if key in {"enabled", "speak_replies"}:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if key == "wake_sensitivity":
            if value is None or value == "" or value == 0 or value == "0":
                return None  # an empty form field is "not set", not "fire on everything"
            number = float(value)
            if not 0.0 < number <= 1.0:
                raise ValueError("wake sensitivity is a threshold between 0 and 1")
            return number
        if key == "volume":
            number = 1.0 if value in (None, "") else float(value)
            return min(1.0, max(0.0, number))
        if key in {"min_utterance_confidence", "duplicate_window_seconds"}:
            return max(0.0, float(value))
        if key == "min_utterance_words":
            return max(1, int(value))
        return "" if value is None else str(value)

    def wake_inputs(self) -> dict[str, Any]:
        """The only settings the wake-word path reads."""

        return {name: getattr(self, name) for name in self.WAKE_INPUTS}

    # -- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None) -> "VoiceSettings":
        settings = cls()
        if path is None:
            return settings
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return settings
        if isinstance(data, dict):
            settings.apply({k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
        return settings

    def save(self, path: str | Path | None) -> None:
        if path is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, target)


_WORD = re.compile(r"[\w']+", re.UNICODE)
#: Vocatives and fillers that precede a request without being part of it.
_ADDRESS = {"zeus", "jarvis", "hey", "ok", "okay", "hallo", "hi", "du", "bitte"}


def normalise_utterance(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


class UtteranceGate:
    """Decides whether a transcript is a request at all.

    The microphone path is the only way text reaches the core without a
    person typing it, and a wake detector has false activations.  What
    follows a false activation is ambient speech -- a fragment ("Toys.",
    "So is...") that the conversation model dutifully answers, seven times
    in two minutes on the live product.  This gate sits between transcription
    and ``send_message`` so such fragments create no message, no action, no
    receipt and no mission.  Every rejection is reported with a reason so the
    behaviour is observable rather than silent.
    """

    def __init__(self, settings: VoiceSettings, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self._clock = clock
        self._recent: dict[str, float] = {}
        self.rejected: list[dict[str, Any]] = []

    def check(self, transcript: Any, *, authorised: bool) -> tuple[bool, str]:
        """(accept, reason).  ``authorised`` = a wake word or an explicit press produced this audio."""

        text = str(getattr(transcript, "text", "") or "").strip()
        words = _WORD.findall(text)
        # Addressing the assistant is not content: "Zeus, Toys." is one word.
        while len(words) > 1 and words[0].lower() in _ADDRESS:
            words = words[1:]
        confidence = float(getattr(transcript, "confidence", 0.0) or 0.0)
        reason = ""
        if not authorised:
            reason = "no listening session: audio without a wake word or a press is not a request"
        elif not words:
            reason = "nothing heard"
        elif len(words) < self.settings.min_utterance_words and not self._is_complete_short_request(text):
            reason = f"fragment: {len(words)} word(s) is not a request"
        elif len({w.lower() for w in words}) == 1 and len(words) > 1:
            reason = "fragment: one word repeated"
        elif text.endswith("...") or text.endswith("…"):
            reason = "incomplete: the transcript trails off"
        elif 0.0 < confidence < self.settings.min_utterance_confidence:
            reason = f"low confidence {confidence:.2f} < {self.settings.min_utterance_confidence:.2f}"
        else:
            key = normalise_utterance(text)
            now = self._clock()
            last = self._recent.get(key)
            window = self.settings.duplicate_window_seconds
            self._recent = {k: t for k, t in self._recent.items() if now - t <= window}
            if last is not None and now - last <= window:
                reason = f"duplicate: heard {now - last:.1f}s ago"
            else:
                self._recent[key] = now
        if reason:
            self.rejected.append({"text": text[:80], "reason": reason, "confidence": round(confidence, 3), "at": time.time()})
            del self.rejected[:-50]
            return False, reason
        return True, ""

    @staticmethod
    def _is_complete_short_request(text: str) -> bool:
        """One-word requests that are complete: stop, pause, weiter, ja, nein, ..."""

        return normalise_utterance(text) in {
            "stop", "stopp", "halt", "pause", "weiter", "ja", "nein", "yes", "no", "danke", "abbrechen", "cancel",
            "lauter", "leiser", "wiederhole", "nochmal", "hilfe", "help",
            "hallo", "hallo zeus", "hallo jarvis", "hi zeus", "hey zeus", "guten morgen", "guten abend", "gute nacht", "hello",
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
        settings_path: str | Path | None = None,
    ) -> None:
        self.bus = bus
        self.settings_path = Path(settings_path) if settings_path else None
        self.settings = settings or VoiceSettings.load(self.settings_path)
        self.gate = UtteranceGate(self.settings)
        # The pronunciation pipeline: what is *spoken* may differ from what is
        # shown; the owner's lexicon lives beside the voice settings.
        from speech.pronounce import Lexicon, Pronouncer

        lexicon_path = self.settings_path.with_name("lexicon.json") if self.settings_path else None
        self.lexicon = Lexicon(lexicon_path)
        self.pronouncer = Pronouncer(self.lexicon, provider="piper_espeak")
        self.last_spoken: list[dict[str, Any]] = []
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
        return {"settings": self.settings.to_dict(), "engine": engine_status, "cached_audio": len(self.store),
                "settings_path": str(self.settings_path) if self.settings_path else "",
                "rejected_utterances": list(self.gate.rejected[-10:])}

    def update_settings(self, changes: dict[str, Any]) -> dict[str, str]:
        """Apply and persist; returns what was refused (empty when everything took)."""

        refused = self.settings.apply(changes)
        try:
            self.settings.save(self.settings_path)
        except OSError as exc:
            refused["_save"] = str(exc)
        return refused

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

        spoken_forms: dict[str, str] = {}

        def synthesize(text: str) -> Audio:
            # Per phrase, after chunking (so the chunker saw the real text):
            # normalise + lexicon for this provider; the displayed text is
            # never touched -- it left through TOKEN/MESSAGE events already.
            rendered = self.pronouncer.render(text, language=self.settings.language or language_hint())
            spoken_forms[text] = rendered.spoken
            if rendered.changed:
                self.last_spoken.append(rendered.to_dict())
                del self.last_spoken[:-20]
            return self.engine.synthesize(
                rendered.spoken, voice=self.settings.voice_id, language=self.settings.language
            )

        def language_hint() -> str:
            from persona.language import stable_language

            try:
                return stable_language(" ".join(spoken_forms.keys())[-200:], current="", default="de") or "de"
            except Exception:  # noqa: BLE001
                return "de"

        def publish(audio: Audio, phrase: Phrase) -> None:
            key = self.store.put(audio)
            self.bus.publish(
                EventType.SPEECH,
                {
                    "audio_id": key,
                    "url": f"/api/voice/audio/{key}.wav",
                    "text": phrase.text,
                    "spoken": spoken_forms.get(phrase.text, phrase.text),
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
