"""Interfaces for the speech components, and the registry that swaps them.

Speech is the part of Jarvis most likely to be replaced piecemeal.  A better
local TTS voice appears; whisper gets faster; the wake word moves onto a
low-power device while recognition stays on the server.  So the contracts here
are narrow on purpose -- audio in, text out; text in, audio out -- and carry no
assumption about where the implementation runs.

That last point is not hypothetical.  On this machine the engines run inside a
separate virtual environment (torch, ctranslate2 and onnxruntime must not be
forced into the system Python), so the "implementation" is a subprocess speaking
JSON over a pipe.  In the planned topology it becomes a small device across the
network.  Neither of those is visible through these protocols, which is the test
of whether the boundary is in the right place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class Transcript:
    """What was heard."""

    text: str
    language: str = ""
    #: 0..1 where the engine reports one.
    confidence: float = 0.0
    duration_seconds: float = 0.0
    #: False while the user is still speaking; True once the phrase is settled.
    final: bool = True
    #: Word timings from the recogniser when it gives them: [{word, start, end, probability}].
    words: list = field(default_factory=list)
    #: The recogniser's own quality signals, when it reports them:
    #: no_speech_probability, avg_logprob, compression_ratio, language_probability, elapsed.
    quality: dict = field(default_factory=dict)
    #: What was heard before any normalisation touched it ("" = same as text).
    raw_text: str = ""

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        out = {
            "text": self.text,
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "final": self.final,
        }
        if self.quality:
            out["quality"] = dict(self.quality)
        if self.raw_text and self.raw_text != self.text:
            out["raw_text"] = self.raw_text
        return out


@dataclass
class Audio:
    """Mono PCM, the only format that crosses these boundaries.

    Deliberately not a file path and not an encoded blob: every component here
    either produces or consumes raw samples, and passing WAV containers around
    would mean each one re-implementing header parsing.
    """

    samples: bytes
    sample_rate: int = 22050
    #: Bytes per sample. 2 = 16-bit signed, which is what everything here uses.
    width: int = 2

    @property
    def seconds(self) -> float:
        if not self.sample_rate or not self.width:
            return 0.0
        return len(self.samples) / self.width / self.sample_rate

    def to_wav(self) -> bytes:
        """Wrap in a WAV container, for playback and for writing to disk."""

        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(self.width)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.samples)
        return buffer.getvalue()

    @classmethod
    def from_wav(cls, data: bytes) -> "Audio":
        import io
        import wave

        with wave.open(io.BytesIO(data), "rb") as handle:
            return cls(
                samples=handle.readframes(handle.getnframes()),
                sample_rate=handle.getframerate(),
                width=handle.getsampwidth(),
            )


@dataclass
class Voice:
    """One installed TTS voice."""

    id: str
    name: str
    language: str
    #: "male" | "female" | "neutral" | "" when unknown.
    gender: str = ""
    engine: str = "piper"
    model_path: str = ""
    #: Where the voice came from, and under what terms.  Recorded because the
    #: brief requires voice licences to be respected, and a registry that
    #: forgets provenance cannot answer whether a voice may be redistributed.
    licence: str = ""
    source: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "engine": self.engine,
            "licence": self.licence,
            "source": self.source,
            "installed": bool(self.model_path) and Path(self.model_path).exists(),
        }


@runtime_checkable
class SpeechToText(Protocol):
    def transcribe(self, audio: Audio, *, language: str = "") -> Transcript:
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """A TTS provider: text (already rendered for this provider) in, audio out."""

    def synthesize(self, text: str, *, voice: str = "", language: str = "") -> Audio:
        ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def is_speech(self, audio: Audio) -> bool:
        ...


class VoiceRegistry:
    """The installed voices, so a new one is configuration rather than code.

    Backed by a JSON file listing model paths.  A user who downloads another
    Piper voice adds it here and selects it; nothing in Jarvis core changes,
    which is the requirement.
    """

    def __init__(self, path: str | Path, *, voices_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.voices_dir = Path(voices_dir) if voices_dir else self.path.parent
        self._voices: dict[str, Voice] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in data.get("voices", []) if isinstance(data, dict) else []:
            try:
                voice = Voice(**item)
            except TypeError:
                continue
            self._voices[voice.id] = voice

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"voices": [voice.__dict__ for voice in self._voices.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, voice: Voice) -> Voice:
        self._voices[voice.id] = voice
        self.save()
        return voice

    def remove(self, voice_id: str) -> None:
        self._voices.pop(voice_id, None)
        self.save()

    def get(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)

    def all(self) -> list[Voice]:
        return list(self._voices.values())

    def installed(self) -> list[Voice]:
        return [voice for voice in self._voices.values() if Path(voice.model_path).exists()]

    def for_language(self, language: str) -> Voice | None:
        """The best installed voice for a language, preferring the default one."""

        wanted = (language or "").lower().split("-")[0]
        candidates = [
            voice
            for voice in self.installed()
            if voice.language.lower().split("-")[0] == wanted
        ]
        if not candidates:
            return None
        # Prefer male, per the configured persona, then fall back to whatever
        # is installed rather than refusing to speak.
        for voice in candidates:
            if voice.gender == "male":
                return voice
        return candidates[0]

    def discover(self) -> list[Voice]:
        """Register any Piper voice sitting in the voices directory.

        Filesystem-first so that dropping a downloaded ``.onnx`` next to its
        ``.onnx.json`` is enough to install a voice, which is how people
        actually obtain them.
        """

        found: list[Voice] = []
        if not self.voices_dir.is_dir():
            return found
        for model in sorted(self.voices_dir.glob("*.onnx")):
            config = model.with_suffix(".onnx.json")
            if not config.is_file():
                continue
            voice_id = model.stem
            if voice_id in self._voices:
                continue
            language = voice_id.split("-")[0] if "-" in voice_id else ""
            found.append(
                self.add(
                    Voice(
                        id=voice_id,
                        name=voice_id.replace("_", " "),
                        language=language.replace("_", "-"),
                        model_path=str(model),
                        engine="piper",
                    )
                )
            )
        return found
