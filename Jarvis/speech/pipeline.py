"""Turning a token stream into speech that starts before the thought finishes.

The pipeline is three stages joined by queues:

    tokens -> PhraseChunker -> synthesis worker -> playback worker

Both workers are single-threaded and ordered, which is the point.  Synthesis
runs *ahead* of playback: while phrase one is being spoken, phrase two is
already being generated, so after the first phrase the audio never has to wait
for the vocoder.  Since Piper synthesises at roughly RTF 0.09 on this machine
and speech plays at RTF 1.0 by definition, one worker keeps ahead comfortably --
a pool would add concurrency bugs to buy nothing.

Interruption is the requirement that shapes everything else.  Barge-in has to
stop audio *now*, not at the end of the current phrase, so playback is fed small
slices and checks a stop flag between them.  A design that handed a whole
utterance to the sound device would be unable to shut up when spoken to, which
is the single most important thing a voice assistant must do.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from speech.chunker import Phrase, PhraseChunker
from speech.contracts import Audio


@dataclass
class SpeechMetrics:
    """Latency, measured rather than asserted."""

    started_at: float = 0.0
    first_token_at: float = 0.0
    first_phrase_at: float = 0.0
    first_audio_at: float = 0.0
    finished_at: float = 0.0
    phrases: int = 0
    audio_seconds: float = 0.0
    synthesis_seconds: float = 0.0

    def _delta(self, mark: float) -> float | None:
        if not mark or not self.started_at:
            return None
        return round(mark - self.started_at, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_to_first_token": self._delta(self.first_token_at),
            "time_to_first_phrase": self._delta(self.first_phrase_at),
            "time_to_first_audio": self._delta(self.first_audio_at),
            "total_seconds": self._delta(self.finished_at),
            "phrases": self.phrases,
            "audio_seconds": round(self.audio_seconds, 2),
            "synthesis_seconds": round(self.synthesis_seconds, 2),
            "realtime_factor": (
                round(self.synthesis_seconds / self.audio_seconds, 3) if self.audio_seconds else None
            ),
        }


class AudioSink:
    """Where synthesized audio goes.  Replaceable for tests and for devices."""

    def play(self, audio: Audio, *, should_stop: Callable[[], bool]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class NullSink(AudioSink):
    """Discards audio, but honours timing.  Used by tests and headless runs."""

    def __init__(self, *, realtime: bool = False) -> None:
        self.realtime = realtime
        self.played: list[Audio] = []

    def play(self, audio: Audio, *, should_stop: Callable[[], bool]) -> None:
        self.played.append(audio)
        if not self.realtime:
            return
        deadline = time.perf_counter() + audio.seconds
        while time.perf_counter() < deadline:
            if should_stop():
                return
            time.sleep(0.02)


class SoundDeviceSink(AudioSink):
    """Plays through the default output device, in interruptible slices.

    Lives in the speech virtualenv's dependency set (``sounddevice``), so it is
    imported lazily and its absence is not fatal -- Jarvis without a speaker is
    still Jarvis.
    """

    #: How much audio to hand the device at a time.  Small enough that
    #: barge-in feels instant, large enough not to underrun.
    SLICE_SECONDS = 0.10

    def __init__(self) -> None:
        self._stream: Any = None

    def play(self, audio: Audio, *, should_stop: Callable[[], bool]) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "sounddevice/numpy are not importable in this interpreter; "
                "play through the speech worker or install them here"
            ) from exc

        samples = np.frombuffer(audio.samples, dtype=np.int16)
        if samples.size == 0:
            return
        step = max(1, int(audio.sample_rate * self.SLICE_SECONDS))
        with sd.OutputStream(samplerate=audio.sample_rate, channels=1, dtype="int16") as stream:
            for start in range(0, samples.size, step):
                if should_stop():
                    return
                stream.write(samples[start : start + step])


class StreamingSpeaker:
    """Speaks a token stream, and can be told to stop mid-word."""

    def __init__(
        self,
        synthesize: Callable[[str], Audio],
        *,
        sink: AudioSink | None = None,
        chunker: PhraseChunker | None = None,
        on_phrase: Callable[[Phrase], None] | None = None,
        on_audio: Callable[[Audio, Phrase], None] | None = None,
    ) -> None:
        self._synthesize = synthesize
        self.sink = sink or NullSink()
        self.chunker = chunker or PhraseChunker()
        self.on_phrase = on_phrase
        self.on_audio = on_audio
        self._stop = threading.Event()
        self.metrics = SpeechMetrics()

    # -- control ---------------------------------------------------------

    def interrupt(self) -> None:
        """Barge-in.  Stops playback and abandons everything still queued."""

        self._stop.set()
        self.sink.stop()

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

    # -- the pipeline ----------------------------------------------------

    def speak_stream(self, tokens: Iterable[str]) -> SpeechMetrics:
        """Consume tokens, speaking each phrase as soon as it is complete."""

        self._stop.clear()
        self.chunker.reset()
        self.metrics = SpeechMetrics(started_at=time.perf_counter())

        to_synthesize: queue.Queue[Phrase | None] = queue.Queue()
        to_play: queue.Queue[tuple[Audio, Phrase] | None] = queue.Queue()

        synth = threading.Thread(
            target=self._synthesis_worker, args=(to_synthesize, to_play), daemon=True, name="tts-synth"
        )
        play = threading.Thread(
            target=self._playback_worker, args=(to_play,), daemon=True, name="tts-play"
        )
        synth.start()
        play.start()

        try:
            for token in tokens:
                if self._stop.is_set():
                    break
                if not self.metrics.first_token_at:
                    self.metrics.first_token_at = time.perf_counter()
                for phrase in self.chunker.feed(token):
                    self._enqueue(phrase, to_synthesize)
            if not self._stop.is_set():
                for phrase in self.chunker.flush():
                    self._enqueue(phrase, to_synthesize)
        finally:
            to_synthesize.put(None)
            synth.join(timeout=120)
            to_play.put(None)
            play.join(timeout=300)
            self.metrics.finished_at = time.perf_counter()

        return self.metrics

    def speak(self, text: str) -> SpeechMetrics:
        """Speak a complete string.  Still chunked, so it can be interrupted."""

        return self.speak_stream([text])

    # -- workers ---------------------------------------------------------

    def _enqueue(self, phrase: Phrase, to_synthesize: "queue.Queue[Phrase | None]") -> None:
        if not self.metrics.first_phrase_at:
            self.metrics.first_phrase_at = time.perf_counter()
        self.metrics.phrases += 1
        if self.on_phrase is not None:
            self.on_phrase(phrase)
        to_synthesize.put(phrase)

    def _synthesis_worker(
        self,
        to_synthesize: "queue.Queue[Phrase | None]",
        to_play: "queue.Queue[tuple[Audio, Phrase] | None]",
    ) -> None:
        while True:
            phrase = to_synthesize.get()
            if phrase is None:
                return
            if self._stop.is_set():
                continue
            started = time.perf_counter()
            try:
                audio = self._synthesize(phrase.text)
            except Exception:
                # A phrase that will not synthesize is skipped rather than
                # silencing the rest of the answer.
                continue
            self.metrics.synthesis_seconds += time.perf_counter() - started
            self.metrics.audio_seconds += audio.seconds
            if self.on_audio is not None:
                self.on_audio(audio, phrase)
            to_play.put((audio, phrase))

    def _playback_worker(self, to_play: "queue.Queue[tuple[Audio, Phrase] | None]") -> None:
        while True:
            item = to_play.get()
            if item is None:
                return
            if self._stop.is_set():
                continue
            audio, _phrase = item
            if not self.metrics.first_audio_at:
                self.metrics.first_audio_at = time.perf_counter()
            try:
                self.sink.play(audio, should_stop=self._stop.is_set)
            except Exception:
                continue
