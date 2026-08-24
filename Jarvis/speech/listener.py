"""Hands-free listening: the first Jarvis device client.

Runs inside the speech virtualenv, holds the microphone, and talks to Jarvis
Core over ordinary HTTP.  It is deliberately a *separate process from the core*,
because that is the shape the eventual hardware takes: a small box with a
microphone and a speaker, and the brain somewhere else.  On this machine the
device and the server happen to be the same computer; nothing in this file
assumes that.

    microphone -> wake word -> record until silence -> POST /api/voice/utterance

What runs here rather than on the server, and why:

*Wake word.*  Streaming continuous audio to the core would be wasteful and a
privacy problem -- the box decides locally when Jarvis is being addressed, and
only then does any audio leave the machine.  openWakeWord's ``hey_jarvis`` model
costs 3-5 ms per 80 ms frame, so listening is roughly 5% of one core.

*Endpointing.*  Deciding when the user stopped talking needs the audio stream,
not a copy of it after the fact.  Energy-based silence detection is enough here
and costs nothing; a neural VAD is a drop-in replacement if it proves necessary.

Measured on this machine: the detector scores 0.995 on a correctly pronounced
"Hey Jarvis" and 0.000 on unrelated speech.  It scores 0.04 when the phrase is
pronounced the German way ("YAR-vis") -- the model expects the English "JAR-vis",
which is how the name is said in either language.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any

#: openWakeWord's frame size at 16 kHz.
FRAME_SAMPLES = 1280
SAMPLE_RATE = 16000


@dataclass
class ListenerConfig:
    url: str = "http://127.0.0.1:8420"
    token: str = ""
    #: Filled from the product identity when not given explicitly. Note this
    #: is the MODEL, which may differ from the word the user is told to say --
    #: a detector is trained weights, not a string.
    wake_model: str = ""
    #: Detection threshold.  0.5 is openWakeWord's own default; measured scores
    #: on this machine were 0.995 for a real utterance and 0.000 for unrelated
    #: speech, so the gap is wide and the exact value is not delicate.
    threshold: float = 0.5
    #: Silence that ends an utterance.  Shorter feels abrupt mid-sentence;
    #: longer makes every exchange feel laggy.
    silence_seconds: float = 0.9
    #: Refuse to record forever if the room is noisy.
    max_utterance_seconds: float = 20.0
    #: Ignore an utterance shorter than this: usually a cough or a door.
    min_utterance_seconds: float = 0.35
    #: Frames of audio kept before the wake word fires, so the first syllable
    #: of the question is not clipped off.
    preroll_frames: int = 8
    #: How long to ignore the wake word after firing, so one "Hey Jarvis"
    #: cannot trigger twice.
    cooldown_seconds: float = 1.5
    #: Multiple of the measured noise floor that counts as speech.
    speech_factor: float = 2.5
    device: int | None = None
    verbose: bool = False


class Endpointer:
    """Decides when the user has stopped talking.

    Extracted from the capture loop so it can be tested without a microphone,
    which is the only way to check the cases that matter: a pause mid-sentence
    must not end the utterance, and a noisy room must not prevent it from ever
    ending.

    The threshold tracks a slow estimate of the room rather than being fixed. A
    fixed value is wrong in both directions -- too high in a quiet room and the
    endpoint never fires; too low next to a fan and recording never stops.
    """

    def __init__(self, config: "ListenerConfig", *, frame_seconds: float) -> None:
        self.config = config
        self.frame_seconds = frame_seconds
        self.noise_floor = 0.0
        self.silence_for = 0.0
        self.elapsed = 0.0

    def reset(self) -> None:
        self.silence_for = 0.0
        self.elapsed = 0.0

    def track_noise(self, level: float) -> None:
        self.noise_floor = level if self.noise_floor == 0.0 else (
            self.noise_floor * 0.995 + level * 0.005
        )

    @property
    def threshold(self) -> float:
        # The floor stops a completely silent room from producing a threshold
        # of zero, where dither alone would read as speech.
        return max(self.noise_floor * self.config.speech_factor, 60.0)

    def feed(self, level: float) -> bool:
        """Add one frame; return True when the utterance is over."""

        self.elapsed += self.frame_seconds
        if level < self.threshold:
            self.silence_for += self.frame_seconds
        else:
            self.silence_for = 0.0
        return (
            self.silence_for >= self.config.silence_seconds
            or self.elapsed >= self.config.max_utterance_seconds
        )

    @property
    def speech_seconds(self) -> float:
        """How much of the utterance was not trailing silence."""

        return max(0.0, self.elapsed - self.silence_for)


class WakeListener:
    def __init__(self, config: ListenerConfig) -> None:
        self.config = config
        self._identity_note = ""
        if not config.wake_model:
            try:
                from core.identity import current

                identity = current()
                config.wake_model = identity.resolved_wake_model
                self._identity_note = identity.wake_word_note()
            except Exception:
                config.wake_model = "hey_jarvis"
        self._model: Any = None

    # -- lifecycle -------------------------------------------------------

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "openwakeword is not installed in this interpreter.\n"
                "  .venv-speech\\Scripts\\python -m pip install openwakeword"
            ) from exc

        try:
            openwakeword.utils.download_models([self.config.wake_model])
        except Exception:
            # Already downloaded, or offline. Loading will fail informatively.
            pass
        self._model = Model(wakeword_models=[self.config.wake_model], inference_framework="onnx")
        return self._model

    def run(self) -> int:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"sounddevice/numpy are required: {exc}") from exc

        model = self._load()
        self._say(f"listening for '{self.config.wake_model.replace('_', ' ')}' - Ctrl-C to stop")
        if self._identity_note:
            self._say(f"  note: {self._identity_note}")

        frame_seconds = FRAME_SAMPLES / SAMPLE_RATE
        endpointer = Endpointer(self.config, frame_seconds=frame_seconds)
        preroll: collections.deque = collections.deque(maxlen=self.config.preroll_frames)
        recording: list[Any] = []
        listening = False
        muted_until = 0.0

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES,
            device=self.config.device,
        ) as stream:
            while True:
                block, overflowed = stream.read(FRAME_SAMPLES)
                frame = block.reshape(-1)
                if overflowed and self.config.verbose:
                    self._say("(audio overflow)")

                level = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9)

                if not listening:
                    # The noise floor is only updated while nobody is speaking
                    # to Jarvis. Letting the utterance itself raise it would
                    # make a long answer progressively harder to end.
                    endpointer.track_noise(level)
                    preroll.append(frame.copy())
                    if time.monotonic() < muted_until:
                        continue
                    score = model.predict(frame).get(self.config.wake_model, 0.0)
                    if score >= self.config.threshold:
                        self._say(f"wake ({score:.2f})")
                        listening = True
                        endpointer.reset()
                        # Keep the pre-roll: people run the question straight
                        # into the wake word, so the first syllable is already
                        # gone by the time detection fires.
                        recording = list(preroll)
                        preroll.clear()
                    continue

                recording.append(frame.copy())
                if not endpointer.feed(level):
                    continue

                listening = False
                muted_until = time.monotonic() + self.config.cooldown_seconds
                model.reset()
                audio = np.concatenate(recording) if recording else np.zeros(0, dtype="int16")
                recording = []
                if endpointer.speech_seconds < self.config.min_utterance_seconds:
                    self._say("(too short - ignored)")
                    continue
                self._say(f"heard {endpointer.speech_seconds:.1f}s, sending...")
                self._send(audio.tobytes())

    # -- helpers ---------------------------------------------------------

    def _send(self, pcm: bytes) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm)

        request = urllib.request.Request(
            f"{self.config.url.rstrip('/')}/api/voice/utterance",
            data=buffer.getvalue(),
            method="POST",
            headers={"Content-Type": "application/octet-stream", "X-Jarvis-Token": self.config.token},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            self._say(f"core refused the utterance: HTTP {exc.code}")
            return
        except (urllib.error.URLError, OSError) as exc:
            self._say(f"core unreachable: {exc}")
            return
        if payload.get("ok"):
            self._say(f"> {payload.get('text', '')}")
        else:
            self._say(f"(nothing recognised: {payload.get('reason', '')})")

    def _say(self, message: str) -> None:
        print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m speech.listener",
        description="Hands-free wake-word listener. Run with the speech virtualenv's python.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8420", help="Jarvis Core URL")
    parser.add_argument("--token", default="", help="the token printed by jarvis.serve")
    parser.add_argument("--wake-model", default="", help="defaults to the product identity's model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--silence", type=float, default=0.9, help="seconds of silence that end an utterance")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"]:
                print(f"{index:3}  {info['name']}")
        return 0

    if not args.token:
        print("--token is required (jarvis.serve prints it in the URL)", file=sys.stderr)
        return 2

    config = ListenerConfig(
        url=args.url,
        token=args.token,
        wake_model=args.wake_model,
        threshold=args.threshold,
        silence_seconds=args.silence,
        device=args.device,
        verbose=args.verbose,
    )
    try:
        return WakeListener(config).run()
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
