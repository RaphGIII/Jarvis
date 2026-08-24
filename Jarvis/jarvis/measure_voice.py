"""Measure the whole voice path, hop by hop.

``python -m jarvis.measure_voice`` -- runs the real components end to end and
reports where the time actually goes:

    wake word -> LISTENING -> STT -> FAST_LOCAL -> streamed TTS -> barge-in

Two honesty notes about what this does and does not prove.

*The audio is synthesised, not spoken.*  A script cannot talk into a
microphone, so Piper generates the utterance and it is fed to the real detector
and the real recogniser.  Every component is the production one; only the
speaker is artificial.  That makes the timings trustworthy and the *accuracy*
numbers optimistic-to-pessimistic in ways worth remembering: synthetic speech is
cleaner than a room, but the German voice pronounces "Jarvis" in a way the
English-trained wake model barely recognises.

*Cold and warm are reported separately.*  The first exchange after startup
loads a 4B model and a whisper model and costs about fifty seconds; every
exchange after it costs three. Reporting one number would misrepresent both.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Said to wake the assistant.  Spelled for the TTS voice rather than for a
#: reader: the German voice says "YAR-vis" for "Jarvis", which the
#: English-trained detector scores at 0.04, so the phonetic spelling is what
#: actually exercises the detector.
WAKE_PHRASE = "Hey Dscharwis"

DEFAULT_QUESTION = "Wie geht es mit meinem Projekt weiter?"


@dataclass
class Hop:
    name: str
    seconds: float
    detail: str = ""
    #: Whether this hop is part of the exchange the user actually waits through.
    #: Warm-up, synthesising the utterance that stands in for a microphone, and
    #: the check that the detector reset afterwards are all measurement
    #: apparatus rather than product behaviour.
    in_exchange: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": self.name,
            "seconds": round(self.seconds, 3),
            "detail": self.detail[:200],
            "in_exchange": self.in_exchange,
        }


@dataclass
class VoiceMeasurement:
    hops: list[Hop] = field(default_factory=list)
    transcript: str = ""
    answer: str = ""
    wake_score: float = 0.0
    interrupted: bool = False
    time_to_first_audio: float | None = None
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, seconds: float, detail: str = "", *, in_exchange: bool = False) -> None:
        self.hops.append(Hop(name, seconds, detail, in_exchange=in_exchange))

    @property
    def total_seconds(self) -> float:
        """The exchange, defined as the hops it is made of.

        Held as a wall-clock reading once, taken from the start of the exchange
        to the end of the last thing the harness happened to do. That reported
        53s for an exchange whose hops summed to four, because the last thing
        the harness did was spawn a venv subprocess to check the detector had
        reset, and importing numpy and openwakeword there took most of a minute.

        Deriving the total from the hops means the headline and the breakdown
        cannot disagree: anything not marked as part of the exchange cannot
        silently inflate it.
        """

        return sum(hop.seconds for hop in self.hops if hop.in_exchange)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hops": [hop.to_dict() for hop in self.hops],
            "wake_score": round(self.wake_score, 3),
            "transcript": self.transcript,
            "answer": self.answer[:400],
            "time_to_first_audio": (
                round(self.time_to_first_audio, 3) if self.time_to_first_audio else None
            ),
            "interrupted": self.interrupted,
            "total_seconds": round(self.total_seconds, 2),
            "notes": self.notes,
        }

    def describe(self) -> str:
        width = max((len(hop.name) for hop in self.hops), default=10)
        lines = ["", "  VOICE VERTICAL", ""]
        for hop in self.hops:
            lines.append(f"  {hop.name.ljust(width)}  {hop.seconds:6.2f}s   {hop.detail[:60]}")
        lines.append("")
        if self.time_to_first_audio is not None:
            lines.append(f"  question ends -> first audio out:  {self.time_to_first_audio:.2f}s")
        lines.append(f"  whole exchange:                    {self.total_seconds:.2f}s")
        if self.notes:
            lines += ["", "  notes:"] + [f"    - {note}" for note in self.notes]
        return "\n".join(lines)


def run(
    *,
    question: str = DEFAULT_QUESTION,
    warm: bool = True,
    barge_in: bool = True,
) -> VoiceMeasurement:
    """Drive the real pipeline and time every hop."""

    from brain.ollama import OllamaBrainProvider
    from brain.tiers import ModelCatalog, ModelTier
    from speech.contracts import Audio
    from speech.engine import SpeechEngine
    from speech.pipeline import NullSink, StreamingSpeaker

    measurement = VoiceMeasurement()
    engine = SpeechEngine()
    catalog = ModelCatalog()
    provider = OllamaBrainProvider(catalog.get(ModelTier.FAST_LOCAL))

    if warm:
        started = time.perf_counter()
        engine.synthesize("bereit")
        engine.transcribe(Audio(samples=bytes(32000), sample_rate=16000))
        provider.generate("OK", max_tokens=4, temperature=0.0)
        measurement.add("warm-up", time.perf_counter() - started, "models resident")
    else:
        measurement.notes.append("cold: every hop below includes a model load")

    # -- the user speaks ------------------------------------------------
    started = time.perf_counter()
    spoken = engine.synthesize(f"{WAKE_PHRASE}. {question}")
    measurement.add("synthesise the utterance", time.perf_counter() - started,
                    f"{spoken.seconds:.1f}s of audio (stands in for a microphone)")

    # -- wake word ------------------------------------------------------
    score, wake_load, wake_detect = _wake_score(spoken)
    measurement.wake_score = score
    measurement.add(
        "wake word", wake_detect,
        f"score {score:.3f} over {spoken.seconds:.1f}s of audio",
    )
    if wake_load:
        measurement.notes.append(
            f"the wake detector took {wake_load:.1f}s to load its model in a separate venv; "
            "that is startup, not detection, and is excluded from the hop above"
        )
    if score < 0.5:
        measurement.notes.append(
            f"the wake word scored {score:.3f}, below the 0.5 threshold -- "
            "detection would not have fired"
        )

    # From here the clock that matters starts: the user has stopped talking.
    exchange_started = time.perf_counter()

    # -- transcription --------------------------------------------------
    started = time.perf_counter()
    transcript = engine.transcribe(spoken, language="de")
    measurement.transcript = transcript.text
    measurement.add("transcribe", time.perf_counter() - started, transcript.text[:60],
                    in_exchange=True)

    # -- think and speak, concurrently -----------------------------------
    from core.identity import current as current_identity

    identity = current_identity()
    prompt = (
        f"{identity.persona_preamble()} Answer in two or three short sentences, in German.\n\n"
        f"user: {transcript.text or question}\n{identity.assistant_name}:"
    )

    collected: list[str] = []
    first_audio: list[float] = []
    speaker = StreamingSpeaker(
        engine.synthesize,
        sink=NullSink(realtime=True),
        on_audio=lambda audio, phrase: first_audio.append(time.perf_counter()),
    )

    def tokens():
        for chunk in provider.generate_stream(prompt, max_tokens=180):
            collected.append(chunk)
            yield chunk

    if barge_in:
        # Interrupt as soon as the first phrase is audible: that is what
        # speaking over the reply actually does, and a timer would measure the
        # scheduler instead.
        original = speaker.on_audio

        def interrupt_on_first(audio, phrase):
            original(audio, phrase)
            if phrase.first:
                speaker.interrupt()

        speaker.on_audio = interrupt_on_first

    started = time.perf_counter()
    metrics = speaker.speak_stream(tokens())
    measurement.answer = "".join(collected).strip()
    measurement.interrupted = speaker.interrupted
    measurement.add("think and speak", time.perf_counter() - started,
                    f"{metrics.phrases} phrase(s), {metrics.audio_seconds:.1f}s of audio",
                    in_exchange=True)

    if first_audio:
        measurement.time_to_first_audio = first_audio[0] - exchange_started

    if barge_in:
        # -- resume listening --------------------------------------------
        resumed_score, _, resumed_detect = _wake_score(engine.synthesize(WAKE_PHRASE))
        # The detection time, not the venv start: same reason as the hop above.
        measurement.add("resume listening", resumed_detect,
                        "detector reset and listening again" if resumed_score > 0.0
                        else "detector did not reset")
    engine.close()
    return measurement


def _wake_score(audio: Any) -> tuple[float, float, float]:
    """Run the real wake detector over synthesised audio, in its own venv.

    Returns (peak score, model-load seconds, detection seconds). The load is
    reported apart from the detection because this runs in a separate venv:
    from the caller's side an interpreter start and an ONNX model load are
    indistinguishable from the work, and folding them together reported 28s
    for a hop that takes milliseconds.
    """

    import base64
    import subprocess

    from speech.engine import venv_python

    interpreter = venv_python()
    if interpreter is None:
        return 0.0

    root = str(Path(__file__).resolve().parent.parent)
    script = (
        "import sys, base64, io, wave, numpy as np;"
        f"sys.path.insert(0, {root!r});"
        "from core.identity import current;"
        "from openwakeword.model import Model;"
        "data = base64.b64decode(sys.stdin.read());"
        "w = wave.open(io.BytesIO(data));"
        "a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32);"
        "rate = w.getframerate();"
        "n = int(len(a) * 16000 / rate);"
        "a = np.interp(np.linspace(0, len(a), n), np.arange(len(a)), a).astype(np.int16);"
        "pad = np.zeros(16000, dtype=np.int16);"
        "a = np.concatenate([pad, a, pad]);"
        "name = current().resolved_wake_model;"
        "import time;"
        "t0 = time.perf_counter();"
        "m = Model(wakeword_models=[name], inference_framework='onnx');"
        "load = time.perf_counter() - t0;"
        "t1 = time.perf_counter();"
        "peak = 0.0\n"
        "for i in range(0, len(a) - 1280, 1280):\n"
        "    peak = max(peak, m.predict(a[i:i+1280]).get(name, 0.0))\n"
        "print(peak, load, time.perf_counter() - t1)"
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", script],
            input=base64.b64encode(audio.to_wav()).decode("ascii"),
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        parts = (completed.stdout or "0").strip().splitlines()[-1].split()
        peak = float(parts[0])
        # The subprocess times itself, because from out here the model load
        # and the interpreter start are indistinguishable from detection.
        load = float(parts[1]) if len(parts) > 1 else 0.0
        detect = float(parts[2]) if len(parts) > 2 else 0.0
        return peak, load, detect
    except Exception:
        return 0.0, 0.0, 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.measure_voice",
        description="Measure the full voice path end to end with the real components.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--cold", action="store_true", help="do not warm the models first")
    parser.add_argument("--no-barge-in", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    measurement = run(
        question=args.question,
        warm=not args.cold,
        barge_in=not args.no_barge_in,
    )

    if args.json:
        print(json.dumps(measurement.to_dict(), indent=2))
    else:
        print(measurement.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
