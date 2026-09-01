"""The speech worker: runs inside the speech virtualenv, talks JSON over stdio.

Why a subprocess at all.  faster-whisper drags in ctranslate2 and Piper drags in
onnxruntime, neither of which belongs in the system Python that runs the rest of
Jarvis -- the brief is explicit about not modifying global packages, and a
speech stack that breaks the project engine on upgrade would be a bad trade for
any amount of convenience.  A virtualenv solves the dependency question but
creates a process boundary, so this is the thing on the far side of it.

The protocol is one JSON object per line in each direction, with audio
base64-encoded.  Unglamorous, but it has the properties that matter here: no
dependency of its own, trivially debuggable by hand, and the same shape a
network transport would take when the speech components move onto a device.

Models are loaded on first use rather than at startup, so the worker can be
spawned eagerly (paying the process cost once, out of the user's way) while the
several-second model load happens only if speech is actually used.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_state: dict[str, Any] = {"stt": None, "tts": {}, "stt_name": "", "voices_dir": ""}


def _load_stt(model: str, download_root: str) -> Any:
    if _state["stt"] is not None and _state["stt_name"] == model:
        return _state["stt"]
    import os

    from faster_whisper import WhisperModel

    # Measured on this machine (2026-09-01): CUDA on the GTX 1070 is a dead
    # end today -- int8/int8_float16/float16 are refused by ctranslate2 on
    # Pascal ("not efficient"), and float32 constructs in ~50s but fails at
    # the first transcribe because cublas64_12.dll is not installed.  CPU
    # int8 transcribes the German corpus at similarity 0.96 (median 3.7s),
    # so "auto" means CPU; only an explicit ZEUS_STT_DEVICE=cuda pays the
    # CUDA attempt (once, remembered on failure).
    wanted = os.environ.get("ZEUS_STT_DEVICE", "auto").strip().lower()
    attempts = []
    if wanted == "cuda" and _state.get("cuda_failed") is not True:
        # Pascal (GTX 1070, CC 6.1): no int8 tensor cores, fp16 at 1/64 rate --
        # ctranslate2 refuses both as "not efficient".  float32 on CUDA still
        # beats int8 on this CPU; measured before enabling.
        attempts.append(("cuda", "int8"))
        attempts.append(("cuda", "float16"))
        attempts.append(("cuda", "float32"))
    if wanted != "cuda":
        attempts.append(("cpu", "int8"))
    last_error: Exception | None = None
    for device, compute in attempts:
        try:
            engine = WhisperModel(model or "base", device=device, compute_type=compute, download_root=download_root or None)
            if device == "cuda":
                # prove it can actually run, not just construct
                import numpy as _np

                list(engine.transcribe(_np.zeros(1600, dtype=_np.float32), language="de", beam_size=1)[0])
            _state["stt"] = engine
            _state["stt_name"] = model
            _state["stt_device"] = f"{device}/{compute}"
            return engine
        except Exception as exc:  # noqa: BLE001 - fall through to the CPU
            last_error = exc
            if device == "cuda":
                _state["cuda_failed"] = True
    raise RuntimeError(f"could not load whisper {model!r}: {last_error}")


def _load_voice(model_path: str) -> Any:
    """Voices are cached by path: reloading costs ~4.5s and they are small."""

    cached = _state["tts"].get(model_path)
    if cached is not None:
        return cached
    from piper import PiperVoice

    voice = PiperVoice.load(model_path)
    _state["tts"][model_path] = voice
    return voice


def handle(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action", "")

    if action == "ping":
        return {"ok": True, "pong": True}

    if action == "capabilities":
        available: dict[str, bool] = {}
        for name in ("faster_whisper", "piper", "sounddevice"):
            try:
                __import__(name)
                available[name] = True
            except Exception:
                available[name] = False
        return {"ok": True, "available": available, "python": sys.version.split()[0]}

    if action == "transcribe":
        started = time.perf_counter()
        model = _load_stt(request.get("model", "base"), request.get("download_root", ""))
        audio_path = request.get("path", "")
        if not audio_path:
            # Write the samples out: faster-whisper reads files or arrays, and
            # a temp file avoids depending on numpy in the protocol.
            import tempfile

            handle_, audio_path = tempfile.mkstemp(suffix=".wav")
            Path(audio_path).write_bytes(base64.b64decode(request.get("wav", "")))
        language = request.get("language") or None
        # Whisper has never seen "Jarvis" and reliably hears it as an ordinary
        # German word -- "wie geht es mit meinem Jarvis-Projekt" came back as
        # "Jahresprojekt". initial_prompt biases decoding toward the vocabulary
        # this system is actually about, which is what it is for.
        #
        # Decoding is deliberately *stateless and suspicious*: one utterance
        # is one request, so nothing conditions on a previous window (that is
        # how one hallucination seeds the next), Whisper's own no-speech and
        # log-probability verdicts are returned rather than swallowed, and a
        # silent stretch is not decoded at all (hallucination_silence_threshold).
        # ``hotwords`` carries the bounded entity list (project names, ZEUS
        # terms) as a decoding bias, not as a prompt that leaks into the text.
        options: dict[str, Any] = dict(
            language=language,
            beam_size=int(request.get("beam_size", 1)),
            initial_prompt=request.get("vocabulary") or None,
            word_timestamps=bool(request.get("word_timestamps", True)),
            condition_on_previous_text=False,
            # One decoding pass.  The default temperature *fallback* re-decodes
            # up to six times when the log-probability or compression check
            # fails -- exactly on silence, noise and mumbling -- and cost 10-15
            # s per bad utterance on this CPU.  A single pass reports the same
            # quality signals; the gate does the rejecting, cheaply.
            temperature=float(request.get("temperature", 0.0)),
            no_speech_threshold=float(request.get("no_speech_threshold", 0.6)),
            log_prob_threshold=float(request.get("log_prob_threshold", -1.0)),
            compression_ratio_threshold=float(request.get("compression_ratio_threshold", 2.4)),
            hallucination_silence_threshold=float(request.get("hallucination_silence_threshold", 1.5)),
            vad_filter=bool(request.get("vad_filter", False)),
        )
        hotwords = str(request.get("hotwords") or "").strip()
        if hotwords:
            options["hotwords"] = hotwords
        try:
            segments, info = model.transcribe(audio_path, **options)
            pieces = list(segments)
        except TypeError:
            # an older faster-whisper without one of the knobs
            for key in ("hotwords", "hallucination_silence_threshold", "vad_filter"):
                options.pop(key, None)
            segments, info = model.transcribe(audio_path, **options)
            pieces = list(segments)
        text = "".join(segment.text for segment in pieces).strip()
        # Word timings let the core tell a wake-word tail from the command.
        words = []
        for segment in pieces:
            for w in (getattr(segment, "words", None) or []):
                words.append({"word": str(getattr(w, "word", "")).strip(), "start": round(float(getattr(w, "start", 0.0) or 0.0), 3),
                              "end": round(float(getattr(w, "end", 0.0) or 0.0), 3), "probability": round(float(getattr(w, "probability", 0.0) or 0.0), 3)})
        # Per-segment quality, reduced to the most pessimistic value across
        # the utterance: one hallucinated segment is enough to distrust it.
        quality: dict[str, Any] = {"segments": len(pieces)}
        if pieces:
            nsp = [float(getattr(s, "no_speech_prob", 0.0) or 0.0) for s in pieces]
            lps = [float(getattr(s, "avg_logprob", 0.0) or 0.0) for s in pieces]
            crs = [float(getattr(s, "compression_ratio", 0.0) or 0.0) for s in pieces]
            quality.update({
                "no_speech_probability": round(max(nsp), 3),
                "avg_logprob": round(min(lps), 3),
                "compression_ratio": round(max(crs), 3),
            })
        if words:
            quality["word_probabilities"] = [w["probability"] for w in words]
        quality["language_probability"] = round(float(getattr(info, "language_probability", 0.0) or 0.0), 3)
        quality["elapsed"] = round(time.perf_counter() - started, 3)
        quality["model"] = str(request.get("model", "base"))
        return {
            "ok": True,
            "text": text,
            "words": words,
            "language": getattr(info, "language", "") or "",
            "confidence": float(getattr(info, "language_probability", 0.0) or 0.0),
            "duration_seconds": float(getattr(info, "duration", 0.0) or 0.0),
            "elapsed": quality["elapsed"],
            "quality": quality,
        }

    if action == "synthesize":
        started = time.perf_counter()
        voice = _load_voice(request["model_path"])
        chunks = list(voice.synthesize(request.get("text", "")))
        samples = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return {
            "ok": True,
            "pcm": base64.b64encode(samples).decode("ascii"),
            "sample_rate": int(voice.config.sample_rate),
            "width": 2,
            "elapsed": round(time.perf_counter() - started, 3),
        }

    if action == "shutdown":
        return {"ok": True, "bye": True}

    return {"ok": False, "error": f"unknown action: {action!r}"}


def main() -> int:
    # Line-buffered so the parent sees each reply as it is produced rather than
    # when the pipe buffer happens to fill.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            _reply({"ok": False, "error": f"bad json: {exc}"})
            continue
        try:
            response = handle(request)
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }
        response["id"] = request.get("id")
        _reply(response)
        if request.get("action") == "shutdown":
            return 0
    return 0


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
