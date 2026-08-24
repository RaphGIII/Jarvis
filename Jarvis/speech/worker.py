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
    from faster_whisper import WhisperModel

    _state["stt"] = WhisperModel(
        model or "base", device="cpu", compute_type="int8", download_root=download_root or None
    )
    _state["stt_name"] = model
    return _state["stt"]


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
        segments, info = model.transcribe(
            audio_path, language=language, beam_size=int(request.get("beam_size", 1))
        )
        pieces = list(segments)
        text = "".join(segment.text for segment in pieces).strip()
        return {
            "ok": True,
            "text": text,
            "language": getattr(info, "language", "") or "",
            "confidence": float(getattr(info, "language_probability", 0.0) or 0.0),
            "duration_seconds": float(getattr(info, "duration", 0.0) or 0.0),
            "elapsed": round(time.perf_counter() - started, 3),
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
