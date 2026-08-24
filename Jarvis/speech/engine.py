"""Main-process view of the speech engines: a worker on the other side of a pipe.

Implements :class:`~speech.contracts.SpeechToText` and
:class:`~speech.contracts.TextToSpeech` by delegating to
:mod:`speech.worker` running in the speech virtualenv.

Two decisions worth stating.

*The worker is started lazily and kept.*  Spawning costs a process and the first
transcription additionally costs a model load (~50 s for whisper-base cold,
~4.5 s for a Piper voice).  Paying that per utterance would make voice unusable,
so the process is long-lived and the models stay resident inside it.

*A dead worker is a recoverable condition, not a crash.*  If it exits -- a bad
model path, an out-of-memory kill -- the next call restarts it.  Speech failing
must degrade to text, never take Jarvis down with it.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speech.contracts import Audio, Transcript, Voice, VoiceRegistry

#: Where the speech virtualenv lives, relative to the repository root.
DEFAULT_VENV = Path(__file__).resolve().parent.parent / ".venv-speech"
DEFAULT_VOICES = Path(__file__).resolve().parent.parent / "data" / "voices"
DEFAULT_MODELS = Path(__file__).resolve().parent.parent / "data" / "models" / "whisper"


class SpeechUnavailable(RuntimeError):
    """The speech stack is not installed or would not start."""


def venv_python(venv: Path | None = None) -> Path | None:
    """The interpreter inside the speech virtualenv, if it exists."""

    root = Path(venv or DEFAULT_VENV)
    for candidate in (root / "Scripts" / "python.exe", root / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


@dataclass
class SpeechConfig:
    #: whisper model size.  "base" chosen on measured evidence: on this machine
    #: it transcribed a German sample at RTF 0.26 against "small"'s 0.79, and
    #: was also the more accurate of the two on that sample.
    stt_model: str = "base"
    language: str = ""
    voice_id: str = ""
    voices_dir: Path = DEFAULT_VOICES
    models_dir: Path = DEFAULT_MODELS
    venv: Path | None = None
    #: Generous: a cold whisper load can take a minute on a spinning disk.
    startup_timeout: float = 240.0
    call_timeout: float = 180.0


class SpeechEngine:
    """Speech to text and text to speech, over the worker protocol."""

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self.registry = VoiceRegistry(
            Path(self.config.voices_dir) / "voices.json", voices_dir=self.config.voices_dir
        )
        self.registry.discover()
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._counter = 0

    # -- availability ----------------------------------------------------

    @property
    def available(self) -> bool:
        return venv_python(self.config.venv) is not None

    def status(self) -> dict[str, Any]:
        python = venv_python(self.config.venv)
        if python is None:
            return {
                "available": False,
                "detail": f"no speech virtualenv at {DEFAULT_VENV}; "
                "create it and install faster-whisper and piper-tts",
                "voices": [],
            }
        voices = [voice.to_dict() for voice in self.registry.all()]
        return {
            "available": True,
            "python": str(python),
            "stt_model": self.config.stt_model,
            "voices": voices,
            "running": self._process is not None and self._process.poll() is None,
        }

    # -- the protocol ----------------------------------------------------

    def _start(self) -> subprocess.Popen[str]:
        python = venv_python(self.config.venv)
        if python is None:
            raise SpeechUnavailable(f"no speech virtualenv at {DEFAULT_VENV}")

        env = dict(os.environ)
        # The worker imports `speech.worker`, so the repository must be on the
        # path even though the interpreter belongs to another environment.
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            [str(python), "-m", "speech.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=root,
            env=env,
            bufsize=1,
        )
        self._process = process
        return process

    def _call(self, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                process = self._start()

            self._counter += 1
            payload = dict(payload, id=self._counter)

            try:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                self._kill()
                raise SpeechUnavailable(f"speech worker died: {exc}") from exc

            if not line:
                stderr = ""
                if process.stderr is not None:
                    try:
                        stderr = process.stderr.read() or ""
                    except Exception:
                        stderr = ""
                self._kill()
                raise SpeechUnavailable(f"speech worker produced no reply. {stderr[-500:]}")

            try:
                response = json.loads(line)
            except ValueError as exc:
                raise SpeechUnavailable(f"speech worker sent invalid JSON: {line[:200]}") from exc

        if not response.get("ok"):
            raise SpeechUnavailable(str(response.get("error") or "speech worker failed"))
        return response

    def _kill(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.kill()
        except Exception:
            pass

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                self._call({"action": "shutdown"})
            except Exception:
                pass
        self._kill()

    def __enter__(self) -> "SpeechEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- speech to text --------------------------------------------------

    def transcribe(self, audio: Audio, *, language: str = "") -> Transcript:
        response = self._call(
            {
                "action": "transcribe",
                "model": self.config.stt_model,
                "download_root": str(self.config.models_dir),
                "language": language or self.config.language,
                "wav": base64.b64encode(audio.to_wav()).decode("ascii"),
            },
            timeout=self.config.call_timeout,
        )
        return Transcript(
            text=str(response.get("text", "")),
            language=str(response.get("language", "")),
            confidence=float(response.get("confidence", 0.0)),
            duration_seconds=float(response.get("duration_seconds", 0.0)),
        )

    def transcribe_file(self, path: str | Path, *, language: str = "") -> Transcript:
        response = self._call(
            {
                "action": "transcribe",
                "model": self.config.stt_model,
                "download_root": str(self.config.models_dir),
                "language": language or self.config.language,
                "path": str(path),
            },
            timeout=self.config.call_timeout,
        )
        return Transcript(
            text=str(response.get("text", "")),
            language=str(response.get("language", "")),
            confidence=float(response.get("confidence", 0.0)),
            duration_seconds=float(response.get("duration_seconds", 0.0)),
        )

    # -- text to speech --------------------------------------------------

    def resolve_voice(self, voice_id: str = "", language: str = "") -> Voice:
        if voice_id:
            voice = self.registry.get(voice_id)
            if voice is None:
                raise SpeechUnavailable(f"no voice {voice_id!r} is installed")
            return voice
        if self.config.voice_id:
            voice = self.registry.get(self.config.voice_id)
            if voice is not None:
                return voice
        voice = self.registry.for_language(language or self.config.language or "de")
        if voice is None:
            installed = self.registry.installed()
            if not installed:
                raise SpeechUnavailable(
                    f"no TTS voice is installed; put a Piper .onnx and .onnx.json in {self.config.voices_dir}"
                )
            return installed[0]
        return voice

    def synthesize(self, text: str, *, voice: str = "", language: str = "") -> Audio:
        chosen = self.resolve_voice(voice, language)
        response = self._call(
            {"action": "synthesize", "model_path": chosen.model_path, "text": text},
            timeout=self.config.call_timeout,
        )
        return Audio(
            samples=base64.b64decode(response.get("pcm", "")),
            sample_rate=int(response.get("sample_rate", 22050)),
            width=int(response.get("width", 2)),
        )
