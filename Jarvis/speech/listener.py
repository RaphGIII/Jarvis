"""Hands-free listening: the first Jarvis device client.

Runs inside the speech virtualenv, holds the microphone, and talks to Jarvis
Core over ordinary HTTP.  It is deliberately a *separate process from the core*,
because that is the shape the eventual hardware takes: a small box with a
microphone and a speaker, and the brain somewhere else.  On this machine the
device and the server happen to be the same computer; nothing in this file
assumes that.

    microphone -> wake word -> listening session -> utterance -> POST /api/voice/utterance

What runs here rather than on the server, and why:

*Wake word.*  Streaming continuous audio to the core would be wasteful and a
privacy problem -- the box decides locally when Jarvis is being addressed, and
only then does any audio leave the machine.

*Endpointing.*  Deciding when the user stopped talking needs the audio stream,
not a copy of it after the fact.  Energy-based silence detection is enough here
and costs nothing; a neural VAD is a drop-in replacement if it proves necessary.

*The listening session is a state machine* (:class:`VoiceSession`), because the
live product failed exactly where the states were implicit: the detector fired
as the word ended, the natural pause before the command counted as 0.9 s of
silence, the session closed with "(too short - ignored)", and every wake also
posted a generic stop that the interface printed as "stopped".  Now:

    IDLE -> WAKE_DETECTED -> LISTENING (armed, grace) -> CAPTURING (speech heard)
         -> UTTERANCE_CAPTURED -> SENT -> IDLE

A session ends exactly once, with a reason code (``no_speech_after_wake``,
``silence``, ``max_length``, ``too_short``); every transition is reported to
the core with the session id so the event log tells the whole story.

*Barge-in* is a request to the core to interrupt what is *speaking or being
generated* (``/api/voice/interrupt``); when nothing is, it is a no-op and
produces no message.  It never touches the session that was just opened.
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
from dataclasses import dataclass, field
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
    #: Detection threshold for openWakeWord's built-in models.  A trained
    #: ZEUS detector applies its own (the core's effective) threshold and
    #: reports 1.0 on the frame it fires, so this only passes that through.
    threshold: float = 0.5
    #: Silence that ends an utterance *after speech was heard*.  Shorter feels
    #: abrupt mid-sentence; longer makes every exchange feel laggy.
    silence_seconds: float = 0.9
    #: The post-wake grace: how long the session stays armed for the first
    #: word of the command.  Derived from the pipeline rather than guessed:
    #: the detector confirms ~160 ms after "Zeus" ends (two 80 ms frames),
    #: the pre-roll already holds the previous 640 ms, and people pause
    #: 0.3-2.0 s between the name and the request.  3.0 s covers that with a
    #: margin and costs nothing when the owner speaks at once, because the
    #: session leaves LISTENING the moment speech is heard.
    arm_seconds: float = 3.0
    #: Refuse to record forever if the room is noisy.
    max_utterance_seconds: float = 20.0
    #: Ignore an utterance shorter than this: usually a cough or a door.
    min_utterance_seconds: float = 0.35
    #: Frames of audio kept before the wake word fires, so the first syllable
    #: of the question is not clipped off.
    preroll_frames: int = 8
    #: How long to ignore the wake word after a session ended, so one
    #: "Zeus" cannot open two sessions.
    cooldown_seconds: float = 1.5
    #: Multiple of the measured noise floor that counts as speech.
    speech_factor: float = 2.5
    device: int | None = None
    verbose: bool = False
    #: Ask the core to interrupt its speech when the wake word fires.
    barge_in: bool = True


class _TrainedWake:
    """A locally trained detector behind openWakeWord's predict/reset shape.

    The detector already applies its threshold, consecutive-frame rule and
    cooldown; ``predict`` reports 1.0 on the frame it fires and the raw score
    otherwise, so the listener's own ``threshold`` (0.5 by default) passes a
    detection through and nothing else.

    The threshold and the weights are the core's to decide: :meth:`sync`
    takes ``/api/voice/wake`` (the same status Voice Studio shows) and applies
    the effective threshold; when the model file changed (a new training) it
    reloads the weights in place, so the owner never has to restart ZEUS for
    the listener to hear the model that was just trained.
    """

    def __init__(self, detector: Any, name: str, *, path: Any = None, loader: Any = None) -> None:
        self.detector = detector
        self.name = name
        self.path = path
        self._loader = loader
        self.reloads = 0

    def predict(self, frame: Any) -> dict[str, float]:
        fired = self.detector.feed(frame)
        return {self.name: 1.0 if fired else min(0.49, float(self.detector.last_score))}

    def reset(self) -> None:
        self.detector.reset()

    @property
    def score(self) -> float:
        return float(getattr(self.detector, "last_score", 0.0))

    def report(self) -> dict[str, Any]:
        import os

        return {"model": str(self.path or ""), "fingerprint": getattr(self.detector, "fingerprint", ""),
                "threshold": float(self.detector.threshold), "pid": os.getpid(), "last_score": round(self.score, 4)}

    def sync(self, status: dict[str, Any]) -> list[str]:
        """Apply the core's wake status; returns what changed (for the log)."""

        changed: list[str] = []
        fingerprint = str(status.get("model_fingerprint") or "")
        if fingerprint and fingerprint != getattr(self.detector, "fingerprint", "") and self._loader is not None:
            try:
                fresh = self._loader()
            except Exception as exc:  # noqa: BLE001 - keep listening with the old weights
                changed.append(f"reload failed: {exc}")
            else:
                fresh.threshold = self.detector.threshold
                self.detector = fresh
                self.reloads += 1
                changed.append(f"model reloaded ({fingerprint})")
        try:
            threshold = float(status.get("effective_threshold"))
        except (TypeError, ValueError):
            threshold = None
        if threshold is not None and abs(threshold - float(self.detector.threshold)) > 1e-9:
            self.detector.threshold = threshold
            changed.append(f"threshold {threshold} ({status.get('threshold_source', '?')})")
        return changed


class Endpointer:
    """Decides when the user has stopped talking.

    Extracted from the capture loop so it can be tested without a microphone,
    which is the only way to check the cases that matter: a pause mid-sentence
    must not end the utterance, a noisy room must not prevent it from ever
    ending, and a cough must not become a question.

    The threshold tracks a slow estimate of the room rather than being fixed. A
    fixed value is wrong in both directions -- too high in a quiet room and the
    endpoint never fires; too low next to a fan and recording never stops.

    ``arm_seconds`` is the post-wake grace: until speech has been heard, silence
    does not end the session -- only the grace running out does (``timed_out``).
    With the default 0.0 the behaviour is the classic one.
    """

    def __init__(self, config: "ListenerConfig", *, frame_seconds: float, arm_seconds: float = 0.0) -> None:
        self.config = config
        self.frame_seconds = frame_seconds
        self.arm_seconds = arm_seconds
        self.noise_floor = 0.0
        self.silence_for = 0.0
        self.elapsed = 0.0
        self.heard_speech = False
        self.timed_out = False
        self.lead_silence = 0.0

    def reset(self) -> None:
        self.silence_for = 0.0
        self.elapsed = 0.0
        self.heard_speech = False
        self.timed_out = False
        self.lead_silence = 0.0

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
            if not self.heard_speech:
                self.lead_silence += self.frame_seconds
        else:
            self.silence_for = 0.0
            self.heard_speech = True
        if self.elapsed >= self.config.max_utterance_seconds:
            return True
        if not self.heard_speech and self.arm_seconds > 0:
            # Armed: waiting for the first word.  Only the grace ends this.
            if self.elapsed >= self.arm_seconds:
                self.timed_out = True
                return True
            return False
        return self.silence_for >= self.config.silence_seconds

    @property
    def speech_seconds(self) -> float:
        """How much of the utterance was speech: neither the leading pause nor the trailing silence."""

        return max(0.0, self.elapsed - self.lead_silence - self.silence_for)


# --------------------------------------------------------------------------
# The listening session: explicit states, one end, reason codes
# --------------------------------------------------------------------------

STATES = ("IDLE", "WAKE_DETECTED", "LISTENING", "CAPTURING", "UTTERANCE_CAPTURED", "SENT")
TRANSITIONS = {
    "IDLE": {"WAKE_DETECTED"},
    "WAKE_DETECTED": {"LISTENING", "IDLE"},
    "LISTENING": {"CAPTURING", "IDLE"},
    "CAPTURING": {"UTTERANCE_CAPTURED", "IDLE"},
    "UTTERANCE_CAPTURED": {"SENT", "IDLE"},
    "SENT": {"IDLE"},
}


@dataclass
class VoiceSession:
    """One listening session, from the wake word to idle.  Ends once."""

    session_id: str
    wake_score: float
    state: str = "WAKE_DETECTED"
    opened_at: float = 0.0
    ended: bool = False
    end_reason: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)

    def transition(self, state: str, reason: str = "") -> bool:
        if state not in TRANSITIONS.get(self.state, set()):
            return False
        self.history.append((state, reason))
        self.state = state
        return True

    def end(self, reason: str) -> bool:
        """Close the session; True the first time only."""

        if self.ended:
            return False
        self.ended = True
        self.end_reason = reason
        self.history.append(("IDLE", reason))
        self.state = "IDLE"
        return True


class CaptureLoop:
    """The per-frame logic of the listener, without a microphone.

    ``step(frame, level, wake_score, now)`` returns a list of actions for the
    process around it: ``("interrupt", session)``, ``("session", session, state,
    reason)`` for reporting, ``("send", session, pcm_frames)`` when an
    utterance is ready, ``("log", text)``.  The loop owns the pre-roll, the
    endpointer and the cooldown; the wake decision is the caller's (it has
    the detector).
    """

    def __init__(self, config: ListenerConfig, *, frame_seconds: float = FRAME_SAMPLES / SAMPLE_RATE) -> None:
        self.config = config
        self.frame_seconds = frame_seconds
        self.endpointer = Endpointer(config, frame_seconds=frame_seconds, arm_seconds=config.arm_seconds)
        self.preroll: collections.deque = collections.deque(maxlen=config.preroll_frames)
        self.recording: list[Any] = []
        self.session: VoiceSession | None = None
        self.muted_until = 0.0
        self.sessions_opened = 0
        self.sessions_ended = 0

    @property
    def state(self) -> str:
        return self.session.state if self.session is not None and not self.session.ended else "IDLE"

    def step(self, frame: Any, level: float, wake_fired: bool, now: float) -> list[tuple]:
        actions: list[tuple] = []
        session = self.session if self.session is not None and not self.session.ended else None
        if session is None:
            # IDLE: the noise floor is only updated while nobody is speaking
            # to ZEUS; letting the utterance itself raise it would make a
            # long answer progressively harder to end.
            self.endpointer.track_noise(level)
            self.preroll.append(frame)
            if now < self.muted_until or not wake_fired:
                return actions
            self.sessions_opened += 1
            session = self.session = VoiceSession(session_id=f"vs{int(now * 1000):x}{self.sessions_opened}", wake_score=1.0, opened_at=now)
            actions.append(("session", session, "WAKE_DETECTED", "wake word"))
            if self.config.barge_in:
                actions.append(("interrupt", session))
            session.transition("LISTENING", "armed")
            actions.append(("session", session, "LISTENING", f"armed for {self.config.arm_seconds:.1f}s"))
            self.endpointer.reset()
            # Keep the pre-roll: people run the question straight into the
            # wake word, so the first syllable is already gone by the time
            # detection fires.
            self.recording = list(self.preroll)
            self.preroll.clear()
            return actions

        # A session is open: LISTENING or CAPTURING.
        self.recording.append(frame)
        over = self.endpointer.feed(level)
        if session.state == "LISTENING" and self.endpointer.heard_speech:
            session.transition("CAPTURING", "speech heard")
            actions.append(("session", session, "CAPTURING", f"speech after {self.endpointer.elapsed:.2f}s"))
        if not over:
            return actions

        self.muted_until = now + self.config.cooldown_seconds
        frames, self.recording = self.recording, []
        if self.endpointer.timed_out:
            self._end(session, "no_speech_after_wake", actions)
            return actions
        if self.endpointer.elapsed >= self.config.max_utterance_seconds and not self.endpointer.heard_speech:
            self._end(session, "no_speech_after_wake", actions)
            return actions
        if self.endpointer.speech_seconds < self.config.min_utterance_seconds:
            self._end(session, "too_short", actions)
            return actions
        session.transition("UTTERANCE_CAPTURED", "silence" if self.endpointer.elapsed < self.config.max_utterance_seconds else "max_length")
        actions.append(("session", session, "UTTERANCE_CAPTURED", f"{self.endpointer.speech_seconds:.1f}s of speech"))
        actions.append(("send", session, frames))
        return actions

    def sent(self, session: VoiceSession, ok: bool, detail: str = "") -> list[tuple]:
        """The core answered the POST; the session is over."""

        actions: list[tuple] = []
        if session.ended:
            return actions
        session.transition("SENT", "posted")
        self._end(session, "sent" if ok else f"rejected: {detail}"[:80], actions)
        return actions

    def _end(self, session: VoiceSession, reason: str, actions: list[tuple]) -> None:
        if session.end(reason):
            self.sessions_ended += 1
            actions.append(("session", session, "IDLE", reason))
            actions.append(("reset", session))


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

        from core.identity import BUILTIN_WAKE_MODELS, trained_wake_model_exists, trained_wake_model_path

        if self.config.wake_model not in BUILTIN_WAKE_MODELS and trained_wake_model_exists(self.config.wake_model):
            # A classifier trained here for this word (speech.wake_training),
            # wrapped so the loop below sees the same predict/reset shape.
            from speech.wake_zeus import ZeusDetector

            path = trained_wake_model_path(self.config.wake_model)
            self._model = _TrainedWake(ZeusDetector.load(path), self.config.wake_model, path=path,
                                       loader=lambda: ZeusDetector.load(path))
            self._sync_wake(force=True)
            return self._model

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

        loop = CaptureLoop(self.config)
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
                now = time.monotonic()
                fired = False
                if loop.state == "IDLE":
                    self._sync_wake()
                    if now >= loop.muted_until:
                        score = model.predict(frame).get(self.config.wake_model, 0.0)
                        fired = score >= self.config.threshold
                        if fired:
                            wake_score = model.score if isinstance(model, _TrainedWake) else score
                self._perform(loop.step(frame.copy(), level, fired, now), loop, model, wake_score if fired else 0.0)

    def _perform(self, actions: list[tuple], loop: CaptureLoop, model: Any, wake_score: float) -> None:
        import numpy as np

        for action in actions:
            kind = action[0]
            if kind == "session":
                _, session, state, reason = action
                if state == "WAKE_DETECTED":
                    session.wake_score = wake_score
                    self._say(f"[{session.session_id}] wake ({wake_score:.2f}) -> LISTENING")
                elif state == "IDLE":
                    self._say(f"[{session.session_id}] -> IDLE ({reason})")
                elif self.config.verbose or state == "CAPTURING":
                    self._say(f"[{session.session_id}] -> {state} ({reason})")
                self._report_session(session, state, reason)
            elif kind == "interrupt":
                self._interrupt(action[1])
            elif kind == "reset":
                model.reset()
            elif kind == "send":
                _, session, frames = action
                audio = np.concatenate(frames) if frames else np.zeros(0, dtype="int16")
                self._say(f"[{session.session_id}] heard {loop.endpointer.speech_seconds:.1f}s, sending...")
                ok, detail = self._send(audio.tobytes(), wake=session.wake_score, session=session.session_id)
                self._perform(loop.sent(session, ok, detail), loop, model, wake_score)

    # -- helpers ---------------------------------------------------------

    #: How often the idle loop asks the core for the wake status.
    SYNC_SECONDS = 5.0

    def _sync_wake(self, *, force: bool = False) -> None:
        """Every few seconds: take the effective threshold, reload a retrained model, report what runs."""

        model = self._model
        if not isinstance(model, _TrainedWake):
            return
        now = time.monotonic()
        if not force and now - getattr(self, "_synced_at", -1e9) < self.SYNC_SECONDS:
            return
        self._synced_at = now
        try:
            status = self._get("/api/voice/wake")
            for change in model.sync(status):
                self._say(f"wake config: {change}")
            self._post_json("/api/voice/wake/listener", model.report())
        except Exception as exc:  # noqa: BLE001 - the core may be restarting; keep listening
            if self.config.verbose:
                self._say(f"(wake sync failed: {exc})")

    def _report_session(self, session: VoiceSession, state: str, reason: str) -> None:
        try:
            self._post_json("/api/voice/session", {"session": session.session_id, "state": state, "reason": reason,
                                                   "wake": round(float(session.wake_score), 4)})
        except Exception as exc:  # noqa: BLE001 - reporting never blocks listening
            if self.config.verbose:
                self._say(f"(session report failed: {exc})")

    def _get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.config.url.rstrip('/')}{path}", method="GET",
                                         headers={"X-Jarvis-Token": self.config.token})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode())

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.config.url.rstrip('/')}{path}", data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "X-Jarvis-Token": self.config.token})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode())

    def _interrupt(self, session: VoiceSession | None = None) -> None:
        """Barge-in: ask the core to interrupt what is speaking or being generated.

        ``/api/voice/interrupt`` is a no-op when nothing is -- it never opens,
        closes or otherwise touches the listening session, and it never
        produces a transcript entry by itself.
        """

        if not getattr(self.config, "barge_in", True):
            return
        try:
            result = self._post_json("/api/voice/interrupt", {"session": session.session_id if session else "",
                                                               "wake": round(float(session.wake_score), 4) if session else 0.0})
            if result.get("interrupted") and self.config.verbose:
                self._say(f"(interrupted: {', '.join(result['interrupted'])})")
        except Exception as exc:  # noqa: BLE001 - listening matters more than the interrupt
            if self.config.verbose:
                self._say(f"(interrupt failed: {exc})")

    def _send(self, pcm: bytes, *, wake: float = 0.0, session: str = "") -> tuple[bool, str]:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm)

        # The wake score that opened this session travels with the audio:
        # the core acts only on utterances that a wake word (or a press in
        # the interface) authorised.
        request = urllib.request.Request(
            f"{self.config.url.rstrip('/')}/api/voice/utterance",
            data=buffer.getvalue(),
            method="POST",
            headers={"Content-Type": "application/octet-stream", "X-Jarvis-Token": self.config.token,
                     "X-Jarvis-Wake": f"{float(wake):.4f}", "X-Jarvis-Session": session},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            self._say(f"core refused the utterance: HTTP {exc.code}")
            return False, f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            self._say(f"core unreachable: {exc}")
            return False, "core unreachable"
        if payload.get("ok"):
            self._say(f"> {payload.get('text', '')}")
            return True, ""
        if payload.get("ignored"):
            self._say(f"(ignored: {payload.get('reason', '')} -- '{payload.get('text', '')}')")
            return False, str(payload.get("reason", "ignored"))
        self._say(f"(nothing recognised: {payload.get('reason', '')})")
        return False, str(payload.get("reason", "nothing recognised"))

    def _say(self, message: str) -> None:
        print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m speech.listener",
        description="Hands-free wake-word listener. Run with the speech virtualenv's python.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8420", help="Jarvis Core URL")
    parser.add_argument("--token", default="", help="the token printed by jarvis.serve")
    parser.add_argument("--no-barge-in", action="store_true", help="do not interrupt ZEUS's speech on the wake word")
    parser.add_argument("--wake-model", default="", help="defaults to the product identity's model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--silence", type=float, default=0.9, help="seconds of silence that end an utterance")
    parser.add_argument("--arm", type=float, default=3.0, help="seconds the session waits for the first word after the wake word")
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
        arm_seconds=args.arm,
        device=args.device,
        verbose=args.verbose,
        barge_in=not args.no_barge_in,
    )
    try:
        return WakeListener(config).run()
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
