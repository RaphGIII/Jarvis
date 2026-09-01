"""One human utterance = one authoritative, evidence-backed event.

The live product produced "ghost sentences": text nobody said appeared as
the owner's words and ZEUS answered it.  Every path that can turn audio into
a visible USER message now goes through the objects in this module, and
each of them has to carry *evidence* rather than a bare string:

``AudioEvidence``
    measured from the PCM itself, independently of what any recogniser says:
    duration, RMS, peak, how much of the recording is speech-like energy, an
    audio fingerprint.  A silent or noise-only segment is recognisable here
    before Whisper gets to invent a sentence for it.

``UtteranceEvidence``
    the identity of the utterance -- session, utterance id, source, timing --
    plus the audio evidence, the recogniser's quality signals (no-speech
    probability, average log-probability, compression ratio, word timings)
    and the raw/normalised transcripts.

``AcceptanceGate``
    an evidence-based decision with reason codes.  Whisper text is *not*
    trusted merely because text came back: silence, fan noise, keyboard
    clicks, a partial repeated transcript, ZEUS's own speech picked up by the
    microphone, a stale or duplicated utterance -- each is refused by a named
    check, and the checks travel with the verdict so Activity can show why.

``UtteranceLedger``
    idempotency.  An utterance may be semantically executed at most once,
    whatever the transport does: the same ``utterance_id`` or the same audio
    fingerprint inside the window is refused as a replay.

Nothing here contains a banned-phrase list.  The defence is measurement.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

_WORD = re.compile(r"[\w']+", re.UNICODE)


# --------------------------------------------------------------------------
# Audio evidence, measured from the samples
# --------------------------------------------------------------------------

def _speech_threshold(floor: float) -> float:
    """Frame RMS above which a frame counts as speech-like.

    Relative to the recording's own floor (a fan at 180 RMS needs 540 to
    count), never below 120 (dither in a quiet room), and never above 700:
    continuous loud speech with few pauses has a *high* floor, and must not
    be able to raise the bar above itself.
    """

    return min(max(floor * 3.0, 120.0), 700.0)

@dataclass
class AudioEvidence:
    duration_seconds: float = 0.0
    rms: float = 0.0
    peak: int = 0
    #: Seconds of frames whose energy stands clearly above the recording's own floor.
    speech_seconds: float = 0.0
    speech_fraction: float = 0.0
    noise_floor: float = 0.0
    #: Exact-bytes digest (replay of the same buffer) ...
    fingerprint: str = ""
    #: ... and an envelope digest (the same recording re-encoded or re-sent).
    envelope: str = ""
    frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("duration_seconds", "rms", "speech_seconds", "speech_fraction", "noise_floor"):
            out[key] = round(float(out[key]), 3)
        return out

    @classmethod
    def from_pcm(cls, samples: bytes, sample_rate: int = 16000, width: int = 2, *, frame_ms: int = 20) -> "AudioEvidence":
        """Measure a mono PCM buffer.  Never raises: bad input is empty evidence."""

        if not samples or width != 2 or sample_rate <= 0:
            return cls(fingerprint=hashlib.sha1(samples or b"").hexdigest()[:16])
        try:
            import numpy as np

            data = np.frombuffer(samples[: len(samples) - (len(samples) % 2)], dtype="<i2").astype(np.float32)
            if data.size == 0:
                return cls(fingerprint=hashlib.sha1(samples).hexdigest()[:16])
            duration = data.size / sample_rate
            rms = float(np.sqrt(np.mean(data * data)) + 1e-9)
            peak = int(np.max(np.abs(data)))
            step = max(1, int(sample_rate * frame_ms / 1000))
            count = data.size // step
            if count == 0:
                frames_rms = np.array([rms], dtype=np.float32)
            else:
                trimmed = data[: count * step].reshape(count, step)
                frames_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-9)
            floor = float(max(np.percentile(frames_rms, 10), 1.0))
            threshold = _speech_threshold(floor)
            speech_frames = int(np.sum(frames_rms > threshold))
            envelope_src = np.round(frames_rms / max(1.0, float(frames_rms.max())) * 15).astype(np.uint8).tobytes()
        except Exception:  # noqa: BLE001 - numpy missing or odd buffer: pure-python path
            import array

            arr = array.array("h")
            arr.frombytes(samples[: len(samples) - (len(samples) % 2)])
            if not arr:
                return cls(fingerprint=hashlib.sha1(samples).hexdigest()[:16])
            duration = len(arr) / sample_rate
            rms = math.sqrt(sum(v * v for v in arr) / len(arr)) + 1e-9
            peak = max(abs(v) for v in arr)
            step = max(1, int(sample_rate * frame_ms / 1000))
            frames_rms_list = []
            for start in range(0, len(arr) - step + 1, step):
                chunk = arr[start:start + step]
                frames_rms_list.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)) + 1e-9)
            if not frames_rms_list:
                frames_rms_list = [rms]
            ordered = sorted(frames_rms_list)
            floor = max(ordered[int(len(ordered) * 0.1)], 1.0)
            threshold = _speech_threshold(floor)
            speech_frames = sum(1 for v in frames_rms_list if v > threshold)
            top = max(frames_rms_list) or 1.0
            envelope_src = bytes(int(round(v / top * 15)) for v in frames_rms_list)
            count = len(frames_rms_list)
        frame_seconds = frame_ms / 1000.0
        total_frames = max(1, count if count else 1)
        return cls(
            duration_seconds=duration,
            rms=rms,
            peak=peak,
            speech_seconds=speech_frames * frame_seconds,
            speech_fraction=speech_frames / total_frames,
            noise_floor=floor,
            fingerprint=hashlib.sha1(samples).hexdigest()[:16],
            envelope=hashlib.sha1(envelope_src).hexdigest()[:16],
            frames=int(total_frames),
        )


# --------------------------------------------------------------------------
# The utterance
# --------------------------------------------------------------------------

@dataclass
class UtteranceEvidence:
    """Everything known about one spoken request, before anything acts on it."""

    utterance_id: str
    session_id: str = ""
    #: microphone | ui_mic | test | unknown
    source: str = "microphone"
    wake_session_id: str = ""
    wake_score: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    language: str = ""
    final: bool = True
    audio: AudioEvidence = field(default_factory=AudioEvidence)
    #: What the device measured while recording (its own endpointer).
    device: dict[str, Any] = field(default_factory=dict)
    #: The recogniser's own quality signals.
    stt: dict[str, Any] = field(default_factory=dict)
    raw_transcript: str = ""
    normalized_transcript: str = ""
    confidence: float = 0.0
    #: True when the core believes it was speaking while this was recorded.
    speaking_overlap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id, "session_id": self.session_id, "source": self.source,
            "wake_session_id": self.wake_session_id, "wake_score": round(float(self.wake_score), 3),
            "started_at": self.started_at, "ended_at": self.ended_at, "language": self.language, "final": self.final,
            "audio": self.audio.to_dict(), "device": dict(self.device), "stt": dict(self.stt),
            "raw_transcript": self.raw_transcript, "normalized_transcript": self.normalized_transcript,
            "confidence": round(float(self.confidence), 3), "speaking_overlap": self.speaking_overlap,
        }


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    observed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    accepted: bool
    reason: str
    checks: list[Check] = field(default_factory=list)
    confidence: float = 0.0
    #: high | medium | low
    level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason, "confidence": round(self.confidence, 3), "level": self.level,
                "checks": [c.to_dict() for c in self.checks]}


@dataclass
class GateSettings:
    """Thresholds, in one place, each with its meaning."""

    #: Below this the buffer is a silent room whatever Whisper says (int16 RMS).
    min_rms: float = 35.0
    min_peak: int = 250
    #: Speech-like energy the recording itself must contain.
    min_speech_seconds: float = 0.25
    min_duration_seconds: float = 0.3
    #: Whisper's own belief that the segment contains no speech.
    max_no_speech_probability: float = 0.6
    #: Mean token log-probability; hallucinated text is usually far below.
    min_avg_logprob: float = -1.25
    #: Gzip-style compression ratio of the text; repetition loops go high.
    max_compression_ratio: float = 2.4
    #: Words per second of speech that a person can plausibly say.
    max_words_per_speech_second: float = 6.5
    min_words: int = 2
    min_confidence: float = 0.35
    duplicate_window_seconds: float = 12.0
    #: How similar a transcript must be to what ZEUS just said to count as echo.
    echo_similarity: float = 0.55
    echo_window_seconds: float = 6.0


_ADDRESS = {"zeus", "jarvis", "hey", "ok", "okay", "hallo", "hi", "du", "bitte"}
_COMPLETE_SHORT = {
    "stop", "stopp", "halt", "pause", "weiter", "ja", "nein", "yes", "no", "danke", "abbrechen", "cancel",
    "lauter", "leiser", "wiederhole", "nochmal", "hilfe", "help",
    "hallo", "hallo zeus", "hallo jarvis", "hi zeus", "hey zeus", "guten morgen", "guten abend", "gute nacht", "hello",
}


def normalise_text(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def similarity(a: str, b: str) -> float:
    a, b = normalise_text(a), normalise_text(b)
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    # Containment matters for echo: the microphone catches a phrase of the
    # answer, not the whole answer.
    if len(a) >= 12 and a in b:
        ratio = max(ratio, 0.9)
    return ratio


class AcceptanceGate:
    """Decides whether a transcript is a request, from evidence, with reasons."""

    def __init__(self, settings: GateSettings | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings or GateSettings()
        self._clock = clock
        self._recent: dict[str, float] = {}
        self.rejected: list[dict[str, Any]] = []

    # -- confidence -----------------------------------------------------

    def confidence(self, evidence: UtteranceEvidence) -> float:
        """0..1 from the recogniser's signals and the audio; 0.5 when nothing is known."""

        stt = evidence.stt or {}
        parts: list[float] = []
        if stt.get("avg_logprob") is not None:
            lp = float(stt["avg_logprob"])
            parts.append(max(0.0, min(1.0, 1.0 + (lp + 0.2) / 1.2)))  # -0.2 -> 1.0, -1.4 -> 0
        if stt.get("no_speech_probability") is not None:
            parts.append(max(0.0, min(1.0, 1.0 - float(stt["no_speech_probability"]))))
        words = stt.get("word_probabilities")
        if isinstance(words, list) and words:
            parts.append(max(0.0, min(1.0, sum(float(w) for w in words) / len(words))))
        if evidence.audio.frames:
            parts.append(max(0.0, min(1.0, evidence.audio.speech_seconds / 0.8)))
        if not parts:
            return max(0.0, min(1.0, evidence.confidence or 0.5))
        return sum(parts) / len(parts)

    @staticmethod
    def level(confidence: float) -> str:
        return "high" if confidence >= 0.72 else "medium" if confidence >= 0.45 else "low"

    # -- the decision ---------------------------------------------------

    def check(self, evidence: UtteranceEvidence, *, authorised: bool, recent_spoken: Iterable[tuple[str, float]] = (),
              ledger: "UtteranceLedger | None" = None) -> Verdict:
        s = self.settings
        checks: list[Check] = []
        text = (evidence.normalized_transcript or evidence.raw_transcript or "").strip()
        words = _WORD.findall(text)
        while len(words) > 1 and words[0].lower() in _ADDRESS:
            words = words[1:]
        audio = evidence.audio
        stt = evidence.stt or {}
        confidence = self.confidence(evidence)
        verdict_level = self.level(confidence)

        def fail(reason: str) -> Verdict:
            self.rejected.append({"text": text[:80], "reason": reason, "confidence": round(confidence, 3), "at": time.time(),
                                  "utterance_id": evidence.utterance_id, "session": evidence.session_id})
            del self.rejected[:-50]
            return Verdict(False, reason, checks, confidence, verdict_level)

        # 1. authority: a wake session or a press produced this audio
        checks.append(Check("authorised by a wake session or a press", authorised, evidence.source))
        if not authorised:
            return fail("no listening session: audio without a wake word or a press is not a request")

        # 2. identity / replay
        if ledger is not None:
            seen = ledger.seen(evidence)
            checks.append(Check("utterance not seen before", not seen, seen or "new"))
            if seen:
                return fail(f"replay: {seen}")

        # 3. the audio itself
        if audio.frames:
            checks.append(Check("recording long enough", audio.duration_seconds >= s.min_duration_seconds, f"{audio.duration_seconds:.2f}s"))
            if audio.duration_seconds < s.min_duration_seconds:
                return fail(f"audio too short: {audio.duration_seconds:.2f}s")
            loud_enough = audio.rms >= s.min_rms and audio.peak >= s.min_peak
            checks.append(Check("audio energy above a silent room", loud_enough, f"rms {audio.rms:.0f}, peak {audio.peak}"))
            if not loud_enough:
                return fail(f"silence: rms {audio.rms:.0f}, peak {audio.peak}")
            device_speech = float((evidence.device or {}).get("speech_seconds", 0.0) or 0.0)
            speech = max(audio.speech_seconds, device_speech)
            checks.append(Check("speech-like energy present", speech >= s.min_speech_seconds,
                                f"{audio.speech_seconds:.2f}s measured, {device_speech:.2f}s on the device"))
            if speech < s.min_speech_seconds:
                return fail(f"no speech energy: {audio.speech_seconds:.2f}s above the floor")
        else:
            checks.append(Check("audio evidence available", False, "no samples measured"))

        # 4. the recogniser's own doubt
        if not text:
            checks.append(Check("something was transcribed", False, "empty"))
            return fail("nothing heard")
        nsp = stt.get("no_speech_probability")
        if nsp is not None:
            checks.append(Check("Whisper believes there is speech", float(nsp) <= s.max_no_speech_probability, f"no_speech {float(nsp):.2f}"))
            if float(nsp) > s.max_no_speech_probability:
                return fail(f"whisper no-speech probability {float(nsp):.2f}")
        lp = stt.get("avg_logprob")
        if lp is not None:
            checks.append(Check("token probability plausible", float(lp) >= s.min_avg_logprob, f"avg_logprob {float(lp):.2f}"))
            if float(lp) < s.min_avg_logprob:
                return fail(f"implausible decoding: avg_logprob {float(lp):.2f}")
        cr = stt.get("compression_ratio")
        if cr is not None:
            checks.append(Check("no repetition loop", float(cr) <= s.max_compression_ratio, f"compression {float(cr):.2f}"))
            if float(cr) > s.max_compression_ratio:
                return fail(f"repetition: compression ratio {float(cr):.2f}")
        speech_for_rate = max(audio.speech_seconds, float((evidence.device or {}).get("speech_seconds", 0.0) or 0.0))
        if audio.frames and speech_for_rate > 0:
            rate = len(words) / max(speech_for_rate, 0.2)
            checks.append(Check("words fit the speech duration", rate <= s.max_words_per_speech_second, f"{len(words)} words in {speech_for_rate:.2f}s"))
            if rate > s.max_words_per_speech_second:
                return fail(f"{len(words)} words cannot fit in {speech_for_rate:.2f}s of speech")

        # 5. the text
        if len(words) < s.min_words and normalise_text(text) not in _COMPLETE_SHORT:
            checks.append(Check("more than a fragment", False, f"{len(words)} word(s)"))
            return fail(f"fragment: {len(words)} word(s) is not a request")
        if len(words) > 1 and len({w.lower() for w in words}) == 1:
            checks.append(Check("not one word repeated", False, text[:40]))
            return fail("fragment: one word repeated")
        if len(words) >= 6:
            counts: dict[str, int] = {}
            for w in words:
                counts[w.lower()] = counts.get(w.lower(), 0) + 1
            top = max(counts.values())
            if top / len(words) > 0.5:
                checks.append(Check("not a repetition loop", False, text[:60]))
                return fail("repetition: one token dominates the transcript")
        if text.endswith("...") or text.endswith("…"):
            checks.append(Check("transcript complete", False, "trails off"))
            return fail("incomplete: the transcript trails off")
        language = str(stt.get("language") or evidence.language or "")
        lang_prob = float(stt.get("language_probability") or 0.0)
        if language and language not in {"de", "en"} and lang_prob >= 0.85:
            checks.append(Check("language plausible for the owner", False, f"{language} ({lang_prob:.2f})"))
            return fail(f"implausible language {language} ({lang_prob:.2f})")
        checks.append(Check("text is a plausible request", True, f"{len(words)} words"))

        # 6. ZEUS hearing itself
        for spoken, at in recent_spoken:
            age = self._clock() - at
            if age > s.echo_window_seconds and not evidence.speaking_overlap:
                continue
            sim = similarity(text, spoken)
            if sim >= s.echo_similarity:
                checks.append(Check("not ZEUS's own speech", False, f"{sim:.2f} similar to what was just said"))
                return fail(f"self-echo: {sim:.2f} similar to ZEUS's own speech")
        checks.append(Check("not ZEUS's own speech", True, "no match with recent speech"))

        # 7. confidence floor
        checks.append(Check("confidence above the floor", confidence >= s.min_confidence, f"{confidence:.2f} ({verdict_level})"))
        if confidence < s.min_confidence:
            return fail(f"low confidence {confidence:.2f}")

        # 8. duplicates by text
        key = normalise_text(text)
        now = self._clock()
        self._recent = {k: t for k, t in self._recent.items() if now - t <= s.duplicate_window_seconds}
        last = self._recent.get(key)
        checks.append(Check("not a duplicate of a recent utterance", last is None, f"heard {now - last:.1f}s ago" if last is not None else "new"))
        if last is not None:
            return fail(f"duplicate: heard {now - last:.1f}s ago")
        self._recent[key] = now
        return Verdict(True, "accepted", checks, confidence, verdict_level)


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------

class UtteranceLedger:
    """Which utterances were accepted, so none is executed twice."""

    def __init__(self, *, window_seconds: float = 120.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._ids: dict[str, float] = {}
        self._fingerprints: dict[str, float] = {}
        self._envelopes: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        for table in (self._ids, self._fingerprints, self._envelopes):
            for key in [k for k, t in table.items() if now - t > self.window]:
                del table[key]

    def seen(self, evidence: UtteranceEvidence) -> str:
        """Why this is a replay, or "" when it is new."""

        now = self._clock()
        with self._lock:
            self._prune(now)
            if evidence.utterance_id and evidence.utterance_id in self._ids:
                return f"utterance {evidence.utterance_id} was already accepted"
            fp = evidence.audio.fingerprint
            if fp and fp in self._fingerprints:
                return f"identical audio was already accepted {now - self._fingerprints[fp]:.1f}s ago"
            env = evidence.audio.envelope
            if env and evidence.audio.frames >= 25 and env in self._envelopes:
                return f"the same recording was already accepted {now - self._envelopes[env]:.1f}s ago"
        return ""

    def accept(self, evidence: UtteranceEvidence) -> None:
        now = self._clock()
        with self._lock:
            if evidence.utterance_id:
                self._ids[evidence.utterance_id] = now
            if evidence.audio.fingerprint:
                self._fingerprints[evidence.audio.fingerprint] = now
            if evidence.audio.envelope and evidence.audio.frames >= 25:
                self._envelopes[evidence.audio.envelope] = now

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)
