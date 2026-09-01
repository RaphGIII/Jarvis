"""The owner wake word as one pipeline: settings -> threshold -> detector -> listener -> gate.

What went wrong on the live product, each pinned here so it stays fixed:

* ``wake_sensitivity`` and ``volume`` were not fields of ``VoiceSettings``;
  Save silently dropped them and the listener ran the manifest's 0.7 while
  the form showed 0.55.
* The Voice Studio test divided int16 PCM by 32768 and the detector cast the
  result back to int16 -- near-silence; scores were noise (0.000 … 0.99).
* ``ZeusDetector.reset()`` kept the wall-clock cooldown, so an offline
  evaluation suppressed most detections.
* Nothing distinguished a wake-triggered utterance from ambient speech, so a
  false activation's "Toys." became a request, an answer and a spoken reply.

No openWakeWord here: the detector takes a fake feature extractor whose
window is a function of the audio it was fed, which is enough to show that
scaling, threshold and cooldown behave -- the acoustic quality of the real
model is measured by ``speech.wake_eval`` on the owner's recordings instead.
"""

from __future__ import annotations

from _audio import speech_wav

import json
from pathlib import Path

import numpy as np
import pytest

from service.voice import UtteranceGate, VoiceService, VoiceSettings, normalise_utterance
from speech.wake_eval import recommend, score_pcm
from speech.wake_zeus import FALLBACK_THRESHOLD, ZeusDetector, resolve_threshold, to_int16_frame


# --------------------------------------------------------------------------
# A detector without openWakeWord
# --------------------------------------------------------------------------

class FakeFeatures:
    """16 x 96 window whose first column is the RMS of the last 16 frames / 1000."""

    def __init__(self) -> None:
        self.levels: list[float] = []
        self.resets = 0

    def __call__(self, frame: np.ndarray) -> None:
        assert frame.dtype == np.int16, "the detector must hand int16 to the feature backbone"
        self.levels.append(float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 1000.0)

    def get_features(self, n: int = 16):
        rows = ([0.0] * n + self.levels)[-n:]
        window = np.zeros((n, 96), dtype=np.float32)
        window[:, 0] = rows
        return window[None]

    def reset(self) -> None:
        self.levels.clear()
        self.resets += 1


def detector(threshold=0.5, **kwargs) -> ZeusDetector:
    # Linear form: score = sigmoid(8 * (level_of_last_frame - 0.5)) -> loud frame (rms 1000) ~ 0.98, silence ~ 0.02.
    W = np.zeros(16 * 96, dtype=np.float32)
    W[15 * 96] = 8.0
    weights = {"W": W, "b": np.float32(-4.0), "mean": np.zeros(16 * 96, np.float32), "std": np.ones(16 * 96, np.float32)}
    return ZeusDetector(weights, threshold=threshold, features=FakeFeatures(), **kwargs)


def loud(n=1280, amplitude=1000) -> np.ndarray:
    return np.full(n, amplitude, dtype=np.int16)


def quiet(n=1280) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


# --------------------------------------------------------------------------
# Threshold semantics: one number, one source
# --------------------------------------------------------------------------

def test_the_owner_setting_is_the_threshold():
    assert resolve_threshold(0.55, 0.7) == (0.55, "owner")


def test_without_an_owner_setting_the_model_recommendation_is_used():
    assert resolve_threshold(None, 0.8) == (0.8, "model")
    assert resolve_threshold("", 0.8) == (0.8, "model")


def test_an_empty_form_field_is_not_a_threshold_of_zero():
    """0 from Number("") would fire on every frame; it means "not set"."""

    assert resolve_threshold(0, 0.7) == (0.7, "model")


def test_with_nothing_configured_anywhere_the_fallback_applies():
    assert resolve_threshold(None, None) == (FALLBACK_THRESHOLD, "default")


def test_the_threshold_is_clamped_to_a_usable_range():
    assert resolve_threshold(5.0, None)[0] == 0.99
    assert resolve_threshold(0.001, None)[0] == 0.05


def test_the_manifest_threshold_is_used_by_load_when_no_threshold_is_given(tmp_path):
    W = np.zeros(16 * 96, dtype=np.float32)
    np.savez(tmp_path / "zeus.npz", W=W, b=np.float32(0), mean=np.zeros(16 * 96, np.float32), std=np.ones(16 * 96, np.float32))
    (tmp_path / "zeus.json").write_text(json.dumps({"threshold": 0.8}), encoding="utf-8")

    assert ZeusDetector.load(tmp_path / "zeus.npz", features=FakeFeatures()).threshold == 0.8
    assert ZeusDetector.load(tmp_path / "zeus.npz", features=FakeFeatures(), threshold=0.55).threshold == 0.55
    assert ZeusDetector.load(tmp_path / "zeus.npz", features=FakeFeatures()).fingerprint


# --------------------------------------------------------------------------
# PCM contract
# --------------------------------------------------------------------------

def test_normalised_float_audio_is_scaled_not_truncated():
    """The Voice Studio bug: int16 / 32768 -> astype(int16) is silence."""

    frame = loud()
    assert np.array_equal(to_int16_frame(frame.astype(np.float32) / 32768.0), (frame.astype(np.float32) / 32768.0 * 32767).astype(np.int16))
    assert int(np.abs(to_int16_frame(frame.astype(np.float32) / 32768.0)).max()) > 900


def test_int16_and_normalised_float_frames_score_the_same():
    a, b = detector(), detector()
    a.feed(loud())
    b.feed(loud().astype(np.float32) / 32768.0)

    assert abs(a.last_score - b.last_score) < 1e-3
    assert a.last_score > 0.9


def test_float_audio_already_in_the_int16_range_is_passed_through():
    a, b = detector(), detector()
    a.feed(loud())
    b.feed(loud().astype(np.float32))

    assert a.last_score == b.last_score


# --------------------------------------------------------------------------
# Firing, cooldown, reset
# --------------------------------------------------------------------------

def test_two_consecutive_frames_over_the_threshold_fire_once():
    det = detector()
    assert det.feed(loud()) is False, "one frame is a spike, not a word"
    assert det.feed(loud()) is True
    assert det.feed(loud()) is False, "cooldown: one word is one wake"


def test_reset_clears_the_cooldown_so_clips_can_be_scored_back_to_back():
    det = detector()
    det.feed(loud()); assert det.feed(loud())
    det.reset()
    det.feed(loud())

    assert det.feed(loud()) is True, "a reset detector must be able to fire immediately"
    assert det.features.resets == 1
    assert det.last_score == 0.0 or det.last_score > 0.9  # reset zeroed it, feeding set it again


def test_the_threshold_decides_whether_the_same_audio_fires():
    strict = detector(threshold=0.99)
    strict.feed(loud()); strict.feed(loud())
    assert strict.feed(loud()) is False and strict.last_score < 0.99

    lenient = detector(threshold=0.5)
    lenient.feed(loud())
    assert lenient.feed(loud()) is True


# --------------------------------------------------------------------------
# Evaluation: the same scorer, honest recommendations
# --------------------------------------------------------------------------

def test_score_pcm_pads_with_noise_not_zeros_and_reports_the_best_frame():
    det = detector()
    result = score_pcm(det, np.concatenate([quiet(), loud(), loud()]).astype(np.float32))

    assert result["score"] > 0.9 and result["detected"] is True
    assert len(result["frames"]) > 3, "the tail after the word was scored too"
    assert result["silent"] is False


def test_a_silent_recording_is_reported_as_silent():
    result = score_pcm(detector(), np.zeros(16000, dtype=np.float32) + 2.0)

    assert result["silent"] is True


def test_the_recommendation_rejects_every_negative_before_maximising_recall():
    positives = [[0.1, 0.9, 0.95, 0.2], [0.1, 0.6, 0.65, 0.1], [0.0, 0.85, 0.9, 0.0]]
    negatives = [[0.0, 0.5, 0.55, 0.0], [0.1, 0.1, 0.1, 0.1]]
    report = recommend(positives, negatives)

    assert report["separates"] is True
    assert report["recommended"] == 0.6
    by_t = {r["threshold"]: r for r in report["rows"]}
    assert by_t[0.5]["false_activations"] == 1 and by_t[0.6]["false_activations"] == 0
    assert by_t[0.6]["recall"] == 1.0 and by_t[0.7]["recall"] == pytest.approx(2 / 3, abs=1e-3)


def test_when_the_samples_do_not_separate_the_report_says_so():
    report = recommend([[0.5, 0.5]], [[0.9, 0.9]])

    assert report["separates"] is False


# --------------------------------------------------------------------------
# Settings: modelled, persisted, and volume kept away from the wake path
# --------------------------------------------------------------------------

def test_wake_sensitivity_and_volume_are_settings_that_persist(tmp_path):
    path = tmp_path / "voice" / "settings.json"
    settings = VoiceSettings()
    assert settings.apply({"wake_sensitivity": "0.55", "volume": 0.3}) == {}
    settings.save(path)

    again = VoiceSettings.load(path)
    assert again.wake_sensitivity == 0.55 and again.volume == 0.3


def test_an_unknown_setting_is_refused_by_name_not_dropped():
    assert VoiceSettings().apply({"sensitivity": 0.5}) == {"sensitivity": "unknown setting"}


def test_an_empty_sensitivity_means_not_set_and_volume_is_clamped():
    settings = VoiceSettings()
    settings.apply({"wake_sensitivity": "", "volume": 7})
    assert settings.wake_sensitivity is None and settings.volume == 1.0


def test_volume_is_not_an_input_of_the_wake_path():
    assert "volume" not in VoiceSettings.WAKE_INPUTS
    assert VoiceSettings(volume=0.1).wake_inputs() == {"wake_sensitivity": None}


def test_the_service_persists_what_it_applies(tmp_path):
    from service.events import EventBus

    service = VoiceService(EventBus(), settings_path=tmp_path / "s.json")
    assert service.update_settings({"wake_sensitivity": 0.6}) == {}

    assert VoiceService(EventBus(), settings_path=tmp_path / "s.json").settings.wake_sensitivity == 0.6


# --------------------------------------------------------------------------
# The core: one effective threshold for the test and the listener
# --------------------------------------------------------------------------

class KernelWithRoot:
    def __init__(self, root: Path) -> None:
        self.state_root = root
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        class P:
            def generate_stream(self, prompt, **_):
                yield "ok"
        return P()


@pytest.fixture()
def core(tmp_path):
    from service.core import JarvisCore

    return JarvisCore(kernel=KernelWithRoot(tmp_path))


def test_saving_the_sensitivity_changes_the_effective_threshold_and_saving_volume_does_not(core, tmp_path):
    before = core.wake_effective_threshold()
    core.voice_settings(wake_sensitivity=0.55)
    assert core.wake_effective_threshold() == (0.55, "owner")

    core.voice_settings(volume=0.2)
    assert core.wake_effective_threshold() == (0.55, "owner")
    assert core.voice.settings.volume == 0.2
    assert json.loads((tmp_path / "voice" / "settings.json").read_text(encoding="utf-8"))["wake_sensitivity"] == 0.55
    assert before[1] in {"model", "default"}


def test_the_status_reports_the_configured_and_effective_threshold(core):
    core.voice_settings(wake_sensitivity=0.55)
    status = core.wake_status()

    assert status["configured_sensitivity"] == 0.55
    assert status["effective_threshold"] == 0.55 and status["threshold_source"] == "owner"


def test_the_test_script_uses_the_effective_threshold_and_the_shared_scorer(core, monkeypatch):
    from service.core import JarvisCore

    script = JarvisCore.wake_test_script("C:/tmp/x.wav", 0.55)

    assert "score_wav" in script and "threshold=0.55" in script
    assert "32768" not in script, "the division that silenced every test recording"


def test_the_test_passes_the_owner_threshold_to_the_detector_regardless_of_volume(core, monkeypatch, tmp_path):
    import subprocess

    model = core._wake_model_path()
    monkeypatch.setattr(core, "_wake_model_path", lambda: tmp_path / "zeus.npz")
    (tmp_path / "zeus.npz").write_bytes(b"fake")
    monkeypatch.setattr(core, "_speech_python", lambda: "python")
    seen: list[str] = []

    def fake_run(command, **_):
        seen.append(command[-1])
        return type("R", (), {"returncode": 0, "stdout": json.dumps({"score": 0.93, "detected": True, "threshold": 0.55}), "stderr": ""})()

    monkeypatch.setattr(subprocess, "run", fake_run)
    core.voice_settings(wake_sensitivity=0.55)
    first = core.wake_test(b"RIFF" + bytes(3000))
    core.voice_settings(volume=0.05)
    second = core.wake_test(b"RIFF" + bytes(3000))

    assert first["ok"] and second["ok"]
    import re

    thresholds = [re.search(r"threshold=([0-9.]+)", script).group(1) for script in seen]
    assert thresholds == ["0.55", "0.55"], "volume changed nothing on the wake path"
    assert core.wake_status()["last_score"]["score"] == 0.93
    assert model.name == "zeus.npz"


def test_the_listener_report_is_compared_with_the_test_configuration(core, monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_wake_model_path", lambda: tmp_path / "zeus.npz")
    (tmp_path / "zeus.npz").write_bytes(b"weights")
    (tmp_path / "zeus.json").write_text(json.dumps({"threshold": 0.7, "dataset": {"owner_positive": 30}}), encoding="utf-8")
    core.voice_settings(wake_sensitivity=0.55)
    fingerprint = core.wake_status()["model_fingerprint"]

    core.wake_listener_report({"fingerprint": fingerprint, "threshold": 0.55, "pid": 1})
    assert core.wake_status()["listener_match"] is True
    assert core.wake_status()["model_kind"] == "OWNER"

    core.wake_listener_report({"fingerprint": fingerprint, "threshold": 0.7, "pid": 1})
    assert core.wake_status()["listener_match"] is False, "a listener on the stale 0.7 is visible"


# --------------------------------------------------------------------------
# The listener follows the core
# --------------------------------------------------------------------------

def test_the_listener_applies_the_effective_threshold_and_reloads_a_retrained_model():
    from speech.listener import _TrainedWake

    old = detector(threshold=0.7, fingerprint="old")
    new = detector(threshold=0.7, fingerprint="new")
    wake = _TrainedWake(old, "zeus", loader=lambda: new)

    changes = wake.sync({"model_fingerprint": "old", "effective_threshold": 0.55, "threshold_source": "owner"})
    assert wake.detector is old and old.threshold == 0.55 and changes == ["threshold 0.55 (owner)"]

    changes = wake.sync({"model_fingerprint": "new", "effective_threshold": 0.55})
    assert wake.detector is new and new.threshold == 0.55 and wake.reloads == 1
    assert wake.report()["fingerprint"] == "new" and wake.report()["threshold"] == 0.55


def test_the_listener_sends_the_wake_score_with_the_utterance(monkeypatch):
    import urllib.request

    from speech.listener import ListenerConfig, WakeListener

    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": true, "text": "hallo"}'

    def fake_urlopen(request, timeout=0):
        seen["headers"] = {k.lower(): v for k, v in request.header_items()}
        seen["url"] = request.full_url
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    listener = WakeListener(ListenerConfig(token="t", wake_model="zeus"))
    listener._send(bytes(3200), wake=0.93, session="abc")

    assert seen["url"].endswith("/api/voice/utterance")
    assert seen["headers"]["x-jarvis-wake"] == "0.9300" and seen["headers"]["x-jarvis-session"] == "abc"


# --------------------------------------------------------------------------
# The gate: ambient speech is not a request
# --------------------------------------------------------------------------

class T:
    def __init__(self, text, confidence=0.9):
        self.text, self.confidence = text, confidence


def test_audio_without_a_listening_session_is_never_a_request():
    gate = UtteranceGate(VoiceSettings())
    accepted, why = gate.check(T("mach das Licht an"), authorised=False)
    assert not accepted and "no listening session" in why


@pytest.mark.parametrize("text", ["Toys.", "So is...", "Jarvis, Toys.", "Toys, Toys, Toys."[:5]])
def test_fragments_are_rejected(text):
    gate = UtteranceGate(VoiceSettings())
    accepted, why = gate.check(T(text), authorised=True)
    assert not accepted, (text, why)


def test_a_real_sentence_passes_and_its_repeat_within_the_window_does_not():
    clock = [100.0]
    gate = UtteranceGate(VoiceSettings(duplicate_window_seconds=12), clock=lambda: clock[0])
    assert gate.check(T("wie spät ist es"), authorised=True) == (True, "")
    clock[0] += 3
    accepted, why = gate.check(T("Wie spät ist es?"), authorised=True)
    assert not accepted and why.startswith("duplicate")
    clock[0] += 20
    assert gate.check(T("wie spät ist es"), authorised=True)[0]


def test_low_confidence_is_rejected_but_unknown_confidence_is_not():
    gate = UtteranceGate(VoiceSettings(min_utterance_confidence=0.35))
    assert not gate.check(T("mach das licht an", confidence=0.1), authorised=True)[0]
    assert gate.check(T("mach das licht an", confidence=0.0), authorised=True)[0], "an engine that reports no confidence is not penalised"


def test_short_complete_requests_pass():
    gate = UtteranceGate(VoiceSettings())
    assert gate.check(T("Stop."), authorised=True)[0]


def test_normalisation_ignores_case_and_punctuation():
    assert normalise_utterance("Toys, Toys!") == normalise_utterance("toys toys")


def test_an_ambient_fragment_creates_no_message_no_receipt_no_mission(core):
    from service.voice import VoiceService
    from speech.contracts import Audio, Transcript

    class Engine:
        def status(self): return {"available": True, "voices": []}
        def transcribe(self, audio, *, language=""): return Transcript(text="Toys.", confidence=0.4)

    core._voice = VoiceService(core.bus, engine_factory=Engine)
    wav = speech_wav()

    result = core.hear(wav, wake=0.93)

    assert result["ok"] is False and result["ignored"] and "fragment" in result["reason"]
    assert core.history == []
    assert len(core.receipts.all()) == 0 if hasattr(core.receipts, "all") else True
    assert core.list_missions().get("missions", []) == []


def test_a_wake_authorised_sentence_is_a_request(core):
    from service.voice import VoiceService
    from speech.contracts import Audio, Transcript

    class Engine:
        def status(self): return {"available": True, "voices": []}
        def transcribe(self, audio, *, language=""): return Transcript(text="wie geht es weiter", confidence=0.9)

    core._voice = VoiceService(core.bus, engine_factory=Engine)
    wav = speech_wav()

    assert core.hear(wav, wake=0.93, answer=False)["ok"] is True
    assert core.hear(wav, answer=False)["ignored"], "the same words without a session are not a request"
