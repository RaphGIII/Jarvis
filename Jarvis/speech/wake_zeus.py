"""The "Zeus" detector at the microphone: openWakeWord features, our classifier.

The listener feeds 80 ms frames.  openWakeWord's :class:`AudioFeatures` keeps
the streaming melspectrogram/embedding state; the classifier trained by
:mod:`speech.wake_training` scores the last 16 embeddings.  Plain numpy at
inference: two small matrix products and a sigmoid per frame.

A detection needs the score above the threshold on ``hits`` consecutive
frames -- one frame is 80 ms, and requiring two costs 80 ms of latency and
removes most single-frame spikes.  After a detection the detector is quiet for
``cooldown`` seconds so one "Zeus" is one wake, not five.

One preprocessing contract, shared by training, the Voice Studio test and the
listener: 16 kHz mono PCM in the int16 range.  :meth:`feed` accepts int16 or
float frames; a float frame whose values lie within [-1, 1] is treated as
normalised audio and scaled up, never truncated -- truncating it to int16 was
the bug that made the Voice Studio test score near-silence (0.000 … random
spikes) while the microphone path scored the real word.

The threshold is resolved by :func:`resolve_threshold`: the owner's explicit
``wake_sensitivity`` when set, otherwise the trained model's recommendation
from its manifest, otherwise 0.7.  ``wake_sensitivity`` *is* the threshold the
score must reach -- lower means more sensitive.  Nothing else (TTS volume,
microphone gain settings) enters the score.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "data" / "models" / "wake" / "zeus.npz"
#: When neither the owner nor a model manifest says otherwise.
FALLBACK_THRESHOLD = 0.7
#: Owner sensitivities outside this range are either never firing or always
#: firing; both are configuration mistakes rather than choices.
MIN_THRESHOLD, MAX_THRESHOLD = 0.05, 0.99


def resolve_threshold(sensitivity: Any, manifest_threshold: Any = None, *, fallback: float = FALLBACK_THRESHOLD) -> tuple[float, str]:
    """(effective threshold, where it came from: "owner" | "model" | "default")."""

    for value, source in ((sensitivity, "owner"), (manifest_threshold, "model")):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:  # NaN
            continue
        if source == "owner" and number <= 0:
            continue  # "" / 0 from an empty form field is "not set", not "fire on everything"
        return min(MAX_THRESHOLD, max(MIN_THRESHOLD, number)), source
    return fallback, "default"


def model_fingerprint(path: str | Path = DEFAULT_MODEL) -> str:
    """Short content hash of the weights, so two processes can prove they run the same model."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def read_manifest(path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    manifest = Path(path).with_suffix(".json")
    try:
        return json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    except (OSError, ValueError):
        return {}


def to_int16_frame(frame: np.ndarray) -> np.ndarray:
    """The one PCM contract: int16 samples.  Normalised floats are scaled, not truncated."""

    array = np.asarray(frame)
    if array.dtype == np.int16:
        return array
    values = array.astype(np.float32).reshape(-1)
    if values.size and float(np.max(np.abs(values))) <= 1.0:
        values = values * 32767.0
    return np.clip(values, -32768, 32767).astype(np.int16)


class ZeusDetector:
    def __init__(self, weights: dict[str, Any], *, threshold: float = FALLBACK_THRESHOLD, hits: int = 2, cooldown: float = 1.5,
                 features: Any = None, fingerprint: str = "") -> None:
        self.layers = []
        i = 1
        while f"W{i}" in weights:
            self.layers.append((np.asarray(weights[f"W{i}"], dtype=np.float32), np.asarray(weights[f"b{i}"], dtype=np.float32)))
            i += 1
        if not self.layers:  # the linear form from an older trainer
            self.layers = [(np.asarray(weights["W"], dtype=np.float32).reshape(-1, 1), np.asarray([weights["b"]], dtype=np.float32))]
        self.mean = np.asarray(weights["mean"], dtype=np.float32)
        self.std = np.asarray(weights["std"], dtype=np.float32)
        self.threshold = float(threshold)
        self.hits = max(1, int(hits))
        self.cooldown = cooldown
        self.fingerprint = fingerprint
        self._streak = 0
        self._last_fire = -1e9
        self.last_score = 0.0
        if features is None:
            from openwakeword.utils import AudioFeatures

            features = AudioFeatures(inference_framework="onnx")
        self.features = features

    @classmethod
    def from_weights(cls, weights: dict[str, Any], **kwargs: Any) -> "ZeusDetector":
        return cls(dict(weights), **kwargs)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL, **kwargs: Any) -> "ZeusDetector":
        """Weights from ``path``; the threshold from ``kwargs`` or, failing that, the manifest."""

        path = Path(path)
        data = dict(np.load(path))
        if kwargs.get("threshold") is None:
            kwargs["threshold"], _source = resolve_threshold(None, read_manifest(path).get("threshold"))
        kwargs.setdefault("fingerprint", model_fingerprint(path))
        return cls(data, **kwargs)

    @staticmethod
    def available(path: str | Path = DEFAULT_MODEL) -> bool:
        return Path(path).is_file()

    def score_window(self, window: np.ndarray) -> float:
        x = (window.reshape(-1).astype(np.float32) - self.mean) / self.std
        for index, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if index < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return float(1.0 / (1.0 + np.exp(-x.reshape(-1)[0])))

    def feed(self, frame: np.ndarray) -> bool:
        """One 1280-sample frame in (int16, or float); True when "Zeus" was just heard."""

        self.features(to_int16_frame(frame))
        window = self.features.get_features(16)[0]
        if window.shape[0] < 16:
            return False
        self.last_score = self.score_window(window)
        if self.last_score >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        now = time.monotonic()
        if self._streak >= self.hits and now - self._last_fire > self.cooldown:
            self._last_fire = now
            self._streak = 0
            return True
        return False

    def reset(self) -> None:
        """Forget everything about the previous audio -- including the cooldown.

        The cooldown is wall-clock time since the last fire.  Left alone across
        a reset, a detector that scores one clip after another suppresses
        every detection within 1.5 s of the previous clip's, which made an
        offline evaluation report recall 7/30 for a model that fires on 27/30.
        """

        self._streak = 0
        self._last_fire = -1e9
        self.last_score = 0.0
        try:
            self.features.reset()
        except Exception:
            pass
