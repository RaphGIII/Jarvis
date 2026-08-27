"""The "Zeus" detector at the microphone: openWakeWord features, our classifier.

The listener feeds 80 ms frames.  openWakeWord's :class:`AudioFeatures` keeps
the streaming melspectrogram/embedding state; the classifier trained by
:mod:`speech.wake_training` scores the last 16 embeddings.  Plain numpy at
inference: two small matrix products and a sigmoid per frame.

A detection needs the score above the threshold on ``hits`` consecutive
frames -- one frame is 80 ms, and requiring two costs 80 ms of latency and
removes most single-frame spikes.  After a detection the detector is quiet for
``cooldown`` seconds so one "Zeus" is one wake, not five.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "data" / "models" / "wake" / "zeus.npz"


class ZeusDetector:
    def __init__(self, weights: dict[str, Any], *, threshold: float = 0.7, hits: int = 2, cooldown: float = 1.5,
                 features: Any = None) -> None:
        self.layers = []
        i = 1
        while f"W{i}" in weights:
            self.layers.append((np.asarray(weights[f"W{i}"], dtype=np.float32), np.asarray(weights[f"b{i}"], dtype=np.float32)))
            i += 1
        if not self.layers:  # the linear form from an older trainer
            self.layers = [(np.asarray(weights["W"], dtype=np.float32).reshape(-1, 1), np.asarray([weights["b"]], dtype=np.float32))]
        self.mean = np.asarray(weights["mean"], dtype=np.float32)
        self.std = np.asarray(weights["std"], dtype=np.float32)
        self.threshold = threshold
        self.hits = max(1, int(hits))
        self.cooldown = cooldown
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
        path = Path(path)
        data = dict(np.load(path))
        manifest = path.with_suffix(".json")
        if manifest.is_file() and "threshold" not in kwargs:
            try:
                kwargs["threshold"] = float(json.loads(manifest.read_text(encoding="utf-8")).get("threshold", 0.7))
            except (OSError, ValueError):
                pass
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
        """One 1280-sample int16 frame in; True when "Zeus" was just heard."""

        self.features(frame.astype(np.int16))
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
        self._streak = 0
        try:
            self.features.reset()
        except Exception:
            pass
