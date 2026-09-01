"""Synthetic audio for the voice tests.

The acceptance gate measures the samples, so a test that wants a transcript
to be *accepted* has to post something that looks like speech -- bursts of
energy with pauses -- and a test that wants silence refused posts silence.
Whisper never runs here; the fake engines decide the words.
"""

from __future__ import annotations

import math
import random

from speech.contracts import Audio

RATE = 16000


def speech_pcm(seconds: float = 1.2, *, level: int = 2500, seed: int = 7, pause_every: float = 0.35) -> bytes:
    """Modulated noise bursts: speech-like energy with syllable pauses."""

    rng = random.Random(seed)
    total = int(RATE * seconds)
    out = bytearray()
    for i in range(total):
        t = i / RATE
        phase = (t % pause_every) / pause_every
        envelope = 0.15 if phase > 0.8 else 0.6 + 0.4 * math.sin(phase * math.pi)
        sample = int(max(-32767, min(32767, rng.gauss(0, level * envelope))))
        out += sample.to_bytes(2, "little", signed=True)
    return bytes(out)


def speech_wav(seconds: float = 1.2, **kwargs) -> bytes:
    return Audio(samples=speech_pcm(seconds, **kwargs), sample_rate=RATE).to_wav()


def silence_wav(seconds: float = 1.5) -> bytes:
    return Audio(samples=bytes(int(RATE * seconds) * 2), sample_rate=RATE).to_wav()


def noise_wav(seconds: float = 1.5, *, level: int = 180, seed: int = 3) -> bytes:
    """A fan: steady, unmodulated noise well above digital silence."""

    rng = random.Random(seed)
    out = bytearray()
    for _ in range(int(RATE * seconds)):
        sample = int(max(-32767, min(32767, rng.gauss(0, level))))
        out += sample.to_bytes(2, "little", signed=True)
    return Audio(samples=bytes(out), sample_rate=RATE).to_wav()


def clicks_wav(seconds: float = 1.5, *, every: float = 0.25, seed: int = 5) -> bytes:
    """Keyboard: short sharp transients on a quiet floor."""

    rng = random.Random(seed)
    out = bytearray()
    total = int(RATE * seconds)
    click_len = int(RATE * 0.012)
    for i in range(total):
        in_click = (i % int(RATE * every)) < click_len
        sample = int(max(-32767, min(32767, rng.gauss(0, 6000 if in_click else 12))))
        out += sample.to_bytes(2, "little", signed=True)
    return Audio(samples=bytes(out), sample_rate=RATE).to_wav()
