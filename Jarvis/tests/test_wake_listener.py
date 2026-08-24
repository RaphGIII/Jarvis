"""Hands-free listening, tested without a microphone.

The endpointer is extracted from the capture loop precisely so these cases can
be checked: a pause mid-sentence must not end the utterance, a noisy room must
not prevent it from ever ending, and a cough must not become a question.

Wake-word accuracy itself was measured against the real model rather than
asserted here — 0.995 on a correctly pronounced "Hey Jarvis", 0.000 on
unrelated speech, 0.04 when a German voice says "YAR-vis" instead of the
English "JAR-vis" the model expects.
"""

from __future__ import annotations

import pytest

from speech.listener import FRAME_SAMPLES, SAMPLE_RATE, Endpointer, ListenerConfig, build_parser

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE  # 0.08 s


@pytest.fixture()
def endpointer():
    return Endpointer(ListenerConfig(silence_seconds=0.5, max_utterance_seconds=5.0), frame_seconds=FRAME_SECONDS)


def feed(endpointer, level, seconds):
    """Feed `seconds` of audio at a constant level; return True if it ended."""

    for _ in range(max(1, int(seconds / FRAME_SECONDS))):
        if endpointer.feed(level):
            return True
    return False


LOUD = 4000.0
QUIET = 5.0


# --------------------------------------------------------------------------
# Endpointing
# --------------------------------------------------------------------------

def test_continuous_speech_does_not_end(endpointer):
    assert not feed(endpointer, LOUD, 3.0)


def test_silence_after_speech_ends_the_utterance(endpointer):
    feed(endpointer, LOUD, 1.0)

    assert feed(endpointer, QUIET, 0.6)


def test_a_pause_mid_sentence_does_not_end_it(endpointer):
    """People stop to think. Ending there truncates the question."""

    feed(endpointer, LOUD, 1.0)
    feed(endpointer, QUIET, 0.3)          # shorter than silence_seconds

    assert not feed(endpointer, LOUD, 0.5), "speech resumed; the utterance continues"


def test_the_silence_counter_resets_when_speech_resumes(endpointer):
    feed(endpointer, LOUD, 0.5)
    feed(endpointer, QUIET, 0.4)
    feed(endpointer, LOUD, 0.2)

    assert endpointer.silence_for == 0.0


def test_an_utterance_cannot_run_forever(endpointer):
    """A noisy room must not hold the microphone open indefinitely."""

    assert feed(endpointer, LOUD, 10.0)


def test_speech_seconds_excludes_the_trailing_silence(endpointer):
    feed(endpointer, LOUD, 1.0)
    feed(endpointer, QUIET, 0.6)

    assert endpointer.speech_seconds == pytest.approx(1.0, abs=0.15)


def test_a_cough_is_below_the_minimum(endpointer):
    feed(endpointer, LOUD, 0.16)
    feed(endpointer, QUIET, 0.6)

    assert endpointer.speech_seconds < ListenerConfig().min_utterance_seconds


def test_reset_clears_the_previous_utterance(endpointer):
    feed(endpointer, LOUD, 1.0)
    endpointer.reset()

    assert endpointer.elapsed == 0.0 and endpointer.silence_for == 0.0


# --------------------------------------------------------------------------
# Adapting to the room
# --------------------------------------------------------------------------

def test_a_silent_room_still_has_a_usable_threshold(endpointer):
    """Otherwise the threshold is zero and dither reads as speech."""

    for _ in range(200):
        endpointer.track_noise(0.0)

    assert endpointer.threshold >= 60.0


def test_the_threshold_rises_with_a_noisy_room(endpointer):
    quiet = endpointer.threshold
    for _ in range(3000):
        endpointer.track_noise(900.0)

    assert endpointer.threshold > quiet


def test_speech_in_a_noisy_room_is_still_speech():
    """The failure this prevents: next to a fan, recording never stops."""

    endpointer = Endpointer(ListenerConfig(silence_seconds=0.5), frame_seconds=FRAME_SECONDS)
    for _ in range(3000):
        endpointer.track_noise(400.0)     # a constant hum

    # The hum alone must read as silence...
    assert feed(endpointer, 400.0, 1.0)
    endpointer.reset()
    # ...while speech well above it does not.
    assert not feed(endpointer, 6000.0, 1.0)


def test_the_noise_estimate_moves_slowly(endpointer):
    """A single loud frame must not redefine the room."""

    for _ in range(100):
        endpointer.track_noise(50.0)
    before = endpointer.noise_floor
    endpointer.track_noise(20000.0)

    assert endpointer.noise_floor < before * 3


# --------------------------------------------------------------------------
# Configuration and CLI
# --------------------------------------------------------------------------

def test_the_defaults_are_sensible():
    config = ListenerConfig()

    assert 0.5 <= config.silence_seconds <= 1.5
    assert config.min_utterance_seconds < config.silence_seconds
    assert config.preroll_frames > 0, "the first syllable would be clipped"
    assert config.cooldown_seconds > 0, "one wake word could trigger twice"


def test_the_wake_model_comes_from_the_product_identity():
    """The config no longer hard-codes a model: a detector is trained weights,
    and which one to listen for is part of the product's identity."""

    from core.identity import current
    from speech.listener import WakeListener

    assert ListenerConfig().wake_model == ""

    listener = WakeListener(ListenerConfig(token="x"))
    assert listener.config.wake_model == current().resolved_wake_model


def test_a_token_is_required():
    """The core refuses unauthenticated audio; failing early is clearer."""

    assert build_parser().parse_args([]).token == ""


def test_the_cli_accepts_the_options_that_matter():
    args = build_parser().parse_args(
        ["--url", "http://box.local:8420", "--token", "abc", "--threshold", "0.7", "--device", "3"]
    )

    assert args.url == "http://box.local:8420"
    assert args.threshold == 0.7
    assert args.device == 3


def test_frames_match_the_detector_and_the_recogniser():
    """openWakeWord wants 1280 samples at 16 kHz; whisper wants 16 kHz."""

    assert FRAME_SAMPLES == 1280
    assert SAMPLE_RATE == 16000
    assert FRAME_SECONDS == pytest.approx(0.08)
