"""The reported exchange must be the hops it is made of.

This file exists because the harness printed two numbers that were wrong in the
same way, and both looked authoritative.

``wake word  28.17s`` was not detection latency. ``_wake_score`` spawns a
subprocess in a separate venv, and the interpreter start plus the numpy and
openwakeword imports plus the ONNX model load were all folded into the hop.
Detection takes about two tenths of a second.

``whole exchange  53.45s`` was a wall-clock reading taken from the start of the
exchange to the end of the last thing the harness happened to do -- which was
spawning that same venv a second time to check the detector had reset. The hops
in that run summed to about four seconds.

Both are the measurement apparatus counted as product behaviour: the mirror
image of an acceptance check that reaches its own answer key. So the total is
now *derived* from the hops rather than held alongside them, and a hop has to
say it belongs to the exchange before it can contribute.
"""

from __future__ import annotations

from jarvis.measure_voice import VoiceMeasurement


def _measured() -> VoiceMeasurement:
    """A measurement shaped like a real run."""

    measurement = VoiceMeasurement()
    measurement.add("warm-up", 42.82, "models resident")
    measurement.add("synthesise the utterance", 0.28, "stands in for a microphone")
    measurement.add("wake word", 0.22, "score 0.997")
    measurement.add("transcribe", 1.09, "Hi Jarvis...", in_exchange=True)
    measurement.add("think and speak", 1.05, "1 phrase(s)", in_exchange=True)
    measurement.add("resume listening", 0.12, "detector reset")
    return measurement


# --------------------------------------------------------------------------
# The total cannot disagree with the breakdown
# --------------------------------------------------------------------------

def test_the_total_is_the_sum_of_the_exchange_hops():
    assert _measured().total_seconds == 1.09 + 1.05


def test_warm_up_is_not_part_of_the_exchange():
    """42s of model loading would otherwise dwarf everything the user waits for."""

    assert _measured().total_seconds < 42.82


def test_the_resume_check_cannot_inflate_the_exchange():
    """The bug: a venv subprocess spawned after the answer reported 53s."""

    measurement = _measured()
    before = measurement.total_seconds
    # However long the harness takes to confirm the detector reset...
    measurement.add("resume listening", 50.0, "slow venv start")

    assert measurement.total_seconds == before


def test_synthesising_the_utterance_is_not_part_of_the_exchange():
    """The user speaks; the harness has to fake that, and faking it is not latency."""

    measurement = VoiceMeasurement()
    measurement.add("synthesise the utterance", 9.0, "stands in for a microphone")

    assert measurement.total_seconds == 0.0


def test_a_measurement_with_no_hops_reports_zero():
    assert VoiceMeasurement().total_seconds == 0.0


# --------------------------------------------------------------------------
# What gets reported
# --------------------------------------------------------------------------

def test_the_report_shows_the_total_it_derived():
    text = _measured().describe()

    assert "whole exchange:                    2.14s" in text


def test_every_hop_is_still_shown_whether_or_not_it_counts():
    """Excluding a hop from the total must not hide it: the reader decides."""

    text = _measured().describe()

    for name in ("warm-up", "wake word", "resume listening", "transcribe"):
        assert name in text


def test_the_serialised_form_says_which_hops_counted():
    """A number in a JSON file outlives the person who knows what it meant."""

    payload = _measured().to_dict()
    counted = {hop["hop"] for hop in payload["hops"] if hop["in_exchange"]}

    assert counted == {"transcribe", "think and speak"}
    assert payload["total_seconds"] == 2.14


def test_time_to_first_audio_is_reported_separately():
    """It is the number that matters for the product, and it is not a hop.

    Speech starts before generation finishes, so first-audio is less than the
    exchange and cannot be derived from summing anything.
    """

    measurement = _measured()
    measurement.time_to_first_audio = 2.14

    assert "first audio out:  2.14s" in measurement.describe()
