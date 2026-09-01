"""The `too_short` fix: blips re-arm the session; speech onset must be sustained.

Live evidence (listener.log): nine sessions entered CAPTURING 0.08-0.96 s
after the wake word -- the word's own tail or a breath -- and died
`too_short` while the owner was still about to speak.  Required behaviour:

    WAKE -> LISTENING (armed) -> [blip] -> LISTENING again (budget permitting)
         -> real speech -> CAPTURING -> UTTERANCE_CAPTURED

and both "Zeus, öffne Spotify" and "Zeus … [pause] … Öffne Spotify" work.
A failed session still ends exactly once, cleanly, with a reason.
"""

from __future__ import annotations

import numpy as np

from speech.listener import CaptureLoop, Endpointer, ListenerConfig

FRAME = 1280
LOUD, QUIET = 4000.0, 5.0


def frame(level: float = QUIET) -> np.ndarray:
    return np.full(FRAME, int(level), dtype=np.int16)


def run(loop: CaptureLoop, *, wake_at: int = 3, speech: list[tuple[int, int]] = (), frames: int = 160) -> list[tuple]:
    actions: list[tuple] = []
    t = 100.0
    for i in range(frames):
        loud = any(a <= i < b for a, b in speech)
        acts = loop.step(frame(LOUD if loud else QUIET), LOUD if loud else QUIET, i == wake_at, t)
        actions.extend(acts)
        for a in acts:
            if a[0] == "send":
                actions.extend(loop.sent(a[1], True))
        t += 0.08
    return actions


def states(actions):
    return [(a[2], a[3]) for a in actions if a[0] == "session"]


def test_the_wake_tail_blip_no_longer_kills_the_session():
    """One loud frame right after the wake (the tail of 'Zeus'), a 1 s pause, then the command."""

    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0, silence_seconds=0.9, total_listen_seconds=8.0))
    # wake at 3; frame 4 is the loud tail of the wake word; the command starts at 24 (1.6 s later)
    actions = run(loop, wake_at=3, speech=[(4, 5), (24, 44)])
    st = [s for s, _ in states(actions)]
    assert "UTTERANCE_CAPTURED" in st, states(actions)
    assert st.count("IDLE") == 1 and states(actions)[-1][1] == "sent"
    sends = [a for a in actions if a[0] == "send"]
    assert len(sends) == 1


def test_a_short_blip_mid_grace_rearms_instead_of_too_short():
    """A 0.16 s breath at 0.8 s, then the real command at 2.5 s."""

    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0, silence_seconds=0.9, total_listen_seconds=8.0, min_voiced_frames=1))
    actions = run(loop, wake_at=3, speech=[(13, 15), (35, 55)])
    st = states(actions)
    reasons = [r for s, r in st if s == "LISTENING"]
    assert any("re-armed" in r for r in reasons), st
    assert [s for s, _ in st].count("IDLE") == 1
    assert any(s == "UTTERANCE_CAPTURED" for s, _ in st)
    assert "too_short" not in [r for _, r in st]


def test_speech_onset_needs_a_sustained_run_of_voiced_frames():
    e = Endpointer(ListenerConfig(min_voiced_frames=3), frame_seconds=0.08, arm_seconds=3.0)
    e.track_noise(50.0)
    e.feed(4000.0)  # one loud frame: a click
    assert e.heard_speech is False
    e.feed(10.0)
    e.feed(4000.0)
    e.feed(4000.0)
    assert e.heard_speech is False, "two in a row is still not onset"
    e.feed(4000.0)
    assert e.heard_speech is True, "three in a row is speech"


def test_a_session_with_no_speech_at_all_still_ends_once_no_loops():
    loop = CaptureLoop(ListenerConfig(arm_seconds=1.0, total_listen_seconds=2.0))
    actions = run(loop, wake_at=3, speech=[], frames=120)
    st = states(actions)
    idles = [(s, r) for s, r in st if s == "IDLE"]
    assert len(idles) == 1 and idles[0][1] == "no_speech_after_wake"
    assert loop.state == "IDLE"


def test_the_budget_is_finite_endless_blips_end_the_session():
    """Blips every second for ever: the session must not re-arm past its budget."""

    cfg = ListenerConfig(arm_seconds=3.0, silence_seconds=0.5, total_listen_seconds=4.0, min_voiced_frames=1)
    loop = CaptureLoop(cfg)
    blips = [(i, i + 1) for i in range(5, 200, 14)]
    actions = run(loop, wake_at=3, speech=blips, frames=220)
    st = states(actions)
    ends = [(s, r) for s, r in st if s == "IDLE"]
    assert len(ends) == 1, st
    assert ends[0][1] in {"too_short", "no_speech_after_wake"}
    assert loop.state == "IDLE"


def test_immediate_speech_still_works_unchanged():
    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0, silence_seconds=0.9, total_listen_seconds=8.0))
    actions = run(loop, wake_at=3, speech=[(4, 24)])
    st = [s for s, _ in states(actions)]
    assert st == ["WAKE_DETECTED", "LISTENING", "CAPTURING", "UTTERANCE_CAPTURED", "SENT", "IDLE"] or \
           st == ["WAKE_DETECTED", "LISTENING", "CAPTURING", "UTTERANCE_CAPTURED", "IDLE"], states(actions)


def test_wake_tail_token_with_low_probability_is_stripped_whatever_it_spells():
    from speech.wake_segment import strip_wake_word

    got = strip_wake_word("Das, öffne Spotify.", wake_word="Zeus",
                          words=[{"word": "Das", "start": 0.0, "end": 0.2, "probability": 0.21},
                                 {"word": "öffne", "start": 0.7, "end": 1.0, "probability": 0.9}])
    assert got.text == "Öffne Spotify." and got.removed
    # a confident early command word stays
    kept = strip_wake_word("Öffne Spotify.", wake_word="Zeus",
                           words=[{"word": "Öffne", "start": 0.1, "end": 0.4, "probability": 0.95}])
    assert kept.text == "Öffne Spotify." and not kept.removed
