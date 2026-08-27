"""The listening session as a state machine, and stop semantics kept apart.

Live failure this pins: saying "Zeus" in the main interface produced a
transcript full of the literal "stopped" and no listening session -- the
listener posted a generic stop on every wake and the natural pause after the
name counted as end-of-speech before the command began.
"""

from __future__ import annotations

import numpy as np
import pytest

from service.core import JarvisCore
from service.events import EventType
from service.state import JarvisState
from speech.listener import CaptureLoop, ListenerConfig, VoiceSession

FRAME = 1280
LOUD, QUIET = 4000.0, 5.0


def frame(level: float = QUIET) -> np.ndarray:
    return np.full(FRAME, int(level), dtype=np.int16)


def run(loop: CaptureLoop, *, wake_at: int = 3, speech: list[tuple[int, int]] = (), frames: int = 120) -> list[tuple]:
    """Feed `frames` 80 ms frames; speech=[(start_frame, end_frame)] are loud."""

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


# --------------------------------------------------------------------------
# Wake -> armed long enough for speech
# --------------------------------------------------------------------------

def test_wake_opens_a_session_that_stays_armed_through_the_pause_before_the_command():
    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0, silence_seconds=0.9))
    # wake at frame 3; the owner starts speaking 1.6 s later (frame 23) for 1.2 s
    actions = run(loop, wake_at=3, speech=[(23, 38)])

    st = states(actions)
    assert [s for s, _ in st] == ["WAKE_DETECTED", "LISTENING", "CAPTURING", "UTTERANCE_CAPTURED", "IDLE"]
    sends = [a for a in actions if a[0] == "send"]
    assert len(sends) == 1, "exactly one captured utterance"
    assert st[-1][1] == "sent"
    assert loop.sessions_opened == 1 and loop.sessions_ended == 1


def test_wake_without_speech_times_out_once_cleanly():
    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0))
    actions = run(loop, wake_at=2, speech=[], frames=120)

    st = states(actions)
    assert [s for s, _ in st] == ["WAKE_DETECTED", "LISTENING", "IDLE"]
    assert st[-1][1] == "no_speech_after_wake"
    assert not [a for a in actions if a[0] == "send"]
    assert loop.sessions_ended == 1
    assert loop.state == "IDLE"
    # 3.0 s of grace = 37-38 frames; the session must not have ended earlier
    idle_index = next(i for i, a in enumerate(actions) if a[0] == "session" and a[2] == "IDLE")
    assert loop.endpointer.elapsed >= 2.9


def test_a_wake_during_cooldown_does_not_open_a_second_session():
    loop = CaptureLoop(ListenerConfig(arm_seconds=1.0, cooldown_seconds=1.5))
    actions = run(loop, wake_at=2, speech=[], frames=20)   # times out at ~1.0 s
    assert loop.sessions_opened == 1
    # another wake 0.3 s later is inside the cooldown
    acts = loop.step(frame(), QUIET, True, 100.0 + 20 * 0.08 + 0.3)
    assert acts == [] and loop.sessions_opened == 1


def test_speech_that_is_too_short_ends_with_that_reason_and_no_post():
    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0, min_utterance_seconds=0.35, silence_seconds=0.5))
    actions = run(loop, wake_at=2, speech=[(10, 12)], frames=60)   # 0.16 s of speech
    st = states(actions)
    assert st[-1] == ("IDLE", "too_short")
    assert not [a for a in actions if a[0] == "send"]


def test_barge_in_is_requested_once_per_wake_and_never_ends_the_session():
    loop = CaptureLoop(ListenerConfig(arm_seconds=3.0, barge_in=True))
    actions = run(loop, wake_at=2, speech=[(20, 35)])
    interrupts = [a for a in actions if a[0] == "interrupt"]
    assert len(interrupts) == 1
    # the interrupt is requested while the session is LISTENING, and the session went on to capture
    assert [s for s, _ in states(actions)][-2:] == ["UTTERANCE_CAPTURED", "IDLE"]


def test_a_session_ends_exactly_once():
    session = VoiceSession(session_id="s1", wake_score=0.9)
    assert session.transition("LISTENING", "armed")
    assert session.end("silence") is True
    assert session.end("silence") is False and session.end("other") is False
    assert session.state == "IDLE" and session.end_reason == "silence"
    assert not session.transition("CAPTURING", "late"), "no transitions after the end"


def test_invalid_transitions_are_refused():
    session = VoiceSession(session_id="s1", wake_score=0.9)
    assert not session.transition("SENT", "skip")
    assert session.state == "WAKE_DETECTED"


# --------------------------------------------------------------------------
# The core: stop semantics kept apart, idempotent, no transcript pollution
# --------------------------------------------------------------------------

class StubKernel:
    def __init__(self):
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        class P:
            def generate_stream(self, prompt, **_):
                yield "ok"
        return P()


def events(core, fn):
    with core.bus.subscribe(replay=False) as sub:
        result = fn()
        return result, sub.drain()


def test_barge_in_with_nothing_running_is_a_silent_no_op():
    core = JarvisCore(kernel=StubKernel())
    result, evs = events(core, lambda: core.voice_interrupt(session="vs1", wake=0.93))

    assert result == {"ok": True, "interrupted": []}
    assert not [e for e in evs if e.type is EventType.NOTIFICATION], "no transcript entry"
    assert [e for e in evs if e.type is EventType.DIAGNOSTIC]
    assert core.state.snapshot.state is JarvisState.IDLE


def test_barge_in_while_speaking_stops_speech_and_enters_listening():
    core = JarvisCore(kernel=StubKernel())
    core.state.set(JarvisState.SPEAKING, detail="reading an answer")
    result, evs = events(core, lambda: core.voice_interrupt(session="vs2", wake=0.9))

    assert "speech" in result["interrupted"] and "answer" in result["interrupted"]
    notes = [e.payload for e in evs if e.type is EventType.NOTIFICATION]
    assert len(notes) == 1 and notes[0]["kind"] == "barge_in" and notes[0]["text"] != "stopped"
    assert core.state.snapshot.state is JarvisState.LISTENING


def test_repeated_stops_are_idempotent_and_do_not_pollute_the_transcript():
    core = JarvisCore(kernel=StubKernel())
    core.state.set(JarvisState.THINKING, detail="answering")
    first, evs1 = events(core, lambda: core.stop_current(reason="esc"))
    second, evs2 = events(core, lambda: core.stop_current(reason="esc"))
    third, evs3 = events(core, lambda: core.voice_interrupt(session="vs3"))

    assert first["stopped"] == ["answer"] and second["stopped"] == [] and third["interrupted"] == []
    assert len([e for e in evs1 if e.type is EventType.NOTIFICATION]) == 1
    assert not [e for e in evs2 + evs3 if e.type is EventType.NOTIFICATION]
    assert all("stopped" != (e.payload.get("text") or "") for e in evs1 + evs2 + evs3)
    assert core.history == []


def test_session_events_mirror_into_the_core_state_and_the_log():
    core = JarvisCore(kernel=StubKernel())
    _, evs = events(core, lambda: [core.voice_session_event("vs4", "WAKE_DETECTED", "wake word", wake=0.9),
                                   core.voice_session_event("vs4", "LISTENING", "armed for 3.0s"),
                                   core.voice_session_event("vs4", "IDLE", "no_speech_after_wake")])
    diags = [e.payload for e in evs if e.type is EventType.DIAGNOSTIC and "voice_session" in e.payload]
    assert [d["voice_session"] for d in diags] == ["WAKE_DETECTED", "LISTENING", "IDLE"]
    assert diags[-1]["reason"] == "no_speech_after_wake"
    assert core.state.snapshot.state is JarvisState.IDLE
    assert not [e for e in evs if e.type is EventType.NOTIFICATION]


def test_a_session_event_never_downgrades_working_state():
    core = JarvisCore(kernel=StubKernel())
    core.state.set(JarvisState.WORKING, detail="mission")
    core.voice_session_event("vs5", "IDLE", "silence")
    assert core.state.snapshot.state is JarvisState.WORKING
