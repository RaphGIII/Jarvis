"""What Jarvis is doing, as one small vocabulary shared by every surface.

The eye animation, the CLI status line, a future device's LED ring and the TV
view must all agree about what is happening.  That only works if "what is
happening" is a closed set of names owned by the core, rather than each client
inferring a mood from whatever it happens to observe.

:class:`JarvisState` is that set.  It answers one question -- what is Jarvis
doing right now -- and deliberately not two others: *why* (that is the event
payload) and *how it should look* (that belongs to the client).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class JarvisState(str, Enum):
    """The states every client knows how to render."""

    #: Nothing to do.  Awake and waiting.
    IDLE = "idle"
    #: Microphone open, capturing speech.
    LISTENING = "listening"
    #: Speech captured, being turned into text.
    TRANSCRIBING = "transcribing"
    #: A model is generating.
    THINKING = "thinking"
    #: Speaking aloud.
    SPEAKING = "speaking"
    #: A project or mission is running.
    WORKING = "working"
    #: An action has run and its effect is being checked independently.
    #: Distinct from WORKING on purpose: "I did it" and "I confirmed it" are
    #: different claims, and the interface should not blur them when the whole
    #: point of the action path is that the second one is earned.
    VERIFYING = "verifying"
    #: Gathering information from documents or the web.
    RESEARCHING = "researching"
    #: Writing or editing code.
    CODING = "coding"
    #: Blocked on the user.
    WAITING = "waiting"
    #: Something failed and the user should know.
    ERROR = "error"
    #: No brain is reachable.
    OFFLINE = "offline"

    @property
    def busy(self) -> bool:
        """True while Jarvis is doing something that takes time."""

        return self in {
            JarvisState.THINKING,
            JarvisState.WORKING,
            JarvisState.VERIFYING,
            JarvisState.RESEARCHING,
            JarvisState.CODING,
            JarvisState.TRANSCRIBING,
        }

    @property
    def accepts_speech(self) -> bool:
        """Whether a wake word or barge-in should be acted on in this state.

        Everything except ERROR and OFFLINE, including SPEAKING: interrupting
        Jarvis mid-sentence is the single most important thing barge-in has to
        support, so a state machine that refused input while talking would
        defeat the feature it exists to serve.
        """

        return self not in {JarvisState.ERROR, JarvisState.OFFLINE}


@dataclass
class StateSnapshot:
    """The current state plus the little that clients need alongside it."""

    state: JarvisState = JarvisState.IDLE
    detail: str = ""
    since: str = ""
    #: Optional project/mission this state belongs to.
    scope: str = ""
    #: 0..1 where a meaningful fraction exists; None when it does not.
    progress: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "detail": self.detail,
            "since": self.since,
            "scope": self.scope,
            "progress": self.progress,
            "busy": self.state.busy,
            **self.extra,
        }


class StateMachine:
    """Holds the current state and notifies on change.

    Transitions are unrestricted on purpose.  A rule table would have to encode
    every legitimate jump -- WORKING straight to LISTENING when the user
    interrupts a build, SPEAKING to LISTENING on barge-in, anything to ERROR --
    and the first time reality disagreed with the table, the truthful state
    would be the one rejected.  The state must always be able to describe what
    is actually happening.
    """

    def __init__(self, on_change: Callable[[StateSnapshot], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._snapshot = StateSnapshot(state=JarvisState.IDLE, since=_now())
        self._on_change = on_change

    @property
    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def state(self) -> JarvisState:
        return self.snapshot.state

    def set(
        self,
        state: JarvisState,
        *,
        detail: str = "",
        scope: str = "",
        progress: float | None = None,
        **extra: Any,
    ) -> StateSnapshot:
        """Move to a state and tell anyone listening."""

        with self._lock:
            previous = self._snapshot
            unchanged = (
                previous.state is state
                and previous.detail == detail
                and previous.scope == scope
                and previous.progress == progress
                and not extra
            )
            if unchanged:
                return previous
            snapshot = StateSnapshot(
                state=state,
                detail=detail,
                # A state that has not really changed keeps its original start
                # time, so "thinking for 40 seconds" stays true across detail
                # updates instead of resetting on every progress line.
                since=previous.since if previous.state is state else _now(),
                scope=scope,
                progress=progress,
                extra=dict(extra),
            )
            self._snapshot = snapshot
            callback = self._on_change

        if callback is not None:
            callback(snapshot)
        return snapshot

    def busy_with(self, state: JarvisState, detail: str = "", **extra: Any) -> "_StateContext":
        """Enter a state for the duration of a block, then return to idle."""

        return _StateContext(self, state, detail, extra)


class _StateContext:
    def __init__(self, machine: StateMachine, state: JarvisState, detail: str, extra: dict[str, Any]) -> None:
        self._machine = machine
        self._state = state
        self._detail = detail
        self._extra = extra
        self._previous: StateSnapshot | None = None

    def __enter__(self) -> StateMachine:
        self._previous = self._machine.snapshot
        self._machine.set(self._state, detail=self._detail, **self._extra)
        return self._machine

    def __exit__(self, exc_type: type | None, exc: BaseException | None, _tb: object) -> None:
        if exc_type is not None:
            self._machine.set(JarvisState.ERROR, detail=str(exc)[:200])
            return
        previous = self._previous
        if previous is not None and not previous.state.busy:
            self._machine.set(previous.state, detail=previous.detail, scope=previous.scope)
        else:
            self._machine.set(JarvisState.IDLE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
