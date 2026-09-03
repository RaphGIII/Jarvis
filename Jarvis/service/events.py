"""The event bus every Jarvis client watches.

One design decision explains the shape of this module: *the core never knows who
is listening*.  A desktop browser, a TV in kiosk mode, a future portable device
and the CLI all consume the same stream, so nothing here may assume a single
client, a connected client, or a client that keeps up.

That produces three requirements which are easy to get wrong:

*A slow subscriber must not stall the core.*  Publishing never blocks.  Each
subscriber owns a bounded queue, and a subscriber that stops draining loses its
oldest events rather than applying backpressure to the thing generating them --
a paused browser tab must not be able to freeze an autonomous build.

*A client that connects late must not see a blank screen.*  The bus keeps a
short replay buffer, so a page opened mid-mission immediately learns the current
state instead of waiting for the next thing to happen.

*Sequence numbers, not timestamps, order the stream.*  Two events in the same
millisecond are common; a clock that steps backwards is rare but ruinous.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator


class EventType(str, Enum):
    """What kinds of thing a client may be told about.

    Kept deliberately small and stable: this is a wire format, and every client
    -- including ones not written yet -- has to understand it.
    """

    #: Jarvis' visible state changed.  Drives the eye.
    STATE = "state"
    #: A chunk of assistant text, streamed as it is generated.
    TOKEN = "token"
    #: A complete assistant message.
    MESSAGE = "message"
    #: The user said something (typed or transcribed).
    USER_MESSAGE = "user_message"
    #: Live transcription, before it is final.
    TRANSCRIPT = "transcript"
    #: A tool started or finished.
    TOOL = "tool"
    #: Project/mission progress.
    PROGRESS = "progress"
    #: Something worth telling the user about unprompted.
    NOTIFICATION = "notification"

    #: A long-running WorkItem changed: created, phase advanced, progressed,
    #: completed, failed or was cancelled.  The Work Center renders these.
    JOB = "job"
    #: Knowledge graph changed.
    KNOWLEDGE = "knowledge"
    #: Health, model tiers, expert quota, resources.
    DIAGNOSTIC = "diagnostic"
    #: Speech synthesis produced audio, or playback state changed.
    SPEECH = "speech"
    #: Something went wrong that the user should see.
    ERROR = "error"


@dataclass
class Event:
    """One thing that happened."""

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    #: Monotonic per-bus counter.  Clients use this to resume and to deduplicate.
    seq: int = 0
    at: str = ""
    #: Optional conversation/project this belongs to, so a client can filter.
    scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "seq": self.seq,
            "at": self.at,
            "scope": self.scope,
        }


class Subscription:
    """One client's view of the stream.

    Bounded on purpose.  When a subscriber falls behind, the oldest events are
    dropped and :attr:`dropped` counts them, so a client can notice it missed
    something and resynchronise rather than silently rendering a stale world.
    """

    def __init__(self, bus: "EventBus", *, maxsize: int = 1000) -> None:
        self._bus = bus
        self._queue: queue.Queue[Event] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False

    def offer(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:  # pragma: no cover - racing with a drain
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:  # pragma: no cover
                self.dropped += 1

    def get(self, timeout: float | None = None) -> Event | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[Event]:
        found: list[Event] = []
        while True:
            try:
                found.append(self._queue.get_nowait())
            except queue.Empty:
                return found

    def listen(self, *, poll: float = 0.5) -> Iterator[Event]:
        """Yield events until the subscription is closed."""

        while not self.closed:
            event = self.get(timeout=poll)
            if event is not None:
                yield event

    def close(self) -> None:
        self.closed = True
        self._bus.unsubscribe(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class EventBus:
    """Fan-out to any number of clients, none of which can slow the core."""

    def __init__(self, *, replay: int = 100) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Subscription] = []
        self._replay_size = replay
        self._replay: list[Event] = []
        self._seq = 0
        self._watchers: list[Callable[[Event], None]] = []

    # -- publishing ------------------------------------------------------

    def publish(
        self,
        type: EventType,
        payload: dict[str, Any] | None = None,
        *,
        scope: str = "",
    ) -> Event:
        """Record an event and hand it to every subscriber.  Never blocks."""

        with self._lock:
            self._seq += 1
            event = Event(
                type=type,
                payload=dict(payload or {}),
                seq=self._seq,
                at=datetime.now(timezone.utc).isoformat(),
                scope=scope,
            )
            self._replay.append(event)
            del self._replay[: max(0, len(self._replay) - self._replay_size)]
            subscribers = list(self._subscribers)
            watchers = list(self._watchers)

        for subscriber in subscribers:
            subscriber.offer(event)
        for watcher in watchers:
            try:
                watcher(event)
            except Exception:
                # A broken watcher must not break the thing being watched.
                continue
        return event

    # -- subscribing -----------------------------------------------------

    def subscribe(self, *, replay: bool = True, maxsize: int = 1000) -> Subscription:
        """Attach a new client, optionally catching it up on recent history."""

        subscription = Subscription(self, maxsize=maxsize)
        with self._lock:
            self._subscribers.append(subscription)
            history = list(self._replay) if replay else []
        for event in history:
            subscription.offer(event)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscribers:
                self._subscribers.remove(subscription)

    def watch(self, callback: Callable[[Event], None]) -> None:
        """Register an in-process listener (logging, persistence, the CLI)."""

        with self._lock:
            self._watchers.append(callback)

    # -- introspection ---------------------------------------------------

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._seq

    def history(self, *, since: int = 0, limit: int = 100) -> list[Event]:
        with self._lock:
            return [event for event in self._replay if event.seq > since][-limit:]
