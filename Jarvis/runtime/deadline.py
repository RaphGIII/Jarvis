"""Time budgets that are actually enforced, and calls that cannot outlive them.

An autonomous system is allowed to work for a long time.  What it is never
allowed to do is work for an *unbounded* time with no way to tell the difference
from being stuck -- which is precisely the state this module exists to make
impossible.

Two problems, two pieces.

**A budget checked only between steps is not a budget.**  The project loop
compared elapsed time against ``max_seconds`` at the top of each iteration, so a
single step that blocked forever never reached the check again.  :class:`Deadline`
is passed *into* the work rather than wrapped around it, so every model call and
every phase can ask how much time is left and refuse to start work it cannot
finish.

**A socket timeout is not a call timeout.**  ``urlopen(timeout=...)`` bounds each
individual read, not the request as a whole; a server that trickles bytes, or one
that keeps a connection open while a 7B model grinds, can hold a caller far past
any per-read limit.  :func:`call_with_timeout` bounds the call itself.

The abandoned-thread trade-off is deliberate.  Python cannot safely kill a thread
blocked in a socket read, so a timed-out call leaves a daemon thread to finish
and discard its result.  That leaks one socket and one thread per timeout, which
is acceptable because timeouts are rare and the process is bounded; the
alternative -- waiting for the call anyway -- is the bug being fixed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")

#: What a budget of "no limit" looks like, so callers can do arithmetic on it.
FOREVER = float("inf")


class DeadlineExceeded(TimeoutError):
    """The mission's time budget ran out."""

    def __init__(self, name: str, budget: float, elapsed: float) -> None:
        super().__init__(f"{name or 'operation'} exceeded its {budget:.0f}s budget after {elapsed:.0f}s")
        self.name = name
        self.budget = budget
        self.elapsed = elapsed


class CallTimeout(TimeoutError):
    """One call did not return within the time allowed for it."""

    def __init__(self, what: str, timeout: float) -> None:
        super().__init__(f"{what} did not return within {timeout:.0f}s")
        self.what = what
        self.timeout = timeout


@dataclass
class Deadline:
    """How much wall-clock time a piece of work may still consume."""

    budget: float = FOREVER
    name: str = ""
    started: float = 0.0

    def __post_init__(self) -> None:
        if not self.started:
            self.started = time.monotonic()
        if self.budget is None:
            self.budget = FOREVER

    @classmethod
    def of(cls, seconds: float | None, name: str = "") -> "Deadline":
        return cls(budget=FOREVER if seconds is None else float(seconds), name=name)

    @classmethod
    def none(cls, name: str = "") -> "Deadline":
        return cls(budget=FOREVER, name=name)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        if self.budget == FOREVER:
            return FOREVER
        return max(0.0, self.budget - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def require(self) -> None:
        """Raise if the budget is gone.  Call before starting expensive work."""

        if self.expired:
            raise DeadlineExceeded(self.name, self.budget, self.elapsed)

    def clamp(self, seconds: float) -> float:
        """Shorten a per-call timeout so it cannot outlive the mission.

        A 900-second model timeout is reasonable on its own and absurd when the
        mission has 40 seconds left; clamping is what stops one call from
        blowing a budget that everything else respected.
        """

        if self.budget == FOREVER:
            return float(seconds)
        return max(1.0, min(float(seconds), self.remaining))

    def child(self, seconds: float | None, name: str = "") -> "Deadline":
        """A sub-budget that can be shorter than this one but never longer."""

        if seconds is None:
            return Deadline(budget=self.remaining, name=name or self.name)
        return Deadline(budget=min(float(seconds), self.remaining), name=name or self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "budget_seconds": None if self.budget == FOREVER else round(self.budget, 1),
            "elapsed_seconds": round(self.elapsed, 1),
            "remaining_seconds": None if self.remaining == FOREVER else round(self.remaining, 1),
            "expired": self.expired,
        }


def call_with_timeout(
    function: Callable[[], T],
    timeout: float,
    *,
    what: str = "call",
    on_abandon: Callable[[], None] | None = None,
) -> T:
    """Run ``function``, raising :class:`CallTimeout` if it takes too long.

    Needed because the timeouts the HTTP layer offers are per-read, not
    per-request.  With ``stream: false`` a local model sends nothing at all
    until it is finished, so a slow generation and a wedged server look
    identical from the socket's point of view -- and only one of them should be
    waited out.

    ``timeout`` of ``inf`` runs the call inline, so an unbudgeted caller pays no
    thread and behaves exactly as before.
    """

    if timeout == FOREVER:
        return function()

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = function()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=worker, name=f"jarvis-{what}", daemon=True)
    thread.start()
    thread.join(max(0.0, float(timeout)))

    if thread.is_alive():
        # The thread is blocked in a read we cannot interrupt.  Let it finish
        # into a box nobody reads; the process is bounded, so this is contained.
        if on_abandon is not None:
            on_abandon()
        raise CallTimeout(what, timeout)

    if "error" in box:
        raise box["error"]
    return box.get("value")  # type: ignore[return-value]
