"""The durable record of what this system actually did, in order.

The Activity button in the running product did nothing.  It flipped a boolean
that decided whether *future* tool events would be echoed into the chat log as
grey notes -- so a user who pressed it saw no panel, no history, and no
indication that anything had happened at all.  Everything the system had done
before the press was invisible, and everything after it was a one-line note in
a transcript.

What makes this worth a module rather than a UI fix is where the entries come
from.  The one thing Activity must never be is a story: a list assembled after
the fact by asking a model what it thinks it did.  So this log has exactly one
input -- :meth:`service.events.EventBus.watch`, the synchronous in-process
listener the bus already provides for persistence -- and no other writer.  An
entry exists here only because an event was really published on the bus by the
code that really did the thing.  There is no code path by which prose becomes
an activity entry, because there is no function here that accepts prose.

Recorded synchronously through ``watch`` rather than by draining a
subscription, because a subscription is bounded and drops its oldest events
when a consumer falls behind.  Dropping a frame of a video is fine; dropping
the record of a file write is not.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: How much of the tail to read when answering a request for recent activity.
#: The file is append-only and unbounded; reading all of it to show the last
#: fifty entries would get slower every day the product is used.
TAIL_BYTES = 512 * 1024


@dataclass(frozen=True)
class ActivityEntry:
    """One thing that happened, as the bus reported it."""

    kind: str
    summary: str
    at: str = ""
    seq: int = 0
    scope: str = ""
    #: The receipt this entry is evidence of, when there is one.
    receipt_id: str = ""
    #: Everything the UI needs to expand the row: target, result, checks.
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "at": self.at,
            "seq": self.seq,
            "scope": self.scope,
            "receipt_id": self.receipt_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActivityEntry":
        return cls(
            kind=str(data.get("kind", "")),
            summary=str(data.get("summary", "")),
            at=str(data.get("at", "")),
            seq=int(data.get("seq", 0) or 0),
            scope=str(data.get("scope", "")),
            receipt_id=str(data.get("receipt_id", "")),
            detail=dict(data.get("detail") or {}),
        )


#: States worth a line in a history.  ``idle`` and ``thinking`` are the resting
#: and default-busy states and occur on every single turn; recording them would
#: bury the transitions that mean something under noise.
NOTABLE_STATES = ("working", "verifying", "error", "researching", "coding", "waiting")

#: Event types that become activity.  ``token`` and ``speech`` are excluded:
#: they are per-chunk streaming frames, thousands per conversation, and the
#: completed message already records what they added up to.
RECORDED = ("user_message", "message", "tool", "progress", "error", "notification", "state", "diagnostic")


class ActivityLog:
    """Append-only activity, fed only by the event bus."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._attached: Any = None

    # -- recording -------------------------------------------------------

    def attach(self, bus: Any) -> "ActivityLog":
        """Start recording from ``bus``.  Idempotent per bus."""

        if self._attached is bus:
            return self
        self._attached = bus
        bus.watch(self._on_event)
        return self

    def _on_event(self, event: Any) -> None:
        entry = self.entry_for(event)
        if entry is not None:
            self.record(entry)

    @staticmethod
    def entry_for(event: Any) -> ActivityEntry | None:
        """Translate a bus event into an activity entry, or ``None`` to skip.

        A pure function of the event so it can be tested without a bus, a file
        or a running server -- and so that what does and does not become
        activity is one readable table rather than a scatter of call sites.
        """

        kind = getattr(event.type, "value", str(event.type))
        if kind not in RECORDED:
            return None
        payload = dict(event.payload or {})
        common = {"at": event.at, "seq": event.seq, "scope": event.scope}

        if kind == "state":
            state = str(payload.get("state", ""))
            if state not in NOTABLE_STATES:
                return None
            return ActivityEntry(
                kind=f"state.{state}",
                summary=str(payload.get("detail", "")) or state,
                detail={"state": state, "busy": payload.get("busy")},
                **common,
            )

        if kind == "user_message":
            meta = payload.get("meta") or {}
            return ActivityEntry(kind="request", summary=str(payload.get("text", ""))[:400], detail={"meta": meta} if meta else {}, **common)

        if kind == "diagnostic":
            # The wake event and the voice trace are activity; the rest stays diagnostics.
            if payload.get("wake") and "score" in payload:
                return ActivityEntry(kind="wake", summary=f"{payload['wake']} · score {float(payload['score']):.3f}",
                                     detail={"session": payload.get("session", ""), "command": payload.get("command", ""),
                                             "utterance_id": payload.get("utterance_id", "")}, **common)
            if payload.get("voice_trace"):
                verdict = payload.get("verdict") or {}
                accepted = bool(verdict.get("accepted"))
                utterance = payload.get("utterance") or {}
                text = str(utterance.get("normalized_transcript") or utterance.get("raw_transcript") or payload.get("text") or "")
                summary = (f"accepted „{text[:120]}“" if accepted else f"rejected: {verdict.get('reason', '?')}" + (f" — „{text[:80]}“" if text else ""))
                return ActivityEntry(kind="voice.accepted" if accepted else "voice.rejected", summary=summary, detail=payload, **common)
            return None

        if kind == "message":
            return ActivityEntry(
                kind="answer",
                summary=str(payload.get("text", ""))[:400],
                # The backend is the useful part: it says whether this answer
                # came from a model, from a registry, or from an executor's
                # receipt -- which is the difference between an opinion and a
                # fact, and the UI colours it accordingly.
                detail={"backend": payload.get("backend", "")},
                **common,
            )

        if kind == "tool":
            receipt = payload.get("receipt")
            if isinstance(receipt, dict):
                verdict = (
                    "verified" if receipt.get("verified")
                    else "ran" if receipt.get("ok")
                    else "failed"
                )
                return ActivityEntry(
                    kind=f"action.{verdict}",
                    summary=str(receipt.get("detail", ""))[:400],
                    receipt_id=str(receipt.get("id", "")),
                    detail=receipt,
                    **common,
                )
            return ActivityEntry(
                kind="tool",
                summary=str(payload.get("summary", ""))[:400],
                detail=payload,
                **common,
            )

        if kind == "error":
            return ActivityEntry(
                kind="error", summary=str(payload.get("error", ""))[:400], detail=payload, **common
            )

        if kind == "progress":
            return ActivityEntry(
                kind="progress", summary=str(payload.get("summary", ""))[:400], detail=payload, **common
            )

        return ActivityEntry(
            kind="notification", summary=str(payload.get("text", ""))[:400], detail=payload, **common
        )

    def record(self, entry: ActivityEntry) -> ActivityEntry:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), default=str) + "\n")
        return entry

    # -- reading ---------------------------------------------------------

    def recent(self, limit: int = 100) -> list[ActivityEntry]:
        """The last ``limit`` entries, oldest first."""

        entries: list[ActivityEntry] = []
        for line in self._tail_lines():
            try:
                entries.append(ActivityEntry.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                # One unreadable line must not hide the rest of the history.
                continue
        return entries[-limit:]

    def _tail_lines(self) -> Iterable[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            if size > TAIL_BYTES:
                handle.seek(size - TAIL_BYTES, os.SEEK_SET)
                handle.readline()  # discard the partial line the seek landed in
            raw = handle.read()
        return [line for line in raw.decode("utf-8", "replace").splitlines() if line.strip()]

    def count(self) -> int:
        return len(self._tail_lines())
