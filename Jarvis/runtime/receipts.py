"""Receipts: the only thing in this system allowed to say an action succeeded.

The defect this exists to make impossible was observed in the running product.
Asked to create a file with exact contents, verify it, and only then confirm,
the assistant replied:

    Datei "zeus_test.txt" wurde erstellt und mit dem Inhalt "ZEUS funktioniert"
    gespeichert.  Speicherort: /home/user/Projekte/Zeus_Testprojekt/zeus_test.txt
    Existenz geprueft: Datei existiert und enthaelt den erwarteten Inhalt.

No file existed.  No path of that shape exists on this machine -- it is a Linux
path invented on a Windows box.  Nothing had been executed at all: the request
went to a language model and the model's prose was shipped to the user as the
outcome.

The system prompt already said, in as many words, "Never claim an action was
performed unless it actually was."  It was ignored, and it was always going to
be: an instruction competes with everything else in the context, and a model
that has been asked to confirm will confirm.  The fix cannot be a better
instruction.  It has to be that *the model does not author the verdict*.

So: an action produces a :class:`Receipt`, written by the executor that actually
ran it.  The user-facing outcome is composed from the receipt.  The model may
propose what to do and may narrate around a receipt that exists; it can no more
manufacture a success than it can manufacture a file.

Two properties are load-bearing and are enforced here rather than left to the
caller:

*A receipt is not verified by having run.*  :attr:`Receipt.verified` requires at
least one verification that actually passed.  An action that executed cleanly
but checked nothing is ``ok`` and *not* ``verified``, and the difference is
visible to the user.  Vacuous truth is the specific way this kind of guard
usually fails -- a check with nothing to check reports green.

*Verification must not be performed by the thing being verified.*  A file write
is confirmed by re-reading the bytes off the filesystem, not by trusting the
writer's return value; a project is confirmed by reloading it from the store,
not by trusting the object still in memory.  Which is why
:class:`Verification` records what was *observed*, not merely a boolean.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Verification:
    """One independent check that an action really had the effect claimed.

    ``observed`` is the point of the class.  A bare pass/fail can be produced by
    a check that never looked at anything; recording what was actually seen
    means a receipt can be audited after the fact by someone who does not trust
    the code that wrote it.
    """

    check: str
    passed: bool
    #: What was actually found.  Kept short; this is evidence, not a log.
    observed: str = ""
    #: What would have counted as correct.
    expected: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "passed": self.passed,
            "observed": self.observed[:600],
            "expected": self.expected[:600],
        }

    def describe(self) -> str:
        mark = "ok" if self.passed else "FAILED"
        return f"{self.check}: {mark}" + (f" ({self.observed[:120]})" if self.observed else "")


@dataclass(frozen=True)
class Receipt:
    """Evidence that a side effect happened, produced by whatever performed it.

    Constructed only by an executor.  Nothing that merely *describes* an action
    -- a model, a prompt, a plan -- may build one, which is the whole point.
    """

    kind: str
    #: The subsystem that actually did the work, e.g. ``tools.write_file``.
    executor: str
    ok: bool
    #: The user's words that led here, for auditing.
    request: str = ""
    #: Why it failed, or what it did.  Written by the executor.
    detail: str = ""
    #: Concrete facts: absolute paths, ids, byte counts.
    evidence: dict[str, Any] = field(default_factory=dict)
    verifications: tuple[Verification, ...] = ()
    id: str = field(default_factory=lambda: f"rcpt_{uuid.uuid4().hex[:12]}")
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0

    @property
    def verified(self) -> bool:
        """Whether this is allowed to be reported to the user as a success.

        Three conditions, and the middle one is the one that matters: an action
        with no verifications is not verified, however cleanly it ran.  A guard
        that passes when there is nothing to check is the failure mode this
        system has already been bitten by, so "checked nothing" is a distinct
        outcome from "checked and passed" rather than collapsing into it.
        """

        return self.ok and bool(self.verifications) and all(v.passed for v in self.verifications)

    @property
    def failures(self) -> tuple[Verification, ...]:
        return tuple(v for v in self.verifications if not v.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "executor": self.executor,
            "ok": self.ok,
            "verified": self.verified,
            "request": self.request[:400],
            "detail": self.detail[:1000],
            "evidence": _plain(self.evidence),
            "verifications": [v.to_dict() for v in self.verifications],
            "at": self.at,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        return cls(
            kind=str(data.get("kind", "")),
            executor=str(data.get("executor", "")),
            ok=bool(data.get("ok", False)),
            request=str(data.get("request", "")),
            detail=str(data.get("detail", "")),
            evidence=dict(data.get("evidence") or {}),
            verifications=tuple(
                Verification(
                    check=str(item.get("check", "")),
                    passed=bool(item.get("passed", False)),
                    observed=str(item.get("observed", "")),
                    expected=str(item.get("expected", "")),
                )
                for item in (data.get("verifications") or [])
                if isinstance(item, dict)
            ),
            id=str(data.get("id") or f"rcpt_{uuid.uuid4().hex[:12]}"),
            at=str(data.get("at", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
        )

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        """One line for the activity feed."""

        verdict = "verified" if self.verified else ("ran, unverified" if self.ok else "FAILED")
        return f"{self.kind} [{verdict}] {self.detail[:120]}"

    def evidence_lines(self) -> list[str]:
        """The checks, as the user should see them."""

        return [v.describe() for v in self.verifications]


def failed(kind: str, executor: str, detail: str, *, request: str = "", **evidence: Any) -> Receipt:
    """A receipt for something that did not happen.

    Present as a named constructor because an honest failure has to be as easy
    to produce as a success; the moment reporting a failure is more work than
    reporting a success, code starts reporting successes.
    """

    return Receipt(
        kind=kind, executor=executor, ok=False, detail=detail, request=request, evidence=dict(evidence)
    )


class ReceiptLedger:
    """Every receipt, appended to disk, newest last.

    Durable rather than in-memory because the user's question is not only "did
    that work" but "what has this thing actually done to my machine", and that
    question outlives a process.  Append-only JSONL for the same reason the
    audit log is: a record that can be rewritten is not evidence.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, receipt: Receipt) -> Receipt:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt.to_dict(), default=str) + "\n")
        return receipt

    def all(self) -> list[Receipt]:
        if not self.path.exists():
            return []
        receipts: list[Receipt] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    receipts.append(Receipt.from_dict(json.loads(line)))
                except (ValueError, TypeError):
                    # One unreadable line must not hide the rest of the history.
                    continue
        return receipts

    def recent(self, limit: int = 50) -> list[Receipt]:
        return self.all()[-limit:]

    def get(self, receipt_id: str) -> Receipt | None:
        for receipt in reversed(self.all()):
            if receipt.id == receipt_id:
                return receipt
        return None


def identifying_tokens(receipt: Receipt) -> set[str]:
    """The concrete things a receipt is *about*: a filename, a project title.

    Used to decide whether a sentence claiming a success is talking about
    something that actually happened.  Only names appear here, never verbs or
    outcomes -- the receipt decides whether it succeeded, and this decides only
    what it succeeded at.
    """

    tokens: set[str] = set()
    for key in ("relative_path", "title", "project_id", "goal"):
        value = receipt.evidence.get(key)
        if isinstance(value, str) and value.strip():
            tokens.add(value.strip().lower())
    path = receipt.evidence.get("path")
    if isinstance(path, str) and path.strip():
        tokens.add(Path(path).name.lower())
    # Short tokens match by accident. A four-character floor keeps "a.txt" and
    # drops anything that would collide with ordinary German or English.
    return {token for token in tokens if len(token) >= 4}


def supporting(receipts: Iterable[Receipt], text: str) -> Receipt | None:
    """A verified receipt that the claim in ``text`` is plausibly about.

    The guard's first version asked "was anything executed *this turn*", which
    is the wrong question one turn later.  Having genuinely created
    ``zeus_test.txt``, the assistant must be able to say so when asked -- and
    blocking that would teach the user that the honesty machinery is noise, at
    which point it protects nobody.

    So the question is whether *this* claim has evidence, not whether the
    conversation has any.  A claim that names nothing concrete ("Done.") can
    match nothing and stays blocked, which is the safe direction.
    """

    lowered = (text or "").lower()
    for receipt in reversed(list(receipts)):
        if not receipt.verified:
            continue
        if any(token in lowered for token in identifying_tokens(receipt)):
            return receipt
    return None


def _plain(value: Any) -> Any:
    """JSON-safe evidence.  Paths in particular must survive as strings."""

    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def all_passed(verifications: Iterable[Verification]) -> bool:
    items = list(verifications)
    return bool(items) and all(item.passed for item in items)
