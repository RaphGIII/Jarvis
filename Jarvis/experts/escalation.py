"""Deciding when the local model has had its chance.

The brief is specific about this and it is the right instinct: *do not rely only
on the LLM saying "this is hard"*.  A model's estimate of its own difficulty is
the least reliable signal available -- it is produced by the same weights that
are about to fail, and small models are systematically overconfident on exactly
the tasks they cannot do.

So escalation here is decided from things that can be counted:

*Failures that actually happened.*  Not "this looks hard" but "three candidates
were generated, applied, and rejected by the tests".

*Failures that stopped being informative.*  Three different diagnoses mean the
loop is still learning; three identical ones mean it is stuck, and a fourth
attempt buys nothing.  That distinction is worth more than the raw count.

*Measured history for this class of task.*  If local self-patching has
succeeded once in five attempts, the sixth should not be discovered the hard
way.  :class:`PerformanceLedger` is what makes "historical evidence shows the
local model performs poorly on this task class" a lookup rather than a hunch.

The bias is deliberately towards trying locally first.  Escalation costs
subscription quota that cannot be bought back when it runs out, and a system
that reaches for the expert whenever a task looks awkward is one that stops
working the moment the quota does.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Attempt:
    """One recorded try at a task, by whichever tier tried it."""

    task_class: str
    tier: str
    succeeded: bool
    seconds: float = 0.0
    failure_kind: str = ""
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceLedger:
    """What has actually worked, per task class and tier.

    Append-only JSONL.  The point is not analytics -- it is that "the local
    model is bad at this" becomes a measured claim rather than an impression,
    and one that improves on its own as the system is used.
    """

    def __init__(self, path: str | Path, *, keep: int = 2000) -> None:
        self.path = Path(path)
        self.keep = keep
        self._lock = threading.Lock()
        self._attempts: list[Attempt] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines[-self.keep :]:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            try:
                self._attempts.append(Attempt(**data))
            except TypeError:
                continue

    def record(self, attempt: Attempt) -> Attempt:
        if not attempt.at:
            attempt.at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._attempts.append(attempt)
            del self._attempts[: max(0, len(self._attempts) - self.keep)]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(attempt.to_dict()) + "\n")
            except OSError:
                # History is an optimisation, never a correctness requirement.
                pass
        return attempt

    def attempts(self, *, task_class: str = "", tier: str = "") -> list[Attempt]:
        with self._lock:
            return [
                item
                for item in self._attempts
                if (not task_class or item.task_class == task_class)
                and (not tier or item.tier == tier)
            ]

    def success_rate(self, task_class: str, tier: str) -> tuple[float | None, int]:
        """Pass rate and sample size.  ``None`` when there is nothing to go on.

        Returning None rather than 0.0 for "never tried" matters: a task class
        with no history is unknown, not known-hopeless, and treating the two
        alike would escalate every new kind of work on its first attempt.
        """

        found = self.attempts(task_class=task_class, tier=tier)
        if not found:
            return None, 0
        passed = sum(1 for item in found if item.succeeded)
        return passed / len(found), len(found)

    def summary(self) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        for attempt in self.attempts():
            key = f"{attempt.task_class}/{attempt.tier}"
            row = rows.setdefault(key, {"attempts": 0, "passed": 0})
            row["attempts"] += 1
            row["passed"] += 1 if attempt.succeeded else 0
        for row in rows.values():
            row["rate"] = round(row["passed"] / row["attempts"], 3)
        return rows


@dataclass
class EscalationSignals:
    """Everything the decision is allowed to look at.

    Deliberately all counts and measurements.  There is no "difficulty" field
    for a model to fill in.
    """

    task_class: str = "general"
    #: Candidates that were produced, applied, and failed verification.
    local_failures: int = 0
    #: How many *distinct* diagnoses those failures produced.
    distinct_diagnoses: int = 0
    #: Tasks abandoned after exhausting their attempts.
    abandoned_tasks: int = 0
    #: Files the goal would have to touch, where that is known.
    files_in_scope: int = 0
    #: The user asked for the expert in so many words.
    user_requested: bool = False
    #: Seconds already spent locally.
    seconds_spent: float = 0.0
    #: Budget for the whole goal, if one was set.
    seconds_budget: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    #: What the decision was based on, so it can be explained and audited.
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"escalate": self.escalate, "reason": self.reason, "evidence": self.evidence}


class EscalationController:
    """Turns signals into a yes or no, and says why."""

    #: Local attempts before escalation is even considered.  The loop is
    #: designed to fail and repair, so one failure is normal operation.
    MIN_LOCAL_FAILURES = 3
    #: Repeating the same diagnosis is worth more than the raw failure count:
    #: it means the loop has stopped learning from its own evidence.
    STUCK_REPEATS = 2
    #: Below this measured local pass rate, stop paying for the lesson again.
    POOR_RATE = 0.34
    #: ...but only once there is enough history to mean anything.
    MIN_SAMPLES = 4
    #: A change touching this many files is beyond what a 7B model plans well.
    WIDE_CHANGE_FILES = 8

    def __init__(self, ledger: PerformanceLedger | None = None, *, enabled: bool = True) -> None:
        self.ledger = ledger
        self.enabled = enabled

    def decide(self, signals: EscalationSignals) -> EscalationDecision:
        evidence: list[str] = []

        if not self.enabled:
            return EscalationDecision(False, "escalation is disabled by configuration")

        # An explicit request settles it. The user may know something the
        # counters do not, and overriding them is their prerogative.
        if signals.user_requested:
            return EscalationDecision(True, "the user asked for the expert", ["user request"])

        rate, samples = (None, 0)
        if self.ledger is not None:
            rate, samples = self.ledger.success_rate(signals.task_class, "build_local")

        # History first: if this class of work is measurably beyond the local
        # model, there is no reason to re-derive that at the user's expense.
        if rate is not None and samples >= self.MIN_SAMPLES and rate < self.POOR_RATE:
            evidence.append(f"local pass rate {rate:.0%} over {samples} attempts at {signals.task_class!r}")
            if signals.local_failures >= 1:
                return EscalationDecision(
                    True,
                    f"this class of task has a {rate:.0%} local success rate and has already failed once here",
                    evidence,
                )

        if signals.local_failures >= self.MIN_LOCAL_FAILURES:
            evidence.append(f"{signals.local_failures} local candidates failed verification")
            # Distinct diagnoses mean the loop is still learning something.
            repeats = signals.local_failures - max(1, signals.distinct_diagnoses)
            if signals.distinct_diagnoses and repeats >= self.STUCK_REPEATS:
                evidence.append(
                    f"only {signals.distinct_diagnoses} distinct diagnoses across those failures"
                )
                return EscalationDecision(
                    True, "the local loop is repeating the same diagnosis rather than learning", evidence
                )
            if signals.abandoned_tasks >= 1:
                evidence.append(f"{signals.abandoned_tasks} task(s) abandoned after exhausting attempts")
                return EscalationDecision(True, "local attempts are exhausted", evidence)

        if signals.files_in_scope >= self.WIDE_CHANGE_FILES:
            evidence.append(f"{signals.files_in_scope} files in scope")
            if signals.local_failures >= 1:
                return EscalationDecision(
                    True, "a wide change that has already failed locally", evidence
                )

        if signals.seconds_budget and signals.seconds_spent >= signals.seconds_budget * 0.6:
            evidence.append(
                f"{signals.seconds_spent:.0f}s of a {signals.seconds_budget:.0f}s budget spent"
            )
            if signals.local_failures >= 2:
                return EscalationDecision(
                    True, "most of the time budget is gone and local attempts are failing", evidence
                )

        return EscalationDecision(
            False,
            "local attempts have not been exhausted" if signals.local_failures else "nothing has failed yet",
            evidence,
        )

    # -- learning from the outcome ---------------------------------------

    def record(self, task_class: str, tier: str, succeeded: bool, *, seconds: float = 0.0, failure_kind: str = "") -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            Attempt(
                task_class=task_class,
                tier=tier,
                succeeded=succeeded,
                seconds=seconds,
                failure_kind=failure_kind,
            )
        )


def classify_goal(goal: str) -> str:
    """A coarse task class, so history has something stable to group by.

    Coarse on purpose.  Classes that are too specific never accumulate enough
    samples to support a decision, which would leave the ledger permanently
    unable to say anything -- the failure mode of a metric that is technically
    correct and practically useless.
    """

    text = (goal or "").lower()
    pairs = (
        ("self_development", ("this repository", "your own code", "jarvis itself", "self-patch", "eigenen code")),
        ("capability", ("capability", "learn to", "acquire", "fähigkeit", "lerne")),
        ("ui", ("ui", "interface", "eye animation", "oberfläche", "frontend")),
        ("vision", ("image", "screenshot", "vision", "ocr", "bild")),
        ("research", ("research", "find out", "investigate", "recherche")),
        ("build", ("build", "create", "implement", "write a", "baue", "erstelle")),
        ("debug", ("fix", "debug", "failing", "error", "repariere", "fehler")),
    )
    import re as _re

    for name, markers in pairs:
        for marker in markers:
            # Word boundaries, not substrings: "ui" is inside "build", so
            # "Build a CLI tool" classified as a UI change until this was fixed.
            # Short markers are exactly the ones that need the boundary.
            pattern = rf"(?<!\w){_re.escape(marker)}(?!\w)" if len(marker) <= 4 else _re.escape(marker)
            if _re.search(pattern, text):
                return name
    return "general"
