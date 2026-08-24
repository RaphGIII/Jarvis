"""The normalized job and result every expert provider speaks.

The point of a contract here is substitutability.  Jarvis should be able to swap
a subscription-backed coding agent for a different one -- or for a future
self-hosted 70B -- by configuration, without the project engine learning
anything about either.  That only holds if the job carries everything a provider
needs and the result carries everything Jarvis needs to *verify*, rather than to
believe.

Hence the asymmetry in what these two objects hold.  :class:`ExpertJob` is
mostly instructions.  :class:`ExpertResult` is mostly evidence: what changed on
disk, what commands ran, what the tests said.  A provider that merely reports
"done" gives Jarvis nothing to check, so the field that would carry that claim
(:attr:`ExpertResult.summary`) is explicitly not what acceptance is decided on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExpertStatus(str, Enum):
    """How an expert attempt ended."""

    #: The provider ran and believes it finished.  Not yet verified.
    COMPLETED = "completed"
    #: The provider ran and reported failure.
    FAILED = "failed"
    #: Subscription quota is spent.  A state, not an error: see
    #: :data:`runtime.cost_policy.EXPERT_UNAVAILABLE`.
    UNAVAILABLE = "unavailable"
    #: The cost policy forbade the channel this provider would use.
    REFUSED = "refused"
    #: The provider exceeded its wall-clock budget.
    TIMEOUT = "timeout"
    #: Something outside the provider's control stopped it (missing credential,
    #: unreachable service, a decision only the user can make).
    BLOCKED = "blocked"
    #: No provider is configured or installed.
    NOT_CONFIGURED = "not_configured"

    @property
    def ran(self) -> bool:
        """True when the provider actually did work worth verifying."""

        return self in {ExpertStatus.COMPLETED, ExpertStatus.FAILED}

    @property
    def retryable_locally(self) -> bool:
        """True when Jarvis should carry on by itself rather than give up."""

        return self in {
            ExpertStatus.UNAVAILABLE,
            ExpertStatus.REFUSED,
            ExpertStatus.NOT_CONFIGURED,
            ExpertStatus.TIMEOUT,
        }


@dataclass
class QuotaState:
    """What the provider says about its own remaining allowance.

    Every field is optional because most providers say nothing useful, and
    inventing a number would be worse than admitting ignorance.

    One caveat matters enough to state here rather than only on the field:
    ``notional_cost_usd`` is what the work WOULD have cost at API rates, and the
    Claude Code CLI reports it even on subscription runs. On a subscription it
    is covered by the flat fee, so it is a usage signal and NOT a charge. It
    must never be shown to the user as money owed.
    """

    exhausted: bool = False
    #: Free-text, straight from the provider.
    detail: str = ""
    #: ISO-8601, when the provider names a reset time.
    resets_at: str = ""
    #: Notional value of the work, when reported.  On a subscription this is
    #: covered by the flat fee -- it is a usage signal, NOT a charge, and must
    #: never be presented to the user as money owed.
    notional_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertJob:
    """A unit of work handed to an expert.

    Carries its own acceptance criteria on purpose.  An expert that is told what
    "done" means can check itself before returning, and -- more importantly --
    Jarvis can check the same thing afterwards without a second source of truth.
    """

    goal: str
    #: The directory the expert may work in.  Normally an isolated worktree.
    workspace: Path
    #: Hard rules: protected paths, style, "do not add dependencies".
    constraints: list[str] = field(default_factory=list)
    #: Retrieved context -- source regions, findings, prior art.
    context: str = ""
    #: (human-readable criterion, runnable command).  The bar for acceptance.
    acceptance: list[tuple[str, list[str]]] = field(default_factory=list)
    #: Commands that must pass; a subset of acceptance in most jobs.
    test_commands: list[list[str]] = field(default_factory=list)
    #: What the expert is permitted to do, in Jarvis' permission vocabulary.
    permissions: list[str] = field(default_factory=list)
    #: What local attempts already tried and how they failed.  The single most
    #: useful thing to send: it stops the expert re-deriving a known dead end.
    previous_failures: list[str] = field(default_factory=list)
    #: Files the job is expected to produce or change.
    expected_artifacts: list[str] = field(default_factory=list)
    #: Paths the expert may write to.  Empty means "anywhere in the workspace".
    allowed_paths: list[str] = field(default_factory=list)
    max_seconds: float = 1800.0
    #: Free-form provider hints; never required for correctness.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        return data

    def brief(self) -> str:
        """The job as prose, for a provider whose interface is a prompt."""

        parts = [f"GOAL:\n{self.goal}\n"]
        if self.constraints:
            parts.append("CONSTRAINTS (these are hard rules):\n" + "\n".join(f"- {item}" for item in self.constraints) + "\n")
        if self.allowed_paths:
            parts.append("You may modify ONLY these paths:\n" + "\n".join(f"- {item}" for item in self.allowed_paths) + "\n")
        if self.previous_failures:
            parts.append(
                "ALREADY TRIED AND FAILED -- do not repeat these:\n"
                + "\n".join(f"- {item}" for item in self.previous_failures)
                + "\n"
            )
        if self.context:
            parts.append(f"CONTEXT:\n{self.context}\n")
        if self.acceptance:
            rows = "\n".join(f"- {text}\n    command: {' '.join(command)}" for text, command in self.acceptance)
            parts.append(
                "ACCEPTANCE CRITERIA -- the work is done when every one of these commands exits zero.\n"
                "Jarvis re-runs them independently after you finish and decides from their exit codes, "
                "so a claim that they pass is not enough:\n"
                f"{rows}\n"
            )
        if self.expected_artifacts:
            parts.append("EXPECTED ARTIFACTS:\n" + "\n".join(f"- {item}" for item in self.expected_artifacts) + "\n")
        return "\n".join(parts)


@dataclass
class ExpertResult:
    """What came back, expressed as evidence rather than assertion."""

    status: ExpertStatus
    provider: str = ""
    #: The provider's own account of what it did.  Informational only: never
    #: the basis for accepting the work.
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    #: (criterion, passed, output) for each acceptance check Jarvis re-ran.
    test_evidence: list[tuple[str, bool, str]] = field(default_factory=list)
    blocker: str = ""
    quota: QuotaState = field(default_factory=QuotaState)
    duration_seconds: float = 0.0
    #: Whatever the provider returned, kept for diagnostics.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        """True only when Jarvis re-ran the acceptance checks and they passed.

        Note what this does *not* consult: :attr:`status` or :attr:`summary`.
        A provider reporting success with no evidence is not verified, which is
        the entire reason this property exists rather than a status check.
        """

        return bool(self.test_evidence) and all(passed for _, passed, _ in self.test_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "provider": self.provider,
            "summary": self.summary[:4000],
            "files_changed": self.files_changed,
            "commits": self.commits,
            "commands_run": self.commands_run,
            "test_evidence": [
                {"criterion": text, "passed": passed, "output": output[-1500:]}
                for text, passed, output in self.test_evidence
            ],
            "blocker": self.blocker,
            "quota": self.quota.to_dict(),
            "duration_seconds": round(self.duration_seconds, 2),
            "verified": self.verified,
        }
