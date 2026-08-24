"""The single door through which stronger intelligence is recruited.

Everything an expert does passes through :meth:`ExpertGateway.submit`, and that
method does three things no provider is trusted to do for itself:

*It asks the cost policy first.*  Before a provider is even consulted, the
channel it bills through must be permitted.  A provider cannot opt itself in.

*It re-runs the acceptance checks.*  A provider's own report of success is
recorded and ignored.  Jarvis executes the criteria in the workspace afterwards
and decides from the exit codes, so an expert that claims a green suite it never
ran is caught by the same mechanism that catches a local model doing it.

*It refuses to convert exhaustion into spending.*  When a subscription runs out,
the gateway returns :attr:`~experts.contracts.ExpertStatus.UNAVAILABLE` and
consults :meth:`~runtime.cost_policy.CostPolicy.fallbacks_for`, which by
construction never names a metered channel.  There is deliberately no code path
from "quota exhausted" to "use the API key".
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from experts.contracts import ExpertJob, ExpertResult, ExpertStatus, QuotaState
from runtime.cost_policy import CostLedger, CostPolicy, CostPolicyViolation, SpendChannel


@dataclass
class ProviderAvailability:
    """Whether a provider could be used right now, and why not if it cannot."""

    available: bool
    detail: str = ""
    #: True when the provider exists but its quota is spent.
    quota_exhausted: bool = False
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "detail": self.detail,
            "quota_exhausted": self.quota_exhausted,
            "version": self.version,
        }


@runtime_checkable
class ExpertProvider(Protocol):
    """What Jarvis needs from any expert, subscription CLI or future 70B."""

    #: Stable identifier used in configuration and diagnostics.
    name: str
    #: How work through this provider is paid for.
    channel: SpendChannel

    def availability(self) -> ProviderAvailability:
        ...

    def execute(self, job: ExpertJob) -> ExpertResult:
        ...


class ExpertGateway:
    """Policy, provider selection, and independent verification."""

    def __init__(
        self,
        providers: list[ExpertProvider] | None = None,
        *,
        policy: CostPolicy | None = None,
        ledger: CostLedger | None = None,
    ) -> None:
        self.policy = policy or CostPolicy.load()
        self.ledger = ledger or CostLedger(self.policy)
        self.providers: list[ExpertProvider] = list(providers or [])

    # -- discovery -------------------------------------------------------

    def register(self, provider: ExpertProvider) -> None:
        self.providers.append(provider)

    def usable_providers(self) -> list[tuple[ExpertProvider, ProviderAvailability]]:
        """Providers whose channel is permitted AND which are actually there."""

        found: list[tuple[ExpertProvider, ProviderAvailability]] = []
        for provider in self.providers:
            if not self.policy.permits(provider.channel):
                continue
            found.append((provider, provider.availability()))
        return found

    def status(self) -> dict[str, Any]:
        """What the UI needs to show LOCAL / EXPERT AVAILABLE / QUOTA EXHAUSTED."""

        rows = []
        for provider in self.providers:
            permitted = self.policy.permits(provider.channel)
            availability = provider.availability() if permitted else ProviderAvailability(
                False, self.policy.explain(provider.channel)
            )
            rows.append(
                {
                    "name": provider.name,
                    "channel": provider.channel.value,
                    "permitted": permitted,
                    **availability.to_dict(),
                }
            )
        return {
            "policy": self.policy.to_dict(),
            "providers": rows,
            "expert_available": any(row["permitted"] and row["available"] for row in rows),
            "quota_exhausted": bool(rows) and all(
                row["quota_exhausted"] for row in rows if row["permitted"]
            ) and any(row["permitted"] for row in rows),
        }

    # -- the one entry point ---------------------------------------------

    def submit(self, job: ExpertJob, *, provider_name: str = "") -> ExpertResult:
        """Run a job through the first usable provider, then verify it."""

        candidates = [
            provider
            for provider in self.providers
            if not provider_name or provider.name == provider_name
        ]
        if not candidates:
            return ExpertResult(
                status=ExpertStatus.NOT_CONFIGURED,
                blocker=f"no expert provider named {provider_name!r} is registered"
                if provider_name
                else "no expert provider is registered",
            )

        last: ExpertResult | None = None
        for provider in candidates:
            # Policy first, always. A provider is never asked whether it may run.
            try:
                self.ledger.require(provider.channel, reason=f"expert job: {job.goal[:80]}")
            except CostPolicyViolation as exc:
                last = ExpertResult(
                    status=ExpertStatus.REFUSED,
                    provider=provider.name,
                    blocker=str(exc),
                )
                continue

            availability = provider.availability()
            if not availability.available:
                last = ExpertResult(
                    status=ExpertStatus.UNAVAILABLE if availability.quota_exhausted else ExpertStatus.NOT_CONFIGURED,
                    provider=provider.name,
                    blocker=availability.detail,
                    quota=QuotaState(exhausted=availability.quota_exhausted, detail=availability.detail),
                )
                continue

            started = time.perf_counter()
            result = provider.execute(job)
            result.provider = result.provider or provider.name
            result.duration_seconds = result.duration_seconds or (time.perf_counter() - started)

            if result.status.ran:
                result.test_evidence = self.verify(job)
                if result.status is ExpertStatus.COMPLETED and not result.verified:
                    # The provider said it was done and the checks disagree.
                    # The checks win; that is what they are for.
                    result.status = ExpertStatus.FAILED
                    result.blocker = result.blocker or "acceptance checks failed after the expert reported success"
            return result

        return last or ExpertResult(status=ExpertStatus.NOT_CONFIGURED, blocker="no provider could run the job")

    # -- verification ----------------------------------------------------

    def verify(self, job: ExpertJob) -> list[tuple[str, bool, str]]:
        """Run the job's acceptance criteria ourselves, in its workspace.

        Deliberately not delegated: an expert reporting on its own tests is the
        same trust failure as a local model reporting on its own patch, and it
        already caught a seeded skeleton certifying itself once.
        """

        evidence: list[tuple[str, bool, str]] = []
        for text, command in job.acceptance:
            passed, output = _run(command, job.workspace)
            evidence.append((text, passed, output))
        return evidence

    def fallback_channels(self, after: SpendChannel) -> list[SpendChannel]:
        return self.policy.fallbacks_for(after)


def _run(command: list[str], cwd: Path, *, timeout: float = 900.0) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        return False, f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"
    output = f"exit={completed.returncode}\n{completed.stdout}\n{completed.stderr}"
    return completed.returncode == 0, output
