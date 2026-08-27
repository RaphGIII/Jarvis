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


#: Availability as a state, not a boolean.  "Configured" is not "available":
#: an installed, signed-in CLI whose session limit was hit ten minutes ago is
#: QUOTA_EXHAUSTED until a later call proves otherwise, and Diagnostics says so.
AVAILABILITY_STATES = ("NOT_INSTALLED", "INSTALLED", "NOT_AUTHENTICATED", "AUTHENTICATED", "AVAILABLE", "BUSY",
                       "QUOTA_EXHAUSTED", "RATE_LIMITED", "UNKNOWN", "ERROR")

#: How long a probe result stands before the next status() re-probes.
AVAILABILITY_TTL_SECONDS = 300.0


@dataclass
class ProviderAvailability:
    """Whether a provider could be used right now, and why not if it cannot."""

    available: bool
    detail: str = ""
    #: True when the provider exists but its quota is spent.
    quota_exhausted: bool = False
    version: str = ""
    state: str = "UNKNOWN"
    checked_at: float = 0.0
    evidence: str = ""

    def resolved_state(self) -> str:
        if self.state != "UNKNOWN":
            return self.state
        if self.quota_exhausted:
            return "QUOTA_EXHAUSTED"
        if self.available:
            return "AVAILABLE"
        lowered = self.detail.lower()
        if "not installed" in lowered or "not on path" in lowered:
            return "NOT_INSTALLED"
        if "log in" in lowered or "login" in lowered or "sign in" in lowered or "authenticat" in lowered:
            return "NOT_AUTHENTICATED"
        return "ERROR" if self.detail else "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "detail": self.detail,
            "state": self.resolved_state(),
            "checked_at": self.checked_at,
            "evidence": self.evidence,
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

    def _cache(self) -> dict[str, ProviderAvailability]:
        cache = getattr(self, "_availability", None)
        if cache is None:
            cache = self._availability = {}
        return cache

    def _probe(self, provider: ExpertProvider) -> ProviderAvailability:
        availability = provider.availability()
        availability.checked_at = time.time()
        availability.state = availability.resolved_state()
        availability.evidence = availability.evidence or "probe: --version"
        self._cache()[provider.name] = availability
        return availability

    def note_result(self, provider_name: str, result: "ExpertResult") -> None:
        """Real execution evidence outranks a probe: quota and rate limits show up here."""

        cache = self._cache()
        previous = cache.get(provider_name)
        version = previous.version if previous else ""
        detail = str(getattr(getattr(result, "quota", None), "detail", "") or result.blocker or "")[:300]
        if result.status is ExpertStatus.UNAVAILABLE and getattr(getattr(result, "quota", None), "exhausted", False):
            lowered = detail.lower()
            state = "RATE_LIMITED" if ("rate limit" in lowered or "429" in lowered or "too many requests" in lowered) else "QUOTA_EXHAUSTED"
            cache[provider_name] = ProviderAvailability(False, detail or "subscription quota exhausted", quota_exhausted=True, version=version,
                                                        state=state, checked_at=time.time(), evidence=f"execution: {result.status.value}")
        elif result.status.ran or result.status is ExpertStatus.COMPLETED:
            cache[provider_name] = ProviderAvailability(True, "subscription CLI answered", version=version, state="AVAILABLE",
                                                        checked_at=time.time(), evidence=f"execution: {result.status.value}")
        elif result.status is ExpertStatus.BLOCKED:
            cache[provider_name] = ProviderAvailability(False, detail or "blocked", version=version, state="ERROR",
                                                        checked_at=time.time(), evidence=f"execution: {result.status.value}")

    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        """What the UI needs: per provider a STATE with the evidence and its age.

        Cached: a probe spawns the CLI, and Diagnostics is drawn far more often
        than availability changes.  A quota or rate-limit state learned from a
        real execution is kept until an execution succeeds or ``refresh`` is
        asked for explicitly -- a ``--version`` probe cannot prove a session
        window has reset.
        """

        rows = []
        cache = self._cache()
        for provider in self.providers:
            permitted = self.policy.permits(provider.channel)
            if not permitted:
                availability = ProviderAvailability(False, self.policy.explain(provider.channel), state="ERROR")
            else:
                cached = cache.get(provider.name)
                stale = cached is None or (time.time() - cached.checked_at) > AVAILABILITY_TTL_SECONDS
                learned = cached is not None and cached.evidence.startswith("execution:") and cached.state in {"QUOTA_EXHAUSTED", "RATE_LIMITED"}
                if refresh or (stale and not learned):
                    try:
                        availability = self._probe(provider)
                    except Exception as exc:  # noqa: BLE001 - a broken probe is a state, not a crash
                        availability = cache[provider.name] = ProviderAvailability(False, f"probe failed: {exc}", state="ERROR", checked_at=time.time())
                else:
                    availability = cached
            rows.append(
                {
                    "name": provider.name,
                    "channel": provider.channel.value,
                    "permitted": permitted,
                    **availability.to_dict(),
                }
            )
        states = [row["state"] for row in rows if row["permitted"]]
        return {
            "policy": self.policy.to_dict(),
            "providers": rows,
            "state": (states[0] if len(states) == 1 else ("AVAILABLE" if "AVAILABLE" in states else (states[0] if states else "UNKNOWN"))),
            "checked_at": max((row.get("checked_at") or 0.0) for row in rows) if rows else 0.0,
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
            try:
                self.note_result(provider.name, result)
            except Exception:  # noqa: BLE001 - bookkeeping never changes the outcome
                pass

            # Verify whenever the provider may have left usable work behind --
            # including a TIMEOUT. Observed live: the expert wrote a complete,
            # correct 20KB implementation and was cut off at its budget while
            # tidying up. Refusing to look at the workspace because the clock
            # ran out threw away work that passed every acceptance check.
            if result.status.ran or result.status is ExpertStatus.TIMEOUT:
                result.test_evidence = self.verify(job)
                if result.status is ExpertStatus.TIMEOUT and result.verified:
                    # The work is done and proven; how it ended is a detail.
                    result.status = ExpertStatus.COMPLETED
                    result.blocker = ""
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
