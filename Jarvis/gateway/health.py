"""Telling a provider problem from a task problem.

A provider that is down, rate-limited, out of quota, rejecting the key or
missing the model has said nothing about how hard the task is.  Treating such
a failure as "needs a stronger model" is how a cheap outage becomes an
expensive bill, so every provider error is classified here first, and only
``TASK_FAILURE`` is allowed to teach the reliability model anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderStatus(str, Enum):
    OK = "ok"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTHENTICATION_ERROR = "authentication_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    TASK_FAILURE = "task_failure"

    @property
    def is_outage(self) -> bool:
        """True for anything that is about the provider, not the task."""

        return self in {ProviderStatus.PROVIDER_UNAVAILABLE, ProviderStatus.RATE_LIMIT, ProviderStatus.QUOTA_EXHAUSTED,
                        ProviderStatus.AUTHENTICATION_ERROR, ProviderStatus.MODEL_UNAVAILABLE}


class GatewayError(RuntimeError):
    """A provider call did not produce a usable answer, and here is what kind of not."""

    def __init__(self, status: ProviderStatus, message: str, *, role: str = "", provider: str = "",
                 http_status: int | None = None, retry_after_seconds: float | None = None) -> None:
        self.status = status
        self.role = role
        self.provider = provider
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"{status.value} from {provider or '?'} ({role or '?'}): {message}")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "role": self.role, "provider": self.provider,
                "http_status": self.http_status, "retry_after_seconds": self.retry_after_seconds, "message": str(self)}


def classify_http(status_code: int, body: str, *, provider_kind: str = "") -> ProviderStatus:
    """Map an HTTP failure onto the six states.  Provider quirks live here."""

    text = (body or "").lower()
    if status_code in {401, 403}:
        # Gemini reports a disabled/invalid key as 400 with API_KEY_INVALID;
        # everyone else uses 401/403.
        return ProviderStatus.AUTHENTICATION_ERROR
    if status_code == 400 and ("api_key_invalid" in text or "api key not valid" in text or "invalid x-api-key" in text):
        return ProviderStatus.AUTHENTICATION_ERROR
    if status_code == 404 or "model_not_found" in text or "not found for api version" in text or "unknown model" in text:
        return ProviderStatus.MODEL_UNAVAILABLE
    if status_code == 429:
        if provider_kind == "gemini":
            # Gemini answers both with RESOURCE_EXHAUSTED.  The quota metric
            # says which: a per-minute limit passes in seconds, a per-day
            # limit is exhausted until the day rolls over.
            if any(marker in text for marker in ("perday", "per_day", "per day", "daily", "generaterequestsperday", "tokensperday")):
                return ProviderStatus.QUOTA_EXHAUSTED
            if any(marker in text for marker in ("perminute", "per_minute", "per minute", "generaterequestsperminute", "tokensperminute",
                                                 "retrydelay", "retry in")):
                return ProviderStatus.RATE_LIMIT
        if any(marker in text for marker in ("quota", "insufficient_quota", "billing", "exceeded your current quota",
                                             "resource_exhausted", "credit")):
            return ProviderStatus.QUOTA_EXHAUSTED
        return ProviderStatus.RATE_LIMIT
    if status_code == 402 or "insufficient_quota" in text or "billing_hard_limit" in text:
        return ProviderStatus.QUOTA_EXHAUSTED
    if status_code >= 500 or status_code == 408:
        return ProviderStatus.PROVIDER_UNAVAILABLE
    return ProviderStatus.TASK_FAILURE


@dataclass
class ProviderHealth:
    """The last thing each provider told us, with a cool-down for outages."""

    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def note(self, provider: str, status: ProviderStatus, *, detail: str = "", retry_after_seconds: float | None = None) -> None:
        cooldown = 0.0
        if status is ProviderStatus.RATE_LIMIT:
            cooldown = retry_after_seconds or 30.0
        elif status is ProviderStatus.QUOTA_EXHAUSTED:
            cooldown = retry_after_seconds or 3600.0
        elif status is ProviderStatus.PROVIDER_UNAVAILABLE:
            cooldown = retry_after_seconds or 60.0
        elif status is ProviderStatus.AUTHENTICATION_ERROR:
            cooldown = 0.0  # a key does not fix itself; report, do not hide
        self.state[provider] = {"status": status.value, "detail": detail[:300], "at": time.time(),
                                "until": time.time() + cooldown if cooldown else 0.0}

    def status(self, provider: str) -> ProviderStatus:
        entry = self.state.get(provider)
        if not entry:
            return ProviderStatus.OK
        until = float(entry.get("until") or 0.0)
        status = ProviderStatus(entry["status"])
        if status is ProviderStatus.AUTHENTICATION_ERROR:
            return status
        if until and time.time() < until:
            return status
        return ProviderStatus.OK

    def usable(self, provider: str) -> bool:
        return not self.status(provider).is_outage

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for provider, entry in self.state.items():
            current = self.status(provider)
            out[provider] = {"status": current.value, "last": entry.get("status"), "detail": entry.get("detail", ""),
                             "cooldown_remaining": max(0.0, float(entry.get("until") or 0.0) - time.time())}
        return out
