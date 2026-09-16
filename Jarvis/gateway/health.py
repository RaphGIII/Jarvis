"""Telling a provider problem from a task problem.

A provider that is down, rate-limited, out of quota, rejecting the key or
missing the model has said nothing about how hard the task is.  Treating such
a failure as "needs a stronger model" is how a cheap outage becomes an
expensive bill, so every provider error is classified here first, and only
``TASK_FAILURE`` is allowed to teach the reliability model anything.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo


class ProviderStatus(str, Enum):
    OK = "ok"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTHENTICATION_ERROR = "authentication_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    #: The provider accepted the request and never answered within the bound.
    #: Kept apart from 429/503/malformed/auth: a hang says nothing about the
    #: key, the quota or the task, and it must never be retried at full length.
    TIMEOUT = "timeout"
    TASK_FAILURE = "task_failure"

    @property
    def is_outage(self) -> bool:
        """True for anything that is about the provider, not the task."""

        return self in {ProviderStatus.PROVIDER_UNAVAILABLE, ProviderStatus.RATE_LIMIT, ProviderStatus.QUOTA_EXHAUSTED,
                        ProviderStatus.AUTHENTICATION_ERROR, ProviderStatus.MODEL_UNAVAILABLE, ProviderStatus.TIMEOUT}


class GatewayError(RuntimeError):
    """A provider call did not produce a usable answer, and here is what kind of not."""

    def __init__(self, status: ProviderStatus, message: str, *, role: str = "", provider: str = "",
                 http_status: int | None = None, retry_after_seconds: float | None = None, model: str = "") -> None:
        self.status = status
        self.role = role
        self.provider = provider
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        #: The concrete model the failed call was addressed to, when the
        #: caller knows it (the pool loops set it): what an interrupted
        #: answer is labelled with, never a placeholder.
        self.model = model
        super().__init__(f"{status.value} from {provider or '?'} ({role or '?'}): {message}")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "role": self.role, "provider": self.provider, "model": self.model,
                "http_status": self.http_status, "retry_after_seconds": self.retry_after_seconds, "message": str(self)}


class FreeIntelligenceUnavailable(GatewayError):
    """The configured zero-cost reasoning pool could not produce an answer."""

    typed_status = "FREE_INTELLIGENCE_UNAVAILABLE"

    def __init__(self, message: str = "free intelligence is temporarily unavailable", *, role: str = "reasoning.free",
                 provider: str = "", attempts: list[dict[str, Any]] | None = None) -> None:
        self.attempts = list(attempts or [])
        # The pool's status is what every attempt said: a pool whose models
        # all hung is a timeout, recorded as such; anything mixed is the
        # general outage.
        classes = {str(a.get("failure_class")) for a in self.attempts} - {"ok"}
        status = ProviderStatus.TIMEOUT if classes and classes == {ProviderStatus.TIMEOUT.value} else ProviderStatus.PROVIDER_UNAVAILABLE
        if classes:
            message = f"{message} ({', '.join(sorted(classes))})"
        super().__init__(status, message, role=role, provider=provider)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["typed_status"] = self.typed_status
        data["attempts"] = list(self.attempts)
        return data


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
        # Groq ("tokens per day (TPD)", "requests per day (RPD)") and OpenRouter
        # ("free-models-per-day") name a daily allowance: it is spent until the
        # provider's day or window resets, not for thirty seconds.
        if _DAILY_QUOTA.search(text):
            return ProviderStatus.QUOTA_EXHAUSTED
        if any(marker in text for marker in ("quota", "insufficient_quota", "billing", "exceeded your current quota",
                                             "resource_exhausted", "credit")):
            return ProviderStatus.QUOTA_EXHAUSTED
        return ProviderStatus.RATE_LIMIT
    if status_code == 402 or "insufficient_quota" in text or "billing_hard_limit" in text:
        return ProviderStatus.QUOTA_EXHAUSTED
    if any(marker in text for marker in ("credit balance is too low", "insufficient credits", "purchase credits", "out of credits",
                                         "billing_error", "insufficient funds")):
        # Anthropic says this with HTTP 400.  An empty prepaid balance is the
        # account's state, not the task's: nothing about the model is learned.
        return ProviderStatus.QUOTA_EXHAUSTED
    if status_code >= 500 or status_code == 408:
        return ProviderStatus.PROVIDER_UNAVAILABLE
    return ProviderStatus.TASK_FAILURE


#: Quota wording that means "resets with the provider's day", not "in a minute".
_DAILY_QUOTA = re.compile(r"per[\s_-]+day|perday|daily|requests_per_day|tokens_per_day|\((?:tpd|rpd)\)|taeglich|täglich", re.I)


def _us_pacific_offset(at: float) -> timezone:
    """UTC-7 during US daylight time (second Sunday of March 02:00 to first Sunday of November 02:00), else UTC-8."""

    year = datetime.fromtimestamp(at, timezone.utc).year

    def nth_sunday(month: int, n: int) -> datetime:
        first = datetime(year, month, 1, tzinfo=timezone.utc)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    start = nth_sunday(3, 2) + timedelta(hours=10)  # 02:00 PST
    end = nth_sunday(11, 1) + timedelta(hours=9)  # 02:00 PDT
    daylight = start <= datetime.fromtimestamp(at, timezone.utc) < end
    return timezone(timedelta(hours=-7 if daylight else -8))


def _zone(name: str, at: float) -> Any:
    """A tzinfo for ``name``: the tz database when present; UTC and US Pacific also without one (Windows ships none)."""

    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - no tz database on this host
        pass
    if name.upper() in {"UTC", "ETC/UTC", "GMT"}:
        return timezone.utc
    if name in {"America/Los_Angeles", "US/Pacific"}:
        return _us_pacific_offset(at)
    return None


def seconds_until_daily_reset(now: float | None = None, zone: str = "America/Los_Angeles") -> float:
    """Seconds until the next midnight in the provider's accounting zone (Google's free tier resets at Pacific midnight)."""

    tz = _zone(zone, now if now is not None else time.time())
    if tz is None:
        return 24 * 3600.0  # an unknown zone without a tz database: a plain day
    current = datetime.fromtimestamp(now if now is not None else time.time(), tz)
    tomorrow = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60.0, (tomorrow - current).total_seconds())


@dataclass
class ProviderHealth:
    """The last thing each provider -- and each model -- told us, with a cool-down for outages.

    Keyed by provider, and additionally by ``provider/model`` when the caller
    names the model: one Gemini model's spent daily quota does not condemn
    its siblings, and a model that does not exist is not retried every turn.
    A route in cool-down is skipped without a network call; that is what
    keeps an exhausted route from being hammered.
    """

    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def _cooldown(status: ProviderStatus, detail: str, retry_after_seconds: float | None, reset_zone: str = "") -> float:
        if status is ProviderStatus.RATE_LIMIT:
            return retry_after_seconds or 30.0
        if status is ProviderStatus.QUOTA_EXHAUSTED:
            # The provider's own reset time wins; otherwise a daily allowance
            # lasts until midnight in the provider's accounting zone.
            if retry_after_seconds:
                return float(retry_after_seconds)
            if _DAILY_QUOTA.search(detail or ""):
                return seconds_until_daily_reset(zone=reset_zone) if reset_zone else seconds_until_daily_reset()
            return 3600.0
        if status is ProviderStatus.PROVIDER_UNAVAILABLE:
            return retry_after_seconds or 60.0
        if status is ProviderStatus.TIMEOUT:
            return retry_after_seconds or 30.0  # a hang under load; short, so a recovered provider is used again soon
        if status is ProviderStatus.MODEL_UNAVAILABLE:
            return retry_after_seconds or 6 * 3600.0  # a model that is not there stays not there for a while
        if status is ProviderStatus.AUTHENTICATION_ERROR:
            return 0.0  # a key does not fix itself; report, do not hide
        return 0.0

    def note(self, provider: str, status: ProviderStatus, *, detail: str = "", retry_after_seconds: float | None = None,
             model: str = "", reset_zone: str = "") -> None:
        cooldown = self._cooldown(status, detail, retry_after_seconds, reset_zone)
        entry = {"status": status.value, "detail": detail[:300], "at": time.time(), "until": time.time() + cooldown if cooldown else 0.0}
        if model:
            self.state[f"{provider}/{model}"] = entry
            # A spent daily quota and a missing model are that model's
            # condition; everything else (a hang, a 5xx, a per-minute limit,
            # a rejected key, an answer) is the provider's.
            if status not in {ProviderStatus.QUOTA_EXHAUSTED, ProviderStatus.MODEL_UNAVAILABLE}:
                self.state[provider] = dict(entry)
        else:
            self.state[provider] = entry

    def cooldown_remaining(self, provider: str, *, model: str = "") -> float:
        remaining = 0.0
        for key in ([provider] + ([f"{provider}/{model}"] if model else [])):
            entry = self.state.get(key)
            if entry and entry.get("until"):
                remaining = max(remaining, float(entry["until"]) - time.time())
        return max(0.0, remaining)

    def status(self, provider: str, *, model: str = "") -> ProviderStatus:
        if model:
            own = self._status_of(f"{provider}/{model}")
            if own.is_outage:
                return own
        return self._status_of(provider)

    def _status_of(self, key: str) -> ProviderStatus:
        entry = self.state.get(key)
        if not entry:
            return ProviderStatus.OK
        until = float(entry.get("until") or 0.0)
        status = ProviderStatus(entry["status"])
        if status is ProviderStatus.AUTHENTICATION_ERROR:
            return status
        if until and time.time() < until:
            return status
        return ProviderStatus.OK

    def usable(self, provider: str, *, model: str = "") -> bool:
        return not self.status(provider, model=model).is_outage

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for provider, entry in self.state.items():
            current = self.status(provider)
            out[provider] = {"status": current.value, "last": entry.get("status"), "detail": entry.get("detail", ""),
                             "cooldown_remaining": max(0.0, float(entry.get("until") or 0.0) - time.time())}
        return out
