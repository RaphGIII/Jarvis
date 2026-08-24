"""What Jarvis is allowed to spend, and the refusal to spend it quietly.

The rule this module exists to enforce is simple to state and easy to violate by
accident: *Jarvis must never silently create usage-based AI costs.*

Accidents happen a particular way.  Something works, then hits a limit, and the
obvious repair is a fallback -- the subscription CLI is rate-limited, an API key
happens to be in the environment, and one `except: ` later the system is quietly
billing per token.  Nobody decided that.  It emerged from a sensible-looking
error handler.

So the design here is deliberately not "check a flag before spending money".  It
is:

*Channels, not vendors.*  What matters is not whether a request goes to Anthropic
or OpenAI but how it is *paid for*: a flat subscription the user already has, a
metered API, purchased credits, rented GPUs.  :class:`SpendChannel` names those,
and policy is expressed over them.

*Deny by default, and only by explicit configuration otherwise.*  Every metered
channel starts closed.  Nothing infers permission from the presence of a
credential: an API key in the environment is a fact about the machine, never a
decision about spending.

*Refusal is loud and typed.*  :meth:`CostPolicy.require` raises
:class:`CostPolicyViolation` rather than returning False, because a silently
skipped escalation and a silently billed one are both failures, and only an
exception makes the caller state what it intends to do about it.

*Exhaustion is not a licence.*  Running out of subscription quota is an ordinary
state -- :data:`EXPERT_UNAVAILABLE` -- and the correct responses are to use a
local model, wait for reset, or checkpoint.  Substituting a metered channel is
not among them, and :meth:`CostPolicy.fallbacks_for` will not offer one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class SpendChannel(str, Enum):
    """How a unit of work would be paid for.

    Grouped by billing model rather than by vendor, because that is what the
    policy is actually about.  Two Anthropic calls can sit in different channels
    -- one inside a subscription the user already pays for, one metered per
    token -- and the difference between them is the whole point.
    """

    #: Models running on the user's own hardware.  Costs electricity.
    LOCAL_MODEL = "local_model"
    #: An officially supported CLI/SDK authenticated against a subscription the
    #: user already pays a flat fee for.  No marginal cost per request.
    SUBSCRIPTION_CLI = "subscription_cli"
    #: Metered per-token API billing.  Off by default, forever, unless the user
    #: turns it on themselves.
    PAID_API = "paid_api"
    #: Prepaid credits, including "auto-reload" arrangements.
    USAGE_CREDITS = "usage_credits"
    #: Rented compute (RunPod and similar).
    RUNPOD = "runpod"
    #: Driving a consumer AI web interface with browser automation.  Disallowed
    #: by default: it is typically against the provider's terms, it is brittle,
    #: and it launders a metered or prohibited service into something that looks
    #: free.
    BROWSER_AI_AUTOMATION = "browser_ai_automation"

    @property
    def metered(self) -> bool:
        """True when using this channel can create a marginal charge."""

        return self in {SpendChannel.PAID_API, SpendChannel.USAGE_CREDITS, SpendChannel.RUNPOD}


#: The status an expert provider reports when its subscription quota is spent.
#: Deliberately a plain constant: it is a state to be handled, not an error to
#: be recovered from by finding another way to pay.
EXPERT_UNAVAILABLE = "EXPERT_UNAVAILABLE"


class CostPolicyViolation(RuntimeError):
    """Raised when something tried to use a channel the policy forbids.

    Carries the channel so a caller can distinguish "expert is off" from "expert
    is broken" without parsing a message.
    """

    def __init__(self, channel: SpendChannel, detail: str = "") -> None:
        self.channel = channel
        self.detail = detail
        message = f"cost policy forbids {channel.value}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class CostDecision:
    """One recorded permit/refuse, for the diagnostics view."""

    channel: SpendChannel
    allowed: bool
    reason: str
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "at": self.at,
        }


@dataclass(frozen=True)
class CostPolicy:
    """The spending rules.  Immutable: changing policy produces a new policy.

    The defaults are the shipped policy, and they are the strict ones.  A user
    who wants metered billing has to say so; no amount of failure elsewhere in
    the system can arrive at it by falling back.
    """

    allow_local_models: bool = True
    allow_subscription_cli: bool = True
    allow_paid_api: bool = False
    allow_usage_credits: bool = False
    allow_runpod: bool = False
    allow_browser_automation_for_ai_chat: bool = False

    #: Where the settings came from, for the diagnostics view.
    source: str = "defaults"

    # -- querying --------------------------------------------------------

    def permits(self, channel: SpendChannel) -> bool:
        return {
            SpendChannel.LOCAL_MODEL: self.allow_local_models,
            SpendChannel.SUBSCRIPTION_CLI: self.allow_subscription_cli,
            SpendChannel.PAID_API: self.allow_paid_api,
            SpendChannel.USAGE_CREDITS: self.allow_usage_credits,
            SpendChannel.RUNPOD: self.allow_runpod,
            SpendChannel.BROWSER_AI_AUTOMATION: self.allow_browser_automation_for_ai_chat,
        }[channel]

    def require(self, channel: SpendChannel, *, detail: str = "") -> None:
        """Permit the channel or raise.

        Raises rather than returning a bool on purpose.  A caller that forgets
        to check a returned False spends the money anyway; a caller that forgets
        to catch an exception crashes loudly and gets fixed.
        """

        if not self.permits(channel):
            raise CostPolicyViolation(channel, detail or self.explain(channel))

    def explain(self, channel: SpendChannel) -> str:
        """Why a channel is closed, in terms a user can act on."""

        if self.permits(channel):
            return f"{channel.value} is allowed by the current policy ({self.source})"
        if channel is SpendChannel.BROWSER_AI_AUTOMATION:
            return (
                "driving a consumer AI web interface is disabled: it generally breaches the "
                "provider's terms and disguises a metered service as a free one"
            )
        if channel.metered:
            return (
                f"{channel.value} would create usage-based charges and is disabled by default; "
                "enable it deliberately in config/cost_policy.json if that is what you want"
            )
        return f"{channel.value} is disabled by the current policy ({self.source})"

    def fallbacks_for(self, channel: SpendChannel) -> list[SpendChannel]:
        """What may be tried instead when ``channel`` is unavailable.

        Never widens the blast radius.  A subscription that ran out of quota
        falls back to local inference and nowhere else -- specifically not to a
        metered channel, which is the exact path by which "the expert was busy"
        silently becomes a bill.
        """

        if channel is SpendChannel.SUBSCRIPTION_CLI:
            return [SpendChannel.LOCAL_MODEL] if self.allow_local_models else []
        return []

    # -- the channels that are on ---------------------------------------

    @property
    def enabled_channels(self) -> list[SpendChannel]:
        return [channel for channel in SpendChannel if self.permits(channel)]

    @property
    def metered_channels_enabled(self) -> list[SpendChannel]:
        return [channel for channel in self.enabled_channels if channel.metered]

    @property
    def is_free(self) -> bool:
        """True when nothing Jarvis may do can produce a marginal charge."""

        return not self.metered_channels_enabled and not self.allow_browser_automation_for_ai_chat

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_local_models": self.allow_local_models,
            "allow_subscription_cli": self.allow_subscription_cli,
            "allow_paid_api": self.allow_paid_api,
            "allow_usage_credits": self.allow_usage_credits,
            "allow_runpod": self.allow_runpod,
            "allow_browser_automation_for_ai_chat": self.allow_browser_automation_for_ai_chat,
            "source": self.source,
            "is_free": self.is_free,
        }

    # -- construction ----------------------------------------------------

    @classmethod
    def strict(cls) -> "CostPolicy":
        """Everything metered is off.  The shipped default."""

        return cls()

    @classmethod
    def load(cls, *, config_dir: str | Path | None = None, environ: dict[str, str] | None = None) -> "CostPolicy":
        """Defaults, then the config file, then the environment.

        The environment can only be *read* for explicit policy switches --
        ``JARVIS_ALLOW_PAID_API`` and friends.  It is never consulted for
        credentials: whether ``ANTHROPIC_API_KEY`` happens to be set says
        nothing about whether the user wants to be billed, and treating a
        present key as consent is precisely the accident this module prevents.
        """

        policy = cls.strict()
        source = "defaults"

        path = Path(config_dir or _default_config_dir()) / "cost_policy.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                fields = {
                    key: bool(value)
                    for key, value in data.items()
                    if key in _BOOLEAN_FIELDS and isinstance(value, (bool, int))
                }
                if fields:
                    policy = replace(policy, **fields)
                    source = str(path)

        env = os.environ if environ is None else environ
        overrides: dict[str, bool] = {}
        for name in _BOOLEAN_FIELDS:
            variable = f"JARVIS_{name.upper()}"
            if variable in env:
                overrides[name] = _truthy(env[variable])
        if overrides:
            policy = replace(policy, **overrides)
            source = f"{source}+env"

        return replace(policy, source=source)


_BOOLEAN_FIELDS = (
    "allow_local_models",
    "allow_subscription_cli",
    "allow_paid_api",
    "allow_usage_credits",
    "allow_runpod",
    "allow_browser_automation_for_ai_chat",
)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


class CostLedger:
    """A record of what was permitted and what was refused.

    Not accounting -- Jarvis cannot see the user's invoice -- but an audit
    trail.  When the diagnostics view claims "no paid channel has been used",
    this is the thing that can substantiate it.
    """

    def __init__(self, policy: CostPolicy | None = None, *, limit: int = 500) -> None:
        self.policy = policy or CostPolicy.load()
        self.limit = limit
        self._decisions: list[CostDecision] = []

    def check(self, channel: SpendChannel, *, reason: str = "") -> bool:
        allowed = self.policy.permits(channel)
        self._record(channel, allowed, reason or self.policy.explain(channel))
        return allowed

    def require(self, channel: SpendChannel, *, reason: str = "") -> None:
        allowed = self.policy.permits(channel)
        self._record(channel, allowed, reason or self.policy.explain(channel))
        if not allowed:
            raise CostPolicyViolation(channel, self.policy.explain(channel))

    def _record(self, channel: SpendChannel, allowed: bool, reason: str) -> None:
        self._decisions.append(
            CostDecision(
                channel=channel,
                allowed=allowed,
                reason=reason,
                at=datetime.now(timezone.utc).isoformat(),
            )
        )
        del self._decisions[: max(0, len(self._decisions) - self.limit)]

    @property
    def decisions(self) -> list[CostDecision]:
        return list(self._decisions)

    @property
    def refusals(self) -> list[CostDecision]:
        return [item for item in self._decisions if not item.allowed]

    def used_metered_channel(self) -> bool:
        """True if any metered channel was ever permitted through this ledger."""

        return any(item.allowed and item.channel.metered for item in self._decisions)

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "decisions": len(self._decisions),
            "refusals": len(self.refusals),
            "used_metered_channel": self.used_metered_channel(),
        }
