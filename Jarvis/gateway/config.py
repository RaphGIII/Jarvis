"""Roles, providers, prices and caps -- read from configuration, never from code.

``config/providers.json`` binds each abstract role to a provider and model:

    reasoning.free      -> gemini / gemini-2.5-flash   (free tier, may train)
    reasoning.deep      -> openai / gpt-5.6            (metered)
    engineer.standard   -> anthropic / claude-opus-5   (metered)
    engineer.frontier   -> anthropic / claude-fable-5-1(metered)
    speech.stt, speech.tts
    local.fast          -> the FAST_LOCAL Ollama tier   (offline fallback only)

Swapping a provider is an edit to that file.  No module outside this one
knows a model name, and the tests assert exactly that.

Prices are per million tokens in EUR and are configuration too.  A metered
role without a price cannot be estimated, cannot be reserved and therefore
cannot be called -- the gateway refuses rather than guesses.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from gateway.modes import CostClass, RoleFamily

ROLE_NAMES: tuple[str, ...] = (
    "reasoning.free",
    "reasoning.deep",
    "engineer.standard",
    "engineer.frontier",
    "speech.stt",
    "speech.tts",
    "local.fast",
    "local.build",
)

PROVIDER_KINDS: tuple[str, ...] = ("gemini", "openai", "anthropic", "ollama", "openai_compatible")

#: Reasoning effort levels the free role exposes.  Mapped per provider to
#: whatever that provider calls it (a thinking budget, a reasoning effort).
THINKING_LEVELS: tuple[str, ...] = ("FREE_LOW", "FREE_MEDIUM", "FREE_HIGH")


@dataclass(frozen=True)
class Pricing:
    """EUR per one million tokens."""

    input_per_m: float = 0.0
    cached_input_per_m: float = 0.0
    output_per_m: float = 0.0
    #: False until the owner has checked these against the provider's price
    #: list.  Estimates from unconfirmed prices are labelled as such in the UI.
    confirmed: bool = False

    @property
    def metered(self) -> bool:
        return any(value > 0 for value in (self.input_per_m, self.cached_input_per_m, self.output_per_m))

    def to_dict(self) -> dict[str, Any]:
        return {"input_per_m": self.input_per_m, "cached_input_per_m": self.cached_input_per_m,
                "output_per_m": self.output_per_m, "confirmed": self.confirmed}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str
    base_url: str = ""
    enabled: bool = False
    #: Secret slot in :class:`gateway.secrets.CredentialStore`; empty for local providers.
    secret: str = ""
    #: True when the provider's terms let it use request content to improve
    #: its products.  Such a provider never receives private context.
    may_train_on_requests: bool = False
    #: Whether requests are billed at all.  A free tier is ``False``.
    metered: bool = True
    #: Monthly ceiling for this provider alone, EUR.  ``None`` = governor default.
    monthly_cap_eur: float | None = None
    timeout_seconds: float = 120.0
    #: Model name -> pricing.  Missing model = cannot be estimated.
    pricing: dict[str, Pricing] = field(default_factory=dict)
    #: Optional per-model settings (e.g. thinking budgets), passed through.
    options: dict[str, Any] = field(default_factory=dict)

    def price_for(self, model: str) -> Pricing | None:
        return self.pricing.get(model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "base_url": self.base_url, "enabled": self.enabled,
            "secret": self.secret, "may_train_on_requests": self.may_train_on_requests, "metered": self.metered,
            "monthly_cap_eur": self.monthly_cap_eur, "timeout_seconds": self.timeout_seconds,
            "pricing": {model: price.to_dict() for model, price in self.pricing.items()},
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class RoleBinding:
    role: str
    provider: str
    model: str
    enabled: bool = True
    #: Reasoning effort configuration: level name -> provider-specific value.
    thinking: dict[str, Any] = field(default_factory=dict)
    #: Prior belief in this role's reliability per task class, before any
    #: observation: {task_class: [successes, failures]} pseudo-counts.
    reliability_prior: dict[str, list[float]] = field(default_factory=dict)
    max_output_tokens: int = 2048
    temperature: float = 0.3
    #: For local roles: which :class:`brain.tiers.ModelTier` serves it.
    tier: str = ""
    offline_fallback: bool = False

    @property
    def family(self) -> RoleFamily:
        return RoleFamily(self.role.split(".", 1)[0])

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "provider": self.provider, "model": self.model, "enabled": self.enabled,
                "thinking": dict(self.thinking), "reliability_prior": {k: list(v) for k, v in self.reliability_prior.items()},
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature, "tier": self.tier,
                "offline_fallback": self.offline_fallback}


@dataclass(frozen=True)
class BudgetConfig:
    """Hard caps in EUR.  Soft allocations are advisory and may move; these do not."""

    monthly_hard_cap: float = 40.0
    daily_hard_cap: float = 6.0
    per_task_hard_cap: float = 5.0
    reasoning_hard_cap: float = 10.0
    engineering_hard_cap: float = 25.0
    provider_hard_caps: dict[str, float] = field(default_factory=dict)
    #: reserved = estimated * safety_factor
    safety_factor: float = 1.5
    soft_allocation: dict[str, float] = field(default_factory=lambda: {"deep_reasoning": 10.0, "engineering": 25.0, "reserve": 5.0})

    def to_dict(self) -> dict[str, Any]:
        return {"monthly_hard_cap": self.monthly_hard_cap, "daily_hard_cap": self.daily_hard_cap,
                "per_task_hard_cap": self.per_task_hard_cap, "reasoning_hard_cap": self.reasoning_hard_cap,
                "engineering_hard_cap": self.engineering_hard_cap, "provider_hard_caps": dict(self.provider_hard_caps),
                "safety_factor": self.safety_factor, "soft_allocation": dict(self.soft_allocation)}


@dataclass(frozen=True)
class GatewayConfig:
    roles: dict[str, RoleBinding]
    providers: dict[str, ProviderConfig]
    budget: BudgetConfig
    currency: str = "EUR"
    source: str = "defaults"
    #: beta in q = p_hat - beta * sigma
    risk_aversion: float = 1.0

    # -- queries ------------------------------------------------------------

    def binding(self, role: str) -> RoleBinding | None:
        return self.roles.get(role)

    def provider_for(self, role: str) -> ProviderConfig | None:
        binding = self.roles.get(role)
        return self.providers.get(binding.provider) if binding else None

    def cost_class(self, role: str) -> CostClass:
        provider = self.provider_for(role)
        binding = self.roles.get(role)
        if provider is None or binding is None:
            return CostClass.METERED  # unknown is never free
        if not provider.metered:
            return CostClass.ZERO
        price = provider.price_for(binding.model)
        if price is not None and not price.metered:
            return CostClass.ZERO
        return CostClass.METERED

    def pricing_for(self, role: str) -> Pricing | None:
        provider = self.provider_for(role)
        binding = self.roles.get(role)
        if provider is None or binding is None:
            return None
        if not provider.metered:
            return Pricing(0.0, 0.0, 0.0, confirmed=True)
        return provider.price_for(binding.model)

    def configured_roles(self) -> list[str]:
        """Roles whose binding and provider are both enabled."""

        out = []
        for role, binding in self.roles.items():
            provider = self.providers.get(binding.provider)
            if binding.enabled and provider is not None and provider.enabled:
                out.append(role)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency, "source": self.source, "risk_aversion": self.risk_aversion,
            "roles": {role: binding.to_dict() for role, binding in self.roles.items()},
            "providers": {name: provider.to_dict() for name, provider in self.providers.items()},
            "budget": self.budget.to_dict(),
        }

    # -- construction -----------------------------------------------------

    @classmethod
    def defaults(cls) -> "GatewayConfig":
        return _parse(DEFAULT_DOCUMENT, source="defaults")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GatewayConfig":
        target = Path(path) if path else default_path()
        if not target.is_file():
            return cls.defaults()
        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid gateway configuration at {target}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"invalid gateway configuration at {target}: not an object")
        return _parse(document, source=str(target))

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, **self.to_dict()}
        document.pop("source", None)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return target

    def with_provider_enabled(self, name: str, enabled: bool) -> "GatewayConfig":
        if name not in self.providers:
            raise KeyError(f"unknown provider: {name}")
        providers = dict(self.providers)
        providers[name] = replace(providers[name], enabled=bool(enabled))
        return replace(self, providers=providers)

    def with_role_binding(self, role: str, *, provider: str | None = None, model: str | None = None,
                          enabled: bool | None = None) -> "GatewayConfig":
        if role not in self.roles:
            raise KeyError(f"unknown role: {role}")
        binding = self.roles[role]
        changes: dict[str, Any] = {}
        if provider is not None:
            if provider not in self.providers:
                raise KeyError(f"unknown provider: {provider}")
            changes["provider"] = provider
        if model is not None:
            changes["model"] = str(model)
        if enabled is not None:
            changes["enabled"] = bool(enabled)
        roles = dict(self.roles)
        roles[role] = replace(binding, **changes)
        return replace(self, roles=roles)


def default_path() -> Path:
    configured = os.environ.get("ZEUS_PROVIDERS_CONFIG", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "config" / "providers.json"


# ---------------------------------------------------------------------------
# The shipped defaults.  Every provider is disabled until the owner enters a
# credential and switches it on; the local roles are on because they cost
# nothing but electricity.  Prices are unconfirmed placeholders until the owner
# checks them -- the UI says so.
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENT: dict[str, Any] = {
    "schema_version": 1,
    "currency": "EUR",
    "risk_aversion": 1.0,
    "roles": {
        "reasoning.free": {
            "provider": "gemini", "model": "gemini-2.5-flash", "enabled": True,
            "thinking": {"FREE_LOW": 0, "FREE_MEDIUM": 2048, "FREE_HIGH": 8192},
            "reliability_prior": {"knowledge": [8, 1], "semantic": [6, 2], "planning": [3, 3], "composition": [2, 3]},
            "max_output_tokens": 2048, "temperature": 0.3,
        },
        "reasoning.deep": {
            "provider": "openai", "model": "gpt-5.6", "enabled": True,
            "thinking": {"FREE_LOW": "low", "FREE_MEDIUM": "medium", "FREE_HIGH": "high"},
            "reliability_prior": {"knowledge": [9, 1], "semantic": [9, 1], "planning": [8, 1], "composition": [7, 1]},
            "max_output_tokens": 4096, "temperature": 0.2,
        },
        "engineer.standard": {
            "provider": "anthropic", "model": "claude-opus-5", "enabled": True,
            "reliability_prior": {"engineering.small": [8, 1], "engineering.medium": [6, 2], "engineering.large": [2, 3]},
            "max_output_tokens": 8192, "temperature": 0.1,
        },
        "engineer.frontier": {
            "provider": "anthropic", "model": "claude-fable-5-1", "enabled": True,
            "reliability_prior": {"engineering.small": [9, 1], "engineering.medium": [9, 1], "engineering.large": [7, 1]},
            "max_output_tokens": 16384, "temperature": 0.1,
        },
        "speech.stt": {"provider": "gemini", "model": "gemini-2.5-flash", "enabled": False},
        "speech.tts": {"provider": "gemini", "model": "gemini-2.5-flash-preview-tts", "enabled": False},
        "local.fast": {"provider": "ollama", "model": "", "tier": "FAST_LOCAL", "enabled": True, "offline_fallback": True,
                       "reliability_prior": {"knowledge": [2, 3], "semantic": [1, 4], "planning": [1, 6], "composition": [1, 6]}},
        "local.build": {"provider": "ollama", "model": "", "tier": "BUILD_LOCAL", "enabled": True, "offline_fallback": True,
                        "reliability_prior": {"engineering.small": [1, 4], "engineering.medium": [1, 8], "engineering.large": [1, 12]}},
    },
    "providers": {
        "gemini": {
            "kind": "gemini", "base_url": "https://generativelanguage.googleapis.com", "enabled": False,
            "secret": "gemini", "may_train_on_requests": True, "metered": False, "timeout_seconds": 120,
            "pricing": {"gemini-2.5-flash": {"input_per_m": 0.0, "cached_input_per_m": 0.0, "output_per_m": 0.0, "confirmed": True}},
        },
        "openai": {
            "kind": "openai", "base_url": "https://api.openai.com", "enabled": False,
            "secret": "openai", "may_train_on_requests": False, "metered": True, "timeout_seconds": 300,
            "pricing": {"gpt-5.6": {"input_per_m": 1.75, "cached_input_per_m": 0.18, "output_per_m": 14.0, "confirmed": False}},
        },
        "anthropic": {
            "kind": "anthropic", "base_url": "https://api.anthropic.com", "enabled": False,
            "secret": "anthropic", "may_train_on_requests": False, "metered": True, "timeout_seconds": 600,
            "pricing": {
                "claude-opus-5": {"input_per_m": 14.0, "cached_input_per_m": 1.4, "output_per_m": 70.0, "confirmed": False},
                "claude-fable-5-1": {"input_per_m": 28.0, "cached_input_per_m": 2.8, "output_per_m": 140.0, "confirmed": False},
            },
        },
        "ollama": {"kind": "ollama", "base_url": "http://127.0.0.1:11434", "enabled": True, "metered": False, "may_train_on_requests": False},
    },
    "budget": {
        "monthly_hard_cap": 40.0, "daily_hard_cap": 6.0, "per_task_hard_cap": 5.0,
        "reasoning_hard_cap": 10.0, "engineering_hard_cap": 25.0, "provider_hard_caps": {},
        "safety_factor": 1.5, "soft_allocation": {"deep_reasoning": 10.0, "engineering": 25.0, "reserve": 5.0},
    },
}


def _parse(document: dict[str, Any], *, source: str) -> GatewayConfig:
    providers: dict[str, ProviderConfig] = {}
    for name, raw in (document.get("providers") or {}).items():
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in PROVIDER_KINDS:
            raise ValueError(f"provider {name!r}: unsupported kind {kind!r}")
        pricing: dict[str, Pricing] = {}
        for model, price in (raw.get("pricing") or {}).items():
            if isinstance(price, dict):
                pricing[str(model)] = Pricing(
                    input_per_m=float(price.get("input_per_m", 0.0)),
                    cached_input_per_m=float(price.get("cached_input_per_m", 0.0)),
                    output_per_m=float(price.get("output_per_m", 0.0)),
                    confirmed=bool(price.get("confirmed", False)),
                )
        cap = raw.get("monthly_cap_eur")
        providers[str(name)] = ProviderConfig(
            name=str(name), kind=kind, base_url=str(raw.get("base_url", "")), enabled=bool(raw.get("enabled", False)),
            secret=str(raw.get("secret", "")), may_train_on_requests=bool(raw.get("may_train_on_requests", False)),
            metered=bool(raw.get("metered", True)), monthly_cap_eur=float(cap) if cap is not None else None,
            timeout_seconds=float(raw.get("timeout_seconds", 120.0)), pricing=pricing,
            options=dict(raw.get("options") or {}),
        )

    roles: dict[str, RoleBinding] = {}
    for role, raw in (document.get("roles") or {}).items():
        if not isinstance(raw, dict):
            continue
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role {role!r}; roles are {', '.join(ROLE_NAMES)}")
        provider = str(raw.get("provider", ""))
        if provider not in providers:
            raise ValueError(f"role {role!r} is bound to unknown provider {provider!r}")
        prior = {str(k): [float(v[0]), float(v[1])] for k, v in (raw.get("reliability_prior") or {}).items()
                 if isinstance(v, (list, tuple)) and len(v) == 2}
        roles[role] = RoleBinding(
            role=role, provider=provider, model=str(raw.get("model", "")), enabled=bool(raw.get("enabled", True)),
            thinking=dict(raw.get("thinking") or {}), reliability_prior=prior,
            max_output_tokens=int(raw.get("max_output_tokens", 2048)), temperature=float(raw.get("temperature", 0.3)),
            tier=str(raw.get("tier", "")), offline_fallback=bool(raw.get("offline_fallback", False)),
        )

    raw_budget = document.get("budget") or {}
    budget = BudgetConfig(
        monthly_hard_cap=float(raw_budget.get("monthly_hard_cap", 40.0)),
        daily_hard_cap=float(raw_budget.get("daily_hard_cap", 6.0)),
        per_task_hard_cap=float(raw_budget.get("per_task_hard_cap", 5.0)),
        reasoning_hard_cap=float(raw_budget.get("reasoning_hard_cap", 10.0)),
        engineering_hard_cap=float(raw_budget.get("engineering_hard_cap", 25.0)),
        provider_hard_caps={str(k): float(v) for k, v in (raw_budget.get("provider_hard_caps") or {}).items()},
        safety_factor=float(raw_budget.get("safety_factor", 1.5)),
        soft_allocation={str(k): float(v) for k, v in (raw_budget.get("soft_allocation") or {}).items()},
    )
    return GatewayConfig(roles=roles, providers=providers, budget=budget, currency=str(document.get("currency", "EUR")),
                         source=source, risk_aversion=float(document.get("risk_aversion", 1.0)))
