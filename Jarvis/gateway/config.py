"""Roles, providers, prices and caps -- read from configuration, never from code.

``config/providers.json`` is the tracked provider defaults/template. It binds
each abstract role to a provider and model, and carries provider definitions,
model IDs, pricing metadata and non-secret static configuration. Owner-local
state such as provider enablement lives in an ignored overlay below
``data/jarvis/owner`` and is merged at load time.
The four intelligence roles ZEUS's core knows, and their initial bindings:

    reasoning.free      -> google    / a Gemini Flash free-tier model   (zero cost; may train)
    reasoning.deep      -> openai    / a GPT reasoning model             (metered)
    engineer.standard   -> anthropic / an Opus-class engineering model   (metered)
    engineer.frontier   -> anthropic / a Fable-class engineering model   (metered)

plus ``engineer.codex`` (a subscription CLI the expert gateway drives), the
optional ``engineer.frontier_alt`` slot for a second frontier engineering
provider (disabled until the owner fills it in; never an automatic cascade),
the speech roles and the local Ollama tiers (offline fallback only).

Swapping a provider or a model is an edit to that file.  No module outside
``gateway/`` knows a model name, and the tests assert exactly that.

Prices are configuration too, dated: each model carries one or more price
entries with ``effective_from`` / ``effective_until`` and the currency they
were listed in; the entry effective today is what estimation and settlement
use.  Accounting keeps the provider-native listed currency alongside the EUR
budget figure.  The EUR conversion comes only from the owner-configured
``exchange_rates`` table; when no rate is configured, ``rate_to_eur`` is
unavailable and estimates are marked unconfirmed.  The internal budget guard
still reserves against the EUR hard caps so metered calls cannot bypass them,
but it is not reported as an exchange rate.  A metered role without an
effective price cannot be estimated, cannot be reserved and therefore cannot
be called -- the gateway refuses rather than guesses.

Reasoning effort is abstract: FAST, NORMAL, DEEP (and MAX where a provider
offers it).  Each role's ``thinking`` map says what its provider calls that
level; the UI never sees provider terminology.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from gateway.modes import CostClass, RoleFamily

ROLE_NAMES: tuple[str, ...] = (
    "reasoning.free",
    "reasoning.smart",
    "reasoning.deep",
    "engineer.standard",
    "engineer.frontier",
    "engineer.frontier_alt",
    "engineer.codex",
    "speech.stt",
    "speech.tts",
    "local.fast",
    "local.build",
)

#: The roles core code may reason about.  Everything else is plumbing.
INTELLIGENCE_ROLES: tuple[str, ...] = ("reasoning.free", "reasoning.smart", "reasoning.deep", "engineer.standard", "engineer.frontier",
                                       "engineer.frontier_alt")

PROVIDER_KINDS: tuple[str, ...] = ("gemini", "openai", "anthropic", "ollama", "openai_compatible", "subscription_cli")

#: Provider kinds the gateway can call itself.  ``subscription_cli`` (Codex)
#: is an engineer the expert gateway drives; the model gateway only ranks it.
MODEL_KINDS: frozenset[str] = frozenset({"gemini", "openai", "anthropic", "ollama", "openai_compatible"})

#: Abstract reasoning effort levels.  Mapped per role to whatever the provider
#: calls it (a thinking level, a thinking budget, a reasoning effort).
THINKING_LEVELS: tuple[str, ...] = ("FAST", "NORMAL", "DEEP", "MAX")

#: Level names from configuration files written before the abstract names.
_LEGACY_THINKING = {"FREE_LOW": "FAST", "FREE_MEDIUM": "NORMAL", "FREE_HIGH": "DEEP"}

OWNER_PROVIDER_STATE = "providers.local.json"

MUTABLE_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"currency", "risk_aversion", "exchange_rates", "budget"})
MUTABLE_PROVIDER_KEYS: frozenset[str] = frozenset({"enabled", "monthly_cap_eur", "timeout_seconds", "options"})
MUTABLE_ROLE_KEYS: frozenset[str] = frozenset({
    "enabled", "provider", "model", "models", "thinking", "reliability_prior",
    "max_output_tokens", "temperature", "tier", "offline_fallback",
})


def _today() -> str:
    return os.environ.get("ZEUS_PRICING_TODAY", "").strip() or date.today().isoformat()


@dataclass(frozen=True)
class ExchangeRate:
    """EUR per unit of a listed currency, set by the owner."""

    currency: str
    eur_per_unit: float
    as_of: str = ""
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"eur_per_unit": self.eur_per_unit, "as_of": self.as_of, "confirmed": self.confirmed}


#: What the EUR view of a price rests on.
RATE_EUR = "eur"                       # listed in EUR: no conversion
RATE_CONFIGURED = "configured"         # the owner's exchange_rates entry
RATE_NOT_NEEDED = "not-needed"         # zero native price: no conversion is needed
RATE_UNAVAILABLE = "unavailable"       # no owner-configured conversion rate


@dataclass(frozen=True)
class Pricing:
    """Provider price plus the EUR budget view.

    ``input_per_m`` / ``cached_input_per_m`` / ``output_per_m`` are EUR and are
    what estimation, reservation and settlement use.  If no exchange rate is
    configured for a non-EUR price, those fields are an unconfirmed budget
    guard; ``rate_to_eur`` is then ``None`` and ``listed`` remains the
    provider-native source of truth.
    """

    input_per_m: float = 0.0
    cached_input_per_m: float = 0.0
    output_per_m: float = 0.0
    #: False until the owner has checked these against the provider's price
    #: list.  Estimates from unconfirmed prices are labelled as such in the UI.
    confirmed: bool = False
    currency: str = "EUR"
    listed: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rate_to_eur: float | None = 1.0
    budget_rate_to_eur: float = 1.0
    rate_source: str = RATE_EUR
    eur_conversion_available: bool = True
    eur_conversion_confirmed: bool = True
    effective_from: str = ""
    effective_until: str = ""
    source: str = ""

    @property
    def metered(self) -> bool:
        return any(value > 0 for value in self.listed)

    @property
    def estimate_confirmed(self) -> bool:
        return self.confirmed and self.eur_conversion_confirmed

    def effective_on(self, day: str) -> bool:
        if self.effective_from and day < self.effective_from:
            return False
        if self.effective_until and day > self.effective_until:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """The configuration form: the listed figures, dated."""

        out: dict[str, Any] = {"input_per_m": self.listed[0], "cached_input_per_m": self.listed[1], "output_per_m": self.listed[2],
                               "currency": self.currency, "confirmed": self.confirmed}
        if self.effective_from:
            out["effective_from"] = self.effective_from
        if self.effective_until:
            out["effective_until"] = self.effective_until
        if self.source:
            out["source"] = self.source
        return out

    def native_cost(self, usage: dict[str, Any]) -> float:
        """What the provider bills, in the currency it lists."""

        fresh = max(0, int(usage.get("input_tokens", 0) or 0) - int(usage.get("cached_input_tokens", 0) or 0))
        return round((fresh * self.listed[0] + int(usage.get("cached_input_tokens", 0) or 0) * self.listed[1]
                      + int(usage.get("output_tokens", 0) or 0) * self.listed[2]) / 1_000_000, 6)

    def to_eur_dict(self) -> dict[str, Any]:
        """The UI form: EUR per million, with the conversion that produced it."""

        has_eur = self.eur_conversion_available
        return {"input_per_m_eur": round(self.input_per_m, 6) if has_eur else None,
                "cached_input_per_m_eur": round(self.cached_input_per_m, 6) if has_eur else None,
                "output_per_m_eur": round(self.output_per_m, 6) if has_eur else None,
                "budget_input_per_m_eur": round(self.input_per_m, 6),
                "budget_cached_input_per_m_eur": round(self.cached_input_per_m, 6),
                "budget_output_per_m_eur": round(self.output_per_m, 6),
                "currency": self.currency, "rate_to_eur": self.rate_to_eur, "budget_rate_to_eur": self.budget_rate_to_eur,
                "rate_source": self.rate_source, "listed": list(self.listed), "confirmed": self.confirmed,
                "pricing_confirmed": self.estimate_confirmed, "eur_conversion_available": self.eur_conversion_available,
                "eur_conversion_confirmed": self.eur_conversion_confirmed,
                "effective_from": self.effective_from, "effective_until": self.effective_until, "source": self.source}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str
    base_url: str = ""
    enabled: bool = False
    #: Secret slot in :class:`gateway.secrets.CredentialStore`; empty for local providers.
    secret: str = ""
    #: Environment variable names whose value is imported into the secret slot
    #: (once, into the encrypted store) when the slot is empty.
    credential_env: tuple[str, ...] = ()
    #: True when the provider's terms let it use request content to improve
    #: its products.  Such a provider never receives private context.
    may_train_on_requests: bool = False
    #: Whether requests are billed at all.  A free tier is ``False``.
    metered: bool = True
    #: Monthly ceiling for this provider alone, EUR.  ``None`` = governor default.
    monthly_cap_eur: float | None = None
    timeout_seconds: float = 120.0
    #: Model name -> the pricing effective today.  Missing model = cannot be estimated.
    pricing: dict[str, Pricing] = field(default_factory=dict)
    #: Model name -> every dated price entry as configured (history included).
    pricing_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: Optional per-model settings (e.g. thinking budgets), passed through.
    options: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    def price_for(self, model: str) -> Pricing | None:
        return self.pricing.get(model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "base_url": self.base_url, "enabled": self.enabled,
            "secret": self.secret, "credential_env": list(self.credential_env),
            "may_train_on_requests": self.may_train_on_requests, "metered": self.metered,
            "monthly_cap_eur": self.monthly_cap_eur, "timeout_seconds": self.timeout_seconds,
            "pricing": {model: list(entries) for model, entries in self.pricing_history.items()},
            "options": dict(self.options), "purpose": self.purpose,
        }


@dataclass(frozen=True)
class RoleBinding:
    role: str
    provider: str
    model: str
    #: Ordered concrete model pool for this role. ``model`` is the first entry
    #: kept for compatibility with older configuration and diagnostics.
    models: tuple[str, ...] = ()
    enabled: bool = True
    #: Reasoning effort configuration: abstract level name -> provider-specific value.
    thinking: dict[str, Any] = field(default_factory=dict)
    #: Prior belief in this role's reliability per task class, before any
    #: observation: {task_class: [successes, failures]} pseudo-counts.
    reliability_prior: dict[str, list[float]] = field(default_factory=dict)
    max_output_tokens: int = 2048
    temperature: float = 0.3
    #: For local roles: which :class:`brain.tiers.ModelTier` serves it.
    tier: str = ""
    offline_fallback: bool = False
    purpose: str = ""

    @property
    def family(self) -> RoleFamily:
        return RoleFamily(self.role.split(".", 1)[0])

    def thinking_for(self, level: str) -> Any:
        """The provider value for an abstract level; MAX falls back to DEEP, DEEP to NORMAL."""

        order = list(THINKING_LEVELS)
        if level not in order:
            return None
        for candidate in reversed(order[: order.index(level) + 1]):
            if candidate in self.thinking:
                return self.thinking[candidate]
        return None

    @property
    def model_pool(self) -> tuple[str, ...]:
        models = tuple(model for model in self.models if str(model).strip())
        if models:
            return models
        return (self.model,) if self.model else ()

    def to_dict(self) -> dict[str, Any]:
        data = {"role": self.role, "provider": self.provider, "model": self.model, "enabled": self.enabled,
                "thinking": dict(self.thinking), "reliability_prior": {k: list(v) for k, v in self.reliability_prior.items()},
                "max_output_tokens": self.max_output_tokens, "temperature": self.temperature, "tier": self.tier,
                "offline_fallback": self.offline_fallback, "purpose": self.purpose}
        if len(self.model_pool) > 1:
            data["models"] = list(self.model_pool)
        return data


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
    exchange_rates: dict[str, ExchangeRate] = field(default_factory=dict)

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
        return self.pricing_for_model(role, binding.model)

    def pricing_for_model(self, role: str, model: str) -> Pricing | None:
        provider = self.provider_for(role)
        binding = self.roles.get(role)
        if provider is None or binding is None:
            return None
        if not provider.metered:
            return provider.price_for(model) or Pricing(0.0, 0.0, 0.0, confirmed=True)
        return provider.price_for(model)

    def is_model_role(self, role: str) -> bool:
        provider = self.provider_for(role)
        return provider is not None and provider.kind in MODEL_KINDS

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
            "exchange_rates": {code: rate.to_dict() for code, rate in self.exchange_rates.items()},
            "roles": {role: binding.to_dict() for role, binding in self.roles.items()},
            "providers": {name: provider.to_dict() for name, provider in self.providers.items()},
            "budget": self.budget.to_dict(),
        }

    # -- construction -----------------------------------------------------

    @classmethod
    def defaults(cls) -> "GatewayConfig":
        return _parse(DEFAULT_DOCUMENT, source="defaults")

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        override_path: str | Path | None = None,
        migrate_owner_state: bool = True,
    ) -> "GatewayConfig":
        target = Path(path) if path else default_path()
        document = _load_defaults_document(target)
        source = str(target) if target.is_file() else "defaults"

        overlay = Path(override_path) if override_path is not None else (owner_override_path() if path is None else None)
        if overlay is not None:
            if migrate_owner_state:
                _migrate_owner_state(document, overlay)
            owner = _load_owner_override(overlay, document)
            if _has_owner_override(owner):
                document = _deep_merge(document, owner)
                source = f"{source} + {overlay}"
        return _parse(document, source=source)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 2, **self.to_dict()}
        document.pop("source", None)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return target

    def save_owner_override(self, defaults_path: str | Path | None = None, override_path: str | Path | None = None) -> Path:
        defaults = _parse(_load_defaults_document(Path(defaults_path) if defaults_path else default_path()), source="defaults")
        document = _owner_override_document(self, defaults)
        return _write_owner_override(Path(override_path) if override_path is not None else owner_override_path(), document)

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
            changes["models"] = (str(model),) if str(model) else ()
        if enabled is not None:
            changes["enabled"] = bool(enabled)
        roles = dict(self.roles)
        roles[role] = replace(binding, **changes)
        return replace(self, roles=roles)

    def provider_for_secret(self, slot: str) -> ProviderConfig | None:
        for provider in self.providers.values():
            if provider.secret == slot:
                return provider
        return None


def default_path() -> Path:
    configured = os.environ.get("ZEUS_PROVIDERS_CONFIG", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "config" / "providers.json"


def owner_override_path(state_root: str | Path | None = None) -> Path:
    configured = os.environ.get("ZEUS_PROVIDERS_OVERRIDE", "").strip()
    if configured:
        return Path(configured)
    if state_root is not None:
        return Path(state_root) / "owner" / OWNER_PROVIDER_STATE
    return Path(__file__).resolve().parent.parent / "data" / "jarvis" / "owner" / OWNER_PROVIDER_STATE


def _load_defaults_document(target: Path) -> dict[str, Any]:
    if not target.is_file():
        return json.loads(json.dumps(DEFAULT_DOCUMENT))
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid gateway configuration at {target}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid gateway configuration at {target}: not an object")
    return document


def _load_owner_override(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid gateway owner override at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid gateway owner override at {path}: not an object")
    return _sanitize_owner_override(document, defaults)


def _sanitize_owner_override(document: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Keep owner state local and non-secret, even if a hand-edited overlay drifts."""

    out: dict[str, Any] = {"schema_version": 2}
    for key in MUTABLE_TOP_LEVEL_KEYS:
        if key in document:
            out[key] = document[key]

    default_providers = set((defaults.get("providers") or {}).keys())
    providers: dict[str, Any] = {}
    for name, raw in (document.get("providers") or {}).items():
        if name not in default_providers or not isinstance(raw, dict):
            continue
        kept = {key: value for key, value in raw.items() if key in MUTABLE_PROVIDER_KEYS}
        if kept:
            providers[str(name)] = kept
    if providers:
        out["providers"] = providers

    default_roles = set((defaults.get("roles") or {}).keys())
    roles: dict[str, Any] = {}
    for name, raw in (document.get("roles") or {}).items():
        if name not in default_roles or not isinstance(raw, dict):
            continue
        kept = {key: value for key, value in raw.items() if key in MUTABLE_ROLE_KEYS}
        if kept:
            roles[str(name)] = kept
    if roles:
        out["roles"] = roles
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if key == "schema_version":
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


_NO_DIFF = object()


def _deep_diff(current: Any, default: Any) -> Any:
    if isinstance(current, dict) and isinstance(default, dict):
        out = {}
        for key, value in current.items():
            diff = _deep_diff(value, default.get(key, _NO_DIFF))
            if diff is not _NO_DIFF:
                out[key] = diff
        return out if out else _NO_DIFF
    return current if current != default else _NO_DIFF


def _owner_override_document(config: GatewayConfig, defaults: GatewayConfig) -> dict[str, Any]:
    current = config.to_dict()
    base = defaults.to_dict()
    out: dict[str, Any] = {"schema_version": 2}

    for key in MUTABLE_TOP_LEVEL_KEYS:
        diff = _deep_diff(current.get(key), base.get(key))
        if diff is not _NO_DIFF:
            out[key] = diff

    providers: dict[str, Any] = {}
    for name, raw in current.get("providers", {}).items():
        default_raw = base.get("providers", {}).get(name)
        if not isinstance(raw, dict) or not isinstance(default_raw, dict):
            continue
        kept = {key: raw[key] for key in MUTABLE_PROVIDER_KEYS if key in raw and raw.get(key) != default_raw.get(key)}
        if kept:
            providers[str(name)] = kept
    if providers:
        out["providers"] = providers

    roles: dict[str, Any] = {}
    for name, raw in current.get("roles", {}).items():
        default_raw = base.get("roles", {}).get(name)
        if not isinstance(raw, dict) or not isinstance(default_raw, dict):
            continue
        kept = {key: raw[key] for key in MUTABLE_ROLE_KEYS if key in raw and raw.get(key) != default_raw.get(key)}
        if kept:
            roles[str(name)] = kept
    if roles:
        out["roles"] = roles
    return out


def _has_owner_override(document: dict[str, Any]) -> bool:
    return any(key != "schema_version" for key in document)


def _write_owner_override(path: Path, document: dict[str, Any]) -> Path:
    if not _has_owner_override(document):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _migrate_owner_state(document: dict[str, Any], override: Path) -> None:
    if override.is_file():
        return
    try:
        current = _parse(document, source="migration")
        defaults = GatewayConfig.defaults()
        owner = _owner_override_document(current, defaults)
    except Exception:  # noqa: BLE001 - loading should not fail because migration could not classify an old file
        return
    if _has_owner_override(owner):
        _write_owner_override(override, owner)


# ---------------------------------------------------------------------------
# The shipped defaults.  Every cloud provider is disabled until the owner
# enters a credential (entering one enables it); the local roles are on
# because they cost nothing but electricity.  Prices are dated entries in the
# currency the provider lists; unconfirmed ones are labelled in the UI.
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENT: dict[str, Any] = {
    "schema_version": 2,
    "currency": "EUR",
    "risk_aversion": 1.0,
    # EUR per unit of a listed currency, e.g. "USD": {"eur_per_unit": 0.9,
    # "as_of": "2026-09-14", "confirmed": true}.  Empty: no EUR conversion is
    # reported for foreign-currency prices.
    "exchange_rates": {},
    "roles": {
        "reasoning.free": {
            "provider": "gemini", "model": "gemini-3.8-flash", "models": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
            "enabled": True,
            "thinking": {"FAST": "low", "NORMAL": "medium", "DEEP": "high"},
            "reliability_prior": {"knowledge": [8, 1], "semantic": [6, 2], "planning": [3, 3], "composition": [2, 3]},
            "max_output_tokens": 2048, "temperature": 0.3,
            "purpose": "public/non-sensitive knowledge, semantic interpretation, context understanding, capability selection and composition",
        },
        "reasoning.smart": {
            "provider": "openai", "model": "gpt-5.6-terra", "enabled": True,
            "thinking": {"FAST": "low", "NORMAL": "medium", "DEEP": "high"},
            "reliability_prior": {"knowledge": [8, 1], "semantic": [8, 1], "planning": [7, 1], "composition": [6, 1]},
            "max_output_tokens": 4096, "temperature": 0.2,
            "purpose": "inexpensive cloud reasoning: everyday interpretation, planning and explanation when the free tier is not enough",
        },
        "reasoning.deep": {
            "provider": "openai", "model": "gpt-5.6-sol", "enabled": True,
            "thinking": {"FAST": "low", "NORMAL": "medium", "DEEP": "high", "MAX": "xhigh"},
            "reliability_prior": {"knowledge": [9, 1], "semantic": [9, 1], "planning": [8, 1], "composition": [7, 1]},
            "max_output_tokens": 4096, "temperature": 0.2,
            "purpose": "difficult reasoning and contextual interpretation, long-horizon planning, complex composition, difficult EngineeringSpecs",
        },
        "engineer.standard": {
            "provider": "anthropic", "model": "claude-opus-5", "enabled": True,
            "thinking": {"FAST": 2048, "NORMAL": 8192, "DEEP": 16384},
            "reliability_prior": {"engineering.small": [8, 1], "engineering.medium": [6, 2], "engineering.large": [2, 3]},
            "max_output_tokens": 8192, "temperature": 0.1,
            "purpose": "serious normal software engineering: isolated capabilities and plugins, bugs needing code changes, medium integrations",
        },
        "engineer.frontier": {
            "provider": "anthropic", "model": "claude-fable-5-1", "enabled": True,
            "thinking": {"FAST": 4096, "NORMAL": 16384, "DEEP": 32768},
            "reliability_prior": {"engineering.small": [9, 1], "engineering.medium": [9, 1], "engineering.large": [7, 1]},
            "max_output_tokens": 16384, "temperature": 0.1,
            "purpose": "major new subsystems, core architecture, multi-module integrations, long autonomous engineering jobs",
        },
        "engineer.frontier_alt": {
            # A second frontier engineer: the owner's Codex subscription with
            # an Astra-class model, when their allowance offers one.  The model
            # and the switch live in the owner override, never here; the paid
            # API is not a fallback for this role.  Chosen only when the
            # engineering router names it; never a cascade.
            "provider": "codex", "model": "", "enabled": False,
            "reliability_prior": {"engineering.small": [9, 1], "engineering.medium": [9, 1], "engineering.large": [7, 1]},
            "max_output_tokens": 16384, "temperature": 0.1,
            "purpose": "optional second frontier engineer through the owner's Codex subscription (Astra-class model); no API fallback",
        },
        "engineer.codex": {
            "provider": "codex", "model": "codex-cli", "enabled": True,
            "reliability_prior": {"engineering.small": [7, 1], "engineering.medium": [6, 2], "engineering.large": [2, 4]},
        },
        "speech.stt": {"provider": "gemini", "model": "gemini-3.8-flash", "enabled": False},
        "speech.tts": {"provider": "gemini", "model": "gemini-2.5-flash-preview-tts", "enabled": False},
        "local.fast": {"provider": "ollama", "model": "", "tier": "FAST_LOCAL", "enabled": True, "offline_fallback": True,
                       "reliability_prior": {"knowledge": [2, 3], "semantic": [1, 4], "planning": [1, 6], "composition": [1, 6]}},
        "local.build": {"provider": "ollama", "model": "", "tier": "BUILD_LOCAL", "enabled": True, "offline_fallback": True,
                        "reliability_prior": {"engineering.small": [1, 4], "engineering.medium": [1, 8], "engineering.large": [1, 12]}},
    },
    "providers": {
        "gemini": {
            "kind": "gemini", "base_url": "https://generativelanguage.googleapis.com", "enabled": False,
            "secret": "gemini", "credential_env": ["GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"],
            "may_train_on_requests": True, "metered": False, "timeout_seconds": 120,
            "options": {
                "output_hard_limit": 65536, "vendor": "google",
                # Google's free-tier day ends at Pacific midnight.
                "daily_quota_reset_zone": "America/Los_Angeles",
                # The zero-cost pool's evidence: a route is used only with
                # verified_zero_cost and a dated source.  The owner verified
                # the Gemini API free tier (token price zero) on 2026-09-14.
                # ``priority`` is the pool order across every provider, 1 first.
                "zero_cost_routes": [
                    {"model": "gemini-3.8-flash", "priority": 1, "verified_zero_cost": True, "verified_at": "2026-09-14",
                     "verified_by": "owner: Gemini API free tier, token price zero", "supports_stream": True,
                     "supports_structured_output": True, "context_limit": 1048576, "output_limit": 65536},
                    {"model": "gemini-3.7-flash", "priority": 2, "verified_zero_cost": True, "verified_at": "2026-09-14",
                     "verified_by": "owner: Gemini API free tier, token price zero", "supports_stream": True,
                     "supports_structured_output": True, "context_limit": 1048576, "output_limit": 65536},
                    {"model": "gemini-3.6-flash", "priority": 3, "verified_zero_cost": True, "verified_at": "2026-09-15",
                     "verified_by": "owner: Gemini API free tier, token price zero (model availability unverified)", "supports_stream": True,
                     "supports_structured_output": True, "context_limit": 1048576, "output_limit": 65536},
                ],
            },
            "purpose": "free-tier reasoning; quotas and rate limits are the resource constraint, not money",
            "pricing": {
                "gemini-3.8-flash": [{"input_per_m": 0.0, "cached_input_per_m": 0.0, "output_per_m": 0.0, "currency": "USD",
                                      "effective_from": "2026-09-14", "confirmed": True, "source": "Gemini API free tier: model token price zero"}],
                "gemini-3.7-flash": [{"input_per_m": 0.0, "cached_input_per_m": 0.0, "output_per_m": 0.0, "currency": "USD",
                                      "effective_from": "2026-09-14", "confirmed": True, "source": "Gemini API free tier: model token price zero"}],
                "gemini-3.6-flash": [{"input_per_m": 0.0, "cached_input_per_m": 0.0, "output_per_m": 0.0, "currency": "USD",
                                      "effective_from": "2026-09-15", "confirmed": True, "source": "Gemini API free tier: model token price zero"}],
            },
        },
        # Independent zero-cost slots.  Disabled until the owner enters a key
        # (in the encrypted store, never here), and their routes stay
        # "unverified" -- listed, never used -- until the owner records that
        # the effective monetary cost for their account is zero.  Model
        # names belong here, in configuration, never in business logic.
        "groq": {
            "kind": "openai_compatible", "base_url": "https://api.groq.com/openai", "enabled": False,
            "secret": "groq", "credential_env": ["GROQ_API_KEY"],
            "may_train_on_requests": False, "metered": False, "timeout_seconds": 60,
            "options": {"output_hard_limit": 32768, "vendor": "groq", "daily_quota_reset_zone": "UTC",
                        "zero_cost_routes": [{"model": "openai/gpt-oss-120b", "priority": 4, "verified_zero_cost": True, "verified_at": "2026-09-16",
                                              "verified_by": "owner: Groq console Current Plan Free, price $0, pay-per-token Developer tier not active; live calls 2026-09-16 (data/acceptance_evidence/live_zero_cost_groq.json)",
                                              "supports_stream": True, "supports_structured_output": True, "context_limit": 131072,
                                              "output_limit": 32768, "note": "first independent route after Google"}]},
            "purpose": "independent zero-cost reasoning route (when verified at zero cost for the owner's account)",
            "pricing": {},
        },
        "cerebras": {
            "kind": "openai_compatible", "base_url": "https://api.cerebras.ai", "enabled": False,
            "secret": "cerebras", "credential_env": ["CEREBRAS_API_KEY"],
            "may_train_on_requests": False, "metered": False, "timeout_seconds": 60,
            "options": {"output_hard_limit": 32768, "vendor": "cerebras", "daily_quota_reset_zone": "UTC",
                        "zero_cost_routes": [{"model": "gpt-oss-120b", "priority": 6, "verified_zero_cost": False, "verified_at": "", "verified_by": "",
                                              "supports_stream": True, "supports_structured_output": True,
                                              "note": "NOT zero cost for this account: HTTP 402 Payment Required for gpt-oss-120b and qwen-3.8-27b on 2026-09-16 (data/acceptance_evidence/live_zero_cost_cerebras_*.json); provider stays disabled"}]},
            "purpose": "independent zero-cost reasoning route (model chosen in configuration when verified at zero cost)",
            "pricing": {},
        },
        "openrouter": {
            "kind": "openai_compatible", "base_url": "https://openrouter.ai/api", "enabled": False,
            "secret": "openrouter", "credential_env": ["OPENROUTER_API_KEY"],
            "may_train_on_requests": True, "metered": False, "timeout_seconds": 60,
            "options": {"output_hard_limit": 16384, "vendor": "openrouter", "daily_quota_reset_zone": "UTC",
                        "zero_cost_routes": [{"model": "openrouter/free", "priority": 5, "verified_zero_cost": True, "verified_at": "2026-09-16",
                                              "verified_by": "owner account: 0 credits, key credit limit 0, is_free_tier; live generation total_cost 0 (data/acceptance_evidence/live_zero_cost_openrouter.json)",
                                              "supports_stream": True, "supports_structured_output": False, "context_limit": 32768,
                                              "output_limit": 16384, "note": "dynamic free-model router: the model that served is recorded as served_model"}]},
            "purpose": "dynamic free-model pool of last resort (when verified at zero cost for the owner's account)",
            "pricing": {},
        },
        "gemini_paid": {
            # The paid tier of the same API and the same key: a metered
            # provider in its own right.  No role is bound to it until the
            # owner binds one; FREE mode can never reach it (metered), and
            # nothing routes here because the free tier ran out.
            "kind": "gemini", "base_url": "https://generativelanguage.googleapis.com", "enabled": False,
            "secret": "gemini", "credential_env": ["GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"],
            "may_train_on_requests": False, "metered": True, "timeout_seconds": 120,
            "options": {"output_hard_limit": 65536},
            "purpose": "paid-tier Gemini, only when the owner binds a role to it; never a silent continuation of the free tier",
            "pricing": {
                "gemini-3.8-flash": [{"input_per_m": 0.30, "cached_input_per_m": 0.03, "output_per_m": 2.50, "currency": "USD",
                                      "effective_from": "2026-09-14", "confirmed": False,
                                      "source": "paid-tier list price of the previous Flash generation; owner to verify for 3.8 Flash before binding a role"}],
            },
        },
        "openai": {
            "kind": "openai", "base_url": "https://api.openai.com", "enabled": False,
            "secret": "openai", "credential_env": ["OPENAI_API_KEY"],
            "may_train_on_requests": False, "metered": True, "timeout_seconds": 300,
            "options": {"output_hard_limit": 128000},
            "purpose": "smart and deep reasoning",
            "pricing": {
                "gpt-5.6-sol": [{"input_per_m": 4.0, "cached_input_per_m": 0.40, "output_per_m": 20.0, "currency": "USD",
                                 "effective_from": "2026-09-14", "confirmed": True, "source": "OpenAI standard pricing, owner-verified 2026-09-14"}],
                # Terra's list price was not provided: charged at Sol's price
                # until the owner confirms it -- an upper bound, never a guess
                # below the truth.  Once set lower, AUTO prefers Terra on cost.
                "gpt-5.6-terra": [{"input_per_m": 4.0, "cached_input_per_m": 0.40, "output_per_m": 20.0, "currency": "USD",
                                   "effective_from": "2026-09-15", "confirmed": False,
                                   "source": "placeholder = Sol list price as a conservative bound; owner to enter Terra's price"}],
            },
        },
        "anthropic": {
            "kind": "anthropic", "base_url": "https://api.anthropic.com", "enabled": False,
            "secret": "anthropic", "credential_env": ["ANTHROPIC_API_KEY"],
            "may_train_on_requests": False, "metered": True, "timeout_seconds": 600,
            "options": {"output_hard_limit": 32000},
            "purpose": "engineering",
            "pricing": {
                # Cache-read prices were not provided: cached input is charged
                # at the full input price until the owner enters the lower one
                # (conservative; the estimate can only be high).
                "claude-opus-5": [{"input_per_m": 5.0, "cached_input_per_m": 5.0, "output_per_m": 25.0, "currency": "USD",
                                   "effective_from": "2026-09-14", "confirmed": True,
                                   "source": "owner-verified 2026-09-14 (input/output); cached input assumed = input"}],
                "claude-fable-5-1": [{"input_per_m": 10.0, "cached_input_per_m": 10.0, "output_per_m": 50.0, "currency": "USD",
                                      "effective_from": "2026-09-14", "confirmed": True,
                                      "source": "owner-verified 2026-09-14 (input/output); cached input assumed = input"}],
            },
        },
        "frontier_alt": {
            "kind": "openai_compatible", "base_url": "", "enabled": False, "secret": "", "credential_env": [],
            "may_train_on_requests": False, "metered": True, "timeout_seconds": 600,
            "purpose": "optional second frontier engineering provider (fill in kind, base_url, secret, model and a dated price)",
            "pricing": {},
        },
        "ollama": {"kind": "ollama", "base_url": "http://127.0.0.1:11434", "enabled": True, "metered": False, "may_train_on_requests": False},
        "codex": {"kind": "subscription_cli", "base_url": "", "enabled": True, "metered": False, "may_train_on_requests": False},
    },
    "budget": {
        "monthly_hard_cap": 40.0, "daily_hard_cap": 6.0, "per_task_hard_cap": 5.0,
        "reasoning_hard_cap": 10.0, "engineering_hard_cap": 25.0, "provider_hard_caps": {},
        "safety_factor": 1.5, "soft_allocation": {"deep_reasoning": 10.0, "engineering": 25.0, "reserve": 5.0},
    },
}


def _parse_pricing(model: str, raw: Any, rates: dict[str, ExchangeRate], *, provider: str) -> tuple[Pricing | None, list[dict[str, Any]]]:
    """One model's price entries -> (the entry effective today in EUR, the raw history)."""

    entries = raw if isinstance(raw, list) else [raw]
    history: list[dict[str, Any]] = []
    parsed: list[Pricing] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        history.append(dict(entry))
        currency = str(entry.get("currency", "EUR") or "EUR").upper()
        listed = (float(entry.get("input_per_m", 0.0)), float(entry.get("cached_input_per_m", 0.0)), float(entry.get("output_per_m", 0.0)))
        zero_price = not any(listed)
        if currency == "EUR":
            rate_to_eur, budget_rate, rate_source, eur_available, eur_confirmed = 1.0, 1.0, RATE_EUR, True, True
        elif zero_price:
            rate_to_eur, budget_rate, rate_source, eur_available, eur_confirmed = None, 1.0, RATE_NOT_NEEDED, True, True
        else:
            fx = rates.get(currency)
            if fx is not None and fx.eur_per_unit > 0:
                rate_to_eur = float(fx.eur_per_unit)
                budget_rate, rate_source, eur_available, eur_confirmed = rate_to_eur, RATE_CONFIGURED, True, bool(fx.confirmed)
            else:
                # No market rate is guessed.  The budget guard keeps the hard
                # cap engaged, while the EUR conversion remains unavailable.
                rate_to_eur, budget_rate, rate_source, eur_available, eur_confirmed = None, 1.0, RATE_UNAVAILABLE, False, False
        parsed.append(Pricing(
            input_per_m=round(listed[0] * budget_rate, 6), cached_input_per_m=round(listed[1] * budget_rate, 6),
            output_per_m=round(listed[2] * budget_rate, 6), confirmed=bool(entry.get("confirmed", False)),
            currency=currency, listed=listed, rate_to_eur=rate_to_eur, budget_rate_to_eur=budget_rate, rate_source=rate_source,
            eur_conversion_available=eur_available, eur_conversion_confirmed=eur_confirmed,
            effective_from=str(entry.get("effective_from", "") or ""), effective_until=str(entry.get("effective_until", "") or ""),
            source=str(entry.get("source", "") or ""),
        ))
    today = _today()
    effective = [p for p in parsed if p.effective_on(today)]
    if not effective:
        return None, history
    effective.sort(key=lambda p: p.effective_from)
    return effective[-1], history


def _parse(document: dict[str, Any], *, source: str) -> GatewayConfig:
    rates: dict[str, ExchangeRate] = {}
    for code, raw in (document.get("exchange_rates") or {}).items():
        if isinstance(raw, dict):
            rates[str(code).upper()] = ExchangeRate(currency=str(code).upper(), eur_per_unit=float(raw.get("eur_per_unit", 0.0)),
                                                    as_of=str(raw.get("as_of", "") or ""), confirmed=bool(raw.get("confirmed", False)))
        elif isinstance(raw, (int, float)):
            rates[str(code).upper()] = ExchangeRate(currency=str(code).upper(), eur_per_unit=float(raw))

    providers: dict[str, ProviderConfig] = {}
    for name, raw in (document.get("providers") or {}).items():
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in PROVIDER_KINDS:
            raise ValueError(f"provider {name!r}: unsupported kind {kind!r}")
        pricing: dict[str, Pricing] = {}
        history: dict[str, list[dict[str, Any]]] = {}
        for model, price in (raw.get("pricing") or {}).items():
            effective, entries = _parse_pricing(str(model), price, rates, provider=str(name))
            history[str(model)] = entries
            if effective is not None:
                pricing[str(model)] = effective
        cap = raw.get("monthly_cap_eur")
        env = raw.get("credential_env")
        providers[str(name)] = ProviderConfig(
            name=str(name), kind=kind, base_url=str(raw.get("base_url", "")), enabled=bool(raw.get("enabled", False)),
            secret=str(raw.get("secret", "")), credential_env=tuple(str(v) for v in (env if isinstance(env, list) else [env] if env else [])),
            may_train_on_requests=bool(raw.get("may_train_on_requests", False)),
            metered=bool(raw.get("metered", True)), monthly_cap_eur=float(cap) if cap is not None else None,
            timeout_seconds=float(raw.get("timeout_seconds", 120.0)), pricing=pricing, pricing_history=history,
            options=dict(raw.get("options") or {}), purpose=str(raw.get("purpose", "") or ""),
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
        thinking = {_LEGACY_THINKING.get(str(k), str(k)): v for k, v in (raw.get("thinking") or {}).items()}
        raw_models = raw.get("models")
        models = tuple(str(v).strip() for v in raw_models if str(v).strip()) if isinstance(raw_models, list) else ()
        model = str(raw.get("model", "") or (models[0] if models else ""))
        if model and not models:
            models = (model,)
        elif model and model not in models:
            models = (model, *models)
        roles[role] = RoleBinding(
            role=role, provider=provider, model=model, models=models, enabled=bool(raw.get("enabled", True)),
            thinking=thinking, reliability_prior=prior,
            max_output_tokens=int(raw.get("max_output_tokens", 2048)), temperature=float(raw.get("temperature", 0.3)),
            tier=str(raw.get("tier", "")), offline_fallback=bool(raw.get("offline_fallback", False)),
            purpose=str(raw.get("purpose", "") or ""),
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
                         source=source, risk_aversion=float(document.get("risk_aversion", 1.0)), exchange_rates=rates)
