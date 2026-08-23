"""Model tiers, their configuration, and honest availability reporting.

Jarvis never asks for "the model".  It asks for a *tier* -- a role in the
system -- and the catalog decides which concrete model and provider currently
fills that role.  Swapping the GTX 1070 for a 30B model on a dedicated server
is then a configuration change, not a code change, which is the whole point of
:class:`ModelTier`.

The second job of this module is to stop Jarvis from lying about availability.
A reachable server is not an available model: this machine's Ollama answers
``GET /v1/models`` with HTTP 200 while the requested model is not pulled, and
answers a chat request for that model with a 404 buried in a JSON body.  So a
tier is reported ``ONLINE`` only after all three of

1. the provider endpoint responds,
2. the requested model is present in its catalog,
3. a real minimal generation completes,

which is what :class:`ModelProbe` does.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


class ModelTier(str, Enum):
    """The roles a model can fill.  Add roles here, not model names."""

    #: Conversation, routing, short answers.  Must stay responsive.
    FAST_LOCAL = "FAST_LOCAL"
    #: The autonomous software-development model.  Primary workhorse.
    BUILD_LOCAL = "BUILD_LOCAL"
    #: Image understanding (screen capture, diagrams).  Optional.
    VISION_LOCAL = "VISION_LOCAL"
    #: Text embeddings for semantic retrieval.  Optional.
    EMBEDDING_LOCAL = "EMBEDDING_LOCAL"
    #: A stronger model on a machine you control (future home server).
    SELF_HOSTED = "SELF_HOSTED"
    #: A paid third-party API.  Always opt-in, never a default.
    EXPERT_CLOUD = "EXPERT_CLOUD"


class HealthState(str, Enum):
    ONLINE = "ONLINE"
    DISABLED = "DISABLED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    PROVIDER_UNREACHABLE = "PROVIDER_UNREACHABLE"
    MODEL_MISSING = "MODEL_MISSING"
    GENERATION_FAILED = "GENERATION_FAILED"


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to talk to the model currently filling a tier."""

    tier: ModelTier
    provider: str = "ollama"
    model: str = ""
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str = ""
    #: Tokens of context to allocate.  On a small GPU this is the single most
    #: important knob: too large and the model spills to CPU or evicts the
    #: desktop's VRAM; see :mod:`brain.resources`.
    context_window: int = 8192
    max_output_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_seconds: float = 600.0
    #: How long the provider should keep weights resident after a request.
    keep_alive: str = "5m"
    #: Layers to offload to GPU; ``None`` lets the provider decide.
    gpu_layers: int | None = None
    enabled: bool = True
    #: True when using this tier spends money.  Gates it behind explicit opt-in.
    paid: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "tier": self.tier.value,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout_seconds": self.timeout_seconds,
            "keep_alive": self.keep_alive,
            "gpu_layers": self.gpu_layers,
            "enabled": self.enabled,
            "paid": self.paid,
            "options": dict(self.options),
        }
        return data

    @property
    def configured(self) -> bool:
        return bool(self.model and self.base_url)

    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "") if self.api_key_env else ""


def default_catalog() -> dict[ModelTier, ModelSpec]:
    """Defaults tuned for the current target machine (GTX 1070, 8 GB).

    Everything local is on by default; everything paid is off by default.  That
    ordering is deliberate and load-bearing: deleting every cloud credential on
    this machine must leave autonomous development fully working.
    """

    return {
        ModelTier.FAST_LOCAL: ModelSpec(
            tier=ModelTier.FAST_LOCAL,
            provider="ollama",
            model="qwen3:4b-instruct",
            context_window=8192,
            max_output_tokens=768,
            temperature=0.3,
            timeout_seconds=180.0,
            keep_alive="10m",
        ),
        ModelTier.BUILD_LOCAL: ModelSpec(
            tier=ModelTier.BUILD_LOCAL,
            provider="ollama",
            model="qwen2.5-coder:7b-instruct-q4_K_M",
            context_window=8192,
            max_output_tokens=1536,
            temperature=0.1,
            timeout_seconds=900.0,
            keep_alive="15m",
        ),
        ModelTier.VISION_LOCAL: ModelSpec(
            tier=ModelTier.VISION_LOCAL,
            provider="ollama",
            model="",  # not pulled on this machine yet
            enabled=False,
            context_window=4096,
        ),
        ModelTier.EMBEDDING_LOCAL: ModelSpec(
            tier=ModelTier.EMBEDDING_LOCAL,
            provider="ollama",
            model="",
            enabled=False,
            context_window=2048,
        ),
        ModelTier.SELF_HOSTED: ModelSpec(
            tier=ModelTier.SELF_HOSTED,
            provider="openai_compatible",
            model="",
            base_url="",
            enabled=False,
            context_window=32768,
            max_output_tokens=4096,
        ),
        ModelTier.EXPERT_CLOUD: ModelSpec(
            tier=ModelTier.EXPERT_CLOUD,
            provider="openai_compatible",
            model="",
            base_url="",
            api_key_env="JARVIS_EXPERT_CLOUD_API_KEY",
            enabled=False,
            paid=True,
            context_window=32768,
            max_output_tokens=4096,
        ),
    }


_SPEC_FIELDS: dict[str, Any] = {
    "provider": str,
    "model": str,
    "base_url": str,
    "api_key_env": str,
    "context_window": int,
    "max_output_tokens": int,
    "temperature": float,
    "top_p": float,
    "timeout_seconds": float,
    "keep_alive": str,
    "gpu_layers": int,
    "enabled": bool,
    "paid": bool,
}


def _coerce(kind: Any, value: Any) -> Any:
    if kind is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return kind(value)


class ModelCatalog:
    """Resolves each :class:`ModelTier` to a concrete :class:`ModelSpec`.

    Configuration is layered, later layers winning:

    1. :func:`default_catalog` -- sensible values for this machine.
    2. A JSON file, ``JARVIS_MODELS_CONFIG`` or ``<repo>/config/models.json``.
    3. Environment variables, ``JARVIS_<TIER>_<FIELD>``, e.g.
       ``JARVIS_BUILD_LOCAL_MODEL`` or ``JARVIS_FAST_LOCAL_CONTEXT_WINDOW``.

    The env layer exists so a single run can be redirected without editing
    files, which is what the acceptance tests and the "no cloud" drill need.
    """

    def __init__(
        self,
        specs: dict[ModelTier, ModelSpec] | None = None,
        *,
        config_path: str | Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self._specs = dict(specs or default_catalog())
        self._environ = dict(environ if environ is not None else os.environ)
        self._config_path = self._resolve_config_path(config_path)
        if specs is None:
            self._apply_file_layer()
            self._apply_env_layer()

    # -- configuration layers -------------------------------------------

    def _resolve_config_path(self, config_path: str | Path | None) -> Path | None:
        if config_path is not None:
            return Path(config_path)
        configured = self._environ.get("JARVIS_MODELS_CONFIG", "").strip()
        if configured:
            return Path(configured)
        default = Path(__file__).resolve().parent.parent / "config" / "models.json"
        return default if default.exists() else None

    def _apply_file_layer(self) -> None:
        if self._config_path is None or not self._config_path.exists():
            return
        try:
            payload = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid model catalog at {self._config_path}: {exc}") from exc
        for tier_name, overrides in (payload.get("tiers") or {}).items():
            tier = _parse_tier(tier_name)
            if tier is None or not isinstance(overrides, dict):
                continue
            self._specs[tier] = self._merge(self._specs[tier], overrides)

    def _apply_env_layer(self) -> None:
        for tier, spec in list(self._specs.items()):
            overrides: dict[str, Any] = {}
            for field_name in _SPEC_FIELDS:
                key = f"JARVIS_{tier.value}_{field_name.upper()}"
                if key in self._environ and self._environ[key] != "":
                    overrides[field_name] = self._environ[key]
            if overrides:
                self._specs[tier] = self._merge(spec, overrides)

    @staticmethod
    def _merge(spec: ModelSpec, overrides: dict[str, Any]) -> ModelSpec:
        changes: dict[str, Any] = {}
        for name, raw in overrides.items():
            if name == "options" and isinstance(raw, dict):
                changes["options"] = {**spec.options, **raw}
                continue
            kind = _SPEC_FIELDS.get(name)
            if kind is None:
                continue
            if raw is None:
                changes[name] = None
                continue
            try:
                changes[name] = _coerce(kind, raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value for {spec.tier.value}.{name}: {raw!r}") from exc
        # A tier that has been given a model is presumed wanted, unless the
        # override says otherwise; otherwise configuring VISION_LOCAL would
        # silently do nothing because the default ships it disabled.
        if changes.get("model") and "enabled" not in changes:
            changes["enabled"] = True
        return replace(spec, **changes)

    # -- lookup ----------------------------------------------------------

    def get(self, tier: ModelTier | str) -> ModelSpec:
        resolved = tier if isinstance(tier, ModelTier) else _parse_tier(str(tier))
        if resolved is None:
            raise KeyError(f"unknown model tier: {tier}")
        return self._specs[resolved]

    def set(self, tier: ModelTier, spec: ModelSpec) -> None:
        self._specs[tier] = spec

    def tiers(self) -> dict[ModelTier, ModelSpec]:
        return dict(self._specs)

    def enabled_tiers(self) -> dict[ModelTier, ModelSpec]:
        return {tier: spec for tier, spec in self._specs.items() if spec.enabled and spec.configured}

    def paid_tiers_enabled(self) -> list[ModelTier]:
        return [tier for tier, spec in self._specs.items() if spec.paid and spec.enabled]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self._config_path) if self._config_path else None,
            "tiers": {tier.value: spec.to_dict() for tier, spec in sorted(self._specs.items(), key=lambda kv: kv[0].value)},
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self._config_path or (Path(__file__).resolve().parent.parent / "config" / "models.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "tiers": {tier.value: spec.to_dict() for tier, spec in sorted(self._specs.items(), key=lambda kv: kv[0].value)}}
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
        self._config_path = target
        return target


def _parse_tier(name: str) -> ModelTier | None:
    try:
        return ModelTier(name.strip().upper())
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------

@dataclass
class ModelHealth:
    """The result of actually trying to use a tier."""

    tier: ModelTier
    state: HealthState
    model: str = ""
    provider: str = ""
    base_url: str = ""
    detail: str = ""
    latency_seconds: float | None = None
    available_models: list[str] = field(default_factory=list)
    checked_at: float = 0.0

    @property
    def online(self) -> bool:
        return self.state is HealthState.ONLINE

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "state": self.state.value,
            "online": self.online,
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "detail": self.detail,
            "latency_seconds": self.latency_seconds,
            "checked_at": self.checked_at,
        }

    def summary(self) -> str:
        if self.online:
            latency = f" ({self.latency_seconds:.1f}s)" if self.latency_seconds else ""
            return f"{self.model} ONLINE{latency}"
        if self.state is HealthState.DISABLED:
            return "disabled"
        if self.state is HealthState.NOT_CONFIGURED:
            return "not configured"
        return f"{self.model or '(no model)'} {self.state.value}: {self.detail}"


class ModelProbe:
    """Establishes whether a tier can actually do work right now.

    Results are cached for ``ttl_seconds`` because a genuine probe costs a real
    (if tiny) generation, and the status line should not pay for a model load
    every time it is drawn.
    """

    def __init__(self, catalog: ModelCatalog, *, ttl_seconds: float = 60.0, provider_factory=None) -> None:
        self.catalog = catalog
        self.ttl_seconds = ttl_seconds
        self._cache: dict[ModelTier, ModelHealth] = {}
        if provider_factory is None:
            from brain.providers import provider_for_spec as provider_factory  # local import: avoids a cycle
        self._provider_factory = provider_factory

    def probe(self, tier: ModelTier, *, force: bool = False) -> ModelHealth:
        spec = self.catalog.get(tier)
        cached = self._cache.get(tier)
        if cached is not None and not force and (time.time() - cached.checked_at) < self.ttl_seconds:
            return cached
        health = self._probe_uncached(spec)
        self._cache[tier] = health
        return health

    def probe_all(self, *, force: bool = False) -> dict[ModelTier, ModelHealth]:
        return {tier: self.probe(tier, force=force) for tier in self.catalog.tiers()}

    def invalidate(self, tier: ModelTier | None = None) -> None:
        if tier is None:
            self._cache.clear()
        else:
            self._cache.pop(tier, None)

    def _probe_uncached(self, spec: ModelSpec) -> ModelHealth:
        now = time.time()
        base = ModelHealth(
            tier=spec.tier,
            state=HealthState.ONLINE,
            model=spec.model,
            provider=spec.provider,
            base_url=spec.base_url,
            checked_at=now,
        )
        if not spec.enabled:
            return replace(base, state=HealthState.DISABLED, detail="tier is disabled in configuration")
        if not spec.configured:
            return replace(base, state=HealthState.NOT_CONFIGURED, detail="no model or base_url configured")

        try:
            provider = self._provider_factory(spec)
        except Exception as exc:
            return replace(base, state=HealthState.NOT_CONFIGURED, detail=f"{type(exc).__name__}: {exc}")

        # 1. Is the endpoint there at all?
        try:
            available = list(provider.list_models())
        except Exception as exc:
            return replace(base, state=HealthState.PROVIDER_UNREACHABLE, detail=f"{type(exc).__name__}: {exc}")

        # 2. Is *this* model there?  An empty catalog means the provider does
        #    not enumerate models, which is not evidence of absence.
        if available and not _model_present(spec.model, available):
            return replace(
                base,
                state=HealthState.MODEL_MISSING,
                detail=f"model {spec.model!r} is not available on {spec.base_url}",
                available_models=available[:32],
            )

        # 3. Can it actually generate?  This is the only step that proves it.
        started = time.perf_counter()
        try:
            text = provider.generate("Reply with the single word: OK", max_tokens=8, temperature=0.0)
        except Exception as exc:
            return replace(
                base,
                state=HealthState.GENERATION_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
                available_models=available[:32],
                latency_seconds=time.perf_counter() - started,
            )
        latency = time.perf_counter() - started
        if not str(text).strip():
            return replace(
                base,
                state=HealthState.GENERATION_FAILED,
                detail="model returned an empty completion",
                latency_seconds=latency,
                available_models=available[:32],
            )
        return replace(base, state=HealthState.ONLINE, latency_seconds=latency, available_models=available[:32])


def _model_present(model: str, available: list[str]) -> bool:
    """Match a configured model name against a provider's catalog.

    Ollama reports ``qwen3:4b-instruct`` but accepts ``qwen3:4b-instruct`` and
    (for the default tag) ``qwen3``; other servers report bare names.  Matching
    on the tag-stripped stem as well keeps a correct configuration from being
    reported as missing over a naming convention.
    """

    wanted = model.strip()
    if not wanted:
        return False
    names = {str(item).strip() for item in available}
    if wanted in names:
        return True
    stem = wanted.split(":", 1)[0]
    return any(name == stem or name.split(":", 1)[0] == stem for name in names)
