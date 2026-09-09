"""The gateway itself: one call, every rule applied, in order.

    request = GatewayRequest(prompt, chunks, mode, facts, ...)
    reply   = gateway.complete(request)

1. Task vector from ZEUS's facts (rules; a semantic estimate may raise soft
   features later).
2. Privacy decision over the context chunks.
3. Route decision: hard overrides, then cheapest-reliable-enough.
4. For a metered route: estimate, reserve.  Refused reservation = no call.
5. Ticket from the transport: FREE mode + metered is impossible here.
6. Provider adapter call; usage read; reservation settled with the actual.
7. Provider outage classified and remembered; only task outcomes teach the
   reliability model.
8. Persona guard on the text.

:class:`GatewayBrainProvider` wraps all of that in the ``BrainProvider``
protocol the rest of ZEUS already speaks (``generate``,
``generate_structured``, ``generate_stream``), so the routing, composition
and conversation code can move off the local model without being rewritten.
The chat mode for those calls comes from a context variable that the
conversation loop sets per request.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from gateway.budget import BudgetGovernor, BudgetRefused, Reservation
from gateway.config import GatewayConfig, RoleBinding
from gateway.estimate import actual_cost, estimate_cost
from gateway.health import GatewayError, ProviderHealth, ProviderStatus
from gateway.learning import Observation, PerformanceLedger, ReliabilityModel
from gateway.modes import MODE_POLICIES, ChatMode, CostClass, policy_for
from gateway.persona import guard_identity, system_prompt_for_role
from gateway.privacy import Chunk, PrivacyDecision, PrivacyRouter
from gateway.providers import ProviderRequest, adapter_for
from gateway.router import ModelRouter, RouteDecision, RouteKind
from gateway.secrets import CredentialStore
from gateway.task import TaskFacts, TaskVector, rule_based
from gateway.transport import ReservationRequired, Ticket, Transport, ZeroCostViolation


class GatewayRefused(RuntimeError):
    """The gateway would not run a model for this request.  Carries the decision."""

    def __init__(self, decision: RouteDecision, message: str = "") -> None:
        self.decision = decision
        super().__init__(message or decision.reason or decision.kind.value)


@dataclass
class RequestContext:
    """Per-request facts the conversation loop knows and the callers deep down do not."""

    mode: ChatMode = ChatMode.AUTO
    task_id: str = ""
    conversation_id: str = ""
    #: Context chunks beyond the prompt itself (memory, documents ...).
    chunks: list[Chunk] = field(default_factory=list)
    facts: TaskFacts | None = None
    #: Soft task features a semantic model estimated (reasoning_depth,
    #: context_dependency, long_horizon, novelty).  Only ever raise the vector.
    soft: dict[str, float] = field(default_factory=dict)
    language: str = ""


_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar("zeus_gateway_context", default=None)


def current_context() -> RequestContext:
    return _context.get() or RequestContext()


def active_context() -> RequestContext | None:
    """The context the current thread opened, or None -- never a fresh default."""

    return _context.get()


def set_context(context: RequestContext) -> contextvars.Token:
    return _context.set(context)


def reset_context(token: contextvars.Token) -> None:
    _context.reset(token)


@dataclass
class GatewayRequest:
    prompt: str
    mode: ChatMode | str = ChatMode.AUTO
    chunks: list[Chunk] = field(default_factory=list)
    facts: TaskFacts | None = None
    task_vector: TaskVector | None = None
    schema: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    system: str = ""
    task_id: str = ""
    #: Force a role family: "reasoning" or "engineer".  Empty = from the task.
    purpose: str = ""
    #: Pin one role (an engineer chosen by the engineering router).  Every
    #: mode, budget and privacy check still applies; only the ranking is skipped.
    role: str = ""
    #: Semantic soft features, merged as max(rule, semantic).
    soft: dict[str, float] = field(default_factory=dict)
    #: False when the caller has already decided a model is to be consulted
    #: (the BrainProvider path): hard overrides then annotate, never refuse.
    overrides: bool = True


@dataclass
class GatewayReply:
    text: str
    decision: RouteDecision
    role: str
    provider: str
    model: str
    usage: dict[str, int]
    estimated_eur: float
    actual_eur: float
    latency_seconds: float
    identity_rewrites: int = 0
    privacy: PrivacyDecision | None = None
    reservation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "provider": self.provider, "model": self.model, "usage": dict(self.usage),
            "estimated_eur": self.estimated_eur, "actual_eur": self.actual_eur, "latency_seconds": round(self.latency_seconds, 3),
            "identity_rewrites": self.identity_rewrites, "decision": self.decision.to_dict(),
            "privacy": self.privacy.to_dict() if self.privacy else None, "reservation_id": self.reservation_id,
        }


class ModelGateway:
    def __init__(
        self,
        *,
        state_root: str | Path,
        config: GatewayConfig | None = None,
        credentials: CredentialStore | None = None,
        local_provider: Callable[[str], Any] | None = None,
        local_available: Callable[[str], bool] | None = None,
        opener: Callable[..., Any] | None = None,
        emit: Callable[[str, dict[str, Any]], None] | None = None,
        cost_policy: Any = None,
        subscription_available: Callable[[str], bool] | None = None,
    ) -> None:
        self.state_root = Path(state_root)
        #: The owner's spending policy (runtime.cost_policy).  A CostPolicy, or
        #: a callable returning one so a change to the owner document is seen
        #: without a restart.  Default: the repository's own configuration.
        self._cost_policy = cost_policy
        #: provider name -> READY, for subscription engineers the expert gateway drives.
        self._subscription_available = subscription_available or (lambda name: False)
        self.config = config or GatewayConfig.load()
        self.credentials = credentials or CredentialStore(self.state_root / "owner" / "provider_credentials.json")
        self.governor = BudgetGovernor(self.state_root / "gateway" / "spend.jsonl", self.config.budget)
        self.ledger = PerformanceLedger(self.state_root / "gateway" / "performance.jsonl")
        self.reliability = ReliabilityModel(self.config, self.ledger)
        self.health = ProviderHealth()
        self.privacy = PrivacyRouter()
        self.transport = Transport(self.credentials, opener=opener)
        #: role -> a local BrainProvider (the kernel's Ollama tier), for local.* roles.
        self._local_provider = local_provider
        self._local_available = local_available or (lambda role: local_provider is not None)
        self.router = self._make_router(self.config)
        self._emit = emit or (lambda kind, payload: None)
        self._lock = threading.RLock()
        self.recent: list[dict[str, Any]] = []
        #: task_id -> replies made for that task and not yet judged.  The
        #: conversation loop reports the verdict later -- an action's receipt,
        #: a composition's GOAL_SATISFIED, the owner's thumbs -- by task id.
        self._pending: dict[str, list[GatewayReply]] = {}
        self._pending_order: list[str] = []

    # -- configuration changes at runtime ---------------------------------------

    def reconfigure(self, config: GatewayConfig) -> None:
        with self._lock:
            self.config = config
            self.governor.config = config.budget
            self.reliability = ReliabilityModel(config, self.ledger)
            self.router = self._make_router(config)

    def _make_router(self, config: GatewayConfig) -> ModelRouter:
        return ModelRouter(config, self.reliability, self.governor, self.health,
                           credential_present=lambda name: self.credentials.has(name),
                           local_available=self._local_available,
                           paid_allowed=lambda: bool(self.cost_policy.allow_paid_api),
                           subscription_available=self._subscription_available)

    @property
    def cost_policy(self) -> Any:
        """The owner's spending policy, read fresh when it was given as a loader."""

        source = self._cost_policy
        if source is None:
            from runtime.cost_policy import CostPolicy

            return CostPolicy.load()
        return source() if callable(source) else source

    def set_subscription_available(self, probe: Callable[[str], bool]) -> None:
        self._subscription_available = probe
        self.router = self._make_router(self.config)

    # -- planning without calling ---------------------------------------------------

    def task_for(self, request: GatewayRequest) -> TaskVector:
        if request.task_vector is not None:
            return request.task_vector.merged_with_semantic(request.soft) if request.soft else request.task_vector
        facts = request.facts or TaskFacts(text=request.prompt)
        if request.purpose == "engineer" and not facts.is_engineering:
            from dataclasses import replace

            facts = replace(facts, is_engineering=True)
        vector = rule_based(facts)
        return vector.merged_with_semantic(request.soft) if request.soft else vector

    def privacy_for(self, request: GatewayRequest, *, provider_may_train: bool) -> PrivacyDecision:
        chunks = [Chunk(text=request.prompt, source="owner_message")] + list(request.chunks)
        return self.privacy.decide(chunks, provider_may_train=provider_may_train)

    def plan(self, request: GatewayRequest) -> tuple[RouteDecision, PrivacyDecision]:
        """Decide without executing.  What the UI shows as the estimate."""

        mode = ChatMode.parse(request.mode)
        task = self.task_for(request)
        # The privacy decision is provider-independent for routing purposes: the
        # ceiling and whether the free lane is open do not depend on the provider.
        privacy = self.privacy_for(request, provider_may_train=True)
        expected_output = request.max_output_tokens or 512
        decision = self.router.decide(task, mode, privacy, prompt=request.prompt, system=request.system or "",
                                      expected_output_tokens=expected_output, task_id=request.task_id, only_role=request.role,
                                      apply_overrides=request.overrides)
        return decision, privacy

    # -- executing ---------------------------------------------------------------------

    def complete(self, request: GatewayRequest) -> GatewayReply:
        mode = ChatMode.parse(request.mode)
        decision, privacy = self.plan(request)
        if decision.kind is not RouteKind.MODEL:
            raise GatewayRefused(decision)
        binding = self.config.binding(decision.role)
        provider = self.config.provider_for(decision.role)
        if binding is None or provider is None:
            decision.kind, decision.reason = RouteKind.REFUSED, f"role {decision.role} is not bound"
            raise GatewayRefused(decision)
        if not self.config.is_model_role(decision.role):
            decision.kind, decision.reason = RouteKind.REFUSED, f"{decision.role} is driven by the expert gateway, not called as a model"
            raise GatewayRefused(decision)

        # Privacy, now for the provider that will actually receive the request.
        privacy = self.privacy_for(request, provider_may_train=provider.may_train_on_requests)
        if provider.may_train_on_requests and not privacy.free_lane_allowed:
            decision.kind, decision.reason = RouteKind.REFUSED, "sensitive request may not go to a may-train provider"
            raise GatewayRefused(decision)
        prompt_text = privacy.prompt_text()

        if binding.family.value == "local":
            return self._complete_local(request, decision, binding, privacy, prompt_text)

        system = request.system or system_prompt_for_role(decision.role)
        max_out = request.max_output_tokens or binding.max_output_tokens
        temperature = binding.temperature if request.temperature is None else request.temperature
        pricing = self.config.pricing_for(decision.role)
        estimate = estimate_cost(prompt=prompt_text, system=system, expected_output_tokens=max_out, pricing=pricing) if pricing else None
        decision.estimate = estimate

        reservation: Reservation | None = None
        if decision.cost_class is CostClass.METERED:
            if estimate is None:
                decision.kind, decision.reason = RouteKind.REFUSED, "metered route without a price"
                raise GatewayRefused(decision)
            try:
                reservation = self.governor.reserve(role=decision.role, provider=provider.name, model=binding.model,
                                                    estimated_eur=estimate.estimated_eur, task_id=request.task_id, mode=mode.value,
                                                    task_cap_eur=policy_for(mode).task_cap_eur, provider_cap_eur=provider.monthly_cap_eur)
            except BudgetRefused as exc:
                decision.kind, decision.reason = RouteKind.REFUSED, str(exc)
                decision.suggestion = "The budget cap is reached; nothing was spent."
                self._observe(decision, binding, provider.name, goal_verified=False, failure_class="budget_refused", mode=mode)
                raise GatewayRefused(decision, str(exc)) from None

        try:
            ticket = self.transport.issue(provider=provider, role=decision.role, mode=mode, cost_class=decision.cost_class,
                                          reservation=reservation)
        except (ZeroCostViolation, ReservationRequired) as exc:
            if reservation is not None:
                self.governor.release(reservation, reason=type(exc).__name__)
            decision.kind, decision.reason = RouteKind.REFUSED, str(exc)
            self._observe(decision, binding, provider.name, goal_verified=False, failure_class="mode_refused", mode=mode)
            raise GatewayRefused(decision, str(exc)) from None

        thinking = binding.thinking.get(decision.thinking_level) if decision.thinking_level else None
        provider_request = ProviderRequest(system=system, prompt=prompt_text, max_output_tokens=max_out, temperature=temperature,
                                           thinking=thinking, schema=request.schema)
        adapter = adapter_for(provider.kind)
        started = time.perf_counter()
        try:
            reply = adapter.call(self.transport, ticket, provider, binding, provider_request)
        except GatewayError as exc:
            if reservation is not None:
                self.governor.release(reservation, reason=exc.status.value)
            if exc.status.is_outage:
                self.health.note(provider.name, exc.status, detail=str(exc), retry_after_seconds=exc.retry_after_seconds)
            self._observe(decision, binding, provider.name, goal_verified=False, failure_class=exc.status.value, mode=mode,
                          latency=time.perf_counter() - started)
            self._emit("gateway.error", exc.to_dict())
            raise

        actual = actual_cost(reply.usage, pricing) if pricing else 0.0
        if reservation is not None:
            self.governor.settle(reservation, actual, usage=reply.usage)
        self.health.note(provider.name, ProviderStatus.OK)
        text, rewrites = guard_identity(reply.text)
        result = GatewayReply(text=text, decision=decision, role=decision.role, provider=provider.name, model=reply.model or binding.model,
                              usage=reply.usage, estimated_eur=estimate.estimated_eur if estimate else 0.0, actual_eur=actual,
                              latency_seconds=reply.latency_seconds, identity_rewrites=rewrites, privacy=privacy,
                              reservation_id=reservation.reservation_id if reservation else "")
        self._remember(result)
        self._track(request.task_id, result)
        return result

    def _complete_local(self, request: GatewayRequest, decision: RouteDecision, binding: RoleBinding,
                        privacy: PrivacyDecision, prompt_text: str) -> GatewayReply:
        if self._local_provider is None:
            decision.kind, decision.reason = RouteKind.REFUSED, "no local provider is wired"
            raise GatewayRefused(decision)
        provider = self._local_provider(decision.role)
        started = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if request.max_output_tokens:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            if request.schema is not None and hasattr(provider, "generate_structured"):
                text = provider.generate_structured(prompt_text, request.schema, **kwargs)
            else:
                text = provider.generate(prompt_text, **kwargs)
        except Exception as exc:  # noqa: BLE001 - local failures are outages too
            self.health.note(binding.provider, ProviderStatus.PROVIDER_UNAVAILABLE, detail=str(exc)[:200])
            raise GatewayError(ProviderStatus.PROVIDER_UNAVAILABLE, str(exc)[:300], role=decision.role, provider=binding.provider) from exc
        meta = dict(getattr(provider, "last_metadata", {}) or {})
        usage = {"input_tokens": int(meta.get("prompt_tokens") or 0), "cached_input_tokens": 0,
                 "output_tokens": int(meta.get("generated_tokens") or 0)}
        text, rewrites = guard_identity(str(text))
        result = GatewayReply(text=text, decision=decision, role=decision.role, provider=binding.provider,
                              model=str(getattr(provider, "model_name", "") or binding.model), usage=usage, estimated_eur=0.0,
                              actual_eur=0.0, latency_seconds=time.perf_counter() - started, identity_rewrites=rewrites, privacy=privacy)
        self._remember(result)
        self._track(request.task_id, result)
        return result

    # -- learning -------------------------------------------------------------------------

    def report_outcome(self, reply: GatewayReply, *, goal_verified: bool, failure_class: str = "") -> None:
        """The caller says whether the goal was verified.  This is what the router learns from."""

        binding = self.config.binding(reply.role)
        if binding is None:
            return
        self._observe(reply.decision, binding, reply.provider, goal_verified=goal_verified,
                      failure_class=failure_class if not goal_verified else "", mode=reply.decision.mode,
                      latency=reply.latency_seconds, usage=reply.usage, estimated=reply.estimated_eur, actual=reply.actual_eur)

    def _observe(self, decision: RouteDecision, binding: RoleBinding, provider: str, *, goal_verified: bool, failure_class: str,
                 mode: ChatMode, latency: float = 0.0, usage: dict[str, int] | None = None, estimated: float = 0.0,
                 actual: float = 0.0) -> None:
        usage = usage or {}
        self.reliability.observe(Observation(
            task_class=decision.task.task_class.value, role=decision.role or binding.role, provider=provider, model=binding.model,
            goal_verified=goal_verified, failure_class=failure_class, estimated_eur=estimated, actual_eur=actual,
            latency_seconds=round(latency, 3), input_tokens=int(usage.get("input_tokens", 0)),
            cached_input_tokens=int(usage.get("cached_input_tokens", 0)), output_tokens=int(usage.get("output_tokens", 0)),
            task_vector=decision.task.as_features(), mode=mode.value,
        ))

    def report_task_outcome(self, task_id: str, *, goal_verified: bool, failure_class: str = "task_failure") -> int:
        """Judge every reply made for ``task_id``.  Returns how many were judged.

        Called once the world has been checked: the receipt verified, the
        goal was satisfied, the owner said it was right or wrong.  A task
        with no gateway reply (a deterministic route) judges nothing.
        """

        if not task_id:
            return 0
        with self._lock:
            replies = self._pending.pop(task_id, [])
            if task_id in self._pending_order:
                self._pending_order.remove(task_id)
        for reply in replies:
            self.report_outcome(reply, goal_verified=goal_verified, failure_class=failure_class)
        return len(replies)

    def _track(self, task_id: str, reply: GatewayReply) -> None:
        if not task_id:
            return
        with self._lock:
            if task_id not in self._pending:
                self._pending[task_id] = []
                self._pending_order.append(task_id)
            self._pending[task_id].append(reply)
            # Unjudged tasks are forgotten, not counted: silence is not success.
            while len(self._pending_order) > 200:
                stale = self._pending_order.pop(0)
                self._pending.pop(stale, None)

    def _remember(self, reply: GatewayReply) -> None:
        with self._lock:
            self.recent.append({"at": time.time(), "role": reply.role, "provider": reply.provider, "model": reply.model,
                                "estimated_eur": reply.estimated_eur, "actual_eur": reply.actual_eur,
                                "latency_seconds": round(reply.latency_seconds, 3), "usage": dict(reply.usage),
                                "task_class": reply.decision.task.task_class.value, "mode": reply.decision.mode.value,
                                "offline_fallback": reply.decision.offline_fallback})
            del self.recent[:-100]
        self._emit("gateway.call", self.recent[-1])

    # -- status for the UI -----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        spend = self.governor.summary().to_dict()
        roles = {}
        for role, binding in self.config.roles.items():
            provider = self.config.providers.get(binding.provider)
            configured = bool(binding.enabled and provider is not None and provider.enabled
                              and (not provider.secret or self.credentials.has(provider.secret)))
            roles[role] = {"provider": binding.provider, "model": binding.model, "enabled": binding.enabled,
                           "configured": configured, "cost_class": self.config.cost_class(role).value,
                           "health": self.health.status(binding.provider).value if provider else "unknown",
                           "offline_fallback": binding.offline_fallback}
        providers = {}
        for name, provider in self.config.providers.items():
            providers[name] = {"kind": provider.kind, "enabled": provider.enabled, "metered": provider.metered,
                               "may_train_on_requests": provider.may_train_on_requests,
                               "credential": self.credentials.has(provider.secret) if provider.secret else True,
                               "health": self.health.status(name).value, "base_url": provider.base_url,
                               "pricing": {m: p.to_dict() for m, p in provider.pricing.items()}}
        return {
            "spend": spend, "roles": roles, "providers": providers, "credentials": self.credentials.status(),
            "modes": {m.value: {"allow_metered": p.allow_metered, "hint_de": p.owner_hint_de, "hint_en": p.owner_hint_en,
                                "task_cap_eur": p.task_cap_eur} for m, p in MODE_POLICIES.items()},
            "budget": self.config.budget.to_dict(), "config_source": self.config.source,
            "paid_api_allowed": bool(self.cost_policy.allow_paid_api), "cost_policy_source": str(getattr(self.cost_policy, "source", "")),
            "transport": {"issued": self.transport.issued, "refused": len(self.transport.refused)},
            "recent": list(self.recent[-20:]),
            "pending_outcomes": len(self._pending),
            "reliability": self.reliability.table(),
            "cloud_reasoning_available": any(roles[r]["configured"] for r in ("reasoning.free", "reasoning.deep") if r in roles),
        }


# ---------------------------------------------------------------------------
# BrainProvider adapter
# ---------------------------------------------------------------------------

class GatewayBrainProvider:
    """Speaks the ``BrainProvider`` protocol; routes through the gateway.

    Given to the kernel in place of the FAST_LOCAL Ollama provider.  Every
    ``generate`` becomes a gateway request in the current chat mode.  When the
    gateway refuses (FREE mode with no free cloud, no credentials, budget) the
    local fallback answers *if the request context permits it*; the decision
    is recorded on ``last_decision`` so the caller can say which happened.
    """

    provider_name = "zeus-gateway"

    def __init__(self, gateway: ModelGateway, *, fallback: Any | None = None, fallback_allowed: Callable[[], bool] | None = None) -> None:
        self.gateway = gateway
        self.fallback = fallback
        self._fallback_allowed = fallback_allowed or (lambda: True)
        self.last_metadata: dict[str, Any] = {}
        self.last_decision: dict[str, Any] = {}
        self.last_reply: GatewayReply | None = None

    @property
    def model_name(self) -> str:
        if self.last_reply is not None:
            return f"{self.last_reply.role}:{self.last_reply.model}"
        return "gateway"

    @property
    def spec(self) -> Any:
        return getattr(self.fallback, "spec", None)

    def _request(self, prompt: str, *, schema: dict[str, Any] | None, max_tokens: int | None, temperature: float | None,
                 system: str | None) -> GatewayRequest:
        context = current_context()
        # The caller (semantic planner, composer, the conversation) has already
        # decided to consult a model; the gateway picks which one.  Hard
        # overrides -- "this is a destructive action", "a capability matches"
        # -- are about executing, and are decided by the caller, not here.
        return GatewayRequest(prompt=prompt, mode=context.mode, chunks=list(context.chunks), facts=context.facts, schema=schema,
                              max_output_tokens=max_tokens, temperature=temperature, system=system or "", task_id=context.task_id,
                              soft=dict(context.soft), overrides=False)

    def _run(self, prompt: str, *, schema: dict[str, Any] | None = None, max_tokens: int | None = None,
             temperature: float | None = None, system: str | None = None) -> str:
        request = self._request(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature, system=system)
        try:
            reply = self.gateway.complete(request)
        except GatewayRefused as exc:
            self.last_decision = exc.decision.to_dict()
            if self.fallback is None or not self._fallback_allowed():
                raise
            return self._fallback(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature, system=system,
                                  why=exc.decision.reason)
        except GatewayError as exc:
            self.last_decision = {"kind": "provider_error", **exc.to_dict()}
            if self.fallback is None or not self._fallback_allowed() or not exc.status.is_outage:
                raise
            return self._fallback(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature, system=system,
                                  why=exc.status.value)
        self.last_reply = reply
        self.last_decision = reply.decision.to_dict()
        self.last_metadata = {
            "latency_seconds": reply.latency_seconds, "generated_tokens": reply.usage.get("output_tokens"),
            "prompt_tokens": reply.usage.get("input_tokens"), "role": reply.role, "provider": reply.provider, "model": reply.model,
            "estimated_eur": reply.estimated_eur, "actual_eur": reply.actual_eur, "offline_fallback": reply.decision.offline_fallback,
        }
        return reply.text

    def _fallback(self, prompt: str, *, schema: dict[str, Any] | None, max_tokens: int | None, temperature: float | None,
                  system: str | None, why: str) -> str:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if schema is not None and hasattr(self.fallback, "generate_structured"):
            text = self.fallback.generate_structured(prompt, schema, **kwargs)
        else:
            if system is not None:
                try:
                    text = self.fallback.generate(prompt, system=system, **kwargs)
                except TypeError:
                    text = self.fallback.generate(f"{system}\n\n{prompt}", **kwargs)
            else:
                text = self.fallback.generate(prompt, **kwargs)
        self.last_metadata = dict(getattr(self.fallback, "last_metadata", {}) or {})
        self.last_metadata.update({"offline_fallback": True, "fallback_reason": why})
        self.last_decision = {**self.last_decision, "fell_back_to_local": True, "fallback_reason": why}
        return str(text)

    # -- BrainProvider protocol -----------------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int | None = None, temperature: float | None = None,
                 top_p: float | None = None, system: str | None = None) -> str:
        return self._run(prompt, max_tokens=max_tokens, temperature=temperature, system=system)

    def generate_coding(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.1, top_p: float = 0.9) -> str:
        return self._run(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_structured(self, prompt: str, schema: dict[str, Any], *, max_tokens: int = 1024, temperature: float = 0.1,
                            top_p: float = 0.9) -> str:
        return self._run(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature)

    def generate_stream(self, prompt: str, *, max_tokens: int | None = None, temperature: float | None = None,
                        top_p: float | None = None, system: str | None = None) -> Iterator[str]:
        # Cloud roles answer in one piece; the conversation loop streams
        # whatever it receives, so yield the answer in sentence-sized chunks
        # rather than blocking the speaker until the end.
        text = self._run(prompt, max_tokens=max_tokens, temperature=temperature, system=system)
        if self.last_metadata.get("offline_fallback") and hasattr(self.fallback, "generate_stream"):
            yield text
            return
        import re

        for piece in re.split(r"(?<=[.!?\n])\s+", text):
            if piece:
                yield piece + (" " if not piece.endswith("\n") else "")

    def think(self, user_prompt: str, max_tokens: int = 512) -> str:
        return self.generate(user_prompt, max_tokens=max_tokens)

    def think_coding(self, user_prompt: str, max_tokens: int = 1024, temperature: float = 0.1, top_p: float = 0.9) -> str:
        return self.generate_coding(user_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def health_check(self) -> dict[str, Any]:
        status = self.gateway.status()
        return {"ok": bool(status.get("cloud_reasoning_available")) or self.fallback is not None, "gateway": True,
                "cloud_reasoning_available": status.get("cloud_reasoning_available"), "spend": status.get("spend")}

    def capabilities(self) -> dict[str, Any]:
        return {"chat": True, "coding": True, "embeddings": False, "local": False, "structured_generation": True, "gateway": True}

    def list_models(self) -> list[str]:
        return [f"{role}:{binding.model}" for role, binding in self.gateway.config.roles.items() if binding.model]

    def unload(self) -> None:
        if hasattr(self.fallback, "unload"):
            self.fallback.unload()

    def __getattr__(self, name: str) -> Any:
        # Anything not part of the gateway's surface (e.g. Ollama-only knobs)
        # reaches the local fallback, so existing callers keep working.
        fallback = self.__dict__.get("fallback")
        if fallback is None:
            raise AttributeError(name)
        return getattr(fallback, name)
