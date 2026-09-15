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
import random
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from gateway.budget import BudgetGovernor, BudgetRefused, Reservation
from gateway.config import GatewayConfig, RoleBinding, THINKING_LEVELS, owner_override_path
from gateway.estimate import actual_cost, estimate_cost
from gateway.health import FreeIntelligenceUnavailable, GatewayError, ProviderHealth, ProviderStatus
from gateway.learning import Observation, PerformanceLedger, ReliabilityModel
from gateway.modes import MODE_POLICIES, ChatMode, CostClass, policy_for
from gateway.output_budget import OutputBudget, decide_output_budget, step_down, truncated_by_limit
from gateway.persona import guard_identity, system_prompt_for_role
from gateway.privacy import Chunk, PrivacyDecision, PrivacyRouter
from gateway.providers import ProviderRequest, adapter_for
from gateway.router import ModelRouter, RouteDecision, RouteKind
from gateway.secrets import CredentialStore
from gateway.task import TaskFacts, TaskVector, rule_based
from gateway.transport import ReservationRequired, Ticket, Transport, ZeroCostViolation


TRANSIENT_FREE_POOL_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
FREE_POOL_MAX_ATTEMPTS_PER_MODEL = 2
FREE_POOL_BASE_DELAY_SECONDS = 1.0
FREE_POOL_MAX_DELAY_SECONDS = 2.0


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
    #: The owner's current words and, when the caller decided one, the output budget.
    owner_text: str = ""
    output_budget: Any = None


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
    #: False for decisions the small local model may never take (GoalSpec,
    #: PlanSpec): when only the offline fallback is reachable the call is
    #: refused instead of made.
    allow_offline_fallback: bool = True
    #: The owner's own words for this request (the prompt may carry the
    #: whole transcript): what the output budget is read from.
    owner_text: str = ""
    #: An OutputBudget the caller decided; otherwise the gateway decides one
    #: from owner_text, the task and the mode.  ``max_output_tokens`` set by
    #: the caller outranks both (a structured decision knows its size).
    output_budget: Any = None


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
    route_attempts: list[dict[str, Any]] = field(default_factory=list)
    #: How the provider stopped, and whether that was the output ceiling.
    finish_reason: str = ""
    truncated: bool = False
    output_budget: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 0
    provider_hard_limit: int = 0
    #: The consumer closed the stream before the provider finished.
    aborted: bool = False

    def completion(self) -> dict[str, Any]:
        """What a reader needs to judge whether the answer is whole."""

        return {"finish_reason": self.finish_reason, "truncated": self.truncated, "output_tokens": int(self.usage.get("output_tokens", 0) or 0),
                "configured_output_budget": self.max_output_tokens, "output_budget": dict(self.output_budget),
                "provider_hard_limit": self.provider_hard_limit, "aborted": self.aborted}

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "provider": self.provider, "model": self.model, "usage": dict(self.usage),
            "estimated_eur": self.estimated_eur, "actual_eur": self.actual_eur, "latency_seconds": round(self.latency_seconds, 3),
            "identity_rewrites": self.identity_rewrites, "decision": self.decision.to_dict(),
            "privacy": self.privacy.to_dict() if self.privacy else None, "reservation_id": self.reservation_id,
            "route_attempts": list(self.route_attempts), "final_selected_model": self.model, "completion": self.completion(),
        }


@dataclass
class _Prepared:
    """Everything decided before a provider is called: route, privacy, budget, reservation, ticket."""

    request: GatewayRequest
    mode: ChatMode
    decision: RouteDecision
    privacy: PrivacyDecision
    binding: RoleBinding
    provider: Any
    prompt_text: str
    system: str = ""
    max_out: int = 0
    temperature: float = 0.0
    pricing: Any = None
    estimate: Any = None
    reservation: Reservation | None = None
    ticket: Any = None
    provider_request: ProviderRequest | None = None
    adapter: Any = None
    budget: OutputBudget | None = None
    hard_limit: int = 0
    local: bool = False
    started: float = 0.0


class GatewayStream:
    """Text as the provider produces it; the settled :class:`GatewayReply` once it has finished.

    Iterate it once.  ``reply`` is set when the iteration ends normally;
    ``error`` when the provider failed; ``aborted`` when the consumer stopped
    early (the reservation is then settled at the estimate, conservatively).
    """

    def __init__(self, gateway: "ModelGateway", prepared: _Prepared) -> None:
        self._gateway = gateway
        self.prepared = prepared
        self.decision = prepared.decision
        self.reply: GatewayReply | None = None
        self.error: Exception | None = None
        self.aborted = False

    def __iter__(self) -> Iterator[str]:
        return self._gateway._run_stream(self.prepared, self)


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
        retry_sleep: Callable[[float], None] | None = None,
        retry_jitter: Callable[[float], float] | None = None,
        import_environment_credentials: bool = True,
    ) -> None:
        self.state_root = Path(state_root)
        #: The owner's spending policy (runtime.cost_policy).  A CostPolicy, or
        #: a callable returning one so a change to the owner document is seen
        #: without a restart.  Default: the repository's own configuration.
        self._cost_policy = cost_policy
        #: provider name -> READY, for subscription engineers the expert gateway drives.
        self._subscription_available = subscription_available or (lambda name: False)
        self.config = config or GatewayConfig.load(override_path=owner_override_path(self.state_root))
        self.credentials = credentials or CredentialStore(self.state_root / "owner" / "provider_credentials.json")
        try:
            # An owner who exported OPENAI_API_KEY (or the Gemini / Anthropic
            # names) has entered the credential: it moves into the encrypted
            # store once and the provider counts as configured.
            imported = self.credentials.import_environment({p.secret: p.credential_env for p in self.config.providers.values()
                                                            if p.secret and p.credential_env}) if import_environment_credentials else []
        except Exception:  # noqa: BLE001 - a store that cannot be written is reported by status(), not here
            imported = []
        if imported:
            for slot in imported:
                provider = self.config.provider_for_secret(slot)
                if provider is not None and not provider.enabled:
                    self.config = self.config.with_provider_enabled(provider.name, True)
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
        self._retry_sleep = retry_sleep or time.sleep
        self._retry_jitter = retry_jitter or (lambda spread: random.uniform(0.0, spread))
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
        def credential_present(name: str) -> bool:
            provider = self.config.providers.get(name)
            if provider is None:
                return False
            return self.credentials.has(provider.secret) if provider.secret else True

        return ModelRouter(config, self.reliability, self.governor, self.health,
                           credential_present=credential_present,
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

    def _prepare(self, request: GatewayRequest) -> _Prepared:
        """Route, privacy, output budget, estimate, reservation, ticket -- all before any byte moves."""

        mode = ChatMode.parse(request.mode)
        decision, privacy = self.plan(request)
        if decision.kind is not RouteKind.MODEL:
            raise GatewayRefused(decision)
        if decision.offline_fallback and not request.allow_offline_fallback:
            # Semantic decisions never go to the small local model: the
            # caller said so, and the call is refused before it is made.
            decision.kind = RouteKind.REFUSED
            decision.reason = "only the offline fallback model is reachable; this request does not accept it (" + decision.reason[:160] + ")"
            decision.suggestion = decision.suggestion or "Configure or re-enable a cloud reasoning provider, or change the mode."
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
        prepared = _Prepared(request=request, mode=mode, decision=decision, privacy=privacy, binding=binding, provider=provider,
                             prompt_text=prompt_text, local=binding.family.value == "local")
        if prepared.local:
            return prepared

        system = request.system or system_prompt_for_role(decision.role)
        temperature = binding.temperature if request.temperature is None else request.temperature
        hard_limit = int((provider.options or {}).get("output_hard_limit") or 0)
        # The output budget: the caller's explicit token count wins (a
        # structured decision knows its size); otherwise a level from the
        # owner's words, the task and the mode, clipped to the provider's
        # real maximum.  Decided here, before the estimate and the reservation.
        budget: OutputBudget | None = request.output_budget
        if request.max_output_tokens:
            max_out = int(request.max_output_tokens)
        else:
            if budget is None:
                budget = decide_output_budget(request.owner_text or request.prompt, task=decision.task, mode=mode,
                                              structured=request.schema is not None, hard_limit=hard_limit or None)
            max_out = int(budget.tokens)
        if hard_limit:
            max_out = min(max_out, hard_limit)
        decision.output_budget = budget.to_dict() if budget else {"level": "explicit", "tokens": max_out, "reason": "caller-specified"}
        pricing = self.config.pricing_for(decision.role)
        estimate = estimate_cost(prompt=prompt_text, system=system, expected_output_tokens=max_out, pricing=pricing) if pricing else None
        decision.estimate = estimate

        reservation: Reservation | None = None
        if decision.cost_class is CostClass.METERED:
            if estimate is None:
                decision.kind, decision.reason = RouteKind.REFUSED, "metered route without a price"
                raise GatewayRefused(decision)
            while True:
                try:
                    reservation = self.governor.reserve(role=decision.role, provider=provider.name, model=binding.model,
                                                        estimated_eur=estimate.estimated_eur, task_id=request.task_id, mode=mode.value,
                                                        task_cap_eur=policy_for(mode).task_cap_eur, provider_cap_eur=provider.monthly_cap_eur)
                    break
                except BudgetRefused as exc:
                    # A paid answer that would not fit is asked for shorter,
                    # one level at a time, never silently sent as it was.
                    lower = step_down(budget, hard_limit=hard_limit or None) if budget is not None else None
                    if lower is None:
                        decision.kind, decision.reason = RouteKind.REFUSED, str(exc)
                        decision.suggestion = "The budget cap is reached; nothing was spent."
                        self._observe(decision, binding, provider.name, goal_verified=False, failure_class="budget_refused", mode=mode)
                        raise GatewayRefused(decision, str(exc)) from None
                    budget = lower
                    max_out = min(int(budget.tokens), hard_limit) if hard_limit else int(budget.tokens)
                    estimate = estimate_cost(prompt=prompt_text, system=system, expected_output_tokens=max_out, pricing=pricing)
                    decision.estimate = estimate
                    decision.output_budget = budget.to_dict()

        try:
            ticket = self.transport.issue(provider=provider, role=decision.role, mode=mode, cost_class=decision.cost_class,
                                          reservation=reservation)
        except (ZeroCostViolation, ReservationRequired) as exc:
            if reservation is not None:
                self.governor.release(reservation, reason=type(exc).__name__)
            decision.kind, decision.reason = RouteKind.REFUSED, str(exc)
            self._observe(decision, binding, provider.name, goal_verified=False, failure_class="mode_refused", mode=mode)
            raise GatewayRefused(decision, str(exc)) from None

        thinking = binding.thinking_for(decision.thinking_level) if decision.thinking_level else None
        prepared.system, prepared.max_out, prepared.temperature = system, max_out, temperature
        prepared.pricing, prepared.estimate, prepared.reservation, prepared.ticket = pricing, estimate, reservation, ticket
        prepared.budget, prepared.hard_limit = budget, hard_limit
        prepared.provider_request = ProviderRequest(system=system, prompt=prompt_text, max_output_tokens=max_out, temperature=temperature,
                                                    thinking=thinking, schema=request.schema, thinking_level=decision.thinking_level)
        prepared.adapter = adapter_for(provider.kind)
        return prepared

    def _fail(self, prepared: _Prepared, exc: GatewayError, *, route_attempts: list[dict[str, Any]] | None = None) -> None:
        if prepared.reservation is not None:
            self.governor.release(prepared.reservation, reason=exc.status.value)
        if exc.status.is_outage:
            self.health.note(prepared.provider.name, exc.status, detail=str(exc), retry_after_seconds=exc.retry_after_seconds)
        attempts = list(route_attempts if route_attempts is not None else (getattr(exc, "attempts", []) or []))
        self._observe(prepared.decision, prepared.binding, prepared.provider.name, goal_verified=False, failure_class=exc.status.value,
                      mode=prepared.mode, latency=time.perf_counter() - prepared.started, route_attempts=attempts,
                      model=(attempts[-1].get("model") if attempts else None))
        self._emit("gateway.error", exc.to_dict())

    def _finish(self, prepared: _Prepared, reply: Any, route_attempts: list[dict[str, Any]], *, text: str | None = None,
                rewrites: int = 0) -> GatewayReply:
        pricing = prepared.pricing
        actual = actual_cost(reply.usage, pricing) if pricing else 0.0
        if prepared.reservation is not None:
            self.governor.settle(prepared.reservation, actual, usage=reply.usage, native=pricing.native_cost(reply.usage) if pricing else None,
                                 currency=pricing.currency if pricing else "EUR", rate_source=pricing.rate_source if pricing else "")
        self.health.note(prepared.provider.name, ProviderStatus.OK)
        if text is None:
            text, rewrites = guard_identity(reply.text)
        decision = prepared.decision
        result = GatewayReply(text=text, decision=decision, role=decision.role, provider=prepared.provider.name,
                              model=reply.model or prepared.binding.model, usage=reply.usage,
                              estimated_eur=prepared.estimate.estimated_eur if prepared.estimate else 0.0, actual_eur=actual,
                              latency_seconds=time.perf_counter() - prepared.started, identity_rewrites=rewrites, privacy=prepared.privacy,
                              reservation_id=prepared.reservation.reservation_id if prepared.reservation else "", route_attempts=route_attempts,
                              finish_reason=str(getattr(reply, "finish_reason", "") or ""), truncated=truncated_by_limit(getattr(reply, "finish_reason", "")),
                              output_budget=dict(decision.output_budget), max_output_tokens=prepared.max_out, provider_hard_limit=prepared.hard_limit)
        self._remember(result)
        self._track(prepared.request.task_id, result)
        return result

    def complete(self, request: GatewayRequest) -> GatewayReply:
        prepared = self._prepare(request)
        if prepared.local:
            return self._complete_local(request, prepared.decision, prepared.binding, prepared.privacy, prepared.prompt_text)
        prepared.started = time.perf_counter()
        try:
            reply, route_attempts = self._call_model_pool(prepared.adapter, prepared.ticket, prepared.provider, prepared.binding,
                                                          prepared.provider_request, decision=prepared.decision, mode=prepared.mode,
                                                          started=prepared.started)
        except GatewayError as exc:
            self._fail(prepared, exc)
            raise
        return self._finish(prepared, reply, route_attempts)

    # -- streaming -----------------------------------------------------------------------

    def stream(self, request: GatewayRequest) -> GatewayStream:
        """The same decision as :meth:`complete`, made now; the answer arrives as the stream is iterated.

        A provider that cannot stream this operation answers in one piece,
        yielded once.  The reservation is settled from the provider's usage
        when the stream ends, and at the estimate if the consumer stops early.
        """

        prepared = self._prepare(request)
        return GatewayStream(self, prepared)

    def _run_stream(self, prepared: _Prepared, holder: GatewayStream) -> Iterator[str]:
        if prepared.local:
            yield from self._stream_local(prepared, holder)
            return
        prepared.started = time.perf_counter()
        adapter = prepared.adapter
        if not hasattr(adapter, "stream"):
            reply = self.complete(prepared.request) if False else None  # never: every configured kind streams; kept for foreign adapters
            try:
                raw, route_attempts = self._call_model_pool(adapter, prepared.ticket, prepared.provider, prepared.binding, prepared.provider_request,
                                                            decision=prepared.decision, mode=prepared.mode, started=prepared.started)
            except GatewayError as exc:
                self._fail(prepared, exc)
                holder.error = exc
                raise
            holder.reply = self._finish(prepared, raw, route_attempts)
            if holder.reply.text:
                yield holder.reply.text
            return
        pieces: list[str] = []
        rewrites = 0
        route_attempts: list[dict[str, Any]] = []
        final = None
        gen = self._stream_model_pool(prepared, route_attempts)
        try:
            for event in gen:
                if "text" in event:
                    text, n = guard_identity(event["text"])
                    rewrites += n
                    pieces.append(text)
                    yield text
                elif "reply" in event:
                    final = event["reply"]
        except GatewayError as exc:
            self._fail(prepared, exc, route_attempts=route_attempts)
            holder.error = exc
            raise
        except GeneratorExit:
            # The consumer stopped listening.  What the provider produced is
            # billed anyway: settle at the estimate, the conservative figure.
            holder.aborted = True
            gen.close()
            if prepared.reservation is not None and prepared.estimate is not None:
                self.governor.settle(prepared.reservation, prepared.estimate.estimated_eur, usage={"note": "stream aborted by the consumer"})
            self._observe(prepared.decision, prepared.binding, prepared.provider.name, goal_verified=False, failure_class="cancelled",
                          mode=prepared.mode, latency=time.perf_counter() - prepared.started, route_attempts=route_attempts)
            raise
        if final is None:
            from gateway.providers import ProviderReply

            final = ProviderReply(text="", usage={"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}, finish_reason="")
        final.text = "".join(pieces)
        holder.reply = self._finish(prepared, final, route_attempts, text=final.text, rewrites=rewrites)

    def _stream_model_pool(self, prepared: _Prepared, route_attempts: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """The free pool's fallback for streams: another model only before any text has been shown."""

        adapter, provider, binding = prepared.adapter, prepared.provider, prepared.binding
        models = binding.model_pool
        if prepared.decision.cost_class is not CostClass.ZERO or len(models) <= 1:
            attempt_started = time.perf_counter()
            yield from adapter.stream(self.transport, prepared.ticket, provider, binding, prepared.provider_request)
            route_attempts.append({"model": binding.model, "attempt": 1, "failure_class": "ok", "http_status": None, "retry_delay_seconds": 0.0,
                                   "latency_seconds": round(time.perf_counter() - attempt_started, 3)})
            return
        last_error: GatewayError | None = None
        for model in models:
            model_price = self.config.pricing_for_model(prepared.decision.role, model)
            if provider.metered or (model_price is not None and model_price.metered):
                raise ZeroCostViolation(f"{prepared.decision.role} model {model} is not zero-cost")
            attempt_binding = replace(binding, model=model, models=(model,))
            for attempt in range(1, FREE_POOL_MAX_ATTEMPTS_PER_MODEL + 1):
                attempt_started = time.perf_counter()
                shown = False
                try:
                    for event in adapter.stream(self.transport, prepared.ticket, provider, attempt_binding, prepared.provider_request):
                        if "text" in event:
                            shown = True
                        if "reply" in event and event["reply"] is not None:
                            event["reply"].model = event["reply"].model or model
                        yield event
                    route_attempts.append({"model": model, "attempt": attempt, "failure_class": "ok", "http_status": None,
                                           "retry_delay_seconds": 0.0, "latency_seconds": round(time.perf_counter() - attempt_started, 3)})
                    self._emit_free_pool(prepared.decision, provider.name, route_attempts, final_model=model, started=prepared.started,
                                         monetary_cost_eur=0.0, goal_verified=None)
                    return
                except GatewayError as exc:
                    last_error = exc
                    transient = self._is_transient_free_pool_error(exc) and not shown
                    retry_delay = self._free_pool_retry_delay(attempt) if transient and attempt < FREE_POOL_MAX_ATTEMPTS_PER_MODEL else 0.0
                    route_attempts.append({"model": model, "attempt": attempt, "failure_class": exc.status.value, "http_status": exc.http_status,
                                           "retry_delay_seconds": round(retry_delay, 3), "latency_seconds": round(time.perf_counter() - attempt_started, 3)})
                    if not transient:
                        raise
                    if retry_delay > 0:
                        self._retry_sleep(retry_delay)
        self._emit_free_pool(prepared.decision, provider.name, route_attempts, final_model="", started=prepared.started, monetary_cost_eur=0.0,
                             goal_verified=False)
        raise FreeIntelligenceUnavailable(role=prepared.decision.role, provider=provider.name, attempts=route_attempts) from last_error

    def _stream_local(self, prepared: _Prepared, holder: GatewayStream) -> Iterator[str]:
        request, decision, binding = prepared.request, prepared.decision, prepared.binding
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
        pieces: list[str] = []
        rewrites = 0
        try:
            if hasattr(provider, "generate_stream"):
                try:
                    iterator = provider.generate_stream(prepared.prompt_text, system=request.system or None, **kwargs)
                except TypeError:
                    iterator = provider.generate_stream(prepared.prompt_text, **kwargs)
                for chunk in iterator:
                    text, n = guard_identity(str(chunk))
                    rewrites += n
                    pieces.append(text)
                    yield text
            else:
                text, rewrites = guard_identity(str(provider.generate(prepared.prompt_text, **kwargs)))
                pieces.append(text)
                yield text
        except GeneratorExit:
            holder.aborted = True
            raise
        except Exception as exc:  # noqa: BLE001 - local failures are outages too
            self.health.note(binding.provider, ProviderStatus.PROVIDER_UNAVAILABLE, detail=str(exc)[:200])
            raise GatewayError(ProviderStatus.PROVIDER_UNAVAILABLE, str(exc)[:300], role=decision.role, provider=binding.provider) from exc
        meta = dict(getattr(provider, "last_metadata", {}) or {})
        usage = {"input_tokens": int(meta.get("prompt_tokens") or 0), "cached_input_tokens": 0,
                 "output_tokens": int(meta.get("generated_tokens") or 0)}
        holder.reply = GatewayReply(text="".join(pieces), decision=decision, role=decision.role, provider=binding.provider,
                                    model=str(getattr(provider, "model_name", "") or binding.model), usage=usage, estimated_eur=0.0,
                                    actual_eur=0.0, latency_seconds=time.perf_counter() - started, identity_rewrites=rewrites,
                                    privacy=prepared.privacy, finish_reason=str(meta.get("finish_reason") or ""))
        self._remember(holder.reply)
        self._track(request.task_id, holder.reply)

    def _call_model_pool(self, adapter: Any, ticket: Ticket, provider: Any, binding: RoleBinding,
                         provider_request: ProviderRequest, *, decision: RouteDecision, mode: ChatMode,
                         started: float) -> tuple[Any, list[dict[str, Any]]]:
        models = binding.model_pool
        if decision.cost_class is not CostClass.ZERO or len(models) <= 1:
            reply_started = time.perf_counter()
            reply = adapter.call(self.transport, ticket, provider, binding, provider_request)
            return reply, [{"model": reply.model or binding.model, "attempt": 1, "failure_class": "ok",
                            "http_status": None, "retry_delay_seconds": 0.0,
                            "latency_seconds": round(time.perf_counter() - reply_started, 3)}]

        attempts: list[dict[str, Any]] = []
        last_error: GatewayError | None = None
        for model in models:
            model_price = self.config.pricing_for_model(decision.role, model)
            if provider.metered or (model_price is not None and model_price.metered):
                raise ZeroCostViolation(f"{decision.role} model {model} is not zero-cost")
            attempt_binding = replace(binding, model=model, models=(model,))
            for attempt in range(1, FREE_POOL_MAX_ATTEMPTS_PER_MODEL + 1):
                attempt_started = time.perf_counter()
                try:
                    reply = adapter.call(self.transport, ticket, provider, attempt_binding, provider_request)
                    attempts.append({"model": model, "attempt": attempt, "failure_class": "ok", "http_status": None,
                                     "retry_delay_seconds": 0.0,
                                     "latency_seconds": round(time.perf_counter() - attempt_started, 3)})
                    self._emit_free_pool(decision, provider.name, attempts, final_model=model, started=started,
                                         monetary_cost_eur=0.0, goal_verified=None)
                    return reply, attempts
                except GatewayError as exc:
                    last_error = exc
                    transient = self._is_transient_free_pool_error(exc)
                    retry_delay = self._free_pool_retry_delay(attempt) if transient and attempt < FREE_POOL_MAX_ATTEMPTS_PER_MODEL else 0.0
                    attempts.append({"model": model, "attempt": attempt, "failure_class": exc.status.value,
                                     "http_status": exc.http_status, "retry_delay_seconds": round(retry_delay, 3),
                                     "latency_seconds": round(time.perf_counter() - attempt_started, 3)})
                    if not transient:
                        raise
                    if retry_delay > 0:
                        self._retry_sleep(retry_delay)
        self._emit_free_pool(decision, provider.name, attempts, final_model="", started=started, monetary_cost_eur=0.0,
                             goal_verified=False)
        raise FreeIntelligenceUnavailable(role=decision.role, provider=provider.name, attempts=attempts) from last_error

    def _is_transient_free_pool_error(self, exc: GatewayError) -> bool:
        return exc.http_status in TRANSIENT_FREE_POOL_HTTP_STATUSES

    def _free_pool_retry_delay(self, attempt: int) -> float:
        base = min(FREE_POOL_MAX_DELAY_SECONDS, FREE_POOL_BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1)))
        jitter = max(0.0, min(0.2, base * 0.15))
        return min(FREE_POOL_MAX_DELAY_SECONDS, base + float(self._retry_jitter(jitter)))

    def _emit_free_pool(self, decision: RouteDecision, provider: str, attempts: list[dict[str, Any]], *,
                        final_model: str, started: float, monetary_cost_eur: float,
                        goal_verified: bool | None) -> None:
        payload = {"role": decision.role, "provider": provider, "attempts": list(attempts),
                   "final_selected_model": final_model, "total_latency_seconds": round(time.perf_counter() - started, 3),
                   "monetary_cost_eur": monetary_cost_eur, "goal_verified": goal_verified}
        self._emit("gateway.free_pool", payload)

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
                      latency=reply.latency_seconds, usage=reply.usage, estimated=reply.estimated_eur, actual=reply.actual_eur,
                      model=reply.model, route_attempts=reply.route_attempts)
        if reply.route_attempts and reply.decision.cost_class is CostClass.ZERO and len(binding.model_pool) > 1:
            self._emit_free_pool(reply.decision, reply.provider, reply.route_attempts, final_model=reply.model,
                                 started=time.perf_counter() - reply.latency_seconds, monetary_cost_eur=reply.actual_eur,
                                 goal_verified=goal_verified)

    def _observe(self, decision: RouteDecision, binding: RoleBinding, provider: str, *, goal_verified: bool, failure_class: str,
                 mode: ChatMode, latency: float = 0.0, usage: dict[str, int] | None = None, estimated: float = 0.0,
                 actual: float = 0.0, model: str | None = None, route_attempts: list[dict[str, Any]] | None = None) -> None:
        usage = usage or {}
        self.reliability.observe(Observation(
            task_class=decision.task.task_class.value, role=decision.role or binding.role, provider=provider,
            model=model or binding.model,
            goal_verified=goal_verified, failure_class=failure_class, estimated_eur=estimated, actual_eur=actual,
            latency_seconds=round(latency, 3), input_tokens=int(usage.get("input_tokens", 0)),
            cached_input_tokens=int(usage.get("cached_input_tokens", 0)), output_tokens=int(usage.get("output_tokens", 0)),
            task_vector=decision.task.as_features(), mode=mode.value, thinking_level=decision.thinking_level,
            route_attempts=list(route_attempts or []), final_selected_model=model or binding.model,
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
                                "monetary_cost_eur": reply.actual_eur,
                                "latency_seconds": round(reply.latency_seconds, 3), "usage": dict(reply.usage),
                                "task_class": reply.decision.task.task_class.value, "mode": reply.decision.mode.value,
                                "offline_fallback": reply.decision.offline_fallback,
                                "route_attempts": list(reply.route_attempts), "final_selected_model": reply.model,
                                "goal_verified": None})
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
            roles[role] = {"provider": binding.provider, "model": binding.model, "models": list(binding.model_pool),
                           "enabled": binding.enabled,
                           "configured": configured, "cost_class": self.config.cost_class(role).value,
                           "health": self.health.status(binding.provider).value if provider else "unknown",
                           "offline_fallback": binding.offline_fallback}
        providers = {}
        for name, provider in self.config.providers.items():
            credential = self.credentials.has(provider.secret) if provider.secret else True
            health = self.health.status(name)
            # CONFIGURED / UNCONFIGURED is about the owner's setup; health is
            # about the provider's answers.  A missing key is not BROKEN.
            if not provider.enabled:
                state = "DISABLED"
            elif not credential:
                state = "UNCONFIGURED"
            elif health.value == "authentication_error":
                state = "KEY_REJECTED"
            elif health.is_outage:
                state = health.value.upper()
            else:
                state = "CONFIGURED"
            providers[name] = {"kind": provider.kind, "enabled": provider.enabled, "metered": provider.metered,
                               "may_train_on_requests": provider.may_train_on_requests, "credential": credential,
                               "credential_env": list(provider.credential_env), "state": state,
                               "health": health.value, "base_url": provider.base_url, "purpose": provider.purpose,
                               "pricing": {m: p.to_eur_dict() for m, p in provider.pricing.items()}}
        return {
            "spend": spend, "roles": roles, "providers": providers, "credentials": self.credentials.status(),
            "modes": {m.value: {"allow_metered": p.allow_metered, "hint_de": p.owner_hint_de, "hint_en": p.owner_hint_en,
                                "task_cap_eur": p.task_cap_eur} for m, p in MODE_POLICIES.items()},
            "budget": self.config.budget.to_dict(), "config_source": self.config.source,
            "exchange_rates": {code: rate.to_dict() for code, rate in self.config.exchange_rates.items()},
            "exchange_rate_note": ("owner-configured rates in use" if self.config.exchange_rates else
                                   "no exchange rate configured: foreign-currency EUR conversion is unavailable; metered estimates use an unconfirmed budget guard"),
            "thinking_levels": list(THINKING_LEVELS),
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
        #: Who produced the most recent generation: role, provider, model, and
        #: whether it was the offline fallback.  Reset at the start of every
        #: generation, so it never describes an earlier answer.
        self.last_provenance: dict[str, Any] = {}

    @property
    def provenance(self) -> dict[str, Any]:
        """The execution receipt of the last generation, for the transcript's backend label."""

        return dict(self.last_provenance)

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
                              soft=dict(context.soft), overrides=False, owner_text=context.owner_text, output_budget=context.output_budget)

    def _run(self, prompt: str, *, schema: dict[str, Any] | None = None, max_tokens: int | None = None,
             temperature: float | None = None, system: str | None = None) -> str:
        request = self._request(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature, system=system)
        self.last_reply = None
        self.last_provenance = {}
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
            if getattr(exc, "typed_status", "") == "FREE_INTELLIGENCE_UNAVAILABLE":
                raise
            if self.fallback is None or not self._fallback_allowed() or not exc.status.is_outage:
                raise
            return self._fallback(prompt, schema=schema, max_tokens=max_tokens, temperature=temperature, system=system,
                                  why=exc.status.value)
        self._note_reply(reply)
        return reply.text

    def _note_reply(self, reply: GatewayReply) -> None:
        self.last_reply = reply
        self.last_decision = reply.decision.to_dict()
        self.last_provenance = {"role": reply.role, "provider": reply.provider, "model": reply.model,
                                "offline_fallback": bool(reply.decision.offline_fallback), "route_attempts": list(reply.route_attempts),
                                "actual_eur": reply.actual_eur, "estimated_eur": reply.estimated_eur, "usage": dict(reply.usage),
                                "latency_seconds": round(reply.latency_seconds, 3), "streamed": False, **reply.completion()}
        self.last_metadata = {
            "latency_seconds": reply.latency_seconds, "generated_tokens": reply.usage.get("output_tokens"),
            "prompt_tokens": reply.usage.get("input_tokens"), "role": reply.role, "provider": reply.provider, "model": reply.model,
            "estimated_eur": reply.estimated_eur, "actual_eur": reply.actual_eur, "offline_fallback": reply.decision.offline_fallback,
        }

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
        self._note_fallback(why)
        return str(text)

    def _note_fallback(self, why: str) -> None:
        self.last_metadata = dict(getattr(self.fallback, "last_metadata", {}) or {})
        self.last_metadata.update({"offline_fallback": True, "fallback_reason": why})
        self.last_decision = {**self.last_decision, "fell_back_to_local": True, "fallback_reason": why}
        self.last_provenance = {"role": "local.fast", "provider": str(getattr(self.fallback, "provider_name", "") or "local"),
                                "model": str(getattr(self.fallback, "model_name", "") or ""), "offline_fallback": True,
                                "fallback_reason": why, "actual_eur": 0.0}

    def _fallback_stream(self, prompt: str, *, max_tokens: int | None, temperature: float | None, system: str | None, why: str) -> Iterator[str]:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if hasattr(self.fallback, "generate_stream"):
            try:
                iterator = self.fallback.generate_stream(prompt, system=system, **kwargs) if system is not None else self.fallback.generate_stream(prompt, **kwargs)
            except TypeError:
                iterator = self.fallback.generate_stream(f"{system}\n\n{prompt}" if system else prompt, **kwargs)
            for chunk in iterator:
                yield str(chunk)
        else:
            yield self._fallback(prompt, schema=None, max_tokens=max_tokens, temperature=temperature, system=system, why=why)
            return
        self._note_fallback(why)

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
        """The answer as the provider produces it -- real streaming, provider-independent.

        The route, budget and reservation are decided before the first byte;
        the first text reaches the caller as soon as the provider sends it.
        When the gateway refuses (or the provider is down before it started)
        the local fallback streams instead, if the request context allows it;
        provenance names whichever answered.
        """

        request = self._request(prompt, schema=None, max_tokens=max_tokens, temperature=temperature, system=system)
        self.last_reply = None
        self.last_provenance = {}
        try:
            stream = self.gateway.stream(request)
        except GatewayRefused as exc:
            self.last_decision = exc.decision.to_dict()
            if self.fallback is None or not self._fallback_allowed():
                raise
            yield from self._fallback_stream(prompt, max_tokens=max_tokens, temperature=temperature, system=system, why=exc.decision.reason)
            return
        iterator = iter(stream)
        started = False
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            except GatewayError as exc:
                self.last_decision = {"kind": "provider_error", **exc.to_dict()}
                if started or getattr(exc, "typed_status", "") == "FREE_INTELLIGENCE_UNAVAILABLE":
                    raise
                if self.fallback is None or not self._fallback_allowed() or not exc.status.is_outage:
                    raise
                yield from self._fallback_stream(prompt, max_tokens=max_tokens, temperature=temperature, system=system, why=exc.status.value)
                return
            started = True
            yield chunk
        if stream.reply is not None:
            self._note_reply(stream.reply)
            self.last_provenance["streamed"] = True

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
