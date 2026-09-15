"""Choosing the route before anything runs.

Rules first, then arithmetic.  The hard overrides in :func:`hard_override`
settle the cases where scoring has no business: a deterministic capability
answers locally, a sensitive request never sees the free lane, a missing
capability is an engineering question, a destructive action is the owner's
decision, too much ambiguity is a clarification -- not a bigger model.

For everything that does need a model, the router selects

    m* = argmin Cost(m, x)   subject to   q(m, x) >= tau(x)

over the roles the chat mode permits, the privacy decision allows, the
provider health admits and the budget can reserve.  ``q`` is the reliability
model's conservative estimate, ``tau`` the task's required reliability.  When
nothing meets the bar, the best available candidate is chosen and the decision
says so; the router never runs a cheaper model first to find out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gateway.budget import BudgetGovernor
from gateway.config import GatewayConfig, RoleBinding
from gateway.estimate import CostEstimate, estimate_cost
from gateway.health import ProviderHealth
from gateway.intelligence_class import ClassDecision, IntelligenceClass
from gateway.learning import ReliabilityModel
from gateway.modes import ChatMode, CostClass, RoleFamily, policy_for
from gateway.privacy import PrivacyDecision, Sensitivity
from gateway.task import TaskClass, TaskVector


class RouteKind(str, Enum):
    #: Run a model in the chosen role.
    MODEL = "model"
    #: A deterministic capability answers; no model is needed.
    DETERMINISTIC = "deterministic"
    #: The request needs engineering (a capability is missing or broken).
    ENGINEERING = "engineering"
    #: A destructive or security-sensitive action: the owner decides first.
    AUTHORIZE = "authorize"
    #: Too ambiguous to act on; ask one good question.
    CLARIFY = "clarify"
    #: Current information is needed: the research/web capability, not a model's memory.
    RESEARCH = "research"
    #: Existing capabilities compose into the goal; do not build a duplicate.
    COMPOSE = "compose"
    #: Nothing permitted can serve this request in this mode.
    REFUSED = "refused"


@dataclass
class Candidate:
    role: str
    binding: RoleBinding
    provider: str
    cost_class: CostClass
    estimate: CostEstimate | None
    q: float
    reason: str = ""
    eligible: bool = True

    @property
    def cost(self) -> float:
        return 0.0 if self.estimate is None else self.estimate.estimated_eur

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "provider": self.provider, "model": self.binding.model, "cost_class": self.cost_class.value,
                "estimated_eur": self.cost, "q": round(self.q, 4), "eligible": self.eligible, "reason": self.reason}


@dataclass
class RouteDecision:
    kind: RouteKind
    mode: ChatMode
    task: TaskVector
    role: str = ""
    provider: str = ""
    model: str = ""
    cost_class: CostClass = CostClass.ZERO
    estimate: CostEstimate | None = None
    thinking_level: str = ""
    #: The output budget decided for this call (level, tokens, reason), before the call.
    output_budget: dict[str, Any] = field(default_factory=dict)
    q: float = 0.0
    tau: float = 0.0
    meets_threshold: bool = True
    offline_fallback: bool = False
    hard_override: str = ""
    reason: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    #: What the owner could change to unlock a refused route.
    suggestion: str = ""
    #: The intelligence class decided BEFORE provider routing (§1-3): the
    #: request's requirement, recorded before any generation.
    intelligence_class: str = ""
    class_decision: dict[str, Any] = field(default_factory=dict)
    #: AUTO only: the one guarded paid generation after the zero-cost pool
    #: was genuinely exhausted.  Never a ladder.
    emergency: bool = False

    @property
    def runs_model(self) -> bool:
        return self.kind is RouteKind.MODEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "mode": self.mode.value, "role": self.role, "provider": self.provider, "model": self.model,
            "cost_class": self.cost_class.value, "estimate": self.estimate.to_dict() if self.estimate else None,
            "thinking_level": self.thinking_level, "output_budget": dict(self.output_budget), "q": round(self.q, 4), "tau": round(self.tau, 4),
            "meets_threshold": self.meets_threshold, "offline_fallback": self.offline_fallback,
            "hard_override": self.hard_override, "reason": self.reason, "suggestion": self.suggestion,
            "candidates": [c.to_dict() for c in self.candidates], "task": self.task.to_dict(),
            "intelligence_class": self.intelligence_class, "class_decision": dict(self.class_decision), "emergency": self.emergency,
        }


#: Above this, the request is a clarification rather than a model call.
AMBIGUITY_CLARIFY_THRESHOLD = 0.8
#: A registry hit at or above this confidence answers deterministically.
DETERMINISTIC_CONFIDENCE = 0.85


def thinking_level_for(task: TaskVector, mode: ChatMode | None = None) -> str:
    """FAST / NORMAL / DEEP from the task; MAX only in DEEP mode for the hardest tasks."""

    depth = max(task.reasoning_depth, task.long_horizon, task.integration_breadth)
    if depth < 0.3:
        return "FAST"
    if depth < 0.65:
        return "NORMAL"
    if mode is ChatMode.DEEP and depth >= 0.85:
        return "MAX"
    return "DEEP"


def hard_override(task: TaskVector, privacy: PrivacyDecision | None) -> tuple[RouteKind | None, str]:
    """The rules that outrank scoring.  Returns (kind, why) or (None, "")."""

    facts = task.facts
    if facts.get("destructive") or facts.get("security_sensitive"):
        return RouteKind.AUTHORIZE, "destructive or security-sensitive action: the owner authorizes first"
    if facts.get("capability_found") and float(facts.get("capability_confidence", 0.0)) >= DETERMINISTIC_CONFIDENCE \
            and not facts.get("capability_broken"):
        return RouteKind.DETERMINISTIC, "an exact capability matches with high confidence: local, no model"
    if facts.get("capability_broken"):
        return RouteKind.ENGINEERING, "the matching capability is BROKEN: engineering repair"
    if task.ambiguity >= AMBIGUITY_CLARIFY_THRESHOLD:
        return RouteKind.CLARIFY, "ambiguity too high: one clarification instead of a bigger model"
    if int(facts.get("subsystems", 0) or 0) >= 2 and not facts.get("is_engineering") and not facts.get("capability_missing"):
        return RouteKind.COMPOSE, "existing capabilities compose into the goal"
    if facts.get("capability_missing") and not facts.get("is_engineering"):
        return RouteKind.ENGINEERING, "no capability can satisfy the goal: engineering classification"
    if task.fresh_information_required >= 0.99:
        return RouteKind.RESEARCH, "current information is needed: research, not recall"
    return None, ""


class ModelRouter:
    def __init__(
        self,
        config: GatewayConfig,
        reliability: ReliabilityModel,
        governor: BudgetGovernor,
        health: ProviderHealth,
        *,
        credential_present=None,
        local_available=None,
        paid_allowed=None,
        subscription_available=None,
    ) -> None:
        self.config = config
        self.reliability = reliability
        self.governor = governor
        self.health = health
        #: provider name -> bool; whether a credential exists for cloud providers.
        self._credential_present = credential_present or (lambda name: False)
        #: role -> bool; whether the local tier behind a local role is usable.
        self._local_available = local_available or (lambda role: True)
        #: Whether the owner's spending policy permits metered API billing at all.
        self._paid_allowed = paid_allowed or (lambda: False)
        #: provider name -> bool; whether a subscription engineer (Codex) is READY.
        self._subscription_available = subscription_available or (lambda name: False)

    # -- candidates -----------------------------------------------------------

    def _families_for(self, task: TaskVector, mode: ChatMode) -> set[RoleFamily]:
        engineering = task.task_class in {TaskClass.ENGINEERING_SMALL, TaskClass.ENGINEERING_MEDIUM, TaskClass.ENGINEERING_LARGE}
        if engineering:
            # Which engineer roles a mode permits is the mode policy's call
            # (metered engineers need BUILD; the subscription engineer is free
            # in every mode); the family is the same everywhere.
            return {RoleFamily.ENGINEER, RoleFamily.LOCAL}
        return {RoleFamily.REASONING, RoleFamily.LOCAL}

    def candidates(self, task: TaskVector, mode: ChatMode, privacy: PrivacyDecision | None, *, prompt: str,
                   system: str = "", expected_output_tokens: int = 512, task_id: str = "",
                   only_role: str = "") -> list[Candidate]:
        policy = policy_for(mode)
        families = self._families_for(task, mode)
        out: list[Candidate] = []
        paid_allowed = bool(self._paid_allowed())
        for role, binding in self.config.roles.items():
            if only_role and role != only_role:
                continue
            if binding.family not in families and not only_role:
                continue
            if binding.family is RoleFamily.LOCAL and (role == "local.build") != (RoleFamily.ENGINEER in families):
                continue
            provider = self.config.providers.get(binding.provider)
            if provider is None:
                continue
            cost_class = self.config.cost_class(role)
            estimate: CostEstimate | None = None
            pricing = self.config.pricing_for(role)
            if pricing is not None:
                estimate = estimate_cost(prompt=prompt, system=system, expected_output_tokens=expected_output_tokens, pricing=pricing)
            q = self.reliability.q(role, task)
            candidate = Candidate(role=role, binding=binding, provider=binding.provider, cost_class=cost_class, estimate=estimate, q=q)

            if not binding.enabled:
                candidate.eligible, candidate.reason = False, "role disabled"
            elif not provider.enabled:
                candidate.eligible, candidate.reason = False, f"provider {provider.name} disabled"
            elif not policy.permits_role(role):
                candidate.eligible, candidate.reason = False, f"mode {mode.value} does not permit {role}"
            elif cost_class is CostClass.METERED and not policy.allow_metered:
                candidate.eligible, candidate.reason = False, f"mode {mode.value} forbids metered routes"
            elif cost_class is CostClass.METERED and not paid_allowed:
                candidate.eligible, candidate.reason = False, "paid API billing is disabled by the owner spending policy"
            elif provider.kind == "subscription_cli" and not self._subscription_available(provider.name):
                candidate.eligible, candidate.reason = False, f"{provider.name} is not available"
            elif cost_class is CostClass.METERED and pricing is None:
                candidate.eligible, candidate.reason = False, "metered role without a price cannot be estimated"
            elif provider.secret and not self._credential_present(provider.name):
                candidate.eligible, candidate.reason = False, f"no credential for {provider.name}"
            elif binding.family is RoleFamily.LOCAL and not self._local_available(role):
                candidate.eligible, candidate.reason = False, "local model not available"
            elif not self.health.usable(provider.name):
                candidate.eligible, candidate.reason = False, f"provider {provider.name}: {self.health.status(provider.name).value}"
            elif provider.may_train_on_requests and privacy is not None and not privacy.free_lane_allowed:
                candidate.eligible, candidate.reason = False, "sensitive request: may-train provider forbidden"
            elif provider.may_train_on_requests and task.privacy_class >= 0.6:
                candidate.eligible, candidate.reason = False, "private context: may-train provider forbidden"
            elif cost_class is CostClass.METERED and estimate is not None:
                ok, cap, _reserved = self.governor.can_reserve(role=role, provider=provider.name, estimated_eur=estimate.estimated_eur,
                                                               task_id=task_id, task_cap_eur=policy.task_cap_eur,
                                                               provider_cap_eur=provider.monthly_cap_eur)
                if not ok:
                    candidate.eligible, candidate.reason = False, f"budget: {cap}"
            out.append(candidate)
        return out

    # -- the decision ---------------------------------------------------------

    def decide(self, task: TaskVector, mode: ChatMode | str, privacy: PrivacyDecision | None = None, *, prompt: str,
               system: str = "", expected_output_tokens: int = 512, task_id: str = "", only_role: str = "",
               apply_overrides: bool = True, intelligence_class: ClassDecision | None = None) -> RouteDecision:
        """Route.  With ``apply_overrides=False`` the hard rules only annotate the
        decision: the caller has already decided that a model is to be consulted
        (interpretation, summary) and only asks which one.

        ``intelligence_class`` is the requirement decided before this call
        (:mod:`gateway.intelligence_class`).  Candidates outside the class
        are ineligible: a SMART request never tries the zero-cost pool
        first, a DEEP request never runs SMART to see whether it copes.  In
        AUTO a paid class whose roles are not configured at all falls back
        to the zero-cost class (nothing is spent); a class whose routes
        merely failed does not climb."""
        mode = ChatMode.parse(mode)
        tau = task.required_reliability
        if mode is ChatMode.DEEP:
            tau = min(0.97, tau + 0.15)
        decision = RouteDecision(kind=RouteKind.MODEL, mode=mode, task=task, tau=tau)
        if intelligence_class is not None:
            decision.intelligence_class = intelligence_class.intelligence_class.value
            decision.class_decision = intelligence_class.to_dict()

        kind, why = hard_override(task, privacy)
        if kind is not None and not only_role:
            decision.hard_override = kind.value
            if apply_overrides:
                decision.kind, decision.reason = kind, why
                if kind is not RouteKind.MODEL:
                    if kind is RouteKind.DETERMINISTIC:
                        decision.intelligence_class = IntelligenceClass.DETERMINISTIC.value
                    return decision

        candidates = self.candidates(task, mode, privacy, prompt=prompt, system=system,
                                     expected_output_tokens=expected_output_tokens, task_id=task_id, only_role=only_role)
        if intelligence_class is not None and not only_role:
            candidates = self._restrict_to_class(candidates, intelligence_class, mode, decision)
        decision.candidates = candidates
        eligible = [c for c in candidates if c.eligible]
        if not eligible:
            decision.kind = RouteKind.REFUSED
            decision.reason = "no permitted route: " + "; ".join(f"{c.role}: {c.reason}" for c in candidates)[:400]
            decision.suggestion = self._suggestion(mode, candidates)
            return decision

        meeting = [c for c in eligible if c.q >= tau]
        if meeting:
            if mode is ChatMode.DEEP:
                # DEEP is the owner asking for the strongest configured
                # reasoner: the most reliable route that meets the bar, chosen
                # directly -- never a cheaper one first to see whether it copes.
                meeting.sort(key=lambda c: (1 if c.binding.offline_fallback else 0, -c.q, c.cost))
            elif mode is ChatMode.SMART:
                # SMART is the owner choosing inexpensive cloud reasoning: the
                # smart role when it meets the bar, the free tier otherwise.
                meeting.sort(key=lambda c: (0 if c.role == "reasoning.smart" else 1, 1 if c.binding.offline_fallback else 0, c.cost, -c.q))
            else:
                # Cheapest first; among equally cheap, the more reliable one.  A
                # local offline fallback is never preferred over a configured
                # zero-cost cloud role that meets the bar: the brief removes the
                # small local model from the normal path.
                meeting.sort(key=lambda c: (c.cost, 1 if c.binding.offline_fallback else 0, -c.q))
            chosen, decision.meets_threshold = meeting[0], True
        else:
            eligible.sort(key=lambda c: (-c.q, c.cost))
            chosen, decision.meets_threshold = eligible[0], False

        decision.role, decision.provider, decision.model = chosen.role, chosen.provider, chosen.binding.model
        decision.cost_class, decision.estimate, decision.q = chosen.cost_class, chosen.estimate, chosen.q
        decision.offline_fallback = chosen.binding.offline_fallback
        decision.thinking_level = thinking_level_for(task, mode) if chosen.binding.thinking else ""
        cheaper_rejected = [c for c in eligible if c.cost < chosen.cost]
        parts = [f"cheapest route with q={chosen.q:.2f} >= tau={tau:.2f}" if decision.meets_threshold
                 else f"no route meets tau={tau:.2f}; best available q={chosen.q:.2f}"]
        if cheaper_rejected:
            parts.append("skipped cheaper " + ", ".join(f"{c.role} (q={c.q:.2f})" for c in cheaper_rejected))
        if decision.offline_fallback:
            parts.append("offline fallback: no cloud role is configured and permitted")
        decision.reason = "; ".join(parts)
        return decision

    def _restrict_to_class(self, candidates: list[Candidate], cls: ClassDecision, mode: ChatMode, decision: RouteDecision) -> list[Candidate]:
        """Only the class's roles stay eligible; in AUTO an unconfigured paid class degrades to zero cost, never the reverse."""

        wanted = set(cls.intelligence_class.roles)
        if not wanted:
            return candidates

        def inside(role: str) -> bool:
            return role in wanted or (cls.intelligence_class.is_engineering and role.startswith("engineer."))

        class_candidates = [c for c in candidates if inside(c.role)]
        if mode is ChatMode.AUTO and cls.intelligence_class in {IntelligenceClass.SMART, IntelligenceClass.DEEP} and not any(
                c.eligible for c in class_candidates):
            unconfigured = all(any(marker in c.reason for marker in ("no credential", "disabled", "not permit", "billing is disabled"))
                               for c in class_candidates) if class_candidates else True
            if unconfigured:
                lower = IntelligenceClass.ZERO_COST
                decision.class_decision["downgrade"] = (f"{cls.intelligence_class.value} has no configured route; "
                                                        f"{lower.value} answers instead (nothing spent)")
                decision.intelligence_class = lower.value
                wanted = set(lower.roles)
        for candidate in candidates:
            if candidate.eligible and candidate.role not in wanted and not (
                    cls.intelligence_class.is_engineering and candidate.role.startswith("engineer.")):
                candidate.eligible, candidate.reason = False, f"outside the selected intelligence class {decision.intelligence_class}"
        return candidates

    def _suggestion(self, mode: ChatMode, candidates: list[Candidate]) -> str:
        reasons = " ".join(c.reason for c in candidates)
        if mode is ChatMode.FREE and ("metered" in reasons or "does not permit" in reasons):
            return "This needs SMART, DEEP or BUILD. Switch the chat mode to allow a paid route."
        if "owner spending policy" in reasons:
            return "Paid API billing is off. Enable paid_api in Owner Settings > Spending (an owner transaction) to allow it."
        if "no credential" in reasons:
            return "Enter a provider API key under Owner Settings > Providers."
        if "budget" in reasons:
            return "The budget cap for this period is reached. Raise it deliberately or wait for the next period."
        if "disabled" in reasons:
            return "Enable a provider under Owner Settings > Providers."
        return ""
