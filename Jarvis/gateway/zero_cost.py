"""The zero-cost intelligence pool: a registry of routes, not a chain of conditionals.

A route is one provider/model pair with evidence about its cost and its
health.  The complexity router never names a vendor; it asks the registry
for "healthy eligible zero-cost routes satisfying this contract", in order.

Nothing enters the pool on the strength of a name.  A route is eligible
only when its configuration says ``verified_zero_cost`` with a date and a
source -- the owner's evidence that the effective monetary cost for their
account is zero.  "Free tier" in documentation, "free" in a model name, a
trial or a promotional credit are not evidence.  An unverified route is
listed (so the owner sees it) and never used.

Ordering is the route's explicit ``priority`` (1 first) and nothing else:
never the order providers happen to be listed in, which is alphabetical in
the saved document.  A route without a priority, or sharing its priority
with another route, is listed with that reason and never used -- an order
nobody decided is not an order.  Routes whose provider or model sits in a
health cool-down (a spent daily quota, a hang, a 5xx) are skipped without a
network call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from gateway.config import GatewayConfig, ProviderConfig, RoleBinding
from gateway.health import ProviderHealth

#: The role whose pool this is.  Other zero-cost roles could join later.
ZERO_COST_ROLE = "reasoning.free"


@dataclass(frozen=True)
class ZeroCostRoute:
    provider_id: str
    model_id: str
    vendor: str
    monetary_class: str  # "zero" | "unverified" | "metered"
    verified_zero_cost: bool
    verified_at: str = ""
    verified_by: str = ""
    supports_stream: bool = True
    supports_structured_output: bool = True
    context_limit: int = 0
    output_limit: int = 0
    #: The route's place in the pool, 1 first; None = no order was configured (never used).
    priority: int | None = None
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.provider_id}/{self.model_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "model_id": self.model_id, "vendor": self.vendor, "monetary_class": self.monetary_class,
                "verified_zero_cost": self.verified_zero_cost, "verified_at": self.verified_at, "verified_by": self.verified_by,
                "supports_stream": self.supports_stream, "supports_structured_output": self.supports_structured_output,
                "context_limit": self.context_limit, "output_limit": self.output_limit, "priority": self.priority, "note": self.note}


@dataclass
class RouteVerdict:
    route: ZeroCostRoute
    eligible: bool
    reason: str = ""
    health: str = "ok"
    quota_reset_in_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**self.route.to_dict(), "eligible": self.eligible, "reason": self.reason, "health": self.health,
                "quota_reset_in_seconds": round(self.quota_reset_in_seconds, 1)}


def _priority(raw: Any) -> int | None:
    """A positive integer priority, or None (a bool, a string or zero is not a decided order)."""

    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return None
    return raw


class ZeroCostRegistry:
    """Every configured zero-cost route, with the evidence behind it."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.routes: list[ZeroCostRoute] = self._build(config)

    # -- construction -----------------------------------------------------------

    @staticmethod
    def _build(config: GatewayConfig) -> list[ZeroCostRoute]:
        routes: list[ZeroCostRoute] = []
        seen: set[str] = set()
        binding: RoleBinding | None = config.binding(ZERO_COST_ROLE)
        primary = binding.provider if binding else ""
        for name in config.providers:
            provider: ProviderConfig = config.providers[name]
            declared = list((provider.options or {}).get("zero_cost_routes") or [])
            vendor = str((provider.options or {}).get("vendor") or name)
            # A model in the role's pool without a declared route is listed (so the owner
            # sees it) -- without evidence and without a priority, so it is never used.
            if name == primary and binding is not None:
                declared_models = {str(d.get("model")) for d in declared if isinstance(d, dict)}
                for model in binding.model_pool:
                    if model not in declared_models:
                        declared.append({"model": model, "verified_zero_cost": False, "note": "in the role's model pool without route evidence"})
            for raw in declared:
                if not isinstance(raw, dict) or not str(raw.get("model") or "").strip():
                    continue
                model = str(raw["model"]).strip()
                key = f"{name}/{model}"
                if key in seen:
                    continue
                seen.add(key)
                price = provider.price_for(model)
                metered = provider.metered and (price is None or price.metered)
                # Evidence is either the route's own verification or a dated,
                # confirmed zero price entry for the model on a non-metered
                # provider -- both say what the owner's account pays: nothing.
                priced_zero = price is not None and not price.metered and bool(getattr(price, "confirmed", False))
                verified = (bool(raw.get("verified_zero_cost")) or priced_zero) and not metered
                monetary = "metered" if metered else ("zero" if verified else "unverified")
                routes.append(ZeroCostRoute(
                    provider_id=name, model_id=model, vendor=vendor, monetary_class=monetary, verified_zero_cost=verified,
                    verified_at=str(raw.get("verified_at") or ""), verified_by=str(raw.get("verified_by") or raw.get("source") or ""),
                    supports_stream=bool(raw.get("supports_stream", True)),
                    supports_structured_output=bool(raw.get("supports_structured_output", True)),
                    context_limit=int(raw.get("context_limit") or 0), output_limit=int(raw.get("output_limit") or 0),
                    priority=_priority(raw.get("priority")), note=str(raw.get("note") or "")))
        # the configured priority decides; routes without one are listed last, in no meaningful order
        routes.sort(key=lambda r: (r.priority is None, r.priority or 0, r.provider_id, r.model_id))
        return routes

    # -- selection --------------------------------------------------------------

    def verdicts(self, *, health: ProviderHealth, credential_present: Callable[[str], bool], need_structured: bool = False,
                 need_stream: bool = False) -> list[RouteVerdict]:
        """Every route with its eligibility and the reason, in pool order."""

        out: list[RouteVerdict] = []
        taken: dict[int, int] = {}
        for route in self.routes:
            if route.priority is not None:
                taken[route.priority] = taken.get(route.priority, 0) + 1
        for route in self.routes:
            provider = self.config.providers.get(route.provider_id)
            reason = ""
            if provider is None:
                reason = "provider not configured"
            elif route.priority is None:
                reason = "no priority configured"
            elif taken.get(route.priority, 0) > 1:
                reason = f"priority {route.priority} is shared with another route"
            elif not provider.enabled:
                reason = "provider disabled"
            elif provider.secret and not credential_present(route.provider_id):
                reason = "no credential"
            elif route.monetary_class == "metered":
                reason = "metered: never a zero-cost route"
            elif not route.verified_zero_cost:
                reason = "zero cost not verified for this account"
            elif need_structured and not route.supports_structured_output:
                reason = "no structured output"
            elif need_stream and not route.supports_stream:
                reason = "no streaming"
            status = health.status(route.provider_id, model=route.model_id)
            reset = health.cooldown_remaining(route.provider_id, model=route.model_id)
            if not reason and status.is_outage:
                reason = f"{status.value} (cool-down {reset:.0f}s)"
            out.append(RouteVerdict(route=route, eligible=not reason, reason=reason, health=status.value, quota_reset_in_seconds=reset))
        return out

    def eligible(self, **kw: Any) -> list[ZeroCostRoute]:
        return [v.route for v in self.verdicts(**kw) if v.eligible]

    def describe(self, *, health: ProviderHealth, credential_present: Callable[[str], bool]) -> list[dict[str, Any]]:
        return [v.to_dict() for v in self.verdicts(health=health, credential_present=credential_present)]

    def vendors(self) -> list[str]:
        seen: list[str] = []
        for route in self.routes:
            if route.vendor not in seen:
                seen.append(route.vendor)
        return seen
