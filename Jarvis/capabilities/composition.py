"""Composition engine: registry + world state + semantic goal -> a plan, or proof of a gap.

    text, world state
        -> SemanticGoalDeriver (closed vocabulary from the contracts; gateway)
        -> CompositionPlanner (effects/preconditions search, scored)
        -> CompositionResult: PLAN | MISSING_CAPABILITY | CLARIFY | UNAVAILABLE | NONE

The engine owns the translation from registry manifests to planner nodes,
the reliability figures (health plus what the gateway has learned about a
capability's verified runs), the context metrics the brief asks for, and
the engineering brief a missing capability produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from capabilities.contracts import SemanticContract
from capabilities.planner import CapabilityNode, CompositionPlanner, MissingCapabilityEvidence, Plan, PlanReport, WorldState, plans_for_goals
from capabilities.semantic_goals import CapabilityCard, GoalDerivation, SemanticGoalDeriver
from catalog.metrics import ContextMetrics


def nodes_from_registry(registry: Any, *, reliability: Callable[[str], float | None] | None = None) -> list[CapabilityNode]:
    nodes: list[CapabilityNode] = []
    for manifest in registry.all():
        try:
            contract = manifest.semantic_contract()
        except Exception:  # noqa: BLE001 - a manifest that cannot describe itself is not plannable
            continue
        learned = reliability(manifest.capability_id) if reliability is not None else None
        nodes.append(CapabilityNode(
            capability_id=manifest.capability_id, contract=contract, health=manifest.health_state().value,
            reliability=learned, failure_count=int(getattr(manifest, "failure_count", 0) or 0), active=manifest.is_active(),
        ))
    return nodes


def cards_from_registry(registry: Any) -> list[CapabilityCard]:
    cards: list[CapabilityCard] = []
    for manifest in registry.all():
        if not manifest.is_active() or manifest.is_broken():
            continue
        try:
            contract = manifest.semantic_contract()
        except Exception:  # noqa: BLE001
            continue
        description = str(manifest.description or "").split("\n")[0][:160]
        cards.append(CapabilityCard(manifest.capability_id, contract, description, tuple(manifest.examples[:4]), manifest.name))
    return cards


def manifest_tokens_text(nodes: Iterable[CapabilityNode]) -> str:
    """What the planner reasoned over, rendered so it can be measured."""

    rows = []
    for node in nodes:
        rows.append({"id": node.capability_id, "health": node.health, **node.contract.to_dict()})
    return json.dumps(rows, ensure_ascii=False)


@dataclass
class CompositionResult:
    status: str  # PLAN | MISSING_CAPABILITY | CLARIFY | UNAVAILABLE | NONE
    derivation: GoalDerivation
    report: PlanReport | None = None
    metrics: ContextMetrics = field(default_factory=ContextMetrics)
    state: WorldState | None = None

    @property
    def plan(self) -> Plan | None:
        return self.report.best if self.report else None

    @property
    def missing(self) -> MissingCapabilityEvidence | None:
        return self.report.missing if self.report else None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "derivation": self.derivation.to_dict(),
                "report": self.report.to_dict() if self.report else None, "metrics": self.metrics.to_dict(),
                "state": self.state.to_dict() if self.state else None}


class CompositionEngine:
    def __init__(self, registry: Any, gateway: Any, *, reliability: Callable[[str], float | None] | None = None,
                 max_depth: int = 6) -> None:
        self.registry = registry
        self.gateway = gateway
        self._reliability = reliability
        self.max_depth = max_depth

    def nodes(self) -> list[CapabilityNode]:
        return nodes_from_registry(self.registry, reliability=self._reliability)

    def planner(self) -> CompositionPlanner:
        return CompositionPlanner(self.nodes(), max_depth=self.max_depth)

    def has_declared_contracts(self) -> bool:
        return any(not node.contract.inferred and node.contract.declared for node in self.nodes() if node.usable)

    def plan_for_goals(self, goals: Iterable[str], state: WorldState) -> PlanReport:
        return plans_for_goals(self.planner(), state, goals)

    def derive(self, text: str, state: WorldState, *, mode: Any = None, task_id: str = "") -> GoalDerivation:
        deriver = SemanticGoalDeriver(self.gateway, cards_from_registry(self.registry))
        return deriver.derive(text, state, mode=mode, task_id=task_id)

    def derive_and_plan(self, text: str, state: WorldState, *, mode: Any = None, task_id: str = "") -> CompositionResult:
        metrics = ContextMetrics()
        nodes = self.nodes()
        metrics.add("manifest", manifest_tokens_text(nodes), note=f"{len(nodes)} capability contract(s)")
        derivation = self.derive(text, state, mode=mode, task_id=task_id)
        metrics.add("other", text, note="owner request")
        if derivation.status != "DERIVED":
            return CompositionResult(status=derivation.status, derivation=derivation, metrics=metrics, state=state)
        report = plans_for_goals(CompositionPlanner(nodes, max_depth=self.max_depth), state, derivation.goals)
        status = "PLAN" if report.plans else "MISSING_CAPABILITY"
        return CompositionResult(status=status, derivation=derivation, report=report, metrics=metrics, state=state)


def engineering_brief(request: str, evidence: MissingCapabilityEvidence, *, goals: Iterable[str] = ()) -> str:
    """The missing capability, stated exactly, for the engineer and its EngineeringSpec."""

    lines = [request.strip(), "", "MISSING CAPABILITY -- proven by the composition planner:",
             f"- semantic goal(s): {', '.join(goals) or ', '.join(evidence.goal)}",
             f"- world state at the time: {', '.join(sorted(evidence.state.facts)) or '(nothing relevant)'}"]
    if evidence.closest_partial_plan and evidence.closest_partial_plan.steps:
        chain = " -> ".join(evidence.closest_partial_plan.capability_ids)
        reached = ", ".join(sorted(evidence.closest_partial_plan.end.facts - evidence.state.facts)) or "nothing new"
        lines.append(f"- existing capabilities already reach: {chain} (producing {reached})")
    else:
        lines.append("- no existing capability applies to this state")
    lines.append(f"- missing effect(s) nobody produces: {', '.join(evidence.missing_effects) or '(the goal is an unmet input)'}")
    if evidence.unmet_inputs:
        lines.append(f"- unmet input(s) nobody produces: {', '.join(evidence.unmet_inputs)}")
    lines += ["",
              "The new capability MUST ship a contract.json next to main.py declaring its semantic contract:",
              json.dumps({"goals": list(goals) or list(evidence.goal),
                          "consumes": sorted(evidence.closest_partial_plan.end.facts - evidence.state.facts)[:3] if evidence.closest_partial_plan else [],
                          "produces": list(evidence.missing_effects), "effects": [],
                          "risk_class": "harmless", "latency_class": "fast", "cost_class": "free"}, ensure_ascii=False),
              "Do not rebuild what the existing capabilities above already do; consume their outputs."]
    return "\n".join(lines)
