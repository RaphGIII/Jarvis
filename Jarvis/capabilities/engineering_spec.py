"""EngineeringSpec: exactly what is missing, for an engineer that has not been called yet.

Built when a validated plan cannot reach the owner's goal.  It carries the
owner's words, the GoalSpec the provider derived, what existing capabilities
already cover, the closest partial plan, the exact missing effects, the
capabilities the new one should reuse, the interfaces it will meet, the
permissions it needs, acceptance criteria, privacy constraints and a minimal
slice of the ZEUS Catalog -- and nothing else.  An engineer is called with
this only when the engineering router has chosen one; otherwise the spec is
kept on disk and the owner is told.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from capabilities.planner import MissingCapabilityEvidence
from catalog.metrics import ContextMetrics


@dataclass
class EngineeringSpec:
    spec_id: str
    owner_goal: str
    goal_spec: dict[str, Any]
    existing_coverage: list[dict[str, Any]]
    closest_partial_plan: list[str]
    missing_effects: list[str]
    reusable_capabilities: list[str]
    relevant_interfaces: list[str]
    required_permissions: list[str]
    acceptance_criteria: list[str]
    privacy_constraints: list[str]
    catalog_context: str
    suggested_contract: dict[str, Any]
    task_class: str = "engineering.medium"
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id, "owner_goal": self.owner_goal, "goal_spec": dict(self.goal_spec),
            "existing_coverage": list(self.existing_coverage), "closest_partial_plan": list(self.closest_partial_plan),
            "missing_effects": list(self.missing_effects), "reusable_capabilities": list(self.reusable_capabilities),
            "relevant_interfaces": list(self.relevant_interfaces), "required_permissions": list(self.required_permissions),
            "acceptance_criteria": list(self.acceptance_criteria), "privacy_constraints": list(self.privacy_constraints),
            "catalog_context": self.catalog_context, "suggested_contract": dict(self.suggested_contract),
            "task_class": self.task_class, "metrics": dict(self.metrics), "created_at": self.created_at,
        }

    def to_brief(self) -> str:
        """The brief an engineer receives: the spec, in prose an engineer can act on."""

        lines = [self.owner_goal.strip(), "",
                 "MISSING CAPABILITY -- proven by the composition planner (EngineeringSpec " + self.spec_id + "):",
                 f"- semantic goal: {self.goal_spec.get('primary_goal', '')}"
                 + (f" (+ {', '.join(self.goal_spec.get('secondary_goals', []))})" if self.goal_spec.get("secondary_goals") else ""),
                 f"- relevant project: {self.goal_spec.get('relevant_project') or '(none)'}",
                 f"- recent events: {', '.join(self.goal_spec.get('relevant_recent_events', [])) or '(none)'}"]
        if self.closest_partial_plan:
            lines.append("- existing capabilities already reach: " + " -> ".join(self.closest_partial_plan))
        else:
            lines.append("- no existing capability applies to this state")
        lines.append(f"- missing effect(s) nobody produces: {', '.join(self.missing_effects) or '(unknown)'}")
        if self.reusable_capabilities:
            lines.append("- reuse, do not rebuild: " + ", ".join(self.reusable_capabilities))
        for row in self.existing_coverage[:8]:
            lines.append(f"  * {row['capability_id']}: produces {', '.join(row.get('produces', [])) or '-'}; needs {', '.join(row.get('requires', [])) or '-'}")
        if self.relevant_interfaces:
            lines.append("- interfaces to meet: " + "; ".join(self.relevant_interfaces))
        if self.required_permissions:
            lines.append("- permissions: " + ", ".join(self.required_permissions))
        if self.privacy_constraints:
            lines.append("- privacy: " + "; ".join(self.privacy_constraints))
        lines.append("- acceptance criteria:")
        lines += [f"  {i + 1}. {c}" for i, c in enumerate(self.acceptance_criteria)]
        lines += ["", "The new capability MUST ship a contract.json next to main.py:", json.dumps(self.suggested_contract, ensure_ascii=False)]
        if self.catalog_context:
            lines += ["", "ZEUS CATALOG CONTEXT (minimal):", self.catalog_context]
        return "\n".join(lines)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.spec_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def build_engineering_spec(owner_goal: str, goal_spec: dict[str, Any], evidence: MissingCapabilityEvidence,
                           cards: Iterable[Any], *, catalog_budget_chars: int = 6_000, repo: Path | None = None,
                           flow_metrics: dict[str, Any] | None = None) -> EngineeringSpec:
    cards = list(cards)
    by_id = {c.capability_id: c for c in cards}
    partial_ids = evidence.closest_partial_plan.capability_ids if evidence.closest_partial_plan else []
    reached = sorted(evidence.closest_partial_plan.end.facts - evidence.state.facts) if evidence.closest_partial_plan else []
    coverage = []
    for cid in partial_ids:
        card = by_id.get(cid)
        if card is None:
            continue
        coverage.append({"capability_id": cid, "produces": sorted(card.contract.provides), "requires": sorted(card.contract.requires),
                         "health": getattr(card, "health", "")})
    missing = [m for m in evidence.missing_effects if not m.startswith("goal.")] or list(evidence.unproducible_effects) \
        or [m[5:] for m in evidence.missing_effects if m.startswith("goal.")]
    consumes = [f for f in reached if not f.startswith("goal.")][:3]
    permissions = sorted({p for cid in partial_ids for p in by_id[cid].contract.permissions} if partial_ids else set())
    privacy = []
    if str(goal_spec.get("privacy_class", "public")) != "public":
        privacy.append(f"the request is {goal_spec.get('privacy_class')}: no data leaves the machine except through configured no-training providers")
    privacy.append("never print secrets; never read outside the named inputs")
    acceptance = [f"contract.json declares produces: {', '.join(m for m in missing if not m.startswith('goal.')) or missing[0] if missing else 'the missing effect'}"]
    if consumes:
        acceptance.append(f"run(payload) accepts the outputs of {', '.join(partial_ids[-1:])} ({', '.join(consumes)}) as its input")
    acceptance += ["run(payload) returns {'ok': True, ...} on success and {'ok': False, 'error': ...} on failure, never raises",
                   "test_capability.py exercises the real behaviour, not a placeholder",
                   "the four standard capability checks pass (tests, contract, implemented, static)"]
    interfaces = ["run(payload: dict) -> dict with 'ok'", "INPUT_SCHEMA: JSON schema of payload", "contract.json (SemanticContract)"]
    catalog_context = ""
    metrics = ContextMetrics()
    try:
        from catalog.context import build_context

        ctx = build_context(["capabilities/contracts.py", "capabilities/service.py"], request=owner_goal[:200], repo=repo,
                            budget_chars=catalog_budget_chars)
        catalog_context = ctx.text
        metrics.add("catalog", catalog_context, note="minimal catalog context")
    except Exception:  # noqa: BLE001 - the catalog is help, not a requirement
        catalog_context = ""
    words = len(owner_goal.split())
    task_class = "engineering.large" if len(missing) >= 3 or words > 70 else ("engineering.medium" if partial_ids or len(missing) > 1 else "engineering.small")
    suggested = {"goals": [g for g in [goal_spec.get("primary_goal", "")] + list(goal_spec.get("secondary_goals", [])) if g and not g.startswith("goal.")],
                 "consumes": consumes, "produces": [m for m in missing if not m.startswith("goal.")], "effects": [],
                 "risk_class": "harmless", "latency_class": "fast", "cost_class": "free"}
    return EngineeringSpec(
        spec_id=uuid.uuid4().hex[:10], owner_goal=owner_goal, goal_spec=dict(goal_spec), existing_coverage=coverage,
        closest_partial_plan=partial_ids, missing_effects=missing, reusable_capabilities=partial_ids,
        relevant_interfaces=interfaces, required_permissions=permissions, acceptance_criteria=acceptance,
        privacy_constraints=privacy, catalog_context=catalog_context, suggested_contract=suggested, task_class=task_class,
        metrics={**metrics.to_dict(), **({"flow": flow_metrics} if flow_metrics else {})},
    )
