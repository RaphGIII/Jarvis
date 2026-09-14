"""The capability composition planner: reach a goal with what already exists.

    state S0 -> capability A -> S1 -> capability B -> S2 -> goal satisfied

A world state is a set of facts (tokens).  A capability is applicable when
its contract's ``requires`` (consumes + preconditions) hold; applying it adds
its ``provides`` (produces + effects).  The goal is a set of facts that must
hold at the end.  The planner searches that space exhaustively up to a depth,
keeps every plan that reaches the goal, and ranks them:

    PlanCost = execution_cost + latency_penalty + risk_penalty
             + uncertainty_penalty + number_of_steps_penalty

The shortest plan is not automatically the best: an unreliable or
irreversible step costs more than an extra reliable one.

When no plan reaches the goal, the planner does not shrug.  It computes the
closure of everything reachable from the state, names the goal facts nobody
can produce, and returns the closest partial plan -- the evidence an
engineering request needs to say exactly which capability is missing.

Nothing here knows chess, files or music.  It knows tokens, and only the
registry's manifests may supply them: the planner cannot invent a capability.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from capabilities.contracts import SemanticContract, normalise_token

COST_WEIGHTS: dict[str, float] = {"free": 0.0, "cheap": 1.0, "metered": 3.0}
LATENCY_WEIGHTS: dict[str, float] = {"fast": 0.0, "medium": 1.0, "slow": 2.0}
RISK_WEIGHTS: dict[str, float] = {"harmless": 0.0, "reversible": 1.0, "irreversible": 4.0}
STEP_PENALTY = 0.5
#: Weighted so that an AT_RISK step (reliability ~0.6 or less) costs more than
#: a slow but healthy one: the shortest path is not the best when it is the
#: least reliable.
UNCERTAINTY_WEIGHT = 5.0

#: Words too generic to tie a fact to a goal's domain when judging partial plans.
GENERIC_WORDS: frozenset[str] = frozenset({"result", "record", "updated", "update", "available", "for", "analysis", "data",
                                            "state", "done", "ready", "new", "the", "of", "and", "profile", "output"})


def _words(token: str) -> set[str]:
    return {w for w in token.replace(".", "_").split("_") if w and w not in GENERIC_WORDS}

#: Health -> assumed reliability when no learned figure exists.
HEALTH_RELIABILITY: dict[str, float] = {"HEALTHY": 0.95, "AT_RISK": 0.6, "UNKNOWN": 0.7, "BROKEN": 0.0}


@dataclass(frozen=True)
class WorldState:
    """What holds right now: facts as tokens, plus where they came from."""

    facts: frozenset[str] = frozenset()
    sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, *, facts: Iterable[str] = (), events: Iterable[str] = (), projects: Iterable[str] = (),
              extra: dict[str, str] | None = None) -> "WorldState":
        tokens: dict[str, str] = {}
        for fact in facts:
            token = normalise_token(fact)
            if token:
                tokens[token] = "fact"
        for event in events:
            token = normalise_token(event)
            if token:
                tokens[token] = "event"
        for project in projects:
            token = normalise_token(project)
            if token:
                tokens[f"project.{token}"] = "project"
        for token, source in (extra or {}).items():
            token = normalise_token(token)
            if token:
                tokens[token] = source
        return cls(facts=frozenset(tokens), sources=tokens)

    def with_facts(self, facts: Iterable[str], source: str = "derived") -> "WorldState":
        added = {normalise_token(f): source for f in facts if normalise_token(f)}
        return WorldState(facts=self.facts | frozenset(added), sources={**self.sources, **added})

    def holds(self, facts: Iterable[str]) -> bool:
        return all(normalise_token(f) in self.facts for f in facts)

    def to_dict(self) -> dict[str, Any]:
        return {"facts": sorted(self.facts), "sources": dict(self.sources)}


@dataclass(frozen=True)
class CapabilityNode:
    """A capability as the planner sees it: id, contract, health, reliability."""

    capability_id: str
    contract: SemanticContract
    health: str = "UNKNOWN"
    reliability: float | None = None
    failure_count: int = 0
    active: bool = True

    @property
    def usable(self) -> bool:
        return self.active and self.health != "BROKEN"

    @property
    def assumed_reliability(self) -> float:
        if self.reliability is not None:
            return max(0.0, min(1.0, float(self.reliability)))
        base = HEALTH_RELIABILITY.get(self.health, 0.7)
        return max(0.05, base - 0.1 * min(self.failure_count, 5))


@dataclass(frozen=True)
class PlanCost:
    execution_cost: float = 0.0
    latency_penalty: float = 0.0
    risk_penalty: float = 0.0
    uncertainty_penalty: float = 0.0
    steps_penalty: float = 0.0

    @property
    def total(self) -> float:
        return round(self.execution_cost + self.latency_penalty + self.risk_penalty + self.uncertainty_penalty + self.steps_penalty, 4)

    def __add__(self, other: "PlanCost") -> "PlanCost":
        return PlanCost(self.execution_cost + other.execution_cost, self.latency_penalty + other.latency_penalty,
                        self.risk_penalty + other.risk_penalty, self.uncertainty_penalty + other.uncertainty_penalty,
                        self.steps_penalty + other.steps_penalty)

    def to_dict(self) -> dict[str, Any]:
        return {"execution_cost": round(self.execution_cost, 3), "latency_penalty": round(self.latency_penalty, 3),
                "risk_penalty": round(self.risk_penalty, 3), "uncertainty_penalty": round(self.uncertainty_penalty, 3),
                "steps_penalty": round(self.steps_penalty, 3), "total": self.total}


def step_cost(node: CapabilityNode) -> PlanCost:
    contract = node.contract
    return PlanCost(
        execution_cost=COST_WEIGHTS.get(contract.cost_class, 1.0),
        latency_penalty=LATENCY_WEIGHTS.get(contract.latency_class, 1.0),
        risk_penalty=RISK_WEIGHTS.get(contract.risk_class, 1.0),
        uncertainty_penalty=UNCERTAINTY_WEIGHT * (1.0 - node.assumed_reliability),
        steps_penalty=STEP_PENALTY,
    )


@dataclass(frozen=True)
class PlanStep:
    capability_id: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    cost: PlanCost
    health: str
    reliability: float
    risk_class: str

    def to_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, "requires": list(self.requires), "provides": list(self.provides),
                "cost": self.cost.to_dict(), "health": self.health, "reliability": round(self.reliability, 3),
                "risk_class": self.risk_class}


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]
    goal: frozenset[str]
    start: WorldState
    end: WorldState
    cost: PlanCost

    @property
    def capability_ids(self) -> list[str]:
        return [step.capability_id for step in self.steps]

    @property
    def irreversible_steps(self) -> int:
        return sum(1 for step in self.steps if step.risk_class == "irreversible")

    @property
    def reliability(self) -> float:
        value = 1.0
        for step in self.steps:
            value *= step.reliability
        return round(value, 4)

    def satisfies(self) -> bool:
        return self.goal <= self.end.facts

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps], "capability_ids": self.capability_ids, "goal": sorted(self.goal),
                "cost": self.cost.to_dict(), "reliability": self.reliability, "irreversible_steps": self.irreversible_steps,
                "end_facts": sorted(self.end.facts)}


@dataclass(frozen=True)
class MissingCapabilityEvidence:
    """Proof that no existing composition reaches the goal, and what would."""

    goal: tuple[str, ...]
    state: WorldState
    available_effects: tuple[str, ...]
    reachable_effects: tuple[str, ...]
    missing_effects: tuple[str, ...]
    closest_partial_plan: Plan | None
    #: Facts the partial plan needs that the state lacks (the other reason a
    #: plan may fail: an input nobody provides).
    unmet_inputs: tuple[str, ...] = ()
    #: Goal facts no capability in the registry produces at all -- the ones
    #: only engineering can supply.
    unproducible_effects: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        """``missing_capability`` when something must be built; ``unmet_input``
        when every needed capability exists and only the world is missing an
        input (no game has finished, no file was named)."""

        if self.unproducible_effects:
            return "missing_capability"
        return "unmet_input" if self.unmet_inputs else "missing_capability"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "goal": list(self.goal), "current_state": sorted(self.state.facts),
                "available_effects": list(self.available_effects), "reachable_effects": list(self.reachable_effects),
                "missing_effects": list(self.missing_effects), "unproducible_effects": list(self.unproducible_effects),
                "closest_partial_plan": self.closest_partial_plan.to_dict() if self.closest_partial_plan else None,
                "unmet_inputs": list(self.unmet_inputs)}

    def describe(self) -> str:
        parts = [f"goal {', '.join(self.goal)}"]
        if self.closest_partial_plan and self.closest_partial_plan.steps:
            parts.append("existing path reaches " + " -> ".join(self.closest_partial_plan.capability_ids)
                         + f" ({', '.join(sorted(self.closest_partial_plan.end.facts & set(self.goal)) or ['nothing of the goal'])})")
        else:
            parts.append("no existing capability applies to the current state")
        if self.missing_effects:
            parts.append("missing effect(s): " + ", ".join(self.missing_effects))
        if self.unmet_inputs:
            parts.append("unmet input(s): " + ", ".join(self.unmet_inputs))
        return "; ".join(parts)


@dataclass
class PlanReport:
    goal: frozenset[str]
    state: WorldState
    plans: list[Plan]
    missing: MissingCapabilityEvidence | None
    explored_states: int
    candidates_considered: int

    @property
    def best(self) -> Plan | None:
        return self.plans[0] if self.plans else None

    @property
    def status(self) -> str:
        if self.plans:
            return "PLAN_FOUND"
        if not self.goal:
            return "NO_GOAL"
        return "MISSING_CAPABILITY"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "goal": sorted(self.goal), "state": self.state.to_dict(),
                "best": self.best.to_dict() if self.best else None,
                "alternatives": [p.to_dict() for p in self.plans[1:6]],
                "missing": self.missing.to_dict() if self.missing else None,
                "explored_states": self.explored_states, "candidates_considered": self.candidates_considered}


class CompositionPlanner:
    def __init__(self, nodes: Iterable[CapabilityNode], *, max_depth: int = 6, max_plans: int = 40,
                 reliability: Callable[[str], float | None] | None = None) -> None:
        self.nodes: dict[str, CapabilityNode] = {}
        for node in nodes:
            if reliability is not None and node.reliability is None:
                learned = reliability(node.capability_id)
                if learned is not None:
                    node = CapabilityNode(node.capability_id, node.contract, node.health, learned, node.failure_count, node.active)
            self.nodes[node.capability_id] = node
        self.max_depth = max_depth
        self.max_plans = max_plans

    # -- queries -------------------------------------------------------------

    def available_effects(self) -> list[str]:
        out: set[str] = set()
        for node in self.nodes.values():
            if node.usable:
                out |= node.contract.provides
        return sorted(out)

    def known_goals(self) -> list[str]:
        out: set[str] = set()
        for node in self.nodes.values():
            if node.usable:
                out |= set(node.contract.goals) | node.contract.provides
        return sorted(out)

    def event_tokens(self) -> frozenset[str]:
        """Facts the world is expected to supply (declared as events by some capability)."""

        out: set[str] = set()
        for node in self.nodes.values():
            out |= set(node.contract.events)
        return frozenset(out)

    def producers_of(self, fact: str) -> list[str]:
        token = normalise_token(fact)
        return sorted(cid for cid, node in self.nodes.items() if node.usable and token in node.contract.provides)

    def relevant_facts(self, goal: frozenset[str]) -> frozenset[str]:
        """Backward closure: the goal, what its producers need, what their producers need ..."""

        relevant: set[str] = set(goal)
        frontier = set(goal)
        while frontier:
            fact = frontier.pop()
            for node in self.nodes.values():
                if node.usable and fact in node.contract.provides:
                    for need in node.contract.requires:
                        if need not in relevant:
                            relevant.add(need)
                            frontier.add(need)
        return frozenset(relevant)

    def reachable(self, state: WorldState) -> frozenset[str]:
        """The closure: everything any sequence of usable capabilities could make true."""

        facts = set(state.facts)
        changed = True
        while changed:
            changed = False
            for node in self.nodes.values():
                if node.usable and node.contract.requires <= facts and not node.contract.provides <= facts:
                    facts |= node.contract.provides
                    changed = True
        return frozenset(facts)

    def _domain_chain(self, state: WorldState, goal_words: set[str]) -> Plan | None:
        subject = [node for node in self.nodes.values() if node.usable and any(
            _words(t) & goal_words for t in (*node.contract.provides, *node.contract.goals, node.capability_id))]
        return _chain_from(self, state, subject)

    # -- planning ------------------------------------------------------------

    def plan(self, state: WorldState, goal: Iterable[str]) -> PlanReport:
        goal_facts = frozenset(normalise_token(g) for g in goal if normalise_token(g))
        if not goal_facts:
            return PlanReport(goal_facts, state, [], None, 0, 0)
        if goal_facts <= state.facts:
            empty = Plan(steps=(), goal=goal_facts, start=state, end=state, cost=PlanCost())
            return PlanReport(goal_facts, state, [empty], None, 1, 0)

        # Only capabilities that contribute something the goal (transitively)
        # needs are expanded; a zip archiver has no place in a chess plan.
        relevant = self.relevant_facts(goal_facts)
        usable = [node for node in self.nodes.values() if node.usable and node.contract.provides & relevant]
        plans: list[Plan] = []
        best_total: dict[frozenset[str], float] = {}
        explored = 0
        considered = 0
        partials: list[tuple[int, float, tuple[PlanStep, ...], frozenset[str]]] = []
        bound = float("inf")

        def search(facts: frozenset[str], steps: tuple[PlanStep, ...], used: frozenset[str], cost: PlanCost) -> None:
            nonlocal explored, considered, bound
            explored += 1
            covered = len(goal_facts & facts)
            partials.append((covered, cost.total, steps, facts))
            if goal_facts <= facts:
                plan = Plan(steps=steps, goal=goal_facts, start=state, end=WorldState(facts=facts), cost=cost)
                plans.append(plan)
                if len(plans) >= self.max_plans:
                    bound = min(bound, sorted(p.cost.total for p in plans)[self.max_plans - 1])
                return
            if len(steps) >= self.max_depth:
                return
            for node in usable:
                if node.capability_id in used:
                    continue
                contract = node.contract
                if not contract.requires <= facts or contract.provides <= facts:
                    continue
                considered += 1
                next_cost = cost + step_cost(node)
                if next_cost.total >= bound:
                    continue
                next_facts = facts | contract.provides
                seen = best_total.get(next_facts)
                if seen is not None and seen <= next_cost.total:
                    continue
                best_total[next_facts] = next_cost.total
                step = PlanStep(node.capability_id, tuple(sorted(contract.requires)), tuple(sorted(contract.provides)),
                                step_cost(node), node.health, node.assumed_reliability, contract.risk_class)
                search(next_facts, steps + (step,), used | {node.capability_id}, next_cost)

        search(state.facts, (), frozenset(), PlanCost())
        plans.sort(key=lambda p: (p.cost.total, -p.reliability, len(p.steps), p.capability_ids))
        plans = plans[: self.max_plans]
        missing = None
        if not plans:
            reachable = self.reachable(state)
            missing_effects = tuple(sorted(goal_facts - reachable))
            partial: Plan | None = None
            goal_words: set[str] = set()
            for token in goal_facts:
                goal_words |= _words(token)

            def closeness(entry: tuple[int, float, tuple[PlanStep, ...], frozenset[str]]) -> tuple[int, int, int, float]:
                covered, total, steps, facts = entry
                # How far the path gets towards the goal's subject: facts it
                # made true that share a domain word with the goal.
                related = sum(1 for f in facts - state.facts if _words(f) & goal_words)
                return (covered, related, len(steps), -total)

            if partials:
                covered, total, steps, facts = max(partials, key=closeness)
                if steps:
                    partial = Plan(steps=steps, goal=goal_facts, start=state, end=WorldState(facts=facts),
                                   cost=sum((s.cost for s in steps), PlanCost()))
            if partial is None and goal_words:
                # Nothing produces the goal, so the relevance filter left no
                # candidates.  The closest existing path is the longest chain
                # of capabilities in the goal's subject area that the state
                # can run -- "reaches record -> analysis; missing: profile".
                partial = self._domain_chain(state, goal_words)
            # Root inputs: facts the relevant chain needs that nobody can
            # produce and the state does not hold (an input, not a capability).
            unmet: set[str] = set()
            for fact in relevant - reachable:
                if fact not in goal_facts and not self.producers_of(fact):
                    unmet.add(fact)
            # An unmet fact that some capability declares as an EVENT is the
            # world's to supply (a game must finish).  Any other unmet fact --
            # an effect no capability produces -- is a missing capability.
            events = self.event_tokens()
            world_inputs = {f for f in unmet if f in events}
            unproducible = sorted({e for e in missing_effects if not self.producers_of(e) and not e.startswith("goal.")}
                                  | {f for f in unmet if f not in events})
            missing = MissingCapabilityEvidence(goal=tuple(sorted(goal_facts)), state=state,
                                                available_effects=tuple(self.available_effects()),
                                                reachable_effects=tuple(sorted(reachable - state.facts)),
                                                missing_effects=tuple(sorted(set(missing_effects) | set(unproducible))),
                                                closest_partial_plan=partial,
                                                unmet_inputs=tuple(sorted(world_inputs)), unproducible_effects=tuple(unproducible))
        return PlanReport(goal_facts, state, plans, missing, explored, considered)


def _chain_from(planner: "CompositionPlanner", state: WorldState, nodes: list[CapabilityNode]) -> Plan | None:
    """Greedy: apply applicable nodes in the subject area until none applies."""

    facts = set(state.facts)
    used: set[str] = set()
    steps: list[PlanStep] = []
    progress = True
    while progress and len(steps) < planner.max_depth:
        progress = False
        for node in sorted(nodes, key=lambda n: -n.assumed_reliability):
            if node.capability_id in used or not node.contract.requires <= facts or node.contract.provides <= facts:
                continue
            facts |= node.contract.provides
            used.add(node.capability_id)
            steps.append(PlanStep(node.capability_id, tuple(sorted(node.contract.requires)), tuple(sorted(node.contract.provides)),
                                  step_cost(node), node.health, node.assumed_reliability, node.contract.risk_class))
            progress = True
            break
    if not steps:
        return None
    return Plan(steps=tuple(steps), goal=frozenset(), start=state, end=WorldState(facts=frozenset(facts)),
                cost=sum((s.cost for s in steps), PlanCost()))


def plans_for_goals(planner: CompositionPlanner, state: WorldState, goals: Iterable[str]) -> PlanReport:
    """Goal tokens may name a capability's ``goals`` entry rather than an effect.

    ``improve_chess`` is a goal a capability serves; it satisfies as soon as a
    capability declaring that goal runs.  The planner works on facts, so each
    goal that is not an effect anyone provides is translated into the effects
    of the capabilities that serve it (the cheapest such translation wins
    naturally through the search).
    """

    goal_tokens = [normalise_token(g) for g in goals if normalise_token(g)]
    effects = set(planner.available_effects())
    facts: set[str] = set()
    unmapped: list[str] = []
    for token in goal_tokens:
        if token in effects:
            facts.add(token)
            continue
        servers = [node for node in planner.nodes.values() if node.usable and token in node.contract.goals]
        if servers:
            # The goal is satisfied by whichever server runs; make each server's
            # provides a way to satisfy it by adding a synthetic fact on run.
            facts.add(f"goal.{token}")
        else:
            unmapped.append(token)
            facts.add(token)
    if any(f.startswith("goal.") for f in facts):
        nodes = []
        for node in planner.nodes.values():
            served = [f"goal.{g}" for g in node.contract.goals if f"goal.{g}" in facts]
            if served:
                contract = SemanticContract.from_dict({**node.contract.to_dict(), "effects": list(node.contract.effects) + served})
                node = CapabilityNode(node.capability_id, contract, node.health, node.reliability, node.failure_count, node.active)
            nodes.append(node)
        planner = CompositionPlanner(nodes, max_depth=planner.max_depth, max_plans=planner.max_plans)
    return planner.plan(state, facts)
