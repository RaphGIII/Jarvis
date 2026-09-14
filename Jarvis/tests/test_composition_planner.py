"""The composition planner: effects and preconditions, scored, with proof of gaps.

Generic tokens throughout: the planner must not know what a chess game is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capabilities.contracts import SemanticContract
from capabilities.planner import CapabilityNode, CompositionPlanner, WorldState, plans_for_goals
from service.world import WorldModel


def node(cid: str, health: str = "HEALTHY", **contract) -> CapabilityNode:
    failures = contract.pop("failure_count", 0)
    return CapabilityNode(cid, SemanticContract.from_dict(contract), health, failure_count=failures)


@pytest.fixture
def chain():
    return [
        node("x.capture", consumes=["active_x_session"], produces=["x_record"], effects=["x_available_for_analysis"]),
        node("engine.analyze", consumes=["x_record"], produces=["x_engine_analysis"], latency_class="slow"),
        node("x.classify", consumes=["x_engine_analysis"], produces=["x_error_profile_update"], goals=["understand_recent_loss", "improve_x"]),
        node("archive.zip.create", produces=["archive.zip.create.result"], inferred=True),
    ]


def test_a_three_step_path_is_found_from_state_to_goal(chain):
    planner = CompositionPlanner(chain)
    report = planner.plan(WorldState.build(events=["active_x_session"]), ["x_error_profile_update"])
    assert report.status == "PLAN_FOUND"
    assert report.best.capability_ids == ["x.capture", "engine.analyze", "x.classify"]
    assert report.best.satisfies() and report.best.end.facts >= {"x_record", "x_engine_analysis", "x_error_profile_update"}
    assert "archive.zip.create" not in sum((p.capability_ids for p in report.plans), []), "unrelated capabilities are never inserted"


def test_a_goal_that_names_a_capability_goal_is_translated_into_its_effects(chain):
    report = plans_for_goals(CompositionPlanner(chain), WorldState.build(events=["active_x_session"]), ["improve_x"])
    assert report.status == "PLAN_FOUND" and report.best.capability_ids[-1] == "x.classify"


def test_the_shortest_path_loses_to_a_more_reliable_one(chain):
    quick = node("x.quick_guess", "AT_RISK", failure_count=3, consumes=["x_record"], produces=["x_error_profile_update"])
    report = CompositionPlanner(chain + [quick]).plan(WorldState.build(events=["active_x_session"]), ["x_error_profile_update"])
    ids = [p.capability_ids for p in report.plans]
    assert ["x.capture", "x.quick_guess"] in ids, "the two-step plan exists"
    assert report.best.capability_ids == ["x.capture", "engine.analyze", "x.classify"], "but the reliable three-step plan ranks first"
    assert report.best.reliability > 0.8 and report.plans[1].reliability < 0.4
    assert report.best.cost.total < report.plans[1].cost.total


def test_irreversible_and_metered_steps_are_penalised():
    cheap = node("a.safe", consumes=["s"], produces=["g"], risk_class="harmless", cost_class="free")
    risky = node("a.risky", consumes=["s"], produces=["g"], risk_class="irreversible", cost_class="metered")
    report = CompositionPlanner([risky, cheap]).plan(WorldState.build(facts=["s"]), ["g"])
    assert report.best.capability_ids == ["a.safe"]
    assert report.plans[1].cost.risk_penalty == 4.0 and report.plans[1].cost.execution_cost == 3.0
    assert report.plans[1].irreversible_steps == 1


def test_broken_capabilities_are_not_used_and_healthy_ones_are_preferred():
    broken = node("a.one", "BROKEN", consumes=["s"], produces=["g"])
    at_risk = node("a.two", "AT_RISK", consumes=["s"], produces=["g"])
    healthy = node("a.three", "HEALTHY", consumes=["s"], produces=["g"])
    report = CompositionPlanner([broken, at_risk, healthy]).plan(WorldState.build(facts=["s"]), ["g"])
    assert [p.capability_ids for p in report.plans] == [["a.three"], ["a.two"]]


def test_a_goal_already_true_needs_no_steps(chain):
    report = CompositionPlanner(chain).plan(WorldState.build(facts=["x_error_profile_update"]), ["x_error_profile_update"])
    assert report.status == "PLAN_FOUND" and report.best.steps == ()


def test_missing_capability_evidence_names_the_gap_and_the_closest_path(chain):
    planner = CompositionPlanner(chain)
    report = planner.plan(WorldState.build(events=["active_x_session"]), ["x_training_profile_updated"])
    assert report.status == "MISSING_CAPABILITY" and report.missing is not None
    evidence = report.missing
    assert evidence.missing_effects == ("x_training_profile_updated",)
    assert evidence.closest_partial_plan is not None
    assert evidence.closest_partial_plan.capability_ids == ["x.capture", "engine.analyze", "x.classify"]
    assert "x_engine_analysis" in evidence.reachable_effects
    assert set(evidence.available_effects) >= {"x_record", "x_engine_analysis", "x_error_profile_update"}
    assert "existing path reaches x.capture -> engine.analyze -> x.classify" in evidence.describe()
    assert "missing effect(s): x_training_profile_updated" in evidence.describe()
    assert evidence.to_dict()["current_state"] == ["active_x_session"]


def test_a_missing_input_is_reported_as_an_input_not_a_capability(chain):
    chain[0] = node("x.capture", consumes=["active_x_session"], produces=["x_record"], effects=["x_available_for_analysis"],
                    events=["active_x_session"])
    report = CompositionPlanner(chain).plan(WorldState.build(), ["x_error_profile_update"])
    assert report.status == "MISSING_CAPABILITY"
    assert report.missing.kind == "unmet_input"
    assert report.missing.unmet_inputs == ("active_x_session",), "x_record and the analysis ARE producible; the session is the world's"
    assert report.missing.unproducible_effects == ()


def test_an_effect_nobody_produces_is_a_missing_capability_even_when_an_input_is_also_unmet(chain):
    """A capability gap outranks a world input: engineering is needed either way."""

    report = CompositionPlanner(chain[:2]).plan(WorldState.build(events=["active_x_session"]), ["x_error_profile_update"])
    assert report.missing.kind == "missing_capability"
    assert report.missing.unproducible_effects == ("x_error_profile_update",)
    assert report.missing.closest_partial_plan.capability_ids == ["x.capture", "engine.analyze"]


def test_the_planner_cannot_invent_a_capability(chain):
    planner = CompositionPlanner(chain)
    report = planner.plan(WorldState.build(events=["active_x_session"]), ["teleport_the_owner"])
    ids = {step.capability_id for plan in report.plans for step in plan.steps}
    assert ids == set() and report.status == "MISSING_CAPABILITY"
    assert set(planner.nodes) == {n.capability_id for n in chain}


def test_learned_reliability_overrides_the_health_assumption(chain):
    learned = {"engine.analyze": 0.2}
    planner = CompositionPlanner(chain, reliability=lambda cid: learned.get(cid))
    assert planner.nodes["engine.analyze"].assumed_reliability == 0.2
    assert planner.nodes["x.capture"].assumed_reliability == 0.95


# ---------------------------------------------------------------------------
# The world model
# ---------------------------------------------------------------------------

def test_the_world_model_turns_events_projects_and_runs_into_facts(tmp_path: Path):
    world = WorldModel(tmp_path / "world.json", projects=lambda: ["Schach Training", "Physikum"])
    world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    world.note_facts(["chess_game_record"], source="chess.capture_game")
    state = world.state()
    assert {"chess_game_finished", "chess_game_finished.result_loss", "chess_game_record", "project.schach_training",
            "project.physikum"} <= state.facts
    assert state.sources["chess_game_finished"] == "event" and state.sources["chess_game_record"] == "chess.capture_game"
    again = WorldModel(tmp_path / "world.json")
    assert "chess_game_finished" in again.state().facts, "events survive a restart until they expire"


def test_events_expire(tmp_path: Path, monkeypatch):
    world = WorldModel(tmp_path / "world.json")
    world.note_event("chess_game_finished", ttl=1)
    import time as _time

    later = _time.time() + 10_000
    monkeypatch.setattr(_time, "time", lambda: later)
    assert "chess_game_finished" not in world.state().facts
