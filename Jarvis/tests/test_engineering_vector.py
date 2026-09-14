"""The EngineeringTaskVector: one engineer, chosen before execution, frontier directly when the task needs it."""

from __future__ import annotations

from pathlib import Path

import pytest

from capabilities.engineering_spec import EngineeringSpec
from gateway.config import GatewayConfig
from gateway.modes import ChatMode
from gateway.secrets import CredentialStore
from service.engineering import Engineer, EngineeringNeed, choose_engineer, estimate_engineering
from service.engineering_vector import EngineeringTaskVector, task_vector_for, vector_from_spec
from test_engineering_router_gateway import Availability
from test_model_gateway import FakeNetwork, make_gateway

CHESS_SYSTEM = ("Ich bin schlecht im Schach. Beobachte meinen Bildschirm nur wenn ich Schach spiele, erfasse meine Partien, "
                "analysiere sie, kategorisiere meine wiederkehrenden Fehler, baue daraus ein langfristiges Spielerprofil und "
                "personalisiertes Training und lege das Projekt Schach Training an.")


def spec(owner_goal: str, *, missing: list[str], partial: list[str] = (), reusable: list[str] = (), permissions: list[str] = (),
         acceptance: int = 4, ambiguity: float = 0.1) -> EngineeringSpec:
    return EngineeringSpec(
        spec_id="t1", owner_goal=owner_goal, goal_spec={"primary_goal": "improve_chess", "ambiguity": ambiguity},
        existing_coverage=[{"capability_id": c, "produces": [], "requires": []} for c in partial], closest_partial_plan=list(partial),
        missing_effects=list(missing), reusable_capabilities=list(reusable), relevant_interfaces=["run(payload) -> dict"],
        required_permissions=list(permissions), acceptance_criteria=[f"criterion {i}" for i in range(acceptance)],
        privacy_constraints=["never print secrets"], catalog_context="", suggested_contract={},
    )


def chess_spec() -> EngineeringSpec:
    return spec(CHESS_SYSTEM, missing=["screen_chess_observation", "chess_game_capture", "chess_error_profile", "chess_training_plan"],
                partial=["stockfish.analyze"], reusable=["stockfish.analyze", "project.create", "file.write"],
                permissions=["screen.read", "filesystem.write"], acceptance=8)


def word_count_spec() -> EngineeringSpec:
    return spec("Zähle die Wörter in einer Textdatei.", missing=["word_count"], acceptance=3)


@pytest.fixture
def cfg() -> GatewayConfig:
    config = GatewayConfig.defaults()
    for name in ("gemini", "openai", "anthropic"):
        config = config.with_provider_enabled(name, True)
    return config


@pytest.fixture
def creds(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    store.set("anthropic", "sk-ant-anthropictestkey00000000000000")
    store.set("openai", "sk-openaitestkey00000000000000000")
    store.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    return store


# --------------------------------------------------------------------------
# The vector
# --------------------------------------------------------------------------

def test_the_chess_system_is_a_large_subsystem_with_frontier_indicators():
    vector = vector_from_spec(chess_spec())
    assert vector.task_class == "engineering.large" and vector.frontier_required
    assert len(vector.frontier_indicators()) >= 3, vector.frontier_indicators()
    assert {"screen", "engine", "projects", "profile", "training"} <= set(vector.systems)
    assert vector.external_system_count >= 0.75 and vector.long_horizon >= 0.6 and vector.integration_breadth >= 0.6
    assert vector.verification_complexity >= 0.6, "screen observation is verified against a live screen, not a unit test"
    features = vector.as_features()
    assert set(features) == {"change_size", "integration_breadth", "technical_novelty", "core_architecture_impact",
                             "codebase_context_required", "verification_complexity", "external_system_count", "long_horizon",
                             "risk", "specification_uncertainty"}
    assert all(0.0 <= v <= 1.0 for v in features.values())


def test_a_word_counter_is_small_and_needs_no_frontier():
    vector = vector_from_spec(word_count_spec())
    assert vector.task_class == "engineering.small" and not vector.frontier_required
    assert vector.frontier_indicators() == []


def test_catalog_impact_raises_core_architecture_and_context_needs():
    plain = vector_from_spec(word_count_spec())
    impacted = vector_from_spec(word_count_spec(), catalog_dependents=14, impacted_core_modules=["service/core.py", "gateway/router.py"])
    assert impacted.core_architecture_impact > plain.core_architecture_impact >= 0.0
    assert impacted.codebase_context_required > plain.codebase_context_required


def test_the_vector_round_trips_and_feeds_the_gateway_task_vector():
    vector = vector_from_spec(chess_spec())
    again = EngineeringTaskVector.from_dict(vector.to_dict())
    assert again.as_features() == vector.as_features() and again.frontier_required
    task = task_vector_for(vector, CHESS_SYSTEM)
    assert task.task_class.value == "engineering.large" and task.facts["frontier_required"] is True
    assert task.facts["engineering_vector"]["external_system_count"] >= 0.75


# --------------------------------------------------------------------------
# The router: frontier directly, never standard-then-frontier
# --------------------------------------------------------------------------

def test_the_chess_system_goes_directly_to_the_frontier_engineer(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    vector = vector_from_spec(chess_spec())
    decision = choose_engineer(EngineeringNeed.CAPABILITY_MISSING, availability=Availability(True), task=task_vector_for(vector, CHESS_SYSTEM),
                               gateway=gateway, mode=ChatMode.BUILD, prompt=chess_spec().to_brief(), vector=vector)
    assert decision.engineer is Engineer.API and decision.role == "engineer.frontier" and decision.provider_name == "engineer.frontier"
    assert "directly" in decision.reason and "frontier engineering required" in decision.reason
    assert all(c["role"] == "engineer.frontier" for c in decision.candidates), "the standard engineer was not even a candidate"
    assert decision.estimated_eur > 0.0 and decision.estimate_range_eur[1] > decision.estimate_range_eur[0]
    assert decision.vector["frontier_required"] is True and decision.task_class == "engineering.large"
    assert gateway.transport.issued == 0, "decided before execution: nothing was sent"


def test_frontier_required_but_unavailable_is_nobody_not_the_standard_engineer(tmp_path, cfg):
    empty = CredentialStore(tmp_path / "none.json", use_dpapi=False)
    empty.set("openai", "sk-openaitestkey00000000000000000")  # a deep reasoner, but no engineering key
    gateway = make_gateway(tmp_path, cfg, empty, FakeNetwork())
    vector = vector_from_spec(chess_spec())
    decision = choose_engineer(EngineeringNeed.CAPABILITY_MISSING, availability=Availability(True), task=task_vector_for(vector, CHESS_SYSTEM),
                               gateway=gateway, mode=ChatMode.BUILD, prompt=CHESS_SYSTEM, vector=vector)
    assert decision.engineer is Engineer.NONE and decision.queued
    assert "not tried instead" in decision.reason and "no credential for anthropic" in decision.reason
    assert decision.role == "" and decision.provider_name == ""


def test_outside_build_mode_the_frontier_task_waits_rather_than_using_codex(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    vector = vector_from_spec(chess_spec())
    decision = choose_engineer(EngineeringNeed.CAPABILITY_MISSING, availability=Availability(True), task=task_vector_for(vector, CHESS_SYSTEM),
                               gateway=gateway, mode=ChatMode.AUTO, prompt=CHESS_SYSTEM, vector=vector)
    assert decision.engineer is Engineer.NONE and "does not permit engineer.frontier" in decision.reason


def test_a_small_task_still_takes_the_cheapest_reliable_engineer(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    vector = vector_from_spec(word_count_spec())
    decision = choose_engineer(EngineeringNeed.CAPABILITY_MISSING, availability=Availability(True), task=task_vector_for(vector, "Zähle die Wörter"),
                               gateway=gateway, mode=ChatMode.BUILD, prompt="Zähle die Wörter", vector=vector)
    assert decision.engineer is Engineer.CODEX and decision.vector["task_class"] == "engineering.small"


def test_the_alternative_frontier_slot_is_used_only_when_configured(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    alt = cfg.roles["engineer.frontier_alt"]
    assert alt.enabled is False and cfg.providers["frontier_alt"].enabled is False, "an empty slot, not a cascade"
    # With the primary frontier disabled and the slot still empty: nobody.
    config = cfg.with_role_binding("engineer.frontier", enabled=False)
    gateway.reconfigure(config)
    vector = vector_from_spec(chess_spec())
    decision = choose_engineer(EngineeringNeed.CAPABILITY_MISSING, availability=Availability(True), task=task_vector_for(vector, CHESS_SYSTEM),
                               gateway=gateway, mode=ChatMode.BUILD, prompt=CHESS_SYSTEM, vector=vector)
    assert decision.engineer is Engineer.NONE and "none configured" in decision.reason


# --------------------------------------------------------------------------
# The engineering cost governor: the estimate before anything is spent
# --------------------------------------------------------------------------

def test_the_engineering_estimate_covers_context_output_and_the_tightest_cap(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    brief = chess_spec().to_brief() + ("x" * 40_000)
    estimate = estimate_engineering(gateway, "engineer.frontier", context_chars=len(brief), mode=ChatMode.BUILD)
    assert estimate["context_tokens"] > 10_000 and estimate["expected_output_tokens"] == cfg.roles["engineer.frontier"].max_output_tokens
    assert estimate["estimated_eur"] > 0.0 and estimate["range_eur"][0] < estimate["estimated_eur"] < estimate["range_eur"][1]
    assert estimate["reserved_eur"] == pytest.approx(estimate["estimated_eur"] * cfg.budget.safety_factor, rel=1e-3)
    assert estimate["hard_max_eur"] <= cfg.budget.per_task_hard_cap
    assert estimate["month_cap_eur"] == 40.0 and estimate["reservable"] is True
    assert estimate["pricing_confirmed"] is False, "the owner verified the USD price, but no EUR conversion is configured"
    too_big = estimate_engineering(gateway, "engineer.frontier", context_chars=len(brief), expected_output_tokens=400_000, mode=ChatMode.BUILD)
    assert too_big["reservable"] is False and too_big["blocking_cap"], "over the cap: no reservation, so no call"
