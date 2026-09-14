"""The held-out scenario: a curse, a recent loss, an active project -- and a plan.

Project *Schach Training*; a chess game finished forty seconds ago, result
loss; the owner says "fuck, schon wieder verloren".  The reasoning provider
reads ``understand_recent_loss`` from the world state and the closed
vocabulary, proposes capture -> analyse -> classify from the contracts it was
shown, the planner validates it, ZEUS runs it and verifies the goal.  None of
the sentences below appear in any manifest; the manifests only carry
contracts.

The scripted provider (``chess_goal``) stands in for a capable model's
contextual judgement: it names the chess goal only when the vocabulary
offers it AND an event or project in the retrieved context carries it.  The
negative controls therefore test two things: that pure chat never pays for a
semantic call at all, and that a goal without grounding is not acted on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gateway.health import ProviderStatus
from service.events import EventType
from test_gateway_integration import answer_text, ask
from test_intelligence_flow import ScriptedNetwork, chess_goal, chess_manifests, chess_plan, make_world, manifest, tool_events

HELD_OUT = [
    "fuck, schon wieder verloren",
    "was mache ich eigentlich dauernd falsch?",
    "nimm die letzte partie mal auseinander",
    "mach daraus was für mein training",
]


@pytest.fixture
def world(tmp_path: Path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    return core, kernel, net, local, executed


# ---------------------------------------------------------------------------
# The scenario, with paraphrases that appear in no manifest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", HELD_OUT)
def test_a_recent_loss_in_context_becomes_a_validated_plan_that_runs(world, sentence):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    events = ask(core, sentence, wait=30)
    assert executed == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"], \
        (sentence, [e.payload.get("summary") for e in events if e.type is EventType.TOOL])
    flow = tool_events(events, "intelligence: PLAN")
    assert flow, [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    result = flow[0]["intelligence"]
    assert result["goal"]["primary_goal"] in {"understand_recent_loss", "improve_chess"}
    assert result["goal"]["relevant_recent_events"] == ["chess_game_finished"]
    assert result["goal"]["relevant_project"] == "Schach Training"
    assert result["plan_spec"]["source"] == "provider", "the model proposed the plan; the planner only validated it"
    assert result["validation"]["ok"] and result["plan"]["capability_ids"] == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
    assert tool_events(events, "goal: SATISFIED"), [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    assert "chess_error_profile_update" in core.world.state().facts, "what ran now holds in the world model"
    assert not any("Verständnisschicht" in c or "Planungsschicht" in c for c in local.calls), "the local model saw no semantic decision"
    assert len(net.goal_prompts) == 1 and len(net.plan_prompts) == 1


def test_the_provider_sees_only_the_relevant_world_and_the_closed_vocabulary(world):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    core.world.note_event("printer_jam", ttl=600)
    core.owner_projects = lambda: [{"title": "Schach Training", "origin": "owner"}, {"title": "Physikum", "origin": "owner"}]  # type: ignore[assignment]
    ask(core, "na super, wieder eine niederlage", wait=30)
    assert net.goal_prompts, "the reasoning provider was asked for the GoalSpec"
    prompt = net.goal_prompts[0]
    assert "chess_game_finished (event)" in prompt and "project: Schach Training" in prompt
    assert "printer_jam" not in prompt and "Physikum" not in prompt, "unrelated events and projects are not sent"
    assert "understand_recent_loss" in prompt and "improve_chess" in prompt
    goal_request = next(r["body"] for r in net.requests if "Verständnisschicht" in json.dumps(r["body"], ensure_ascii=False))
    schema = goal_request["generationConfig"]["responseSchema"]
    assert set(schema["properties"]["primary_goal"]["enum"]) >= {"understand_recent_loss", "improve_chess", "chess_game_record", "none"}
    assert schema["properties"]["relevant_project"]["enum"] == ["Schach Training", "none"]
    assert "stockfish.analyze" in net.plan_prompts[0] and '"contract"' in net.plan_prompts[0], "stage 2: full contracts, only for the retrieved cards"
    assert "archive" not in net.plan_prompts[0]


# ---------------------------------------------------------------------------
# Negative controls: the same words without supporting context
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", HELD_OUT[:3] + ["ich hab schon wieder verloren"])
def test_without_context_the_same_sentence_is_plain_conversation(world, sentence):
    """No recent event, no related project, no capability word: the reasoning provider is not even asked."""

    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    events = ask(core, sentence, wait=30)
    assert executed == []
    assert not tool_events(events, "intelligence:"), "pure chat pays for no semantic reading"
    assert net.goal_prompts == [] and net.plan_prompts == []
    text = answer_text(events).lower()
    assert "tut mir leid" in text or "sag mir genauer" in text, "answered as conversation, or asked what should exist -- never acted"
    assert "planner" not in text, "no internal diagnostics reach the owner"


def test_a_capability_word_without_context_is_read_but_not_acted_on(world):
    """'training' touches a contract's vocabulary, so the provider is asked; without an event or project it says none."""

    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    events = ask(core, "mach daraus was für mein training", wait=30)
    assert executed == []
    assert len(net.goal_prompts) == 1 and net.plan_prompts == [], "one GoalSpec call, no PlanSpec call"
    assert not tool_events(events, "intelligence: PLAN")
    assert tool_events(events, "intelligence: CLARIFY"), "the provider said none and asked; the system relays the question"
    assert answer_text(events) == "Worum geht es?"


def test_an_unrelated_recent_event_does_not_ground_the_chess_goal(world):
    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    core.world.note_event("printer_jam", ttl=600)
    events = ask(core, "verdammt, schon wieder verloren", wait=30)
    assert executed == [] and not tool_events(events, "intelligence: PLAN")
    assert net.plan_prompts == []


def test_an_expired_game_no_longer_supports_the_reading(world, monkeypatch):
    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=1)
    import time as _time

    later = _time.time() + 3_600
    monkeypatch.setattr(_time, "time", lambda: later)
    assert "chess_game_finished" not in core.world.state().facts
    events = ask(core, "schon wieder verloren", wait=30)
    assert executed == [] and not tool_events(events, "intelligence: PLAN")


def test_an_ungrounded_goal_from_the_provider_becomes_a_question_not_an_action(world):
    """A provider that insists on the chess goal with nothing to carry it -- no event, no project, no word -- is asked back, not obeyed."""

    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    # "stockfish" retrieves the engine card by its word; capture and classify
    # follow as chain connectors.  A connector's goal is in the vocabulary,
    # but nothing in the request or the world supports it.
    net.goal_fn = lambda prompt: {"primary_goal": "understand_recent_loss", "confidence": 0.95, "ambiguity": 0.1, "reason": "I insist"}
    events = ask(core, "stockfish ist gerade irgendwie komisch", wait=30)
    assert executed == []
    clarify = tool_events(events, "intelligence: CLARIFY")
    assert clarify, [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    assert "neither an event" in clarify[0]["intelligence"]["reason"]
    assert clarify[0]["intelligence"]["context"]["cards"] == ["stockfish.analyze", "chess.capture_game", "chess.classify_errors"]
    assert net.plan_prompts == [], "no PlanSpec is requested for an ungrounded goal"
    assert "Meinst du" in answer_text(events)


def test_a_word_grounded_goal_without_the_game_is_an_unmet_input_not_an_action(world):
    """'schach' is a contract's word, so the goal is grounded; but no game finished, so nothing runs and nothing is built."""

    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    net.goal_fn = lambda prompt: {"primary_goal": "understand_recent_loss", "confidence": 0.9, "ambiguity": 0.1, "reason": "chess words"}
    events = ask(core, "im schach läuft es gerade nicht", wait=30)
    assert executed == []
    unmet = tool_events(events, "intelligence: unmet input")
    assert unmet and unmet[0]["missing_capability"]["kind"] == "unmet_input"
    assert not tool_events(events, "missing capability proven") and not any(e.type is EventType.PROGRESS for e in events)


def test_an_active_project_alone_grounds_the_reading_but_the_input_is_missing(world):
    core, kernel, net, local, executed = world
    events = ask(core, "ich werde im schach einfach nicht besser", wait=30)
    # No game has finished: the capture step has nothing to consume.  The
    # provider's plan is rejected with the exact unmet requirement, once; the
    # planner then proves it is an unmet INPUT, not a missing capability.
    assert executed == []
    unmet = tool_events(events, "intelligence: unmet input")
    assert unmet and unmet[0]["missing_capability"]["kind"] == "unmet_input"
    assert "chess_game_finished" in unmet[0]["missing_capability"]["unmet_inputs"]
    assert len(net.plan_prompts) == 2 and "chess_game_finished" in net.plan_prompts[1]
    assert "chess_game_finished" in answer_text(events)
    assert not tool_events(events, "missing capability proven")


# ---------------------------------------------------------------------------
# Missing capability: proof, not opinion; an EngineeringSpec, not a prompt
# ---------------------------------------------------------------------------

def test_a_missing_effect_is_proven_and_specified_for_engineering(world, monkeypatch):
    core, kernel, net, local, executed = world
    core.capabilities.registry.disable("chess.classify_errors", reason="test")
    core.capabilities.registry.register(manifest(
        "chess.training_planner", "Turns an error profile into a training plan.",
        {"consumes": ["chess_error_profile_update"], "produces": ["chess_training_plan"], "goals": ["improve_chess"],
         "related_projects": ["Schach Training"], "domain": "chess"}))
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    net.plan_fn = lambda prompt: {"steps": [], "reason": "nobody classifies errors", "missing": "chess_error_profile_update"}
    briefs: list[tuple[str, object]] = []
    monkeypatch.setattr(core, "_start_capability_engineering",
                        lambda goal, original, scope, **kw: briefs.append((kw.get("evidence", ""), kw.get("spec")))
                        or core._deliver("engineering", scope=scope, backend="x"))
    events = ask(core, "schon wieder verloren, ich will endlich besser werden", wait=30)
    assert executed == []
    proven = tool_events(events, "missing capability proven")
    assert proven, [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    evidence = proven[0]["missing_capability"]
    assert evidence["kind"] == "missing_capability"
    assert "chess_error_profile_update" in evidence["missing_effects"], "the effect nobody produces any more"
    assert evidence["unmet_inputs"] == [], "the game did finish; the gap is a capability, not an input"
    assert evidence["closest_partial_plan"]["capability_ids"] == ["chess.capture_game", "stockfish.analyze"]
    spec = proven[0]["engineering_spec"]
    assert spec["owner_goal"] == "schon wieder verloren, ich will endlich besser werden"
    assert spec["goal_spec"]["primary_goal"] == "improve_chess" and spec["goal_spec"]["relevant_project"] == "Schach Training"
    assert spec["missing_effects"] == ["chess_error_profile_update"]
    assert spec["reusable_capabilities"] == ["chess.capture_game", "stockfish.analyze"]
    assert spec["required_permissions"] == [] and spec["privacy_constraints"]
    brief, handed = briefs[0]
    assert handed is not None and handed.spec_id == spec["spec_id"]
    assert "MISSING CAPABILITY -- proven by the composition planner" in brief
    assert "chess.capture_game -> stockfish.analyze" in brief and "contract.json" in brief


# ---------------------------------------------------------------------------
# No 4B decision authority
# ---------------------------------------------------------------------------

def test_with_only_the_offline_model_reachable_no_semantic_decision_is_taken(world):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")  # no metered route either: only the local model is left
    local.generate_structured = lambda prompt, schema, **kw: json.dumps({"primary_goal": "understand_recent_loss", "confidence": 0.95,
                                                                         "reason": "local guess"})
    events = ask(core, "fuck, schon wieder verloren", wait=30)
    assert executed == [], "the legacy local model's goal was not acted on"
    unavailable = tool_events(events, "intelligence: INTELLIGENCE_UNAVAILABLE")
    assert unavailable and unavailable[0]["intelligence"]["goal"] is None
    assert "offline fallback" in unavailable[0]["intelligence"]["reason"]
    text = answer_text(events)
    assert "nicht erreichbar" in text and "lokale Modell entscheidet" in text
    assert not any(e.type is EventType.PROGRESS for e in events), "no engineering was started"


def test_semantic_authority_is_reported_truthfully(world):
    core, kernel, net, local, executed = world
    assert core.semantic_authority()["available"] is True
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    authority = core.semantic_authority()
    assert authority["available"] is False and authority["role"] == "local.fast"


# ---------------------------------------------------------------------------
# Metrics and the API
# ---------------------------------------------------------------------------

def test_the_preview_runs_the_flow_without_executing_and_reports_its_context(world):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    preview = core.compose_contract_preview("schon wieder verloren")
    assert preview["ok"] and preview["status"] == "PLAN"
    metrics = preview["metrics"]
    assert 0 < metrics["capability_summary_tokens"] < 400 and metrics["world_context_tokens"] > 0
    assert metrics["total_model_context_tokens"] < 3000, "three contracts and a sentence: a few hundred tokens, not a repository"
    assert preview["plan"]["capability_ids"] == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
    assert preview["context"]["cards"] == ["chess.capture_game", "chess.classify_errors", "stockfish.analyze"]
    assert executed == [], "a preview executes nothing"
    state = core.note_world_event("chess_game_finished", detail={"result": "win"})
    assert state["ok"] and "chess_game_finished.result_win" in state["state"]["facts"]
    assert core.world_state()["ok"]


# ---------------------------------------------------------------------------
# Live: the real reasoning provider, when a key is present
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ZEUS_LIVE_SEMANTIC"), reason="set ZEUS_LIVE_SEMANTIC=1 with a configured provider key")
def test_live_reasoning_provider_grounds_the_scenario(tmp_path: Path):
    from capabilities.intelligence import IntelligenceFlow, ProjectSummary, capability_cards
    from capabilities.registry import CapabilityRegistry
    from core.kernel import JarvisKernel, KernelConfig
    from service.world import WorldModel

    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state"))
    registry = CapabilityRegistry(tmp_path / "registry.json")
    for m in chess_manifests():
        registry.register(m)
    world_model = WorldModel(projects=lambda: ["Schach Training"])
    world_model.note_event("chess_game_finished", detail={"result": "loss"})
    flow = IntelligenceFlow(kernel.gateway, capability_cards(registry), [ProjectSummary("Schach Training")])
    positive = flow.run("fuck, schon wieder verloren", world_model.state())
    assert positive.status == "PLAN" and positive.goal and positive.goal.primary_goal
    negative = IntelligenceFlow(kernel.gateway, capability_cards(registry), [ProjectSummary("Schach Training")]).run(
        "ich habe meine schlüssel verloren", world_model.state())
    assert negative.status != "PLAN"
