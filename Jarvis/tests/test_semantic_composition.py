"""The held-out scenario: a curse, a recent loss, an active project -- and a plan.

Project *Schach Training*; a chess game finished forty seconds ago, result
loss; the owner says "fuck, schon wieder verloren".  ZEUS derives
``understand_recent_loss``, finds capture -> analyse -> classify in the
registry's contracts, and runs it.  None of the sentences below appear in
any manifest; the manifests only carry contracts.

What the fake semantic provider is, and is not: it is *trigger-happy* -- it
answers the chess goal whenever the closed vocabulary offers one, whatever
the sentence and whatever the context.  So every negative control here is
the SYSTEM refusing an ungrounded reading, not the fake being polite.  The
fake stands in for a capable model's contextual judgement only in the
positive cases; the model's real linguistic ability is exercised by the
optional live test at the end.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from capabilities.models import CapabilityHealth, CapabilityLifecycle, CapabilityManifest
from capabilities.service import ExecutionOutcome
from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from gateway.config import GatewayConfig
from gateway.health import ProviderStatus
from service.core import JarvisCore
from service.events import EventType
from test_gateway_integration import LocalChat, answer_text, ask
from test_model_gateway import FakeNetwork, FakeResponse, gemini_reply, openai_reply

CHESS_GOALS = ["understand_recent_loss", "improve_chess"]


class TriggerHappyNetwork(FakeNetwork):
    """A semantic provider that always picks the chess goal when it is offered."""

    def __init__(self) -> None:
        super().__init__()
        self.responses["api.openai.com"] = openai_reply("Eine Antwort.")

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        host = request.full_url.split("/")[2]
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        if host != "generativelanguage.googleapis.com":
            return super().__call__(request, timeout)
        answer = self.responses.get(host)
        if isinstance(answer, Exception):
            raise answer
        prompt = json.dumps(body, ensure_ascii=False)
        if "semantische Zielableitung" in prompt:
            offered = [g for g in CHESS_GOALS if f"- {g} (" in prompt]
            if offered:
                return FakeResponse(gemini_reply(json.dumps({"goals": offered[:1], "confidence": 0.9,
                                                             "reason": "the owner just lost a game"})))
            return FakeResponse(gemini_reply(json.dumps({"goals": [], "confidence": 0.3, "reason": "nothing fits"})))
        if "semantische Steuerung" in prompt:
            return FakeResponse(gemini_reply(json.dumps({"operation": "conversation", "target": "", "confidence": 0.9, "reason": "chat"})))
        return FakeResponse(gemini_reply("Das tut mir leid. Nächstes Mal klappt es."))


def _manifest(cid: str, description: str, contract: dict, **extra) -> CapabilityManifest:
    return CapabilityManifest(
        cid, description, lifecycle=CapabilityLifecycle.ACTIVE.value,
        health={"state": "healthy", "health": CapabilityHealth.HEALTHY.value}, contract=contract, family=cid.split(".", 1)[0],
        source_location=f"/installed/{cid}", **extra,
    )


def chess_manifests() -> list[CapabilityManifest]:
    return [
        _manifest("chess.capture_game", "Stores the game that was just played as a record.",
                  {"consumes": ["chess_game_finished"], "produces": ["chess_game_record"],
                   "effects": ["chess_game_available_for_analysis"], "events": ["chess_game_finished"],
                   "related_projects": ["Schach Training"], "domain": "chess"}),
        _manifest("stockfish.analyze", "Runs an engine analysis over a game record.",
                  {"consumes": ["chess_game_record"], "produces": ["chess_engine_analysis"], "latency_class": "slow", "domain": "chess"}),
        _manifest("chess.classify_errors", "Classifies the mistakes in an analysed game and updates the error profile.",
                  {"consumes": ["chess_engine_analysis"], "produces": ["chess_error_profile_update"],
                   "goals": CHESS_GOALS, "related_projects": ["Schach Training"], "domain": "chess"}),
    ]


@pytest.fixture
def world(tmp_path: Path):
    net = TriggerHappyNetwork()
    local = LocalChat()
    config_root = tmp_path / "config"
    config_root.mkdir()
    cfg = GatewayConfig.defaults().with_provider_enabled("gemini", True).with_provider_enabled("openai", True)
    cfg.save(config_root / "providers.json")
    (config_root / "owner").mkdir()
    (config_root / "owner" / "spending.json").write_text(json.dumps({"paid_api": True}), encoding="utf-8")
    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state", config_root=config_root, enable_research_tools=False))
    kernel.local_provider = lambda tier: local  # type: ignore[assignment]
    gateway = kernel.gateway
    gateway.transport._opener = net
    gateway.credentials.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    gateway.credentials.set("openai", "sk-openaitestkey00000000000000000")
    core = JarvisCore(kernel=kernel, identity=Identity())
    core.language = "de"
    for manifest in chess_manifests():
        core.capabilities.registry.register(manifest)
    executed: list[str] = []

    def fake_execute(capability_id: str, payload=None) -> ExecutionOutcome:
        executed.append(capability_id)
        return ExecutionOutcome(capability_id=capability_id, ok=True, output={"ok": True, "capability": capability_id})

    core.capabilities.execute = fake_execute  # type: ignore[assignment]
    core.owner_projects = lambda: [{"title": "Schach Training", "origin": "owner"}]  # type: ignore[assignment]
    return core, kernel, net, local, executed


def tool_events(events, needle: str):
    return [e.payload for e in events if e.type is EventType.TOOL and needle in str(e.payload.get("summary", ""))]


# ---------------------------------------------------------------------------
# The scenario, with paraphrases that appear in no manifest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "fuck, schon wieder verloren",
    "mist, das ging gerade wieder komplett schief",
    "warum verliere ich eigentlich dauernd?",
    "hab die partie eben vergeigt, keine ahnung woran es lag",
])
def test_a_recent_loss_in_context_becomes_a_composed_plan(world, sentence):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    events = ask(core, sentence, wait=30)
    assert executed == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"], \
        (sentence, [e.payload.get("summary") for e in events if e.type is EventType.TOOL])
    composition = tool_events(events, "composition: PLAN")
    assert composition and composition[0]["composition"]["derivation"]["goals"] == ["understand_recent_loss"]
    grounding = composition[0]["composition"]["derivation"]["grounding"]["understand_recent_loss"]
    assert grounding["grounded"] and ("event chess_game_finished" in grounding["why"] or "project" in grounding["why"])
    assert tool_events(events, "goal: SATISFIED"), [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    assert "chess_error_profile_update" in core.world.state().facts, "what ran now holds in the world model"
    assert not any("Zielableitung" in c for c in local.calls), "the local model saw no semantic decision"


def test_the_derivation_sees_the_world_state_and_the_closed_vocabulary(world):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    ask(core, "na super, wieder eine niederlage", wait=30)
    prompts = [r["body"] for r in net.requests if "semantische Zielableitung" in json.dumps(r["body"], ensure_ascii=False)]
    assert prompts, "the semantic provider was asked"
    text = json.dumps(prompts[0], ensure_ascii=False)
    assert "chess_game_finished (event)" in text and "project.schach_training (project)" in text
    assert "- understand_recent_loss (" in text and "- improve_chess (" in text
    schema = prompts[0]["generationConfig"]["responseSchema"]
    assert set(schema["properties"]["goals"]["items"]["enum"]) >= {"understand_recent_loss", "improve_chess", "chess_game_record"}


# ---------------------------------------------------------------------------
# Negative controls: the same words without supporting context
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", ["fuck, schon wieder verloren", "ich hab schon wieder verloren"])
def test_without_context_the_same_sentence_is_plain_conversation(world, sentence):
    """No recent event, no related project, no capability word: the semantic provider is not even asked."""

    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]  # no chess project either
    events = ask(core, sentence, wait=30)
    assert executed == []
    assert not tool_events(events, "composition:"), "pure chat pays for no semantic reading"
    assert not any("Zielableitung" in json.dumps(r["body"], ensure_ascii=False) for r in net.requests)
    assert "tut mir leid" in answer_text(events).lower(), "answered as conversation"


def test_an_unrelated_recent_event_does_not_ground_the_chess_goal(world):
    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    core.world.note_event("printer_jam", ttl=600)
    events = ask(core, "verdammt, schon wieder verloren", wait=30)
    assert executed == [] and tool_events(events, "composition: CLARIFY")


def test_an_expired_game_no_longer_supports_the_reading(world, monkeypatch):
    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=1)
    import time as _time

    later = _time.time() + 3_600
    monkeypatch.setattr(_time, "time", lambda: later)
    assert "chess_game_finished" not in core.world.state().facts
    events = ask(core, "schon wieder verloren", wait=30)
    assert executed == [] and not tool_events(events, "composition: PLAN")


def test_a_capability_word_without_context_is_asked_about_not_acted_on(world):
    """The trigger-happy model names the chess goal; with no event and no project the system asks."""

    core, kernel, net, local, executed = world
    core.owner_projects = lambda: []  # type: ignore[assignment]
    events = ask(core, "im schach läuft es gerade nicht", wait=30)
    assert executed == []
    composition = tool_events(events, "composition:")
    assert composition, "a capability word makes the reading worth a look"
    status = composition[0]["composition"]["status"]
    assert status == "MISSING_CAPABILITY", "grounded by the word 'schach', but no game has finished"
    assert tool_events(events, "composition: unmet input") and not tool_events(events, "missing capability proven")


def test_an_active_project_alone_grounds_the_reading_without_an_event(world):
    core, kernel, net, local, executed = world
    events = ask(core, "ich werde im schach einfach nicht besser", wait=30)
    # No game has finished: the capture step has nothing to consume, so the
    # goal is grounded (project) but the plan cannot start -- an unmet INPUT,
    # said as such; nothing is built and nothing runs.
    assert executed == []
    unmet = tool_events(events, "composition: unmet input")
    assert unmet and unmet[0]["missing_capability"]["kind"] == "unmet_input"
    assert "chess_game_finished" in unmet[0]["missing_capability"]["unmet_inputs"]
    assert "chess_game_finished" in answer_text(events)
    assert not tool_events(events, "missing capability proven")


# ---------------------------------------------------------------------------
# Missing capability: proof, not opinion
# ---------------------------------------------------------------------------

def test_a_missing_effect_is_proven_and_handed_to_engineering_with_evidence(world, monkeypatch):
    core, kernel, net, local, executed = world
    core.capabilities.registry.disable("chess.classify_errors", reason="test")
    core.capabilities.registry.register(_manifest(
        "chess.training_planner", "Turns an error profile into a training plan.",
        {"consumes": ["chess_error_profile_update"], "produces": ["chess_training_plan"], "goals": ["improve_chess"],
         "related_projects": ["Schach Training"], "domain": "chess"}))
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    briefs: list[str] = []
    monkeypatch.setattr(core, "_start_capability_engineering",
                        lambda goal, original, scope, **kw: briefs.append(kw.get("evidence", "")) or core._deliver("engineering", scope=scope, backend="x"))
    events = ask(core, "schon wieder verloren, ich will endlich besser werden", wait=30)
    assert executed == []
    proven = tool_events(events, "missing capability proven")
    assert proven, [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    evidence = proven[0]["missing_capability"]
    assert evidence["kind"] == "missing_capability"
    assert "chess_error_profile_update" in evidence["missing_effects"], "the effect nobody produces any more"
    assert evidence["unmet_inputs"] == [], "the game did finish; the gap is a capability, not an input"
    assert evidence["closest_partial_plan"]["capability_ids"] == ["chess.capture_game", "stockfish.analyze"]
    assert briefs and "MISSING CAPABILITY -- proven by the composition planner" in briefs[0]
    assert "chess.capture_game -> stockfish.analyze" in briefs[0]
    assert "contract.json" in briefs[0]


# ---------------------------------------------------------------------------
# No 4B decision authority
# ---------------------------------------------------------------------------

def test_with_only_the_offline_model_reachable_no_semantic_decision_is_taken(world):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")  # no metered route either: only the local model is left
    local.generate_structured = lambda prompt, schema, **kw: json.dumps({"goals": ["understand_recent_loss"], "confidence": 0.95,
                                                                         "reason": "local guess"})
    events = ask(core, "fuck, schon wieder verloren", wait=30)
    assert executed == [], "the legacy local model's goal was not acted on"
    composition = tool_events(events, "composition: UNAVAILABLE")
    assert composition and composition[0]["composition"]["derivation"]["offline"] is True
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

def test_the_composition_reports_how_much_context_it_needed(world):
    core, kernel, net, local, executed = world
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    preview = core.compose_contract_preview("schon wieder verloren")
    assert preview["ok"] and preview["status"] == "PLAN"
    metrics = preview["metrics"]
    assert metrics["manifest_tokens"] > 0 and metrics["total_tokens"] >= metrics["manifest_tokens"]
    assert metrics["total_tokens"] < 2000, "three contracts and a sentence: a few hundred tokens, not a repository"
    assert preview["report"]["best"]["capability_ids"] == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
    assert executed == [], "a preview executes nothing"
    state = core.note_world_event("chess_game_finished", detail={"result": "win"})
    assert state["ok"] and "chess_game_finished.result_win" in state["state"]["facts"]
    assert core.world_state()["ok"]


# ---------------------------------------------------------------------------
# Live: the real semantic provider, when a key is present
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ZEUS_LIVE_SEMANTIC"), reason="set ZEUS_LIVE_SEMANTIC=1 with a configured provider key")
def test_live_semantic_provider_grounds_the_scenario(tmp_path: Path):
    from capabilities.composition import CompositionEngine
    from capabilities.registry import CapabilityRegistry
    from service.world import WorldModel

    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state"))
    registry = CapabilityRegistry(tmp_path / "registry.json")
    for manifest in chess_manifests():
        registry.register(manifest)
    engine = CompositionEngine(registry, kernel.gateway)
    world_model = WorldModel(projects=lambda: ["Schach Training"])
    world_model.note_event("chess_game_finished", detail={"result": "loss"})
    positive = engine.derive_and_plan("fuck, schon wieder verloren", world_model.state())
    assert positive.status == "PLAN" and positive.derivation.goals
    negative = engine.derive_and_plan("ich habe meine schlüssel verloren", world_model.state())
    assert negative.status != "PLAN"
