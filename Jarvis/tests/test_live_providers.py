"""Live acceptance against the real providers.  Explicit opt-in only; never part of ordinary CI.

    ZEUS_LIVE_GEMINI=1    the free reasoning role (quota, not money)
    ZEUS_LIVE_OPENAI=1    one small deep-reasoning call (metered; a few cents)
    ZEUS_LIVE_ENGINEER=1  one tiny standard-engineer job (metered; a few cents)

Credentials come from the owner's encrypted store (``data/jarvis/owner/
provider_credentials.json``) or from the environment (GOOGLE_GEMINI_API_KEY /
GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY); the store is read, never
written to by these tests.  Every run writes its evidence -- prompt sizes,
tokens, latency, cost, the GoalSpec and the plan, never a secret -- under
``data/acceptance_evidence/live_*.json``.

The pass condition of the sprint is here, not in the scripted tests: the
model receives a German utterance it has never seen, reads its context,
produces a grounded GoalSpec, selects the existing capabilities, and the
deterministic validator accepts or rightly rejects its plan.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from capabilities.intelligence import IntelligenceFlow, ProjectSummary, builtin_cards, capability_cards
from capabilities.registry import CapabilityRegistry
from gateway.config import GatewayConfig
from gateway.gateway import GatewayRequest, ModelGateway
from gateway.modes import ChatMode
from gateway.secrets import CredentialStore, redact
from gateway.task import TaskFacts
from service.world import WorldModel

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "data" / "acceptance_evidence"
LIVE_STORE = ROOT / "data" / "jarvis" / "owner" / "provider_credentials.json"

gemini_live = pytest.mark.skipif(not os.environ.get("ZEUS_LIVE_GEMINI"), reason="set ZEUS_LIVE_GEMINI=1 to call the free reasoning provider")
openai_live = pytest.mark.skipif(not os.environ.get("ZEUS_LIVE_OPENAI"), reason="set ZEUS_LIVE_OPENAI=1 to spend a few cents on the deep reasoner")
engineer_live = pytest.mark.skipif(not os.environ.get("ZEUS_LIVE_ENGINEER"), reason="set ZEUS_LIVE_ENGINEER=1 to spend a few cents on the standard engineer")

pytestmark = pytest.mark.live


def _credentials(tmp_path: Path) -> CredentialStore:
    """The owner's store when it exists; otherwise a scratch store filled from the environment."""

    if LIVE_STORE.is_file():
        store = CredentialStore(LIVE_STORE)
    else:
        store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    config = GatewayConfig.defaults()
    if not LIVE_STORE.is_file():
        store.import_environment({p.secret: p.credential_env for p in config.providers.values() if p.secret and p.credential_env})
    return store


def _gateway(tmp_path: Path, store: CredentialStore, *providers: str, paid: bool = False) -> ModelGateway:
    from runtime.cost_policy import CostPolicy

    config = GatewayConfig.load()
    for name in providers:
        if not store.has(name):
            pytest.skip(f"no {name} credential in the store or the environment")
        config = config.with_provider_enabled(name, True)
    for name in config.providers:
        if name not in providers and config.providers[name].kind not in {"ollama", "subscription_cli"}:
            config = config.with_provider_enabled(name, False)
    return ModelGateway(state_root=tmp_path / "state", config=config, credentials=store,
                        cost_policy=CostPolicy(allow_paid_api=paid, source="live-test"))


def _record(name: str, payload: dict, store: CredentialStore) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"live_{name}.json"
    text = json.dumps({"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}, indent=2, ensure_ascii=False, default=str)
    path.write_text(redact(text, store), encoding="utf-8")
    return path


def _chess_registry(tmp_path: Path) -> CapabilityRegistry:
    from test_intelligence_flow import chess_manifests

    registry = CapabilityRegistry(tmp_path / "registry.json")
    for manifest in chess_manifests():
        registry.register(manifest)
    return registry


def _flow(gateway: ModelGateway, registry: CapabilityRegistry, *, projects: list[str], mode: ChatMode = ChatMode.FREE) -> IntelligenceFlow:
    return IntelligenceFlow(gateway, capability_cards(registry) + builtin_cards(), [ProjectSummary(p) for p in projects], mode=mode)


def _summary(result) -> dict:
    return {"status": result.status, "provider": result.provider, "model": result.model, "goal": result.goal.to_dict() if result.goal else None,
            "plan_spec": result.plan_spec.to_dict() if result.plan_spec else None,
            "validation": result.validation.to_dict() if result.validation else None,
            "plan": result.plan.capability_ids if result.plan else None, "reason": result.reason, "question": result.question,
            "context": result.context.to_dict() if result.context else None, "metrics": result.metrics.to_dict()}


# --------------------------------------------------------------------------
# Gemini, the free reasoning role
# --------------------------------------------------------------------------

@gemini_live
def test_a_public_knowledge_question_is_answered_on_the_free_route_at_zero_cost(tmp_path):
    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "gemini")
    text = "Erkläre mir kurz den Unterschied zwischen kompetitiver und nichtkompetitiver Enzymhemmung."
    started = time.perf_counter()
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_question=True), mode=ChatMode.FREE))
    elapsed = time.perf_counter() - started
    _record("gemini_public_knowledge", {"request": text, "role": reply.role, "provider": reply.provider, "model": reply.model,
                                        "usage": reply.usage, "latency_seconds": round(reply.latency_seconds, 3), "wall_seconds": round(elapsed, 3),
                                        "estimated_eur": reply.estimated_eur, "actual_eur": reply.actual_eur,
                                        "thinking_level": reply.decision.thinking_level, "answer": reply.text}, store)
    assert reply.role == "reasoning.free" and reply.actual_eur == 0.0 and reply.estimated_eur == 0.0
    assert not reply.decision.offline_fallback
    lower = reply.text.lower()
    assert "kompetitiv" in lower and ("substrat" in lower or "aktives zentrum" in lower or "bindungsstelle" in lower or "allosterisch" in lower)
    assert gateway.governor.summary().month == 0.0
    assert gateway.status()["providers"]["gemini"]["state"] == "CONFIGURED"


@gemini_live
def test_the_recent_loss_in_context_is_read_as_the_chess_goal_and_planned_from_the_contracts(tmp_path):
    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "gemini")
    registry = _chess_registry(tmp_path)
    world = WorldModel(projects=lambda: ["Schach Training"])
    world.note_event("chess_game_finished", detail={"result": "loss"})
    flow = _flow(gateway, registry, projects=["Schach Training"])
    result = flow.run("fuck, schon wieder verloren", world.state())
    _record("gemini_chess_context", {"request": "fuck, schon wieder verloren", **_summary(result)}, store)
    assert result.status == "PLAN", (result.status, result.reason, result.question)
    assert result.goal.primary_goal in {"understand_recent_loss", "improve_chess"} and result.goal.rejected == []
    assert result.goal.relevant_recent_events == ["chess_game_finished"] or result.goal.relevant_project == "Schach Training"
    assert {"chess.capture_game", "chess.classify_errors", "stockfish.analyze"} <= set(result.context.to_dict()["cards"])
    assert result.plan.capability_ids == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
    assert result.validation.ok and result.plan_spec.source in {"provider", "reproposal"}, "the model proposed it; the validator accepted it"
    assert result.metrics.total_model_context_tokens < 3000 and result.metrics.provider_calls <= 3


@gemini_live
@pytest.mark.parametrize("sentence", [
    "boah, das war wieder nix eben",
    "die letzte runde ist mir komplett entglitten, keine ahnung warum",
    "kannst du dir mal anschauen, was da gerade schiefgelaufen ist?",
])
def test_unseen_paraphrases_reach_the_same_plan(tmp_path, sentence):
    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "gemini")
    registry = _chess_registry(tmp_path)
    world = WorldModel(projects=lambda: ["Schach Training"])
    world.note_event("chess_game_finished", detail={"result": "loss"})
    result = _flow(gateway, registry, projects=["Schach Training"]).run(sentence, world.state())
    _record("gemini_paraphrase_" + "".join(c for c in sentence[:24] if c.isalnum() or c == " ").strip().replace(" ", "_"),
            {"request": sentence, **_summary(result)}, store)
    assert result.status in {"PLAN", "CLARIFY"}, (result.status, result.reason)
    if result.status == "PLAN":
        assert result.plan.capability_ids == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
        assert result.validation.ok
    else:
        assert result.plan is None, "asking is allowed; acting on an unverified reading is not"


@gemini_live
def test_without_chess_context_the_same_words_are_not_a_chess_goal(tmp_path):
    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "gemini")
    registry = _chess_registry(tmp_path)
    world = WorldModel(projects=lambda: ["Physikum"])
    world.note_event("printer_jam")
    flow = _flow(gateway, registry, projects=["Physikum"])
    result = flow.run("fuck, schon wieder verloren", world.state())
    _record("gemini_negative_control", {"request": "fuck, schon wieder verloren", **_summary(result)}, store)
    # Nothing chess-related is retrieved without the event, the project or a
    # chess word; the model is not even asked.  If it were, it may not act.
    assert result.status != "PLAN" and result.plan is None
    if result.context is not None and not result.context.empty:
        assert result.goal is None or result.goal.primary_goal in {"", "none"} or result.status == "CLARIFY"


@gemini_live
def test_an_ambiguous_request_with_thin_context_becomes_a_question_not_a_plan(tmp_path):
    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "gemini")
    registry = _chess_registry(tmp_path)
    world = WorldModel(projects=lambda: ["Schach Training"])
    flow = _flow(gateway, registry, projects=["Schach Training"])
    result = flow.run("mach das mal irgendwie besser", world.state())
    _record("gemini_ambiguity", {"request": "mach das mal irgendwie besser", **_summary(result)}, store)
    assert result.status != "PLAN", (result.status, result.reason)
    assert result.plan is None
    if result.status == "CLARIFY":
        assert result.question, "a question, not an invented plan"


# --------------------------------------------------------------------------
# The deep reasoner, one small metered call
# --------------------------------------------------------------------------

@openai_live
def test_a_small_deep_reasoning_call_reserves_first_and_records_its_usage(tmp_path):
    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "openai", paid=True)
    text = ("Drei Fähigkeiten: A liefert aus einem Bild eine Liste von Zügen, B braucht eine Zugliste und liefert eine Bewertung, "
            "C braucht eine Bewertung und liefert einen Trainingsplan. Ziel: Trainingsplan aus einem Bild. "
            "Antworte als JSON: {\"order\": [ids], \"why\": kurz}.")
    schema = {"type": "object", "properties": {"order": {"type": "array", "items": {"type": "string"}}, "why": {"type": "string"}},
              "required": ["order", "why"]}
    request = GatewayRequest(prompt=text, facts=TaskFacts(text=text, subsystems=3, refers_to_context=True), mode=ChatMode.DEEP, schema=schema,
                             max_output_tokens=300, task_id="live-sol")
    decision, _ = gateway.plan(request)
    assert decision.role == "reasoning.deep", decision.reason
    reply = gateway.complete(request)
    gateway.report_task_outcome("live-sol", goal_verified=True)
    history = gateway.governor.history()
    _record("openai_deep_reasoning", {"request": text, "role": reply.role, "provider": reply.provider, "model": reply.model,
                                      "thinking_level": reply.decision.thinking_level, "usage": reply.usage,
                                      "latency_seconds": round(reply.latency_seconds, 3), "estimated_eur": reply.estimated_eur,
                                      "actual_eur": reply.actual_eur, "answer": reply.text, "ledger": history[-2:]}, store)
    assert reply.role == "reasoning.deep" and reply.provider == "openai"
    data = json.loads(reply.text)
    assert [x.strip().upper() for x in data["order"]] == ["A", "B", "C"]
    assert reply.usage["input_tokens"] > 0 and reply.usage["output_tokens"] > 0
    assert 0.0 < reply.actual_eur < 0.05, "a few cents at most"
    assert [row["kind"] for row in history[-2:]] == ["reserve", "settle"], "reserved before the call, settled after it"
    assert history[-1]["actual_eur"] == pytest.approx(reply.actual_eur)
    assert all("gemini" not in r.get("provider", "") for r in gateway.recent), "no free-provider attempt was made first"


# --------------------------------------------------------------------------
# The standard engineer, one tiny job
# --------------------------------------------------------------------------

@engineer_live
def test_the_standard_engineer_makes_a_one_line_change_that_verifies(tmp_path):
    from experts.api_engineer import ApiEngineerExpert
    from experts.contracts import ExpertJob, ExpertStatus
    from experts.gateway import ExpertGateway
    from runtime.cost_policy import CostLedger, CostPolicy

    store = _credentials(tmp_path)
    gateway = _gateway(tmp_path, store, "anthropic", paid=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "greet.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    expert = ApiEngineerExpert(gateway, "engineer.standard")
    assert expert.availability().available, expert.availability().detail
    experts = ExpertGateway([expert], policy=CostPolicy(allow_paid_api=True), ledger=CostLedger(CostPolicy(allow_paid_api=True)))
    import sys

    job = ExpertJob(goal="Change greet() so it returns 'hello world' instead of 'hello'. Touch nothing else.", workspace=workspace,
                    acceptance=[("greet says hello world", [sys.executable, "-c",
                                 "import sys; sys.path.insert(0, '.'); from greet import greet; assert greet() == 'hello world'; print('OK')"])],
                    metadata={"files": ["greet.py"], "task_id": "live-engineer"})
    result = experts.submit(job, provider_name="engineer.standard")
    _record("anthropic_standard_engineer", {"status": result.status.value, "blocker": result.blocker, "files": result.files_changed,
                                            "raw": result.raw, "evidence": result.test_evidence, "summary": result.summary}, store)
    assert result.status is ExpertStatus.COMPLETED and result.verified, (result.blocker, result.test_evidence)
    assert result.raw["cost_eur"] > 0.0 and result.raw["context_tokens"]["total_engineering_context_tokens"] > 0


def test_the_frontier_engineer_is_configured_but_not_exercised_without_authorization(tmp_path):
    """Configuration and adapter only: no money is spent on a meaningless frontier task."""

    config = GatewayConfig.load()
    binding = config.roles["engineer.frontier"]
    assert binding.enabled and binding.model and config.providers[binding.provider].kind == "anthropic"
    assert config.pricing_for("engineer.frontier") is not None, "priced, so estimable, so reservable"
    from gateway.providers import adapter_for

    assert adapter_for(config.providers[binding.provider].kind).kind == "anthropic"
