"""The authoritative semantic flow: the provider understands and proposes, the planner grounds and validates.

Helpers here (``ScriptedNetwork``, ``make_world``, ``chess_goal``, ``chess_plan``)
are shared by the held-out scenario tests and the acceptance tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from capabilities.contracts import SemanticContract
from capabilities.engineering_spec import build_engineering_spec
from capabilities.intelligence import Card, PlanSpec, PlanStepSpec, ProjectSummary, builtin_cards, retrieve, validate_plan
from capabilities.models import CapabilityHealth, CapabilityLifecycle, CapabilityManifest
from capabilities.planner import WorldState
from capabilities.service import ExecutionOutcome
from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from gateway.config import GatewayConfig
from gateway.health import ProviderStatus
from service.core import JarvisCore
from service.events import EventType
from test_gateway_integration import LocalChat, answer_text, ask
from test_model_gateway import FakeNetwork, FakeResponse, anthropic_reply, gemini_reply, openai_reply

GOAL_MARK = "Verständnisschicht"
PLAN_MARK = "Planungsschicht"

_REPLY = {"generativelanguage.googleapis.com": gemini_reply, "api.openai.com": openai_reply, "api.anthropic.com": anthropic_reply}


def _prompt_of(host: str, body: dict) -> str:
    if host == "generativelanguage.googleapis.com":
        return "".join(part.get("text", "") for c in body.get("contents", []) for part in c.get("parts", []))
    if host == "api.openai.com":
        parts = [str(body.get("instructions", ""))]
        for item in body.get("input", []):
            content = item.get("content", "") if isinstance(item, dict) else ""
            parts.append(content if isinstance(content, str) else "".join(str(c.get("text", "")) for c in content if isinstance(c, dict)))
        return "\n".join(parts)
    if host == "api.anthropic.com":
        parts = [str(body.get("system", ""))]
        for m in body.get("messages", []):
            content = m.get("content", "") if isinstance(m, dict) else ""
            parts.append(content if isinstance(content, str) else "".join(str(c.get("text", "")) for c in content if isinstance(c, dict)))
        return "\n".join(parts)
    return json.dumps(body, ensure_ascii=False)


class ScriptedNetwork(FakeNetwork):
    """The reasoning provider, scripted: a function of the prompt for GoalSpecs and one for PlanSpecs.

    Whichever configured provider the router picks (the roles are abstract;
    the vendor is configuration), the script answers it in that vendor's
    wire format.  ``goal_prompts`` / ``plan_prompts`` record what the model saw.
    """

    def __init__(self, goal: Callable[[str], dict], plan: Callable[[str], dict] | None = None) -> None:
        super().__init__()
        self.goal_fn = goal
        self.plan_fn = plan or (lambda prompt: {"steps": [], "reason": "no plan", "missing": ""})
        self.goal_prompts: list[str] = []
        self.plan_prompts: list[str] = []
        self.hosts: list[str] = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        host = request.full_url.split("/")[2]
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        reply = _REPLY.get(host)
        if reply is None:
            return super().__call__(request, timeout)
        answer = self.responses.get(host)
        if isinstance(answer, Exception):
            raise answer
        prompt = _prompt_of(host, body) or json.dumps(body, ensure_ascii=False)
        self.hosts.append(host)
        if GOAL_MARK in prompt:
            self.goal_prompts.append(prompt)
            return FakeResponse(reply(json.dumps(self.goal_fn(prompt))))
        if PLAN_MARK in prompt:
            self.plan_prompts.append(prompt)
            return FakeResponse(reply(json.dumps(self.plan_fn(prompt))))
        if "semantische Steuerung" in prompt:
            return FakeResponse(reply(json.dumps({"operation": "conversation", "target": "", "confidence": 0.9, "reason": "chat"})))
        return FakeResponse(reply("Das tut mir leid. Nächstes Mal klappt es."))


def manifest(cid: str, description: str, contract: dict, **extra) -> CapabilityManifest:
    return CapabilityManifest(cid, description, lifecycle=CapabilityLifecycle.ACTIVE.value,
                              health={"state": "healthy", "health": CapabilityHealth.HEALTHY.value}, contract=contract,
                              family=cid.split(".", 1)[0], source_location=f"/installed/{cid}", **extra)


def make_world(tmp_path: Path, net: FakeNetwork, *, manifests=(), projects=(), execute=None):
    """A real core and kernel; only the network and the capability executor are ours."""

    local = LocalChat()
    config_root = tmp_path / "config"
    config_root.mkdir(exist_ok=True)
    cfg = GatewayConfig.defaults().with_provider_enabled("gemini", True).with_provider_enabled("openai", True)
    cfg.save(config_root / "providers.json")
    (config_root / "owner").mkdir(exist_ok=True)
    (config_root / "owner" / "spending.json").write_text(json.dumps({"paid_api": True}), encoding="utf-8")
    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state", config_root=config_root, enable_research_tools=False))
    kernel.local_provider = lambda tier: local  # type: ignore[assignment]
    gateway = kernel.gateway
    gateway.transport._opener = net
    gateway.credentials.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    gateway.credentials.set("openai", "sk-openaitestkey00000000000000000")
    core = JarvisCore(kernel=kernel, identity=Identity())
    core.language = "de"
    for m in manifests:
        core.capabilities.registry.register(m)
    executed: list[str] = []
    if execute is None:
        def execute(capability_id: str, payload=None) -> ExecutionOutcome:  # noqa: E306
            executed.append(capability_id)
            return ExecutionOutcome(capability_id=capability_id, ok=True, output={"ok": True, "capability": capability_id})
    core.capabilities.execute = execute  # type: ignore[assignment]
    core.owner_projects = lambda: [{"title": t, "origin": "owner", "goal": g} for t, g in projects]  # type: ignore[assignment]
    return core, kernel, local, executed


def tool_events(events, needle: str):
    return [e.payload for e in events if e.type is EventType.TOOL and needle in str(e.payload.get("summary", ""))]


def chess_manifests():
    return [
        manifest("chess.capture_game", "Stores the game that was just played as a record.",
                 {"consumes": ["chess_game_finished"], "produces": ["chess_game_record"], "events": ["chess_game_finished"],
                  "related_projects": ["Schach Training"], "domain": "chess"}),
        manifest("stockfish.analyze", "Runs an engine analysis over a game record.",
                 {"consumes": ["chess_game_record"], "produces": ["chess_engine_analysis"], "latency_class": "slow", "domain": "chess"}),
        manifest("chess.classify_errors", "Classifies the mistakes in an analysed game and updates the error profile.",
                 {"consumes": ["chess_engine_analysis"], "produces": ["chess_error_profile_update"],
                  "goals": ["understand_recent_loss", "improve_chess"], "related_projects": ["Schach Training"], "domain": "chess"}),
    ]


def chess_goal(prompt: str) -> dict:
    """A capable provider's reading: a chess goal only when the offered vocabulary AND the context carry it."""

    has_event = "chess_game_finished (event)" in prompt
    has_project = "(project: Schach Training" in prompt
    request = prompt.lower().split("anfrage:")[-1]
    wants_training = "training" in request or "besser" in request
    offered = [g for g in ("understand_recent_loss", "improve_chess") if g in prompt.split("Ziel-Vokabular")[-1]]
    if wants_training and "improve_chess" in offered:
        offered = ["improve_chess"] + [g for g in offered if g != "improve_chess"]
    if offered and (has_event or has_project):
        return {"primary_goal": offered[0], "secondary_goals": offered[1:2] if wants_training else [],
                "referenced_entities": [], "relevant_project": "Schach Training" if has_project else "none",
                "relevant_recent_events": ["chess_game_finished"] if has_event else [],
                "constraints": [], "privacy_class": "public", "ambiguity": 0.1, "confidence": 0.9,
                "reason": "the owner just lost a game in the chess training project"}
    return {"primary_goal": "none", "secondary_goals": [], "referenced_entities": [], "relevant_project": "none",
            "relevant_recent_events": [], "constraints": [], "privacy_class": "public", "ambiguity": 0.6, "confidence": 0.3,
            "reason": "nothing in the context supports a goal", "clarification_question": "Worum geht es?"}


def chess_plan(prompt: str) -> dict:
    return {"steps": [
        {"capability_id": "chess.capture_game", "intended_effect": "chess_game_record", "why": "the game must be recorded first",
         "arguments_json": "{}", "role": "required"},
        {"capability_id": "stockfish.analyze", "intended_effect": "chess_engine_analysis", "why": "engine analysis of the record",
         "arguments_json": "{}", "role": "required"},
        {"capability_id": "chess.classify_errors", "intended_effect": "chess_error_profile_update", "why": "the mistakes, classified",
         "arguments_json": "{}", "role": "required"},
    ], "expected_final_goal": ["chess_error_profile_update"], "required_permissions": [], "reason": "capture, analyse, classify"}


# ---------------------------------------------------------------------------
# Retrieval: two stages, a few hundred tokens
# ---------------------------------------------------------------------------

def _chess_cards() -> list[Card]:
    return [
        Card("chess.capture_game", SemanticContract.from_dict({"consumes": ["chess_game_finished"], "produces": ["chess_game_record"],
                                                               "events": ["chess_game_finished"], "related_projects": ["Schach Training"],
                                                               "domain": "chess"}), "Stores the game just played.", words=frozenset({"chess", "schach", "partie"})),
        Card("stockfish.analyze", SemanticContract.from_dict({"consumes": ["chess_game_record"], "produces": ["chess_engine_analysis"],
                                                              "latency_class": "slow", "domain": "chess"}), "Engine analysis of a record.",
             words=frozenset({"stockfish", "analyse", "chess"})),
        Card("chess.classify_errors", SemanticContract.from_dict({"consumes": ["chess_engine_analysis"], "produces": ["chess_error_profile_update"],
                                                                  "goals": ["understand_recent_loss", "improve_chess"],
                                                                  "related_projects": ["Schach Training"], "domain": "chess"}),
             "Classifies mistakes and updates the error profile.", words=frozenset({"chess", "fehler", "schach"})),
        Card("archive.zip.create", SemanticContract.from_dict({"produces": ["archive.zip.create.result"], "inferred": True}),
             "Package a folder into a zip.", words=frozenset({"zip", "archiv", "ordner"})),
    ]


def test_retrieval_sends_only_what_the_request_is_about():
    state = WorldState.build(projects=["Schach Training", "Physikum", "Steuer 2026"], extra={"chess_game_finished": "event",
                                                                                              "chess_game_finished.result_loss": "event",
                                                                                              "printer_jam": "event"})
    projects = [ProjectSummary("Schach Training", "besser werden"), ProjectSummary("Physikum", "lernen"), ProjectSummary("Steuer 2026", "")]
    context = retrieve("fuck, schon wieder verloren", state, _chess_cards() + builtin_cards(), projects)
    ids = [c.capability_id for c in context.cards]
    assert "chess.capture_game" in ids and "chess.classify_errors" in ids
    assert "stockfish.analyze" in ids and "connects the chain" in context.reasons["stockfish.analyze"]
    assert "archive.zip.create" not in ids and not any(c.builtin for c in context.cards), "nothing the request is not about"
    assert [t for t, _ in context.events] == ["chess_game_finished", "chess_game_finished.result_loss"], "the printer jam is not sent"
    assert [p.title for p in context.projects] == ["Schach Training"], "Physikum and Steuer are not sent"
    world_text = context.world_text()
    assert "Physikum" not in world_text and "printer" not in world_text
    from gateway.estimate import estimate_tokens

    assert estimate_tokens(context.summaries_text()) < 400, "stage 1 is a few hundred tokens at most"
    empty = retrieve("wie wird das wetter", WorldState.build(projects=["Physikum"]), _chess_cards() + builtin_cards(), projects[1:])
    assert empty.empty, "no event, no chess project, no capability word: nothing is retrieved"


def test_stage_two_details_are_only_loaded_for_retrieved_cards():
    card = _chess_cards()[1]
    assert "chess_engine_analysis" in card.detail() and "inputs" in card.detail()
    assert len(card.summary()) < len(card.detail())


# ---------------------------------------------------------------------------
# GoalSpec: closed vocabulary, grounded
# ---------------------------------------------------------------------------

def test_an_invented_goal_or_project_is_dropped_not_acted_on(tmp_path):
    inventive = ScriptedNetwork(lambda prompt: {"primary_goal": "teleport_owner", "secondary_goals": ["understand_recent_loss"],
                                                 "relevant_project": "Weltherrschaft", "relevant_recent_events": ["nuclear_launch"],
                                                 "confidence": 0.99, "ambiguity": 0.0, "reason": "x"})
    core, kernel, local, executed = make_world(tmp_path, inventive, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    preview = core.compose_contract_preview("schon wieder verloren")
    goal = preview["goal"]
    assert goal["primary_goal"] == "" and "teleport_owner" in goal["rejected"], "an invented goal is dropped, not replaced"
    assert goal["secondary_goals"] == ["understand_recent_loss"], "the real token survives, as what it was: secondary"
    assert goal["relevant_project"] == "" and goal["relevant_recent_events"] == []
    assert preview["status"] != "PLAN" and preview["plan"] is None
    ask(core, "schon wieder verloren", wait=30)
    assert executed == []


def test_no_provider_route_is_typed_intelligence_unavailable_not_a_local_guess(tmp_path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    local.generate_structured = lambda prompt, schema, **kw: json.dumps({"primary_goal": "understand_recent_loss", "confidence": 0.95,
                                                                         "reason": "local guess"})
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    assert preview["status"] == "FREE_INTELLIGENCE_UNAVAILABLE" and preview["goal"] is None and preview["plan"] is None
    assert "offline fallback" in preview["reason"]
    assert not any("Verständnisschicht" in c for c in local.calls), "the local model was never asked for a GoalSpec"
    events = ask(core, "fuck, schon wieder verloren", wait=30)
    assert executed == [] and "weder ein bezahltes Modell noch das lokale Modell" in answer_text(events)
    core.set_chat_mode("AUTO")
    kernel.gateway.health.note("openai", ProviderStatus.QUOTA_EXHAUSTED)  # the paid route is gone too
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    assert preview["status"] == "INTELLIGENCE_UNAVAILABLE", "outside FREE the same situation is the general typed status"
    assert not any("Verständnisschicht" in c for c in local.calls), "still no GoalSpec from the local model"


# ---------------------------------------------------------------------------
# PlanSpec: validated, corrected once, proven
# ---------------------------------------------------------------------------

def test_plan_validation_names_the_exact_problem():
    cards = {c.capability_id: c for c in _chess_cards()}
    state = WorldState.build(extra={"chess_game_finished": "event"})
    bad = PlanSpec(steps=[PlanStepSpec("stockfish.analyze", "chess_engine_analysis"),
                          PlanStepSpec("chess.classify_errors", "world_peace"),
                          PlanStepSpec("chess.teleport", "x")])
    validation = validate_plan(bad, state, cards, ["chess_error_profile_update"])
    kinds = [(p.step, p.kind) for p in validation.problems]
    assert (0, "unmet_requirement") in kinds and (1, "invalid_effect") in kinds and (2, "unknown_capability") in kinds
    assert validation.problems[0].missing == ["chess_game_record"]
    good = PlanSpec(steps=[PlanStepSpec("chess.capture_game", "chess_game_record"), PlanStepSpec("stockfish.analyze", "chess_engine_analysis"),
                           PlanStepSpec("chess.classify_errors", "chess_error_profile_update")])
    ok = validate_plan(good, state, cards, ["chess_error_profile_update"])
    assert ok.ok and ok.plan is not None and ok.plan.capability_ids == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
    served = validate_plan(good, state, cards, ["goal.understand_recent_loss"])
    assert served.ok, "a goal token is reached by a step whose contract serves it"
    short = validate_plan(PlanSpec(steps=[PlanStepSpec("chess.capture_game", "chess_game_record")]), state, cards, ["goal.understand_recent_loss"])
    assert [p.kind for p in short.problems] == ["goal_not_reached"]


def test_an_invalid_proposal_gets_the_exact_problem_back_once(tmp_path):
    proposals: list[dict] = [
        {"steps": [{"capability_id": "stockfish.analyze", "intended_effect": "chess_engine_analysis", "arguments_json": "{}"},
                   {"capability_id": "chess.classify_errors", "intended_effect": "chess_error_profile_update", "arguments_json": "{}"}],
         "reason": "skipped the capture"},
        chess_plan(""),
    ]
    net = ScriptedNetwork(chess_goal, lambda prompt: proposals.pop(0))
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    assert preview["status"] == "PLAN" and preview["plan"]["capability_ids"] == ["chess.capture_game", "stockfish.analyze", "chess.classify_errors"]
    assert len(net.plan_prompts) == 2
    assert "UNGÜLTIG" in net.plan_prompts[1] and "chess_game_record" in net.plan_prompts[1], "the exact missing effect went back"
    assert preview["plan_spec"]["source"] == "reproposal"


def test_when_the_provider_cannot_propose_a_valid_plan_the_planner_decides(tmp_path):
    net = ScriptedNetwork(chess_goal, lambda prompt: {"steps": [{"capability_id": "chess.classify_errors", "intended_effect": "chess_error_profile_update",
                                                                 "arguments_json": "{}"}], "reason": "wrong"})
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    assert preview["status"] == "PLAN" and preview["plan_spec"]["source"] == "deterministic"
    assert len(net.plan_prompts) == 2, "one round of exact feedback, then the planner"
    assert "deterministic path used" in preview["metrics"]["notes"][0]


def test_a_missing_effect_yields_an_engineering_spec_and_no_engineer_without_a_route(tmp_path, monkeypatch):
    net = ScriptedNetwork(chess_goal, lambda prompt: {"steps": [], "reason": "nothing produces a training plan", "missing": "chess_training_plan"})
    manifests = chess_manifests()[:2] + [manifest("chess.training_planner", "Turns an error profile into a training plan.",
                                                 {"consumes": ["chess_error_profile_update"], "produces": ["chess_training_plan"],
                                                  "goals": ["improve_chess"], "related_projects": ["Schach Training"], "domain": "chess"})]
    core, kernel, local, executed = make_world(tmp_path, net, manifests=manifests, projects=[("Schach Training", "besser werden")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    from service.engineering import Engineer, EngineerDecision, EngineeringNeed

    monkeypatch.setattr(core, "_engineer_for_capability",
                        lambda goal, spec=None: EngineerDecision(EngineeringNeed.CAPABILITY_MISSING, Engineer.NONE,
                                                                 "Codex is NOT_INSTALLED and no metered engineer is permitted"))
    events = ask(core, "mach daraus was für mein training", wait=30)
    assert executed == []
    proven = tool_events(events, "missing capability proven")
    assert proven, [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    spec = proven[0]["engineering_spec"]
    assert spec["goal_spec"]["primary_goal"] == "improve_chess"
    assert "chess_error_profile_update" in spec["missing_effects"]
    assert spec["closest_partial_plan"] == ["chess.capture_game", "stockfish.analyze"]
    assert spec["reusable_capabilities"] == ["chess.capture_game", "stockfish.analyze"]
    assert spec["suggested_contract"]["produces"] == ["chess_error_profile_update"]
    assert any("contract.json" in c for c in spec["acceptance_criteria"])
    assert spec["catalog_context"] and spec["metrics"]["catalog_context_tokens"] > 0
    assert spec["metrics"]["flow"]["provider_calls"] == 2
    text = answer_text(events)
    assert "Codex is NOT_INSTALLED" in text and "spezifiziert" in text
    saved = list((tmp_path / "state" / "engineering_specs").glob("*.json"))
    assert saved and json.loads(saved[0].read_text(encoding="utf-8"))["spec_id"] == spec["spec_id"]
    routing = tool_events(events, "engineering routing")
    assert routing and routing[0]["engineering"]["engineer"] == "NONE"
    assert not any(e.type is EventType.PROGRESS for e in events), "no engineer was called"


def test_engineering_routing_for_a_capability_is_decided_before_execution(tmp_path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests())
    core.set_chat_mode("BUILD")
    from capabilities.planner import MissingCapabilityEvidence

    evidence = MissingCapabilityEvidence(goal=("chess_training_plan",), state=WorldState.build(), available_effects=(),
                                         reachable_effects=(), missing_effects=("chess_training_plan",), closest_partial_plan=None,
                                         unproducible_effects=("chess_training_plan",))
    spec = build_engineering_spec("mach daraus was für mein training", {"primary_goal": "improve_chess"}, evidence, [])
    decision = core._engineer_for_capability("mach daraus was für mein training", spec)
    assert decision.engineer.value in {"API", "CODEX", "NONE"}
    assert decision.task_class.startswith("engineering."), "a typed engineering class, decided before any engineer runs"
    roles = {c["role"] for c in decision.candidates}
    assert {"engineer.standard", "engineer.frontier"} <= roles, "the router ranked the abstract engineer roles"
    if decision.engineer.value == "API":
        assert decision.provider_name in {"engineer.standard", "engineer.frontier"}, "a role, never a vendor name"


# ---------------------------------------------------------------------------
# Real event sources
# ---------------------------------------------------------------------------

def test_verified_receipts_become_world_events_and_unverified_ones_do_not(tmp_path):
    core, kernel, local, executed = make_world(tmp_path, ScriptedNetwork(chess_goal, chess_plan))
    from runtime.receipts import Receipt, Verification

    verified = Receipt(kind="project.create", executor="projects", ok=True, detail="created",
                       verifications=(Verification(check="record exists", passed=True, observed="yes"),),
                       evidence={"project": {"title": "Schach Training"}})
    unverified = Receipt(kind="file.write", executor="files", ok=True, detail="written", verifications=(), evidence={"path": "x.md"})
    core._note_world_receipt(verified)
    core._note_world_receipt(unverified)
    facts = core.world.state().facts
    assert "project_created" in facts and "project_created.title_schach_training" in facts
    assert "file_written" not in facts, "an unverified receipt is not a world event"
    playing = Receipt(kind="music.play", executor="music", ok=True, detail="playing",
                      verifications=(Verification(check="media session reports playing", passed=True, observed="Spotify"),),
                      evidence={"app": "Spotify", "title": "Sonne"})
    core._note_world_receipt(playing)
    facts = core.world.state().facts
    assert "media_playing" in facts and "media_playing.app_spotify" in facts
    written = Receipt(kind="file.write", executor="files", ok=True, detail="written",
                      verifications=(Verification(check="file exists", passed=True, observed="12 bytes"),), evidence={"path": "notes/x.md"})
    core._note_world_receipt(written)
    assert "file_written" in core.world.state().facts


# ---------------------------------------------------------------------------
# Metrics and the single path
# ---------------------------------------------------------------------------

def test_the_flow_reports_its_four_context_measurements(tmp_path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    m = preview["metrics"]
    assert 0 < m["capability_summary_tokens"] < 400
    assert m["detailed_contract_tokens"] > m["capability_summary_tokens"], "stage 2 is loaded only after selection"
    assert 0 < m["world_context_tokens"] < 200
    assert m["total_model_context_tokens"] >= m["capability_summary_tokens"] + m["world_context_tokens"]
    assert m["total_model_context_tokens"] < 3000 and m["provider_calls"] == 2


def test_the_composer_no_longer_plans():
    from service.composer import Composer

    assert not hasattr(Composer, "plan") and not hasattr(Composer, "replan"), "one authoritative composition path"
    assert hasattr(Composer, "parse") and hasattr(Composer, "execute")
