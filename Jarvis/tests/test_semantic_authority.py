"""The legacy local model never decides what the owner means -- on any path, after any provider failure.

Observed live on 2026-09-15: a FREE request beginning with "schreibe" was
routed as an action, the Gemini GoalSpec call hung for 134 s, and after the
typed FREE_INTELLIGENCE_UNAVAILABLE the local 4B model was asked to plan.
It chose knowledge.search and the owner got no answer.  These tests pin the
three fixes: writing verbs route by their object, semantic calls are bounded
and a timeout is its own failure class, and a provider outage ends in a
deterministic execution or one clean typed message -- never in a local guess
and never in a paid escalation from FREE.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error

import pytest

from capabilities.intelligence import IntelligenceFlow, ProjectSummary, builtin_cards, capability_cards
from gateway.config import GatewayConfig
from gateway.gateway import SEMANTIC_CALL_TIMEOUT_SECONDS, GatewayBrainProvider, GatewayRequest
from gateway.health import GatewayError, ProviderStatus
from gateway.modes import ChatMode
from gateway.task import TaskFacts
from service.actions import parse_file_write
from service.events import EventType
from service.intent import Intent, classify
from service.routing import route
from service.semantic import SemanticAuthorityUnavailable, SemanticGoal
from test_gateway_integration import answer_text, ask
from test_intelligence_flow import ScriptedNetwork, chess_goal, chess_manifests, chess_plan, make_world
from test_model_gateway import FakeNetwork, LocalStub, cfg, creds, gemini_reply, make_gateway, openai_reply  # noqa: F401 - fixtures

GEMINI = "generativelanguage.googleapis.com"
OPENAI = "api.openai.com"
FREE_MESSAGE = "weder ein bezahltes Modell noch das lokale Modell"
GENERAL_MESSAGE = "Das lokale Modell entscheidet so etwas nicht"


def http_503() -> urllib.error.HTTPError:
    body = io.BytesIO(json.dumps({"error": {"code": 503, "message": "This model is currently experiencing high demand"}}).encode("utf-8"))
    return urllib.error.HTTPError("https://" + GEMINI + "/x", 503, "Service Unavailable", hdrs=None, fp=body)


def tools(events) -> list[dict]:
    return [e.payload for e in events if e.type is EventType.TOOL]


def semantic_local_calls(local) -> list[str]:
    """Prompts the local model received that would have made it decide something."""

    return [c for c in local.calls if any(mark in c for mark in ("Verständnisschicht", "operation", "machine-readable action", "JSON"))]


def outage_world(tmp_path, failure: Exception, *, mode: str = "FREE"):
    net = FakeNetwork()
    net.responses[GEMINI] = failure
    net.responses[OPENAI] = openai_reply("bezahlte Antwort")
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001 - the free pool's back-off is real time otherwise
    core.set_chat_mode(mode)
    return net, core, kernel, local, executed


# ---------------------------------------------------------------------------
# 1 + 7 + 8: a semantic timeout is bounded, typed, and never reaches the local model or a paid provider
# ---------------------------------------------------------------------------

def test_gemini_semantic_timeout_never_reaches_the_local_model_and_never_escalates(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, TimeoutError("timed out"))
    started = time.perf_counter()
    events = ask(core, "Lege das Ergebnis von gestern ab.", wait=30)
    elapsed = time.perf_counter() - started
    text = answer_text(events)
    assert FREE_MESSAGE in text, text
    assert executed == [] and semantic_local_calls(local) == [], local.calls
    assert not any(OPENAI in r["url"] for r in net.requests), "FREE never escalates to a paid provider"
    gemini = [r for r in net.requests if GEMINI in r["url"]]
    assert 1 <= len(gemini) <= 2, "one bounded try per pool model; a model that hung is not waited on twice"
    assert all(r["timeout"] == SEMANTIC_CALL_TIMEOUT_SECONDS for r in gemini), [r["timeout"] for r in gemini]
    assert kernel.gateway.health.status("gemini") is ProviderStatus.TIMEOUT, "a hang is recorded as a timeout, not a generic outage"
    summaries = [str(p.get("summary")) for p in tools(events)]
    assert any("FREE_INTELLIGENCE_UNAVAILABLE" in s and "timeout" in s for s in summaries), summaries
    assert kernel.gateway.status()["spend"]["month"] == 0.0
    assert elapsed < 10.0, f"the fake provider fails instantly; nothing in the path may wait: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 2: the whole free pool unavailable (503) -> the same typed outcome, no local semantic call
# ---------------------------------------------------------------------------

def test_free_pool_unavailable_never_asks_the_local_model_to_plan(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, http_503())
    events = ask(core, "Lege das Ergebnis von gestern ab.", wait=30)
    assert FREE_MESSAGE in answer_text(events)
    assert executed == [] and semantic_local_calls(local) == []
    assert not any(OPENAI in r["url"] for r in net.requests)
    assert kernel.gateway.health.status("gemini") is ProviderStatus.PROVIDER_UNAVAILABLE
    assert any("provider_unavailable" in str(p.get("summary")) for p in tools(events))
    # The message is delivered once, as a message -- the owner is never left without an answer.
    assert len([e for e in events if e.type is EventType.MESSAGE]) == 1


# ---------------------------------------------------------------------------
# 3: an ambiguous request after the outage gets one clean message, in FREE and outside it
# ---------------------------------------------------------------------------

def test_ambiguous_request_after_the_outage_gets_a_clean_typed_message(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, TimeoutError("timed out"))
    events = ask(core, "Leg das mal ordentlich ab, du weißt schon was.", wait=30)
    text = answer_text(events)
    assert FREE_MESSAGE in text and "schreibe Datei" in text, "FREE: the outage, and the one shape that runs without a model"
    assert semantic_local_calls(local) == [] and executed == []
    # Outside FREE the same outage is the general typed status; still no local decision, no paid call on this path.
    core.set_chat_mode("AUTO")
    net.responses[OPENAI] = TimeoutError("timed out")
    events = ask(core, "Leg das mal ordentlich ab, du weißt schon was.", wait=30)
    text = answer_text(events)
    assert GENERAL_MESSAGE in text, text
    assert semantic_local_calls(local) == [] and executed == []
    assert any("INTELLIGENCE_UNAVAILABLE" in str(p.get("summary")) and "FREE_" not in str(p.get("summary")) for p in tools(events))


def test_a_free_chat_question_during_the_outage_gets_the_typed_message_not_a_blame_on_the_local_model(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, http_503())
    events = ask(core, "Erkläre mir die kompetitive Enzymhemmung.", wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert FREE_MESSAGE in message["text"] and "lokale KI" not in message["text"], message["text"]
    assert message["backend"] == "intelligence"
    assert local.calls == [], "FREE: the local model is not asked for prose in place of the free provider either"
    assert not any(OPENAI in r["url"] for r in net.requests)
    states = [e.payload.get("state") for e in events if e.type is EventType.STATE]
    assert "error" not in states, states
    assert any("conversation path" in str(p.get("summary")) for p in tools(events))


class StallingStream:
    """An event stream that serves some chunks and then hangs until the socket times out."""

    def __init__(self, blocks: list[str]) -> None:
        self._blocks = blocks
        self.status = 200

    def __iter__(self):
        for block in self._blocks:
            for line in block.split("\n"):
                yield (line + "\n").encode("utf-8")
            yield b"\n"
        raise TimeoutError("timed out")

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StallingNetwork(FakeNetwork):
    """Serves the free provider's stream from a body that stalls after its first chunks."""

    def __init__(self, blocks: list[str]) -> None:
        super().__init__()
        self.blocks = blocks

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        if GEMINI in request.full_url and "alt=sse" in request.full_url:
            return StallingStream(self.blocks)
        return super().__call__(request, timeout)


def test_a_stream_that_stalls_mid_answer_is_stored_incomplete_not_blamed_on_the_local_model(tmp_path):
    from test_streaming import gemini_chunks

    chunks = gemini_chunks(["Bei der kompetitiven Hemmung ", "konkurriert ein Inhibitor"])[:-1]  # no finish, then the stall
    net = StallingNetwork(chunks)
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001
    core.set_chat_mode("FREE")
    events = ask(core, "Erkläre mir die kompetitive Enzymhemmung.", wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "Bei der kompetitiven Hemmung", message["text"]
    assert message["backend"] == "gemini/gemini-3.8-flash" and "lokale KI" not in message["text"], message["backend"]
    assert message["meta"]["provenance"]["model"] == "gemini-3.8-flash" and message["meta"]["provenance"]["interrupted"] == "timeout"
    completion = message["meta"]["completion"]
    assert completion["complete"] is False and completion["finish_reason"] == "stream_interrupted:timeout" and completion["aborted"] is False
    assert local.calls == [] and kernel.gateway.health.status("gemini") is ProviderStatus.TIMEOUT
    assert "error" not in [e.payload.get("state") for e in events if e.type is EventType.STATE]


# ---------------------------------------------------------------------------
# 4 + 6: a spelled-out file write runs from its syntax -- with or without a reasoning provider
# ---------------------------------------------------------------------------

def test_a_spelled_out_file_write_runs_without_any_model(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, TimeoutError("timed out"))
    events = ask(core, "Schreibe eine Datei zeus_sem.txt mit Inhalt Hallo Welt.", wait=30)
    target = core.actions.workspace / "zeus_sem.txt"
    assert target.read_text(encoding="utf-8") == "Hallo Welt"
    assert net.requests == [] and local.calls == [], "deterministic syntax: no provider and no local model was consulted"
    assert any("deterministic action: file.write zeus_sem.txt" in str(p.get("summary")) for p in tools(events))
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["backend"] == "tools.write_file" and "zeus_sem.txt" in message["text"]


def test_schreibe_datei_routes_to_the_file_action_and_writes_it_with_the_provider_up(tmp_path):
    classification = classify("Schreibe eine Datei test.txt mit Inhalt Hallo.")
    assert classification.intent is Intent.ACTION and "datei" in classification.reason
    assert parse_file_write("Schreibe die Datei notiz.txt mit dem Inhalt: „Milch kaufen“").arguments == {"path": "notiz.txt", "content": "Milch kaufen"}
    assert parse_file_write("Schreibe mir eine Erklärung zur Glykolyse.") is None
    assert parse_file_write("Erstelle test.txt") is None, "a file without content is not deterministic"
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("nicht gefragt")
    core, kernel, local, executed = make_world(tmp_path, net)
    events = ask(core, "Schreibe eine Datei test.txt mit Inhalt Hallo.", wait=30)
    assert (core.actions.workspace / "test.txt").read_text(encoding="utf-8") == "Hallo"
    assert net.requests == [], "a write that is spelled out needs no model, even when one is reachable"
    assert "test.txt" in answer_text(events)


# ---------------------------------------------------------------------------
# 5: writing verbs route by their object, not by themselves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Schreibe mir eine Erklärung zur Glykolyse.",
    "Schreibe mir eine kurze Erklärung zur Keto-Enol-Tautomerie.",
    "Erkläre kurz die Rolle von H₂O und schreibe die Gleichung ΔG = ΔG° + RT ln Q mit einer kurzen Erklärung.",
    "Write me a summary of the Krebs cycle.",
])
def test_writing_prose_stays_conversational(text):
    classification = classify(text)
    assert classification.intent is Intent.CONVERSATION and classification.matched == "prose-composition", classification
    assert route(text).intent.value == "conversation" and route(text).reading.composition


def test_a_message_is_an_action_only_where_something_can_send_it():
    text = "Schreibe meiner Freundin eine Nachricht."
    assert classify(text).intent is Intent.CONVERSATION and "nothing registered that could send" in classify(text).reason
    assert classify(text, capability_names=["messaging.send"]).intent is Intent.ACTION
    assert classify("Schreibe eine Datei test.txt mit Inhalt Hallo.").intent is Intent.ACTION
    assert classify("schreib das in notizen.md").intent is Intent.ACTION
    assert classify("Leg eine Notiz an: Milch kaufen").intent is Intent.ACTION


def test_schreibe_mir_eine_erklaerung_is_answered_by_the_free_provider_not_executed(tmp_path):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("Die Glykolyse ist der Abbau von Glucose zu Pyruvat.")
    core, kernel, local, executed = make_world(tmp_path, net)
    core.set_chat_mode("FREE")
    events = ask(core, "Schreibe mir eine Erklärung zur Glykolyse.", wait=30)
    assert "Glykolyse ist der Abbau" in answer_text(events)
    summaries = [str(p.get("summary")) for p in tools(events)]
    assert any(s.startswith("routed: conversation") for s in summaries), summaries
    assert not any(s.startswith("executing") or "semantic goal" in s for s in summaries), summaries
    assert executed == [] and semantic_local_calls(local) == []
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["backend"].startswith("gemini/")


# ---------------------------------------------------------------------------
# 8: the decision path is bounded -- per call, and as a whole
# ---------------------------------------------------------------------------

def test_a_socket_timeout_is_its_own_status_distinct_from_429_503_auth_and_malformed(tmp_path, cfg, creds):
    def failing(exc):
        net = FakeNetwork()
        net.responses[GEMINI] = exc
        return net

    request = GatewayRequest(prompt="Klassifiziere", facts=TaskFacts(text="Klassifiziere", is_question=True), mode=ChatMode.FREE,
                             timeout_seconds=7.5)
    seen = {}
    for name, exc in {"timeout": TimeoutError("timed out"), "wrapped_timeout": urllib.error.URLError(TimeoutError("_ssl.c: timed out")),
                      "unreachable": urllib.error.URLError("no route to host"), "overloaded": http_503()}.items():
        net = failing(exc)
        gateway = make_gateway(tmp_path / name, cfg, creds, net)
        with pytest.raises(GatewayError) as caught:
            gateway.complete(request)
        seen[name] = caught.value
        assert all(r["timeout"] == 7.5 for r in net.requests), "the request's own bound reaches the socket"
    assert seen["timeout"].status is ProviderStatus.TIMEOUT and seen["wrapped_timeout"].status is ProviderStatus.TIMEOUT
    assert seen["unreachable"].status is ProviderStatus.PROVIDER_UNAVAILABLE and seen["overloaded"].status is ProviderStatus.PROVIDER_UNAVAILABLE
    assert {a["failure_class"] for a in seen["timeout"].attempts} == {"timeout"}
    assert [a["model"] for a in seen["timeout"].attempts] == ["gemini-3.8-flash", "gemini-3.7-flash"], "each pool model once, no second wait"
    assert len(seen["overloaded"].attempts) == 4, "a 503 keeps the bounded retry per model"
    assert ProviderStatus.TIMEOUT.is_outage and ProviderStatus.TIMEOUT.value == "timeout"


def test_the_semantic_decision_path_stops_at_its_deadline_without_another_call(tmp_path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    core.set_chat_mode("FREE")
    cards = capability_cards(core.capabilities.registry) + builtin_cards()
    projects = [ProjectSummary("Schach Training", "")]
    flow = IntelligenceFlow(kernel.gateway, cards, projects, mode=core.chat_mode, deadline_seconds=0.0)
    result = flow.run("fuck, schon wieder verloren", core.world.state())
    assert result.status == "FREE_INTELLIGENCE_UNAVAILABLE" and result.reason.startswith("timeout") and "before the GoalSpec" in result.reason
    assert flow.metrics.provider_calls == 0 and net.requests == []
    assert semantic_local_calls(local) == []
    # With time, the same flow asks -- and each call carries the semantic bound, not the provider default of 120 s.
    flow = IntelligenceFlow(kernel.gateway, cards, projects, mode=core.chat_mode)
    result = flow.run("fuck, schon wieder verloren", core.world.state())
    assert result.status == "PLAN"
    assert net.requests and all(r["timeout"] == SEMANTIC_CALL_TIMEOUT_SECONDS for r in net.requests), [r["timeout"] for r in net.requests]


# ---------------------------------------------------------------------------
# 9: composition, capability selection, missing-capability and replan all refuse the local model
# ---------------------------------------------------------------------------

def test_composition_with_the_provider_hanging_is_typed_unavailable_not_a_local_plan(tmp_path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    core.set_chat_mode("FREE")
    net.responses[GEMINI] = TimeoutError("timed out")
    local.generate_structured = lambda prompt, schema, **kw: (local.calls.append(prompt), json.dumps({"primary_goal": "understand_recent_loss",
                                                                                                        "confidence": 0.95, "reason": "local guess"}))[1]
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    assert preview["status"] == "FREE_INTELLIGENCE_UNAVAILABLE" and preview["goal"] is None and preview["plan"] is None
    assert "timeout" in preview["reason"], preview["reason"]
    events = ask(core, "fuck, schon wieder verloren", wait=30)
    assert executed == [] and FREE_MESSAGE in answer_text(events)
    assert semantic_local_calls(local) == [], "the local model was never asked for a GoalSpec, a PlanSpec or a goal"


def test_replan_after_a_failed_step_refuses_the_local_model_when_the_provider_is_gone(tmp_path):
    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    core.set_chat_mode("FREE")
    cards = capability_cards(core.capabilities.registry) + builtin_cards()
    flow = IntelligenceFlow(kernel.gateway, cards, [ProjectSummary("Schach Training", "")], mode=core.chat_mode)
    result = flow.run("fuck, schon wieder verloren", core.world.state())
    assert result.status == "PLAN" and result.goal is not None
    net.responses[GEMINI] = TimeoutError("timed out")
    local.generate_structured = lambda prompt, schema, **kw: (local.calls.append(prompt), json.dumps({"steps": [], "reason": "local"}))[1]
    fresh = flow.repropose("fuck, schon wieder verloren", result.goal, result.context, sorted(result.report.goal), "stockfish.analyze",
                           "engine crashed", ["chess.capture_game"])
    assert fresh is None
    assert semantic_local_calls(local) == [] and not any("stockfish" in c for c in local.calls)


def test_capability_missing_from_a_goal_is_not_engineered_without_the_reasoning_provider(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, TimeoutError("timed out"))
    kernel.gateway.health.note("gemini", ProviderStatus.TIMEOUT)
    goal = SemanticGoal(operation="capability.missing", target="", confidence=0.9, reason="no tool", question="")
    started: list = []
    core._start_capability_teaching_for_request = lambda *a, **k: started.append(a)  # noqa: SLF001
    with core.bus.subscribe(replay=False) as sub:
        handled = core._dispatch_semantic_goal(goal, "Mach mir daraus eine Tabelle", "", classify("Mach mir daraus eine Tabelle"))  # noqa: SLF001
        events = sub.drain()
    assert handled is True and started == []
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert GENERAL_MESSAGE in message["text"] or FREE_MESSAGE in message["text"], message["text"]
    assert semantic_local_calls(local) == []


def test_the_decision_provider_has_no_way_down_to_the_local_model(tmp_path, cfg, creds):
    net = FakeNetwork()
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net, local=local)
    everyday = GatewayBrainProvider(gateway, fallback=local)
    decisions = everyday.for_decisions()
    assert decisions.fallback is None and decisions.decision is True and everyday.decision is False
    from gateway.gateway import GatewayRefused, RequestContext, reset_context, set_context

    token = set_context(RequestContext(mode=ChatMode.FREE))
    try:
        assert json.loads(everyday.generate_structured("x", {"type": "object"})) == {"ok": True}, "prose-side calls may still fall back"
        with pytest.raises(GatewayRefused):
            decisions.generate_structured("x", {"type": "object"})
        with pytest.raises(GatewayRefused):
            decisions.generate("machine-readable action please")
    finally:
        reset_context(token)
    assert len(local.calls) == 1 and net.requests == []


def test_a_kernel_with_only_the_local_model_makes_no_semantic_decision(tmp_path):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("unused")
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.provider = lambda tier: local  # the gateway exists, but the conversational tier was not routed through it
    with pytest.raises(SemanticAuthorityUnavailable):
        core._decision_provider()  # noqa: SLF001
    events = ask(core, "Lege das Ergebnis von gestern ab.", wait=30)
    assert GENERAL_MESSAGE in answer_text(events) or FREE_MESSAGE in answer_text(events)
    assert semantic_local_calls(local) == [] and executed == [] and net.requests == []
