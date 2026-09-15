"""§47 -- the intelligence class is decided before any provider, and no request climbs a ladder of models."""

from __future__ import annotations

import pytest

from gateway.gateway import GatewayRequest
from gateway.intelligence_class import IntelligenceClass, classify
from gateway.modes import ChatMode
from gateway.task import TaskFacts, rule_based
from service.events import EventType
from test_gateway_integration import answer_text, ask
from test_intelligence_flow import make_world
from test_model_gateway import FakeNetwork, cfg, creds, gemini_reply, make_gateway, openai_reply  # noqa: F401 - fixtures

GEMINI = "generativelanguage.googleapis.com"
OPENAI = "api.openai.com"

EXAMPLES = [
    ("Was bedeutet Glukoneogenese?", IntelligenceClass.ZERO_COST),
    ("Erkläre mir die Regulation der Glukoneogenese.", IntelligenceClass.ZERO_COST),
    ("Erkläre mir bitte ausführlich die Frank-Starling-Mechanik.", IntelligenceClass.ZERO_COST),
    ("Vergleiche diese drei Studien, finde methodische Schwächen und leite die wahrscheinlichste kausale Interpretation ab.",
     {IntelligenceClass.SMART, IntelligenceClass.DEEP}),
    ("Entwirf eine robuste Architektur unter 15 technischen Constraints.", IntelligenceClass.DEEP),
]


def _classify(text: str, mode: ChatMode = ChatMode.AUTO, **facts):
    task = rule_based(TaskFacts(text=text, is_question=text.rstrip().endswith("?"), **facts))
    return classify(task, mode, text=text)


@pytest.mark.parametrize("text,expected", EXAMPLES)
def test_the_examples_classify_as_specified(text, expected):
    decision = _classify(text)
    allowed = expected if isinstance(expected, set) else {expected}
    assert decision.intelligence_class in allowed, (decision.intelligence_class, decision.reason, decision.signals)
    assert decision.source == "classifier" and 0.0 <= decision.score <= 1.0


def test_engineering_goes_to_the_engineering_router_directly():
    decision = _classify("Implementiere dieses neue Subsystem in ZEUS.", is_engineering=True, new_subsystem=True, estimated_files_changed=9)
    assert decision.intelligence_class is IntelligenceClass.BUILD_FRONTIER and decision.source == "engineering"
    small = _classify("Repariere den Tippfehler in der Fehlermeldung.", is_engineering=True, estimated_files_changed=1)
    assert small.intelligence_class is IntelligenceClass.BUILD_STANDARD
    assert decision.intelligence_class.is_engineering and decision.intelligence_class.is_paid


def test_owner_override_wins_over_the_classifier():
    text = "Entwirf eine robuste Architektur unter 15 technischen Constraints."
    assert _classify(text, ChatMode.FREE).intelligence_class is IntelligenceClass.ZERO_COST
    assert _classify(text, ChatMode.FREE).owner_override is True
    assert _classify("Was bedeutet Glukoneogenese?", ChatMode.DEEP).intelligence_class is IntelligenceClass.DEEP
    assert _classify("Was bedeutet Glukoneogenese?", ChatMode.SMART).intelligence_class is IntelligenceClass.SMART
    assert _classify("Was bedeutet Glukoneogenese?", ChatMode.BUILD).intelligence_class.is_engineering


def test_a_deterministic_trivial_task_never_reaches_a_model(tmp_path):
    net = FakeNetwork()  # no responses at all: any provider call would fail loudly
    core, kernel, local, executed = make_world(tmp_path, net)
    events = ask(core, "Wie spät ist es?", wait=30)
    text = answer_text(events)
    assert net.requests == [] and local.calls == [], "no model, no provider"
    assert any(ch.isdigit() for ch in text) and ":" in text, text


def test_the_class_exists_before_any_provider_request_and_is_recorded(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("Glukoneogenese ist die Neubildung von Glucose.")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    request = GatewayRequest(prompt="Was bedeutet Glukoneogenese?", facts=TaskFacts(text="Was bedeutet Glukoneogenese?", is_question=True), mode=ChatMode.AUTO)
    decision, _ = gateway.plan(request)
    assert net.requests == [], "planning makes no request"
    assert decision.intelligence_class == "ZERO_COST" and decision.class_decision["source"] == "classifier"
    reply = gateway.complete(request)
    assert reply.decision.intelligence_class == "ZERO_COST" and reply.completion()["intelligence_class"] == "ZERO_COST"
    assert len(net.requests) == 1 and GEMINI in net.requests[0]["url"]
    assert gateway.recent[-1]["provider"] == "gemini"


def test_smart_never_tries_the_zero_cost_pool_first(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("free answer")
    net.responses[OPENAI] = openai_reply("smart answer")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(GatewayRequest(prompt="Was bedeutet Glukoneogenese?", facts=TaskFacts(text="x", is_question=True), mode=ChatMode.SMART))
    assert reply.text == "smart answer" and reply.role == "reasoning.smart" and reply.decision.intelligence_class == "SMART"
    assert [r["url"].split("/")[2] for r in net.requests] == [OPENAI], "one request, to the smart class only"


def test_deep_never_tries_zero_cost_or_smart_first(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("free answer")
    net.responses[OPENAI] = openai_reply("deep answer")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(GatewayRequest(prompt="Was bedeutet Glukoneogenese?", facts=TaskFacts(text="x", is_question=True), mode=ChatMode.DEEP))
    assert reply.role == "reasoning.deep" and reply.decision.intelligence_class == "DEEP"
    assert [r["url"].split("/")[2] for r in net.requests] == [OPENAI]
    assert net.requests[0]["body"]["model"] == gateway.config.roles["reasoning.deep"].model


def test_auto_with_a_smart_class_goes_to_smart_directly_no_cheap_trial(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("free answer")
    net.responses[OPENAI] = openai_reply("smart answer")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    text = "Vergleiche diese drei Studien, finde methodische Schwächen und leite die wahrscheinlichste kausale Interpretation ab."
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.AUTO, owner_text=text))
    assert reply.decision.intelligence_class in {"SMART", "DEEP"} and reply.provider == "openai"
    assert [r["url"].split("/")[2] for r in net.requests] == [OPENAI], "no zero-cost attempt was burnt first"


def test_auto_with_a_zero_cost_class_uses_the_zero_cost_pool(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("free answer")
    net.responses[OPENAI] = openai_reply("smart answer")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(GatewayRequest(prompt="Was bedeutet Glukoneogenese?", facts=TaskFacts(text="Was bedeutet Glukoneogenese?", is_question=True),
                                            mode=ChatMode.AUTO))
    assert reply.provider == "gemini" and reply.decision.intelligence_class == "ZERO_COST" and reply.actual_eur == 0.0
    assert [r["url"].split("/")[2] for r in net.requests] == [GEMINI]


def test_the_class_reaches_the_product_transcript(tmp_path):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("Glukoneogenese ist die Neubildung von Glucose.")
    core, kernel, local, executed = make_world(tmp_path, net)
    core.set_chat_mode("AUTO")
    events = ask(core, "Was bedeutet Glukoneogenese?", wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["meta"]["provenance"]["intelligence_class"] == "ZERO_COST"
    assert message["meta"]["completion"]["delivery_mode"] in {"provider_stream", "complete_response"}
