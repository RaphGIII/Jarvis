"""The model gateway: the invariants the migration brief calls critical.

Everything here runs offline.  The network is a fake ``urlopen`` that records
what would have been sent, so a test can prove that in FREE mode *nothing*
was sent -- not that a call was discouraged.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from gateway.budget import BudgetGovernor, BudgetRefused
from gateway.config import GatewayConfig, Pricing
from gateway.gateway import (GatewayBrainProvider, GatewayRefused, GatewayRequest, ModelGateway, RequestContext,
                             reset_context, set_context)
from gateway.health import FreeIntelligenceUnavailable, GatewayError, ProviderStatus, classify_http
from gateway.learning import Observation
from gateway.modes import ChatMode, CostClass
from gateway.persona import guard_identity, system_prompt_for_role
from gateway.privacy import Chunk, PrivacyRouter, Sensitivity, classify_text
from gateway.router import RouteKind
from gateway.secrets import CredentialStore, redact
from gateway.task import TaskClass, TaskFacts, rule_based
from gateway.transport import ReservationRequired, Transport, ZeroCostViolation


# ---------------------------------------------------------------------------
# A fake network
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeNetwork:
    """Answers by host; remembers every request that reached it."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict[str, object] = {}

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        host = request.full_url.split("/")[2]
        answer = self.responses.get(host)
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            raise urllib.error.URLError("no route to host")
        return FakeResponse(answer)


class GeminiPoolNetwork(FakeNetwork):
    """Answers Gemini requests by concrete model, in sequence."""

    def __init__(self, by_model: dict[str, list[object]]) -> None:
        super().__init__()
        self.by_model = {model: list(replies) for model, replies in by_model.items()}

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        model = request.full_url.split("/models/", 1)[1].split(":generateContent", 1)[0]
        replies = self.by_model.get(model, [])
        answer = replies.pop(0) if replies else None
        if answer is None:
            raise urllib.error.URLError(f"no scripted answer for {model}")
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(answer)


def gemini_reply(text: str, *, prompt_tokens: int = 100, out_tokens: int = 50) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": prompt_tokens, "candidatesTokenCount": out_tokens}}


def openai_reply(text: str, *, prompt_tokens: int = 1000, out_tokens: int = 500, cached: int = 0, reasoning: int = 0) -> dict:
    """A Responses API reply: a reasoning item, then the message, then the usage with cached and reasoning tokens."""

    return {"id": "resp_test", "object": "response", "status": "completed", "model": "gpt-x",
            "output": [{"type": "reasoning", "id": "rs_test", "summary": []},
                       {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": prompt_tokens, "input_tokens_details": {"cached_tokens": cached},
                      "output_tokens": out_tokens, "output_tokens_details": {"reasoning_tokens": reasoning},
                      "total_tokens": prompt_tokens + out_tokens}}


def anthropic_reply(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "model": "claude-x",
            "usage": {"input_tokens": 2000, "output_tokens": 800, "cache_read_input_tokens": 500}}


def http_error(url: str, code: int, body: dict | str) -> urllib.error.HTTPError:
    raw = json.dumps(body) if isinstance(body, dict) else body
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(raw.encode("utf-8")))


class LocalStub:
    provider_name = "ollama"
    model_name = "stub-local"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_metadata = {"prompt_tokens": 10, "generated_tokens": 5}

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return "local answer"

    def generate_structured(self, prompt, schema, **kwargs):
        self.calls.append(prompt)
        return json.dumps({"ok": True})


@pytest.fixture
def cfg() -> GatewayConfig:
    config = GatewayConfig.defaults()
    for name in ("gemini", "openai", "anthropic"):
        config = config.with_provider_enabled(name, True)
    return config


@pytest.fixture
def creds(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    store.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    store.set("openai", "sk-openaitestkey00000000000000000")
    store.set("anthropic", "sk-ant-anthropictestkey00000000000000")
    return store


@pytest.fixture
def net() -> FakeNetwork:
    network = FakeNetwork()
    network.responses["generativelanguage.googleapis.com"] = gemini_reply("Beta-Oxidation baut Fettsäuren ab.")
    network.responses["api.openai.com"] = openai_reply("A deep answer.")
    network.responses["api.anthropic.com"] = anthropic_reply("diff --git a/x b/x")
    return network


def make_gateway(tmp_path: Path, cfg: GatewayConfig, creds: CredentialStore, net: FakeNetwork, *, local: LocalStub | None = None,
                 paid_api: bool = True, retry_sleep=None, retry_jitter=None) -> ModelGateway:
    from runtime.cost_policy import CostPolicy

    # The owner has enabled metered billing (an owner transaction in the
    # product); without it every metered route is ineligible -- see the test
    # for exactly that.
    return ModelGateway(state_root=tmp_path / "state", config=cfg, credentials=creds, opener=net,
                        local_provider=(lambda role: local) if local else None,
                        local_available=(lambda role: local is not None),
                        cost_policy=CostPolicy(allow_paid_api=paid_api, source="test"),
                        retry_sleep=retry_sleep or (lambda _delay: None), retry_jitter=retry_jitter or (lambda _spread: 0.0),
                        import_environment_credentials=False)


def knowledge(text: str = "Was ist Beta-Oxidation?") -> GatewayRequest:
    return GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_question=True))


# ---------------------------------------------------------------------------
# FREE mode is a wall, not a preference
# ---------------------------------------------------------------------------

def test_free_mode_never_reaches_a_metered_provider(tmp_path, cfg, creds, net):
    # Make the free lane unusable so the only cloud candidates are metered.
    cfg = cfg.with_provider_enabled("gemini", False)
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg, creds, net, local=local)
    request = knowledge("Erkläre mir die Frank-Starling-Mechanik ausführlich mit Herleitung.")
    request.mode = ChatMode.FREE
    reply = gateway.complete(request)
    assert reply.provider == "ollama" and reply.decision.offline_fallback
    hosts = {r["url"].split("/")[2] for r in net.requests}
    assert "api.openai.com" not in hosts and "api.anthropic.com" not in hosts
    assert gateway.governor.summary().month == 0.0


def test_free_mode_with_nothing_free_refuses_and_says_which_mode_would_work(tmp_path, cfg, creds, net):
    cfg = cfg.with_provider_enabled("gemini", False)
    gateway = make_gateway(tmp_path, cfg, creds, net)  # no local provider either
    request = knowledge()
    request.mode = ChatMode.FREE
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(request)
    assert info.value.decision.kind is RouteKind.REFUSED
    assert "SMART" in info.value.decision.suggestion
    assert net.requests == []


def test_transport_refuses_metered_ticket_in_free_mode_before_any_network(creds):
    transport = Transport(creds, opener=lambda *a, **k: pytest.fail("network must not be touched"))
    cfg = GatewayConfig.defaults()
    with pytest.raises(ZeroCostViolation):
        transport.issue(provider=cfg.providers["openai"], role="reasoning.deep", mode=ChatMode.FREE, cost_class=CostClass.METERED,
                        reservation=None)
    with pytest.raises(ReservationRequired):
        transport.issue(provider=cfg.providers["openai"], role="reasoning.deep", mode=ChatMode.AUTO, cost_class=CostClass.METERED,
                        reservation=None)
    assert transport.issued == 0 and len(transport.refused) == 2


def test_free_lane_answers_a_public_knowledge_question_at_zero_cost(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    request = knowledge()
    request.mode = ChatMode.FREE
    reply = gateway.complete(request)
    assert reply.provider == "gemini" and reply.actual_eur == 0.0 and reply.decision.cost_class is CostClass.ZERO
    sent = net.requests[-1]
    assert "generativelanguage.googleapis.com" in sent["url"]
    assert sent["headers"].get("X-goog-api-key", "").startswith("AIza")
    assert "Beta-Oxidation" in reply.text


def test_reasoning_free_38_success_never_calls_37(tmp_path, cfg, creds):
    net = GeminiPoolNetwork({"gemini-3.8-flash": [gemini_reply("OK from 3.8")],
                             "gemini-3.7-flash": [gemini_reply("should not be used")]})
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True),
                                            mode=ChatMode.FREE))
    assert reply.model == "gemini-3.8-flash" and reply.actual_eur == 0.0
    assert len(net.requests) == 1 and "gemini-3.7-flash" not in net.requests[0]["url"]
    assert len(reply.route_attempts) == 1
    attempt = reply.route_attempts[0]
    assert {k: attempt[k] for k in ("model", "attempt", "failure_class", "http_status", "retry_delay_seconds")} == {
        "model": "gemini-3.8-flash", "attempt": 1, "failure_class": "ok", "http_status": None, "retry_delay_seconds": 0.0}


def test_reasoning_free_38_503_twice_falls_back_to_37(tmp_path, cfg, creds):
    error = http_error("https://generativelanguage.googleapis.com/v1beta/x", 503,
                       {"error": {"status": "UNAVAILABLE", "message": "high demand"}})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [error, error], "gemini-3.7-flash": [gemini_reply("OK from 3.7")]})
    delays: list[float] = []
    gateway = make_gateway(tmp_path, cfg, creds, net, retry_sleep=delays.append)
    reply = gateway.complete(GatewayRequest(prompt="public knowledge", facts=TaskFacts(text="public knowledge", is_question=True),
                                            mode=ChatMode.FREE))
    assert reply.model == "gemini-3.7-flash" and reply.text == "OK from 3.7" and reply.actual_eur == 0.0
    assert [r["url"].split("/models/", 1)[1].split(":", 1)[0] for r in net.requests] == [
        "gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.7-flash"]
    assert delays == [1.0]
    assert [a["failure_class"] for a in reply.route_attempts] == ["provider_unavailable", "provider_unavailable", "ok"]


def test_reasoning_free_38_429_retries_then_falls_back(tmp_path, cfg, creds):
    rate_limited = http_error("https://generativelanguage.googleapis.com/v1beta/x", 429,
                              {"error": {"status": "RESOURCE_EXHAUSTED", "message": "per minute",
                                         "details": [{"retryDelay": "30s"}]}})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [rate_limited, rate_limited],
                             "gemini-3.7-flash": [gemini_reply("OK from 3.7")]})
    delays: list[float] = []
    gateway = make_gateway(tmp_path, cfg, creds, net, retry_sleep=delays.append)
    reply = gateway.complete(GatewayRequest(prompt="public knowledge", facts=TaskFacts(text="public knowledge", is_question=True),
                                            mode=ChatMode.FREE))
    assert reply.model == "gemini-3.7-flash" and delays == [1.0]
    assert [a["http_status"] for a in reply.route_attempts] == [429, 429, None]
    assert sum(a["retry_delay_seconds"] for a in reply.route_attempts) <= 2.0


def test_both_reasoning_free_models_unavailable_is_typed_zero_cost_failure(tmp_path, cfg, creds):
    unavailable = http_error("https://generativelanguage.googleapis.com/v1beta/x", 503,
                             {"error": {"status": "UNAVAILABLE", "message": "high demand"}})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [unavailable, unavailable],
                             "gemini-3.7-flash": [unavailable, unavailable]})
    gateway = make_gateway(tmp_path, cfg, creds, net)
    with pytest.raises(FreeIntelligenceUnavailable) as info:
        gateway.complete(GatewayRequest(prompt="public knowledge", facts=TaskFacts(text="public knowledge", is_question=True),
                                        mode=ChatMode.FREE))
    assert info.value.typed_status == "FREE_INTELLIGENCE_UNAVAILABLE"
    assert len(info.value.attempts) == 4 and gateway.governor.summary().month == 0.0
    assert all("api.openai.com" not in r["url"] and "api.anthropic.com" not in r["url"] for r in net.requests)


@pytest.mark.parametrize(("status", "expected"), [
    (400, ProviderStatus.TASK_FAILURE),
    (401, ProviderStatus.AUTHENTICATION_ERROR),
    (403, ProviderStatus.AUTHENTICATION_ERROR),
])
def test_reasoning_free_non_transient_http_errors_do_not_model_hop(tmp_path, cfg, creds, status, expected):
    err = http_error("https://generativelanguage.googleapis.com/v1beta/x", status, {"error": {"message": "bad request"}})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [err], "gemini-3.7-flash": [gemini_reply("should not be used")]})
    gateway = make_gateway(tmp_path, cfg, creds, net)
    with pytest.raises(GatewayError) as info:
        gateway.complete(GatewayRequest(prompt="public knowledge", facts=TaskFacts(text="public knowledge", is_question=True),
                                        mode=ChatMode.FREE))
    assert info.value.status is expected
    assert len(net.requests) == 1 and "gemini-3.7-flash" not in net.requests[0]["url"]


def test_reasoning_free_pool_never_invokes_paid_providers(tmp_path, cfg, creds):
    unavailable = http_error("https://generativelanguage.googleapis.com/v1beta/x", 503, {"error": "down"})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [unavailable, unavailable],
                             "gemini-3.7-flash": [unavailable, unavailable]})
    net.responses["api.openai.com"] = openai_reply("paid")
    net.responses["api.anthropic.com"] = anthropic_reply("paid")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(GatewayRequest(prompt="public knowledge", facts=TaskFacts(text="public knowledge", is_question=True),
                                        mode=ChatMode.FREE))
    hosts = {r["url"].split("/")[2] for r in net.requests}
    assert hosts == {"generativelanguage.googleapis.com"}


def test_reasoning_free_pool_never_invokes_legacy_4b_fallback(tmp_path, cfg, creds):
    unavailable = http_error("https://generativelanguage.googleapis.com/v1beta/x", 503, {"error": "down"})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [unavailable, unavailable],
                             "gemini-3.7-flash": [unavailable, unavailable]})
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg, creds, net, local=local)
    provider = GatewayBrainProvider(gateway, fallback=local)
    token = set_context(RequestContext(mode=ChatMode.FREE, facts=TaskFacts(text="Was ist NAT?", is_question=True)))
    try:
        with pytest.raises(FreeIntelligenceUnavailable):
            provider.generate("Was ist NAT?")
    finally:
        reset_context(token)
    assert local.calls == []


def test_reasoning_free_total_retry_delay_is_bounded(tmp_path, cfg, creds):
    unavailable = http_error("https://generativelanguage.googleapis.com/v1beta/x", 503, {"error": "down"})
    net = GeminiPoolNetwork({"gemini-3.8-flash": [unavailable, unavailable],
                             "gemini-3.7-flash": [unavailable, unavailable]})
    delays: list[float] = []
    gateway = make_gateway(tmp_path, cfg, creds, net, retry_sleep=delays.append)
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(GatewayRequest(prompt="public knowledge", facts=TaskFacts(text="public knowledge", is_question=True),
                                        mode=ChatMode.FREE))
    assert len(delays) == 2
    assert sum(delays) <= 4.0 and max(delays) <= 2.0


# ---------------------------------------------------------------------------
# Budget: reserve before, settle after, refuse past the cap
# ---------------------------------------------------------------------------

def test_metered_call_reserves_with_safety_factor_then_settles_to_actual(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    # A long planning request that the free prior does not cover well enough.
    text = "Plane für mich die nächsten sechs Wochen Lernplan für Physiologie, Biochemie und Anatomie, " * 4
    request = GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_question=False), mode=ChatMode.DEEP)
    reply = gateway.complete(request)
    assert reply.provider == "openai" and reply.role == "reasoning.deep"
    assert reply.estimated_eur > 0.0 and reply.actual_eur > 0.0
    history = gateway.governor.history()
    kinds = [row["kind"] for row in history]
    assert kinds == ["reserve", "settle"]
    reserve, settle = history
    assert reserve["reserved_eur"] == pytest.approx(reserve["estimated_eur"] * 1.5, rel=1e-3)
    assert settle["actual_eur"] == pytest.approx(reply.actual_eur)
    assert gateway.governor.summary().open_reservations == 0
    assert gateway.governor.summary().month == pytest.approx(reply.actual_eur)


def test_reservation_past_the_monthly_cap_is_refused_and_nothing_is_sent(tmp_path, cfg, creds, net):
    from dataclasses import replace

    cfg = replace(cfg, budget=replace(cfg.budget, daily_hard_cap=100.0, per_task_hard_cap=100.0, engineering_hard_cap=100.0))
    gateway = make_gateway(tmp_path, cfg, creds, net)
    # Spend the month first.
    res = gateway.governor.reserve(role="engineer.frontier", provider="anthropic", model="m", estimated_eur=26.0, task_id="t0")
    gateway.governor.settle(res, 39.98)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    request = GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP)
    cfg2 = cfg.with_provider_enabled("gemini", False)
    gateway.reconfigure(cfg2)
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(request)
    assert "budget" in info.value.decision.reason
    assert net.requests == []


def test_governor_caps_are_all_enforced(tmp_path):
    from gateway.config import BudgetConfig

    governor = BudgetGovernor(tmp_path / "spend.jsonl", BudgetConfig(monthly_hard_cap=40, daily_hard_cap=6, per_task_hard_cap=5,
                                                                      reasoning_hard_cap=10, engineering_hard_cap=25,
                                                                      provider_hard_caps={"openai": 3.0}, safety_factor=1.5))
    with pytest.raises(BudgetRefused) as info:
        governor.reserve(role="reasoning.deep", provider="openai", model="m", estimated_eur=2.5, task_id="a")  # 3.75 > provider cap
    assert info.value.cap == "provider_hard_cap:openai"
    with pytest.raises(BudgetRefused) as info:
        governor.reserve(role="reasoning.deep", provider="other", model="m", estimated_eur=4.0, task_id="a")  # 6.0 > per-task 5
    assert info.value.cap == "per_task_hard_cap"
    with pytest.raises(BudgetRefused) as info:
        governor.reserve(role="reasoning.deep", provider="other", model="m", estimated_eur=3.0, task_cap_eur=0.5)  # SMART cap
    assert info.value.cap == "per_task_hard_cap"
    ok = governor.reserve(role="engineer.frontier", provider="anthropic", model="m", estimated_eur=3.0, task_id="b")
    governor.settle(ok, 4.5)
    with pytest.raises(BudgetRefused) as info:
        governor.reserve(role="engineer.frontier", provider="anthropic", model="m", estimated_eur=1.2, task_id="c")  # day 4.5+1.8 > 6
    assert info.value.cap == "daily_hard_cap"


def test_ledger_survives_a_restart(tmp_path):
    governor = BudgetGovernor(tmp_path / "spend.jsonl")
    res = governor.reserve(role="reasoning.deep", provider="openai", model="m", estimated_eur=1.0, task_id="t")
    governor.settle(res, 0.8)
    open_res = governor.reserve(role="reasoning.deep", provider="openai", model="m", estimated_eur=1.0, task_id="t2")
    again = BudgetGovernor(tmp_path / "spend.jsonl")
    summary = again.summary()
    assert summary.month == pytest.approx(0.8 + 1.5)  # settled + still-open reservation counts
    assert summary.open_reservations == 1
    again.release(open_res)
    assert BudgetGovernor(tmp_path / "spend.jsonl").summary().month == pytest.approx(0.8)


def test_failed_provider_call_releases_the_reservation(tmp_path, cfg, creds, net):
    net.responses["api.openai.com"] = http_error("https://api.openai.com/v1/chat/completions", 503, {"error": "down"})
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    with pytest.raises(GatewayError) as info:
        gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP))
    assert info.value.status is ProviderStatus.PROVIDER_UNAVAILABLE
    assert gateway.governor.summary().month == 0.0
    assert [row["kind"] for row in gateway.governor.history()] == ["reserve", "release"]


# ---------------------------------------------------------------------------
# No cascade; outages are outages
# ---------------------------------------------------------------------------

def test_a_provider_outage_is_not_evidence_the_task_needs_a_stronger_model(tmp_path, cfg, creds, net):
    net.responses["generativelanguage.googleapis.com"] = http_error(
        "https://generativelanguage.googleapis.com/x", 503, {"error": {"status": "UNAVAILABLE", "message": "high demand"}})
    gateway = make_gateway(tmp_path, cfg, creds, net)
    before = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    with pytest.raises(FreeIntelligenceUnavailable) as info:
        gateway.complete(knowledge())
    assert info.value.typed_status == "FREE_INTELLIGENCE_UNAVAILABLE"
    after = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert (after.alpha, after.beta) == (before.alpha, before.beta), "an outage must not move the reliability estimate"
    assert len(net.requests) == 4, "bounded retries stay inside the free pool"
    assert all("api.openai.com" not in r["url"] for r in net.requests)
    assert gateway.health.status("gemini") is ProviderStatus.PROVIDER_UNAVAILABLE


def test_after_quota_exhaustion_auto_mode_routes_around_the_free_lane_within_budget(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    reply = gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True),
                                            mode=ChatMode.AUTO))
    assert reply.provider == "openai"
    assert all("googleapis" not in r["url"] for r in net.requests)


def test_after_quota_exhaustion_free_mode_does_not_spend(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net, local=LocalStub())
    gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    reply = gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True),
                                            mode=ChatMode.FREE))
    assert reply.provider == "ollama" and net.requests == []


def test_router_skips_a_cheaper_model_it_does_not_trust_without_trying_it(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    # Teach the model that the free role fails at composition.
    for _ in range(12):
        gateway.reliability.observe(Observation(task_class="composition", role="reasoning.free", provider="gemini", model="g",
                                                goal_verified=False, failure_class="task_failure"))
    text = "Nimm die letzte Partie, analysiere sie und aktualisiere mein Trainingsprofil"
    facts = TaskFacts(text=text, subsystems=1, refers_to_context=True)
    request = GatewayRequest(prompt=text, facts=facts, mode=ChatMode.AUTO)
    request.task_vector = rule_based(facts)
    from dataclasses import replace

    request.task_vector = replace(request.task_vector, task_class=TaskClass.COMPOSITION, failure_cost=0.6)
    reply = gateway.complete(request)
    assert reply.provider == "openai"
    assert len(net.requests) == 1 and "api.openai.com" in net.requests[0]["url"]
    assert "skipped cheaper reasoning.free" in reply.decision.reason


def test_classify_http_distinguishes_the_six_states():
    assert classify_http(401, "") is ProviderStatus.AUTHENTICATION_ERROR
    assert classify_http(400, '{"error":{"status":"INVALID_ARGUMENT","message":"API key not valid"}}') is ProviderStatus.AUTHENTICATION_ERROR
    assert classify_http(404, "model not found") is ProviderStatus.MODEL_UNAVAILABLE
    assert classify_http(429, "rate limit exceeded, retry") is ProviderStatus.RATE_LIMIT
    assert classify_http(429, "You exceeded your current quota, please check your plan and billing") is ProviderStatus.QUOTA_EXHAUSTED
    assert classify_http(503, "") is ProviderStatus.PROVIDER_UNAVAILABLE
    assert classify_http(400, "bad request") is ProviderStatus.TASK_FAILURE


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def test_private_context_never_reaches_a_may_train_provider(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    request = knowledge("Fasse mir das kurz zusammen.")
    request.chunks = [Chunk(text="Befund vom 03.05.: Kreatinin 1.9 mg/dl, Diagnose: N18.3", source="document"),
                      Chunk(text="Allgemeinwissen: Kreatinin ist ein Abbauprodukt.", source="public_knowledge")]
    reply = gateway.complete(request)
    assert reply.provider == "gemini"
    sent = json.dumps(net.requests[-1]["body"])
    assert "Kreatinin 1.9" not in sent and "N18.3" not in sent
    assert "Abbauprodukt" in sent
    assert reply.privacy is not None and len(reply.privacy.withheld) == 1


def test_a_sensitive_owner_message_closes_the_free_lane_entirely(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    text = "Hier mein Kontoauszug: IBAN DE89 3704 0044 0532 0130 00 – wie viel habe ich diesen Monat ausgegeben?"
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_question=True), mode=ChatMode.AUTO))
    assert reply.provider == "openai", "a no-training provider may see private data; the free lane may not"
    assert all("googleapis" not in r["url"] for r in net.requests)


def test_secrets_never_leave_even_to_a_no_training_provider(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    text = "Analysiere diese Konfiguration"
    request = GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP,
                             chunks=[Chunk(text="OPENAI_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz", source="document")])
    reply = gateway.complete(request)
    assert "sk-live" not in json.dumps(net.requests[-1]["body"])
    assert reply.privacy.ceiling is Sensitivity.SECRET


def test_classify_text_tells_learning_questions_from_records():
    assert classify_text("Was ist die Frank-Starling-Mechanik?")[0] is Sensitivity.PUBLIC
    assert classify_text("Erkläre kompetitive Hemmung.")[0] is Sensitivity.PUBLIC
    assert classify_text("Laborbericht: Hb 11.2 g/dl, Patient: R. G.")[0] is Sensitivity.PRIVATE
    assert classify_text("passwort: hunter2!!")[0] is Sensitivity.SECRET
    router = PrivacyRouter()
    decision = router.decide([Chunk("Was ist NAT?", "owner_message"), Chunk("From: a@b.de\nTo: c@d.de\nhallo", "email")],
                             provider_may_train=True)
    assert decision.free_lane_allowed and len(decision.allowed) == 1 and len(decision.withheld) == 1


# ---------------------------------------------------------------------------
# Hard overrides
# ---------------------------------------------------------------------------

def test_hard_overrides_outrank_scoring(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    cases = [
        (TaskFacts(text="lösche alle logs", destructive=True), RouteKind.AUTHORIZE),
        (TaskFacts(text="prüfsumme", capability_found=True, capability_confidence=0.95), RouteKind.DETERMINISTIC),
        (TaskFacts(text="prüfsumme", capability_found=True, capability_confidence=0.95, capability_broken=True), RouteKind.ENGINEERING),
        (TaskFacts(text="mach was mit schach", capability_missing=True), RouteKind.ENGINEERING),
        (TaskFacts(text="analysiere und trainiere", subsystems=3), RouteKind.COMPOSE),
        (TaskFacts(text="wer hat heute das spiel gewonnen", needs_fresh_information=True), RouteKind.RESEARCH),
    ]
    for facts, expected in cases:
        decision, _ = gateway.plan(GatewayRequest(prompt=facts.text, facts=facts))
        assert decision.kind is expected, (facts.text, decision.kind, decision.reason)
        assert net.requests == []


def test_too_much_ambiguity_is_a_clarification_not_an_escalation(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    facts = TaskFacts(text="mach das mal")
    vector = rule_based(facts).merged_with_semantic({"ambiguity": 0.9})
    decision, _ = gateway.plan(GatewayRequest(prompt="mach das mal", task_vector=vector))
    assert decision.kind is RouteKind.CLARIFY


def test_semantic_estimate_can_only_raise_soft_features():
    facts = TaskFacts(text="x", destructive=True)
    base = rule_based(facts)
    merged = base.merged_with_semantic({"ambiguity": 0.7, "action_risk": 0.0, "reasoning_depth": 0.2})
    assert merged.ambiguity == 0.7 and merged.action_risk == 1.0 and merged.reasoning_depth == 0.2
    assert merged.provenance["ambiguity"] == "semantic"
    lowered = merged.merged_with_semantic({"ambiguity": 0.1})
    assert lowered.ambiguity == 0.7, "D_final = max(rule, semantic): nothing averages down"


# ---------------------------------------------------------------------------
# Engineering roles are selected before execution, no standard-then-frontier
# ---------------------------------------------------------------------------

def test_large_new_subsystem_goes_directly_to_the_frontier_engineer(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    text = ("Beobachte meinen Bildschirm nur beim Schach, zeichne Partien auf, analysiere sie mit Stockfish, "
            "kategorisiere meine Fehler, führe ein Langzeitprofil und erstelle daraus Training und ein Projekt.")
    facts = TaskFacts(text=text, is_engineering=True, new_subsystem=True, subsystems=5, needs_screen=True, external_systems=1)
    reply = gateway.complete(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.BUILD, purpose="engineer"))
    assert reply.role == "engineer.frontier" and reply.provider == "anthropic"
    assert len(net.requests) == 1
    assert reply.decision.task.task_class is TaskClass.ENGINEERING_LARGE


def test_small_engineering_change_uses_the_standard_engineer(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    text = "Korrigiere den Tippfehler in der Fehlermeldung von fs.largest"
    facts = TaskFacts(text=text, is_engineering=True, estimated_files_changed=1)
    reply = gateway.complete(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.BUILD, purpose="engineer"))
    assert reply.role == "engineer.standard"


def test_engineering_is_not_available_outside_build_mode(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    text = "Korrigiere den Tippfehler"
    facts = TaskFacts(text=text, is_engineering=True, estimated_files_changed=1)
    with pytest.raises(GatewayRefused):
        gateway.complete(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.SMART, purpose="engineer"))
    assert net.requests == []


# ---------------------------------------------------------------------------
# Persona and identity
# ---------------------------------------------------------------------------

def test_the_answer_stays_zeus_whoever_computed_it(tmp_path, cfg, creds, net):
    net.responses["generativelanguage.googleapis.com"] = gemini_reply(
        "Ich bin Gemini, ein großes Sprachmodell, trainiert von Google. NAT übersetzt private in öffentliche Adressen.")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(knowledge("Was ist NAT?"))
    assert "Gemini" not in reply.text and "Google" not in reply.text
    assert reply.text.startswith("Ich bin ZEUS")
    assert "NAT übersetzt" in reply.text
    assert reply.identity_rewrites == 1
    # Provider identity is still available where it belongs: diagnostics.
    assert reply.provider == "gemini" and reply.to_dict()["provider"] == "gemini"


def test_system_prompt_carries_the_identity_clause_and_no_provider_names():
    prompt = system_prompt_for_role("reasoning.free", assistant="ZEUS")
    assert "You are ZEUS" in prompt
    assert "designed and orchestrat" in prompt
    assert "trained a foundation model" in prompt


def test_identity_guard_leaves_ordinary_text_alone():
    text = "Google Maps zeigt den Weg; Anthropic ist ein Unternehmen. Ich bin mir sicher."
    assert guard_identity(text) == (text, 0)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def test_credentials_are_encrypted_at_rest_and_redacted_everywhere(tmp_path):
    store = CredentialStore(tmp_path / "c.json", use_dpapi=False)
    store.set("openai", "sk-verysecretkey1234567890abcdef")
    assert store.status()["openai"]["configured"] and store.status()["openai"]["hint"] == "…cdef"
    assert "sk-verysecretkey" not in json.dumps(store.status())
    assert redact("Authorization: Bearer sk-verysecretkey1234567890abcdef", store) == "Authorization: Bearer [secret]"
    with pytest.raises(KeyError):
        store.set("unknown", "x")
    with pytest.raises(ValueError):
        store.set("openai", "   ")
    assert store.clear("openai") and not store.has("openai")


def test_provider_error_bodies_are_redacted(tmp_path, cfg, creds, net):
    key = creds.get("gemini")
    net.responses["generativelanguage.googleapis.com"] = http_error(
        f"https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key={key}", 400,
        {"error": {"message": f"API key not valid: {key}", "status": "INVALID_ARGUMENT"}})
    gateway = make_gateway(tmp_path, cfg, creds, net)
    with pytest.raises(GatewayError) as info:
        gateway.complete(knowledge())
    assert key not in str(info.value)
    assert info.value.status is ProviderStatus.AUTHENTICATION_ERROR


def test_missing_credential_is_not_a_route(tmp_path, cfg, net):
    empty = CredentialStore(tmp_path / "none.json", use_dpapi=False)
    gateway = make_gateway(tmp_path, cfg, empty, net)
    decision, _ = gateway.plan(knowledge())
    assert decision.kind is RouteKind.REFUSED and "API key" in decision.suggestion


# ---------------------------------------------------------------------------
# Configuration, not code
# ---------------------------------------------------------------------------

def test_no_model_name_lives_outside_the_configuration():
    root = Path(__file__).resolve().parent.parent / "gateway"
    for path in root.glob("*.py"):
        if path.name == "config.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("gemini-2.5", "gpt-5", "claude-opus", "claude-fable", "claude-sonnet"):
            assert needle not in text, f"{path.name} hardcodes {needle}"


def test_rebinding_a_role_changes_the_provider_without_code(tmp_path, cfg, creds, net):
    net.responses["api.anthropic.com"] = anthropic_reply("Deep from anthropic")
    rebound = cfg.with_role_binding("reasoning.deep", provider="anthropic", model="claude-opus-5")
    gateway = make_gateway(tmp_path, rebound.with_provider_enabled("gemini", False), creds, net)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP))
    assert reply.provider == "anthropic" and reply.text == "Deep from anthropic"


def test_config_round_trips_through_the_file(tmp_path):
    path = tmp_path / "providers.json"
    cfg = GatewayConfig.defaults().with_provider_enabled("gemini", True)
    cfg.save(path)
    loaded = GatewayConfig.load(path)
    assert loaded.providers["gemini"].enabled and loaded.roles["reasoning.free"].provider == "gemini"
    assert loaded.budget.monthly_hard_cap == 40.0
    assert loaded.to_dict()["roles"] == cfg.to_dict()["roles"]


def test_a_metered_role_without_a_price_cannot_be_called(tmp_path, cfg, creds, net):
    from dataclasses import replace

    providers = dict(cfg.providers)
    providers["openai"] = replace(providers["openai"], pricing={})
    cfg2 = replace(cfg, providers=providers).with_provider_enabled("gemini", False)
    gateway = make_gateway(tmp_path, cfg2, creds, net)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP))
    assert "price" in info.value.decision.reason
    assert net.requests == []


# ---------------------------------------------------------------------------
# Learning from verified goals only
# ---------------------------------------------------------------------------

def test_reliability_moves_on_verified_goals_and_ignores_self_reports(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    before = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    reply = gateway.complete(knowledge())
    unchanged = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert (unchanged.alpha, unchanged.beta) == (before.alpha, before.beta), "a returned answer proves nothing yet"
    gateway.report_outcome(reply, goal_verified=True)
    after = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert after.alpha == before.alpha + 1 and after.observations == 1
    rows = gateway.ledger.observations()
    assert rows and rows[-1].goal_verified and "prompt" not in json.dumps(rows[-1].to_dict())


def test_performance_ledger_never_contains_prompt_text(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(knowledge("Erkläre die kompetitive Hemmung"))
    gateway.report_outcome(reply, goal_verified=True)
    raw = (tmp_path / "state" / "gateway" / "performance.jsonl").read_text(encoding="utf-8")
    assert "kompetitive" not in raw and "Hemmung" not in raw


# ---------------------------------------------------------------------------
# The BrainProvider adapter the rest of ZEUS talks to
# ---------------------------------------------------------------------------

def test_brain_provider_adapter_routes_generate_through_the_gateway_in_the_current_mode(tmp_path, cfg, creds, net):
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg, creds, net, local=local)
    provider = GatewayBrainProvider(gateway, fallback=local)
    token = set_context(RequestContext(mode=ChatMode.FREE, facts=TaskFacts(text="Was ist NAT?", is_question=True)))
    try:
        text = provider.generate("Was ist NAT?", max_tokens=200, temperature=0.0)
    finally:
        reset_context(token)
    assert "Beta-Oxidation" in text  # the fake gemini answer
    assert provider.last_metadata["provider"] == "gemini" and provider.last_metadata["actual_eur"] == 0.0
    assert local.calls == []
    assert provider.model_name.startswith("reasoning.free:")


def test_brain_provider_adapter_falls_back_locally_only_when_the_gateway_refuses(tmp_path, cfg, creds, net):
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net, local=local)
    provider = GatewayBrainProvider(gateway, fallback=local)
    token = set_context(RequestContext(mode=ChatMode.FREE))
    try:
        out = provider.generate_structured("Plane etwas Langes " * 30, {"type": "object"}, max_tokens=100)
    finally:
        reset_context(token)
    assert json.loads(out) == {"ok": True}
    assert provider.last_metadata["offline_fallback"] is True
    assert net.requests == []


def test_brain_provider_stream_yields_the_gateway_answer(tmp_path, cfg, creds, net):
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("Erster Satz. Zweiter Satz! Dritter?")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    provider = GatewayBrainProvider(gateway)
    pieces = list(provider.generate_stream("Was ist NAT?", system="sys"))
    assert "".join(pieces).strip() == "Erster Satz. Zweiter Satz! Dritter?"
    assert len(pieces) == 3
    assert net.requests[-1]["body"]["system_instruction"]["parts"][0]["text"] == "sys"


# ---------------------------------------------------------------------------
# Adapters read usage correctly (what the budget settles on)
# ---------------------------------------------------------------------------

def test_openai_usage_and_cached_tokens_drive_the_actual_cost(tmp_path, cfg, creds, net):
    net.responses["api.openai.com"] = openai_reply("x", prompt_tokens=1_000_000, out_tokens=0, cached=500_000)
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP))
    price = cfg.providers["openai"].pricing[cfg.roles["reasoning.deep"].model]
    expected = 0.5 * price.input_per_m + 0.5 * price.cached_input_per_m
    assert reply.actual_eur == pytest.approx(expected, rel=1e-6)
    assert price.currency == "USD" and price.rate_to_eur is None and price.rate_source == "unavailable"
    settle = gateway.governor.history()[-1]
    assert settle["actual_native"] == pytest.approx(2.2, rel=1e-6) and settle["currency"] == "USD"
    assert settle["rate_source"] == "unavailable"
    body = net.requests[-1]["body"]
    assert net.requests[-1]["url"].endswith("/v1/responses"), "the Responses API"
    assert body["reasoning"]["effort"] in {"low", "medium", "high", "xhigh"} and "temperature" not in body
    assert body["store"] is False and body["input"][0]["content"][0]["text"]


def test_gemini_schema_is_sanitised_for_structured_output(tmp_path, cfg, creds, net):
    net.responses["generativelanguage.googleapis.com"] = gemini_reply('{"operation": "x"}')
    gateway = make_gateway(tmp_path, cfg, creds, net)
    schema = {"type": "object", "additionalProperties": False, "$schema": "x",
              "properties": {"operation": {"type": "string", "enum": ["x", "y"]}, "n": {"type": ["integer", "null"]}},
              "required": ["operation"]}
    request = knowledge("Klassifiziere")
    request.schema = schema
    gateway.complete(request)
    sent = net.requests[-1]["body"]["generationConfig"]
    assert sent["responseMimeType"] == "application/json"
    assert "additionalProperties" not in json.dumps(sent["responseSchema"])
    assert sent["responseSchema"]["properties"]["n"] == {"type": "integer", "nullable": True}


def test_status_report_carries_spend_roles_and_no_secret(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    status = gateway.status()
    assert status["spend"]["monthly_hard_cap"] == 40.0
    assert status["roles"]["reasoning.free"]["configured"] is True
    assert status["cloud_reasoning_available"] is True
    for secret in creds.values():
        assert secret not in json.dumps(status)


# ---------------------------------------------------------------------------
# Outcomes arrive later, by task id
# ---------------------------------------------------------------------------

def test_replies_are_judged_by_task_id_when_the_world_has_been_checked(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    request = knowledge("Was ist NAT?")
    request.task_id = "req-1"
    gateway.complete(request)
    gateway.complete(request)  # a second call in the same request (e.g. interpretation + answer)
    before = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert gateway.status()["pending_outcomes"] == 1
    judged = gateway.report_task_outcome("req-1", goal_verified=True)
    assert judged == 2
    after = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert after.observations == 2 and after.alpha == pytest.approx(before.alpha + 2, abs=0.05)  # decay on the older one
    assert gateway.report_task_outcome("req-1", goal_verified=True) == 0, "a task is judged once"
    assert gateway.report_task_outcome("never-seen", goal_verified=False) == 0


def test_unjudged_tasks_are_forgotten_not_counted(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    for i in range(205):
        request = knowledge("Was ist NAT?")
        request.task_id = f"req-{i}"
        gateway.complete(request)
    assert gateway.status()["pending_outcomes"] == 200
    assert gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE).observations == 0


# ---------------------------------------------------------------------------
# The owner's spending document outranks everything below it
# ---------------------------------------------------------------------------

def test_without_the_owner_enabling_paid_api_no_metered_route_exists(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net, paid_api=False)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP))
    assert "owner spending policy" in info.value.decision.reason
    assert "Owner Settings" in info.value.decision.suggestion
    assert net.requests == []
    assert gateway.status()["paid_api_allowed"] is False


def test_the_cost_policy_reads_the_owner_spending_document(tmp_path):
    from runtime.cost_policy import CostPolicy

    config_dir = tmp_path / "config"
    (config_dir / "owner").mkdir(parents=True)
    assert CostPolicy.load(config_dir=config_dir, environ={}).allow_paid_api is False
    (config_dir / "owner" / "spending.json").write_text(json.dumps({"paid_api": True, "cloud_gpu": False}), encoding="utf-8")
    policy = CostPolicy.load(config_dir=config_dir, environ={})
    assert policy.allow_paid_api is True and policy.allow_runpod is False and "owner" in policy.source
    # The owner's document outranks the machine file.
    (config_dir / "cost_policy.json").write_text(json.dumps({"allow_paid_api": False}), encoding="utf-8")
    assert CostPolicy.load(config_dir=config_dir, environ={}).allow_paid_api is True


def test_codex_is_ranked_as_a_zero_cost_engineer_but_never_called_as_a_model(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    gateway.set_subscription_available(lambda name: name == "codex")
    text = "Korrigiere den Tippfehler in der Fehlermeldung"
    facts = TaskFacts(text=text, is_engineering=True, estimated_files_changed=1)
    decision, _ = gateway.plan(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.BUILD, purpose="engineer"))
    assert decision.role == "engineer.codex" and decision.cost_class is CostClass.ZERO
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.BUILD, purpose="engineer"))
    assert "expert gateway" in info.value.decision.reason
    assert net.requests == []
    # Pinning the frontier role skips the ranking but keeps every gate.
    reply = gateway.complete(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.BUILD, purpose="engineer", role="engineer.frontier"))
    assert reply.role == "engineer.frontier"
    with pytest.raises(GatewayRefused):
        gateway.complete(GatewayRequest(prompt=text, facts=facts, mode=ChatMode.FREE, purpose="engineer", role="engineer.frontier"))


# ---------------------------------------------------------------------------
# Soft features raise the vector; advisory overrides only annotate
# ---------------------------------------------------------------------------

def test_semantic_soft_features_raise_thinking_and_never_lower_it(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    plain, _ = gateway.plan(knowledge("Was ist NAT?"))
    assert plain.thinking_level == "FAST"
    request = knowledge("Was ist NAT?")
    request.soft = {"reasoning_depth": 0.9}
    deep, _ = gateway.plan(request)
    assert deep.thinking_level == "DEEP"
    assert deep.task.reasoning_depth == 0.9 and deep.task.provenance["reasoning_depth"] == "semantic"
    # A caller that does pass ambiguity gets the clarify override -- unless it
    # has already decided to consult a model (the BrainProvider path).
    request.soft = {"ambiguity": 0.95}
    clarify, _ = gateway.plan(request)
    assert clarify.kind is RouteKind.CLARIFY
    request.overrides = False
    advisory, _ = gateway.plan(request)
    assert advisory.kind is RouteKind.MODEL and advisory.hard_override == "clarify"


def test_brain_provider_requests_are_advisory_about_overrides(tmp_path, cfg, creds, net):
    gateway = make_gateway(tmp_path, cfg, creds, net)
    facts = TaskFacts(text="lösche alle logs", destructive=True)
    strict, _ = gateway.plan(GatewayRequest(prompt="lösche alle logs", facts=facts))
    assert strict.kind is RouteKind.AUTHORIZE
    provider = GatewayBrainProvider(gateway)
    token = set_context(RequestContext(mode=ChatMode.AUTO, facts=facts))
    try:
        text = provider.generate("Fasse zusammen, was der Owner will.")
    finally:
        reset_context(token)
    assert text  # the model was consulted; executing is the caller's decision
    assert provider.last_decision["hard_override"] == "authorize" and provider.last_decision["kind"] == "model"


def test_the_semantic_planner_reads_soft_features_from_the_model():
    from service.semantic import SemanticPlanner

    class Provider:
        def generate_structured(self, prompt, schema, **_):
            assert "features" in schema["properties"]
            return json.dumps({"operation": "conversation", "target": "", "confidence": 0.9, "reason": "x",
                               "features": {"reasoning_depth": 0.8, "context_dependency": "0.3", "bogus": 1, "novelty": 7}})

    goal = SemanticPlanner().plan("Erkläre die Frank-Starling-Mechanik", Provider())
    assert goal is not None and goal.features == {"reasoning_depth": 0.8, "context_dependency": 0.3, "novelty": 1.0}


def test_the_semantic_planner_offers_installed_capabilities_by_id_only():
    from service.semantic import OPERATIONS, SemanticPlanner

    assert "capability.run" in OPERATIONS
    seen = {}

    class Provider:
        def generate_structured(self, prompt, schema, **_):
            seen["prompt"] = prompt
            return json.dumps({"operation": "capability.run", "target": "local.berechne.sha.256_pruefsumme", "confidence": 0.88,
                               "reason": "Fingerprint = Prüfsumme"})

    goal = SemanticPlanner().plan("Gib mir den Fingerprint der Datei x.bin", Provider(),
                                  capabilities=[{"id": "local.berechne.sha.256_pruefsumme", "description": "SHA-256 einer Datei",
                                                 "examples": ["Berechne die SHA-256-Prüfsumme der Datei X."]}])
    assert goal is not None and goal.operation == "capability.run" and goal.target == "local.berechne.sha.256_pruefsumme"
    assert "local.berechne.sha.256_pruefsumme: SHA-256 einer Datei" in seen["prompt"]
    assert "capability.run" in seen["prompt"] and "EXAKT aus der Liste" in seen["prompt"]
