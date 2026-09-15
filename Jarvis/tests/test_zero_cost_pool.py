"""§48 + §49 -- the zero-cost pool is provider-independent and verified; AUTO answers, with at most one guarded paid call."""

from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, _parse
from gateway.gateway import INTERACTIVE_STREAM_TIMEOUTS, GatewayError, GatewayRefused, GatewayRequest
from gateway.health import FreeIntelligenceUnavailable, ProviderStatus
from gateway.modes import ChatMode
from gateway.secrets import CredentialStore
from gateway.task import TaskFacts
from runtime.cost_policy import CostPolicy
from test_model_gateway import FakeNetwork, FakeResponse, LocalStub, gemini_reply, make_gateway, openai_reply
from test_streaming import SSENetwork, gemini_chunks

GEMINI = "generativelanguage.googleapis.com"
GROQ = "api.groq.com"
OPENROUTER = "openrouter.ai"
CEREBRAS = "api.cerebras.ai"
OPENAI = "api.openai.com"


class _Body(io.BytesIO):
    """An error body that can be read once per attempt (the fake raises one error object for every call)."""

    def read(self, *args):  # noqa: D401
        return self.getvalue()


def quota_error() -> urllib.error.HTTPError:
    body = _Body(json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                       "message": "Quota exceeded for quota metric 'Generate Content API requests per day'"}}).encode())
    return urllib.error.HTTPError("https://" + GEMINI + "/x", 429, "Too Many Requests", hdrs=None, fp=body)


def chat_completion(text: str, *, model: str = "openai/gpt-oss-120b") -> dict:
    return {"model": model, "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


def chat_stream(pieces: list[str], *, model: str = "openrouter/free") -> list[str]:
    blocks = ["data: " + json.dumps({"model": model, "choices": [{"delta": {"content": p}, "finish_reason": None}]}) for p in pieces]
    blocks.append("data: " + json.dumps({"model": model, "choices": [{"delta": {}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 90, "completion_tokens": 12}}))
    blocks.append("data: [DONE]")
    return blocks


def pool_config(*, groq_verified: bool = True, openrouter_verified: bool = True, cerebras_model: str = "llama-3.3-70b") -> GatewayConfig:
    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    for name in ("gemini", "openai", "groq", "openrouter", "cerebras"):
        document["providers"][name]["enabled"] = True
    document["providers"]["groq"]["options"]["zero_cost_routes"][0].update(
        {"verified_zero_cost": groq_verified, "verified_at": "2026-09-15", "verified_by": "owner: account shows zero monetary cost"})
    document["providers"]["openrouter"]["options"]["zero_cost_routes"][0].update(
        {"verified_zero_cost": openrouter_verified, "verified_at": "2026-09-15", "verified_by": "owner: free router at zero cost"})
    document["providers"]["cerebras"]["options"]["zero_cost_routes"][0].update({"model": cerebras_model})  # stays unverified
    return _parse(document, source="test")


@pytest.fixture
def pool_creds(tmp_path) -> CredentialStore:
    store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    store.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    store.set("openai", "sk-openaitestkey00000000000000000")
    store.set("groq", "gsk_testkey0000000000000000000000")
    store.set("openrouter", "sk-or-testkey000000000000000000")
    store.set("cerebras", "csk-testkey0000000000000000000000")
    return store


def knowledge(mode: ChatMode = ChatMode.FREE) -> GatewayRequest:
    return GatewayRequest(prompt="Was bedeutet Glukoneogenese?", facts=TaskFacts(text="Was bedeutet Glukoneogenese?", is_question=True), mode=mode)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_the_registry_lists_routes_with_evidence_and_only_verified_ones_are_eligible(tmp_path, pool_creds):
    gateway = make_gateway(tmp_path, pool_config(groq_verified=False), pool_creds, FakeNetwork(), local=LocalStub())
    routes = gateway.status()["zero_cost_routes"]
    by_key = {f"{r['provider_id']}/{r['model_id']}": r for r in routes}
    assert by_key["gemini/gemini-3.8-flash"]["eligible"] and by_key["gemini/gemini-3.8-flash"]["verified_at"] == "2026-09-14"
    assert [r["model_id"] for r in routes if r["provider_id"] == "gemini"] == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
    assert not by_key["groq/openai/gpt-oss-120b"]["eligible"] and "not verified" in by_key["groq/openai/gpt-oss-120b"]["reason"]
    assert by_key["groq/openai/gpt-oss-120b"]["monetary_class"] == "unverified"
    assert not by_key["cerebras/llama-3.3-70b"]["eligible"], "an unverified route is listed and never used"
    assert by_key["openrouter/openrouter/free"]["eligible"] and by_key["openrouter/openrouter/free"]["vendor"] == "openrouter"
    assert all(r["monetary_class"] != "metered" for r in routes)
    assert "openai" not in {r["provider_id"] for r in routes}, "a metered provider is never a zero-cost route"
    assert gateway.status()["zero_cost_vendors"][0] == "google"


# ---------------------------------------------------------------------------
# resilience
# ---------------------------------------------------------------------------

def test_gemini_healthy_answers_normally_and_the_pool_order_is_the_configured_one(tmp_path, pool_creds):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("Neubildung von Glucose.")
    net.responses[GROQ] = chat_completion("groq answer")
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    reply = gateway.complete(knowledge())
    assert reply.provider == "gemini" and reply.model == "gemini-3.8-flash" and reply.text == "Neubildung von Glucose."
    assert [r["url"].split("/")[2] for r in net.requests] == [GEMINI]


def test_google_vendor_exhausted_the_independent_route_answers_at_zero_cost(tmp_path, pool_creds):
    net = FakeNetwork()
    net.responses[GEMINI] = quota_error()
    net.responses[GROQ] = chat_completion("Glukoneogenese: Glucose aus Nicht-Kohlenhydraten.")
    net.responses[OPENAI] = openai_reply("paid")
    local = LocalStub()
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=local)
    reply = gateway.complete(knowledge())
    assert reply.provider == "groq" and reply.model == "openai/gpt-oss-120b" and reply.actual_eur == 0.0
    assert reply.text.startswith("Glukoneogenese")
    hosts = [r["url"].split("/")[2] for r in net.requests]
    assert hosts == [GEMINI, GEMINI, GEMINI, GROQ], "each Google model once (a daily quota is not retried), then the first independent route"
    assert [(a["provider"], a["failure_class"]) for a in reply.route_attempts] == [("gemini", "quota_exhausted")] * 3 + [("groq", "ok")]
    assert OPENAI not in hosts and gateway.governor.summary().month == 0.0
    assert local.calls == [], "the local model never generated"
    # exact provenance stays internally correct
    assert gateway.recent[-1]["provider"] == "groq" and gateway.recent[-1]["model"] == "openai/gpt-oss-120b"
    assert gateway.health.status("gemini", model="gemini-3.8-flash") is ProviderStatus.QUOTA_EXHAUSTED
    assert gateway.health.cooldown_remaining("gemini", model="gemini-3.8-flash") > 3600, "a daily quota cools down until the day resets"


def test_a_route_known_to_be_exhausted_is_not_hammered(tmp_path, pool_creds):
    net = FakeNetwork()
    net.responses[GEMINI] = quota_error()
    net.responses[GROQ] = chat_completion("groq answer")
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    gateway.complete(knowledge())
    before = len([r for r in net.requests if GEMINI in r["url"]])
    reply = gateway.complete(knowledge())
    after = len([r for r in net.requests if GEMINI in r["url"]])
    assert reply.provider == "groq" and after == before, "the second request skipped every Google route without a network call"
    verdicts = {f"{r['provider_id']}/{r['model_id']}": r for r in gateway.status()["zero_cost_routes"]}
    assert all(not verdicts[f"gemini/{m}"]["eligible"] and verdicts[f"gemini/{m}"]["quota_reset_in_seconds"] > 0
               for m in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"))


def test_an_unverified_provider_cannot_enter_the_pool_even_when_google_is_out(tmp_path, pool_creds):
    net = FakeNetwork()
    net.responses[GEMINI] = quota_error()
    net.responses[CEREBRAS] = chat_completion("cerebras answer", model="llama-3.3-70b")
    gateway = make_gateway(tmp_path, pool_config(groq_verified=False, openrouter_verified=False), pool_creds, net, local=LocalStub())
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(knowledge())
    hosts = {r["url"].split("/")[2] for r in net.requests}
    assert CEREBRAS not in hosts and GROQ not in hosts and OPENROUTER not in hosts, "unverified cost: listed, never used"


def test_a_stream_only_failure_completes_on_the_same_model(tmp_path, pool_creds):
    class StallThenAnswer(FakeNetwork):
        """The SSE endpoint stalls before the first token; the plain completion endpoint answers."""

        def __call__(self, request, timeout=None):
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
            if "alt=sse" in request.full_url:
                sock = SimpleNamespace(timeouts=[], settimeout=lambda s: sock.timeouts.append(s))

                class Stalled:
                    status = 200
                    fp = SimpleNamespace(raw=SimpleNamespace(_sock=sock))

                    def __iter__(self):
                        raise TimeoutError("timed out")

                    def read(self):
                        return b""

                    def close(self):
                        pass

                return Stalled()
            return FakeResponse(gemini_reply("Vollständige Antwort als Ganzes."))

    net = StallThenAnswer()
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    stream = gateway.stream(knowledge())
    text = "".join(stream)
    assert text == "Vollständige Antwort als Ganzes."
    reply = stream.reply
    assert reply.provider == "gemini" and reply.model == "gemini-3.8-flash" and reply.delivery_mode == "complete_response"
    assert reply.completion()["delivery_mode"] == "complete_response"
    assert [(a["model"], a["failure_class"], a["delivery_mode"]) for a in reply.route_attempts] == [
        ("gemini-3.8-flash", "timeout", "provider_stream"), ("gemini-3.8-flash", "ok", "complete_response")]
    completion = [r for r in net.requests if "alt=sse" not in r["url"]]
    assert len(completion) == 1 and completion[0]["timeout"] == INTERACTIVE_STREAM_TIMEOUTS.first_token
    assert "gemini-3.7-flash" not in " ".join(r["url"] for r in net.requests), "a stream failure is not an intelligence failure"


def test_the_openrouter_free_route_streams_when_configured(tmp_path, pool_creds):
    net = SSENetwork()
    net.responses[GEMINI] = quota_error()
    net.responses[GROQ] = urllib.error.URLError("groq down")
    net.streams[OPENROUTER] = chat_stream(["Glukoneo", "genese ist ", "die Neubildung."])
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    stream = gateway.stream(knowledge())
    pieces = list(stream)
    assert "".join(pieces) == "Glukoneogenese ist die Neubildung." and len(pieces) == 3
    reply = stream.reply
    assert reply.provider == "openrouter" and reply.model == "openrouter/free" and reply.delivery_mode == "provider_stream"
    assert reply.usage["output_tokens"] == 12 and reply.actual_eur == 0.0
    assert net.requests[-1]["body"]["stream"] is True and net.requests[-1]["body"]["stream_options"] == {"include_usage": True}
    assert net.requests[-1]["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_zero_cost_mode_never_spends_even_with_every_free_route_down(tmp_path, pool_creds):
    net = FakeNetwork()
    net.responses[GEMINI] = quota_error()
    net.responses[GROQ] = urllib.error.URLError("down")
    net.responses[OPENROUTER] = urllib.error.URLError("down")
    net.responses[OPENAI] = openai_reply("paid")
    local = LocalStub()
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=local)
    with pytest.raises(FreeIntelligenceUnavailable) as info:
        gateway.complete(knowledge(ChatMode.FREE))
    assert info.value.typed_status == "FREE_INTELLIGENCE_UNAVAILABLE"
    assert OPENAI not in {r["url"].split("/")[2] for r in net.requests}
    assert gateway.governor.summary().month == 0.0 and local.calls == []
    assert {a["provider"] for a in info.value.attempts} == {"gemini", "groq", "openrouter"}


# ---------------------------------------------------------------------------
# AUTO availability
# ---------------------------------------------------------------------------

def _all_free_down(net: FakeNetwork) -> None:
    net.responses[GEMINI] = quota_error()
    net.responses[GROQ] = urllib.error.URLError("down")
    net.responses[OPENROUTER] = urllib.error.URLError("down")


def test_auto_makes_exactly_one_guarded_emergency_call_when_every_free_route_is_out(tmp_path, pool_creds):
    net = FakeNetwork()
    _all_free_down(net)
    net.responses[OPENAI] = openai_reply("Notfall-Antwort.", prompt_tokens=800, out_tokens=200)
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    gateway._cost_policy = CostPolicy(allow_paid_api=True, auto_emergency_paid_fallback=True, emergency_max_cost_per_request_eur=0.03, source="test")
    reply = gateway.complete(knowledge(ChatMode.AUTO))
    assert reply.text == "Notfall-Antwort." and reply.provider == "openai" and reply.role == "reasoning.smart"
    assert reply.decision.emergency is True and reply.decision.intelligence_class == "ZERO_COST"
    assert reply.estimated_eur <= 0.03 and 0.0 < reply.actual_eur <= 0.03
    paid = [r for r in net.requests if OPENAI in r["url"]]
    assert len(paid) == 1 and paid[0]["body"]["model"] == gateway.config.roles["reasoning.smart"].model
    assert gateway.governor.summary().open_reservations == 0


def test_the_emergency_is_never_a_ladder(tmp_path, pool_creds):
    net = FakeNetwork()
    _all_free_down(net)
    net.responses[OPENAI] = urllib.error.HTTPError("https://api.openai.com/v1/responses", 503, "down", hdrs=None, fp=io.BytesIO(b"{}"))
    net.responses["api.anthropic.com"] = {"content": [{"type": "text", "text": "never"}]}
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    gateway._cost_policy = CostPolicy(allow_paid_api=True, auto_emergency_paid_fallback=True, emergency_max_cost_per_request_eur=0.03, source="test")
    with pytest.raises(GatewayError) as info:
        gateway.complete(knowledge(ChatMode.AUTO))
    assert info.value.provider == "openai"
    hosts = [r["url"].split("/")[2] for r in net.requests]
    assert hosts.count(OPENAI) == 1 and "api.anthropic.com" not in hosts, "one paid attempt; no second paid route, no engineer"
    assert gateway.governor.summary().month == 0.0 and gateway.governor.summary().open_reservations == 0


def test_the_emergency_is_off_by_default_and_needs_the_owner_switch(tmp_path, pool_creds):
    net = FakeNetwork()
    _all_free_down(net)
    net.responses[OPENAI] = openai_reply("paid")
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    assert CostPolicy.strict().auto_emergency_paid_fallback is False
    gateway._cost_policy = CostPolicy(allow_paid_api=True, source="test")  # paid billing on, emergency off
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(knowledge(ChatMode.AUTO))
    assert OPENAI not in {r["url"].split("/")[2] for r in net.requests}


def test_the_per_request_emergency_ceiling_is_enforced(tmp_path, pool_creds):
    net = FakeNetwork()
    _all_free_down(net)
    net.responses[OPENAI] = openai_reply("paid")
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    gateway._cost_policy = CostPolicy(allow_paid_api=True, auto_emergency_paid_fallback=True, emergency_max_cost_per_request_eur=0.0005, source="test")
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(knowledge(ChatMode.AUTO))
    assert OPENAI not in {r["url"].split("/")[2] for r in net.requests}, "over the ceiling: no paid call at all"
    assert gateway.governor.summary().month == 0.0 and gateway.governor.summary().open_reservations == 0


def test_the_monthly_hard_cap_is_enforced_for_the_emergency_too(tmp_path, pool_creds):
    net = FakeNetwork()
    _all_free_down(net)
    net.responses[OPENAI] = openai_reply("paid")
    document = json.loads(json.dumps(pool_config().to_dict()))
    document["budget"]["monthly_hard_cap"] = 0.01  # the month is (almost) spent: the cap no longer admits the emergency estimate
    gateway = make_gateway(tmp_path, _parse(document, source="test"), pool_creds, net, local=LocalStub())
    gateway._cost_policy = CostPolicy(allow_paid_api=True, auto_emergency_paid_fallback=True, emergency_max_cost_per_request_eur=0.03, source="test")
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(knowledge(ChatMode.AUTO))
    assert OPENAI not in {r["url"].split("/")[2] for r in net.requests}
    assert gateway.governor.summary().month == 0.0 and gateway.governor.summary().open_reservations == 0


def test_the_emergency_also_fires_when_the_pool_is_refused_before_any_call(tmp_path, pool_creds):
    """Every free route already in cool-down: the router refuses at plan time; AUTO still gets its one guarded call."""

    net = FakeNetwork()
    net.responses[OPENAI] = openai_reply("Notfall.")
    gateway = make_gateway(tmp_path, pool_config(), pool_creds, net, local=LocalStub())
    for name in ("gemini", "groq", "openrouter"):
        gateway.health.note(name, ProviderStatus.QUOTA_EXHAUSTED)
    gateway._cost_policy = CostPolicy(allow_paid_api=True, auto_emergency_paid_fallback=True, emergency_max_cost_per_request_eur=0.03, source="test")
    reply = gateway.complete(knowledge(ChatMode.AUTO))
    assert reply.decision.emergency is True and reply.provider == "openai"
    assert [r["url"].split("/")[2] for r in net.requests] == [OPENAI]
    gateway._cost_policy = CostPolicy(allow_paid_api=True, source="test")
    with pytest.raises(GatewayRefused):
        gateway.complete(knowledge(ChatMode.AUTO))
