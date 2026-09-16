"""Zero-cost resilience: an explicit route priority, each verified route able to answer alone, daily quotas that stay
spent until the provider's reset, transient limits that honour Retry-After, the served model recorded, no paid cascade."""

from __future__ import annotations

import io
import json
import time
import urllib.error

import pytest

from gateway.config import GatewayConfig, _parse
from gateway.gateway import GatewayError, GatewayRequest
from gateway.health import FreeIntelligenceUnavailable, ProviderHealth, ProviderStatus, seconds_until_daily_reset
from gateway.modes import ChatMode
from gateway.task import TaskFacts
from gateway.transport import retry_after_from
from gateway.zero_cost import ZeroCostRegistry
from test_model_gateway import FakeNetwork, FakeResponse, LocalStub, gemini_reply, make_gateway, openai_reply
from test_streaming import SSEResponse, gemini_chunks
from test_zero_cost_pool import chat_completion, chat_stream, pool_creds, quota_error  # noqa: F401 - fixture

GEMINI = "generativelanguage.googleapis.com"
GROQ = "api.groq.com"
OPENROUTER = "openrouter.ai"
CEREBRAS = "api.cerebras.ai"
OPENAI = "api.openai.com"

#: The owner's pool, in the order the owner decided.
POOL = [("gemini", "gemini-3.8-flash"), ("gemini", "gemini-3.7-flash"), ("gemini", "gemini-3.6-flash"),
        ("groq", "openai/gpt-oss-120b"), ("openrouter", "openrouter/free")]


class _Body(io.BytesIO):
    def read(self, *args):  # noqa: D401 - readable once per attempt
        return self.getvalue()


def http_error(host: str, code: int, body: dict, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"https://{host}/x", code, "error", hdrs=headers or {}, fp=_Body(json.dumps(body).encode()))


def groq_daily_limit(*, try_again: str = "7m12.3s") -> urllib.error.HTTPError:
    tail = f" Please try again in {try_again}." if try_again else ""
    return http_error(GROQ, 429, {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
                                                       f"`on_demand` on tokens per day (TPD): Limit 200000, Used 199990, Requested 900.{tail}",
                                            "type": "tokens", "code": "rate_limit_exceeded"}})


class RouteNetwork(FakeNetwork):
    """Answers by route -- Gemini by the model in its URL, OpenAI-compatible providers by host -- as JSON or as a stream."""

    def __init__(self, answers: dict[str, object]) -> None:
        super().__init__()
        self.answers = answers

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        host = request.full_url.split("/")[2]
        key = request.full_url.split("/models/", 1)[1].split(":", 1)[0] if host == GEMINI else host
        answer = self.answers.get(key)
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            raise urllib.error.URLError(f"no route for {key}")
        streaming = "alt=sse" in request.full_url or bool(body.get("stream"))
        if streaming:
            if host == GEMINI:
                return SSEResponse(gemini_chunks([answer["candidates"][0]["content"]["parts"][0]["text"]]))
            return SSEResponse(chat_stream([answer["choices"][0]["message"]["content"]], model=answer.get("model", "")))
        return FakeResponse(answer)


def owner_pool(changes: dict | None = None) -> GatewayConfig:
    """The shipped defaults with the owner's keys entered (providers enabled); Cerebras stays disabled."""

    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    for name in ("gemini", "openai", "groq", "openrouter"):
        document["providers"][name]["enabled"] = True
    for (provider, model), priority in (changes or {}).items():
        for route in document["providers"][provider]["options"]["zero_cost_routes"]:
            if route["model"] == model:
                if priority is None:
                    route.pop("priority", None)
                else:
                    route["priority"] = priority
    return _parse(document, source="test")


def knowledge(mode: ChatMode = ChatMode.FREE) -> GatewayRequest:
    text = "Warum ist der Himmel blau?"
    return GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_question=True), mode=mode, owner_text=text)


def hosts(net: FakeNetwork) -> list[str]:
    return [r["url"].split("/")[2] for r in net.requests]


def route_answer(provider: str, model: str) -> object:
    text = f"Antwort von Route {provider}/{model}."
    return gemini_reply(text) if provider == "gemini" else chat_completion(text, model=model)


def route_down(provider: str, model: str) -> Exception:
    if provider == "gemini":
        return quota_error()
    return http_error(GROQ if provider == "groq" else OPENROUTER, 503, {"error": {"message": "service unavailable"}})


# ---------------------------------------------------------------------------
# the configured pool and its order
# ---------------------------------------------------------------------------

def test_the_shipped_pool_is_the_owner_order_with_evidence_and_cerebras_out(tmp_path):
    registry = ZeroCostRegistry(owner_pool())
    ordered = [(r.provider_id, r.model_id, r.priority) for r in registry.routes]
    assert ordered[:5] == [(p, m, i) for i, (p, m) in enumerate(POOL, start=1)]
    by_key = {r.key: r for r in registry.routes}
    for key in ("groq/openai/gpt-oss-120b", "openrouter/openrouter/free"):
        assert by_key[key].verified_zero_cost and by_key[key].verified_at == "2026-09-16" and "live_zero_cost" in by_key[key].verified_by
    cerebras = by_key["cerebras/gpt-oss-120b"]
    assert not cerebras.verified_zero_cost and "402" in cerebras.note
    assert GatewayConfig.defaults().providers["cerebras"].enabled is False
    loaded = GatewayConfig.load(GatewayConfig.defaults().save(tmp_path / "providers.json"), override_path=tmp_path / "no_override.json")
    assert [(r.provider_id, r.model_id) for r in ZeroCostRegistry(loaded).routes][:5] == POOL, "the order survives a save and a load"


def test_the_order_is_the_priority_never_the_provider_names(tmp_path, pool_creds):
    # the same routes with Groq and OpenRouter swapped: OpenRouter ("o") now answers before Groq ("g")
    config = owner_pool({("groq", "openai/gpt-oss-120b"): 5, ("openrouter", "openrouter/free"): 4})
    assert [r.provider_id for r in ZeroCostRegistry(config).routes][3:5] == ["openrouter", "groq"]
    net = RouteNetwork({m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: route_answer("groq", "openai/gpt-oss-120b"),
                                                                                OPENROUTER: route_answer("openrouter", "openrouter/free")})
    reply = make_gateway(tmp_path, config, pool_creds, net, local=LocalStub()).complete(knowledge())
    assert reply.provider == "openrouter" and GROQ not in hosts(net)


def test_a_route_without_a_priority_or_with_a_shared_one_is_never_used(tmp_path, pool_creds):
    missing = owner_pool({("groq", "openai/gpt-oss-120b"): None})
    verdicts = {v.route.key: v for v in ZeroCostRegistry(missing).verdicts(health=ProviderHealth(), credential_present=lambda _n: True)}
    assert not verdicts["groq/openai/gpt-oss-120b"].eligible and verdicts["groq/openai/gpt-oss-120b"].reason == "no priority configured"
    shared = owner_pool({("openrouter", "openrouter/free"): 4})
    gateway = make_gateway(tmp_path, shared, pool_creds, RouteNetwork({}), local=LocalStub())
    reasons = {f"{r['provider_id']}/{r['model_id']}": r["reason"] for r in gateway.status()["zero_cost_routes"]}
    assert "shared" in reasons["groq/openai/gpt-oss-120b"] and "shared" in reasons["openrouter/openrouter/free"]


# ---------------------------------------------------------------------------
# every route can answer alone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("survivor", POOL, ids=[f"{p}/{m}" for p, m in POOL])
@pytest.mark.parametrize("path", ["complete", "stream"])
def test_each_route_answers_when_every_other_route_is_unavailable(tmp_path, pool_creds, survivor, path):
    answers: dict[str, object] = {OPENAI: openai_reply("paid -- must never be asked")}
    for provider, model in POOL:
        key = model if provider == "gemini" else (GROQ if provider == "groq" else OPENROUTER)
        answers[key] = route_answer(provider, model) if (provider, model) == survivor else route_down(provider, model)
    net = RouteNetwork(answers)
    local = LocalStub()
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, net, local=local)
    if path == "complete":
        reply = gateway.complete(knowledge())
        text = reply.text
    else:
        stream = gateway.stream(knowledge())
        text = "".join(stream)
        reply = stream.reply
    provider, model = survivor
    assert (reply.provider, reply.model) == survivor and text == f"Antwort von Route {provider}/{model}."
    assert reply.actual_eur == 0.0 and reply.decision.intelligence_class == "ZERO_COST" and reply.decision.emergency is False
    assert OPENAI not in hosts(net) and CEREBRAS not in hosts(net) and gateway.governor.summary().month == 0.0 and local.calls == []
    attempted = list(dict.fromkeys((a["provider"], a["model"]) for a in reply.route_attempts))
    assert attempted == POOL[: POOL.index(survivor) + 1], "the routes before it were tried in priority order (a 5xx gets its one bounded retry)"
    assert all(a["failure_class"] != "ok" for a in reply.route_attempts[:-1]) and reply.route_attempts[-1]["failure_class"] == "ok"


def test_with_every_zero_cost_route_down_free_mode_refuses_and_nothing_paid_is_called(tmp_path, pool_creds):
    answers = {OPENAI: openai_reply("paid")}
    answers |= {model if p == "gemini" else (GROQ if p == "groq" else OPENROUTER): route_down(p, model) for p, model in POOL}
    net = RouteNetwork(answers)
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, net, local=LocalStub())
    with pytest.raises(FreeIntelligenceUnavailable) as info:
        gateway.stream(knowledge()).__iter__().__next__()
    assert list(dict.fromkeys((a["provider"], a["model"]) for a in info.value.attempts)) == POOL
    assert OPENAI not in hosts(net) and CEREBRAS not in hosts(net) and gateway.governor.summary().month == 0.0


def test_the_zero_cost_pool_never_serves_a_paid_class(tmp_path, pool_creds):
    """SMART is decided before any provider: its outage stays SMART's, the free routes are not a fallback for it."""

    net = RouteNetwork({OPENAI: http_error(OPENAI, 503, {"error": {"message": "down"}}), GROQ: route_answer("groq", "openai/gpt-oss-120b"),
                        OPENROUTER: route_answer("openrouter", "openrouter/free")} | {m: route_answer(p, m) for p, m in POOL if p == "gemini"})
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, net, local=LocalStub())
    with pytest.raises(GatewayError):
        gateway.complete(knowledge(ChatMode.SMART))
    assert set(hosts(net)) == {OPENAI}


# ---------------------------------------------------------------------------
# quota, rate limits, reset times
# ---------------------------------------------------------------------------

def test_a_groq_daily_quota_stays_spent_until_the_time_groq_names(tmp_path, pool_creds):
    answers = {m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: groq_daily_limit(), OPENROUTER: route_answer("openrouter", "openrouter/free")}
    net = RouteNetwork(answers)
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, net, local=LocalStub())
    first = gateway.complete(knowledge())
    assert first.provider == "openrouter"
    assert ("groq", "quota_exhausted") in [(a["provider"], a["failure_class"]) for a in first.route_attempts]
    assert gateway.health.status("groq", model="openai/gpt-oss-120b") is ProviderStatus.QUOTA_EXHAUSTED
    remaining = gateway.health.cooldown_remaining("groq", model="openai/gpt-oss-120b")
    assert 420 < remaining <= 433, "Groq said 7m12.3s: that, not thirty seconds"
    before = hosts(net).count(GROQ)
    for _ in range(3):
        assert gateway.complete(knowledge()).provider == "openrouter"
    assert hosts(net).count(GROQ) == before, "a spent daily quota is skipped without a network call"


def test_a_daily_quota_without_a_named_time_lasts_until_utc_midnight_for_groq(tmp_path, pool_creds):
    answers = {m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: groq_daily_limit(try_again=""), OPENROUTER: route_answer("openrouter", "openrouter/free")}
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, RouteNetwork(answers), local=LocalStub())
    gateway.complete(knowledge())
    remaining = gateway.health.cooldown_remaining("groq", model="openai/gpt-oss-120b")
    assert abs(remaining - seconds_until_daily_reset(zone="UTC")) < 5 and remaining != 30.0


def test_a_transient_rate_limit_uses_retry_after_and_recovers(tmp_path, pool_creds):
    limited = http_error(GROQ, 429, {"error": {"message": "Rate limit reached ... on requests per minute (RPM): Limit 30."}}, headers={"Retry-After": "7"})
    answers = {m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: limited, OPENROUTER: route_answer("openrouter", "openrouter/free")}
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, RouteNetwork(answers), local=LocalStub())
    assert gateway.complete(knowledge()).provider == "openrouter"
    assert gateway.health.status("groq", model="openai/gpt-oss-120b") is ProviderStatus.RATE_LIMIT
    assert 5 < gateway.health.cooldown_remaining("groq") <= 7


def test_retry_after_is_read_from_every_place_providers_put_it():
    assert retry_after_from({"Retry-After": "12"}, "") == 12.0
    assert retry_after_from({}, '{"error": {"details": [{"retryDelay": "23s"}]}}') == 23.0
    assert retry_after_from({}, "Please try again in 1m26.4s.") == pytest.approx(86.4)
    assert retry_after_from({}, "Please try again in 950ms.") == pytest.approx(0.95)
    assert retry_after_from({"x-ratelimit-reset-requests": "2m0s", "x-ratelimit-reset-tokens": "3.5s"}, "") == 120.0
    assert 99 < retry_after_from({"X-RateLimit-Reset": str(int((time.time() + 100) * 1000))}, "") <= 100
    assert retry_after_from({}, "nothing here") is None


def test_the_openrouter_daily_free_allowance_is_a_quota_until_utc_midnight(tmp_path, pool_creds):
    spent = http_error(OPENROUTER, 429, {"error": {"message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
                                                   "code": 429}})
    answers = {m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: route_down("groq", ""), OPENROUTER: spent}
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, RouteNetwork(answers), local=LocalStub())
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(knowledge())
    assert gateway.health.status("openrouter", model="openrouter/free") is ProviderStatus.QUOTA_EXHAUSTED
    assert abs(gateway.health.cooldown_remaining("openrouter", model="openrouter/free") - seconds_until_daily_reset(zone="UTC")) < 5


def test_the_pacific_day_is_computed_without_a_tz_database():
    from datetime import datetime, timezone

    winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc).timestamp()  # 04:00 PST
    summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp()  # 05:00 PDT
    assert seconds_until_daily_reset(now=winter) == pytest.approx(20 * 3600)
    assert seconds_until_daily_reset(now=summer) == pytest.approx(19 * 3600)
    assert seconds_until_daily_reset(now=summer, zone="UTC") == pytest.approx(12 * 3600)


# ---------------------------------------------------------------------------
# the model that served
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["complete", "stream"])
def test_openrouter_records_the_model_that_actually_served(tmp_path, pool_creds, path):
    answers = {m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: route_down("groq", ""),
                                                                       OPENROUTER: chat_completion("Rayleigh-Streuung.", model="nex-agi/nex-n2.5-pro:free")}
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, RouteNetwork(answers), local=LocalStub())
    if path == "complete":
        reply = gateway.complete(knowledge())
    else:
        stream = gateway.stream(knowledge())
        assert "".join(stream) == "Rayleigh-Streuung."
        reply = stream.reply
    assert reply.model == "openrouter/free" and reply.served_model == "nex-agi/nex-n2.5-pro:free"
    assert reply.route_attempts[-1]["served_model"] == "nex-agi/nex-n2.5-pro:free"
    assert reply.to_dict()["served_model"] == "nex-agi/nex-n2.5-pro:free"
    assert gateway.health.status("openrouter", model="openrouter/free") is ProviderStatus.OK
    assert "openrouter/nex-agi/nex-n2.5-pro:free" not in gateway.health.state, "health is kept for the configured route, not the served model"


# ---------------------------------------------------------------------------
# a provider-wide outage of the bound provider does not close the pool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("survivor", [("groq", "openai/gpt-oss-120b"), ("openrouter", "openrouter/free")], ids=["groq", "openrouter"])
def test_gemini_in_a_provider_wide_cool_down_still_leaves_the_pool_answering(tmp_path, pool_creds, survivor):
    """Found live: Gemini unreachable put the provider in a 60 s cool-down; the next FREE question was refused at planning
    ("provider gemini: provider_unavailable") while Groq and OpenRouter were healthy.  The pool decides the role's availability."""

    tunnel_refused = urllib.error.URLError(OSError("Tunnel connection failed: 503 Service Unavailable"))
    answers: dict[str, object] = {m: tunnel_refused for p, m in POOL if p == "gemini"} | {OPENAI: openai_reply("paid")}
    answers[GROQ] = route_answer("groq", "openai/gpt-oss-120b") if survivor[0] == "groq" else tunnel_refused
    answers[OPENROUTER] = route_answer("openrouter", "openrouter/free")
    net = RouteNetwork(answers)
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, net, local=LocalStub())
    first = gateway.stream(knowledge())
    "".join(first)
    assert (first.reply.provider, first.reply.model) == survivor
    assert not gateway.health.usable("gemini"), "Gemini is in its provider-wide cool-down now"
    calls_before = len(net.requests)
    second = gateway.stream(knowledge())
    text = "".join(second)
    assert (second.reply.provider, second.reply.model) == survivor and text.startswith("Antwort von Route")
    assert second.reply.actual_eur == 0.0 and second.reply.decision.intelligence_class == "ZERO_COST"
    assert GEMINI not in hosts(net)[calls_before:], "the cooling provider is skipped without a network call"
    assert OPENAI not in hosts(net) and gateway.governor.summary().month == 0.0


def test_with_the_whole_pool_cooling_down_free_is_refused_before_any_call(tmp_path, pool_creds):
    net = RouteNetwork({OPENAI: openai_reply("paid")})
    gateway = make_gateway(tmp_path, owner_pool(), pool_creds, net, local=LocalStub())
    for name in ("gemini", "groq", "openrouter"):
        gateway.health.note(name, ProviderStatus.PROVIDER_UNAVAILABLE)
    decision, _ = gateway.plan(knowledge())
    assert decision.kind.value == "refused" and "provider gemini" in decision.reason
    with pytest.raises(Exception):
        "".join(gateway.stream(knowledge()))
    assert net.requests == []


def test_sensitive_content_never_reaches_a_may_train_pool_route(tmp_path, pool_creds):
    from gateway.privacy import Chunk, Sensitivity

    document = json.loads(json.dumps(owner_pool().to_dict()))
    document["providers"]["gemini"]["may_train_on_requests"] = False  # the bound provider would accept it; OpenRouter may train
    config = _parse(document, source="test")
    answers = {m: quota_error() for p, m in POOL if p == "gemini"} | {GROQ: route_down("groq", ""), OPENROUTER: route_answer("openrouter", "openrouter/free")}
    net = RouteNetwork(answers)
    gateway = make_gateway(tmp_path, config, pool_creds, net, local=LocalStub())
    request = knowledge()
    request.chunks = [Chunk(text="Kontostand und Befund", source="owner_message", sensitivity=Sensitivity.PRIVATE)]
    with pytest.raises(FreeIntelligenceUnavailable):
        gateway.complete(request)
    assert OPENROUTER not in hosts(net) and GROQ in hosts(net)
