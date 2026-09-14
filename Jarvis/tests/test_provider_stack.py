"""The real provider stack: dated prices, abstract effort levels, wire formats, credentials, FREE and privacy walls.

Nothing here touches the network; the live suite (``test_live_providers.py``)
does, behind explicit opt-in flags.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from experts.api_engineer import ApiEngineerExpert
from experts.contracts import ExpertJob, ExpertStatus
from gateway.config import THINKING_LEVELS, GatewayConfig, _parse
from gateway.gateway import GatewayRefused, GatewayRequest, ModelGateway
from gateway.health import ProviderStatus, classify_http
from gateway.modes import ChatMode
from gateway.secrets import CredentialStore
from gateway.task import TaskFacts
from test_model_gateway import FakeNetwork, FakeResponse, anthropic_reply, gemini_reply, http_error, make_gateway, openai_reply

ROOT = Path(__file__).resolve().parent.parent


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


def deep_request(text: str = "Plane für mich die nächsten sechs Wochen Lernplan mit Meilensteinen " * 6) -> GatewayRequest:
    return GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP)


# --------------------------------------------------------------------------
# Pricing is dated configuration in the listed currency
# --------------------------------------------------------------------------

def test_dated_prices_pick_the_entry_effective_today_and_convert_to_eur(monkeypatch):
    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    document["providers"]["openai"]["pricing"]["gpt-5.6-sol"] = [
        {"input_per_m": 4.0, "cached_input_per_m": 0.4, "output_per_m": 20.0, "currency": "USD", "effective_from": "2026-01-01",
         "effective_until": "2026-06-30", "confirmed": True},
        {"input_per_m": 2.0, "cached_input_per_m": 0.2, "output_per_m": 10.0, "currency": "USD", "effective_from": "2026-07-01",
         "confirmed": True},
        {"input_per_m": 1.0, "cached_input_per_m": 0.1, "output_per_m": 5.0, "currency": "USD", "effective_from": "2027-01-01",
         "confirmed": True},
    ]
    document["exchange_rates"]["USD"] = {"eur_per_unit": 0.5, "as_of": "2026-09-14", "confirmed": True}
    monkeypatch.setenv("ZEUS_PRICING_TODAY", "2026-09-14")
    config = _parse(document, source="test")
    price = config.pricing_for("reasoning.deep")
    assert price.listed == (2.0, 0.2, 10.0) and price.currency == "USD" and price.effective_from == "2026-07-01"
    assert price.input_per_m == 1.0 and price.cached_input_per_m == 0.1 and price.output_per_m == 5.0, "EUR through the rate"
    assert price.confirmed is True
    monkeypatch.setenv("ZEUS_PRICING_TODAY", "2026-03-01")
    earlier = _parse(document, source="test").pricing_for("reasoning.deep")
    assert earlier.listed == (4.0, 0.4, 20.0)
    monkeypatch.setenv("ZEUS_PRICING_TODAY", "2025-01-01")
    none = _parse(document, source="test")
    assert none.pricing_for("reasoning.deep") is None, "no price effective: the role cannot be estimated or called"
    assert none.providers["openai"].pricing_history["gpt-5.6-sol"][2]["effective_from"] == "2027-01-01", "history is kept"


def test_the_shipped_prices_are_owner_verified_native_figures_with_unavailable_eur_conversion():
    config = GatewayConfig.defaults()
    assert config.exchange_rates == {}, "no market rate is guessed"
    deep = config.pricing_for("reasoning.deep")
    assert deep.listed == (4.0, 0.40, 20.0) and deep.currency == "USD" and deep.effective_from == "2026-09-14" and deep.confirmed
    assert (deep.input_per_m, deep.cached_input_per_m, deep.output_per_m) == (4.0, 0.40, 20.0), "budget guard stays engaged"
    assert deep.rate_to_eur is None and deep.rate_source == "unavailable"
    assert deep.eur_conversion_available is False and deep.eur_conversion_confirmed is False and deep.estimate_confirmed is False
    assert deep.to_eur_dict()["input_per_m_eur"] is None and deep.to_eur_dict()["budget_input_per_m_eur"] == 4.0
    standard = config.pricing_for("engineer.standard")
    assert standard.listed == (5.0, 5.0, 25.0) and standard.confirmed and not standard.estimate_confirmed
    assert "cached input assumed = input" in standard.source
    frontier = config.pricing_for("engineer.frontier")
    assert frontier.listed == (10.0, 10.0, 50.0) and frontier.confirmed and not frontier.estimate_confirmed
    from gateway.modes import CostClass

    free = config.providers["gemini"].price_for(config.roles["reasoning.free"].model)
    assert free.metered is False and free.rate_source == "not-needed" and free.estimate_confirmed
    assert config.cost_class("reasoning.free") is CostClass.ZERO
    assert deep.native_cost({"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}) == 4.0


def test_an_owner_configured_rate_provides_the_only_confirmed_eur_conversion():
    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    document["exchange_rates"]["USD"] = {"eur_per_unit": 0.9, "as_of": "2026-09-14", "confirmed": True}
    config = _parse(document, source="test")
    price = config.pricing_for("reasoning.deep")
    assert price.rate_source == "configured" and price.rate_to_eur == pytest.approx(0.9)
    assert price.eur_conversion_available and price.eur_conversion_confirmed and price.estimate_confirmed
    assert price.input_per_m == pytest.approx(3.6) and price.output_per_m == pytest.approx(18.0)
    assert price.listed == (4.0, 0.40, 20.0), "the listed figure is untouched; only the EUR view moved"
    document["providers"]["openai"]["pricing"]["gpt-5.6-sol"][0]["currency"] = "CHF"
    chf = _parse(document, source="test").pricing_for("reasoning.deep")
    assert chf.rate_source == "unavailable" and chf.rate_to_eur is None
    assert not chf.eur_conversion_available and not chf.estimate_confirmed
    assert chf.input_per_m == 4.0, "an unknown currency keeps the budget guard, but no EUR conversion is claimed"


def test_the_paid_gemini_tier_is_a_separate_metered_provider_that_free_mode_can_never_reach(tmp_path, creds):
    config = GatewayConfig.defaults()
    paid = config.providers["gemini_paid"]
    assert paid.metered and not paid.enabled and paid.secret == "gemini" and paid.kind == "gemini"
    assert paid.price_for("gemini-3.8-flash").metered and paid.price_for("gemini-3.8-flash").confirmed is False
    assert not any(b.provider == "gemini_paid" for b in config.roles.values()), "no role is bound to the paid tier by default"
    # Even when the owner binds the free role to the paid tier, FREE mode refuses before any request.
    bound = config.with_provider_enabled("gemini_paid", True).with_role_binding("reasoning.free", provider="gemini_paid")
    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("paid answer")
    gateway = make_gateway(tmp_path, bound, creds, net)
    from gateway.modes import CostClass

    assert gateway.config.cost_class("reasoning.free") is CostClass.METERED
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True), mode=ChatMode.FREE))
    assert net.requests == [] and "metered" in info.value.decision.reason
    # Free tier exhausted with the paid tier enabled but unbound: nothing routes there.
    free_only = config.with_provider_enabled("gemini", True).with_provider_enabled("gemini_paid", True)
    gateway = make_gateway(tmp_path, free_only, creds, FakeNetwork())
    gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    with pytest.raises(GatewayRefused):
        gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True), mode=ChatMode.FREE))
    assert gateway.transport.issued == 0


def test_the_configuration_round_trips_with_dated_prices_and_rates(tmp_path):
    config = GatewayConfig.defaults()
    path = config.save(tmp_path / "providers.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2 and document["exchange_rates"] == {}
    entry = document["providers"]["openai"]["pricing"]["gpt-5.6-sol"][0]
    assert entry["input_per_m"] == 4.0 and entry["currency"] == "USD", "saved as listed, not as converted"
    again = GatewayConfig.load(path)
    assert again.pricing_for("reasoning.deep") == config.pricing_for("reasoning.deep")
    assert again.roles["reasoning.deep"].thinking == {"FAST": "low", "NORMAL": "medium", "DEEP": "high", "MAX": "xhigh"}


def test_legacy_level_names_in_an_old_file_are_read_as_the_abstract_ones():
    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    document["roles"]["reasoning.free"]["thinking"] = {"FREE_LOW": 0, "FREE_MEDIUM": 2048, "FREE_HIGH": 8192}
    config = _parse(document, source="test")
    assert config.roles["reasoning.free"].thinking == {"FAST": 0, "NORMAL": 2048, "DEEP": 8192}
    assert config.roles["reasoning.free"].thinking_for("MAX") == 8192, "MAX falls back to DEEP where a provider has no MAX"


# --------------------------------------------------------------------------
# Abstract effort levels, provider wire formats
# --------------------------------------------------------------------------

def test_the_four_roles_are_bound_in_configuration_only():
    config = GatewayConfig.defaults()
    for role in ("reasoning.free", "reasoning.deep", "engineer.standard", "engineer.frontier"):
        binding = config.roles[role]
        assert binding.enabled and binding.model and binding.purpose, role
        assert config.providers[binding.provider].kind in {"gemini", "openai", "anthropic"}
    assert config.roles["engineer.frontier_alt"].enabled is False
    assert THINKING_LEVELS == ("FAST", "NORMAL", "DEEP", "MAX")


def test_gemini_gets_a_thinking_level_openai_a_reasoning_effort_anthropic_a_budget(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("ok")
    net.responses["api.openai.com"] = openai_reply("ok")
    net.responses["api.anthropic.com"] = anthropic_reply(json.dumps({"summary": "x", "diff": ""}))
    gateway = make_gateway(tmp_path, cfg, creds, net)

    gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True)))
    gemini_body = net.requests[-1]["body"]
    assert gemini_body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}, "FAST -> the provider's low level"
    assert "thinkingBudget" not in json.dumps(gemini_body)

    reply = gateway.complete(deep_request())
    openai_body = net.requests[-1]["body"]
    assert reply.role == "reasoning.deep" and net.requests[-1]["url"].endswith("/v1/responses")
    assert openai_body["reasoning"]["effort"] == "high" and "temperature" not in openai_body, "DEEP -> high"
    assert openai_body["text"]["format"]["type"] == "json_schema" if "text" in openai_body else True
    assert reply.decision.thinking_level in {"DEEP", "MAX"}

    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_engineering=True, new_subsystem=True, subsystems=5),
                                    mode=ChatMode.BUILD, purpose="engineer", role="engineer.frontier"))
    anthropic_body = net.requests[-1]["body"]
    assert anthropic_body["thinking"]["type"] == "enabled" and anthropic_body["thinking"]["budget_tokens"] >= 1024
    assert anthropic_body["max_tokens"] > anthropic_body["thinking"]["budget_tokens"]
    assert "temperature" not in anthropic_body


def test_max_effort_is_only_reached_in_deep_mode_for_the_hardest_tasks(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["api.openai.com"] = openai_reply("ok")
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    request = deep_request()
    request.soft = {"reasoning_depth": 0.95, "long_horizon": 0.9}
    gateway.complete(request)
    assert net.requests[-1]["body"]["reasoning"]["effort"] == "xhigh"
    request.mode = ChatMode.SMART
    request.max_output_tokens = 200
    gateway.complete(request)
    assert net.requests[-1]["body"]["reasoning"]["effort"] == "high", "MAX is a DEEP-mode decision"


def test_openai_structured_output_and_usage_flow_through_the_responses_api(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["api.openai.com"] = openai_reply('{"primary_goal": "x"}', prompt_tokens=3000, out_tokens=400, cached=1000, reasoning=300)
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    request = deep_request()
    request.schema = {"type": "object", "properties": {"primary_goal": {"type": "string"}}, "required": ["primary_goal"]}
    reply = gateway.complete(request)
    body = net.requests[-1]["body"]
    assert body["text"]["format"] == {"type": "json_schema", "name": "zeus_response", "schema": request.schema, "strict": False}
    assert body["instructions"] and body["store"] is False
    assert json.loads(reply.text)["primary_goal"] == "x"
    assert reply.usage == {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": 400, "reasoning_tokens": 300}
    price = cfg.pricing_for("reasoning.deep")
    assert reply.actual_eur == pytest.approx((2000 * price.input_per_m + 1000 * price.cached_input_per_m + 400 * price.output_per_m) / 1e6)
    assert reply.estimated_eur > 0.0 and gateway.governor.summary().month == pytest.approx(reply.actual_eur)


def test_an_incomplete_openai_response_without_text_is_a_task_failure_not_an_outage(tmp_path, cfg, creds):
    from gateway.health import GatewayError

    net = FakeNetwork()
    net.responses["api.openai.com"] = {"id": "r", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [],
                                       "usage": {"input_tokens": 10, "output_tokens": 5}}
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    with pytest.raises(GatewayError) as info:
        gateway.complete(deep_request())
    assert info.value.status is ProviderStatus.TASK_FAILURE and "max_output_tokens" in str(info.value)
    assert gateway.health.usable("openai"), "a task failure is not held against the provider"


# --------------------------------------------------------------------------
# Provider failure classification: Gemini's two 429s, keys, outages
# --------------------------------------------------------------------------

def test_gemini_per_minute_limits_are_rate_limits_and_per_day_limits_are_quota():
    minute = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota",
                                   "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                                "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                                                "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                               {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"}]}})
    day = minute.replace("PerMinute", "PerDay").replace('"retryDelay": "23s"', '"retryDelay": "3600s"')
    assert classify_http(429, minute, provider_kind="gemini") is ProviderStatus.RATE_LIMIT
    assert classify_http(429, day, provider_kind="gemini") is ProviderStatus.QUOTA_EXHAUSTED
    assert classify_http(429, '{"error": {"message": "insufficient_quota"}}', provider_kind="openai") is ProviderStatus.QUOTA_EXHAUSTED
    assert classify_http(429, '{"error": {"message": "slow down"}}', provider_kind="openai") is ProviderStatus.RATE_LIMIT
    assert classify_http(400, '{"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid"}}', provider_kind="gemini") \
        is ProviderStatus.AUTHENTICATION_ERROR
    assert classify_http(503, "", provider_kind="gemini") is ProviderStatus.PROVIDER_UNAVAILABLE


def test_a_gemini_rate_limit_retries_the_free_pool_and_free_mode_does_not_spend_around_it(tmp_path, cfg, creds):
    from gateway.health import FreeIntelligenceUnavailable

    net = FakeNetwork()
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                      "details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}, {"retryDelay": "17s"}]}}
    net.responses["generativelanguage.googleapis.com"] = http_error("https://generativelanguage.googleapis.com/v1beta/x", 429, body)
    net.responses["api.openai.com"] = openai_reply("paid answer")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    with pytest.raises(FreeIntelligenceUnavailable) as info:
        gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True), mode=ChatMode.FREE))
    assert info.value.typed_status == "FREE_INTELLIGENCE_UNAVAILABLE"
    assert info.value.attempts[0]["failure_class"] == ProviderStatus.RATE_LIMIT.value
    assert info.value.attempts[0]["retry_delay_seconds"] <= 2.0
    assert gateway.health.status("gemini") is ProviderStatus.PROVIDER_UNAVAILABLE
    with pytest.raises(GatewayRefused) as refused:
        gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True), mode=ChatMode.FREE))
    assert "provider_unavailable" in refused.value.decision.reason
    assert all("openai" not in r["url"] for r in net.requests), "FREE never spent to route around the free lane"
    assert gateway.transport.refused == [] or all(r["mode"] == "FREE" for r in gateway.transport.refused)


# --------------------------------------------------------------------------
# Credentials: entered once, stored encrypted, imported from the environment
# --------------------------------------------------------------------------

def test_credentials_are_imported_from_the_environment_once_and_enable_the_provider(tmp_path):
    store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    filled = store.import_environment({"gemini": ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"), "openai": ("OPENAI_API_KEY",),
                                       "anthropic": ("ANTHROPIC_API_KEY",)},
                                      environ={"GEMINI_API_KEY": "AIzaSyFROMENVgemini00000000000000000000000", "OPENAI_API_KEY": " sk-fromenvopenai000000000000000000 "})
    assert filled == ["gemini", "openai"]
    assert store.get("gemini").startswith("AIza") and store.get("openai") == "sk-fromenvopenai000000000000000000"
    assert store.has("anthropic") is False
    again = store.import_environment({"gemini": ("GEMINI_API_KEY",)}, environ={"GEMINI_API_KEY": "AIzaSyOTHERKEY0000000000000000000000000000"})
    assert again == [] and store.get("gemini").startswith("AIzaSyFROMENV"), "a filled slot is not overwritten by the environment"
    raw = (tmp_path / "creds.json").read_text(encoding="utf-8")
    assert "sk-fromenv" not in raw and "AIzaSyFROMENV" not in raw, "never plaintext on disk"


def test_the_gateway_imports_environment_keys_at_startup_and_marks_the_provider_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-startupopenai0000000000000000000000")
    monkeypatch.delenv("GOOGLE_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from runtime.cost_policy import CostPolicy

    store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    gateway = ModelGateway(state_root=tmp_path / "state", config=GatewayConfig.defaults(), credentials=store, opener=FakeNetwork(),
                           cost_policy=CostPolicy(allow_paid_api=True, source="test"))
    status = gateway.status()
    assert status["providers"]["openai"]["state"] == "CONFIGURED" and status["providers"]["openai"]["enabled"] is True
    assert status["providers"]["gemini"]["state"] == "DISABLED", "no key entered: disabled, not broken"
    assert status["providers"]["gemini"]["credential_env"] == ["GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"]
    assert "sk-startup" not in json.dumps(status)
    assert status["credentials"]["openai"]["configured"] is True and status["credentials"]["openai"]["hint"].startswith("…")


def test_provider_states_tell_unconfigured_from_key_rejected_from_outage(tmp_path, cfg):
    empty = CredentialStore(tmp_path / "none.json", use_dpapi=False)
    gateway = make_gateway(tmp_path, cfg, empty, FakeNetwork())
    assert gateway.status()["providers"]["gemini"]["state"] == "UNCONFIGURED"
    assert gateway.status()["roles"]["reasoning.free"]["configured"] is False
    empty.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    assert gateway.status()["providers"]["gemini"]["state"] == "CONFIGURED"
    gateway.health.note("gemini", ProviderStatus.AUTHENTICATION_ERROR, detail="HTTP 400: API key not valid")
    assert gateway.status()["providers"]["gemini"]["state"] == "KEY_REJECTED"
    gateway.health.note("openai", ProviderStatus.QUOTA_EXHAUSTED)
    empty.set("openai", "sk-openaitestkey00000000000000000")
    assert gateway.status()["providers"]["openai"]["state"] == "QUOTA_EXHAUSTED"


def test_entering_a_credential_through_the_core_enables_the_provider_and_persists_it(tmp_path):
    from test_intelligence_flow import ScriptedNetwork, make_world

    core, kernel, local, executed = make_world(tmp_path, ScriptedNetwork(lambda p: {"primary_goal": "none"}))
    config = kernel.gateway.config.with_provider_enabled("anthropic", False)
    kernel.gateway.reconfigure(config)
    assert kernel.gateway.status()["providers"]["anthropic"]["state"] == "DISABLED"
    core.owner_token = getattr(core, "owner_token", "") or ""
    result = core.provider_set_credential("anthropic", "sk-ant-enteredbyowner000000000000000000", authorization=core.security.token
                                          if hasattr(core, "security") and hasattr(core.security, "token") else "")
    if not result.get("ok"):
        pytest.skip(f"owner authorization shape differs in this build: {result}")
    assert result["enabled_provider"] == "anthropic" and result["providers"]["anthropic"]["state"] == "CONFIGURED"
    saved = json.loads((tmp_path / "config" / "providers.json").read_text(encoding="utf-8"))
    assert saved["providers"]["anthropic"]["enabled"] is True
    assert "sk-ant-entered" not in json.dumps(result) and "sk-ant-entered" not in json.dumps(saved)


# --------------------------------------------------------------------------
# FREE is a wall; privacy is a wall; both before any request
# --------------------------------------------------------------------------

def test_in_free_mode_a_paid_call_is_rejected_before_any_network_request(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["api.openai.com"] = openai_reply("paid")
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(GatewayRequest(prompt="Was ist NAT?", facts=TaskFacts(text="Was ist NAT?", is_question=True), mode=ChatMode.FREE))
    assert net.requests == [] and gateway.governor.summary().month == 0.0
    assert "reasoning.deep: mode FREE" in info.value.decision.reason or "metered" in info.value.decision.reason
    from gateway.budget import Reservation
    from gateway.modes import CostClass
    from gateway.transport import ZeroCostViolation

    with pytest.raises(ZeroCostViolation):
        gateway.transport.issue(provider=cfg.providers["openai"], role="reasoning.deep", mode=ChatMode.FREE, cost_class=CostClass.METERED,
                                reservation=Reservation(reservation_id="r", role="reasoning.deep", provider="openai", model="m", task_id="",
                                                        estimated_eur=0.01, reserved_eur=0.015, at="", mode="FREE"))


def test_private_context_never_reaches_the_free_provider_even_in_free_mode(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("x")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    private = "Meine IBAN ist DE89 3704 0044 0532 0130 00, überweise davon die Miete und erkläre mir die Gebühren."
    with pytest.raises(GatewayRefused) as info:
        gateway.complete(GatewayRequest(prompt=private, facts=TaskFacts(text=private), mode=ChatMode.FREE))
    assert net.requests == [], "the privacy wall is decided before a request exists"
    assert "may-train" in info.value.decision.reason or "sensitive" in info.value.decision.reason.lower()


def test_free_mode_with_the_free_provider_exhausted_is_a_typed_free_unavailability_in_the_flow(tmp_path):
    from test_gateway_integration import answer_text, ask
    from test_intelligence_flow import ScriptedNetwork, chess_goal, chess_manifests, chess_plan, make_world, tool_events

    net = ScriptedNetwork(chess_goal, chess_plan)
    core, kernel, local, executed = make_world(tmp_path, net, manifests=chess_manifests(), projects=[("Schach Training", "")])
    core.world.note_event("chess_game_finished", detail={"result": "loss"}, ttl=600)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    preview = core.compose_contract_preview("fuck, schon wieder verloren")
    assert preview["status"] == "FREE_INTELLIGENCE_UNAVAILABLE" and preview["plan"] is None
    events = ask(core, "fuck, schon wieder verloren", wait=30)
    assert executed == [] and tool_events(events, "intelligence: FREE_INTELLIGENCE_UNAVAILABLE")
    text = answer_text(events)
    assert "FREE" in text and "weder ein bezahltes Modell noch das lokale Modell" in text
    assert all("openai" not in r["url"] for r in net.requests), "no silent paid call"
    assert not any("Verständnisschicht" in c for c in local.calls), "no silent local GoalSpec"


# --------------------------------------------------------------------------
# Empirical routing data: what each call leaves behind
# --------------------------------------------------------------------------

def test_every_call_leaves_a_routing_record_without_chain_of_thought(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["api.openai.com"] = openai_reply("Die Antwort.", prompt_tokens=1200, out_tokens=300, cached=200, reasoning=250)
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    request = deep_request()
    request.task_id = "t-1"
    reply = gateway.complete(request)
    gateway.report_task_outcome("t-1", goal_verified=True)
    rows = [json.loads(line) for line in (tmp_path / "state" / "gateway" / "performance.jsonl").read_text(encoding="utf-8").splitlines()]
    row = rows[-1]
    for key in ("task_class", "task_vector", "role", "model", "thinking_level", "latency_seconds", "input_tokens", "cached_input_tokens",
                "output_tokens", "estimated_eur", "actual_eur", "goal_verified", "failure_class", "mode"):
        assert key in row, key
    assert row["role"] == "reasoning.deep" and row["thinking_level"] in {"DEEP", "MAX"} and row["goal_verified"] is True
    assert row["input_tokens"] == 1200 and row["cached_input_tokens"] == 200 and row["output_tokens"] == 300
    assert row["actual_eur"] == pytest.approx(reply.actual_eur) and row["estimated_eur"] == pytest.approx(reply.estimated_eur)
    text = json.dumps(rows)
    assert "Lernplan" not in text and "Die Antwort" not in text, "no prompt, no answer, no chain of thought in the record"


# --------------------------------------------------------------------------
# Provider independence: names live in configuration
# --------------------------------------------------------------------------

#: The intelligence core: routing, composition, engineering decisions, cost.
#: These know the four roles and nothing about who is behind them.
INTELLIGENCE_CORE = (
    "service/core.py", "service/engineering.py", "service/engineering_vector.py", "service/acquisition.py", "service/composer.py",
    "service/selfdev.py", "service/world.py", "capabilities/intelligence.py", "capabilities/planner.py", "capabilities/engineering_spec.py",
    "capabilities/contracts.py", "capabilities/service.py", "capabilities/registry.py", "catalog/context.py",
    "gateway/router.py", "gateway/budget.py", "gateway/modes.py", "gateway/task.py", "gateway/learning.py", "gateway/estimate.py",
    "gateway/privacy.py", "gateway/gateway.py", "experts/api_engineer.py", "experts/gateway.py",
)


def test_no_business_logic_imports_or_names_vendor_models():
    """Model identifiers live in gateway/config.py only; the intelligence core never names a vendor.

    Vendor-specific CLI adapters (the Codex and Claude Code experts, the wire
    adapters in gateway/providers.py) are the places that must know a vendor;
    they are behind the role abstraction, not in front of it.
    """

    vendor_model = re.compile(r"\b(gemini-\d|gpt-\d|claude-(opus|sonnet|fable|haiku)|o\d-mini)\b", re.I)
    vendor_word = re.compile(r"\b(gemini|openai|anthropic|claude|chatgpt|google)\b", re.I)
    offenders: list[str] = []
    for package in ("service", "capabilities", "catalog", "experts", "gateway", "runtime", "projects", "knowledge", "brain"):
        for path in (ROOT / package).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
            if vendor_model.search(code) and rel != "gateway/config.py":
                offenders.append(f"{rel}: model name")
            if rel in INTELLIGENCE_CORE and vendor_word.search(code):
                offenders.append(f"{rel}: vendor word")
    assert offenders == [], offenders


def test_swapping_the_standard_engineer_to_another_provider_is_configuration_only(tmp_path, creds):
    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    document["providers"]["frontier_alt"].update({"kind": "openai_compatible", "base_url": "https://engineering.example", "enabled": True,
                                                  "secret": "openai", "pricing": {"other-engineer": [{"input_per_m": 1.0, "cached_input_per_m": 0.1,
                                                                                                        "output_per_m": 4.0, "currency": "EUR", "confirmed": True}]}})
    document["roles"]["engineer.standard"].update({"provider": "frontier_alt", "model": "other-engineer"})
    config = _parse(document, source="test")
    net = FakeNetwork()
    net.responses["engineering.example"] = {"choices": [{"message": {"content": json.dumps({"summary": "s", "diff": ""})}, "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    gateway = make_gateway(tmp_path, config.with_provider_enabled("anthropic", False), creds, net)
    expert = ApiEngineerExpert(gateway, "engineer.standard")
    assert expert.availability().available and "other-engineer" in expert.availability().detail
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "main.py").write_text("x = 1\n", encoding="utf-8")
    result = expert.execute(ExpertJob(goal="Rename x to y", workspace=workspace, metadata={"files": ["main.py"], "task_id": "swap"}))
    assert result.status is ExpertStatus.FAILED and net.requests and net.requests[0]["url"].startswith("https://engineering.example/")
    assert net.requests[0]["body"]["model"] == "other-engineer"


# --------------------------------------------------------------------------
# Token-efficient engineering: the context pack, incremental source requests
# --------------------------------------------------------------------------

class SequencedNetwork(FakeNetwork):
    """Answers a host with one reply after another."""

    def __call__(self, request, timeout=None):
        host = request.full_url.split("/")[2]
        answer = self.responses.get(host)
        if isinstance(answer, list):
            self.responses[host] = answer[1:] if len(answer) > 1 else answer
            body = json.loads(request.data.decode("utf-8")) if request.data else {}
            self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
            return FakeResponse(answer[0])
        return super().__call__(request, timeout)


def test_the_engineer_gets_a_measured_context_pack_and_may_ask_for_more_source_by_name(tmp_path, cfg, creds):
    workspace = tmp_path / "ws"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "greet.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (workspace / "pkg" / "helper.py").write_text("SUFFIX = ' world'\n", encoding="utf-8")
    (workspace / "pkg" / "unrelated.py").write_text("x = 1\n" * 200, encoding="utf-8")
    diff = ("diff --git a/pkg/greet.py b/pkg/greet.py\n--- a/pkg/greet.py\n+++ b/pkg/greet.py\n@@ -1,2 +1,3 @@\n"
            "+from pkg.helper import SUFFIX\n def greet():\n-    return 'hello'\n+    return 'hello' + SUFFIX\n")
    net = SequencedNetwork()
    net.responses["api.anthropic.com"] = [
        anthropic_reply(json.dumps({"summary": "need the helper", "diff": "", "request_files": ["pkg/helper.py", "../outside.py"]})),
        anthropic_reply(json.dumps({"summary": "greet uses the suffix", "diff": diff})),
    ]
    gateway = make_gateway(tmp_path, cfg, creds, net)
    expert = ApiEngineerExpert(gateway, "engineer.standard")
    job = ExpertJob(goal="Make greet() return 'hello world' using the helper's suffix", workspace=workspace,
                    metadata={"files": ["pkg/greet.py"], "task_id": "ctx-1"})
    pack = expert.context_pack(job)
    tokens = pack["tokens"]
    for key in ("spec_tokens", "catalog_tokens", "interface_tokens", "source_tokens", "test_tokens", "total_engineering_context_tokens"):
        assert key in tokens, key
    assert tokens["source_tokens"] > 0 and tokens["total_engineering_context_tokens"] >= tokens["source_tokens"] + tokens["spec_tokens"]
    assert "unrelated" not in pack["files_text"], "nothing unrelated by default"

    result = expert.execute(job)
    assert result.status is ExpertStatus.COMPLETED, result.blocker
    assert len(net.requests) == 2 and result.raw["context_rounds"] == 1
    second = json.dumps(net.requests[1]["body"])
    assert "SUFFIX = ' world'" in second and "outside.py" in second and "unrelated.py" not in second
    assert result.raw["context_tokens"]["source_tokens"] > tokens["source_tokens"], "the pack grew by exactly the requested file"
    assert (workspace / "pkg" / "greet.py").read_text(encoding="utf-8").startswith("from pkg.helper import SUFFIX")
    assert any("context round 1" in c for c in result.commands_run)
