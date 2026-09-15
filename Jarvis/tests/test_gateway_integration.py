"""The gateway inside the real product: kernel, core, HTTP.

A real :class:`JarvisKernel` on a temporary state root, its local Ollama tier
replaced by a stub, the network replaced by a recorder.  What these tests
prove is the wiring the brief asks for: the chat mode set through the API is
the mode every model call in that answer runs under; FREE mode sends nothing
to a paid provider; the conversation reply comes from the free cloud lane
when it is configured and from the local model only as the fallback; the
credential endpoints never echo a key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from brain.tiers import ModelTier
from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from gateway.config import GatewayConfig, owner_override_path
from gateway.gateway import GatewayBrainProvider
from gateway.modes import ChatMode
from service.core import JarvisCore
from service.events import EventType
from test_model_gateway import FakeNetwork, LocalStub, gemini_reply, openai_reply


class LocalChat(LocalStub):
    """The local 4B stand-in: answers anything, streams too."""

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return "lokale Antwort"

    def generate_stream(self, prompt, **kwargs):
        self.calls.append(prompt)
        yield "lokale Antwort"


@pytest.fixture
def world(tmp_path: Path, monkeypatch):
    """A core whose kernel is real and whose network is ours."""

    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("Beta-Oxidation ist der Abbau von Fettsäuren.")
    net.responses["api.openai.com"] = openai_reply("Eine tiefe Antwort.")
    local = LocalChat()

    config_root = tmp_path / "config"
    config_root.mkdir()
    cfg = GatewayConfig.defaults().with_provider_enabled("gemini", True).with_provider_enabled("openai", True)
    cfg.save(config_root / "providers.json")
    # The owner has enabled metered billing through their spending document.
    (config_root / "owner").mkdir()
    (config_root / "owner" / "spending.json").write_text(json.dumps({"paid_api": True}), encoding="utf-8")

    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state", config_root=config_root, enable_research_tools=False))
    kernel.local_provider = lambda tier: local  # type: ignore[assignment]
    # The gateway's transport must use the fake network.
    gateway = kernel.gateway
    gateway.transport._opener = net
    gateway.credentials.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    gateway.credentials.set("openai", "sk-openaitestkey00000000000000000")
    core = JarvisCore(kernel=kernel, identity=Identity())
    core.language = "de"
    return core, kernel, net, local


def ask(core: JarvisCore, text: str, *, mode: str = "", wait: float = 20.0) -> list:
    with core.bus.subscribe(replay=False) as sub:
        meta = {"source": "test"}
        if mode:
            meta["mode"] = mode
        core.send_message(text, meta=meta)
        deadline = time.time() + wait
        events = []
        while time.time() < deadline:
            events.extend(sub.drain())
            if any(e.type is EventType.MESSAGE for e in events):
                break
            time.sleep(0.05)
    return events


def answer_text(events) -> str:
    for event in events:
        if event.type is EventType.MESSAGE:
            return str(event.payload.get("text", ""))
    return ""


def test_kernel_hands_out_a_gateway_provider_for_the_conversational_tier(world):
    core, kernel, net, local = world
    provider = kernel.provider(ModelTier.FAST_LOCAL)
    assert isinstance(provider, GatewayBrainProvider)
    assert kernel.provider(ModelTier.BUILD_LOCAL) is local, "only the conversational tier is routed; building stays local"
    assert hasattr(provider, "generate_stream") and hasattr(provider, "generate_structured")


def test_a_knowledge_question_is_answered_by_the_free_lane_not_the_local_model(world):
    core, kernel, net, local = world
    events = ask(core, "Was ist Beta-Oxidation?")
    text = answer_text(events)
    assert "Beta-Oxidation ist der Abbau" in text, [e.payload for e in events if e.type is EventType.ERROR]
    assert any("googleapis" in r["url"] for r in net.requests)
    assert not any("Beta-Oxidation" in call for call in local.calls), "the 4B model is off the normal path"


def test_free_mode_set_through_the_message_never_pays(world):
    core, kernel, net, local = world
    kernel.gateway.health.note("gemini", __import__("gateway.health", fromlist=["ProviderStatus"]).ProviderStatus.QUOTA_EXHAUSTED)
    events = ask(core, "Erkläre mir bitte ausführlich die Frank-Starling-Mechanik.", mode="FREE")
    assert core.chat_mode is ChatMode.FREE
    text = answer_text(events)
    assert "Die kostenlose KI ist gerade ausgelastet" in text and "lokale Antwort" not in text
    assert local.calls == [], "the local model answers nobody"
    assert not any("api.openai.com" in r["url"] for r in net.requests)
    assert kernel.gateway.governor.summary().month == 0.0


def test_auto_mode_routes_around_an_exhausted_free_lane_and_accounts_for_it(world):
    core, kernel, net, local = world
    from gateway.health import ProviderStatus

    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("AUTO")
    events = ask(core, "Was ist NAT?")
    assert "Eine tiefe Antwort" in answer_text(events)
    assert any("api.openai.com" in r["url"] for r in net.requests)
    spend = kernel.gateway.governor.summary()
    assert spend.month > 0.0 and spend.open_reservations == 0
    status = core.gateway_status()
    assert status["spend"]["month"] == pytest.approx(spend.month, abs=1e-4)
    assert status["mode"] == "AUTO"


def test_mode_endpoint_validates_and_persists(world):
    core, kernel, net, local = world
    assert core.set_chat_mode("deep")["mode"] == "DEEP"
    assert core.chat_mode is ChatMode.DEEP
    assert core.preferences.get("gateway.mode") == "DEEP"
    bad = core.set_chat_mode("TURBO")
    assert bad["ok"] is False and core.chat_mode is ChatMode.DEEP


def test_estimate_without_running(world):
    core, kernel, net, local = world
    before = len(net.requests)
    result = core.gateway_estimate("Was ist NAT?", mode="AUTO")
    assert result["ok"] and result["decision"]["kind"] == "model" and result["decision"]["role"] == "reasoning.free"
    assert result["decision"]["estimate"]["estimated_eur"] == 0.0
    assert len(net.requests) == before


def test_credential_endpoints_are_owner_gated_and_never_echo_the_key(world):
    core, kernel, net, local = world
    refused = core.provider_set_credential("anthropic", "sk-ant-secret000000000000000000")
    assert refused.get("ok") is False
    assert not kernel.gateway.credentials.has("anthropic")
    core.security.setup("correct horse battery")
    token = core.security.unlock("correct horse battery", "CREDENTIALS")["authorization"]
    stored = core.provider_set_credential("anthropic", "sk-ant-secret000000000000000000", authorization=token)
    assert stored["ok"] and stored["credentials"]["anthropic"]["configured"]
    assert "sk-ant-secret" not in json.dumps(stored)
    assert "sk-ant-secret" not in json.dumps(core.providers_status())
    assert "sk-ant-secret" not in json.dumps(core.diagnostics(refresh=False))
    token = core.security.unlock("correct horse battery", "CREDENTIALS")["authorization"]
    enabled = core.provider_enable("anthropic", True, authorization=token)
    assert enabled["ok"] and enabled["providers"]["anthropic"]["enabled"]
    assert GatewayConfig.load(
        kernel.config_root / "providers.json",
        override_path=owner_override_path(kernel.state_root),
    ).providers["anthropic"].enabled


def test_diagnostics_show_the_provider_identity_the_conversation_hides(world):
    core, kernel, net, local = world
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("Ich bin Gemini, ein Modell von Google. NAT ist Adressübersetzung.")
    events = ask(core, "Was ist NAT?")
    text = answer_text(events)
    assert "Gemini" not in text and "Google" not in text and "NAT ist" in text
    recent = core.gateway_status()["recent"]
    assert recent and recent[-1]["provider"] == "gemini"


def test_owner_feedback_on_an_answer_teaches_the_router(world):
    core, kernel, net, local = world
    gateway = kernel.gateway
    from gateway.task import TaskClass

    before = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    with core.bus.subscribe(replay=False) as sub:
        core.send_message("Was ist Beta-Oxidation?", meta={"source": "test"}, request_id="fb-1")
        deadline = time.time() + 20
        while time.time() < deadline and not any(e.type is EventType.MESSAGE for e in sub.drain()):
            time.sleep(0.05)
    unchanged = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert (unchanged.alpha, unchanged.beta) == (before.alpha, before.beta), "an answer alone proves nothing"
    core.feedback("response", rating="down", request_id="fb-1")
    after = gateway.reliability.reliability("reasoning.free", TaskClass.KNOWLEDGE)
    assert after.beta > before.beta and after.observations >= 1
    rows = gateway.ledger.observations()
    assert rows[-1].goal_verified is False and rows[-1].failure_class == "task_failure"


def _checksum_manifest():
    from capabilities.models import CapabilityHealth, CapabilityLifecycle, CapabilityManifest

    return CapabilityManifest(
        "local.berechne.sha.256_pruefsumme", "Berechnet die SHA-256-Prüfsumme einer Datei.",
        source_location="skills/installed/local.berechne.sha.256_pruefsumme/1.0.1",
        lifecycle=CapabilityLifecycle.ACTIVE.value,
        health={"state": CapabilityHealth.HEALTHY.value.lower(), "health": CapabilityHealth.HEALTHY.value},
        family="local", name="SHA-256 Prüfsumme", goal_types=["CHECKSUM"], target_types=["FILE"],
        examples=["Berechne die SHA-256-Prüfsumme der Datei X."], aliases=["sha256"], runtime_brain="none",
        codex_required=False, created_by="codex",
    )


def _planner_json(capability_id: str) -> str:
    return json.dumps({"operation": "capability.run", "target": capability_id, "confidence": 0.9,
                       "reason": "Fingerprint einer Datei ist die Prüfsumme"})


def test_a_capable_model_finds_a_capability_by_meaning_and_zeus_runs_it(world, monkeypatch):
    """§10: "Fingerprint der Datei" reaches the checksum capability without that phrase having been taught."""



    core, kernel, net, local = world
    manifest = _checksum_manifest()
    core.capabilities.registry.register(manifest)
    net.responses["generativelanguage.googleapis.com"] = gemini_reply(_planner_json(manifest.capability_id))
    executed: list[str] = []
    monkeypatch.setattr(core, "_execute_capability", lambda m, goal, text, scope, phrase="": executed.append(m.capability_id)
                        or core._deliver("ausgeführt", scope=scope, backend=m.capability_id))
    events = ask(core, "Gib mir den Fingerprint der Datei D:/x.bin")
    assert executed == [manifest.capability_id], [e.payload.get("summary") for e in events if e.type is EventType.TOOL]
    sent = json.dumps(net.requests[-1]["body"])
    assert "Installierte ZEUS-F" in sent and manifest.capability_id in sent, "the planner was shown the installed capabilities"


def test_the_offline_model_is_never_asked_to_name_a_capability(world, monkeypatch):
    """Formerly: the offline model's match was asked about.  Now it is not consulted at all.

    A capability choice is a semantic decision; without the reasoning
    provider it is not made.  The request still gets an answer -- prose from
    whatever may answer prose -- and nothing runs.
    """

    core, kernel, net, local = world
    manifest = _checksum_manifest()
    core.capabilities.registry.register(manifest)
    from gateway.health import ProviderStatus

    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    structured: list[str] = []
    local.generate_structured = lambda prompt, schema, **kwargs: structured.append(prompt) or _planner_json(manifest.capability_id)
    executed: list[str] = []
    monkeypatch.setattr(core, "_execute_capability", lambda m, goal, text, scope, phrase="": executed.append(m.capability_id)
                        or core._deliver("ausgeführt", scope=scope, backend=m.capability_id))
    events = ask(core, "Gib mir den Fingerprint der Datei D:/x.bin")
    assert executed == [] and structured == [], "the offline model is not asked which capability the owner means"
    assert "Meinst du" not in answer_text(events) and not getattr(core, "_pending", None)
    assert answer_text(events), "the owner still gets an answer"
