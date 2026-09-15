"""Two defects from the first real Gemini acceptance run, and the rules that fix them.

1. A false wake word (score 0.91, no speech followed) cancelled a typed
   question's finished cloud answer; the owner received an empty message.
   A wake word may interrupt ZEUS's *voice*; it is no evidence against an
   answer nobody is listening to, and a stop belongs to the generation it
   was issued in.
2. The delivered message was labelled with the FAST_LOCAL tier's catalog
   model although reasoning.free / Gemini had answered.  The label comes
   from the execution receipt of whatever produced the answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from brain.tiers import ModelTier
from gateway.gateway import GatewayBrainProvider, GatewayRequest
from gateway.health import ProviderStatus
from gateway.modes import ChatMode
from gateway.task import TaskFacts
from service.events import EventType
from service.state import JarvisState
from test_gateway_integration import answer_text, ask
from test_intelligence_flow import GOAL_MARK, PLAN_MARK, ScriptedNetwork, make_world
from test_model_gateway import FakeNetwork, FakeResponse, LocalStub, gemini_reply, http_error, make_gateway

ANSWER = "Das tut mir leid. Nächstes Mal klappt es."
QUESTION = "Warum ist der Himmel blau?"


class HookedNetwork(ScriptedNetwork):
    """The scripted provider, with a hook that fires while the conversation answer is being generated."""

    def __init__(self) -> None:
        super().__init__(lambda prompt: {"primary_goal": "none", "confidence": 0.2, "reason": "chat"})
        self.on_conversation: Callable[[], None] | None = None
        self.fail_models: dict[str, int] = {}

    def __call__(self, request, timeout=None):
        url = request.full_url
        for model, remaining in list(self.fail_models.items()):
            if model in url and remaining > 0:
                self.fail_models[model] = remaining - 1
                body = json.loads(request.data.decode("utf-8")) if request.data else {}
                self.requests.append({"url": url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
                raise http_error(url, 429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                                                      "details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"},
                                                                  {"retryDelay": "1s"}]}})
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        prompt = json.dumps(body, ensure_ascii=False)
        if self.on_conversation is not None and GOAL_MARK not in prompt and PLAN_MARK not in prompt and "semantische Steuerung" not in prompt:
            self.on_conversation()
        return super().__call__(request, timeout)


class SpokenVoice:
    """A voice that speaks replies: what a wake word may legitimately interrupt."""

    class settings:  # noqa: N801 - mirrors VoiceSettings' attribute access
        enabled = True
        speak_replies = True

    def __init__(self) -> None:
        self._speaker = None
        self.interrupts = 0
        self.spoken: list[str] = []

    @property
    def speaking(self) -> bool:
        return self._speaker is not None

    def interrupt(self) -> None:
        self.interrupts += 1
        self._speaker = None

    def speak_stream(self, tokens, *, scope: str = ""):
        self._speaker = object()
        try:
            for token in tokens:
                self.spoken.append(token)
        finally:
            self._speaker = None
        return {"ok": True}


def notifications(events, kind: str):
    return [e.payload for e in events if e.type is EventType.NOTIFICATION and e.payload.get("kind") == kind]


def states(events):
    return [e.payload.get("state") for e in events if e.type is EventType.STATE]


# --------------------------------------------------------------------------
# Defect 1: wake words, stops and generations
# --------------------------------------------------------------------------

def test_a_false_wake_during_a_typed_answer_does_not_destroy_the_answer(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    interrupts: list[dict] = []
    net.on_conversation = lambda: interrupts.append(core.voice_interrupt(session="vs-false", wake=0.91))
    events = ask(core, QUESTION, wait=30)
    assert interrupts and interrupts[0]["interrupted"] == [] and interrupts[0].get("protected") == ["answer"]
    assert answer_text(events) == ANSWER, "the finished cloud answer reached the owner"
    assert notifications(events, "barge_in") == [], "no 'Unterbrochen' for an answer nobody was listening to"
    assert "listening" not in states(events), "the core did not leave the answer for a listening session"
    assert not core._stop_requested.is_set()


def test_a_stale_stop_flag_never_poisons_a_later_answer(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    # An earlier generation was stopped; the flag is still set when the next request arrives.
    core._request_stop()
    assert core._stop_requested.is_set()
    events = ask(core, QUESTION, wait=30)
    assert answer_text(events) == ANSWER
    # The binding itself: a stop issued for an older generation does not apply now, even with the event set.
    core._stop_requested.set()
    core._stop_generation = core._answer_generation - 1
    assert core._stop_applies() is False
    core._stop_generation = core._answer_generation
    assert core._stop_applies() is True


def test_a_genuine_barge_in_still_interrupts_a_spoken_answer(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    voice = SpokenVoice()
    core._voice = voice
    interrupts: list[dict] = []
    net.on_conversation = lambda: interrupts.append(core.voice_interrupt(session="vs-real", wake=0.93))
    events = ask(core, QUESTION, wait=30)
    assert interrupts and "answer" in interrupts[0]["interrupted"] and "speech" in interrupts[0]["interrupted"]
    assert voice.interrupts >= 1 and voice.spoken == [], "the speaker was cut off and nothing of the answer was spoken"
    assert answer_text(events) != ANSWER, "the interrupted spoken answer was not delivered as if nothing happened"
    barge = notifications(events, "barge_in")
    assert barge and "answer" in barge[0]["stopped"] and barge[0]["session"] == "vs-real"
    assert "listening" in states(events)


def test_a_false_ambient_wake_with_nothing_running_is_a_no_op(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    with core.bus.subscribe(replay=False) as sub:
        result = core.voice_interrupt(session="vs-ambient", wake=0.88)
        events = sub.drain()
    assert result == {"ok": True, "interrupted": [], "speaking": False}
    assert notifications(events, "barge_in") == [] and core.state.snapshot.state is not JarvisState.LISTENING
    assert not core._stop_requested.is_set()
    # The same wake with a typed answer in flight: the answer is protected, the voice (idle) untouched.
    core._voice = SpokenVoice()
    core._voice.settings = type("S", (), {"enabled": True, "speak_replies": False})()
    interrupts: list[dict] = []
    net.on_conversation = lambda: interrupts.append(core.voice_interrupt(session="vs-ambient-2", wake=0.9))
    events = ask(core, QUESTION, wait=30)
    assert interrupts[0]["interrupted"] == [] and answer_text(events) == ANSWER and core._voice.interrupts == 0


def test_the_owners_explicit_stop_still_cancels_the_current_answer(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    net.on_conversation = lambda: core.stop_current(reason="owner")
    events = ask(core, QUESTION, wait=30)
    assert answer_text(events) != ANSWER and notifications(events, "stop")


# --------------------------------------------------------------------------
# Defect 2: the backend label is the execution receipt
# --------------------------------------------------------------------------

def test_the_delivered_message_names_the_provider_and_model_that_answered(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == ANSWER
    assert message["backend"] == "gemini/gemini-3.8-flash"
    provenance = message["meta"]["provenance"]
    assert provenance["role"] == "reasoning.free" and provenance["provider"] == "gemini" and provenance["model"] == "gemini-3.8-flash"
    assert provenance["offline_fallback"] is False and provenance["actual_eur"] == 0.0
    tier_label = getattr(kernel.catalog.get(ModelTier.FAST_LOCAL), "model", "") or ModelTier.FAST_LOCAL.value
    assert message["backend"] != tier_label, "the resident local tier is not the answer's source"
    assert not any(QUESTION in c for c in local.calls), "the local model did not take part"


def test_when_the_first_pool_model_fails_the_label_names_the_model_that_actually_answered(tmp_path):
    net = HookedNetwork()
    net.fail_models["gemini-3.8-flash"] = 10  # every 3.8 attempt is rate-limited
    core, kernel, local, executed = make_world(tmp_path, net)
    pool = kernel.gateway.config.roles["reasoning.free"].model_pool
    if len(pool) < 2:
        pytest.skip("the free role has a single model; no pool fallback to label")
    events = ask(core, QUESTION, wait=60)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == ANSWER
    assert message["backend"] == f"gemini/{pool[1]}"
    provenance = message["meta"]["provenance"]
    assert provenance["model"] == pool[1] and provenance["actual_eur"] == 0.0
    assert any(a["model"] == "gemini-3.8-flash" and a["failure_class"] != "ok" for a in provenance["route_attempts"])


def test_an_offline_fallback_answer_is_labelled_as_the_local_model_it_came_from(tmp_path):
    net = HookedNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "lokale Antwort"
    assert message["backend"] == "ollama/stub-local"
    assert message["meta"]["provenance"]["offline_fallback"] is True and message["meta"]["provenance"]["role"] == "local.fast"


def test_the_gateway_provider_provenance_is_per_generation(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.secrets import CredentialStore

    cfg = GatewayConfig.defaults().with_provider_enabled("gemini", True)
    creds = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    creds.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("Cloud.")
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg, creds, net, local=local)
    provider = GatewayBrainProvider(gateway, fallback=local)
    assert provider.provenance == {}
    assert provider.generate("Was ist NAT?") == "Cloud."
    first = provider.provenance
    assert first["role"] == "reasoning.free" and first["model"] == "gemini-3.8-flash" and first["offline_fallback"] is False
    gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    text = provider.generate("Was ist NAT?")
    second = provider.provenance
    assert text == "local answer" and second["offline_fallback"] is True and second["model"] == "stub-local"
    assert second is not first and second["role"] == "local.fast"
    assert provider.model_name != "reasoning.free:gemini-3.8-flash" or provider.last_reply is None


def test_provenance_beats_the_catalog_label_for_any_provider():
    from service.core import JarvisCore

    class Cloud:
        provenance = {"role": "reasoning.deep", "provider": "some_vendor", "model": "model-x", "offline_fallback": False}
        model_name = "ignored"

    class Plain:
        provider_name = "ollama"
        model_name = "qwen-local"

    class Mute:
        pass

    assert JarvisCore._answer_provenance(Cloud(), "tier-label") == ("some_vendor/model-x", Cloud.provenance)
    assert JarvisCore._answer_provenance(Plain(), "tier-label") == ("qwen-local", {"model": "qwen-local", "provider": "ollama"})
    assert JarvisCore._answer_provenance(Mute(), "tier-label") == ("tier-label", {})
