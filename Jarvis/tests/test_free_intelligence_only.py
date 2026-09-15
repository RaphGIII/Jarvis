"""The legacy local model answers nobody; FREE is Gemini or one deterministic sentence; streams are bounded per phase.

Live on 2026-09-15 the free pool sat in health cool-down and the owner's
prose came from qwen3:4b-instruct -- chemically wrong.  A Gemini stream held
its connection 115 s before a 503, and a stalled stream waited the full
120 s silence bound.  These tests pin what replaced that: no mode permits
``local.fast``; the kernel's gateway provider has no local fallback; startup
and the health badge generate nothing; the owner reads one fixed sentence
when the free intelligence is unavailable, and SMART is named, never taken;
a streamed answer has a connect, first-token, idle and total bound.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from gateway.gateway import INTERACTIVE_STREAM_TIMEOUTS, LONG_STREAM_TIMEOUTS, GatewayRefused, GatewayRequest
from gateway.health import ProviderStatus
from gateway.modes import ChatMode, policy_for
from gateway.task import TaskFacts
from gateway.transport import StreamTimeouts
from service.core import JarvisCore
from service.events import EventType
from test_gateway_integration import answer_text, ask
from test_intelligence_flow import make_world
from test_model_gateway import FakeNetwork, LocalStub, cfg, creds, gemini_reply, make_gateway, openai_reply  # noqa: F401 - fixtures
from test_semantic_authority import GEMINI, OPENAI, http_503, outage_world, semantic_local_calls, tools
from test_streaming import gemini_chunks

FREE_TEXT_DE = JarvisCore.FREE_UNAVAILABLE_DE
GENERAL_TEXT_DE = JarvisCore.INTELLIGENCE_UNAVAILABLE_DE
QUESTION = "Erkläre mir die kompetitive Enzymhemmung."


# ---------------------------------------------------------------------------
# 1 + 2 + 3: Gemini unavailable -> no local generation, no paid call, one deterministic sentence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failure", [http_503, lambda: TimeoutError("timed out")], ids=["503", "timeout"])
def test_free_outage_is_one_deterministic_sentence_with_no_local_and_no_paid_call(tmp_path, failure):
    net, core, kernel, local, executed = outage_world(tmp_path, failure())
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == FREE_TEXT_DE, message["text"]
    assert message["backend"] == "intelligence"
    assert local.calls == [], "no local generation of any kind"
    assert not any(OPENAI in r["url"] for r in net.requests), "no paid provider call"
    assert kernel.gateway.status()["spend"]["month"] == 0.0
    text = message["text"]
    assert "429" not in text and "503" not in text and "Traceback" not in text and "gemini" not in text.lower()


def test_free_in_health_cooldown_is_the_same_sentence_never_the_local_model(tmp_path):
    """The live defect: the free pool was in cool-down and the router handed prose to the 4B."""

    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("unused")
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    events = ask(core, "Schreibe mir eine kurze Erklärung zur Keto-Enol-Tautomerie.", wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == FREE_TEXT_DE and message["backend"] == "intelligence"
    assert local.calls == [] and net.requests == []
    assert any("no local model stands in, no paid escalation" in str(p.get("summary")) for p in tools(events))
    states = [e.payload.get("state") for e in events if e.type is EventType.STATE]
    assert "error" not in states


def test_outside_free_the_sentence_is_the_general_one_and_still_not_local(tmp_path):
    net, core, kernel, local, executed = outage_world(tmp_path, TimeoutError("timed out"), mode="AUTO")
    net.responses[OPENAI] = TimeoutError("timed out")
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"].startswith(GENERAL_TEXT_DE), message["text"]
    assert local.calls == []


# ---------------------------------------------------------------------------
# 4: startup and the health badge generate nothing on the local model
# ---------------------------------------------------------------------------

def test_startup_marks_ready_from_the_gateway_without_loading_the_local_model(tmp_path):
    net = FakeNetwork()
    net.responses[GEMINI] = gemini_reply("unused")
    core, kernel, local, executed = make_world(tmp_path, net)
    assert not core.lifecycle.ready
    thread = core.warm(speech=False)
    thread.join(timeout=120)
    assert not thread.is_alive()
    assert core.lifecycle.ready and core.lifecycle.readiness()["AI_READY"] is True
    stage = core.lifecycle.stages["intelligence"]
    assert stage["ok"] is True and "reasoning.free" in stage["detail"]
    assert "fast_local" not in core.lifecycle.stages
    assert local.calls == [], "READY was earned without a single local generation"
    assert net.requests == [], "and without a provider call either"
    core._probe_health()  # noqa: SLF001 - the badge's probe
    assert local.calls == [] and core._health_ok is True  # noqa: SLF001
    health = core.lifecycle.health()
    assert health["ready"] is True and health["detail"] == "ready"


def test_a_core_with_no_reasoning_provider_is_still_ready_and_says_so(tmp_path, monkeypatch):
    net = FakeNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway.credentials.delete("gemini") if hasattr(kernel.gateway.credentials, "delete") else None
    for name in ("gemini", "openai"):
        kernel.gateway.config = kernel.gateway.config.with_provider_enabled(name, False)
    thread = core.warm(speech=False)
    thread.join(timeout=120)
    assert core.lifecycle.ready
    assert "deterministic answers only" in core.lifecycle.stages["intelligence"]["detail"] or "configured" in core.lifecycle.stages["intelligence"]["detail"]
    assert local.calls == []


# ---------------------------------------------------------------------------
# 5 + 6: no mode routes to local.fast; the gateway provider has no way down
# ---------------------------------------------------------------------------

def test_no_mode_permits_the_conversational_local_model():
    for mode in ChatMode:
        policy = policy_for(mode)
        assert not policy.permits_role("local.fast"), mode
        assert policy.permits_role("local.build"), "the engineering coder stays permitted"
    assert policy_for(ChatMode.SMART).permits_role("reasoning.smart") and policy_for(ChatMode.DEEP).permits_role("reasoning.deep")
    assert not policy_for(ChatMode.FREE).permits_role("reasoning.smart")


def test_the_kernel_gateway_provider_has_no_local_fallback(tmp_path):
    net = FakeNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    from brain.tiers import ModelTier

    provider = kernel.provider(ModelTier.FAST_LOCAL)
    assert provider.fallback is None
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    kernel.gateway.health.note("openai", ProviderStatus.QUOTA_EXHAUSTED)
    from gateway.gateway import RequestContext, reset_context, set_context

    token = set_context(RequestContext(mode=ChatMode.FREE))
    try:
        with pytest.raises(GatewayRefused):
            provider.generate("Was ist NAT?")
        with pytest.raises(GatewayRefused):
            list(provider.generate_stream("Was ist NAT?"))
    finally:
        reset_context(token)
    assert local.calls == []


def test_free_with_no_free_lane_is_refused_not_answered_locally(tmp_path, cfg, creds):
    net = FakeNetwork()
    local = LocalStub()
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net, local=local)
    request = GatewayRequest(prompt="Erkläre mir die Frank-Starling-Mechanik.", facts=TaskFacts(text="x", is_question=True), mode=ChatMode.FREE)
    decision, _ = gateway.plan(request)
    assert decision.kind.value != "model"
    with pytest.raises(GatewayRefused):
        gateway.complete(request)
    assert local.calls == [] and net.requests == []
    candidates = gateway.router.candidates(decision.task, ChatMode.FREE, None, prompt="x") if hasattr(gateway, "router") else []
    for candidate in candidates:
        if candidate.role == "local.fast":
            assert not candidate.eligible and "does not permit" in candidate.reason


# ---------------------------------------------------------------------------
# 7: deterministic local capabilities work with no LLM anywhere
# ---------------------------------------------------------------------------

def test_a_spelled_out_file_write_runs_with_no_provider_and_no_local_model(tmp_path):
    net = FakeNetwork()
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway.health.note("gemini", ProviderStatus.QUOTA_EXHAUSTED)
    core.set_chat_mode("FREE")
    events = ask(core, "Schreibe eine Datei ohne_llm.txt mit Inhalt Deterministisch.", wait=30)
    assert (core.actions.workspace / "ohne_llm.txt").read_text(encoding="utf-8") == "Deterministisch"
    assert local.calls == [] and net.requests == []
    assert next(e.payload for e in events if e.type is EventType.MESSAGE)["backend"] == "tools.write_file"


# ---------------------------------------------------------------------------
# 8 + 9 + 10 + 11 + 12: phased stream bounds, and the pool's walk from 3.8 to 3.7
# ---------------------------------------------------------------------------

class FakeSock:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, seconds: float) -> None:
        self.timeouts.append(seconds)


class PhasedStream:
    """An event stream whose blocks arrive after scripted delays; a delay above the armed socket timeout raises."""

    def __init__(self, blocks: list[tuple[float, str]]) -> None:
        self._blocks = blocks
        self.sock = FakeSock()
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.sock))
        self.status = 200

    def __iter__(self):
        for delay, block in self._blocks:
            armed = self.sock.timeouts[-1] if self.sock.timeouts else None
            if armed is not None and delay > armed:
                raise TimeoutError("timed out")
            for line in block.split("\n"):
                yield (line + "\n").encode("utf-8")
            yield b"\n"

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class PhasedNetwork(FakeNetwork):
    """Serves each Gemini model's stream from a script of (delay, block) pairs; remembers every stream served."""

    def __init__(self, scripts: dict[str, list[tuple[float, str]]]) -> None:
        super().__init__()
        self.scripts = scripts
        self.served: dict[str, list[PhasedStream]] = {}

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
        if GEMINI in request.full_url and "alt=sse" in request.full_url:
            model = request.full_url.split("/models/", 1)[1].split(":", 1)[0]
            stream = PhasedStream(list(self.scripts.get(model, [])))
            self.served.setdefault(model, []).append(stream)
            return stream
        return super().__call__(request, timeout)


def timed(pieces: list[str], delays: list[float], **kw) -> list[tuple[float, str]]:
    return list(zip(delays, gemini_chunks(pieces, **kw)))


def test_a_first_token_stall_is_bounded_and_the_pool_moves_to_the_next_model(tmp_path):
    net = PhasedNetwork({"gemini-3.8-flash": timed(["nie"], [25.0]),
                         "gemini-3.7-flash": timed(["Die kompetitive ", "Hemmung."], [2.0, 1.0])})
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001
    core.set_chat_mode("FREE")
    started = time.perf_counter()
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "Die kompetitive Hemmung." and message["backend"] == "gemini/gemini-3.7-flash"
    assert message["meta"]["completion"]["complete"] is True
    assert all(r["timeout"] == INTERACTIVE_STREAM_TIMEOUTS.connect for r in net.requests), "the connect bound reaches urlopen"
    stalled = net.served["gemini-3.8-flash"][0]
    assert stalled.sock.timeouts == [INTERACTIVE_STREAM_TIMEOUTS.first_token], "3.8 was armed for the first token and cut there"
    attempts = message["meta"]["provenance"]["route_attempts"]
    assert [(a["model"], a["failure_class"]) for a in attempts] == [("gemini-3.8-flash", "timeout"), ("gemini-3.7-flash", "ok")]
    assert local.calls == [] and time.perf_counter() - started < 10.0


def test_an_inter_chunk_stall_is_bounded_and_the_shown_text_is_kept_as_incomplete(tmp_path):
    net = PhasedNetwork({"gemini-3.8-flash": timed(["Bei der kompetitiven ", "Hemmung ", "konkurriert"], [1.0, 2.0, 40.0])})
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001
    core.set_chat_mode("FREE")
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "Bei der kompetitiven Hemmung" and message["backend"] == "gemini/gemini-3.8-flash"
    assert message["meta"]["completion"]["complete"] is False and message["meta"]["completion"]["finish_reason"] == "stream_interrupted:timeout"
    stream = net.served["gemini-3.8-flash"][0]
    assert stream.sock.timeouts == [INTERACTIVE_STREAM_TIMEOUTS.first_token, INTERACTIVE_STREAM_TIMEOUTS.idle, INTERACTIVE_STREAM_TIMEOUTS.idle]
    assert "gemini-3.7-flash" not in net.served, "text was already shown: no silent switch to another model"
    assert local.calls == []


def test_a_daily_quota_on_the_first_model_reaches_the_second_without_a_retry(tmp_path, cfg, creds):
    from test_model_gateway import GeminiPoolNetwork
    import urllib.error
    import io

    def quota() -> urllib.error.HTTPError:
        body = io.BytesIO(json.dumps({"error": {"code": 429, "message": "Quota exceeded for quota metric 'Generate Content API requests per day'"}}).encode())
        return urllib.error.HTTPError("https://" + GEMINI + "/x", 429, "Too Many Requests", hdrs=None, fp=body)

    net = GeminiPoolNetwork({"gemini-3.8-flash": [quota(), quota()], "gemini-3.7-flash": [gemini_reply("Antwort von 3.7")]})
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(GatewayRequest(prompt="Was ist Beta-Oxidation?", facts=TaskFacts(text="x", is_question=True), mode=ChatMode.FREE))
    assert reply.text == "Antwort von 3.7" and reply.model == "gemini-3.7-flash"
    assert [(a["model"], a["failure_class"], a["attempt"]) for a in reply.route_attempts] == \
        [("gemini-3.8-flash", "quota_exhausted", 1), ("gemini-3.7-flash", "ok", 1)]


def test_both_pool_models_unavailable_terminates_cleanly_and_quickly(tmp_path):
    net = PhasedNetwork({"gemini-3.8-flash": timed(["nie"], [25.0]), "gemini-3.7-flash": timed(["nie"], [25.0])})
    core, kernel, local, executed = make_world(tmp_path, net)
    kernel.gateway._retry_sleep = lambda _delay: None  # noqa: SLF001
    core.set_chat_mode("FREE")
    started = time.perf_counter()
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == FREE_TEXT_DE and message["backend"] == "intelligence"
    assert kernel.gateway.health.status("gemini") is ProviderStatus.TIMEOUT
    assert local.calls == [] and not any(OPENAI in r["url"] for r in net.requests)
    assert time.perf_counter() - started < 10.0


def test_normal_streaming_is_unaffected_and_carries_the_interactive_bounds(tmp_path):
    net = PhasedNetwork({"gemini-3.8-flash": timed(["Der Himmel ", "ist blau."], [1.0, 1.0])})
    core, kernel, local, executed = make_world(tmp_path, net)
    core.set_chat_mode("FREE")
    events = ask(core, "Warum ist der Himmel blau?", wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "Der Himmel ist blau." and message["meta"]["completion"]["complete"] is True
    tokens = [e for e in events if e.type is EventType.TOKEN]
    assert len(tokens) >= 2 and "".join(e.payload["text"] for e in tokens) == message["text"]
    assert local.calls == []


# ---------------------------------------------------------------------------
# 13: SMART and DEEP are untouched; deliberate long generations keep long bounds
# ---------------------------------------------------------------------------

def test_smart_and_deep_routes_and_long_generation_bounds_are_unchanged(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses[OPENAI] = openai_reply("Eine Antwort.")
    net.responses[GEMINI] = gemini_reply("frei")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    smart = gateway._prepare(GatewayRequest(prompt="Antworte kurz: Farbe von Chlorophyll?", facts=TaskFacts(text="x", is_question=True),  # noqa: SLF001
                                            mode=ChatMode.SMART))
    assert smart.decision.role == "reasoning.smart" and smart.timeouts == INTERACTIVE_STREAM_TIMEOUTS
    deep_text = "Plane für mich die nächsten sechs Wochen Lernplan ausführlich " * 8
    deep = gateway._prepare(GatewayRequest(prompt=deep_text, facts=TaskFacts(text=deep_text), mode=ChatMode.DEEP, owner_text=deep_text))  # noqa: SLF001
    assert deep.decision.role == "reasoning.deep"
    assert deep.timeouts == LONG_STREAM_TIMEOUTS, (deep.decision.thinking_level, deep.decision.output_budget)
    explicit = gateway._prepare(GatewayRequest(prompt="x", facts=TaskFacts(text="x", is_question=True), mode=ChatMode.FREE,  # noqa: SLF001
                                               timeouts=StreamTimeouts(connect=3.0, first_token=4.0, idle=5.0, total=6.0)))
    assert explicit.timeouts.to_dict() == {"connect": 3.0, "first_token": 4.0, "idle": 5.0, "total": 6.0}
    assert INTERACTIVE_STREAM_TIMEOUTS.to_dict() == {"connect": 10.0, "first_token": 20.0, "idle": 20.0, "total": 150.0}
