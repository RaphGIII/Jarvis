"""True streaming, output budgets, completeness, and stream-safe delivery -- provider-independent.

The fake providers here speak Server-Sent Events the way the real ones do
(Gemini ``alt=sse`` chunks, OpenAI Responses events, Anthropic message
events).  A hook fires when each chunk is *served*, so a test can prove the
client already had the earlier chunks before the provider produced the next.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from gateway.config import GatewayConfig
from gateway.gateway import GatewayRequest, ModelGateway
from gateway.modes import ChatMode
from gateway.output_budget import LEVELS, decide_output_budget
from gateway.secrets import CredentialStore
from gateway.task import TaskFacts
from service.events import EventType
from service.scitext import StreamNormalizer, normalize
from test_gateway_integration import answer_text, ask
from test_intelligence_flow import make_world
from test_model_gateway import FakeNetwork, FakeResponse, gemini_reply, make_gateway, openai_reply

QUESTION = "Warum ist der Himmel blau?"


# --------------------------------------------------------------------------
# SSE fakes
# --------------------------------------------------------------------------

class SSEResponse:
    """An event-stream body served block by block; ``on_serve(index)`` fires before each block."""

    def __init__(self, blocks: list[str], on_serve: Callable[[int], None] | None = None) -> None:
        self._blocks = blocks
        self._on_serve = on_serve
        self.status = 200
        self.closed = False

    def __iter__(self):
        for index, block in enumerate(self._blocks):
            if self._on_serve is not None:
                self._on_serve(index)
            for line in block.split("\n"):
                yield (line + "\n").encode("utf-8")
            yield b"\n"

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def gemini_chunks(pieces: list[str], *, finish: str = "STOP", prompt_tokens: int = 100, out_tokens: int = 50) -> list[str]:
    blocks = []
    for k, piece in enumerate(pieces):
        payload: dict[str, Any] = {"candidates": [{"content": {"parts": [{"text": piece}]}}]}
        if k == len(pieces) - 1:
            payload["candidates"][0]["finishReason"] = finish
            payload["usageMetadata"] = {"promptTokenCount": prompt_tokens, "candidatesTokenCount": out_tokens}
        blocks.append("data: " + json.dumps(payload, ensure_ascii=False))
    return blocks


def openai_events(pieces: list[str], *, status: str = "completed", prompt_tokens: int = 800, out_tokens: int = 120, cached: int = 100) -> list[str]:
    blocks = ["event: response.created\ndata: " + json.dumps({"type": "response.created", "response": {"id": "r"}})]
    for piece in pieces:
        blocks.append("event: response.output_text.delta\ndata: " + json.dumps({"type": "response.output_text.delta", "delta": piece}, ensure_ascii=False))
    final = {"id": "r", "status": status, "model": "gpt-x", "usage": {"input_tokens": prompt_tokens, "input_tokens_details": {"cached_tokens": cached},
                                                                      "output_tokens": out_tokens, "output_tokens_details": {"reasoning_tokens": 0}}}
    if status == "incomplete":
        final["incomplete_details"] = {"reason": "max_output_tokens"}
    kind = "response.completed" if status == "completed" else "response.incomplete"
    blocks.append(f"event: {kind}\ndata: " + json.dumps({"type": kind, "response": final}))
    return blocks


def anthropic_events(pieces: list[str], *, stop: str = "end_turn", input_tokens: int = 500, out_tokens: int = 80) -> list[str]:
    blocks = ["event: message_start\ndata: " + json.dumps({"type": "message_start", "message": {"model": "claude-x", "usage": {"input_tokens": input_tokens}}})]
    for piece in pieces:
        blocks.append("event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": piece}}, ensure_ascii=False))
    blocks.append("event: message_delta\ndata: " + json.dumps({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": out_tokens}}))
    blocks.append("event: message_stop\ndata: " + json.dumps({"type": "message_stop"}))
    return blocks


class SSENetwork(FakeNetwork):
    """Serves an event stream for streaming endpoints and JSON otherwise."""

    def __init__(self) -> None:
        super().__init__()
        self.streams: dict[str, Any] = {}
        self.on_serve: Callable[[int], None] | None = None
        self.served: list[SSEResponse] = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8")) if request.data else {}
        host = request.full_url.split("/")[2]
        streaming = "alt=sse" in request.full_url or bool(body.get("stream"))
        if streaming and host in self.streams:
            self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body, "timeout": timeout})
            blocks = self.streams[host]
            blocks = blocks(request) if callable(blocks) else blocks
            if isinstance(blocks, Exception):
                raise blocks
            response = SSEResponse(list(blocks), on_serve=self.on_serve)
            self.served.append(response)
            return response
        return super().__call__(request, timeout)


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


def streamed_world(tmp_path: Path, pieces: list[str], **kw):
    net = SSENetwork()
    net.streams["generativelanguage.googleapis.com"] = gemini_chunks(pieces, **kw)
    core, kernel, local, executed = make_world(tmp_path, net)
    return net, core, kernel, local


# --------------------------------------------------------------------------
# 1 + 2 + 14: chunks reach the client while the provider is still producing; the record is the stream
# --------------------------------------------------------------------------

def test_first_chunks_reach_the_client_before_generation_completes(tmp_path):
    pieces = ["Der Himmel ", "ist blau, weil ", "kurzwelliges Licht ", "stärker gestreut wird."]
    net, core, kernel, local = streamed_world(tmp_path, pieces)
    seen_before: dict[int, int] = {}
    with core.bus.subscribe(replay=False) as watcher:
        def on_serve(index: int) -> None:
            tokens = [e for e in watcher.drain() if e.type is EventType.TOKEN]
            seen_before[index] = seen_before.get(index - 1, 0) + len(tokens)

        net.on_serve = on_serve
        events = ask(core, QUESTION, wait=30)
    assert seen_before[0] == 0 and seen_before[2] >= 2, seen_before  # by the third chunk, two had reached the client
    assert seen_before[3] >= 3
    assert answer_text(events) == "".join(pieces)
    assert net.requests[-1]["url"].endswith(":streamGenerateContent?alt=sse")


def test_the_stored_answer_is_exactly_the_completed_stream(tmp_path):
    pieces = ["Bikarbonat \\ce{HC", "O3-} puffert; Ca^", "2+ fällt &am", "p; CO₂ steigt.\n\n```py\nx &amp; y\n```"]
    net, core, kernel, local = streamed_world(tmp_path, pieces)
    events = ask(core, QUESTION, wait=30)
    tokens = "".join(e.payload["text"] for e in events if e.type is EventType.TOKEN)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == tokens, "what was shown is what was stored"
    assert message["text"] == normalize("".join(pieces)), "and it is the normalised stream, chunk boundaries notwithstanding"
    assert "HCO₃⁻" in message["text"] and "Ca^2+ fällt & CO₂" in message["text"]
    assert "x &amp; y" in message["text"], "fenced code is not touched"
    assert message["backend"] == "gemini/gemini-3.8-flash" and message["meta"]["provenance"]["streamed"] is True
    assert message["meta"]["completion"]["truncated"] is False and message["meta"]["completion"]["finish_reason"] == "STOP"


def test_the_stream_normaliser_matches_the_whole_text_normalisation():
    text = "Puffer: \\ce{H2CO3} ⇌ \\ce{HCO3-} + \\ce{H+}; \\\\Delta G &lt; 0 � ```a &amp; b``` und &amp;rarr; Ende"
    for size in (1, 3, 7, 16, 40):
        norm = StreamNormalizer()
        parts = [norm.feed(text[i:i + size]) for i in range(0, len(text), size)] + [norm.finish()]
        assert "".join(parts) == normalize(text), size


# --------------------------------------------------------------------------
# 3-8 are the renderer's: run its deterministic checks under node when available
# --------------------------------------------------------------------------

def test_the_renderer_checks_pass_under_node():
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, str(root / "ui" / "tests" / "render.test.mjs")], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0 and "ALL OK" in completed.stdout, completed.stdout[-2000:] + completed.stderr[-500:]


# --------------------------------------------------------------------------
# 9-12: the output budget, decided before the call, estimated, clipped
# --------------------------------------------------------------------------

def test_a_brief_request_gets_a_lower_budget_than_a_detailed_medical_one():
    brief = decide_output_budget("Erkläre mir kurz, warum Hyperventilation zu einer Tetanie führt.")
    detailed = decide_output_budget("Erkläre detailliert und Schritt für Schritt die Pathophysiologie der hyperventilationsbedingten Tetanie.")
    default = decide_output_budget("Warum führt Hyperventilation zu einer Tetanie?")
    deepest = decide_output_budget("Erkläre maximal ausführlich die Pathophysiologie der Tetanie.")
    assert brief.level == "brief" and brief.tokens == LEVELS["brief"][1]
    assert detailed.level == "detailed" and detailed.tokens == LEVELS["detailed"][1]
    assert default.level == "normal" and brief.tokens < default.tokens < detailed.tokens < deepest.tokens
    assert deepest.level == "deep"
    structured = decide_output_budget("Erkläre maximal ausführlich x", structured=True)
    assert structured.level == "deep", "the owner's words outrank the structured default"
    assert decide_output_budget("Klassifiziere das", structured=True).level == "brief"


def test_the_budget_reaches_the_provider_and_the_estimate(tmp_path, cfg, creds):
    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("ok")
    net.responses["api.openai.com"] = openai_reply("ok")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    gateway.complete(GatewayRequest(prompt="transcript...\nOwner: Erkläre mir kurz die Enzymhemmung.", facts=TaskFacts(text="x", is_question=True),
                                    owner_text="Erkläre mir kurz die Enzymhemmung."))
    assert net.requests[-1]["body"]["generationConfig"]["maxOutputTokens"] == LEVELS["brief"][1]
    reply = gateway.complete(GatewayRequest(prompt="Erkläre detailliert die Atmungskette.", facts=TaskFacts(text="x", is_question=True),
                                            owner_text="Erkläre detailliert die Atmungskette."))
    assert net.requests[-1]["body"]["generationConfig"]["maxOutputTokens"] == LEVELS["detailed"][1]
    assert reply.output_budget["level"] == "detailed" and reply.max_output_tokens == LEVELS["detailed"][1]
    # A metered role: the estimate is computed on the chosen budget, before the call.
    text = "Plane detailliert die nächsten sechs Wochen Lernplan " * 6
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP, owner_text=text))
    assert reply.role == "reasoning.deep"
    assert reply.decision.estimate.output_tokens == reply.max_output_tokens == LEVELS[reply.output_budget["level"]][1]
    assert net.requests[-1]["body"]["max_output_tokens"] == reply.max_output_tokens
    assert reply.estimated_eur > 0.0 and reply.decision.output_budget["level"] in {"detailed", "deep"}


def test_provider_hard_limits_are_never_exceeded(tmp_path, creds):
    document = json.loads(json.dumps(GatewayConfig.defaults().to_dict()))
    document["providers"]["gemini"]["enabled"] = True
    document["providers"]["gemini"]["options"] = {"output_hard_limit": 1500}
    from gateway.config import _parse

    cfg = _parse(document, source="test")
    net = FakeNetwork()
    net.responses["generativelanguage.googleapis.com"] = gemini_reply("ok")
    gateway = make_gateway(tmp_path, cfg, creds, net)
    reply = gateway.complete(GatewayRequest(prompt="Erkläre maximal ausführlich alles.", facts=TaskFacts(text="x", is_question=True),
                                            owner_text="Erkläre maximal ausführlich alles."))
    assert net.requests[-1]["body"]["generationConfig"]["maxOutputTokens"] == 1500 and reply.provider_hard_limit == 1500
    gateway.complete(GatewayRequest(prompt="x", facts=TaskFacts(text="x", is_question=True), max_output_tokens=9000))
    assert net.requests[-1]["body"]["generationConfig"]["maxOutputTokens"] == 1500, "an explicit request is clipped too"


def test_a_paid_budget_that_would_not_fit_is_reduced_before_the_call_never_silently_sent(tmp_path, cfg, creds):
    from dataclasses import replace

    tight = replace(cfg, budget=replace(cfg.budget, per_task_hard_cap=0.05))
    net = FakeNetwork()
    net.responses["api.openai.com"] = openai_reply("ok")
    gateway = make_gateway(tmp_path, tight.with_provider_enabled("gemini", False), creds, net)
    text = "Erkläre maximal ausführlich die Herzphysiologie."
    reply = gateway.complete(GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_question=True), mode=ChatMode.DEEP, owner_text=text))
    assert reply.output_budget["reduced"] is True and reply.output_budget["requested_level"] == "deep"
    assert reply.output_budget["level"] == "brief" and reply.max_output_tokens == LEVELS["brief"][1]
    assert net.requests[-1]["body"]["max_output_tokens"] == LEVELS["brief"][1]
    assert gateway.governor.summary().month == pytest.approx(reply.actual_eur)


# --------------------------------------------------------------------------
# 13: truncation is detected and reported, never presented as complete
# --------------------------------------------------------------------------

def test_a_ceiling_truncated_stream_is_reported_and_continuation_offered(tmp_path):
    net, core, kernel, local = streamed_world(tmp_path, ["Die Antwort beginnt und ", "endet mitten im"], finish="MAX_TOKENS", out_tokens=4000)
    events = ask(core, QUESTION, wait=30)
    message = next(e.payload for e in events if e.type is EventType.MESSAGE)
    assert message["text"] == "Die Antwort beginnt und endet mitten im"
    completion = message["meta"]["completion"]
    assert completion["truncated"] is True and completion["finish_reason"] == "MAX_TOKENS"
    assert completion["output_tokens"] == 4000 and completion["configured_output_budget"] == LEVELS["normal"][1]
    assert completion["provider_hard_limit"] == 65536
    assert any("truncated at the output ceiling" in str(e.payload.get("summary", "")) for e in events if e.type is EventType.TOOL)


def test_openai_and_anthropic_streams_carry_text_usage_cost_and_stop_reasons(tmp_path, cfg, creds):
    net = SSENetwork()
    net.streams["api.openai.com"] = openai_events(["Erste ", "Antwort."], status="incomplete", out_tokens=300)
    net.streams["api.anthropic.com"] = anthropic_events(["diff ", "here"], stop="max_tokens")
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    stream = gateway.stream(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP, owner_text=text))
    pieces = list(stream)
    reply = stream.reply
    assert pieces == ["Erste ", "Antwort."] and reply.text == "Erste Antwort." and reply.role == "reasoning.deep"
    assert reply.usage["input_tokens"] == 800 and reply.usage["cached_input_tokens"] == 100 and reply.usage["output_tokens"] == 300
    assert reply.truncated is True and reply.finish_reason == "max_output_tokens" and reply.actual_eur > 0.0
    assert net.requests[-1]["body"]["stream"] is True
    assert gateway.governor.summary().month == pytest.approx(reply.actual_eur), "settled from the stream's usage"

    stream = gateway.stream(GatewayRequest(prompt=text, facts=TaskFacts(text=text, is_engineering=True, new_subsystem=True, subsystems=5),
                                           mode=ChatMode.BUILD, purpose="engineer", role="engineer.frontier"))
    pieces = list(stream)
    reply = stream.reply
    assert pieces == ["diff ", "here"] and reply.role == "engineer.frontier" and reply.model == "claude-x"
    assert reply.usage == {"input_tokens": 500, "cached_input_tokens": 0, "output_tokens": 80}
    assert reply.truncated is True and reply.finish_reason == "max_tokens"


# --------------------------------------------------------------------------
# cancellation: the right request only, and the money accounted conservatively
# --------------------------------------------------------------------------

def test_closing_a_paid_stream_early_settles_at_the_estimate_and_frees_nothing_silently(tmp_path, cfg, creds):
    net = SSENetwork()
    net.streams["api.openai.com"] = openai_events(["eins ", "zwei ", "drei"], out_tokens=3)
    gateway = make_gateway(tmp_path, cfg.with_provider_enabled("gemini", False), creds, net)
    text = "Plane für mich die nächsten sechs Wochen Lernplan " * 8
    stream = gateway.stream(GatewayRequest(prompt=text, facts=TaskFacts(text=text), mode=ChatMode.DEEP, owner_text=text, task_id="cancel-1"))
    iterator = iter(stream)
    first = next(iterator)
    iterator.close()
    assert first == "eins " and stream.aborted is True and stream.reply is None
    assert net.served[-1].closed, "the connection was closed"
    rows = gateway.governor.history()
    assert rows[-1]["kind"] == "settle" and rows[-1]["actual_eur"] == pytest.approx(rows[-2]["estimated_eur"])
    assert "aborted" in json.dumps(rows[-1])
    assert gateway.governor.summary().open_reservations == 0


# --------------------------------------------------------------------------
# 15: streaming does not reintroduce the false-wake / stale-stop defect
# --------------------------------------------------------------------------

def test_a_false_wake_during_a_streamed_answer_does_not_destroy_it(tmp_path):
    pieces = ["Der Himmel ", "ist blau, ", "weil Rayleigh-Streuung."]
    net, core, kernel, local = streamed_world(tmp_path, pieces)
    interrupts: list[dict] = []
    net.on_serve = lambda index: interrupts.append(core.voice_interrupt(session="vs-false", wake=0.92)) if index == 1 else None
    core._request_stop()  # and a stale stop from an earlier generation, still set
    events = ask(core, QUESTION, wait=30)
    assert interrupts and interrupts[0]["interrupted"] == [] and interrupts[0].get("protected") == ["answer"]
    assert answer_text(events) == "".join(pieces)
    assert not [e for e in events if e.type is EventType.NOTIFICATION and e.payload.get("kind") == "barge_in"]


def test_the_owners_stop_cancels_only_the_current_stream(tmp_path):
    pieces = ["Der Himmel ", "ist blau, ", "weil Rayleigh-Streuung."]
    net, core, kernel, local = streamed_world(tmp_path, pieces)
    net.on_serve = lambda index: core.stop_current(reason="owner") if index == 1 else None
    events = ask(core, QUESTION, wait=30)
    assert answer_text(events) != "".join(pieces)
    net.on_serve = None
    events = ask(core, QUESTION, wait=30)
    assert answer_text(events) == "".join(pieces), "the next request is a new generation; the old stop does not apply"
