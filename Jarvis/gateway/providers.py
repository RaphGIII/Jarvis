"""Provider adapters: request shape in, text and usage out.

Each adapter knows one wire protocol and nothing else -- no credentials, no
HTTP, no policy.  It builds a body for :class:`gateway.transport.Transport`
and reads the reply into a :class:`ProviderReply` with normalised usage
counts, so the budget can settle the same way whoever answered.

Adding a provider is a new class here, registered in :func:`adapter_for`, and
a new ``kind`` in the configuration.  Nothing above this module changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from gateway.config import ProviderConfig, RoleBinding
from gateway.health import GatewayError, ProviderStatus, classify_http
from gateway.transport import Ticket, Transport

#: A streaming adapter yields ``{"text": piece}`` as the answer arrives and,
#: last, ``{"reply": ProviderReply}`` with the usage and the finish reason.
StreamEvent = dict[str, Any]


@dataclass
class ProviderRequest:
    system: str
    prompt: str
    max_output_tokens: int
    temperature: float
    #: Provider-specific thinking setting, already resolved from the role binding.
    thinking: Any = None
    #: JSON schema when structured output is wanted.
    schema: dict[str, Any] | None = None
    #: The abstract level the setting came from (FAST / NORMAL / DEEP / MAX), for the record.
    thinking_level: str = ""
    #: A bound on this one call, in seconds; None = the provider's configured timeout.
    timeout_seconds: float | None = None


@dataclass
class ProviderReply:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0
    finish_reason: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"usage": dict(self.usage), "latency_seconds": round(self.latency_seconds, 3),
                "finish_reason": self.finish_reason, "model": self.model, "chars": len(self.text)}


_SCHEMA_KEEP = {"type", "properties", "required", "items", "enum", "description", "nullable", "format", "minimum", "maximum",
                "minItems", "maxItems"}


def gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini accepts an OpenAPI subset; strip what it rejects (additionalProperties, $schema, ...)."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key not in _SCHEMA_KEEP:
                    continue
                if key == "properties" and isinstance(value, dict):
                    out[key] = {name: walk(sub) for name, sub in value.items()}
                elif key == "items":
                    out[key] = walk(value)
                elif key == "type" and isinstance(value, list):
                    non_null = [v for v in value if v != "null"]
                    out[key] = non_null[0] if non_null else "string"
                    if "null" in value:
                        out["nullable"] = True
                else:
                    out[key] = value
            return out
        return node

    return walk(schema)


class GeminiAdapter:
    kind = "gemini"
    auth = "x-goog-api-key"

    def _payload(self, provider: ProviderConfig, binding: RoleBinding, request: ProviderRequest) -> tuple[str, dict[str, Any]]:
        url = f"{provider.base_url.rstrip('/')}/v1beta/models/{binding.model}:generateContent"
        generation: dict[str, Any] = {"temperature": request.temperature, "maxOutputTokens": request.max_output_tokens}
        if request.schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = gemini_schema(request.schema)
        if request.thinking is not None:
            # Two generations of thinking control: a token budget (2.x) or a
            # named level (3.x).  Configuration says which; an unusable value
            # is dropped rather than sent.
            if isinstance(request.thinking, bool):
                pass
            elif isinstance(request.thinking, (int, float)):
                generation["thinkingConfig"] = {"thinkingBudget": int(request.thinking)}
            elif isinstance(request.thinking, str) and request.thinking.strip():
                value = request.thinking.strip().lower()
                if value.lstrip("-").isdigit():
                    generation["thinkingConfig"] = {"thinkingBudget": int(value)}
                else:
                    generation["thinkingConfig"] = {"thinkingLevel": value}
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": generation,
        }
        if request.system:
            body["system_instruction"] = {"parts": [{"text": request.system}]}
        return url, body

    @staticmethod
    def _usage(usage_raw: dict[str, Any]) -> dict[str, int]:
        return {
            "input_tokens": int(usage_raw.get("promptTokenCount", 0) or 0),
            "cached_input_tokens": int(usage_raw.get("cachedContentTokenCount", 0) or 0),
            "output_tokens": int(usage_raw.get("candidatesTokenCount", 0) or 0) + int(usage_raw.get("thoughtsTokenCount", 0) or 0),
        }

    def stream(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
               request: ProviderRequest) -> Iterator[StreamEvent]:
        url, body = self._payload(provider, binding, request)
        url = url.replace(":generateContent", ":streamGenerateContent") + "?alt=sse"
        usage_raw: dict[str, Any] = {}
        finish = ""
        for _event, data in transport.post_sse(ticket, provider, url, body, auth=self.auth, timeout=request.timeout_seconds):
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                err = chunk["error"] if isinstance(chunk["error"], dict) else {"message": str(chunk["error"])}
                code = int(err.get("code", 0) or 0)
                raise GatewayError(classify_http(code or 500, json.dumps(err), provider_kind=provider.kind),
                                   f"stream error: {str(err.get('message', err))[:300]}", role=ticket.role, provider=provider.name,
                                   http_status=code or None)
            candidates = chunk.get("candidates") or []
            if candidates:
                candidate = candidates[0]
                parts = (candidate.get("content") or {}).get("parts") or []
                text = "".join(str(part.get("text", "")) for part in parts if not part.get("thought"))
                if text:
                    yield {"text": text}
                if candidate.get("finishReason"):
                    finish = str(candidate["finishReason"])
            elif (chunk.get("promptFeedback") or {}).get("blockReason"):
                raise GatewayError(ProviderStatus.TASK_FAILURE, f"blocked: {chunk['promptFeedback']['blockReason']}", role=ticket.role,
                                   provider=provider.name)
            if chunk.get("usageMetadata"):
                usage_raw = chunk["usageMetadata"]
        yield {"reply": ProviderReply(text="", usage=self._usage(usage_raw), finish_reason=finish, model=binding.model)}

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        url, body = self._payload(provider, binding, request)
        reply = transport.post_json(ticket, provider, url, body, auth=self.auth, timeout=request.timeout_seconds)
        data = reply.data
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason", "")
            raise GatewayError(ProviderStatus.TASK_FAILURE, f"no candidates{': ' + block if block else ''}", role=ticket.role,
                               provider=provider.name)
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if not part.get("thought"))
        usage = self._usage(data.get("usageMetadata") or {})
        return ProviderReply(text=text, usage=usage, latency_seconds=reply.latency_seconds,
                             finish_reason=str(candidates[0].get("finishReason", "")), model=binding.model)


OPENAI_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


class OpenAIAdapter:
    """The Responses API: reasoning effort, structured output, cached-token usage."""

    kind = "openai"
    auth = "bearer"

    def _payload(self, provider: ProviderConfig, binding: RoleBinding, request: ProviderRequest) -> tuple[str, dict[str, Any]]:
        url = f"{provider.base_url.rstrip('/')}/v1/responses"
        body: dict[str, Any] = {
            "model": binding.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": request.prompt}]}],
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if request.system:
            body["instructions"] = request.system
        effort = str(request.thinking or "").strip().lower()
        if effort in OPENAI_EFFORTS:
            # Reasoning models take an effort level and reject a temperature.
            body["reasoning"] = {"effort": effort}
        else:
            body["temperature"] = request.temperature
        if request.schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": "zeus_response", "schema": request.schema, "strict": False}}
        return url, body

    @staticmethod
    def _usage(usage_raw: dict[str, Any]) -> dict[str, int]:
        in_details = usage_raw.get("input_tokens_details") or {}
        out_details = usage_raw.get("output_tokens_details") or {}
        return {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "cached_input_tokens": int(in_details.get("cached_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "reasoning_tokens": int(out_details.get("reasoning_tokens", 0) or 0),
        }

    def stream(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
               request: ProviderRequest) -> Iterator[StreamEvent]:
        url, body = self._payload(provider, binding, request)
        body["stream"] = True
        final: dict[str, Any] = {}
        for event, data in transport.post_sse(ticket, provider, url, body, auth=self.auth, timeout=request.timeout_seconds):
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            kind = str(payload.get("type") or event or "")
            if kind == "response.output_text.delta":
                delta = str(payload.get("delta") or "")
                if delta:
                    yield {"text": delta}
            elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                final = payload.get("response") or {}
                if kind == "response.failed":
                    message = str(((final.get("error") or {}).get("message")) or "response failed")
                    raise GatewayError(ProviderStatus.TASK_FAILURE, message[:300], role=ticket.role, provider=provider.name)
            elif "output" in payload and not kind.startswith("response."):
                # A completed Responses object in one piece (a provider that
                # did not stream this operation): its text, then its usage.
                final = payload
                text = str(payload.get("output_text") or "")
                if not text:
                    for item in payload.get("output") or []:
                        if isinstance(item, dict) and item.get("type") == "message":
                            for part in item.get("content") or []:
                                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                                    text += str(part.get("text", ""))
                if text:
                    yield {"text": text}
            elif kind == "error":
                raise GatewayError(ProviderStatus.TASK_FAILURE, str(payload.get("message") or payload.get("error") or "stream error")[:300],
                                   role=ticket.role, provider=provider.name)
        incomplete = final.get("incomplete_details") if isinstance(final.get("incomplete_details"), dict) else {}
        finish = str((incomplete or {}).get("reason") or final.get("status") or "completed")
        yield {"reply": ProviderReply(text="", usage=self._usage(final.get("usage") or {}), finish_reason=finish,
                                      model=str(final.get("model") or binding.model))}

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        url, body = self._payload(provider, binding, request)
        reply = transport.post_json(ticket, provider, url, body, auth=self.auth, timeout=request.timeout_seconds)
        data = reply.data
        if data.get("error"):
            raise GatewayError(ProviderStatus.TASK_FAILURE, str(data["error"])[:300], role=ticket.role, provider=provider.name)
        text = str(data.get("output_text") or "")
        if not text:
            pieces: list[str] = []
            for item in data.get("output") or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        pieces.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") == "refusal":
                        raise GatewayError(ProviderStatus.TASK_FAILURE, f"refusal: {str(part.get('refusal', ''))[:200]}",
                                           role=ticket.role, provider=provider.name)
            text = "".join(pieces)
        status = str(data.get("status") or "")
        if not text and status not in {"completed", ""}:
            detail = (data.get("incomplete_details") or {}).get("reason", "") if isinstance(data.get("incomplete_details"), dict) else ""
            raise GatewayError(ProviderStatus.TASK_FAILURE, f"response {status}{': ' + detail if detail else ''}", role=ticket.role,
                               provider=provider.name)
        usage = self._usage(data.get("usage") or {})
        incomplete = data.get("incomplete_details") if isinstance(data.get("incomplete_details"), dict) else {}
        finish = str((incomplete or {}).get("reason") or status or "")
        return ProviderReply(text=text, usage=usage, latency_seconds=reply.latency_seconds, finish_reason=finish,
                             model=str(data.get("model") or binding.model))


class OpenAICompatibleAdapter(OpenAIAdapter):
    kind = "openai_compatible"

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        # Self-hosted servers usually predate max_completion_tokens; keep the
        # classic field and let temperature through.
        url = f"{provider.base_url.rstrip('/')}/v1/chat/completions"
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        body: dict[str, Any] = {"model": binding.model, "messages": messages, "max_tokens": request.max_output_tokens,
                                "temperature": request.temperature}
        if request.schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "zeus_response", "schema": request.schema}}
        reply = transport.post_json(ticket, provider, url, body, auth=self.auth, timeout=request.timeout_seconds)
        data = reply.data
        try:
            choice = data["choices"][0]
            text = str(choice["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise GatewayError(ProviderStatus.TASK_FAILURE, "malformed completion", role=ticket.role, provider=provider.name) from None
        usage_raw = data.get("usage") or {}
        usage = {"input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0), "cached_input_tokens": 0,
                 "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0)}
        return ProviderReply(text=text, usage=usage, latency_seconds=reply.latency_seconds,
                             finish_reason=str(choice.get("finish_reason", "")), model=binding.model)


class AnthropicAdapter:
    kind = "anthropic"
    auth = "x-api-key"

    def _payload(self, provider: ProviderConfig, binding: RoleBinding, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        url = f"{provider.base_url.rstrip('/')}/v1/messages"
        prompt = request.prompt
        if request.schema is not None:
            prompt = (f"{prompt}\n\nAnswer with one JSON object only, matching this JSON schema, no prose:\n"
                      f"{json.dumps(request.schema)}")
        body: dict[str, Any] = {"model": binding.model, "max_tokens": request.max_output_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if request.system:
            body["system"] = request.system
        budget = None
        if isinstance(request.thinking, (int, float)) and not isinstance(request.thinking, bool):
            budget = int(request.thinking)
        elif isinstance(request.thinking, str) and request.thinking.strip().isdigit():
            budget = int(request.thinking.strip())
        if budget and budget >= 1024:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            if body["max_tokens"] <= budget:
                body["max_tokens"] = budget + request.max_output_tokens
        else:
            body["temperature"] = request.temperature
        headers = {"anthropic-version": "2023-06-01"}
        return url, body, headers

    def stream(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
               request: ProviderRequest) -> Iterator[StreamEvent]:
        url, body, headers = self._payload(provider, binding, request)
        body["stream"] = True
        input_tokens = cached = output_tokens = 0
        stop = ""
        model = binding.model
        for event, data in transport.post_sse(ticket, provider, url, body, headers=headers, auth=self.auth, timeout=request.timeout_seconds):
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            kind = str(payload.get("type") or event or "")
            if kind == "message_start":
                message = payload.get("message") or {}
                usage = message.get("usage") or {}
                input_tokens = (int(usage.get("input_tokens", 0) or 0) + int(usage.get("cache_read_input_tokens", 0) or 0)
                                + int(usage.get("cache_creation_input_tokens", 0) or 0))
                cached = int(usage.get("cache_read_input_tokens", 0) or 0)
                model = str(message.get("model") or model)
            elif kind == "content_block_delta":
                delta = payload.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield {"text": str(delta["text"])}
            elif kind == "message_delta":
                stop = str((payload.get("delta") or {}).get("stop_reason") or stop)
                usage = payload.get("usage") or {}
                output_tokens = int(usage.get("output_tokens", output_tokens) or output_tokens)
            elif kind == "error":
                err = payload.get("error") or {}
                raise GatewayError(ProviderStatus.TASK_FAILURE, str(err.get("message") if isinstance(err, dict) else err)[:300],
                                   role=ticket.role, provider=provider.name)
            elif "content" in payload and isinstance(payload.get("content"), list):
                # A completed message in one piece.
                text = "".join(str(block.get("text", "")) for block in payload["content"] if isinstance(block, dict) and block.get("type") == "text")
                usage = payload.get("usage") or {}
                input_tokens = (int(usage.get("input_tokens", 0) or 0) + int(usage.get("cache_read_input_tokens", 0) or 0)
                                + int(usage.get("cache_creation_input_tokens", 0) or 0))
                cached = int(usage.get("cache_read_input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                stop = str(payload.get("stop_reason") or stop)
                model = str(payload.get("model") or model)
                if text:
                    yield {"text": text}
        yield {"reply": ProviderReply(text="", usage={"input_tokens": input_tokens, "cached_input_tokens": cached, "output_tokens": output_tokens},
                                      finish_reason=stop, model=model)}

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        url, body, headers = self._payload(provider, binding, request)
        reply = transport.post_json(ticket, provider, url, body, headers=headers, auth=self.auth, timeout=request.timeout_seconds)
        data = reply.data
        blocks = data.get("content") or []
        text = "".join(str(block.get("text", "")) for block in blocks if block.get("type") == "text")
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens", 0) or 0) + int(usage_raw.get("cache_read_input_tokens", 0) or 0)
                            + int(usage_raw.get("cache_creation_input_tokens", 0) or 0),
            "cached_input_tokens": int(usage_raw.get("cache_read_input_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        }
        return ProviderReply(text=text, usage=usage, latency_seconds=reply.latency_seconds,
                             finish_reason=str(data.get("stop_reason", "")), model=str(data.get("model") or binding.model))


_ADAPTERS = {
    "gemini": GeminiAdapter(),
    "openai": OpenAIAdapter(),
    "openai_compatible": OpenAICompatibleAdapter(),
    "anthropic": AnthropicAdapter(),
}


def adapter_for(kind: str):
    try:
        return _ADAPTERS[str(kind).strip().lower()]
    except KeyError:
        raise ValueError(f"no gateway adapter for provider kind {kind!r}") from None
