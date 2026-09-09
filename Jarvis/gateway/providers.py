"""Provider adapters: request shape in, text and usage out.

Each adapter knows one wire protocol and nothing else -- no credentials, no
HTTP, no policy.  It builds a body for :class:`gateway.transport.Transport`
and reads the reply into a :class:`ProviderReply` with normalised usage
counts, so the budget can settle the same way whoever answered.

Adding a provider is a new class here, registered in :func:`adapter_for`, and
a new ``kind`` in the configuration.  Nothing above this module changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gateway.config import ProviderConfig, RoleBinding
from gateway.health import GatewayError, ProviderStatus
from gateway.transport import Ticket, Transport


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

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        url = f"{provider.base_url.rstrip('/')}/v1beta/models/{binding.model}:generateContent"
        generation: dict[str, Any] = {"temperature": request.temperature, "maxOutputTokens": request.max_output_tokens}
        if request.schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = gemini_schema(request.schema)
        if request.thinking is not None:
            try:
                generation["thinkingConfig"] = {"thinkingBudget": int(request.thinking)}
            except (TypeError, ValueError):
                pass
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": generation,
        }
        if request.system:
            body["system_instruction"] = {"parts": [{"text": request.system}]}
        reply = transport.post_json(ticket, provider, url, body, auth=self.auth)
        data = reply.data
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason", "")
            raise GatewayError(ProviderStatus.TASK_FAILURE, f"no candidates{': ' + block if block else ''}", role=ticket.role,
                               provider=provider.name)
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if not part.get("thought"))
        usage_raw = data.get("usageMetadata") or {}
        usage = {
            "input_tokens": int(usage_raw.get("promptTokenCount", 0) or 0),
            "cached_input_tokens": int(usage_raw.get("cachedContentTokenCount", 0) or 0),
            "output_tokens": int(usage_raw.get("candidatesTokenCount", 0) or 0) + int(usage_raw.get("thoughtsTokenCount", 0) or 0),
        }
        return ProviderReply(text=text, usage=usage, latency_seconds=reply.latency_seconds,
                             finish_reason=str(candidates[0].get("finishReason", "")), model=binding.model)


class OpenAIAdapter:
    kind = "openai"
    auth = "bearer"

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        url = f"{provider.base_url.rstrip('/')}/v1/chat/completions"
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        body: dict[str, Any] = {"model": binding.model, "messages": messages, "max_completion_tokens": request.max_output_tokens}
        if request.thinking:
            # Reasoning models take an effort level and reject a temperature.
            body["reasoning_effort"] = str(request.thinking)
        else:
            body["temperature"] = request.temperature
        if request.schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "zeus_response", "schema": request.schema}}
        reply = transport.post_json(ticket, provider, url, body, auth=self.auth)
        data = reply.data
        try:
            choice = data["choices"][0]
            text = str(choice["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise GatewayError(ProviderStatus.TASK_FAILURE, "malformed completion", role=ticket.role, provider=provider.name) from None
        usage_raw = data.get("usage") or {}
        details = usage_raw.get("prompt_tokens_details") or {}
        usage = {
            "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "cached_input_tokens": int(details.get("cached_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
        }
        return ProviderReply(text=text, usage=usage, latency_seconds=reply.latency_seconds,
                             finish_reason=str(choice.get("finish_reason", "")), model=str(data.get("model") or binding.model))


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
        reply = transport.post_json(ticket, provider, url, body, auth=self.auth)
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

    def call(self, transport: Transport, ticket: Ticket, provider: ProviderConfig, binding: RoleBinding,
             request: ProviderRequest) -> ProviderReply:
        url = f"{provider.base_url.rstrip('/')}/v1/messages"
        prompt = request.prompt
        if request.schema is not None:
            import json as _json

            prompt = (f"{prompt}\n\nAnswer with one JSON object only, matching this JSON schema, no prose:\n"
                      f"{_json.dumps(request.schema)}")
        body: dict[str, Any] = {"model": binding.model, "max_tokens": request.max_output_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if request.system:
            body["system"] = request.system
        if request.thinking:
            try:
                body["thinking"] = {"type": "enabled", "budget_tokens": int(request.thinking)}
            except (TypeError, ValueError):
                body["temperature"] = request.temperature
        else:
            body["temperature"] = request.temperature
        headers = {"anthropic-version": "2023-06-01"}
        reply = transport.post_json(ticket, provider, url, body, headers=headers, auth=self.auth)
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
