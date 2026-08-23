from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from config import MAX_NEW_TOKENS, MODEL_ID, SYSTEM_PROMPT


class BrainProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, prompt: str, *, max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.2, top_p: float | None = None) -> str:
        ...

    def generate_coding(self, prompt: str, *, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 450,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        ...

    def health_check(self) -> dict[str, Any]:
        ...

    def capabilities(self) -> dict[str, Any]:
        ...


class StructuredGenerationUnsupported(NotImplementedError):
    """Raised only when a provider genuinely cannot do guided JSON."""


@dataclass
class ProviderError(RuntimeError):
    kind: str
    status: int | None = None
    message: str = ""
    model: str = ""
    attempt: int = 0
    stage: str = ""

    def __post_init__(self) -> None:
        RuntimeError.__init__(
            self,
            f"ProviderError(kind={self.kind!r}, status={self.status!r}, model={self.model!r}, attempt={self.attempt!r}, message={self.message!r})",
        )


class LocalTransformersBrainProvider:
    """Provider wrapper around the existing local Transformers/JarvisBrain path."""

    provider_name = "local_transformers"

    def __init__(self, model_id: str | None = None, brain: Any | None = None) -> None:
        self.model_id = model_id or os.getenv("JARVIS_BRAIN_MODEL") or MODEL_ID
        if brain is None:
            from brain.model import JarvisBrain

            brain = JarvisBrain(model_id=self.model_id)
        self.brain = brain
        self.model_name = self.model_id
        self.model = getattr(brain, "model", None)
        self.tokenizer = getattr(brain, "tokenizer", None)

    def generate(self, prompt: str, *, max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.2, top_p: float | None = None) -> str:
        return self.brain.ask(SYSTEM_PROMPT, prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def generate_coding(self, prompt: str, *, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 450,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        raise StructuredGenerationUnsupported("Local Transformers provider does not support guided JSON generation.")

    def think(self, user_prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
        return self.generate(user_prompt, max_tokens=max_tokens)

    def think_coding(self, user_prompt: str, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        return self.generate_coding(user_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def health_check(self) -> dict[str, Any]:
        return {"ok": self.model is not None, "provider": self.provider_name, "model": self.model_name}

    def capabilities(self) -> dict[str, Any]:
        return {"chat": True, "coding": True, "embeddings": False, "local": True, "structured_generation": False}


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 450
    retries: int = 1
    backoff_seconds: float = 0.5
    context_window: int = 8192


class OpenAICompatibleBrainProvider:
    """Provider for OpenAI-compatible /v1/chat/completions endpoints."""

    provider_name = "openai_compatible"

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self.model_name = config.model
        self.last_metadata: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> "OpenAICompatibleBrainProvider":
        base_url = os.environ["JARVIS_BRAIN_BASE_URL"]
        api_key = os.getenv("JARVIS_BRAIN_API_KEY", "")
        model = os.getenv("JARVIS_BRAIN_MODEL", MODEL_ID)
        return cls(
            OpenAICompatibleConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=float(os.getenv("JARVIS_BRAIN_TIMEOUT", "60")),
                temperature=float(os.getenv("JARVIS_BRAIN_TEMPERATURE", "0.2")),
                top_p=float(os.getenv("JARVIS_BRAIN_TOP_P", "0.9")),
                max_tokens=int(os.getenv("JARVIS_BRAIN_MAX_TOKENS", "450")),
                retries=int(os.getenv("JARVIS_BRAIN_RETRIES", "1")),
                context_window=int(os.getenv("JARVIS_BRAIN_CONTEXT_WINDOW", "8192")),
            )
        )

    def generate(self, prompt: str, *, max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None) -> str:
        return self._chat_completion(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p if top_p is None else top_p,
        )

    def generate_coding(self, prompt: str, *, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        return self._chat_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 450,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        return self._chat_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_schema=schema,
        )

    def think(self, user_prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
        return self.generate(user_prompt, max_tokens=max_tokens)

    def think_coding(self, user_prompt: str, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        return self.generate_coding(user_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def health_check(self) -> dict[str, Any]:
        try:
            self.generate("Return OK.", max_tokens=4, temperature=0.0, top_p=1.0)
            return {"ok": True, "provider": self.provider_name, "model": self.model_name, "context_window": self.config.context_window}
        except Exception as exc:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model_name,
                "error": _provider_error_payload(exc),
                "context_window": self.config.context_window,
            }

    def capabilities(self) -> dict[str, Any]:
        return {"chat": True, "coding": True, "embeddings": False, "local": False, "structured_generation": True}

    def _chat_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            structured_mode = os.getenv("JARVIS_BRAIN_STRUCTURED_MODE", "guided_json").strip().lower()
            if structured_mode == "response_format":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "jarvis_response",
                        "schema": response_schema,
                    },
                }
            else:
                payload["guided_json"] = response_schema

        reasoning_effort = os.getenv("JARVIS_BRAIN_REASONING_EFFORT", "").strip()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        body = json.dumps(payload).encode("utf-8")
        headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Jarvis/0.3",
}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            started = time.perf_counter()
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = data["choices"][0]
                content = choice["message"]["content"]
                usage = data.get("usage") or {}
                self.last_metadata = {
                    "latency_seconds": time.perf_counter() - started,
                    "generated_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "attempts": attempt + 1,
                }
                return str(content)
            except urllib.error.HTTPError as exc:
                error = self._classify_http_error(exc, attempt + 1)
                last_error = error
                if not _retryable_provider_error(error) or attempt >= self.config.retries:
                    raise error from exc
                time.sleep(self.config.backoff_seconds * (2**attempt))
            except (urllib.error.URLError, TimeoutError) as exc:
                error = ProviderError(kind="provider_connection_error", message=_safe_error_message(str(exc)), model=self.config.model, attempt=attempt + 1)
                last_error = error
                if attempt >= self.config.retries:
                    raise error from exc
                time.sleep(self.config.backoff_seconds * (2**attempt))
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                error = ProviderError(kind="bad_provider_response", message=_safe_error_message(str(exc)), model=self.config.model, attempt=attempt + 1)
                last_error = error
                raise error from exc
            except Exception as exc:
                last_error = exc
                raise
        if isinstance(last_error, ProviderError):
            raise last_error
        raise ProviderError(kind="provider_unavailable", message=type(last_error).__name__ if last_error else "unknown", model=self.config.model) from last_error

    def _classify_http_error(self, exc: urllib.error.HTTPError, attempt: int) -> ProviderError:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:1200]
        except Exception:
            body = str(exc)
        message = _safe_error_message(body or str(exc))
        lowered = message.lower()
        if "maximum context" in lowered or "context length" in lowered or "too many tokens" in lowered:
            kind = "context_overflow"
        elif exc.code == 429:
            kind = "rate_limited"
        elif exc.code in {500, 502, 503, 504}:
            kind = "server_error"
        elif exc.code in {400, 404, 422}:
            kind = "deterministic_provider_error"
        else:
            kind = "provider_http_error"
        return ProviderError(kind=kind, status=exc.code, message=message, model=self.config.model, attempt=attempt)


def _retryable_provider_error(error: ProviderError) -> bool:
    return error.kind in {"rate_limited", "server_error", "provider_connection_error"}


def _safe_error_message(message: str) -> str:
    redacted = message
    for key in ["JARVIS_BRAIN_API_KEY", "API_KEY", "TOKEN", "SECRET"]:
        secret = os.getenv(key)
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted[:1200]


def _provider_error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ProviderError):
        return {
            "kind": exc.kind,
            "status": exc.status,
            "message": exc.message,
            "model": exc.model,
            "attempt": exc.attempt,
        }
    return {"kind": type(exc).__name__, "message": _safe_error_message(str(exc))}


def make_brain_provider_from_env() -> BrainProvider:
    provider = os.getenv("JARVIS_BRAIN_PROVIDER", "local_transformers").strip().lower()
    if provider == "local_transformers":
        return LocalTransformersBrainProvider(model_id=os.getenv("JARVIS_BRAIN_MODEL") or MODEL_ID)
    if provider == "openai_compatible":
        return OpenAICompatibleBrainProvider.from_env()
    raise ValueError(f"Unsupported JARVIS_BRAIN_PROVIDER: {provider}")



def make_build_remote_brain_provider_from_env() -> BrainProvider:
    """
    Build a remote coding provider from JARVIS_BUILD_REMOTE_* variables.

    The local FAST_LOCAL JARVIS_BRAIN_* configuration is temporarily
    hidden while the remote provider is constructed, preventing the
    BUILD_REMOTE tier from accidentally inheriting Ollama/Qwen settings.
    """

    model = os.getenv("JARVIS_BUILD_REMOTE_MODEL", "").strip()
    base_url = os.getenv(
        "JARVIS_BUILD_REMOTE_BASE_URL", ""
    ).strip()

    if not model:
        raise ValueError(
            "JARVIS_BUILD_REMOTE_MODEL is not configured"
        )

    if not base_url:
        raise ValueError(
            "JARVIS_BUILD_REMOTE_BASE_URL is not configured"
        )

    remote_env: dict[str, str] = {}

    prefix = "JARVIS_BUILD_REMOTE_"

    for key, value in list(os.environ.items()):
        if not key.startswith(prefix):
            continue

        suffix = key[len(prefix):]

        if suffix == "ENABLED":
            continue

        remote_env[f"JARVIS_BRAIN_{suffix}"] = value

    remote_env.setdefault(
        "JARVIS_BRAIN_PROVIDER",
        "openai_compatible",
    )

    existing_brain_keys = [
        key
        for key in list(os.environ)
        if key.startswith("JARVIS_BRAIN_")
    ]

    saved = {
        key: os.environ.get(key)
        for key in existing_brain_keys
    }

    try:
        for key in existing_brain_keys:
            os.environ.pop(key, None)

        os.environ.update(remote_env)

        return make_brain_provider_from_env()

    finally:
        for key in list(os.environ):
            if key.startswith("JARVIS_BRAIN_"):
                os.environ.pop(key, None)

        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
