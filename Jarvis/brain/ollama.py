"""Ollama provider using the native ``/api/chat`` endpoint.

Ollama also speaks an OpenAI-compatible dialect, and
:class:`~brain.providers.OpenAICompatibleBrainProvider` can drive it.  This
provider exists because the compatible dialect silently drops the three
settings that matter most on an 8 GB GPU:

``num_ctx``
    How much KV cache to allocate.  The OpenAI route uses whatever the model's
    Modelfile says -- 32k for qwen2.5-coder, 262k for qwen3 -- which on a GTX
    1070 either fails to load or evicts the desktop's VRAM.  Pinning it is the
    difference between a usable machine and an unusable one.
``keep_alive``
    How long weights stay resident.  A cold load of the 7B coder costs ~47 s on
    this machine, so keeping it warm across an autonomous run matters; equally,
    releasing it afterwards is what gives the GPU back to the user.
``format``
    A JSON schema Ollama enforces during sampling.  Constrained decoding is the
    single most effective mitigation for a small model's tendency to emit prose
    around its JSON.

The provider deliberately keeps the same surface as the other providers
(``generate`` / ``generate_coding`` / ``generate_structured`` / ``health_check``)
so nothing downstream needs to know which one it is talking to.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from brain.providers import ProviderError, _safe_error_message
from brain.tiers import ModelSpec


class OllamaBrainProvider:
    """Talks to a local Ollama daemon over its native API."""

    provider_name = "ollama"

    def __init__(self, spec: ModelSpec, *, system_prompt: str | None = None) -> None:
        self.spec = spec
        self.model_name = spec.model
        self.config = spec  # parity with OpenAICompatibleBrainProvider.config
        self.system_prompt = system_prompt
        self.last_metadata: dict[str, Any] = {}

    # -- capability surface ---------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        return {
            "chat": True,
            "coding": True,
            "embeddings": False,
            "local": True,
            "structured_generation": True,
        }

    def list_models(self) -> list[str]:
        payload = self._get("/api/tags")
        return [str(item.get("name", "")) for item in (payload.get("models") or []) if item.get("name")]

    def health_check(self) -> dict[str, Any]:
        """Prove the model works; never infer it from the server being up."""

        try:
            models = self.list_models()
        except Exception as exc:
            return {"ok": False, "provider": self.provider_name, "model": self.model_name, "error": _error_payload(exc)}
        if models and self.model_name not in models and self.model_name.split(":", 1)[0] not in {m.split(":", 1)[0] for m in models}:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model_name,
                "error": {"kind": "model_missing", "message": f"{self.model_name} is not pulled"},
                "available_models": models,
            }
        try:
            text = self.generate("Reply with the single word: OK", max_tokens=8, temperature=0.0)
        except Exception as exc:
            return {"ok": False, "provider": self.provider_name, "model": self.model_name, "error": _error_payload(exc)}
        return {
            "ok": bool(str(text).strip()),
            "provider": self.provider_name,
            "model": self.model_name,
            "context_window": self.spec.context_window,
            "metadata": dict(self.last_metadata),
        }

    # -- generation ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        return self._chat(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def generate_coding(
        self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.1, top_p: float = 0.9
    ) -> str:
        return self._chat(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        return self._chat(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, schema=schema
        )

    # Older call sites in this repository use the JarvisBrain vocabulary.
    def think(self, user_prompt: str, max_tokens: int = 512) -> str:
        return self.generate(user_prompt, max_tokens=max_tokens)

    def think_coding(self, user_prompt: str, max_tokens: int = 1024, temperature: float = 0.1, top_p: float = 0.9) -> str:
        return self.generate_coding(user_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def unload(self) -> None:
        """Ask Ollama to evict the model now, freeing VRAM for the user."""

        try:
            self._post("/api/chat", {"model": self.model_name, "messages": [], "keep_alive": 0}, timeout=30.0)
        except Exception:
            # Unloading is a courtesy, never a correctness requirement.
            pass

    # -- transport -------------------------------------------------------

    def _chat(
        self,
        prompt: str,
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "num_ctx": int(self.spec.context_window),
            "num_predict": int(max_tokens if max_tokens is not None else self.spec.max_output_tokens),
            "temperature": float(self.spec.temperature if temperature is None else temperature),
            "top_p": float(self.spec.top_p if top_p is None else top_p),
        }
        if self.spec.gpu_layers is not None:
            options["num_gpu"] = int(self.spec.gpu_layers)
        options.update(self.spec.options or {})

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.spec.keep_alive,
            "options": options,
        }
        if schema is not None:
            body["format"] = schema

        started = time.perf_counter()
        payload = self._post("/api/chat", body, timeout=self.spec.timeout_seconds)
        content = ((payload.get("message") or {}).get("content")) or ""

        self.last_metadata = {
            "latency_seconds": time.perf_counter() - started,
            "generated_tokens": payload.get("eval_count"),
            "prompt_tokens": payload.get("prompt_eval_count"),
            "load_seconds": _nanos_to_seconds(payload.get("load_duration")),
            "eval_seconds": _nanos_to_seconds(payload.get("eval_duration")),
            "finish_reason": payload.get("done_reason"),
            "num_ctx": options["num_ctx"],
        }
        eval_seconds = self.last_metadata.get("eval_seconds")
        eval_count = payload.get("eval_count")
        if eval_seconds and eval_count:
            self.last_metadata["tokens_per_second"] = round(float(eval_count) / float(eval_seconds), 2)
        return str(content)

    def _get(self, path: str, *, timeout: float = 10.0) -> dict[str, Any]:
        url = self.spec.base_url.rstrip("/") + path
        request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        return self._send(request, timeout)

    def _post(self, path: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        url = self.spec.base_url.rstrip("/") + path
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "Jarvis/1.0"},
        )
        return self._send(request, timeout)

    def _send(self, request: urllib.request.Request, timeout: float) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                detail = str(exc)
            kind = "model_missing" if exc.code == 404 or "not found" in detail.lower() else "provider_http_error"
            raise ProviderError(
                kind=kind, status=exc.code, message=_safe_error_message(detail), model=self.model_name
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                kind="provider_connection_error", message=_safe_error_message(str(exc)), model=self.model_name
            ) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                kind="bad_provider_response", message=_safe_error_message(raw[:400]), model=self.model_name
            ) from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise ProviderError(
                kind="provider_error", message=_safe_error_message(str(payload["error"])), model=self.model_name
            )
        return payload if isinstance(payload, dict) else {}


def _nanos_to_seconds(value: Any) -> float | None:
    try:
        return round(float(value) / 1e9, 3)
    except (TypeError, ValueError):
        return None


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ProviderError):
        return {"kind": exc.kind, "status": exc.status, "message": exc.message}
    return {"kind": type(exc).__name__, "message": _safe_error_message(str(exc))}
