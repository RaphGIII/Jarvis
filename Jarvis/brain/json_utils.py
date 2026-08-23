"""Shared helpers for parsing JSON emitted by LLM providers.

Local/small models served through vLLM-style guided-JSON decoding
occasionally emit raw, unescaped control characters (literal newlines,
tabs) inside JSON string values instead of the correctly escaped ``\\n``
sequence. That output is otherwise well-formed and semantically valid, but
Python's strict-mode :func:`json.loads` rejects it outright per the JSON
spec, which wastes an entire LLM call/repair-cycle attempt for something
that is trivially recoverable. Every call site that parses raw LLM text as
JSON should go through :func:`lenient_json_loads` instead of calling
``json.loads`` directly.
"""
from __future__ import annotations

import json
from typing import Any


def lenient_json_loads(text: str) -> Any:
    """Parse JSON, tolerating literal control characters inside strings.

    Tries strict parsing first (the common, well-formed case) and only
    falls back to ``strict=False`` if that fails, so this never changes
    behavior for already-valid JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text, strict=False)
