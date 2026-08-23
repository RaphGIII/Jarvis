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
import re
from typing import Any

# The JSON spec only allows a backslash inside a string to be followed by one
# of ", \, /, b, f, n, r, t, or a \uXXXX escape. Code-generating small models
# routinely emit Python source (which allows `\'`) as a JSON string value and
# over-escape single quotes as `\'`, which is not valid JSON. A backslash
# never legitimately appears outside of a string in JSON, so it is safe to
# strip an invalid escape's backslash anywhere in the document.
_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _strip_invalid_escapes(text: str) -> str:
    return _INVALID_ESCAPE.sub("", text)


def lenient_json_loads(text: str) -> Any:
    """Parse JSON, tolerating common small-model JSON-generation mistakes.

    Tries strict parsing first (the common, well-formed case), then
    ``strict=False`` (tolerates literal control characters inside strings),
    then repairs invalid backslash escapes (e.g. ``\\'``) that are valid in
    Python string literals but not in JSON. Never changes behavior for
    already-valid JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    return json.loads(_strip_invalid_escapes(text), strict=False)
