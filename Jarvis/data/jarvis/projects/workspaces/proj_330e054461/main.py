"""Capability implementation. Replace the body of run()."""

from __future__ import annotations

import shutil
from typing import Any

# Every payload key run() accepts. A caller cannot pass what is not declared.
INPUT_SCHEMA = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    # Discover the environment with the STANDARD LIBRARY, like this. Jarvis
    # tools are not importable here -- only stdlib and declared dependencies.
    player = shutil.which("some-program")
    if payload.get("dry_run"):
        return {"ok": bool(player), "would_use": player}
    return {"ok": False, "error": "JARVIS_CAPABILITY_NOT_IMPLEMENTED"}
