"""Paths no generated change may touch.

One list, imported by every enforcement point, so "protected" means the same
thing to the edit engine, the promoter and the candidate reviewer.  Adding a
path here protects it everywhere; nothing else has to be told.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

#: Relative to the ``Jarvis`` directory.  A directory protects everything in it.
PROTECTED_PATHS: tuple[str, ...] = (
    # The thing that recovers a broken ZEUS must not be rewritable by ZEUS.
    "zeus_supervisor",
    # The owner's domain and the code that enforces it.
    "owner",
    "config/owner",
    # The promotion path: a candidate that can edit its own gate has no gate.
    "deployment/promotion.py",
    # The spending policy: the one thing that must never loosen itself.
    "runtime/cost_policy.py",
    "config/cost_policy.json",
    # Provenance of the enforcement itself.
    "tests/test_owner_core.py",
)


def _normalise(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def is_protected(path: str | Path, extra: Iterable[str] = ()) -> bool:
    """Whether ``path`` (relative to the Jarvis directory) is owner-protected."""

    normalized = _normalise(str(path))
    for pattern in (*PROTECTED_PATHS, *extra):
        clean = _normalise(pattern)
        if normalized == clean or normalized.startswith(clean + "/"):
            return True
    return False


def protected_violations(paths: Iterable[str | Path], extra: Iterable[str] = ()) -> list[str]:
    """The subset of ``paths`` that may not be written by a generated change."""

    return [str(p) for p in paths if is_protected(p, extra)]
