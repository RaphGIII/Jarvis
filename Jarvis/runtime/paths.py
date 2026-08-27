"""One path model for everything that hands a file to a capability or action.

Three call sites used to mean three different things by "relative path": the
composer prefixed the workspace unconditionally, the capability service ran
with the capability's install directory as cwd, and the sandbox executor used
a fresh copy.  The learned word-counter therefore received
``…\\workspace\\workspace\\notes\\x.txt`` (prefixed twice), ``plan.txt``
(resolved against its own install dir) and a file the plan never wrote --
three failures, one cause.

The rules, in order:

1. An absolute path is taken as it is.
2. A path that already lies inside the workspace (absolute or written with
   the ``workspace/`` prefix) is not prefixed again.
3. Anything else is relative to the workspace root.
4. The result must stay inside the workspace (no ``..`` escapes).
5. When the caller says the file is an *input* (``must_exist``), a missing
   file is a :class:`PathError` naming the path -- before the capability runs,
   so the failure is ZEUS's honest answer rather than a subprocess's
   ``File not found`` for a path the planner invented.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


class PathError(ValueError):
    """A path that cannot be handed to a capability, and why."""


def resolve_workspace_path(workspace: str | Path, value: str, *, must_exist: bool = False) -> Path:
    root = Path(workspace).resolve()
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        raise PathError("empty path")
    candidate = Path(raw)
    if candidate.is_absolute() or PureWindowsPath(raw).is_absolute():
        resolved = Path(os.path.normpath(str(candidate)))
    else:
        parts = [p for p in candidate.parts if p not in ("", ".")]
        # "workspace/notes/x.txt" means the workspace's notes/x.txt, not
        # workspace/workspace/notes/x.txt.
        while parts and parts[0].lower() == root.name.lower():
            parts = parts[1:]
        resolved = Path(os.path.normpath(str(root.joinpath(*parts)))) if parts else root
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PathError(f"{raw} lies outside the workspace {root}") from None
    if must_exist and not resolved.exists():
        raise PathError(f"no such file in the workspace: {raw} (looked at {resolved})")
    return resolved
