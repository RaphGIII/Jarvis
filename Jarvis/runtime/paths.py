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

Rule 4 is a *write* rule.  :func:`resolve_read_path` is the read model, and
the difference between them is the difference between "ZEUS may not change
this machine outside its workspace" and "ZEUS may not look at a file the owner
named" -- only the first of which is a security property.  See the read
section at the bottom of this module.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Iterable


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


# -- reading a file the owner named, outside the workspace -----------------
#
# ``resolve_workspace_path`` is the WRITE model: anything a capability creates,
# overwrites or deletes stays inside the workspace, and that must not loosen.
# Reading is a different act.  A checksum, a line count, a "what is in this
# file" is not a change to the machine, and confining it to the workspace made
# ZEUS unable to answer the one question the owner actually asked -- live
# mission m_ea63437b82 refused to hash
# ``C:\Users\rapha\OneDrive\Desktop\...\ffmpeg-7.0.1`` because it "lies
# outside the workspace", which is true and is not a reason.
#
# So the two are separated here, by name, and only the read side opens up:
#
# * a RELATIVE path still means the workspace, for reads as for writes, and
#   still may not ``..`` its way out of it -- the owner who means a file
#   elsewhere names it absolutely;
# * an ABSOLUTE path the owner named is read where it is;
# * except under a secret-bearing root, which is denied for reads too, because
#   "read any file" and "read the credential store" are not the same permission.
#
# The second bullet's exception is two lists: names that are secrets wherever
# they appear (below), and roots the caller declares -- which is how ZEUS's own
# owner directory, holding the scrypt verifier for the password that authorises
# promotion, stays unreadable by a capability that was asked for a checksum.

#: Directory names whose contents are secrets wherever they appear.
SECRET_DIR_NAMES = frozenset({
    "secrets", ".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".kube", ".docker",
    "credentials", "keyrings",
})

#: File names that are secrets wherever they appear.
SECRET_FILE_NAMES = frozenset({
    ".env", ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials.json",
})


def secret_reason(path: Path) -> str:
    """Why this path may not be read, or ``""`` when it may.

    Deny-list rather than allow-list on purpose: an allow-list of readable
    roots is the same confinement with more steps, and the owner naming a file
    is the authorisation.  What it protects is the small set of places whose
    contents are credentials no request should ever be able to print.
    """

    parts = [p.lower() for p in path.parts]
    for part in parts[:-1]:
        if part in SECRET_DIR_NAMES:
            return f"{path} lies under a secret store ({part}); reading it is not allowed"
    if path.name.lower() in SECRET_FILE_NAMES:
        return f"{path.name} is a credential file; reading it is not allowed"
    return ""


def protected_roots(state_root: str | Path, repository: str | Path | None = None) -> tuple[Path, ...]:
    """ZEUS's own unreadable places, as absolute paths.

    The owner directory holds ``auth.json`` -- the scrypt verifier for the
    password that is the only thing able to promote code into the product. A
    capability asked for a checksum has no business reading it, and neither
    has any other read this module resolves.
    """

    roots: list[Path] = []
    try:
        state = Path(state_root).resolve()
        roots += [state / "owner", state / "secrets"]
    except (OSError, TypeError, ValueError):
        pass
    if repository is not None:
        try:
            from owner.protected import PROTECTED_PATHS

            repo = Path(repository).resolve()
            roots += [repo / part for part in PROTECTED_PATHS]
        except Exception:  # noqa: BLE001 - a missing list is not a reason to open up
            pass
    return tuple(roots)


def _root_denial(path: Path, roots: Iterable[Path]) -> str:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return f"{path} lies inside {root}, which ZEUS does not read out to anyone"
    return ""


def resolve_read_path(workspace: str | Path, value: str, *, must_exist: bool = True,
                      denied_roots: Iterable[Path] = ()) -> Path:
    """An input a capability READS: the workspace, or a file the owner named.

    Raises :class:`PathError` with the reason -- outside-ness is not one of
    them, a secret store and a missing file are.
    """

    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        raise PathError("empty path")
    candidate = Path(raw)
    if not (candidate.is_absolute() or PureWindowsPath(raw).is_absolute()):
        # Relative still means the workspace, ``..`` escapes included.
        return resolve_workspace_path(workspace, raw, must_exist=must_exist)
    resolved = Path(os.path.normpath(str(candidate.expanduser())))
    denial = secret_reason(resolved) or _root_denial(resolved, denied_roots)
    if denial:
        raise PathError(denial)
    if must_exist and not resolved.exists():
        raise PathError(f"no such file: {raw}")
    return resolved


#: The argument names that mean "a file this capability WRITES".  Everything
#: else path-shaped is an input it reads.
WRITE_KEYS = frozenset({"output", "output_path", "outfile", "out", "target",
                        "destination", "dest", "save_to", "write_to"})


def is_write_key(key: str) -> bool:
    return str(key or "").strip().lower() in WRITE_KEYS


def resolve_argument_path(workspace: str | Path, key: str, value: str,
                          *, denied_roots: Iterable[Path] = ()) -> Path:
    """One path argument, resolved under the policy its NAME implies.

    A write target is confined to the workspace and need not exist yet; a read
    input may be anywhere the owner can name -- outside ``denied_roots`` -- and
    must exist before the capability runs.
    """

    if is_write_key(key):
        return resolve_workspace_path(workspace, value, must_exist=False)
    return resolve_read_path(workspace, value, must_exist=True, denied_roots=denied_roots)
