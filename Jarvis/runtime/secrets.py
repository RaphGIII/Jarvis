"""Credentials, kept out of configuration and out of the repository.

The standing rule for this project is that secrets must not live in normal
config files.  :mod:`runtime.preferences` is a normal config file -- printable
in diagnostics, safe to show, checked when something looks wrong -- and a client
secret in it would be none of those things.

Two sources, in order:

1. **The environment.**  This is how every other credential in the system is
   supplied (see ``ModelSpec.api_key``), and it is the right answer for a
   machine that already has one.
2. **A file under the state root**, which is gitignored.  This exists because
   the environment is a poor place to put something a user has to paste in
   once: it means editing system settings and restarting the process, and the
   most likely outcome is the credential ending up in a shell history file
   instead.

Nothing here is ever written back.  ZEUS reads credentials; it does not manage
them, rotate them, or put them anywhere new.

A missing credential is a *fact*, reported as one.  :class:`SecretRequirement`
carries what is needed and how to supply it, so a capability that cannot run
can say exactly what would let it run -- rather than failing with something the
user cannot act on, or worse, quietly doing something else instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SecretRequirement:
    """What is missing, and what would satisfy it."""

    name: str
    fields: tuple[str, ...]
    #: Environment variables that would supply it.
    env_vars: tuple[str, ...] = ()
    #: Where a file would go.
    path: str = ""
    #: Plain instructions, shown to the user verbatim.
    how: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fields": list(self.fields),
            "env_vars": list(self.env_vars),
            "path": self.path,
            "how": self.how,
        }

    def describe(self) -> str:
        return self.how or f"{self.name} needs: {', '.join(self.fields)}"


@dataclass(frozen=True)
class Secret:
    """A resolved credential and where it came from."""

    name: str
    values: dict[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def present(self) -> bool:
        return bool(self.values) and all(self.values.values())

    def get(self, key: str) -> str:
        return self.values.get(key, "")

    def redacted(self) -> dict[str, Any]:
        """Safe for diagnostics: says what exists, never what it is."""

        return {
            "name": self.name,
            "source": self.source,
            "present": self.present,
            "fields": {key: ("set" if value else "empty") for key, value in self.values.items()},
        }


class SecretStore:
    """Reads credentials from the environment, then from the state root."""

    def __init__(self, root: str | Path) -> None:
        #: ``<state_root>/secrets``.  Under ``data/jarvis/``, which .gitignore
        #: already excludes, so a pasted credential cannot reach a commit.
        self.root = Path(root)

    def path_for(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def read(self, name: str, fields: tuple[str, ...], *, env_prefix: str = "") -> Secret:
        prefix = env_prefix or name.upper()
        env_values = {field: os.getenv(f"{prefix}_{field.upper()}", "") for field in fields}
        if all(env_values.values()):
            return Secret(name=name, values=env_values, source="environment")

        path = self.path_for(name)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                file_values = {field: str(data.get(field, "") or "") for field in fields}
                if all(file_values.values()):
                    return Secret(name=name, values=file_values, source=str(path))

        # Partial is reported as absent rather than as half-present: a client id
        # with no secret cannot authenticate, and pretending otherwise turns a
        # clear "not configured" into a confusing "unauthorised" later.
        return Secret(name=name, values={}, source="")

    def requirement(self, name: str, fields: tuple[str, ...], *, env_prefix: str = "", how: str = "") -> SecretRequirement:
        prefix = env_prefix or name.upper()
        return SecretRequirement(
            name=name,
            fields=fields,
            env_vars=tuple(f"{prefix}_{field.upper()}" for field in fields),
            path=str(self.path_for(name)),
            how=how,
        )
