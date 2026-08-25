"""What the user prefers, kept out of the code that acts on it.

"Spotify is my music provider" is a fact about this user, not a fact about how
music works.  Writing it into the conversational handler -- ``if "music" in
text: open_spotify()`` -- would make the preference unchangeable without an
edit, unreadable without grepping, and invisible to the user whose preference
it is.  It would also quietly make Spotify the *only* provider, because there
would be nowhere for a second one to be named.

So preferences are stored, dotted, and read by a resolver:

    music.default_provider  -> "spotify"
    music.default_output    -> "this_pc"

The intent layer never learns the word Spotify.  It produces ``music.play``,
and the resolver looks up which provider this user wants and finds the
capability that implements it.  Adding a second provider is a registered
capability plus one preference change; it is not a branch in a prompt.

Secrets deliberately do not live here.  This file is ordinary configuration --
readable, printable in diagnostics, safe to show -- and a credential in it would
be none of those things.  See :mod:`runtime.secrets`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

#: Shipped defaults.  A preference the user has never set still has an answer,
#: and the answer is visible here rather than implied by a fallback buried in
#: whichever function happened to need it first.
DEFAULTS: dict[str, Any] = {
    "music.default_provider": "spotify",
    "music.default_output": "this_pc",
}


class Preferences:
    """Dotted key/value settings, persisted as one JSON file."""

    def __init__(self, path: str | Path, *, defaults: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._defaults = dict(DEFAULTS if defaults is None else defaults)
        self._values: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> "Preferences":
        if not self.path.exists():
            return self
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt preferences file must not stop the product starting;
            # the defaults are always a working answer.
            return self
        if isinstance(data, dict):
            self._values = {str(key): value for key, value in data.items()}
        return self

    def save(self) -> Path:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._values, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        return self.path

    # -- access ----------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._values:
            return self._values[key]
        if key in self._defaults:
            return self._defaults[key]
        return default

    def set(self, key: str, value: Any) -> Any:
        self._values[key] = value
        self.save()
        return value

    def unset(self, key: str) -> None:
        self._values.pop(key, None)
        self.save()

    def namespace(self, prefix: str) -> dict[str, Any]:
        """Everything under ``prefix``, defaults included."""

        head = prefix.rstrip(".") + "."
        keys = {key for key in self._defaults if key.startswith(head)}
        keys |= {key for key in self._values if key.startswith(head)}
        return {key: self.get(key) for key in sorted(keys)}

    def to_dict(self) -> dict[str, Any]:
        keys = set(self._defaults) | set(self._values)
        return {
            key: {"value": self.get(key), "set_by_user": key in self._values}
            for key in sorted(keys)
        }
