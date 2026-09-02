"""Owner aliases: "Uni-Planer" means D:\\Studium\\Planer.xlsx.

Teach-by-explanation is a core behaviour: when ZEUS cannot resolve a name
("Öffne meinen Uni-Planer") it asks ONE question, and the owner's answer is
remembered here — scoped to a kind (app, file, folder, url, project) so the
next request resolves instantly and deterministically.

The store is a small JSON file under the owner's state directory.  Keys are
folded (case/umlaut-insensitive) and stripped of leading possessives, so
"mein Uni-Planer", "Uni-Planer" and "uniplaner" resolve to the same entry.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

KINDS = ("app", "file", "folder", "url", "project")

_POSSESSIVE = re.compile(r"^(?:mein[esrmn]{0,2}|unser[esrmn]{0,2}|der|die|das|den|dem|my|our|the)\s+", re.I)


def fold(text: str) -> str:
    lowered = str(text or "").lower().replace("ß", "ss")
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(a, b)
    return "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)).strip()


def normalize(name: str) -> str:
    out = fold(name).strip(" .!?\"'„“”‚'")
    prev = None
    while prev != out:
        prev = out
        out = _POSSESSIVE.sub("", out).strip()
    return out.replace("-", " ").replace("  ", " ")


def classify_target(value: str) -> tuple[str, str]:
    """(kind, normalized value) for an owner-given location.

    A Windows path becomes file/folder (checked on disk), something URL-shaped
    becomes url.  An empty kind means the value is neither.
    """

    v = str(value or "").strip().strip("\"'„“”‚'")
    if re.match(r"^[a-zA-Z]:[\\/]", v) or v.startswith("\\\\"):
        path = Path(v)
        if path.is_dir():
            return "folder", str(path)
        if path.is_file():
            return "file", str(path)
        # a path that does not exist is still a path; the caller decides
        return "file", str(path)
    if re.match(r"^https?://", v, re.I):
        return "url", v
    if re.match(r"^(?:www\.)?[\w-]{2,}(?:\.[a-z]{2,})+(?:/\S*)?$", v, re.I):
        return "url", "https://" + v if not v.lower().startswith("http") else v
    return "", v


#: "Wenn ich 'Lernplan' sage, meine ich D:\\Studium\\plan.xlsx"
_TEACH = re.compile(
    r"^\s*(?:hey\s+|ok\s+|okay\s+)?(?:zeus|jarvis)?[,:\s]*"
    r"(?:wenn\s+ich\s+[„\"'‚]?(?P<name>[^„\"'‚''“”]{1,60}?)[“\"'']?\s+sage[,]?\s*(?:dann\s+)?meine\s+ich\s+(?P<value>.+)"
    r"|merk\s+dir[,:]?\s+[„\"'‚]?(?P<name2>[^„\"'‚''“”]{1,60}?)[“\"'']?\s+(?:ist|=|bedeutet|meint|liegt\s+(?:unter|in|bei))\s+(?P<value2>.+))\s*$",
    re.I,
)


def parse_teach(text: str) -> tuple[str, str] | None:
    """An explicit alias lesson in the owner's words, or None."""

    m = _TEACH.match((text or "").strip())
    if not m:
        return None
    name = (m.group("name") or m.group("name2") or "").strip(" .!?")
    value = (m.group("value") or m.group("value2") or "").strip(" .!?")
    if not name or not value:
        return None
    return name, value


class AliasStore:
    """A persistent name → (kind, value) mapping taught by the owner."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = dict(raw.get("aliases") or {})
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"aliases": self._data or {}}, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._load())

    def get(self, name: str) -> dict[str, Any] | None:
        key = normalize(name)
        if not key:
            return None
        with self._lock:
            data = self._load()
            entry = data.get(key)
            return dict(entry) if entry else None

    def learn(self, name: str, kind: str, value: str) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"unknown alias kind {kind!r}; one of {KINDS}")
        key = normalize(name)
        if not key:
            raise ValueError("an alias needs a name")
        entry = {"name": str(name).strip(), "kind": kind, "value": str(value).strip(),
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self._lock:
            data = self._load()
            data[key] = entry
            self._save()
        return dict(entry)

    def forget(self, name: str) -> bool:
        key = normalize(name)
        with self._lock:
            data = self._load()
            if key in data:
                del data[key]
                self._save()
                return True
        return False

    def matches(self, text: str) -> list[dict[str, Any]]:
        """Aliases whose name occurs in the text — hints for the planner."""

        probe = " " + fold(text).replace("-", " ") + " "
        out: list[dict[str, Any]] = []
        with self._lock:
            for key, entry in self._load().items():
                if key and key in probe:
                    out.append(dict(entry))
        return out[:5]
