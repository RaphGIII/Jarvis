"""The world model: what ZEUS currently knows to be true, as planner facts.

Small and generic.  Three sources of facts, each a token with an origin and,
for events, an expiry:

* **events** noted by capabilities, devices or the owner's tools through
  :meth:`WorldModel.note_event` (or ``POST /api/world/event``): "a chess game
  finished, result loss".  Events fade -- a loss forty seconds ago is context,
  a loss last week is not.
* **facts** produced by capability executions: whatever a capability's
  contract says it produces holds once it has run and verified, until a
  later run replaces it or it expires.
* **projects** the owner has, as ``project.<slug>`` facts, so a contract can
  say it relates to one.

The model never interprets anything.  It is the ``S0`` the composition
planner starts from and the context the semantic goal derivation sees.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from capabilities.contracts import normalise_token
from capabilities.planner import WorldState

DEFAULT_EVENT_TTL = 15 * 60.0
DEFAULT_FACT_TTL = 60 * 60.0


@dataclass
class WorldEntry:
    token: str
    source: str
    at: float
    ttl: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.ttl > 0 and time.time() > self.at + self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {"token": self.token, "source": self.source, "at": self.at, "ttl": self.ttl,
                "age_seconds": round(time.time() - self.at, 1), "detail": dict(self.detail)}


class WorldModel:
    def __init__(self, path: str | Path | None = None, *, projects: Any = None) -> None:
        self.path = Path(path) if path else None
        self._entries: dict[str, WorldEntry] = {}
        self._lock = threading.Lock()
        #: A callable returning the owner's project titles, read when a state is built.
        self._projects = projects
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in rows if isinstance(rows, list) else []:
            try:
                entry = WorldEntry(str(row["token"]), str(row.get("source", "")), float(row.get("at", 0)), float(row.get("ttl", 0)),
                                   dict(row.get("detail") or {}))
            except (KeyError, TypeError, ValueError):
                continue
            if not entry.expired:
                self._entries[entry.token] = entry
        self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rows = [e.to_dict() for e in self._entries.values() if not e.expired]
            self.path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # -- writing -----------------------------------------------------------------

    def note_event(self, token: str, *, source: str = "event", ttl: float = DEFAULT_EVENT_TTL,
                   detail: dict[str, Any] | None = None) -> WorldEntry:
        """An event happened: ``chess_game_finished`` with ``{"result": "loss"}``.

        Detail keys become facts too (``chess_game_finished.result_loss``), so
        a contract can require the outcome and not merely the event.
        """

        entry = self._put(token, source, ttl, detail)
        for key, value in (detail or {}).items():
            fact = normalise_token(f"{token}.{key}_{value}")
            if fact:
                self._put(fact, source, ttl, {})
        self._save()
        return entry

    def note_facts(self, facts: Iterable[str], *, source: str, ttl: float = DEFAULT_FACT_TTL) -> list[str]:
        added = []
        for fact in facts:
            entry = self._put(fact, source, ttl, {})
            if entry is not None:
                added.append(entry.token)
        self._save()
        return added

    def forget(self, token: str) -> bool:
        with self._lock:
            removed = self._entries.pop(normalise_token(token), None) is not None
        self._save()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._save()

    def _put(self, token: str, source: str, ttl: float, detail: dict[str, Any] | None) -> WorldEntry | None:
        norm = normalise_token(token)
        if not norm:
            return None
        entry = WorldEntry(norm, source, time.time(), float(ttl), dict(detail or {}))
        with self._lock:
            self._entries[norm] = entry
        return entry

    # -- reading -------------------------------------------------------------------

    def entries(self) -> list[WorldEntry]:
        with self._lock:
            live = [e for e in self._entries.values() if not e.expired]
            self._entries = {e.token: e for e in live}
        return sorted(live, key=lambda e: -e.at)

    def state(self, *, extra_facts: Iterable[str] = ()) -> WorldState:
        titles: list[str] = []
        if self._projects is not None:
            try:
                titles = [str(t) for t in self._projects() if str(t).strip()]
            except Exception:  # noqa: BLE001 - projects are context, never a failure
                titles = []
        extra: dict[str, str] = {}
        for entry in self.entries():
            extra[entry.token] = entry.source
        for fact in extra_facts:
            token = normalise_token(fact)
            if token:
                extra[token] = "request"
        return WorldState.build(projects=titles, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries()], "state": self.state().to_dict()}
