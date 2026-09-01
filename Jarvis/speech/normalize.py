"""Conservative post-STT normalisation: fix what is known, never rewrite meaning.

Whisper hears "spottify", "Starkfisch", "Self Def".  The owner has told
ZEUS once what these are, or ZEUS knows them because they are its own
projects and capabilities.  This layer applies exactly that knowledge and
nothing else:

* punctuation and sentence capitalisation;
* known entity spelling (product terms, project titles, capability names,
  the owner's own vocabulary) -- only when the heard token is *close* to a
  known entity by edit distance **and** the entity is at least as long as a
  real word, so "Bio" is never turned into "Biochemie" and random words are
  never pulled toward a lookalike;
* the owner's explicit corrections ("Starkfisch" -> "Stockfish"), stored
  bounded and applied only to the heard form they were given for.

Both transcripts are kept: what was heard (``raw``) and what was made of it
(``normalized``), with every replacement listed, so Activity can show the
owner exactly what changed and the owner can undo a wrong rule.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

_TOKEN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)

#: Product vocabulary Whisper has never seen in German context.
BUILTIN_ENTITIES = (
    "Zeus", "Spotify", "GitHub", "Stockfish", "SelfDev", "Mission Control", "Knowledge", "Ollama", "Whisper",
    "Piper", "Python", "PowerShell", "Windows", "Chrome", "Screenshot", "Playlist", "Physikum", "Biochemie",
    "Anatomie", "Physiologie", "Repository", "Capability", "Worktree", "Supervisor", "Listener", "Wakeword",
    "Voice Studio", "Galaxy", "Timer", "Rammstein",
)


@dataclass
class Replacement:
    heard: str
    meant: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"heard": self.heard, "meant": self.meant, "reason": self.reason}


@dataclass
class Normalized:
    raw: str
    text: str
    replacements: list[Replacement] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.text != self.raw

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "text": self.text, "changed": self.changed, "replacements": [r.to_dict() for r in self.replacements]}


class Vocabulary:
    """The owner's heard->meant corrections, bounded, on disk next to the voice settings."""

    LIMIT = 300

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in data.get("entries", []) if isinstance(data, dict) else []:
            heard = str(row.get("heard", "")).strip().lower()
            meant = str(row.get("meant", "")).strip()
            if heard and meant:
                self.entries[heard] = {"heard": heard, "meant": meant, "note": str(row.get("note", "")), "at": str(row.get("at", "")),
                                       "applied": int(row.get("applied", 0) or 0)}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": list(self.entries.values())[-self.LIMIT:]}, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def learn(self, heard: str, meant: str, *, note: str = "") -> dict[str, Any]:
        """Store one heard->meant pair.  Refuses trivial or dangerous rules."""

        heard_key = " ".join(_TOKEN.findall(heard.lower()))
        meant = meant.strip()
        if not heard_key or not meant:
            return {"ok": False, "error": "both the heard form and the meant form are needed"}
        if len(heard_key) < 3:
            return {"ok": False, "error": "the heard form is too short to be a safe rule"}
        if heard_key == " ".join(_TOKEN.findall(meant.lower())):
            return {"ok": False, "error": "heard and meant are the same word"}
        import time

        with self._lock:
            self.entries[heard_key] = {"heard": heard_key, "meant": meant, "note": note[:200], "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "applied": 0}
            while len(self.entries) > self.LIMIT:
                del self.entries[next(iter(self.entries))]
            self.save()
        return {"ok": True, "heard": heard_key, "meant": meant}

    def forget(self, heard: str) -> bool:
        key = " ".join(_TOKEN.findall(heard.lower()))
        with self._lock:
            removed = self.entries.pop(key, None) is not None
            if removed:
                self.save()
        return removed

    def list(self) -> list[dict[str, Any]]:
        return list(self.entries.values())

    def meant_terms(self) -> list[str]:
        return [row["meant"] for row in self.entries.values()]


def _close(heard: str, entity: str) -> bool:
    """Whether ``heard`` is a mis-spelling of ``entity`` rather than another word."""

    h, e = heard.lower(), entity.lower()
    if h == e:
        return False
    if len(e) < 5 or abs(len(h) - len(e)) > 2:
        return False
    if h[0] != e[0]:
        # a different first letter is a different word ("Bio"/"Rio")
        return False
    ratio = SequenceMatcher(None, h, e).ratio()
    return ratio >= 0.8


class Normalizer:
    def __init__(self, *, entities: Iterable[str] = (), vocabulary: Vocabulary | None = None) -> None:
        self.entities = list(dict.fromkeys(list(BUILTIN_ENTITIES) + [e for e in entities if e]))
        self.vocabulary = vocabulary

    def with_entities(self, entities: Iterable[str]) -> "Normalizer":
        return Normalizer(entities=list(self.entities) + list(entities), vocabulary=self.vocabulary)

    def apply(self, text: str, *, word_probabilities: dict[str, float] | None = None) -> Normalized:
        raw = text or ""
        if not raw.strip():
            return Normalized(raw, raw.strip())
        replacements: list[Replacement] = []
        out = raw

        # 1. the owner's explicit corrections: exact heard forms only (multi-word first)
        if self.vocabulary is not None:
            rules = sorted(self.vocabulary.entries.values(), key=lambda r: -len(r["heard"]))
            for rule in rules:
                pattern = re.compile(r"(?<![\w'])" + r"\s+".join(re.escape(part) for part in rule["heard"].split()) + r"(?![\w'])", re.I | re.U)
                if pattern.search(out):
                    out = pattern.sub(rule["meant"], out)
                    replacements.append(Replacement(rule["heard"], rule["meant"], "owner correction"))
                    rule["applied"] = int(rule.get("applied", 0)) + 1

        # 2. known entities: only near-misses of a known term (never exact, never far)
        single = [e for e in self.entities if " " not in e]
        def fix(match: re.Match[str]) -> str:
            token = match.group(0)
            for entity in single:
                if token.lower() == entity.lower():
                    # the same word: canonical casing only ("spotify" -> "Spotify")
                    return entity
                if _close(token, entity):
                    replacements.append(Replacement(token, entity, "known entity spelling"))
                    return entity
            return token
        out = _TOKEN.sub(fix, out)
        for entity in (e for e in self.entities if " " in e):
            pattern = re.compile(r"\b" + r"[\s-]+".join(re.escape(p) for p in entity.split()) + r"\b", re.I)
            if pattern.search(out) and not re.search(re.escape(entity), out):
                out = pattern.sub(entity, out)
                replacements.append(Replacement(entity, entity, "known entity spelling"))

        # 3. capitalisation and terminal punctuation
        out = out.strip()
        if out and out[0].islower():
            out = out[0].upper() + out[1:]
        if out and out[-1] not in ".!?…":
            out += "?" if re.match(r"^(wie|was|warum|wieso|wann|wo|wer|welche|kannst|hast|bist|gibt|how|what|why|when|where|who|which|can|do|is|are)\b", out, re.I) else "."
        out = re.sub(r"\s+([,.!?])", r"\1", out)
        return Normalized(raw, out, replacements)


def entity_hints(entities: Iterable[str], *, limit: int = 24) -> str:
    """A bounded, comma-separated hint for the recogniser: names, not a prompt."""

    seen: list[str] = []
    for item in entities:
        item = str(item or "").strip()
        if item and len(item) <= 40 and item not in seen:
            seen.append(item)
        if len(seen) >= limit:
            break
    return ", ".join(seen)
