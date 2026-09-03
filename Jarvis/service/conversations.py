"""Conversation archive: recent chats survive, with compact summaries.

Every time the owner starts a fresh conversation (or the product shuts
down), the outgoing transcript is archived as one JSON record: turns, a
short meaningful title, and — filled in asynchronously by FAST_LOCAL — a
compact summary (what was discussed, decisions, open tasks, artifacts).
The rail lists them grouped by day; opening one restores the transcript.

The raw transcript, the summary and durable facts stay SEPARATE: the
archive never becomes a prompt dump, and candidate long-term facts go
through the owner-memory rules, not silently into context.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

MIN_TURNS = 2
LIST_LIMIT = 60

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "open_tasks": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary"],
}

SUMMARY_PROMPT = """Fasse dieses Gespräch zwischen dem Owner und ZEUS kompakt zusammen.
- title: 3-6 Wörter, aussagekräftig, keine Anführungszeichen.
- summary: 2-4 Sätze — was besprochen wurde, Entscheidungen, Ergebnisse.
- open_tasks: offene Punkte (leer, wenn keine).
- facts: höchstens 3 dauerhaft nützliche Fakten über den Owner/das System (leer, wenn keine).

Gespräch:
{transcript}

JSON:"""


class ConversationArchive:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, conv_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "", str(conv_id))
        return self.root / f"{safe}.json"

    def archive(self, turns: list[dict[str, Any]], *, language: str = "", reason: str = "") -> dict[str, Any] | None:
        turns = [t for t in turns if isinstance(t, dict) and str(t.get("text", "")).strip()]
        if len(turns) < MIN_TURNS:
            return None
        first_user = next((str(t.get("text", "")) for t in turns if t.get("role") == "user"), "")
        record = {
            "id": "conv_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "title": (first_user[:64].strip() or "Gespräch"),
            "summary": "", "open_tasks": [], "facts": [],
            "turns": turns[-80:], "turn_count": len(turns),
            "language": language, "reason": reason, "summarized": False,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path(record["id"]).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(record)

    def set_summary(self, conv_id: str, *, title: str = "", summary: str = "",
                    open_tasks: list[str] | None = None, facts: list[str] | None = None) -> bool:
        with self._lock:
            path = self._path(conv_id)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if title.strip():
                record["title"] = title.strip()[:80]
            if summary.strip():
                record["summary"] = summary.strip()[:1200]
            if open_tasks is not None:
                record["open_tasks"] = [str(t)[:200] for t in open_tasks][:8]
            if facts is not None:
                record["facts"] = [str(f)[:200] for f in facts][:5]
            record["summarized"] = True
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def list(self, limit: int = 30) -> list[dict[str, Any]]:
        try:
            paths = sorted(self.root.glob("conv_*.json"), reverse=True)[: max(1, min(limit, LIST_LIMIT))]
        except OSError:
            return []
        out = []
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({k: record.get(k) for k in ("id", "at", "title", "summary", "turn_count", "summarized", "open_tasks")})
        return out

    def get(self, conv_id: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(conv_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def delete(self, conv_id: str) -> bool:
        try:
            self._path(conv_id).unlink()
            return True
        except OSError:
            return False

    def attach_result(self, conv_id: str, result: dict[str, Any]) -> bool:
        """A background job finished after the conversation was archived: keep the result with it."""

        with self._lock:
            path = self._path(conv_id)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            record.setdefault("late_results", []).append(result)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return True


def transcript_text(turns: list[dict[str, Any]], *, limit_chars: int = 4000) -> str:
    lines = []
    for turn in turns:
        who = "Owner" if turn.get("role") == "user" else "ZEUS"
        lines.append(f"{who}: {str(turn.get('text', '')).strip()}")
    text = "\n".join(lines)
    return text[-limit_chars:]
