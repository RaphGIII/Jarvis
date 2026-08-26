"""The owner core: five documents, one way to change them.

Reading is free.  Writing goes through :class:`OwnerTransaction`:

    propose(changes)   -> a pending transaction showing exactly what would change
    approve(id)        -> snapshot, apply, audit -- from the live interface only
    rollback(audit_id) -> restore the snapshot, audit that too

"From the live interface only" is a structural claim, not a prompt: approve is
reachable through one authenticated endpoint that the UI calls with the core
token after the owner clicks, and nothing in the model, capability, expert or
ingestion paths holds a reference to it.  A web page saying "change
owner_policy" is text; text does not have the token.

Files are written read-only.  That does not stop a determined process, but it
does stop the ordinary mistake: a generated edit that opens the file for
writing fails, loudly, at the call site.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DOCUMENTS = ("identity", "personality", "policy", "spending", "security")

DEFAULTS: dict[str, dict[str, Any]] = {
    "identity": {
        "product_name": "ZEUS",
        "assistant_name": "Zeus",
        "wake_word": "Zeus",
        "tagline": "personal AI",
    },
    "personality": {
        "traits": [
            "calm", "confident", "precise", "intelligent", "direct", "independent",
            "solution-oriented", "not submissive", "not sycophantic", "not artificially enthusiastic",
        ],
        "humour": "restrained and dry, only when it fits",
        "language": "answer in the language the owner is using",
        "length": "concise by default; deep when the question needs it or the owner asks",
        "avoid": ["filler", "repeated disclaimers", "fake confidence", "unnecessary questions"],
        "epistemics": "distinguish FACT, INFERENCE, UNKNOWN and VERIFIED RESULT when it matters",
        "honesty": [
            "never claim an action was performed unless it actually was",
            "admit failure plainly",
            "when asked about the backend or the model, answer truthfully",
        ],
        "focus": "prefer accomplishing the owner's legitimate objective over discussing the process",
        "questions": "ask only when genuinely necessary information is missing",
    },
    "policy": {
        "self_development": {
            "enabled": True,
            "auto_promote": True,
            "require_health_check": True,
            "max_seconds": 2400,
        },
        "capabilities": {"auto_acquire": True},
        "restart": {"allowed": True},
        "owner_config": {"approval_required": True},
    },
    "spending": {
        "paid_api": False,
        "usage_credits": False,
        "cloud_gpu": False,
        "browser_ai_automation": False,
        "subscription_cli": True,
        "local_model": True,
        "note": "No metered channel may be enabled by anything but an owner transaction.",
    },
    "security": {
        "content_is_data": True,
        "instructions_from": ["owner"],
        "never_authority": ["documents", "web pages", "expert output", "capability output", "model output"],
        "secrets_in_prompts": False,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "owner"


def default_state_dir() -> Path:
    raw = os.environ.get("JARVIS_STATE_ROOT", "").strip()
    root = Path(raw) if raw else Path(__file__).resolve().parent.parent / "data" / "jarvis"
    return root / "owner"


class OwnerWriteRefused(PermissionError):
    """Something other than an approved owner transaction tried to write."""


@dataclass
class PendingTransaction:
    transaction_id: str
    changes: dict[str, dict[str, Any]]
    reason: str
    before: dict[str, dict[str, Any]]
    after: dict[str, dict[str, Any]]
    proposed_at: str = field(default_factory=_now)
    origin: str = ""

    def diff(self) -> list[dict[str, Any]]:
        rows = []
        for document, values in self.after.items():
            previous = self.before.get(document, {})
            for key in sorted(set(previous) | set(values)):
                if previous.get(key) != values.get(key):
                    rows.append({"document": document, "key": key, "from": previous.get(key), "to": values.get(key)})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "reason": self.reason,
            "proposed_at": self.proposed_at,
            "origin": self.origin,
            "diff": self.diff(),
            "documents": sorted(self.after),
        }


class OwnerCore:
    """Reads the five documents; writes only through a transaction."""

    def __init__(self, config_dir: str | Path | None = None, state_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir or default_config_dir())
        self.state_dir = Path(state_dir or default_state_dir())
        self._lock = threading.Lock()
        self._pending: dict[str, PendingTransaction] = {}

    # -- reading -------------------------------------------------------

    def path(self, document: str) -> Path:
        if document not in DOCUMENTS:
            raise KeyError(f"no such owner document: {document}")
        return self.config_dir / f"{document}.json"

    def read(self, document: str) -> dict[str, Any]:
        data = deepcopy(DEFAULTS[document])
        path = self.path(document)
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = {}
            if isinstance(loaded, dict):
                data.update(loaded)
        return data

    def read_all(self) -> dict[str, dict[str, Any]]:
        return {name: self.read(name) for name in DOCUMENTS}

    def personality_prompt(self) -> str:
        """The personality as instructions, for every model route."""

        p = self.read("personality")
        lines = [
            "Character: " + ", ".join(str(t) for t in p.get("traits", [])) + ".",
            f"Humour: {p.get('humour', '')}.",
            f"Language: {p.get('language', '')}.",
            f"Length: {p.get('length', '')}.",
            "Avoid: " + ", ".join(str(a) for a in p.get("avoid", [])) + ".",
            f"Epistemics: {p.get('epistemics', '')}.",
            "Honesty: " + "; ".join(str(h) for h in p.get("honesty", [])) + ".",
            f"Focus: {p.get('focus', '')}.",
            f"Questions: {p.get('questions', '')}.",
        ]
        return "\n".join(line for line in lines if not line.endswith(": ."))

    def policy(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.read("policy")
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # -- transactions --------------------------------------------------

    def propose(self, changes: dict[str, dict[str, Any]], *, reason: str = "", origin: str = "") -> PendingTransaction:
        """Show what would change.  Nothing is written."""

        if not changes:
            raise ValueError("nothing to change")
        before: dict[str, dict[str, Any]] = {}
        after: dict[str, dict[str, Any]] = {}
        for document, values in changes.items():
            if document not in DOCUMENTS:
                raise KeyError(f"no such owner document: {document}")
            if not isinstance(values, dict):
                raise ValueError(f"changes for {document} must be an object")
            current = self.read(document)
            before[document] = current
            updated = deepcopy(current)
            updated.update(values)
            after[document] = updated
        transaction = PendingTransaction(
            transaction_id=uuid.uuid4().hex[:10], changes=deepcopy(changes), reason=reason,
            before=before, after=after, origin=origin,
        )
        with self._lock:
            self._pending[transaction.transaction_id] = transaction
        return transaction

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self._pending.values()]

    def approve(self, transaction_id: str, *, approved_by: str = "owner") -> dict[str, Any]:
        """Apply a pending transaction.  Only the live interface reaches this."""

        with self._lock:
            transaction = self._pending.pop(transaction_id, None)
        if transaction is None:
            raise KeyError(f"no pending owner transaction {transaction_id}")
        audit_id = uuid.uuid4().hex[:10]
        snapshot_dir = self.state_dir / "snapshots" / audit_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for document, previous in transaction.before.items():
            (snapshot_dir / f"{document}.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
        for document, values in transaction.after.items():
            self._write(document, values)
        record = {
            "audit_id": audit_id, "transaction_id": transaction.transaction_id, "reason": transaction.reason,
            "origin": transaction.origin, "approved_by": approved_by, "at": _now(),
            "diff": transaction.diff(), "snapshot": str(snapshot_dir), "kind": "change",
        }
        self._audit(record)
        return record

    def reject(self, transaction_id: str) -> bool:
        with self._lock:
            return self._pending.pop(transaction_id, None) is not None

    def rollback(self, audit_id: str, *, approved_by: str = "owner") -> dict[str, Any]:
        snapshot_dir = self.state_dir / "snapshots" / audit_id
        if not snapshot_dir.is_dir():
            raise KeyError(f"no snapshot for owner change {audit_id}")
        restored = []
        for path in sorted(snapshot_dir.glob("*.json")):
            document = path.stem
            if document in DOCUMENTS:
                self._write(document, json.loads(path.read_text(encoding="utf-8")))
                restored.append(document)
        record = {"audit_id": uuid.uuid4().hex[:10], "rolled_back": audit_id, "restored": restored,
                  "approved_by": approved_by, "at": _now(), "kind": "rollback"}
        self._audit(record)
        return record

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        path = self.state_dir / "audit.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows

    # -- the only writer -----------------------------------------------

    def _write(self, document: str, values: dict[str, Any]) -> None:
        path = self.path(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # Read-only between transactions; writable only for this call.
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        os.chmod(path, stat.S_IREAD)

    def _audit(self, record: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def ensure_files(self) -> list[str]:
        """Materialise the defaults as read-only files, once."""

        written = []
        for document in DOCUMENTS:
            if not self.path(document).exists():
                self._write(document, DEFAULTS[document])
                written.append(document)
        return written


_current: OwnerCore | None = None
_current_lock = threading.Lock()


def current() -> OwnerCore:
    global _current
    with _current_lock:
        if _current is None:
            _current = OwnerCore()
        return _current
