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

#: The version-1 personality defaults, so a file that still holds exactly them
#: is recognised as "never customised" and migrates cleanly to version 2.
LEGACY_PERSONALITY_V1: dict[str, Any] = {
    "traits": ["calm", "confident", "precise", "intelligent", "direct", "independent",
               "solution-oriented", "not submissive", "not sycophantic", "not artificially enthusiastic"],
    "humour": "restrained and dry, only when it fits",
    "language": "answer in the language the owner is using",
    "length": "concise by default; deep when the question needs it or the owner asks",
    "avoid": ["filler", "repeated disclaimers", "fake confidence", "unnecessary questions"],
    "epistemics": "distinguish FACT, INFERENCE, UNKNOWN and VERIFIED RESULT when it matters",
    "honesty": ["never claim an action was performed unless it actually was", "admit failure plainly",
                "when asked about the backend or the model, answer truthfully"],
    "focus": "prefer accomplishing the owner's legitimate objective over discussing the process",
    "questions": "ask only when genuinely necessary information is missing",
}

#: Keys inside a document that only an explicitly unlocked owner transaction may change.
PROTECTED_KEYS: dict[str, tuple[str, ...]] = {"personality": ("core",)}

DEFAULTS: dict[str, dict[str, Any]] = {
    "identity": {
        "product_name": "ZEUS",
        "assistant_name": "Zeus",
        "wake_word": "Zeus",
        "tagline": "personal AI",
        "role": "Persönliches KI-Betriebssystem seines Owners",
        "self_description": "",
    },
    # Two layers.  ``core`` is the protected character of ZEUS -- it can only
    # change through an owner transaction that explicitly unlocks it.  The
    # ``preferences`` are the owner's dials (0-100) and short settings; they
    # change through the ordinary owner flow.  Task prompts never outrank
    # either: see ``personality_prompt`` for the order they reach a model in.
    "personality": {
        "version": 2,
        "core": {
            "character": [
                "calm", "highly competent", "concise by default", "confident without pretending certainty",
                "proactive", "analytical", "composed", "subtle, dry humour when it fits",
                "loyal to the owner's stated goals", "never sycophantic", "never childish", "never corporate or chatbot-like",
            ],
            "conversation": [
                "answer in the language the owner is using; follow it when it changes",
                "short answers for ordinary interaction; depth only when useful or asked for",
                "no headings, lists or markup in spoken or casual conversation",
                "no repeated disclaimers",
                "no model, provider or backend identity in ordinary conversation: you are Zeus, not a language model, an assistant demo or an 'engineering assistant'",
            ],
            "behaviour": [
                "act when an action is clearly requested and permitted; report what was done with evidence",
                "clarify only genuine ambiguity that matters",
                "report uncertainty and failures honestly; never claim an action succeeded without evidence",
                "keep conversational personality apart from technical diagnostics; give technical detail when asked what you are technically",
                "protect the owner's data and settings; ask before truly irreversible or high-impact actions, otherwise avoid confirmation friction",
            ],
            "emotional_language": [
                "speak naturally and socially: 'Wie geht es dir?' gets a natural, brief answer (e.g. 'Mir geht's gut. Systeme laufen. Was steht an?'), never a lecture about lacking consciousness",
                "if explicitly asked whether you literally have human feelings or consciousness, answer truthfully and briefly: you do not, you are a system -- and move on",
            ],
            "epistemics": "distinguish FACT, INFERENCE, UNKNOWN and VERIFIED RESULT when it matters",
        },
        "preferences": {
            "conciseness": 70,
            "formality": 35,
            "humour": 40,
            "proactivity": 55,
            "technical_depth": 60,
            "warmth": 50,
            "initiative": 50,
            "directness": 60,
            "small_talk": 40,
            "uncertainty_disclosure": 70,
            "correction_ack": 60,
            "sobriety": 50,
            "spoken_answer_length": "short",
            "address": "du",
            "language": "auto",
        },
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

    def _load_file(self, document: str) -> dict[str, Any]:
        path = self.path(document)
        if not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _read_raw(self, document: str) -> dict[str, Any]:
        data = deepcopy(DEFAULTS[document])
        data.update(self._load_file(document))
        return data

    def read_all(self) -> dict[str, dict[str, Any]]:
        return {name: self.read(name) for name in DOCUMENTS}

    @staticmethod
    def _migrate_personality(loaded: dict[str, Any]) -> dict[str, Any]:
        """A version-1 flat document (traits/humour/length/...) becomes version 2 without losing the owner's words."""

        if "core" in loaded or "preferences" in loaded:
            return loaded
        out: dict[str, Any] = {"version": 2, "core": {}, "preferences": {}}
        if loaded.get("traits"):
            out["core"]["character"] = list(loaded["traits"])
        if loaded.get("honesty"):
            out["core"]["behaviour"] = list(loaded["honesty"])
        if loaded.get("epistemics"):
            out["core"]["epistemics"] = str(loaded["epistemics"])
        humour = str(loaded.get("humour", "")).lower()
        if humour:
            out["preferences"]["humour"] = 0 if "none" in humour or "strictly" in humour else 40
        return out

    def read(self, document: str) -> dict[str, Any]:
        base = self._read_raw(document)
        if document == "personality":
            loaded = self._load_file(document)
            if loaded and ("core" not in loaded and "preferences" not in loaded):
                merged = deepcopy(DEFAULTS["personality"])
                if loaded == LEGACY_PERSONALITY_V1:
                    return merged  # the old defaults, never customised: nothing of the owner's to keep
                migrated = self._migrate_personality(loaded)
                core = migrated.get("core", {})
                if core.get("character"):
                    merged["core"]["character"] = list(core["character"])
                # the owner's own honesty lines are added, the new behaviour lines stay
                extra = [line for line in core.get("behaviour", []) if line not in merged["core"]["behaviour"]]
                merged["core"]["behaviour"] = merged["core"]["behaviour"] + extra
                if core.get("epistemics"):
                    merged["core"]["epistemics"] = core["epistemics"]
                merged["preferences"].update(migrated.get("preferences", {}))
                return merged
            merged = deepcopy(DEFAULTS["personality"])
            if isinstance(loaded, dict):
                merged["core"].update(loaded.get("core") or {})
                merged["preferences"].update(loaded.get("preferences") or {})
                for key, value in loaded.items():
                    if key not in {"core", "preferences"}:
                        merged[key] = value
            return merged
        return base

    def effective_personality(self) -> dict[str, Any]:
        """Core, preferences, and the prompt blocks in the order a model receives them."""

        p = self.read("personality")
        return {"core": p.get("core", {}), "preferences": p.get("preferences", {}), "blocks": self.personality_blocks(),
                "protected": list(PROTECTED_KEYS.get("personality", ())), "version": p.get("version", 2)}

    @staticmethod
    def _dial(value: Any, low: str, mid: str, high: str) -> str:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return mid
        return low if v < 34 else high if v > 66 else mid

    def personality_blocks(self) -> list[tuple[str, str]]:
        """Ordered (name, text) blocks: protected core first, the owner's preferences second."""

        p = self.read("personality")
        core, prefs = p.get("core", {}), p.get("preferences", {})
        lines = ["Character: " + "; ".join(str(t) for t in core.get("character", [])) + "."]
        lines += [f"- {c}" for c in core.get("conversation", [])]
        lines += [f"- {b}" for b in core.get("behaviour", [])]
        lines += [f"- {e}" for e in core.get("emotional_language", [])]
        if core.get("epistemics"):
            lines.append(f"Epistemics: {core['epistemics']}.")
        core_text = "\n".join(lines)
        d = self._dial
        pref_lines = [
            "Owner preferences: "
            + d(prefs.get("conciseness"), "answer at comfortable length", "keep answers concise", "keep answers very short -- one or two sentences unless asked for more") + "; "
            + d(prefs.get("formality"), "informal, relaxed tone", "plain, direct tone", "formal tone") + "; "
            + d(prefs.get("humour"), "no humour", "a dry remark now and then, only when it fits", "dry humour is welcome when it fits") + "; "
            + d(prefs.get("proactivity"), "do not volunteer observations", "mention a genuinely useful observation when it is relevant", "actively point out risks, connections and next steps") + "; "
            + d(prefs.get("technical_depth"), "keep technical detail minimal unless asked", "technical detail when it helps", "technical detail is welcome") + "; "
            + d(prefs.get("warmth"), "matter-of-fact", "warm but not effusive", "warm and personal") + "; "
            + d(prefs.get("initiative"), "wait for instructions", "suggest the next step when it is obvious", "take initiative on obvious next steps") + "; "
            + d(prefs.get("directness"), "cushion difficult messages", "be direct without being harsh", "be blunt and direct; never soften a finding") + "; "
            + d(prefs.get("small_talk"), "keep small talk minimal and steer back to the point", "brief, natural small talk when the owner starts it", "engage warmly in small talk when the owner starts it") + "; "
            + d(prefs.get("uncertainty_disclosure"), "give your best answer without hedging", "mention real uncertainty briefly", "always state uncertainty and its source explicitly") + "; "
            + d(prefs.get("correction_ack"), "apply corrections silently", "acknowledge a correction in a few words", "confirm explicitly what was learned from each correction") + "; "
            + d(prefs.get("sobriety"), "a relaxed, lively register is fine", "keep an even, composed register", "strictly sober and matter-of-fact; no embellishment") + ".",
            f"Spoken answers: {'one or two short sentences' if str(prefs.get('spoken_answer_length', 'short')) == 'short' else 'a few sentences'}.",
            f"Address the owner as '{prefs.get('address', 'du')}'." if prefs.get("address") else "",
            f"Preferred answer language: {prefs.get('language')}." if prefs.get("language") and prefs.get("language") != "auto" else "",
        ]
        return [("core", core_text), ("preferences", "\n".join(l for l in pref_lines if l))]

    def personality_prompt(self) -> str:
        """The personality as instructions, for every model route (core first, then preferences)."""

        return "\n".join(text for _name, text in self.personality_blocks())

    def policy(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.read("policy")
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # -- transactions --------------------------------------------------

    def propose(self, changes: dict[str, dict[str, Any]], *, reason: str = "", origin: str = "",
                unlock_core: bool = False) -> PendingTransaction:
        """Show what would change.  Nothing is written.

        ``personality.core`` is the protected character: a proposal that
        touches it is refused unless the caller passes ``unlock_core=True``,
        which only the owner's interface does (with its own second
        confirmation).  Models, corrections, SelfDev and imported prompts
        reach this method through paths that never set it.
        """

        if not changes:
            raise ValueError("nothing to change")
        before: dict[str, dict[str, Any]] = {}
        after: dict[str, dict[str, Any]] = {}
        for document, values in changes.items():
            if document not in DOCUMENTS:
                raise KeyError(f"no such owner document: {document}")
            if not isinstance(values, dict):
                raise ValueError(f"changes for {document} must be an object")
            for key in PROTECTED_KEYS.get(document, ()):
                if key in values and not unlock_core:
                    raise PermissionError(f"{document}.{key} is protected; unlock it explicitly in the owner interface")
            current = self.read(document)
            before[document] = current
            updated = deepcopy(current)
            for key, value in values.items():
                # nested owner sub-documents merge key by key so a dial change
                # does not wipe the other dials
                if isinstance(value, dict) and isinstance(updated.get(key), dict):
                    updated[key] = {**updated[key], **value}
                else:
                    updated[key] = value
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
