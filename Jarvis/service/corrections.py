"""Korrigieren: what the owner meant, kept apart from what the model guessed.

EXECUTION VERIFIED does not imply USER INTENT SATISFIED.  A file was written,
the receipt proves it, and it was still the wrong file.  The owner says what
was wrong; this module turns that sentence into a durable, scoped correction
with provenance, and hands the relevant ones to the next interpretation
*before* the model guesses -- where they outrank the guess.

Classification decides what a correction changes:

    INTENT_ERROR              the request was read as the wrong kind of thing
    ENTITY_RESOLUTION_ERROR   the right kind, the wrong object (file, track, app)
    PARAMETER_ERROR           right object, wrong detail (path, format, amount)
    OWNER_PREFERENCE          not an error: how the owner wants it from now on
    CAPABILITY_DEFECT         the capability itself misbehaved -> repair, not memory
    EXECUTION_FAILURE         the tool failed -> transient, no rule learned
    VERIFICATION_DEFECT       the check passed something it should have refused

Scope decides how far it reaches:

    THIS_REQUEST              one-shot; recorded, never retrieved
    ENTITY_SPECIFIC           whenever the same object is named
    INTENT_SPECIFIC           whenever the same kind of request comes
    DOMAIN_SPECIFIC           the same area (files, music, ...)
    GLOBAL_OWNER_PREFERENCE   always

Classification is deterministic from the owner's words; the owner sees it and
can change it before saving.  Only owner corrections are stored here, and only
through the owner's own endpoint -- never from a document, an expert or a
model.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

#: "du sprichst X falsch aus, sag Y": stored in the pronunciation lexicon, never in the personality.
PRONUNCIATION = "PRONUNCIATION"

#: "du hast X gehört, ich sagte Y": a transcription error, stored as a bounded vocabulary rule.
STT_CORRECTION = "STT_CORRECTION"

CLASSES = (
    PRONUNCIATION, STT_CORRECTION,
    "INTENT_ERROR", "ENTITY_RESOLUTION_ERROR", "PARAMETER_ERROR", "OWNER_PREFERENCE",
    "CAPABILITY_DEFECT", "EXECUTION_FAILURE", "VERIFICATION_DEFECT",
)
SCOPES = ("THIS_REQUEST", "ENTITY_SPECIFIC", "INTENT_SPECIFIC", "DOMAIN_SPECIFIC", "GLOBAL_OWNER_PREFERENCE")

#: What the owner picks in the dialog, in the owner's words, and the class it
#: becomes.  The owner writes one sentence; the category says which system
#: learns from it (the recogniser's vocabulary, the router, the resolver, the
#: verifier, the lexicon) so the same mistake is not repeated.
OWNER_CATEGORIES: dict[str, str] = {
    "MISHEARD": STT_CORRECTION,
    "WRONG_INTENT": "INTENT_ERROR",
    "WRONG_TARGET": "ENTITY_RESOLUTION_ERROR",
    "WRONG_RESULT": "VERIFICATION_DEFECT",
    "INCOMPLETE": "PARAMETER_ERROR",
    "PRONUNCIATION": PRONUNCIATION,
    "OTHER": "OWNER_PREFERENCE",
}

_HEARD_MEANT = (
    re.compile(r"(?:nicht|not)\s+[„\"“']?(?P<heard>[^„\"“”'\s,]+)[“\"”']?\s*[,]?\s*(?:sondern|but)\s+[„\"“']?(?P<meant>[^„\"“”'.!?]+)", re.I),
    re.compile(r"[„\"“']?(?P<heard>[^„\"“”'\s]+)[“\"”']?\s*(?:->|→|=>|heisst|heißt|means|ist|is)\s+[„\"“']?(?P<meant>[^„\"“”'.!?]+)", re.I),
    re.compile(r"(?:ich\s+meinte|i\s+meant|gemeint\s+war|ich\s+sagte|i\s+said)\s+[„\"“']?(?P<meant>[^„\"“”'.!?]+)", re.I),
)


def heard_meant_pair(text: str, *, heard_text: str = "") -> tuple[str, str] | None:
    """(heard, meant) from the owner's sentence; the heard token is looked up in the transcript when the sentence names only what was meant."""

    from difflib import SequenceMatcher

    for pattern in _HEARD_MEANT:
        m = pattern.search(text or "")
        if not m:
            continue
        meant = m.group("meant").strip(" .!?\"'„“”")
        heard = (m.groupdict().get("heard") or "").strip(" .!?\"'„“”")
        if heard and meant and heard.lower() != meant.lower():
            return heard, meant
        if meant and heard_text:
            best, score = "", 0.0
            for token in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", heard_text, re.UNICODE):
                if token.lower() == meant.lower():
                    continue
                ratio = SequenceMatcher(None, token.lower(), meant.lower()).ratio()
                if ratio > score:
                    best, score = token, ratio
            if best and score >= 0.5 and len(best) >= 3:
                return best, meant
    return None

#: What a correction may change about a plan, by parameter name.
OVERRIDABLE = ("path", "directory", "folder", "provider", "format", "language", "output", "device")

_STOP = {
    "der", "die", "das", "ein", "eine", "einen", "und", "oder", "nicht", "ist", "war", "ich", "du", "es", "zu",
    "the", "a", "an", "and", "or", "not", "is", "was", "i", "you", "it", "to", "of", "in", "im", "bitte",
    "please", "mir", "mich", "dir", "dich", "meinte", "meant", "immer", "always", "nur", "only", "mal", "so",
    "dass", "that", "this", "diesmal", "künftig", "kuenftig", "future", "from", "now", "on", "ab", "jetzt",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-ZäöüÄÖÜß_][\w\-]{2,}", (text or "").lower()) if w not in _STOP]


@dataclass
class OwnerCorrection:
    original_request: str
    what_was_wrong: str
    classification: str
    scope: str
    parsed_intent: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    executed_action: str = ""
    observed_result: str = ""
    receipt_id: str = ""
    #: The reusable rule: ``when`` names what must be present in a future
    #: request for this to apply; ``then`` is guidance and/or overrides.
    when: dict[str, Any] = field(default_factory=dict)
    then: dict[str, Any] = field(default_factory=dict)
    correction_id: str = field(default_factory=lambda: f"corr_{uuid.uuid4().hex[:10]}")
    at: str = field(default_factory=_now)
    provenance: str = "owner-ui"
    active: bool = True
    applied_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OwnerCorrection":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

_ALWAYS = ("immer", "künftig", "kuenftig", "ab jetzt", "grundsätzlich", "grundsaetzlich", "in zukunft",
           "always", "from now on", "in future", "in the future", "every time", "jedes mal")
_ONCE = ("nur diesmal", "nur dieses mal", "nur jetzt", "diesmal", "just this once", "only this time", "this time only")
_FAILED = ("hat nicht funktioniert", "funktioniert nicht", "ist abgestürzt", "abgestuerzt", "fehler", "crash",
           "did not work", "didn't work", "failed", "error", "broken", "kaputt")
_WRONG_CHECK = ("hast behauptet", "angeblich", "stimmt nicht", "war gar nicht", "nicht wirklich", "claimed",
                "said it was", "wasn't actually", "was not actually", "not really")
_WRONG_OBJECT = ("falsche datei", "falscher ordner", "falsches lied", "falscher track", "falsche app",
                 "falsches programm", "das falsche", "die falsche", "den falschen", "wrong file", "wrong folder",
                 "wrong track", "wrong song", "wrong app", "wrong one", "not that one", "nicht das")
_WRONG_KIND = ("ich meinte", "ich wollte", "das war keine", "das war kein", "sollte nicht", "nicht erstellen",
               "i meant", "i wanted", "was not asking", "wasn't asking", "should not have", "shouldn't have",
               "das war eine frage", "that was a question")
_CAPABILITY = ("die fähigkeit", "die faehigkeit", "der provider", "spotify", "screenshot", "capability", "the tool",
               "das tool", "die funktion")
_PARAM = ("pfad", "ordner", "verzeichnis", "format", "name", "dateiname", "endung", "sprache", "path", "folder",
          "directory", "filename", "extension", "language", "lautstärke", "volume")

_DOMAINS = {
    "files": ("datei", "dateien", "ordner", "notiz", "notizen", "dokument", "file", "files", "folder", "note", "notes", "document"),
    "music": ("musik", "lied", "song", "track", "playlist", "spotify", "music", "play", "spiel", "abspielen"),
    "system": ("screenshot", "bildschirm", "fenster", "programm", "app", "screen", "window", "gpu", "cpu"),
    "self": ("zeus", "du selbst", "yourself", "dein", "your"),
}


def domain_of(text: str) -> str:
    lowered = (text or "").lower()
    for name, words in _DOMAINS.items():
        if any(w in lowered for w in words):
            return name
    return "general"


_PRONOUNCE = re.compile(r"(ausspr\w*|pronounc\w*|\bsprich\b.*\baus\b|\bsag\b.*\bwie\b|betonst|betonung)", re.I)
_PRONOUNCE_PAIR = re.compile(
    r"[\"'„‚«]([^\"'“”‘’»]+)[\"'“”‘’»].{0,40}?(?:wie|as|like|so:?|als)\s*[\"'„‚«]([^\"'“”‘’»]+)[\"'“”‘’»]", re.I | re.S)


def pronunciation_pair(text: str) -> tuple[str, str] | None:
    """('X', 'Y') from "du sprichst 'X' falsch aus. Sprich es wie 'Y' aus." -- None when the sentence is not that."""

    if not _PRONOUNCE.search(text or ""):
        return None
    m = _PRONOUNCE_PAIR.search(text or "")
    if not m:
        return None
    surface, spoken = m.group(1).strip(), m.group(2).strip()
    return (surface, spoken) if surface and spoken else None


def classify_correction(what_was_wrong: str, *, receipt_ok: bool | None = None, request: str = "") -> tuple[str, str, str]:
    """(classification, scope, reason) from the owner's words, deterministically."""

    text = (what_was_wrong or "").lower()
    has = lambda words: any(w in text for w in words)  # noqa: E731

    if pronunciation_pair(what_was_wrong):
        return PRONUNCIATION, "GLOBAL_OWNER_PREFERENCE", "a pronunciation correction: stored in the owner lexicon"

    if has(_ONCE):
        scope = "THIS_REQUEST"
    elif has(_ALWAYS):
        scope = "GLOBAL_OWNER_PREFERENCE" if not has(_PARAM) and domain_of(request) == "general" else "DOMAIN_SPECIFIC"
    elif has(_WRONG_OBJECT):
        scope = "ENTITY_SPECIFIC"
    else:
        scope = "INTENT_SPECIFIC"

    # The owner's words decide. A failed receipt only tips the balance when
    # the sentence itself carries no rule -- "künftig immer" is a preference
    # whatever the last attempt did.
    states_rule = has(_ALWAYS) or has(_ONCE) or has(_PARAM) or has(_WRONG_KIND) or has(_WRONG_OBJECT)
    if has(_FAILED) or (receipt_ok is False and not states_rule):
        return "EXECUTION_FAILURE", "THIS_REQUEST", "the owner describes a failure of the tool, not of the reading"
    if has(_WRONG_CHECK):
        return "VERIFICATION_DEFECT", scope, "the owner says the check accepted something false"
    if has(_CAPABILITY) and (has(_FAILED) or "falsch" in text or "wrong" in text):
        return "CAPABILITY_DEFECT", "INTENT_SPECIFIC", "the owner blames the capability's behaviour"
    if has(_WRONG_KIND):
        return "INTENT_ERROR", scope, "the owner says the request was read as the wrong kind of thing"
    if has(_WRONG_OBJECT):
        return "ENTITY_RESOLUTION_ERROR", "ENTITY_SPECIFIC", "the owner names the wrong object"
    if has(_ALWAYS):
        return "OWNER_PREFERENCE", scope, "the owner states how it should be from now on"
    if has(_PARAM):
        return "PARAMETER_ERROR", scope, "the owner names a detail that was wrong"
    return "OWNER_PREFERENCE", scope, "no error vocabulary; recorded as a preference"


def rule_for(what_was_wrong: str, *, request: str, classification: str, scope: str,
             parsed_intent: str = "", entities: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``when``/``then`` a future interpretation checks and applies."""

    when: dict[str, Any] = {"domain": domain_of(request)}
    if scope == "INTENT_SPECIFIC" and parsed_intent:
        when["intent"] = parsed_intent
    if scope == "ENTITY_SPECIFIC":
        when["terms"] = sorted(set(_terms(request)) & set(_terms(what_was_wrong))) or _terms(request)[:4]
    if scope == "GLOBAL_OWNER_PREFERENCE":
        when = {}
    then: dict[str, Any] = {"note": what_was_wrong.strip()}
    overrides = _extract_overrides(what_was_wrong)
    if overrides:
        then["overrides"] = overrides
    return when, then


#: The owner naming what kind of request it was.  Stored as an ``intent``
#: override, which :func:`service.routing.route` applies before it decides.
_INTENT_WORDS = (
    ("self_development", re.compile(
        r"(selbstentwicklung|self[-\s]?development|an dir selbst|an zeus selbst|(?:aenderung|änderung) an zeus|dich selbst (?:aendern|ändern|verbessern)"
        r"|(?:eine|die) (?:aenderung|änderung|änderung) an dir|change (?:to )?yourself|modify yourself|kein(?:e)? (?:lied|song|musik)|not (?:a )?(?:song|music))", re.I)),
    ("real_world_action", re.compile(r"(einfach ausfuehren|einfach ausführen|eine aktion|just do it|an action, not)", re.I)),
    ("conversation", re.compile(r"(nur eine frage|just a question|only a question|nur reden)", re.I)),
    ("capability_acquisition", re.compile(r"(neue faehigkeit|neue fähigkeit|new capability|lernen sollst|should learn)", re.I)),
)

_OVERRIDE_PATTERNS = (
    (("path", "directory", "folder"), re.compile(
        r"(?:ordner|verzeichnis|folder|directory|unter|into|in den ordner|nach)\s+[\"'`]?([\w\-./\\]+?)[\"'`]?(?:[\s,.;]|$)", re.I)),
    (("format",), re.compile(r"(?:format|als|as)\s+(md|markdown|txt|json|csv|pdf|yaml)\b", re.I)),
    (("provider",), re.compile(r"\b(spotify|youtube|local|lokal)\b", re.I)),
)


def _extract_overrides(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for intent, pattern in _INTENT_WORDS:
        if pattern.search(text or ""):
            out["intent"] = intent
            break
    for names, pattern in _OVERRIDE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            value = match.group(1).strip().rstrip("/\\")
            if names[0] == "path":
                out["directory"] = value
            else:
                out[names[0]] = value.lower()
    return out


# --------------------------------------------------------------------------
# Store and retrieval
# --------------------------------------------------------------------------

class CorrectionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> list[OwnerCorrection]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(OwnerCorrection.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                continue
        return rows

    def _save(self, rows: Iterable[OwnerCorrection]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, correction: OwnerCorrection) -> OwnerCorrection:
        with self._lock:
            rows = self._load()
            rows.append(correction)
            self._save(rows)
        return correction

    def list(self, *, include_inactive: bool = False) -> list[OwnerCorrection]:
        with self._lock:
            rows = self._load()
        return [r for r in rows if include_inactive or r.active]

    def get(self, correction_id: str) -> OwnerCorrection | None:
        return next((r for r in self.list(include_inactive=True) if r.correction_id == correction_id), None)

    def update(self, correction_id: str, **changes: Any) -> OwnerCorrection | None:
        with self._lock:
            rows = self._load()
            found = None
            for row in rows:
                if row.correction_id == correction_id:
                    for key, value in changes.items():
                        if key in row.__dataclass_fields__ and key not in {"correction_id", "at", "provenance"}:
                            setattr(row, key, value)
                    found = row
            if found is not None:
                self._save(rows)
        return found

    def delete(self, correction_id: str) -> bool:
        with self._lock:
            rows = self._load()
            kept = [r for r in rows if r.correction_id != correction_id]
            if len(kept) == len(rows):
                return False
            self._save(kept)
        return True

    # -- retrieval, before interpretation --------------------------------

    def relevant(self, request: str, *, intent: str = "", limit: int = 5) -> list[OwnerCorrection]:
        """Corrections whose scope reaches this request.  Narrow first."""

        text = (request or "").lower()
        terms = set(_terms(request))
        domain = domain_of(request)
        matched: list[tuple[int, OwnerCorrection]] = []
        for row in self.list():
            if row.scope == "THIS_REQUEST" or row.classification in {"EXECUTION_FAILURE"}:
                continue
            when = row.when or {}
            if row.scope == "GLOBAL_OWNER_PREFERENCE":
                matched.append((1, row))
                continue
            if when.get("domain") and when["domain"] != domain:
                continue
            if row.scope == "DOMAIN_SPECIFIC":
                matched.append((2, row))
                continue
            if row.scope == "INTENT_SPECIFIC":
                if not when.get("intent") or when["intent"] == intent:
                    matched.append((3, row))
                continue
            if row.scope == "ENTITY_SPECIFIC":
                needed = set(when.get("terms") or [])
                if needed and (needed & terms or any(t in text for t in needed)):
                    matched.append((4, row))
        matched.sort(key=lambda item: (-item[0], item[1].at), reverse=False)
        # Narrowest scope wins ties; most recent within a scope.
        matched.sort(key=lambda item: (item[0], item[1].at), reverse=True)
        return [row for _, row in matched[:limit]]

    def note_applied(self, corrections: Iterable[OwnerCorrection]) -> None:
        ids = {c.correction_id for c in corrections}
        if not ids:
            return
        with self._lock:
            rows = self._load()
            for row in rows:
                if row.correction_id in ids:
                    row.applied_count += 1
            self._save(rows)


def guidance_lines(corrections: Iterable[OwnerCorrection]) -> list[str]:
    """What the model is told, verbatim from the owner, with its scope."""

    lines = []
    for row in corrections:
        lines.append(f"- [{row.scope.lower().replace('_', ' ')}] {row.then.get('note', '')}")
    return lines


WRITE_ACTIONS = {"file.write", "note.create", "project.create", ""}


def apply_overrides(arguments: dict[str, Any], corrections: Iterable[OwnerCorrection], *, action: str = "") -> tuple[dict[str, Any], list[str]]:
    """Owner overrides applied to a plan's arguments.  Returns (arguments, applied).

    ``action`` keeps a correction inside its meaning: "notes go into
    notizen/" moves the *creation* of a note, not the reading of an unrelated
    file -- live, a directory override sent ``file.read plan.txt`` to
    ``notizen/plan.txt`` and failed.
    """

    updated = dict(arguments)
    applied: list[str] = []
    for row in corrections:
        overrides = dict((row.then or {}).get("overrides") or {})
        if not overrides:
            continue
        directory = overrides.pop("directory", None)
        if directory and updated.get("path") and action in WRITE_ACTIONS:
            name = str(updated["path"]).replace("\\", "/").rsplit("/", 1)[-1]
            updated["path"] = f"{directory.strip('/').replace(chr(92), '/')}/{name}"
            applied.append(f"{row.correction_id}: directory={directory}")
        for key, value in overrides.items():
            if key in OVERRIDABLE:
                updated[key] = value
                applied.append(f"{row.correction_id}: {key}={value}")
    return updated, applied
