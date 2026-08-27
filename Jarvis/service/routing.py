"""Top-level routing: *what kind of thing is being asked* before *which tool*.

The failure this module answers, observed twice through the live product: the
owner wrote a long German paragraph asking ZEUS to repair its own desktop
lifecycle, and the sentence "Wenn ich ZEUS.exe starte" contains the word
"start".  The music parser ran before anything else had looked at the request,
matched "start" as a play verb, and the whole paragraph went to Spotify as a
search query.  Spotify answered 400.

The defect was structural rather than a missing keyword: domain parsers were
being consulted before anyone had decided what the request *was*.  A Spotify
verb list has no business deciding whether a request is self-development.

So routing is now two stages, in a fixed order:

1. **Top level** (this module) -- decided from two things that every request
   has: an OPERATION (act, modify, learn, ask, correct, research) and an
   OBJECT (the world, or ZEUS itself, or ZEUS's protected owner core).
   "Play a song" is *act* on *world*.  "Improve how you choose songs" is
   *modify* on *self*.  Both mention songs; only the second is self-
   development, and music vocabulary plays no part in telling them apart.
2. **Domain** (:mod:`service.intent`) -- only after the top level says the
   request is a real-world action or plain conversation does any domain
   parser (music, files, ...) get to look at it.

Everything here is deterministic and cheap: it runs on every message, and a
model call in front of "Pause." would cost more than the mistake it prevents.
Owner corrections are consulted *before* the final choice, so an owner who has
said "that was a change to yourself, not a song" once is not made to say it
again.  The residual risk is stated rather than hidden: a phrasing this module
has never seen can still be misread, which is why the domain layer and the
claim guard remain behind it, and why every decision is recorded with its
reason so the owner can correct it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class TopLevelIntent(str, Enum):
    """What the request is, before any domain has been chosen."""

    CONVERSATION = "conversation"
    REAL_WORLD_ACTION = "real_world_action"
    SELF_DEVELOPMENT = "self_development"
    CAPABILITY_ACQUISITION = "capability_acquisition"
    CAPABILITY_REPAIR = "capability_repair"
    OWNER_CONFIG_CHANGE = "owner_config_change"
    OWNER_CORRECTION = "owner_correction"
    PROJECT = "project"
    RESEARCH = "research"
    COMPLEX_MISSION = "complex_mission"

    @property
    def modifies_zeus(self) -> bool:
        return self in {TopLevelIntent.SELF_DEVELOPMENT, TopLevelIntent.CAPABILITY_REPAIR,
                        TopLevelIntent.OWNER_CONFIG_CHANGE}

    @property
    def domain_eligible(self) -> bool:
        """Whether a domain parser (music, files, ...) may claim this request.

        Media routing only becomes possible once the request is known to be an
        action on the world or ordinary conversation.  A self-modification,
        an acquisition, a project or a correction is never a song.
        """

        return self in {TopLevelIntent.REAL_WORLD_ACTION, TopLevelIntent.CONVERSATION}


@dataclass(frozen=True)
class Reading:
    """The object/operation analysis a route is derived from."""

    #: act | modify | learn | ask | correct | research | none
    operation: str
    #: self | self_core | self_capability | world | none
    object: str
    self_score: int = 0
    world_score: int = 0
    is_question: bool = False
    #: The phrases that decided it, for Activity and for tuning.
    self_refs: tuple[str, ...] = ()
    modify_verbs: tuple[str, ...] = ()
    action_verbs: tuple[str, ...] = ()
    world_objects: tuple[str, ...] = ()
    core_terms: tuple[str, ...] = ()
    capability_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation, "object": self.object,
            "self_score": self.self_score, "world_score": self.world_score,
            "is_question": self.is_question,
            "self_refs": list(self.self_refs)[:6], "modify_verbs": list(self.modify_verbs)[:6],
            "action_verbs": list(self.action_verbs)[:6], "world_objects": list(self.world_objects)[:6],
            "core_terms": list(self.core_terms)[:4], "capability_terms": list(self.capability_terms)[:4],
        }


@dataclass(frozen=True)
class Route:
    intent: TopLevelIntent
    #: high | medium | low
    confidence: str
    reason: str
    reading: Reading
    #: Signals that pointed elsewhere and were overruled, with why.
    conflicts: tuple[str, ...] = ()
    #: Owner corrections that forced or confirmed the route.
    corrections: tuple[str, ...] = ()
    #: Whether an owner correction overrode the analysis.
    forced_by_owner: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_level": self.intent.value, "confidence": self.confidence, "reason": self.reason,
            "reading": self.reading.to_dict(), "conflicts": list(self.conflicts),
            "corrections": list(self.corrections), "forced_by_owner": self.forced_by_owner,
        }


# --------------------------------------------------------------------------
# Vocabulary.  Stems and shapes, never exact sentences.
# --------------------------------------------------------------------------

def fold(text: str) -> str:
    lowered = (text or "").lower().replace("ß", "ss")
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(source, target)
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


#: The owner addressing ZEUS is not the owner talking *about* ZEUS.  "Zeus, spiel
#: ..." is a vocative and carries no self-reference.
_VOCATIVE = re.compile(r"^\s*(hey\s+|ok\s+|okay\s+)?(zeus|jarvis)\s*[,:!.\-]?\s*", re.I)

#: ZEUS as the object of a sentence: possessives, reflexives, its own name used
#: as a thing rather than an address, its executable.
_SELF_REFERENCE = re.compile(
    r"\b(dein|deine|deinen|deinem|deiner|deines|dich|dir|dich\s+selbst|dir\s+selbst|selbst\s+aendern"
    r"|your|yours|yourself|zeus\.exe|zeus\s+selbst|an\s+zeus|in\s+zeus|zeus'?s|zeus-)\b"
)

#: "dass du", "wie du", "the way you" become a self-reference only together with
#: a behaviour-change signal: "ich möchte, dass du mir eine Datei anlegst" is a
#: request; "ich möchte, dass du künftig kürzer antwortest" is a change to ZEUS.
_SELF_BEHAVIOUR = re.compile(r"\b(dass\s+du|wie\s+du|wenn\s+du|the\s+way\s+you|how\s+you|when\s+you|that\s+you)\b")
_BEHAVIOUR_CHANGE = re.compile(
    r"\b(kuenftig|zukuenftig|in\s+zukunft|ab\s+jetzt|ab\s+sofort|von\s+nun\s+an|immer|nie\s+wieder|nicht\s+mehr"
    r"|from\s+now\s+on|in\s+future|going\s+forward|always|never\s+again|no\s+longer|should|sollst|sollte|soll|sollen)\b"
)

#: Operations that change something that already exists.  Stems, so that
#: "ändere", "änderst", "geändert" all count.
_MODIFY = re.compile(
    r"\b(aender\w*|veraender\w*|umbau\w*|umgestalt\w*|ueberarbeit\w*|repar\w*|behebe?\w*|fix\w*|verbesser\w*"
    r"|optimier\w*|erweiter\w*|entwickl\w*|implementier\w*|einbau\w*|anpass\w*|ersetz\w*|entfern\w*|streich\w*"
    r"|ergaenz\w*|hinzufueg\w*|korrigier\w*|beschleunig\w*|refactor\w*|redesign\w*|umschreib\w*|neu\s+schreib\w*"
    r"|change|changes|changing|modify|modif\w*|improve\w*|repair\w*|extend\w*|rewrite|rework\w*|update\w*"
    r"|upgrade\w*|add|adding|remove|removing|replace|replacing|implement\w*|develop\w*|build\s+in|make\s+(?:it|this|that|yourself|your))\b"
)
#: "füge ... hinzu", "baue ... ein/um", "passe ... an": separable verbs whose
#: particle lands at the end of the clause.
_MODIFY_SEPARABLE = re.compile(
    r"\b(fueg\w*|bau\w*|pass\w*|stell\w*|schalt\w*)\b[^.!?\n]{0,80}?\b(hinzu|ein|um|an|ab|aus)\b"
)

#: Operations on the world that create/act rather than change ZEUS.
_ACT = re.compile(
    r"\b(erstell\w*|erzeug\w*|leg\w*|anleg\w*|ableg\w*|schreib\w*|speicher\w*|sicher\w*|loesch\w*|benenn\w*|oeffne\w*|zeig\w*|starte?|spiel\w*"
    r"|mach\w*|schick\w*|send\w*|such\w*|finde?|lade?|download\w*|kopier\w*|verschieb\w*|druck\w*|installier\w*"
    r"|create|make|write|save|store|delete|open|show|play|put\s+on|send|search|find|copy|move|print|install|take|record)\b"
)

#: "learn to", "teach yourself": acquisition.  Checked before self-development
#: because "bring dir bei" contains "dir".
_LEARN = re.compile(
    r"\b(lerne?(?=\s*[,:]|\s+(?:wie|zu|es|das|was|selbst|neue)|\s*[.!]?\s*$)|lern\s+(?:wie|zu|es|das)|bring\w*\s+dir\s+bei|beibring\w*|eigne\s+dir\s+an|aneign\w*|learn\s+(?:how\s+)?to|teach\s+yourself"
    r"|acquire\s+(?:the\s+)?(?:ability|capability|skill)|new\s+capability|neue\s+faehigkeit)\b"
)

_RESEARCH = re.compile(
    r"\b(recherchier\w*|recherche|finde\s+heraus|find\s+out|research|investigate|untersuch\w*|vergleiche\s+quellen"
    r"|was\s+ist\s+der\s+aktuelle\s+stand|aktuelle\w*\s+stand|latest\s+news|neueste\w*\s+erkenntnis\w*|quellen\s+suchen|look\s+up)\b"
)

_CORRECTION = re.compile(
    r"^\s*(nein[,.!]?\s|falsch[,.!]?\s|das\s+war\s+falsch|das\s+ist\s+falsch|so\s+nicht|nicht\s+so|ich\s+meinte"
    r"|das\s+meinte\s+ich\s+nicht|korrektur[:\s]|korrigiere[:\s]|no[,.!]?\s+that|that\s+was\s+wrong|wrong[,.!]?\s|i\s+meant|correction[:\s])"
)

#: ZEUS's own parts.  Alone these are ordinary nouns ("Fenster"); next to a
#: self-reference they name the thing to be changed.
_SELF_COMPONENTS = re.compile(
    r"\b(oberflaeche|ui|interface|fenster|window|auge|eye|startvorgang|start-?up|boot|supervisor|core|kern|lifecycle"
    r"|wakeword|wake-?word|listener|worker|router|routing|selfdev|self-?development|code|quellcode|codebase|activity"
    r"|projects|projekte-ansicht|knowledge|wissensgraph|diagnose|diagnostics|einstellungen|settings|header|kopfzeile"
    r"|antwort\w*|stimme|voice|verhalten|behaviou?r|funktion\w*|feature\w*|modul\w*|logik|antwortzeit|latenz|prompt\w*"
    r"|persoenlichkeit|personality|identitaet|identity)\b"
)

#: The owner core: changes here are a protected owner transaction, never a
#: SelfDev mission.
_OWNER_CORE = re.compile(
    r"\b(kern-?persoenlichkeit|persoenlichkeit|charakter|identitaet|identity|personality|core\s+personality"
    r"|owner[-\s]?policy|richtlinie\w*|policy|policies|ausgaben\w*|budget|spending|kosten-?regel\w*"
    r"|sicherheitsrichtlinie\w*|security\s+policy|berechtigungen|permissions|antivirus|virenscanner)\b"
)

#: Things in the world a request can act on.  A possessive in front of one of
#: these ("deine Screenshot-Funktion") turns it into a part of ZEUS instead.
_WORLD_OBJECTS = re.compile(
    r"\b(datei\w*|ordner|verzeichnis\w*|notiz\w*|dokument\w*|lied\w*|song\w*|musik|playlist\w*|album\w*|track\w*"
    r"|screenshot\w*|bildschirmfoto\w*|bildschirm|foto\w*|bild\w*|e-?mail\w*|mail\w*|kalender|termin\w*|erinnerung\w*"
    r"|timer|wecker|archiv\w*|zip|app|programm\w*|skript\w*|script\w*|website\w*|webseite\w*|browser|tab|fenster"
    r"|file\w*|folder\w*|directory|note\w*|document\w*|music|photo\w*|picture\w*|image\w*|calendar|reminder\w*|alarm)\b"
)

#: Acquired capabilities by common name.  The registry's real names are passed
#: in by the caller; these cover how an owner would refer to them.
_CAPABILITY_TERMS = re.compile(
    r"\b(spotify|screenshot\w*|bildschirmfoto\w*|screen\s+capture|zip|archiv\w*|musik-?funktion|music\s+capability"
    r"|faehigkeit\w*|capability|capabilities|skill\w*)\b"
)
_REPAIR = re.compile(r"\b(repar\w*|behebe?\w*|fix\w*|repair\w*|kaputt|defekt|broken|funktioniert\s+nicht|geht\s+nicht|does\s+not\s+work|doesn'?t\s+work|fails?|schlaegt\s+fehl)\b")

#: Durable engineering on something that is not ZEUS.
_PROJECT = re.compile(
    r"\b(programmier\w*|baue?\s+(?:mir\s+)?(?:eine|ein|an|a)\s|schreibe?\s+(?:mir\s+)?(?:ein\s+)?(?:programm|skript|script|code)"
    r"|build\s+(?:me\s+)?(?:an?\s+)?(?:app|tool|program|script|website)|write\s+(?:me\s+)?(?:a\s+)?(?:program|script)"
    r"|implementier\w*|implement|refactor\w*|debugg?\w*|fix\s+the\s+bug|fixe\s+den\s+bug|behebe\s+den\s+bug|run\s+the\s+tests)\b"
)

_COMPARATIVE = re.compile(
    r"\b(groesser|kleiner|schneller|langsamer|besser|leiser|lauter|heller|dunkler|kuerzer|laenger|ruhiger|schlichter"
    r"|bigger|larger|smaller|faster|slower|better|quieter|louder|brighter|darker|shorter|longer|calmer|simpler)\b"
)
_DISPLAY = re.compile(r"\b(zeig\w*|anzeig\w*|blende?\w*|show|display|render|put)\b")
_PLACEMENT = re.compile(
    r"\b(neben\s+dein\w*|in\s+dein\w*|an\s+dein\w*|unter\s+dein\w*|ueber\s+dein\w*|in\s+der\s+kopfzeile|im\s+header"
    r"|next\s+to\s+your|in\s+your|on\s+your|beside\s+your|under\s+your|above\s+your|in\s+the\s+header)\b"
)

_INTERROGATIVE = re.compile(
    r"^(wie|was|warum|wieso|weshalb|wann|wo|wer|welche[srn]?|wieviel|kannst\s+du|koenntest\s+du|weisst\s+du"
    r"|how|what|why|when|where|who|which|can\s+you|could\s+you|do\s+you)\b"
)

#: A request that names a whole pipeline of stages is a mission even when it
#: names no project verb.
_MISSION_SHAPE = re.compile(r"\b(schritt\s+fuer\s+schritt|step\s+by\s+step|plane?\s+und|plan\s+and|mehrstufig|multi-?step|langfristig|long-?term)\b")


def _findall(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    out: list[str] = []
    for match in pattern.finditer(text):
        piece = match.group(0).strip()
        if piece and piece not in out:
            out.append(piece)
    return tuple(out)


def read(text: str, *, capability_names: Iterable[str] = ()) -> Reading:
    """The object/operation analysis, with every signal that fed it."""

    folded = fold(text).strip()
    body = _VOCATIVE.sub("", folded, count=1)
    stripped = body.rstrip()
    is_question = stripped.endswith("?") or bool(_INTERROGATIVE.match(stripped)) and not stripped.endswith((".", "!"))

    self_refs = list(_findall(_SELF_REFERENCE, body))
    # "bring dir bei" is acquisition, and its "dir" is not a self-reference.
    if _LEARN.search(body):
        self_refs = [r for r in self_refs if r != "dir"]
    behaviour = _BEHAVIOUR_CHANGE.search(body) is not None

    modify_verbs = list(_findall(_MODIFY, body)) + list(_findall(_MODIFY_SEPARABLE, body))
    # "mach dein Auge groesser", "make your answers shorter": a comparative on a
    # part of ZEUS is a modification even though "mach" is an act verb.
    if _SELF_REFERENCE.search(body) and _COMPARATIVE.search(body):
        modify_verbs.append(_COMPARATIVE.search(body).group(0))
    # "how you choose songs", "dass du kuenftig ...": ZEUS's behaviour is the
    # object when something is to change about it.
    if behaviour or modify_verbs:
        self_refs.extend(_findall(_SELF_BEHAVIOUR, body))
    # "show X next to your eye", "zeig ... in deiner Kopfzeile": a display placed
    # into a part of ZEUS is a change to ZEUS, not a thing shown to the owner.
    placement = _PLACEMENT.search(body)
    if placement and _DISPLAY.search(body) and not modify_verbs:
        modify_verbs.append(placement.group(0))
    action_verbs = list(_findall(_ACT, body))
    core_terms = _findall(_OWNER_CORE, body)
    capability_terms = list(_findall(_CAPABILITY_TERMS, body))
    for name in capability_names:
        token = fold(str(name)).split(".")[-1]
        if token and len(token) > 3 and re.search(rf"\b{re.escape(token)}\b", body):
            capability_terms.append(token)

    # World objects owned by ZEUS ("deine Screenshot-Funktion", "your eye") are
    # parts of ZEUS; the rest are things in the world.
    owned = re.compile(r"\b(dein\w*|your|zeus'?s)\s+(?:\w+[\s-]+){0,2}?(\w+)")
    owned_words = {m.group(2) for m in owned.finditer(body)}
    world_objects = tuple(w for w in _findall(_WORLD_OBJECTS, body) if w not in owned_words)
    components = _findall(_SELF_COMPONENTS, body)

    self_score = 0
    self_score += 2 * min(len([r for r in self_refs if r not in {"dass du", "wie du", "wenn du", "the way you", "how you", "when you", "that you"}]), 4)
    self_score += 1 * min(len([r for r in self_refs if r in {"dass du", "wie du", "wenn du", "the way you", "how you", "when you", "that you"}]), 2)
    if self_refs and components:
        self_score += 1
    if "zeus.exe" in self_refs or "zeus selbst" in self_refs:
        self_score += 2
    world_score = min(len(world_objects), 4)

    if _CORRECTION.match(body):
        operation = "correct"
    elif _LEARN.search(body) and not (self_refs and modify_verbs and _MODIFY.search(body) and "lern" not in body[:30]):
        operation = "learn"
    elif modify_verbs and (self_score > 0 or not action_verbs):
        operation = "modify"
    elif behaviour and self_refs and not is_question:
        operation = "modify"
    elif _RESEARCH.search(body) and not action_verbs:
        operation = "research"
    elif action_verbs or modify_verbs:
        operation = "act"
    elif is_question:
        operation = "ask"
    else:
        operation = "none"

    if operation in {"modify", "learn"} and self_score > 0 and self_score >= world_score:
        if core_terms and operation == "modify":
            obj = "self_core"
        elif capability_terms and operation == "modify":
            obj = "self_capability"
        else:
            obj = "self"
    elif self_score > 0 and self_score > world_score and operation == "act" and components and not world_objects:
        # "Zeig dein Auge größer" -- an imperative on a part of ZEUS.
        obj = "self"
    elif world_objects or action_verbs:
        obj = "world"
    elif self_score > 0 and operation in {"ask", "none"}:
        obj = "self"
    else:
        obj = "none"

    return Reading(
        operation=operation, object=obj, self_score=self_score, world_score=world_score,
        is_question=is_question, self_refs=tuple(self_refs), modify_verbs=tuple(modify_verbs),
        action_verbs=tuple(action_verbs), world_objects=world_objects, core_terms=core_terms,
        capability_terms=tuple(capability_terms),
    )


#: Owner-correction override values that name a top-level route.
INTENT_OVERRIDES = {member.value: member for member in TopLevelIntent}


def route(text: str, *, corrections: Iterable[Any] = (), capability_names: Iterable[str] = ()) -> Route:
    """The top-level intent of a request, decided before any domain parser runs."""

    reading = read(text, capability_names=capability_names)
    conflicts: list[str] = []
    applied: list[str] = []

    # Owner corrections first: an explicit "intent" override wins outright and
    # says so.  Everything else in the correction store is about arguments and
    # is applied downstream, after the route.
    forced: TopLevelIntent | None = None
    for row in corrections:
        overrides = dict(((getattr(row, "then", None) or {}).get("overrides") or {}))
        wanted = INTENT_OVERRIDES.get(str(overrides.get("intent", "")).lower())
        if wanted is not None:
            forced = wanted
            applied.append(getattr(row, "correction_id", "?"))
            break

    intent, confidence, reason = _decide(reading, conflicts)

    if forced is not None and forced is not intent:
        conflicts.append(f"analysis said {intent.value}; owner correction {applied[-1]} forces {forced.value}")
        return Route(forced, "high", f"owner correction {applied[-1]} forces {forced.value}", reading,
                     tuple(conflicts), tuple(applied), forced_by_owner=True)
    return Route(intent, confidence, reason, reading, tuple(conflicts), tuple(applied), forced_by_owner=False)


def _decide(r: Reading, conflicts: list[str]) -> tuple[TopLevelIntent, str, str]:
    self_target = r.object in {"self", "self_core", "self_capability"}

    if r.operation == "correct":
        return TopLevelIntent.OWNER_CORRECTION, "medium", "the owner is correcting a previous reading"

    if r.operation == "learn":
        if self_target and r.modify_verbs and r.object == "self":
            conflicts.append("learn-verb with a self-modification; acquisition kept because learning was the head verb")
        return TopLevelIntent.CAPABILITY_ACQUISITION, "high" if not r.is_question else "low", \
            "asks to acquire an ability it does not have"

    if r.operation == "modify" and self_target and not r.is_question:
        if r.world_objects:
            conflicts.append(
                f"world objects {list(r.world_objects)[:3]} were mentioned, but ZEUS itself is the target "
                f"(self {r.self_score} >= world {r.world_score}); a domain capability does not override self-development"
            )
        if r.object == "self_core":
            return TopLevelIntent.OWNER_CONFIG_CHANGE, "high", \
                f"asks to change the owner core ({', '.join(r.core_terms[:2])}): protected owner transaction"
        if r.object == "self_capability" and r.capability_terms and any(_REPAIR.search(v) for v in r.modify_verbs):
            return TopLevelIntent.CAPABILITY_REPAIR, "high", \
                f"asks to repair its own capability ({', '.join(r.capability_terms[:2])})"
        confidence = "high" if (r.self_score >= 2 and r.self_score > r.world_score) else "medium"
        return TopLevelIntent.SELF_DEVELOPMENT, confidence, \
            f"asks to change ZEUS itself: {', '.join(r.modify_verbs[:2]) or 'behaviour change'} on {', '.join(r.self_refs[:2])}"

    if r.operation == "modify" and self_target and r.is_question:
        conflicts.append("a question about changing ZEUS is answered, not started as a mission")
        return TopLevelIntent.CONVERSATION, "medium", "a question about ZEUS changing itself, not an instruction"

    if r.operation == "research":
        return TopLevelIntent.RESEARCH, "medium", "asks to find something out from sources"

    if r.operation in {"act", "modify"}:
        if _PROJECT.search(fold(" ".join(r.action_verbs + r.modify_verbs))) or (r.modify_verbs and not r.world_objects and r.object == "world" and r.self_score == 0 and any(m.startswith(("implement", "refactor", "debug", "programmier")) for m in r.modify_verbs)):
            return TopLevelIntent.PROJECT, "medium", "describes durable engineering work on something that is not ZEUS"
        if r.self_score > 0 and r.world_score >= r.self_score:
            conflicts.append(
                f"self-references {list(r.self_refs)[:2]} present, but the request acts on the world "
                f"({list(r.world_objects)[:2]}); routed as an action"
            )
        return TopLevelIntent.REAL_WORLD_ACTION, "high" if r.action_verbs else "medium", \
            f"acts on the world: {', '.join((r.action_verbs or r.modify_verbs)[:2])}"

    if r.operation == "ask" or r.operation == "none":
        return TopLevelIntent.CONVERSATION, "high" if not r.self_refs else "medium", \
            "a question or a remark with no side effect"

    return TopLevelIntent.CONVERSATION, "low", "no operation recognised"


def looks_like_prose(query: str, *, max_words: int = 12, max_chars: int = 120) -> str:
    """Why a string must not be sent to a media provider as a search, or ``""``.

    A provider-level guard, independent of the router above: whatever path a
    request took, a paragraph is not a track name.
    """

    text = (query or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        return f"{len(words)} words is a paragraph, not a track name"
    if len(text) > max_chars:
        return f"{len(text)} characters is a paragraph, not a track name"
    if "\n" in text:
        return "multi-line text is not a track name"
    sentences = [s for s in re.split(r"[.!?]\s", text) if s.strip()]
    if len(sentences) >= 2:
        return "several sentences are not a track name"
    if re.search(r"\b(wenn|dass|damit|sodass|soll|sollen|darf|duerfen|dürfen|muss|muessen|müssen|entwickle|verifiziere|teste|repariere|should|must|so that|whenever)\b", text, re.I):
        return "instruction vocabulary inside a search query"
    return ""
