"""Top-level intent and typed action contracts, decided before any domain parser.

Every accepted request becomes exactly one :class:`TopIntent` first:

    CONVERSATION  ACTION  PROJECT_OPERATION  MISSION  SELF_DEVELOPMENT
    CORRECTION  KNOWLEDGE_OPERATION  SYSTEM_CONTROL  CLARIFICATION

and, for the operations ZEUS can perform deterministically, an
:class:`ActionIntent` -- verb, object type, target, typed arguments,
constraints, success criteria, forbidden effects, confidence.  A contract
like that is what lets execution be deterministic ("project.create" with a
title and three tasks) and goal verification independent ("the project
persists with exactly those three tasks"), instead of a model narrating what
it might have done.

The live defect this answers: "Erstelle ein neues Projekt für meine
Prüfungsvorbereitung" was sometimes answered with prose, sometimes not
executed at all, because the only thing that could turn it into an action was
a 4B model asked for JSON.  Natural German project requests are now parsed
here, without a model, in every paraphrase the owner actually uses:

    Erstelle ein neues Projekt.                      -> CLARIFICATION (title)
    Mach mir ein Projekt für M1.                      -> create title=M1
    Lege ein neues Projekt Biochemie an.              -> create title=Biochemie
    Ich möchte ein neues Projekt für meinen Hausbau.  -> create title=Hausbau
    Erstell unter ZEUS ein Teilprojekt Voice.         -> create title=Voice parent=ZEUS
    Erstelle ein Projekt M1 und leg drei Aufgaben an: A, B, C -> create + tasks

Semantic purpose decides, not word overlap: "Was ist ein Projekt?" is a
question, "Verbessere deine Projektansicht" is self-development, and neither
reaches the project parser.  Text similarity is at most a hint for the
capability resolver; it never selects an operation here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class TopIntent(str, Enum):
    CONVERSATION = "conversation"
    ACTION = "action"
    PROJECT_OPERATION = "project_operation"
    MISSION = "mission"
    SELF_DEVELOPMENT = "self_development"
    CORRECTION = "correction"
    KNOWLEDGE_OPERATION = "knowledge_operation"
    SYSTEM_CONTROL = "system_control"
    CLARIFICATION = "clarification"


#: How much a mistake costs.  Speech uncertainty and action safety are
#: different axes; this is the safety one.
class Consequence(str, Enum):
    HARMLESS = "harmless"        # reversible or read-only
    MODERATE = "moderate"        # reversible with effort (archive, rename, move)
    IRREVERSIBLE = "irreversible"  # delete, send, pay


@dataclass
class ActionIntent:
    """The typed contract for one operation."""

    operation: str                  # e.g. project.create, project.open, system.open_view
    verb: str = ""                  # create | read | open | update | rename | delete | archive | add_tasks | stop | ...
    object_type: str = ""           # project | knowledge | view | music | screenshot | ...
    target: str = ""                # the thing acted on (a title, an id, a view name)
    arguments: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    forbidden_effects: list[str] = field(default_factory=list)
    confidence: float = 0.0         # parser confidence 0..1
    consequence: Consequence = Consequence.HARMLESS
    #: What is missing before this can run (empty = executable now).
    missing: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def executable(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["consequence"] = self.consequence.value
        out["confidence"] = round(self.confidence, 2)
        return out


@dataclass
class Understanding:
    top: TopIntent
    reason: str
    action: ActionIntent | None = None
    #: The clarification question, when ``top`` is CLARIFICATION.
    question: str = ""
    is_action_request: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"top": self.top.value, "reason": self.reason, "action": self.action.to_dict() if self.action else None,
                "question": self.question, "is_action_request": self.is_action_request}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def fold(text: str) -> str:
    lowered = (text or "").lower().replace("ß", "ss")
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(a, b)
    return "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))


_VOCATIVE = re.compile(r"^\s*(?:hey\s+|ok\s+|okay\s+|hallo\s+|hi\s+)?(?:zeus|jarvis)\s*[,:!.\-]?\s*", re.I)
_POLITE = re.compile(r"^\s*(?:bitte\s+)?(?:kannst|koenntest|könntest|wuerdest|würdest|willst|magst)\s+du\s+(?:mir\s+|uns\s+)?(?:bitte\s+)?|^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?", re.I)
_WISH = re.compile(r"^\s*(?:ich\s+(?:moechte|möchte|will|brauche|haette\s+gern|hätte\s+gern|wuensche|wünsche)(?:\s+gerne?)?|i\s+(?:want|need|would\s+like))\s+", re.I)
_PLEASE = re.compile(r"\b(?:bitte|please|mal|doch|einfach|jetzt|schnell|kurz)\b", re.I)
_QUOTED = re.compile(r"[„\"“'‚‘«»]([^„\"“”'‚‘’«»]{1,80})[“\"”'‘’»«]")

_INTERROGATIVE = re.compile(r"^(?:was|wer|wie|warum|wieso|weshalb|wann|wo|welche[srnm]?|wieviel|wie\s+viele?|gibt\s+es|ist\s+das|what|who|how|why|when|where|which|is\s+there|does|do\s+you)\b", re.I)

#: Imperative action verbs (stems) that make a request an ACTION REQUEST.
_ACTION_VERB = re.compile(
    r"\b(erstell\w*|erzeug\w*|leg\w*|anleg\w*|mach\w*|oeffne\w*|zeig\w*|speicher\w*|sicher\w*|spiel\w*|pausier\w*|stopp?\w*|halt|"
    r"verschieb\w*|loesch\w*|entfern\w*|start\w*|aender\w*|benenn\w*|archivier\w*|fueg\w*|hinzufueg\w*|notier\w*|merk\w*|"
    r"schick\w*|send\w*|such\w*|finde?|lade?|kopier\w*|druck\w*|installier\w*|setz\w*|stell\w*|schalt\w*|nimm|screenshot|"
    r"create|make|open|show|save|store|play|pause|stop|move|delete|remove|start|change|rename|archive|add|note|send|search|find|copy|print|install|set|take)\b"
)

_PROJECT_WORD = re.compile(r"\b(projekt\w*|project\w*|teilprojekt\w*|subproject\w*|unterprojekt\w*)\b", re.I)


def is_question(text: str) -> bool:
    body = _VOCATIVE.sub("", (text or "").strip(), count=1)
    if body.rstrip().endswith("?") and not _POLITE.match(body):
        return True
    return bool(_INTERROGATIVE.match(body)) and not body.rstrip().endswith("!")


def is_action_request(text: str) -> bool:
    """An imperative (or a polite/wished imperative) with an action verb -- not a question about a concept."""

    body = _VOCATIVE.sub("", (text or "").strip(), count=1)
    polite = bool(_POLITE.match(body)) or bool(_WISH.match(body))
    body2 = _WISH.sub("", _POLITE.sub("", body, count=1), count=1)
    folded = fold(body2)
    if not folded:
        return False
    if is_question(body) and not polite:
        return False
    first = folded.split()[0] if folded.split() else ""
    if _ACTION_VERB.match(first):
        return True
    # "ich moechte ein neues Projekt", "kannst du ein Projekt anlegen"
    if polite and (_ACTION_VERB.search(folded) or _PROJECT_WORD.search(folded)):
        return True
    # separable verbs: "leg ... an", "mach ... auf"
    return bool(re.match(r"^(leg|lege|mach|schalt|stell|fueg|fuege|nimm)\b", folded))


# --------------------------------------------------------------------------
# Project operations
# --------------------------------------------------------------------------

_CREATE = re.compile(r"\b(erstell\w*|erzeug\w*|anleg\w*|leg\w*|mach\w*|neu\w*|create|make|new|set\s+up|start)\b", re.I)
_LIST = re.compile(r"\b(meine\s+projekte|die\s+projekte|alle\s+projekte|projektliste|projekt(?:e|uebersicht|übersicht)|my\s+projects|the\s+projects|all\s+projects|projects)\b", re.I)
_OPEN = re.compile(r"\b(oeffne\w*|öffne\w*|zeig\w*|anzeig\w*|geh\s+(?:zu|in)|open|show|go\s+to)\b", re.I)
_RENAME = re.compile(r"\b(benenn\w*\s+.*\bum|umbenenn\w*|rename|heisst\s+nicht|heißt\s+nicht|nenn\w*)\b", re.I)
_DELETE = re.compile(r"\b(loesch\w*|lösch\w*|entfern\w*|delete|remove)\b", re.I)
_ARCHIVE = re.compile(r"\b(archivier\w*|archive|ins\s+archiv)\b", re.I)
_ADD_TASK = re.compile(r"\b(aufgabe\w*|task\w*|todo\w*)\b", re.I)
_TASK_LIST = re.compile(
    r"(?:(?:leg|lege|erstell|erstelle|fueg|fuege|füge|mit|und|sowie|add|with|create)\s+(?:mir\s+)?(?:die|den|drei|zwei|vier|fuenf|fünf|folgende[n]?|these|the|three|two|four|five|\d+)?\s*"
    r"(?:aufgaben|tasks|todos)\s*(?:an|hinzu)?\s*[:\-–]?\s*)(.+)$", re.I)
_PARENT = re.compile(r"\b(?:unter(?:halb)?|im\s+projekt|in\s+dem\s+projekt|innerhalb\s+von|als\s+teilprojekt\s+von|als\s+unterprojekt\s+von|under|inside|within|as\s+a\s+subproject\s+of)\s+(?:dem\s+projekt\s+|the\s+project\s+|projekt\s+|project\s+)?([A-ZÄÖÜ][\w\-]*(?:\s+[A-ZÄÖÜ0-9][\w\-]*)*|[„\"“'][^„\"“”']+[“\"”'])", re.U)
_TITLE_BY_NAME = re.compile(r"\b(?:namens|mit\s+dem\s+namen|mit\s+namen|mit\s+(?:dem\s+)?titel|genannt|called|named|titled|heissen\s+soll|heißen\s+soll)\s*[:\-–]?\s*(.+)$", re.I)
_TITLE_FOR = re.compile(r"\b(?:fuer|für|for|zu|zum|zur|ueber|über|about|on)\s+(.+)$", re.I)
_POSSESSIVE = re.compile(r"^(?:meine[nmrs]?|mein|dein\w*|unser\w*|die|der|das|den|dem|des|eine?[nmrs]?|my|the|a|an|our)\s+", re.I)
_TRAILING = re.compile(r"\s*(?:\ban\b|\bauf\b|\bein\b|\bhinzu\b|\bplease\b|\bbitte\b|[.!?,;:]+)\s*$", re.I)
_SEPARATOR = re.compile(r"\s*(?:,\s*und\s+|\s+und\s+leg|\s+und\s+erstell|\s+und\s+fueg|\s+und\s+füge|\s+mit\s+(?:den|drei|zwei|vier|folgenden|\d+)?\s*aufgaben|\s+with\s+(?:the\s+)?tasks|;|:)\s*", re.I)
_IMPORTANCE = (
    ("FOCUS", re.compile(r"\b(fokus|focus|wichtig\w*|hohe\s+prioritaet|hohe\s+priorität|high\s+priority|prioritaet\s+hoch|priorität\s+hoch)\b", re.I)),
    ("LOW_PRIORITY", re.compile(r"\b(niedrige\s+prioritaet|niedrige\s+priorität|low\s+priority|unwichtig|nebenbei)\b", re.I)),
    # "nur zum Testen", "als Test": a marker.  "Testprojekt" is a *name* and stays one.
    ("TEST", re.compile(r"\b(zum\s+testen|nur\s+ein\s+test|nur\s+zum\s+test\w*|als\s+test|just\s+a\s+test|for\s+testing)\b", re.I)),
)
_DEADLINE = re.compile(r"\b(?:bis|deadline|until|by)\s+(?:zum\s+|spaetestens\s+|spätestens\s+)?((?:\d{1,2}\.\s?\d{1,2}\.(?:\s?\d{2,4})?)|(?:\d{4}-\d{2}-\d{2})|(?:montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|morgen|uebermorgen|übermorgen|ende\s+\w+|naechste\w*\s+\w+|nächste\w*\s+\w+|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next\s+\w+|end\s+of\s+\w+))", re.I)
_NUMBER_WORDS = {"zwei": 2, "drei": 3, "vier": 4, "fuenf": 5, "fünf": 5, "sechs": 6, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_THIS = re.compile(r"\b(dieses|das|dies|this|that|it|es)\s+(?:projekt|project)?\b|\b(?:projekt|project)\s+(?:hier|here)\b", re.I)


def _clean_title(raw: str) -> str:
    title = raw.strip()
    title = re.sub(r"^\s*(?:ein|eine|einen|neues|neue|neuen|new|a|an)\s+", "", title, flags=re.I)
    title = _POSSESSIVE.sub("", title)
    title = _TRAILING.sub("", title)
    title = title.strip(" \"'„“”‚‘’«»")
    title = re.sub(r"\s+", " ", title)
    # "Physikumsvorbereitung an" / "Biochemie an." -- separable verb particle;
    # "Biochemie anlegen?" -- the infinitive of a polite request
    title = re.sub(r"\s+(?:an|auf|ein|anlegen|erstellen|machen|erzeugen|anzulegen|zu\s+erstellen|create|make|set\s+up)$", "", title, flags=re.I)
    return title.strip(" .,:;!?")


def _split_list(text: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|\bund\b|\band\b|\bsowie\b)\s*", text)
    out: list[str] = []
    for part in parts:
        item = part.strip(" .:;!\"'„“”")
        # the separable particle of "leg ... an" / "füge ... hinzu" lands on the last item
        item = re.sub(r"\s+(?:an|hinzu|auf|ein)$", "", item, flags=re.I).strip()
        if item:
            out.append(item)
    return out


def _extract_tasks(body: str) -> tuple[list[str], str]:
    """(tasks, body without the task clause)."""

    match = _TASK_LIST.search(body)
    if not match:
        return [], body
    tasks = _split_list(match.group(1))
    tasks = [t for t in tasks if t and not re.fullmatch(r"(an|hinzu)", t, flags=re.I)]
    remainder = body[: match.start()].rstrip(" ,;:")
    remainder = re.sub(r"\s+(?:und|and|sowie)\s*$", "", remainder, flags=re.I)
    return tasks, remainder


def _wanted_task_count(text: str) -> int | None:
    m = re.search(r"\b(zwei|drei|vier|fuenf|fünf|sechs|two|three|four|five|six|\d+)\s+(?:aufgaben|tasks|todos)\b", text, re.I)
    if not m:
        return None
    word = m.group(1).lower()
    return _NUMBER_WORDS.get(word) or (int(word) if word.isdigit() else None)


def parse_project_operation(text: str, *, project_titles: Iterable[str] = ()) -> ActionIntent | None:
    """A typed project operation from a natural sentence, or None when the sentence is not one."""

    original = (text or "").strip()
    body = _VOCATIVE.sub("", original, count=1)
    polite = bool(_POLITE.match(body)) or bool(_WISH.match(body))
    body = _WISH.sub("", _POLITE.sub("", body, count=1), count=1).strip()
    body = _PLEASE.sub("", body).strip()
    body = re.sub(r"\s+", " ", body)
    folded = fold(body)
    titles = [t for t in project_titles if t]

    if not _PROJECT_WORD.search(body):
        return None
    # A question about the concept, or a question about the list, is a read at most.
    question = is_question(original) and not polite

    # -- read / list ---------------------------------------------------
    if _LIST.search(body) and not _CREATE.search(folded.split("projekt")[0] if "projekt" in folded else folded[:0]):
        if question or _OPEN.search(body) or re.search(r"\b(liste|list|welche|which|alle|all|meine|my)\b", folded):
            return ActionIntent("project.list", verb="read", object_type="project", confidence=0.9,
                                success_criteria=["the owner's projects are listed"], reason="asks for the project list")

    # -- delete / archive ----------------------------------------------
    if _DELETE.search(body) and not question:
        target = _target_project(body, titles)
        intent = ActionIntent("project.delete", verb="delete", object_type="project", target=target, confidence=0.85 if target else 0.5,
                              consequence=Consequence.IRREVERSIBLE, success_criteria=["the project no longer exists"],
                              reason="asks to delete a project")
        if not target:
            intent.missing.append("target")
        return intent
    if _ARCHIVE.search(body) and not question:
        target = _target_project(body, titles)
        intent = ActionIntent("project.archive", verb="archive", object_type="project", target=target, confidence=0.85 if target else 0.5,
                              consequence=Consequence.MODERATE, success_criteria=["the project is marked ARCHIVED"], reason="asks to archive a project")
        if not target:
            intent.missing.append("target")
        return intent

    # -- rename: "Das Projekt heisst nicht Bio sondern Biochemie", "benenne X in Y um"
    m = re.search(r"(?:heisst|heißt)\s+nicht\s+(.+?)\s+sondern\s+(.+)$", body, re.I)
    if m:
        old, new = _clean_title(m.group(1)), _clean_title(m.group(2))
        return ActionIntent("project.rename", verb="rename", object_type="project", target=old, arguments={"title": new},
                            confidence=0.9, consequence=Consequence.MODERATE, success_criteria=[f"a project titled {new!r} exists", f"no project titled {old!r} remains"],
                            reason="corrects the project title")
    m = re.search(r"\bbenenn\w*\s+(?:das\s+)?(?:projekt\s+)?(.+?)\s+(?:in|zu|nach)\s+(.+?)\s+um\b", body, re.I) or re.search(r"\brename\s+(?:the\s+)?(?:project\s+)?(.+?)\s+to\s+(.+)$", body, re.I)
    if m:
        old, new = _clean_title(m.group(1)), _clean_title(m.group(2))
        return ActionIntent("project.rename", verb="rename", object_type="project", target=old, arguments={"title": new}, confidence=0.9,
                            consequence=Consequence.MODERATE, success_criteria=[f"a project titled {new!r} exists"], reason="asks to rename a project")

    # -- add tasks to an existing project --------------------------------
    if _ADD_TASK.search(body) and not _CREATE.search(folded.split("aufgabe")[0] if "aufgabe" in folded else folded[:0]) and re.search(r"\b(fueg|füg|hinzu|add|ergaenz|ergänz)", folded):
        tasks, _ = _extract_tasks(body)
        if not tasks:
            m2 = re.search(r"(?:aufgabe|task)\s*[:\-–]?\s*[„\"“']?([^„\"“”']+?)[“\"”']?\s*(?:hinzu|zum\s+projekt|to\s+the\s+project|$)", body, re.I)
            if m2:
                tasks = [m2.group(1).strip(" .")]
        target = _target_project(body, titles)
        intent = ActionIntent("project.add_tasks", verb="add_tasks", object_type="project", target=target, arguments={"tasks": tasks},
                              confidence=0.8 if tasks and target else 0.5, success_criteria=[f"{len(tasks)} task(s) persist on the project"],
                              reason="adds tasks to a project")
        if not target:
            intent.missing.append("target")
        if not tasks:
            intent.missing.append("tasks")
        return intent

    # -- open a project ----------------------------------------------------
    if _OPEN.search(body) and not _CREATE.search(folded) and not question:
        target = _target_project(body, titles)
        this = bool(_THIS.search(body)) and not target
        intent = ActionIntent("project.open", verb="open", object_type="project", target=target or ("__last__" if this else ""),
                              confidence=0.9 if target or this else 0.5, success_criteria=["the project view is opened on the project"],
                              reason="asks to open a project")
        if not intent.target:
            intent.missing.append("target")
        return intent

    # -- create ------------------------------------------------------------
    if _CREATE.search(folded) and not question:
        tasks, rest = _extract_tasks(body)
        wanted = _wanted_task_count(body)
        parent = ""
        pm = _PARENT.search(rest)
        if pm:
            parent = _clean_title(pm.group(1))
            rest = (rest[: pm.start()] + " " + rest[pm.end():]).strip()
        importance = ""
        for level, pattern in _IMPORTANCE:
            if pattern.search(rest):
                importance = level
                rest = pattern.sub("", rest)
                break
        deadline = ""
        dm = _DEADLINE.search(rest)
        if dm:
            deadline = dm.group(1).strip()
            rest = (rest[: dm.start()] + " " + rest[dm.end():]).strip()
        title, goal = "", ""
        q = _QUOTED.search(rest)
        after_project = ""
        pm2 = _PROJECT_WORD.search(rest)
        if pm2:
            after_project = rest[pm2.end():].strip(" ,:-–")
            after_project = _SEPARATOR.split(after_project)[0]
        if q:
            title = _clean_title(q.group(1))
        else:
            nm = _TITLE_BY_NAME.search(after_project or rest)
            if nm:
                title = _clean_title(_SEPARATOR.split(nm.group(1))[0])
            else:
                fm = _TITLE_FOR.search(after_project or rest)
                if fm:
                    phrase = _SEPARATOR.split(fm.group(1))[0]
                    goal = _clean_title(phrase) if phrase else ""
                    title = _title_from_phrase(phrase)
                elif after_project:
                    candidate = _clean_title(after_project)
                    if candidate and not _CREATE.fullmatch(fold(candidate)) and fold(candidate) not in {"an", "anlegen", "erstellen", "auf"}:
                        title = candidate
        if not goal:
            goal = title
        intent = ActionIntent(
            "project.create", verb="create", object_type="project", target=title,
            arguments={"title": title, "goal": goal, "tasks": tasks, "parent": parent, "importance": importance, "deadline": deadline,
                       "description": body},
            constraints=[], success_criteria=["the project persists with this title"] + ([f"{len(tasks)} tasks persist"] if tasks else []),
            forbidden_effects=["file", "note"], confidence=0.92 if title else 0.6, consequence=Consequence.HARMLESS,
            reason="asks to create a project",
        )
        if wanted and tasks and len(tasks) != wanted:
            intent.constraints.append(f"{wanted} tasks were named; {len(tasks)} parsed")
        if not title:
            intent.missing.append("title")
        return intent
    return None


def _title_from_phrase(phrase: str) -> str:
    """"meine Prüfungsvorbereitung" -> "Prüfungsvorbereitung"; "meinen Hausbau" -> "Hausbau"; "M1 Vorbereitung" stays."""

    cleaned = _clean_title(phrase)
    cleaned = re.sub(r"^(?:das|die|der|den|dem|the)\s+(?:thema|fach|bereich)\s+", "", cleaned, flags=re.I)
    words = cleaned.split()
    if not words:
        return ""
    # keep short noun phrases whole; capitalise the head
    if len(words) > 6:
        words = words[:6]
    out = " ".join(words)
    return out[0].upper() + out[1:]


def _target_project(body: str, titles: Iterable[str]) -> str:
    """Which existing project a sentence names, by title (exact, then contained, then compound like 'Stockfish-Projekt')."""

    q = _QUOTED.search(body)
    if q:
        return _clean_title(q.group(1))
    folded = fold(body)
    best = ""
    for title in sorted((t for t in titles if t), key=len, reverse=True):
        ft = fold(title)
        if ft and re.search(r"(?<![\w-])" + re.escape(ft) + r"(?![\w])", folded):
            return title
    # "das Stockfish-Projekt", "das Projekt Stockfish", "the Stockfish project"
    m = re.search(r"\b(?:das|dem|des|the|mein|meinem)?\s*([A-Za-zÄÖÜäöü0-9][\w]*)[-\s](?:projekt|project)\b", body) or \
        re.search(r"\b(?:projekt|project)\s+[„\"“']?([A-Za-zÄÖÜäöü0-9][\w\-]*(?:\s+[A-ZÄÖÜ0-9][\w\-]*)*)", body)
    if m:
        candidate = m.group(1).strip(" \"'„“”")
        if fold(candidate) not in {"das", "dieses", "dies", "ein", "neues", "mein", "this", "the", "a", "new"}:
            for title in titles:
                if fold(candidate) in fold(title) or fold(title) in fold(candidate):
                    return title
            best = candidate
    return best


# --------------------------------------------------------------------------
# System control and knowledge
# --------------------------------------------------------------------------

_VIEWS = {
    "projekte": "projects", "projects": "projects", "projektansicht": "projects", "galaxie": "projects", "galaxy": "projects",
    "missionen": "missions", "missions": "missions", "mission control": "missions",
    "activity": "activity", "aktivitaet": "activity", "aktivitäten": "activity", "protokoll": "activity",
    "knowledge": "knowledge", "wissen": "knowledge", "wissensgraph": "knowledge",
    "korrekturen": "corrections", "corrections": "corrections", "diagnose": "diagnostics", "diagnostics": "diagnostics",
    "einstellungen": "owner", "owner": "owner", "voice studio": "voice", "sprachstudio": "voice", "gedanken": "thoughts", "thoughts": "thoughts",
    "capabilities": "capabilities", "faehigkeiten": "capabilities", "fähigkeiten": "capabilities", "release": "release",
}
_OPEN_VIEW = re.compile(r"\b(oeffne|öffne|zeig(?:e)?\s+mir|zeig|geh\s+(?:zu|in)|open|show\s+me|show|go\s+to)\b\s*(?:die|das|den|the|meine|my)?\s*(?P<view>[\w\s]+?)\s*(?:an|auf|bitte)?\s*[.!]?$", re.I)
_STOP = re.compile(r"^\s*(stopp?|halt|stop|abbrechen|cancel|sei\s+still|ruhe|be\s+quiet|shut\s+up|hoer\s+auf|hör\s+auf)\b", re.I)
_SCREENSHOT = re.compile(r"\b(screenshot|bildschirmfoto|bildschirmaufnahme|screen\s+capture)\b", re.I)
_KNOWLEDGE_SAVE = re.compile(r"\b(speicher\w*|sicher\w*|leg\w*|notier\w*|merk\w*|save|store|note|remember)\b.*\b(knowledge|wissen|wissensgraph|knowledge\s+graph)\b|\b(knowledge|wissen)\b.*\b(speicher\w*|ablegen|save|store)\b", re.I)


def parse_system_control(text: str) -> ActionIntent | None:
    body = _VOCATIVE.sub("", (text or "").strip(), count=1)
    if _STOP.match(body) and len(body.split()) <= 4:
        return ActionIntent("system.stop", verb="stop", object_type="system", confidence=0.95, success_criteria=["speech and the current answer stop"], reason="stop")
    if _SCREENSHOT.search(body) and is_action_request(text):
        return ActionIntent("screenshot.capture", verb="capture", object_type="screenshot", confidence=0.85,
                            success_criteria=["an image file exists"], reason="asks for a screenshot")
    if _PROJECT_WORD.search(body):
        return None
    m = _OPEN_VIEW.search(body)
    if m and is_action_request(text):
        name = fold(m.group("view")).strip()
        for key, view in _VIEWS.items():
            if fold(key) == name or name.endswith(fold(key)):
                return ActionIntent("system.open_view", verb="open", object_type="view", target=view, confidence=0.9,
                                    success_criteria=[f"the {view} view is open"], reason=f"asks to open {view}")
    return None


def parse_knowledge_operation(text: str) -> ActionIntent | None:
    body = _VOCATIVE.sub("", (text or "").strip(), count=1)
    if _KNOWLEDGE_SAVE.search(body) and is_action_request(text):
        return ActionIntent("knowledge.create", verb="create", object_type="knowledge", confidence=0.8,
                            success_criteria=["a knowledge node exists and is searchable"], forbidden_effects=["file", "note", "project"],
                            reason="asks to store something in Knowledge")
    return None


# --------------------------------------------------------------------------
# Correction and self-development shapes
# --------------------------------------------------------------------------

_CORRECTION = re.compile(r"^\s*(nein[,.!]?\s|falsch[,.!]?\s|das\s+war\s+falsch|das\s+ist\s+falsch|so\s+nicht|nicht\s+so|ich\s+meinte|das\s+meinte\s+ich\s+nicht|du\s+hast\s+mich\s+falsch\s+verstanden|korrektur[:\s]|no[,.!]?\s+(?:that|i)|that\s+was\s+wrong|wrong[,.!]?\s|i\s+meant|correction[:\s])", re.I)
_MEANT = re.compile(r"\b(?:ich\s+meinte|meinte\s+ich|i\s+meant|gemeint\s+war|nicht\s+\S+\s+sondern)\s+[„\"“']?([^„\"“”'.!?]+)", re.I)
_SELF_REPAIR = re.compile(r"\b(finde\s+den\s+fehler|repariere?\s+dich|reparier\s+dich|fix\s+yourself|repair\s+yourself|verbesser\w*\s+dich|behebe\s+das\s+bei\s+dir|behebe\s+deinen|fix\s+your|korrigiere\s+dich)\b", re.I)


def parse_correction(text: str) -> ActionIntent | None:
    body = (text or "").strip()
    if _SELF_REPAIR.search(body):
        return None  # a repair request is self-development, not a correction
    if not _CORRECTION.match(_VOCATIVE.sub("", body, count=1)):
        return None
    meant = ""
    m = _MEANT.search(body)
    if m:
        meant = m.group(1).strip(" .!?\"'„“”")
    return ActionIntent("correction.note", verb="correct", object_type="correction", target=meant, arguments={"meant": meant},
                        confidence=0.85, success_criteria=["the correction is attached to the last request"], reason="the owner corrects the previous reading")


def is_self_repair_request(text: str) -> bool:
    return bool(_SELF_REPAIR.search(text or ""))


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------

def understand(text: str, *, route: Any = None, project_titles: Iterable[str] = (), capability_names: Iterable[str] = ()) -> Understanding:
    """One TopIntent, and a typed ActionIntent where the operation is deterministic.

    ``route`` is :func:`service.routing.route`'s result for the same text,
    when the caller already has it; self-development, acquisition and
    owner-core routes are honoured as they are.
    """

    text = (text or "").strip()
    action_request = is_action_request(text)
    top_value = getattr(getattr(route, "intent", None), "value", "") if route is not None else ""

    if is_self_repair_request(text):
        return Understanding(TopIntent.SELF_DEVELOPMENT, "asks ZEUS to find and repair its own mistake", is_action_request=True)
    correction = parse_correction(text)
    if correction is not None:
        return Understanding(TopIntent.CORRECTION, correction.reason, correction, is_action_request=False)
    if top_value in {"self_development", "capability_repair", "owner_config_change"}:
        return Understanding(TopIntent.SELF_DEVELOPMENT, f"router: {top_value}", is_action_request=action_request)
    if top_value == "capability_acquisition":
        return Understanding(TopIntent.MISSION, "asks to acquire an ability", is_action_request=action_request)

    control = parse_system_control(text)
    if control is not None:
        return Understanding(TopIntent.SYSTEM_CONTROL if control.object_type in {"system", "view"} else TopIntent.ACTION, control.reason, control, is_action_request=True)

    project = parse_project_operation(text, project_titles=project_titles)
    if project is not None:
        if project.missing:
            question = clarification_for(project)
            return Understanding(TopIntent.CLARIFICATION, f"{project.operation}: missing {', '.join(project.missing)}", project, question=question, is_action_request=True)
        return Understanding(TopIntent.PROJECT_OPERATION, project.reason, project, is_action_request=True)

    knowledge = parse_knowledge_operation(text)
    if knowledge is not None:
        return Understanding(TopIntent.KNOWLEDGE_OPERATION, knowledge.reason, knowledge, is_action_request=True)

    if top_value in {"project", "complex_mission"}:
        return Understanding(TopIntent.MISSION, f"router: {top_value}", is_action_request=action_request)
    if action_request:
        return Understanding(TopIntent.ACTION, "an imperative with an action verb", is_action_request=True)
    return Understanding(TopIntent.CONVERSATION, "a question or a remark with no side effect", is_action_request=False)


def clarification_for(intent: ActionIntent, *, language: str = "de") -> str:
    de = (language or "de").startswith("de")
    if "title" in intent.missing:
        return "Wie soll das Projekt heißen?" if de else "What should the project be called?"
    if "target" in intent.missing:
        return "Welches Projekt meinst du?" if de else "Which project do you mean?"
    if "tasks" in intent.missing:
        return "Welche Aufgaben soll ich anlegen?" if de else "Which tasks should I add?"
    return ("Mir fehlt noch: " + ", ".join(intent.missing)) if de else ("I still need: " + ", ".join(intent.missing))
