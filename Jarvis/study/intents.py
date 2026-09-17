"""Study commands as typed actions, read without a model.

"Zeig mir die Seite zum Frank-Starling-Mechanismus" is not a question for a language
model; it is ``study.locate(topic)``.  A sentence is read as three things, not matched
against a list of sentences:

    action   show / open / find / navigate / inspect / answer / summarize / quiz / explain
    scope    Studium, study files, notes, GoodNotes, slides, script, handwriting -- or the
             document that is open right now ("diese Seite", "davor", "daneben")
    subject  what it is about ("Frank-Starling", "die Niere", "Herzphysiologie")

and becomes one operation:

    study.locate      open the best place                "Zeig mir (die Seite zu) Frank-Starling", "Öffne meine GoodNotes-Notizen dazu"
    study.search      list the places                    "Wo habe ich etwas zur Niere?", "Welche Unterlagen habe ich zu X?"
    study.which_page  name the open place                "Welche Seite war das?"
    study.figure      the figure for the current topic   "Zeig mir die Abbildung."
    study.page        previous / next page or slide      "Seite davor", "Seite danach", "nächste Folie"
    study.return      back to the found place            "Zurück zu der Stelle", "Öffne genau die Stelle", "Zeig mir den Ausschnitt"
    study.view        zoom / whole page                  "Mach größer", "Zeig die ganze Seite"
    study.next_hit    another place for the same topic   "Zeig mir noch einen Treffer"
    study.below       what follows the found place       "Was steht darunter?"
    study.beside      handwriting next to it             "Was habe ich daneben geschrieben?", "Was steht handschriftlich daneben?"
    study.open_recent recently used material             "Öffne das Skript von gestern"
    study.answer      answer from the owner's material   "Was habe ich zu X notiert?"
    study.summarize / study.quiz / study.explain / study.compare / study.missing   on the open place

A subject with no study scope at all ("Zeig mir Frank-Starling.") is a study command only
when the owner's own library literally contains it (``probe``) -- "Zeig mir das Wetter"
never is.  Words that also mean the web ("Seite von Amazon", "Webseite") never are.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from service.intents import ActionIntent

_MATERIAL = (r"(?:seite|stelle|abschnitt|folie|abbildung|grafik|schema|notizen|notiz|mitschrift|unterlagen|skript|script|vorlesung(?:snotizen)?"
             r"|kapitel|material|lernzettel|goodnotes(?:-?notizen|-?unterlagen)?)")
_WEB = re.compile(r"(?i)\b(webseite|website|homepage|internetseite|url|https?://|www\.|\.(?:de|com|org|net)\b|seite\s+von\s+[A-Z][a-z]+\s*$)")

#: words that put a sentence into Studium on their own
_SCOPE = re.compile(r"(?i)\b(studium\w*|studien\w*|studiums?dateien|lernmaterial\w*|unterlagen|notizen|notiz|mitschrift\w*|skript\w*|script|vorlesung\w*"
                    r"|folien|folie|lernzettel\w*|goodnotes\w*|handschrift\w*|handgeschrieben\w*|abbildung|schaubild|schema|grafik|diagramm"
                    r"|pdf|seite|kapitel|abschnitt)\b")
#: words that name something outside Studium: never guessed into it without the owner's own words
_ELSEWHERE = re.compile(r"(?i)\b(wetter|nachrichten|kalender|termin\w*|spotify|musik|song|lied|playlist|projekt\w*|mission\w*|einstellung\w*"
                        r"|webseite|internet|google|youtube|foto\w*|bilder\s+von|video\w*|datei(en)?\s+(auf|im|in)\s+(dem\s+)?(desktop|download|laufwerk)"
                        r"|laufwerk|ordner|explorer|tv|fernseher|licht|lampe|timer|wecker|erinnerung\w*|datei)\b"
                        r"|\b[\w-]+\.(?:txt|md|pdf|docx?|pptx?|xlsx?|png|jpe?g|csv|json|py|js|html?)\b|[a-z]:\\")

_SHOW = r"(?:zeig(?:e|t)?|anzeigen|pr[äa]sentier\w*)"
_OPEN = r"(?:[öo]ffne\w*|oeffne\w*|mach\s+.{0,40}\s+auf|bring\s+mich\s+zu|geh\s+zu|spring\s+zu|navigier\w*)"
_FIND = r"(?:such\w*|find\w*)"

_PAGE_PREV = re.compile(r"(?i)^\s*(?:und\s+)?(?:(?:die\s+)?(?:seite|folie)\s+(?:davor|zur[üu]ck)|(?:die\s+)?vorherige\s+(?:seite|folie)|eine\s+(?:seite|folie)\s+zur[üu]ck"
                        r"|zur[üu]ckbl[äa]ttern|bl[äa]ttere?\s+zur[üu]ck)\s*[.!?]?\s*$")
_PAGE_NEXT = re.compile(r"(?i)^\s*(?:und\s+)?(?:(?:die\s+)?(?:seite|folie)\s+(?:danach|weiter)|(?:die\s+)?n[äa]chste\s+(?:seite|folie)|eine\s+(?:seite|folie)\s+weiter"
                        r"|weiterbl[äa]ttern|bl[äa]ttere?\s+weiter)\s*[.!?]?\s*$")
_PAGE_N = re.compile(r"(?i)^\s*(?:geh\s+(?:auf|zu)|spring\s+(?:auf|zu)|zeig(?:e)?\s+mir|[öo]ffne)\s+(?:die\s+)?(seite|folie)\s+(\d{1,4})\s*[.!?]?\s*$")
_RETURN = re.compile(r"(?i)^\s*(?:und\s+)?(?:zur[üu]ck\s+zu(?:r|\s+der)\s+stelle|(?:[öo]ffne|zeig(?:e)?\s+mir)\s+(?:genau\s+)?(?:die|diese)\s+stelle"
                     r"|zeig(?:e)?\s+mir\s+(?:den|diesen)\s+ausschnitt|(?:[öo]ffne|zeig(?:e)?\s+mir)\s+(?:den|diesen)\s+ausschnitt)\s*(?:wieder|nochmal)?\s*[.!?]?\s*$")
_ZOOM_IN = re.compile(r"(?i)^\s*(?:mach(?:e)?\s+(?:es\s+|das\s+|sie\s+)?gr[öo](?:ß|ss)er|vergr[öo](?:ß|ss)er\w*|zoom\w*\s+(?:rein|hinein|n[äa]her)|n[äa]her\s+ran)\s*[.!?]?\s*$")
_ZOOM_OUT = re.compile(r"(?i)^\s*(?:mach(?:e)?\s+(?:es\s+|das\s+|sie\s+)?kleiner|verkleiner\w*|zoom\w*\s+(?:raus|heraus))\s*[.!?]?\s*$")
_FIT = re.compile(r"(?i)^\s*(?:zeig(?:e)?\s+(?:mir\s+)?die\s+ganze\s+(?:seite|folie)|ganze\s+(?:seite|folie)(?:\s+zeigen)?)\s*[.!?]?\s*$")
_NEXT_HIT = re.compile(r"(?i)^\s*(?:zeig(?:e)?\s+(?:mir\s+)?(?:noch\s+)?(?:einen|ein)\s+(?:anderen\s+|weiteren\s+)?(?:treffer|stelle)|n[äa]chster\s+treffer"
                       r"|(?:ein\s+)?anderer\s+treffer|noch\s+(?:ein|einen)\s+treffer)\s*[.!?]?\s*$")
_BELOW = re.compile(r"(?i)^\s*was\s+steht\s+(?:da)?(?:runter|drunter|darunter|unten\s+drunter|danach)\s*[.!?]?\s*$")
_BESIDE = re.compile(r"(?i)^\s*was\s+(?:habe|hab)\s+ich\s+(?:da)?(?:neben|daneben)\s+(?:handschriftlich\s+)?(?:geschrieben|notiert|vermerkt)\s*[.!?]?\s*$"
                     r"|^\s*was\s+steht\s+(?:handschriftlich\s+)?(?:da)?neben(?:\s+handschriftlich)?\s*[.!?]?\s*$")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("study.figure", re.compile(r"(?i)^\s*(?:und\s+)?(?:zeig(?:e)?\s+mir|öffne|oeffne)\s+(?:bitte\s+)?(?:die|das|den)\s+(?:abbildung|grafik|schema|bild|schaubild)\s*(?:dazu)?\s*[.!?]?\s*$")),
    ("study.which_page", re.compile(r"(?i)^\s*(?:und\s+)?(?:welche|auf\s+welcher)\s+(?:seite|folie)\s+(?:war|ist|stand)\s+(?:das|es|dies)\b")),
    ("study.explain", re.compile(r"(?i)\berkl[äa]r(?:e)?\s+(?:mir\s+)?(?:nur\s+)?(?:den|diesen|die|diese|das|dieses)\s+(?:markierten?|ausgew[äa]hlten?|diesen|diese|dieses)?\s*(?:abschnitt|stelle|teil|absatz|passage|seite|folie|kapitel)\b")),
    ("study.summarize", re.compile(r"(?i)\b(?:fass(?:e)?|zusammenfassen|zusammenfassung)\b.*\b(?:diese|dieses|diesen|die|das)\s+(?:seite|kapitel|abschnitt|folie|skript|dokument)\b"
                                   r"|\b(?:diese|dieses)\s+(?:seite|kapitel|abschnitt|folie)\s+zusammen")),
    ("study.quiz", re.compile(r"(?i)\b(?:pr[üu]f(?:e)?\s+mich|frag(?:e)?\s+mich\s+ab|quiz|mach(?:e)?\s+(?:daraus|mir)?\s*\d*\s*(?:schwere\s+|leichte\s+)?(?:mc|multiple[- ]choice|pr[üu]fungs)?-?\s*fragen|karteikarten)\b")),
    ("study.missing", re.compile(r"(?i)\b(?:was|welche\s+themen)\s+fehl(?:t|en)\s+(?:meinen?\s+notizen|mir)\b")),
    ("study.compare", re.compile(r"(?i)\bvergleich(?:e)?\b.*\b(?:notizen|skript|mitschrift|unterlagen|vorlesung)")),
    ("study.open_recent", re.compile(r"(?i)\b(?:öffne|oeffne|zeig(?:e)?\s+mir)\s+(?:das|die|den|mein(?:e|en)?)\s+(?:skript|notizen|unterlagen|vorlesung|mitschrift|folien)\s+von\s+(?:gestern|heute|vorgestern|letzter\s+woche)\b")),
    ("study.answer", re.compile(r"(?i)\bwas\s+(?:habe|hab)\s+ich\s+(?:zu[mr]?|über|ueber)\s+.+\s+(?:notiert|geschrieben|aufgeschrieben|mitgeschrieben)\b"
                                r"|\bwas\s+steht\s+in\s+meine[mn]?\s+" + _MATERIAL + r"\s+(?:zu[mr]?|über|ueber)\b")),
    ("study.search", re.compile(r"(?i)\bwo\s+(?:habe|hab|hatte)\s+ich\s+(?:etwas|was|infos?|notizen|material)\s+(?:zu[mr]?|über|ueber)\b"
                                r"|\bwo\s+stand\s+(?:nochmal\s+|noch\s+mal\s+)?"
                                r"|\bwo\s+(?:steht|stehen|finde\s+ich)\s+(?:in\s+meine[mnr]?\s+\w+\s+)?(?:etwas|was|infos?)\s+(?:zu[mr]?|über|ueber)\b"
                                r"|\bwelche\s+(?:unterlagen|notizen|skripte|materialien)\s+habe\s+ich\s+(?:zu[mr]?|über|ueber)\b"
                                r"|\bsuch(?:e)?\s+in\s+meinen\s+(?:unterlagen|notizen|skripten|materialien|studium\w*)\b")),
    ("study.locate", re.compile(r"(?i)\b(?:zeig(?:e)?\s+mir|öffne|oeffne)\s+(?:bitte\s+)?(?:genau\s+)?(?:die|den|das|meine|meinen)\s+(?:genaue\s+)?"
                                + _MATERIAL + r"\b.*\b(?:zu[mr]?|über|ueber|mit)\b")),
]
_CONTEXT_OPS = {"study.figure", "study.which_page", "study.explain", "study.summarize", "study.quiz", "study.compare", "study.missing"}

#: the study words around a subject, removed before the subject is searched for
_SCOPE_PHRASES = re.compile(
    r"(?i)\s*\b(?:in|aus|bei|unter)\s+(?:meinen|meine|meiner|den|der|dem)\s+(?:eigenen\s+)?(?:studium\w*|studien\w*|unterlagen|notizen|mitschriften?|skripten?|scripts?"
    r"|vorlesungs\w*|folien|lernzettel\w*|goodnotes\w*|dateien|materialien|material)\b(?:\s+(?:im|in)\s+studium)?"
    r"|\s*\b(?:im|aus\s+dem)\s+studium\b|\s*\bhandschriftlich\b")


_NOTED = re.compile(r"(?i)\bwo\s+(?:habe|hab|hatte)\s+ich\s+(?P<t>.+?)\s+(?:handschriftlich\s+|von\s+hand\s+)?(?:notiert|geschrieben|aufgeschrieben|mitgeschrieben|vermerkt)\b")
_KIND_NOUNS = re.compile(r"(?i)^(?:(?:die|den|das|meine[nr]?|eine[nr]?)\s+)?(?:goodnotes-?)?(?:bild|abbildung|grafik|schema|schaubild|diagramm|tabelle|folie|seite|stelle"
                         r"|notizen|notiz|unterlagen|mitschrift|skript|lernzettel|goodnotes(?:-?notizen|-?unterlagen)?)(?:\s+(?:zu[mr]?|über|ueber|mit|von))?\s*")
#: a subject that points back at what was just found
_DEICTIC = {"", "das", "es", "dazu", "davon", "darüber", "hierzu", "dies", "diese", "dieses"}


def _topic(text: str) -> str:
    from study.retrieval import read_query

    noted = _NOTED.search(text)
    if noted:
        return _clean_topic(noted.group("t"))
    return _clean_topic(read_query(_SCOPE_PHRASES.sub("", text)).topic)


def _clean_topic(topic: str) -> str:
    cleaned = _KIND_NOUNS.sub("", topic.strip(" .!?")).strip()
    cleaned = re.sub(r"(?i)\s+(?:dazu|davon|darüber)$", "", cleaned).strip()
    # "Wo habe ich etwas zu Kalium notiert?": the subject is "Kalium", not "etwas zu Kalium"
    cleaned = re.sub(r"(?i)^(?:(?:etwas|was|irgendwas|infos?|informationen|notizen|material)\s+)?(?:zu[mr]?|über|ueber|zum\s+thema)\s+(?:den|die|das|dem|der)?\s*", "", cleaned).strip() or cleaned
    return cleaned


def _handwriting(body: str) -> bool:
    return bool(re.search(r"(?i)\b(handschrift\w*|handgeschrieben\w*|von\s+hand)\b", body))


def _goodnotes(body: str) -> bool:
    return bool(re.search(r"(?i)\bgoodnotes", body))


def parse_study_operation(text: str, *, has_material: bool = True, probe: Callable[[str], bool] | None = None,
                          has_context: bool = False) -> ActionIntent | None:
    """The typed study action for a sentence, or None when it is not a study command.

    ``probe(topic)`` answers whether the owner's library literally contains a topic (lexical, local, no model);
    ``has_context`` says a study place is open or was just found -- only then do "davor", "darunter" ... mean a page."""

    body = " ".join(str(text or "").split())
    if not body or _WEB.search(body):
        return None
    navigation = _navigation(body, has_context=has_context)
    if navigation is not None:
        return navigation
    if not has_context and _navigation(body, has_context=True) is not None:
        return None  # "Zeig die ganze Seite" with nothing open is not a search for the words "ganze Seite"
    for operation, pattern in _PATTERNS:
        if not pattern.search(body):
            continue
        if operation in _CONTEXT_OPS:
            count = re.search(r"\b(\d{1,3})\b", body)
            arguments: dict[str, Any] = {"count": int(count.group(1)) if count and operation == "study.quiz" else None,
                                         "difficulty": "schwer" if re.search(r"(?i)\bschwer", body) else ("leicht" if re.search(r"(?i)\bleicht", body) else ""),
                                         "multiple_choice": bool(re.search(r"(?i)\b(mc|multiple[- ]choice)\b", body))}
            if operation in {"study.compare", "study.missing"}:
                arguments["topic"] = ""
            return ActionIntent(operation, verb="read", object_type="study", target="", arguments=arguments, confidence=0.85,
                                success_criteria=["uses only the owner's study material as source"], reason=f"study command: {operation}")
        if operation == "study.open_recent":
            when = re.search(r"(?i)\b(gestern|heute|vorgestern|letzter\s+woche)\b", body)
            kind = re.search(r"(?i)\b(skript|notizen|unterlagen|vorlesung|mitschrift|folien)\b", body)
            return ActionIntent(operation, verb="open", object_type="study", target=kind.group(1).lower() if kind else "",
                                arguments={"when": when.group(1).lower() if when else "gestern"}, confidence=0.85,
                                success_criteria=["opens material used or changed in that period"], reason="study command: open recent material")
        if not has_material and operation != "study.locate":
            return None
        topic = _topic(body)
        if topic.lower() in _DEICTIC or len(re.sub(r"\W", "", topic)) < 3:
            return _from_context(operation, body) if has_context else None
        return _topic_intent(operation, body, topic)
    return _structured(body, has_material=has_material, probe=probe, has_context=has_context)


def _topic_intent(operation: str, body: str, topic: str, *, reason: str = "") -> ActionIntent:
    figure = bool(re.search(r"(?i)\b(abbildung|abb\.?|grafik|schema|schaubild|diagramm|bild)\b", body))
    return ActionIntent(operation, verb="open" if operation == "study.locate" else "read", object_type="study", target=topic,
                        arguments={"query": body, "figure": figure, "handwriting": _handwriting(body), "goodnotes": _goodnotes(body)},
                        confidence=0.9, success_criteria=["names the document and the exact location", "never invents a source"],
                        reason=reason or f"study command: {operation}")


def _from_context(operation: str, body: str) -> ActionIntent | None:
    """"Öffne meine GoodNotes-Notizen dazu", "Wo habe ich das handschriftlich notiert?", "Wo ist die Grafik?": the subject is the open topic."""

    figure = bool(re.search(r"(?i)\b(abbildung|abb\.?|grafik|schema|schaubild|diagramm|bild)\b", body))
    if figure and not _handwriting(body) and not _goodnotes(body):
        return ActionIntent("study.figure", verb="read", object_type="study", target="", arguments={}, confidence=0.85,
                            success_criteria=["uses only the owner's study material as source"], reason="study command: figure of the open topic")
    if operation not in {"study.locate", "study.search"} and not (_handwriting(body) or _goodnotes(body)):
        return None
    intent = _topic_intent(operation if operation in {"study.locate", "study.search"} else "study.search", body, "",
                           reason="study command: the open topic in other material")
    intent.arguments["from_context"] = True
    return intent


def _return_to_place(body: str) -> ActionIntent:
    return ActionIntent("study.return", verb="open", object_type="study", target="", arguments={"query": body}, confidence=0.85,
                        success_criteria=["reopens the found place"], reason="study navigation: back to the place")


def _navigation(body: str, *, has_context: bool) -> ActionIntent | None:
    """Commands about the open document.  Without an open place they are not study commands (other parts of ZEUS may own them)."""

    def nav(operation: str, **arguments: Any) -> ActionIntent:
        return ActionIntent(operation, verb="open", object_type="study", target="", arguments=arguments, confidence=0.85,
                            success_criteria=["acts on the open study document"], reason=f"study navigation: {operation}")

    if _PAGE_PREV.match(body) and has_context:
        return nav("study.page", step=-1)
    if _PAGE_NEXT.match(body) and has_context:
        return nav("study.page", step=1)
    number = _PAGE_N.match(body)
    if number and has_context:
        return nav("study.page", number=int(number.group(2)))
    if _RETURN.match(body) and has_context:
        return _return_to_place(body)
    if _ZOOM_IN.match(body) and has_context:
        return nav("study.view", zoom=0.25)
    if _ZOOM_OUT.match(body) and has_context:
        return nav("study.view", zoom=-0.25)
    if _FIT.match(body) and has_context:
        return nav("study.view", fit=True)
    if _NEXT_HIT.match(body) and has_context:
        return nav("study.next_hit")
    if _BELOW.match(body) and has_context:
        return nav("study.below")
    if _BESIDE.match(body) and has_context:
        return nav("study.beside")
    return None


def _structured(body: str, *, has_material: bool, probe: Callable[[str], bool] | None, has_context: bool) -> ActionIntent | None:
    """action + scope + subject, for the sentences no fixed form covers."""

    if not has_material:
        return None
    lowered = body.lower()
    action = ""
    if re.match(r"(?i)^\s*(?:und\s+)?(?:bitte\s+)?(?:kannst\s+du\s+(?:mir\s+)?)?" + _SHOW + r"\b", body) or re.search(r"(?i)\b" + _SHOW + r"\s+mir\b", body):
        action = "show"
    elif re.match(r"(?i)^\s*(?:und\s+)?(?:bitte\s+)?" + _OPEN, body):
        action = "open"
    elif re.match(r"(?i)^\s*(?:und\s+)?(?:bitte\s+)?" + _FIND + r"\b", body):
        action = "find"
    elif re.match(r"(?i)^\s*wo\s+(?:habe|hab|hatte|steht|stand|stehen|finde|ist|war)\b", body):
        action = "where"
    if not action:
        return None
    scoped = bool(_SCOPE.search(body))
    if action == "where" and re.match(r"(?i)^\s*wo\s+steht\s+das\s*[.!?]?\s*$", body):
        return _return_to_place(body) if has_context else None
    topic = _topic(body)
    topic = re.sub(r"(?i)^(?:etwas|was|mal|bitte|alles|infos?|informationen)\s+(?:zu[mr]?|über|ueber|von)\s+", "", topic).strip(" .!?")
    topic = re.sub(r"(?i)^(?:zu[mr]?|über|ueber|von)\s+", "", topic).strip(" .!?")
    topic = re.sub(r"(?i)^wo\s+(?:ist|war|steht|stand|sind|finde\s+ich)\s+", "", topic).strip(" .!?")
    topic = _clean_topic(topic)
    if len(re.sub(r"\W", "", topic)) < 3 or topic.lower() in _DEICTIC | {"sie", "den ausschnitt", "die stelle", "ausschnitt", "stelle"}:
        if not has_context:
            return None
        if action in {"show", "open"} and re.search(r"(?i)\b(stelle|ausschnitt)\b", body):
            return _return_to_place(body)
        return _from_context("study.search" if action in {"where", "find"} else "study.locate", body)
    if not scoped:
        # no study word at all: only when the owner's own library literally holds the subject, and never for other domains
        if _ELSEWHERE.search(body) or probe is None or not probe(topic):
            return None
    elif _ELSEWHERE.search(lowered) and not re.search(r"(?i)\b(studium\w*|unterlagen|notizen|skript\w*|goodnotes\w*|folien|lernzettel\w*|handschrift\w*)\b", body):
        return None
    operation = "study.search" if action in {"where", "find"} else "study.locate"
    return _topic_intent(operation, body, topic, reason=f"study command: {action} + {'scope' if scoped else 'library match'} + subject")
