"""Catch a success claim that no receipt supports.

:mod:`service.intent` routes anything that looks like an action to a path that
actually executes it.  This module exists because "looks like" is not "is": no
keyword list covers every phrasing, and the request that gets misclassified is
exactly the one nobody thought of.

So the answer is checked as well as the question.  If the reply says the thing
was done -- *"wurde erstellt"*, *"I have created"*, *"verified that it exists"*
-- and no receipt was produced during that turn, the claim is false by
construction, because a receipt is the only way anything gets done here.  It is
then not shipped as written.

This is the second of two defences and it is the weaker one, which is worth
being precise about.  It fires after generation, so on a streaming turn the user
may briefly see the claim before the final message replaces it.  The strong
defence is the first one: a turn classified as an action never streams model
prose at all, because its outcome is composed from the receipt.  This one exists
to make a *classifier miss* honest rather than dangerous.

Why detect a claim rather than instruct the model not to make one: the running
product already had "Never claim an action was performed unless it actually
was" in its system prompt, ranked last so nothing could crowd it out, and it
claimed anyway -- inventing a Linux path on a Windows machine to do it.  A rule
the model is free to ignore is not a mechanism.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _fold(text: str) -> str:
    """Lowercase and strip accents, so "geprüft" and "geprueft" both match."""

    lowered = (text or "").lower().replace("ß", "ss")
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(source, target)
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


#: Past-tense completion claims about a side effect.  Deliberately narrow: these
#: match "it was done", not "I can do that" or "shall I do that", because
#: offering to act is honest and must stay unimpeded.
_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(habe|hab)\b[^.!?]{0,60}\b(erstellt|gespeichert|angelegt|geschrieben|geloescht|geaendert|verifiziert|geprueft)\b",
     "first-person German completion claim"),
    (r"\b(wurde|wurden|ist|sind)\b[^.!?]{0,60}\b(erstellt|gespeichert|angelegt|geschrieben|geloescht|geaendert)\b",
     "German passive completion claim"),
    (r"\b(erfolgreich)\b[^.!?]{0,40}\b(erstellt|gespeichert|angelegt|geschrieben|verifiziert|geprueft)\b",
     "German success claim"),
    (r"\bexistenz\b[^.!?]{0,30}\b(geprueft|bestaetigt|verifiziert)\b", "German verification claim"),
    (r"\b(datei|ordner|verzeichnis|projekt)\b[^.!?]{0,40}\bexistiert\b", "German existence claim"),
    (r"\bi\s+(have\s+)?(created|saved|written|wrote|deleted|removed|renamed|verified|checked)\b",
     "first-person English completion claim"),
    (r"\b(has|have|was|were)\s+been\s+(created|saved|written|deleted|removed|verified)\b",
     "English passive completion claim"),
    (r"\bsuccessfully\s+(created|saved|written|wrote|deleted|removed|verified)\b", "English success claim"),
    (r"\b(the\s+)?(file|folder|directory|project)\b[^.!?]{0,30}\b(now\s+)?exists\b", "English existence claim"),
    (r"^\s*(done|erledigt|fertig)\b", "bare completion token"),
)

_COMPILED = tuple((re.compile(pattern, re.I | re.M), label) for pattern, label in _CLAIM_PATTERNS)


@dataclass(frozen=True)
class Claim:
    """A success claim found in generated text."""

    phrase: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {"phrase": self.phrase, "kind": self.kind}


def find_claim(text: str) -> Claim | None:
    """The first completion claim in ``text``, or ``None``.

    Returns the *first* rather than all of them because one unsupported claim is
    already disqualifying; counting them would be precision nobody acts on.
    """

    folded = _fold(text)
    for pattern, label in _COMPILED:
        match = pattern.search(folded)
        if match:
            return Claim(phrase=match.group(0).strip()[:120], kind=label)
    return None


def claims_success(text: str) -> bool:
    return find_claim(text) is not None


def correction(claim: Claim, *, language: str = "") -> str:
    """What the user is told instead of the unsupported claim.

    It names what was actually attempted -- nothing -- rather than apologising,
    because the user's next question is going to be "so did it happen or not"
    and the answer has to be in the first sentence.
    """

    german = language.startswith("de")
    if german:
        return (
            "Ich habe darauf geantwortet, als waere die Aktion ausgefuehrt worden -- das war sie nicht. "
            "Es wurde nichts ausgefuehrt und es gibt keinen Beleg (Receipt) fuer diesen Schritt. "
            "Formuliere die Anfrage bitte als konkrete Aktion (zum Beispiel: "
            "\"Erstelle die Datei X mit dem Inhalt Y\"), dann fuehre ich sie wirklich aus und zeige den Beleg."
        )
    return (
        "I answered as though that had been carried out. It was not. "
        "Nothing was executed and there is no receipt for this step. "
        "Ask for it as a concrete action (for example: \"create the file X containing Y\") "
        "and I will actually run it and show you the receipt."
    )
