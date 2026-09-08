"""Turn one owner request into the capability it is an instance of.

A request always arrives with its particulars in it -- *this* file, *that*
folder, "Rammstein". Handing that sentence to the engineer verbatim produces a
capability whose stored subject words include the particulars, so
``zeus_acceptance.txt`` becomes part of what a checksum capability declares
itself to be for. Every later request naming that file then matches it, and no
request naming a different file does. That is the opposite of reusable, and it
is how a registry fills up with near-duplicates of the same primitive.

So the sentence is split before anything is built: the *shape* of the request
becomes the capability's goal and its indexed vocabulary, and the particulars
stay with the request that is waiting to be answered. Nothing is thrown away --
the original goal is still what the owner asked and still what gets resumed.

Deliberately syntactic, not a model call. Paths, filenames, quoted literals and
drive letters are recognisable without understanding the sentence, and a
resolver that is wrong here is wrong in a way that is visible in the manifest
rather than in a model's reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: An absolute Windows path, a UNC path, or a POSIX path with at least one
#: separator. Bare relative names are handled by :data:`_FILENAME`.
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|(?<![\w.])/)[^\s\"'<>|]{2,}")

#: A bare filename with a recognisable extension.
_FILENAME = re.compile(r"\b[\w\-.]{1,60}\.(?:txt|md|py|json|csv|ya?ml|log|ini|cfg|html?|js|toml|pdf|png|jpe?g|zip|mp3|mp4|docx?|xlsx?)\b", re.I)

#: A quoted literal: almost always the specific thing, not the kind of thing.
_QUOTED = re.compile(r"[\"'\u201c\u201e\u00ab]([^\"'\u201d\u201c\u00bb]{1,80})[\"'\u201d\u00bb]")

#: A bare drive root ("D:", "D:\").
_DRIVE = re.compile(r"\b[A-Za-z]:\\?(?![\w\\/])")

_GERMAN_MARKERS = ("der ", "die ", "das ", "eine", "einen", "welche", "wie ", "mir ", "mach", "datei", "ordner", "bitte")


@dataclass
class GeneralizedGoal:
    """The reusable shape of a request, with what was removed kept beside it."""

    goal: str
    original: str
    particulars: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.particulars)

    def to_dict(self) -> dict[str, object]:
        return {"goal": self.goal, "original": self.original, "particulars": list(self.particulars)}


def generalize(text: str) -> GeneralizedGoal:
    """Replace the particulars of a request with what they are an example of."""

    original = " ".join(str(text or "").split())
    if not original:
        return GeneralizedGoal("", "", [])
    german = _looks_german(original)
    particulars: list[str] = []

    def swap(pattern: re.Pattern[str], replacement: str, source: str, *, group: int = 0) -> str:
        def _sub(match: re.Match[str]) -> str:
            taken = match.group(group).strip()
            if taken and taken not in particulars:
                particulars.append(taken)
            return replacement

        return pattern.sub(_sub, source)

    goal = original
    goal = swap(_PATH, "einem Pfad" if german else "a path", goal)
    goal = swap(_FILENAME, "einer Datei" if german else "a file", goal)
    goal = swap(_QUOTED, "einem Wert" if german else "a given value", goal, group=1)
    goal = swap(_DRIVE, "einem Laufwerk" if german else "a drive", goal)
    goal = _COLLAPSE_DE.sub(r"\2", goal)
    goal = _COLLAPSE_EN.sub(r"\2", goal)
    goal = " ".join(goal.split())
    return GeneralizedGoal(goal or original, original, particulars)


#: "der Datei einer Datei" is what a naive substitution leaves behind, because
#: the owner named the kind of thing and then the thing. Only the kind survives.
_COLLAPSE_DE = re.compile(r"\b(der|die|das|des|dem|den)\s+(?:Datei|Ordner|Verzeichnis|Pfad)\s+(einer Datei|einem Pfad|einem Laufwerk)\b", re.I)
_COLLAPSE_EN = re.compile(r"\b(the)\s+(?:file|folder|directory|path|drive)\s+(a file|a path|a drive)\b", re.I)


def generic_keywords(keywords: list[str], particulars: list[str]) -> list[str]:
    """Drop the vocabulary that came from the particulars.

    A capability indexed under ``zeus_acceptance`` and ``txt`` answers requests
    about one file and no others, which is a lookup table with extra steps.
    """

    banned = {
        term.lower()
        for value in particulars
        for term in re.split(r"[^A-Za-z0-9_]+", str(value))
        if term
    }
    banned |= {term.lower().rstrip("_") for term in banned}
    return [word for word in keywords if word.lower() not in banned]


def _looks_german(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(marker in lowered for marker in _GERMAN_MARKERS)
