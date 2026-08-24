"""Working out what language the user is speaking, without asking a model.

Three parts of Jarvis need this answer and two of them need it *before* any
model has run: whisper decodes better when told the language, and the TTS voice
has to be chosen before there is anything to say.  Asking an LLM would put a
generation on the critical path of every utterance to learn something a lookup
settles in microseconds.

The method is stopword frequency plus a few orthographic signals.  It is not a
general-purpose language identifier and does not pretend to be -- it is tuned
for the handful of languages this system has voices and models for, and it says
so when it is unsure rather than guessing confidently.

Short input is where naive detectors embarrass themselves: "ok", "ja", "stop"
carry almost no signal, and a confident wrong answer means whisper gets told the
wrong language for the next ten minutes.  So :func:`detect` reports confidence,
and callers are expected to keep the previous language when it is low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Words that are common in one language and rare in the others here.  Chosen
#: for discrimination rather than raw frequency: "in" is common everywhere and
#: therefore worthless.
_MARKERS: dict[str, frozenset[str]] = {
    "de": frozenset(
        """der die das und nicht ist ich du wir ihr sie mit auf für von zu den dem
        eine einen einem eines auch noch nur schon wenn aber oder weil dass wie
        was wer wo wann warum kann soll muss will habe hast hat haben wird werden
        wurde bitte danke gut sehr mehr immer nie jetzt heute morgen gestern
        machen gemacht geht gehen kommt kommen sagen gesagt""".split()
    ),
    "en": frozenset(
        """the and is are was were you your they them this that these those with
        for from have has had will would can could should must please thanks
        what where when why how which who because but or not don't doesn't
        didn't there their here about into over under again still just only
        make made going come came said say""".split()
    ),
    "fr": frozenset(
        """le la les des une un est sont avec pour dans sur que qui quoi mais ou
        donc parce vous nous ils elles avoir être fait faire très bien merci
        s'il plaît aujourd'hui demain hier toujours jamais encore""".split()
    ),
    "es": frozenset(
        """el la los las una uno es son con para por que quien pero como donde
        cuando porque muy bien gracias favor hoy mañana ayer siempre nunca
        hacer hecho tiene tener puede poder""".split()
    ),
    "it": frozenset(
        """il lo la gli le una uno sono con per che chi come dove quando perché
        molto bene grazie favore oggi domani ieri sempre mai fare fatto
        avere essere puoi può""".split()
    ),
}

#: Orthography that is near-decisive on its own.
_LETTERS: dict[str, str] = {
    "de": "äöüßÄÖÜ",
    "fr": "àâçéèêëîïôùûœÀÂÇÉÈÊ",
    "es": "áéíóúñ¿¡ÁÉÍÓÚÑ",
    "it": "àèéìòù",
}

#: Below this many recognisable words, any verdict is a guess.
_MIN_WORDS = 3


@dataclass(frozen=True)
class LanguageGuess:
    """A language, and how much to trust it."""

    language: str
    #: 0..1.  Below ~0.5 the caller should keep whatever it was using.
    confidence: float
    #: What the decision was based on, for diagnostics.
    detail: str = ""

    @property
    def confident(self) -> bool:
        return self.confidence >= 0.5

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "confident": self.confident,
            "detail": self.detail,
        }


def detect(text: str, *, default: str = "") -> LanguageGuess:
    """Guess the language of a short piece of conversational text."""

    cleaned = (text or "").strip()
    if not cleaned:
        return LanguageGuess(default, 0.0, "empty input")

    words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", cleaned.lower(), flags=re.UNICODE)
    if not words:
        return LanguageGuess(default, 0.0, "no words")

    scores = {language: 0.0 for language in _MARKERS}
    hits = {language: 0 for language in _MARKERS}
    for word in words:
        for language, markers in _MARKERS.items():
            if word in markers:
                scores[language] += 1.0
                hits[language] += 1

    # Distinctive letters are strong evidence but must not by themselves
    # outvote a whole sentence of the other language's stopwords -- a single
    # borrowed "café" in English text should not make it French.
    for language, letters in _LETTERS.items():
        found = sum(1 for char in cleaned if char in letters)
        if found:
            scores[language] += min(2.0, found * 0.75)

    best = max(scores, key=lambda key: scores[key])
    best_score = scores[best]
    if best_score <= 0:
        return LanguageGuess(default, 0.0, "no markers matched")

    total = sum(scores.values()) or 1.0
    share = best_score / total

    # Confidence combines "how clearly did this language win" with "was there
    # enough text to mean anything". A one-word message can be right but is
    # never trustworthy, and treating it as such is how a detector locks onto
    # the wrong language and stays there.
    evidence = min(1.0, len(words) / 12.0)
    recognised = hits[best] + (1 if any(c in _LETTERS.get(best, "") for c in cleaned) else 0)
    if recognised < _MIN_WORDS:
        evidence *= 0.5

    confidence = round(min(1.0, share * evidence * 1.6), 3)
    detail = f"{hits[best]} marker(s) of {len(words)} word(s), share {share:.2f}"
    return LanguageGuess(best, confidence, detail)


def stable_language(text: str, *, current: str = "", default: str = "de") -> str:
    """The language to actually use, given what was being used before.

    Switching only on a confident detection is the point.  A conversation in
    German peppered with "ok" and "stop" must not flip back and forth, because
    every flip changes the recogniser's hint and the voice mid-conversation --
    which sounds far worse than occasionally answering in the wrong language.
    """

    guess = detect(text, default=current or default)
    if guess.confident:
        return guess.language
    return current or default


#: Languages Jarvis can currently hear, think and speak in.  Whisper handles far
#: more, but a voice has to exist for a language to be usable end to end.
SUPPORTED = ("de", "en", "fr", "es", "it")


def language_name(code: str) -> str:
    return {
        "de": "German",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
    }.get((code or "").lower().split("-")[0], code or "unknown")
