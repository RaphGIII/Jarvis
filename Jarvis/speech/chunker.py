"""Deciding when Jarvis has heard enough of its own thought to start saying it.

This is the piece that makes spoken answers feel immediate.  A model generating
at 77 tok/s takes several seconds to finish a paragraph, and waiting for it
means several seconds of silence after every question.  Speaking the first
phrase while the rest is still being generated removes almost all of that.

The tension is entirely about chunk size, and it pulls in both directions:

*Small chunks start sooner.*  Time-to-first-audio is roughly the time to
generate the first chunk plus the time to synthesise it.

*Small chunks sound worse.*  A TTS engine given three words has no room for
sentence prosody, so a reply chopped into fragments comes out flat and clipped
-- the robotic cadence the brief specifically warns against.

The resolution is that these two costs are not paid at the same moment.  Only
the FIRST chunk is on the latency path; by the time it is playing, generation is
several seconds ahead and later chunks can be as long as they like.  So the
chunker is deliberately *asymmetric*: it takes the earliest respectable boundary
it can find for the opening phrase, then relaxes into full sentences.

Everything else here is about not splitting in the wrong place.  "z.B." and
"3.14" and "Dr." all contain a full stop followed by something, and cutting
there produces an audible stumble.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Abbreviations whose trailing full stop does not end a sentence.  German and
#: English both, since the persona answers in whichever language it is asked in.
ABBREVIATIONS = frozenset(
    {
        # German
        "z.b", "u.a", "d.h", "bzw", "usw", "ggf", "evtl", "inkl", "exkl", "ca",
        "vgl", "bspw", "sog", "u.u", "i.d.r", "z.t", "nr", "abb", "bzgl", "etc",
        # English
        "mr", "mrs", "ms", "dr", "prof", "st", "vs", "eg", "ie", "approx",
        "fig", "no", "inc", "ltd", "jr", "sr", "e.g", "i.e",
    }
)

#: Strong boundaries: a sentence really ended here.
_SENTENCE_END = ".!?…"
#: Weak boundaries: acceptable places to breathe when a sentence runs long.
_CLAUSE_END = ",;:—–"


@dataclass
class Phrase:
    """One speakable unit."""

    text: str
    #: True for the phrase that starts playback; it is the latency-critical one.
    first: bool = False
    #: Why it was emitted, for diagnostics and tests.
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


@dataclass
class PhraseChunker:
    """Turns a token stream into phrases worth speaking.

    ``feed`` returns the phrases that became complete because of this token --
    usually none, occasionally one, rarely more.  ``flush`` returns whatever is
    left when generation ends.
    """

    #: The opening phrase may be as short as a complete sentence happens to be.
    #: "Ja." and "Good morning." are worth speaking immediately -- a complete
    #: sentence carries its own prosody at any length, and the first phrase is
    #: the only one whose generation time the user actually experiences. The
    #: minimum exists solely to reject a fragment that is not a sentence at all.
    first_min_chars: int = 2
    #: Later phrases aim for at least this, so the engine has prosodic room.
    min_chars: int = 80
    #: Above this, break at any clause boundary rather than run on forever.
    max_chars: int = 260
    #: Above this, break anywhere at all -- a model emitting no punctuation
    #: must not be able to withhold speech indefinitely.
    hard_max_chars: int = 400

    _buffer: str = field(default="", init=False)
    _emitted: int = field(default=0, init=False)

    # -- public ----------------------------------------------------------

    def feed(self, text: str) -> list[Phrase]:
        if not text:
            return []
        self._buffer += text
        found: list[Phrase] = []
        while True:
            phrase = self._take()
            if phrase is None:
                return found
            found.append(phrase)

    def flush(self) -> list[Phrase]:
        """Emit the remainder.  Called when the model has finished."""

        remaining = self._buffer.strip()
        self._buffer = ""
        if not remaining:
            return []
        phrase = Phrase(text=remaining, first=self._emitted == 0, reason="flush")
        self._emitted += 1
        return [phrase]

    def reset(self) -> None:
        self._buffer = ""
        self._emitted = 0

    @property
    def pending(self) -> str:
        return self._buffer

    # -- internals -------------------------------------------------------

    @property
    def _target(self) -> int:
        return self.first_min_chars if self._emitted == 0 else self.min_chars

    def _take(self) -> Phrase | None:
        """Cut one phrase off the front of the buffer, if one is ready."""

        buffer = self._buffer
        if not buffer.strip():
            return None

        cut, reason = self._find_cut(buffer)
        if cut is None:
            return None

        text = buffer[:cut].strip()
        if not text:
            return None
        self._buffer = buffer[cut:].lstrip()
        phrase = Phrase(text=text, first=self._emitted == 0, reason=reason)
        self._emitted += 1
        return phrase

    def _find_cut(self, buffer: str) -> tuple[int | None, str]:
        stripped_length = len(buffer.strip())

        # 1. A real sentence end, once we have enough to be worth speaking.
        for index, char in enumerate(buffer):
            if char not in _SENTENCE_END:
                continue
            end = self._end_of_terminator(buffer, index)
            if end is None:
                continue
            if len(buffer[:end].strip()) < self._target:
                continue
            return end, "sentence"

        # 2. Long enough that a clause boundary is better than waiting.
        if stripped_length >= self.max_chars:
            for index in range(len(buffer) - 1, -1, -1):
                if buffer[index] in _CLAUSE_END and len(buffer[: index + 1].strip()) >= self._target:
                    return index + 1, "clause"

        # 3. No punctuation at all. Break on a word boundary rather than let a
        #    model that never uses a full stop hold the speaker hostage.
        if stripped_length >= self.hard_max_chars:
            window = buffer[: self.hard_max_chars]
            space = window.rfind(" ")
            if space > self._target:
                return space + 1, "hard_limit"
            return self.hard_max_chars, "hard_limit"

        return None, ""

    def _end_of_terminator(self, buffer: str, index: int) -> int | None:
        """Where a sentence ending at ``index`` finishes, or None if it doesn't.

        Returns None when the character is not really a terminator -- a decimal
        point, an abbreviation, or a full stop we cannot yet judge because the
        next character has not been generated.
        """

        char = buffer[index]

        # Consume a run: "..." and "?!" are one ending, not three.
        end = index
        while end + 1 < len(buffer) and buffer[end + 1] in _SENTENCE_END:
            end += 1
        end += 1

        if end >= len(buffer):
            # The terminator is the last thing generated so far. We cannot tell
            # "end of sentence" from "3." mid-number until more arrives, and
            # guessing wrong is audible, so wait for the next token.
            return None

        following = buffer[end]
        if char == "." :
            if not (following.isspace() or following in "\"')]}"):
                # 3.14, file.txt, example.com -- not a sentence.
                return None
            if self._ends_with_abbreviation(buffer[:index]):
                return None
            if self._is_list_marker(buffer, index):
                return None
        elif not (following.isspace() or following in "\"')]}"):
            return None

        # Take any closing quote/bracket with the sentence it belongs to.
        while end < len(buffer) and buffer[end] in "\"')]}":
            end += 1
        return end

    @staticmethod
    def _is_list_marker(buffer: str, index: int) -> bool:
        """True for the "1." in "1. Erstens ..." -- an item number, not an end.

        Only when the digits open a line, because "Es kostet 5." really is a
        sentence and the two are otherwise indistinguishable.
        """

        start = buffer.rfind("\n", 0, index) + 1
        return buffer[start:index].strip().isdigit()

    @staticmethod
    def _ends_with_abbreviation(text: str) -> bool:
        match = re.search(r"([A-Za-zÄÖÜäöüß.]+)$", text)
        if not match:
            return False
        word = match.group(1).lower().strip(".")
        if not word:
            return False
        if word in ABBREVIATIONS:
            return True
        # A single letter before a full stop is an initial ("J. Smith"), not an
        # ending.
        return len(word) == 1


def split_for_speech(text: str, **options) -> list[str]:
    """Convenience: chunk a complete string as if it had been streamed."""

    chunker = PhraseChunker(**options)
    phrases = chunker.feed(text)
    phrases += chunker.flush()
    return [phrase.text for phrase in phrases]
