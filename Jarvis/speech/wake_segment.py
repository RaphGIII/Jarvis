"""Keep the wake word out of the owner's request.

The wake detector already knows the trigger was "Zeus"; asking Whisper to
transcribe the same audio and then routing its guess ("Solls, wie geht es
dir?") is the live defect this module ends.  Two layers, both scoped to a
verified wake session:

1. **Audio.**  The listener starts the command recording at the end of the
   wake word (the detector confirms ~160 ms after it; the recording keeps
   exactly that much pre-roll), so the wake word is mostly not in the audio.
   A fast speaker still leaves its tail; hence:

2. **Text, session-scoped.**  When -- and only when -- the utterance came
   from a wake session, a leading token that (a) sits inside the wake-tail
   window by Whisper's word timestamps or has no timestamps, and (b) sounds
   like the wake word by a *shape* derived from the wake word itself (first
   consonant class, last consonant class, length) is removed.  "Solls",
   "Seus", "Zoiß", "Zeus" go; "Servus" (ends in s but starts differently
   after the vowel? no -- it keeps its length rule), "Jesus" and every
   ordinary word stay because the rule never runs outside a wake session
   and never rewrites anything but the leading token.

Nothing here is a global word replacement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

#: How far into the recording the wake word's tail can reach (pre-roll kept
#: by the listener, plus one frame of slack).
WAKE_TAIL_SECONDS = 0.45

_LEAD = re.compile(r"^\s*(?:hey|ok|okay|hallo|hi)?\s*", re.I)
_TOKEN = re.compile(r"^([^\W\d_]+)([\s,.;:!?–-]*)(.*)$", re.S | re.U)
_SIBILANT = set("szßc")
_VOWELS = set("aeiouyäöüàéè")


def _shape(word: str) -> tuple[str, str, int]:
    w = word.lower()
    first = "s" if w[:1] in _SIBILANT or w[:2] in {"ts", "ds", "tz"} else w[:1]
    last = "s" if w[-1:] in _SIBILANT or w[-2:] in {"ss", "ts", "tz"} else w[-1:]
    return first, last, len(w)


def sounds_like(token: str, wake_word: str) -> bool:
    """Does ``token`` have the wake word's sound shape (for a leading, isolated token)?"""

    t = re.sub(r"[^\w]", "", token.lower())
    if not t or not wake_word:
        return False
    if t == wake_word.lower():
        return True
    tf, tl, tn = _shape(t)
    wf, wl, wn = _shape(wake_word)
    if (tf, tl) != (wf, wl):
        return False
    if not any(ch in _VOWELS for ch in t):
        return False
    # the same number of vowel groups (one for "Zeus": "eu"; "Servus" has two)
    # and about the wake word's length: "solls" (5) vs "zeus" (4)
    groups = lambda w: len(re.findall(r"[aeiouyäöüàéè]+", w))
    return groups(t) <= groups(wake_word.lower()) and abs(tn - wn) <= 2 and tn <= wn + 1


@dataclass
class Segmentation:
    text: str
    removed: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "removed": self.removed, "reason": self.reason}


def strip_wake_word(text: str, *, wake_word: str, words: Iterable[dict[str, Any]] | None = None,
                    wake_session: bool = True) -> Segmentation:
    """The command content of a wake-session transcript."""

    original = text or ""
    if not wake_session or not original.strip():
        return Segmentation(original.strip(), reason="no wake session" if not wake_session else "empty")
    lead = _LEAD.match(original)
    body = original[lead.end():] if lead else original
    match = _TOKEN.match(body)
    if not match:
        return Segmentation(original.strip(), reason="no leading token")
    token, sep, rest = match.group(1), match.group(2), match.group(3)
    if not sounds_like(token, wake_word):
        return Segmentation(original.strip(), reason="leading token is not the wake word")
    first_word = next(iter(words or []), None)
    if first_word is not None:
        try:
            end = float(first_word.get("end", 0.0))
        except (TypeError, ValueError):
            end = 0.0
        if end > WAKE_TAIL_SECONDS + 0.4:
            return Segmentation(original.strip(), reason=f"leading token ends at {end:.2f}s, outside the wake tail")
    remainder = rest.strip()
    if remainder:
        remainder = remainder[0].upper() + remainder[1:]
    return Segmentation(remainder, removed=(original[:lead.end()] if lead else "") + token + sep.strip(), reason="wake word removed from the command")
