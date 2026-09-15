"""Provider output, made safe to render -- without touching what it says.

Models answer medical and chemical questions in a mix of Markdown, LaTeX and
mhchem; transports and JSON layers add their own damage: doubled
backslashes, HTML entities shown literally, U+FFFD where a byte was lost.
This module repairs the *encoding* of an answer and normalises chemistry
markup the renderer does not support into Unicode -- ``\\ce{HCO3-}`` becomes
HCO₃⁻ -- and changes nothing else.  Fenced code is left exactly as written.

It runs on the stream as it arrives (with a small carry for a token cut by a
chunk boundary) and once more, identically, on the completed answer, so the
persisted message is the normalised stream and nothing more.
"""

from __future__ import annotations

import html
import re

SUBSCRIPT = str.maketrans("0123456789+-()", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎")
SUPERSCRIPT = str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾")

_ARROWS = [("<=>>", "⇌"), ("<<=>", "⇌"), ("<=>", "⇌"), ("<->", "↔"), ("->", "→"), ("<-", "←")]
_FENCE = re.compile(r"(```.*?(?:```|$))", re.S)
_ENTITY = re.compile(r"&(#\d{1,6}|#x[0-9a-fA-F]{1,6}|[a-zA-Z]{2,10});")
_CE = re.compile(r"\\(?:ce|pu)\{((?:[^{}]|\{[^{}]*\})*)\}")
_DOUBLE_BACKSLASH_CMD = re.compile(r"\\\\(?=[a-zA-Z(\[\]){}])")
_CHARGE = re.compile(r"\^(\{[^}]*\}|[0-9]*[+\-])")
_SUBSCRIPT_DIGITS = re.compile(r"(?<=[A-Za-z)\]])(\d+)")
_CARRY = re.compile(r"(\\[a-zA-Z]{0,12}\{?[^{}]{0,40}|&#?[a-zA-Z0-9]{0,10}|\\|`{1,2})$")


def chemistry_to_unicode(formula: str) -> str:
    """``H2O`` -> H₂O, ``Ca^2+`` -> Ca²⁺, ``HCO3-`` -> HCO₃⁻, ``2H2 + O2 -> 2H2O`` -> 2H₂ + O₂ → 2H₂O."""

    text = formula.strip()
    for old, new in _ARROWS:
        text = text.replace(old, new)

    def charge(match: re.Match) -> str:
        body = match.group(1).strip("{}")
        return body.translate(SUPERSCRIPT)

    text = _CHARGE.sub(charge, text)
    # A trailing charge sign after a subscripted group: HCO3- -> HCO₃⁻, Na+ -> Na⁺.
    text = re.sub(r"(?<=[A-Za-z0-9)\]])([+\-])(?=$|[\s,;.)\]→⇌↔←])", lambda m: m.group(1).translate(SUPERSCRIPT), text)
    text = _SUBSCRIPT_DIGITS.sub(lambda m: m.group(1).translate(SUBSCRIPT), text)
    text = re.sub(r"_\{?(\d+)\}?", lambda m: m.group(1).translate(SUBSCRIPT), text)
    return text


def _normalise_prose(text: str) -> str:
    if not text:
        return text
    out = text.replace("\ufffd", "")
    out = _DOUBLE_BACKSLASH_CMD.sub("\\\\", out) if "\\\\" in out else out
    if "&" in out:
        out = _ENTITY.sub(lambda m: html.unescape(m.group(0)), out)
    if "\\ce{" in out or "\\pu{" in out:
        out = _CE.sub(lambda m: chemistry_to_unicode(m.group(1)), out)
    return out


def normalize(text: str) -> str:
    """The whole answer: prose repaired, chemistry markup normalised, code untouched."""

    parts = _FENCE.split(str(text or ""))
    return "".join(part if part.startswith("```") else _normalise_prose(part) for part in parts)


class StreamNormalizer:
    """The same normalisation, chunk by chunk, with a carry for tokens a chunk boundary cut.

    ``feed`` returns what may be shown now; ``finish`` returns the rest.  The
    concatenation of everything returned equals ``normalize`` of the whole
    stream, so the persisted answer and the shown one are one text.
    """

    def __init__(self) -> None:
        self._carry = ""
        self._in_fence = False

    def feed(self, chunk: str) -> str:
        text = self._carry + str(chunk or "")
        self._carry = ""
        if not text:
            return ""
        # Inside a fence nothing is normalised; a fence marker itself may straddle chunks.
        fence_count = text.count("```")
        if self._in_fence:
            if fence_count == 0:
                # A closing fence may be cut mid-way: hold back a trailing backtick or two.
                partial = re.search(r"`{1,2}$", text)
                if partial:
                    self._carry = text[partial.start():]
                    return text[: partial.start()]
                return text
            head, _, rest = text.partition("```")
            self._in_fence = False
            return head + "```" + self.feed(rest)
        if fence_count:
            head, _, rest = text.partition("```")
            self._in_fence = True
            return _normalise_prose(self._release(head)) + "```" + self.feed(rest)
        match = _CARRY.search(text)
        if match and match.start() > 0 or (match and match.start() == 0 and len(text) < 60):
            self._carry = text[match.start():]
            text = text[: match.start()]
        elif match:
            self._carry = ""
        return _normalise_prose(text)

    def _release(self, text: str) -> str:
        return text

    def finish(self) -> str:
        rest, self._carry = self._carry, ""
        return rest if self._in_fence else _normalise_prose(rest)
