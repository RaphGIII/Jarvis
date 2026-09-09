"""What kind of failure this was, and therefore what to do about it.

A capability that fails is not one event.  ``AttributeError: '_hashlib.HASH'
object has no attribute 'hex_digest'`` and ``no such file: report.txt`` are
both "the capability returned ok=False", and treating them the same is what
produced two live defects at once:

* every failure was reported to the owner as an INSIGHT ("Fähigkeit ... ist
  ausgefallen"), a note about the system instead of the repair the system
  needed -- so a real implementation defect sat in the registry and the next
  request hit it again; and
* a request that named a file ZEUS was not allowed to read counted against the
  capability's health, so a correct capability was on its way to BROKEN for
  refusing to do something it was right to refuse.

So the failure is classified once, here, from the evidence the run produced,
and the classification -- not the fact of failure -- decides the next move:

===============  ==================================  ==========================
kind             what it means                       what ZEUS does
===============  ==================================  ==========================
``INPUT``        the request named something that    ask the owner for the
                 is not there, or did not name it    right input; the
                                                     capability is not at fault
``PERMISSION``   the act is refused by policy        explain what was refused
                                                     and what would allow it
``DEFECT``       the capability's own code broke     mark it BROKEN and send it
                                                     to Codex for repair
``TRANSIENT``    something outside failed for now    bounded retry / replan
``UNSUPPORTED``  the capability does not cover this  acquisition path
===============  ==================================  ==========================

Only ``DEFECT`` and ``TRANSIENT`` are the capability's health.  Only
``DEFECT`` is repairable by changing its code, and it is repairable on the
FIRST occurrence: a traceback out of the capability's own module is not a
flake that a second call might not reproduce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Kinds, as strings, so they survive a round trip through a receipt.
INPUT = "INPUT"
PERMISSION = "PERMISSION"
DEFECT = "DEFECT"
TRANSIENT = "TRANSIENT"
UNSUPPORTED = "UNSUPPORTED"
UNKNOWN = "UNKNOWN"

#: Checked in this order.  A ``PermissionError`` is a permission denial and not
#: a defect even though it is an exception name; a ``FileNotFoundError`` is bad
#: input and not a defect for the same reason.  Python's exception vocabulary
#: is therefore matched LAST, against what is left.
_PERMISSION = re.compile(
    r"permissionerror|permission denied|access is denied|zugriff verweigert"
    r"|is outside the action workspace|lies outside the workspace"
    r"|secret store|credential file|is not allowed|not authori[sz]ed"
    r"|requires? (?:owner )?authori|forbidden|\b40[13]\b",
    re.I,
)
_INPUT = re.compile(
    r"filenotfounderror|isadirectoryerror|notadirectoryerror"
    r"|no such file|file not found|does not exist|nicht gefunden"
    r"|\bnot a file\b|\bis a directory\b|kein[e]? datei"
    r"|is required\b|missing (?:required )?(?:argument|input|parameter)"
    r"|empty path|invalid path|no filename",
    re.I,
)
_TRANSIENT = re.compile(
    r"timeouterror|timed out|\btimeout\b|connectionerror|connectionreset"
    r"|connection (?:refused|reset|aborted)|temporarily unavailable"
    r"|\b(?:429|500|502|503|504)\b|rate limit|network is unreachable"
    r"|dns|ssl(?:error|certificate)|try again later",
    re.I,
)
_UNSUPPORTED = re.compile(
    r"notimplementederror|not implemented|not supported|unsupported"
    r"|no handler for|cannot handle",
    re.I,
)
#: What a broken implementation looks like from outside: a Python exception
#: raised by the capability's own code, or a runner that produced nothing.
_DEFECT = re.compile(
    r"\btraceback \(most recent call last\)"
    r"|attributeerror|nameerror|typeerror|valueerror|keyerror|indexerror"
    r"|importerror|modulenotfounderror|syntaxerror|indentationerror"
    r"|unboundlocalerror|zerodivisionerror|recursionerror|assertionerror"
    r"|unicodedecodeerror|unicodeencodeerror|jsondecodeerror"
    r"|did not produce valid output|produced no output"
    r"|has no attribute|object is not (?:callable|subscriptable|iterable)",
    re.I,
)


@dataclass(frozen=True)
class FailureClassification:
    """One failure, named -- with the evidence that named it."""

    kind: str
    reason: str
    #: Whether this failure is evidence about the capability's own health.
    counts_against_health: bool
    #: Whether changing the capability's code is what would fix it.
    repairable: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {"kind": self.kind, "reason": self.reason,
                "counts_against_health": self.counts_against_health,
                "repairable": self.repairable}


def _hit(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(0) if match else ""


def classify_failure(error: str, *, return_code: int | None = None) -> FailureClassification:
    """Name the failure from the text the run produced.

    ``return_code`` is read only as corroboration: a capability that crashed
    out of its interpreter is a defect even when its message says nothing.
    """

    text = str(error or "").strip()
    if not text and return_code not in (None, 0):
        return FailureClassification(DEFECT, f"the capability exited with code {return_code} and said nothing",
                                     counts_against_health=True, repairable=True)
    if not text:
        return FailureClassification(UNKNOWN, "the capability failed without saying why",
                                     counts_against_health=True, repairable=False)

    hit = _hit(_PERMISSION, text)
    if hit:
        return FailureClassification(PERMISSION, f"refused by policy ({hit})",
                                     counts_against_health=False, repairable=False)
    hit = _hit(_INPUT, text)
    if hit:
        return FailureClassification(INPUT, f"the request did not name a usable input ({hit})",
                                     counts_against_health=False, repairable=False)
    hit = _hit(_TRANSIENT, text)
    if hit:
        return FailureClassification(TRANSIENT, f"something outside failed for now ({hit})",
                                     counts_against_health=True, repairable=False)
    hit = _hit(_UNSUPPORTED, text)
    if hit:
        return FailureClassification(UNSUPPORTED, f"the capability does not cover this ({hit})",
                                     counts_against_health=False, repairable=False)
    hit = _hit(_DEFECT, text)
    if hit:
        return FailureClassification(DEFECT, f"the capability's own code failed ({hit})",
                                     counts_against_health=True, repairable=True)
    return FailureClassification(UNKNOWN, text[:160], counts_against_health=True, repairable=False)
