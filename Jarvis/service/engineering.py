"""Who does engineering work, decided once, for every path that needs it.

There were two answers to that question and they disagreed. The capability
paths asked Codex first and only built locally when told to. The
self-development path -- the one an owner actually reaches by saying "bring dir
das bei" -- ran the local coder unconditionally and treated Codex as the
fallback for its failures. Measured live on 2026-09-08, mission ``d1309425e9``:

    12:29:59  BUILD      BUILD_LOCAL, attempt 1, budget 2400s
    12:48:15  BUILD      SELF_DEVELOPMENT_CANDIDATE_REJECTED
    12:50:56  ESCALATE   submitting to expert
    13:04:49  ESCALATE   expert completed in 832.9s; 7 changed files

Eighteen minutes of the local model producing a rejected candidate, then Codex
doing the work in fourteen. The owner's sentence had said "bring dir diese
Fähigkeit bei"; the architecture says an engineering need checks Codex first;
the code did the opposite, and nothing in the routing evidence showed it.

So the decision lives here, once, and every entry point asks this module:

    chat request · explicit self-development · missing capability ·
    broken capability repair · mission-triggered engineering · corrections
    that need code

The policy, in full:

*   An existing HEALTHY capability is not engineering. It runs. This module is
    never consulted for it.
*   Anything else -- a missing capability, a broken one, a change to ZEUS
    itself -- checks :class:`~capabilities.codex.CodexAvailability` FIRST.
*   Codex READY: Codex is the engineer.
*   Codex anything else: the work is queued and the owner is told the truth.
    Not attempted locally, not faked, not quietly downgraded.
*   BUILD_LOCAL is never an automatic fallback. It runs only when the owner
    has explicitly authorized it for that piece of work.

The last rule is the one the old path broke, and it is the reason this module
returns a decision object rather than a boolean: "we did not do this, and here
is exactly why" has to survive all the way to the Activity log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EngineeringNeed(str, Enum):
    """What kind of work is being asked for."""

    #: A capability the registry does not have.
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    #: A capability that exists and cannot be trusted.
    CAPABILITY_BROKEN = "CAPABILITY_BROKEN"
    #: A change to ZEUS itself -- its own code, its own behaviour.
    CORE_ENGINEERING = "CORE_ENGINEERING"


class Engineer(str, Enum):
    """Who is going to write the code."""

    CODEX = "CODEX"
    #: The local coder. Only ever by explicit owner authorization.
    BUILD_LOCAL = "BUILD_LOCAL"
    #: Nobody, right now. The work is queued and the owner is told.
    NONE = "NONE"


@dataclass
class EngineerDecision:
    """The routing decision, and enough of its reasoning to audit it."""

    need: EngineeringNeed
    engineer: Engineer
    reason: str
    codex_state: str = ""
    codex_detail: str = ""
    codex_checked: bool = True
    owner_authorized_local: bool = False
    queued: bool = True

    @property
    def is_codex(self) -> bool:
        return self.engineer is Engineer.CODEX

    @property
    def is_local(self) -> bool:
        return self.engineer is Engineer.BUILD_LOCAL

    @property
    def proceeds(self) -> bool:
        return self.engineer is not Engineer.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "need": self.need.value,
            "engineer": self.engineer.value,
            "reason": self.reason,
            "codex_state": self.codex_state,
            "codex_detail": self.codex_detail[:200],
            "codex_checked": self.codex_checked,
            "owner_authorized_local": self.owner_authorized_local,
            "queued": self.queued,
            "build_local_invocations": 1 if self.is_local else 0,
        }

    def owner_sentence(self, *, german: bool = True) -> str:
        """What to tell the owner, without dressing any of it up."""

        if self.engineer is Engineer.CODEX:
            return ("Codex übernimmt das (isolierter Arbeitsbaum, Verifikation, dann Freigabe durch dich)."
                    if german else
                    "Codex is taking this (isolated worktree, verification, then your authorization).")
        if self.engineer is Engineer.BUILD_LOCAL:
            return ("Du hast das lokale Coder-Modell freigegeben — ich baue es lokal."
                    if german else
                    "You authorized the local coder — I am building it locally.")
        return (f"Codex ist gerade nicht verfügbar ({self.codex_state}). "
                f"Ich baue das nicht ersatzweise lokal, sondern habe es vorgemerkt."
                if german else
                f"Codex is unavailable right now ({self.codex_state}). "
                f"I am not building it locally instead; it is queued.")


def choose_engineer(
    need: EngineeringNeed,
    *,
    availability: Any = None,
    owner_authorized_local: bool = False,
) -> EngineerDecision:
    """The one decision. Codex first, always; BUILD_LOCAL only by authorization.

    ``availability`` is anything with a ``status()`` returning a
    :class:`~capabilities.codex.CodexAvailability`. It is consulted before
    anything else, because "is an engineer available" has to be answered before
    "who does the work", and the old path answered them in the other order.

    ``owner_authorized_local`` is the only thing that can select the local
    coder. It is not a policy flag, not a mission field, and not a model's
    opinion: it is the owner having said so for this piece of work.
    """

    state, detail, checked = "UNKNOWN", "", False
    ready = False
    if availability is not None:
        try:
            status = availability.status()
            checked = True
            state = str(getattr(getattr(status, "state", ""), "value", getattr(status, "state", "")) or "UNKNOWN")
            detail = str(getattr(status, "detail", "") or "")
            ready = bool(getattr(status, "ready", False))
        except Exception as exc:  # noqa: BLE001 - availability is a state, not a crash
            state, detail, checked = "ERROR", f"{type(exc).__name__}: {exc}", True
    else:
        state, detail = "NOT_CONFIGURED", "no Codex availability was wired in"

    if ready:
        return EngineerDecision(
            need=need, engineer=Engineer.CODEX,
            reason="Codex is READY and owns engineering work",
            codex_state=state, codex_detail=detail, codex_checked=checked, queued=False,
        )
    if owner_authorized_local:
        # The owner asked for it explicitly, knowing Codex is not available.
        # This is the only door to the local coder, and it is not automatic.
        return EngineerDecision(
            need=need, engineer=Engineer.BUILD_LOCAL,
            reason=f"Codex is {state}; the owner explicitly authorized the local coder",
            codex_state=state, codex_detail=detail, codex_checked=checked,
            owner_authorized_local=True, queued=False,
        )
    return EngineerDecision(
        need=need, engineer=Engineer.NONE,
        reason=f"Codex is {state} and the local coder is not an automatic fallback",
        codex_state=state, codex_detail=detail, codex_checked=checked, queued=True,
    )


def default_availability(gateway: Any = None) -> Any:
    """The availability source the whole system shares."""

    from capabilities.codex import CodexAvailabilityCache

    return CodexAvailabilityCache(gateway=gateway) if gateway is not None else CodexAvailabilityCache()


#: What the owner has to say, in their own words, to put the local coder to
#: work. Deliberately explicit: "mach das lokal" is a decision, and nothing
#: shorter should be readable as one.
_LOCAL_AUTHORIZATION = (
    "build_local", "buildlocal", "lokal bauen", "lokales modell", "lokalem modell",
    "lokale modell", "lokaler coder", "lokalen coder", "lokale coder",
    "local coder", "local model", "locally instead",
    "mach es lokal", "mach das lokal", "bau es lokal", "bau das lokal",
)


def owner_authorized_local_build(text: str) -> bool:
    """Whether this owner sentence authorizes the local coder for this work.

    Read from the owner's own words, never inferred from a failure, a policy
    default or a model's suggestion. "Bring dir das bei" is a request for the
    capability, not permission to build it with the weaker engineer.
    """

    from capabilities.models import _fold

    folded = _fold(str(text or ""))
    return any(marker in folded for marker in _LOCAL_AUTHORIZATION)
