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

With the model gateway the question gains candidates and a method.  The
engineering task is a typed vector; every engineer -- Codex (subscription,
zero marginal cost), ``engineer.standard`` and ``engineer.frontier`` (metered
API roles) -- has a learned reliability per engineering class; the cheapest
one predicted to be reliable enough is chosen *before* anything runs.  No
"try Codex, then Opus, then Fable": a large new subsystem goes to the
frontier engineer directly.  Metered engineers exist only in BUILD mode,
only with the owner's spending policy allowing paid billing, and only within
the budget governor's caps; without those, Codex remains the engineer when
it is READY, and otherwise the work is queued as before.
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
    #: A metered engineer role behind an API (engineer.standard / engineer.frontier),
    #: driven by :class:`experts.api_engineer.ApiEngineerExpert`.
    API = "API"
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
    #: The gateway role that does the work (engineer.codex / engineer.standard / engineer.frontier).
    role: str = ""
    #: The expert-gateway provider name to submit to ("codex" or the role name).
    provider_name: str = ""
    task_class: str = ""
    q: float = 0.0
    tau: float = 0.0
    estimated_eur: float = 0.0
    estimate_range_eur: tuple[float, float] = (0.0, 0.0)
    mode: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineerDecision":
        """The decision a mission recorded, back as an object (to continue after the owner's go)."""

        data = dict(data or {})
        rng = data.get("estimate_range_eur") or (0.0, 0.0)
        return cls(
            need=EngineeringNeed(str(data.get("need") or EngineeringNeed.CORE_ENGINEERING.value)),
            engineer=Engineer(str(data.get("engineer") or Engineer.NONE.value)),
            reason=str(data.get("reason", "")), codex_state=str(data.get("codex_state", "")),
            codex_detail=str(data.get("codex_detail", "")), codex_checked=bool(data.get("codex_checked", True)),
            owner_authorized_local=bool(data.get("owner_authorized_local", False)), queued=bool(data.get("queued", False)),
            role=str(data.get("role", "")), provider_name=str(data.get("provider_name", "")),
            task_class=str(data.get("task_class", "")), q=float(data.get("q", 0.0) or 0.0), tau=float(data.get("tau", 0.0) or 0.0),
            estimated_eur=float(data.get("estimated_eur", 0.0) or 0.0),
            estimate_range_eur=(float(rng[0]), float(rng[1])) if len(rng) == 2 else (0.0, 0.0),
            mode=str(data.get("mode", "")), candidates=list(data.get("candidates") or []),
        )

    @property
    def is_codex(self) -> bool:
        return self.engineer is Engineer.CODEX

    @property
    def is_api(self) -> bool:
        return self.engineer is Engineer.API

    @property
    def uses_expert_gateway(self) -> bool:
        return self.engineer in {Engineer.CODEX, Engineer.API}

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
            "role": self.role,
            "provider_name": self.provider_name,
            "task_class": self.task_class,
            "q": round(self.q, 4),
            "tau": round(self.tau, 4),
            "estimated_eur": round(self.estimated_eur, 4),
            "estimate_range_eur": [round(self.estimate_range_eur[0], 4), round(self.estimate_range_eur[1], 4)],
            "mode": self.mode,
            "candidates": list(self.candidates),
        }

    def owner_sentence(self, *, german: bool = True) -> str:
        """What to tell the owner, without dressing any of it up."""

        if self.engineer is Engineer.CODEX:
            return ("Codex übernimmt das (isolierter Arbeitsbaum, Verifikation, dann Freigabe durch dich)."
                    if german else
                    "Codex is taking this (isolated worktree, verification, then your authorization).")
        if self.engineer is Engineer.API:
            low, high = self.estimate_range_eur
            who = "der Frontier-Engineer" if self.role.endswith("frontier") else "der Standard-Engineer"
            who_en = "the frontier engineer" if self.role.endswith("frontier") else "the standard engineer"
            return (f"{who.capitalize()} ({self.role}) übernimmt das – geschätzt €{low:.2f}–€{high:.2f}, "
                    f"hartes Maximum durch das Budget; isolierter Arbeitsbaum, Verifikation, dann Freigabe durch dich."
                    if german else
                    f"{who_en.capitalize()} ({self.role}) is taking this – estimated €{low:.2f}–€{high:.2f}, "
                    f"hard maximum set by the budget; isolated worktree, verification, then your authorization.")
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
    task: Any = None,
    gateway: Any = None,
    mode: Any = None,
    prompt: str = "",
) -> EngineerDecision:
    """The one decision. Codex first, always; BUILD_LOCAL only by authorization.

    With ``task`` (a :class:`gateway.task.TaskVector`) and ``gateway`` (the
    :class:`gateway.gateway.ModelGateway`) the choice is made the gateway's
    way: the cheapest engineer whose learned reliability meets the task's
    requirement, decided before execution.  Codex is that engineer whenever
    it is READY and reliable enough for the class; a metered engineer is
    chosen directly when Codex is not predicted to manage it and the mode,
    the owner's spending policy and the budget allow.

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

    if task is not None and gateway is not None:
        decided = _choose_with_gateway(need, task, gateway, mode=mode, prompt=prompt, codex_ready=ready,
                                       codex_state=state, codex_detail=detail, codex_checked=checked)
        if decided is not None:
            return decided

    if ready:
        return EngineerDecision(
            need=need, engineer=Engineer.CODEX,
            reason="Codex is READY and owns engineering work",
            codex_state=state, codex_detail=detail, codex_checked=checked, queued=False,
            role="engineer.codex", provider_name="codex",
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


def _choose_with_gateway(need: EngineeringNeed, task: Any, gateway: Any, *, mode: Any, prompt: str, codex_ready: bool,
                         codex_state: str, codex_detail: str, codex_checked: bool) -> EngineerDecision | None:
    """Rank Codex and the API engineers on reliability and cost; pick before running."""

    from gateway.modes import ChatMode
    from gateway.router import RouteKind

    chat_mode = ChatMode.parse(mode) if mode is not None else ChatMode.AUTO
    try:
        gateway.set_subscription_available(lambda name: codex_ready if name == "codex" else False)
        decision = gateway.router.decide(task, chat_mode, None, prompt=prompt or task.facts.get("text", "") or "engineering",
                                         expected_output_tokens=4096)
    except Exception:  # noqa: BLE001 - a gateway that cannot answer leaves the classic rule in charge
        return None
    candidates = [c.to_dict() for c in decision.candidates]
    common = dict(need=need, codex_state=codex_state, codex_detail=codex_detail, codex_checked=codex_checked,
                  task_class=task.task_class.value, tau=decision.tau, mode=chat_mode.value, candidates=candidates)
    if decision.kind is RouteKind.MODEL and decision.role:
        estimate = decision.estimate
        low, high = estimate.range_eur() if estimate else (0.0, 0.0)
        if decision.role == "engineer.codex":
            reason = (f"Codex is READY and {decision.reason}" if decision.meets_threshold else
                      f"Codex is READY; no engineer met the reliability bar for {task.task_class.value} "
                      f"(q={decision.q:.2f} < tau={decision.tau:.2f}) and no metered engineer is permitted in {chat_mode.value}")
            return EngineerDecision(engineer=Engineer.CODEX, reason=reason, queued=False,
                                    role="engineer.codex", provider_name="codex", q=decision.q, **common)
        return EngineerDecision(engineer=Engineer.API, reason=decision.reason, queued=False, role=decision.role,
                                provider_name=decision.role, q=decision.q, estimated_eur=estimate.estimated_eur if estimate else 0.0,
                                estimate_range_eur=(low, high), **common)
    # Nothing met the bar or nothing was permitted.  Codex, when READY, still
    # does zero-cost work the owner asked for -- said plainly as "best
    # available", never as "reliable enough".
    if codex_ready:
        codex = next((c for c in candidates if c["role"] == "engineer.codex"), None)
        return EngineerDecision(engineer=Engineer.CODEX, queued=False, role="engineer.codex", provider_name="codex",
                                q=float(codex["q"]) if codex else 0.0,
                                reason=("Codex is READY; no engineer met the reliability bar and no metered engineer is permitted "
                                        f"({decision.reason[:160]})"), **common)
    return None


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
