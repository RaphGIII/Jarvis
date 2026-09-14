"""Capability semantic contracts (manifest v2): what a capability does to the world.

A capability is not a phrase to be matched; it is a transformation of world
state.  The contract says, in typed tokens, what it needs and what it leaves
behind:

    consumes:       facts that must hold (and are used up conceptually)
    preconditions:  facts that must hold (and stay)
    produces:       facts that hold afterwards (artefacts, records)
    effects:        facts that hold afterwards (state changes, events)
    goals:          the owner intents this capability serves
    events:         world events that make this capability relevant

Tokens are ``snake_case`` identifiers, optionally dotted (``chess.game_record``
and ``chess_game_record`` are both fine); they are the vocabulary the planner
searches over, so the same word must mean the same thing across capabilities.
The contract is generic: nothing here knows what chess, a file or a song is.

Classes for scoring are small closed sets so the planner can price a step:
``risk_class`` (harmless / reversible / irreversible), ``latency_class``
(fast / medium / slow), ``cost_class`` (free / cheap / metered).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Iterable

TOKEN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")

RISK_CLASSES: tuple[str, ...] = ("harmless", "reversible", "irreversible")
LATENCY_CLASSES: tuple[str, ...] = ("fast", "medium", "slow")
COST_CLASSES: tuple[str, ...] = ("free", "cheap", "metered")

#: Contract fields that hold token lists, in the order the manifest shows them.
TOKEN_FIELDS: tuple[str, ...] = ("goals", "events", "consumes", "produces", "preconditions", "effects")


class ContractError(ValueError):
    """A contract that cannot be planned over: malformed tokens or classes."""


def normalise_token(value: Any) -> str:
    """``Chess Game Record`` -> ``chess_game_record``; dotted ids kept."""

    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_.]", "", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text


def _tokens(values: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        token = normalise_token(value)
        if token and token not in out:
            out.append(token)
    return out


@dataclass(frozen=True)
class SemanticContract:
    goals: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    implementation_entrypoint: str = ""
    related_projects: list[str] = field(default_factory=list)
    domain: str = ""
    risk_class: str = "harmless"
    latency_class: str = "fast"
    cost_class: str = "free"
    #: True when the tokens were inferred from an older manifest rather than
    #: declared by its author.  The planner still uses them; the catalog says so.
    inferred: bool = False

    # -- derived -----------------------------------------------------------

    @property
    def requires(self) -> frozenset[str]:
        """Everything that must hold before the capability runs."""

        return frozenset(self.consumes) | frozenset(self.preconditions)

    @property
    def provides(self) -> frozenset[str]:
        """Everything that holds afterwards."""

        return frozenset(self.produces) | frozenset(self.effects)

    @property
    def declared(self) -> bool:
        return bool(self.provides or self.goals or self.events)

    def validate(self) -> list[str]:
        problems: list[str] = []
        for name in TOKEN_FIELDS:
            for token in getattr(self, name):
                if not TOKEN.match(token):
                    problems.append(f"{name}: {token!r} is not a snake_case token")
        if self.risk_class not in RISK_CLASSES:
            problems.append(f"risk_class {self.risk_class!r} not in {RISK_CLASSES}")
        if self.latency_class not in LATENCY_CLASSES:
            problems.append(f"latency_class {self.latency_class!r} not in {LATENCY_CLASSES}")
        if self.cost_class not in COST_CLASSES:
            problems.append(f"cost_class {self.cost_class!r} not in {COST_CLASSES}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Compact: empty fields and default classes are omitted."""

        defaults = {"risk_class": "harmless", "latency_class": "fast", "cost_class": "free"}
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, list):
                if value:
                    out[f.name] = list(value)
            elif isinstance(value, bool):
                if value:
                    out[f.name] = value
            elif value not in ("", None) and value != defaults.get(f.name):
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, strict: bool = False) -> "SemanticContract":
        data = dict(data or {})
        contract = cls(
            goals=_tokens(data.get("goals")),
            events=_tokens(data.get("events")),
            consumes=_tokens(data.get("consumes")),
            produces=_tokens(data.get("produces")),
            preconditions=_tokens(data.get("preconditions")),
            effects=_tokens(data.get("effects")),
            permissions=[str(p) for p in (data.get("permissions") or []) if str(p).strip()],
            dependencies=[str(p) for p in (data.get("dependencies") or []) if str(p).strip()],
            tests=[str(p) for p in (data.get("tests") or []) if str(p).strip()],
            implementation_entrypoint=str(data.get("implementation_entrypoint") or ""),
            related_projects=[str(p) for p in (data.get("related_projects") or []) if str(p).strip()],
            domain=normalise_token(data.get("domain")) if data.get("domain") else "",
            risk_class=str(data.get("risk_class") or "harmless").strip().lower(),
            latency_class=str(data.get("latency_class") or "fast").strip().lower(),
            cost_class=str(data.get("cost_class") or "free").strip().lower(),
            inferred=bool(data.get("inferred", False)),
        )
        problems = contract.validate()
        if problems and strict:
            raise ContractError("; ".join(problems))
        return contract


def infer_contract(capability_id: str, *, description: str = "", family: str = "", permissions: Iterable[str] = (),
                   goal_types: Iterable[str] = (), preconditions: Iterable[str] = (), latency_class: str = "",
                   entrypoint: str = "", tests: Iterable[str] = ()) -> SemanticContract:
    """A minimal contract for a manifest written before contracts existed.

    Honest and generic: the capability's own id becomes its result fact
    (``archive.zip.create`` produces ``archive.zip.create.result``), its
    ``goal_types`` become goals, its ``preconditions`` stay preconditions.
    Such a capability can end a plan but cannot feed another one until an
    author declares what it really produces; the catalog marks it inferred.
    """

    cid = normalise_token(capability_id)
    goals = _tokens(goal_types)
    domain = normalise_token(family) if family else (cid.split(".", 1)[0] if "." in cid else "")
    lat = str(latency_class or "").strip().lower()
    return SemanticContract(
        goals=goals,
        produces=[f"{cid}.result"] if cid else [],
        preconditions=_tokens(preconditions),
        permissions=[str(p) for p in permissions if str(p).strip()],
        implementation_entrypoint=str(entrypoint or ""),
        tests=[str(t) for t in tests if str(t).strip()],
        domain=domain,
        latency_class=lat if lat in LATENCY_CLASSES else ("slow" if lat in {"long", "batch"} else "fast"),
        risk_class="reversible" if any("write" in str(p) or "delete" in str(p) for p in permissions) else "harmless",
        inferred=True,
    )
