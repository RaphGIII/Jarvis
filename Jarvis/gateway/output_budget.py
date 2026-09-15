"""How long an answer may be, decided before the call -- provider-independent.

One small fixed output limit for every request truncates a physiology
explanation at the third mechanism and wastes reservation on a one-word
answer.  The budget is a *level* with a soft token range, chosen from what
ZEUS knows before any model runs:

- the owner's own words ("kurz", "detailliert", "maximal ausführlich");
- the task class and the expected deliverable (a JSON decision, an
  explanation, a plan, code);
- the reasoning depth of the task vector and the chat mode;
- the cost policy: for a metered role the governor may step the level down
  until the reservation fits the remaining budget, never up.

Each provider adapter maps the token figure to its native parameter, clipped
to the provider's real hard maximum (configuration, per provider).  No model
is asked how long the answer should be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: level -> (soft minimum, soft maximum) visible-output tokens.  The budget
#: uses the maximum: what is reserved is what the answer may need.
LEVELS: dict[str, tuple[int, int]] = {
    "brief": (500, 1000),
    "normal": (2000, 4000),
    "detailed": (4000, 8000),
    "deep": (8000, 16000),
    "large": (16000, 32000),
}
ORDER: tuple[str, ...] = ("brief", "normal", "detailed", "deep", "large")

_BRIEF = re.compile(r"\b(kurz|knapp|in einem satz|ein(em)? satz|ein wort|einem wort|stichpunkt|stichworte|"
                    r"kurzfassung|tl;?dr|briefly|short(ly)?|one word|one sentence|in a sentence|concise)\b", re.I)
_DETAILED = re.compile(r"\b(detailliert|ausführlich|ausfuehrlich|umfassend|gründlich|gruendlich|im detail|schritt für schritt|"
                       r"detailed|thorough(ly)?|in depth|in-depth|comprehensive|step by step|erkläre genau|erklär genau)\b", re.I)
_DEEP = re.compile(r"\b(maximal ausführlich|so ausführlich wie möglich|vollständig|erschöpfend|lückenlos|alle details|"
                   r"exhaustive(ly)?|as detailed as possible|complete treatment|leave nothing out|everything you know)\b", re.I)
_LARGE = re.compile(r"\b(vollständiges? (dokument|skript|programm|modul|kapitel)|kompletten? (code|skript|programm|dokument)|"
                    r"ganzes? (kapitel|dokument)|full (document|script|program|module|chapter)|entire (chapter|document)|"
                    r"complete (implementation|codebase|module))\b", re.I)


@dataclass(frozen=True)
class OutputBudget:
    level: str
    tokens: int
    reason: str = ""
    #: The level the request asked for before the cost governor stepped it down.
    requested_level: str = ""
    #: Where the budget came from: owner words, task class, mode, governor.
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def reduced(self) -> bool:
        return bool(self.requested_level) and self.requested_level != self.level

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "tokens": self.tokens, "reason": self.reason, "requested_level": self.requested_level or self.level,
                "reduced": self.reduced, "signals": dict(self.signals)}


def _level_index(level: str) -> int:
    return ORDER.index(level) if level in ORDER else 1


def _bump(level: str, steps: int) -> str:
    return ORDER[max(0, min(len(ORDER) - 1, _level_index(level) + steps))]


def budget_for_level(level: str, *, reason: str = "", requested_level: str = "", signals: dict[str, Any] | None = None,
                     hard_limit: int | None = None) -> OutputBudget:
    tokens = LEVELS.get(level, LEVELS["normal"])[1]
    if hard_limit:
        tokens = min(tokens, int(hard_limit))
    return OutputBudget(level=level, tokens=tokens, reason=reason, requested_level=requested_level, signals=dict(signals or {}))


def decide_output_budget(text: str, *, task: Any = None, mode: Any = None, structured: bool = False,
                         hard_limit: int | None = None) -> OutputBudget:
    """The level an answer to ``text`` needs, from the owner's words, the task and the mode.

    ``task`` is the gateway's TaskVector (class and reasoning depth); ``mode``
    the chat mode; ``structured`` says the caller wants a JSON object (a
    decision, not prose), which is brief by nature unless the owner asked.
    """

    owner = str(text or "")
    signals: dict[str, Any] = {}
    level = "normal"
    reasons: list[str] = []

    task_class = str(getattr(getattr(task, "task_class", None), "value", getattr(task, "task_class", "")) or "")
    if task_class.startswith("engineering"):
        level, signals["task_class"] = "detailed", task_class
        reasons.append("engineering deliverable")
    elif task_class in {"planning", "composition"}:
        level, signals["task_class"] = "normal", task_class
    elif task_class:
        signals["task_class"] = task_class
    if structured:
        level = "brief"
        signals["structured"] = True
        reasons.append("structured output")

    depth = float(getattr(task, "reasoning_depth", 0.0) or 0.0)
    if depth >= 0.65 and not structured:
        level = _bump(level, 1)
        signals["reasoning_depth"] = round(depth, 2)
        reasons.append(f"reasoning depth {depth:.2f}")

    mode_name = str(getattr(mode, "value", mode) or "").upper()
    if mode_name == "DEEP" and not structured:
        level = _bump(level, 1)
        signals["mode"] = mode_name
        reasons.append("DEEP mode")

    # The owner's explicit instruction outranks every inference.
    if _LARGE.search(owner):
        level, signals["owner_words"] = "large", "large deliverable"
        reasons.append("the owner asked for a complete deliverable")
    elif _DEEP.search(owner):
        level, signals["owner_words"] = "deep", "maximal"
        reasons.append("the owner asked for the fullest answer")
    elif _DETAILED.search(owner):
        level, signals["owner_words"] = max(("detailed", level), key=_level_index), "detailed"
        reasons.append("the owner asked for detail")
    elif _BRIEF.search(owner):
        level, signals["owner_words"] = "brief", "brief"
        reasons.append("the owner asked for brevity")

    return budget_for_level(level, reason="; ".join(reasons) or "default for a conversational answer", signals=signals, hard_limit=hard_limit)


def step_down(budget: OutputBudget, *, hard_limit: int | None = None) -> OutputBudget | None:
    """One level less, for a governor that cannot reserve the current one.  None at the floor."""

    index = _level_index(budget.level)
    if index <= 0:
        return None
    lower = ORDER[index - 1]
    return budget_for_level(lower, reason=f"{budget.reason}; reduced by the cost governor", requested_level=budget.requested_level or budget.level,
                            signals={**budget.signals, "reduced_by": "cost governor"}, hard_limit=hard_limit)


TRUNCATING_FINISH_REASONS = frozenset({"max_tokens", "max_output_tokens", "length", "maxtokens"})


def truncated_by_limit(finish_reason: str) -> bool:
    """Whether a provider stopped because it hit the output ceiling, whatever it calls that."""

    return str(finish_reason or "").strip().lower().replace("_", "").replace("-", "") in {r.replace("_", "") for r in TRUNCATING_FINISH_REASONS}
