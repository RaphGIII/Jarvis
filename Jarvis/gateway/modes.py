"""The five owner-facing chat modes, and what each one permits.

The owner never picks a provider.  They pick how much ZEUS may spend on this
conversation and what kind of work it is; the router does the rest.

FREE is not "prefer free".  It is a hard boundary enforced in the transport
layer (:mod:`gateway.transport`): a paid provider call in FREE mode is
rejected before any network request is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ``engineer.codex`` -- the subscription CLI engineer -- costs nothing per
# request and is permitted in every mode, as it always was; the metered
# engineer roles need BUILD.


class ChatMode(str, Enum):
    #: ZEUS chooses the cheapest route predicted to satisfy the task reliably.
    AUTO = "AUTO"
    #: Hard zero-cost mode.  Paid provider calls are impossible.
    FREE = "FREE"
    #: Inexpensive capable cloud reasoning within the configured budget.
    SMART = "SMART"
    #: The strongest configured reasoning model when the task needs it.
    DEEP = "DEEP"
    #: Engineering: creating and repairing capabilities and ZEUS itself.
    BUILD = "BUILD"

    @classmethod
    def parse(cls, value: object, default: "ChatMode | None" = None) -> "ChatMode":
        if isinstance(value, cls):
            return value
        # str(Enum) is "ChatMode.FREE" on modern Pythons, so never str() a member.
        text = str(getattr(value, "value", value) or "").strip().upper()
        try:
            return cls(text)
        except ValueError:
            return default or cls.AUTO


class CostClass(str, Enum):
    """What a route costs, as the policy sees it."""

    #: Local compute, deterministic tools, or a provider's free tier.
    ZERO = "ZERO"
    #: Metered per token.
    METERED = "METERED"


class RoleFamily(str, Enum):
    REASONING = "reasoning"
    ENGINEER = "engineer"
    SPEECH = "speech"
    LOCAL = "local"


@dataclass(frozen=True)
class ModePolicy:
    """Which roles a mode may use, and whether metered routes are allowed."""

    mode: ChatMode
    allow_metered: bool
    #: Role names permitted in this mode (``*`` means any configured role).
    roles: tuple[str, ...]
    #: Per-task ceiling this mode imposes on top of the governor's caps, EUR.
    task_cap_eur: float | None = None
    owner_hint_de: str = ""
    owner_hint_en: str = ""

    def permits_role(self, role: str) -> bool:
        return "*" in self.roles or role in self.roles or any(role.startswith(prefix.rstrip("*")) for prefix in self.roles if prefix.endswith("*"))


MODE_POLICIES: dict[ChatMode, ModePolicy] = {
    ChatMode.FREE: ModePolicy(
        ChatMode.FREE, allow_metered=False,
        roles=("local.*", "reasoning.free", "engineer.codex"),
        task_cap_eur=0.0,
        owner_hint_de="FREE: nur lokale und kostenlose Wege; bezahlte Anbieter sind technisch gesperrt.",
        owner_hint_en="FREE: local and zero-cost routes only; paid providers are technically blocked.",
    ),
    ChatMode.AUTO: ModePolicy(
        ChatMode.AUTO, allow_metered=True,
        roles=("local.*", "reasoning.free", "reasoning.deep", "engineer.codex"),
        owner_hint_de="AUTO: der günstigste Weg, der die Aufgabe voraussichtlich zuverlässig löst.",
        owner_hint_en="AUTO: the cheapest route predicted to solve the task reliably.",
    ),
    ChatMode.SMART: ModePolicy(
        ChatMode.SMART, allow_metered=True,
        roles=("local.*", "reasoning.free", "reasoning.deep", "engineer.codex"),
        task_cap_eur=0.50,
        owner_hint_de="SMART: günstiges Cloud-Denken innerhalb des Budgets.",
        owner_hint_en="SMART: inexpensive cloud reasoning within budget.",
    ),
    ChatMode.DEEP: ModePolicy(
        ChatMode.DEEP, allow_metered=True,
        roles=("local.*", "reasoning.free", "reasoning.deep", "engineer.codex"),
        owner_hint_de="DEEP: das stärkste konfigurierte Denkmodell, wenn die Aufgabe es braucht.",
        owner_hint_en="DEEP: the strongest configured reasoning model when the task needs it.",
    ),
    ChatMode.BUILD: ModePolicy(
        ChatMode.BUILD, allow_metered=True,
        roles=("local.*", "reasoning.free", "reasoning.deep", "engineer.codex", "engineer.standard", "engineer.frontier"),
        owner_hint_de="BUILD: Engineering-Modus zum Erstellen und Reparieren von Fähigkeiten.",
        owner_hint_en="BUILD: engineering mode for creating and repairing capabilities.",
    ),
}


def policy_for(mode: ChatMode | str) -> ModePolicy:
    return MODE_POLICIES[ChatMode.parse(mode)]
