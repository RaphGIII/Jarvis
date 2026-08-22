from __future__ import annotations

from dataclasses import dataclass, field

from capabilities.models import SkillSpecification


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    blocked_permissions: list[str] = field(default_factory=list)
    reason: str = ""


class PermissionPolicy:
    """Conservative v0.4 permission gate for acquired capabilities."""

    default_auto_allowed = frozenset({"filesystem.read", "filesystem.write"})
    blocked_prefixes = ("credentials.", "email.", "browser", "network.")

    def __init__(self, auto_allowed: set[str] | None = None) -> None:
        self.auto_allowed = set(auto_allowed or self.default_auto_allowed)

    def evaluate(self, spec: SkillSpecification) -> PermissionDecision:
        blocked = []
        for permission in spec.permissions:
            if permission in self.auto_allowed:
                continue
            if permission.startswith(self.blocked_prefixes) or permission not in self.auto_allowed:
                blocked.append(permission)
        if blocked:
            return PermissionDecision(
                allowed=False,
                blocked_permissions=blocked,
                reason="Capability requires permissions that Jarvis cannot silently grant.",
            )
        return PermissionDecision(True, [], "Safe local capability permissions.")
