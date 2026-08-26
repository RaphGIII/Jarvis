"""The owner's domain: what ZEUS is, how it behaves, what it may spend.

Five documents, structurally separate from everything ZEUS learns or builds:

    identity      the names it answers to
    personality   how it speaks and reasons
    policy        what it may do on its own
    spending      which channels cost money, and that none are on
    security      what content may and may not command

Ordinary self-development, capability building, experts, documents, web pages
and research may read them and may not write them.  That is enforced in four
places -- the edit engine, the promotion copy, the git-level protected-path
list, and read-only file attributes -- so a system prompt is the least of it.

The owner changes them through :class:`owner.core.OwnerTransaction`: propose,
see the diff, approve from the live interface, and every change is snapshotted,
audited and reversible.
"""

from __future__ import annotations

from owner.protected import PROTECTED_PATHS, is_protected, protected_violations

__all__ = ["PROTECTED_PATHS", "is_protected", "protected_violations"]
