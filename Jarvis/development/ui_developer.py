"""Letting Jarvis change its own face, safely.

"Change the eye animation" has to be an ordinary request, and the UI was built
with no build step precisely so that it could be.  What makes it safe is not
that the change is small -- it is that the whole path exists:

    request -> isolated worktree -> candidate -> health check -> promote
                                                      |
                                                      +-- fails -> rolled back

Two things make this different from ordinary self-development, and both are why
this module exists rather than just calling the repository engineer directly.

*A broken UI fails silently.*  A syntax error in ``app.js`` does not stop the
server, does not fail a test, and appears only as a blank page.  So the
acceptance criterion is :mod:`jarvis.verify_ui`, which starts a real server,
fetches every asset, and checks that the page still has the parts the client
looks up by name.

*The blast radius is the interface only.*  ``allowed_paths`` is the ui directory
and nothing else.  A UI request that starts editing the project engine has
misunderstood the task, and the edit engine refuses it rather than the reviewer
having to notice.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The only paths a UI change may touch.
UI_PATHS = ("ui/index.html", "ui/app.js", "ui/eye.js", "ui/graph.js")

#: What the candidate must satisfy before anything is copied into the live tree.
UI_HEALTH_COMMAND = ("-m", "jarvis.verify_ui")


@dataclass
class UIChangeResult:
    """What happened, in enough detail to explain a refusal."""

    ok: bool = False
    status: str = ""
    detail: str = ""
    changed_files: list[str] = field(default_factory=list)
    health: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    rolled_back: bool = False
    worktree: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "detail": self.detail[:2000],
            "changed_files": self.changed_files,
            "health": self.health,
            "promotion": self.promotion,
            "rolled_back": self.rolled_back,
            "worktree": self.worktree,
        }


class UIDeveloper:
    """Runs a UI change request through the full candidate/promote/rollback path."""

    def __init__(
        self,
        repository: str | Path,
        *,
        engineer: Any = None,
        promoter: Any = None,
        python: str = "",
    ) -> None:
        import sys

        self.repository = Path(repository).resolve()
        self._engineer = engineer
        self._promoter = promoter
        self.python = python or sys.executable

    # -- the health check ------------------------------------------------

    def health_check(self) -> Any:
        """The command that decides whether a candidate may be promoted."""

        from deployment.promotion import HealthCheck

        return HealthCheck(
            command=[self.python, *UI_HEALTH_COMMAND],
            timeout_seconds=180.0,
            # Exit zero is not enough: a verify_ui that failed to import would
            # also exit zero from a shell's point of view if something swallowed
            # it. The marker proves the checks actually ran.
            expect_output="UI_OK",
        )

    def verify_candidate(self, worktree: str | Path) -> tuple[bool, str]:
        """Run the UI checks against a candidate worktree, before promotion."""

        from jarvis.verify_ui import verify

        report = verify(Path(worktree) / "ui", serve=True)
        return report.ok, report.describe()

    # -- the whole path --------------------------------------------------

    def change(
        self,
        request: str,
        *,
        max_cycles: int = 3,
        max_seconds: float = 900.0,
        promote: bool = True,
    ) -> UIChangeResult:
        """Develop a UI change in isolation, verify it, and promote or discard it.

        ``promote=False`` stops after verification, which is what a preview
        wants: the candidate stays in its worktree and the live interface is
        untouched.
        """

        from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal

        engineer = self._engineer or RepositoryEngineer(max_seconds=max_seconds)
        goal = SelfImprovementGoal(
            objective=(
                f"{request}\n\n"
                "This is a change to the Jarvis web interface. Edit only the files under ui/. "
                "The page has no build step: index.html loads eye.js and app.js directly, so "
                "the files must remain valid on their own. Do not remove any element id the "
                "client looks up, and leave the __JARVIS_TOKEN__ placeholder in place."
            ),
            allowed_paths=list(UI_PATHS),
        )

        result = UIChangeResult()
        try:
            # The UI health check IS the acceptance criterion. Passing it to the
            # engineer rather than only checking afterwards means the loop can
            # see itself failing it and repair, instead of producing a broken
            # candidate and learning about it at the very end.
            candidate = engineer.improve(
                self.repository,
                goal,
                acceptance_commands=[[self.python, *UI_HEALTH_COMMAND]],
                max_cycles=max_cycles,
            )
        except Exception as exc:
            result.status = "development_failed"
            result.detail = f"{type(exc).__name__}: {exc}"
            return result

        result.worktree = str(getattr(candidate, "worktree", "") or "")
        result.changed_files = list(getattr(candidate, "changed_files", []) or [])
        result.status = str(getattr(candidate, "status", "") or "")

        ready = getattr(candidate, "status", "") == "SELF_DEVELOPMENT_CANDIDATE_READY"
        if not ready:
            result.detail = str(getattr(candidate, "error", "") or "the candidate was not accepted")
            return result

        if not result.worktree:
            result.status = "no_candidate"
            result.detail = "development produced no worktree to verify"
            return result

        healthy, detail = self.verify_candidate(result.worktree)
        result.health = {"ok": healthy, "detail": detail}
        if not healthy:
            # Refused before anything is copied. The live interface never saw it.
            result.status = "unhealthy_candidate"
            result.detail = detail
            return result

        if not promote:
            result.ok = True
            result.status = "verified_not_promoted"
            result.detail = "the candidate is healthy and waiting in its worktree"
            return result

        record = self._promote(result.worktree, result.changed_files, request)
        result.promotion = record.to_dict() if hasattr(record, "to_dict") else {}
        result.rolled_back = str(result.promotion.get("outcome", "")) == "rolled_back"
        result.ok = bool(getattr(record, "success", False))
        result.status = str(result.promotion.get("outcome", "")) or ("promoted" if result.ok else "refused")
        if not result.ok:
            result.detail = str(result.promotion.get("error", "")) or "promotion refused"
        return result

    def _promote(self, worktree: str, changed_files: list[str], request: str) -> Any:
        from deployment.promotion import Promoter

        promoter = self._promoter or Promoter(repository=self.repository)
        # Only ui/ files move, whatever the candidate touched. A UI change that
        # edited something else does not get to smuggle it through promotion.
        movable = [name for name in changed_files if name.replace("\\", "/").startswith("ui/")]
        return promoter.promote(
            worktree,
            changed_files=movable or list(UI_PATHS),
            health_check=self.health_check(),
            commit_message=f"UI: {request[:72]}",
        )
