"""Promoting a verified candidate into the live installation -- and undoing it.

This is the module that stands between a local language model and a working
Jarvis.  A 7B model will, sooner or later, produce a change that passes its
tests and still breaks the installation in some way the tests did not cover.
The design assumption is therefore not "the candidate is probably fine" but
"the candidate will eventually be wrong, and recovery must be automatic".

The sequence is:

    record known-good  ->  require a clean tree  ->  snapshot
      ->  apply the candidate  ->  verify  ->  restart  ->  health check
      ->  keep, or roll back to the known-good revision

Every step is deterministic, every step is recorded in an append-only audit
log, and every failure path leads to :meth:`Promoter.rollback` rather than to a
half-promoted installation.

Two rules are absolute:

*A dirty tree is never promoted onto.*  Uncommitted work would be destroyed by
a rollback, so promotion refuses to start rather than risk it.

*The health check is a real command.*  "Health" means a process ran and exited
zero after the change was applied -- not that the diff looked reasonable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence


class PromotionStage(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    SNAPSHOT = "SNAPSHOT"
    APPLY = "APPLY"
    VERIFY = "VERIFY"
    RESTART = "RESTART"
    HEALTH_CHECK = "HEALTH_CHECK"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"
    DONE = "DONE"


class PromotionOutcome(str, Enum):
    PROMOTED = "PROMOTED"
    #: Refused before anything was changed.
    REJECTED = "REJECTED"
    #: Applied, found wanting, and successfully undone.
    ROLLED_BACK = "ROLLED_BACK"
    #: Applied, found wanting, and the undo ALSO failed.  Needs a human.
    ROLLBACK_FAILED = "ROLLBACK_FAILED"


@dataclass
class PromotionRecord:
    """The full account of one promotion attempt."""

    outcome: PromotionOutcome
    candidate: str = ""
    target: str = ""
    known_good_revision: str = ""
    promoted_revision: str = ""
    snapshot_path: str = ""
    changed_files: list[str] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    promotion_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def success(self) -> bool:
        return self.outcome is PromotionOutcome.PROMOTED

    @property
    def needs_human(self) -> bool:
        return self.outcome is PromotionOutcome.ROLLBACK_FAILED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outcome"] = self.outcome.value
        data["success"] = self.success
        return data


class PromotionRefused(RuntimeError):
    """A precondition failed.  Nothing was changed."""


@dataclass
class HealthCheck:
    """A command that proves the installation still works after a change."""

    command: list[str]
    timeout_seconds: float = 300.0
    #: Text that must appear in the output for the check to count as passing.
    #: Guards against a command that exits zero without doing anything.
    expect_output: str = ""

    def run(self, cwd: Path, *, env: dict[str, str] | None = None) -> tuple[bool, str]:
        try:
            completed = subprocess.run(
                self.command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                shell=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, f"health check timed out after {self.timeout_seconds:.0f}s: {' '.join(self.command)}"
        except FileNotFoundError:
            return False, f"health check executable not found: {self.command[0]}"

        output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            return False, f"exit={completed.returncode}\n{output[-2000:]}"
        if self.expect_output and self.expect_output not in output:
            return False, f"expected {self.expect_output!r} in the output but it was absent\n{output[-2000:]}"
        return True, output[-2000:]


class PromotionAudit:
    """Append-only record of every promotion and rollback."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: PromotionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")

    def history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def last_known_good(self) -> str:
        """The newest revision a promotion actually succeeded from or to."""

        for record in reversed(self.history(limit=200)):
            if record.get("outcome") == PromotionOutcome.PROMOTED.value:
                return str(record.get("promoted_revision") or record.get("known_good_revision") or "")
        return ""


class Snapshot:
    """A restorable copy of the tracked files, taken before a promotion.

    Git already lets us return to a commit, but a snapshot covers what git will
    not: files that are tracked-but-modified at the moment of promotion, and the
    case where the git operation itself is what went wrong.  Belt and braces is
    the right posture for the one mechanism protecting the installation.
    """

    def __init__(self, root: Path, files: Sequence[str]) -> None:
        self.root = Path(root)
        self.files = list(files)

    @classmethod
    def create(cls, repository: Path, destination: Path, files: Sequence[str]) -> "Snapshot":
        destination.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for relative in files:
            source = repository / relative
            if not source.exists() or not source.is_file():
                # A file that does not exist yet still needs recording, so a
                # rollback can delete it again.
                (destination / (relative.replace("/", "__") + ".absent")).write_text("", encoding="utf-8")
                saved.append(relative)
                continue
            target = destination / relative.replace("/", "__")
            target.write_bytes(source.read_bytes())
            saved.append(relative)
        (destination / "MANIFEST.json").write_text(json.dumps({"files": saved}, indent=2), encoding="utf-8")
        return cls(destination, saved)

    @classmethod
    def load(cls, destination: Path) -> "Snapshot":
        manifest = destination / "MANIFEST.json"
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"] if manifest.exists() else []
        return cls(destination, files)

    def restore(self, repository: Path) -> list[str]:
        restored: list[str] = []
        for relative in self.files:
            flattened = relative.replace("/", "__")
            absent = self.root / (flattened + ".absent")
            target = repository / relative
            if absent.exists():
                if target.exists():
                    target.unlink()
                    restored.append(relative)
                continue
            source = self.root / flattened
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                restored.append(relative)
        return restored


class Promoter:
    """Moves a verified candidate into the live installation, reversibly."""

    def __init__(
        self,
        repository: str | Path,
        *,
        audit: PromotionAudit | None = None,
        snapshot_root: str | Path | None = None,
        restart: Callable[[], tuple[bool, str]] | None = None,
        git_timeout: float = 120.0,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.audit = audit or PromotionAudit(self.repository / "data" / "promotions.jsonl")
        self.snapshot_root = Path(snapshot_root or (self.repository / "data" / "snapshots"))
        #: Restarting is deployment-specific.  The default is a no-op that
        #: reports success, because in a single-process install the health check
        #: below already runs in a fresh interpreter.
        self._restart = restart or (lambda: (True, "no restart hook configured"))
        self.git_timeout = git_timeout

    # -- git helpers -----------------------------------------------------

    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        return subprocess.run(
            ["git", *args],
            cwd=str(self.repository),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.git_timeout,
            check=check,
            env=env,
        )

    def current_revision(self) -> str:
        completed = self._git("rev-parse", "HEAD")
        return completed.stdout.strip() if completed.returncode == 0 else ""

    def is_clean(self) -> tuple[bool, str]:
        completed = self._git("status", "--porcelain")
        if completed.returncode != 0:
            return False, f"git status failed: {completed.stderr.strip()}"
        # Runtime state under data/ is rewritten by the running product (voice
        # registry, evidence files) and is never what a promotion moves or a
        # rollback resets, so it does not count as uncommitted work.
        dirty = [
            line for line in completed.stdout.splitlines()
            if line.strip() and not line[3:].strip().strip('"').replace("\\", "/").lstrip("Jarvis/").startswith("data/")
        ]
        return (not dirty), "\n".join(dirty[:20])

    # -- the pipeline ----------------------------------------------------

    def promote(
        self,
        candidate: str | Path,
        *,
        changed_files: Sequence[str],
        health_check: HealthCheck,
        commit_message: str = "",
        verify: Callable[[Path], tuple[bool, str]] | None = None,
        allow_dirty: bool = False,
    ) -> PromotionRecord:
        """Apply a candidate worktree's changes to the live repository.

        ``changed_files`` is the caller's declaration of what may move; anything
        outside it is not copied, so a candidate cannot smuggle in a file the
        reviewer never saw.
        """

        candidate = Path(candidate).resolve()
        record = PromotionRecord(outcome=PromotionOutcome.REJECTED, candidate=str(candidate), target=str(self.repository))

        def stage(name: PromotionStage, ok: bool, detail: str = "") -> None:
            record.stages.append(
                {"stage": name.value, "ok": ok, "detail": detail[:2000], "at": datetime.now(timezone.utc).isoformat()}
            )

        def refuse(reason: str) -> PromotionRecord:
            record.reason = reason
            record.finished_at = datetime.now(timezone.utc).isoformat()
            self.audit.record(record)
            return record

        # ---- preflight ------------------------------------------------
        if not candidate.exists():
            stage(PromotionStage.PREFLIGHT, False, "candidate does not exist")
            return refuse(f"candidate does not exist: {candidate}")

        files = [str(item).replace("\\", "/") for item in changed_files if str(item).strip()]
        if not files:
            stage(PromotionStage.PREFLIGHT, False, "no changed files declared")
            return refuse("nothing to promote: the candidate declared no changed files")

        known_good = self.current_revision()
        if not known_good:
            stage(PromotionStage.PREFLIGHT, False, "target is not a git repository")
            return refuse("cannot promote into a directory that is not a git repository")
        record.known_good_revision = known_good

        clean, dirty = self.is_clean()
        if not clean and not allow_dirty:
            stage(PromotionStage.PREFLIGHT, False, f"working tree is dirty:\n{dirty}")
            return refuse(
                "refusing to promote onto a dirty working tree -- a rollback would destroy the uncommitted work. "
                "Commit or stash it first."
            )
        stage(PromotionStage.PREFLIGHT, True, f"known good {known_good[:12]}, {len(files)} file(s)")

        # ---- snapshot -------------------------------------------------
        snapshot_dir = self.snapshot_root / f"{record.promotion_id}_{known_good[:8]}"
        try:
            snapshot = Snapshot.create(self.repository, snapshot_dir, files)
        except OSError as exc:
            stage(PromotionStage.SNAPSHOT, False, str(exc))
            return refuse(f"could not create a recovery snapshot: {exc}")
        record.snapshot_path = str(snapshot_dir)
        stage(PromotionStage.SNAPSHOT, True, f"{len(snapshot.files)} file(s) saved to {snapshot_dir}")

        # ---- apply ----------------------------------------------------
        try:
            applied = self._copy_files(candidate, files)
        except (OSError, ValueError) as exc:
            stage(PromotionStage.APPLY, False, str(exc))
            snapshot.restore(self.repository)
            return refuse(f"could not apply the candidate: {exc}")
        record.changed_files = applied
        stage(PromotionStage.APPLY, True, f"copied {len(applied)} file(s)")

        # From here on every failure must undo the change.
        def rollback(reason: str, outcome_on_success: PromotionOutcome = PromotionOutcome.ROLLED_BACK) -> PromotionRecord:
            ok, detail = self._rollback(snapshot, known_good)
            stage(PromotionStage.ROLLBACK, ok, detail)
            record.outcome = outcome_on_success if ok else PromotionOutcome.ROLLBACK_FAILED
            record.reason = reason if ok else f"{reason}; AND THE ROLLBACK FAILED: {detail}"
            record.finished_at = datetime.now(timezone.utc).isoformat()
            self.audit.record(record)
            return record

        # ---- verify ---------------------------------------------------
        if verify is not None:
            ok, detail = verify(self.repository)
            stage(PromotionStage.VERIFY, ok, detail)
            if not ok:
                return rollback(f"post-apply verification failed: {detail[:400]}")

        # ---- restart --------------------------------------------------
        ok, detail = self._restart()
        stage(PromotionStage.RESTART, ok, detail)
        if not ok:
            return rollback(f"restart failed: {detail[:400]}")

        # ---- health check ---------------------------------------------
        ok, detail = health_check.run(self.repository)
        stage(PromotionStage.HEALTH_CHECK, ok, detail)
        if not ok:
            return rollback(f"health check failed after promotion: {detail[:400]}")

        # ---- commit ---------------------------------------------------
        if commit_message:
            committed, detail = self._commit(applied, commit_message)
            stage(PromotionStage.COMMIT, committed, detail)
            if not committed:
                return rollback(f"could not commit the promoted change: {detail[:400]}")
            record.promoted_revision = self.current_revision()

        record.outcome = PromotionOutcome.PROMOTED
        record.reason = "candidate verified, promoted and healthy"
        record.finished_at = datetime.now(timezone.utc).isoformat()
        stage(PromotionStage.DONE, True, record.promoted_revision or "(uncommitted)")
        self.audit.record(record)
        return record

    # -- pieces ----------------------------------------------------------

    def _copy_files(self, candidate: Path, files: list[str]) -> list[str]:
        """Copy exactly the declared files, refusing anything that escapes."""

        from owner.protected import protected_violations

        # The owner's domain and the recovery machinery never move through a
        # promotion. This is the last gate before the live tree, and it does
        # not trust that an earlier one was consulted.
        violations = protected_violations(files)
        if violations:
            raise ValueError(f"owner-protected paths may not be promoted: {violations[:5]}")

        applied: list[str] = []
        for relative in files:
            raw = Path(relative)
            if raw.is_absolute() or ".." in raw.parts:
                raise ValueError(f"unsafe path in candidate: {relative}")
            source = (candidate / raw).resolve(strict=False)
            target = (self.repository / raw).resolve(strict=False)
            if not source.is_relative_to(candidate):
                raise ValueError(f"candidate path escapes the worktree: {relative}")
            if not target.is_relative_to(self.repository):
                raise ValueError(f"target path escapes the repository: {relative}")

            if not source.exists():
                # The candidate deleted it; mirror that.
                if target.exists():
                    target.unlink()
                    applied.append(relative)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            applied.append(relative)
        return applied

    def _commit(self, files: list[str], message: str) -> tuple[bool, str]:
        add = self._git("add", "--", *files)
        if add.returncode != 0:
            return False, add.stderr.strip()
        commit = self._git("-c", "commit.gpgsign=false", "commit", "-m", message, "--", *files)
        if commit.returncode != 0:
            output = f"{commit.stdout}\n{commit.stderr}"
            if "nothing to commit" in output.lower():
                return True, "nothing to commit; the candidate matched the live tree"
            return False, output.strip()[:1000]
        return True, commit.stdout.strip()[:500]

    def _rollback(self, snapshot: Snapshot, known_good: str) -> tuple[bool, str]:
        """Restore the files, then make sure git agrees we are back."""

        details: list[str] = []
        try:
            restored = snapshot.restore(self.repository)
            details.append(f"restored {len(restored)} file(s) from the snapshot")
        except OSError as exc:
            return False, f"snapshot restore failed: {exc}"

        current = self.current_revision()
        if current and known_good and current != known_good:
            reset = self._git("reset", "--hard", known_good)
            if reset.returncode != 0:
                return False, "; ".join(details) + f"; git reset to {known_good[:12]} failed: {reset.stderr.strip()}"
            details.append(f"reset to known-good {known_good[:12]}")

        # The rollback is only real if the tree now matches the known-good tree.
        diff = self._git("diff", "--name-only", known_good, "--")
        if diff.returncode == 0 and diff.stdout.strip():
            remaining = [line for line in diff.stdout.splitlines() if line.strip()]
            return False, "; ".join(details) + f"; files still differ from known-good: {remaining[:10]}"
        details.append("working tree matches the known-good revision")
        return True, "; ".join(details)

    def rollback_to(self, known_good_revision: str, *, snapshot_path: str | Path | None = None) -> PromotionRecord:
        """Manually return the installation to a known-good state."""

        record = PromotionRecord(
            outcome=PromotionOutcome.REJECTED,
            target=str(self.repository),
            known_good_revision=known_good_revision,
            snapshot_path=str(snapshot_path or ""),
        )
        snapshot = Snapshot.load(Path(snapshot_path)) if snapshot_path else Snapshot(self.repository, [])
        ok, detail = self._rollback(snapshot, known_good_revision)
        record.stages.append({"stage": PromotionStage.ROLLBACK.value, "ok": ok, "detail": detail[:2000]})
        record.outcome = PromotionOutcome.ROLLED_BACK if ok else PromotionOutcome.ROLLBACK_FAILED
        record.reason = detail
        record.finished_at = datetime.now(timezone.utc).isoformat()
        self.audit.record(record)
        return record

    def prune_snapshots(self, *, keep: int = 10) -> int:
        """Keep the most recent snapshots; delete the rest."""

        if not self.snapshot_root.exists():
            return 0
        directories = sorted(
            (item for item in self.snapshot_root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for directory in directories[keep:]:
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
        return removed


def default_health_check(python_executable: str | None = None) -> HealthCheck:
    """Prove Jarvis still imports and its kernel still assembles.

    Deliberately more than ``import jarvis``: a broken model catalog or a
    circular import in the project engine would pass a bare import check and
    still leave the installation unable to do anything.
    """

    import sys

    return HealthCheck(
        command=[
            python_executable or sys.executable,
            "-c",
            "from core.kernel import JarvisKernel; "
            "k = JarvisKernel(); "
            "assert k.tools.names(); "
            "assert k.catalog.tiers(); "
            "print('JARVIS_HEALTH_OK')",
        ],
        timeout_seconds=300.0,
        expect_output="JARVIS_HEALTH_OK",
    )
