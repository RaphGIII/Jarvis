"""Hard isolation for self-development: experimental code never touches the live tree.

The invariant, stated once:

    EXPERIMENTAL SELFDEV CODE MUST NEVER WRITE DIRECTLY TO THE ACTIVE
    PRODUCTION WORKING TREE.

Before this module, isolation was real but one-directional.  The candidate was
built in a worktree, and that part held.  What leaked around it:

* the candidate's health check ran with the parent's full environment, so
  ``JARVIS_STATE_ROOT`` pointing at the live tree would have made the
  *candidate* write state into *production*;
* the expert (a CLI with a shell) was started in the worktree and asked
  nicely to stay there;
* nothing ever removed a worktree, and the main repository's ``.git`` grew a
  registry of 141 dead candidates;
* a promotion copied files one by one with no journal, so a process death in
  the middle left a tree that was neither the old revision nor the new one.

This module makes the boundary a thing rather than a convention:

:class:`CandidateWorkspace`
    Creates the worktree from the repository's git root (outside it, always),
    and is the only object that knows the candidate's paths.  Every
    subprocess a mission runs in the candidate gets :meth:`env` -- an
    allow-listed environment with ``PYTHONPATH`` pinned to the candidate and
    every ``JARVIS_*_ROOT`` variable pointed *inside* it -- so the code under
    test cannot find production even if it goes looking.  ``release`` keeps
    the candidate's diff as evidence and then removes the worktree,
    whichever way the mission ended.

:class:`LiveTreeGuard`
    A fingerprint of the live tree taken before anything experimental runs,
    and re-checked after every phase that ran foreign code.  A file that
    changed in the live tree *and is byte-identical to the candidate's copy*
    is contamination by construction -- nothing else produces that -- and is
    restored from git.  The owner's own uncommitted edits never match a
    candidate's bytes, so they are never touched.

The guard is the second line, and it says so: prevention is the environment,
the working directory, the tool policy on the expert and the promoter's
containment checks.  The guard exists because a boundary that is not checked
is a hope.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(cwd: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
    )


def git_root(path: Path) -> Path | None:
    completed = _git(path, "rev-parse", "--show-toplevel")
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve()


def _make_writable_and_retry(func, path, _exc_info):  # pragma: no cover - platform dependent
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


#: Runtime state the product rewrites while it runs.  Never contamination,
#: never restored, never counted as dirt.
RUNTIME_PREFIXES = ("data/", "Jarvis/data/", ".pytest_cache/", ".pytest_tmp/", ".agent_tmp/")


def _is_runtime(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return rel.startswith(RUNTIME_PREFIXES) or "__pycache__" in rel or rel.endswith(".pyc")


#: Environment variables a candidate may see.  Everything credential-shaped
#: and every ZEUS root pointer stays out; the roots are re-added pointing into
#: the candidate.
ALLOWED_ENV = frozenset({
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "LANG", "LC_ALL",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "USERNAME", "COMPUTERNAME",
    "JARVIS_BRAIN_PROVIDER", "JARVIS_BRAIN_BASE_URL", "JARVIS_BRAIN_MODEL", "JARVIS_BRAIN_TIMEOUT",
    "JARVIS_BRAIN_TEMPERATURE", "JARVIS_BRAIN_TOP_P", "JARVIS_BRAIN_MAX_TOKENS", "JARVIS_BRAIN_RETRIES",
    "OLLAMA_HOST", "OLLAMA_MODELS", "CUDA_VISIBLE_DEVICES",
})


@dataclass
class CandidateWorkspace:
    """A git worktree of the repository, outside it, for one mission."""

    #: The directory ZEUS runs from (``.../repo/Jarvis``).  May be a
    #: subdirectory of the git root; the candidate mirrors that layout.
    repository: Path
    mission_id: str
    base: Path | None = None
    path: Path = field(default_factory=Path)  # the worktree root
    root: Path = field(default_factory=Path)  # the candidate's ZEUS directory
    created_at: str = ""
    baseline: str = ""

    def __post_init__(self) -> None:
        self.repository = Path(self.repository).resolve()
        self.base = Path(self.base or Path(os.environ.get("TEMP", "/tmp")) / "jarvis_selfdev").resolve()

    # -- registry --------------------------------------------------------

    @staticmethod
    def registry_path(repository: Path) -> Path:
        return Path(repository) / "data" / "jarvis" / "selfdev" / "worktrees.json"

    @classmethod
    def _registry(cls, repository: Path) -> dict[str, Any]:
        path = cls.registry_path(repository)
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            return {}

    @classmethod
    def _save_registry(cls, repository: Path, rows: dict[str, Any]) -> None:
        path = cls.registry_path(repository)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        tmp.replace(path)

    # -- lifecycle -------------------------------------------------------

    def create(self) -> "CandidateWorkspace":
        top = git_root(self.repository)
        if top is None:
            raise RuntimeError(f"{self.repository} is not inside a git repository; a candidate needs one")
        self.path = (self.base / f"candidate_{self.mission_id}").resolve()
        self._assert_external(top, self.path)
        if self.path.exists():
            self._remove_worktree(top, self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        completed = _git(top, "worktree", "add", "--detach", str(self.path), "HEAD")
        if completed.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {completed.stderr.strip()[:400]}")
        relative = self.repository.relative_to(top)
        self.root = (self.path / relative).resolve()
        self.baseline = _git(top, "rev-parse", "HEAD").stdout.strip()
        self.created_at = _now()
        rows = self._registry(self.repository)
        rows[self.mission_id] = {"path": str(self.path), "root": str(self.root), "created_at": self.created_at,
                                 "baseline": self.baseline}
        self._save_registry(self.repository, rows)
        return self

    @classmethod
    def attach(cls, repository: Path, mission_id: str, worktree_root: str | Path) -> "CandidateWorkspace":
        """An existing candidate (a resumed mission), without creating anything."""

        ws = cls(repository=Path(repository), mission_id=mission_id)
        ws.root = Path(worktree_root).resolve()
        top = git_root(ws.root)
        ws.path = top if top is not None else ws.root
        return ws

    @staticmethod
    def _assert_external(source: Path, destination: Path) -> None:
        source_root = source.resolve()
        destination_root = destination.resolve(strict=False)
        if destination_root == source_root or destination_root.is_relative_to(source_root):
            raise ValueError("a candidate must live outside the repository")
        if source_root.is_relative_to(destination_root):
            raise ValueError("a candidate must not contain the repository")

    def exists(self) -> bool:
        return bool(self.root) and self.root.is_dir()

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """The only environment a candidate's subprocesses ever see.

        Allow-listed, ``PYTHONPATH`` pinned to the candidate so its modules
        beat anything installed on the host, and every ZEUS root variable
        pointed inside the candidate so the code under test cannot address
        the production state root even by accident.
        """

        env = {k: v for k, v in os.environ.items() if k.upper() in ALLOWED_ENV}
        env["PYTHONPATH"] = str(self.root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["JARVIS_STATE_ROOT"] = str(self.root / "data" / "jarvis")
        env["JARVIS_CONFIG_ROOT"] = str(self.root / "config")
        env["ZEUS_CANDIDATE"] = "1"
        env["ZEUS_CANDIDATE_ROOT"] = str(self.root)
        # A candidate must never think it is the supervised product.
        for name in ("ZEUS_SUPERVISED", "ZEUS_SUPERVISOR_DIR", "ZEUS_SUPERVISOR_PID"):
            env.pop(name, None)
        if extra:
            env.update(extra)
        return env

    def diff(self) -> str:
        """The candidate's whole change against its baseline, untracked included."""

        if not self.exists():
            return ""
        parts = []
        tracked = _git(self.path, "diff", "HEAD", "--", ".", ":(exclude)*.pyc")
        if tracked.returncode == 0 and tracked.stdout.strip():
            parts.append(tracked.stdout)
        status = _git(self.path, "status", "--porcelain", "--untracked-files=all")
        for line in status.stdout.splitlines():
            if line.startswith("??"):
                rel = line[3:].strip().strip('"')
                if _is_runtime(rel):
                    continue
                target = self.path / rel
                try:
                    if target.is_file() and target.stat().st_size < 400_000:
                        body = target.read_text(encoding="utf-8", errors="replace")
                        parts.append(f"--- /dev/null\n+++ b/{rel}\n" + "".join(f"+{l}\n" for l in body.splitlines()))
                except OSError:
                    continue
        return "".join(parts)

    def changed_files(self) -> list[str]:
        """Paths relative to the candidate's ZEUS directory, files only."""

        if not self.exists():
            return []
        status = _git(self.path, "status", "--porcelain", "--untracked-files=all")
        if status.returncode != 0:
            return []
        prefix = self.root.relative_to(self.path).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        out = []
        for line in status.stdout.splitlines():
            if len(line) <= 3:
                continue
            name = line[3:].strip().strip('"')
            if " -> " in name:
                name = name.split(" -> ", 1)[1]
            name = name.replace("\\", "/")
            if prefix and not name.startswith(prefix):
                continue
            rel = name[len(prefix):] if prefix else name
            if _is_runtime(rel) or rel.endswith("/") or (self.root / rel).is_dir():
                continue
            out.append(rel)
        return out

    def keep_evidence(self, evidence_root: Path, *, reason: str = "") -> Path | None:
        """The candidate's diff, kept where the mission record can point to it."""

        patch = self.diff()
        if not patch.strip():
            return None
        evidence_root.mkdir(parents=True, exist_ok=True)
        target = evidence_root / f"{self.mission_id}.patch"
        header = f"# ZEUS self-development candidate {self.mission_id}\n# baseline {self.baseline}\n# kept {_now()}: {reason}\n"
        target.write_text(header + patch, encoding="utf-8")
        return target

    def release(self, *, evidence_root: Path | None = None, reason: str = "") -> dict[str, Any]:
        """Remove the worktree, keeping the diff as evidence first."""

        report: dict[str, Any] = {"mission_id": self.mission_id, "removed": False, "evidence": ""}
        if evidence_root is not None and self.exists():
            try:
                kept = self.keep_evidence(evidence_root, reason=reason)
                report["evidence"] = str(kept) if kept else ""
            except OSError as exc:  # pragma: no cover - disk trouble
                report["evidence_error"] = str(exc)
        top = git_root(self.repository)
        if top is not None and self.path and self.path.exists():
            report["removed"] = self._remove_worktree(top, self.path)
        rows = self._registry(self.repository)
        if rows.pop(self.mission_id, None) is not None:
            self._save_registry(self.repository, rows)
        return report

    @staticmethod
    def _remove_worktree(top: Path, path: Path) -> bool:
        completed = _git(top, "worktree", "remove", "--force", str(path))
        if path.exists():
            shutil.rmtree(path, onerror=_make_writable_and_retry)
        _git(top, "worktree", "prune")
        return not path.exists() or completed.returncode == 0

    @classmethod
    def reap(cls, repository: Path, *, keep: Iterable[str] = (), base: Path | None = None) -> list[str]:
        """Remove every candidate that does not belong to a mission in ``keep``.

        Runs at startup: whatever a killed process left behind is collected
        here, from the registry and from git's own worktree list.
        """

        repository = Path(repository).resolve()
        keep_ids = set(keep)
        top = git_root(repository)
        removed: list[str] = []
        base = Path(base or Path(os.environ.get("TEMP", "/tmp")) / "jarvis_selfdev").resolve()
        rows = cls._registry(repository)
        for mission_id, row in list(rows.items()):
            if mission_id in keep_ids:
                continue
            path = Path(row.get("path", ""))
            if top is not None and path and str(path).startswith(str(base)):
                cls._remove_worktree(top, path)
            rows.pop(mission_id, None)
            removed.append(mission_id)
        cls._save_registry(repository, rows)
        # Directories under the base that belong to no registered mission and
        # are older than a working day: the copytree candidates of earlier
        # sprints (148 of them, 9.9 GB, at the time this was written).
        if base.is_dir():
            cutoff = time.time() - 6 * 3600
            registered = {Path(row.get("path", "")).resolve() for row in rows.values()}
            for path in base.iterdir():
                try:
                    if not path.is_dir() or path.resolve() in registered or path.stat().st_mtime > cutoff:
                        continue
                    if any(path.name.endswith(mid) or path.name == f"candidate_{mid}" for mid in keep_ids):
                        continue
                    shutil.rmtree(path, onerror=_make_writable_and_retry)
                    removed.append(path.name)
                except OSError:
                    continue
        if top is not None:
            listing = _git(top, "worktree", "list", "--porcelain")
            for line in listing.stdout.splitlines():
                if not line.startswith("worktree "):
                    continue
                path = Path(line[len("worktree "):].strip())
                try:
                    inside_base = path.resolve().is_relative_to(base)
                except OSError:
                    inside_base = False
                if not inside_base:
                    continue
                mission_id = path.name.replace("candidate_", "", 1)
                if mission_id in keep_ids:
                    continue
                cls._remove_worktree(top, path)
                if mission_id not in removed:
                    removed.append(mission_id)
        return removed


# --------------------------------------------------------------------------
# The live tree
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TreeFingerprint:
    head: str
    #: relative path -> sha256 of the working copy for every dirty tracked or
    #: untracked non-runtime file.  Clean files are implied by ``head``.
    dirty: dict[str, str]
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"head": self.head, "dirty": dict(self.dirty), "at": self.at}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


class LiveTreeGuard:
    """Proves, after each experimental phase, that the live tree is unchanged.

    ``repository`` is ZEUS's directory; the fingerprint covers the whole git
    working tree it lives in.
    """

    def __init__(self, repository: Path) -> None:
        self.repository = Path(repository).resolve()
        self.top = git_root(self.repository) or self.repository

    def fingerprint(self) -> TreeFingerprint:
        head = _git(self.top, "rev-parse", "HEAD").stdout.strip()
        dirty: dict[str, str] = {}
        status = _git(self.top, "status", "--porcelain", "--untracked-files=all")
        for line in status.stdout.splitlines():
            if len(line) <= 3:
                continue
            rel = line[3:].strip().strip('"')
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            rel = rel.replace("\\", "/")
            if _is_runtime(rel):
                continue
            target = self.top / rel
            dirty[rel] = _sha(target) if target.is_file() else "deleted"
        return TreeFingerprint(head=head, dirty=dirty, at=_now())

    def changes_since(self, before: TreeFingerprint) -> dict[str, str]:
        """Files whose working copy differs from ``before``: path -> what happened."""

        now = self.fingerprint()
        out: dict[str, str] = {}
        if now.head != before.head:
            out["HEAD"] = f"{before.head[:12]} -> {now.head[:12]}"
        for rel, digest in now.dirty.items():
            if before.dirty.get(rel) != digest:
                out[rel] = "modified" if rel in before.dirty else "appeared"
        for rel in before.dirty:
            if rel not in now.dirty:
                out[rel] = "reverted"
        return out

    def contamination(self, before: TreeFingerprint, candidate: CandidateWorkspace | None) -> list[str]:
        """Changed live files that are byte-identical to the candidate's copy.

        That identity is the proof: the owner's own edits do not match a
        candidate's bytes, and neither does runtime state.
        """

        changed = [rel for rel, what in self.changes_since(before).items() if rel != "HEAD" and what != "reverted"]
        if candidate is None or not candidate.exists() or not changed:
            return []
        hits = []
        for rel in changed:
            live = self.top / rel
            cand = candidate.path / rel
            if live.is_file() and cand.is_file() and _sha(live) == _sha(cand):
                # Identical to the candidate AND different from the committed
                # version (or new): that is a candidate file in the live tree.
                hits.append(rel)
        return hits

    def restore(self, files: Iterable[str], before: TreeFingerprint) -> list[str]:
        """Put contaminated files back: committed content, or gone if new."""

        restored = []
        for rel in files:
            target = self.top / rel
            if rel in before.dirty:
                # It was dirty before the mission (the owner's edit) -- we
                # cannot know its previous bytes; leave it and report.
                continue
            tracked = _git(self.top, "ls-files", "--error-unmatch", "--", rel).returncode == 0
            if tracked:
                if _git(self.top, "checkout", "--", rel).returncode == 0:
                    restored.append(rel)
            else:
                try:
                    target.unlink()
                    restored.append(rel)
                except OSError:
                    continue
        return restored

    def check(self, before: TreeFingerprint, candidate: CandidateWorkspace | None, *, phase: str = "") -> dict[str, Any]:
        """One call after a phase: what changed, what was contamination, what was restored."""

        changed = self.changes_since(before)
        hits = self.contamination(before, candidate)
        restored = self.restore(hits, before) if hits else []
        return {"phase": phase, "changed": changed, "contamination": hits, "restored": restored,
                "clean": not hits, "at": _now()}


class MissionCancelled(RuntimeError):
    """Raised inside a mission when the owner asked it to stop."""
