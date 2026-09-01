"""The release pipeline: a ZEUS.exe that ZEUS can rebuild, verify, promote and roll back.

The gap this closes: ZEUS could promote Python changes into its own tree and
restart into them, but ``ZEUS.exe`` is the supervisor *frozen*, so a change
to the launcher or supervisor stayed invisible until a person rebuilt the
executable by hand.  This module makes the executable one more thing with a
known-good version, a candidate, a verification and a rollback.

Layout (``dist/`` beside the repository -- the one folder the owner excluded
from the antivirus, by their decision; nothing here touches that setting):

    dist/ZEUS/                 the known-good release (the shortcuts point here)
    dist/ZEUS.previous/        the release before the last promotion (rollback)
    dist/candidates/<id>/      candidate releases, each a complete onedir build

Flow:

    build_candidate()   PyInstaller into dist/candidates/<id>/ZEUS/, with
                        VERSION.json carrying the git commit and a
                        *launcher fingerprint* (a hash of zeus_supervisor/)
    verify_candidate()  the frozen exe answers ``--version``, ``check``
                        passes (preflight, no core), the fingerprint matches
                        the source it claims to be built from
    promote()           known-good -> ZEUS.previous (rename, works while the
                        old exe runs), candidate -> ZEUS; then a *relaunch*
                        through the supervisor and a watchdog
                        (zeus_supervisor.relaunch) that restores
                        ZEUS.previous if the new exe never reaches READY
    rollback()          ZEUS.previous -> ZEUS, explicit

``needs_rebuild()`` compares the fingerprint of the source with the one the
known-good exe was built from, so a source-only promotion never triggers a
rebuild and a launcher change is never silently left unbuilt.

No signing: a self-invented certificate would be worse than none.  The place
for real code signing is ``sign()`` below, which today records that it did
nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


#: What the executable is made of.  A change anywhere here means a rebuild.
LAUNCHER_SOURCES = ("zeus_supervisor",)


def launcher_fingerprint(repository: Path) -> str:
    """A hash of every file the frozen launcher is built from, plus the toolchain."""

    h = hashlib.sha256()
    for name in LAUNCHER_SOURCES:
        root = Path(repository) / name
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            h.update(path.relative_to(repository).as_posix().encode())
            h.update(path.read_bytes())
    icon = Path(repository) / "ui" / "zeus.ico"
    if icon.is_file():
        h.update(icon.read_bytes())
    h.update(sys.version.split()[0].encode())
    try:
        import PyInstaller  # noqa: WPS433

        h.update(str(PyInstaller.__version__).encode())
    except Exception:  # noqa: BLE001
        h.update(b"no-pyinstaller")
    return h.hexdigest()[:16]


def read_version(release_dir: Path) -> dict[str, Any]:
    path = Path(release_dir) / "VERSION.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


@dataclass
class ReleaseRecord:
    release_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    kind: str = ""  # build | verify | promote | rollback | relaunch
    candidate: str = ""
    outcome: str = ""
    reason: str = ""
    revision: str = ""
    fingerprint: str = ""
    seconds: float = 0.0
    at: str = field(default_factory=_now)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class ReleaseManager:
    def __init__(self, repository: Path, *, dist: Path | None = None, python: str = "", log: Callable[[str], None] | None = None) -> None:
        self.repository = Path(repository).resolve()
        self.dist = Path(dist) if dist else self.repository.parent / "dist"
        self.python = python or sys.executable
        self.log = log or (lambda _msg: None)
        self.ledger = self.repository / "data" / "jarvis" / "supervisor" / "releases.jsonl"

    # -- paths -------------------------------------------------------

    @property
    def known_good(self) -> Path:
        return self.dist / "ZEUS"

    @property
    def previous(self) -> Path:
        return self.dist / "ZEUS.previous"

    @property
    def candidates(self) -> Path:
        return self.dist / "candidates"

    def _record(self, record: ReleaseRecord) -> ReleaseRecord:
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), default=str) + "\n")
        except OSError:
            pass
        return record

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            rows = [json.loads(l) for l in self.ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (OSError, ValueError):
            return []
        return rows[-limit:]

    # -- status ------------------------------------------------------

    def revision(self) -> str:
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.repository), capture_output=True, text=True,
                                  timeout=30, creationflags=_no_window()).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def needs_rebuild(self) -> tuple[bool, str]:
        """Whether the known-good exe was built from different launcher sources."""

        version = read_version(self.known_good)
        if not (self.known_good / "ZEUS.exe").is_file():
            return True, "no known-good release exists"
        built = str(version.get("launcher_fingerprint", ""))
        current = launcher_fingerprint(self.repository)
        if not built:
            return True, f"the known-good release records no launcher fingerprint (built before this pipeline); source is {current}"
        if built != current:
            return True, f"launcher sources changed: built from {built}, source is {current}"
        return False, f"launcher unchanged ({current})"

    def status(self) -> dict[str, Any]:
        rebuild, why = self.needs_rebuild()
        cands = []
        if self.candidates.is_dir():
            for path in sorted(self.candidates.iterdir()):
                if path.is_dir():
                    cands.append({"id": path.name, "path": str(path), "version": read_version(path / "ZEUS"),
                                  "verified": (path / "VERIFIED.json").is_file()})
        return {
            "known_good": {"path": str(self.known_good), "exists": (self.known_good / "ZEUS.exe").is_file(),
                           "version": read_version(self.known_good)},
            "previous": {"path": str(self.previous), "exists": (self.previous / "ZEUS.exe").is_file(),
                         "version": read_version(self.previous)},
            "candidates": cands[-10:],
            "needs_rebuild": rebuild, "needs_rebuild_reason": why,
            "source_fingerprint": launcher_fingerprint(self.repository),
            "source_revision": self.revision(),
            "history": self.history(10),
        }

    # -- build -------------------------------------------------------

    def build_candidate(self, *, builder: Callable[[Path, Path], Path] | None = None) -> ReleaseRecord:
        """PyInstaller into ``dist/candidates/<id>/ZEUS``.  Never touches the known-good release."""

        started = time.monotonic()
        revision = self.revision()
        fingerprint = launcher_fingerprint(self.repository)
        cid = f"{revision[:12] or 'nogit'}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        target = self.candidates / cid
        record = ReleaseRecord(kind="build", candidate=str(target), revision=revision, fingerprint=fingerprint)
        try:
            target.mkdir(parents=True, exist_ok=False)
            if builder is not None:
                exe = builder(self.repository, target)
            else:
                from zeus_supervisor.build import build

                exe = build(self.repository, dist=target, shortcuts=False)
            out_dir = Path(exe).parent
            version = read_version(out_dir)
            version.update({"launcher_fingerprint": fingerprint, "candidate_id": cid, "revision": revision})
            (out_dir / "VERSION.json").write_text(json.dumps(version, indent=2), encoding="utf-8")
            record.outcome, record.reason = "built", f"{exe} ({Path(exe).stat().st_size // 1024} KB)"
            record.detail = {"exe": str(exe), "version": version}
        except Exception as exc:  # noqa: BLE001 - a failed build is a record, not a crash
            record.outcome, record.reason = "failed", f"{type(exc).__name__}: {exc}"[:600]
        record.seconds = round(time.monotonic() - started, 1)
        self.log(f"release build {record.outcome}: {record.reason}")
        return self._record(record)

    # -- verify ------------------------------------------------------

    def verify_candidate(self, candidate: str | Path, *, runner: Callable[[list[str]], tuple[int, str]] | None = None) -> ReleaseRecord:
        """Independent checks on the frozen candidate, none of them its own report.

        1. the files are there and sane (ZEUS.exe, _internal/, VERSION.json);
        2. VERSION.json's fingerprint equals the source it claims (it was
           built from this tree, not an older one);
        3. the exe runs: ``--version`` prints the supervisor version;
        4. ``check`` -- the full preflight from *inside the frozen program*,
           against the real repository, Ollama and models -- exits 0.
        The exe is never started as a supervisor here: the running ZEUS holds
        the instance lock, and a boot is what the relaunch watchdog verifies.
        """

        started = time.monotonic()
        cand_root = Path(candidate)
        out_dir = cand_root / "ZEUS" if (cand_root / "ZEUS").is_dir() else cand_root
        exe = out_dir / "ZEUS.exe"
        record = ReleaseRecord(kind="verify", candidate=str(cand_root))
        checks: list[dict[str, Any]] = []

        def check(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail[:400]})

        check("files", exe.is_file() and (out_dir / "_internal").is_dir() and (out_dir / "VERSION.json").is_file(),
              f"{exe} {'exists' if exe.is_file() else 'missing'}")
        version = read_version(out_dir)
        record.revision = str(version.get("revision", ""))
        record.fingerprint = str(version.get("launcher_fingerprint", ""))
        check("fingerprint", record.fingerprint == launcher_fingerprint(self.repository),
              f"built from {record.fingerprint or '?'}, source is {launcher_fingerprint(self.repository)}")
        size = exe.stat().st_size if exe.is_file() else 0
        check("size", 100_000 < size < 200_000_000, f"{size} bytes")

        run = runner or self._run
        if exe.is_file():
            code, out = run([str(exe), "--version"])
            check("runs", code == 0 and "supervisor" in out.lower(), out.strip()[:200] or f"exit {code}")
            code, out = run([str(exe), "check"])
            check("preflight", code == 0, (out.strip().splitlines() or [f"exit {code}"])[-1][:300])
        ok = all(c["ok"] for c in checks)
        record.outcome = "verified" if ok else "rejected"
        record.reason = "; ".join(f"{c['name']}: {'ok' if c['ok'] else 'FAILED'}" for c in checks)
        record.detail = {"checks": checks, "version": version}
        record.seconds = round(time.monotonic() - started, 1)
        if ok:
            (cand_root / "VERIFIED.json").write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")
        self.log(f"release verify {record.outcome}: {record.reason}")
        return self._record(record)

    def _run(self, command: list[str], timeout: float = 240.0) -> tuple[int, str]:
        env = dict(os.environ)
        env["ZEUS_REPO"] = str(self.repository)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                                       errors="replace", env=env, cwd=str(Path(command[0]).parent), creationflags=_no_window())
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout:.0f}s"
        except OSError as exc:
            return 127, str(exc)
        return completed.returncode, f"{completed.stdout}\n{completed.stderr}"

    # -- promote / rollback ------------------------------------------

    def promote(self, candidate: str | Path, *, require_verified: bool = True) -> ReleaseRecord:
        """Known-good -> previous, candidate -> known-good.  Renames, so the running exe is unaffected."""

        started = time.monotonic()
        cand_root = Path(candidate)
        out_dir = cand_root / "ZEUS" if (cand_root / "ZEUS").is_dir() else cand_root
        record = ReleaseRecord(kind="promote", candidate=str(cand_root))
        version = read_version(out_dir)
        record.revision, record.fingerprint = str(version.get("revision", "")), str(version.get("launcher_fingerprint", ""))
        if require_verified and not (cand_root / "VERIFIED.json").is_file():
            record.outcome, record.reason = "refused", "the candidate has not been verified"
            return self._record(record)
        if not (out_dir / "ZEUS.exe").is_file():
            record.outcome, record.reason = "refused", "the candidate has no ZEUS.exe"
            return self._record(record)
        try:
            if self.previous.exists():
                shutil.rmtree(self.previous, ignore_errors=True)
                if self.previous.exists():
                    # A previous release still running cannot be removed; park it.
                    self.previous.rename(self.dist / f"ZEUS.previous.{int(time.time())}")
            if self.known_good.exists():
                try:
                    self.known_good.rename(self.previous)
                except PermissionError:
                    # Windows: the running ZEUS.exe holds its own directory
                    # open, so the swap cannot happen while it runs.  Stage
                    # the candidate beside it; the relaunch watchdog performs
                    # the swap the moment the old supervisor has exited, then
                    # starts the new release (and restores the previous one if
                    # it never gets READY).  Nothing known-good is touched here.
                    return self._record(self._stage(cand_root, out_dir, record))
            shutil.copytree(out_dir, self.known_good)
            (self.known_good / "PROMOTED.json").write_text(json.dumps({
                "candidate": str(cand_root), "at": _now(), "revision": record.revision, "fingerprint": record.fingerprint,
                "previous": str(self.previous) if (self.previous / "ZEUS.exe").is_file() else "",
            }, indent=2), encoding="utf-8")
            record.outcome, record.reason = "promoted", f"{self.known_good} now {record.fingerprint} @ {record.revision[:12]}; previous kept"
        except OSError as exc:
            record.outcome, record.reason = "failed", f"{type(exc).__name__}: {exc}"[:400]
            # Put the known-good back if the rename half happened.
            if not self.known_good.exists() and self.previous.exists():
                try:
                    self.previous.rename(self.known_good)
                    record.reason += "; known-good restored"
                except OSError:
                    record.reason += "; AND THE KNOWN-GOOD COULD NOT BE RESTORED"
        record.seconds = round(time.monotonic() - started, 1)
        self.log(f"release promote {record.outcome}: {record.reason}")
        return self._record(record)

    @property
    def staged(self) -> Path:
        return self.dist / "ZEUS.staged"

    @property
    def staged_pointer(self) -> Path:
        return self.repository / "data" / "jarvis" / "supervisor" / "control" / "staged.json"

    def _stage(self, cand_root: Path, out_dir: Path, record: ReleaseRecord) -> ReleaseRecord:
        """Copy the verified candidate to ``dist/ZEUS.staged`` and leave the swap to the watchdog."""

        try:
            if self.staged.exists():
                shutil.rmtree(self.staged, ignore_errors=True)
            shutil.copytree(out_dir, self.staged)
            self.staged_pointer.parent.mkdir(parents=True, exist_ok=True)
            self.staged_pointer.write_text(json.dumps({
                "staged": str(self.staged), "known_good": str(self.known_good), "previous": str(self.previous),
                "candidate": str(cand_root), "revision": record.revision, "fingerprint": record.fingerprint, "at": _now(),
            }, indent=2), encoding="utf-8")
            record.outcome = "staged"
            record.reason = (f"the running ZEUS.exe holds {self.known_good}; candidate staged at {self.staged} "
                             f"({record.fingerprint} @ {record.revision[:12]}) -- the relaunch watchdog swaps it in")
        except OSError as exc:
            record.outcome, record.reason = "failed", f"staging failed: {type(exc).__name__}: {exc}"[:400]
        self.log(f"release promote {record.outcome}: {record.reason}")
        return record

    def rollback(self, reason: str = "explicit rollback") -> ReleaseRecord:
        record = ReleaseRecord(kind="rollback", reason=reason)
        if not (self.previous / "ZEUS.exe").is_file():
            record.outcome, record.reason = "refused", "no previous release to roll back to"
            return self._record(record)
        try:
            failed = self.dist / f"ZEUS.failed.{int(time.time())}"
            if self.known_good.exists():
                self.known_good.rename(failed)
            self.previous.rename(self.known_good)
            record.outcome = "rolled_back"
            record.revision = str(read_version(self.known_good).get("revision", ""))
            record.detail = {"failed_release_kept_at": str(failed)}
        except OSError as exc:
            record.outcome, record.reason = "failed", f"{type(exc).__name__}: {exc}"[:400]
        self.log(f"release rollback {record.outcome}: {record.reason}")
        return self._record(record)

    def prune(self, keep: int = 3) -> list[str]:
        """Old candidates and parked failed releases, oldest first."""

        removed = []
        if self.candidates.is_dir():
            dirs = sorted((p for p in self.candidates.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
            for path in dirs[:-keep] if keep else dirs:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
        for path in self.dist.glob("ZEUS.failed.*"):
            if time.time() - path.stat().st_mtime > 7 * 86400:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(str(path))
        return removed

    def sign(self, release_dir: Path) -> dict[str, Any]:
        """Code signing: not done.  Documented, not faked.

        A real signature needs a certificate the owner obtains (an EV or OV
        code-signing certificate from a CA, or a Windows Store identity);
        ``signtool sign /fd SHA256 /tr <timestamp> /td SHA256 /f <pfx>`` is the
        step that would go here, with the certificate held outside the
        repository.  Self-signed certificates buy nothing with SmartScreen
        or an antivirus and are not used.
        """

        return {"signed": False, "reason": "no code-signing certificate is configured; see ReleaseManager.sign"}
