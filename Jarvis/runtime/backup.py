"""Backup and restore of non-secret runtime state.

Git protects the code; nothing protected the state that makes this ZEUS *this*
ZEUS: projects, missions, owner corrections, the capability registry and its
installed capabilities, the knowledge graph, preferences, verified lessons and
experience.  A backup is one zip with a manifest (what, when, which revision,
sha256 per file); ``verify`` re-reads the archive and checks every hash;
``restore`` unpacks into the state root (never over the live tree's code),
keeping the current state aside first.

Secrets never enter a backup: the secret store directory is excluded, and any
file whose name looks like a credential is skipped and listed in the manifest
so the owner can see what was left out.  Backups stay local (``data/backups``
is ignored by git); uploading private runtime state is an owner decision the
product does not make.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: State-root subdirectories that belong in a backup.
INCLUDE = ("projects", "missions", "engine", "selfdev", "owner", "capabilities", "knowledge", "preferences", "experience",
           "devices", "expert_lessons.jsonl", "receipts.jsonl", "activity.jsonl", "performance.jsonl", "corrections.jsonl",
           "window", "supervisor/known_good.json", "supervisor/deployments.jsonl", "supervisor/releases.jsonl")
#: Never: the secret store, tokens, anything credential-shaped, caches, worktrees, evidence patches of candidates.
EXCLUDE_DIRS = ("secrets", "supervisor/logs", "selfdev/evidence", "__pycache__", "ui-health", "capabilities/installed/.venv")
SECRET_NAME = re.compile(r"(secret|token|credential|password|\.pem$|\.key$|client_secret)", re.I)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _revision(repository: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repository), capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class BackupManager:
    def __init__(self, state_root: str | Path, *, backups: str | Path | None = None, repository: Path | None = None) -> None:
        self.state_root = Path(state_root).resolve()
        self.backups = Path(backups) if backups else self.state_root.parent / "backups"
        self.repository = Path(repository) if repository else self.state_root.parents[1]

    def _files(self) -> tuple[list[Path], list[str]]:
        chosen: list[Path] = []
        skipped: list[str] = []
        for entry in INCLUDE:
            path = self.state_root / entry
            if path.is_file():
                candidates = [path]
            elif path.is_dir():
                candidates = [p for p in path.rglob("*") if p.is_file()]
            else:
                continue
            for p in candidates:
                rel = p.relative_to(self.state_root).as_posix()
                if any(rel.startswith(ex) or f"/{ex}/" in f"/{rel}" for ex in EXCLUDE_DIRS):
                    continue
                if SECRET_NAME.search(p.name) and not rel.startswith("owner/"):
                    skipped.append(rel)
                    continue
                if rel.endswith((".tmp", ".pyc")):
                    continue
                chosen.append(p)
        return chosen, skipped

    def create(self, *, label: str = "") -> dict[str, Any]:
        started = time.perf_counter()
        files, skipped = self._files()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"zeus-state-{stamp}{('-' + re.sub(r'[^a-z0-9]+', '-', label.lower())[:30]) if label else ''}.zip"
        self.backups.mkdir(parents=True, exist_ok=True)
        target = self.backups / name
        manifest: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(), "revision": _revision(self.repository),
                                    "state_root": str(self.state_root), "files": {}, "skipped_secret_like": skipped, "label": label}
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                rel = p.relative_to(self.state_root).as_posix()
                zf.write(p, f"state/{rel}")
                manifest["files"][rel] = {"sha256": _sha(p), "bytes": p.stat().st_size}
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        report = {"ok": True, "path": str(target), "files": len(files), "bytes": target.stat().st_size, "skipped_secret_like": skipped,
                  "revision": manifest["revision"], "seconds": round(time.perf_counter() - started, 2)}
        report["verified"] = self.verify(target)["ok"]
        return report

    def verify(self, path: str | Path) -> dict[str, Any]:
        """Every file in the archive hashes to what the manifest says."""

        path = Path(path)
        if not path.is_file():
            return {"ok": False, "error": "no such backup"}
        bad: list[str] = []
        missing: list[str] = []
        try:
            with zipfile.ZipFile(path) as zf:
                manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
                names = set(zf.namelist())
                for rel, meta in manifest.get("files", {}).items():
                    member = f"state/{rel}"
                    if member not in names:
                        missing.append(rel)
                        continue
                    digest = hashlib.sha256(zf.read(member)).hexdigest()
                    if digest != meta.get("sha256"):
                        bad.append(rel)
        except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            return {"ok": False, "error": f"unreadable backup: {exc}"}
        return {"ok": not bad and not missing, "checked": len(manifest.get("files", {})), "corrupt": bad, "missing": missing,
                "revision": manifest.get("revision", ""), "created_at": manifest.get("created_at", "")}

    def restore(self, path: str | Path, *, confirm: bool = False) -> dict[str, Any]:
        """Unpack into the state root, keeping the current state aside first."""

        if not confirm:
            return {"ok": False, "error": "restore needs confirm=true"}
        check = self.verify(path)
        if not check["ok"]:
            return {"ok": False, "error": f"backup does not verify: {check}"}
        aside = self.backups / f"before-restore-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        aside.mkdir(parents=True, exist_ok=True)
        restored: list[str] = []
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if not member.startswith("state/"):
                    continue
                rel = member[len("state/"):]
                if ".." in rel or rel.startswith("/") or SECRET_NAME.search(Path(rel).name) and not rel.startswith("owner/"):
                    continue
                target = self.state_root / rel
                if target.exists():
                    keep = aside / rel
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, keep)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                restored.append(rel)
        return {"ok": True, "restored": len(restored), "kept_aside": str(aside), "revision": check.get("revision", "")}

    def list(self) -> list[dict[str, Any]]:
        if not self.backups.is_dir():
            return []
        out = []
        for p in sorted(self.backups.glob("zeus-state-*.zip"), key=lambda x: x.stat().st_mtime, reverse=True):
            out.append({"path": str(p), "name": p.name, "bytes": p.stat().st_size, "at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()})
        return out
