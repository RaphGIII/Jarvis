"""The internal doctor: "is ZEUS healthy?", answered from deterministic checks.

Every check here reads something that already exists -- a process table, a
file, a registry, a git revision, the last measured tier status -- and none of
them wakes a model.  Drawing Diagnostics used to be a real generation that
evicted the conversational model; the doctor is the opposite: it can run every
minute and nobody would notice.

Each check returns ``ok`` plus a *level*: ``ok``, ``warn`` (works, but
something is off), ``error`` (does not work).  ``healthy`` is "no errors".
Remedies are sentences a person can act on, not codes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class Check:
    name: str
    ok: bool
    level: str  # ok | warn | error
    detail: str
    remedy: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "level": self.level, "detail": self.detail[:400],
                "remedy": self.remedy[:300], "data": self.data}


def _ok(name: str, detail: str, **data: Any) -> Check:
    return Check(name, True, "ok", detail, data=data)


def _warn(name: str, detail: str, remedy: str = "", **data: Any) -> Check:
    return Check(name, True, "warn", detail, remedy, data=data)


def _error(name: str, detail: str, remedy: str = "", **data: Any) -> Check:
    return Check(name, False, "error", detail, remedy, data=data)


def _git_head(repository: Path) -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repository), capture_output=True, text=True,
                                   timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


class Doctor:
    """Runs the checks against a core.  Everything optional degrades to a warning."""

    def __init__(self, core: Any, *, repository: Path | None = None) -> None:
        self.core = core
        self.repository = Path(repository) if repository else Path(__file__).resolve().parents[1]

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        checks: list[Check] = []
        for step in (self._supervisor, self._core, self._fast_local, self._build_local, self._expert, self._ollama,
                     self._gpu, self._voice_processes, self._duplicates, self._window, self._revision, self._release,
                     self._capabilities, self._mission_stores, self._pending_rollback, self._isolation):
            try:
                checks.append(step())
            except Exception as exc:  # noqa: BLE001 - a broken check is a finding, not a crash
                checks.append(_error(step.__name__.strip("_"), f"check crashed: {type(exc).__name__}: {exc}"))
        errors = [c for c in checks if c.level == "error"]
        warnings = [c for c in checks if c.level == "warn"]
        return {
            "healthy": not errors,
            "summary": ("healthy" if not errors and not warnings else
                        f"{len(errors)} error(s), {len(warnings)} warning(s)"),
            "errors": [c.name for c in errors], "warnings": [c.name for c in warnings],
            "checks": [c.to_dict() for c in checks],
            "at": datetime.now(timezone.utc).isoformat(), "seconds": round(time.perf_counter() - started, 3),
        }

    # -- checks ------------------------------------------------------

    def _supervisor(self) -> Check:
        life = self.core.lifecycle
        status = life.supervisor_status()
        if not life.supervised:
            return _warn("supervisor", "not running under the supervisor (started by hand)",
                         "Start ZEUS through ZEUS.exe or python -m zeus_supervisor for restart/rollback protection")
        st = status.get("status") or {}
        updated = str(st.get("updated_at", ""))
        return _ok("supervisor", f"{st.get('phase', '?')}: {str(st.get('detail', ''))[:80]} (pid {st.get('supervisor_pid')})",
                   phase=st.get("phase"), updated_at=updated, known_good=(status.get("known_good") or {}).get("revision", "")[:12])

    def _core(self) -> Check:
        health = self.core.lifecycle.health()
        rd = health.get("readiness") or {}
        if not health.get("ready"):
            return _error("core", f"not READY: {health.get('detail', '')}", "Read data/jarvis/supervisor/logs/core.log",
                          readiness=rd)
        return _ok("core", f"READY, pid {health.get('pid')}, up {health.get('uptime_seconds')}s", readiness=rd,
                   pid=health.get("pid"))

    def _tier(self, name: str) -> dict[str, Any]:
        try:
            status = self.core.kernel.status(force=False, probe=False)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        tiers = status.get("tiers") or status.get("brains") or {}
        return tiers.get(name) or tiers.get(name.lower()) or {}

    def _fast_local(self) -> Check:
        stage = (self.core.lifecycle.stages or {}).get("fast_local") or {}
        tier = self._tier("FAST_LOCAL")
        model = tier.get("model") or tier.get("spec", {}).get("model") if isinstance(tier, dict) else ""
        if stage.get("ok"):
            return _ok("fast_local", f"{model or 'FAST_LOCAL'}: a real generation succeeded in this process", model=model)
        return _error("fast_local", f"no generation yet: {stage.get('detail', 'not measured')}",
                      "Check that Ollama is running and serves the FAST_LOCAL model", model=model)

    def _build_local(self) -> Check:
        tier = self._tier("BUILD_LOCAL")
        model = tier.get("model") or (tier.get("spec") or {}).get("model") if isinstance(tier, dict) else ""
        available = tier.get("available") if isinstance(tier, dict) else None
        if available is False:
            return _warn("build_local", f"{model or 'BUILD_LOCAL'} not available (last measured)",
                         "Self-development will escalate to the expert; pull the coder model into the Ollama store", model=model)
        return _ok("build_local", f"{model or 'BUILD_LOCAL'} configured (not probed: probing evicts the chat model)", model=model)

    def _expert(self) -> Check:
        try:
            status = self.core.experts.status()
        except Exception as exc:  # noqa: BLE001
            return _warn("expert", f"gateway status unavailable: {exc}")
        if status.get("quota_exhausted"):
            return _warn("expert", "expert quota exhausted", "Wait for the subscription window to reset", **{k: status[k] for k in ("provider",) if k in status})
        if status.get("expert_available"):
            return _ok("expert", f"available via {status.get('provider', 'subscription')}")
        return _warn("expert", "no expert available (local only)", "Log in to the subscription CLI if escalation is wanted",
                     blocker=str(status.get("blocker", ""))[:120])

    def _ollama(self) -> Check:
        try:
            from zeus_supervisor.config import SupervisorConfig
            from zeus_supervisor.preflight import Preflight, PreflightCache

            config = SupervisorConfig.load(self.repository)
            report = Preflight(config, log=lambda _m: None,
                               cache=PreflightCache(config.state_dir / "preflight_cache.json")).run(generation=False)
        except Exception as exc:  # noqa: BLE001
            return _warn("ollama", f"preflight could not run: {exc}")
        failed = [c for c in report.checks if not c.ok and c.required]
        soft = [c for c in report.checks if not c.ok and not c.required]
        detail = ", ".join(f"{c.name}={'ok' if c.ok else 'FAIL'}" for c in report.checks)
        if failed:
            return _error("ollama", f"{failed[0].name}: {failed[0].detail}", failed[0].remedy, checks=detail)
        if soft:
            return _warn("ollama", f"{soft[0].name}: {soft[0].detail}", soft[0].remedy, checks=detail,
                         version=report.ollama_version)
        return _ok("ollama", f"Ollama {report.ollama_version}, models present", checks=detail, version=report.ollama_version)

    def _gpu(self) -> Check:
        try:
            gpu = self.core.gpu_usage()
        except Exception as exc:  # noqa: BLE001
            return _warn("gpu", f"unreadable: {exc}")
        if not gpu or gpu.get("error"):
            return _warn("gpu", f"no GPU reading: {gpu.get('error', 'nvidia-smi absent') if gpu else 'none'}")
        used, total = gpu.get("memory_used_mib"), gpu.get("memory_total_mib")
        detail = f"{gpu.get('name', 'GPU')}: {gpu.get('utilization_percent', '?')}% busy, {used}/{total} MiB"
        if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total and used / total > 0.95:
            return _warn("gpu", detail + " (VRAM nearly full)", "Two models cannot coexist on this GPU; avoid probing BUILD_LOCAL while chatting")
        return _ok("gpu", detail, **{k: gpu.get(k) for k in ("utilization_percent", "memory_used_mib", "memory_total_mib")})

    def _voice_processes(self) -> Check:
        from service.processes import counts

        c = counts()
        stages = self.core.lifecycle.stages or {}
        voice_ok = bool(stages.get("voice", {}).get("ok")) and bool(stages.get("recogniser", {}).get("ok"))
        if c.get("listener", 0) == 0:
            return _warn("voice", f"no wake-word listener running (worker {c.get('worker', 0)})",
                         "The supervisor starts the listener after READY; check data/jarvis/supervisor/logs/listener.log", counts=c)
        if not voice_ok:
            return _warn("voice", f"listener {c.get('listener')}, worker {c.get('worker')}; speech stack not yet warm", counts=c)
        return _ok("voice", f"listener {c.get('listener')}, worker {c.get('worker')}, speech warm", counts=c)

    def _duplicates(self) -> Check:
        from service.processes import counts

        c = counts()
        dup = {role: n for role, n in c.items() if n > 1}
        if dup:
            return _error("duplicates", f"more than one of: {dup}", "Run 'ZEUS vollständig beenden' and start ZEUS once", counts=c)
        return _ok("duplicates", "one of each: " + ", ".join(f"{r}={n}" for r, n in c.items()), counts=c)

    def _window(self) -> Check:
        desktop = getattr(self.core.lifecycle, "desktop", None)
        if desktop is None:
            return _warn("window", "no managed desktop window (headless or no Chromium engine)")
        st = desktop.status()
        if st.get("windows", 0) > 1:
            return _warn("window", f"{st['windows']} ZEUS windows", "The next show() closes duplicates", **st)
        if not st.get("exists"):
            return _ok("window", "closed (core keeps running; ZEUS.exe or /api/window/show reopens it)", **{k: st[k] for k in ("engine",)})
        return _ok("window", f"{'visible' if st.get('visible') else 'hidden'} ({st.get('engine')}, identity {st.get('identity')})")

    def _revision(self) -> Check:
        head = _git_head(self.repository)
        life = self.core.lifecycle
        running = life.revision()
        known = ((life.supervisor_status().get("known_good") or {}).get("revision") or "")
        detail = f"running {running[:12]}, HEAD {head[:12]}, known-good {known[:12] or '?'}"
        if running and head and running != head:
            return _warn("revision", detail + " (the tree moved under the running process)", "Restart to run HEAD")
        if known and running and known != running:
            return _warn("revision", detail + " (not yet known-good)", "The supervisor marks it after a live READY")
        return _ok("revision", detail, head=head, running=running, known_good=known)

    def _release(self) -> Check:
        try:
            status = self.core.releases.status()
        except Exception as exc:  # noqa: BLE001
            return _warn("release", f"release status unavailable: {exc}")
        kg = status.get("known_good") or {}
        if not kg.get("exists"):
            return _warn("release", "no known-good ZEUS.exe built", "python -m zeus_supervisor.build, or /api/release/build")
        if status.get("needs_rebuild"):
            return _warn("release", f"ZEUS.exe is stale: {status.get('needs_rebuild_reason')}", "Build and promote a release (/api/release/build)",
                         revision=str(kg.get("version", {}).get("revision", ""))[:12])
        return _ok("release", f"ZEUS.exe at {str(kg.get('version', {}).get('revision', ''))[:12]}, launcher {kg.get('version', {}).get('launcher_fingerprint', '')}",
                   previous=bool((status.get("previous") or {}).get("exists")))

    def _capabilities(self) -> Check:
        try:
            manifests = self.core.capabilities.registry.all()
        except Exception as exc:  # noqa: BLE001
            return _error("capabilities", f"registry unreadable: {exc}", "Check data/jarvis/capabilities/registry.json")
        problems = []
        stale = []
        for m in manifests:
            cid = str(getattr(m, "capability_id", "?"))
            source = Path(str(getattr(m, "source_location", "") or ""))
            entry = str(getattr(m, "entrypoint", "") or "")
            status = str(getattr(m, "status", "") or "")
            if not source.is_dir() or (entry and not (source / entry).is_file()):
                problems.append(f"{cid}: source or entrypoint missing")
            if status not in {"active", "disabled", "degraded", "acquiring", "repairing", "unverified", "deprecated"}:
                problems.append(f"{cid}: unknown status {status!r}")
            if status == "disabled":
                stale.append(cid)
        if problems:
            return _error("capabilities", "; ".join(problems[:4]), "Repair or remove the broken entries", count=len(manifests))
        if stale:
            return _warn("capabilities", f"{len(manifests)} registered, disabled: {stale[:4]}", "A repair that was killed leaves a capability disabled; resume or re-verify it", count=len(manifests))
        return _ok("capabilities", f"{len(manifests)} registered, all with source and entrypoint", count=len(manifests),
                   ids=[str(getattr(m, 'capability_id', '')) for m in manifests][:12])

    def _mission_stores(self) -> Check:
        state = Path(self.core.kernel.state_root)
        broken = []
        counts = {}
        for name in ("missions", "selfdev"):
            root = state / name
            n = 0
            if root.is_dir():
                for path in root.glob("*.json"):
                    n += 1
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        broken.append(f"{name}/{path.name}")
            counts[name] = n
        active = []
        try:
            for m in self.core.selfdev_store.list():
                if not m.finished:
                    active.append(f"{m.mission_id}:{m.phase}")
        except Exception:  # noqa: BLE001
            pass
        if broken:
            return _error("missions", f"unreadable mission files: {broken[:4]}", "Move them aside; the store skips what it cannot parse", counts=counts)
        return _ok("missions", f"{counts.get('missions', 0)} capability + {counts.get('selfdev', 0)} selfdev mission files"
                   + (f", active: {active}" if active else ""), counts=counts, active=active)

    def _pending_rollback(self) -> Check:
        from deployment.promotion import Journal

        pending = []
        snapshots = self.repository / "data" / "snapshots"
        if snapshots.is_dir():
            for directory in snapshots.iterdir():
                if directory.is_dir() and Journal(directory).read().get("status") in {"applying", "applied"}:
                    pending.append(directory.name)
        failed = sorted(p.name for p in (self.repository.parent / "dist").glob("ZEUS.failed.*")) if (self.repository.parent / "dist").is_dir() else []
        if pending:
            return _error("rollback", f"promotion(s) applied but never committed: {pending[:3]}",
                          "They are restored at the next start (recover_interrupted); restart ZEUS", pending=pending)
        if failed:
            return _warn("rollback", f"a failed release is parked: {failed[-1]}", "Inspect dist/ZEUS.failed.* and delete it when understood", failed=failed)
        return _ok("rollback", "nothing pending")

    def _isolation(self) -> Check:
        from service.isolation import CandidateWorkspace, git_root

        top = git_root(self.repository)
        if top is None:
            return _warn("isolation", "not a git repository")
        try:
            listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(top), capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return _warn("isolation", "git worktree list failed")
        candidates = [l.split(" ", 1)[1] for l in listing.splitlines() if l.startswith("worktree ") and "candidate_" in l]
        active = []
        try:
            active = [m.mission_id for m in self.core.selfdev_store.list() if not m.finished]
        except Exception:  # noqa: BLE001
            pass
        stray = [c for c in candidates if not any(mid in c for mid in active)]
        if stray:
            return _warn("isolation", f"{len(stray)} candidate worktree(s) without an active mission", "They are reaped at the next start", stray=stray[:4])
        return _ok("isolation", f"{len(candidates)} candidate worktree(s), all belonging to active missions" if candidates else "no candidate worktrees")
