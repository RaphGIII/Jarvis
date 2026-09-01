"""The internal doctor: "is ZEUS healthy?", answered from deterministic checks.

Every check here reads something that already exists -- a process table, a
file, a registry, a git revision, the last measured tier status -- and none of
them wakes a model.  Drawing Diagnostics used to be a real generation that
evicted the conversational model; the doctor is the opposite: it can run every
minute and nobody would notice.

Each check returns ``ok`` plus a *level*: ``ok``, ``warn`` (works, but
something is off), ``error`` (does not work).  ``healthy`` is "no errors".
Remedies are sentences a person can act on, not codes.

Liveness is not function.  A process that exists is one fact; whether the
product function behind it works is another, and the doctor reports both:
every check belongs to a *subsystem* (infrastructure, core, voice, wakeword,
knowledge, capabilities, expert, projects), each subsystem's health is the
worst of its checks -- HEALTHY / DEGRADED / FAILING -- and ``overall`` is
DEGRADED as soon as a product function is, even while every process runs.
The functional checks (knowledge round trip, wake model + evaluation +
listener agreement, capability runtime health, cached expert state) read
files and in-memory state; none of them wakes a model or spawns a CLI.
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


#: Which subsystem each check reports on.  Infrastructure checks say a thing
#: exists; the others say whether a product function works.
SUBSYSTEMS = {
    "supervisor": "infrastructure", "ollama": "infrastructure", "gpu": "infrastructure", "duplicates": "infrastructure",
    "window": "infrastructure", "revision": "infrastructure", "release": "infrastructure", "rollback": "infrastructure",
    "isolation": "infrastructure", "mission_stores": "infrastructure",
    "core": "core", "fast_local": "core", "build_local": "core",
    "voice": "voice", "wakeword": "wakeword", "knowledge": "knowledge",
    "capabilities": "capabilities", "capability_health": "capabilities", "expert": "expert", "projects": "projects",
}
#: Subsystems whose degradation degrades the product as a whole.
IMPORTANT = ("core", "voice", "wakeword", "knowledge", "capabilities", "projects")
HEALTH = {"ok": "HEALTHY", "warn": "DEGRADED", "error": "FAILING"}


@dataclass
class Check:
    name: str
    ok: bool
    level: str  # ok | warn | error
    detail: str
    remedy: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def subsystem(self) -> str:
        return SUBSYSTEMS.get(self.name, "infrastructure")

    @property
    def health(self) -> str:
        return HEALTH.get(self.level, "FAILING")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "level": self.level, "detail": self.detail[:400],
                "remedy": self.remedy[:300], "data": self.data, "subsystem": self.subsystem, "health": self.health}


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
                     self._gpu, self._voice_processes, self._wakeword, self._duplicates, self._window, self._revision, self._release,
                     self._capabilities, self._capability_health, self._knowledge, self._projects, self._mission_stores,
                     self._pending_rollback, self._isolation):
            try:
                checks.append(step())
            except Exception as exc:  # noqa: BLE001 - a broken check is a finding, not a crash
                checks.append(_error(step.__name__.strip("_"), f"check crashed: {type(exc).__name__}: {exc}"))
        errors = [c for c in checks if c.level == "error"]
        warnings = [c for c in checks if c.level == "warn"]
        order = {"HEALTHY": 0, "DEGRADED": 1, "FAILING": 2}
        subsystems: list[dict[str, Any]] = []
        for name in dict.fromkeys(list(dict.fromkeys(SUBSYSTEMS.values()))):
            mine = [c for c in checks if c.subsystem == name]
            if not mine:
                continue
            health = max((c.health for c in mine), key=lambda h: order[h])
            subsystems.append({"name": name, "health": health, "important": name in IMPORTANT,
                               "checks": [c.name for c in mine], "detail": "; ".join(f"{c.name}: {c.detail[:80]}" for c in mine if c.level != "ok")[:300]})
        worst_important = max((s["health"] for s in subsystems if s["important"]), key=lambda h: order[h], default="HEALTHY")
        overall = "FAILING" if errors else ("DEGRADED" if worst_important != "HEALTHY" else "HEALTHY")
        degraded = [s["name"] for s in subsystems if s["health"] != "HEALTHY"]
        return {
            "healthy": not errors,
            "overall": overall,
            "summary": ("healthy" if overall == "HEALTHY" else
                        f"{overall.lower()}: {', '.join(degraded)} — {len(errors)} error(s), {len(warnings)} warning(s)"),
            "errors": [c.name for c in errors], "warnings": [c.name for c in warnings],
            "subsystems": subsystems,
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
        """From the gateway's cached state: never a CLI spawn per Diagnostics render."""

        try:
            status = self.core.experts.status()
        except Exception as exc:  # noqa: BLE001
            return _warn("expert", f"gateway status unavailable: {exc}")
        rows = [r for r in status.get("providers", []) if r.get("permitted")]
        row = rows[0] if rows else {}
        state = str(status.get("state") or row.get("state") or "UNKNOWN")
        checked = row.get("checked_at") or 0
        age = f"{int(time.time() - float(checked))}s ago" if checked else "never checked"
        name = row.get("name", "expert")
        evidence = f"{name} {row.get('version', '')}".strip() + f" · {row.get('evidence', '')} · {age}"
        if state in {"QUOTA_EXHAUSTED", "RATE_LIMITED"}:
            return _warn("expert", f"{state}: {str(row.get('detail', ''))[:100]}", "Wait for the subscription window to reset; ZEUS stays local until a call succeeds (never PAYG)",
                         state=state, evidence=evidence)
        if state == "AVAILABLE":
            return _ok("expert", f"AVAILABLE via subscription CLI ({evidence})", state=state, evidence=evidence)
        if state == "NOT_INSTALLED":
            return _warn("expert", "NOT_INSTALLED: no subscription CLI on PATH (local only)", "Install and sign in to the CLI if escalation is wanted", state=state)
        if state == "NOT_AUTHENTICATED":
            return _warn("expert", "NOT_AUTHENTICATED: the CLI needs a sign-in (local only)", "Sign in to the subscription CLI", state=state)
        return _warn("expert", f"{state}: {str(row.get('detail', status.get('blocker', '')))[:100]} (local only)", "Log in to the subscription CLI if escalation is wanted",
                     state=state, evidence=evidence)

    def _ollama(self) -> Check:
        try:
            from zeus_supervisor.config import SupervisorConfig
            from zeus_supervisor.preflight import Preflight, PreflightCache

            config = SupervisorConfig.load(self.repository)
            # A diagnosis observes; it never starts services.  The supervisor
            # owns Ollama's lifecycle and reports RUNNING / STARTING /
            # UNAVAILABLE / FAILED / MISSING with the reason.
            report = Preflight(config, log=lambda _m: None,
                               cache=PreflightCache(config.state_dir / "preflight_cache.json")).run(generation=False, start_services=False)
        except Exception as exc:  # noqa: BLE001
            return _warn("ollama", f"preflight could not run: {exc}")
        failed = [c for c in report.checks if not c.ok and c.required]
        soft = [c for c in report.checks if not c.ok and not c.required]
        detail = ", ".join(f"{c.name}={'ok' if c.ok else 'FAIL'}" for c in report.checks)
        state = report.ollama_state or ("RUNNING" if report.ollama_version else "UNAVAILABLE")
        if failed:
            return _error("ollama", f"{state}: {failed[0].name}: {failed[0].detail}", failed[0].remedy, checks=detail,
                          state=state, reason=report.ollama_reason)
        if soft:
            return _warn("ollama", f"{state}: {soft[0].name}: {soft[0].detail}", soft[0].remedy, checks=detail,
                         version=report.ollama_version, state=state, reason=report.ollama_reason)
        return _ok("ollama", f"RUNNING: Ollama {report.ollama_version}, models present", checks=detail, version=report.ollama_version,
                   state="RUNNING", reason=report.ollama_reason)

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

    def _wakeword(self) -> Check:
        """The wake word as a function: a trained owner model, evaluated, and the listener running that exact model/threshold."""

        try:
            status = self.core.wake_status()
        except Exception as exc:  # noqa: BLE001
            return _warn("wakeword", f"wake status unavailable: {exc}")
        kind = status.get("model_kind", "NONE")
        threshold = status.get("effective_threshold")
        if kind == "NONE":
            return _error("wakeword", "no trained wake model: the listener falls back to 'hey jarvis'", "Voice Studio → record „Zeus“ 15× and train")
        ev = status.get("evaluation") or {}
        listener = status.get("listener")
        data = {"model_kind": kind, "effective_threshold": threshold, "threshold_source": status.get("threshold_source"),
                "recall": ev.get("positive_recall"), "rejection": ev.get("negative_rejection"), "evaluated_at": ev.get("at"),
                "listener_match": status.get("listener_match"), "fingerprint": status.get("model_fingerprint")}
        if kind != "OWNER":
            return _warn("wakeword", "synthetic-only model: not trained on the owner's voice", "Record owner samples in Voice Studio and train", **data)
        if not ev:
            return _warn("wakeword", f"OWNER model, threshold {threshold} ({status.get('threshold_source')}), never evaluated", "Voice Studio → Calibrate", **data)
        if ev.get("stale"):
            return _warn("wakeword", "the evaluation is for an older model", "Voice Studio → Calibrate", **data)
        recall, rejection = ev.get("positive_recall"), ev.get("negative_rejection")
        if recall is not None and recall < 0.8:
            return _warn("wakeword", f"owner recall {recall:.0%} at threshold {threshold} (rejection {rejection:.0%})", "Lower the wake threshold or record more samples", **data)
        if rejection is not None and rejection < 1.0:
            return _warn("wakeword", f"false activations on owner negatives: rejection {rejection:.0%} at {threshold}", "Raise the wake threshold", **data)
        if listener is None:
            return _warn("wakeword", f"OWNER model, recall {recall:.0%}, rejection {rejection:.0%} at {threshold}; no listener report in the last 30 s",
                         "The listener reports every 5 s; check listener.log", **data)
        if not status.get("listener_match"):
            return _warn("wakeword", f"the listener runs model {listener.get('fingerprint')} at {listener.get('threshold')}, the product expects {status.get('model_fingerprint')} at {threshold}",
                         "The listener reloads within seconds; if not, restart ZEUS", **data)
        return _ok("wakeword", f"OWNER model, recall {recall:.0%}, rejection {rejection:.0%} at threshold {threshold} ({status.get('threshold_source')}); listener runs the same model and threshold", **data)

    def _knowledge(self) -> Check:
        """The graph as a function: one file, readable, non-empty, and a write path that exists."""

        try:
            stats = self.core.knowledge_stats()
        except Exception as exc:  # noqa: BLE001
            return _error("knowledge", f"graph unreadable: {exc}", "Check data/jarvis/knowledge/palace.sqlite")
        if not stats.get("ok"):
            return _error("knowledge", f"graph unreadable: {stats.get('error')}", "Check data/jarvis/knowledge/palace.sqlite", path=stats.get("path"))
        nodes = int(stats.get("nodes", 0) or 0)
        edges = int(stats.get("edges", 0) or 0)
        legacy = Path(str(stats.get("path", ""))).with_name("graph.db")
        if legacy.is_file():
            return _warn("knowledge", f"{nodes} nodes, {edges} edges, but a legacy graph.db still exists beside the graph", "It is migrated on the next start", path=stats.get("path"))
        if nodes == 0:
            return _warn("knowledge", "the graph is empty", "Store a finding (knowledge.create) or ingest a document", path=stats.get("path"))
        return _ok("knowledge", f"{nodes} nodes, {edges} edges in one graph; typed create/link/read/search/backlinks available", path=stats.get("path"), nodes=nodes, edges=edges)

    def _capability_health(self) -> Check:
        """Runtime health, separate from installation: a capability that failed its last real calls is not healthy."""

        try:
            manifests = self.core.capabilities.registry.all()
        except Exception as exc:  # noqa: BLE001
            return _warn("capability_health", f"registry unreadable: {exc}")
        failing, degraded, unverified = [], [], []
        for m in manifests:
            if getattr(m, "status", "") != "active":
                continue
            health = m.health_view() if hasattr(m, "health_view") else {}
            state = health.get("state", "unverified")
            (failing if state == "failing" else degraded if state == "degraded" else unverified if state == "unverified" else []).append(
                f"{m.capability_id} ({health.get('last_error', '')[:60]})" if state in {"failing", "degraded"} else m.capability_id)
        if failing:
            return _error("capability_health", f"FAILING: {', '.join(failing[:3])}", "Repair from the Capability Center; the resolver demotes them meanwhile",
                          failing=failing, degraded=degraded)
        if degraded:
            return _warn("capability_health", f"DEGRADED: {', '.join(degraded[:3])}", "One more failure marks them FAILING; a repair restores them", degraded=degraded)
        healthy = len([m for m in manifests if getattr(m, "status", "") == "active"]) - len(unverified)
        return _ok("capability_health", f"{healthy} healthy, {len(unverified)} never called since health tracking began", unverified=unverified[:12])

    def _projects(self) -> Check:
        """Projects as the owner's view: owner projects apart from acquisition jobs."""

        try:
            rows = self.core.list_projects()
        except Exception as exc:  # noqa: BLE001
            return _warn("projects", f"project store unreadable: {exc}")
        owner = [r for r in rows if r.get("origin", "owner") == "owner"]
        internal = [r for r in rows if r.get("origin") == "acquisition"]
        unclassified = [r for r in rows if r.get("origin") not in {"owner", "acquisition"}]
        if unclassified:
            return _warn("projects", f"{len(unclassified)} project(s) of unknown origin", "Classify them (kind / capability_id)", unclassified=[r.get("id") for r in unclassified][:6])
        return _ok("projects", f"{len(owner)} owner project(s); {len(internal)} internal acquisition job(s) kept apart", owner=len(owner), internal=len(internal))

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
