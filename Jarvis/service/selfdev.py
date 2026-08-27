"""Self-development through ZEUS: "change something about yourself."

    LIVE REQUEST
      -> SelfDevMission (durable, one JSON file per mission)
         UNDERSTAND    acceptance: what must be true, and how it will be checked
         INVESTIGATE   deterministic: the code index names the files that matter
         BUILD         RepositoryEngineer in an isolated git worktree, BUILD_LOCAL
         VERIFY        the acceptance commands, re-run here, plus targeted tests
         ESCALATE      ExpertGateway, only after the local tier has failed, and
                       the expert's work is verified by the same commands
         PROMOTE       Promoter: snapshot, copy the declared files, static
                       health check, commit -- owner-protected paths refused
         RESTARTING    the supervisor restarts ZEUS and runs the live health
                       check; unhealthy means it reverts to known-good
         DONE / FAILED reported into the conversation after the restart

Every phase writes the mission file before it runs and after it finishes, so a
crash or a restart in the middle leaves a record that says exactly where it
stopped.  Nothing here trusts a model's account of what it did: changed files
come from git, verification from exit codes, the restart verdict from the
supervisor's deployment receipt.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from service.events import EventType
from service.isolation import CandidateWorkspace, LiveTreeGuard, MissionCancelled
from service.state import JarvisState

UI_TERMS = (
    "ui", "interface", "oberfläche", "oberflaeche", "auge", "eye", "anzeige", "anzeigen", "zeige", "zeig ",
    "show", "display", "button", "panel", "farbe", "colour", "color", "animation", "seite", "page", "badge",
)

#: Words that carry no signal for the code index.
STOPWORDS = {
    "zeus", "bitte", "please", "the", "and", "und", "der", "die", "das", "ein", "eine", "einen", "mein", "meine",
    "my", "your", "dein", "deine", "deinem", "deinen", "neben", "next", "to", "of", "in", "an", "auf", "mit",
    "with", "für", "for", "dezent", "subtly", "aktuelle", "current", "show", "zeige", "zeig", "change", "ändere",
    "aendere", "about", "yourself", "dich", "selbst", "dir", "so", "dass", "that", "it", "es", "is", "ist",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SelfDevMission:
    request: str
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    scope: str = ""
    language: str = ""
    phase: str = "UNDERSTAND"
    started_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    #: Acceptance: human-readable criteria and the commands that decide them.
    acceptance: list[dict[str, Any]] = field(default_factory=list)
    area: str = "code"  # ui | code
    #: What INVESTIGATE found: files and why.
    investigation: dict[str, Any] = field(default_factory=dict)
    worktree: str = ""
    changed_files: list[str] = field(default_factory=list)
    local_attempts: int = 0
    escalated: bool = False
    expert: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    promotion_id: str = ""
    expected_revision: str = ""
    outcome: str = ""  # promoted | failed | rolled_back
    reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    model_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    #: The top-level routing decision that created this mission (evidence).
    routing: dict[str, Any] = field(default_factory=dict)
    #: Set through the API; the runner stops at the next phase boundary.
    cancel_requested: bool = False
    #: Live-tree guard reports, one per experimental phase (evidence that the
    #: active tree stayed untouched -- or what was put back).
    isolation: list[dict[str, Any]] = field(default_factory=list)
    #: Where the candidate's diff was kept after the worktree was released.
    evidence_patch: str = ""
    #: The goal question, answered separately from execution: {answer, why, kind}.
    verification_goal: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.phase in {"DONE", "FAILED", "CANCELLED"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfDevMission":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class SelfDevStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def path_for(self, mission_id: str) -> Path:
        return self.root / f"{mission_id}.json"

    def save(self, mission: SelfDevMission) -> Path:
        mission.updated_at = _now()
        path = self.path_for(mission.mission_id)
        with self._lock:
            # A cancel arrives from another thread through this file.  Saving
            # an in-memory copy must never un-cancel a mission.
            if not mission.cancel_requested and path.is_file():
                try:
                    if json.loads(path.read_text(encoding="utf-8")).get("cancel_requested"):
                        mission.cancel_requested = True
                except (OSError, ValueError):
                    pass
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(mission.to_dict(), indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        return path

    def load(self, mission_id: str) -> SelfDevMission | None:
        path = self.path_for(mission_id)
        if not path.is_file():
            return None
        try:
            return SelfDevMission.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def list(self) -> list[SelfDevMission]:
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                out.append(SelfDevMission.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue
        return out

    def awaiting_restart(self) -> list[SelfDevMission]:
        return [m for m in self.list() if m.phase == "RESTARTING"]

    def active(self) -> SelfDevMission | None:
        for mission in reversed(self.list()):
            if not mission.finished and mission.phase != "RESTARTING":
                return mission
        return None


class SelfDevRunner:
    """Drives one mission through every phase.  One runner per mission."""

    def __init__(
        self,
        *,
        repository: Path,
        store: SelfDevStore,
        kernel: Any,
        owner: Any,
        lifecycle: Any,
        gateway: Any = None,
        emit: Callable[[EventType, dict[str, Any]], None],
        set_state: Callable[..., Any],
        python: str = "",
    ) -> None:
        self.repository = Path(repository).resolve()
        self.store = store
        self.kernel = kernel
        self.owner = owner
        self.lifecycle = lifecycle
        self.gateway = gateway
        self.emit = emit
        self.set_state = set_state
        self.python = python or sys.executable
        #: The static health check every candidate must pass: ZEUS imports and
        #: its kernel assembles. Replaceable, so a test repository can bring
        #: its own definition of "still works".
        self.health_command: list[str] = [
            self.python, "-c",
            "from core.kernel import JarvisKernel; k = JarvisKernel(); assert k.tools.names(); print('JARVIS_HEALTH_OK')",
        ]
        self.health_marker = "JARVIS_HEALTH_OK"
        #: The isolation boundary.  Created at BUILD, released at the end of
        #: the mission whichever way it ends; see :mod:`service.isolation`.
        self._workspace: Any = None
        self._engineer: Any = None
        self._guard = LiveTreeGuard(self.repository)
        self._baseline: Any = None
        self.evidence_root = Path(self.store.root) / "evidence"
        #: Verified experience from earlier missions: read before BUILD, written after the verdict.
        from development.experience import ExperienceStore

        self.experience = ExperienceStore(Path(self.store.root).parent / "experience" / "selfdev.jsonl")

    # -- isolation -----------------------------------------------------

    def _check_cancel(self, mission: SelfDevMission) -> None:
        """The owner's cancel arrives through the store, from another thread."""

        latest = self.store.load(mission.mission_id)
        if (latest is not None and latest.cancel_requested) or mission.cancel_requested:
            mission.cancel_requested = True
            if self._engineer is not None:
                # The engineer's loop checks its deadline before every model
                # call; an expired budget is the one exit it already has.
                try:
                    self._engineer.deadline.budget = 0.0
                except Exception:  # noqa: BLE001
                    pass
            raise MissionCancelled(f"mission {mission.mission_id} cancelled by the owner")

    def _audit(self, mission: SelfDevMission, phase: str) -> None:
        """After foreign code ran: prove the live tree is what it was.

        Contamination -- a live file byte-identical to the candidate's copy --
        is restored and the mission fails, because a boundary that was
        crossed is a defect in the boundary, not a detail of the mission.
        """

        if self._baseline is None:
            return
        report = self._guard.check(self._baseline, self._workspace, phase=phase)
        mission.isolation.append(report)
        self.store.save(mission)
        if report["contamination"]:
            self.emit(EventType.ERROR, {"error": f"self-development {phase} wrote into the live tree: "
                                                 f"{report['contamination'][:6]}; restored {report['restored'][:6]}",
                                        "kind": "selfdev", "mission_id": mission.mission_id, "isolation": report})
            raise RuntimeError(f"isolation breach in {phase}: {report['contamination'][:6]} (restored: {report['restored'][:6]})")

    def _finish(self, mission: SelfDevMission) -> None:
        """Release the candidate, keeping its diff; the live tree is never involved."""

        self._remember(mission)
        ws = self._workspace
        if ws is None:
            return
        # A verified candidate whose promotion failed is worth keeping: a
        # resume re-verifies and promotes it in seconds instead of rebuilding
        # it in half an hour.  Everything else is released, diff first.
        keep = bool(mission.verification.get("ok")) and mission.outcome == "failed" and mission.phase == "FAILED"
        if keep:
            return
        try:
            report = ws.release(evidence_root=self.evidence_root, reason=f"{mission.phase}: {mission.reason[:120]}")
            if report.get("evidence"):
                mission.evidence_patch = report["evidence"]
            mission.events.append({"at": _now(), "phase": "RELEASE", "detail": f"worktree removed={report.get('removed')}, evidence={mission.evidence_patch}"})
            if mission.phase != "RESTARTING":
                mission.worktree = ""
            self.store.save(mission)
        except Exception as exc:  # noqa: BLE001 - releasing must never mask the verdict
            mission.events.append({"at": _now(), "phase": "RELEASE", "detail": f"release failed: {exc}"[:300]})
            self.store.save(mission)
        finally:
            self._workspace = None
            self._engineer = None

    def _remember(self, mission: SelfDevMission) -> None:
        """What this mission learned, as one compact verified-experience entry."""

        if mission.phase not in {"DONE", "FAILED", "CANCELLED", "RESTARTING"} or getattr(mission, "_remembered", False):
            return
        try:
            from development.experience import from_selfdev_mission

            entry = from_selfdev_mission(mission)
            if mission.phase == "RESTARTING":
                entry.outcome = "promoted"
            self.experience.add(entry)
            mission.events.append({"at": _now(), "phase": "EXPERIENCE", "detail": f"kept as {entry.experience_id}"})
            setattr(mission, "_remembered", True)
        except Exception as exc:  # noqa: BLE001
            mission.events.append({"at": _now(), "phase": "EXPERIENCE", "detail": f"not kept: {exc}"[:200]})

    # -- bookkeeping ---------------------------------------------------

    def _phase(self, mission: SelfDevMission, phase: str, detail: str = "") -> None:
        if phase not in {"FAILED", "CANCELLED", "DONE", "RELEASE"}:
            self._check_cancel(mission)
        mission.phase = phase
        mission.events.append({"at": _now(), "phase": phase, "detail": detail[:400]})
        self.store.save(mission)
        self.emit(EventType.PROGRESS, {"summary": f"selfdev {phase}: {detail}"[:300], "kind": "selfdev",
                                       "mission_id": mission.mission_id, "phase": phase})

    def _cancel(self, mission: SelfDevMission, reason: str) -> SelfDevMission:
        mission.outcome = "cancelled"
        mission.reason = reason[:1000]
        self._phase(mission, "CANCELLED", reason)
        self.emit(EventType.NOTIFICATION, {"text": f"self-development {mission.mission_id} cancelled", "kind": "selfdev",
                                           "mission_id": mission.mission_id})
        return mission

    def _fail(self, mission: SelfDevMission, reason: str) -> SelfDevMission:
        mission.outcome = "failed"
        mission.reason = reason[:1000]
        self._phase(mission, "FAILED", reason)
        return mission

    def _timed(self, mission: SelfDevMission, name: str, fn: Callable[[], Any]) -> Any:
        started = time.monotonic()
        try:
            return fn()
        finally:
            mission.timings[name] = round(mission.timings.get(name, 0.0) + time.monotonic() - started, 1)

    # -- the run -------------------------------------------------------

    def run(self, mission: SelfDevMission) -> SelfDevMission:
        policy = self.owner.read("policy").get("self_development", {})
        if not policy.get("enabled", True):
            return self._fail(mission, "self-development is disabled by the owner policy")
        max_seconds = float(policy.get("max_seconds", 2400))

        try:
            self._baseline = self._guard.fingerprint()
            self._timed(mission, "understand", lambda: self._understand(mission))
            self._timed(mission, "investigate", lambda: self._investigate(mission))
            self.set_state(JarvisState.CODING, detail="developing a change to myself", scope=mission.scope)
            candidate = self._timed(mission, "build", lambda: self._build(mission, max_seconds))
            self._audit(mission, "BUILD")
            self.set_state(JarvisState.VERIFYING, detail="verifying the candidate", scope=mission.scope)
            verified = self._timed(mission, "verify", lambda: self._verify(mission))
            self._audit(mission, "VERIFY")
            if not verified and self.gateway is not None:
                self.set_state(JarvisState.CODING, detail="asking an expert", scope=mission.scope)
                self._timed(mission, "escalate", lambda: self._escalate(mission))
                self._audit(mission, "ESCALATE")
                self.set_state(JarvisState.VERIFYING, detail="verifying the expert's work", scope=mission.scope)
                verified = self._timed(mission, "verify", lambda: self._verify(mission))
                self._audit(mission, "VERIFY")
            if not verified:
                return self._fail(mission, f"no verified candidate: {mission.verification.get('detail', '')[:400]}")
            if not policy.get("auto_promote", True):
                mission.outcome = "verified_not_promoted"
                self._phase(mission, "DONE", "verified; the owner policy does not promote automatically")
                return mission
            self._timed(mission, "promote", lambda: self._promote(mission))
            if mission.outcome == "failed":
                return mission
            self._restart(mission)
            return mission
        except MissionCancelled as exc:
            return self._cancel(mission, str(exc))
        except Exception as exc:  # noqa: BLE001 - the mission file must say what raised
            import traceback

            return self._fail(mission, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
        finally:
            self._finish(mission)
            if mission.phase not in {"RESTARTING"}:
                self.set_state(JarvisState.IDLE, scope=mission.scope)

    def resume(self, mission: SelfDevMission) -> SelfDevMission:
        """Continue a mission whose candidate still exists on disk.

        A failure in PROMOTE or RESTARTING does not invalidate a verified
        worktree; re-verifying it and promoting costs seconds, rebuilding it
        costs the half hour that was already spent.  Anything earlier than a
        candidate starts over.
        """

        if mission.worktree and Path(mission.worktree).is_dir():
            self._workspace = CandidateWorkspace.attach(self.repository, mission.mission_id, mission.worktree)
            mission.changed_files = self._changed_files(mission.worktree)
        if not mission.changed_files:
            mission.phase, mission.outcome, mission.reason = "UNDERSTAND", "", ""
            mission.worktree = ""
            self._workspace = None
            return self.run(mission)
        policy = self.owner.read("policy").get("self_development", {})
        mission.outcome, mission.reason, mission.cancel_requested = "", "", False
        self._baseline = self._guard.fingerprint()
        if not mission.acceptance:
            self._understand(mission)
        self._phase(mission, "RESUME", f"resuming with the existing candidate ({len(mission.changed_files)} files)")
        try:
            self.set_state(JarvisState.VERIFYING, detail="re-verifying the candidate", scope=mission.scope)
            if not self._timed(mission, "verify", lambda: self._verify(mission)):
                return self._fail(mission, f"the candidate no longer verifies: {mission.verification.get('detail', '')[:400]}")
            if not policy.get("auto_promote", True):
                mission.outcome = "verified_not_promoted"
                self._phase(mission, "DONE", "verified; the owner policy does not promote automatically")
                return mission
            self._timed(mission, "promote", lambda: self._promote(mission))
            if mission.outcome == "failed":
                return mission
            self._restart(mission)
            return mission
        except MissionCancelled as exc:
            return self._cancel(mission, str(exc))
        except Exception as exc:  # noqa: BLE001
            import traceback

            return self._fail(mission, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
        finally:
            self._finish(mission)
            if mission.phase != "RESTARTING":
                self.set_state(JarvisState.IDLE, scope=mission.scope)

    # -- phases --------------------------------------------------------

    def _understand(self, mission: SelfDevMission) -> None:
        text = mission.request.lower()
        mission.area = "ui" if any(term in text for term in UI_TERMS) else "code"
        acceptance = [
            {"criterion": "ZEUS still imports and its kernel still assembles",
             "command": list(self.health_command), "expect": self.health_marker},
        ]
        if mission.area == "ui":
            acceptance.append({"criterion": "the interface is served whole and its parts are still there",
                               "command": [self.python, "-m", "jarvis.verify_ui"], "expect": "UI_OK"})
        mission.acceptance = acceptance
        self._phase(mission, "UNDERSTAND", f"area={mission.area}, {len(acceptance)} acceptance checks")

    def _investigate(self, mission: SelfDevMission) -> None:
        """Deterministic: which files does this request touch?  No model."""

        from development.code_index import CodeIndex

        terms = [w for w in re.findall(r"[a-zA-ZäöüÄÖÜß_]{3,}", mission.request.lower()) if w not in STOPWORDS]
        files: dict[str, int] = {}
        index = CodeIndex(self.repository)
        for term in terms[:12]:
            try:
                for hit in index.find_literal(term, limit=20):
                    rel = index.relative(Path(hit.path)) if hasattr(hit, "path") else str(hit)
                    # Promotion snapshots and other runtime copies under data/
                    # are not the code; the second live mission ranked two of
                    # them above ui/index.html.
                    if rel.replace("\\", "/").startswith(("data/", ".venv", ".pytest")):
                        continue
                    files[rel] = files.get(rel, 0) + 1
            except Exception:
                continue
        # The UI is not Python; the index does not see it, so name it directly.
        if mission.area == "ui":
            for name in ("ui/index.html", "ui/app.js", "ui/eye.js", "service/http.py", "service/core.py"):
                files[name] = files.get(name, 0) + 3
        ranked = sorted(files.items(), key=lambda kv: -kv[1])[:8]
        mission.investigation = {"terms": terms[:12], "files": [f for f, _ in ranked]}
        self._phase(mission, "INVESTIGATE", f"{len(terms)} terms, {len(ranked)} candidate files")

    def _goal_text(self, mission: SelfDevMission) -> str:
        from owner.protected import PROTECTED_PATHS

        files = mission.investigation.get("files", [])
        lines = [
            mission.request.strip(),
            "",
            "This is a change to ZEUS itself -- the program you are running inside. Edit the repository "
            "so the request is satisfied. Keep every existing behaviour working.",
            "Files most likely relevant (from a deterministic index; read them first): " + ", ".join(files)
            if files else "",
            "The web interface has no build step: ui/index.html loads ui/eye.js and ui/app.js directly, "
            "the server is service/http.py (stdlib, JSON API under /api/ plus an SSE stream at /events) "
            "and the state/values it serves come from service/core.py." if mission.area == "ui" else "",
            "Do not remove any element id the client looks up and leave the __JARVIS_TOKEN__ placeholder in place."
            if mission.area == "ui" else "",
            "Never touch these owner-protected paths: " + ", ".join(PROTECTED_PATHS) + ".",
            "Do not add dependencies. Do not print secrets. Small, targeted edits.",
        ]
        try:
            guidance = self.experience.guidance(mission.request, subsystem=mission.area)
        except Exception:  # noqa: BLE001 - experience is help, never a blocker
            guidance = ""
        if guidance:
            lines.append("")
            lines.append(guidance)
            mission.events.append({"at": _now(), "phase": "EXPERIENCE", "detail": f"{guidance.count(chr(10))} lines of verified experience retrieved"})
        return "\n".join(line for line in lines if line)

    def _build(self, mission: SelfDevMission, max_seconds: float) -> Any:
        from brain.tiers import ModelTier
        from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal

        from development.repository_engineer import ModelRequestBudget

        brain = self.kernel.provider(ModelTier.BUILD_LOCAL)
        # The prompt budget must match the window the provider was configured
        # with; left at its 8192 default it compacts prompts the model could
        # have read in full (the tuner measured 24576 here).
        window = getattr(getattr(brain, "spec", None), "context_window", None)
        # The boundary first.  The candidate is a git worktree outside the
        # repository; the engineer is handed it and never chooses a path.
        workspace = CandidateWorkspace(repository=self.repository, mission_id=mission.mission_id).create()
        self._workspace = workspace
        mission.worktree = str(workspace.root)
        engineer = RepositoryEngineer(
            brain=brain, timeout_seconds=600.0, max_cycles=3, max_seconds=max_seconds,
            worktree_root=workspace.base / mission.mission_id,
            context_budget=ModelRequestBudget.from_env(window) if window else None,
        )
        self._engineer = engineer
        goal = SelfImprovementGoal(objective=self._goal_text(mission), allowed_paths=["."])
        commands = [item["command"] for item in mission.acceptance]
        mission.local_attempts += 1
        self._phase(mission, "BUILD", f"BUILD_LOCAL, attempt {mission.local_attempts}, budget {max_seconds:.0f}s, candidate {workspace.root}")
        candidate = engineer.improve(self.repository, goal, acceptance_commands=commands, max_cycles=3,
                                     worktree=workspace.root)
        mission.changed_files = self._changed_files(mission.worktree)
        trajectory = getattr(engineer, "_active_trajectory", {}) or {}
        mission.model_calls += sum(1 for e in trajectory.get("events", []) if "model" in str(e.get("stage", "")).lower()
                                   or str(e.get("kind", "")) in {"generation", "model_call"})
        status = str(getattr(candidate, "status", ""))
        mission.events.append({"at": _now(), "phase": "BUILD", "detail": f"{status}; {len(mission.changed_files)} files: {mission.changed_files[:6]}",
                               "error": str(getattr(candidate, "error", ""))[:300], "cycles": getattr(candidate, "cycles", 0)})
        self.store.save(mission)
        return candidate

    def _changed_files(self, worktree: str) -> list[str]:
        if not worktree or not Path(worktree).is_dir():
            return []
        try:
            out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=worktree, capture_output=True, text=True, timeout=60).stdout
            top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=worktree, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return []
        files = []
        root = Path(worktree)
        # git reports paths from the top of the worktree; ZEUS may live in a
        # subdirectory of it, and every consumer wants paths relative to ZEUS.
        prefix = ""
        try:
            rel = root.resolve().relative_to(Path(top).resolve()).as_posix() if top else "."
            prefix = "" if rel == "." else rel + "/"
        except ValueError:
            prefix = ""
        for line in out.splitlines():
            if len(line) > 3:
                name = line[3:].strip().strip('"')
                if " -> " in name:
                    name = name.split(" -> ", 1)[1]
                name = name.replace("\\", "/")
                if prefix:
                    if not name.startswith(prefix):
                        continue
                    name = name[len(prefix):]
                # A candidate worktree has no .gitignore at its root, so
                # bytecode caches and runtime state show up as untracked.
                # None of that is a change; only files are promoted.
                if name.endswith("/") or "__pycache__" in name or name.endswith(".pyc"):
                    continue
                if name.startswith(("data/", ".pytest_cache/", ".pytest_tmp/", ".agent_tmp/", ".venv")):
                    continue
                if (root / name).is_dir():
                    continue
                files.append(name)
        return files

    def _env_for(self, cwd: str) -> dict[str, str]:
        """Inside the candidate: the scrubbed workspace environment, with every
        ZEUS root pointed into the candidate.  On the live tree (the promoter's
        re-run): the process environment, with the tree first on the path."""

        ws = self._workspace
        try:
            inside = ws is not None and Path(cwd).resolve().is_relative_to(ws.path)
        except (OSError, ValueError):
            inside = False
        if inside:
            return ws.env()
        env = dict(os.environ)
        for name in ("JARVIS_STATE_ROOT", "JARVIS_CONFIG_ROOT", "ZEUS_CANDIDATE", "ZEUS_CANDIDATE_ROOT"):
            env.pop(name, None)
        env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _run(self, command: list[str], cwd: str, timeout: float = 600.0) -> tuple[bool, str]:
        env = self._env_for(cwd)
        try:
            completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", timeout=timeout, env=env,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout:.0f}s"
        except OSError as exc:
            return False, str(exc)
        output = f"{completed.stdout}\n{completed.stderr}"
        return completed.returncode == 0, output[-3000:]

    def _targeted_tests(self, mission: SelfDevMission) -> list[str]:
        """Test files that mention a module the candidate changed."""

        stems = {Path(f).stem for f in mission.changed_files if f.endswith(".py") and not f.startswith("tests/")}
        chosen = {f for f in mission.changed_files if f.startswith("tests/") and f.endswith(".py")}
        tests_dir = self.repository / "tests"
        if stems and tests_dir.is_dir():
            for path in tests_dir.glob("test_*.py"):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if any(re.search(rf"\b{re.escape(stem)}\b", text) for stem in stems):
                    chosen.add(f"tests/{path.name}")
        return sorted(chosen)[:6]

    def _verify(self, mission: SelfDevMission) -> bool:
        """Independent: acceptance commands and targeted tests, re-run here."""

        if not mission.worktree or not Path(mission.worktree).is_dir():
            mission.verification = {"ok": False, "detail": "no candidate worktree"}
            self._phase(mission, "VERIFY", "no candidate")
            return False
        if not mission.changed_files:
            mission.verification = {"ok": False, "detail": "the candidate changed nothing"}
            self._phase(mission, "VERIFY", "no changes in the candidate")
            return False
        from owner.protected import protected_violations

        violations = protected_violations(mission.changed_files)
        if violations:
            mission.verification = {"ok": False, "detail": f"candidate touched owner-protected paths: {violations}"}
            self._phase(mission, "VERIFY", mission.verification["detail"])
            return False

        checks: list[dict[str, Any]] = []
        ok_all = True
        for item in mission.acceptance:
            ok, output = self._run(list(item["command"]), mission.worktree)
            if ok and item.get("expect") and item["expect"] not in output:
                ok = False
                output = f"expected {item['expect']!r} in the output\n" + output
            checks.append({"criterion": item["criterion"], "ok": ok, "output": output[-800:]})
            ok_all &= ok
        tests = self._targeted_tests(mission)
        if tests:
            ok, output = self._run([self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], mission.worktree, timeout=900)
            checks.append({"criterion": f"targeted tests: {', '.join(tests)}", "ok": ok, "output": output[-800:]})
            ok_all &= ok

        # EXECUTION_VERIFIED is not GOAL_SATISFIED.  Live mission f16dc926c6
        # passed every check above with a candidate that added one empty
        # <div> and no behaviour: the interface still served, the kernel still
        # imported, and the request was not implemented.  Two goal checks,
        # both stated as what they are: a deterministic shape check (data the
        # request names needs logic, not markup), and a model's reading of
        # the diff -- recorded as inference, gating escalation, never promotion
        # on its own.
        shape_ok, shape_detail = self._goal_shape(mission)
        checks.append({"criterion": "the change has the shape the request needs", "ok": shape_ok, "output": shape_detail})
        ok_all &= shape_ok
        inferred = self._goal_inference(mission)
        mission.verification_goal = inferred
        if inferred.get("answer") == "no":
            checks.append({"criterion": "a reader of the diff sees the request implemented (model inference)", "ok": False,
                           "output": inferred.get("why", "")[:400]})
            ok_all = False
        mission.verification = {"ok": ok_all, "checks": checks, "tests": tests,
                                "detail": "; ".join(f"{c['criterion']}: {'ok' if c['ok'] else 'FAILED'}" for c in checks)}
        self._phase(mission, "VERIFY", mission.verification["detail"])
        return ok_all

    def _goal_shape(self, mission: SelfDevMission) -> tuple[bool, str]:
        """Deterministic: a request that names data or behaviour cannot be met by markup alone."""

        files = [f.replace("\\", "/") for f in mission.changed_files]
        text = mission.request.lower()
        names_data = any(w in text for w in ("/api/", "anzahl", "zahl", "wert", "count", "number", "aus /", "from /", "sekunden", "seconds",
                                               "status", "aktuell", "current", "live", "wenn ", "when ", "sobald", "nur wenn"))
        logic = [f for f in files if f.endswith((".js", ".py", ".ts"))]
        markup = [f for f in files if f.endswith((".html", ".css"))]
        if mission.area == "ui" and names_data and markup and not logic:
            return False, f"markup only ({', '.join(markup)}): the request names data or behaviour, which needs ui/app.js or a view module"
        if not files:
            return False, "no files changed"
        return True, f"{len(logic)} logic file(s), {len(markup)} markup file(s)"

    def _goal_inference(self, mission: SelfDevMission) -> dict[str, Any]:
        """FAST_LOCAL reads the diff against the request.  Inference, labelled as such."""

        try:
            from brain.tiers import ModelTier
            from service.isolation import CandidateWorkspace

            ws = self._workspace or CandidateWorkspace.attach(self.repository, mission.mission_id, mission.worktree)
            diff = ws.diff()[:6000]
            if not diff.strip():
                return {"answer": "no", "why": "empty diff", "kind": "model_inference"}
            provider = self.kernel.provider(ModelTier.FAST_LOCAL)
            prompt = ("You review a code change against a request. Answer with one JSON object: "
                      '{"implemented": "yes"|"partly"|"no", "why": "<one sentence>"}.\n\n'
                      f"Request:\n{mission.request[:800]}\n\nDiff:\n{diff}\n\nJSON:")
            try:
                raw = provider.generate(prompt, max_tokens=200, temperature=0.0)
            except TypeError:
                raw = provider.generate(prompt)
            from brain.json_utils import lenient_json_loads

            data = lenient_json_loads(str(raw))
            answer = str((data or {}).get("implemented", "")).lower().strip() if isinstance(data, dict) else ""
            return {"answer": "yes" if answer.startswith("y") else "no" if answer.startswith("n") else "partly" if answer else "unknown",
                    "why": str((data or {}).get("why", ""))[:300] if isinstance(data, dict) else str(raw)[:200], "kind": "model_inference"}
        except Exception as exc:  # noqa: BLE001
            return {"answer": "unknown", "why": f"{type(exc).__name__}: {exc}"[:200], "kind": "model_inference"}

    def _escalate(self, mission: SelfDevMission) -> None:
        from experts.contracts import ExpertJob
        from owner.protected import PROTECTED_PATHS

        mission.escalated = True
        try:
            status = self.gateway.status()
        except Exception as exc:
            status = {"error": str(exc)}
        if not status.get("expert_available"):
            mission.expert = {"status": "unavailable", "gateway": status}
            self._phase(mission, "ESCALATE", f"expert unavailable: {status}")
            return
        failures = [str(e.get("error") or e.get("detail", "")) for e in mission.events if e.get("phase") == "BUILD"]
        failures.append(str(mission.verification.get("detail", "")))
        job = ExpertJob(
            goal=self._goal_text(mission),
            workspace=Path(mission.worktree),
            constraints=[f"never modify: {', '.join(PROTECTED_PATHS)}", "no new dependencies", "small targeted edits",
                         "the change must be complete and working in this worktree; ZEUS verifies with the commands below"],
            acceptance=[(item["criterion"], list(item["command"])) for item in mission.acceptance],
            previous_failures=[f for f in failures if f][-4:],
            max_seconds=1500.0,
        )
        self._phase(mission, "ESCALATE", f"submitting to {status.get('provider', 'expert')}")
        started = time.monotonic()
        result = self.gateway.submit(job)
        mission.changed_files = self._changed_files(mission.worktree)
        mission.expert = {
            "status": str(getattr(getattr(result, "status", None), "value", getattr(result, "status", ""))),
            "verified": bool(getattr(result, "verified", False)),
            "seconds": round(time.monotonic() - started, 1),
            "provider": str(getattr(result, "provider", "")),
            "summary": str(getattr(result, "summary", ""))[:500],
        }
        self._phase(mission, "ESCALATE", f"expert {mission.expert['status']} in {mission.expert['seconds']}s; "
                                         f"{len(mission.changed_files)} changed files")

    def _promote(self, mission: SelfDevMission) -> None:
        from deployment.promotion import HealthCheck, Promoter

        acceptance = mission.acceptance[0]
        health = HealthCheck(command=list(acceptance["command"]), timeout_seconds=300.0, expect_output=str(acceptance.get("expect", "")))
        promoter = Promoter(repository=self.repository)
        self._phase(mission, "PROMOTE", f"{len(mission.changed_files)} files into the live tree")

        def verify(repo: Path) -> tuple[bool, str]:
            for item in mission.acceptance:
                ok, output = self._run(list(item["command"]), str(repo))
                if not ok or (item.get("expect") and item["expect"] not in output):
                    return False, f"{item['criterion']}: {output[-600:]}"
            return True, "acceptance re-run on the live tree"

        record = promoter.promote(
            mission.worktree, changed_files=mission.changed_files, health_check=health, verify=verify,
            commit_message=f"ZEUS self-development: {mission.request[:64]}\n\nMission {mission.mission_id}. "
                           f"Developed in an isolated worktree, verified by {len(mission.acceptance)} acceptance "
                           f"check(s){' and targeted tests' if mission.verification.get('tests') else ''}, "
                           f"{'with' if mission.escalated else 'without'} expert help.",
        )
        mission.promotion = record.to_dict()
        mission.promotion_id = record.promotion_id
        mission.expected_revision = record.promoted_revision
        if not record.success:
            self._fail(mission, f"promotion {record.outcome.value}: {record.reason[:400]}")
            return
        self.store.save(mission)

    def _restart(self, mission: SelfDevMission) -> None:
        if not self.lifecycle.supervised:
            mission.outcome = "promoted"
            self._phase(mission, "DONE", "promoted and committed; not under the supervisor, so no restart was performed")
            return
        self._phase(mission, "RESTARTING", f"asking the supervisor to restart at {mission.expected_revision[:12]}")
        self.lifecycle.request_restart(
            f"self-development {mission.mission_id}: {mission.request[:80]}",
            expected_revision=mission.expected_revision, promotion_id=mission.promotion_id, requested_by="selfdev",
        )


def settle_after_restart(store: SelfDevStore, lifecycle: Any) -> list[SelfDevMission]:
    """After a restart: which missions were waiting, and what the supervisor
    decided.  The verdict is the deployment receipt, never the mission's own
    expectation."""

    settled = []
    receipts = lifecycle.supervisor_status().get("deployments", [])
    for mission in store.awaiting_restart():
        verdict = next((r for r in reversed(receipts) if r.get("promotion_id") == mission.promotion_id), None)
        if verdict is None:
            # Restarted for some other reason, or the receipt is not there yet.
            if lifecycle.revision() == mission.expected_revision:
                mission.outcome = "promoted"
                mission.phase = "DONE"
                mission.reason = "running at the promoted revision"
            else:
                mission.outcome = "rolled_back"
                mission.phase = "FAILED"
                mission.reason = f"running at {lifecycle.revision()[:12]}, not the promoted {mission.expected_revision[:12]}"
        elif verdict.get("outcome") == "healthy":
            mission.outcome = "promoted"
            mission.phase = "DONE"
            mission.reason = f"restart verified in {verdict.get('duration_seconds', '?')}s"
        else:
            mission.outcome = "rolled_back"
            mission.phase = "FAILED"
            mission.reason = f"{verdict.get('outcome')}: {verdict.get('reason', '')}"[:600]
        mission.events.append({"at": _now(), "phase": mission.phase, "detail": mission.reason})
        store.save(mission)
        settled.append(mission)
    return settled


def describe(mission: SelfDevMission, language: str = "") -> str:
    """The sentence the owner reads.  German or English, from the record."""

    de = language.startswith("de")
    files = ", ".join(mission.changed_files[:5]) + (" …" if len(mission.changed_files) > 5 else "")
    total = sum(mission.timings.values())
    if mission.outcome == "promoted":
        if de:
            return (f"Selbst-Update abgeschlossen: „{mission.request[:80]}“. Geändert: {files}. "
                    f"Verifiziert, übernommen als {mission.expected_revision[:10]}, Neustart bestätigt "
                    f"({mission.reason}). {total:.0f}s, {'mit' if mission.escalated else 'ohne'} Experten.")
        return (f"Self-update done: “{mission.request[:80]}”. Changed: {files}. Verified, promoted as "
                f"{mission.expected_revision[:10]}, restart confirmed ({mission.reason}). {total:.0f}s, "
                f"{'with' if mission.escalated else 'without'} an expert.")
    if mission.outcome == "cancelled":
        if de:
            return (f"Selbst-Update abgebrochen: „{mission.request[:80]}“. Nichts wurde übernommen; der Arbeitsbaum ist "
                    f"entfernt{' (Diff aufbewahrt)' if mission.evidence_patch else ''}. Aktiver Code unverändert.")
        return (f"Self-update cancelled: “{mission.request[:80]}”. Nothing was promoted; the candidate is "
                f"released{' (diff kept)' if mission.evidence_patch else ''}. Live code unchanged.")
    if mission.outcome == "rolled_back":
        if de:
            return (f"Selbst-Update zurückgerollt: „{mission.request[:80]}“ hat den Neustart nicht überstanden "
                    f"({mission.reason}). Ich laufe wieder auf der letzten bekannt-guten Version.")
        return (f"Self-update rolled back: “{mission.request[:80]}” did not survive the restart ({mission.reason}). "
                f"I am running the last known-good revision again.")
    if de:
        return f"Selbst-Update fehlgeschlagen: „{mission.request[:80]}“ — {mission.reason[:300]}"
    return f"Self-update failed: “{mission.request[:80]}” — {mission.reason[:300]}"
