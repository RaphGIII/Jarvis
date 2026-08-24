"""Capabilities: things Jarvis can actually do, acquired when it cannot.

A capability is not knowledge in a prompt.  It is an installed, executable,
tested unit with a stable identifier, and the difference matters: knowing *about*
playing music is not the same as being able to play a file when asked, twice, a
week apart, after a reboot.

The lifecycle is:

    ask -> is it installed?  -> yes: execute it
                             -> no: acquire it, verify it, register it, execute it

Acquisition runs on the ordinary :class:`~projects.engine.ProjectEngine` rather
than on a pipeline of its own.  That is deliberate.  Acquiring a capability is
investigating an environment, deciding an approach, writing code, testing it and
repairing it -- which is exactly what the project loop already does, including
all of its recovery behaviour.  A second implementation would be a second set of
bugs.

Two contracts make the difference between capabilities that can be tested and
capabilities that can only be hoped about:

``run(payload) -> dict``
    Every capability is a Python module exposing this one function.  Uniform,
    trivially callable, and serialisable across a future process boundary.

``payload["dry_run"]``
    A capability with an external effect -- playing audio, opening a window,
    sending something -- must support a dry run that performs every check and
    reports what it *would* do without doing it.  Without this, a side-effecting
    capability cannot be verified at all, and "verified" would come to mean
    "the code imported successfully".  With it, acquisition can prove the player
    was found, the file exists, and the command is well-formed, and be honest
    that the sound itself was not observed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capabilities.models import CapabilityManifest
from capabilities.registry import CapabilityRegistry
from knowledge.graph import KnowledgeGraph, NodeType
from knowledge.memory import ExperienceMemory
from projects.engine import ProjectEngine
from projects.models import Project, ResourceLimits, StopReason


@dataclass
class CapabilityOutcome:
    """The result of asking for a capability."""

    goal: str
    capability_id: str = ""
    #: "available" (already installed), "acquired", or "failed".
    status: str = "failed"
    acquired: bool = False
    manifest: CapabilityManifest | None = None
    project_id: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status in {"available", "acquired"} and bool(self.capability_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest"] = self.manifest.to_dict() if self.manifest else None
        data["usable"] = self.usable
        return data


@dataclass
class ExecutionOutcome:
    capability_id: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Marker carried by the seeded skeleton.  Verification refuses to register a
#: capability while it is still present.
#:
#: This exists because of a real failure: the skeleton was seeded to help the
#: model, and the skeleton passed its own verification. run() returned a dict,
#: the template test asserted only that "ok" was in the result, and the stub
#: satisfied both -- so a capability that did nothing at all was registered as
#: verified. A scaffold must never be able to certify itself.
NOT_IMPLEMENTED = "JARVIS_CAPABILITY_NOT_IMPLEMENTED"

#: Written into every capability workspace.  Giving the model a working
#: skeleton to edit is far more reliable than asking it to produce the shape
#: from a description -- and it pins the run()/dry_run contract in code rather
#: than in prose the model may skim.
#: The seeded skeleton, deliberately tiny.
#:
#: An earlier version carried the full contract in its docstring, which made it
#: 32 lines. Replacing it with a 15-line implementation then looked like
#: file truncation to the edit engine's shrink guard, and the write was refused
#: -- one safety mechanism blocking another. The rules live in the project's
#: constraints, where the model reads them anyway; the skeleton only has to pin
#: the shape.
_TEMPLATE_MAIN = '''"""Capability implementation. Replace the body of run()."""

from __future__ import annotations

from typing import Any

# Every payload key run() accepts. A caller cannot pass what is not declared.
INPUT_SCHEMA = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": False, "error": "JARVIS_CAPABILITY_NOT_IMPLEMENTED"}
'''

_TEMPLATE_TEST = '''"""Tests for this capability. Replace these with real ones."""

import main


def test_placeholder():
    raise AssertionError("JARVIS_CAPABILITY_NOT_IMPLEMENTED: write real tests")
'''


class CapabilityService:
    """Resolves, acquires, registers and executes capabilities."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        engine: ProjectEngine,
        graph: KnowledgeGraph | None = None,
        root: str | Path | None = None,
        execution_timeout: float = 120.0,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.graph = graph
        self.memory = ExperienceMemory(graph) if graph is not None else None
        self.root = Path(root or (Path(registry.path).parent / "installed"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_timeout = execution_timeout

    # ------------------------------------------------------------------
    # Resolving
    # ------------------------------------------------------------------

    def resolve(self, goal: str) -> CapabilityManifest | None:
        """Find an installed capability that can satisfy ``goal``.

        The registry's own term matching comes first, then the knowledge graph,
        which knows the vocabulary a capability declared for itself -- that is
        what lets "play some music" reach ``audio.play_file``.
        """

        matches = self.registry.find(goal, limit=1)
        if matches:
            return matches[0]

        if self.memory is not None:
            for node in self.memory.known_capabilities(goal, limit=3):
                manifest = self.registry.get(node.title)
                if manifest is not None and manifest.status == "active":
                    return manifest
        return None

    def has(self, capability_id: str) -> bool:
        return self.registry.has(capability_id)

    def list(self) -> list[CapabilityManifest]:
        return [item for item in self.registry.all() if item.status == "active"]

    # ------------------------------------------------------------------
    # Acquiring
    # ------------------------------------------------------------------

    def ensure(self, goal: str, *, max_steps: int = 40, keywords: list[str] | None = None) -> CapabilityOutcome:
        """Return a capability for ``goal``, acquiring one if none exists."""

        existing = self.resolve(goal)
        if existing is not None:
            return CapabilityOutcome(
                goal=goal,
                capability_id=existing.capability_id,
                status="available",
                manifest=existing,
                reason="an installed capability already covers this",
            )
        return self.acquire(goal, max_steps=max_steps, keywords=keywords)

    def acquire(self, goal: str, *, max_steps: int = 40, keywords: list[str] | None = None) -> CapabilityOutcome:
        """Build, verify and register a new capability."""

        capability_id = self.suggest_id(goal)
        project = self._start_project(goal, capability_id, max_steps=max_steps)
        session = self.engine.run(project, max_steps=max_steps)

        workspace = Path(project.workspace)
        verification = self._verify(workspace)

        if not (session.accepted and verification["ok"]):
            reason = (
                f"acquisition did not verify ({session.stop_reason.value}): "
                f"{session.message or verification.get('detail', '')}"
            )
            self._remember_attempt(project, capability_id, succeeded=False, reason=reason)
            return CapabilityOutcome(
                goal=goal,
                capability_id="",
                status="failed",
                project_id=project.id,
                verification=verification,
                reason=reason,
            )

        manifest = self._install(capability_id, goal, workspace, verification, keywords=keywords)
        self._remember_attempt(project, capability_id, succeeded=True, reason="verified and registered")
        return CapabilityOutcome(
            goal=goal,
            capability_id=manifest.capability_id,
            status="acquired",
            acquired=True,
            manifest=manifest,
            project_id=project.id,
            verification=verification,
            reason="implemented, verified by its own tests, and registered",
        )

    def _start_project(self, goal: str, capability_id: str, *, max_steps: int) -> Project:
        project = self.engine.create_project(
            f"Build a reusable capability that can: {goal}",
            kind="capability",
            title=capability_id,
            limits=ResourceLimits(max_steps=max_steps, max_seconds=3600, max_consecutive_failures=8),
            constraints=[
                "Implement everything in main.py, exposing exactly one function: run(payload: dict) -> dict.",
                "run() must always return a dict and must never raise for an expected failure.",
                "Read every value out of payload with .get(), never with [] -- a missing key must be a "
                "clean error, not a KeyError.",
                "If the capability affects anything outside this process, honour payload.get('dry_run') by "
                "performing every check and reporting what it would do, without doing it.",
                # Reinventing this is a reliable failure: a hand-rolled PATH walk misses
                # powershell.exe on Windows because it ignores PATHEXT, and the capability
                # then concludes the machine has no player at all.
                "To find a program, use shutil.which() from the standard library. Do not write your own "
                "PATH search: shutil.which handles the .exe/.cmd extensions Windows needs.",
                "Prefer the Python standard library over an external program whenever it can do the job. "
                "A capability with no external dependency is more reliable and needs no installation.",
                "Use the 'which' tool to check what is actually installed on this machine before choosing "
                "an approach. Do not assume a program exists.",
                "Keep the module-level INPUT_SCHEMA in main.py accurate: it must list every payload key "
                "run() reads. It is the only way a caller can know what to pass, so a key that is not "
                "declared there is a key nobody will ever send.",
                "Write real tests in test_capability.py that prove behaviour, not just key presence.",
                # The failure this prevents: a test asserting a specific player is installed,
                # which is false on this machine, so a correct implementation still fails.
                "Tests must NOT assert that any particular external program is installed. Assert on "
                "behaviour instead: that a dry run reports whichever mechanism it chose, that a missing "
                "input fails cleanly, that the returned dict has the documented shape.",
            ],
            acceptance=[
                (
                    "the capability's own tests pass",
                    [sys.executable, "-m", "pytest", "-q", "test_capability.py"],
                ),
                (
                    "main.run is implemented, importable, and returns a dict",
                    [
                        sys.executable,
                        "-c",
                        "import main; r = main.run({'dry_run': True}); "
                        "assert isinstance(r, dict), type(r); "
                        f"assert '{NOT_IMPLEMENTED}' not in open('main.py', encoding='utf-8').read(), "
                        "'the skeleton has not been replaced with a real implementation'; "
                        "print('CONTRACT_OK')",
                    ],
                ),
            ],
        )
        project.metadata["capability_id"] = capability_id

        workspace = self.engine.store.workspace_for(project)
        # Seed the workspace so the model edits a working skeleton rather than
        # inventing the file layout, which it gets wrong far more often.
        (workspace / "main.py").write_text(_TEMPLATE_MAIN, encoding="utf-8")
        (workspace / "test_capability.py").write_text(_TEMPLATE_TEST, encoding="utf-8")
        self.engine.store.save(project)
        return project

    # ------------------------------------------------------------------
    # Verification and installation
    # ------------------------------------------------------------------

    def _verify(self, workspace: Path) -> dict[str, Any]:
        """Prove the capability works, independently of the project's own claim.

        Run separately from the loop's acceptance check on purpose: the loop
        decides when to stop, and something that did not participate in that
        decision should decide whether the result is real.
        """

        main = workspace / "main.py"
        if not main.exists():
            return {"ok": False, "detail": "main.py was never created"}

        checks: list[dict[str, Any]] = []

        # First and most important: is there an implementation at all?  The
        # seeded skeleton returns a dict and satisfies its own placeholder test,
        # so without this a capability that does nothing registers as verified.
        source = main.read_text(encoding="utf-8", errors="replace")
        test_source = ""
        test_file = workspace / "test_capability.py"
        if test_file.exists():
            test_source = test_file.read_text(encoding="utf-8", errors="replace")
        checks.append(
            {
                "name": "implemented",
                "ok": NOT_IMPLEMENTED not in source and NOT_IMPLEMENTED not in test_source,
                "detail": f"{NOT_IMPLEMENTED} is still present: the skeleton was never replaced",
            }
        )

        contract = self._run(
            [
                sys.executable,
                "-c",
                "import main; "
                "assert callable(getattr(main, 'run', None)), 'main.run is missing'; "
                "r = main.run({'dry_run': True}); "
                "assert isinstance(r, dict), f'run() returned {type(r)}'; "
                "print('CONTRACT_OK')",
            ],
            workspace,
        )
        checks.append({"name": "contract", "ok": contract["ok"], "detail": contract["detail"][-800:]})

        tests = self._run([sys.executable, "-m", "pytest", "-q", "test_capability.py"], workspace)
        checks.append({"name": "tests", "ok": tests["ok"], "detail": tests["detail"][-1500:]})

        # A test file with no assertions passes trivially and proves nothing.
        substantive = test_source.count("assert") >= 2 and "main.run" in test_source
        checks.append(
            {
                "name": "tests_are_substantive",
                "ok": substantive,
                "detail": "the test file must call main.run and make at least two assertions",
            }
        )

        return {"ok": all(item["ok"] for item in checks), "checks": checks, "detail": "; ".join(
            f"{item['name']}={'ok' if item['ok'] else 'FAILED'}" for item in checks
        )}

    def _install(
        self,
        capability_id: str,
        goal: str,
        workspace: Path,
        verification: dict[str, Any],
        *,
        keywords: list[str] | None = None,
    ) -> CapabilityManifest:
        """Copy the verified workspace into the permanent catalog and register it."""

        version = self._next_version(capability_id)
        target = self.root / capability_id.replace(".", "_") / version
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            workspace,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".venv"),
        )

        terms = sorted(set((keywords or []) + _keywords_from(goal)))
        manifest = CapabilityManifest(
            capability_id=capability_id,
            description=goal,
            version=version,
            entrypoint="main.py",
            source_location=str(target.resolve()),
            tests_location=str((target / "test_capability.py").resolve()),
            input_schema=self._input_schema_of(target),
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            creation_metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "capability_service",
                "goal": goal,
                "keywords": terms,
            },
            validation_status={"verified": True, "checks": verification.get("checks", [])},
        )
        self.registry.register(manifest)

        if self.memory is not None:
            self.memory.record_capability(capability_id, goal, keywords=terms, version=version)
        return manifest

    def _input_schema_of(self, source: Path) -> dict[str, Any]:
        """Read the capability's declared INPUT_SCHEMA out of its own module.

        A capability that invents its own payload key is useless if nobody can
        discover the key: an acquired audio player expecting ``audio_path`` was
        unusable by a caller that passed ``path``. Reading the declaration makes
        the capability self-describing, which is what a registry is for.

        Extracted in a subprocess, since importing model-authored code into the
        Jarvis process is exactly what :meth:`execute` avoids.
        """

        fallback = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}
        result = self._run(
            [
                sys.executable,
                "-c",
                "import json, main; "
                "schema = getattr(main, 'INPUT_SCHEMA', None); "
                "print('SCHEMA:' + json.dumps(schema if isinstance(schema, dict) else {}))",
            ],
            source,
        )
        if not result["ok"] or "SCHEMA:" not in result["detail"]:
            return fallback
        try:
            raw = result["detail"].split("SCHEMA:", 1)[1].splitlines()[0]
            schema = json.loads(raw)
        except (IndexError, json.JSONDecodeError):
            return fallback
        return schema if schema.get("properties") else fallback

    def _next_version(self, capability_id: str) -> str:
        existing = self.registry.get(capability_id)
        if existing is None:
            return "1.0.0"
        try:
            major, minor, patch = (int(part) for part in existing.version.split("."))
        except ValueError:
            return "1.0.1"
        return f"{major}.{minor}.{patch + 1}"

    # ------------------------------------------------------------------
    # Executing
    # ------------------------------------------------------------------

    def execute(self, capability_id: str, payload: dict[str, Any] | None = None) -> ExecutionOutcome:
        """Run an installed capability.

        Runs from the installed copy, in a subprocess, with a timeout. A
        subprocess rather than an import because a capability is model-authored
        code: it should not be able to take the Jarvis process down with it, and
        it must be reloadable without a restart.
        """

        import time

        manifest = self.registry.get(capability_id)
        if manifest is None or manifest.status != "active":
            return ExecutionOutcome(capability_id=capability_id, ok=False, error=f"no active capability {capability_id!r}")

        source = Path(manifest.source_location)
        if not source.exists():
            return ExecutionOutcome(
                capability_id=capability_id, ok=False, error=f"installed source is missing: {source}"
            )

        request = dict(payload or {})
        started = time.perf_counter()
        run_dir = source.parent / f".run_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            (run_dir / "request.json").write_text(json.dumps(request, default=str), encoding="utf-8")
            (run_dir / "runner.py").write_text(_runner_source(manifest.entrypoint), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(run_dir / "runner.py")],
                cwd=str(source),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.execution_timeout,
                env={**_safe_env(), "JARVIS_CAPABILITY_RUN_DIR": str(run_dir)},
            )
            duration = time.perf_counter() - started

            if completed.returncode != 0:
                return ExecutionOutcome(
                    capability_id=capability_id,
                    ok=False,
                    error=f"exit={completed.returncode}\n{(completed.stderr or completed.stdout)[-1500:]}",
                    duration_seconds=duration,
                )
            output_path = run_dir / "output.json"
            if not output_path.exists():
                return ExecutionOutcome(
                    capability_id=capability_id, ok=False, error="the capability produced no output", duration_seconds=duration
                )
            output = json.loads(output_path.read_text(encoding="utf-8"))
            return ExecutionOutcome(
                capability_id=capability_id,
                ok=bool(output.get("ok", True)),
                output=output,
                error=str(output.get("error", "")),
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            return ExecutionOutcome(
                capability_id=capability_id,
                ok=False,
                error=f"the capability did not finish within {self.execution_timeout:.0f}s",
                duration_seconds=time.perf_counter() - started,
            )
        except (OSError, json.JSONDecodeError) as exc:
            return ExecutionOutcome(
                capability_id=capability_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - started,
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def use(self, goal: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> tuple[CapabilityOutcome, ExecutionOutcome | None]:
        """Acquire if needed, then run.  The whole lifecycle in one call."""

        outcome = self.ensure(goal, **kwargs)
        if not outcome.usable:
            return outcome, None
        return outcome, self.execute(outcome.capability_id, payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_id(goal: str) -> str:
        """A stable, readable identifier derived from the goal."""

        import re

        words = [word for word in re.split(r"[^a-z0-9]+", goal.lower()) if len(word) > 2]
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "can", "able", "ability",
            "build", "make", "create", "reusable", "capability", "jarvis", "please", "want", "need",
            "der", "die", "das", "und", "fuer", "mit", "eine", "einen", "kannst", "kann",
        }
        useful = [word for word in words if word not in stopwords][:4]
        if not useful:
            useful = ["capability", uuid.uuid4().hex[:6]]
        return "local." + ".".join(useful[:2]) + ("." + "_".join(useful[2:]) if len(useful) > 2 else "")

    def _run(self, command: list[str], cwd: Path) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.execution_timeout,
                env=_safe_env(),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": completed.returncode == 0,
            "detail": f"exit={completed.returncode}\n{completed.stdout}\n{completed.stderr}",
        }

    def _remember_attempt(self, project: Project, capability_id: str, *, succeeded: bool, reason: str) -> None:
        if self.memory is None:
            return
        self.memory.record_project(project)
        if not succeeded:
            from knowledge.memory import Lesson

            self.memory.record_lesson(
                Lesson(
                    text=f"acquiring {capability_id} failed: {reason[:180]}",
                    worked=False,
                    evidence=reason[:400],
                    source_project=project.id,
                    tags=("capability",),
                )
            )


def _runner_source(entrypoint: str) -> str:
    module = Path(entrypoint).stem
    return (
        "import importlib\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "\n"
        'run_dir = pathlib.Path(os.environ["JARVIS_CAPABILITY_RUN_DIR"])\n'
        "sys.path.insert(0, str(pathlib.Path.cwd()))\n"
        'payload = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))\n'
        f'module = importlib.import_module("{module}")\n'
        "try:\n"
        "    result = module.run(payload)\n"
        "except Exception as exc:\n"
        '    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}\n'
        'if not isinstance(result, dict):\n'
        '    result = {"ok": False, "error": f"run() returned {type(result).__name__}, expected dict"}\n'
        '(run_dir / "output.json").write_text(json.dumps(result, default=str), encoding="utf-8")\n'
    )


def _safe_env() -> dict[str, str]:
    import os

    allowed = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME", "USERPROFILE",
        "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "LANG",
    )
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _keywords_from(goal: str) -> list[str]:
    import re

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "can", "able", "build",
        "make", "create", "reusable", "capability", "jarvis", "please", "want", "need", "should",
    }
    return [word for word in re.split(r"[^a-z0-9]+", goal.lower()) if len(word) > 3 and word not in stopwords][:8]
