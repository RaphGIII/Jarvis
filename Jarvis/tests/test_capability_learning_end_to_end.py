"""The whole loop on one state root: taught, registered, restarted, reused.

This is the shape of the live acceptance, with the Codex CLI replaced by a
stand-in that writes the files a real engineer would write. Everything else is
the production machinery: the same workspace, the same verification gates, the
same registry, the same resolver, the same executor.

What it proves, and what it does not: it proves that a capability which did not
exist can be engineered, verified, registered, executed, survive a restart, and
then answer a *differently worded* request locally with the engineer absent. It
does not prove the Codex CLI itself works -- that is only provable by running
it, and it is recorded in the sprint report from a live run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from capabilities.codex import CodexAvailabilityState, StaticCodexAvailability
from capabilities.models import CapabilityHealth, CapabilityLifecycle
from capabilities.registry import CapabilityRegistry
from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from experts.contracts import ExpertResult, ExpertStatus
from service.core import JarvisCore


MAIN = '''"""Compute the SHA-256 digest of a file."""

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string", "description": "the file to digest"}},
    "required": ["file_path"],
}


def run(payload):
    import hashlib
    import pathlib

    if payload.get("dry_run"):
        return {"ok": True, "detail": "would read the file and return its SHA-256 digest", "dry_run": True}
    raw = payload.get("file_path")
    if not raw:
        return {"ok": False, "error": "file_path is required"}
    target = pathlib.Path(str(raw))
    if not target.is_file():
        return {"ok": False, "error": f"no such file: {target}"}
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"ok": True, "sha256": digest, "algorithm": "sha256",
            "detail": f"SHA-256 von {target.name}: {digest}"}
'''

TESTS = '''import hashlib
import pathlib

import main


def test_a_missing_path_is_a_clean_error():
    result = main.run({})
    assert result["ok"] is False
    assert "file_path" in result["error"]


def test_the_digest_matches_hashlib(tmp_path):
    target = tmp_path / "sample.bin"
    target.write_bytes(b"zeus")
    result = main.run({"file_path": str(target)})
    assert result["ok"] is True
    assert result["sha256"] == hashlib.sha256(b"zeus").hexdigest()


def test_a_dry_run_reads_nothing_and_says_so():
    result = main.run({"dry_run": True})
    assert result["ok"] is True and result["dry_run"] is True
'''


class StandInEngineer:
    """What a working engineer leaves behind: two files and a summary."""

    name = "codex"

    def __init__(self) -> None:
        self.jobs: list[Any] = []

    def status(self) -> dict[str, Any]:
        return {"expert_available": True, "state": "AVAILABLE"}

    def submit(self, job: Any) -> ExpertResult:
        self.jobs.append(job)
        workspace = Path(job.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "main.py").write_text(MAIN, encoding="utf-8")
        (workspace / "test_capability.py").write_text(TESTS, encoding="utf-8")
        return ExpertResult(status=ExpertStatus.COMPLETED, provider="codex",
                            summary="implemented the SHA-256 digest capability",
                            files_changed=["main.py", "test_capability.py"])


class ExplodingEngineer:
    """Any call at all is the defect under test."""

    def status(self) -> dict[str, Any]:
        raise AssertionError("a learned capability must not consult the engineer")

    def submit(self, job: Any) -> ExpertResult:
        raise AssertionError("a learned capability must not invoke the engineer")

    def availability(self) -> Any:
        raise AssertionError("a learned capability must not probe the engineer")


class _NoModel:
    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("the capability path must not call a model")

    def generate_stream(self, *args: Any, **kwargs: Any):
        raise AssertionError("the capability path must not call a model")


def _core(state_root: Path) -> JarvisCore:
    kernel = JarvisKernel(KernelConfig(state_root=state_root))
    kernel.provider = lambda tier: _NoModel()  # type: ignore[assignment]
    kernel.catalog.get = lambda tier: type("S", (), {"model": "stub"})()  # type: ignore[assignment]
    return JarvisCore(kernel=kernel, identity=Identity())


def _drive(core: JarvisCore, goal: str, text: str, *, gateway: Any, engineer_ready: bool = True) -> dict[str, Any]:
    """One owner request through the production routing path, synchronously."""

    from service.acquisition import AcquisitionMission

    delivered: list[str] = []
    events: list[dict[str, Any]] = []
    core.emit = lambda kind, payload, scope="": events.append({"kind": str(kind), **payload})  # type: ignore[assignment]
    core._deliver = lambda text_, **kwargs: delivered.append(text_)  # type: ignore[assignment]

    real_thread_start = None
    if gateway is not None:
        def _mission(**kwargs: Any) -> AcquisitionMission:
            kwargs.setdefault("gateway", gateway)
            kwargs.setdefault(
                "codex_availability",
                StaticCodexAvailability(
                    CodexAvailabilityState.READY if engineer_ready else CodexAvailabilityState.QUOTA_EXHAUSTED,
                    "stand-in engineer" if engineer_ready else "allowance spent",
                ),
            )
            return AcquisitionMission(**kwargs)

        import service.acquisition as acquisition_module

        real_mission = acquisition_module.AcquisitionMission
        acquisition_module.AcquisitionMission = _mission  # type: ignore[assignment]
    # The engineering runs on a worker thread in production; here it is joined
    # so the assertions see the finished state rather than a race.
    import threading

    real_thread_start = threading.Thread.start
    threads: list[threading.Thread] = []

    def _start(self: threading.Thread) -> None:
        threads.append(self)
        real_thread_start(self)

    threading.Thread.start = _start  # type: ignore[assignment]
    try:
        core._route_capability_goal(goal, text, "test")
        for thread in threads:
            thread.join(timeout=300)
    finally:
        threading.Thread.start = real_thread_start  # type: ignore[assignment]
        if gateway is not None:
            acquisition_module.AcquisitionMission = real_mission  # type: ignore[assignment]
    routes = [e["capability_routing"] for e in events if "capability_routing" in e]
    return {"delivered": delivered, "events": events, "routes": routes}


def test_a_capability_is_taught_registered_and_then_reused_after_a_restart(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    workspace = state_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    fixture = workspace / "bericht.txt"
    fixture.write_bytes(b"ZEUS capability acceptance fixture\n")
    expected = hashlib.sha256(fixture.read_bytes()).hexdigest()

    # ---- 1. the capability does not exist -------------------------------
    core = _core(state_root)
    assert core.capabilities.resolve_request("Berechne die SHA-256-Pruefsumme der Datei bericht.txt").outcome.value == "MISSING"

    # ---- 2. the request is routed, the engineer is asked, the work lands -
    engineer = StandInEngineer()
    first = _drive(
        core,
        "Berechne die SHA-256-Pruefsumme der Datei bericht.txt",
        "Zeus, berechne die SHA-256-Pruefsumme der Datei bericht.txt",
        gateway=engineer,
    )

    assert [route["result"] for route in first["routes"]][0] == "MISSING"
    assert len(engineer.jobs) == 1, "the engineer is asked exactly once"
    assert engineer.jobs[0].constraints, "and is told the capability contract it will be judged against"

    # ---- 3. what was registered -----------------------------------------
    registered = [m for m in core.capabilities.registry.all() if m.capability_id != "none"]
    trace = "\n".join(f"  {e.get('kind')} {str(e.get('summary') or e.get('error'))[:180]}" for e in first["events"])
    assert registered, f"nothing was registered.\n{first['delivered']}\n{trace}"
    manifest = registered[0]
    assert manifest.lifecycle_state() is CapabilityLifecycle.ACTIVE
    assert manifest.health_state() is CapabilityHealth.HEALTHY
    assert manifest.codex_required is False
    assert manifest.runtime_brain == "none"
    assert manifest.created_by == "codex", "who wrote it is recorded, not assumed"
    assert manifest.source == "codex_generated"
    assert Path(manifest.implementation_path).is_dir()
    indexed = " ".join(manifest.creation_metadata.get("keywords") or [])
    assert "bericht" not in indexed, "the capability must not be indexed under the one file it was first asked about"

    # ---- 4. the original request was answered ---------------------------
    assert any(expected in text for text in first["delivered"]), first["delivered"]

    # ---- 5. a restart: nothing in memory, only what is on disk ----------
    restarted = _core(state_root)
    assert restarted is not core
    reloaded = CapabilityRegistry(state_root / "capabilities" / "registry.json").get(manifest.capability_id)
    assert reloaded is not None and reloaded.is_active()

    # ---- 6. a different wording, with the engineer absent ---------------
    second = _drive(
        restarted,
        "Welchen sha256 Fingerprint hat die Datei bericht.txt?",
        "Zeus, welchen sha256 Fingerprint hat die Datei bericht.txt?",
        gateway=ExplodingEngineer(),
    )

    assert [route["result"] for route in second["routes"]] == ["FOUND"]
    assert second["routes"][0]["capability_id"] == manifest.capability_id
    assert second["routes"][0]["codex_checked"] is False
    assert any(expected in text for text in second["delivered"]), second["delivered"]

    # ---- 7. and the wording that worked is now part of what it answers to
    learned = CapabilityRegistry(state_root / "capabilities" / "registry.json").get(manifest.capability_id)
    assert any("fingerprint" in alias.lower() for alias in learned.aliases)
