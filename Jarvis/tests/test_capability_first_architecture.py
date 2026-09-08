from __future__ import annotations

from pathlib import Path
from typing import Any

from capabilities.codex import CodexAvailabilityState, StaticCodexAvailability
from capabilities.models import (
    CapabilityHealth,
    CapabilityLifecycle,
    CapabilityManifest,
    CapabilityResolutionStatus,
    SkillSpecification,
)
from capabilities.registry import CapabilityRegistry
from capabilities.resolver import CapabilityResolver
from environments.coding.sandbox_backend import LocalTestSandboxBackend
from experts.contracts import ExpertResult, ExpertStatus
from runtime.capability_runtime import CapabilityAcquisitionRuntime, CapabilityRuntimeConfig
from service.acquisition import AcquisitionMission


class ExplodingBrain:
    provider_name = "exploding"
    model_name = "ExplodingBrain"
    last_metadata = {"generated_tokens": 0, "total_tokens": 0}

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise AssertionError("local/LLM resolver should not be called")

    def generate_structured(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise AssertionError("capability engineering should not be called")


def _spotify_manifest(*, health: CapabilityHealth = CapabilityHealth.HEALTHY) -> CapabilityManifest:
    return CapabilityManifest(
        "media.spotify.play",
        "Starts playback of a requested artist, album or track in Spotify.",
        source_location="skills/installed/media.spotify.play/1.0.0",
        lifecycle=CapabilityLifecycle.ACTIVE.value,
        health={"state": health.value.lower(), "health": health.value},
        family="media",
        name="Spotify Playback",
        goal_types=["PLAY_MEDIA"],
        target_types=["MEDIA_QUERY"],
        examples=["Spiel Rammstein.", "Mach Sonne von Rammstein an.", "Play Discover Weekly."],
        anti_examples=["Öffne Wikipedia.", "Wie viele Dateien sind in D:?"],
        aliases=["spotify play", "music playback"],
        runtime_brain="none",
        codex_required=False,
        created_by="codex",
    )


def test_codex_created_manifest_defaults_to_local_runtime() -> None:
    spec = SkillSpecification(
        capability_id="filesystem.largest_child",
        objective="Find the largest direct child under a directory.",
        acceptance_criteria=["Returns the largest child path and size."],
        public_tests=[{"input": {"root_path": "."}, "expected_keys": ["path", "size"]}],
        metadata={
            "goal_types": ["FILESYSTEM_LARGEST_CHILD"],
            "target_types": ["DIRECTORY"],
            "examples": ["Welcher Ordner auf D: ist am größten?"],
            "anti_examples": ["Spiel Rammstein."],
        },
    )

    manifest = spec.to_manifest()

    assert manifest.runtime_brain == "none"
    assert manifest.codex_required is False
    assert manifest.source == "codex_generated"
    assert manifest.created_by == "codex"
    assert manifest.lifecycle_state() is CapabilityLifecycle.VERIFIED
    assert manifest.validate() == []


def test_resolver_finds_healthy_typed_capability_without_brain(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_spotify_manifest())
    brain = ExplodingBrain()

    result = CapabilityResolver(registry, brain=brain).resolve("Spiel Rammstein.")

    assert result.outcome is CapabilityResolutionStatus.FOUND
    assert result.status == "available"
    assert result.capability_id == "media.spotify.play"
    assert brain.calls == 0


def test_resolver_rejects_semantic_mismatch_even_with_action_words(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_spotify_manifest())

    result = CapabilityResolver(registry).resolve("Öffne Wikipedia.")

    assert result.outcome is CapabilityResolutionStatus.MISSING
    assert result.capability_id != "media.spotify.play"


def test_resolver_reports_broken_capability_instead_of_executing(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_spotify_manifest(health=CapabilityHealth.BROKEN))

    result = CapabilityResolver(registry).resolve("Spiel Rammstein.")

    assert result.outcome is CapabilityResolutionStatus.BROKEN
    assert result.status == "broken"
    assert result.capability_id == "media.spotify.play"


def test_resolver_does_not_treat_a_weak_second_candidate_as_ambiguity(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(
        CapabilityManifest(
            "text.line_count",
            "Count non-empty lines in supplied text.",
            lifecycle=CapabilityLifecycle.ACTIVE.value,
            health={"state": "healthy", "health": CapabilityHealth.HEALTHY.value},
        )
    )
    registry.register(
        CapabilityManifest(
            "data.csv_column_mode",
            "Read CSV text and return the most common value in a selected column.",
            lifecycle=CapabilityLifecycle.ACTIVE.value,
            health={"state": "healthy", "health": CapabilityHealth.HEALTHY.value},
        )
    )

    result = CapabilityResolver(registry).resolve(
        "Please handle this related request: read csv text and return the most common value in a selected column."
    )

    assert result.outcome is CapabilityResolutionStatus.FOUND
    assert result.capability_id == "data.csv_column_mode"
    assert [item["capability_id"] for item in result.candidates] == ["data.csv_column_mode", "text.line_count"]


def test_runtime_executes_known_capability_without_codex_or_brain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "installed" / "media.spotify.play" / "1.0.0"
    source.mkdir(parents=True)
    (source / "main.py").write_text(
        "def run(payload):\n"
        "    return {'ok': True, 'playing': payload.get('query')}\n",
        encoding="utf-8",
    )
    brain = ExplodingBrain()
    codex = StaticCodexAvailability(CodexAvailabilityState.READY, "ready")
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(
            data_dir=str(tmp_path / "runtime"),
            skills_root=str(tmp_path / "skills"),
            use_docker=False,
        ),
        codex_availability=codex,
    )
    manifest = _spotify_manifest()
    manifest.source_location = str(source)
    manifest.implementation_path = str(source)
    runtime.registry.register(manifest)

    result = runtime.handle_goal(
        "Spiel Rammstein.",
        request_payload={"query": "Rammstein"},
        expected_output={"ok": True, "playing": "Rammstein"},
    )

    assert result.success
    assert result.promoted is False
    assert result.codex_checked is False
    assert result.resolution.outcome is CapabilityResolutionStatus.FOUND
    assert result.output == {"ok": True, "playing": "Rammstein"}
    assert codex.calls == 0
    assert brain.calls == 0


def test_runtime_missing_goal_checks_codex_before_generation_and_queues(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    brain = ExplodingBrain()
    codex = StaticCodexAvailability(CodexAvailabilityState.QUOTA_EXHAUSTED, "daily limit reached")
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(
            data_dir=str(tmp_path / "runtime"),
            skills_root=str(tmp_path / "skills"),
            use_docker=False,
        ),
        codex_availability=codex,
    )

    result = runtime.handle_goal("Count the copper widgets in this folder.")

    assert not result.success
    assert result.queued
    assert result.codex_checked
    assert result.codex_state == "QUOTA_EXHAUSTED"
    assert result.resolution.outcome is CapabilityResolutionStatus.MISSING
    assert codex.calls == 1
    assert brain.calls == 0
    queued = runtime.request_queue.list()
    assert len(queued) == 1
    assert queued[0].goal == "Count the copper widgets in this folder."


class _NoMemory:
    def __init__(self) -> None:
        self.recorded: list[Any] = []

    def context_for(self, goal: str, *, task_class: str = "", limit: int = 2) -> str:
        return ""

    def record(self, lesson: Any) -> Any:
        self.recorded.append(lesson)
        return lesson


class _NoLedger:
    def __init__(self) -> None:
        self.attempts: list[Any] = []

    def record(self, attempt: Any) -> None:
        self.attempts.append(attempt)


class _NoMissions:
    def resumable(self, *args: Any, **kwargs: Any) -> None:
        return None

    def load(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save(self, *args: Any, **kwargs: Any) -> None:
        return None

    def clear(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Kernel:
    def __init__(self, root: Path) -> None:
        self.state_root = root


class _CodexFirstService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ensure_calls = 0
        self.started = 0
        self.installed: list[str] = []

    def ensure(self, goal: str, **kwargs: Any) -> None:
        self.ensure_calls += 1
        raise AssertionError("codex_first must not call BUILD_LOCAL first")

    def _start_project(self, goal: str, capability_id: str, **kwargs: Any) -> Any:
        self.started += 1
        workspace = self.root / "workspace" / capability_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "main.py").write_text("def run(payload):\n    return {'ok': True}\n", encoding="utf-8")
        return type("Project", (), {"workspace": str(workspace)})()

    def _verify(self, workspace: Path, *, extra_checks: list[Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "detail": "verified", "checks": [{"name": "tests", "ok": True}]}

    def _install(
        self,
        capability_id: str,
        goal: str,
        workspace: Path,
        verification: dict[str, Any],
        *,
        keywords: list[str] | None = None,
        built_by: str = "local_build",
    ) -> Any:
        self.installed.append(capability_id)
        return type("Manifest", (), {"capability_id": capability_id})()

    @staticmethod
    def suggest_id(goal: str) -> str:
        return "filesystem.largest_child"


class _Gateway:
    def __init__(self) -> None:
        self.jobs: list[Any] = []

    def status(self) -> dict[str, Any]:
        return {"expert_available": True, "state": "AVAILABLE"}

    def submit(self, job: Any) -> ExpertResult:
        self.jobs.append(job)
        return ExpertResult(
            status=ExpertStatus.COMPLETED,
            provider="codex",
            summary="implemented",
            test_evidence=[("tests", True, "")],
        )


def test_acquisition_mission_codex_first_queues_when_codex_unavailable(tmp_path: Path) -> None:
    service = _CodexFirstService(tmp_path)
    codex = StaticCodexAvailability(CodexAvailabilityState.OFFLINE, "not signed in")
    mission = AcquisitionMission(
        service=service,
        kernel=_Kernel(tmp_path),
        codex_availability=codex,
        memory=_NoMemory(),
        ledger=_NoLedger(),
        missions=_NoMissions(),
    )

    result = mission.run("Find the largest direct child folder.", codex_first=True)

    assert not result.acquired
    assert result.queued
    assert result.codex_checked
    assert result.codex_state == "OFFLINE"
    assert result.local_attempts == 0
    assert service.ensure_calls == 0
    assert codex.calls == 1
    assert (tmp_path / "capabilities" / "requests.jsonl").exists()


def test_acquisition_mission_codex_first_uses_isolated_codex_workspace_not_local_builder(tmp_path: Path) -> None:
    service = _CodexFirstService(tmp_path)
    gateway = _Gateway()
    mission = AcquisitionMission(
        service=service,
        kernel=_Kernel(tmp_path),
        gateway=gateway,
        codex_availability=StaticCodexAvailability(CodexAvailabilityState.READY, "ready"),
        memory=_NoMemory(),
        ledger=_NoLedger(),
        missions=_NoMissions(),
    )

    result = mission.run("Find the largest direct child folder.", codex_first=True)

    assert result.acquired
    assert result.escalated
    assert result.capability_id == "filesystem.largest_child"
    assert service.ensure_calls == 0
    assert service.started == 1
    assert service.installed == ["filesystem.largest_child"]
    assert len(gateway.jobs) == 1
    assert gateway.jobs[0].workspace.name == "filesystem.largest_child"
