from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import brain.providers as providers
import jarvis.build_executor as build_executor_module
from brain.router import BrainRouter, BrainTier, RemoteBrainUnavailable
from brain.tiers import ModelCatalog
from jarvis.build_executor import BuildExecutor


class FakeBrain:
    provider_name = "fake_remote"
    model_name = "fake-coder"

    def generate(
        self,
        prompt,
        *,
        max_tokens=512,
        temperature=0.2,
        top_p=None,
    ):
        return "OK"


def test_router_exposes_injected_build_brain():
    brain = FakeBrain()

    router = BrainRouter(fast_brain=brain, build_brain=brain)

    assert router.brain(BrainTier.BUILD_LOCAL) is brain


def test_router_blocks_a_disabled_build_tier():
    """A tier turned off in the catalog must never hand back a brain."""

    catalog = ModelCatalog(environ={"JARVIS_BUILD_LOCAL_ENABLED": "0"})
    router = BrainRouter(catalog=catalog)

    try:
        router.brain(BrainTier.BUILD_LOCAL)
    except RemoteBrainUnavailable:
        pass
    else:
        raise AssertionError(
            "a disabled BUILD_LOCAL unexpectedly returned a brain"
        )


def test_remote_provider_configuration_isolated(monkeypatch):
    observed = {}

    def fake_factory():
        observed.update(
            {
                key: value
                for key, value in os.environ.items()
                if key.startswith("JARVIS_BRAIN_")
            }
        )
        return FakeBrain()

    monkeypatch.setattr(
        providers,
        "make_brain_provider_from_env",
        fake_factory,
    )

    monkeypatch.setenv(
        "JARVIS_BRAIN_MODEL",
        "qwen3:4b-instruct",
    )
    monkeypatch.setenv(
        "JARVIS_BRAIN_BASE_URL",
        "http://127.0.0.1:11434",
    )

    monkeypatch.setenv(
        "JARVIS_BUILD_REMOTE_MODEL",
        "remote-coder",
    )
    monkeypatch.setenv(
        "JARVIS_BUILD_REMOTE_BASE_URL",
        "https://remote.example/v1",
    )
    monkeypatch.setenv(
        "JARVIS_BUILD_REMOTE_API_KEY",
        "remote-key",
    )

    brain = (
        providers.make_build_remote_brain_provider_from_env()
    )

    assert isinstance(brain, FakeBrain)

    assert (
        observed["JARVIS_BRAIN_MODEL"]
        == "remote-coder"
    )
    assert (
        observed["JARVIS_BRAIN_BASE_URL"]
        == "https://remote.example/v1"
    )
    assert (
        observed["JARVIS_BRAIN_API_KEY"]
        == "remote-key"
    )

    # FAST_LOCAL environment must be restored afterwards.
    assert (
        os.environ["JARVIS_BRAIN_MODEL"]
        == "qwen3:4b-instruct"
    )
    assert (
        os.environ["JARVIS_BRAIN_BASE_URL"]
        == "http://127.0.0.1:11434"
    )


def test_build_executor_delegates_to_repository_engineer(
    monkeypatch,
    tmp_path,
):
    # This test verifies BuildRemoteExecutor defaults and must not
    # inherit BUILD_REMOTE settings from the caller's shell.
    for name in (
        "JARVIS_BUILD_TARGETED_TEST_COMMAND",
        "JARVIS_BUILD_FULL_TEST_COMMAND",
        "JARVIS_BUILD_ALLOWED_PATHS",
        "JARVIS_BUILD_PROTECTED_PATHS",
        "JARVIS_BUILD_CONTEXT_WINDOW",
        "JARVIS_BUILD_MAX_CYCLES",
        "JARVIS_BUILD_COMMAND_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)

    captured = {}

    class FakeEngineer:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def preflight(self):
            return {
                "provider": "fake_remote",
                "structured_generation": "OK",
            }

        def improve(
            self,
            repository_path,
            goal,
            acceptance_commands,
            *,
            full_test_commands=None,
            benchmark_commands=None,
            max_cycles=None,
        ):
            captured["repository_path"] = repository_path
            captured["goal"] = goal
            captured["acceptance_commands"] = (
                acceptance_commands
            )
            captured["full_test_commands"] = (
                full_test_commands
            )
            captured["benchmark_commands"] = (
                benchmark_commands
            )
            captured["max_cycles"] = max_cycles

            return SimpleNamespace(
                status="SELF_DEVELOPMENT_CANDIDATE_READY",
                success=True,
                cycles=1,
                changed_files=["example.py"],
                worktree=str(
                    tmp_path / "fake-worktree"
                ),
                diff_path="diff.patch",
                result_path="result.json",
                error="",
            )

    monkeypatch.setattr(
        build_executor_module,
        "RepositoryEngineer",
        FakeEngineer,
    )

    repo = tmp_path / "repo"
    repo.mkdir()

    executor = BuildExecutor(
        repository_path=repo,
        brain=FakeBrain(),
        run_root=tmp_path / "runs",
    )

    result = executor.execute(
        "Implementiere Feature X"
    )

    assert result.success is True

    assert (
        captured["goal"].objective
        == "Implementiere Feature X"
    )

    assert (
        Path(captured["repository_path"]).resolve()
        == repo.resolve()
    )

    assert (
        captured["full_test_commands"][0][:3]
        == [sys.executable, "-m", "pytest"]
    )
