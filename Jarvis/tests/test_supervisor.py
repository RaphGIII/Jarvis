"""The supervisor: known-good pointer, control channel, rollback, readiness."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from zeus_supervisor import EXIT_RESTART_REQUESTED, EXIT_SHUTDOWN_REQUESTED
from zeus_supervisor.config import DEFAULT_MODELS, SupervisorConfig, discover_ollama_store, find_repository
from zeus_supervisor.control import ControlChannel
from zeus_supervisor.known_good import DeploymentReceipt, KnownGoodStore
from zeus_supervisor.supervisor import Supervisor


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "Jarvis"
    (root / "jarvis").mkdir(parents=True)
    (root / "service").mkdir()
    (root / "jarvis" / "serve.py").write_text("print('serve')\n", encoding="utf-8")
    (root / "service" / "core.py").write_text("# core\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "known good")
    return root


def test_default_models_match_the_catalog() -> None:
    from brain.tiers import ModelCatalog, ModelTier

    catalog = ModelCatalog()
    assert DEFAULT_MODELS["FAST_LOCAL"] == catalog.get(ModelTier.FAST_LOCAL).model
    assert DEFAULT_MODELS["BUILD_LOCAL"] == catalog.get(ModelTier.BUILD_LOCAL).model


def test_find_repository_from_inside_and_above(repo: Path) -> None:
    assert find_repository(repo / "jarvis") == repo.resolve()
    assert find_repository(repo.parent) == repo.resolve()


def test_discover_store_prefers_one_with_every_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    partial = tmp_path / "partial"
    complete = tmp_path / "complete"
    for store, models in ((partial, ["a:1"]), (complete, ["a:1", "b:2"])):
        for model in models:
            name, tag = model.split(":")
            path = store / "manifests" / "registry.ollama.ai" / "library" / name / tag
            path.parent.mkdir(parents=True)
            path.write_text("{}")
    monkeypatch.setattr("zeus_supervisor.config.OLLAMA_STORE_CANDIDATES", (str(partial), str(complete)))
    assert discover_ollama_store(["a:1", "b:2"]) == str(complete)
    assert discover_ollama_store(["a:1"]) == str(partial)
    assert discover_ollama_store(["zzz:9"]) == ""


def test_known_good_pointer_and_receipts(tmp_path: Path) -> None:
    store = KnownGoodStore(tmp_path / "sup")
    assert store.load().revision == ""
    first = store.mark("aaa", {"ready": True})
    second = store.mark("bbb", {"ready": True})
    assert first.previous == ""
    assert second.previous == "aaa"
    store.record(DeploymentReceipt(kind="promotion", revision="bbb", outcome="healthy"))
    assert store.history()[-1]["outcome"] == "healthy"


def test_control_channel_round_trip(tmp_path: Path) -> None:
    channel = ControlChannel(tmp_path / "sup")
    assert channel.take() is None
    channel.request("restart", reason="promotion", promotion_id="p1", expected_revision="abc")
    taken = channel.take()
    assert taken is not None and taken.action == "restart" and taken.promotion_id == "p1"
    assert channel.take() is None  # consumed
    channel.write_status({"phase": "ready"})
    assert channel.read_status()["phase"] == "ready"


def _supervisor(repo: Path) -> Supervisor:
    config = SupervisorConfig(repository=repo, python="python", open_browser=False, voice=False, port=0)
    return Supervisor(config)


def test_rollback_reverts_to_known_good_and_keeps_history(repo: Path) -> None:
    known_good = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "jarvis" / "serve.py").write_text("raise SystemExit('broken')\n", encoding="utf-8")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qam", "bad candidate")
    bad = _git(repo, "rev-parse", "HEAD").stdout.strip()

    sup = _supervisor(repo)
    ok, detail = sup._rollback(known_good)
    assert ok, detail
    assert (repo / "jarvis" / "serve.py").read_text(encoding="utf-8") == "print('serve')\n"
    log = _git(repo, "log", "--oneline").stdout
    assert "bad candidate" in log and "Revert" in log, "history is kept, the bad commit is undone by a revert"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() not in {known_good, bad}


def test_rollback_stashes_uncommitted_work_instead_of_destroying_it(repo: Path) -> None:
    known_good = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "jarvis" / "serve.py").write_text("raise SystemExit('broken')\n", encoding="utf-8")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qam", "bad candidate")
    (repo / "notes.txt").write_text("owner's unsaved work", encoding="utf-8")

    ok, detail = _supervisor(repo)._rollback(known_good)
    assert ok, detail
    assert "stashed" in detail
    assert not (repo / "notes.txt").exists()
    assert "owner's unsaved work" in _git(repo, "stash", "show", "-p", "--include-untracked").stdout


def test_boot_loop_guard_holds_after_max_failures(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sup = _supervisor(repo)
    sup.config.max_failures = 2
    now = time.monotonic()
    sup.failures = [now, now]
    monkeypatch.setattr(sup.status_page, "start", lambda: True)
    monkeypatch.setattr(sup.status_page, "stop", lambda: None)
    assert sup._start_and_verify(None) == "held"
    assert sup.phase == "held"
    assert sup.known_good.history()[-1]["outcome"] == "held"


def test_exit_codes_are_distinct() -> None:
    assert EXIT_RESTART_REQUESTED != EXIT_SHUTDOWN_REQUESTED


def test_lifecycle_health_requires_real_generation() -> None:
    from service.core import JarvisCore

    core = JarvisCore()
    health = core.lifecycle.health()
    assert health["ready"] is False
    assert "loading" in health["detail"]
    core.lifecycle.mark("fast_local", True, "OK")
    assert core.lifecycle.health()["ready"] is True
    core.lifecycle.mark("fast_local", False, "boom")
    health = core.lifecycle.health()
    assert health["ready"] is False and "boom" in health["detail"]


def test_lifecycle_restart_refused_when_unsupervised(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.core import JarvisCore

    monkeypatch.delenv("ZEUS_SUPERVISED", raising=False)
    core = JarvisCore()
    result = core.lifecycle.request_restart("test")
    assert result["ok"] is False and result["supervised"] is False
    assert not core.lifecycle.exit_event.is_set()


def test_lifecycle_restart_writes_control_and_saves_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.core import ConversationTurn, JarvisCore

    monkeypatch.setenv("ZEUS_SUPERVISED", "1")
    monkeypatch.setenv("ZEUS_SUPERVISOR_DIR", str(tmp_path / "sup"))
    monkeypatch.setenv("JARVIS_STATE_ROOT", str(tmp_path / "state"))
    core = JarvisCore()
    core._history.append(ConversationTurn(role="user", text="update yourself", at="now"))
    result = core.lifecycle.request_restart("promotion", promotion_id="p9", expected_revision="deadbeef")
    assert result["ok"] and core.lifecycle.exit_code == EXIT_RESTART_REQUESTED
    request = json.loads((tmp_path / "sup" / "control" / "request.json").read_text(encoding="utf-8"))
    assert request["action"] == "restart" and request["promotion_id"] == "p9"
    saved = json.loads((tmp_path / "state" / "conversation_resume.json").read_text(encoding="utf-8"))
    assert saved["turns"][0]["text"] == "update yourself"

    fresh = JarvisCore()
    resumed = fresh.lifecycle.restore_conversation()
    assert resumed["turns"] == 1 and fresh.history[0].text == "update yourself"
    assert not (tmp_path / "state" / "conversation_resume.json").exists()
    assert core.lifecycle.exit_event.wait(timeout=3)
