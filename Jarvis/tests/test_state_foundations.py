"""Backup/restore, verified experience, device context."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from development.experience import Experience, ExperienceStore, from_selfdev_mission
from runtime.backup import BackupManager
from runtime.device_context import current_context, set_context


def _state(tmp_path: Path) -> Path:
    root = tmp_path / "Jarvis" / "data" / "jarvis"
    (root / "projects").mkdir(parents=True)
    (root / "projects" / "p1.json").write_text('{"goal": "x"}')
    (root / "owner").mkdir()
    (root / "owner" / "corrections.jsonl").write_text('{"id": 1}\n')
    (root / "secrets").mkdir()
    (root / "secrets" / "spotify.json").write_text('{"client_secret": "nope"}')
    (root / "capabilities").mkdir()
    (root / "capabilities" / "registry.json").write_text('{"capabilities": []}')
    (root / "capabilities" / "api_token.txt").write_text("secret-looking")
    return root


def test_backup_excludes_secrets_verifies_and_restores(tmp_path):
    root = _state(tmp_path)
    bm = BackupManager(root, backups=tmp_path / "backups", repository=tmp_path / "Jarvis")
    report = bm.create(label="test")
    assert report["ok"] and report["verified"] and report["files"] == 3
    assert report["skipped_secret_like"] == ["capabilities/api_token.txt"]
    import zipfile

    names = zipfile.ZipFile(report["path"]).namelist()
    assert not any("secrets" in n for n in names) and "MANIFEST.json" in names
    # damage the state, restore it
    (root / "projects" / "p1.json").write_text("garbage")
    assert bm.restore(report["path"])["ok"] is False  # needs confirm
    restored = bm.restore(report["path"], confirm=True)
    assert restored["ok"] and restored["restored"] == 3
    assert json.loads((root / "projects" / "p1.json").read_text()) == {"goal": "x"}
    assert Path(restored["kept_aside"]).is_dir()
    # a corrupted archive does not verify
    data = bytearray(Path(report["path"]).read_bytes())
    Path(report["path"]).write_bytes(bytes(data[:-40]) + b"x" * 40)
    assert bm.verify(report["path"])["ok"] is False


def test_experience_is_compact_relevant_and_measured(tmp_path):
    store = ExperienceStore(tmp_path / "exp.jsonl")
    m1 = SimpleNamespace(request="Zeus, show your uptime in the header of your UI", area="ui", outcome="promoted",
                         changed_files=["ui/app.js", "ui/index.html"], investigation={"files": ["ui/index.html", "ui/app.js"]},
                         events=[{"phase": "BUILD", "error": "anchor 'header' not found"}], expert={"provider": "claude_code"},
                         verification={"tests": ["tests/test_desktop_window.py"], "detail": "kernel import ok; verify_ui ok"},
                         timings={"investigate": 13.0, "build": 692.0, "escalate": 591.0}, model_calls=9, local_attempts=1,
                         escalated=True, mission_id="m1", expected_revision="abc123def456")
    e1 = store.add(from_selfdev_mission(m1))
    assert e1.subsystem == "ui" and e1.outcome == "promoted" and e1.relevant_files == ["ui/app.js", "ui/index.html"]
    assert "anchor" in e1.failed_hypotheses[0] and "claude_code" in e1.strategy
    guidance = store.guidance("Zeus, show the GPU load in your header", subsystem="ui")
    assert "ui/app.js" in guidance and "anchor" in guidance and len(guidance) < 900
    assert store.guidance("recherchiere die Bedeutung von Photosynthese") == ""
    assert store.list()[0].used == 1
    m2 = SimpleNamespace(**{**m1.__dict__, "request": "Zeus, show the mission count in your header", "timings": {"investigate": 9.0, "build": 300.0}, "mission_id": "m2", "escalated": False, "model_calls": 4})
    store.add(from_selfdev_mission(m2))
    cmp = store.compare("show something in your header", subsystem="ui")
    assert [m["mission_id"] for m in cmp["missions"]] == ["m1", "m2"]
    assert cmp["trend"]["total_s"]["change"] < 0 and cmp["trend"]["model_calls"]["last"] == 4


def test_device_context_describes_this_machine_and_takes_owner_facts(tmp_path):
    core = SimpleNamespace(kernel=SimpleNamespace(state_root=tmp_path / "state"),
                           lifecycle=SimpleNamespace(stages={"voice": {"ok": True}, "recogniser": {"ok": False}}, desktop=None),
                           capabilities=SimpleNamespace(registry=SimpleNamespace(all=lambda: [SimpleNamespace(capability_id="music.provider.spotify")])))
    ctx = current_context(core)
    assert ctx.device_type == "desktop" and ctx.speaker and not ctx.microphone
    assert "speaker" in ctx.available and "microphone" not in ctx.available
    assert ctx.capabilities == ["music.provider.spotify"]
    again = set_context(core, room="Wohnzimmer", name="Desktop")
    assert again.room == "Wohnzimmer" and current_context(core).room == "Wohnzimmer"
    assert (tmp_path / "state" / "devices" / "this_device.json").is_file()
