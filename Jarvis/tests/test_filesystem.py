"""The File Galaxy's backend: real paths only, bounded listings, live watching."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from service.filesystem import MAX_ENTRIES, FilesystemIndex, categorize, list_drives


@pytest.fixture
def fs(tmp_path):
    events = []
    index = FilesystemIndex(emit=events.append, log=lambda _m: None)
    index._events = events  # type: ignore[attr-defined]
    yield index
    index.unwatch()


def test_drives_are_real_and_d_is_primary():
    drives = list_drives()
    assert drives, "at least one drive"
    for d in drives:
        assert Path(d["path"]).exists()
    if any(d["path"].upper().startswith("D:") for d in drives):
        assert next(d for d in drives if d["path"].upper().startswith("D:"))["primary"] is True


def test_listing_returns_only_real_entries_with_metadata(fs, tmp_path):
    (tmp_path / "Projekte").mkdir()
    (tmp_path / "Projekte" / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    out = fs.list(str(tmp_path))
    assert out["ok"] and out["path"] == str(tmp_path)
    names = {e["name"]: e for e in out["entries"]}
    assert names["Projekte"]["type"] == "dir" and names["Projekte"]["children_count"] == 1
    assert names["notes.txt"]["type"] == "file" and names["notes.txt"]["size"] == 2
    for entry in out["entries"]:
        assert Path(entry["path"]).exists(), "nothing is invented"
    assert names["Projekte"]["category"] == "PROJECTS"


def test_listing_is_bounded_and_marks_truncation(fs, tmp_path):
    for i in range(MAX_ENTRIES + 20):
        (tmp_path / f"f{i:04}.txt").write_text("x")
    out = fs.list(str(tmp_path))
    assert len(out["entries"]) == MAX_ENTRIES and out["truncated"] is True


def test_missing_paths_are_an_error_not_a_fabrication(fs, tmp_path):
    out = fs.list(str(tmp_path / "nope"))
    assert out["ok"] is False and "does not exist" in out["error"]


def test_cache_serves_and_invalidation_refreshes(fs, tmp_path):
    fs.list(str(tmp_path))
    (tmp_path / "new").mkdir()
    cached = fs.list(str(tmp_path))
    assert all(e["name"] != "new" for e in cached["entries"]), "TTL cache"
    fs.invalidate(str(tmp_path))
    fresh = fs.list(str(tmp_path))
    assert any(e["name"] == "new" for e in fresh["entries"])


@pytest.mark.skipif(sys.platform != "win32", reason="ReadDirectoryChangesW")
def test_watcher_sees_create_rename_delete_live(fs, tmp_path):
    assert fs.watch(str(tmp_path))["ok"]
    time.sleep(0.3)
    target = tmp_path / "NewFolder"
    target.mkdir()
    deadline = time.time() + 5
    while time.time() < deadline and not fs._events:
        time.sleep(0.1)
    assert fs._events, "creation was seen"
    changed = fs._events[-1]["changed"]
    assert any(str(tmp_path).lower() in c["path"].lower() for c in changed)
    fs._events.clear()
    target.rename(tmp_path / "RenamedFolder")
    deadline = time.time() + 5
    while time.time() < deadline and not fs._events:
        time.sleep(0.1)
    assert fs._events, "rename was seen"
    fs._events.clear()
    (tmp_path / "RenamedFolder").rmdir()
    deadline = time.time() + 5
    while time.time() < deadline and not fs._events:
        time.sleep(0.1)
    assert fs._events, "deletion was seen"
    # events invalidate the cache so the next list is fresh
    out = fs.list(str(tmp_path))
    assert all(e["name"] != "RenamedFolder" for e in out["entries"])


def test_watch_is_idempotent_and_bursts_are_batched(fs, tmp_path):
    fs.watch(str(tmp_path))
    again = fs.watch(str(tmp_path))
    assert again.get("already") is True
    if sys.platform == "win32":
        time.sleep(0.3)
        for i in range(30):
            (tmp_path / f"burst{i}.txt").write_text("x")
        deadline = time.time() + 5
        while time.time() < deadline and not fs._events:
            time.sleep(0.1)
        time.sleep(1.0)
        assert 0 < len(fs._events) <= 6, f"debounced batches, not 30 events: {len(fs._events)}"


def test_open_in_explorer_validates_the_path(fs, tmp_path):
    out = FilesystemIndex.open_in_explorer(str(tmp_path / "missing"))
    assert out["ok"] is False and "does not exist" in out["error"]


def test_categorize_is_metadata_only():
    assert categorize("Games") == "GAMES"
    assert categorize("Dokumente") == "DOCUMENTS"
    assert categorize("random-thing") == "OTHER"
