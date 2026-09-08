"""The Release view has to put the newest candidate first.

A candidate directory is named ``<revision12>-<timestamp>-<uuid>``, and the
listing sorted those names alphabetically -- which orders them by git revision
hash, a value with no relation to time at all. Observed on this machine: with
sixteen candidates present, the one built minutes earlier sat eighth of ten and
two builds from a week before appeared last, where the UI's ``.reverse()`` put
them at the top.

"Which one do I promote?" is the question that view exists to answer, and the
ordering was answering it wrongly.
"""

from __future__ import annotations

import json
from pathlib import Path

from deployment.release import ReleaseManager


def _candidate(root: Path, name: str, *, built_at: str | None, verified: bool = True) -> Path:
    target = root / "candidates" / name / "ZEUS"
    target.mkdir(parents=True, exist_ok=True)
    (target / "ZEUS.exe").write_bytes(b"MZ")
    if built_at is not None:
        (target / "VERSION.json").write_text(
            json.dumps({"product": "ZEUS", "revision": name.split("-")[0], "built_at": built_at}),
            encoding="utf-8",
        )
    if verified:
        (target.parent / "VERIFIED.json").write_text(json.dumps({"outcome": "verified"}), encoding="utf-8")
    return target.parent


def _manager(tmp_path: Path) -> ReleaseManager:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return ReleaseManager(repo, dist=tmp_path / "dist")


def test_the_newest_candidate_is_first(tmp_path: Path) -> None:
    """The exact shape seen live: hash order buries today's build."""

    dist = tmp_path / "dist"
    # Names deliberately chosen so alphabetical order and time order disagree,
    # the way real revision hashes do.
    _candidate(dist, "73f5e6f6998d-20260908T055949Z-14ae44", built_at="2026-09-08T06:00:19+00:00")
    _candidate(dist, "df0a34fb0807-20260908T113235Z-ee06fb", built_at="2026-09-08T11:32:50+00:00")
    _candidate(dist, "f8e0c3984973-20260901T234204Z-811759", built_at="2026-09-01T23:42:04+00:00")
    _candidate(dist, "f93a6a907c8e-20260901T175054Z-3a1de6", built_at="2026-09-01T17:50:54+00:00")

    candidates = _manager(tmp_path).status()["candidates"]

    assert [c["id"][:12] for c in candidates] == ["df0a34fb0807", "73f5e6f6998d", "f8e0c3984973", "f93a6a907c8e"]
    assert candidates[0]["verified"] is True


def test_the_timestamp_in_the_name_is_used_when_there_is_no_version_file(tmp_path: Path) -> None:
    """A candidate whose build died before VERSION.json still has its id."""

    dist = tmp_path / "dist"
    _candidate(dist, "aaaaaaaaaaaa-20260908T120000Z-000001", built_at=None)
    _candidate(dist, "zzzzzzzzzzzz-20260101T000000Z-000002", built_at=None)

    candidates = _manager(tmp_path).status()["candidates"]

    assert [c["id"][:12] for c in candidates] == ["aaaaaaaaaaaa", "zzzzzzzzzzzz"]


def test_a_candidate_with_no_usable_time_sorts_last_rather_than_first(tmp_path: Path) -> None:
    """An unreadable candidate must not be offered as the newest thing to promote."""

    dist = tmp_path / "dist"
    _candidate(dist, "not-a-candidate-name", built_at=None)
    _candidate(dist, "bbbbbbbbbbbb-20260908T120000Z-000003", built_at="2026-09-08T12:00:00+00:00")

    candidates = _manager(tmp_path).status()["candidates"]

    assert candidates[0]["id"].startswith("bbbbbbbbbbbb")
    assert candidates[-1]["id"] == "not-a-candidate-name"


def test_only_ten_are_returned_and_they_are_the_ten_newest(tmp_path: Path) -> None:
    """Truncation has to keep the newest, which is what it was dropping."""

    dist = tmp_path / "dist"
    for day in range(1, 15):
        _candidate(dist, f"{day:012d}-202609{day:02d}T120000Z-0000{day:02d}",
                   built_at=f"2026-09-{day:02d}T12:00:00+00:00")

    candidates = _manager(tmp_path).status()["candidates"]

    assert len(candidates) == 10
    assert candidates[0]["id"].startswith("000000000014")
    assert candidates[-1]["id"].startswith("000000000005")
