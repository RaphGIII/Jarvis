from __future__ import annotations

from pathlib import Path

import pytest

from development.repository_engineer import (
    RepositoryEngineer,
    SelfImprovementGoal,
    _focused_edit_context,
)


class FakeBrain:
    pass


def make_engineer(tmp_path):
    return RepositoryEngineer(
        brain=FakeBrain(),
        worktree_root=tmp_path / "worktrees",
    )


def make_goal():
    return SelfImprovementGoal(
        objective="Add /bye while keeping /quit and /exit working",
        allowed_paths=["cli.py"],
    )


def test_focused_context_finds_relevant_code_near_end(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    filler = "\n".join(f"x_{i} = {i}" for i in range(1200))
    target = '\nif command in {"/quit", "/exit", "quit", "exit"}:\n    return\n'

    (root / "cli.py").write_text(
        filler + target,
        encoding="utf-8",
    )

    result = _focused_edit_context(
        root,
        make_goal(),
        {},
        max_chars=5000,
    )

    assert "/quit" in result
    assert "/exit" in result


def test_search_mismatch_is_regenerated_automatically(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    path = root / "cli.py"
    path.write_text(
        'if command in {"/quit", "/exit"}:\n'
        '    return\n',
        encoding="utf-8",
    )

    eng = make_engineer(tmp_path)
    calls = []

    corrected = {
        "analysis": "corrected",
        "files": [{
            "path": "cli.py",
            "search": '{"/quit", "/exit"}',
            "replace": '{"/quit", "/exit", "/bye"}',
        }],
        "new_files": [],
        "deleted_files": [],
    }

    def regenerate(prompt):
        calls.append(prompt)
        return corrected

    eng._generate_patch_bundle = regenerate

    eng._apply_proposal(
        root,
        root,
        make_goal(),
        {
            "analysis": "bad first attempt",
            "files": [{
                "path": "cli.py",
                "search": "THIS TEXT DOES NOT EXIST ANYWHERE",
                "replace": "replacement",
            }],
            "new_files": [],
            "deleted_files": [],
        },
        {},
    )

    assert calls
    assert "/bye" in path.read_text(encoding="utf-8")


def test_failed_multi_edit_is_transactional(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    path = root / "cli.py"
    original = "x = 1\ny = 2\ny = 2\n"
    path.write_text(original, encoding="utf-8")

    eng = make_engineer(tmp_path)

    with pytest.raises(ValueError):
        eng._apply_proposal(
            root,
            root,
            make_goal(),
            {
                "analysis": "two edits",
                "files": [
                    {
                        "path": "cli.py",
                        "search": "x = 1",
                        "replace": "x = 9",
                    },
                    {
                        "path": "cli.py",
                        "search": "y = 2",
                        "replace": "y = 8",
                    },
                ],
                "new_files": [],
                "deleted_files": [],
            },
            {},
        )

    assert path.read_text(encoding="utf-8") == original


def test_crlf_is_preserved(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    path = root / "cli.py"
    path.write_bytes(
        b'x = 1\r\n'
        b'y = 2\r\n'
    )

    make_engineer(tmp_path)._apply_proposal(
        root,
        root,
        make_goal(),
        {
            "analysis": "small edit",
            "files": [{
                "path": "cli.py",
                "search": "x = 1",
                "replace": "x = 3",
            }],
            "new_files": [],
            "deleted_files": [],
        },
        {},
    )

    data = path.read_bytes()

    assert b"x = 3\r\n" in data
    assert b"y = 2\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_changed_line_budget_blocks_file_wipe(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()

    text = "\n".join(f"x_{i} = {i}" for i in range(50)) + "\n"
    path = root / "cli.py"
    path.write_text(text, encoding="utf-8")

    monkeypatch.setenv("JARVIS_BUILD_MAX_CHANGED_LINES", "20")

    with pytest.raises(ValueError, match="changed-line budget"):
        make_engineer(tmp_path)._apply_proposal(
            root,
            root,
            make_goal(),
            {
                "analysis": "bad wipe",
                "files": [{
                    "path": "cli.py",
                    "search": text,
                    "replace": "",
                }],
                "new_files": [],
                "deleted_files": [],
            },
            {},
        )

    assert path.read_text(encoding="utf-8") == text
