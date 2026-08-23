from __future__ import annotations

from pathlib import Path

import pytest

from development.edit_engine import EditBudget
from development.repository_engineer import (
    RepositoryEngineer,
    SelfImprovementGoal,
    _patch_schema,
)


class FakeBrain:
    pass


def engineer(tmp_path):
    return RepositoryEngineer(
        brain=FakeBrain(),
        worktree_root=tmp_path / "worktrees",
    )


def goal():
    return SelfImprovementGoal(
        objective="minimal edit",
        allowed_paths=["cli.py"],
    )


def test_schema_offers_targeted_edits_and_bounds_whole_file_rewrites():
    """Targeted edits are the primary verb; a rewrite is possible but bounded.

    The earlier version of this test asserted that ``content`` was absent from
    the schema altogether, as a blunt way of stopping a weak model from
    rewriting a large file.  The bound now lives where it belongs -- in
    :class:`~development.edit_engine.EditBudget` -- so the schema can also
    express legitimate small rewrites and new-file creation.
    """

    item = _patch_schema()["properties"]["files"]["items"]

    assert "search" in item["properties"]
    assert "replace" in item["properties"]
    assert set(item["properties"]["op"]["enum"]) == {
        "replace",
        "insert_before",
        "insert_after",
        "rewrite",
    }

    # The real protection against whole-file clobbering.
    assert EditBudget().max_rewrite_chars <= 12000


def test_exact_edit_changes_only_target(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    path = root / "cli.py"
    path.write_text(
        'if command in {"/quit", "/exit"}:\n'
        '    return\n',
        encoding="utf-8",
    )

    eng = engineer(tmp_path)

    eng._apply_proposal(
        root,
        root,
        goal(),
        {
            "files": [{
                "path": "cli.py",
                "search": '{"/quit", "/exit"}',
                "replace": '{"/quit", "/exit", "/bye"}',
            }],
            "new_files": [],
            "deleted_files": [],
        },
        {},
    )

    result = path.read_text(encoding="utf-8")

    assert '{"/quit", "/exit", "/bye"}' in result
    assert "return" in result


def test_non_unique_search_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    (root / "cli.py").write_text(
        "x = 1\nx = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly once"):
        engineer(tmp_path)._apply_proposal(
            root,
            root,
            goal(),
            {
                "files": [{
                    "path": "cli.py",
                    "search": "x = 1",
                    "replace": "x = 2",
                }],
                "new_files": [],
                "deleted_files": [],
            },
            {},
        )


def test_whole_file_sized_edit_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    large = "a" * 4001
    (root / "cli.py").write_text(large, encoding="utf-8")

    with pytest.raises(ValueError, match="edit too large"):
        engineer(tmp_path)._apply_proposal(
            root,
            root,
            goal(),
            {
                "files": [{
                    "path": "cli.py",
                    "search": large,
                    "replace": "replacement",
                }],
                "new_files": [],
                "deleted_files": [],
            },
            {},
        )


def test_new_file_cannot_replace_existing_file(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    (root / "cli.py").write_text("original\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot replace existing"):
        engineer(tmp_path)._apply_proposal(
            root,
            root,
            goal(),
            {
                "files": [],
                "new_files": [{
                    "path": "cli.py",
                    "content": "",
                }],
                "deleted_files": [],
            },
            {},
        )
