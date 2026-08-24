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


def test_a_truncated_read_does_not_poison_the_staleness_baseline(tmp_path):
    """A digest over a partial read can never match the real file again.

    Recorded from a live self-patch run: once jarvis/cli.py grew past the
    8000-character read limit, investigation stored a hash of the truncated
    text, every subsequent edit was rejected as `stale_context`, and four
    development cycles produced no change at all.
    """

    from development.repository_engineer import RepositoryContextManager

    root = tmp_path / "repo"
    root.mkdir()
    big = "\n".join(f"line_{index} = {index}" for index in range(2000)) + "\n"
    (root / "big.py").write_text(big, encoding="utf-8")
    (root / "small.py").write_text("value = 1\n", encoding="utf-8")

    engineer = make_engineer(tmp_path)
    manager = RepositoryContextManager()

    content, complete = engineer.read_file_complete(root, "big.py")
    assert not complete, "the fixture must actually exceed the read limit"
    manager.add_file("big.py", content, complete=complete)

    content, complete = engineer.read_file_complete(root, "small.py")
    assert complete
    manager.add_file("small.py", content, complete=complete)

    assert "big.py" not in manager.file_hashes, "no baseline is better than a wrong one"
    assert "small.py" in manager.file_hashes


def test_a_large_file_can_still_be_edited(tmp_path):
    """The end the previous test protects: the edit actually lands."""

    root = tmp_path / "repo"
    root.mkdir()
    body = "\n".join(f"line_{index} = {index}" for index in range(2000))
    (root / "cli.py").write_text(f"{body}\n\nMARKER = 'before'\n", encoding="utf-8")

    engineer = make_engineer(tmp_path)
    content, complete = engineer.read_file_complete(root, "cli.py")
    context = {"inspected_files": {}, "file_hashes": {}}
    from development.repository_engineer import RepositoryContextManager

    manager = RepositoryContextManager()
    manager.add_file("cli.py", content, complete=complete)
    context["file_hashes"] = dict(manager.file_hashes)

    engineer._apply_proposal(
        root,
        root,
        SelfImprovementGoal(objective="change the marker", allowed_paths=["cli.py"]),
        {
            "analysis": "update the marker",
            "files": [{"path": "cli.py", "search": "MARKER = 'before'", "replace": "MARKER = 'after'"}],
            "new_files": [],
            "deleted_files": [],
        },
        context,
    )

    assert "MARKER = 'after'" in (root / "cli.py").read_text(encoding="utf-8")


def test_the_focus_window_scales_with_the_measured_context():
    """Prompting as though the machine had not been measured is the worst of both.

    A live self-patch run missed its anchor thirteen times because the model was
    shown a 5000-character keyhole view of a 430-line file, while its context
    window -- measured by the tuner -- was 24576 tokens.
    """

    from development.repository_engineer import ModelRequestBudget

    class Brain:
        pass

    small = RepositoryEngineer(brain=Brain(), context_budget=ModelRequestBudget(context_window=8192))
    large = RepositoryEngineer(brain=Brain(), context_budget=ModelRequestBudget(context_window=24576))

    assert large._focus_budget() > small._focus_budget()
    assert small._focus_budget() >= 5000, "never smaller than the old fixed window"
    assert large._focus_budget() <= 48000, "and bounded, so the prompt still leaves room to answer"


def test_a_large_file_is_shown_in_full_when_the_context_allows(tmp_path):
    """The point of the budget: an anchor can only be copied from what is shown."""

    from development.repository_engineer import ModelRequestBudget, _focused_edit_context

    root = tmp_path / "repo"
    root.mkdir()
    body = "\n".join(f"def handler_{index}():\n    return {index}\n" for index in range(120))
    (root / "cli.py").write_text(f"{body}\nMARKER_TO_FIND = 'here'\n", encoding="utf-8")

    goal = SelfImprovementGoal(objective="change MARKER_TO_FIND", allowed_paths=["cli.py"])
    engineer = RepositoryEngineer(brain=object(), context_budget=ModelRequestBudget(context_window=24576))

    shown = _focused_edit_context(root, goal, {}, max_chars=engineer._focus_budget())
    assert "MARKER_TO_FIND = 'here'" in shown, "the line to edit must actually appear"
    assert "00001:" not in shown, "and unnumbered, so the anchor can be copied verbatim"
