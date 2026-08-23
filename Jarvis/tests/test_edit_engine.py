from __future__ import annotations

import pytest

from development.edit_engine import (
    EditBudget,
    EditEngine,
    EditError,
    EditOp,
    PathPolicy,
    parse_bundle,
    stable_hash,
)


def _engine(root, **kwargs):
    policy = PathPolicy(
        root,
        allowed_paths=kwargs.pop("allowed_paths", None),
        protected_paths=kwargs.pop("protected_paths", None),
    )
    return EditEngine(policy, **kwargs)


def _write(root, name, text):
    r"""Write bytes verbatim.

    ``Path.write_text`` translates ``\n`` to ``os.linesep`` on Windows, which
    would silently turn every LF fixture into a CRLF one and make the
    newline-preservation tests meaningless.
    """

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# ---------------------------------------------------------------- parsing


def test_parse_accepts_search_replace_dialect():
    plan = parse_bundle({"analysis": "a", "files": [{"path": "a.py", "search": "x", "replace": "y"}]})
    assert [op.op for op in plan.operations] == [EditOp.REPLACE]


def test_parse_accepts_whole_file_content_dialect():
    plan = parse_bundle({"analysis": "a", "files": [{"path": "a.py", "content": "x\n"}]})
    assert [op.op for op in plan.operations] == [EditOp.REWRITE]


def test_parse_accepts_explicit_ops_and_new_and_deleted_files():
    plan = parse_bundle(
        {
            "analysis": "a",
            "files": [{"path": "a.py", "op": "insert_after", "search": "x", "replace": "y"}],
            "new_files": [{"path": "b.py", "content": "b\n"}],
            "deleted_files": ["c.py"],
        }
    )
    assert [op.op for op in plan.operations] == [EditOp.INSERT_AFTER, EditOp.CREATE, EditOp.DELETE]


def test_parse_rejects_empty_bundle():
    with pytest.raises(EditError) as excinfo:
        parse_bundle({"analysis": "a", "files": []})
    assert excinfo.value.kind == "empty_plan"


def test_parse_rejects_edit_without_replacement_but_marks_it_recoverable():
    with pytest.raises(EditError) as excinfo:
        parse_bundle({"analysis": "a", "files": [{"path": "a.py", "search": "x"}]})
    assert excinfo.value.recoverable


# ---------------------------------------------------------------- matching


def test_exact_search_replace(tmp_path):
    _write(tmp_path, "a.py", "def add(a, b):\n    return a - b\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "a - b", "replace": "a + b"}]}))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"


def test_line_number_gutter_in_search_is_stripped(tmp_path):
    """A model that copies from a numbered listing must still land its edit."""

    _write(tmp_path, "a.py", "def add(a, b):\n    return a - b\n")
    plan = parse_bundle(
        {"files": [{"path": "a.py", "search": "00001: def add(a, b):\n00002:     return a - b", "replace": "def add(a, b):\n    return a + b"}]}
    )
    result = _engine(tmp_path).apply(plan)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    assert result.applied[0].match_mode == "line_numbers_stripped"


def test_whitespace_differences_still_match_canonically(tmp_path):
    _write(tmp_path, "a.py", "def add(a, b):\n    return a - b\n")
    plan = parse_bundle({"files": [{"path": "a.py", "search": "def add(a,   b):\n\n        return a - b", "replace": "def add(a, b):\n    return a + b"}]})
    result = _engine(tmp_path).apply(plan)
    assert "a + b" in (tmp_path / "a.py").read_text(encoding="utf-8")
    assert result.applied[0].match_mode == "canonical"


def test_ambiguous_search_is_rejected_as_recoverable(tmp_path):
    _write(tmp_path, "a.py", "x = 1\ny = 2\nx = 1\n")
    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "x = 1", "replace": "x = 9"}]}))
    assert excinfo.value.kind == "ambiguous_search"
    assert excinfo.value.recoverable
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 2\nx = 1\n"


def test_occurrence_selects_among_identical_matches(tmp_path):
    _write(tmp_path, "a.py", "x = 1\ny = 2\nx = 1\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "x = 1", "replace": "x = 9", "occurrence": 2}]}))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 2\nx = 9\n"


def test_unmatched_search_is_recoverable(tmp_path):
    _write(tmp_path, "a.py", "def add(a, b):\n    return a - b\n")
    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "def subtract(q, r):", "replace": "z"}]}))
    assert excinfo.value.kind in {"no_unique_match", "ambiguous_search"}
    assert excinfo.value.recoverable


# ---------------------------------------------------------------- inserts


def test_insert_after_anchor(tmp_path):
    _write(tmp_path, "a.py", "import os\n\ndef f():\n    pass\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "op": "insert_after", "search": "import os", "replace": "import sys"}]}))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "import os\nimport sys\n\ndef f():\n    pass\n"


def test_insert_before_anchor(tmp_path):
    _write(tmp_path, "a.py", "def f():\n    pass\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "op": "insert_before", "search": "def f():", "replace": "# comment"}]}))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "# comment\ndef f():\n    pass\n"


# ---------------------------------------------------------------- lifecycle


def test_create_and_delete(tmp_path):
    _write(tmp_path, "old.py", "gone\n")
    _engine(tmp_path).apply(parse_bundle({"files": [], "new_files": [{"path": "pkg/new.py", "content": "fresh\n"}], "deleted_files": ["old.py"]}))
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "fresh\n"
    assert not (tmp_path / "old.py").exists()


def test_create_over_existing_is_rejected(tmp_path):
    _write(tmp_path, "a.py", "keep\n")
    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(parse_bundle({"files": [], "new_files": [{"path": "a.py", "content": "clobber\n"}]}))
    assert excinfo.value.kind == "create_over_existing"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "keep\n"


def test_rewrite_of_missing_file_creates_it(tmp_path):
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "content": "made\n"}]}))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "made\n"


def test_oversized_rewrite_is_refused_in_favour_of_targeted_edits(tmp_path):
    _write(tmp_path, "a.py", "x\n")
    engine = _engine(tmp_path, budget=EditBudget(max_rewrite_chars=10))
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "a.py", "content": "y" * 50}]}))
    assert excinfo.value.kind == "rewrite_too_large"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x\n"


# ---------------------------------------------------------------- safety


def test_protected_path_is_refused_and_file_is_byte_identical(tmp_path):
    path = _write(tmp_path, "tests/test_x.py", "assert True\n")
    before = path.read_bytes()
    engine = _engine(tmp_path, protected_paths=["tests"])
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "tests/test_x.py", "content": "assert False\n"}]}))
    assert excinfo.value.kind == "protected_path"
    assert not excinfo.value.recoverable
    assert path.read_bytes() == before


def test_path_outside_allow_list_is_refused(tmp_path):
    _write(tmp_path, "src/a.py", "a\n")
    _write(tmp_path, "other/b.py", "b\n")
    engine = _engine(tmp_path, allowed_paths=["src"])
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "other/b.py", "content": "c\n"}]}))
    assert excinfo.value.kind == "path_not_allowed"
    assert (tmp_path / "other" / "b.py").read_text(encoding="utf-8") == "b\n"


@pytest.mark.parametrize("escape", ["../outside.py", "/abs.py", "a/../../outside.py"])
def test_path_escapes_are_refused(tmp_path, escape):
    with pytest.raises(EditError):
        _engine(tmp_path).apply(parse_bundle({"files": [{"path": escape, "content": "x\n"}]}))


def test_stale_context_is_detected(tmp_path):
    _write(tmp_path, "a.py", "current\n")
    engine = _engine(tmp_path, expected_hashes={"a.py": stable_hash("what the model was shown\n")})
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "a.py", "search": "current", "replace": "new"}]}))
    assert excinfo.value.kind == "stale_context"
    assert excinfo.value.recoverable
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "current\n"


def test_changed_line_budget(tmp_path):
    _write(tmp_path, "a.py", "x\n")
    engine = _engine(tmp_path, budget=EditBudget(max_changed_lines=2, max_rewrite_chars=100000))
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "a.py", "content": "\n".join(str(i) for i in range(50))}]}))
    assert excinfo.value.kind == "changed_line_budget"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x\n"


# ---------------------------------------------------------------- atomicity


def test_failed_multi_edit_leaves_no_partial_mutation(tmp_path):
    """The headline atomicity guarantee: op 1 is good, op 2 is not."""

    a = _write(tmp_path, "a.py", "value = 1\n")
    b = _write(tmp_path, "b.py", "value = 2\n")
    before = {"a": a.read_bytes(), "b": b.read_bytes()}

    plan = parse_bundle(
        {
            "files": [
                {"path": "a.py", "search": "value = 1", "replace": "value = 11"},
                {"path": "b.py", "search": "this anchor does not exist anywhere", "replace": "value = 22"},
            ]
        }
    )
    with pytest.raises(EditError):
        _engine(tmp_path).apply(plan)

    assert a.read_bytes() == before["a"]
    assert b.read_bytes() == before["b"]


def test_failed_plan_does_not_leave_created_files_behind(tmp_path):
    _write(tmp_path, "a.py", "value = 1\n")
    plan = parse_bundle(
        {
            "files": [{"path": "a.py", "search": "nope nope nope", "replace": "x"}],
            "new_files": [{"path": "leftover.py", "content": "x\n"}],
        }
    )
    with pytest.raises(EditError):
        _engine(tmp_path).apply(plan)
    assert not (tmp_path / "leftover.py").exists()


def test_multi_file_success_applies_everything(tmp_path):
    _write(tmp_path, "a.py", "value = 1\n")
    _write(tmp_path, "b.py", "value = 2\n")
    result = _engine(tmp_path).apply(
        parse_bundle(
            {
                "files": [
                    {"path": "a.py", "search": "value = 1", "replace": "value = 11"},
                    {"path": "b.py", "search": "value = 2", "replace": "value = 22"},
                ],
                "new_files": [{"path": "c.py", "content": "value = 33\n"}],
            }
        )
    )
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "value = 11\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "value = 22\n"
    assert (tmp_path / "c.py").read_text(encoding="utf-8") == "value = 33\n"
    assert sorted(result.changed_paths()) == ["a.py", "b.py", "c.py"]


def test_sequential_edits_to_one_file_compose(tmp_path):
    _write(tmp_path, "a.py", "one = 1\ntwo = 2\n")
    _engine(tmp_path).apply(
        parse_bundle(
            {
                "files": [
                    {"path": "a.py", "search": "one = 1", "replace": "one = 111"},
                    {"path": "a.py", "search": "two = 2", "replace": "two = 222"},
                ]
            }
        )
    )
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "one = 111\ntwo = 222\n"


# ---------------------------------------------------------------- encoding


def test_crlf_and_bom_are_preserved(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes("﻿value = 1\r\nother = 2\r\n".encode("utf-8"))
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "value = 1", "replace": "value = 9"}]}))
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in raw.replace(b"\r\n", b"")  # every LF is still part of a CRLF pair
    assert raw == "﻿value = 9\r\nother = 2\r\n".encode("utf-8")


def test_lf_file_stays_lf(tmp_path):
    path = _write(tmp_path, "a.py", "value = 1\nother = 2\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "value = 1", "replace": "value = 9"}]}))
    assert b"\r" not in path.read_bytes()


# ------------------------------------------- dialects seen from real models

def test_new_content_is_accepted_as_a_synonym_for_content(tmp_path):
    """A live local model emitted `new_content`; rejecting it cost three cycles."""

    _write(tmp_path, "a.py", "old\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "new_content": "fresh\n"}]}))
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "fresh\n"


def test_empty_search_with_a_replacement_is_refused_on_a_file_with_content(tmp_path):
    """An anchorless replacement cannot mean "discard everything else".

    An earlier version of this engine read it as a whole-file rewrite. On a
    live run that deleted a 243-line module, so the reading is now restricted
    to files with nothing to lose (see the truncation tests below).
    """

    _write(tmp_path, "a.py", "old\n")
    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(parse_bundle({"files": [{"path": "a.py", "search": "", "replace": "fresh\n"}]}))
    assert excinfo.value.kind == "empty_search"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "old\n"


def test_an_empty_search_rewrite_still_obeys_the_size_budget(tmp_path):
    """Where the reading IS allowed, it is still bounded like any rewrite."""

    _write(tmp_path, "blank.py", "")  # an existing file, so this is a rewrite, not a create
    engine = _engine(tmp_path, budget=EditBudget(max_rewrite_chars=10))
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "blank.py", "search": "", "replace": "y" * 200}]}))
    assert excinfo.value.kind == "rewrite_too_large"
    assert (tmp_path / "blank.py").read_text(encoding="utf-8") == ""


def test_an_empty_search_rewrite_still_obeys_protected_paths(tmp_path):
    path = _write(tmp_path, "tests/test_x.py", "assert True\n")
    engine = _engine(tmp_path, protected_paths=["tests"])
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "tests/test_x.py", "search": "", "replace": "assert False\n"}]}))
    assert excinfo.value.kind == "protected_path"
    assert path.read_text(encoding="utf-8") == "assert True\n"


def test_the_rejection_message_names_the_keys_it_received(tmp_path):
    """A model can only correct an edit if the error says what was wrong with it."""

    with pytest.raises(EditError) as excinfo:
        parse_bundle({"files": [{"path": "a.py", "notes": "I forgot the content"}]})
    assert "notes" in str(excinfo.value)
    assert excinfo.value.recoverable


def test_new_files_also_accept_the_content_synonyms(tmp_path):
    _engine(tmp_path).apply(parse_bundle({"files": [], "new_files": [{"path": "b.py", "new_content": "made\n"}]}))
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "made\n"


# ------------------------------------- protecting a file from being gutted

def _big_module(lines=60):
    body = "\n".join(f"def function_{index}():\n    return {index}\n" for index in range(lines))
    return f"from __future__ import annotations\n\n{body}"


def test_a_fragment_cannot_replace_a_whole_module(tmp_path):
    """The live failure: a 243-line CLI replaced by a two-line fragment.

    Both size budgets passed, because the *result* was tiny. Nothing bounded
    how much a rewrite may remove, and removal is the direction that destroys
    work.
    """

    original = _big_module()
    path = _write(tmp_path, "cli.py", original)
    fragment = 'if command.lower() in {"/quit", "/exit", "/bye"}:\n    return\n'

    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(parse_bundle({"files": [{"path": "cli.py", "content": fragment}]}))

    assert excinfo.value.kind == "rewrite_truncates_file"
    assert excinfo.value.recoverable, "the model should be asked to send a real edit instead"
    assert path.read_text(encoding="utf-8") == original


def test_an_anchorless_replacement_never_overwrites_an_existing_file(tmp_path):
    """The precise shape the local model emitted, with an empty search."""

    original = _big_module()
    path = _write(tmp_path, "cli.py", original)

    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(
            parse_bundle({"files": [{"path": "cli.py", "search": "", "replace": "x = 1\n"}]})
        )

    assert excinfo.value.kind == "empty_search"
    assert excinfo.value.recoverable
    assert path.read_text(encoding="utf-8") == original


def test_an_anchorless_replacement_still_creates_a_missing_file(tmp_path):
    """The reading is only dangerous when there is content to destroy."""

    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "fresh.py", "search": "", "replace": "x = 1\n"}]}))
    assert (tmp_path / "fresh.py").read_text(encoding="utf-8") == "x = 1\n"


def test_an_anchorless_replacement_may_fill_an_empty_file(tmp_path):
    _write(tmp_path, "empty.py", "")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "empty.py", "search": "", "replace": "x = 1\n"}]}))
    assert (tmp_path / "empty.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_genuine_whole_file_rewrite_is_still_allowed(tmp_path):
    """The guard must not block legitimate refactoring of a whole module."""

    _write(tmp_path, "mod.py", _big_module(40))
    rewritten = _big_module(38).replace("return 0", "return 999")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "mod.py", "content": rewritten}]}))
    assert "999" in (tmp_path / "mod.py").read_text(encoding="utf-8")


def test_small_files_may_still_be_gutted(tmp_path):
    """Deleting most of a ten-line file is ordinary editing, not destruction."""

    _write(tmp_path, "tiny.py", "a = 1\nb = 2\nc = 3\nd = 4\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "tiny.py", "content": "a = 1\n"}]}))
    assert (tmp_path / "tiny.py").read_text(encoding="utf-8") == "a = 1\n"


def test_deletion_still_works_and_is_not_confused_with_truncation(tmp_path):
    _write(tmp_path, "gone.py", _big_module())
    _engine(tmp_path).apply(parse_bundle({"files": [], "deleted_files": ["gone.py"]}))
    assert not (tmp_path / "gone.py").exists()


# --------------------------------------------- syntax as a post-condition

def test_an_edit_that_breaks_python_syntax_is_refused(tmp_path):
    """The live case: right anchor, right place, wrong indentation.

    Without this the broken file reaches the test runner, where the failure
    looks like a test error rather than the edit mistake it actually is.
    """

    original = "def main():\n    if flag:\n        print('a')\n        return\n"
    path = _write(tmp_path, "cli.py", original)

    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(
            parse_bundle(
                {"files": [{"path": "cli.py", "search": "        print('a')", "replace": "if other:\n        print('b')"}]}
            )
        )

    assert excinfo.value.kind == "syntax_error"
    assert excinfo.value.recoverable
    assert "indentation" in str(excinfo.value).lower()
    assert path.read_text(encoding="utf-8") == original


def test_a_syntax_failure_rolls_back_the_other_files_too(tmp_path):
    good = _write(tmp_path, "good.py", "x = 1\n")
    _write(tmp_path, "bad.py", "def f():\n    return 1\n")

    with pytest.raises(EditError):
        _engine(tmp_path).apply(
            parse_bundle(
                {
                    "files": [
                        {"path": "good.py", "search": "x = 1", "replace": "x = 2"},
                        {"path": "bad.py", "search": "    return 1", "replace": "  return ("},
                    ]
                }
            )
        )

    assert good.read_text(encoding="utf-8") == "x = 1\n", "atomicity must survive the syntax gate"


def test_a_file_that_was_already_broken_can_still_be_repaired(tmp_path):
    """The gate must not make a pre-existing syntax error unfixable."""

    _write(tmp_path, "broken.py", "def f(:\n    pass\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "broken.py", "search": "def f(:", "replace": "def f():"}]}))
    assert (tmp_path / "broken.py").read_text(encoding="utf-8") == "def f():\n    pass\n"


def test_non_python_files_are_not_syntax_checked(tmp_path):
    _write(tmp_path, "notes.md", "# title\n")
    _engine(tmp_path).apply(parse_bundle({"files": [{"path": "notes.md", "search": "# title", "replace": "def f(: ["}]}))
    assert "def f(: [" in (tmp_path / "notes.md").read_text(encoding="utf-8")


def test_a_newly_created_python_file_is_syntax_checked(tmp_path):
    with pytest.raises(EditError) as excinfo:
        _engine(tmp_path).apply(parse_bundle({"files": [], "new_files": [{"path": "new.py", "content": "def f(:\n"}]}))
    assert excinfo.value.kind == "syntax_error"
    assert not (tmp_path / "new.py").exists()
