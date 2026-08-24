"""Executing one path proves one path.

This check exists because of a near-miss worth stating plainly. A generated
music capability passed tests, contract and implemented -- every runtime check
-- while containing:

    else:
        media_control('playpause', dry_run=True)   # not defined anywhere

`media_control` is a Jarvis tool, available while investigating and not
importable from a capability. The line sits in a branch that
run({'dry_run': True}) never reaches, so nothing executed it and the capability
would have raised NameError the first time the user actually needed that path.

A side-effecting capability is verified almost entirely through its dry run,
which makes the branch that does the real work the branch least likely to have
been run. Static analysis covers what execution did not.
"""

from __future__ import annotations

import subprocess

import pytest

from capabilities.static_check import JARVIS_TOOLS, check_file, check_source


# --------------------------------------------------------------------------
# The case this was built for
# --------------------------------------------------------------------------

def test_a_jarvis_tool_called_from_a_capability_is_caught():
    source = (
        "import shutil\n"
        "\n"
        "def run(payload):\n"
        "    if shutil.which('vlc'):\n"
        "        return {'ok': True}\n"
        "    media_control('playpause')\n"
        "    return {'ok': False}\n"
    )

    report = check_source(source)

    assert not report.ok
    assert report.issues[0].name == "media_control"


def test_the_message_explains_what_a_tool_is():
    """"Not defined" is a symptom the model diagnoses correctly and acts on wrongly."""

    report = check_source("def run(p):\n    return find_media()\n")

    assert "Jarvis TOOL" in report.describe()
    assert "shutil.which" in report.describe()


def test_an_unreached_branch_is_still_checked():
    """The whole point: the dry run never enters this branch."""

    source = (
        "def run(payload):\n"
        "    if payload.get('dry_run'):\n"
        "        return {'ok': True}\n"
        "    return launch_application('vlc')\n"
    )

    assert not check_source(source).ok


def test_each_undefined_name_is_reported_once():
    """ast.walk descends into functions, which reported everything twice."""

    source = "def run(p):\n    return media_control(1)\n"

    assert len(check_source(source).issues) == 1


# --------------------------------------------------------------------------
# Ordinary undefined names
# --------------------------------------------------------------------------

def test_a_plain_typo_is_caught():
    report = check_source("def run(p):\n    return reslt\n")

    assert not report.ok
    assert "never defined" in report.describe()


def test_a_name_defined_later_at_module_level_is_fine():
    source = "def run(p):\n    return helper()\n\ndef helper():\n    return 1\n"

    assert check_source(source).ok


def test_imports_count_as_defined():
    assert check_source("import shutil\n\ndef run(p):\n    return shutil.which('x')\n").ok


def test_aliased_imports_count():
    assert check_source("import subprocess as sp\n\ndef run(p):\n    return sp.run(['x'])\n").ok


def test_from_imports_count():
    assert check_source("from pathlib import Path\n\ndef run(p):\n    return Path('.')\n").ok


def test_builtins_are_available():
    source = "def run(p):\n    return {'n': len(str(list(range(3))))}\n"

    assert check_source(source).ok


def test_parameters_are_available():
    assert check_source("def run(payload):\n    return payload\n").ok


def test_locals_are_available():
    assert check_source("def run(p):\n    x = 1\n    return x\n").ok


def test_comprehension_variables_are_available():
    assert check_source("def run(p):\n    return [i for i in range(3)]\n").ok


def test_exception_names_are_available():
    source = "def run(p):\n    try:\n        return 1\n    except ValueError as exc:\n        return str(exc)\n"

    assert check_source(source).ok


def test_class_attributes_and_methods():
    source = (
        "class Player:\n"
        "    def play(self):\n"
        "        return 1\n"
        "\n"
        "def run(p):\n"
        "    return Player().play()\n"
    )

    assert check_source(source).ok


def test_a_nested_function_sees_the_enclosing_scope():
    source = "def run(p):\n    value = 1\n    def inner():\n        return value\n    return inner()\n"

    assert check_source(source).ok


def test_module_level_code_is_checked_too():
    assert not check_source("VALUE = undefined_thing\n").ok


def test_a_module_level_constant_is_visible_in_functions():
    assert check_source("LIMIT = 3\n\ndef run(p):\n    return LIMIT\n").ok


# --------------------------------------------------------------------------
# Failure modes of the checker itself
# --------------------------------------------------------------------------

def test_a_file_that_does_not_parse_is_reported_clearly():
    report = check_source("def run(:\n    pass\n")

    assert not report.ok
    assert "does not parse" in report.describe()


def test_a_missing_file_is_reported_not_raised(tmp_path):
    report = check_file(str(tmp_path / "nope.py"))

    assert not report.ok


def test_an_empty_file_is_fine(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("", encoding="utf-8")

    assert check_file(str(target)).ok


def test_a_clean_capability_passes(tmp_path):
    target = tmp_path / "main.py"
    target.write_text(
        "import shutil\n"
        "\n"
        "INPUT_SCHEMA = {'type': 'object'}\n"
        "\n"
        "def run(payload):\n"
        "    player = shutil.which('vlc')\n"
        "    return {'ok': bool(player), 'player': player}\n",
        encoding="utf-8",
    )

    assert check_file(str(target)).ok


def test_the_tool_list_covers_the_desktop_pack():
    from tools.desktop import desktop_tools

    for spec in desktop_tools():
        assert spec.name in JARVIS_TOOLS, f"{spec.name} is missing from JARVIS_TOOLS"


# --------------------------------------------------------------------------
# As an acceptance check
# --------------------------------------------------------------------------

def test_the_static_check_is_part_of_the_acceptance_bar():
    from capabilities.service import capability_checks

    assert "static" in {check.name for check in capability_checks()}


def test_the_acceptance_command_fails_on_a_bad_capability(tmp_path):
    from capabilities.service import capability_checks

    (tmp_path / "main.py").write_text(
        "def run(p):\n    if p.get('dry_run'):\n        return {}\n    return media_control(1)\n",
        encoding="utf-8",
    )
    check = next(item for item in capability_checks() if item.name == "static")

    completed = subprocess.run(list(check.command), cwd=tmp_path, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "media_control" in completed.stderr


def test_the_acceptance_command_passes_on_a_good_capability(tmp_path):
    from capabilities.service import capability_checks

    (tmp_path / "main.py").write_text(
        "import shutil\n\ndef run(p):\n    return {'ok': bool(shutil.which('vlc'))}\n",
        encoding="utf-8",
    )
    check = next(item for item in capability_checks() if item.name == "static")

    completed = subprocess.run(list(check.command), cwd=tmp_path, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "STATIC_OK" in completed.stdout
