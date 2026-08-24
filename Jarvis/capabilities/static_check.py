"""Static checks a capability must pass before it is believed.

This module exists because of a specific and instructive near-miss.  A generated
music capability passed every runtime check -- tests, contract, placeholder
removed -- while containing this:

    else:
        media_control('playpause', dry_run=True)   # not defined anywhere

``media_control`` is a *Jarvis tool*, available while investigating and not
importable from a capability.  The line sits in a branch that
``run({'dry_run': True})`` never reaches, so every runtime check passed and the
capability would have raised ``NameError`` the first time the user actually
needed that path.

The general lesson is that **executing one path proves one path**.  A
side-effecting capability is verified almost entirely through its dry run, which
means the branch that does the real work is the branch least likely to have been
run.  Static analysis is the cheap way to cover what execution did not: an
undefined name is a defect whether or not the test happened to reach it.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import Any

#: Names available to any module without being defined or imported.
_BUILTINS = frozenset(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}

#: Jarvis tools.  Not special-cased for correctness -- an undefined name is a
#: defect regardless -- but naming them lets the error say what went wrong
#: instead of just that something is missing.
JARVIS_TOOLS = frozenset(
    {
        "list_files", "read_file", "search_text", "find_definition", "apply_edits",
        "write_file", "make_directory", "check_syntax", "run_command", "run_python",
        "run_tests", "git", "git_diff", "create_virtualenv", "install_packages",
        "find_program", "web_search", "fetch_document",
        "running_processes", "find_applications", "media_folders", "find_media",
        "open_path", "open_url", "launch_application", "media_control",
        "clipboard_write", "notify", "screenshot",
    }
)


@dataclass
class StaticIssue:
    kind: str
    name: str
    line: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "line": self.line, "message": self.message}


@dataclass
class StaticReport:
    ok: bool = True
    issues: list[StaticIssue] = field(default_factory=list)

    def add(self, issue: StaticIssue) -> None:
        self.issues.append(issue)
        self.ok = False

    def describe(self) -> str:
        if self.ok:
            return "no static issues"
        return "\n".join(f"line {item.line}: {item.message}" for item in self.issues)


def check_source(source: str, *, filename: str = "main.py") -> StaticReport:
    """Find names used but never defined, imported, or built in."""

    report = StaticReport()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        report.add(
            StaticIssue("syntax", "", exc.lineno or 0, f"{filename} does not parse: {exc.msg}")
        )
        return report

    module_names = _bound_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(node, module_names, report)

    # Module-level code, outside any function.
    _check_scope(tree, module_names | _BUILTINS, report, skip_functions=True)
    return report


def check_file(path: str) -> StaticReport:
    from pathlib import Path

    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report = StaticReport()
        report.add(StaticIssue("unreadable", "", 0, str(exc)))
        return report
    return check_source(source, filename=Path(path).name)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

def _bound_names(node: ast.AST) -> set[str]:
    """Everything a module or function body binds: defs, imports, assignments."""

    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                names.add(alias.asname or alias.name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
    return names


def _check_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef, module_names: set[str], report: StaticReport
) -> None:
    available = module_names | _bound_names(function) | _BUILTINS
    _check_scope(function, available, report, skip_functions=False)


def _walk_scope(scope: ast.AST, *, skip_functions: bool):
    """Walk a scope's own nodes, without descending into nested definitions.

    ``ast.walk`` descends unconditionally, so filtering function nodes out of
    its output still visits their bodies -- which reported every issue inside a
    function twice, once from the function's own pass and once from the module's.
    """

    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if skip_functions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _check_scope(
    scope: ast.AST, available: set[str], report: StaticReport, *, skip_functions: bool
) -> None:
    seen: set[str] = set()
    for node in _walk_scope(scope, skip_functions=skip_functions):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id in available or node.id in seen:
            continue
        seen.add(node.id)
        if node.id in JARVIS_TOOLS:
            message = (
                f"{node.id!r} is a Jarvis TOOL, not something this module can call. "
                "Tools exist only while investigating. Use the standard library "
                "(shutil.which, pathlib, subprocess, os) instead."
            )
        else:
            message = f"{node.id!r} is used but never defined or imported"
        report.add(StaticIssue("undefined", node.id, node.lineno, message))


def main(argv: list[str] | None = None) -> int:
    """``python -m capabilities.static_check main.py`` -- used as an acceptance check."""

    import sys

    paths = list(argv if argv is not None else sys.argv[1:]) or ["main.py"]
    failed = False
    for path in paths:
        report = check_file(path)
        if not report.ok:
            failed = True
            print(f"{path}:\n{report.describe()}", file=sys.stderr)
    if failed:
        return 1
    print("STATIC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
