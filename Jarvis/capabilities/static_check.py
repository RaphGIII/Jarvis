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


#: Characters that can only be in a string because a backslash was eaten.
#: A Windows path in a non-raw literal turns a tab escape into a real tab and a
#: return escape into a real carriage return, and the path stops existing.
_ACCIDENTAL_ESCAPES = {
    chr(9): "a tab escape",
    chr(13): "a carriage-return escape",
    chr(10): "a newline escape",
    chr(8): "a backspace escape",
    chr(12): "a form-feed escape",
    chr(11): "a vertical-tab escape",
    chr(7): "a bell escape",
}

_BACKSLASH = chr(92)


def _looks_like_a_path(value: str) -> bool:
    """Whether a string was probably meant to be a filesystem path."""

    lowered = value.lower()
    return (
        ":" in value[:3]                       # a drive letter
        or lowered.endswith((".exe", ".dll", ".json", ".txt", ".py", ".png", ".onnx"))
        or _BACKSLASH in value
        or value.count("/") >= 2
    )


def check_windows_paths(source: str, *, filename: str = "main.py") -> StaticReport:
    """Find Windows paths mangled by Python's own string escaping.

    Written because a generated engine.py assigned a Stockfish path as an
    ordinary quoted string full of backslashes. Python read the escapes, the
    path acquired a carriage return and a tab, and the process died with
    FileNotFoundError pointing at something that looked perfectly correct in
    the source.

    That is the interesting part: the bug is invisible when reading the code
    and obvious when reading the VALUE, which is exactly what a static pass can
    do and a person skimming a diff cannot.
    """

    report = StaticReport()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return report

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if not _looks_like_a_path(value):
            continue
        found = [name for char, name in _ACCIDENTAL_ESCAPES.items() if char in value]
        if found:
            report.add(
                StaticIssue(
                    "path_escape",
                    "",
                    getattr(node, "lineno", 0),
                    f"this path contains {', '.join(found)}: a backslash was interpreted as an "
                    "escape, so the path is not what it looks like. Use a raw string, forward "
                    "slashes, or pathlib.",
                )
            )
    return report


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

    for issue in check_windows_paths(source, filename=filename).issues:
        report.add(issue)

    module_names = _bound_names(tree)
    annotations = _annotation_nodes(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(node, module_names, report, annotations)

    # Module-level code, outside any function.
    _check_scope(tree, module_names | _BUILTINS, report, skip_functions=True, annotations=annotations)
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
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module_names: set[str],
    report: StaticReport,
    annotations: set[int],
) -> None:
    available = module_names | _bound_names(function) | _BUILTINS
    _check_scope(function, available, report, skip_functions=False, annotations=annotations)


def _annotation_nodes(scope: ast.AST) -> set[int]:
    """Every node that lives inside a type annotation.

    Annotations are not evaluated at runtime under PEP 649 (Python 3.14's
    default), so a missing ``Any`` in ``def run(p: dict[str, Any])`` imports and
    runs perfectly well.  Flagging it would be a false positive, and a false
    positive here is expensive: this check gates acceptance, so it would reject
    a capability that works.

    Found live -- a generated capability lost its ``from typing import Any``
    while being edited, and the runtime checks were right that nothing was
    broken by it.
    """

    marked: set[int] = set()

    def mark(node: ast.AST | None) -> None:
        if node is None:
            return
        for item in ast.walk(node):
            marked.add(id(item))

    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark(node.returns)
            arguments = node.args
            for group in (arguments.args, arguments.posonlyargs, arguments.kwonlyargs):
                for argument in group:
                    mark(argument.annotation)
            for argument in (arguments.vararg, arguments.kwarg):
                if argument is not None:
                    mark(argument.annotation)
        elif isinstance(node, ast.AnnAssign):
            mark(node.annotation)
    return marked


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
    scope: ast.AST,
    available: set[str],
    report: StaticReport,
    *,
    skip_functions: bool,
    annotations: set[int],
) -> None:
    seen: set[str] = set()
    for node in _walk_scope(scope, skip_functions=skip_functions):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if id(node) in annotations:
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
