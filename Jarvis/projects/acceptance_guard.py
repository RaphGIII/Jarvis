"""Making sure an acceptance check cannot be satisfied by reading the answers.

This module exists because of a failure that was entirely my own design's
fault. The chess project was asked to reconstruct a position from a board
image, and it produced:

    def position(path) -> str:
        with open('.../fixtures.json') as f:
            data = json.load(f)
        for p in data['positions']:
            if p['image'] == path.split('/')[-1]:
                return p['fen']

That passed. It passed because the ground-truth file was in the workspace so
the model could check its own work, and the acceptance command compared against
the same file. The model did not solve the problem; it looked up the answers,
and the check confirmed that the answers matched the answers.

The general rule this encodes: **an acceptance check whose answer key is
reachable from the solution is not a check.** It is the same failure as a
capability whose dry run returns "would play music" -- verification satisfiable
without doing the work.

Two defences, because either alone is weak:

*Hold the answers out.*  The strongest fix, and the caller's job: keep the key
somewhere the solution has no reason to look and does not receive as a path.

*Notice when the solution reads it anyway.*  Held-out is not airtight -- a
determined search of the filesystem would find it -- so :func:`check_no_oracle`
looks for the solution referencing the key at all, and says so plainly.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class OracleReport:
    """Whether a solution appears to have consulted the answers."""

    clean: bool = True
    findings: list[dict[str, Any]] = field(default_factory=list)

    def add(self, path: str, line: int, detail: str) -> None:
        self.findings.append({"file": path, "line": line, "detail": detail})
        self.clean = False

    def describe(self) -> str:
        if self.clean:
            return "no reference to the answer key"
        return "\n".join(
            f"{item['file']}:{item['line']}: {item['detail']}" for item in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        return {"clean": self.clean, "findings": self.findings}


def check_no_oracle(
    workspace: str | Path,
    *,
    forbidden: Iterable[str],
    suffixes: Iterable[str] = (".py",),
) -> OracleReport:
    """Look for the solution reading anything it was supposed to derive.

    ``forbidden`` are substrings that should not appear in the solution: the
    name of the answer file, the key it is stored under, a distinctive value.
    Matching is textual because that is what catches it -- a model that reads
    the answers does so by naming the file.
    """

    report = OracleReport()
    root = Path(workspace)
    needles = [item for item in forbidden if item]
    if not root.is_dir() or not needles:
        return report

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in tuple(suffixes) or not path.is_file():
            continue
        if any(part in {"__pycache__", ".git"} for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(source.splitlines(), start=1):
            for needle in needles:
                if needle.lower() in line.lower():
                    report.add(
                        str(path.relative_to(root)).replace("\\", "/"),
                        number,
                        f"references {needle!r}, which is the answer key -- the value must be "
                        "derived, not looked up",
                    )
                    break
    return report


def check_not_constant(
    workspace: str | Path,
    function: str,
    *,
    module: str,
) -> OracleReport:
    """Notice a function that ignores its input and returns a literal.

    The other half of the same failure. Before it read the answers, the same
    model wrote ``return 'rnbqkbnr/pppppppp/...'`` -- a constant that happened
    to be right for one of four fixtures. A single-fixture test would have
    passed it.
    """

    report = OracleReport()
    path = Path(workspace) / f"{module}.py"
    if not path.is_file():
        return report

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return report

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function:
            continue
        body = [item for item in node.body if not _is_docstring(item)]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            if isinstance(value, ast.Constant):
                report.add(
                    f"{module}.py",
                    node.lineno,
                    f"{function}() ignores its arguments and returns a constant",
                )
    return report


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def static_check_command(python: str, repo_root: str | Path, *modules: str) -> list[str]:
    """An acceptance command that fails on a name the code never defines.

    The capability path has had this since a generated module called a Jarvis
    tool from a branch its tests never reached. Projects had no equivalent, and
    the chess project then failed exactly the same way -- `image_region(...)`
    called from inside position.py, a tool name in solution code.

    Offered as an acceptance criterion rather than run after the fact so the
    loop sees it fail and repairs, which is the difference between a guard and
    a post-mortem.
    """

    names = ", ".join(repr(f"{name}.py") for name in modules) or "'main.py'"
    return [
        python,
        "-c",
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from capabilities.static_check import check_file; "
        f"paths = [{names}]; "
        "import pathlib; "
        # A file that does not exist must FAIL, not pass silently. A vacuous
        # pass is indistinguishable from a real one in the acceptance report,
        # and the chess project showed three of four criteria green while the
        # module they were checking had never been written.
        "missing = [p for p in paths if not pathlib.Path(p).is_file()]; "
        "[print(f'{p}: does not exist') for p in missing]; "
        "reports = [(p, check_file(p)) for p in paths if pathlib.Path(p).is_file()]; "
        "bad = missing + [p for p, r in reports if not r.ok]; "
        "[print(f'{p}: {r.describe()}') for p, r in reports if not r.ok]; "
        "print('STATIC_OK') if not bad else None; "
        "raise SystemExit(1 if bad else 0)",
    ]


def guard_command(
    python: str,
    repo_root: str | Path,
    *,
    forbidden: Iterable[str],
) -> list[str]:
    """An acceptance command that fails when the solution consulted the answers.

    Returned as a command rather than run inline so it can sit alongside the
    other acceptance criteria -- which means the loop sees it fail and repairs,
    instead of being told at the end that its solution does not count.
    """

    needles = ", ".join(repr(item) for item in forbidden if item)
    return [
        python,
        "-c",
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from projects.acceptance_guard import check_no_oracle; "
        f"r = check_no_oracle('.', forbidden=[{needles}]); "
        "print(r.describe()); "
        "raise SystemExit(0 if r.clean else 1)",
    ]
