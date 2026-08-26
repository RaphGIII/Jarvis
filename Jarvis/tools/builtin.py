"""The standard Jarvis toolset: filesystem, search, git, processes, packages.

Every adapter here is deliberately boring.  Boring is the point: these are the
deterministic hands that carry out whatever the model decided, and the less
cleverness they contain, the fewer ways an autonomous run can surprise its
operator.

Two rules hold throughout:

*Containment first.*  Reads are confined to the workspace plus explicitly
declared readable roots; writes go through :mod:`development.edit_engine`, which
is the only code in the system that puts model-authored bytes on disk.

*Subprocesses are fenced.*  Commands run with a scrubbed environment, an
executable allow-list, a timeout and a bounded output.  A hung or chatty child
process must not be able to stall or flood the agent loop.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from development.edit_engine import EditEngine, EditError, PathPolicy, parse_bundle
from tools.registry import RiskLevel, ToolContext, ToolError, ToolSpec

# Directories that are never interesting to an agent and would swamp any listing.
IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".pytest_tmp",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".runtime",
        "node_modules",
        "build",
        "dist",
        ".idea",
        ".vscode",
        ".agent_tmp",
    }
)

BINARY_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".pt", ".onnx", ".sqlite", ".db", ".zip", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2"}
)

#: Executables an autonomous run may launch without individual approval.
DEFAULT_COMMAND_ALLOWLIST = (
    "python",
    "python3",
    "py",
    "pytest",
    "pip",
    "git",
    "node",
    "npm",
    "ruff",
    "pyflakes",
    "mypy",
    "black",
)


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def resolve_readable(context: ToolContext, relative: str) -> Path:
    """Resolve a path for reading, inside the workspace or a declared root."""

    raw = Path(str(relative).replace("\\", "/"))
    if raw.is_absolute():
        candidate = raw.resolve(strict=False)
        for root in [context.workspace, *context.readable_roots]:
            if candidate.is_relative_to(Path(root).resolve()):
                return candidate
        raise ToolError(f"path is outside every readable root: {relative}", kind="path_denied", retryable=False)
    if ".." in raw.parts:
        raise ToolError(f"path may not contain '..': {relative}", kind="path_denied", retryable=False)
    candidate = (context.workspace / raw).resolve(strict=False)
    if not candidate.is_relative_to(context.workspace.resolve()):
        raise ToolError(f"path escapes the workspace: {relative}", kind="path_denied", retryable=False)
    return candidate


def iter_files(root: Path, *, limit: int = 2000, subdir: str | None = None) -> list[Path]:
    base = (root / subdir).resolve(strict=False) if subdir else root
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    found: list[Path] = []
    for current, directories, filenames in os.walk(base):
        directories[:] = sorted(name for name in directories if name not in IGNORED_DIRECTORY_NAMES)
        for name in sorted(filenames):
            path = Path(current) / name
            if path.suffix.lower() in BINARY_SUFFIXES:
                continue
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def relative_to(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------
# Filesystem adapters
# --------------------------------------------------------------------------

def list_files(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    subdir = str(arguments.get("path") or "").strip() or None
    limit = int(arguments.get("limit") or 300)
    if subdir:
        resolve_readable(context, subdir)  # containment check
    paths = iter_files(context.workspace, limit=limit, subdir=subdir)
    return {
        "root": str(context.workspace),
        "count": len(paths),
        "files": [relative_to(context.workspace, path) for path in paths],
    }


def read_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_readable(context, str(arguments["path"]))
    if not path.exists() or not path.is_file():
        raise ToolError(f"file does not exist: {arguments['path']}", kind="missing_file")
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    lines = text.split("\n")

    start = arguments.get("start_line")
    end = arguments.get("end_line")
    if start is not None or end is not None:
        low = max(1, int(start or 1))
        high = min(len(lines), int(end or len(lines)))
        selected = "\n".join(lines[low - 1 : high])
        return {"path": str(arguments["path"]), "start_line": low, "end_line": high, "total_lines": len(lines), "content": selected}

    max_chars = int(arguments.get("max_chars") or context.max_output_chars)
    truncated = len(text) > max_chars
    return {
        "path": str(arguments["path"]),
        "total_lines": len(lines),
        "truncated": truncated,
        "content": text[:max_chars],
    }


def search_text(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    query = str(arguments["query"])
    subdir = str(arguments.get("path") or "").strip() or None
    limit = int(arguments.get("limit") or 60)
    use_regex = bool(arguments.get("regex"))

    try:
        pattern = re.compile(query if use_regex else re.escape(query), re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"invalid regular expression: {exc}", kind="invalid_arguments") from None

    matches: list[dict[str, Any]] = []
    for path in iter_files(context.workspace, limit=4000, subdir=subdir):
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append({"path": relative_to(context.workspace, path), "line": number, "text": line.strip()[:240]})
                if len(matches) >= limit:
                    return {"query": query, "count": len(matches), "matches": matches, "truncated": True}
    return {"query": query, "count": len(matches), "matches": matches, "truncated": False}


def find_definition(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Locate where a Python symbol is defined, rather than merely mentioned."""

    symbol = str(arguments["symbol"]).strip()
    pattern = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(symbol)}\b|^\s*{re.escape(symbol)}\s*=")
    hits: list[dict[str, Any]] = []
    for path in iter_files(context.workspace, limit=4000, subdir=str(arguments.get("path") or "") or None):
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.match(line):
                hits.append({"path": relative_to(context.workspace, path), "line": number, "text": line.strip()[:240]})
    return {"symbol": symbol, "count": len(hits), "definitions": hits[:40]}


#: Marks a bundle assembled by write_file rather than by the model, so the
#: anchor requirement below does not apply to it.
_INTERNAL = "__jarvis_internal__"


def _is_internal(arguments: dict[str, Any]) -> bool:
    return bool(arguments.get(_INTERNAL))


def _has_content(context: ToolContext, relative: str) -> bool:
    """Whether a path exists and holds anything worth protecting."""

    if not relative:
        return False
    try:
        path = resolve_readable(context, relative)
    except ToolError:
        return True  # unresolvable: be conservative and demand an anchor
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8-sig", errors="replace").strip())
    except OSError:
        return True


def apply_edits(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Write model-authored changes through the deterministic edit engine.

    ``apply_edits`` is for *anchored* changes only: every entry needs a search
    string. Whole-file content belongs in ``write_file``, and the split is not
    cosmetic. Left free to do both, a local model reaches for "replace the file,
    here is everything" almost every time -- and gets it slightly wrong, so an
    edit meant to add one import arrives as a file containing only that import.
    Refusing the shape here, with a pointer to the right tool, is far more
    effective than asking it not to in the prompt.
    """

    edits = arguments.get("files") or []
    if isinstance(edits, list) and not _is_internal(arguments):
        for item in edits:
            if not isinstance(item, dict) or str(item.get("search", "")).strip():
                continue
            # An anchor is only required where there is something to lose.
            # Filling a file that is empty or absent destroys nothing, and
            # refusing it cost a live run three cycles: the model had created
            # an empty test file and simply wanted to put the tests in it.
            if _has_content(context, str(item.get("path", ""))):
                raise ToolError(
                    f"apply_edits needs a 'search' anchor for {item.get('path', '?')}. "
                    "Copy the exact lines you want to change into 'search' and the new version into "
                    "'replace'. To set a file's entire contents instead, use write_file.",
                    kind="empty_search",
                    retryable=True,
                )

    bundle = {
        "analysis": str(arguments.get("analysis", "")),
        "files": edits,
        "new_files": arguments.get("new_files") or [],
        "deleted_files": arguments.get("deleted_files") or [],
    }
    policy = PathPolicy(
        context.workspace,
        allowed_paths=context.allowed_paths,
        protected_paths=context.protected_paths,
        protected_reason=context.protected_reason,
    )
    engine = EditEngine(policy)
    try:
        result = engine.apply(parse_bundle(bundle))
    except EditError as exc:
        raise ToolError(_with_small_file_hint(exc, context), kind=exc.kind, retryable=exc.recoverable) from None
    return result.to_dict()


#: Kinds where the model knows what it wants to change but cannot express it as
#: an anchor.  On a short file there is a much easier way out.
_ANCHOR_TROUBLE = frozenset({"no_unique_match", "ambiguous_search", "no_effective_edit", "stale_context"})

#: Above this many lines, sending the whole file back is worse than a bad
#: anchor: the model will not reproduce it faithfully.
_SMALL_FILE_LINES = 60


def _with_small_file_hint(error: EditError, context: ToolContext) -> str:
    """Point at ``write_file`` when the anchor matcher is winning.

    A local model that cannot land an anchor keeps failing the same way --
    observed as eight consecutive misses on a twenty-line module, each with a
    perfectly correct diagnosis attached. Saying so converts a dead end into one
    more attempt that can actually succeed.

    Two signals, because the first one alone was tuned too tightly. File size
    was the original trigger, and during a live capability build it withheld
    this advice from a 69-line file for being nine lines over the threshold; the
    model then missed the same anchor three times running while diagnosing the
    underlying bug correctly every time.

    The stronger signal is *repetition*. A second failure on the same path means
    the model's picture of the file has drifted from what is on disk, and no
    further anchor it invents from that picture is going to match. That needs no
    threshold and cannot be mistuned.
    """

    if error.kind not in _ANCHOR_TROUBLE or not error.path:
        return error.detail

    # `scratch` is shared between the agent loop and its tools, which is what
    # lets a tool notice it is being asked the same impossible thing again.
    misses = context.scratch.setdefault("anchor_misses", {})
    misses[error.path] = misses.get(error.path, 0) + 1
    repeated = misses[error.path] >= 2

    lines: int | None = None
    try:
        path = resolve_readable(context, error.path)
        lines = len(path.read_text(encoding="utf-8-sig", errors="replace").splitlines())
    except (ToolError, OSError):
        lines = None

    small = lines is not None and lines <= _SMALL_FILE_LINES
    if not small and not repeated:
        return error.detail

    if small:
        return (
            f"{error.detail}\n{error.path} is only {lines} lines. Rather than fight the anchor, "
            f"call write_file with the complete corrected contents of {error.path}."
        )

    # Repeated misses on a file too large to retype. "Rewrite the whole file"
    # is the right advice for sixty lines and actively harmful for seven
    # hundred: told that during a live repair, the model replaced a 688-line
    # module with a 37-line sketch, and only the edit engine's shrink guard
    # stopped it from destroying a working implementation to fix one number.
    #
    # This used to end there, and for a long file it was a dead end with advice
    # attached: anchors would not land, a rewrite would be refused, and the
    # attempt ended having changed nothing. Measured on the Spotify latency
    # repair -- eight consecutive anchor failures on 708 lines, one write_file
    # refused for shrinking it to 61, a correct diagnosis on every one.
    #
    # `replace_definition` is named here, unlike the rewrite tool, because it
    # cannot do the damage the rewrite tool can: everything outside the named
    # definition is preserved byte for byte, so a model that reaches for it too
    # eagerly loses nothing.
    return (
        f"{error.detail}\nThat is {misses[error.path]} failed anchors in {error.path} in a row, "
        f"so the text you are matching against is not what the file contains. {error.path} is "
        f"{lines} lines -- far too long to retype from memory. If what you want to change is one "
        f"function or class, call replace_definition with its NAME and the complete new source: "
        f"it needs no anchor and leaves the rest of the file untouched. Otherwise call read_file "
        f"on the region you want to change, then anchor on a short unique line you have just read."
    )


def replace_definition(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Replace one named function or class, located by the parser rather than by text.

    Anchored editing has a hole in the middle of it, and every large repair
    falls in. On a short file a model that cannot land an anchor is told to send
    the whole file back, and that works. On a long one it cannot: retyping seven
    hundred lines from memory produces a sketch, and the shrink guard -- rightly
    -- refuses it. So both doors close. Measured on the Spotify latency repair:
    eight consecutive anchor failures against a 708-line module, one write_file
    refused for shrinking it to 61 lines, and the attempt ended having changed
    nothing, with a correct diagnosis attached to every failure.

    What the model wanted was expressible in one sentence -- *replace what
    `_powershell` does* -- and there was no way to say it. The name of a
    definition is a far better handle than a copy of its first line: the parser
    finds it exactly, it cannot be ambiguous, it cannot drift from what the
    model remembers, and everything outside it is preserved byte for byte, so
    nothing else in the file can be lost.

    ``name`` may be dotted for a method: ``ClassName.method``. The replacement
    is parsed on its own and must define exactly that name, so a model that
    sends a fragment or the wrong function is refused before anything is
    written rather than after.
    """

    import ast
    import textwrap

    path = str(arguments["path"])
    name = str(arguments["name"]).strip()
    source = str(arguments.get("source") or "")
    target = resolve_readable(context, path)

    if not target.is_file():
        raise ToolError(
            f"{path} does not exist, so there is no definition in it to replace. "
            "Use write_file to create it.",
            kind="missing_file", retryable=False,
        )
    if not name:
        raise ToolError("replace_definition needs the name of the function or class to replace.",
                        kind="invalid_arguments", retryable=False)
    if not source.strip():
        raise ToolError(
            f"replace_definition needs the complete new source for {name}, including its "
            "'def' or 'class' line.",
            kind="invalid_arguments", retryable=True,
        )

    text = target.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ToolError(
            f"{path} does not currently parse ({exc.msg} at line {exc.lineno}), so a definition "
            "cannot be located in it. Fix the syntax first.",
            kind="syntax_error", retryable=True,
        ) from None

    replacement = textwrap.dedent(source).strip("\n")
    try:
        parsed = ast.parse(replacement)
    except SyntaxError as exc:
        raise ToolError(
            f"YOUR REPLACEMENT for {name} is not valid Python: {exc.msg} at line {exc.lineno} "
            "of the text you sent. NOTHING WAS WRITTEN.",
            kind="syntax_error", retryable=True,
        ) from None
    defined = _definition_names(parsed)

    node, container = _locate_definition(tree, name)
    if node is None and len(defined) == 1:
        # The replacement says what it defines, in code, and it has to define
        # exactly one thing anyway. So `name` is a second statement of a fact
        # already established -- and a redundant parameter is somewhere to be
        # wrong. Observed live during the archive-count repair: the model sent
        # a perfectly good `def run(payload)` under the name
        # "run.payload.get('source')", having over-generalised the dotted form
        # documented for methods, and the edit was refused for the one part of
        # the call that carried no information.
        #
        # Nothing is loosened by this. The file must still contain the
        # definition, the replacement must still define exactly one, and
        # everything outside it is still preserved byte for byte.
        node, container = _locate_definition(tree, defined[0])
        if node is not None:
            name = defined[0]

    if node is None:
        available = ", ".join(_definition_names(tree)[:25]) or "none"
        raise ToolError(
            f"{path} has no top-level definition called {name!r}. It defines: {available}. "
            "Use the exact name, or ClassName.method for a method.",
            kind="unknown_definition", retryable=True,
        )

    wanted = name.split(".")[-1]
    if defined != [wanted]:
        raise ToolError(
            f"the replacement must be exactly one definition called {wanted!r}; what you sent "
            f"defines {defined or 'nothing'}. NOTHING WAS WRITTEN.",
            kind="invalid_arguments", retryable=True,
        )
    if len(parsed.body) != 1:
        # Only def/class statements were being counted, so a module-level
        # assignment or import travelled along with the definition and was
        # spliced into the middle of the file. Observed live during the
        # archive-count repair: a replacement carrying both `INPUT_SCHEMA = {...}`
        # and `def run(...)`, refused downstream by the duplicate-definition
        # guard with a message about INPUT_SCHEMA -- which is not what the model
        # thought it was doing and not where it would have looked.
        #
        # The rule is the one the tool's name already implies: this replaces a
        # definition, so it takes a definition. Anything else in the file is
        # changed with a different call.
        extra = len(parsed.body) - 1
        raise ToolError(
            f"the replacement must contain ONLY the definition of {wanted!r} -- you also sent "
            f"{extra} other top-level statement(s), which would be inserted into the middle of "
            f"{path} and duplicate what is already there. Send just the def or class. "
            "NOTHING WAS WRITTEN.",
            kind="invalid_arguments", retryable=True,
        )

    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])]) - 1
    end = int(node.end_lineno or node.lineno)

    # A method keeps the indentation of the class it lives in; a top-level
    # definition has none. Taken from the definition being replaced rather than
    # guessed, so the file's own style survives.
    indent = text.splitlines()[start][: len(text.splitlines()[start]) - len(text.splitlines()[start].lstrip())]
    body = "\n".join((indent + item if item.strip() else item) for item in replacement.split("\n"))

    updated = "".join(lines[:start]) + body + "\n" + "".join(lines[end:])
    if updated == text:
        raise ToolError(
            f"{name} in {path} already has exactly that body, so nothing changed. If the "
            "behaviour is still wrong, the defect is somewhere else.",
            kind="no_effective_edit", retryable=True,
        )

    bundle = {
        "analysis": f"replace_definition {name}",
        "files": [{"path": path, "content": updated}],
        _INTERNAL: True,
    }
    result = apply_edits(bundle, context)
    result["definition"] = name
    result["container"] = container
    return result


def _locate_definition(tree: "Any", name: str) -> tuple["Any", str]:
    """The node for ``name``, and what it was found inside.

    Only definitions the model can name unambiguously: top level, or one level
    inside a class. A closure nested in a function has no stable address a
    caller could write down.
    """

    import ast

    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if "." in name:
        outer, inner = name.split(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == outer:
                for child in node.body:
                    if isinstance(child, kinds) and child.name == inner:
                        return child, outer
        return None, ""
    for node in tree.body:
        if isinstance(node, kinds) and node.name == name:
            return node, ""
    return None, ""


def _definition_names(tree: "Any") -> list[str]:
    """Every name a module defines at the top level, plus its methods."""

    import ast

    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, kinds):
            continue
        names.append(node.name)
        if isinstance(node, ast.ClassDef):
            names += [f"{node.name}.{child.name}" for child in node.body if isinstance(child, kinds)]
    return names


def write_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Create or overwrite one file, still via the edit engine's guarantees.

    Idempotent by design.  ``write_file`` is a declaration of desired content,
    so a file that already holds that content is a success, not a failure.  The
    edit engine's "no effective edit" guard exists for *search/replace*, where a
    no-op means the model's anchor did not do what it believed -- applying that
    same rule here made an agent that re-asserted a correct file look like it
    was failing, which then triggered pointless repair cycles.
    """

    path = str(arguments["path"])
    content = str(arguments["content"])
    target = resolve_readable(context, path)

    if target.exists() and not target.is_file():
        # Named explicitly, because the generic path produced "existing edit
        # target does not exist: zeus_fail.txt" about something that plainly
        # does exist -- it is a directory. A message that misnames the problem
        # cannot be acted on by a user or repaired by a model.
        raise ToolError(
            f"{path} already exists and is a directory, not a file; "
            "nothing can be written there. Choose another name or remove the directory.",
            kind="path_conflict",
            retryable=False,
        )

    if target.exists() and target.is_file():
        try:
            current = target.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
        except OSError:
            current = None
        if current is not None and current == content.replace("\r\n", "\n"):
            return {"applied": [], "total_changed_lines": 0, "unchanged": True, "path": path}

    bundle = (
        {"analysis": "write_file", "files": [{"path": path, "content": content}], _INTERNAL: True}
        if target.exists()
        else {"analysis": "write_file", "files": [], "new_files": [{"path": path, "content": content}], _INTERNAL: True}
    )
    return apply_edits(bundle, context)


def make_directory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_readable(context, str(arguments["path"]))
    path.mkdir(parents=True, exist_ok=True)
    return {"path": relative_to(context.workspace, path), "created": True}


# --------------------------------------------------------------------------
# Process adapters
# --------------------------------------------------------------------------

def safe_environment(context: ToolContext) -> dict[str, str]:
    """A scrubbed environment for child processes.

    Only the variables a build genuinely needs are forwarded.  Credentials in
    particular must never reach a subprocess that a model chose to run.
    """

    passthrough = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "LANG",
        "LC_ALL",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
    )
    environment = {key: value for key, value in os.environ.items() if key in passthrough}
    environment.update(context.environment)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    # The workspace wins over anything installed on the host. Without this, a
    # stray top-level package in user site-packages shadows the workspace's own
    # modules -- this machine has a `tests` package installed that did exactly
    # that, making a candidate's own test suite unimportable.
    environment["PYTHONPATH"] = str(context.workspace.resolve())
    return environment


def _normalise_command(raw: Any) -> list[str]:
    if isinstance(raw, list):
        command = [str(part) for part in raw if str(part) != ""]
    elif isinstance(raw, str):
        import shlex

        command = [part.strip("\"'") for part in shlex.split(raw, posix=False)]
    else:
        raise ToolError("command must be a string or an array of strings", kind="invalid_arguments")
    if not command:
        raise ToolError("command was empty", kind="invalid_arguments")
    return command


def _check_allowlist(command: list[str], allowlist: Iterable[str]) -> None:
    executable = Path(command[0]).name.lower()
    for suffix in (".exe", ".cmd", ".bat"):
        if executable.endswith(suffix):
            executable = executable[: -len(suffix)]
    allowed = {name.lower() for name in allowlist}
    if executable not in allowed:
        raise ToolError(
            f"executable {command[0]!r} is not in the allow-list for this run ({', '.join(sorted(allowed))})",
            kind="command_denied",
            retryable=False,
        )


def run_command(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    command = _normalise_command(arguments.get("command"))
    allowlist = context.scratch.get("command_allowlist", DEFAULT_COMMAND_ALLOWLIST)
    _check_allowlist(command, allowlist)

    cwd = context.workspace
    if arguments.get("cwd"):
        cwd = resolve_readable(context, str(arguments["cwd"]))
    timeout = float(arguments.get("timeout_seconds") or context.timeout_seconds)

    # Prefer the interpreter running Jarvis, or the workspace venv, over
    # whatever "python" happens to mean on PATH.
    if Path(command[0]).name.lower() in {"python", "python3", "py"}:
        command = [_python_executable(context), *command[1:]]

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            env=safe_environment(context),
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            f"command timed out after {timeout:.0f}s: {' '.join(command)}\n"
            f"{_tail(exc.stdout)}\n{_tail(exc.stderr)}",
            kind="timeout",
        ) from None
    except FileNotFoundError:
        raise ToolError(f"executable not found: {command[0]}", kind="missing_executable", retryable=False) from None

    cap = context.max_output_chars // 2
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stdout": _tail(completed.stdout, cap),
        "stderr": _tail(completed.stderr, cap),
    }


def run_python(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Execute a snippet of Python inside the workspace."""

    code = str(arguments["code"])
    return run_command({"command": [_python_executable(context), "-c", code], "timeout_seconds": arguments.get("timeout_seconds")}, context)


def run_tests(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Run pytest, defaulting to the whole workspace."""

    target = str(arguments.get("target") or "").strip()
    extra = ["-q"]
    if arguments.get("verbose"):
        extra = ["-v"]
    command = [_python_executable(context), "-m", "pytest", *extra]
    if target:
        resolve_readable(context, target)
        command.append(target)
    result = run_command({"command": command, "timeout_seconds": arguments.get("timeout_seconds")}, context)
    result["summary"] = _pytest_summary(f"{result.get('stdout', '')}\n{result.get('stderr', '')}")
    return result


def check_syntax(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Compile Python files without running them -- the cheapest useful gate."""

    import ast

    targets = arguments.get("paths")
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        paths = [path for path in iter_files(context.workspace, limit=800) if path.suffix == ".py"]
    else:
        paths = [resolve_readable(context, str(item)) for item in targets]

    errors = []
    for path in paths:
        if not path.exists() or path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError as exc:
            errors.append(
                {
                    "path": relative_to(context.workspace, path),
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                }
            )
        except OSError:
            continue
    return {"checked": len(paths), "ok": not errors, "errors": errors}


def _python_executable(context: ToolContext) -> str:
    """Prefer the workspace virtualenv, so project dependencies are visible."""

    configured = context.scratch.get("python_executable")
    if configured:
        return str(configured)
    for candidate in (
        context.workspace / ".venv" / "Scripts" / "python.exe",
        context.workspace / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable or "python"


def _tail(text: str | None, limit: int = 8000) -> str:
    value = text or ""
    return value if len(value) <= limit else "...[head truncated]\n" + value[-limit:]


_PYTEST_SUMMARY = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")


def _pytest_summary(output: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for count, label in _PYTEST_SUMMARY.findall(output):
        key = "errors" if label.startswith("error") else label
        summary[key] = summary.get(key, 0) + int(count)
    return summary


# --------------------------------------------------------------------------
# Git adapters
# --------------------------------------------------------------------------

def git_command(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    args = arguments.get("args")
    if isinstance(args, str):
        import shlex

        args = shlex.split(args, posix=False)
    if not isinstance(args, list) or not args:
        raise ToolError("git args must be a non-empty array", kind="invalid_arguments")

    readonly = {"status", "diff", "log", "show", "ls-files", "rev-parse", "branch", "blame", "describe"}
    subcommand = str(args[0]).lstrip("-")
    if subcommand not in readonly and not context.scratch.get("allow_git_writes"):
        raise ToolError(
            f"git {subcommand} modifies history and is not enabled for this run",
            kind="command_denied",
            retryable=False,
        )
    return run_command({"command": ["git", *[str(item) for item in args]]}, context)


def git_diff(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    args = ["diff"]
    if arguments.get("staged"):
        args.append("--cached")
    if arguments.get("name_only"):
        args.append("--name-only")
    if arguments.get("path"):
        args.extend(["--", str(arguments["path"])])
    return git_command({"args": args}, context)


# --------------------------------------------------------------------------
# Dependency adapters
# --------------------------------------------------------------------------

def create_virtualenv(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Create a project-local virtualenv.  Never touch the global interpreter."""

    name = str(arguments.get("path") or ".venv")
    target = resolve_readable(context, name)
    if target.exists():
        return {"path": relative_to(context.workspace, target), "created": False, "reason": "already exists"}
    result = run_command(
        {"command": [sys.executable, "-m", "venv", str(target)], "timeout_seconds": arguments.get("timeout_seconds") or 300},
        context,
    )
    if not result["success"]:
        raise ToolError(f"could not create virtualenv: {result['stderr'][-800:]}", kind="venv_failed")
    context.scratch["python_executable"] = _venv_python(target)
    return {"path": relative_to(context.workspace, target), "created": True, "python": _venv_python(target)}


def install_packages(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Install into the workspace virtualenv, creating it if necessary.

    Installing into the host interpreter is refused outright: an autonomous run
    that can mutate the system Python can break every other project on the
    machine, and that is not a risk the operator agreed to by asking for a
    feature.
    """

    packages = arguments.get("packages")
    if isinstance(packages, str):
        packages = [packages]
    if not isinstance(packages, list) or not packages:
        raise ToolError("packages must be a non-empty array", kind="invalid_arguments")
    for package in packages:
        if not re.fullmatch(r"[A-Za-z0-9._\-\[\]]+(?:[=<>!~]=?[A-Za-z0-9._*\-]+)?", str(package)):
            raise ToolError(f"refusing suspicious package specifier: {package!r}", kind="invalid_arguments", retryable=False)

    venv = resolve_readable(context, ".venv")
    if not venv.exists():
        create_virtualenv({"path": ".venv"}, context)
    python = _venv_python(venv)

    result = run_command(
        {
            "command": [python, "-m", "pip", "install", "--disable-pip-version-check", *[str(item) for item in packages]],
            "timeout_seconds": arguments.get("timeout_seconds") or 900,
        },
        context,
    )
    if result["success"]:
        _record_requirements(context.workspace, [str(item) for item in packages])
    result["packages"] = [str(item) for item in packages]
    result["python"] = python
    return result


def _venv_python(venv: Path) -> str:
    windows = venv / "Scripts" / "python.exe"
    return str(windows if windows.exists() or os.name == "nt" else venv / "bin" / "python")


def _record_requirements(workspace: Path, packages: list[str]) -> None:
    """Keep an explicit install record so the environment is reproducible."""

    path = workspace / "requirements.txt"
    existing = []
    if path.exists():
        existing = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    merged = list(existing)
    for package in packages:
        stem = re.split(r"[=<>!~\[]", package)[0].strip().lower()
        if not any(re.split(r"[=<>!~\[]", item)[0].strip().lower() == stem for item in merged):
            merged.append(package)
    path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def find_program(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Report whether ``name`` is available on this machine, and in what form.

    Deliberately NOT called ``which``. It was, and a local model writing a
    capability confused this tool with ``shutil.which``: it called the stdlib
    function and then subscripted the result as ``player["found"]``, because
    that is the shape *this* tool returns. A tool name that shadows a
    well-known standard-library function with different semantics is a trap,
    and the fix belongs in the name rather than in a warning nobody reads.

    It used to answer a narrower question than the one it was asked. It ran
    ``shutil.which`` and nothing else, so *is pyautogui available here* came
    back ``found: false`` -- true of the PATH, false of the machine, and read
    by the model as the second. Measured live during the screen-capture
    acquisition:

        find_program({"name": "pyautogui"}) -> {"found": false, "path": ""}
        find_program({"name": "mss"})       -> {"found": false, "path": ""}

    pyautogui is installed and importable here; the project's own constraints
    said so two paragraphs above. Both answers looked identical, so the model
    treated the installed package and the absent one as equally unavailable,
    picked ``mss`` anyway, and spent the rest of its budget diagnosing an
    ImportError. An executable check standing in for an availability check is
    the same substitution as description overlap standing in for relevance.

    So the answer distinguishes the two ways a thing can be here. ``found``
    means available *somehow*; ``kind`` says which, and is what decides how to
    use it. A python package reports no ``path`` on purpose -- there is nothing
    to hand to ``subprocess`` -- and ``answer`` says in words what the fields
    mean, because the field the model reads first is whichever one it reads
    first.
    """

    name = str(arguments["name"])
    location = shutil.which(name)
    if location:
        return {
            "name": name,
            "found": True,
            "kind": "executable",
            "path": location,
            "importable": False,
            "answer": f"{name} is an executable program at {location}. Run it with subprocess.",
        }

    if _importable(name):
        return {
            "name": name,
            "found": True,
            "kind": "python_package",
            "path": "",
            "importable": True,
            "answer": (
                f"{name} is NOT a command you can run, but it IS an installed Python package on "
                f"this machine: `import {name}` works. Use it by importing it. There is no path "
                f"to pass to subprocess."
            ),
        }

    return {
        "name": name,
        "found": False,
        "kind": "absent",
        "path": "",
        "importable": False,
        "answer": (
            f"{name} is neither a program on PATH nor an installed Python package. Do not import "
            f"it and do not try to run it -- choose something that is here instead."
        ),
    }


def _importable(name: str) -> bool:
    """Whether ``import name`` would work, without running the module.

    ``find_spec`` locates a module without executing it, which matters for a
    package like pyautogui whose import has side effects on the desktop. A
    dotted or otherwise unusable name is not importable rather than an error:
    this is a question, and every question here has an answer.
    """

    if not name or not name.isidentifier():
        return False
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError, TypeError):
        return False


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def builtin_tools() -> list[ToolSpec]:
    """The standard toolset, with declared risk levels."""

    return [
        ToolSpec(
            name="list_files",
            purpose="List files in the workspace, optionally under a subdirectory.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": [],
            },
            adapter=list_files,
            risk=RiskLevel.SAFE,
            tags=("filesystem", "investigate"),
            example='{"name": "list_files", "arguments": {"path": "src"}}',
        ),
        ToolSpec(
            name="read_file",
            purpose="Read a text file. Use start_line/end_line for a slice of a big file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["path"],
            },
            adapter=read_file,
            risk=RiskLevel.SAFE,
            tags=("filesystem", "investigate"),
            example='{"name": "read_file", "arguments": {"path": "src/app.py"}}',
        ),
        ToolSpec(
            name="search_text",
            purpose="Find lines matching a string (or regex) anywhere in the workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            adapter=search_text,
            risk=RiskLevel.SAFE,
            tags=("filesystem", "investigate"),
            example='{"name": "search_text", "arguments": {"query": "def main"}}',
        ),
        ToolSpec(
            name="find_definition",
            purpose="Find where a Python function, class or module-level name is defined.",
            input_schema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}, "path": {"type": "string"}},
                "required": ["symbol"],
            },
            adapter=find_definition,
            risk=RiskLevel.SAFE,
            tags=("filesystem", "investigate"),
            example='{"name": "find_definition", "arguments": {"symbol": "BrainRouter"}}',
        ),
        ToolSpec(
            name="apply_edits",
            purpose=(
                "Change PART of a file. Every edit needs a 'search' anchor copied exactly from the "
                "file plus the new text in 'replace'. Also creates and deletes files. Atomic. "
                "To set a whole file's contents, use write_file instead."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "files": {"type": "array"},
                    "new_files": {"type": "array"},
                    "deleted_files": {"type": "array"},
                },
                "required": [],
            },
            adapter=apply_edits,
            risk=RiskLevel.LOW,
            tags=("filesystem", "implement"),
            example='{"name": "apply_edits", "arguments": {"files": [{"path": "a.py", "search": "x = 1", "replace": "x = 2"}]}}',
        ),
        ToolSpec(
            name="replace_definition",
            purpose=(
                "Replace one whole function or class by NAME, with no search anchor. "
                "Use this when an anchored edit keeps missing, or when the file is too long to "
                "rewrite. Everything outside the named definition is preserved exactly. "
                "name may be dotted for a method: ClassName.method."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["path", "name", "source"],
            },
            adapter=replace_definition,
            risk=RiskLevel.MODERATE,
            tags=("filesystem", "edit"),
            example=('{"name": "replace_definition", "arguments": {"path": "main.py", '
                     '"name": "_powershell", "source": "def _powershell(verb):\\n    return {}\\n"}}'),
        ),
        ToolSpec(
            name="write_file",
            purpose="Set a file's ENTIRE contents. Use this when you have the complete file, not a fragment.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            adapter=write_file,
            risk=RiskLevel.LOW,
            tags=("filesystem", "implement"),
            example='{"name": "write_file", "arguments": {"path": "tests/test_app.py", "content": "def test_x():\\n    assert True\\n"}}',
        ),
        ToolSpec(
            name="make_directory",
            purpose="Create a directory inside the workspace.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            adapter=make_directory,
            risk=RiskLevel.LOW,
            tags=("filesystem", "implement"),
        ),
        ToolSpec(
            name="check_syntax",
            purpose="Parse Python files and report syntax errors without executing anything.",
            input_schema={"type": "object", "properties": {"paths": {"type": "array"}}, "required": []},
            adapter=check_syntax,
            risk=RiskLevel.SAFE,
            tags=("verify",),
            example='{"name": "check_syntax", "arguments": {"paths": ["src/app.py"]}}',
        ),
        ToolSpec(
            name="run_command",
            purpose="Run an allow-listed program in the workspace and capture its output.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "array"},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["command"],
            },
            adapter=run_command,
            risk=RiskLevel.MODERATE,
            tags=("process", "verify"),
            example='{"name": "run_command", "arguments": {"command": ["python", "main.py"]}}',
        ),
        ToolSpec(
            name="run_python",
            purpose="Execute a short Python snippet in the workspace and capture its output.",
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string"}, "timeout_seconds": {"type": "number"}},
                "required": ["code"],
            },
            adapter=run_python,
            risk=RiskLevel.MODERATE,
            tags=("process", "verify"),
            example='{"name": "run_python", "arguments": {"code": "import app; print(app.add(2, 3))"}}',
        ),
        ToolSpec(
            name="run_tests",
            purpose="Run pytest in the workspace and report the pass/fail summary.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}, "verbose": {"type": "boolean"}, "timeout_seconds": {"type": "number"}},
                "required": [],
            },
            adapter=run_tests,
            risk=RiskLevel.MODERATE,
            tags=("process", "verify"),
            example='{"name": "run_tests", "arguments": {"target": "tests"}}',
        ),
        ToolSpec(
            name="git",
            purpose="Run a read-only git command (status, diff, log, show, ls-files).",
            input_schema={"type": "object", "properties": {"args": {"type": "array"}}, "required": ["args"]},
            adapter=git_command,
            risk=RiskLevel.LOW,
            tags=("repository", "investigate"),
            example='{"name": "git", "arguments": {"args": ["status", "--short"]}}',
        ),
        ToolSpec(
            name="git_diff",
            purpose="Show the current uncommitted diff.",
            input_schema={
                "type": "object",
                "properties": {"staged": {"type": "boolean"}, "name_only": {"type": "boolean"}, "path": {"type": "string"}},
                "required": [],
            },
            adapter=git_diff,
            risk=RiskLevel.LOW,
            tags=("repository", "verify"),
        ),
        ToolSpec(
            name="create_virtualenv",
            purpose="Create an isolated Python environment for this project.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
            adapter=create_virtualenv,
            risk=RiskLevel.MODERATE,
            tags=("dependencies",),
        ),
        ToolSpec(
            name="install_packages",
            purpose="Install Python packages into the project virtualenv (never the system Python).",
            input_schema={
                "type": "object",
                "properties": {"packages": {"type": "array"}, "timeout_seconds": {"type": "number"}},
                "required": ["packages"],
            },
            adapter=install_packages,
            risk=RiskLevel.MODERATE,
            tags=("dependencies",),
            example='{"name": "install_packages", "arguments": {"packages": ["pillow"]}}',
        ),
        ToolSpec(
            name="find_program",
            purpose=(
                "Check whether something is available on this machine -- an executable OR an "
                "importable Python package -- and how to use it. Returns "
                "{found: bool, kind: 'executable'|'python_package'|'absent', path: str, answer: str}. "
                "kind='python_package' means import it; there is no path to run. "
                "This is a Jarvis tool, not Python's shutil.which()."
            ),
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            adapter=find_program,
            risk=RiskLevel.SAFE,
            tags=("environment", "investigate"),
            example='{"name": "find_program", "arguments": {"name": "stockfish"}}',
        ),
    ]
