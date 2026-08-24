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
    if isinstance(edits, list):
        for item in edits:
            if isinstance(item, dict) and not str(item.get("search", "")).strip() and not _is_internal(arguments):
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
    """Point at ``write_file`` when a short file is fighting the anchor matcher.

    A local model that cannot land an anchor will usually keep failing the same
    way -- observed as eight consecutive misses on a twenty-line module, each
    with a perfectly correct diagnosis attached. When the file is short enough
    to reproduce reliably, saying so converts a dead end into one more attempt
    that can actually succeed.
    """

    if error.kind not in _ANCHOR_TROUBLE or not error.path:
        return error.detail
    try:
        path = resolve_readable(context, error.path)
        lines = len(path.read_text(encoding="utf-8-sig", errors="replace").splitlines())
    except (ToolError, OSError):
        return error.detail
    if lines > _SMALL_FILE_LINES:
        return error.detail
    return (
        f"{error.detail}\n"
        f"{error.path} is only {lines} lines. Rather than fight the anchor, call write_file with the "
        f"complete corrected contents of {error.path}."
    )


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


def which(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Report whether a program exists on this machine, and where."""

    name = str(arguments["name"])
    location = shutil.which(name)
    return {"name": name, "found": bool(location), "path": location or ""}


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
            name="which",
            purpose="Check whether a program is installed on this machine and where it lives.",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            adapter=which,
            risk=RiskLevel.SAFE,
            tags=("environment", "investigate"),
            example='{"name": "which", "arguments": {"name": "stockfish"}}',
        ),
    ]
