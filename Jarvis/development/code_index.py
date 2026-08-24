"""Deterministic repository intelligence: find the code, don't make the model hunt.

The division of labour that makes a small local model usable is that *software*
locates the code and the *model* reasons about it.  Handing a 7B model a
430-line file and asking it to find the right line is asking it to do the one
thing it is worst at, and it duly fails: asked four times to add an exit word to
a CLI, it four times edited the help string, because a line reading

    /quit  /exit  /bye    leave

matches the goal more densely than the line that actually implements it.

Lexical scoring cannot tell those apart -- both contain the literal.  What
separates them is *syntactic role*, and that is a fact about the code, available
for free from the AST:

    HELP = \"\"\"... /quit /exit /bye leave ...\"\"\"      module-level string constant
    if command in {"/quit", "/exit", "/bye"}:         a branch condition

Both occurrences sit inside string literals, so "is it a string?" is the wrong
question and would rank the real target no higher.  The question that works is
"what is this string *doing*?", answered by walking ancestors:

* inside a comparison or branch test          -> this is control flow
* inside a collection inside a function       -> this is data the code acts on
* anywhere else inside a function body        -> executable
* a module-level constant assignment          -> configuration or help text
* a docstring or comment                      -> documentation

:class:`CodeIndex` answers that question for a whole repository, plus the
ordinary structural queries -- where is this symbol defined, what references it,
what does this module import -- so a retrieval step can be a lookup instead of a
guess.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Iterator


class Role(IntEnum):
    """What a piece of text is *doing*, ordered so that higher means more executable.

    The ordering is the whole point: it is what lets "rank real code above help
    text" be a sort rather than a heuristic pile.
    """

    #: A comment, or a docstring.  Describes code; is not code.
    DOCUMENTATION = 0
    #: A module-level string constant: help text, banners, templates.
    CONSTANT_TEXT = 10
    #: Any other module-level code.
    MODULE_CODE = 30
    #: Inside a function or method body.
    FUNCTION_CODE = 50
    #: Inside a collection literal that a function body evaluates.
    FUNCTION_DATA = 60
    #: Inside a branch condition or comparison -- control flow.
    CONTROL_FLOW = 90

    @property
    def executable(self) -> bool:
        return self >= Role.MODULE_CODE


@dataclass(frozen=True)
class Symbol:
    """A definition, with the lines it spans."""

    name: str
    kind: str  # "function" | "method" | "class"
    path: str
    start_line: int
    end_line: int
    qualname: str = ""
    #: Lines occupied by the definition's own docstring, if any.
    docstring_lines: tuple[int, int] | None = None

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line


@dataclass
class Occurrence:
    """One place a search term appears, and what that place *is*."""

    path: str
    line: int
    text: str
    role: Role
    symbol: str = ""
    #: Set when the occurrence is inside a string literal.
    in_string: bool = False

    @property
    def score(self) -> float:
        return float(self.role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "text": self.text[:200],
            "role": self.role.name,
            "symbol": self.symbol,
            "executable": self.role.executable,
        }


@dataclass
class Region:
    """A contiguous, copyable slice of a file, with why it was chosen."""

    path: str
    start_line: int
    end_line: int
    text: str
    reason: str = ""
    role: Role = Role.MODULE_CODE
    symbol: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)

    def render(self) -> str:
        """Unnumbered, so an anchor copied out of it can actually match."""

        header = f"--- {self.path} lines {self.start_line}-{self.end_line}"
        if self.symbol:
            header += f"  ({self.symbol})"
        if self.reason:
            header += f"  [{self.reason}]"
        return f"\n{header} ---\n{self.text}\n--- end {self.path} ---\n"


@dataclass
class FileIndex:
    """Everything worth knowing about one Python file, computed once."""

    path: str
    source: str
    lines: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    #: line -> Role, for every line that carries one.
    line_roles: dict[int, Role] = field(default_factory=dict)
    parse_error: str = ""

    def symbol_at(self, line: int) -> Symbol | None:
        """The innermost definition containing ``line``."""

        best: Symbol | None = None
        for symbol in self.symbols:
            if symbol.contains(line):
                if best is None or symbol.start_line > best.start_line:
                    best = symbol
        return best

    def role_at(self, line: int) -> Role:
        return self.line_roles.get(line, Role.MODULE_CODE)


class CodeIndex:
    """A queryable, deterministic view of a repository's Python source."""

    #: Directories that are never the answer to "where is this implemented".
    IGNORED = frozenset(
        {
            ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".pytest_tmp",
            ".mypy_cache", ".ruff_cache", ".cache", ".runtime", "node_modules", "build",
            "dist", ".idea", ".vscode", ".agent_tmp", "site-packages",
        }
    )

    #: How many raw matches to collect per query before ranking.  Generous:
    #: the point is that ranking sees everything, not that scanning is cheap.
    _SCAN_CEILING = 5000

    def __init__(self, root: str | Path, *, max_files: int = 4000) -> None:
        self.root = Path(root).resolve()
        self.max_files = max_files
        self._cache: dict[str, FileIndex] = {}

    # -- enumeration -----------------------------------------------------

    def python_files(self, subdir: str | None = None) -> list[Path]:
        import os

        base = (self.root / subdir).resolve(strict=False) if subdir else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [base] if base.suffix == ".py" else []

        found: list[Path] = []
        for current, directories, filenames in os.walk(base):
            directories[:] = sorted(name for name in directories if name not in self.IGNORED)
            for name in sorted(filenames):
                if name.endswith(".py"):
                    found.append(Path(current) / name)
                    if len(found) >= self.max_files:
                        return found
        return found

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    # -- indexing --------------------------------------------------------

    def index(self, path: str | Path) -> FileIndex:
        """Index one file, cached by path."""

        resolved = (self.root / path).resolve(strict=False) if not Path(path).is_absolute() else Path(path)
        relative = self.relative(resolved)
        if relative in self._cache:
            return self._cache[relative]

        try:
            source = resolved.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
        except OSError as exc:
            index = FileIndex(path=relative, source="", parse_error=str(exc))
            self._cache[relative] = index
            return index

        index = _build_file_index(relative, source)
        self._cache[relative] = index
        return index

    # -- queries ---------------------------------------------------------

    def find_literal(
        self,
        term: str,
        *,
        subdir: str | None = None,
        executable_only: bool = False,
        paths: Iterable[str] | None = None,
        limit: int = 200,
    ) -> list[Occurrence]:
        """Every place ``term`` appears, ranked with executable code first.

        This is the query that matters for a goal phrased in literals -- "/quit",
        a config key, an error string -- and it is the one plain text search gets
        wrong, because it has no way to prefer the branch that acts on the
        literal over the paragraph that documents it.
        """

        occurrences: list[Occurrence] = []
        needle = term.lower().rstrip(".,;:!?)")
        if not needle:
            return []
        # Restrict BEFORE ranking and truncating, never after: filtering a
        # repository-wide top-N down to one file can legitimately leave nothing.
        wanted_paths = {str(item).replace("\\", "/") for item in paths} if paths else None
        for path in self.python_files(subdir):
            index = self.index(path)
            if wanted_paths is not None and index.path not in wanted_paths:
                continue
            if needle not in index.source.lower():
                continue
            for number, line in enumerate(index.lines, start=1):
                if needle not in line.lower():
                    continue
                role = index.role_at(number)
                if executable_only and not role.executable:
                    continue
                symbol = index.symbol_at(number)
                occurrences.append(
                    Occurrence(
                        path=index.path,
                        line=number,
                        text=line.rstrip(),
                        role=role,
                        symbol=symbol.qualname if symbol else "",
                        in_string='"' in line or "'" in line,
                    )
                )
                if len(occurrences) >= self._SCAN_CEILING:
                    break

        # Rank first, THEN truncate.  Truncating during the scan is the same
        # mistake as lexical ranking, one level down: in a file with 240 decoy
        # mentions and the implementation on the last line, a limit applied
        # while scanning discards the only occurrence that mattered before
        # anything has had the chance to notice it is the executable one.
        occurrences.sort(key=lambda item: (-item.score, item.path, item.line))
        return occurrences[:limit]

    def find_symbol(self, name: str, *, subdir: str | None = None) -> list[Symbol]:
        """Where a function, method or class is *defined* (not merely mentioned)."""

        found: list[Symbol] = []
        for path in self.python_files(subdir):
            index = self.index(path)
            found.extend(symbol for symbol in index.symbols if symbol.name == name)
        return found

    def references(self, name: str, *, subdir: str | None = None, limit: int = 100) -> list[Occurrence]:
        """Executable mentions of a name, excluding its own definition."""

        pattern = re.compile(rf"\b{re.escape(name)}\b")
        found: list[Occurrence] = []
        for path in self.python_files(subdir):
            index = self.index(path)
            if name not in index.source:
                continue
            definitions = {symbol.start_line for symbol in index.symbols if symbol.name == name}
            for number, line in enumerate(index.lines, start=1):
                if number in definitions or not pattern.search(line):
                    continue
                role = index.role_at(number)
                if not role.executable:
                    continue
                symbol = index.symbol_at(number)
                found.append(
                    Occurrence(
                        path=index.path, line=number, text=line.rstrip(), role=role,
                        symbol=symbol.qualname if symbol else "",
                    )
                )
                if len(found) >= limit:
                    return found
        return found

    def imports_of(self, path: str | Path) -> list[str]:
        return list(self.index(path).imports)

    def importers_of(self, module: str, *, subdir: str | None = None) -> list[str]:
        """Which files import a module -- the dependency edge, reversed."""

        stem = module.split(".")[-1]
        found: list[str] = []
        for path in self.python_files(subdir):
            index = self.index(path)
            if any(stem == item.split(".")[-1] or item.startswith(module) for item in index.imports):
                found.append(index.path)
        return found

    # -- the retrieval the model actually consumes -----------------------

    def regions_for_terms(
        self,
        terms: Iterable[str],
        *,
        paths: Iterable[str] | None = None,
        budget_chars: int = 20000,
        context_lines: int = 8,
        max_regions: int = 8,
    ) -> list[Region]:
        """The source regions most likely to be the thing a goal is about.

        Regions are whole enclosing definitions where one exists, because an
        anchor copied from a half-shown function tends to be ambiguous, and a
        model that can see the surrounding branch writes a better replacement.
        """

        wanted = []
        for term in terms:
            cleaned = str(term).strip().rstrip(".,;:!?)")
            if len(cleaned) > 1 and cleaned not in wanted:
                wanted.append(cleaned)
        if not wanted:
            return []

        scored: list[Occurrence] = []
        for term in wanted:
            scored.extend(self.find_literal(term, paths=paths, limit=60))

        if not scored:
            return []

        # Prefer executable roles, then the terms that are most distinctive:
        # a hit on "/goodbye" says more than a hit on "command".
        rarity = {term: sum(1 for item in scored if term.lower() in item.text.lower()) or 1 for term in wanted}
        def weight(item: Occurrence) -> float:
            distinct = sum(1 / rarity[term] for term in wanted if term.lower() in item.text.lower())
            return item.score + distinct * 10

        scored.sort(key=lambda item: (-weight(item), item.path, item.line))

        regions: list[Region] = []
        used: list[tuple[str, int, int]] = []
        remaining = budget_chars

        for occurrence in scored:
            if len(regions) >= max_regions or remaining <= 0:
                break
            index = self.index(occurrence.path)
            symbol = index.symbol_at(occurrence.line)
            if symbol is not None:
                start, end = symbol.start_line, symbol.end_line
                reason = f"{occurrence.role.name} in {symbol.qualname}"
            else:
                start = max(1, occurrence.line - context_lines)
                end = min(len(index.lines), occurrence.line + context_lines)
                reason = occurrence.role.name

            if any(path == occurrence.path and start >= low and end <= high for path, low, high in used):
                continue

            text = "\n".join(index.lines[start - 1 : end])
            if len(text) > remaining:
                if regions:
                    continue
                text = text[:remaining]
            regions.append(
                Region(
                    path=occurrence.path, start_line=start, end_line=end, text=text,
                    reason=reason, role=occurrence.role, symbol=symbol.qualname if symbol else "",
                )
            )
            used.append((occurrence.path, start, end))
            remaining -= len(text)

        return regions

    def describe_literal(self, term: str, *, limit: int = 8) -> str:
        """A short report a model can act on: where the term lives, and as what."""

        occurrences = self.find_literal(term, limit=60)
        if not occurrences:
            return f"'{term}' does not appear in any Python file."
        rows = [f"'{term}' appears in {len(occurrences)} place(s), most executable first:"]
        for occurrence in occurrences[:limit]:
            marker = "CODE" if occurrence.role.executable else "TEXT"
            where = f" in {occurrence.symbol}" if occurrence.symbol else ""
            rows.append(f"  [{marker}/{occurrence.role.name}] {occurrence.path}:{occurrence.line}{where}: {occurrence.text.strip()[:110]}")
        return "\n".join(rows)


# --------------------------------------------------------------------------
# Building one file's index
# --------------------------------------------------------------------------

def _build_file_index(relative: str, source: str) -> FileIndex:
    lines = source.split("\n")
    index = FileIndex(path=relative, source=source, lines=lines)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        index.parse_error = f"{exc.msg} at line {exc.lineno}"
        # Still useful: comments can be classified without a parse tree.
        index.line_roles = _comment_roles(source)
        return index

    index.symbols = _collect_symbols(relative, tree)
    index.imports = _collect_imports(tree)
    index.line_roles = _classify_lines(source, tree, index.symbols)
    return index


def _collect_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return sorted(set(imports))


def _collect_symbols(relative: str, tree: ast.AST) -> list[Symbol]:
    symbols: list[Symbol] = []

    def docstring_span(node: ast.AST) -> tuple[int, int] | None:
        body = getattr(node, "body", None)
        if not body:
            return None
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            return (first.lineno, getattr(first, "end_lineno", first.lineno))
        return None

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(child, ast.ClassDef) else ("method" if prefix else "function")
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                symbols.append(
                    Symbol(
                        name=child.name,
                        kind=kind,
                        path=relative,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        qualname=qualname,
                        docstring_lines=docstring_span(child),
                    )
                )
                walk(child, qualname)
            else:
                walk(child, prefix)

    walk(tree, "")
    return symbols


def _comment_roles(source: str) -> dict[int, Role]:
    roles: dict[int, Role] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                roles[token.start[0]] = Role.DOCUMENTATION
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return roles


def _classify_lines(source: str, tree: ast.AST, symbols: list[Symbol]) -> dict[int, Role]:
    """Give every meaningful line the role its syntax implies.

    The ordering below is deliberate: a later, more specific assignment wins, so
    a comment inside a function is still DOCUMENTATION, and a literal inside a
    branch condition is CONTROL_FLOW even though it also sits inside a function.
    """

    roles: dict[int, Role] = {}
    total_lines = source.count("\n") + 1

    # 1. Baseline by containment: module level, then function bodies.
    for line in range(1, total_lines + 1):
        roles[line] = Role.MODULE_CODE
    for symbol in symbols:
        if symbol.kind in {"function", "method"}:
            for line in range(symbol.start_line, symbol.end_line + 1):
                roles[line] = Role.FUNCTION_CODE

    parents = _parent_map(tree)

    # 2. Module-level string data: help text, banners, templates, tables.
    #
    # A table of strings is only "help text" if nothing decides anything with
    # it.  EXIT_WORDS = {"/quit", "/exit"} paired with `if word in EXIT_WORDS`
    # is control-flow data and must not be demoted to the same rank as a
    # banner, or hoisting the set out of the function would hide it from
    # retrieval -- punishing the better-factored version of the same code.
    decisive = _names_used_in_decisions(tree, parents=_parent_map(tree))
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not _is_string_data(node):
            continue
        role = Role.FUNCTION_DATA if _assigned_names(node) & decisive else Role.CONSTANT_TEXT
        for line in _span(node):
            roles[line] = role

    # 3. Data the code acts on, and control flow, both inside functions.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        enclosing_function = _enclosing_function(node, parents)
        if enclosing_function is None:
            continue
        role = Role.FUNCTION_CODE
        if _inside_collection(node, parents):
            role = Role.FUNCTION_DATA
        if _inside_control_flow(node, parents):
            role = Role.CONTROL_FLOW
        for line in _span(node):
            if roles.get(line, Role.MODULE_CODE) < role:
                roles[line] = role

    # 4. Branch conditions generally, string or not.
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            for line in _span(node.test):
                roles[line] = max(roles.get(line, Role.MODULE_CODE), Role.CONTROL_FLOW)
        elif isinstance(node, ast.Compare):
            if _enclosing_function(node, parents) is not None:
                for line in _span(node):
                    roles[line] = max(roles.get(line, Role.MODULE_CODE), Role.CONTROL_FLOW)

    # 5. Docstrings and comments override everything: they are not code.
    for symbol in symbols:
        if symbol.docstring_lines:
            for line in range(symbol.docstring_lines[0], symbol.docstring_lines[1] + 1):
                roles[line] = Role.DOCUMENTATION
    module_doc = _module_docstring_span(tree)
    if module_doc:
        for line in range(module_doc[0], module_doc[1] + 1):
            roles[line] = Role.DOCUMENTATION
    roles.update(_comment_roles(source))

    return roles


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _ancestors(node: ast.AST, parents: dict[int, ast.AST]) -> Iterator[ast.AST]:
    current = parents.get(id(node))
    while current is not None:
        yield current
        current = parents.get(id(current))


def _enclosing_function(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    for ancestor in _ancestors(node, parents):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ancestor
    return None


def _inside_collection(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    for ancestor in _ancestors(node, parents):
        if isinstance(ancestor, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
            return True
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
    return False


def _inside_control_flow(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """True when the value participates in a decision rather than merely existing."""

    previous: ast.AST = node
    for ancestor in _ancestors(node, parents):
        if isinstance(ancestor, ast.Compare):
            return True
        if isinstance(ancestor, (ast.If, ast.While, ast.IfExp)) and previous is getattr(ancestor, "test", None):
            return True
        if isinstance(ancestor, ast.match_case):
            return True
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        previous = ancestor
    return False


def _is_string_data(node: ast.AST) -> bool:
    """A string constant, or a collection literal made only of string constants."""

    value = getattr(node, "value", None)
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
        elements = value.elts
    elif isinstance(value, ast.Dict):
        elements = [item for item in list(value.keys) + list(value.values) if item is not None]
    else:
        return False
    return bool(elements) and all(
        isinstance(item, ast.Constant) and isinstance(item.value, str) for item in elements
    )


def _assigned_names(node: ast.AST) -> set[str]:
    targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
    return {item.id for item in targets if isinstance(item, ast.Name)}


def _names_used_in_decisions(tree: ast.AST, *, parents: dict[int, ast.AST]) -> set[str]:
    """Names that some branch condition reads -- i.e. names the program decides with."""

    names: set[str] = set()
    for node in ast.walk(tree):
        test: ast.AST | None = None
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            test = node.test
        elif isinstance(node, ast.Compare):
            test = node
        if test is None:
            continue
        for child in ast.walk(test):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _span(node: ast.AST) -> range:
    start = getattr(node, "lineno", None)
    if start is None:
        return range(0)
    return range(start, getattr(node, "end_lineno", start) + 1)


def _module_docstring_span(tree: ast.AST) -> tuple[int, int] | None:
    body = getattr(tree, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return (first.lineno, getattr(first, "end_lineno", first.lineno))
    return None
