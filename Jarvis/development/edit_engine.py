"""Deterministic, atomic, multi-operation source edit engine.

This module is the single place where Jarvis turns a model-proposed change
bundle into bytes on disk.  Everything that is *decided* by a language model
(which file, which anchor, what the replacement text is) arrives here as data;
everything that is *performed* is deterministic Python with fail-closed
validation.

Design goals, in priority order:

1. **Never partially mutate.**  Every operation is resolved and validated in
   memory first.  Disk writes only start once the whole plan is known-good, and
   any failure during the write phase restores the original bytes.
2. **Fail loudly and recoverably.**  A bad edit raises :class:`EditError` with a
   machine-readable ``kind``.  Callers use ``kind`` to decide whether asking the
   model for a correction is worthwhile (``recoverable``) or whether the plan
   was structurally illegal (a protected-path write, say).
3. **Tolerate weak models without trusting them.**  Small local models copy
   anchors imprecisely: they re-indent, collapse whitespace, and -- notoriously
   -- paste back the ``00042: `` line-number gutter from whatever context we
   showed them.  The matcher degrades from exact, to line-number-stripped, to
   whitespace-canonical, to a similarity match with a mandatory margin between
   the best and second-best candidate.  It never guesses when two places in the
   file are comparably good.
4. **Preserve the file as the repo has it.**  BOM, CRLF/LF and trailing-newline
   conventions survive an edit untouched.

Supported operations (see :class:`EditOp`) cover exact search/replace, anchored
insertion, whole-file rewrite, creation and deletion, so a caller never has to
express an edit as "rewrite this 2000-line file" just because the schema had no
better verb.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable


class EditOp(str, Enum):
    """The deterministic operations an edit plan may request."""

    REPLACE = "replace"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    REWRITE = "rewrite"
    CREATE = "create"
    DELETE = "delete"


#: Operation kinds that create a file which must not already exist.
_CREATING_OPS = {EditOp.CREATE}
#: Operation kinds that require the target file to already exist.
_EXISTING_OPS = {EditOp.REPLACE, EditOp.INSERT_BEFORE, EditOp.INSERT_AFTER, EditOp.REWRITE, EditOp.DELETE}


class EditError(ValueError):
    """A rejected edit plan.

    ``kind`` is stable and machine-readable so the caller can branch on it;
    ``recoverable`` says whether re-prompting the model with fresh file content
    is a sensible next move (a mismatched anchor) or a waste of a cycle (a
    protected-path write, which the model must not retry at all).
    """

    def __init__(self, kind: str, message: str, *, path: str = "", recoverable: bool = False) -> None:
        super().__init__(f"patch quality gate failed: {message}")
        self.kind = kind
        self.path = path
        self.recoverable = recoverable
        self.detail = message


@dataclass(frozen=True)
class EditOperation:
    """One resolved operation in an :class:`EditPlan`."""

    op: EditOp
    path: str
    search: str = ""
    replace: str = ""
    content: str = ""
    occurrence: int | None = None
    #: True when this became a REWRITE only because the model supplied a
    #: replacement with no anchor.  Such an operation is a *guess* at intent,
    #: not a declaration of it, and the engine holds it to a stricter standard.
    anchorless: bool = False

    def describe(self) -> str:
        return f"{self.op.value}:{self.path}"


@dataclass
class EditPlan:
    """An ordered, de-duplicated set of operations plus the model's rationale."""

    operations: list[EditOperation] = field(default_factory=list)
    analysis: str = ""

    def __bool__(self) -> bool:
        return bool(self.operations)

    def paths(self) -> list[str]:
        seen: list[str] = []
        for operation in self.operations:
            if operation.path not in seen:
                seen.append(operation.path)
        return seen


@dataclass
class EditBudget:
    """Hard limits that bound how much damage one plan can do."""

    max_operations: int = 12
    max_changed_lines: int = 400
    max_new_file_chars: int = 24000
    #: An anchor larger than this is a whole-file rewrite wearing a
    #: search/replace costume, and gets refused as one.
    max_search_chars: int = 4000
    max_replace_chars: int = 12000
    #: Whole-file rewrites of large files are how weak models destroy code.
    #: Above this size a rewrite must be expressed as targeted edits instead.
    max_rewrite_chars: int = 12000
    #: Files shorter than this are exempt from the shrink guard below --
    #: gutting a ten-line file is ordinary editing.
    truncation_floor_lines: int = 25
    #: Fraction of a file's non-blank lines a rewrite must retain.  Below this
    #: the "rewrite" is almost certainly a fragment the model meant to splice
    #: in, and applying it would delete the rest of the file.
    min_retained_fraction: float = 0.5

    @classmethod
    def from_env(cls, getenv: Callable[[str, str], str]) -> "EditBudget":
        """Read limits from the environment.

        ``JARVIS_BUILD_*`` are the names operators of this repository already
        use for the two limits that existed before this engine did; they win
        over the generic ``JARVIS_EDIT_*`` spellings so existing configuration
        and runbooks keep working.
        """

        defaults = cls()

        def pick(*names: str, fallback: int) -> int:
            for name in names:
                raw = getenv(name, "")
                if raw:
                    return int(raw)
            return fallback

        return cls(
            max_operations=pick("JARVIS_EDIT_MAX_OPERATIONS", fallback=defaults.max_operations),
            max_changed_lines=pick(
                "JARVIS_BUILD_MAX_CHANGED_LINES",
                "JARVIS_EDIT_MAX_CHANGED_LINES",
                fallback=defaults.max_changed_lines,
            ),
            max_new_file_chars=pick(
                "JARVIS_BUILD_MAX_NEW_FILE_CHARS",
                "JARVIS_EDIT_MAX_NEW_FILE_CHARS",
                fallback=defaults.max_new_file_chars,
            ),
            max_search_chars=pick("JARVIS_EDIT_MAX_SEARCH_CHARS", fallback=defaults.max_search_chars),
            max_replace_chars=pick("JARVIS_EDIT_MAX_REPLACE_CHARS", fallback=defaults.max_replace_chars),
            max_rewrite_chars=pick("JARVIS_EDIT_MAX_REWRITE_CHARS", fallback=defaults.max_rewrite_chars),
        )


@dataclass
class AppliedEdit:
    """What actually happened to one path."""

    path: str
    op: str
    match_mode: str = ""
    changed_lines: int = 0
    created: bool = False
    deleted: bool = False


@dataclass
class EditResult:
    """The outcome of a successfully applied plan."""

    applied: list[AppliedEdit] = field(default_factory=list)
    total_changed_lines: int = 0

    def changed_paths(self) -> list[str]:
        return [item.path for item in self.applied]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": [vars(item) for item in self.applied],
            "total_changed_lines": self.total_changed_lines,
        }


class PathPolicy:
    """Resolves a repo-relative path to an absolute one, or refuses.

    Containment, protected paths and the allow-list all live here so that every
    write in the system goes through one implementation of the rules.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_paths: Iterable[str] | None = None,
        protected_paths: Iterable[str] | None = None,
        live_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.allowed_paths = [str(item) for item in (allowed_paths or [])]
        self.protected_paths = [str(item) for item in (protected_paths or [])]
        self.live_root = Path(live_root).resolve() if live_root is not None else None

    def resolve(self, relative_path: str) -> Path:
        raw = Path(str(relative_path).replace("\\", "/"))
        normalized = raw.as_posix()

        if not normalized or normalized in {".", "/"}:
            raise EditError("unsafe_path", f"empty edit path: {relative_path!r}", path=normalized)
        if raw.is_absolute() or ".." in raw.parts:
            raise EditError("unsafe_path", f"unsafe repository path: {relative_path}", path=normalized)
        if path_matches(normalized, self.protected_paths):
            raise EditError("protected_path", f"protected repository path: {relative_path}", path=normalized)
        if self.allowed_paths and not path_matches(normalized, self.allowed_paths):
            raise EditError("path_not_allowed", f"path not allowed for this goal: {relative_path}", path=normalized)

        candidate = (self.root / raw).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise EditError("path_escape", f"path escapes workspace: {relative_path}", path=normalized)

        # A symlink in the *live* tree can redirect a write out of the sandbox
        # even when the candidate path itself looks contained.
        if self.live_root is not None:
            live_candidate = (self.live_root / raw).resolve(strict=False)
            if live_candidate.exists() and live_candidate.is_symlink():
                raise EditError("symlink_target", f"refuse to edit symlinked live path: {relative_path}", path=normalized)
        if candidate.is_symlink():
            raise EditError("symlink_target", f"refuse to edit symlinked path: {relative_path}", path=normalized)

        return candidate


def path_matches(path: str, patterns: list[str]) -> bool:
    """True when ``path`` equals or sits under any of ``patterns``."""

    if not patterns:
        return False
    normalized = path.strip("/").replace("\\", "/")
    for pattern in patterns:
        clean = str(pattern).strip("/").replace("\\", "/")
        if clean in {"", "."}:
            return True
        if normalized == clean or normalized.startswith(clean + "/"):
            return True
    return False


# --------------------------------------------------------------------------
# Text normalisation helpers
# --------------------------------------------------------------------------

#: Matches the ``00042: `` / ``42: `` / ``42 | `` gutters produced by code
#: viewers.  Weak models paste these straight back into their anchors.
_LINE_NUMBER_GUTTER = re.compile(r"^\s*\d{1,6}\s*[:|]\s?")


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def strip_line_number_gutter(value: str) -> str:
    """Remove a leading line-number gutter from *every* line, or from none.

    Applying this per-line would corrupt genuine source such as ``1: x`` inside
    a dict literal, so the gutter is only stripped when it is present on all
    non-blank lines -- the signature of a numbered listing rather than of code.
    """

    lines = normalize_newlines(value).split("\n")
    meaningful = [line for line in lines if line.strip()]
    if not meaningful or not all(_LINE_NUMBER_GUTTER.match(line) for line in meaningful):
        return value
    return "\n".join(_LINE_NUMBER_GUTTER.sub("", line) if line.strip() else line for line in lines)


def canonical(value: str) -> str:
    """Whitespace-insensitive, blank-line-insensitive form used for matching."""

    rows = []
    for line in normalize_newlines(value).splitlines():
        compact = re.sub(r"[ \t]+", " ", line.strip())
        if compact:
            rows.append(compact)
    return "\n".join(rows)


def changed_line_count(before: str, after: str) -> int:
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False)
    total = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            total += max(i2 - i1, j2 - j1)
    return total


def stable_hash(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


@dataclass(frozen=True)
class FileFormat:
    """The on-disk conventions of a file, so an edit can put them back."""

    bom: bool = False
    newline: str = "\n"

    def encode(self, text: str) -> bytes:
        if self.newline == "\r\n":
            text = text.replace("\n", "\r\n")
        payload = text.encode("utf-8")
        if self.bom:
            payload = b"\xef\xbb\xbf" + payload
        return payload


def read_source(path: Path) -> tuple[bytes, str, FileFormat]:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    decoded = raw.decode("utf-8-sig", errors="replace")
    newline = "\r\n" if "\r\n" in decoded else "\n"
    return raw, normalize_newlines(decoded), FileFormat(bom=bom, newline=newline)


# --------------------------------------------------------------------------
# Bundle parsing
# --------------------------------------------------------------------------

def parse_bundle(bundle: dict[str, Any]) -> EditPlan:
    """Turn a model-produced change bundle into a validated :class:`EditPlan`.

    Several bundle dialects are accepted because different call sites and
    different prompt templates in this repository grew different shapes, and
    because a small model will use whichever one it saw an example of:

    * ``files: [{path, search, replace}]``  -> :attr:`EditOp.REPLACE`
    * ``files: [{path, content}]``          -> :attr:`EditOp.REWRITE`
    * ``files: [{path, op, ...}]``          -> that operation, explicitly
    * ``edits: [...]``                      -> alias for ``files``
    * ``new_files: [{path, content}]``      -> :attr:`EditOp.CREATE`
    * ``deleted_files: [path]``             -> :attr:`EditOp.DELETE`

    Accepting the dialects here -- rather than at the point of application --
    keeps the applier free of format guesswork.
    """

    if not isinstance(bundle, dict):
        raise EditError("invalid_bundle", "edit bundle must be an object")

    raw_edits = bundle.get("files")
    if raw_edits is None:
        raw_edits = bundle.get("edits")
    if raw_edits is None:
        raw_edits = []
    if not isinstance(raw_edits, list):
        raise EditError("invalid_bundle", "'files' must be an array of edits")

    operations: list[EditOperation] = []

    for item in raw_edits:
        if not isinstance(item, dict):
            raise EditError("invalid_edit", "each edit must be an object")
        operations.append(_parse_edit(item))

    for item in bundle.get("new_files") or []:
        if not isinstance(item, dict):
            raise EditError("invalid_edit", "each new file must be an object")
        path = _require_path(item)
        content = _first_present(item, _CONTENT_KEYS)
        operations.append(
            EditOperation(op=EditOp.CREATE, path=path, content=normalize_newlines(str(content if content is not None else "")))
        )

    for value in bundle.get("deleted_files") or []:
        path = str(value).replace("\\", "/").strip()
        if not path:
            raise EditError("invalid_edit", "deleted_files entries must be non-empty paths")
        operations.append(EditOperation(op=EditOp.DELETE, path=path))

    if not operations:
        raise EditError("empty_plan", "edit bundle contained no operations")

    return EditPlan(operations=operations, analysis=str(bundle.get("analysis", "")))


def _require_path(item: dict[str, Any]) -> str:
    path = str(item.get("path", "")).replace("\\", "/").strip()
    if not path:
        raise EditError("invalid_edit", "edit is missing a 'path'")
    return path


#: Keys a model reaches for when it means "the whole new file contents".
#: Every one of these was produced by a real local model in a live run; each
#: costs a wasted cycle if it is rejected instead of understood.
_CONTENT_KEYS = ("content", "new_content", "new_text", "text", "body", "file_content")


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if item.get(key) is not None:
            return item[key]
    return None


def _parse_edit(item: dict[str, Any]) -> EditOperation:
    path = _require_path(item)
    declared = str(item.get("op") or item.get("operation") or "").strip().lower()

    search = normalize_newlines(str(item.get("search", item.get("anchor", "")) or ""))
    replace_value = item.get("replace")
    content_value = _first_present(item, _CONTENT_KEYS)
    occurrence = item.get("occurrence")
    occurrence_index = int(occurrence) if isinstance(occurrence, (int, float)) and int(occurrence) > 0 else None

    if declared:
        try:
            op = EditOp(declared)
        except ValueError:
            raise EditError("invalid_edit", f"unsupported edit operation {declared!r} for {path}", path=path) from None
    elif search.strip() and replace_value is not None:
        op = EditOp.REPLACE
    elif content_value is not None:
        op = EditOp.REWRITE
    elif replace_value is not None:
        # A replacement with a blank anchor cannot be an anchored edit -- there
        # is nothing to anchor to.  It is recorded as a rewrite *candidate*;
        # whether that reading is safe depends on what is currently in the file,
        # which only the engine can see.  See EditEngine._stage_rewrite.
        #
        # This distinction is not academic.  Asked to add "/bye" to a 243-line
        # CLI, the local model emitted exactly this shape with a two-line
        # replacement, meaning "change this line".  Read as a rewrite it deleted
        # the entire file, and both size budgets passed because the *result* was
        # small.
        op = EditOp.REWRITE
        content_value = replace_value
        return EditOperation(
            op=op,
            path=path,
            search=search,
            replace=normalize_newlines(str(replace_value)),
            content=normalize_newlines(str(content_value)),
            occurrence=occurrence_index,
            anchorless=True,
        )
    elif search.strip():
        raise EditError(
            "invalid_edit",
            f"edit for {path} has a search anchor but no 'replace' text",
            path=path,
            recoverable=True,
        )
    else:
        raise EditError(
            "invalid_edit",
            f"edit for {path} needs either search+replace (to change part of a file) "
            f"or content (to set the whole file). Got keys: {sorted(item)}",
            path=path,
            recoverable=True,
        )

    if op in {EditOp.REPLACE, EditOp.INSERT_BEFORE, EditOp.INSERT_AFTER} and not search.strip():
        raise EditError("empty_search", f"empty search for {path}", path=path, recoverable=True)

    text = normalize_newlines(str(content_value if content_value is not None else ""))
    replacement = normalize_newlines(str(replace_value if replace_value is not None else ""))

    if op in {EditOp.INSERT_BEFORE, EditOp.INSERT_AFTER} and not replacement:
        # Insert operations carry their payload in whichever field the model
        # chose; both readings mean the same thing here.
        replacement = text

    if op in {EditOp.REWRITE, EditOp.CREATE} and not text and replacement:
        text = replacement

    return EditOperation(op=op, path=path, search=search, replace=replacement, content=text, occurrence=occurrence_index)


# --------------------------------------------------------------------------
# Anchor location
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Match:
    start: int
    end: int
    mode: str


def locate(current: str, search: str, relative: str, *, occurrence: int | None = None) -> Match:
    """Find exactly one span in ``current`` corresponding to ``search``.

    The escalation order is deliberate.  Each step is strictly more forgiving
    than the last, and the first step that yields a *single* answer wins, so a
    file that contains a literal exact match is never resolved by similarity.
    """

    search = normalize_newlines(search)
    if not search.strip():
        raise EditError("empty_search", f"empty search for {relative}", path=relative, recoverable=True)

    # 1. Exact substring.
    exact_spans = _all_spans(current, search)
    if len(exact_spans) == 1:
        start = exact_spans[0]
        return Match(start, start + len(search), "exact")
    if len(exact_spans) > 1:
        if occurrence is not None and 1 <= occurrence <= len(exact_spans):
            start = exact_spans[occurrence - 1]
            return Match(start, start + len(search), "exact_occurrence")
        raise EditError(
            "ambiguous_search",
            f"search must match exactly once in {relative}; matched {len(exact_spans)}. "
            "Extend the search text with surrounding lines until it is unique.",
            path=relative,
            recoverable=True,
        )

    # 2. Same, after stripping a pasted line-number gutter.
    deguttered = strip_line_number_gutter(search)
    if deguttered != search:
        spans = _all_spans(current, deguttered)
        if len(spans) == 1:
            start = spans[0]
            return Match(start, start + len(deguttered), "line_numbers_stripped")
        if len(spans) > 1 and occurrence is not None and 1 <= occurrence <= len(spans):
            start = spans[occurrence - 1]
            return Match(start, start + len(deguttered), "line_numbers_stripped")
        search = deguttered

    # 3./4. Whitespace-canonical, then similarity, over line windows.
    return _locate_by_window(current, search, relative)


def _all_spans(haystack: str, needle: str) -> list[int]:
    spans: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        spans.append(start)
        start = haystack.find(needle, start + 1)
        if len(spans) > 64:
            break
    return spans


def _locate_by_window(current: str, search: str, relative: str) -> Match:
    search_lines = search.splitlines() or [search]
    if len(search_lines) > 40:
        raise EditError(
            "no_unique_match",
            f"search block too long to match safely in {relative}; use a shorter unique anchor",
            path=relative,
            recoverable=True,
        )

    current_lines = current.splitlines(keepends=True)
    if not current_lines:
        raise EditError("no_unique_match", f"no safe unique match for {relative} (file is empty)", path=relative, recoverable=True)

    offsets = [0]
    for line in current_lines:
        offsets.append(offsets[-1] + len(line))

    target = canonical(search)
    if not target:
        raise EditError("empty_search", f"empty search for {relative}", path=relative, recoverable=True)

    window_sizes = sorted({max(1, len(search_lines) - 1), len(search_lines), len(search_lines) + 1})
    candidates: list[tuple[float, int, int]] = []
    canonical_hits: list[tuple[int, int]] = []

    for size in window_sizes:
        if size < 1 or size > len(current_lines):
            continue
        for index in range(0, len(current_lines) - size + 1):
            start = offsets[index]
            end = offsets[index + size]
            window_canonical = canonical(current[start:end])
            if not window_canonical:
                continue
            if window_canonical == target:
                canonical_hits.append((start, end))
                continue
            ratio = difflib.SequenceMatcher(None, target, window_canonical, autojunk=False).ratio()
            candidates.append((ratio, start, end))

    if canonical_hits:
        # Overlapping windows of different sizes can describe the same region;
        # that is one hit, not several.
        merged = _merge_overlapping(canonical_hits)
        if len(merged) == 1:
            start, end = merged[0]
            return Match(start, end, "canonical")
        raise EditError(
            "ambiguous_search",
            f"search matches {len(merged)} different places in {relative}; make the anchor unique",
            path=relative,
            recoverable=True,
        )

    if not candidates:
        raise EditError("no_unique_match", f"no safe unique match for {relative}", path=relative, recoverable=True)

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, start, end = candidates[0]

    # A short anchor is cheap to get accidentally-similar, so it needs a higher
    # bar than a long one.
    minimum = 0.88 if len(search_lines) <= 2 else 0.91
    if best_score < minimum:
        raise EditError(
            "no_unique_match",
            f"no safe unique match for {relative} (best similarity {best_score:.2f} < {minimum})",
            path=relative,
            recoverable=True,
        )

    runner_up = next((score for score, s, e in candidates[1:] if not _overlaps(s, e, start, end)), 0.0)
    if runner_up >= best_score - 0.035:
        raise EditError(
            "ambiguous_search",
            f"no safe unique match for {relative} (two candidates within {best_score - runner_up:.3f})",
            path=relative,
            recoverable=True,
        )

    return Match(start, end, "fuzzy")


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _merge_overlapping(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

class EditEngine:
    """Applies :class:`EditPlan` objects atomically to a workspace."""

    def __init__(
        self,
        policy: PathPolicy,
        *,
        budget: EditBudget | None = None,
        expected_hashes: dict[str, str] | None = None,
    ) -> None:
        self.policy = policy
        self.budget = budget or EditBudget()
        #: repo-relative path -> hash of the content the model was shown.  A
        #: mismatch means the model is editing a version of the file that no
        #: longer exists, which is never safe to apply blind.
        self.expected_hashes = dict(expected_hashes or {})

    # -- public API ------------------------------------------------------

    def apply(self, plan: EditPlan) -> EditResult:
        """Apply the whole plan, or change nothing at all."""

        if len(plan.operations) > self.budget.max_operations:
            raise EditError(
                "too_many_operations",
                f"too many edits: {len(plan.operations)} > {self.budget.max_operations}",
            )

        state = _WorkingState()
        result = EditResult()

        for operation in plan.operations:
            self._stage(operation, state, result)

        self._enforce_change_budget(state, result)
        self._check_syntax(state)
        self._commit(state)
        self._refresh_hashes(state)
        return result

    def preview(self, plan: EditPlan) -> dict[str, str]:
        """Resolve the plan without writing, returning the would-be contents.

        Used by callers that want to type-check or lint a candidate before it
        touches the workspace.
        """

        state = _WorkingState()
        result = EditResult()
        for operation in plan.operations:
            self._stage(operation, state, result)
        self._enforce_change_budget(state, result)
        return {state.relatives[path]: text for path, text in state.working.items()}

    # -- staging ---------------------------------------------------------

    def _stage(self, operation: EditOperation, state: "_WorkingState", result: EditResult) -> None:
        path = self.policy.resolve(operation.path)
        relative = operation.path.replace("\\", "/")

        if operation.op is EditOp.CREATE:
            self._stage_create(operation, path, relative, state, result)
            return
        if operation.op is EditOp.DELETE:
            self._stage_delete(path, relative, state, result)
            return
        if operation.op is EditOp.REWRITE:
            self._stage_rewrite(operation, path, relative, state, result)
            return
        self._stage_anchored(operation, path, relative, state, result)

    def _stage_create(
        self, operation: EditOperation, path: Path, relative: str, state: "_WorkingState", result: EditResult
    ) -> None:
        if relative in state.deleted_relatives:
            raise EditError("conflicting_ops", f"cannot delete and create the same file: {relative}", path=relative)
        if path.exists() or path in state.working:
            raise EditError(
                "create_over_existing",
                f"new_files cannot replace existing file: {relative}; edit it instead",
                path=relative,
                recoverable=True,
            )
        content = operation.content
        if len(content) > self.budget.max_new_file_chars:
            raise EditError(
                "file_too_large",
                f"new file too large: {relative} ({len(content)} > {self.budget.max_new_file_chars} chars)",
                path=relative,
            )
        state.working[path] = content
        state.formats[path] = FileFormat()
        state.originals[path] = None
        state.relatives[path] = relative
        state.created.add(path)
        result.applied.append(AppliedEdit(path=relative, op=operation.op.value, created=True))

    def _stage_delete(self, path: Path, relative: str, state: "_WorkingState", result: EditResult) -> None:
        if path in state.working and path not in state.created:
            raise EditError("conflicting_ops", f"cannot edit and delete the same file: {relative}", path=relative)
        if not path.exists() or not path.is_file():
            raise EditError(
                "missing_target", f"delete target does not exist: {relative}", path=relative, recoverable=True
            )
        raw, current, _ = read_source(path)
        state.deleted[path] = raw
        state.deleted_relatives.add(relative)
        state.relatives[path] = relative
        state.working.pop(path, None)
        state.created.discard(path)
        result.applied.append(AppliedEdit(path=relative, op=EditOp.DELETE.value, changed_lines=len(current.splitlines()), deleted=True))

    def _stage_rewrite(
        self, operation: EditOperation, path: Path, relative: str, state: "_WorkingState", result: EditResult
    ) -> None:
        content = operation.content
        if not path.exists():
            # A model that says "rewrite" about a file that is not there means
            # "create"; treating it as an error would burn a repair cycle on a
            # vocabulary mismatch.
            self._stage_create(
                EditOperation(op=EditOp.CREATE, path=operation.path, content=content), path, relative, state, result
            )
            return
        current = self._load(path, relative, state)

        if operation.anchorless and current.strip():
            # The model gave a replacement with no anchor for a file that
            # already has content.  Reading that as "make the file this" would
            # discard everything else in it, which is never what was meant.
            raise EditError(
                "empty_search",
                f"edit for {relative} has replacement text but an empty search anchor. "
                f"{relative} already has content, so copy the exact lines you want to change "
                "into 'search', or send the complete new file as 'content'.",
                path=relative,
                recoverable=True,
            )

        if len(content) > self.budget.max_rewrite_chars:
            raise EditError(
                "rewrite_too_large",
                f"whole-file rewrite of {relative} is {len(content)} chars "
                f"(> {self.budget.max_rewrite_chars}); express this as targeted search/replace edits",
                path=relative,
                recoverable=True,
            )
        if content == current:
            raise EditError("no_effective_edit", f"no effective edit for {relative}", path=relative, recoverable=True)

        self._guard_against_truncation(relative, current, content)
        state.working[path] = content
        result.applied.append(
            AppliedEdit(path=relative, op=EditOp.REWRITE.value, match_mode="whole_file", changed_lines=changed_line_count(current, content))
        )

    def _guard_against_truncation(self, relative: str, current: str, replacement: str) -> None:
        """Refuse a rewrite that would delete most of an existing file.

        The size budgets bound how *large* a change may be; nothing bounded how
        much a change may *remove*, and shrinking is the direction that destroys
        work.  A model that replies with a fragment when asked to modify a file
        produces a tiny, cheap, budget-passing edit that deletes hundreds of
        lines -- exactly what happened to a 243-line CLI during a live run.

        Deliberately not a total ban: deleting most of a small file is ordinary
        editing, and an explicit whole-file rewrite of a large file is still
        allowed as long as it is recognisably the same file.
        """

        current_lines = len([line for line in current.splitlines() if line.strip()])
        if current_lines < self.budget.truncation_floor_lines:
            return

        new_lines = len([line for line in replacement.splitlines() if line.strip()])
        if new_lines >= current_lines * self.budget.min_retained_fraction:
            return

        raise EditError(
            "rewrite_truncates_file",
            f"refusing to shrink {relative} from {current_lines} to {new_lines} non-blank lines "
            f"({new_lines / max(1, current_lines):.0%} retained, minimum "
            f"{self.budget.min_retained_fraction:.0%}). If you meant to change part of the file, "
            "send a search/replace edit; if you really meant to replace all of it, send the complete "
            "new file, not a fragment.",
            path=relative,
            recoverable=True,
        )

    def _stage_anchored(
        self, operation: EditOperation, path: Path, relative: str, state: "_WorkingState", result: EditResult
    ) -> None:
        if len(operation.search) > self.budget.max_search_chars:
            raise EditError("edit_too_large", f"edit too large for {relative}: search anchor is {len(operation.search)} chars", path=relative, recoverable=True)
        if len(operation.replace) > self.budget.max_replace_chars:
            raise EditError("edit_too_large", f"edit too large for {relative}: replacement is {len(operation.replace)} chars", path=relative, recoverable=True)

        current = self._load(path, relative, state)
        match = locate(current, operation.search, relative, occurrence=operation.occurrence)
        matched = current[match.start : match.end]

        if operation.op is EditOp.REPLACE:
            replacement = operation.replace
            # A canonical/fuzzy match may have absorbed the trailing newline of
            # the last matched line; put it back so the following line does not
            # get glued onto the replacement.
            if match.mode not in {"exact", "exact_occurrence"} and matched.endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
            updated = current[: match.start] + replacement + current[match.end :]
        elif operation.op is EditOp.INSERT_BEFORE:
            payload = operation.replace
            if not payload.endswith("\n"):
                payload += "\n"
            updated = current[: match.start] + payload + current[match.start :]
        else:  # INSERT_AFTER
            payload = operation.replace
            if not payload.endswith("\n"):
                payload += "\n"
            insert_at = match.end
            if not matched.endswith("\n"):
                # Anchor did not include its line break; step past it so the
                # insertion lands on its own line rather than mid-line.
                newline_at = current.find("\n", match.end)
                insert_at = len(current) if newline_at == -1 else newline_at + 1
                if newline_at == -1:
                    payload = "\n" + payload
            updated = current[:insert_at] + payload + current[insert_at:]

        if updated == current:
            raise EditError("no_effective_edit", f"no effective edit for {relative}", path=relative, recoverable=True)

        state.working[path] = updated
        result.applied.append(
            AppliedEdit(
                path=relative,
                op=operation.op.value,
                match_mode=match.mode,
                changed_lines=changed_line_count(current, updated),
            )
        )

    # -- helpers ---------------------------------------------------------

    def _load(self, path: Path, relative: str, state: "_WorkingState") -> str:
        if path in state.working:
            return state.working[path]
        if path in state.deleted:
            raise EditError("conflicting_ops", f"cannot edit a deleted file: {relative}", path=relative)
        if not path.exists() or not path.is_file():
            raise EditError(
                "missing_target",
                f"existing edit target does not exist: {relative}",
                path=relative,
                recoverable=True,
            )

        raw, current, file_format = read_source(path)

        expected = self.expected_hashes.get(relative)
        if expected and expected != stable_hash(current):
            raise EditError(
                "stale_context",
                f"stale file context for {relative}; re-read the file before editing it",
                path=relative,
                recoverable=True,
            )

        state.originals[path] = current
        state.original_bytes[path] = raw
        state.working[path] = current
        state.formats[path] = file_format
        state.relatives[path] = relative
        return current

    def _enforce_change_budget(self, state: "_WorkingState", result: EditResult) -> None:
        total = sum(item.changed_lines for item in result.applied)
        result.total_changed_lines = total
        if total > self.budget.max_changed_lines:
            raise EditError(
                "changed_line_budget",
                f"changed-line budget exceeded: {total} > {self.budget.max_changed_lines}",
            )

    def _check_syntax(self, state: "_WorkingState") -> None:
        """Refuse a plan that would leave a Python file unable to parse.

        Checked in memory, before anything is written, so the guarantee stays
        all-or-nothing.  This is the cheapest verification in the system and one
        of the most valuable: a small model editing by anchor routinely lands
        the right text at the right place with the wrong indentation, and
        without this the broken file reaches the test runner, where the failure
        is reported as a test error rather than as the edit mistake it is.

        A file that did not parse *before* the edit is left alone -- the edit is
        not to blame for a pre-existing syntax error, and refusing would make it
        impossible to repair one.
        """

        import ast

        for path, text in state.working.items():
            relative = state.relatives.get(path, path.name)
            if not relative.endswith(".py"):
                continue
            try:
                ast.parse(text)
            except SyntaxError as exc:
                original = state.originals.get(path)
                if original is not None:
                    try:
                        ast.parse(original)
                    except SyntaxError:
                        continue  # it was already broken; not this edit's doing
                line = exc.lineno or 0
                context = "\n".join(
                    f"{number:>5}: {content}"
                    for number, content in enumerate(text.splitlines()[max(0, line - 4) : line + 2], start=max(1, line - 3))
                )
                raise EditError(
                    "syntax_error",
                    f"the edit would leave {relative} unparseable: {exc.msg} at line {line}. "
                    f"Check the indentation of your replacement text.\n{context}",
                    path=relative,
                    recoverable=True,
                ) from None

    def _commit(self, state: "_WorkingState") -> None:
        """Write everything, restoring the original bytes if anything fails."""

        written: list[Path] = []
        try:
            for path, text in state.working.items():
                file_format = state.formats.get(path, FileFormat())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file_format.encode(text))
                written.append(path)
            for path in state.deleted:
                if path.exists():
                    path.unlink()
                    written.append(path)
        except Exception:
            self._rollback(state)
            raise

    def _rollback(self, state: "_WorkingState") -> None:
        for path, raw in state.original_bytes.items():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            except OSError:
                pass
        for path, raw in state.deleted.items():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            except OSError:
                pass
        for path in state.created:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    def _refresh_hashes(self, state: "_WorkingState") -> None:
        """Keep the staleness baseline in step with what is now on disk."""

        for path, text in state.working.items():
            self.expected_hashes[state.relatives[path]] = stable_hash(text)
        for path in state.deleted:
            self.expected_hashes.pop(state.relatives.get(path, ""), None)


@dataclass
class _WorkingState:
    originals: dict[Path, str | None] = field(default_factory=dict)
    original_bytes: dict[Path, bytes] = field(default_factory=dict)
    working: dict[Path, str] = field(default_factory=dict)
    formats: dict[Path, FileFormat] = field(default_factory=dict)
    relatives: dict[Path, str] = field(default_factory=dict)
    created: set[Path] = field(default_factory=set)
    deleted: dict[Path, bytes] = field(default_factory=dict)
    deleted_relatives: set[str] = field(default_factory=set)


def edit_schema(*, max_edits: int = 8, max_new_files: int = 4, allow_rewrite: bool = True) -> dict[str, Any]:
    """JSON schema for the change bundle, for guided/structured generation.

    Placeholders in examples are deliberately meaningless (``"..."``): a
    plausible-looking English phrase in a schema example gets echoed back
    verbatim as content by small models.

    ``allow_rewrite=False`` removes whole-file replacement from the schema
    entirely, leaving only anchored edits for existing files and ``new_files``
    for new ones.  Under constrained decoding that makes a destructive rewrite
    *unrepresentable* rather than merely discouraged.  That distinction is not
    theoretical: asked to add one command to a 189-line CLI, the local model
    emitted a whole-file rewrite sixteen times in a row despite the prompt
    telling it not to, because the schema still offered the field.
    """

    edit_properties: dict[str, Any] = {
        "path": {"type": "string"},
        "op": {
            "type": "string",
            "enum": ["replace", "insert_before", "insert_after"] + (["rewrite"] if allow_rewrite else []),
        },
        "search": {"type": "string"},
        "replace": {"type": "string"},
    }
    if allow_rewrite:
        edit_properties["content"] = {"type": "string"}

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "analysis": {"type": "string"},
            "files": {
                "type": "array",
                "maxItems": max_edits,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": edit_properties,
                    "required": ["path", "search", "replace"] if not allow_rewrite else ["path"],
                },
            },
            "new_files": {
                "type": "array",
                "maxItems": max_new_files,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
            "deleted_files": {"type": "array", "maxItems": max_new_files, "items": {"type": "string"}},
        },
        "required": ["analysis", "files"],
    }
