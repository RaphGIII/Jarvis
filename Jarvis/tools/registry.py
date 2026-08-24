"""A declarative, audited tool runtime.

The division of labour throughout Jarvis is: the model decides *what* should
happen, deterministic Python decides *how* and *whether*.  This module is the
gate between those two halves.  A tool is data -- a name, a purpose, an input
schema, a risk level -- plus one adapter function.  Everything a model can cause
to happen goes through :meth:`ToolRegistry.invoke`, which means there is exactly
one place that enforces permissions, timeouts, output caps and the audit trail.

Three properties matter more than convenience here:

*Risk is declared, not inferred.*  Every tool carries a :class:`RiskLevel`.  A
policy admits a maximum level and an optional approval callback, so raising the
ceiling for one autonomous run is a deliberate act rather than a side effect of
registering a new tool.

*Failure is data.*  An adapter that raises produces a :class:`ToolResult` with
``ok=False`` and the error text, not an exception that unwinds the agent loop.
An autonomous agent has to be able to observe a failed tool call and try
something else -- that is the whole point of the loop.

*Everything is recorded.*  Each invocation appends to an audit log with its
arguments, outcome and duration, which is also the raw material for the training
datasets exported later.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable


class RiskLevel(IntEnum):
    """How much damage a tool can do if the model is wrong about using it."""

    #: Pure reads inside the workspace.
    SAFE = 0
    #: Writes inside the workspace; reversible via version control.
    LOW = 1
    #: Runs code, installs into an isolated environment, reaches the network.
    MODERATE = 2
    #: Touches state outside the workspace, or is hard to undo.
    HIGH = 3
    #: Can affect the host or the user's real accounts.
    CRITICAL = 4


class ToolError(RuntimeError):
    """An adapter's structured failure.  Becomes ``ToolResult.error``."""

    def __init__(self, message: str, *, kind: str = "tool_error", retryable: bool = True) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class ToolDenied(ToolError):
    """The policy refused the call.  Never retryable by re-asking the model."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="permission_denied", retryable=False)


@dataclass
class ToolContext:
    """Ambient state every adapter receives.

    Passing this explicitly, rather than letting adapters read globals, is what
    makes a tool usable against a project workspace, a candidate worktree or a
    test fixture without changing the tool.
    """

    workspace: Path
    #: Extra roots an adapter may read from (e.g. the repo under study).
    readable_roots: list[Path] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    protected_paths: list[str] = field(default_factory=list)
    timeout_seconds: float = 120.0
    max_output_chars: int = 20000
    environment: dict[str, str] = field(default_factory=dict)
    #: Free-form state shared between the agent loop and its tools.
    scratch: dict[str, Any] = field(default_factory=dict)

    def with_workspace(self, workspace: str | Path) -> "ToolContext":
        return ToolContext(
            workspace=Path(workspace).resolve(),
            readable_roots=list(self.readable_roots),
            allowed_paths=list(self.allowed_paths),
            protected_paths=list(self.protected_paths),
            timeout_seconds=self.timeout_seconds,
            max_output_chars=self.max_output_chars,
            environment=dict(self.environment),
            scratch=self.scratch,
        )


@dataclass(frozen=True)
class ToolSpec:
    """The declaration of one tool."""

    name: str
    purpose: str
    input_schema: dict[str, Any]
    adapter: Callable[[dict[str, Any], ToolContext], Any]
    risk: RiskLevel = RiskLevel.SAFE
    timeout_seconds: float | None = None
    tags: tuple[str, ...] = ()
    #: A one-line example, shown to the model.  Concrete examples raise the hit
    #: rate of small models dramatically compared with schema alone.
    example: str = ""

    def describe(self) -> dict[str, Any]:
        """The form shown to a model.  Deliberately excludes the adapter."""

        payload = {
            "name": self.name,
            "purpose": self.purpose,
            "risk": self.risk.name,
            "input_schema": self.input_schema,
        }
        if self.example:
            payload["example"] = self.example
        return payload


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ToolCall":
        """Build a call from a model's JSON, tolerating the usual variations."""

        if not isinstance(payload, dict):
            raise ToolError("tool call must be an object", kind="invalid_call", retryable=True)
        name = str(payload.get("name") or payload.get("tool") or payload.get("action") or "").strip()
        if not name:
            raise ToolError("tool call is missing a name", kind="invalid_call", retryable=True)
        raw_args = payload.get("arguments")
        if raw_args is None:
            raw_args = payload.get("args")
        if raw_args is None:
            raw_args = payload.get("input")
        if raw_args is None:
            # Some models inline the arguments alongside the name.
            raw_args = {k: v for k, v in payload.items() if k not in {"name", "tool", "action", "call_id", "thought"}}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raise ToolError(f"arguments for {name} were not valid JSON", kind="invalid_call", retryable=True) from None
        if not isinstance(raw_args, dict):
            raise ToolError(f"arguments for {name} must be an object", kind="invalid_call", retryable=True)
        return cls(name=name, arguments=raw_args, call_id=str(payload.get("call_id") or uuid.uuid4().hex[:12]))


@dataclass
class ToolResult:
    """The observation an agent gets back.  Failures are results, not raises."""

    name: str
    ok: bool
    output: Any = None
    error: str = ""
    error_kind: str = ""
    retryable: bool = True
    duration_seconds: float = 0.0
    truncated: bool = False
    call_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def observation(self, *, max_chars: int = 4000) -> str:
        """A compact rendering suitable for putting back into a prompt."""

        if self.ok:
            body = self.output if isinstance(self.output, str) else json.dumps(self.output, indent=2, sort_keys=True, default=str)
        else:
            body = f"ERROR ({self.error_kind}): {self.error}"
        if len(body) > max_chars:
            body = body[: max_chars - 40] + "\n...[observation truncated]"
        return f"[{self.name}] {'ok' if self.ok else 'failed'}\n{body}"


class ToolPolicy:
    """Decides whether a call is allowed before the adapter ever runs."""

    def __init__(
        self,
        *,
        max_risk: RiskLevel = RiskLevel.MODERATE,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        approve: Callable[[ToolSpec, ToolCall], bool] | None = None,
    ) -> None:
        self.max_risk = max_risk
        #: When set, acts as an exclusive allow-list.
        self.allow = set(allow) if allow is not None else None
        self.deny = set(deny or ())
        #: Consulted for tools above ``max_risk``.  Without one, they are
        #: refused -- an unattended run must never silently escalate itself.
        self.approve = approve

    def check(self, spec: ToolSpec, call: ToolCall) -> None:
        if spec.name in self.deny:
            raise ToolDenied(f"tool {spec.name} is denied by policy")
        if self.allow is not None and spec.name not in self.allow:
            raise ToolDenied(f"tool {spec.name} is not in the allow-list for this run")
        if spec.risk > self.max_risk:
            if self.approve is None:
                raise ToolDenied(
                    f"tool {spec.name} has risk {spec.risk.name}, above the limit {self.max_risk.name} for this run"
                )
            if not self.approve(spec, call):
                raise ToolDenied(f"approval for {spec.name} ({spec.risk.name}) was declined")


class AuditLog:
    """Append-only JSONL record of every tool invocation."""

    #: Argument names whose values must never be written down.
    _SECRET_KEYS = ("api_key", "token", "secret", "password", "authorization", "credential")

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []

    def record(self, result: ToolResult) -> None:
        entry = {
            "timestamp": result.started_at or datetime.now(timezone.utc).isoformat(),
            "tool": result.name,
            "call_id": result.call_id,
            "arguments": self.redact(result.arguments),
            "ok": result.ok,
            "error": result.error,
            "error_kind": result.error_kind,
            "duration_seconds": round(result.duration_seconds, 4),
            "truncated": result.truncated,
        }
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    @classmethod
    def redact(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("<redacted>" if any(marker in str(key).lower() for marker in cls._SECRET_KEYS) else cls.redact(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls.redact(item) for item in value]
        if isinstance(value, str) and len(value) > 2000:
            return value[:2000] + "...[truncated]"
        return value


class ToolRegistry:
    """Holds tool declarations and is the only way to invoke them."""

    def __init__(self, *, policy: ToolPolicy | None = None, audit: AuditLog | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.policy = policy or ToolPolicy()
        self.audit = audit or AuditLog(None)

    # -- registration ----------------------------------------------------

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def register_many(self, specs: Iterable[ToolSpec], *, replace: bool = False) -> None:
        for spec in specs:
            self.register(spec, replace=replace)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name] for name in self.names()]

    # -- description for prompting ---------------------------------------

    def describe(
        self,
        *,
        tags: Iterable[str] | None = None,
        max_risk: RiskLevel | None = None,
        exclude: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List the tools a model should be told about.

        Filtered by the same policy that will judge the call, so the model is
        never shown a tool it would be refused -- offering one is a reliable way
        to waste a cycle of a small model's attention.
        """

        wanted = set(tags) if tags else None
        hidden = set(exclude or ())
        ceiling = max_risk if max_risk is not None else self.policy.max_risk
        described = []
        for spec in self.specs():
            if spec.name in hidden:
                continue
            if wanted and not wanted.intersection(spec.tags):
                continue
            if spec.risk > ceiling and self.policy.approve is None:
                continue
            if self.policy.allow is not None and spec.name not in self.policy.allow:
                continue
            if spec.name in self.policy.deny:
                continue
            described.append(spec.describe())
        return described

    def render_for_prompt(self, *, tags: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> str:
        lines = []
        for item in self.describe(tags=tags, exclude=exclude):
            required = item["input_schema"].get("required") or []
            properties = item["input_schema"].get("properties") or {}
            args = ", ".join(
                f"{name}{'' if name in required else '?'}: {str(details.get('type', 'any'))}"
                for name, details in properties.items()
            )
            lines.append(f"- {item['name']}({args}) -- {item['purpose']}")
            if item.get("example"):
                lines.append(f"    example: {item['example']}")
        return "\n".join(lines)

    # -- invocation ------------------------------------------------------

    def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Run one tool call.  Never raises for an ordinary failure."""

        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        spec = self._tools.get(call.name)

        def finish(result: ToolResult) -> ToolResult:
            result.duration_seconds = time.perf_counter() - started
            result.started_at = started_at
            result.call_id = call.call_id
            result.arguments = dict(call.arguments)
            self.audit.record(result)
            return result

        if spec is None:
            close = _closest(call.name, self.names())
            hint = f" Did you mean {close}?" if close else ""
            return finish(
                ToolResult(
                    name=call.name,
                    ok=False,
                    error=f"unknown tool {call.name!r}.{hint} Available: {', '.join(self.names())}",
                    error_kind="unknown_tool",
                )
            )

        try:
            self.policy.check(spec, call)
            arguments = _validate_arguments(spec, call.arguments)
        except ToolError as exc:
            return finish(ToolResult(name=spec.name, ok=False, error=str(exc), error_kind=exc.kind, retryable=exc.retryable))

        scoped = context
        if spec.timeout_seconds is not None:
            scoped = context.with_workspace(context.workspace)
            scoped.timeout_seconds = spec.timeout_seconds

        try:
            output = spec.adapter(arguments, scoped)
        except ToolError as exc:
            return finish(ToolResult(name=spec.name, ok=False, error=str(exc), error_kind=exc.kind, retryable=exc.retryable))
        except Exception as exc:  # an adapter bug must not kill the agent loop
            return finish(
                ToolResult(name=spec.name, ok=False, error=f"{type(exc).__name__}: {exc}", error_kind="adapter_exception")
            )

        output, truncated = _cap_output(output, context.max_output_chars)
        return finish(ToolResult(name=spec.name, ok=True, output=output, truncated=truncated))

    def invoke_payload(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            call = ToolCall.from_payload(payload)
        except ToolError as exc:
            result = ToolResult(name=str(payload.get("name", "?")), ok=False, error=str(exc), error_kind=exc.kind)
            result.started_at = datetime.now(timezone.utc).isoformat()
            self.audit.record(result)
            return result
        return self.invoke(call, context)


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Check required keys and coerce obvious type mismatches.

    Full JSON-Schema validation would be overkill and would reject calls that
    are trivially fixable -- a small model writing ``"3"`` where an integer was
    wanted should not cost a whole cycle.
    """

    schema = spec.input_schema or {}
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    missing = [name for name in required if name not in arguments or arguments[name] is None]
    if missing:
        raise ToolError(
            f"{spec.name} is missing required argument(s): {', '.join(missing)}",
            kind="invalid_arguments",
            retryable=True,
        )

    coerced: dict[str, Any] = {}
    for name, value in arguments.items():
        details = properties.get(name)
        if not isinstance(details, dict):
            coerced[name] = value
            continue
        coerced[name] = _coerce_value(spec.name, name, details.get("type"), value)
    return coerced


def _coerce_value(tool: str, name: str, wanted: Any, value: Any) -> Any:
    if wanted is None or value is None:
        return value
    try:
        if wanted == "string" and not isinstance(value, str):
            return value if isinstance(value, (dict, list)) else str(value)
        if wanted == "integer" and not isinstance(value, int):
            return int(str(value).strip())
        if wanted == "number" and not isinstance(value, (int, float)):
            return float(str(value).strip())
        if wanted == "boolean" and not isinstance(value, bool):
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if wanted == "array" and isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            raise ValueError("not a list")
    except (TypeError, ValueError):
        raise ToolError(
            f"{tool}: argument {name!r} should be {wanted}, got {value!r}", kind="invalid_arguments", retryable=True
        ) from None
    return value


def _cap_output(output: Any, max_chars: int) -> tuple[Any, bool]:
    if isinstance(output, str) and len(output) > max_chars:
        return output[:max_chars] + "\n...[output truncated]", True
    if isinstance(output, (dict, list)):
        rendered = json.dumps(output, default=str)
        if len(rendered) > max_chars:
            return {"truncated": True, "preview": rendered[:max_chars]}, True
    return output, False


def _closest(name: str, candidates: list[str]) -> str:
    import difflib

    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else ""
