"""Execute what the user asked for, and report only what actually happened.

The division of labour here is the whole point, so it is stated plainly:

    **The model proposes. The executor disposes. Code writes the verdict.**

A model is good at turning *"erstelle die Datei zeus_test.txt mit exakt dem
Inhalt ZEUS funktioniert"* into ``{"action": "file.write", "path":
"zeus_test.txt", "content": "ZEUS funktioniert"}``.  That is extraction, and
extraction is checkable -- if it gets the filename wrong, the user sees the
wrong filename in the receipt and says so.

A model is not good at knowing whether a write succeeded, because it cannot see
the filesystem, and when it cannot see something it produces the most plausible
continuation instead.  That is why nothing in this module asks it to.  The
outcome sentence the user reads is built by :func:`compose` out of a
:class:`~runtime.receipts.Receipt`, and a receipt is only ever constructed by
the code that ran the thing.

There is no prompt anywhere in this file telling the model to be honest.  There
is nowhere for it to be dishonest.

Scope is deliberately narrow.  The live chat surface can write a file, read a
file, create a project, and answer from the registries -- and it declines
anything else in so many words.  Widening it means adding an executor with real
verification, not adding a sentence to a prompt; a chat turn that could invoke
``run_command`` would be a much larger security decision than this fix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.receipts import Receipt, Verification, failed

#: Where a file action is allowed to write.  Not the repository: a chat turn
#: that can drop files into the source tree is a different, larger decision than
#: the one this fix is making.  Under the state root, so it is discoverable,
#: backed up with everything else, and inside the tree the user searches.
WORKSPACE_DIRNAME = "workspace"

#: How much of a file the executor will read back to verify it.  A readback is
#: evidence, not a transfer; comparing the first megabyte of a log file proves
#: what needs proving.
MAX_VERIFY_BYTES = 1_000_000


@dataclass
class ActionPlan:
    """What the model proposed doing.  Not yet a claim that anything happened."""

    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Why the model declined, when ``action`` is ``none``.
    reason: str = ""

    @property
    def declined(self) -> bool:
        return self.action in {"", "none"}

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "arguments": dict(self.arguments), "reason": self.reason}


SUPPORTED_ACTIONS = ("file.write", "file.read", "project.create", "capability", "none")


PLANNER_PROMPT = """You turn a user's request into one machine-readable action. You do not perform it.

Available actions:
  {{"action": "file.write", "path": "<relative filename>", "content": "<exact file contents>"}}
  {{"action": "file.read", "path": "<relative filename>"}}
  {{"action": "project.create", "name": "<project name>"}}
  {{"action": "capability", "goal": "<the real-world thing to do>"}}
  {{"action": "none", "reason": "<why none of the above fits>"}}

Use "capability" when the user is asking for something to actually HAPPEN on
this computer that the first three cannot do -- taking a screenshot, controlling
a device, reading a sensor, converting a file. Describe the outcome, not how to
achieve it.

Use "none" for anything that is not a real-world action: questions, opinions,
explanations, jokes, poems, chat. Writing a poem is not a capability.

Rules:
- Reply with one JSON object and nothing else. No explanation, no code fence.
- Copy content and names EXACTLY as the user gave them, including capitalisation.
- Never invent a path the user did not give. Use a bare filename.

User request:
{request}

JSON:"""


class ActionExecutor:
    """Runs real actions against real subsystems and returns receipts."""

    def __init__(
        self,
        kernel: Any,
        *,
        tools: Any = None,
        workspace: str | Path | None = None,
    ) -> None:
        self.kernel = kernel
        #: Injectable so a test can substitute a registry whose ``write_file``
        #: genuinely fails.  Acceptance requires proving that a failing tool
        #: produces an honest failure, and the only way to prove that is to
        #: make the tool fail for real rather than to assert about a mock.
        self._tools = tools
        self._workspace = Path(workspace) if workspace else None

    # -- wiring ----------------------------------------------------------

    @property
    def tools(self) -> Any:
        if self._tools is None:
            self._tools = self.kernel.tools
        return self._tools

    @property
    def workspace(self) -> Path:
        if self._workspace is None:
            self._workspace = Path(self.kernel.state_root) / WORKSPACE_DIRNAME
        self._workspace.mkdir(parents=True, exist_ok=True)
        return self._workspace

    # -- planning --------------------------------------------------------

    def plan(self, request: str, provider: Any) -> ActionPlan:
        """Ask the model what to do.  Its answer is a proposal, never a result."""

        from brain.json_utils import lenient_json_loads

        prompt = PLANNER_PROMPT.format(request=request.strip())
        try:
            raw = provider.generate(prompt, max_tokens=400, temperature=0.0)
        except TypeError:
            # Providers differ on which knobs they accept; the prompt is what
            # matters and a planner that dies on a keyword argument is worse
            # than one that runs with defaults.
            raw = provider.generate(prompt)
        except Exception as exc:
            return ActionPlan("none", reason=f"the planner could not be reached: {exc}")

        try:
            payload = lenient_json_loads(str(raw))
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            return ActionPlan("none", reason="the planner did not return a usable action")

        action = str(payload.get("action") or payload.get("name") or "none").strip()
        if action not in SUPPORTED_ACTIONS:
            return ActionPlan("none", reason=f"{action!r} is not an action I can execute")
        # "name" is dropped only when the model used it *as* the action name.
        # Reserving it unconditionally silently swallowed the project name, so
        # project.create arrived with nothing to create.
        reserved = {"action", "reason"} | ({"name"} if not payload.get("action") else set())
        arguments = {key: value for key, value in payload.items() if key not in reserved}
        inner = payload.get("arguments")
        if isinstance(inner, dict):
            arguments.update(inner)
        return ActionPlan(action, arguments=arguments, reason=str(payload.get("reason", "")))

    # -- execution -------------------------------------------------------

    def execute(self, plan: ActionPlan, *, request: str = "") -> Receipt:
        """Perform the plan.  Every path out of here returns a receipt."""

        started = time.perf_counter()
        handlers = {
            "file.write": self._write_file,
            "file.read": self._read_file,
            "project.create": self._create_project,
        }
        handler = handlers.get(plan.action)
        if handler is None:
            receipt = failed(
                plan.action or "none",
                "service.actions",
                plan.reason or "I cannot execute that. Nothing was attempted.",
                request=request,
            )
        else:
            try:
                receipt = handler(plan, request)
            except Exception as exc:
                # An executor bug must surface as a failed action, not as a
                # silent success and not as a 500 that loses the attempt.
                receipt = failed(
                    plan.action, "service.actions", f"{type(exc).__name__}: {exc}", request=request
                )
        return Receipt(
            kind=receipt.kind,
            executor=receipt.executor,
            ok=receipt.ok,
            request=request or receipt.request,
            detail=receipt.detail,
            evidence=receipt.evidence,
            verifications=receipt.verifications,
            id=receipt.id,
            at=receipt.at,
            duration_seconds=time.perf_counter() - started,
        )

    # -- file actions ----------------------------------------------------

    def _resolve(self, relative: str) -> Path:
        """A path inside the workspace, or an error.

        Containment is checked after resolution rather than by inspecting the
        string, because ``a/../../b`` and a symlink both look harmless as text.
        """

        root = self.workspace.resolve()
        candidate = (root / str(relative).strip().lstrip("/\\")).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"{relative!r} is outside the action workspace") from None
        return candidate

    def _write_file(self, plan: ActionPlan, request: str) -> Receipt:
        from tools.registry import ToolCall, ToolContext

        relative = str(plan.arguments.get("path") or plan.arguments.get("file") or "").strip()
        if not relative:
            return failed("file.write", "tools.write_file", "no filename was given", request=request)
        content = plan.arguments.get("content")
        if content is None:
            return failed(
                "file.write", "tools.write_file", f"no content was given for {relative}", request=request
            )
        content = str(content)

        try:
            target = self._resolve(relative)
        except ValueError as exc:
            return failed("file.write", "tools.write_file", str(exc), request=request, path=relative)

        context = ToolContext(workspace=self.workspace.resolve())
        result = self.tools.invoke(ToolCall("write_file", {"path": relative, "content": content}), context)

        if not result.ok:
            # The executor's own failure, reported as it came back. This is the
            # branch acceptance D exercises, and it must never be reachable
            # from a success message.
            return failed(
                "file.write",
                "tools.write_file",
                f"the file tool refused: {result.error}",
                request=request,
                path=str(target),
                error_kind=result.error_kind,
            )

        # -- independent readback ----------------------------------------
        # Deliberately not through the tool that just wrote it. A writer that
        # reports success and a reader that reads the writer's own buffer would
        # agree with each other about a file that is not there.
        verifications: list[Verification] = []
        exists = target.is_file()
        verifications.append(
            Verification(
                check="file exists on disk",
                passed=exists,
                observed=f"{target} ({target.stat().st_size} bytes)" if exists else f"{target} is not a file",
                expected=str(target),
            )
        )

        if exists:
            try:
                observed = target.read_text(encoding="utf-8", errors="replace")[:MAX_VERIFY_BYTES]
            except OSError as exc:
                observed = ""
                verifications.append(
                    Verification("file is readable", False, observed=str(exc), expected="readable")
                )
            else:
                # Compared after newline normalisation only: Windows may store
                # CRLF for content written with LF, and calling that a content
                # mismatch would report a failure the user cannot act on.
                matches = observed.replace("\r\n", "\n") == content.replace("\r\n", "\n")
                verifications.append(
                    Verification(
                        check="content matches exactly",
                        passed=matches,
                        observed=repr(observed[:200]),
                        expected=repr(content[:200]),
                    )
                )

        ok = all(item.passed for item in verifications)
        return Receipt(
            kind="file.write",
            executor="tools.write_file",
            ok=ok,
            request=request,
            detail=(
                f"wrote {len(content)} characters to {target}"
                if ok
                else f"the write reported success but verification failed for {target}"
            ),
            evidence={
                "path": str(target),
                "relative_path": relative,
                "characters": len(content),
                "workspace": str(self.workspace),
            },
            verifications=tuple(verifications),
        )

    def _read_file(self, plan: ActionPlan, request: str) -> Receipt:
        relative = str(plan.arguments.get("path") or "").strip()
        if not relative:
            return failed("file.read", "service.actions", "no filename was given", request=request)
        try:
            target = self._resolve(relative)
        except ValueError as exc:
            return failed("file.read", "service.actions", str(exc), request=request)
        if not target.is_file():
            return failed(
                "file.read", "service.actions", f"{target} does not exist", request=request, path=str(target)
            )
        content = target.read_text(encoding="utf-8", errors="replace")[:MAX_VERIFY_BYTES]
        return Receipt(
            kind="file.read",
            executor="service.actions",
            ok=True,
            request=request,
            detail=f"read {len(content)} characters from {target}",
            evidence={"path": str(target), "content": content[:4000]},
            verifications=(
                Verification(
                    check="file exists on disk",
                    passed=True,
                    observed=f"{target} ({target.stat().st_size} bytes)",
                    expected=str(target),
                ),
            ),
        )

    # -- project actions -------------------------------------------------

    def _create_project(self, plan: ActionPlan, request: str) -> Receipt:
        name = str(
            plan.arguments.get("name")
            or plan.arguments.get("title")
            or plan.arguments.get("goal")
            or ""
        ).strip()
        if not name:
            return failed("project.create", "projects.store", "no project name was given", request=request)

        goal = str(plan.arguments.get("goal") or name).strip()
        project = self.kernel.start_project(goal, title=name)

        # -- independent reload ------------------------------------------
        # From a *new* store over the same directory, so what is checked is
        # what is on disk rather than the object still held in memory. An
        # in-memory check would have passed for a project that was never
        # written, which is the whole failure being fixed.
        from projects.store import ProjectStore

        store = ProjectStore(Path(self.kernel.state_root) / "projects")
        reloaded = store.try_load(project.id)
        path = store.path_for(project.id)

        verifications = [
            Verification(
                check="project file written to disk",
                passed=path.is_file(),
                observed=f"{path} ({path.stat().st_size} bytes)" if path.is_file() else f"{path} missing",
                expected=str(path),
            ),
            Verification(
                check="reloaded from a fresh store",
                passed=reloaded is not None and reloaded.id == project.id,
                observed=(
                    f"id={reloaded.id} title={reloaded.title!r} goal={reloaded.goal!r}"
                    if reloaded is not None
                    else "the store returned nothing"
                ),
                expected=f"id={project.id}",
            ),
            Verification(
                check="visible through the Projects API the UI uses",
                passed=any(item.id == project.id for item in store.list_projects()),
                observed=f"{len(store.list_projects())} project(s) listed",
                expected=f"a listing containing {project.id}",
            ),
        ]

        ok = all(item.passed for item in verifications)
        return Receipt(
            kind="project.create",
            executor="projects.store",
            ok=ok,
            request=request,
            detail=(
                f"created project {name!r} as {project.id}"
                if ok
                else f"project {name!r} was not persisted correctly"
            ),
            evidence={
                "project_id": project.id,
                "title": project.title,
                "goal": project.goal,
                "path": str(path),
                "workspace": str(project.workspace),
            },
            verifications=tuple(verifications),
        )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def compose(receipt: Receipt, *, language: str = "") -> str:
    """The sentence the user reads, built from the receipt and nothing else.

    Written by code rather than generated, because this is the one sentence in
    the whole exchange that asserts something about the world. If a model wrote
    it, the model would be able to assert a success -- and the observed failure
    is precisely a model asserting a success. Deterministic text is not a
    stylistic preference here; it is the mechanism.
    """

    german = language.startswith("de")
    lines: list[str] = []

    if receipt.verified:
        lines.append(_headline_ok(receipt, german))
    elif receipt.ok:
        lines.append(
            "Ausgefuehrt, aber nicht verifiziert -- ich behandle das nicht als Erfolg."
            if german
            else "Executed, but not verified -- I am not treating that as success."
        )
    else:
        lines.append(
            f"Fehlgeschlagen. {receipt.detail}" if german else f"That failed. {receipt.detail}"
        )

    if receipt.verifications:
        lines.append("")
        lines.append("Belege:" if german else "Evidence:")
        lines.extend(f"  - {line}" for line in receipt.evidence_lines())

    lines.append("")
    lines.append(f"receipt {receipt.id}")
    return "\n".join(lines)


def _headline_ok(receipt: Receipt, german: bool) -> str:
    evidence = receipt.evidence
    if receipt.kind == "file.write":
        path = evidence.get("path", "")
        return (
            f"Datei geschrieben und anschliessend unabhaengig zurueckgelesen: {path}"
            if german
            else f"File written, then independently read back: {path}"
        )
    if receipt.kind == "file.read":
        content = str(evidence.get("content", ""))
        head = "Inhalt von" if german else "Contents of"
        return f"{head} {evidence.get('path', '')}:\n\n{content}"
    if receipt.kind == "project.create":
        return (
            f"Projekt {evidence.get('title', '')!r} angelegt und aus dem Speicher neu geladen "
            f"(id {evidence.get('project_id', '')})."
            if german
            else f"Project {evidence.get('title', '')!r} created and reloaded from the store "
            f"(id {evidence.get('project_id', '')})."
        )
    return receipt.detail
