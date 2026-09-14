"""An engineer behind a metered API, driven through the model gateway.

Codex is a subscription CLI that edits a worktree by itself.  The engineer
roles the gateway configures (``engineer.standard``, ``engineer.frontier``)
are models behind an API: they cannot touch files.  This provider gives them
hands -- the smallest honest pair:

1. The brief, plus the relevant files read from the candidate worktree, go to
   the pinned role in BUILD mode.  Every gateway rule applies: the owner's
   spending policy, the budget reservation, the privacy router, the FREE wall.
2. The model answers with a unified diff.  ``git apply`` puts it into the
   worktree.  If it does not apply, the model gets the error once and answers
   again; a second failure is a FAILED result, never a guess.
3. What changed is read back from ``git status``; the expert gateway then
   re-runs the acceptance commands itself, as it does for every engineer.

The model's account of what it did is recorded and never believed: the
verification decides.  Secrets never appear in the files sent (the privacy
router withholds them) and never in the brief.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from experts.contracts import ExpertJob, ExpertResult, ExpertStatus, QuotaState
from experts.gateway import ProviderAvailability
from runtime.cost_policy import SpendChannel

#: Roughly 60k tokens of source, which is what the brief calls for: tens of
#: thousands of context tokens, not hundreds of thousands.
FILE_BUDGET_CHARS = 200_000
PER_FILE_CHARS = 60_000

DIFF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "one paragraph: what was changed and why"},
        "diff": {"type": "string", "description": "a unified diff (git format, a/ and b/ prefixes) applying at the workspace root"},
        "files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "diff"],
}


def _git(cwd: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


class ApiEngineerExpert:
    """One configured engineer role, as the expert gateway sees it."""

    channel = SpendChannel.PAID_API
    #: Never reached by iteration: the engineering router names this expert or it does not run.
    explicit_only = True

    def __init__(self, gateway: Any, role: str, *, mode: Any = None) -> None:
        self.gateway = gateway
        self.role = role
        self.name = role
        from gateway.modes import ChatMode

        self.mode = mode or ChatMode.BUILD
        self.last: dict[str, Any] = {}

    # -- availability -----------------------------------------------------------

    def availability(self) -> ProviderAvailability:
        from gateway.health import ProviderStatus

        config = self.gateway.config
        binding = config.binding(self.role)
        provider = config.provider_for(self.role)
        if binding is None or provider is None or not binding.enabled or not provider.enabled:
            return ProviderAvailability(available=False, detail=f"{self.role} is not enabled in config/providers.json",
                                        state="NOT_CONFIGURED")
        if provider.secret and not self.gateway.credentials.has(provider.secret):
            return ProviderAvailability(available=False, detail=f"no API key stored for {provider.name}", state="NOT_AUTHENTICATED")
        if not bool(self.gateway.cost_policy.allow_paid_api):
            return ProviderAvailability(available=False, detail="paid API billing is disabled by the owner spending policy",
                                        state="NOT_CONFIGURED")
        health = self.gateway.health.status(provider.name)
        if health is ProviderStatus.QUOTA_EXHAUSTED:
            return ProviderAvailability(available=False, detail=f"{provider.name}: quota exhausted", quota_exhausted=True,
                                        state="QUOTA_EXHAUSTED")
        if health is ProviderStatus.RATE_LIMIT:
            return ProviderAvailability(available=False, detail=f"{provider.name}: rate limited", state="RATE_LIMITED")
        if health is ProviderStatus.AUTHENTICATION_ERROR:
            return ProviderAvailability(available=False, detail=f"{provider.name}: the API key was rejected", state="NOT_AUTHENTICATED")
        if health.is_outage:
            return ProviderAvailability(available=False, detail=f"{provider.name}: {health.value}", state="ERROR")
        return ProviderAvailability(available=True, detail=f"{self.role} -> {provider.name}/{binding.model}", state="AVAILABLE")

    # -- the work --------------------------------------------------------------------

    def _files(self, job: ExpertJob) -> list[str]:
        listed = [str(f) for f in (job.metadata.get("files") or []) if str(f).strip()]
        listed += [str(f) for f in (job.metadata.get("tests") or []) if str(f).strip()]
        seen: list[str] = []
        for item in listed:
            rel = item.replace("\\", "/")
            if rel not in seen:
                seen.append(rel)
        return seen

    def _read_files(self, job: ExpertJob) -> tuple[str, list[str]]:
        root = Path(job.workspace)
        parts: list[str] = []
        included: list[str] = []
        budget = FILE_BUDGET_CHARS
        for rel in self._files(job):
            path = root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > PER_FILE_CHARS:
                text = text[:PER_FILE_CHARS] + f"\n... [{len(text) - PER_FILE_CHARS} more characters not shown]\n"
            if len(text) > budget:
                break
            budget -= len(text)
            parts.append(f"===== {rel} =====\n{text}")
            included.append(rel)
        return "\n\n".join(parts), included

    def _catalog_context(self, job: ExpertJob) -> str:
        """The narrow engineering context from the catalog: contracts, dependents, tests."""

        try:
            from catalog.context import build_context

            files = [f for f in self._files(job) if f.endswith(".py") and not f.startswith("tests/")]
            if not files:
                return ""
            return build_context(files, request=job.goal[:300], repo=Path(job.workspace), budget_chars=30_000).text
        except Exception:  # noqa: BLE001 - the catalog is help, not a requirement
            return ""

    def _prompt(self, job: ExpertJob, files_text: str, included: list[str], *, previous_error: str = "") -> str:
        catalog_text = self._catalog_context(job)
        lines = [
            job.brief(),
            *([catalog_text] if catalog_text else []),
            "WORKSPACE: paths below are relative to the workspace root you are editing. "
            "Answer with ONE JSON object {\"summary\", \"diff\", \"files\"}.",
            "The diff MUST be a unified diff in git format (`diff --git a/<path> b/<path>`, `---`/`+++` lines, hunks with "
            "correct line counts) that applies cleanly with `git apply` at the workspace root. Include full context lines. "
            "Create new files with `--- /dev/null`. Do not include any file you were not shown unless you create it.",
        ]
        if included:
            lines.append("FILES (current content):\n" + files_text)
        if previous_error:
            lines.append("YOUR PREVIOUS DIFF DID NOT APPLY. git said:\n" + previous_error[:2000]
                         + "\nProduce a corrected diff against the files as shown above.")
        return "\n\n".join(lines)

    def _apply(self, job: ExpertJob, diff: str) -> tuple[bool, str]:
        workspace = Path(job.workspace)
        top_raw = _git(workspace, "rev-parse", "--show-toplevel")
        if top_raw.returncode != 0:
            # A capability workspace is a plain directory: git apply works there
            # too, as a patch tool, with paths relative to the workspace.
            top, relative = workspace, Path(".")
        else:
            top = Path(top_raw.stdout.strip())
            try:
                relative = workspace.resolve().relative_to(top.resolve())
            except ValueError:
                relative = Path(".")
        if not diff.strip():
            return False, "the model returned an empty diff"
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8", newline="\n") as handle:
            handle.write(diff if diff.endswith("\n") else diff + "\n")
            patch_path = handle.name
        try:
            args = ["apply", "--whitespace=nowarn"]
            if str(relative) not in {".", ""}:
                args.append(f"--directory={relative.as_posix()}")
            check = _git(top, *args, "--check", patch_path)
            if check.returncode != 0:
                return False, check.stderr.strip() or check.stdout.strip() or "git apply --check failed"
            applied = _git(top, *args, patch_path)
            if applied.returncode != 0:
                return False, applied.stderr.strip() or "git apply failed"
            return True, "applied"
        finally:
            try:
                Path(patch_path).unlink()
            except OSError:
                pass

    def _snapshot(self, job: ExpertJob) -> dict[str, str]:
        import hashlib

        workspace = Path(job.workspace)
        out: dict[str, str] = {}
        for path in workspace.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
                try:
                    out[path.relative_to(workspace).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    continue
        return out

    def _changed(self, job: ExpertJob, before: dict[str, str] | None = None) -> list[str]:
        workspace = Path(job.workspace)
        status = _git(workspace, "status", "--porcelain", "--untracked-files=all")
        top_raw = _git(workspace, "rev-parse", "--show-toplevel")
        out: list[str] = []
        if status.returncode != 0:
            after = self._snapshot(job)
            return sorted(rel for rel, digest in after.items() if (before or {}).get(rel) != digest)
        top = Path(top_raw.stdout.strip()) if top_raw.returncode == 0 else workspace
        for line in status.stdout.splitlines():
            if len(line) < 4:
                continue
            rel = line[3:].strip().strip('"').replace("\\", "/")
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            full = (top / rel).resolve()
            try:
                out.append(full.relative_to(workspace.resolve()).as_posix())
            except ValueError:
                out.append(rel)
        return out

    def execute(self, job: ExpertJob) -> ExpertResult:
        from gateway.gateway import GatewayError, GatewayRefused, GatewayRequest
        from gateway.task import TaskFacts

        started = time.perf_counter()
        files_text, included = self._read_files(job)
        facts = TaskFacts(text=job.goal, is_engineering=True, estimated_files_changed=max(1, len(included)),
                          subsystems=int(job.metadata.get("subsystems", 0) or 0),
                          new_subsystem=bool(job.metadata.get("new_subsystem", False)))
        commands: list[str] = []
        previous_error = ""
        before = self._snapshot(job)
        total_cost = 0.0
        usage_total: dict[str, int] = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        summary = ""
        for attempt in range(2):
            prompt = self._prompt(job, files_text, included, previous_error=previous_error)
            request = GatewayRequest(prompt=prompt, mode=self.mode, facts=facts, schema=DIFF_SCHEMA, purpose="engineer",
                                     role=self.role, task_id=str(job.metadata.get("task_id") or job.metadata.get("mission_id") or ""))
            try:
                reply = self.gateway.complete(request)
            except GatewayRefused as exc:
                return ExpertResult(status=ExpertStatus.REFUSED, provider=self.name, blocker=str(exc),
                                    raw={"decision": exc.decision.to_dict()}, duration_seconds=time.perf_counter() - started)
            except GatewayError as exc:
                quota = QuotaState(exhausted=exc.status.value == "quota_exhausted", detail=str(exc))
                return ExpertResult(status=ExpertStatus.UNAVAILABLE if exc.status.is_outage else ExpertStatus.FAILED,
                                    provider=self.name, blocker=str(exc), quota=quota, duration_seconds=time.perf_counter() - started)
            total_cost += reply.actual_eur
            for key in usage_total:
                usage_total[key] += int(reply.usage.get(key, 0) or 0)
            data = _parse(reply.text)
            summary = str(data.get("summary", ""))[:800]
            diff = str(data.get("diff", ""))
            ok, detail = self._apply(job, diff)
            commands.append(f"git apply ({'ok' if ok else 'failed'})")
            if ok:
                changed = self._changed(job, before)
                self.last = {"cost_eur": round(total_cost, 6), "usage": usage_total, "attempts": attempt + 1, "files": changed}
                return ExpertResult(status=ExpertStatus.COMPLETED, provider=self.name, summary=summary, files_changed=changed,
                                    commands_run=commands, duration_seconds=time.perf_counter() - started,
                                    raw={"cost_eur": round(total_cost, 6), "usage": usage_total, "attempts": attempt + 1,
                                         "role": reply.role, "model": reply.model, "files_shown": included})
            previous_error = detail
        return ExpertResult(status=ExpertStatus.FAILED, provider=self.name, summary=summary, commands_run=commands,
                            blocker=f"the diff did not apply after two attempts: {previous_error[:300]}",
                            duration_seconds=time.perf_counter() - started,
                            raw={"cost_eur": round(total_cost, 6), "usage": usage_total, "attempts": 2})


def _parse(text: str) -> dict[str, Any]:
    body = str(text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body
        body = body.rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except ValueError:
        try:
            from brain.json_utils import lenient_json_loads

            data = lenient_json_loads(body)
        except Exception:  # noqa: BLE001
            data = None
    if isinstance(data, dict):
        return data
    # A bare diff is still usable.
    if "diff --git" in body:
        return {"summary": "", "diff": body[body.index("diff --git"):]}
    return {"summary": body[:200], "diff": ""}
