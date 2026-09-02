"""Expert provider backed by the Claude Code CLI on the user's subscription.

Built against the tool actually installed on this machine (``claude 2.1.241``)
rather than against assumed flags, as the brief requires.  Three findings from
that inspection shaped the adapter:

``-p/--print`` with ``--output-format json`` returns a single JSON object whose
``result`` field holds the final text, alongside ``is_error``, ``subtype``,
``num_turns``, ``permission_denials``, ``total_cost_usd`` and ``usage``.  That
is enough to drive a provider without scraping human-readable output.

``--bare`` must NOT be used.  Its own help text says Anthropic auth becomes
"strictly ANTHROPIC_API_KEY or ..." -- which is the metered channel.  The whole
value of this provider is that it runs on a subscription the user already pays
for, so a flag that quietly switches billing models is exactly the accident
:mod:`runtime.cost_policy` exists to prevent.

Belt and braces on the same point: the subprocess environment is *scrubbed* of
metered credentials before launch.  If ``ANTHROPIC_API_KEY`` is set on this
machine for some unrelated reason, this provider will not inherit it, so it
cannot silently bill per token even if a future CLI version would prefer a key
over the subscription session.  Removing a variable is a much stronger guarantee
than trusting a flag.

``total_cost_usd`` is reported even on subscription runs.  It is the notional
value of the work, covered by the flat fee -- recorded as a usage signal, never
surfaced as money owed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from experts.contracts import ExpertJob, ExpertResult, ExpertStatus, QuotaState
from experts.gateway import ProviderAvailability
from runtime.cost_policy import SpendChannel

#: Environment variables that would route this CLI onto metered billing.
#: Stripped from the child environment so subscription auth is the only
#: possibility rather than merely the intention.
_METERED_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_BEARER_TOKEN_BEDROCK",
    "OPENAI_API_KEY",
)

#: Phrases the CLI uses when a subscription allowance is spent.  Matched
#: case-insensitively against stderr and the result text.
_QUOTA_MARKERS = (
    "usage limit reached",
    "rate limit",
    "quota exceeded",
    "out of credit",
    "insufficient credit",
    "upgrade to continue",
    "limit will reset",
    "session limit",
    "too many requests",
    "429",
    "hit your limit",
    "reached your limit",
)


#: Tool patterns the expert may not use.  ``cd`` because the worktree is its
#: whole world; the git verbs because a worktree's ``.git`` is a pointer into
#: the live repository's metadata and these would rewrite it.
CONFINEMENT_DISALLOWED_TOOLS = (
    "Bash(cd:*)", "Bash(cd *)", "Bash(pushd:*)", "Bash(Set-Location:*)",
    "Bash(git worktree:*)", "Bash(git stash:*)", "Bash(git reset:*)", "Bash(git checkout:*)",
    "Bash(git switch:*)", "Bash(git branch:*)", "Bash(git push:*)", "Bash(git gc:*)", "Bash(git config:*)",
    "Bash(git commit:*)", "Bash(git rebase:*)", "Bash(git merge:*)", "Bash(git clean:*)",
    "Bash(powershell:*)", "Bash(pwsh:*)", "Bash(cmd:*)",
)


class ClaudeCodeExpert:
    """Runs an :class:`~experts.contracts.ExpertJob` through ``claude -p``."""

    name = "claude_code"
    channel = SpendChannel.SUBSCRIPTION_CLI

    def __init__(
        self,
        *,
        executable: str | None = None,
        model: str = "",
        permission_mode: str = "acceptEdits",
    ) -> None:
        # `None` means "find it"; an explicit "" means "there isn't one", which
        # `or` would have silently overridden with whatever is on PATH.
        self.executable = (shutil.which("claude") or "") if executable is None else executable
        self.model = model
        self.permission_mode = permission_mode

    # -- availability ----------------------------------------------------

    def availability(self) -> ProviderAvailability:
        if not self.executable:
            return ProviderAvailability(
                False,
                "the claude CLI is not installed or not on PATH; install it and sign in "
                "with your existing subscription",
            )
        try:
            completed = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._environment(),
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ProviderAvailability(False, f"could not run the claude CLI: {exc}")

        if completed.returncode != 0:
            text = (completed.stderr or completed.stdout).strip()[:300]
            lowered = text.lower()
            state = "NOT_AUTHENTICATED" if any(m in lowered for m in ("log in", "login", "sign in", "not authenticated", "unauthorized")) else "ERROR"
            return ProviderAvailability(False, text, state=state)

        return ProviderAvailability(True, "subscription CLI available (installed, version answered; quota unknown until a call)",
                                    version=completed.stdout.strip()[:80], state="AVAILABLE")

    # -- execution -------------------------------------------------------

    def execute(self, job: ExpertJob) -> ExpertResult:
        if not self.executable:
            return ExpertResult(
                status=ExpertStatus.NOT_CONFIGURED,
                provider=self.name,
                blocker="the claude CLI is not installed",
            )

        workspace = Path(job.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        before = _snapshot(workspace)

        command = [
            self.executable,
            "-p",
            self._prompt(job),
            "--output-format",
            "json",
            "--permission-mode",
            self.permission_mode,
            "--add-dir",
            str(workspace),
            # Edits outside the workspace already need a permission nobody is
            # there to grant.  The shell is the remaining way out, so the
            # commands that leave a directory or rewrite shared git state are
            # withheld -- from the tool menu, not from a sentence in the prompt.
            "--disallowedTools",
            *CONFINEMENT_DISALLOWED_TOOLS,
        ]
        if self.model:
            command += ["--model", self.model]

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=max(30.0, job.max_seconds),
                env=self._environment(),
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            return ExpertResult(
                status=ExpertStatus.TIMEOUT,
                provider=self.name,
                blocker=f"the expert exceeded its {job.max_seconds:.0f}s budget",
                duration_seconds=time.perf_counter() - started,
                files_changed=_changed(before, _snapshot(workspace)),
            )
        except OSError as exc:
            return ExpertResult(status=ExpertStatus.BLOCKED, provider=self.name, blocker=str(exc))

        duration = time.perf_counter() - started
        payload = _parse(completed.stdout)
        text = str(payload.get("result", "") or completed.stdout)
        combined = f"{text}\n{completed.stderr}".lower()

        quota = QuotaState(
            exhausted=any(marker in combined for marker in _QUOTA_MARKERS),
            detail=(completed.stderr or text)[-400:].strip(),
            notional_cost_usd=_as_float(payload.get("total_cost_usd")),
        )

        if quota.exhausted:
            # A state, not a failure to route around. The gateway will consult
            # the cost policy for fallbacks, and the policy never names one that
            # costs money.
            return ExpertResult(
                status=ExpertStatus.UNAVAILABLE,
                provider=self.name,
                summary=text[:4000],
                blocker="subscription quota exhausted",
                quota=quota,
                duration_seconds=duration,
                raw=payload,
                files_changed=_changed(before, _snapshot(workspace)),
            )

        failed = bool(payload.get("is_error")) or completed.returncode != 0
        status = ExpertStatus.FAILED if failed else ExpertStatus.COMPLETED

        return ExpertResult(
            status=status,
            provider=self.name,
            summary=text[:4000],
            files_changed=_changed(before, _snapshot(workspace)),
            commands_run=[" ".join(command[:2]) + " ..."],
            blocker=(completed.stderr or "").strip()[-400:] if failed else "",
            quota=quota,
            duration_seconds=duration,
            raw={
                key: payload.get(key)
                for key in ("subtype", "num_turns", "session_id", "stop_reason", "permission_denials", "usage")
                if key in payload
            },
        )

    # -- internals -------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        """The parent environment minus anything that implies metered billing."""

        env = dict(os.environ)
        for name in _METERED_CREDENTIALS:
            env.pop(name, None)
        return env

    def _prompt(self, job: ExpertJob) -> str:
        return (
            "You are being asked by Jarvis, an autonomous development system, to complete one "
            "well-specified job in the current working directory.\n\n"
            "Work directly in the files. Do not ask questions -- everything you need is below, and "
            "there is nobody to answer. When you are finished, reply with a short summary of what "
            "you changed and why.\n\n"
            "Your work will be verified independently: Jarvis re-runs the acceptance commands after "
            "you exit and decides from their exit codes, so claiming success without running them "
            "achieves nothing.\n\n"
            f"{job.brief()}"
        )


def _parse(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except ValueError:
        # Some CLI versions print progress lines before the JSON object.
        start = text.find("{")
        if start < 0:
            return {}
        try:
            payload = json.loads(text[start:])
        except ValueError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot(root: Path) -> dict[str, float]:
    """Path -> mtime, so a diff can name what the expert touched."""

    found: dict[str, float] = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if path.is_file() and not any(part in {".git", "__pycache__", "node_modules"} for part in path.parts):
            try:
                found[str(path.relative_to(root)).replace("\\", "/")] = path.stat().st_mtime
            except OSError:
                continue
    return found


def _changed(before: dict[str, float], after: dict[str, float]) -> list[str]:
    changed = [name for name, mtime in after.items() if before.get(name) != mtime]
    changed += [name for name in before if name not in after]
    return sorted(set(changed))
