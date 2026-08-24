"""Expert provider backed by the Codex CLI on a ChatGPT subscription.

**Status: written, not verified.** The Codex CLI is not installed on this
machine, so unlike :mod:`experts.claude_code` -- which was built by inspecting
``claude 2.1.241`` and proved on a real job -- this adapter has never actually
run. :meth:`CodexExpert.availability` says so rather than reporting a cheerful
"ready" for something nobody has tested, and the gateway will simply not select
it until the CLI exists.

What *is* load-bearing here regardless of whether the tool is present is the
cost safety, and it is the same shape as the Claude adapter for the same
reason. The Codex CLI authenticates either against a ChatGPT subscription (flat
fee, no marginal cost) or against ``OPENAI_API_KEY`` (metered per token). Which
one it picks is decided by the environment, so the environment is scrubbed:
``OPENAI_API_KEY`` and its relatives are removed from the child process, which
means subscription auth is the only possibility rather than merely the
intention.

Removing a variable is a far stronger guarantee than passing a flag that asks
politely, and it is the same lesson ``--bare`` taught on the Claude side.

When the CLI is installed, the two things to verify before trusting this are:
the non-interactive subcommand and its output format (assumed ``codex exec``
with ``--json``), and that a subscription-authenticated session really is what
runs. Until then it is scaffolding with the safety already in place.
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

#: Variables that would route this CLI onto metered billing.  Removed from the
#: child environment so a subscription session is the only thing that can run.
_METERED_CREDENTIALS = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
)

#: Phrases that mean the subscription allowance is spent.
_QUOTA_MARKERS = (
    "usage limit",
    "rate limit",
    "quota exceeded",
    "out of credit",
    "insufficient_quota",
    "upgrade your plan",
    "limit will reset",
)


class CodexExpert:
    """Runs an :class:`~experts.contracts.ExpertJob` through the Codex CLI."""

    name = "codex"
    channel = SpendChannel.SUBSCRIPTION_CLI

    #: Set once the CLI has actually been driven successfully on this machine.
    #: Until then the adapter reports itself unverified rather than ready.
    verified_on_this_machine = False

    def __init__(
        self,
        *,
        executable: str | None = None,
        model: str = "",
        subcommand: str = "exec",
        full_auto: bool = True,
    ) -> None:
        # None means detect; an explicit "" means absent. The Claude adapter
        # had a bug here where `or` silently fell through to PATH.
        self.executable = (shutil.which("codex") or "") if executable is None else executable
        self.model = model
        self.subcommand = subcommand
        self.full_auto = full_auto

    # -- availability ----------------------------------------------------

    def availability(self) -> ProviderAvailability:
        if not self.executable:
            return ProviderAvailability(
                False,
                "the codex CLI is not installed. Install it and sign in with your ChatGPT "
                "subscription (not an API key) to use this provider.",
            )
        try:
            completed = subprocess.run(
                [self.executable, "--version"],
                capture_output=True, text=True, timeout=30,
                env=self._environment(), encoding="utf-8", errors="replace",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ProviderAvailability(False, f"could not run the codex CLI: {exc}")

        if completed.returncode != 0:
            return ProviderAvailability(False, (completed.stderr or completed.stdout).strip()[:300])

        version = completed.stdout.strip()[:80]
        if not self.verified_on_this_machine:
            # Present but untested. Honest rather than optimistic: this adapter
            # was written against documented behaviour, not observed behaviour.
            return ProviderAvailability(
                False,
                f"codex {version} is installed but this adapter has never been verified against it. "
                "Set CodexExpert.verified_on_this_machine = True once a real job has run.",
                version=version,
            )
        return ProviderAvailability(True, "subscription CLI available", version=version)

    # -- execution -------------------------------------------------------

    def execute(self, job: ExpertJob) -> ExpertResult:
        if not self.executable:
            return ExpertResult(
                status=ExpertStatus.NOT_CONFIGURED,
                provider=self.name,
                blocker="the codex CLI is not installed",
            )

        workspace = Path(job.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        before = _snapshot(workspace)

        command = [self.executable, self.subcommand]
        if self.full_auto:
            # Non-interactive: there is nobody at the terminal to approve
            # anything, and a prompt would simply hang until the budget expires.
            command.append("--full-auto")
        if self.model:
            command += ["--model", self.model]
        command.append(self._prompt(job))

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True, text=True,
                timeout=max(30.0, job.max_seconds),
                env=self._environment(), encoding="utf-8", errors="replace",
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
        text = completed.stdout or ""
        combined = f"{text}\n{completed.stderr}".lower()

        quota = QuotaState(
            exhausted=any(marker in combined for marker in _QUOTA_MARKERS),
            detail=(completed.stderr or text)[-400:].strip(),
        )
        if quota.exhausted:
            return ExpertResult(
                status=ExpertStatus.UNAVAILABLE,
                provider=self.name,
                summary=text[:4000],
                blocker="subscription quota exhausted",
                quota=quota,
                duration_seconds=duration,
                files_changed=_changed(before, _snapshot(workspace)),
            )

        failed = completed.returncode != 0
        return ExpertResult(
            status=ExpertStatus.FAILED if failed else ExpertStatus.COMPLETED,
            provider=self.name,
            summary=text[:4000],
            files_changed=_changed(before, _snapshot(workspace)),
            commands_run=[" ".join(command[:2]) + " ..."],
            blocker=(completed.stderr or "").strip()[-400:] if failed else "",
            quota=quota,
            duration_seconds=duration,
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
            "Work directly in the files. Do not ask questions -- everything you need is below, "
            "and there is nobody to answer. When finished, summarise what you changed.\n\n"
            "Your work will be verified independently: Jarvis re-runs the acceptance commands "
            "after you exit and decides from their exit codes.\n\n"
            f"{job.brief()}"
        )


def _snapshot(root: Path) -> dict[str, float]:
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
