"""Expert provider backed by the Codex CLI on a ChatGPT subscription.

The adapter drives the installed Codex CLI directly, with the flags that CLI
actually accepts -- see :data:`_AUTONOMOUS_FLAGS`, which replaced a set taken
from documentation that ``codex exec`` rejects outright.

:meth:`CodexExpert.availability` is deliberately cheap: it checks that the CLI
can start and report a version, and it remembers a refusal. It cannot do more
than that, because ``--version`` reads a binary on disk and knows nothing about
the account behind it: a spent allowance is only ever discovered by asking, so
:meth:`CodexExpert.note_quota_exhausted` records the answer when a real call
comes back refused and availability stops claiming READY until the reset time
the CLI itself named.

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

If the user is not signed in through the ChatGPT subscription client, Codex
reports an auth error instead of silently falling back to PAYG.

The same argument reaches past billing: the owner's own secrets are stripped
from the child environment too. The engineer is given a workspace and a brief,
which is everything it needs, so it is given nothing else.
"""

from __future__ import annotations

import json
import os
import re
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
    "OPENAI_PROJECT",
    "OPENAI_API_TYPE",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)

#: Anything whose *name* says it carries a secret.  The engineer is given a
#: workspace and a brief, and nothing else: it never needs the owner's
#: credentials, so it is never handed them.  Name-based rather than a fixed
#: list, because the list of secrets a machine happens to hold is not knowable
#: in advance and the cost of guessing wrong is one-directional.
_SECRET_NAME = re.compile(
    r"(secret|token|password|passwd|credential|api[_-]?key|access[_-]?key|private[_-]?key|"
    r"session|cookie|auth)", re.I,
)

#: Names that match the pattern but carry no secret and are load-bearing for a
#: normal child process on Windows.
_SECRET_EXCEPTIONS = frozenset({"SESSIONNAME", "AUTHORITY"})

#: How to run the CLI with nobody at the terminal, verified against
#: ``codex-cli 0.153.4`` on this machine.
#:
#: The previous value was ``--full-auto``, taken from documentation rather than
#: from the tool: ``codex exec`` rejects it outright ("unexpected argument
#: '--full-auto' found") and exits in under four seconds with an empty summary,
#: which reads in the acquisition log exactly like an expert that tried and
#: failed. It had never run, which is what the adapter used to say about itself
#: before that admission was deleted.
#:
#: ``--sandbox workspace-write`` is what actually grants write access to the
#: capability workspace, and ``exec`` never prompts for approval anyway.
#: ``--skip-git-repo-check`` is required because a capability workspace is a
#: plain directory: without it the CLI refuses to start at all.
_AUTONOMOUS_FLAGS = ("--sandbox", "workspace-write", "--skip-git-repo-check")

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
        #: Kept as the caller-facing name for "run without asking anybody".
        #: What it emits is :data:`_AUTONOMOUS_FLAGS`, which is what the
        #: installed CLI accepts; the flag it used to emit does not exist.
        self.full_auto = full_auto
        #: Set by :meth:`note_quota_exhausted` when a real call comes back
        #: refused; until then the CLI is presumed usable.
        self._quota_until = 0.0
        self._quota_detail = ""

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
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ProviderAvailability(False, f"could not run the codex CLI: {exc}")

        if completed.returncode != 0:
            return ProviderAvailability(False, (completed.stderr or completed.stdout).strip()[:300])

        version = completed.stdout.strip()[:80]
        if self._quota_block_active():
            return ProviderAvailability(
                False,
                f"subscription usage limit reached; {self._quota_detail}",
                version=version,
            )
        return ProviderAvailability(True, "subscription CLI available", version=version)

    def note_quota_exhausted(self, detail: str, *, seconds: float = 3600.0) -> None:
        """Remember that the allowance is spent, and for roughly how long.

        ``codex --version`` answers from a binary on disk and knows nothing
        about the account behind it, so availability said READY while every
        real call came straight back with "You've hit your usage limit" --
        measured on this machine, 2026-09-08. A provider that reports itself
        available and then cannot do anything is worse than one that reports
        itself unavailable: the caller queues honestly in the second case and
        burns a workspace and a verification cycle in the first.
        """

        self._quota_until = time.time() + max(60.0, seconds)
        self._quota_detail = str(detail or "").strip()[:300]

    def _quota_block_active(self) -> bool:
        return time.time() < getattr(self, "_quota_until", 0.0)

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
            command += list(_AUTONOMOUS_FLAGS)
        command += ["-C", str(workspace)]
        if self.model:
            command += ["--model", self.model]
        command.append(self._prompt(job))

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True, text=True,
                # Without this the CLI inherits whatever stdin this process
                # has and announces "Reading additional input from stdin...",
                # which in a service with no terminal is a wait with no end.
                stdin=subprocess.DEVNULL,
                timeout=max(30.0, job.max_seconds),
                env=self._environment(), encoding="utf-8", errors="replace",
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
        text = completed.stdout or ""
        combined = f"{text}\n{completed.stderr}".lower()

        quota = QuotaState(
            exhausted=any(marker in combined for marker in _QUOTA_MARKERS),
            detail=(completed.stderr or text)[-400:].strip(),
        )
        if quota.exhausted:
            self.note_quota_exhausted(quota.detail, seconds=_reset_seconds(combined))
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
        """The parent environment minus metered billing and minus every secret.

        Removing a variable is a far stronger guarantee than a flag that asks
        politely -- the CLI cannot reach an API key it was never given -- and
        the same argument applies to the owner's own secrets. The owner
        password is never in the environment in any form (it is a scrypt
        verifier on disk, and the plaintext lives only inside the call frame
        that checks it), and this makes sure nothing else that names itself a
        credential travels into the engineer's process either.
        """

        env = {
            name: value
            for name, value in os.environ.items()
            if not (_SECRET_NAME.search(name) and name.upper() not in _SECRET_EXCEPTIONS)
        }
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


def _reset_seconds(text: str, *, default: float = 3600.0) -> float:
    """How long to treat the allowance as spent, from what the CLI said.

    The message carries a wall-clock reset time ("try again at 5:29 AM"). Using
    it means the block lifts when the allowance actually returns instead of on
    a fixed guess, and a message without one falls back to an hour.
    """

    import datetime as _dt

    match = re.search(r"try again at\s+(\d{1,2}):(\d{2})\s*(am|pm)?", text, re.I)
    if not match:
        return default
    hour, minute = int(match.group(1)), int(match.group(2))
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    now = _dt.datetime.now()
    target = now.replace(hour=min(23, hour), minute=min(59, minute), second=0, microsecond=0)
    if target <= now:
        target += _dt.timedelta(days=1)
    return max(60.0, min(24 * 3600.0, (target - now).total_seconds()))
