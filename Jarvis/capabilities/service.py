"""Capabilities: things Jarvis can actually do, acquired when it cannot.

A capability is not knowledge in a prompt.  It is an installed, executable,
tested unit with a stable identifier, and the difference matters: knowing *about*
playing music is not the same as being able to play a file when asked, twice, a
week apart, after a reboot.

The lifecycle is:

    ask -> is it installed?  -> yes: execute it
                             -> no: acquire it, verify it, register it, execute it

Acquisition runs on the ordinary :class:`~projects.engine.ProjectEngine` rather
than on a pipeline of its own.  That is deliberate.  Acquiring a capability is
investigating an environment, deciding an approach, writing code, testing it and
repairing it -- which is exactly what the project loop already does, including
all of its recovery behaviour.  A second implementation would be a second set of
bugs.

Two contracts make the difference between capabilities that can be tested and
capabilities that can only be hoped about:

``run(payload) -> dict``
    Every capability is a Python module exposing this one function.  Uniform,
    trivially callable, and serialisable across a future process boundary.

``payload["dry_run"]``
    A capability with an external effect -- playing audio, opening a window,
    sending something -- must support a dry run that performs every check and
    reports what it *would* do without doing it.  Without this, a side-effecting
    capability cannot be verified at all, and "verified" would come to mean
    "the code imported successfully".  With it, acquisition can prove the player
    was found, the file exists, and the command is well-formed, and be honest
    that the sound itself was not observed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capabilities.models import CapabilityManifest
from capabilities.registry import CapabilityRegistry
from knowledge.graph import KnowledgeGraph, NodeType
from knowledge.memory import ExperienceMemory
from projects.engine import ProjectEngine
from projects.models import Project, ResourceLimits, StopReason


@dataclass
class CapabilityOutcome:
    """The result of asking for a capability."""

    goal: str
    capability_id: str = ""
    #: "available" (already installed), "acquired", or "failed".
    status: str = "failed"
    acquired: bool = False
    manifest: CapabilityManifest | None = None
    project_id: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status in {"available", "acquired"} and bool(self.capability_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest"] = self.manifest.to_dict() if self.manifest else None
        data["usable"] = self.usable
        return data


@dataclass
class ExecutionOutcome:
    capability_id: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Marker carried by the seeded skeleton.  Verification refuses to register a
#: capability while it is still present.
#:
#: This exists because of a real failure: the skeleton was seeded to help the
#: model, and the skeleton passed its own verification. run() returned a dict,
#: the template test asserted only that "ok" was in the result, and the stub
#: satisfied both -- so a capability that did nothing at all was registered as
#: verified. A scaffold must never be able to certify itself.
NOT_IMPLEMENTED = "JARVIS_CAPABILITY_NOT_IMPLEMENTED"

#: The seeded skeleton, deliberately tiny.
#:
#: Giving the model a working skeleton to edit is far more reliable than asking
#: it to produce the shape from a description.
#:
#: An earlier version carried the full contract in its docstring, which made it
#: 32 lines. Replacing it with a 15-line implementation then looked like
#: file truncation to the edit engine's shrink guard, and the write was refused
#: -- one safety mechanism blocking another. The rules live in the project's
#: constraints, where the model reads them anyway; the skeleton only has to pin
#: the shape.
_TEMPLATE_MAIN = '''"""Capability implementation. Replace the body of run()."""

from __future__ import annotations

import shutil
from typing import Any

# Every payload key run() accepts. A caller cannot pass what is not declared.
INPUT_SCHEMA = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    # Discover the environment with the STANDARD LIBRARY, like this. Jarvis
    # tools are not importable here -- only stdlib and declared dependencies.
    player = shutil.which("some-program")
    if payload.get("dry_run"):
        return {"ok": bool(player), "would_use": player}
    return {"ok": False, "error": "JARVIS_CAPABILITY_NOT_IMPLEMENTED"}
'''

_TEMPLATE_TEST = '''"""Tests for this capability. Replace these with real ones."""

import main


def test_placeholder():
    raise AssertionError("JARVIS_CAPABILITY_NOT_IMPLEMENTED: write real tests")
'''


#: Where Jarvis itself lives, so a check running in a capability workspace can
#: import the static checker without depending on how it was launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CapabilityCheck:
    """One objective check, phrased for the loop and runnable by anything.

    The same object is used twice on purpose: the project loop takes ``text``
    and ``command`` as an acceptance criterion it can watch fail, and
    :meth:`CapabilityService._verify` re-runs ``command`` from a clean process
    once the loop has stopped.

    That keeps the two properties that matter and were previously in tension.
    The bar is *identical*, so the loop cannot be marked against a rubric it was
    never shown -- the failure mode that ended both live capability runs with
    ``contract=ok`` and ``implemented=FAILED``.  And verification stays
    *independent*, because it executes the checks itself rather than believing
    the loop's report of them.
    """

    name: str
    text: str
    command: tuple[str, ...]


def _audio_python() -> str:
    """The interpreter that can read the audio meter, or the current one."""

    for candidate in (
        _REPO_ROOT / ".venv-speech" / "Scripts" / "python.exe",
        _REPO_ROOT / ".venv-speech" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


#: Cached once per process. The cache itself is durable across runs; this only
#: avoids re-reading the file for every project created in one session.
_ENVIRONMENT: list[str] | None = None


def _environment_briefing() -> list[str]:
    """What is true about this machine, from the deterministic probe cache.

    Falls back to an empty list rather than to guesses: a briefing that is
    wrong about the environment is worse than one that says nothing, which is
    the lesson of four capability attempts that failed on packages the model
    had no way to know were absent.
    """

    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        try:
            from runtime.environment import EnvironmentCache

            cache = EnvironmentCache(_REPO_ROOT / "data" / "jarvis" / "environment.json")
            _ENVIRONMENT = cache.briefing()
        except Exception:
            _ENVIRONMENT = []
    return _ENVIRONMENT


def _importable_packages(limit: int = 24) -> str:
    """Third-party packages the capability's interpreter can actually import.

    Checked by importlib rather than listed by hand: a hand-written list drifts
    the moment anything is installed or removed, and a briefing that is wrong
    about the environment is worse than one that says nothing.

    What is returned is what was *probed*, which is not the same as what exists.
    The briefing used to follow this list with "anything else is NOT installed",
    and that sentence was false: mss was installed here and on no candidate
    list, so the model was told to avoid the very library that would have done
    the job. A bounded probe is the right thing in a prompt; the absolute
    claimed around it was not. Anything outside this list is *unknown*, and
    ``find_program`` is what answers for it.
    """

    import importlib.util

    # Probed, not assumed. Adding a name here costs one find_spec and makes the
    # briefing more informative; asserting a name is absent without probing it
    # is what made the briefing wrong. mss, pyperclip, keyboard, pyscreeze,
    # pygetwindow and pytesseract were all installed on this machine and on no
    # list, so a capability that needed one was told to use the standard
    # library instead.
    candidates = (
        "numpy", "requests", "chess", "cv2", "PIL", "yaml", "bs4", "lxml",
        "pandas", "scipy", "matplotlib", "pygame", "pydub", "mutagen",
        "psutil", "pyautogui", "comtypes", "pycaw", "win32api", "torch",
        "mss", "pyperclip", "keyboard", "pyscreeze", "pygetwindow", "pytesseract",
    )
    found = []
    for name in candidates:
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except (ImportError, ValueError):
            continue
        if len(found) >= limit:
            break
    return ", ".join(found) if found else "none beyond the standard library"


def audible_playback_check(python: str | None = None) -> CapabilityCheck:
    """Proof that a playback capability actually produced sound.

    Every other check is a proxy. run() returning a dict, the tests passing and
    no name being undefined do not distinguish a capability that plays music
    from one that describes playing music -- and a live acquisition produced
    exactly that, passing all four while every branch returned
    {"message": "Dry run: ..."}.

    This calls run() for real, with the audio meter watching. It belongs to the
    caller rather than to the standard bar because only a capability whose
    purpose is an external effect can be held to it.
    """

    # The audio meter lives in the extras virtualenv (pycaw/comtypes are
    # Windows COM bindings that do not belong in the interpreter running the
    # project engine). The capability itself only needs the standard library,
    # so running it under that interpreter costs nothing.
    executable = python or _audio_python()
    return CapabilityCheck(
        name="audible",
        text=(
            "calling run() with dry_run false actually produces sound, measured by the "
            "system audio meter"
        ),
        command=(
            executable,
            "-c",
            # Import the capability FIRST, then put Jarvis on the path.
            # Inserting the repository at sys.path[0] shadowed the workspace:
            # Jarvis has its own top-level main.py, so `import main` resolved to
            # that instead of the capability under test, and the check failed
            # with ModuleNotFoundError: transformers -- an error from a file the
            # capability had never heard of.
            "import sys; "
            "sys.path.insert(0, '.'); "
            "import main; "
            f"sys.path.append({str(_REPO_ROOT)!r}); "
            "from tools.audio_probe import observe_around; "
            "e = observe_around(lambda: main.run({}), seconds=2.5); "
            "print(e.detail); "
            "raise SystemExit(0 if e.audible else 1)",
        ),
    )


def capability_checks(python: str | None = None) -> list[CapabilityCheck]:
    """The complete, single definition of what makes a capability real."""

    executable = python or sys.executable
    return [
        CapabilityCheck(
            name="tests",
            text="the capability's own tests pass",
            command=(executable, "-m", "pytest", "-q", "test_capability.py"),
        ),
        CapabilityCheck(
            name="contract",
            text="main.run is implemented, importable, and returns a dict",
            command=(
                executable,
                "-c",
                # A bare NameError is a symptom the model diagnoses correctly
                # and then acts on wrongly: told "media_folders is not defined"
                # it tries to define it. The cause is that Jarvis TOOLS are not
                # importable from a capability, and saying so here turns three
                # wasted repair cycles into one actionable message.
                "import main\n"
                "assert callable(getattr(main, 'run', None)), 'main.run is missing'\n"
                "try:\n"
                "    r = main.run({'dry_run': True})\n"
                "except NameError as exc:\n"
                "    raise SystemExit(\n"
                "        f'{exc}. That name is a JARVIS TOOL, not something main.py can call. '\n"
                "        'Tools exist only while investigating. Use the standard library "
                "(shutil.which, pathlib, subprocess, os) instead, or hard-code what the tool told you.'\n"
                "    )\n"
                "assert isinstance(r, dict), f'run() returned {type(r)}'\n"
                "print('CONTRACT_OK')\n",
            ),
        ),
        CapabilityCheck(
            name="implemented",
            text=(
                "the placeholder marker is gone from BOTH main.py and test_capability.py, "
                "and the tests really exercise run()"
            ),
            command=(
                executable,
                "-c",
                "import pathlib; "
                "main_src = pathlib.Path('main.py').read_text(encoding='utf-8'); "
                "test_src = pathlib.Path('test_capability.py').read_text(encoding='utf-8'); "
                f"assert '{NOT_IMPLEMENTED}' not in main_src, "
                "'main.py still contains the placeholder marker: run() has not been implemented'; "
                f"assert '{NOT_IMPLEMENTED}' not in test_src, "
                "'test_capability.py still contains the placeholder marker: "
                "replace the seeded test with real ones'; "
                "assert 'main.run' in test_src, "
                "'the tests never call main.run, so they prove nothing'; "
                "assert test_src.count('assert') >= 2, "
                "'the tests make fewer than two assertions; write at least two that check behaviour'; "
                "print('SUBSTANCE_OK')",
            ),
        ),
        CapabilityCheck(
            name="static",
            text=(
                "every name main.py uses is defined, imported, or a builtin -- including in "
                "branches the tests never reach"
            ),
            command=(
                executable,
                "-c",
                # Executing one path proves one path. A side-effecting
                # capability is verified almost entirely through its dry run,
                # so the branch that does the real work is the branch least
                # likely to have been run. Observed live: a music capability
                # passed tests, contract and implemented while carrying
                # `media_control(...)` -- an undefined name -- in the else
                # branch its dry run never entered.
                "import sys; "
                f"sys.path.insert(0, {str(_REPO_ROOT)!r}); "
                "from capabilities.static_check import main; "
                "raise SystemExit(main(['main.py']))",
            ),
        ),
    ]


class CapabilityService:
    """Resolves, acquires, registers and executes capabilities."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        engine: ProjectEngine,
        graph: KnowledgeGraph | None = None,
        root: str | Path | None = None,
        execution_timeout: float = 120.0,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.graph = graph
        self.memory = ExperienceMemory(graph) if graph is not None else None
        self.root = Path(root or (Path(registry.path).parent / "installed"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_timeout = execution_timeout

    # ------------------------------------------------------------------
    # Resolving
    # ------------------------------------------------------------------

    def resolve(self, goal: str) -> CapabilityManifest | None:
        """Find an installed capability that can satisfy ``goal``.

        The registry's own term matching comes first, then the knowledge graph,
        which knows the vocabulary a capability declared for itself -- that is
        what lets "play some music" reach ``audio.play_file``.
        """

        matches = self.registry.find(goal, limit=1)
        if matches:
            return matches[0]

        if self.memory is not None:
            for node in self.memory.known_capabilities(goal, limit=3):
                manifest = self.registry.get(node.title)
                if manifest is not None and manifest.status == "active":
                    return manifest
        return None

    def has(self, capability_id: str) -> bool:
        return self.registry.has(capability_id)

    def list(self) -> list[CapabilityManifest]:
        return [item for item in self.registry.all() if item.status == "active"]

    # ------------------------------------------------------------------
    # Acquiring
    # ------------------------------------------------------------------

    def ensure(
        self,
        goal: str,
        *,
        max_steps: int = 40,
        keywords: list[str] | None = None,
        max_seconds: float | None = None,
        extra_checks: list[CapabilityCheck] | None = None,
        capability_id: str = "",
    ) -> CapabilityOutcome:
        """Return a capability for ``goal``, acquiring one if none exists.

        ``capability_id`` lets a caller name the thing being built instead of
        having one derived from the goal text.  That matters for anything with
        a contract rather than a description -- a music provider is looked up
        as ``music.provider.spotify``, not as whatever ``suggest_id`` makes of
        a paragraph -- and it is what lets a rebuild find the installed version
        to start from.  Without it a repair silently began from a blank
        skeleton, because the id it looked up was not the id it registered.
        """

        # A named capability is looked up by name. Similarity is the wrong
        # question once the caller has said which one it wants, and answering
        # the wrong question produced the worst false positive in this project:
        # a mission for `system.screen.capture` resolved by description overlap
        # to `music.provider.spotify` and reported the screen-capture
        # capability as acquired, having built nothing.
        if capability_id:
            named = self.registry.get(capability_id)
            if named is not None and named.status == "active":
                return CapabilityOutcome(
                    goal=goal,
                    capability_id=named.capability_id,
                    status="available",
                    manifest=named,
                    reason=f"{capability_id} is already installed",
                )
        else:
            existing = self.resolve(goal)
            if existing is not None:
                return CapabilityOutcome(
                    goal=goal,
                    capability_id=existing.capability_id,
                    status="available",
                    manifest=existing,
                    reason="an installed capability already covers this",
                )
        return self.acquire(
            goal,
            max_steps=max_steps,
            keywords=keywords,
            max_seconds=max_seconds,
            extra_checks=extra_checks,
            capability_id=capability_id,
        )

    def acquire(
        self,
        goal: str,
        *,
        max_steps: int = 40,
        keywords: list[str] | None = None,
        max_seconds: float | None = None,
        extra_checks: list[CapabilityCheck] | None = None,
        capability_id: str = "",
    ) -> CapabilityOutcome:
        """Build, verify and register a new capability.

        ``extra_checks`` are goal-specific criteria the standard bar cannot
        carry -- proof that a playback capability made a sound, say. They are
        given to the loop as acceptance criteria and re-run by verification,
        exactly like the standard ones, so a capability cannot be accepted
        without satisfying them and cannot self-certify them either.
        """

        capability_id = capability_id or self.suggest_id(goal)
        project = self._start_project(
            goal, capability_id, max_steps=max_steps, max_seconds=max_seconds,
            extra_checks=extra_checks,
        )
        session = self.engine.run(project, max_steps=max_steps)

        workspace = Path(project.workspace)
        verification = self._verify(workspace, extra_checks=extra_checks)

        if not (session.accepted and verification["ok"]):
            reason = (
                f"acquisition did not verify ({session.stop_reason.value}): "
                f"{session.message or verification.get('detail', '')}"
            )
            self._remember_attempt(project, capability_id, succeeded=False, reason=reason)
            return CapabilityOutcome(
                goal=goal,
                capability_id="",
                status="failed",
                project_id=project.id,
                verification=verification,
                reason=reason,
            )

        manifest = self._install(capability_id, goal, workspace, verification, keywords=keywords)
        self._remember_attempt(project, capability_id, succeeded=True, reason="verified and registered")
        return CapabilityOutcome(
            goal=goal,
            capability_id=manifest.capability_id,
            status="acquired",
            acquired=True,
            manifest=manifest,
            project_id=project.id,
            verification=verification,
            reason="implemented, verified by its own tests, and registered",
        )

    def _start_project(
        self,
        goal: str,
        capability_id: str,
        *,
        max_steps: int,
        max_seconds: float | None = None,
        extra_checks: list[CapabilityCheck] | None = None,
    ) -> Project:
        project = self.engine.create_project(
            f"Build a reusable capability that can: {goal}",
            kind="capability",
            title=capability_id,
            limits=ResourceLimits(
                max_steps=max_steps,
                # Acquisition is a long job, but never an unbounded one.
                max_seconds=min(3600.0, max_seconds) if max_seconds else 3600.0,
                max_consecutive_failures=8,
            ),
            constraints=[
                "Implement everything in main.py, exposing exactly one function: run(payload: dict) -> dict.",
                "run() must always return a dict and must never raise for an expected failure.",
                "Read every value out of payload with .get(), never with [] -- a missing key must be a "
                "clean error, not a KeyError.",
                "If the capability affects anything outside this process, honour payload.get('dry_run') by "
                "performing every check and reporting what it would do, without doing it.",
                # Reinventing this is a reliable failure: a hand-rolled PATH walk misses
                # powershell.exe on Windows because it ignores PATHEXT, and the capability
                # then concludes the machine has no player at all.
                "To find a program, use shutil.which() from the standard library. Do not write your own "
                "PATH search: shutil.which handles the .exe/.cmd extensions Windows needs.",
                # Measured twice on this machine, four months apart, in two
                # different files doing the same job. `tools.media_session` was
                # changed from an inline script to a file in `2f3a4a2` -- 4.4 s
                # a call became 1.1 s including PowerShell's own startup, while
                # the work inside was about a millisecond. The Spotify
                # capability was written before that and still passes its
                # script with -EncodedCommand: 8.8 s of wall clock for 0.3 s of
                # work, on every "what's playing".
                #
                # It is stated here because the lesson lived in a Jarvis module
                # that a model-authored capability cannot see. A capability
                # briefing is the only place a fact like this reaches the code
                # that needs it.
                "If you run PowerShell, write the script to a file once and run it with -File, passing "
                "values through a param() block. Do NOT pass a long script with -Command or "
                "-EncodedCommand: PowerShell re-parses the whole string on every invocation, and "
                "measured on this machine that is 8.8 seconds of wall clock for 0.3 seconds of work. "
                "One -Command, never two -- a second one is read as an argument to the first and the "
                "command silently does nothing.",
                "Prefer the Python standard library over an external program whenever it can do the job. "
                "A capability with no external dependency is more reliable and needs no installation.",
                "Use the 'find_program' tool to check what is actually installed on this machine before "
                "choosing an approach. Do not assume a program exists. Note that find_program is a Jarvis "
                "tool returning {found, path}, while shutil.which() returns a path string or None -- in "
                "your code use shutil.which() and treat its result as a string.",
                # Observed live, repeatedly: the model investigated with the
                # media_folders tool, then wrote `media_folders()` into
                # main.py. Tools and library functions look identical in a
                # transcript -- both are names that were called and returned
                # useful data -- so the distinction has to be stated, not
                # assumed. Three consecutive VERIFY failures came from exactly
                # this, each diagnosed as "media_folders is not defined".
                # Observed live: the model wrote `from playsound import playsound`
                # for a package that is not installed. It had no way to know --
                # nothing told it what this interpreter can import, so it
                # guessed a plausible name. Listing them is cheap and removes an
                # entire class of failure.
                f"Third-party packages known to be importable here: {_importable_packages()}. "
                "That is what was checked, NOT everything that exists -- before importing anything "
                "else, call find_program with the package name: it answers kind='python_package' for "
                "something you can import and kind='absent' for something you cannot. Prefer the "
                "standard library wherever it can do the job. "
                "winsound, subprocess, shutil, pathlib and os are always available on Windows.",
                "Jarvis TOOLS (media_folders, find_media, running_processes, find_applications, "
                "find_program, read_file, ...) exist only while you are investigating. They are NOT "
                "importable and NOT callable from main.py. Anything main.py needs at runtime must come "
                "from the Python standard library or a declared dependency. If a tool told you something "
                "useful -- a folder path, a program location -- put the ANSWER in your code, or "
                "rediscover it there with shutil.which(), pathlib and os.",
                "Keep the module-level INPUT_SCHEMA in main.py accurate: it must list every payload key "
                "run() reads. It is the only way a caller can know what to pass, so a key that is not "
                "declared there is a key nobody will ever send.",
                "Write real tests in test_capability.py that prove behaviour, not just key presence.",
                # Facts about this machine, probed deterministically and cached
                # rather than rediscovered by the loop. Four of the six earlier
                # music attempts failed on environment facts nobody had told the
                # model; INVESTIGATE then spent a tool call per attempt
                # re-establishing things that had not changed in days. Each line
                # carries how it was determined, so the difference between a
                # checked fact and an assumed one stays visible.
                *(
                    [
                        "Facts about this machine, each established by the probe named in brackets. "
                        "You do not need to rediscover these:",
                        *(f"  - {line}" for line in _environment_briefing()),
                    ]
                    if _environment_briefing()
                    else []
                ),
                # The failure this prevents: a test asserting a specific player is installed,
                # which is false on this machine, so a correct implementation still fails.
                "Tests must NOT assert that any particular external program is installed. Assert on "
                "behaviour instead: that a dry run reports whichever mechanism it chose, that a missing "
                "input fails cleanly, that the returned dict has the documented shape.",
            ],
            # The loop is graded by exactly the checks that decide acceptance.
            # Keeping two lists in step by hand is what failed before: a
            # criterion that can veto acceptance but never appears in the
            # loop's evidence is a hidden rubric, and the loop converges on
            # failing it.  Both live F failures ended contract=ok and
            # implemented=FAILED, because `implemented` also inspected the test
            # file and the loop's contract check did not.
            acceptance=[
                (check.text, list(check.command))
                for check in capability_checks() + list(extra_checks or [])
            ],
        )
        project.metadata["capability_id"] = capability_id

        workspace = self.engine.store.workspace_for(project)
        # Seed the workspace so the model edits a working skeleton rather than
        # inventing the file layout, which it gets wrong far more often.
        #
        # When a version of this capability already exists, seed from *that*
        # instead. Rebuilding a capability means improving the one there is;
        # starting from the skeleton throws away everything that worked in
        # order to fix the one thing that did not. Observed: a Spotify provider
        # of 794 working lines failed a single check because Spotify rejects
        # `limit=20` despite documenting a maximum of 50, and repairing it from
        # a blank template would have discarded the WinRT interop, the token
        # flow and the URI handoff to change one number.
        seeded = self._seed_from_installed(capability_id, workspace)
        if not seeded:
            # Nothing installed under this id -- but an earlier attempt at the
            # same capability may have been interrupted, and its workspace is
            # still on disk with real code in it.
            seeded = self._seed_from_interrupted(capability_id, project.id, workspace)
        if not seeded:
            (workspace / "main.py").write_text(_TEMPLATE_MAIN, encoding="utf-8")
            (workspace / "test_capability.py").write_text(_TEMPLATE_TEST, encoding="utf-8")
        self.engine.store.save(project)
        return project

    def _seed_from_interrupted(self, capability_id: str, current_id: str, workspace: Path) -> bool:
        """Continue from a previous attempt that was killed before it finished.

        The checkpoint records attempts that *completed*. An attempt that was
        interrupted -- the machine restarted, the process was killed, the time
        budget ran out mid-step -- records nothing, so a resumed mission began
        again from a blank skeleton. Its predecessor's workspace was still on
        disk the whole time, holding thirty or forty minutes of model output,
        and nothing looked at it. Observed four times over during the
        screen-capture benchmark: each restart threw away a partly-written
        implementation and paid for it again.

        Only a workspace that represents real progress is taken. A file that
        does not parse cannot be edited by anchors and is worse to inherit than
        a skeleton, and one that still carries the placeholder marker *is* the
        skeleton. Newest first, because that is the attempt that got furthest.
        """

        import ast

        store = getattr(self.engine, "store", None)
        if store is None:
            return False
        try:
            projects = [
                item for item in store.list_projects()
                if item.id != current_id
                and str(getattr(item, "title", "")) == capability_id
                and str(getattr(item, "state", "")) not in {"COMPLETED", "ProjectState.COMPLETED"}
            ]
        except Exception:
            return False

        projects.sort(key=lambda item: str(getattr(item, "updated_at", "")), reverse=True)
        for candidate in projects:
            source = Path(str(candidate.workspace or ""))
            main = source / "main.py"
            if not main.is_file():
                continue
            try:
                text = main.read_text(encoding="utf-8")
            except OSError:
                continue
            if NOT_IMPLEMENTED in text:
                continue
            try:
                ast.parse(text)
            except SyntaxError:
                continue
            copied = False
            for item in source.iterdir():
                if item.name in {"__pycache__", ".pytest_cache", ".venv"} or item.suffix == ".pyc":
                    continue
                try:
                    if item.is_dir():
                        shutil.copytree(item, workspace / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, workspace / item.name)
                    copied = True
                except OSError:
                    continue
            if copied:
                return True
        return False

    def _seed_from_installed(self, capability_id: str, workspace: Path) -> bool:
        """Copy the installed version of a capability into a fresh workspace.

        Returns whether anything was copied.  A disabled capability counts:
        being disabled is usually *why* it is being rebuilt.
        """

        manifest = self.registry.get(capability_id)
        if manifest is None:
            return False
        source = Path(str(getattr(manifest, "source_location", "") or ""))
        if not source.is_dir() or not (source / "main.py").is_file():
            return False
        copied = False
        for item in source.iterdir():
            if item.name in {"__pycache__", ".pytest_cache", ".venv"} or item.suffix == ".pyc":
                continue
            try:
                if item.is_dir():
                    shutil.copytree(item, workspace / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, workspace / item.name)
                copied = True
            except OSError:
                continue
        return copied

    # ------------------------------------------------------------------
    # Verification and installation
    # ------------------------------------------------------------------

    def _verify(
        self, workspace: Path, *, extra_checks: list[CapabilityCheck] | None = None
    ) -> dict[str, Any]:
        """Prove the capability works, independently of the project's own claim.

        Run separately from the loop's acceptance check on purpose: the loop
        decides when to stop, and something that did not participate in that
        decision should decide whether the result is real.
        """

        main = workspace / "main.py"
        if not main.exists():
            return {"ok": False, "detail": "main.py was never created"}

        checks: list[dict[str, Any]] = []

        # Re-run the very checks the loop was graded on, from a clean process.
        # Same bar, independently executed: the loop cannot be marked against a
        # rubric it never saw, and cannot certify itself either.
        for check in capability_checks() + list(extra_checks or []):
            outcome = self._run(list(check.command), workspace)
            checks.append(
                {
                    "name": check.name,
                    "ok": outcome["ok"],
                    "detail": outcome["detail"][-1500:],
                }
            )

        return {"ok": all(item["ok"] for item in checks), "checks": checks, "detail": "; ".join(
            f"{item['name']}={'ok' if item['ok'] else 'FAILED'}" for item in checks
        )}

    def _install(
        self,
        capability_id: str,
        goal: str,
        workspace: Path,
        verification: dict[str, Any],
        *,
        keywords: list[str] | None = None,
    ) -> CapabilityManifest:
        """Copy the verified workspace into the permanent catalog and register it."""

        version = self._next_version(capability_id)
        target = self.root / capability_id.replace(".", "_") / version
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            workspace,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".venv"),
        )

        terms = sorted(set((keywords or []) + _keywords_from(goal)))
        manifest = CapabilityManifest(
            capability_id=capability_id,
            description=goal,
            version=version,
            entrypoint="main.py",
            source_location=str(target.resolve()),
            tests_location=str((target / "test_capability.py").resolve()),
            input_schema=self._input_schema_of(target),
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            creation_metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "capability_service",
                "goal": goal,
                "keywords": terms,
            },
            validation_status={"verified": True, "checks": verification.get("checks", [])},
        )
        self.registry.register(manifest)

        if self.memory is not None:
            self.memory.record_capability(capability_id, goal, keywords=terms, version=version)
        return manifest

    def _input_schema_of(self, source: Path) -> dict[str, Any]:
        """Read the capability's declared INPUT_SCHEMA out of its own module.

        A capability that invents its own payload key is useless if nobody can
        discover the key: an acquired audio player expecting ``audio_path`` was
        unusable by a caller that passed ``path``. Reading the declaration makes
        the capability self-describing, which is what a registry is for.

        Extracted in a subprocess, since importing model-authored code into the
        Jarvis process is exactly what :meth:`execute` avoids.
        """

        fallback = {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": []}
        result = self._run(
            [
                sys.executable,
                "-c",
                "import json, main; "
                "schema = getattr(main, 'INPUT_SCHEMA', None); "
                "print('SCHEMA:' + json.dumps(schema if isinstance(schema, dict) else {}))",
            ],
            source,
        )
        if not result["ok"] or "SCHEMA:" not in result["detail"]:
            return fallback
        try:
            raw = result["detail"].split("SCHEMA:", 1)[1].splitlines()[0]
            schema = json.loads(raw)
        except (IndexError, json.JSONDecodeError):
            return fallback
        return schema if schema.get("properties") else fallback

    def _next_version(self, capability_id: str) -> str:
        existing = self.registry.get(capability_id)
        if existing is None:
            return "1.0.0"
        try:
            major, minor, patch = (int(part) for part in existing.version.split("."))
        except ValueError:
            return "1.0.1"
        return f"{major}.{minor}.{patch + 1}"

    # ------------------------------------------------------------------
    # Executing
    # ------------------------------------------------------------------

    def execute(self, capability_id: str, payload: dict[str, Any] | None = None) -> ExecutionOutcome:
        """Run an installed capability.

        Runs from the installed copy, in a subprocess, with a timeout. A
        subprocess rather than an import because a capability is model-authored
        code: it should not be able to take the Jarvis process down with it, and
        it must be reloadable without a restart.
        """

        import time

        manifest = self.registry.get(capability_id)
        if manifest is None or manifest.status != "active":
            return ExecutionOutcome(capability_id=capability_id, ok=False, error=f"no active capability {capability_id!r}")
        outcome = self._execute(manifest, capability_id, payload)
        # Runtime health is written by what actually happened, every time.
        try:
            self.registry.note_execution(capability_id, bool(outcome.ok), str(outcome.error or ((outcome.output or {}).get("error", "") if isinstance(outcome.output, dict) else ""))[:300])
        except Exception:  # noqa: BLE001 - health bookkeeping never masks the outcome
            pass
        return outcome

    def _execute(self, manifest: Any, capability_id: str, payload: dict[str, Any] | None = None) -> ExecutionOutcome:
        import time

        source = Path(manifest.source_location)
        if not source.exists():
            return ExecutionOutcome(
                capability_id=capability_id, ok=False, error=f"installed source is missing: {source}"
            )

        request = dict(payload or {})
        started = time.perf_counter()
        run_dir = source.parent / f".run_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            (run_dir / "request.json").write_text(json.dumps(request, default=str), encoding="utf-8")
            (run_dir / "runner.py").write_text(_runner_source(manifest.entrypoint), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(run_dir / "runner.py")],
                cwd=str(source),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.execution_timeout,
                env={**_safe_env(), "JARVIS_CAPABILITY_RUN_DIR": str(run_dir)},
            )
            duration = time.perf_counter() - started

            if completed.returncode != 0:
                return ExecutionOutcome(
                    capability_id=capability_id,
                    ok=False,
                    error=f"exit={completed.returncode}\n{(completed.stderr or completed.stdout)[-1500:]}",
                    duration_seconds=duration,
                )
            output_path = run_dir / "output.json"
            if not output_path.exists():
                return ExecutionOutcome(
                    capability_id=capability_id, ok=False, error="the capability produced no output", duration_seconds=duration
                )
            output = json.loads(output_path.read_text(encoding="utf-8"))
            return ExecutionOutcome(
                capability_id=capability_id,
                ok=bool(output.get("ok", True)),
                output=output,
                error=str(output.get("error", "")),
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            return ExecutionOutcome(
                capability_id=capability_id,
                ok=False,
                error=f"the capability did not finish within {self.execution_timeout:.0f}s",
                duration_seconds=time.perf_counter() - started,
            )
        except (OSError, json.JSONDecodeError) as exc:
            return ExecutionOutcome(
                capability_id=capability_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - started,
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def use(self, goal: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> tuple[CapabilityOutcome, ExecutionOutcome | None]:
        """Acquire if needed, then run.  The whole lifecycle in one call."""

        outcome = self.ensure(goal, **kwargs)
        if not outcome.usable:
            return outcome, None
        return outcome, self.execute(outcome.capability_id, payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_id(goal: str) -> str:
        """A stable, readable identifier derived from the goal."""

        import re

        words = [word for word in re.split(r"[^a-z0-9]+", goal.lower()) if len(word) > 2]
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "can", "able", "ability",
            "build", "make", "create", "reusable", "capability", "jarvis", "please", "want", "need",
            "der", "die", "das", "und", "fuer", "mit", "eine", "einen", "kannst", "kann",
        }
        useful = [word for word in words if word not in stopwords][:4]
        if not useful:
            useful = ["capability", uuid.uuid4().hex[:6]]
        return "local." + ".".join(useful[:2]) + ("." + "_".join(useful[2:]) if len(useful) > 2 else "")

    def _run(self, command: list[str], cwd: Path) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.execution_timeout,
                env=_safe_env(),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": completed.returncode == 0,
            "detail": f"exit={completed.returncode}\n{completed.stdout}\n{completed.stderr}",
        }

    def _remember_attempt(self, project: Project, capability_id: str, *, succeeded: bool, reason: str) -> None:
        if self.memory is None:
            return
        self.memory.record_project(project)
        if not succeeded:
            from knowledge.memory import Lesson

            self.memory.record_lesson(
                Lesson(
                    text=f"acquiring {capability_id} failed: {reason[:180]}",
                    worked=False,
                    evidence=reason[:400],
                    source_project=project.id,
                    tags=("capability",),
                )
            )


def _runner_source(entrypoint: str) -> str:
    module = Path(entrypoint).stem
    return (
        "import importlib\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "\n"
        'run_dir = pathlib.Path(os.environ["JARVIS_CAPABILITY_RUN_DIR"])\n'
        "sys.path.insert(0, str(pathlib.Path.cwd()))\n"
        'payload = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))\n'
        f'module = importlib.import_module("{module}")\n'
        "try:\n"
        "    result = module.run(payload)\n"
        "except Exception as exc:\n"
        '    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}\n'
        'if not isinstance(result, dict):\n'
        '    result = {"ok": False, "error": f"run() returned {type(result).__name__}, expected dict"}\n'
        '(run_dir / "output.json").write_text(json.dumps(result, default=str), encoding="utf-8")\n'
    )


def _safe_env() -> dict[str, str]:
    import os

    allowed = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME", "USERPROFILE",
        "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "LANG",
    )
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _keywords_from(goal: str) -> list[str]:
    """Subject words from a goal, for the registry to index the capability under.

    The first eight long words of a goal, which is fine for the goal that
    creates a capability and wrong for the goal that repairs one. A repair
    brief opens with the defect -- deliberately, so the planner plans a repair
    rather than a rebuild -- so its first eight long words are *defect,
    existing, implementation, rebuild, repair, working*, and those are what
    v1.0.4 of a music provider ended up indexed under. It then answered
    "rebuild the existing implementation because of a defect and repair the
    working code" with a music player.

    Filtered against the same vocabulary resolution ignores, so a word that
    could never contribute to a match is never stored as one either. Every
    capability's goal contains these words; a term shared by everything
    distinguishes nothing.
    """

    import re

    from capabilities.registry import BOILERPLATE

    stopwords = BOILERPLATE | {"jarvis", "please", "want", "need", "should", "able"}
    return [
        word
        for word in re.split(r"[^a-z0-9]+", goal.lower())
        if len(word) > 3 and word not in stopwords
    ][:8]
