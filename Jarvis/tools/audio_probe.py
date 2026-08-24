"""Did sound actually come out of the speakers?

Every other check a playback capability can pass is a proxy.  ``run()`` returned
a dict, the tests passed, no name is undefined -- none of that distinguishes a
capability that plays music from one that describes playing music.  The live
acquisition produced exactly that: a module whose every branch returned
``{"message": "Dry run: ..."}`` and passed all four checks without emitting a
sound.

Windows keeps a peak meter per audio session, so the question has a real answer.
Measured on this machine: the meter reads ~0.02 in a quiet room and exactly 0.30
while a 0.30-amplitude tone plays.  That is the difference between a capability
that works and one that claims to, and it is the only evidence worth accepting
for something whose entire purpose is an external effect.

Two things this deliberately does NOT do:

*It does not identify the music.*  Whether the right track is playing is not
something a meter can answer, and pretending otherwise would be the same kind of
false confidence this module exists to remove.

*It does not run in the main interpreter.*  pycaw and comtypes live in the
extras virtualenv, and this re-executes itself there rather than adding two
Windows-only COM dependencies to the interpreter that runs the project engine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Peak above which we are willing to say "audio is playing".  The measured
#: quiet floor on this machine is ~0.02; a real signal was 0.30.  0.05 sits
#: clear of the floor without demanding the user's volume be high.
AUDIBLE_PEAK = 0.05

#: How long a single measurement listens for.  Long enough to survive a gap
#: between notes, short enough not to stall an acceptance check.
DEFAULT_WINDOW = 1.5


@dataclass
class PlaybackEvidence:
    """What the meter saw, before and during."""

    baseline_peak: float = 0.0
    observed_peak: float = 0.0
    audible: bool = False
    detail: str = ""
    samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_peak": round(self.baseline_peak, 5),
            "observed_peak": round(self.observed_peak, 5),
            "audible": self.audible,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# The measurement itself (only runs where pycaw is importable)
# --------------------------------------------------------------------------

def _session_peak() -> float:
    """The loudest thing any audio session is currently playing."""

    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation

    loudest = 0.0
    for session in AudioUtilities.GetAllSessions():
        try:
            meter = session._ctl.QueryInterface(IAudioMeterInformation)
            loudest = max(loudest, float(meter.GetPeakValue()))
        except Exception:
            # A session can disappear between enumeration and query; that is
            # ordinary, not an error worth propagating.
            continue
    return loudest


def measure(seconds: float = DEFAULT_WINDOW, *, interval: float = 0.1) -> tuple[float, list[float]]:
    """Poll the meter for ``seconds`` and return the peak seen, plus the samples.

    Sampled rather than taken once because music has gaps: a single reading
    between two notes says "silent" about something that is plainly playing.
    """

    samples: list[float] = []
    deadline = time.perf_counter() + max(0.1, seconds)
    while time.perf_counter() < deadline:
        samples.append(_session_peak())
        time.sleep(interval)
    return (max(samples) if samples else 0.0), samples


def observe(seconds: float = DEFAULT_WINDOW, *, baseline_seconds: float = 0.6) -> PlaybackEvidence:
    """Measure the room, then measure again, and compare.

    The comparison matters: a machine with a fan, a browser tab or a
    notification sound has a non-zero floor, and treating "peak > 0" as proof
    would accept any capability that did nothing while something else made
    noise.
    """

    baseline, _ = measure(baseline_seconds)
    observed, samples = measure(seconds)

    audible = observed >= AUDIBLE_PEAK and observed > baseline * 1.5
    if audible:
        detail = f"peak {observed:.3f} against a baseline of {baseline:.3f}"
    elif observed < AUDIBLE_PEAK:
        detail = f"nothing audible: peak {observed:.3f} is below {AUDIBLE_PEAK}"
    else:
        detail = (
            f"peak {observed:.3f} is not meaningfully above the baseline {baseline:.3f}; "
            "something was already making noise"
        )
    return PlaybackEvidence(
        baseline_peak=baseline,
        observed_peak=observed,
        audible=audible,
        detail=detail,
        samples=samples,
    )


# --------------------------------------------------------------------------
# Running it from an interpreter that lacks pycaw
# --------------------------------------------------------------------------

def available() -> bool:
    try:
        import pycaw  # noqa: F401
    except ImportError:
        return False
    return True


def extras_python() -> Path | None:
    root = Path(__file__).resolve().parent.parent / ".venv-speech"
    for candidate in (root / "Scripts" / "python.exe", root / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def observe_anywhere(seconds: float = DEFAULT_WINDOW) -> PlaybackEvidence:
    """Measure, re-executing in the extras virtualenv when necessary."""

    if available():
        return observe(seconds)

    interpreter = extras_python()
    if interpreter is None:
        return PlaybackEvidence(
            detail="no audio meter available: pycaw is not installed in any known interpreter"
        )

    root = str(Path(__file__).resolve().parent.parent)
    try:
        completed = subprocess.run(
            [str(interpreter), "-c",
             f"import sys; sys.path.insert(0, {root!r}); "
             "from tools.audio_probe import main; raise SystemExit(main(['--json', '--seconds', "
             f"'{seconds}']))"],
            capture_output=True, text=True, timeout=max(30.0, seconds * 6),
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PlaybackEvidence(detail=f"audio probe failed: {exc}")

    try:
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return PlaybackEvidence(detail=f"audio probe returned nothing usable: {completed.stderr[-200:]}")

    return PlaybackEvidence(
        baseline_peak=float(payload.get("baseline_peak", 0.0)),
        observed_peak=float(payload.get("observed_peak", 0.0)),
        audible=bool(payload.get("audible")),
        detail=str(payload.get("detail", "")),
    )


def observe_around(
    start: "Callable[[], Any]",
    *,
    seconds: float = DEFAULT_WINDOW,
    baseline_seconds: float = 0.6,
    settle: float = 0.4,
) -> PlaybackEvidence:
    """Measure the room, start something, then measure again.

    This ordering is the whole point and is easy to get wrong -- measuring the
    baseline while the thing is already playing makes the two readings identical
    and reports "something was already making noise", which is technically true
    and useless.  Taking the baseline first is what turns the meter into a test.
    """

    baseline, _ = measure(baseline_seconds)

    error = ""
    try:
        start()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    # Give the audio stack a moment: a player that has just been asked to start
    # is not producing samples yet, and measuring immediately reports silence
    # about something that is about to be perfectly audible.
    time.sleep(settle)
    observed, samples = measure(seconds)

    audible = observed >= AUDIBLE_PEAK and observed > baseline * 1.5
    if error:
        detail = f"the action failed: {error}"
    elif audible:
        detail = f"peak {observed:.3f} against a baseline of {baseline:.3f}"
    elif observed < AUDIBLE_PEAK:
        detail = f"nothing audible: peak {observed:.3f} is below {AUDIBLE_PEAK}"
    else:
        detail = (
            f"peak {observed:.3f} is not meaningfully above the baseline {baseline:.3f}; "
            "something was already making noise"
        )
    return PlaybackEvidence(
        baseline_peak=baseline,
        observed_peak=observed,
        audible=audible and not error,
        detail=detail,
        samples=samples,
    )


def _run_command(command: str) -> "Callable[[], Any]":
    def start() -> None:
        # Popen, not run: the point is to measure while it plays, and a
        # blocking call would only return once the sound had finished.
        subprocess.Popen(command, shell=True)

    return start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.audio_probe",
        description="Report whether audio is currently coming out of this machine.",
    )
    parser.add_argument("--seconds", type=float, default=DEFAULT_WINDOW)
    parser.add_argument("--json", action="store_true", help="print the evidence as JSON")
    parser.add_argument(
        "--require-audible",
        action="store_true",
        help="exit non-zero unless something audible was heard (for acceptance checks)",
    )
    parser.add_argument(
        "--run",
        default="",
        help="shell command to start BEFORE measuring; the baseline is taken first",
    )
    args = parser.parse_args(argv)

    if args.run:
        evidence = observe_around(_run_command(args.run), seconds=args.seconds)
    else:
        evidence = observe(args.seconds) if available() else observe_anywhere(args.seconds)

    if args.json:
        print(json.dumps(evidence.to_dict()))
    else:
        print(("AUDIBLE: " if evidence.audible else "SILENT: ") + evidence.detail)

    if args.require_audible and not evidence.audible:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
