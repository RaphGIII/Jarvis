"""Gates for a capability that claims to have produced something.

The music provider needed a check that asked Windows what was playing, because
"it ran without error" is not the same as "a sound came out".  Every capability
with a real-world effect needs the equivalent, and most of them produce a *file*
-- a screenshot, an export, a report, a download, a conversion.

So this is the general form of that check.  A capability says where it put
something; these gates go and look, from outside, with no help from the
capability.  They are deliberately domain-blind: nothing here knows what a
screenshot is.  What they know is the shape of the lie a capability tells when
it has not done the work.

Four lies, four checks:

*It said it wrote a file and there is no file.*  The commonest, and the one that
"returned ok=True" cannot distinguish from success.

*The file is the wrong kind.*  A capability asked for a PNG that writes a text
file containing the word "screenshot" satisfies every check that only looks at
the path.  Magic bytes are read from the file, never from its extension, which
the capability also chose.

*The file was already there.*  A capability that reports the path of something
produced an hour ago -- or by a previous run of itself -- passes an existence
check trivially.  Freshness is what separates *produced* from *found*.

*The file is empty, or content-free.*  A zero-byte PNG is a valid path with a
valid extension.  A screenshot of nothing is a single colour repeated a million
times, and compresses to almost nothing; a real one does not.  The size floor is
crude on purpose, because a precise one would need to know what was being made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Magic bytes, so the *content* decides the type rather than the name the
#: capability chose for it.
SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "gif": (b"GIF87a", b"GIF89a"),
    "pdf": (b"%PDF-",),
    "zip": (b"PK\x03\x04",),
    "wav": (b"RIFF",),
    "sqlite": (b"SQLite format 3\x00",),
}

#: Below this, a file of any type is very unlikely to contain anything. A blank
#: 1920x1080 PNG is about 8 kB; a real screenshot is hundreds. Deliberately far
#: below both, because the floor exists to catch zero and near-zero, not to
#: judge quality.
MIN_BYTES = 1024

#: How recently the artifact must have been written to count as produced by
#: this run rather than found lying around.
MAX_AGE_SECONDS = 120.0


def artifact_check_source(
    payload: dict,
    *,
    kind: str = "",
    min_bytes: int = MIN_BYTES,
    max_age_seconds: float = MAX_AGE_SECONDS,
    result_key: str = "path",
) -> str:
    """Python source for a gate that runs the capability and inspects its output.

    Returned as source rather than run here because capability checks execute
    in the capability's own workspace, in a separate process, exactly as the
    acquisition loop will run them.
    """

    return (
        "import sys, json, time, pathlib;"
        "import main;"
        f"payload = {json.dumps(payload)};"
        "result = main.run(payload);"
        "assert isinstance(result, dict), 'run() did not return a dict: ' + repr(result);"
        "assert result.get('ok') is True, 'the capability reported failure: ' "
        "+ str(result.get('error'));"
        f"raw = result.get({result_key!r});"
        f"assert raw, 'run() reported success but no {result_key} came back: ' + repr(result);"
        "path = pathlib.Path(str(raw));"
        "assert path.is_file(), 'nothing exists at the reported path: ' + str(path);"
        "size = path.stat().st_size;"
        f"assert size >= {min_bytes}, 'the artifact is ' + str(size) + ' bytes, which is not a "
        f"file with anything in it';"
        "age = time.time() - path.stat().st_mtime;"
        f"assert age <= {max_age_seconds}, 'the artifact is ' + str(int(age)) + 's old, so it was "
        f"found rather than produced';"
        + (
            "head = path.open('rb').read(16);"
            f"signatures = {json.dumps([sig.hex() for sig in SIGNATURES.get(kind, ())])};"
            "assert any(head.hex().startswith(sig) for sig in signatures), "
            f"'the file at ' + str(path) + ' is not a {kind}: it begins ' + repr(head[:8]);"
            if kind in SIGNATURES
            else ""
        )
        + "print('ARTIFACT_OK', path, size, 'bytes', round(age, 1), 's old')"
    )


def artifact_check(
    name: str,
    text: str,
    payload: dict,
    *,
    kind: str = "",
    python: str | None = None,
    min_bytes: int = MIN_BYTES,
    max_age_seconds: float = MAX_AGE_SECONDS,
    result_key: str = "path",
):
    """A :class:`~capabilities.service.CapabilityCheck` that verifies an artifact."""

    from capabilities.service import CapabilityCheck

    return CapabilityCheck(
        name=name,
        text=text,
        command=(
            python or sys.executable,
            "-c",
            artifact_check_source(
                payload,
                kind=kind,
                min_bytes=min_bytes,
                max_age_seconds=max_age_seconds,
                result_key=result_key,
            ),
        ),
    )


def screen_size() -> tuple[int, int]:
    """The primary display's pixel size, from Windows rather than from a guess.

    Used by a gate that wants to know whether an image is plausibly a capture of
    *this* screen. Read here, in the checking process, so a capability cannot
    influence the number it will be compared against.
    """

    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return (0, 0)
