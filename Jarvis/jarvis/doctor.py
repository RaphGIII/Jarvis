"""Tell whether an autonomous run is alive, stalled, dead or finished.

    python -m jarvis.doctor --run <dir>     # a specific run
    python -m jarvis.doctor                 # every run it can find
    python -m jarvis.doctor --watch         # keep looking

This exists because of a concrete misdiagnosis. An evidence run went quiet, and
from the outside there was no way to distinguish "the 7B model is four minutes
into a large prompt" from "the process is wedged"; it had in fact finished
cleanly hours earlier, and the only way to find that out was to go rummaging
through process tables and file timestamps by hand.

The distinction the doctor reports is the one that matters:

``alive``      breathing, and moving forward
``stalled``    breathing, but nothing has advanced for a long time
``dead``       the process is gone and never wrote a finish
``finished``   it completed; the outcome is printed
``unknown``    no heartbeat here

Exit codes let it be used from a script: 0 alive or finished, 1 stalled, 2 dead,
3 nothing found.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from runtime.heartbeat import Liveness, check_liveness

DEFAULT_ROOTS = (
    Path(__file__).resolve().parent.parent / "data" / "jarvis",
    Path(__file__).resolve().parent.parent / "data",
)


def find_heartbeats(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            found.append(root)
            continue
        found.extend(sorted(root.rglob("heartbeat.json")))
    # Newest first: the run someone is asking about is almost always the latest.
    return sorted(set(found), key=lambda path: path.stat().st_mtime, reverse=True)


def render(path: Path, liveness: Liveness) -> str:
    mark = {"alive": "ok  ", "finished": "done", "stalled": "STALL", "dead": "DEAD", "unknown": "?   "}
    beat = f"{liveness.seconds_since_beat:.0f}s" if liveness.seconds_since_beat is not None else "-"
    progress = f"{liveness.seconds_since_progress:.0f}s" if liveness.seconds_since_progress is not None else "-"
    payload = liveness.heartbeat
    lines = [
        f"{mark.get(liveness.state, '?'):>5}  {liveness.state.upper():<9} {path}",
        f"       run={payload.get('run', '?')} stage={payload.get('stage', '?')} steps={payload.get('steps', 0)}",
        f"       last beat {beat} ago, last progress {progress} ago, elapsed {payload.get('elapsed_seconds', 0)}s",
        f"       {liveness.detail}",
    ]
    if payload.get("detail"):
        lines.append(f"       detail: {str(payload['detail'])[:120]}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report whether autonomous runs are alive, stalled or dead.")
    parser.add_argument("--run", action="append", default=[], help="a run directory or heartbeat.json (repeatable)")
    parser.add_argument("--dead-after", type=float, default=60.0, help="seconds without a heartbeat before 'dead'")
    parser.add_argument(
        "--stalled-after",
        type=float,
        default=900.0,
        help="seconds without progress before 'stalled' (generous: one local step can take minutes)",
    )
    parser.add_argument("--watch", action="store_true", help="keep reporting until interrupted")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(item) for item in args.run] or list(DEFAULT_ROOTS)

    while True:
        paths = find_heartbeats(roots)
        if not paths:
            print(f"No heartbeat found under: {', '.join(str(root) for root in roots)}")
            if not args.watch:
                raise SystemExit(3)
        else:
            results = [(path, check_liveness(path, dead_after=args.dead_after, stalled_after=args.stalled_after)) for path in paths]
            if args.json:
                print(json.dumps([{"path": str(p), **l.__dict__} for p, l in results], indent=2, default=str))
            else:
                for path, liveness in results:
                    print(render(path, liveness))
                    print()
            if not args.watch:
                states = {liveness.state for _, liveness in results}
                if "dead" in states:
                    raise SystemExit(2)
                if "stalled" in states:
                    raise SystemExit(1)
                raise SystemExit(0)

        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
