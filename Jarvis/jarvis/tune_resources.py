"""CLI: benchmark this machine and persist the resulting resource policy.

    python -m jarvis.tune_resources                     # tune FAST_LOCAL + BUILD_LOCAL
    python -m jarvis.tune_resources --tier BUILD_LOCAL  # just the build model
    python -m jarvis.tune_resources --show              # print the stored policy

Each candidate context size costs a real model load, so a full sweep takes a few
minutes.  That is the point: the numbers in the policy are measured on the
machine that will run the work, not inferred from a spec sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brain.resources import HostProbe, ResourcePolicyStore, ResourceTuner
from brain.tiers import ModelCatalog, ModelTier

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "resources.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local inference and store a resource policy.")
    parser.add_argument("--tier", action="append", default=[], help="tier to tune (repeatable); default FAST_LOCAL and BUILD_LOCAL")
    parser.add_argument("--candidate", action="append", type=int, default=[], help="context size to try (repeatable)")
    parser.add_argument("--min-tokens-per-second", type=float, default=3.0)
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--show", action="store_true", help="print the stored policy and exit")
    parser.add_argument("--json", action="store_true", help="emit the policy as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = ResourcePolicyStore(args.policy_path)

    if args.show:
        policy = store.load()
        if policy is None:
            print(f"No stored policy at {args.policy_path}. Run without --show to create one.")
            return
        print(json.dumps(policy.to_dict(), indent=2, sort_keys=True) if args.json else _render(policy))
        return

    host = HostProbe().detect()
    print(f"Host:  {host.platform}, {host.cpu_count} CPUs, {host.total_ram_mib} MiB RAM")
    for gpu in host.gpus:
        print(f"GPU:   {gpu.name}, {gpu.total_mib} MiB total, {gpu.free_mib} MiB free")
    if not host.gpus:
        print("GPU:   none detected (CPU inference)")

    tiers = [ModelTier(name.upper()) for name in args.tier] or [ModelTier.FAST_LOCAL, ModelTier.BUILD_LOCAL]
    print(f"\nTuning {', '.join(tier.value for tier in tiers)} -- this loads each model once per candidate size.\n")

    tuner = ResourceTuner(ModelCatalog())
    policy = tuner.tune(
        tiers=tiers,
        candidates=args.candidate or None,
        min_tokens_per_second=args.min_tokens_per_second,
    )
    path = store.save(policy)

    print(_render(policy))
    print(f"\nSaved to {path}")


def _render(policy) -> str:
    lines = ["Resource policy", "---------------"]
    lines.append(f"max concurrent generations: {policy.max_concurrent_generations}")
    lines.append(f"reserved VRAM:              {policy.reserved_vram_mib} MiB")
    for tier, window in sorted(policy.context_windows.items()):
        lines.append(f"{tier:<16} context {window}, keep_alive {policy.keep_alive.get(tier, '-')}")
    if policy.measurements:
        lines.append("")
        lines.append("Measurements")
        for tier, rows in sorted(policy.measurements.items()):
            for row in rows:
                status = "ok " if row.get("ok") else "FAIL"
                lines.append(
                    f"  {tier:<16} ctx {row.get('context_window'):>6}  {status}  "
                    f"{row.get('tokens_per_second', 0):>6.2f} tok/s  "
                    f"load {row.get('load_seconds', 0):>6.2f}s  "
                    f"VRAM free {row.get('vram_free_mib', 0):>6} MiB"
                    + (f"  {row.get('error')}" if row.get("error") else "")
                )
    if policy.notes:
        lines.append("")
        lines.append("Notes")
        lines.extend(f"  - {note}" for note in policy.notes)
    if policy.tuned_at:
        lines.append(f"\ntuned at {policy.tuned_at}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
