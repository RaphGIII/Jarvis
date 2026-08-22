from __future__ import annotations

import argparse
import json
import os
import shlex
import tempfile
import uuid
from pathlib import Path

from brain.providers import make_brain_provider_from_env
from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal, SelfImprovementMemory


def run_self_developer_from_args(args: argparse.Namespace) -> dict:
    if args.brain_provider:
        os.environ["JARVIS_BRAIN_PROVIDER"] = args.brain_provider
    repo = Path(args.repo).resolve()
    benchmark_root = Path(args.benchmark_dir or (repo / "data" / "self_development")).resolve()
    benchmark_root.mkdir(parents=True, exist_ok=True)
    configured_worktree_root = getattr(args, "worktree_root", None)
    worktree_root = Path(configured_worktree_root).resolve() if configured_worktree_root else Path(tempfile.gettempdir()) / "jarvis_selfdev" / uuid.uuid4().hex[:12]
    goal = SelfImprovementGoal(
        objective=args.goal,
        success_criteria=list(args.success_criteria or []),
        allowed_paths=list(args.allowed_path or ["."]),
        protected_paths=list(args.protected_path or []),
        tests=[_split_command(item) for item in args.test_command],
        full_tests=[_split_command(item) for item in args.full_test_command],
        benchmarks=[_split_command(item) for item in args.benchmark_command],
        require_benchmark_improvement=bool(args.require_benchmark_improvement),
        metric_name=args.metric_name,
    )
    brain = make_brain_provider_from_env()
    engineer = RepositoryEngineer(
        brain=brain,
        worktree_root=worktree_root,
        memory=SelfImprovementMemory(benchmark_root / "self_development_trajectories.jsonl"),
        timeout_seconds=float(args.timeout_seconds),
        max_cycles=int(args.max_cycles),
    )
    result = engineer.improve(
        repo,
        goal,
        goal.tests,
        full_test_commands=goal.full_tests,
        benchmark_commands=goal.benchmark_commands(),
        max_cycles=args.max_cycles,
    )
    payload = {
        "STATUS": result.status,
        "SUCCESS": result.success,
        "WORKTREE": result.worktree,
        "CHANGED_FILES": result.changed_files,
        "DIFF_PATH": result.diff_path,
        "RESULT_PATH": result.result_path,
        "TARGETED_TESTS_PASSED": all(item.success for item in result.tests) if result.tests else True,
        "FULL_TESTS_PASSED": all(item.success for item in result.full_tests) if result.full_tests else True,
        "BENCHMARKS_BEFORE": [item.to_dict() for item in result.benchmarks_before],
        "BENCHMARKS_AFTER": [item.to_dict() for item in result.benchmarks_after],
        "REVIEW": result.review.to_dict() if result.review else None,
        "PROTECTED_PRISTINE": result.protected_pristine,
        "INSPECT_COMMAND": f"git -C {result.worktree} diff",
        "APPLY_COMMAND": f"git -C {repo} diff --no-index . {result.worktree}",
    }
    output_path = benchmark_root / "self_development_result.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["SUMMARY_PATH"] = str(output_path)
    return payload


def _split_command(value: str) -> list[str]:
    return [part.strip("\"'") for part in shlex.split(value, posix=False)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run JARVIS Self-Developer v1 against a real repository worktree.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--success-criteria", action="append", default=[])
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--protected-path", action="append", default=[])
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--full-test-command", action="append", default=[])
    parser.add_argument("--benchmark-command", action="append", default=[])
    parser.add_argument("--require-benchmark-improvement", action="store_true")
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--brain-provider", choices=["local_transformers", "openai_compatible"], default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--worktree-root", default=None)
    return parser


def main() -> None:
    payload = run_self_developer_from_args(build_parser().parse_args())
    for key in [
        "STATUS",
        "SUCCESS",
        "WORKTREE",
        "CHANGED_FILES",
        "DIFF_PATH",
        "RESULT_PATH",
        "SUMMARY_PATH",
        "TARGETED_TESTS_PASSED",
        "FULL_TESTS_PASSED",
        "PROTECTED_PRISTINE",
        "INSPECT_COMMAND",
        "APPLY_COMMAND",
    ]:
        if key in payload:
            print(f"{key}: {payload[key]}", flush=True)


if __name__ == "__main__":
    main()
