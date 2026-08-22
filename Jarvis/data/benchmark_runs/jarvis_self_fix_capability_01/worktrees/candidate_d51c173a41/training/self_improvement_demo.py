from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brain.providers import make_brain_provider_from_env
from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal, SelfImprovementMemory


@dataclass
class SelfImprovementDemoConfig:
    goal: str = "Improve semantic capability reuse in the fixture resolver."
    benchmark_dir: str | None = None
    brain_provider: str | None = None
    mock_brain: bool = True
    quiet: bool = False


class MockRepositoryBrain:
    provider_name = "mock_repository_engineer"
    model_name = "MockRepositoryBrain"

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        return json.dumps(
            {
                "analysis": "Add simple synonym expansion so semantically equivalent count-line requests resolve.",
                "files": [
                    {
                        "path": "resolver.py",
                        "content": _improved_resolver_source(),
                    }
                ],
                "new_files": [],
                "deleted_files": [],
            }
        )


def run_self_improvement_demo(config: SelfImprovementDemoConfig | None = None) -> dict[str, Any]:
    config = config or SelfImprovementDemoConfig()
    root = (Path(config.benchmark_dir) if config.benchmark_dir else Path(tempfile.mkdtemp(prefix="jarvis_self_improve_"))).resolve()
    fixture = _create_fixture_repo(root / "fixture_repo")
    brain = MockRepositoryBrain() if config.mock_brain else make_brain_provider_from_env()
    goal = SelfImprovementGoal(
        objective=config.goal,
        success_criteria=["Semantically equivalent content-line requests resolve to the installed line-count capability."],
        allowed_paths=["resolver.py"],
        protected_paths=["test_resolver.py"],
        tests=[["python", "-m", "unittest", "test_resolver.py"]],
    )
    engineer = RepositoryEngineer(
        brain=brain,
        worktree_root=root / "worktrees",
        memory=SelfImprovementMemory(root / "self_improvement_trajectories.jsonl"),
    )
    result = engineer.improve(fixture, goal, goal.tests)
    metrics = {
        "VERSION": "self_improvement_v0.1",
        "STATUS": result.status,
        "SUCCESS": result.success,
        "WORKTREE": result.worktree,
        "TESTS_PASSED": all(item.success for item in result.tests),
        "PROTECTED_PRISTINE": result.protected_pristine,
        "DIFF": result.diff,
        "TRAJECTORY_PATH": str(engineer.memory.path),
        "CONFIG": asdict(config),
    }
    output_path = root / "self_improvement_results.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metrics["RESULT_PATH"] = str(output_path)
    if not config.quiet:
        for key in ["VERSION", "STATUS", "SUCCESS", "WORKTREE", "TESTS_PASSED", "PROTECTED_PRISTINE", "TRAJECTORY_PATH", "RESULT_PATH"]:
            print(f"{key}: {metrics[key]}", flush=True)
    return metrics


def _create_fixture_repo(root: Path) -> Path:
    if root.exists():
        import shutil

        shutil.rmtree(root, onexc=_make_writable_and_retry)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolver.py").write_text(_baseline_resolver_source(), encoding="utf-8")
    (root / "test_resolver.py").write_text(_resolver_tests_source(), encoding="utf-8")
    env = dict(os.environ)
    env["GIT_CEILING_DIRECTORIES"] = str(root.resolve().parent)
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, timeout=20, env=env)
    subprocess.run(["git", "config", "user.email", "jarvis@example.invalid"], cwd=root, capture_output=True, text=True, timeout=20, env=env)
    subprocess.run(["git", "config", "user.name", "Jarvis Fixture"], cwd=root, capture_output=True, text=True, timeout=20, env=env)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, timeout=20, env=env)
    subprocess.run(["git", "commit", "-m", "fixture baseline"], cwd=root, capture_output=True, text=True, timeout=20, env=env)
    return root


def _make_writable_and_retry(function, path, excinfo) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except Exception:
        raise excinfo[1]


def _baseline_resolver_source() -> str:
    return '''def resolve(goal, catalog):
    terms = set(goal.lower().split())
    best = None
    best_score = 0
    for capability_id, description in catalog.items():
        target = set((capability_id + " " + description).lower().replace(".", " ").split())
        score = len(terms & target)
        if score > best_score:
            best = capability_id
            best_score = score
    return best if best_score >= 2 else None
'''


def _improved_resolver_source() -> str:
    return '''def _terms(text):
    raw = {part for part in text.lower().replace(".", " ").split() if len(part) > 2}
    expanded = set(raw)
    synonyms = {"actual": {"non", "empty"}, "content": {"text", "line"}, "string": {"text"}, "many": {"count"}}
    for term in list(raw):
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        expanded.update(synonyms.get(term, set()))
    return expanded


def resolve(goal, catalog):
    terms = _terms(goal)
    best = None
    best_score = 0
    for capability_id, description in catalog.items():
        target = _terms(capability_id + " " + description)
        score = len(terms & target)
        if score > best_score:
            best = capability_id
            best_score = score
    return best if best_score >= 2 else None
'''


def _resolver_tests_source() -> str:
    return '''import unittest

from resolver import resolve


class ResolverTests(unittest.TestCase):
    def test_semantic_line_count_reuse(self):
        catalog = {"text.line_count": "Count non-empty lines in supplied text."}
        self.assertEqual(resolve("How many actual lines of content are in this string?", catalog), "text.line_count")


if __name__ == "__main__":
    unittest.main()
'''


def _parse_args() -> SelfImprovementDemoConfig:
    parser = argparse.ArgumentParser(description="Run a safe JARVIS self-improvement candidate demo.")
    parser.add_argument("--goal", default=SelfImprovementDemoConfig.goal)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--brain-provider", choices=["local_transformers", "openai_compatible"], default=None)
    parser.add_argument("--mock-brain", action="store_true", help="Use deterministic mock provider.")
    parser.add_argument("--real-brain", action="store_true", help="Use configured frozen Qwen provider.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.brain_provider:
        import os

        os.environ["JARVIS_BRAIN_PROVIDER"] = args.brain_provider
    return SelfImprovementDemoConfig(
        goal=args.goal,
        benchmark_dir=args.benchmark_dir,
        brain_provider=args.brain_provider,
        mock_brain=not args.real_brain,
        quiet=args.quiet,
    )


def main() -> None:
    run_self_improvement_demo(_parse_args())


if __name__ == "__main__":
    main()
