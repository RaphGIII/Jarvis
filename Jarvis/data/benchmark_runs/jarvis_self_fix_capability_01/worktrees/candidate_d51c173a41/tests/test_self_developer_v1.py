from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal, SelfImprovementMemory
from runtime.self_developer import run_self_developer_from_args


class IterativeRepoBrain:
    def __init__(
        self,
        *,
        malformed_first_patch: bool = False,
        reviewer_block_once: bool = False,
        optional_review: bool = False,
        initial_multiplier: str = "*",
    ) -> None:
        self.malformed_first_patch = malformed_first_patch
        self.reviewer_block_once = reviewer_block_once
        self.optional_review = optional_review
        self.initial_multiplier = initial_multiplier
        self.patch_calls = 0
        self.repair_calls = 0
        self.review_calls = 0
        self.prompts: list[str] = []

    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        self.prompts.append(prompt)
        props = schema.get("properties") or {}
        if "requests" in props:
            return json.dumps(
                {
                    "requests": [
                        {"tool": "search_text", "query": "add"},
                        {"tool": "read_file", "path": "calc.py"},
                        {"tool": "inspect_tests"},
                    ],
                    "notes": "Inspect calculator implementation and tests.",
                }
            )
        if "plan" in props:
            return json.dumps({"analysis": "Fix add implementation.", "plan": "Modify calc.py and run tests.", "files_to_change": ["calc.py"]})
        if "approved" in props:
            self.review_calls += 1
            if self.reviewer_block_once and self.review_calls == 1:
                return json.dumps(
                    {
                        "approved": False,
                        "blocking_findings": ["calc.add still does not satisfy addition contract"],
                        "optional_findings": [],
                        "recommended_tests": [],
                    }
                )
            return json.dumps(
                {
                    "approved": True,
                    "blocking_findings": [],
                    "optional_findings": ["Consider property tests later."] if self.optional_review else [],
                    "recommended_tests": [],
                }
            )
        if "Repairer" in prompt:
            self.repair_calls += 1
            return json.dumps(
                {
                    "analysis": "Repair failing arithmetic behavior.",
                    "files": [{"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}],
                    "new_files": [],
                    "deleted_files": [],
                }
            )
        self.patch_calls += 1
        if self.malformed_first_patch and self.patch_calls == 1:
            return "not json"
        return json.dumps(
            {
                "analysis": "Initial implementation attempt.",
                "files": [{"path": "calc.py", "content": f"def add(a, b):\n    return a {self.initial_multiplier} b\n"}],
                "new_files": [{"path": "notes.md", "content": "candidate notes\n"}],
                "deleted_files": [],
            }
        )


class ProtectedEditBrain(IterativeRepoBrain):
    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        props = schema.get("properties") or {}
        if "requests" in props or "plan" in props or "approved" in props:
            return super().generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return json.dumps({"analysis": "bad", "files": [{"path": "test_calc.py", "content": "pass\n"}], "new_files": [], "deleted_files": []})


class RegressionBrain(IterativeRepoBrain):
    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        props = schema.get("properties") or {}
        if "requests" in props or "plan" in props or "approved" in props:
            return super().generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return json.dumps(
            {
                "analysis": "Pass targeted but break full regression.",
                "files": [
                    {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
                    {"path": "other.py", "content": "def stable():\n    return False\n"},
                ],
                "new_files": [],
                "deleted_files": [],
            }
        )


def _make_real_repo(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "other.py").write_text("def stable():\n    return True\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\nclass CalcTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    (root / "test_full.py").write_text(
        "import unittest\nfrom calc import add\nfrom other import stable\n\nclass FullTests(unittest.TestCase):\n    def test_all(self):\n        self.assertEqual(add(2, 3), 5)\n        self.assertTrue(stable())\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    (root / "bench.py").write_text("from calc import add\nprint('SCORE:', 1 if add(2, 3) == 5 else 0)\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "config", "user.email", "jarvis@example.invalid"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "config", "user.name", "Jarvis Test"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, capture_output=True, text=True, timeout=20)
    return root


def _goal() -> SelfImprovementGoal:
    return SelfImprovementGoal(
        objective="Improve calculator addition behavior.",
        success_criteria=["Targeted tests pass.", "Full regression passes.", "SCORE improves."],
        allowed_paths=["calc.py", "notes.md", "other.py"],
        protected_paths=["test_calc.py", "test_full.py", "bench.py"],
        tests=[["python", "-m", "unittest", "test_calc.py"]],
        full_tests=[["python", "-m", "unittest", "test_full.py"]],
        benchmarks=[["python", "bench.py"]],
        require_benchmark_improvement=True,
        metric_name="SCORE",
    )


def test_self_developer_real_repo_mode_repairs_and_passes_full_benchmark(tmp_path):
    repo = _make_real_repo(tmp_path / "repo")
    original = {path.name: path.read_bytes() for path in repo.glob("*.py")}
    memory = SelfImprovementMemory(tmp_path / "memory.jsonl")
    brain = IterativeRepoBrain(initial_multiplier="*")
    engineer = RepositoryEngineer(brain=brain, worktree_root=tmp_path / "worktrees", memory=memory, max_cycles=3)

    result = engineer.improve(repo, _goal())

    assert result.status == "SELF_DEVELOPMENT_CANDIDATE_READY"
    assert result.success
    assert result.tests and all(item.success for item in result.tests)
    assert result.full_tests and all(item.success for item in result.full_tests)
    assert result.benchmarks_before[0].metrics["SCORE"] == 0.0
    assert result.benchmarks_after[0].metrics["SCORE"] == 1.0
    assert brain.repair_calls == 1
    assert "calc.py" in result.changed_files
    assert Path(result.diff_path).exists()
    assert Path(result.result_path).exists()
    assert all((repo / name).read_bytes() == data for name, data in original.items())
    assert any(event["stage"] == "UNDERSTAND" for event in memory.load_all()[-1]["events"])


def test_self_developer_malformed_generation_recovers(tmp_path):
    repo = _make_real_repo(tmp_path / "repo")
    brain = IterativeRepoBrain(malformed_first_patch=True, initial_multiplier="+")
    result = RepositoryEngineer(brain=brain, worktree_root=tmp_path / "worktrees", max_cycles=1).improve(repo, _goal())

    assert result.success
    assert brain.patch_calls == 2


def test_self_developer_protected_tests_are_immutable(tmp_path):
    repo = _make_real_repo(tmp_path / "repo")
    result = RepositoryEngineer(brain=ProtectedEditBrain(), worktree_root=tmp_path / "worktrees", max_cycles=1).improve(repo, _goal())

    assert not result.success
    assert "protected" in result.error
    assert (repo / "test_calc.py").read_text(encoding="utf-8").startswith("import unittest")


def test_self_developer_rejects_full_regression_failure(tmp_path):
    repo = _make_real_repo(tmp_path / "repo")
    result = RepositoryEngineer(brain=RegressionBrain(), worktree_root=tmp_path / "worktrees", max_cycles=1).improve(repo, _goal())

    assert not result.success
    assert result.tests and all(item.success for item in result.tests)
    assert result.full_tests == [] or not all(item.success for item in result.full_tests)
    assert (repo / "other.py").read_text(encoding="utf-8") == "def stable():\n    return True\n"


def test_self_developer_reviewer_blocking_vs_optional(tmp_path):
    repo = _make_real_repo(tmp_path / "repo")
    blocked = RepositoryEngineer(
        brain=IterativeRepoBrain(reviewer_block_once=True, initial_multiplier="+"),
        worktree_root=tmp_path / "blocked",
        max_cycles=1,
    ).improve(repo, _goal())
    optional = RepositoryEngineer(
        brain=IterativeRepoBrain(optional_review=True, initial_multiplier="+"),
        worktree_root=tmp_path / "optional",
        max_cycles=1,
    ).improve(repo, _goal())

    assert not blocked.success
    assert blocked.review and blocked.review.blocking_findings
    assert optional.success
    assert optional.review and optional.review.optional_findings


def test_self_developer_cli_real_repo_mode_with_mock_provider(tmp_path, monkeypatch):
    repo = _make_real_repo(tmp_path / "repo")
    brain = IterativeRepoBrain(initial_multiplier="+")

    import runtime.self_developer as cli

    monkeypatch.setattr(cli, "make_brain_provider_from_env", lambda: brain)
    payload = run_self_developer_from_args(
        argparse.Namespace(
            repo=str(repo),
            goal="Improve calculator addition behavior.",
            success_criteria=["SCORE improves."],
            allowed_path=["calc.py", "notes.md"],
            protected_path=["test_calc.py", "test_full.py", "bench.py"],
            test_command=["python -m unittest test_calc.py"],
            full_test_command=["python -m unittest test_full.py"],
            benchmark_command=["python bench.py"],
            require_benchmark_improvement=True,
            metric_name="SCORE",
            max_cycles=1,
            timeout_seconds=30.0,
            brain_provider=None,
            benchmark_dir=str(tmp_path / "selfdev"),
        )
    )

    assert payload["STATUS"] == "SELF_DEVELOPMENT_CANDIDATE_READY"
    assert payload["SUCCESS"] is True
    assert Path(payload["SUMMARY_PATH"]).exists()
    assert Path(payload["WORKTREE"]) != repo
