from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from environments.coding.task import CodingTask
from capabilities.models import SkillSpecification


@dataclass
class StagedSkillWorkspace:
    root: Path
    spec_path: Path
    public_tests_path: Path
    protected_hashes: dict[str, str]

    def to_task(self, description: str, hidden_workspace: Path | None = None, hidden_test_command: list[str] | None = None, max_steps: int = 14) -> CodingTask:
        return CodingTask(
            description=description,
            workspace=self.root,
            test_command=["python", "-m", "unittest", "test_public.py"],
            hidden_workspace=hidden_workspace,
            hidden_test_command=hidden_test_command,
            protected_paths={"test_public.py", "skill_spec.json"},
            task_id=f"skill-{self.root.name}",
            max_steps=max_steps,
            metadata={"capability_acquisition": True},
        )

    def protected_files_pristine(self) -> bool:
        for relative, expected in self.protected_hashes.items():
            path = self.root / relative
            if not path.exists() or _file_hash(path) != expected:
                return False
        return True


class SkillWorkspaceManager:
    def __init__(self, staging_root: str | Path) -> None:
        self.staging_root = Path(staging_root)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def create(self, spec: SkillSpecification, candidate_id: str) -> StagedSkillWorkspace:
        root = self.staging_root / _safe_id(spec.capability_id) / _safe_id(candidate_id)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        spec_path = root / "skill_spec.json"
        public_tests_path = root / "test_public.py"
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        public_tests_path.write_text(_render_public_tests(spec), encoding="utf-8")
        return StagedSkillWorkspace(
            root=root,
            spec_path=spec_path,
            public_tests_path=public_tests_path,
            protected_hashes={
                "skill_spec.json": _file_hash(spec_path),
                "test_public.py": _file_hash(public_tests_path),
            },
        )


def _render_public_tests(spec: SkillSpecification) -> str:
    cases = json.dumps(spec.public_tests, indent=2, sort_keys=True)
    return f'''import unittest

import main


PUBLIC_CASES = {cases}


class PublicSkillTests(unittest.TestCase):
    def test_public_cases(self):
        for case in PUBLIC_CASES:
            payload = case.get("input", {{}})
            if case.get("raises"):
                with self.assertRaises(Exception):
                    main.run(payload)
                continue
            result = main.run(payload)
            if "expected" in case:
                self.assertEqual(result, case["expected"], case.get("name", "public case"))
            for key in case.get("expected_keys", []):
                self.assertIsInstance(result, dict)
                self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
'''


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "capability"


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
