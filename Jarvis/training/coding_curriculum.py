from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from environments.coding.task import CodingTask
from learning.curriculum.curriculum import CurriculumManager, TaskCandidate
from learning.curriculum.difficulty import TaskFeatures


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


@dataclass
class CodingTaskFactory:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def make_addition_bug_task(self, task_id: str, variant: int = 0) -> CodingTask:
        workspace = self._fresh_workspace(task_id)
        function_name = "add" if variant % 2 == 0 else "sum_numbers"
        module = "calculator"
        (workspace / f"{module}.py").write_text(
            f"def {function_name}(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        (workspace / f"test_{module}.py").write_text(
            "import unittest\n"
            f"from {module} import {function_name}\n\n"
            "class CalculatorTests(unittest.TestCase):\n"
            "    def test_adds_positive_numbers(self):\n"
            f"        self.assertEqual({function_name}(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        return CodingTask(
            description=f"Repair the addition bug in {module}.py so tests pass.",
            workspace=workspace,
            test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
            task_id=task_id,
            max_steps=8,
            metadata={"level": 2, "bug_type": "logic", "function": function_name},
        )

    def make_syntax_bug_task(self, task_id: str) -> CodingTask:
        workspace = self._fresh_workspace(task_id)
        (workspace / "calculator.py").write_text("def add(a, b):\n    retun a + b\n", encoding="utf-8")
        (workspace / "test_calculator.py").write_text(
            "import unittest\n"
            "from calculator import add\n\n"
            "class CalculatorTests(unittest.TestCase):\n"
            "    def test_adds(self):\n"
            "        self.assertEqual(add(1, 4), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        return CodingTask(
            description="Repair the syntax error in calculator.py so tests pass.",
            workspace=workspace,
            test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
            task_id=task_id,
            max_steps=8,
            metadata={"level": 1, "bug_type": "syntax"},
        )

    def make_hidden_addition_task(self, task_id: str, variant: int = 0, split: DatasetSplit = DatasetSplit.TRAIN) -> CodingTask:
        workspace = self._fresh_workspace(task_id)
        hidden_workspace = self._fresh_workspace(f"{task_id}_hidden")
        function_name = "combine_values" if split == DatasetSplit.HOLDOUT else "add"
        left, right = ("x", "y") if split == DatasetSplit.HOLDOUT else ("a", "b")
        (workspace / "calculator.py").write_text(
            f"def {function_name}({left}, {right}):\n    return {left} - {right}\n",
            encoding="utf-8",
        )
        (workspace / "test_public.py").write_text(
            "import unittest\n"
            f"from calculator import {function_name}\n\n"
            "class PublicTests(unittest.TestCase):\n"
            "    def test_public_addition(self):\n"
            f"        self.assertEqual({function_name}(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        workspace_literal = str(workspace).replace("\\", "\\\\")
        (hidden_workspace / "hidden_verifier.py").write_text(
            "import sys\n"
            f"sys.path.insert(0, r'{workspace_literal}')\n"
            f"from calculator import {function_name}\n"
            f"assert {function_name}(-2, 5) == 3\n"
            f"assert {function_name}(10, 7) == 17\n",
            encoding="utf-8",
        )
        return CodingTask(
            description="Repair the arithmetic implementation so public and hidden verifier tests pass.",
            workspace=workspace,
            test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
            hidden_workspace=hidden_workspace,
            hidden_test_command=[sys.executable, "hidden_verifier.py"],
            protected_paths={"test_public.py"},
            task_id=task_id,
            max_steps=8,
            metadata={"level": 2, "bug_type": "logic", "split": split.value, "variant": variant},
        )

    def make_split_tasks(self, split: DatasetSplit, count: int = 4) -> list[CodingTask]:
        tasks = []
        for index in range(count):
            if split == DatasetSplit.TRAIN and index % 3 == 2:
                tasks.append(self.make_syntax_bug_task(f"{split.value}_syntax_{index}"))
            else:
                tasks.append(self.make_hidden_addition_task(f"{split.value}_addition_{index}", variant=index, split=split))
        return tasks

    def make_curriculum_candidates(self, count: int = 4) -> list[TaskCandidate]:
        candidates = []
        for index in range(count):
            features = TaskFeatures(
                normalized_steps=min(1.0, (index + 2) / 8.0),
                normalized_tools=0.45,
                uncertainty=0.4 + index * 0.05,
                novelty=0.3 + index * 0.03,
            )
            candidates.append(
                TaskCandidate(
                    task_id=f"curriculum_addition_{index}",
                    features=features,
                    predicted_success=max(0.55, 0.82 - index * 0.05),
                    metadata={"variant": index},
                )
            )
        return candidates

    def select_task(self, curriculum: CurriculumManager, count: int = 4) -> CodingTask:
        selected = curriculum.select_next_task(self.make_curriculum_candidates(count))
        variant = int(selected.metadata.get("variant", 0)) if selected else 0
        return self.make_addition_bug_task(f"selected_{variant}", variant=variant)

    def _fresh_workspace(self, task_id: str) -> Path:
        workspace = self.root / task_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        return workspace
