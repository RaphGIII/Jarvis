from __future__ import annotations

import shutil
import sys
import hashlib
from textwrap import dedent
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
        (hidden_workspace / "hidden_verifier.py").write_text(
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))\n"
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

    def make_v03_split_tasks(self, split: DatasetSplit, count: int | None = None) -> list[CodingTask]:
        defaults = {DatasetSplit.TRAIN: 30, DatasetSplit.VALIDATION: 10, DatasetSplit.HOLDOUT: 20}
        target_count = count if count is not None else defaults[split]
        templates = self._v03_templates(split)
        if target_count > len(templates):
            raise ValueError(f"Not enough distinct v0.3 templates for {split.value}: requested {target_count}, have {len(templates)}")
        tasks: list[CodingTask] = []
        for index in range(target_count):
            template = templates[index]
            tasks.append(self._make_v03_task(split, index, template))
        return tasks

    @staticmethod
    def structural_fingerprint(task: CodingTask) -> str:
        source = (task.workspace / "solution.py").read_text(encoding="utf-8") if (task.workspace / "solution.py").exists() else ""
        public = (task.workspace / "test_public.py").read_text(encoding="utf-8") if (task.workspace / "test_public.py").exists() else ""
        payload = "\n".join([task.description, source, public, str(task.metadata.get("family", ""))])
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

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

    def _make_v03_task(self, split: DatasetSplit, index: int, template: dict[str, str]) -> CodingTask:
        task_id = f"v03_{split.value}_{template['family']}_{index}"
        workspace = self._fresh_workspace(task_id)
        hidden_workspace = self._fresh_workspace(f"{task_id}_hidden")
        (workspace / "solution.py").write_text(dedent(template["source"]).strip() + "\n", encoding="utf-8")
        (workspace / "test_public.py").write_text(dedent(template["public_test"]).strip() + "\n", encoding="utf-8")
        (hidden_workspace / "hidden_verifier.py").write_text(
            "import os\n"
            "import sys\n"
            "sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))\n"
            + dedent(template["hidden_test"]).strip()
            + "\n",
            encoding="utf-8",
        )
        return CodingTask(
            description=template["description"],
            workspace=workspace,
            test_command=[sys.executable, "-m", "unittest", "discover", "-v"],
            hidden_workspace=hidden_workspace,
            hidden_test_command=[sys.executable, "hidden_verifier.py"],
            protected_paths={"test_public.py"},
            task_id=task_id,
            max_steps=int(template.get("max_steps", "10")),
            metadata={"version": "v0.3", "split": split.value, "family": template["family"], "index": index},
        )

    def _v03_templates(self, split: DatasetSplit) -> list[dict[str, str]]:
        train = [
            {
                "family": "arithmetic",
                "description": "Repair solution.add_numbers so it returns the numeric sum for arbitrary integers.",
                "source": "def add_numbers(a, b):\n    return a - b\n",
                "public_test": """
                    import unittest
                    from solution import add_numbers

                    class PublicTests(unittest.TestCase):
                        def test_positive(self):
                            self.assertEqual(add_numbers(2, 3), 5)

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import add_numbers\nassert add_numbers(-2, 5) == 3\nassert add_numbers(10, 7) == 17\n",
            },
            {
                "family": "string",
                "description": "Repair solution.shout_name so it strips whitespace and returns an uppercase greeting.",
                "source": "def shout_name(name):\n    return 'hello ' + name.lower()\n",
                "public_test": """
                    import unittest
                    from solution import shout_name

                    class PublicTests(unittest.TestCase):
                        def test_name(self):
                            self.assertEqual(shout_name(' Ada '), 'HELLO ADA')

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import shout_name\nassert shout_name('bob') == 'HELLO BOB'\nassert shout_name('\\tLin ') == 'HELLO LIN'\n",
            },
            {
                "family": "list",
                "description": "Repair solution.positive_total so it sums only positive numbers from the list.",
                "source": "def positive_total(values):\n    total = 0\n    for value in values:\n        total -= value\n    return total\n",
                "public_test": """
                    import unittest
                    from solution import positive_total

                    class PublicTests(unittest.TestCase):
                        def test_mixed(self):
                            self.assertEqual(positive_total([3, -4, 5]), 8)

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import positive_total\nassert positive_total([-5, -1]) == 0\nassert positive_total([1, 2, 3]) == 6\n",
            },
            {
                "family": "dict",
                "description": "Repair solution.invert_lookup so it maps dictionary values back to their keys.",
                "source": "def invert_lookup(mapping):\n    return {key: value for key, value in mapping.items()}\n",
                "public_test": """
                    import unittest
                    from solution import invert_lookup

                    class PublicTests(unittest.TestCase):
                        def test_invert(self):
                            self.assertEqual(invert_lookup({'a': 1, 'b': 2}), {1: 'a', 2: 'b'})

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import invert_lookup\nassert invert_lookup({'x': 'y'}) == {'y': 'x'}\n",
            },
            {
                "family": "boundary",
                "description": "Repair solution.clamp so it bounds a value inclusively between low and high.",
                "source": "def clamp(value, low, high):\n    if value < low:\n        return high\n    if value > high:\n        return low\n    return value\n",
                "public_test": """
                    import unittest
                    from solution import clamp

                    class PublicTests(unittest.TestCase):
                        def test_low(self):
                            self.assertEqual(clamp(-1, 0, 10), 0)

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import clamp\nassert clamp(11, 0, 10) == 10\nassert clamp(5, 0, 10) == 5\n",
            },
            {
                "family": "loop",
                "description": "Repair solution.factorial so it computes factorial for non-negative integers.",
                "source": "def factorial(n):\n    result = 0\n    for value in range(1, n + 1):\n        result *= value\n    return result\n",
                "public_test": """
                    import unittest
                    from solution import factorial

                    class PublicTests(unittest.TestCase):
                        def test_factorial(self):
                            self.assertEqual(factorial(4), 24)

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import factorial\nassert factorial(0) == 1\nassert factorial(5) == 120\n",
            },
        ]
        validation = [
            {
                "family": "string",
                "description": "Repair solution.initials so it returns uppercase initials from words in a name.",
                "source": "def initials(name):\n    return ''.join(part[-1].lower() for part in name.split())\n",
                "public_test": """
                    import unittest
                    from solution import initials

                    class PublicTests(unittest.TestCase):
                        def test_initials(self):
                            self.assertEqual(initials('Ada Lovelace'), 'AL')

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import initials\nassert initials('Grace Brewster Hopper') == 'GBH'\n",
            },
            {
                "family": "list",
                "description": "Repair solution.evens so it returns only even integers in their original order.",
                "source": "def evens(values):\n    return [value for value in values if value % 2 == 1]\n",
                "public_test": """
                    import unittest
                    from solution import evens

                    class PublicTests(unittest.TestCase):
                        def test_evens(self):
                            self.assertEqual(evens([1, 2, 3, 4]), [2, 4])

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import evens\nassert evens([0, -2, 5]) == [0, -2]\n",
            },
        ]
        holdout = [
            {
                "family": "parsing",
                "description": "Repair solution.parse_scores so it converts comma-separated integers into a list of ints.",
                "source": "def parse_scores(text):\n    return text.split(',')\n",
                "public_test": """
                    import unittest
                    from solution import parse_scores

                    class PublicTests(unittest.TestCase):
                        def test_scores(self):
                            self.assertEqual(parse_scores('1,2,3'), [1, 2, 3])

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import parse_scores\nassert parse_scores('10,-2,0') == [10, -2, 0]\n",
            },
            {
                "family": "class_behavior",
                "description": "Repair Counter so increment changes the stored count and value returns it.",
                "source": "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count -= 1\n\n    def value(self):\n        return 0\n",
                "public_test": """
                    import unittest
                    from solution import Counter

                    class PublicTests(unittest.TestCase):
                        def test_counter(self):
                            counter = Counter()
                            counter.increment()
                            self.assertEqual(counter.value(), 1)

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import Counter\ncounter = Counter()\nfor _ in range(3):\n    counter.increment()\nassert counter.value() == 3\n",
            },
            {
                "family": "exceptions",
                "description": "Repair solution.safe_divide so division by zero returns None and valid division returns a float.",
                "source": "def safe_divide(a, b):\n    return a / b\n",
                "public_test": """
                    import unittest
                    from solution import safe_divide

                    class PublicTests(unittest.TestCase):
                        def test_zero(self):
                            self.assertIsNone(safe_divide(4, 0))

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": "from solution import safe_divide\nassert safe_divide(6, 3) == 2\nassert safe_divide(1, 0) is None\n",
            },
        ]
        if split == DatasetSplit.TRAIN:
            return self._expand_v03_templates(train, split, 30)
        if split == DatasetSplit.VALIDATION:
            return self._expand_v03_templates(validation, split, 10)
        return self._expand_v03_templates(holdout, split, 20)

    def _expand_v03_templates(self, base_templates: list[dict[str, str]], split: DatasetSplit, target: int) -> list[dict[str, str]]:
        expanded = []
        for index in range(target):
            expanded.append(self._variant_template(base_templates[index % len(base_templates)], split, index))
        return expanded

    def _variant_template(self, template: dict[str, str], split: DatasetSplit, index: int) -> dict[str, str]:
        variant = dict(template)
        token = f"{split.value}_{index}"
        replacements = {
            "add_numbers": f"add_numbers_{token}",
            "shout_name": f"shout_name_{token}",
            "positive_total": f"positive_total_{token}",
            "invert_lookup": f"invert_lookup_{token}",
            "clamp": f"clamp_{token}",
            "factorial": f"factorial_{token}",
            "initials": f"initials_{token}",
            "evens": f"evens_{token}",
            "parse_scores": f"parse_scores_{token}",
            "safe_divide": f"safe_divide_{token}",
            "Counter": f"Counter{split.value.title()}{index}",
        }
        for field in ["source", "public_test", "hidden_test", "description"]:
            text = variant[field]
            for old, new in replacements.items():
                text = text.replace(old, new)
            variant[field] = text
        variant["family"] = f"{variant['family']}_{token}"
        variant["source"] = f"# v0.3 {split.value} variant {index}\n" + variant["source"]
        variant["description"] = f"{variant['description']} Variant {index}."
        return variant
