from __future__ import annotations

import shutil
import sys
import hashlib
import ast
import re
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
        payload = "\n".join(
            [
                CodingTaskFactory._normalized_ast(source),
                CodingTaskFactory._normalized_ast(public),
                re.sub(r"\b(train|validation|holdout|v03|variant|task|split|id)\b|\d+", "", task.description.lower()),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _normalized_ast(text: str) -> str:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return re.sub(r"#.*|[A-Za-z_][A-Za-z0-9_]*|\d+", "_", text)

        class Normalizer(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                node.name = "FUNC"
                self.generic_visit(node)
                return node

            def visit_ClassDef(self, node: ast.ClassDef):
                node.name = "CLASS"
                self.generic_visit(node)
                return node

            def visit_arg(self, node: ast.arg):
                node.arg = "ARG"
                return node

            def visit_Name(self, node: ast.Name):
                node.id = "NAME"
                return node

            def visit_Attribute(self, node: ast.Attribute):
                self.generic_visit(node)
                node.attr = "ATTR"
                return node

            def visit_alias(self, node: ast.alias):
                node.name = "MODULE"
                node.asname = None
                return node

        normalized = Normalizer().visit(tree)
        ast.fix_missing_locations(normalized)
        return ast.dump(normalized, include_attributes=False)

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
        return self._v03_distinct_catalog()[split]

    def _v03_distinct_catalog(self) -> dict[DatasetSplit, list[dict[str, str]]]:
        def fn(family: str, name: str, description: str, body: str, public: str, hidden: str) -> dict[str, str]:
            return {
                "family": family,
                "description": description,
                "source": f"def {name}(*args):\n" + "\n".join(f"    {line}" for line in body.splitlines()) + "\n",
                "public_test": f"""
                    import unittest
                    from solution import {name}

                    class PublicTests(unittest.TestCase):
                        def test_public(self):
                            {public}

                    if __name__ == '__main__':
                        unittest.main()
                """,
                "hidden_test": f"from solution import {name}\n{hidden}\n",
            }

        def raw(family: str, description: str, source: str, public_test: str, hidden_test: str) -> dict[str, str]:
            return {"family": family, "description": description, "source": source, "public_test": public_test, "hidden_test": hidden_test}

        train = [
            fn("arithmetic_add", "add_numbers", "Return the sum of two integers.", "a, b = args\nreturn a - b", "self.assertEqual(add_numbers(2, 3), 5)", "assert add_numbers(-2, 5) == 3"),
            fn("numeric_square", "square_plus_one", "Return n squared plus one.", "n = args[0]\nreturn n + 1", "self.assertEqual(square_plus_one(3), 10)", "assert square_plus_one(0) == 1"),
            fn("numeric_abs", "absolute_delta", "Return the absolute difference between two numbers.", "a, b = args\nreturn a - b", "self.assertEqual(absolute_delta(2, 7), 5)", "assert absolute_delta(9, 4) == 5"),
            fn("string_upper", "clean_upper", "Strip text and return uppercase.", "text = args[0]\nreturn text.lower()", "self.assertEqual(clean_upper(' Ada '), 'ADA')", "assert clean_upper('\\tbob ') == 'BOB'"),
            fn("string_reverse", "reverse_text", "Return text reversed.", "text = args[0]\nreturn text", "self.assertEqual(reverse_text('abc'), 'cba')", "assert reverse_text('Jarvis') == 'sivraJ'"),
            fn("normalization_slug", "slugify", "Normalize spaces to lowercase hyphenated words.", "text = args[0]\nreturn text.lower()", "self.assertEqual(slugify('Hello World'), 'hello-world')", "assert slugify(' Two  Words ') == 'two-words'"),
            fn("parsing_ints", "parse_ints", "Parse comma separated integers.", "text = args[0]\nreturn text.split(',')", "self.assertEqual(parse_ints('1,2,3'), [1, 2, 3])", "assert parse_ints('10,-2') == [10, -2]"),
            fn("parsing_pairs", "parse_pairs", "Parse key=value pairs into a dictionary.", "text = args[0]\nreturn dict(part.split(':') for part in text.split(','))", "self.assertEqual(parse_pairs('a=1,b=2'), {'a': '1', 'b': '2'})", "assert parse_pairs('x=9') == {'x': '9'}"),
            fn("list_filter", "positive_values", "Return only positive values.", "values = args[0]\nreturn [v for v in values if v < 0]", "self.assertEqual(positive_values([-1, 2, 3]), [2, 3])", "assert positive_values([-5, 0, 4]) == [4]"),
            fn("list_transform", "double_values", "Double each value in order.", "values = args[0]\nreturn [v + 2 for v in values]", "self.assertEqual(double_values([1, 3]), [2, 6])", "assert double_values([]) == []"),
            fn("aggregation_sum", "sum_even", "Sum only even integers.", "values = args[0]\ntotal = 0\nfor v in values:\n    total += v\nreturn total", "self.assertEqual(sum_even([1, 2, 4]), 6)", "assert sum_even([1, 3]) == 0"),
            fn("aggregation_mean", "mean_or_zero", "Return mean or zero for empty input.", "values = args[0]\nreturn sum(values) / len(values)", "self.assertEqual(mean_or_zero([]), 0)", "assert mean_or_zero([2, 4]) == 3"),
            fn("dict_invert", "invert_lookup", "Invert a dictionary.", "mapping = args[0]\nreturn {k: v for k, v in mapping.items()}", "self.assertEqual(invert_lookup({'a': 1}), {1: 'a'})", "assert invert_lookup({'x': 'y'}) == {'y': 'x'}"),
            fn("dict_count", "count_words", "Count words in a list.", "words = args[0]\nreturn {word: 1 for word in words}", "self.assertEqual(count_words(['a', 'a', 'b']), {'a': 2, 'b': 1})", "assert count_words([]) == {}"),
            fn("set_unique", "unique_sorted", "Return unique values sorted.", "values = args[0]\nreturn list(values)", "self.assertEqual(unique_sorted([3, 1, 3]), [1, 3])", "assert unique_sorted([]) == []"),
            fn("set_intersection", "common_items", "Return sorted common items.", "a, b = args\nreturn sorted(set(a) | set(b))", "self.assertEqual(common_items([1, 2], [2, 3]), [2])", "assert common_items([], [1]) == []"),
            fn("boundary_clamp", "clamp", "Clamp value inclusively.", "value, low, high = args\nif value < low:\n    return high\nif value > high:\n    return low\nreturn value", "self.assertEqual(clamp(-1, 0, 10), 0)", "assert clamp(11, 0, 10) == 10"),
            fn("boundary_index", "safe_get", "Return item or default when index is out of range.", "values, index, default = args\nreturn values[index]", "self.assertEqual(safe_get([1], 5, None), None)", "assert safe_get([7], 0, None) == 7"),
            fn("loop_factorial", "factorial", "Compute factorial.", "n = args[0]\nresult = 0\nfor v in range(1, n + 1):\n    result *= v\nreturn result", "self.assertEqual(factorial(4), 24)", "assert factorial(0) == 1"),
            fn("loop_countdown", "countdown", "Return descending integers to one.", "n = args[0]\nreturn list(range(1, n + 1))", "self.assertEqual(countdown(3), [3, 2, 1])", "assert countdown(0) == []"),
            fn("conditional_grade", "letter_grade", "Map numeric score to pass/fail.", "score = args[0]\nif score > 60:\n    return 'fail'\nreturn 'pass'", "self.assertEqual(letter_grade(80), 'pass')", "assert letter_grade(50) == 'fail'"),
            fn("conditional_sign", "sign_label", "Return negative, zero, or positive.", "n = args[0]\nif n >= 0:\n    return 'positive'\nreturn 'negative'", "self.assertEqual(sign_label(0), 'zero')", "assert sign_label(-1) == 'negative'"),
            fn("search_contains", "contains_casefold", "Case-insensitive containment check.", "text, needle = args\nreturn needle in text", "self.assertTrue(contains_casefold('Hello', 'he'))", "assert not contains_casefold('abc', 'z')"),
            fn("search_first", "first_index", "Return first index or -1.", "values, target = args\nreturn values.index(target)", "self.assertEqual(first_index([1, 2], 3), -1)", "assert first_index([4, 5], 5) == 1"),
            fn("sorting_numbers", "sort_desc", "Sort numbers descending.", "values = args[0]\nreturn sorted(values)", "self.assertEqual(sort_desc([1, 3, 2]), [3, 2, 1])", "assert sort_desc([]) == []"),
            fn("sorting_records", "sort_by_name", "Sort dictionaries by name.", "records = args[0]\nreturn sorted(records, key=lambda r: r['id'])", "self.assertEqual(sort_by_name([{'name':'b'}, {'name':'a'}]), [{'name':'a'}, {'name':'b'}])", "assert sort_by_name([]) == []"),
            raw("class_state", "Repair Counter increment and value behavior.", "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count -= 1\n\n    def value(self):\n        return 0\n", "import unittest\nfrom solution import Counter\n\nclass PublicTests(unittest.TestCase):\n    def test_public(self):\n        c = Counter(); c.increment(); self.assertEqual(c.value(), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n", "from solution import Counter\nc = Counter()\nfor _ in range(3): c.increment()\nassert c.value() == 3"),
            raw("class_collection", "Repair Bag add and size behavior.", "class Bag:\n    def __init__(self):\n        self.items = []\n\n    def add(self, item):\n        return None\n\n    def size(self):\n        return 0\n", "import unittest\nfrom solution import Bag\n\nclass PublicTests(unittest.TestCase):\n    def test_public(self):\n        b = Bag(); b.add('x'); self.assertEqual(b.size(), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n", "from solution import Bag\nb = Bag(); b.add('a'); b.add('b'); assert b.size() == 2"),
            fn("exception_divide", "safe_divide", "Return None for division by zero.", "a, b = args\nreturn a / b", "self.assertIsNone(safe_divide(4, 0))", "assert safe_divide(6, 3) == 2"),
            fn("nested_flatten", "flatten_once", "Flatten one level of nested lists.", "values = args[0]\nreturn values", "self.assertEqual(flatten_once([[1, 2], [3]]), [1, 2, 3])", "assert flatten_once([]) == []"),
        ]
        validation = [
            fn("validation_initials", "make_initials", "Return uppercase initials from words in a name.", "name = args[0]\nreturn ''.join(part[-1].lower() for part in name.split())", "self.assertEqual(make_initials('Ada Lovelace'), 'AL')", "assert make_initials('Grace Brewster Hopper') == 'GBH'"),
            fn("validation_even_filter", "even_values", "Return only even integers in original order.", "values = args[0]\nreturn [value for value in values if value % 2 == 1]", "self.assertEqual(even_values([1, 2, 3, 4]), [2, 4])", "assert even_values([0, -2, 5]) == [0, -2]"),
            fn("validation_minmax", "min_max", "Return the smallest and largest values as a tuple.", "values = args[0]\nreturn (values[0], values[-1])", "self.assertEqual(min_max([3, 1, 4]), (1, 4))", "assert min_max([-2]) == (-2, -2)"),
            fn("validation_parse_lines", "nonempty_lines", "Return stripped non-empty lines.", "text = args[0]\nreturn text.split('\\n')", "self.assertEqual(nonempty_lines('a\\n\\n b '), ['a', 'b'])", "assert nonempty_lines('') == []"),
            fn("validation_dict_merge", "merge_counts", "Merge count dictionaries by summing values.", "left, right = args\nmerged = dict(left)\nmerged.update(right)\nreturn merged", "self.assertEqual(merge_counts({'a': 1}, {'a': 2, 'b': 1}), {'a': 3, 'b': 1})", "assert merge_counts({}, {'x': 4}) == {'x': 4}"),
            fn("validation_prefix", "starts_with_any", "Check whether text starts with any prefix.", "text, prefixes = args\nreturn text in prefixes", "self.assertTrue(starts_with_any('jarvis', ['ja', 'co']))", "assert not starts_with_any('alpha', [])"),
            fn("validation_nested_sum", "sum_matrix", "Sum all numbers in a matrix.", "matrix = args[0]\nreturn sum(matrix)", "self.assertEqual(sum_matrix([[1, 2], [3]]), 6)", "assert sum_matrix([]) == 0"),
            fn("validation_sort_length", "sort_by_length", "Sort strings by length then alphabetically.", "words = args[0]\nreturn sorted(words)", "self.assertEqual(sort_by_length(['bbb', 'a', 'cc']), ['a', 'cc', 'bbb'])", "assert sort_by_length(['ba', 'ab']) == ['ab', 'ba']"),
            raw("validation_stack", "Repair Stack push/pop behavior.", "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, value):\n        return None\n\n    def pop(self):\n        return None\n", "import unittest\nfrom solution import Stack\n\nclass PublicTests(unittest.TestCase):\n    def test_public(self):\n        s = Stack(); s.push(3); self.assertEqual(s.pop(), 3)\n\nif __name__ == '__main__':\n    unittest.main()\n", "from solution import Stack\ns = Stack(); s.push('a'); s.push('b'); assert s.pop() == 'b'"),
            fn("validation_exception_lookup", "lookup_default", "Return default when a key is missing.", "mapping, key, default = args\nreturn mapping[key]", "self.assertEqual(lookup_default({}, 'x', 7), 7)", "assert lookup_default({'x': 1}, 'x', 0) == 1"),
        ]
        holdout = [
            fn("holdout_parse_bool", "parse_bool", "Parse yes/no strings into booleans.", "text = args[0]\nreturn bool(text)", "self.assertFalse(parse_bool('no'))", "assert parse_bool('yes') is True"),
            fn("holdout_roman", "roman_one_to_three", "Convert 1..3 to roman numerals.", "n = args[0]\nreturn str(n)", "self.assertEqual(roman_one_to_three(2), 'II')", "assert roman_one_to_three(3) == 'III'"),
            fn("holdout_palindrome", "is_palindrome", "Ignore case and spaces when checking palindrome.", "text = args[0]\nreturn text == text[::-1]", "self.assertTrue(is_palindrome('Never odd or even'))", "assert not is_palindrome('jarvis')"),
            fn("holdout_chunks", "chunk_pairs", "Group values into pairs.", "values = args[0]\nreturn values", "self.assertEqual(chunk_pairs([1, 2, 3, 4]), [[1, 2], [3, 4]])", "assert chunk_pairs([1]) == [[1]]"),
            fn("holdout_merge", "merge_sorted", "Merge two sorted lists.", "a, b = args\nreturn sorted(a) + sorted(b)", "self.assertEqual(merge_sorted([1, 3], [2]), [1, 2, 3])", "assert merge_sorted([], [1]) == [1]"),
            fn("holdout_anagram", "is_anagram", "Check whether two words are anagrams.", "a, b = args\nreturn a == b", "self.assertTrue(is_anagram('listen', 'silent'))", "assert not is_anagram('abc', 'abd')"),
            fn("holdout_window", "moving_sum", "Return sums of adjacent pairs.", "values = args[0]\nreturn values", "self.assertEqual(moving_sum([1, 2, 3]), [3, 5])", "assert moving_sum([5]) == []"),
            fn("holdout_nested_get", "nested_get", "Safely read a nested dictionary key.", "mapping, outer, inner, default = args\nreturn mapping[outer][inner]", "self.assertEqual(nested_get({}, 'a', 'b', 9), 9)", "assert nested_get({'a': {'b': 2}}, 'a', 'b', 0) == 2"),
            fn("holdout_title", "title_words", "Capitalize each word after trimming.", "text = args[0]\nreturn text.upper()", "self.assertEqual(title_words('hello world'), 'Hello World')", "assert title_words(' ada ') == 'Ada'"),
            fn("holdout_run_lengths", "run_lengths", "Return consecutive character run lengths.", "text = args[0]\nreturn []", "self.assertEqual(run_lengths('aabb'), [('a', 2), ('b', 2)])", "assert run_lengths('') == []"),
            fn("holdout_rotate", "rotate_left", "Rotate a list left by n positions.", "values, n = args\nreturn values", "self.assertEqual(rotate_left([1, 2, 3], 1), [2, 3, 1])", "assert rotate_left([], 3) == []"),
            fn("holdout_mode", "most_common", "Return most common item.", "values = args[0]\nreturn values[0]", "self.assertEqual(most_common(['b', 'a', 'a']), 'a')", "assert most_common([3, 2, 2]) == 2"),
            fn("holdout_validate", "is_valid_port", "Return whether value is a valid TCP port.", "value = args[0]\nreturn value > 0", "self.assertFalse(is_valid_port(70000))", "assert is_valid_port(443)"),
            fn("holdout_dedupe", "dedupe_keep_order", "Remove duplicates while preserving order.", "values = args[0]\nreturn list(set(values))", "self.assertEqual(dedupe_keep_order([2, 1, 2]), [2, 1])", "assert dedupe_keep_order([]) == []"),
            fn("holdout_matrix", "transpose", "Transpose a rectangular matrix.", "matrix = args[0]\nreturn matrix", "self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])", "assert transpose([]) == []"),
            raw("holdout_class_toggle", "Repair Toggle switch state behavior.", "class Toggle:\n    def __init__(self):\n        self.on = False\n\n    def flip(self):\n        self.on = False\n\n    def state(self):\n        return False\n", "import unittest\nfrom solution import Toggle\n\nclass PublicTests(unittest.TestCase):\n    def test_public(self):\n        t = Toggle(); t.flip(); self.assertTrue(t.state())\n\nif __name__ == '__main__':\n    unittest.main()\n", "from solution import Toggle\nt = Toggle(); t.flip(); t.flip(); assert t.state() is False"),
            fn("holdout_exception_int", "parse_optional_int", "Parse integer or return None.", "text = args[0]\nreturn int(text)", "self.assertIsNone(parse_optional_int('x'))", "assert parse_optional_int('7') == 7"),
            fn("holdout_algorithm_gcd", "gcd", "Compute greatest common divisor.", "a, b = args\nreturn min(a, b)", "self.assertEqual(gcd(12, 8), 4)", "assert gcd(7, 3) == 1"),
            fn("holdout_algorithm_fib", "fib", "Compute fibonacci number.", "n = args[0]\nreturn n", "self.assertEqual(fib(4), 3)", "assert fib(6) == 8"),
            fn("holdout_structure", "group_by_first", "Group words by first letter.", "words = args[0]\nreturn {}", "self.assertEqual(group_by_first(['ant', 'bat', 'ape']), {'a': ['ant', 'ape'], 'b': ['bat']})", "assert group_by_first([]) == {}"),
        ]
        return {DatasetSplit.TRAIN: train, DatasetSplit.VALIDATION: validation, DatasetSplit.HOLDOUT: holdout}

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
