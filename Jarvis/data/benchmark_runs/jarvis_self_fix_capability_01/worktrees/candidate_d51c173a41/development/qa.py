from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capabilities.models import SkillSpecification
from development.memory import classify_failure
from development.software_engineer import SoftwareTestResult
from environments.coding.sandbox_backend import SandboxBackend


@dataclass(frozen=True)
class ExecutableContract:
    entrypoint: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    functional_requirements: list[str]
    edge_conditions: list[str]
    prohibited_side_effects: list[str]
    dependency_restrictions: list[str]
    permission_restrictions: list[str]
    deterministic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InternalTestCase:
    name: str
    payload: dict[str, Any]
    expected: dict[str, Any] | None = None
    raises: bool = False
    invariant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InternalTestSuite:
    path: Path
    cases: list[InternalTestCase] = field(default_factory=list)
    invariant_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "cases": [case.to_dict() for case in self.cases],
            "invariant_source": self.invariant_source,
        }


@dataclass
class ReviewFinding:
    approved: bool
    contract_violations: list[str] = field(default_factory=list)
    risk_cases: list[str] = field(default_factory=list)
    recommended_tests: list[InternalTestCase] = field(default_factory=list)
    repair_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recommended_tests"] = [case.to_dict() for case in self.recommended_tests]
        return data


def compile_contract(spec: SkillSpecification) -> ExecutableContract:
    requirements = list(spec.functional_requirements)
    criteria = list(spec.acceptance_criteria)
    edge_conditions = _edge_conditions(spec)
    return ExecutableContract(
        entrypoint="main.run(payload: dict) -> dict",
        input_schema=dict(spec.inputs),
        output_schema=dict(spec.outputs),
        functional_requirements=requirements + criteria,
        edge_conditions=edge_conditions,
        prohibited_side_effects=[
            "No network access.",
            "No subprocess execution.",
            "No writes outside the capability workspace.",
            "Do not mutate protected tests or skill_spec.json.",
        ],
        dependency_restrictions=list(spec.constraints or ["Python standard library only."]),
        permission_restrictions=list(spec.permissions),
    )


class InternalTestEngineer:
    """Builds Jarvis-owned acceptance tests from the public contract, never hidden tests."""

    def __init__(self, brain: Any | None = None, *, max_cases: int = 12) -> None:
        self.brain = brain
        self.max_cases = max_cases

    def create_suite(self, spec: SkillSpecification, contract: ExecutableContract, directory: Path) -> InternalTestSuite:
        directory.mkdir(parents=True, exist_ok=True)
        cases = self._brain_cases(spec, contract)
        cases.extend(self._deterministic_cases(spec))
        cases = _dedupe_cases(cases)[: self.max_cases]
        if len(cases) < 5:
            cases.extend(self._generic_cases(spec, cases))
        cases = _dedupe_cases(cases)[: self.max_cases]
        suite = InternalTestSuite(directory / "test_internal_qa.py", cases, self._invariant_source(spec))
        self.write_suite(suite)
        return suite

    def append_review_tests(self, suite: InternalTestSuite, tests: list[InternalTestCase]) -> InternalTestSuite:
        if not tests:
            return suite
        suite.cases = _dedupe_cases([*suite.cases, *tests])[: self.max_cases]
        self.write_suite(suite)
        return suite

    def write_suite(self, suite: InternalTestSuite) -> None:
        source = _render_internal_test_source(suite.cases, suite.invariant_source)
        suite.path.write_text(source, encoding="utf-8")

    def _brain_cases(self, spec: SkillSpecification, contract: ExecutableContract) -> list[InternalTestCase]:
        if self.brain is None or not hasattr(self.brain, "generate_structured"):
            return []
        prompt = (
            "Return JSON only. You are Jarvis TestEngineer. Create internal black-box tests from the contract.\n"
            "Do not use hidden verifier tests. Do not include implementation code.\n"
            "Schema: {\"cases\":[{\"name\":\"...\",\"input\":{},\"expected\":{},\"raises\":false}]}\n"
            f"Specification:\n{json.dumps(spec.to_dict(), indent=2, sort_keys=True)}\n"
            f"Executable contract:\n{json.dumps(contract.to_dict(), indent=2, sort_keys=True)}"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "name": {"type": "string"},
                            "input": {"type": "object"},
                            "expected": {"type": "object"},
                            "raises": {"type": "boolean"},
                        },
                        "required": ["name", "input"],
                    },
                }
            },
            "required": ["cases"],
        }
        try:
            raw = self.brain.generate_structured(prompt, schema, max_tokens=1600, temperature=0.2, top_p=0.9)
            data = json.loads(_extract_json(raw))
        except Exception:
            return []
        cases: list[InternalTestCase] = []
        for item in data.get("cases", []):
            if not isinstance(item, dict):
                continue
            payload = item.get("input")
            expected = item.get("expected")
            if isinstance(payload, dict) and (isinstance(expected, dict) or item.get("raises")):
                cases.append(
                    InternalTestCase(
                        name=str(item.get("name") or "brain_case"),
                        payload=payload,
                        expected=expected if isinstance(expected, dict) else None,
                        raises=bool(item.get("raises", False)),
                    )
                )
        return cases

    def _deterministic_cases(self, spec: SkillSpecification) -> list[InternalTestCase]:
        cases = [_case_from_public(item) for item in spec.public_tests]
        capability_id = spec.capability_id
        if capability_id == "text.line_count":
            cases.extend(
                [
                    InternalTestCase("empty_text", {"text": ""}, {"lines": 0}),
                    InternalTestCase("whitespace_only", {"text": " \n\t\n"}, {"lines": 0}),
                    InternalTestCase("blank_line_invariant", {"text": "a\n\nb\n"}, {"lines": 2}),
                    InternalTestCase("trailing_newline", {"text": "a\n"}, {"lines": 1}),
                ]
            )
        elif capability_id == "data.csv_column_mode":
            cases.extend(
                [
                    InternalTestCase("lexicographic_tie", {"csv_text": "c\nb\na\n", "column": "c"}, {"value": "a", "frequency": 1}),
                    InternalTestCase("missing_column", {"csv_text": "a,b\n1,2\n", "column": "z"}, {"value": None, "frequency": 0}),
                    InternalTestCase("empty_csv", {"csv_text": "", "column": "x"}, {"value": None, "frequency": 0}),
                ]
            )
        elif capability_id == "files.extension_summary":
            cases.extend(
                [
                    InternalTestCase("case_normalization", {"paths": ["A.PY", "b.py", "README"]}, {"": 1, ".py": 2}),
                    InternalTestCase("empty_paths", {"paths": []}, {}),
                    InternalTestCase("nested_names", {"paths": ["dir/archive.tar.gz", ".env"]}, {"": 1, ".gz": 1}),
                ]
            )
        elif capability_id == "data.json_records_to_csv":
            cases.extend(
                [
                    InternalTestCase("empty_records", {"records": [], "fields": ["a"]}, {"csv": "a\r\n"}),
                    InternalTestCase("missing_field", {"records": [{"a": 1}], "fields": ["a", "b"]}, {"csv": "a,b\r\n1,\r\n"}),
                ]
            )
        elif capability_id == "text.markdown_table":
            cases.extend(
                [
                    InternalTestCase("empty_rows", {"headers": ["A"], "rows": []}, {"markdown": "| A |\n| --- |"}),
                    InternalTestCase("stringify_cells", {"headers": ["A"], "rows": [[None]]}, {"markdown": "| A |\n| --- |\n| None |"}),
                ]
            )
        elif capability_id == "text.duplicate_lines":
            cases.extend(
                [
                    InternalTestCase("ignore_blank", {"text": "a\n\n a \na\n"}, {"duplicates": ["a"]}),
                    InternalTestCase("no_duplicates", {"text": "a\nb\n"}, {"duplicates": []}),
                ]
            )
        elif capability_id == "files.normalize_names":
            cases.extend(
                [
                    InternalTestCase("punctuation", {"names": ["A+B.py", "no ext"]}, {"names": ["a_b.py", "no_ext"]}),
                    InternalTestCase("empty_names", {"names": []}, {"names": []}),
                ]
            )
        elif capability_id == "logs.level_counts":
            cases.extend(
                [
                    InternalTestCase("case_insensitive", {"lines": ["debug x", "WARNING y"]}, {"DEBUG": 1, "INFO": 0, "WARNING": 1, "ERROR": 0}),
                    InternalTestCase("unknown_ignored", {"lines": ["TRACE t"]}, {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0}),
                ]
            )
        elif capability_id == "records.filter_equals":
            cases.extend(
                [
                    InternalTestCase("missing_field", {"records": [{"a": 1}, {"b": 1}], "field": "a", "value": 1}, {"records": [{"a": 1}]}),
                    InternalTestCase("empty_records", {"records": [], "field": "x", "value": 1}, {"records": []}),
                ]
            )
        elif capability_id == "local.kv_utility":
            cases.extend(
                [
                    InternalTestCase("delete_missing", {"operations": [["delete", "x"], ["get", "x"]]}, {"store": {}, "results": [None]}),
                    InternalTestCase("last_set_wins", {"operations": [["set", "x", 1], ["set", "x", 2], ["get", "x"]]}, {"store": {"x": 2}, "results": [2]}),
                ]
            )
        elif capability_id == "text.rule_transform":
            cases.extend(
                [
                    InternalTestCase("lower", {"text": "Ada", "rule": "lower"}, {"text": "ada"}),
                    InternalTestCase("reverse", {"text": "ab", "rule": "reverse"}, {"text": "ba"}),
                    InternalTestCase("unknown_rule", {"text": "ab", "rule": "unknown"}, raises=True),
                ]
            )
        elif capability_id == "data.json_key_compare":
            cases.extend(
                [
                    InternalTestCase("empty_left", {"left": {}, "right": {"z": 1}}, {"added": ["z"], "removed": [], "common": []}),
                    InternalTestCase("sorted_keys", {"left": {"b": 1, "a": 1}, "right": {"c": 1, "a": 2}}, {"added": ["c"], "removed": ["b"], "common": ["a"]}),
                ]
            )
        elif capability_id == "numbers.aggregate":
            cases.extend(
                [
                    InternalTestCase("empty_numbers", {"values": []}, {"count": 0, "sum": 0, "mean": None, "min": None, "max": None}),
                    InternalTestCase("negative_numbers", {"values": [-1, 1]}, {"count": 2, "sum": 0, "mean": 0.0, "min": -1, "max": 1}),
                ]
            )
        elif capability_id == "text.parse_key_values":
            cases.extend(
                [
                    InternalTestCase("last_value_wins", {"text": "x=1\nx=2\nbad"}, {"values": {"x": "2"}}),
                    InternalTestCase("ignore_empty_key", {"text": " =bad\na = ok"}, {"values": {"a": "ok"}}),
                ]
            )
        elif capability_id == "sets.unique_sorted":
            cases.extend(
                [
                    InternalTestCase("string_sort", {"values": ["10", "2", "1"]}, {"values": ["1", "10", "2"]}),
                    InternalTestCase("empty_values", {"values": []}, {"values": []}),
                ]
            )
        return [case for case in cases if case is not None]

    def _generic_cases(self, spec: SkillSpecification, existing: list[InternalTestCase]) -> list[InternalTestCase]:
        cases: list[InternalTestCase] = []
        for item in spec.public_tests:
            payload = dict(item.get("input") or {})
            expected = item.get("expected")
            if not isinstance(expected, dict):
                continue
            for key, value in list(payload.items())[:3]:
                variant = dict(payload)
                if isinstance(value, str):
                    variant[key] = ""
                elif isinstance(value, list):
                    variant[key] = []
                elif isinstance(value, dict):
                    variant[key] = {}
                else:
                    continue
                cases.append(InternalTestCase(f"generic_empty_{key}", variant, expected=None, invariant="returns_dict"))
        while len(existing) + len(cases) < 5:
            cases.append(InternalTestCase(f"deterministic_smoke_{len(cases)}", {}, expected=None, invariant="returns_dict"))
        return cases

    @staticmethod
    def _invariant_source(spec: SkillSpecification) -> str:
        if spec.capability_id == "text.line_count":
            return """
    def test_line_count_metamorphic_blank_line(self):
        base = main.run({"text": "a\\nb"})
        with_blank = main.run({"text": "a\\n\\nb"})
        self.assertEqual(base, with_blank)
"""
        if spec.capability_id == "sets.unique_sorted":
            return """
    def test_unique_sorted_duplicate_invariant(self):
        base = main.run({"values": ["b", "a"]})
        duplicated = main.run({"values": ["b", "a", "b"]})
        self.assertEqual(base, duplicated)
"""
        if spec.capability_id == "data.csv_column_mode":
            return """
    def test_csv_mode_frequency_matches_occurrences(self):
        result = main.run({"csv_text": "x\\na\\na\\nb\\n", "column": "x"})
        self.assertEqual(result, {"value": "a", "frequency": 2})
"""
        return ""


class DeterministicReviewer:
    def review(
        self,
        *,
        spec: SkillSpecification,
        contract: ExecutableContract,
        implementation_root: Path,
        public_result: SoftwareTestResult,
        internal_result: SoftwareTestResult,
        internal_suite: InternalTestSuite,
    ) -> ReviewFinding:
        violations: list[str] = []
        risks: list[str] = []
        main_path = implementation_root / "main.py"
        if not public_result.success:
            violations.append("public tests failed")
        if not internal_result.success:
            violations.append("internal QA tests failed")
        if not main_path.exists():
            violations.append("main.py missing")
        else:
            source = main_path.read_text(encoding="utf-8")
            violations.extend(_static_contract_violations(source, contract))
            risks.extend(_static_risks(source, spec))
        approved = not violations and not risks
        return ReviewFinding(approved=approved, contract_violations=violations, risk_cases=risks, repair_required=not approved)


def run_internal_tests(backend: SandboxBackend, workspace: Path, suite: InternalTestSuite, *, timeout_seconds: float = 20.0) -> SoftwareTestResult:
    completed = backend.run(
        ["python", suite.path.name],
        cwd=workspace,
        timeout_seconds=timeout_seconds,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        verifier_workspace=suite.path.parent,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    return SoftwareTestResult(
        success=completed.returncode == 0,
        stdout=completed.stdout[-6000:],
        stderr=completed.stderr[-6000:],
        return_code=completed.returncode,
        failure_classes=classify_failure(output),
    )


def _render_internal_test_source(cases: list[InternalTestCase], invariant_source: str) -> str:
    serializable = [case.to_dict() for case in cases]
    return f'''import unittest

import main


INTERNAL_CASES = {repr(serializable)}


class InternalQATests(unittest.TestCase):
    def test_contract_cases(self):
        for case in INTERNAL_CASES:
            payload = case.get("payload", {{}})
            if case.get("raises"):
                with self.assertRaises(Exception, msg=case.get("name", "raises")):
                    main.run(payload)
                continue
            result = main.run(payload)
            self.assertIsInstance(result, dict, case.get("name", "returns dict"))
            if case.get("expected") is not None:
                self.assertEqual(result, case["expected"], case.get("name", "internal case"))
{invariant_source}


if __name__ == "__main__":
    unittest.main()
'''


def _case_from_public(item: dict[str, Any]) -> InternalTestCase | None:
    payload = item.get("input")
    expected = item.get("expected")
    if not isinstance(payload, dict):
        return None
    if item.get("raises"):
        return InternalTestCase(str(item.get("name") or "public_raises"), payload, raises=True)
    if isinstance(expected, dict):
        return InternalTestCase(str(item.get("name") or "public_case"), payload, expected)
    return InternalTestCase(str(item.get("name") or "public_keys"), payload, invariant="returns_dict")


def _dedupe_cases(cases: list[InternalTestCase | None]) -> list[InternalTestCase]:
    deduped: list[InternalTestCase] = []
    seen: set[str] = set()
    for case in cases:
        if case is None:
            continue
        key = json.dumps(case.to_dict(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def _edge_conditions(spec: SkillSpecification) -> list[str]:
    text = " ".join([spec.objective, *spec.functional_requirements, *spec.acceptance_criteria]).lower()
    conditions = []
    markers = {
        "empty inputs": ("empty", "blank", "missing"),
        "duplicates": ("duplicate", "unique"),
        "ordering/ties": ("order", "sorted", "tie", "deterministic"),
        "malformed inputs": ("malformed", "invalid", "reject"),
        "case normalization": ("case", "lower", "upper"),
    }
    for label, words in markers.items():
        if any(word in text for word in words):
            conditions.append(label)
    return conditions or ["ordinary cases", "boundary cases", "deterministic repeated execution"]


def _static_contract_violations(source: str, contract: ExecutableContract) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]
    has_run = any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body)
    if not has_run:
        violations.append("entrypoint run(payload) missing")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in {"socket", "subprocess", "requests", "urllib"}:
                    violations.append(f"prohibited dependency import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in {"socket", "subprocess", "requests", "urllib"}:
                violations.append(f"prohibited dependency import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            violations.append(f"prohibited dynamic execution: {node.func.id}")
    return sorted(set(violations))


def _static_risks(source: str, spec: SkillSpecification) -> list[str]:
    risks = []
    lowered = source.lower()
    requirements = " ".join(spec.functional_requirements).lower()
    if "ignore empty" in requirements and "strip" not in lowered:
        risks.append("requirements mention empty/whitespace handling but implementation does not obviously strip values")
    if "break ties" in requirements and "sorted" not in lowered:
        risks.append("requirements mention tie handling but implementation does not obviously sort")
    if "reject unknown" in requirements and "raise" not in lowered:
        risks.append("requirements mention rejection behavior but implementation does not obviously raise")
    return risks


def _extract_json(text: str) -> str:
    import re

    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text
