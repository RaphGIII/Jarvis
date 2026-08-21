from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capabilities.models import SkillSpecification


@dataclass
class CapabilityBenchmarkTask:
    task_id: str
    goal: str
    request_payload: dict[str, Any]
    expected_output: dict[str, Any]
    specification: SkillSpecification
    hidden_tests: list[dict[str, Any]]


class CapabilityAcquisitionTaskFactory:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def make_tasks(self, count: int | None = None) -> list[CapabilityBenchmarkTask]:
        tasks = _catalog()
        if count is not None:
            if count < 1 or count > len(tasks):
                raise ValueError(f"count must be between 1 and {len(tasks)}")
            tasks = tasks[:count]
        return tasks

    def create_hidden_verifier(self, task: CapabilityBenchmarkTask) -> Path:
        verifier = self.root / "hidden" / task.task_id
        verifier.mkdir(parents=True, exist_ok=True)
        source = _hidden_verifier_source(task.hidden_tests)
        (verifier / "hidden_verifier.py").write_text(source, encoding="utf-8")
        return verifier


def _spec(
    capability_id: str,
    objective: str,
    requirements: list[str],
    public_tests: list[dict[str, Any]],
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> SkillSpecification:
    return SkillSpecification(
        capability_id=capability_id,
        objective=objective,
        functional_requirements=requirements,
        inputs=inputs or {"type": "object", "additionalProperties": True},
        outputs=outputs or {"type": "object", "additionalProperties": True},
        constraints=["Use Python standard library only.", "No network access.", "Expose run(payload: dict) -> dict."],
        permissions=[],
        acceptance_criteria=["Public tests pass.", "Hidden verifier passes.", "Original request executes through the installed skill."],
        public_tests=public_tests,
        proposed_file_structure=["main.py"],
    )


def _task(
    task_id: str,
    capability_id: str,
    objective: str,
    requirements: list[str],
    public_tests: list[dict[str, Any]],
    hidden_tests: list[dict[str, Any]],
    request_payload: dict[str, Any],
    expected_output: dict[str, Any],
) -> CapabilityBenchmarkTask:
    return CapabilityBenchmarkTask(
        task_id=task_id,
        goal=f"{objective} Use capability {capability_id}.",
        request_payload=request_payload,
        expected_output=expected_output,
        specification=_spec(capability_id, objective, requirements, public_tests),
        hidden_tests=hidden_tests,
    )


def _catalog() -> list[CapabilityBenchmarkTask]:
    return [
        _task(
            "text_line_count",
            "text.line_count",
            "Count non-empty lines in supplied text.",
            ["Split text into lines.", "Ignore empty or whitespace-only lines."],
            [{"name": "basic", "input": {"text": "a\n\n b \n"}, "expected": {"lines": 2}}],
            [{"input": {"text": ""}, "expected": {"lines": 0}}, {"input": {"text": "x\ny\nz"}, "expected": {"lines": 3}}],
            {"text": "one\n\nthree"},
            {"lines": 2},
        ),
        _task(
            "csv_mode",
            "data.csv_column_mode",
            "Read CSV text and return the most common value in a selected column.",
            ["Use the header row.", "Skip rows with missing columns.", "Break ties lexicographically."],
            [{"name": "mode", "input": {"csv_text": "name,color\nAda,red\nBob,blue\nCy,red\n", "column": "color"}, "expected": {"value": "red", "frequency": 2}}],
            [{"input": {"csv_text": "a,b\n1,x\n2,y\n3,y\n", "column": "b"}, "expected": {"value": "y", "frequency": 2}}],
            {"csv_text": "team,lang\nA,py\nB,js\nC,py\nD,go\n", "column": "lang"},
            {"value": "py", "frequency": 2},
        ),
        _task(
            "extension_summary",
            "files.extension_summary",
            "Summarize file extensions from a list of paths.",
            ["Normalize extension case.", "Use empty string for extensionless files."],
            [{"name": "extensions", "input": {"paths": ["a.py", "b.TXT", "README"]}, "expected": {"": 1, ".py": 1, ".txt": 1}}],
            [{"input": {"paths": ["x.JSON", "dir/y.json", "z"]}, "expected": {"": 1, ".json": 2}}],
            {"paths": ["src/a.py", "src/b.py", "notes.md"]},
            {".md": 1, ".py": 2},
        ),
        _task(
            "json_records_to_csv",
            "data.json_records_to_csv",
            "Convert JSON records to deterministic CSV text.",
            ["Preserve requested field order.", "Render missing values as empty strings."],
            [{"name": "csv", "input": {"records": [{"a": 1, "b": 2}, {"a": 3}], "fields": ["a", "b"]}, "expected": {"csv": "a,b\r\n1,2\r\n3,\r\n"}}],
            [{"input": {"records": [{"name": "Ada", "age": 7}], "fields": ["name", "age"]}, "expected": {"csv": "name,age\r\nAda,7\r\n"}}],
            {"records": [{"x": "A"}, {"x": "B", "y": "C"}], "fields": ["x", "y"]},
            {"csv": "x,y\r\nA,\r\nB,C\r\n"},
        ),
        _task(
            "markdown_table",
            "text.markdown_table",
            "Generate a Markdown table from headers and rows.",
            ["Include separator row.", "Stringify cell values."],
            [{"name": "table", "input": {"headers": ["A", "B"], "rows": [[1, 2]]}, "expected": {"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}}],
            [{"input": {"headers": ["Name"], "rows": [["Ada"], ["Bob"]]}, "expected": {"markdown": "| Name |\n| --- |\n| Ada |\n| Bob |"}}],
            {"headers": ["K", "V"], "rows": [["x", 1], ["y", 2]]},
            {"markdown": "| K | V |\n| --- | --- |\n| x | 1 |\n| y | 2 |"},
        ),
        _task(
            "duplicate_lines",
            "text.duplicate_lines",
            "Find duplicate non-empty lines while preserving first duplicate order.",
            ["Trim line whitespace.", "Ignore blank lines."],
            [{"name": "dupes", "input": {"text": "a\nb\na\n"}, "expected": {"duplicates": ["a"]}}],
            [{"input": {"text": " x \ny\nx\ny\nx\n"}, "expected": {"duplicates": ["x", "y"]}}],
            {"text": "red\nblue\nred\ngreen\nblue\n"},
            {"duplicates": ["red", "blue"]},
        ),
        _task(
            "filename_normalize",
            "files.normalize_names",
            "Normalize filenames into lowercase underscore names.",
            ["Replace non-alphanumeric runs with underscores.", "Preserve final extension."],
            [{"name": "names", "input": {"names": ["My File.TXT"]}, "expected": {"names": ["my_file.txt"]}}],
            [{"input": {"names": ["Report 2026.final.PDF", "no ext"]}, "expected": {"names": ["report_2026_final.pdf", "no_ext"]}}],
            {"names": ["Hello World.md", "A+B.py"]},
            {"names": ["hello_world.md", "a_b.py"]},
        ),
        _task(
            "log_level_counts",
            "logs.level_counts",
            "Parse simple log lines and count levels.",
            ["Recognize DEBUG, INFO, WARNING, ERROR.", "Return zero for missing known levels."],
            [{"name": "logs", "input": {"lines": ["INFO start", "ERROR bad", "INFO ok"]}, "expected": {"DEBUG": 0, "INFO": 2, "WARNING": 0, "ERROR": 1}}],
            [{"input": {"lines": ["WARNING w", "debug low"]}, "expected": {"DEBUG": 1, "INFO": 0, "WARNING": 1, "ERROR": 0}}],
            {"lines": ["INFO a", "INFO b", "ERROR c"]},
            {"DEBUG": 0, "INFO": 2, "WARNING": 0, "ERROR": 1},
        ),
        _task(
            "records_filter",
            "records.filter_equals",
            "Filter dictionaries where a field equals a requested value.",
            ["Preserve original record order.", "Ignore records missing the field."],
            [{"name": "filter", "input": {"records": [{"x": 1}, {"x": 2}], "field": "x", "value": 2}, "expected": {"records": [{"x": 2}]}}],
            [{"input": {"records": [{"a": "yes"}, {"b": "yes"}, {"a": "no"}], "field": "a", "value": "yes"}, "expected": {"records": [{"a": "yes"}]}}],
            {"records": [{"kind": "a", "n": 1}, {"kind": "b", "n": 2}], "field": "kind", "value": "a"},
            {"records": [{"kind": "a", "n": 1}]},
        ),
        _task(
            "kv_utility",
            "local.kv_utility",
            "Apply set/get/delete operations to a small local key/value state.",
            ["Return final store.", "Return values produced by get operations."],
            [{"name": "kv", "input": {"operations": [["set", "a", 1], ["get", "a"]]}, "expected": {"store": {"a": 1}, "results": [1]}}],
            [{"input": {"operations": [["set", "a", 1], ["delete", "a"], ["get", "a"]]}, "expected": {"store": {}, "results": [None]}}],
            {"operations": [["set", "x", 4], ["set", "y", 5], ["get", "x"]]},
            {"store": {"x": 4, "y": 5}, "results": [4]},
        ),
        _task(
            "rule_transform",
            "text.rule_transform",
            "Transform text according to a named rule.",
            ["Support upper, lower, title, and reverse.", "Reject unknown rules."],
            [{"name": "upper", "input": {"text": "Ada", "rule": "upper"}, "expected": {"text": "ADA"}}],
            [{"input": {"text": "Ada Lovelace", "rule": "reverse"}, "expected": {"text": "ecalevoL adA"}}],
            {"text": "hello world", "rule": "title"},
            {"text": "Hello World"},
        ),
        _task(
            "json_key_compare",
            "data.json_key_compare",
            "Compare top-level keys of two JSON-like dictionaries.",
            ["Return added, removed, and common sorted key lists."],
            [{"name": "compare", "input": {"left": {"a": 1, "b": 2}, "right": {"b": 3, "c": 4}}, "expected": {"added": ["c"], "removed": ["a"], "common": ["b"]}}],
            [{"input": {"left": {}, "right": {"z": 1}}, "expected": {"added": ["z"], "removed": [], "common": []}}],
            {"left": {"id": 1, "name": "a"}, "right": {"id": 2, "email": "x"}},
            {"added": ["email"], "removed": ["name"], "common": ["id"]},
        ),
        _task(
            "number_aggregate",
            "numbers.aggregate",
            "Aggregate numeric values into count, sum, mean, min, and max.",
            ["Handle an empty list with null min/max/mean."],
            [{"name": "numbers", "input": {"values": [1, 2, 3]}, "expected": {"count": 3, "sum": 6, "mean": 2.0, "min": 1, "max": 3}}],
            [{"input": {"values": []}, "expected": {"count": 0, "sum": 0, "mean": None, "min": None, "max": None}}],
            {"values": [10, -2, 4]},
            {"count": 3, "sum": 12, "mean": 4.0, "min": -2, "max": 10},
        ),
        _task(
            "parse_key_values",
            "text.parse_key_values",
            "Parse lines of key=value text into a dictionary.",
            ["Trim whitespace.", "Ignore malformed lines.", "Last value wins."],
            [{"name": "kvtext", "input": {"text": "a=1\nb = two\nbad"}, "expected": {"values": {"a": "1", "b": "two"}}}],
            [{"input": {"text": "x=1\nx=2\n =bad"}, "expected": {"values": {"x": "2"}}}],
            {"text": "name=Ada\nlang=Python"},
            {"values": {"name": "Ada", "lang": "Python"}},
        ),
        _task(
            "unique_sorted",
            "sets.unique_sorted",
            "Return unique values sorted by their string representation.",
            ["Remove duplicates.", "Keep deterministic ordering across mixed scalar values."],
            [{"name": "unique", "input": {"values": [3, 1, 3, 2]}, "expected": {"values": [1, 2, 3]}}],
            [{"input": {"values": ["b", "a", "b"]}, "expected": {"values": ["a", "b"]}}],
            {"values": ["10", "2", "10", "1"]},
            {"values": ["1", "10", "2"]},
        ),
    ]


def _hidden_verifier_source(hidden_tests: list[dict[str, Any]]) -> str:
    cases = json.dumps(hidden_tests, indent=2, sort_keys=True)
    return f'''import main

HIDDEN_CASES = {cases}

for case in HIDDEN_CASES:
    result = main.run(case.get("input", {{}}))
    assert result == case["expected"], (case.get("input"), result, case["expected"])
'''
