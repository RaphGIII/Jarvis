from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from brain.providers import make_brain_provider_from_env
from capabilities.models import SkillSpecification
from environments.coding.sandbox_backend import DockerSandboxBackend, LocalTestSandboxBackend, SandboxBackend
from runtime.capability_runtime import CapabilityAcquisitionRuntime, CapabilityRuntimeConfig
from training.capability_curriculum import CapabilityAcquisitionTaskFactory, CapabilityBenchmarkTask


@dataclass
class CapabilityAcquisitionV04Config:
    mock_brain: bool = False
    brain_provider: str | None = None
    task_count: int = 3
    persistent: bool = False
    benchmark_dir: str | None = None
    quiet: bool = False
    local_test_backend: bool = False


class MockCapabilityBrain:
    """Test/demo stand-in for frozen Qwen. It emits specs and complete file bundles."""

    provider_name = "mock_capability"
    model_name = "MockCapabilityBrain"

    def __init__(
        self,
        tasks: list[CapabilityBenchmarkTask],
        *,
        fail_first: bool = True,
        malformed_first: bool = False,
        multi_file: bool = False,
        repair_failures: int = 0,
    ) -> None:
        self.tasks = tasks
        self.by_capability = {task.specification.capability_id: task for task in tasks}
        self.by_goal = {task.goal: task for task in tasks}
        self.fail_first = fail_first
        self.malformed_first = malformed_first
        self.multi_file = multi_file
        self.repair_failures = repair_failures
        self.review_calls = 0
        self.test_engineer_calls = 0
        self.implementation_calls = 0
        self.repair_calls = 0
        self.model = None
        self.last_metadata: dict[str, Any] = {}

    def capabilities(self) -> dict[str, Any]:
        return {"chat": True, "coding": True, "structured_generation": True, "local": True}

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int = 700,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        self.last_metadata = {"generated_tokens": 1, "total_tokens": 1, "finish_reason": "stop", "attempts": 1}
        properties = schema.get("properties") or {}
        if "cases" in properties:
            self.test_engineer_calls += 1
            return json.dumps({"cases": []})
        if "approved" in properties:
            self.review_calls += 1
            return json.dumps(
                {
                    "approved": True,
                    "contract_violations": [],
                    "risk_cases": [],
                    "recommended_tests": [],
                    "repair_required": False,
                }
            )
        if "files" in properties:
            is_repair = "diagnosis" in properties or "Repair the current project" in prompt
            return self._file_bundle(prompt, repair=is_repair)
        task = self._task_from_prompt(prompt)
        return json.dumps(task.specification.to_dict())

    def generate_coding(self, prompt: str, *, max_tokens: int = 450, temperature: float = 0.6, top_p: float = 0.9) -> str:
        return self._file_bundle(prompt, repair="Repair the current project" in prompt)

    def generate(self, prompt: str, *, max_tokens: int = 700, temperature: float = 0.2, top_p: float | None = None) -> str:
        if "Decide if one installed capability" in prompt:
            return json.dumps({"status": "missing", "capability_id": "", "reason": "No installed capability in mock catalog.", "confidence": 0.0})
        return json.dumps(self._task_from_prompt(prompt).specification.to_dict())

    def think(self, user_prompt: str, max_tokens: int = 700) -> str:
        return self.generate(user_prompt, max_tokens=max_tokens)

    def _file_bundle(self, prompt: str, *, repair: bool) -> str:
        if self.malformed_first and self.implementation_calls == 0 and not repair:
            self.implementation_calls += 1
            return "not json"
        capability_id = _extract_capability_id(prompt)
        if repair:
            self.repair_calls += 1
        else:
            self.implementation_calls += 1
        broken = (self.fail_first and not repair) or (repair and self.repair_calls <= self.repair_failures)
        files = implementation_files_for_capability(capability_id, broken=broken, multi_file=self.multi_file)
        payload = {
            "summary": "Create a complete local skill implementation.",
            "plan": "Implement run(payload) using deterministic standard-library code.",
            "files": files,
        }
        if repair:
            payload["diagnosis"] = "Public tests failed; replace implementation with schema-correct behavior."
        return json.dumps(payload)

    def _task_from_prompt(self, prompt: str) -> CapabilityBenchmarkTask:
        for task in self.tasks:
            if task.specification.capability_id in prompt or task.goal in prompt or task.second_goal in prompt:
                return task
        return self.tasks[0]


def run_capability_acquisition_v04_demo(config: CapabilityAcquisitionV04Config | None = None) -> dict[str, Any]:
    config = config or CapabilityAcquisitionV04Config()
    if config.benchmark_dir:
        root = Path(config.benchmark_dir)
    elif config.persistent:
        root = Path("data") / "capability_acquisition_v04"
    else:
        root = Path(tempfile.mkdtemp(prefix="jarvis_capability_v04_"))
    factory = CapabilityAcquisitionTaskFactory(root / "tasks")
    tasks = factory.make_tasks(config.task_count)
    brain = MockCapabilityBrain(tasks) if config.mock_brain else make_brain_provider_from_env()
    backend = _backend(config)
    if isinstance(backend, DockerSandboxBackend) and not DockerSandboxBackend.is_available():
        return {"SANDBOX_AVAILABLE": False, "MESSAGE": "Docker sandbox unavailable; unsafe host fallback is disabled."}
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=backend,
        config=CapabilityRuntimeConfig(
            data_dir=str(root / "runtime"),
            skills_root=str(root / "skills"),
            use_docker=not config.local_test_backend,
            trace=not config.quiet,
        ),
    )

    results = []
    started = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        hidden_workspace = factory.create_hidden_verifier(task)
        _log(config, f"[GOAL] {index}/{len(tasks)} {task.goal}")
        _log(config, "[CAPABILITY GAP] resolving registry match")
        result = runtime.handle_goal(
            task.goal,
            request_payload=task.request_payload,
            expected_output=task.expected_output,
            second_goal=task.second_goal,
            hidden_workspace=hidden_workspace,
        )
        _log(config, f"[SPEC] {task.specification.capability_id}")
        _log(config, f"[PLAN] state={result.development_state}")
        _log(config, f"[IMPLEMENT] llm_calls={result.llm_calls}")
        _log(config, f"[BUILD] cycles={result.repair_iterations} invalid={result.invalid_action_rate:.3f}")
        _log(config, f"[TEST] public={result.public_success}")
        _log(config, f"[INTERNAL QA] pass={result.internal_verification_success}")
        _log(config, f"[REVIEW] approved={result.reviewer_approved}")
        if result.repair_iterations:
            _log(config, f"[REPAIR {result.repair_iterations}/{runtime.config.max_repair_cycles}]")
        _log(config, f"[VERIFY] hidden={result.hidden_success} blind_repair={result.blind_repair_success}")
        _log(config, f"[PROMOTE] promoted={result.promoted}")
        _log(config, f"[EXECUTE] success={result.execution_success}")
        _log(config, f"[SECOND CALL] success={result.second_call_success}")
        results.append(result.to_dict())

    successes = [1.0 if item["success"] else 0.0 for item in results]
    first_attempts = [1.0 if item.get("initial_implementation_pass") else 0.0 for item in results]
    repair_successes = [1.0 if item["success"] and int(item["repair_iterations"]) > 0 else 0.0 for item in results if int(item["repair_iterations"]) > 0]
    total_tokens = [float((item.get("token_usage") or {}).get("total_tokens", 0)) for item in results]
    metrics = {
        "VERSION": "v0.4",
        "TASK_COUNT": len(tasks),
        "CAPABILITY_ACQUISITION_SUCCESS_RATE": mean(successes) if successes else 0.0,
        "INITIAL_IMPLEMENTATION_PASS_RATE": mean(first_attempts) if first_attempts else 0.0,
        "INTERNAL_QA_PASS_RATE": mean([1.0 if item.get("internal_verification_success") else 0.0 for item in results]) if results else 0.0,
        "REVIEW_PASS_RATE": mean([1.0 if item.get("reviewer_approved") else 0.0 for item in results]) if results else 0.0,
        "HIDDEN_PASS_RATE": mean([1.0 if item.get("hidden_success") else 0.0 for item in results]) if results else 0.0,
        "BLIND_REPAIR_SUCCESS_RATE": mean([1.0 if item.get("blind_repair_success") else 0.0 for item in results]) if results else 0.0,
        "REPAIR_SUCCESS_RATE": mean(repair_successes) if repair_successes else 0.0,
        "MEAN_REPAIR_CYCLES": mean([float(item["repair_iterations"]) for item in results]) if results else 0.0,
        "MEAN_STEPS_TO_ACQUISITION": mean([float(item["steps_to_acquisition"]) for item in results]) if results else 0.0,
        "MEAN_LLM_CALLS_PER_CAPABILITY": mean([float(item["llm_calls"]) for item in results]) if results else 0.0,
        "MEAN_TOKENS_PER_CAPABILITY": mean(total_tokens) if total_tokens else 0.0,
        "PROMOTION_FAILURES": sum(1 for item in results if not item["promoted"]),
        "SECOND_CALL_DIRECT_USE_SUCCESS_RATE": mean([1.0 if item["second_call_success"] else 0.0 for item in results]) if results else 0.0,
        "INVALID_ACTION_RATE": mean([float(item["invalid_action_rate"]) for item in results]) if results else 0.0,
        "DEVELOPMENT_WALL_TIME_SECONDS": time.perf_counter() - started,
        "REGISTRY_PATH": str(runtime.registry.path),
        "TRAJECTORY_PATH": str(runtime.trajectory_store.path),
        "RESULTS": results,
        "CONFIG": asdict(config),
    }
    output_path = root / "capability_acquisition_v04_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metrics["RESULT_PATH"] = str(output_path)
    return metrics


def implementation_files_for_capability(capability_id: str, *, broken: bool = False, multi_file: bool = False) -> list[dict[str, str]]:
    if broken:
        return [{"path": "main.py", "content": "def run(payload):\n    return {'result': None}\n"}]
    if multi_file:
        return [
            {"path": "main.py", "content": "from helper import run_impl\n\n\ndef run(payload):\n    return run_impl(payload)\n"},
            {"path": "helper.py", "content": implementation_for_capability(capability_id).replace("def run(payload: dict) -> dict:", "def run_impl(payload: dict) -> dict:")},
        ]
    return [{"path": "main.py", "content": implementation_for_capability(capability_id)}]


def implementation_for_capability(capability_id: str) -> str:
    return _UNIVERSAL_IMPLEMENTATION.replace("__CAPABILITY_ID__", capability_id)


_UNIVERSAL_IMPLEMENTATION = r'''import csv
import io
import json
import re
from collections import Counter
from pathlib import PurePath

CAPABILITY_ID = "__CAPABILITY_ID__"


def run(payload: dict) -> dict:
    if CAPABILITY_ID == "text.line_count":
        return {"lines": sum(1 for line in str(payload.get("text", "")).splitlines() if line.strip())}
    if CAPABILITY_ID == "data.csv_column_mode":
        reader = csv.DictReader(io.StringIO(str(payload.get("csv_text", ""))))
        column = payload.get("column")
        values = [row[column] for row in reader if column in row and row.get(column) not in (None, "")]
        counts = Counter(values)
        if not counts:
            return {"value": None, "frequency": 0}
        value, frequency = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0]
        return {"value": value, "frequency": frequency}
    if CAPABILITY_ID == "files.extension_summary":
        counts = Counter(PurePath(str(path)).suffix.lower() for path in payload.get("paths", []))
        return dict(sorted(counts.items()))
    if CAPABILITY_ID == "data.json_records_to_csv":
        output = io.StringIO()
        fields = list(payload.get("fields", []))
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        for record in payload.get("records", []):
            writer.writerow({field: record.get(field, "") for field in fields})
        return {"csv": output.getvalue()}
    if CAPABILITY_ID == "text.markdown_table":
        headers = [str(item) for item in payload.get("headers", [])]
        rows = payload.get("rows", [])
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(item) for item in row) + " |")
        return {"markdown": "\n".join(lines)}
    if CAPABILITY_ID == "text.duplicate_lines":
        seen = set()
        duplicates = []
        for line in str(payload.get("text", "")).splitlines():
            value = line.strip()
            if not value:
                continue
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        return {"duplicates": duplicates}
    if CAPABILITY_ID == "files.normalize_names":
        names = []
        for name in payload.get("names", []):
            text = str(name).strip().lower()
            if "." in text:
                stem, ext = text.rsplit(".", 1)
                normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") + "." + re.sub(r"[^a-z0-9]+", "", ext)
            else:
                normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
            names.append(normalized)
        return {"names": names}
    if CAPABILITY_ID == "logs.level_counts":
        counts = {level: 0 for level in ["DEBUG", "INFO", "WARNING", "ERROR"]}
        for line in payload.get("lines", []):
            match = re.match(r"\s*(debug|info|warning|error)\b", str(line), flags=re.I)
            if match:
                counts[match.group(1).upper()] += 1
        return counts
    if CAPABILITY_ID == "records.filter_equals":
        field = payload.get("field")
        value = payload.get("value")
        return {"records": [record for record in payload.get("records", []) if record.get(field) == value]}
    if CAPABILITY_ID == "local.kv_utility":
        store = {}
        results = []
        for operation in payload.get("operations", []):
            op = operation[0]
            key = operation[1] if len(operation) > 1 else None
            if op == "set":
                store[key] = operation[2] if len(operation) > 2 else None
            elif op == "get":
                results.append(store.get(key))
            elif op == "delete":
                store.pop(key, None)
        return {"store": store, "results": results}
    if CAPABILITY_ID == "text.rule_transform":
        text = str(payload.get("text", ""))
        rule = str(payload.get("rule", "")).lower()
        if rule == "upper":
            return {"text": text.upper()}
        if rule == "lower":
            return {"text": text.lower()}
        if rule == "title":
            return {"text": text.title()}
        if rule == "reverse":
            return {"text": text[::-1]}
        raise ValueError("unknown rule")
    if CAPABILITY_ID == "data.json_key_compare":
        left = set((payload.get("left") or {}).keys())
        right = set((payload.get("right") or {}).keys())
        return {"added": sorted(right - left), "removed": sorted(left - right), "common": sorted(left & right)}
    if CAPABILITY_ID == "numbers.aggregate":
        values = list(payload.get("values", []))
        if not values:
            return {"count": 0, "sum": 0, "mean": None, "min": None, "max": None}
        total = sum(values)
        return {"count": len(values), "sum": total, "mean": total / len(values), "min": min(values), "max": max(values)}
    if CAPABILITY_ID == "text.parse_key_values":
        values = {}
        for line in str(payload.get("text", "")).splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                values[key] = value.strip()
        return {"values": values}
    if CAPABILITY_ID == "sets.unique_sorted":
        return {"values": sorted(set(payload.get("values", [])), key=lambda item: str(item))}
    raise ValueError(f"Unsupported capability: {CAPABILITY_ID}")
'''


def _extract_capability_id(prompt: str) -> str:
    match = re.search(r'"capability_id"\s*:\s*"([^"]+)"', prompt)
    return match.group(1) if match else "local.utility"


def _backend(config: CapabilityAcquisitionV04Config) -> SandboxBackend:
    if config.local_test_backend:
        return LocalTestSandboxBackend()
    return DockerSandboxBackend()


def _log(config: CapabilityAcquisitionV04Config, message: str) -> None:
    if not config.quiet:
        print(message, flush=True)


def _parse_args() -> CapabilityAcquisitionV04Config:
    parser = argparse.ArgumentParser(description="Run JARVIS v0.4 Capability Acquisition MVP benchmark.")
    parser.add_argument("--mock-brain", action="store_true", help="Use deterministic mock Qwen provider.")
    parser.add_argument("--brain-provider", choices=["local_transformers", "openai_compatible"], default=None)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--local-test-backend", action="store_true", help="Unit-test-only backend; production uses Docker.")
    args = parser.parse_args()
    if args.brain_provider:
        import os

        os.environ["JARVIS_BRAIN_PROVIDER"] = args.brain_provider
    return CapabilityAcquisitionV04Config(
        mock_brain=args.mock_brain,
        brain_provider=args.brain_provider,
        task_count=args.task_count,
        persistent=args.persistent,
        benchmark_dir=args.benchmark_dir,
        quiet=args.quiet,
        local_test_backend=args.local_test_backend,
    )


def main() -> None:
    metrics = run_capability_acquisition_v04_demo(_parse_args())
    for key in [
        "VERSION",
        "TASK_COUNT",
        "CAPABILITY_ACQUISITION_SUCCESS_RATE",
        "INITIAL_IMPLEMENTATION_PASS_RATE",
        "INTERNAL_QA_PASS_RATE",
        "REVIEW_PASS_RATE",
        "HIDDEN_PASS_RATE",
        "BLIND_REPAIR_SUCCESS_RATE",
        "REPAIR_SUCCESS_RATE",
        "MEAN_REPAIR_CYCLES",
        "MEAN_STEPS_TO_ACQUISITION",
        "MEAN_LLM_CALLS_PER_CAPABILITY",
        "MEAN_TOKENS_PER_CAPABILITY",
        "PROMOTION_FAILURES",
        "SECOND_CALL_DIRECT_USE_SUCCESS_RATE",
        "INVALID_ACTION_RATE",
        "REGISTRY_PATH",
        "TRAJECTORY_PATH",
        "RESULT_PATH",
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
