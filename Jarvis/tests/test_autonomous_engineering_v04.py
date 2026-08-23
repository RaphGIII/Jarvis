from __future__ import annotations

import json
import subprocess
from pathlib import Path

from capabilities.models import SkillSpecification
from development.memory import DevelopmentMemory
from development.qa import InternalTestEngineer, compile_contract
from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal, SelfImprovementMemory
from development.software_engineer import AutonomousSoftwareEngineer, ProjectRequest
from environments.coding.sandbox_backend import LocalTestSandboxBackend
from runtime.capability_runtime import CapabilityAcquisitionRuntime, CapabilityRuntimeConfig
from training.capability_acquisition_v04_demo import MockCapabilityBrain
from training.capability_curriculum import CapabilityAcquisitionTaskFactory
from training.self_improvement_demo import SelfImprovementDemoConfig, run_self_improvement_demo


def _runtime(tmp_path: Path, brain) -> CapabilityAcquisitionRuntime:
    return CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(
            data_dir=str(tmp_path / "runtime"),
            skills_root=str(tmp_path / "skills"),
            use_docker=False,
            max_repair_cycles=2,
            max_blind_repair_cycles=2,
        ),
    )


def _hidden_workspace(root: Path, source: str) -> Path:
    path = root / "hidden"
    path.mkdir(parents=True, exist_ok=True)
    (path / "hidden_verifier.py").write_text(source, encoding="utf-8")
    return path


def _simple_spec(capability_id: str = "custom.scale") -> SkillSpecification:
    return SkillSpecification(
        capability_id=capability_id,
        objective="Double an integer x from the request payload.",
        functional_requirements=["Return y equal to x multiplied by two."],
        inputs={"type": "object"},
        outputs={"type": "object"},
        constraints=["Use Python standard library only.", "No network access.", "Expose run(payload: dict) -> dict."],
        acceptance_criteria=["Public tests pass.", "Internal QA passes.", "Hidden verifier passes."],
        public_tests=[{"name": "visible", "input": {"x": 1}, "expected": {"y": 2}}],
        proposed_file_structure=["main.py"],
    )


class StaticBundleBrain:
    provider_name = "static_bundle"
    model_name = "StaticBundleBrain"
    last_metadata = {"generated_tokens": 1, "total_tokens": 1}

    def __init__(self, files: list[dict[str, str]], *, review: dict | None = None) -> None:
        self.files = files
        self.review = review
        self.prompts: list[str] = []

    def generate_structured(self, prompt, schema, *, max_tokens=700, temperature=0.2, top_p=0.9):
        self.prompts.append(prompt)
        props = schema.get("properties") or {}
        if "cases" in props:
            return json.dumps({"cases": []})
        if "approved" in props:
            return json.dumps(
                self.review
                or {
                    "approved": True,
                    "contract_violations": [],
                    "risk_cases": [],
                    "recommended_tests": [],
                    "repair_required": False,
                }
            )
        return json.dumps({"summary": "static", "plan": "static", "files": self.files, "diagnosis": "static"})

    def generate(self, prompt, *, max_tokens=700, temperature=0.2, top_p=None):
        return json.dumps({"status": "missing", "capability_id": "", "reason": "mock", "confidence": 0.0})


class MalformedRepairBrain(StaticBundleBrain):
    def __init__(self) -> None:
        super().__init__([{"path": "main.py", "content": "def run(payload):\n    return {'lines': None}\n"}])
        self.repair_calls = 0

    def generate_structured(self, prompt, schema, *, max_tokens=700, temperature=0.2, top_p=0.9):
        props = schema.get("properties") or {}
        if "cases" in props or "approved" in props:
            return super().generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        if "Repair the current project" in prompt:
            self.repair_calls += 1
            if self.repair_calls == 1:
                return "not json"
            return json.dumps(
                {
                    "diagnosis": "fix line counting",
                    "files": [
                        {
                            "path": "main.py",
                            "content": "def run(payload):\n    return {'lines': sum(1 for line in str(payload.get('text', '')).splitlines() if line.strip())}\n",
                        }
                    ],
                }
            )
        return super().generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=top_p)


class BlindRepairBrain(StaticBundleBrain):
    def __init__(self) -> None:
        super().__init__([{"path": "main.py", "content": "def run(payload):\n    return {'y': int(payload.get('x', 0)) + 1}\n"}])
        self.blind_prompts: list[str] = []

    def generate_structured(self, prompt, schema, *, max_tokens=700, temperature=0.2, top_p=0.9):
        props = schema.get("properties") or {}
        if "files" in props and "External acceptance verification failed" in prompt:
            self.blind_prompts.append(prompt)
            return json.dumps(
                {
                    "diagnosis": "generalize arithmetic rule from contract",
                    "files": [{"path": "main.py", "content": "def run(payload):\n    return {'y': int(payload.get('x', 0)) * 2}\n"}],
                }
            )
        return super().generate_structured(prompt, schema, max_tokens=max_tokens, temperature=temperature, top_p=top_p)


def test_internal_tests_are_external_to_implementation_workspace(tmp_path):
    task = CapabilityAcquisitionTaskFactory(tmp_path / "tasks").make_tasks(1)[0]
    staged = tmp_path / "candidate"
    staged.mkdir()
    suite = InternalTestEngineer().create_suite(task.specification, compile_contract(task.specification), tmp_path / "internal")

    assert suite.path.parent != staged
    assert suite.path.exists()
    assert "HIDDEN" not in suite.path.read_text(encoding="utf-8")


def test_public_pass_only_cannot_promote_when_internal_qa_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = SkillSpecification(
        capability_id="text.line_count",
        objective="Count non-empty lines in supplied text.",
        functional_requirements=["Split text into lines.", "Ignore empty or whitespace-only lines."],
        acceptance_criteria=["Public tests pass.", "Internal QA passes.", "Hidden verifier passes."],
        public_tests=[{"name": "weak", "input": {"text": "a\nb"}, "expected": {"lines": 2}}],
        proposed_file_structure=["main.py"],
    )
    brain = StaticBundleBrain([{"path": "main.py", "content": "def run(payload):\n    return {'lines': len(str(payload.get('text', '')).splitlines())}\n"}])
    hidden = _hidden_workspace(tmp_path, "import main\nassert main.run({'text': 'a\\nb'}) == {'lines': 2}\n")

    result = _runtime(tmp_path, brain).handle_goal(
        spec.objective,
        request_payload={"text": "a\nb"},
        expected_output={"lines": 2},
        spec=spec,
        hidden_workspace=hidden,
    )

    assert result.public_success
    assert not result.internal_verification_success
    assert not result.promoted
    assert not result.success


def test_reviewer_can_reject_missing_contract_behavior_after_tests_pass(tmp_path):
    spec = SkillSpecification(
        capability_id="custom.rule",
        objective="Apply a named text rule.",
        functional_requirements=["Reject unknown rules."],
        acceptance_criteria=["Public tests pass.", "Reviewer approves."],
        public_tests=[{"name": "upper", "input": {"text": "a", "rule": "upper"}, "expected": {"text": "A"}}],
        proposed_file_structure=["main.py"],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engineer = AutonomousSoftwareEngineer(
        brain=StaticBundleBrain(
            [{"path": "main.py", "content": "def run(payload):\n    return {'text': str(payload.get('text', '')).upper()}\n"}],
            review={
                "approved": False,
                "contract_violations": [],
                "risk_cases": ["unknown rules are not rejected"],
                "recommended_tests": [],
                "repair_required": True,
            },
        ),
        backend=LocalTestSandboxBackend(),
        memory=DevelopmentMemory(tmp_path / "memory.jsonl"),
    )
    (workspace / "test_public.py").write_text(
        "import unittest, main\n\nclass T(unittest.TestCase):\n    def test_upper(self):\n        self.assertEqual(main.run({'text':'a','rule':'upper'}), {'text':'A'})\n",
        encoding="utf-8",
    )

    result = engineer.build(
        ProjectRequest(
            goal=spec.objective,
            specification=spec,
            workspace=workspace,
            test_command=["python", "-m", "unittest", "test_public.py"],
            protected_paths={"test_public.py"},
            max_repair_cycles=0,
        )
    )

    assert result.public_test_result and result.public_test_result.success
    assert result.internal_verification_success
    assert not result.reviewer_approved
    assert any("reject" in " ".join(item.get("risk_cases", [])).lower() for item in result.review_findings)


def test_malformed_repair_output_is_regenerated_within_cycle(tmp_path):
    task = CapabilityAcquisitionTaskFactory(tmp_path / "tasks").make_tasks(1)[0]
    staged = tmp_path / "workspace"
    staged.mkdir()
    (staged / "test_public.py").write_text(
        "import unittest, main\n\nclass T(unittest.TestCase):\n    def test_lines(self):\n        self.assertEqual(main.run({'text':'a\\n\\n b \\n'}), {'lines': 2})\n",
        encoding="utf-8",
    )
    brain = MalformedRepairBrain()
    result = AutonomousSoftwareEngineer(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        memory=DevelopmentMemory(tmp_path / "memory.jsonl"),
    ).build(
        ProjectRequest(
            goal=task.goal,
            specification=task.specification,
            workspace=staged,
            test_command=["python", "-m", "unittest", "test_public.py"],
            protected_paths={"test_public.py"},
            max_repair_cycles=1,
        )
    )

    assert result.success
    assert result.repair_cycles == 1
    assert brain.repair_calls == 2


def test_blind_hidden_failure_repair_does_not_expose_verifier_details(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _simple_spec()
    brain = BlindRepairBrain()
    hidden = _hidden_workspace(tmp_path, "import main\nassert main.run({'x': 3}) == {'y': 6}\n")

    result = _runtime(tmp_path, brain).handle_goal(
        spec.objective,
        request_payload={"x": 2},
        expected_output={"y": 4},
        spec=spec,
        hidden_workspace=hidden,
    )

    assert result.success
    assert result.blind_repair_success
    assert brain.blind_prompts
    blind_prompt = brain.blind_prompts[0]
    assert "External acceptance verification failed" in blind_prompt
    assert "hidden_verifier" not in blind_prompt
    assert "{'x': 3}" not in blind_prompt
    assert "{'y': 6}" not in blind_prompt


def test_development_memory_records_partial_and_final_success_separately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _simple_spec()
    brain = BlindRepairBrain()
    hidden = _hidden_workspace(tmp_path, "import main\nassert main.run({'x': 3}) == {'y': 6}\n")
    runtime = _runtime(tmp_path, brain)

    result = runtime.handle_goal(
        spec.objective,
        request_payload={"x": 2},
        expected_output={"y": 4},
        spec=spec,
        hidden_workspace=hidden,
    )
    records = runtime.development_memory.load_all()

    assert result.success
    assert records[-1].public_success
    assert records[-1].internal_verification_success
    assert records[-1].reviewer_approved
    assert records[-1].hidden_success
    assert records[-1].promotion_success
    assert records[-1].execution_success
    assert records[-1].second_call_success
    assert records[-1].final_success


def test_semantically_equivalent_second_call_reuses_installed_capability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = CapabilityAcquisitionTaskFactory(tmp_path / "tasks").make_tasks(1)[0]
    hidden = CapabilityAcquisitionTaskFactory(tmp_path / "tasks").create_hidden_verifier(task)
    runtime = _runtime(tmp_path, MockCapabilityBrain([task], fail_first=False))

    first = runtime.handle_goal(
        task.goal,
        request_payload=task.request_payload,
        expected_output=task.expected_output,
        second_goal="How many actual lines of content are in this string?",
        hidden_workspace=hidden,
    )
    second = CapabilityAcquisitionRuntime(
        brain=MockCapabilityBrain([task], fail_first=False),
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), skills_root=str(tmp_path / "skills"), use_docker=False),
    ).handle_goal(
        "How many actual lines of content are in this string?",
        request_payload=task.request_payload,
        expected_output=task.expected_output,
    )

    assert first.success
    assert first.second_call_success
    assert second.resolution.status == "available"
    assert second.success


class RepoPatchBrain:
    def __init__(self, proposal: dict) -> None:
        self.proposal = proposal

    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        return json.dumps(self.proposal)


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "util.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (root / "test_util.py").write_text(
        "import unittest\nfrom util import answer\n\nclass T(unittest.TestCase):\n    def test_answer(self):\n        self.assertEqual(answer(), 2)\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "config", "user.email", "jarvis@example.invalid"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "config", "user.name", "Jarvis Test"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, capture_output=True, text=True, timeout=20)
    return root


def test_repository_engineer_creates_isolated_multifile_candidate(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    proposal = {
        "analysis": "Use helper module and fix answer.",
        "files": [{"path": "util.py", "content": "from helper import value\n\n\ndef answer():\n    return value()\n"}],
        "new_files": [{"path": "helper.py", "content": "def value():\n    return 2\n"}],
        "deleted_files": [],
    }
    goal = SelfImprovementGoal(
        objective="Fix answer implementation.",
        success_criteria=["unit tests pass"],
        allowed_paths=["util.py", "helper.py"],
        protected_paths=["test_util.py"],
        tests=[["python", "-m", "unittest", "test_util.py"]],
    )
    engineer = RepositoryEngineer(
        brain=RepoPatchBrain(proposal),
        worktree_root=tmp_path / "worktrees",
        memory=SelfImprovementMemory(tmp_path / "memory.jsonl"),
    )

    result = engineer.improve(repo, goal, goal.tests)

    assert result.status == "SELF_DEVELOPMENT_CANDIDATE_READY"
    assert Path(result.worktree) != repo
    assert (repo / "util.py").read_text(encoding="utf-8") == "def answer():\n    return 1\n"
    assert "helper.py" in result.diff
    assert all(item.success for item in result.tests)
    assert engineer.memory.load_all()


def test_repository_engineer_rejects_protected_path_changes(tmp_path):
    repo = _make_repo(tmp_path / "repo")
    proposal = {
        "analysis": "bad",
        "files": [{"path": "test_util.py", "content": "pass\n"}],
        "new_files": [],
        "deleted_files": [],
    }
    goal = SelfImprovementGoal(
        objective="Try to fake tests.",
        allowed_paths=["."],
        protected_paths=["test_util.py"],
        tests=[["python", "-m", "unittest", "test_util.py"]],
    )

    result = RepositoryEngineer(brain=RepoPatchBrain(proposal), worktree_root=tmp_path / "worktrees").improve(repo, goal, goal.tests)

    assert not result.success
    assert "protected" in result.error
    assert (repo / "test_util.py").read_text(encoding="utf-8").startswith("import unittest")


def test_self_improvement_demo_runs_on_disposable_fixture(tmp_path):
    metrics = run_self_improvement_demo(
        SelfImprovementDemoConfig(
            benchmark_dir=str(tmp_path / "demo"),
            mock_brain=True,
            quiet=True,
        )
    )

    assert metrics["SUCCESS"] is True
    assert metrics["STATUS"] == "SELF_DEVELOPMENT_CANDIDATE_READY"
    assert Path(metrics["RESULT_PATH"]).exists()


def test_valid_bundle_rejects_placeholder_only_file_content():
    """A weak local model can literally return the "..." schema-example
    placeholder as a file's actual `content` instead of real code. That must
    be treated as an invalid bundle (triggering retry/repair) rather than
    written to disk as a broken file (discovered via live small-model
    testing on a multi-file capability build)."""
    from development.software_engineer import _valid_bundle

    assert not _valid_bundle({"summary": "x", "files": [{"path": "main.py", "content": "..."}]})
    assert not _valid_bundle({"summary": "x", "files": [{"path": "main.py", "content": "   ...   "}]})
    assert _valid_bundle({"summary": "x", "files": [{"path": "main.py", "content": "def run(payload):\n    return {}\n"}]})
    assert not _valid_bundle(
        {
            "summary": "x",
            "files": [
                {"path": "main.py", "content": "def run(payload):\n    return {}\n"},
                {"path": "helper.py", "content": "..."},
            ],
        }
    )


def test_missing_required_files_flags_specification_mandated_files_not_provided():
    """When the specification's proposed_file_structure mandates a helper
    module (e.g. a multi-file design), a bundle that only returns main.py
    must be flagged as missing that file so the caller retries/repairs with
    an explicit corrective instruction, instead of silently accepting a
    single-file bundle that can never satisfy a multi-file acceptance
    criterion (discovered via live small-model testing on a multi-file
    capability build: the model kept inlining everything into main.py)."""
    from development.software_engineer import _missing_required_files

    bundle_single_file = {"files": [{"path": "main.py", "content": "def run(payload):\n    return {}\n"}]}
    assert _missing_required_files(bundle_single_file, ["aggregator.py"]) == ["aggregator.py"]
    assert _missing_required_files(bundle_single_file, []) == []
    assert _missing_required_files(bundle_single_file, None) == []

    bundle_with_helper = {
        "files": [
            {"path": "main.py", "content": "import aggregator\n"},
            {"path": "aggregator.py", "content": "def aggregate():\n    return {}\n"},
        ]
    }
    assert _missing_required_files(bundle_with_helper, ["aggregator.py"]) == []

    bundle_placeholder_helper = {
        "files": [
            {"path": "main.py", "content": "import aggregator\n"},
            {"path": "aggregator.py", "content": "..."},
        ]
    }
    assert _missing_required_files(bundle_placeholder_helper, ["aggregator.py"]) == ["aggregator.py"]


def test_review_from_payload_rejects_contradictory_approved_and_repair_required():
    """A weak local model can emit approved=true with repair_required=true (both
    literally, e.g. echoing schema example placeholders). Trusting `approved` in
    that case would silently promote a candidate the model itself flagged as
    needing repair, so this must be treated as malformed and fall back to the
    deterministic reviewer (discovered via live small-model testing)."""
    from development.software_engineer import _review_from_payload

    finding = _review_from_payload(
        {
            "approved": True,
            "contract_violations": [],
            "risk_cases": [],
            "recommended_tests": [],
            "repair_required": True,
        }
    )
    assert finding is None


def test_review_from_payload_strips_placeholder_echoes():
    """A weak local model can literally echo the "..." placeholder tokens from
    the response-schema example as if they were real findings/tests. Those must
    be filtered out rather than treated as genuine contract violations, risks,
    or recommended tests."""
    from development.software_engineer import _review_from_payload

    finding = _review_from_payload(
        {
            "approved": True,
            "contract_violations": ["..."],
            "risk_cases": ["..."],
            "recommended_tests": [{"name": "...", "input": {"value": "..."}, "expected": None, "raises": False}],
            "repair_required": False,
        }
    )
    assert finding is not None
    assert finding.approved is True
    assert finding.contract_violations == []
    assert finding.risk_cases == []
    assert finding.recommended_tests == []
    assert finding.repair_required is False


def test_review_from_payload_keeps_genuine_findings():
    from development.software_engineer import _review_from_payload

    finding = _review_from_payload(
        {
            "approved": False,
            "contract_violations": ["main.py missing return type handling"],
            "risk_cases": [],
            "recommended_tests": [{"name": "negative_case", "input": {"value": -1}, "expected": {"ok": False}, "raises": False}],
            "repair_required": True,
        }
    )
    assert finding is not None
    assert finding.approved is False
    assert finding.contract_violations == ["main.py missing return type handling"]
    assert len(finding.recommended_tests) == 1
    assert finding.recommended_tests[0].name == "negative_case"
