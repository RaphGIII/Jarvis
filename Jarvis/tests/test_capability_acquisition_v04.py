from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

from capabilities.models import CapabilityManifest, SkillSpecification
from capabilities.permissions import PermissionPolicy
from capabilities.promotion import SkillPromoter
from capabilities.registry import CapabilityRegistry
from capabilities.resolver import CapabilityResolver
from capabilities.workspace import SkillWorkspaceManager
from development.memory import DevelopmentMemory
from development.software_engineer import AutonomousSoftwareEngineer, ProjectRequest
from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.environment import CodingEnvironment
from environments.coding.sandbox_backend import LocalTestSandboxBackend
from runtime.capability_runtime import CapabilityAcquisitionRuntime, CapabilityRuntimeConfig
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.capability_acquisition_v04_demo import MockCapabilityBrain, run_capability_acquisition_v04_demo, CapabilityAcquisitionV04Config
from training.capability_curriculum import CapabilityAcquisitionTaskFactory


def _factory(tmp_path: Path) -> CapabilityAcquisitionTaskFactory:
    return CapabilityAcquisitionTaskFactory(tmp_path / "benchmark")


def _runtime(tmp_path: Path, tasks):
    return CapabilityAcquisitionRuntime(
        brain=MockCapabilityBrain(tasks),
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_build_steps=6),
    )


class ProtectedEditBrain:
    provider_name = "protected_edit"
    model_name = "ProtectedEditBrain"
    last_metadata = {"generated_tokens": 1, "total_tokens": 1}

    def generate_structured(self, prompt, schema, *, max_tokens=700, temperature=0.2, top_p=0.9):
        return (
            '{"summary":"bad","files":['
            '{"path":"main.py","content":"def run(payload):\\n    return {}\\n"},'
            '{"path":"test_public.py","content":"pass"}]}'
        )


def test_v04_persistent_capability_registry(tmp_path):
    path = tmp_path / "registry.json"
    registry = CapabilityRegistry(path)
    manifest = CapabilityManifest("text.statistics", "Compute text statistics.", source_location="skills/installed/text.statistics/1.0.0")
    registry.register(manifest)

    restored = CapabilityRegistry(path)
    assert restored.has("text.statistics")
    assert restored.get("text.statistics").description == "Compute text statistics."
    restored.disable("text.statistics", "test")
    assert not CapabilityRegistry(path).has("text.statistics")


def test_v04_capability_matching_and_missing_detection(tmp_path):
    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(CapabilityManifest("data.csv.mode", "Read CSV text and compute the most common column value."))
    resolver = CapabilityResolver(registry)

    available = resolver.resolve("Please compute the most common value in a CSV column using data.csv.mode")
    missing = resolver.resolve("Build a note store capability")

    assert available.status == "available"
    assert available.capability_id == "data.csv.mode"
    assert missing.status == "missing"


def test_v04_skill_specification_validation_and_permission_blocking():
    spec = SkillSpecification(
        capability_id="net.fetch",
        objective="Fetch a URL",
        acceptance_criteria=["works"],
        public_tests=[{"input": {}, "expected_keys": ["result"]}],
        permissions=["network.http"],
    )
    assert spec.validate() == []
    decision = PermissionPolicy().evaluate(spec)
    assert not decision.allowed
    assert decision.blocked_permissions == ["network.http"]


def test_v04_staging_workspace_and_protected_public_tests(tmp_path):
    task = _factory(tmp_path).make_tasks(1)[0]
    staged = SkillWorkspaceManager(tmp_path / "staging").create(task.specification, "candidate")
    coding_task = staged.to_task("develop skill", max_steps=2)
    env = CodingEnvironment(coding_task, backend=LocalTestSandboxBackend())

    assert (staged.root / "test_public.py").exists()
    assert (staged.root / "skill_spec.json").exists()
    blocked = env.step(ActionCandidate(ActionType.WRITE_FILE, {"path": "test_public.py", "content": "pass"}))
    assert not blocked.action_result.ok
    assert blocked.objective_metrics["invalid_action"] is True
    assert staged.protected_files_pristine()


def test_v04_hidden_verifier_is_not_in_staged_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    runtime = _runtime(tmp_path, [task])
    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, hidden_workspace=hidden)

    assert result.success
    staged_roots = list((tmp_path / "skills" / "_staging").rglob(task.specification.capability_id.replace(".", "_") + "*"))
    assert not any("hidden_verifier.py" in str(path) for path in staged_roots)
    assert "hidden_verifier.py" not in "\n".join(event["stage"] + str(event["payload"]) for event in runtime.trajectory_store.load_all()[0]["events"])


def test_v04_greenfield_engine_does_not_use_low_level_action_loop():
    source = inspect.getsource(CapabilityAcquisitionRuntime.handle_goal)
    assert "run_episode" not in source
    assert "ActionCandidate" not in source


def test_v04_software_engineer_initial_build_runs_tests_automatically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    brain = MockCapabilityBrain([task], fail_first=False)
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_build_steps=6),
    )

    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    assert result.success
    assert result.initial_implementation_pass
    assert result.repair_iterations == 0
    assert brain.implementation_calls == 1
    assert brain.repair_calls == 0


def test_v04_failure_output_reaches_repair_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    brain = MockCapabilityBrain([task], fail_first=True)
    prompts = []
    original = brain.generate_structured

    def capture(prompt, schema, **kwargs):
        if "Repair the current project" in prompt:
            prompts.append(prompt)
        return original(prompt, schema, **kwargs)

    brain.generate_structured = capture
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_build_steps=6),
    )

    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    assert result.success
    assert brain.repair_calls == 1
    assert prompts
    assert "Exact public test" in prompts[0]
    assert "AssertionError" in prompts[0] or "FAIL:" in prompts[0]


def test_v04_multi_file_greenfield_project_promotes_and_executes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(2)[1]
    hidden = factory.create_hidden_verifier(task)
    runtime = CapabilityAcquisitionRuntime(
        brain=MockCapabilityBrain([task], fail_first=False, multi_file=True),
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_build_steps=6),
    )

    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)
    manifest = runtime.registry.get(task.specification.capability_id)

    assert result.success
    assert manifest is not None
    assert (Path(manifest.source_location) / "helper.py").exists()


def test_v04_multiple_repair_cycles_are_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    brain = MockCapabilityBrain([task], fail_first=True, repair_failures=1)
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_repair_cycles=3),
    )

    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    assert result.success
    assert result.repair_iterations == 2
    assert brain.repair_calls == 2


def test_v04_malformed_structured_output_regenerates_without_qwen_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    brain = MockCapabilityBrain([task], malformed_first=True)
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_repair_cycles=1),
    )

    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    assert result.success
    assert result.promoted
    assert brain.implementation_calls == 2
    assert runtime.registry.has(task.specification.capability_id)


def test_v04_software_engineer_rejects_protected_file_materialization(tmp_path):
    task = _factory(tmp_path).make_tasks(1)[0]
    staged = SkillWorkspaceManager(tmp_path / "staging").create(task.specification, "candidate")
    engineer = AutonomousSoftwareEngineer(
        brain=ProtectedEditBrain(),
        backend=LocalTestSandboxBackend(),
        memory=DevelopmentMemory(tmp_path / "memory.jsonl"),
    )

    result = engineer.build(
        ProjectRequest(
            goal=task.goal,
            specification=task.specification,
            workspace=staged.root,
            test_command=["python", "-m", "unittest", "test_public.py"],
            protected_paths={"test_public.py", "skill_spec.json"},
            max_repair_cycles=0,
        )
    )

    assert not result.success
    assert "protected path" in result.error


def test_v04_successful_acquisition_promotion_execution_and_second_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    runtime = _runtime(tmp_path, [task])

    first = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)
    second = CapabilityAcquisitionRuntime(
        brain=MockCapabilityBrain([task]),
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, max_build_steps=6),
    ).handle_goal(task.second_goal, request_payload=task.request_payload, expected_output=task.expected_output)

    assert first.success
    assert first.promoted
    assert first.public_success
    assert first.hidden_success
    assert first.execution_success
    assert first.second_call_success
    assert second.success
    assert not second.promoted
    assert second.resolution.status == "available"
    assert second.output == task.expected_output


def test_v04_failed_promotion_when_hidden_verifier_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    bad_task = replace(task, hidden_tests=[{"input": task.request_payload, "expected": {"wrong": True}}])
    hidden = factory.create_hidden_verifier(bad_task)
    runtime = _runtime(tmp_path, [task])

    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    assert not result.success
    assert not result.promoted
    assert result.public_success
    assert not result.hidden_success
    assert not runtime.registry.has(task.specification.capability_id)


def test_v04_versioned_skill_promotion_never_overwrites(tmp_path):
    registry = CapabilityRegistry(tmp_path / "registry.json")
    task = _factory(tmp_path).make_tasks(1)[0]
    staged = SkillWorkspaceManager(tmp_path / "staging").create(task.specification, "candidate")
    (staged.root / "main.py").write_text("def run(payload):\n    return {'ok': True}\n", encoding="utf-8")
    promoter = SkillPromoter(tmp_path / "installed", registry)

    first = promoter.promote(task.specification, staged, public_success=True, hidden_success=True)
    second = promoter.promote(task.specification, staged, public_success=True, hidden_success=True)

    assert first.promoted
    assert not second.promoted
    assert any("already exists" in error for error in second.errors)


def test_v04_trajectory_records_successful_and_failed_attempts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    runtime = _runtime(tmp_path, [task])
    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    records = runtime.trajectory_store.load_all()
    stages = [event["stage"] for event in records[0]["events"]]
    assert result.trajectory_id == records[0]["trajectory_id"]
    assert "gap_detection" in stages
    assert "specification" in stages
    assert "promote" in stages
    assert records[0]["outcome"]["promoted"] is True


def test_v04_development_memory_persists_and_retrieves_success_and_failure_patterns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factory = _factory(tmp_path)
    task = factory.make_tasks(1)[0]
    hidden = factory.create_hidden_verifier(task)
    runtime = _runtime(tmp_path, [task])
    result = runtime.handle_goal(task.goal, request_payload=task.request_payload, expected_output=task.expected_output, second_goal=task.second_goal, hidden_workspace=hidden)

    records = runtime.development_memory.load_all()
    retrieved = runtime.development_memory.retrieve(task.second_goal, task.specification.to_dict(), failure_text="AssertionError")

    assert result.success
    assert records
    assert records[0].failures
    assert retrieved


def test_v04_shadow_learning_mode_cannot_change_selected_action(tmp_path):
    runtime = JarvisRuntime(
        action_generator=lambda goal, observation: [],
        sandbox_backend=LocalTestSandboxBackend(),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
        config=JarvisRuntimeConfig(
            latent_dim=8,
            hidden_dim=8,
            replay_capacity=10,
            eval_controller="full",
            production_controller="heuristic",
            learned_controller_mode="shadow",
            learned_gate_min_experiences=0,
            learned_gate_warmup_experiences=1,
        ),
    )
    task = _factory(tmp_path).make_tasks(1)[0]
    staged = SkillWorkspaceManager(tmp_path / "staging").create(task.specification, "candidate")
    coding_task = staged.to_task("shadow", max_steps=2)
    observation = runtime.start_task(coding_task, RuntimeMode.EVAL)
    candidates = [
        ActionCandidate(ActionType.RUN_TESTS, confidence=0.2, estimated_cost=2.0),
        ActionCandidate(ActionType.WRITE_FILE, {"path": "main.py", "content": "def run(payload): return {'x': 1}"}, confidence=1.0, estimated_cost=0.1),
    ]
    latent = runtime._encode_features(runtime._observation_features(observation))
    scored = runtime._score_candidates(latent, candidates, observation)
    selected = runtime._select(scored)

    heuristic = max([item for item in scored if item.feasible], key=lambda item: item.heuristic_score)
    assert selected.candidate == heuristic.candidate
    assert all(item.score == item.heuristic_score for item in scored)
    assert any("q" in item.normalized_score_components for item in scored)


def test_v04_benchmark_catalog_has_15_distinct_capability_tasks(tmp_path):
    tasks = _factory(tmp_path).make_tasks()
    capability_ids = [task.specification.capability_id for task in tasks]

    assert len(tasks) == 15
    assert len(capability_ids) == len(set(capability_ids))
    assert all(task.specification.capability_id not in task.goal for task in tasks)
    assert len({tuple(sorted(task.expected_output.keys())) for task in tasks}) > 8


def test_v04_research_note_reaches_specification_metadata_and_trajectory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from capabilities.research import CapabilityResearcher, ResearchNote

    class _StubResearcher(CapabilityResearcher):
        def research(self, goal):
            return ResearchNote(
                query="sha-256 checksum hexdigest",
                source="https://docs.python.org/3/library/hashlib.html",
                summary="hashlib.sha256(data).hexdigest() returns the hex digest as a string.",
                fetched=True,
            )

    class _SpecEchoingBrain:
        provider_name = "spec_echo"
        model_name = "SpecEchoingBrain"
        last_metadata = {"generated_tokens": 1, "total_tokens": 1}

        def __init__(self) -> None:
            self.seen_research_in_prompt = False

        def generate(self, prompt, *, max_tokens=700, temperature=0.2, top_p=None):
            if "Decide if one installed capability" in prompt:
                return '{"status":"missing","capability_id":"","reason":"none","confidence":0.0}'
            if "hashlib.sha256(data).hexdigest()" in prompt:
                self.seen_research_in_prompt = True
            return (
                '{"capability_id":"local.sha256.checksum","objective":"Compute a sha256 checksum of a string.",'
                '"functional_requirements":["Hash the input text."],'
                '"inputs":{"type":"object","additionalProperties":true},'
                '"outputs":{"type":"object","additionalProperties":true},'
                '"constraints":["Use Python standard library only."],'
                '"allowed_dependencies":[],"permissions":[],'
                '"acceptance_criteria":["Public examples pass."],'
                '"public_tests":[{"name":"hashes","input":{"text":"abc"},'
                '"expected":{"result":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}],'
                '"proposed_file_structure":["main.py"]}'
            )

    brain = _SpecEchoingBrain()
    runtime = CapabilityAcquisitionRuntime(
        brain=brain,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False),
        researcher=_StubResearcher(),
    )

    result = runtime.handle_goal("Compute a sha256 checksum of a string.")

    assert brain.seen_research_in_prompt
    records = runtime.trajectory_store.load_all()
    research_events = [event for event in records[-1]["events"] if event["stage"] == "research"]
    assert research_events
    assert research_events[0]["payload"]["fetched"] is True
    assert "hashlib" in research_events[0]["payload"]["source"]
    spec_events = [event for event in records[-1]["events"] if event["stage"] == "specification"]
    assert spec_events[0]["payload"]["specification"]["metadata"]["research"]["fetched"] is True


def test_v04_research_disabled_by_default_config_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = CapabilityAcquisitionRuntime(
        brain=None,
        backend=LocalTestSandboxBackend(),
        config=CapabilityRuntimeConfig(data_dir=str(tmp_path / "runtime"), use_docker=False, enable_research=False),
    )
    assert runtime.researcher is None


def test_v04_mock_demo_runs_without_qwen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = run_capability_acquisition_v04_demo(
        CapabilityAcquisitionV04Config(
            mock_brain=True,
            task_count=2,
            benchmark_dir=str(tmp_path / "demo"),
            quiet=True,
            local_test_backend=True,
        )
    )

    assert metrics["CAPABILITY_ACQUISITION_SUCCESS_RATE"] == 1.0
    assert metrics["SECOND_CALL_DIRECT_USE_SUCCESS_RATE"] == 1.0
    assert "INITIAL_IMPLEMENTATION_PASS_RATE" in metrics
    assert "MEAN_LLM_CALLS_PER_CAPABILITY" in metrics
    assert Path(metrics["RESULT_PATH"]).exists()


def test_generated_public_test_file_executes_bool_and_null_expected_values(tmp_path):
    """Regression test: public_tests containing JSON true/false/null must render as
    executable Python (json.loads at import time), not raw JSON literals spliced
    into source code (which previously produced `NameError: name 'true' is not
    defined` -- discovered via a live local-LLM run whose spec legitimately used
    boolean outputs, e.g. a leap-year predicate)."""
    spec = SkillSpecification(
        capability_id="local.bool_and_null_probe",
        objective="Return booleans and nulls to exercise public-test rendering.",
        public_tests=[
            {"name": "true_case", "input": {"x": 1}, "expected": {"ok": True}},
            {"name": "false_case", "input": {"x": 0}, "expected": {"ok": False}},
            {"name": "null_case", "input": {"x": None}, "expected": {"ok": None}},
        ],
        proposed_file_structure=["main.py"],
    )
    staged = SkillWorkspaceManager(tmp_path / "staging").create(spec, "candidate")
    (staged.root / "main.py").write_text(
        "def run(payload: dict) -> dict:\n"
        "    x = payload.get('x')\n"
        "    if x == 1:\n"
        "        return {'ok': True}\n"
        "    if x == 0:\n"
        "        return {'ok': False}\n"
        "    return {'ok': None}\n",
        encoding="utf-8",
    )

    import subprocess
    import sys as _sys

    completed = subprocess.run(
        [_sys.executable, "-m", "unittest", "test_public.py"],
        cwd=staged.root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NameError" not in completed.stderr


def test_generated_hidden_verifier_executes_bool_and_null_expected_values(tmp_path):
    """Same regression as above, for the hidden-verifier source generator used by
    the benchmark curriculum (training/capability_curriculum.py)."""
    from training.capability_curriculum import _hidden_verifier_source

    hidden_tests = [
        {"input": {"x": 1}, "expected": {"ok": True}},
        {"input": {"x": 0}, "expected": {"ok": False}},
        {"input": {"x": None}, "expected": {"ok": None}},
    ]
    source = _hidden_verifier_source(hidden_tests)
    workdir = tmp_path / "hidden_probe"
    workdir.mkdir()
    (workdir / "hidden_verifier.py").write_text(source, encoding="utf-8")
    (workdir / "main.py").write_text(
        "def run(payload: dict) -> dict:\n"
        "    x = payload.get('x')\n"
        "    if x == 1:\n"
        "        return {'ok': True}\n"
        "    if x == 0:\n"
        "        return {'ok': False}\n"
        "    return {'ok': None}\n",
        encoding="utf-8",
    )

    import subprocess
    import sys as _sys

    completed = subprocess.run(
        [_sys.executable, "hidden_verifier.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NameError" not in completed.stderr
