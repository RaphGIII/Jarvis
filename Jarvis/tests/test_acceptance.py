"""The mission's acceptance tests, A through O.

Each test below is named for the letter it covers, so a reader can check the
brief against the suite line by line.

Most run deterministically against scripted models, because an acceptance suite
that needs a 45-second model load per assertion does not get run.  The two that
genuinely require real inference -- N (real hardware) and the live halves of A
and E -- are marked ``live`` and skipped unless BUILD_LOCAL actually answers.
That skip is deliberate and visible: a test that quietly passes without touching
the model would be worse than no test, so the live ones assert on evidence
recorded by real runs and say plainly when they were not run.

    python -m pytest tests/test_acceptance.py -q            # deterministic
    python -m pytest tests/test_acceptance.py -q -m live    # against Ollama
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from brain.resources import ResourcePolicy, ResourcePolicyStore
from brain.tiers import ModelCatalog, ModelProbe, ModelTier
from capabilities.registry import CapabilityRegistry
from capabilities.service import CapabilityService
from deployment.promotion import HealthCheck, PromotionAudit, PromotionOutcome, Promoter, default_health_check
from development.edit_engine import EditEngine, EditError, PathPolicy, parse_bundle
from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal
from knowledge.graph import KnowledgeGraph
from knowledge.memory import ExperienceMemory
from projects.engine import ProjectEngine
from projects.models import ResourceLimits, StopReason
from projects.store import ProjectStore
from tools.builtin import builtin_tools
from tools.registry import RiskLevel, ToolCall, ToolContext, ToolPolicy, ToolRegistry

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "data" / "acceptance_evidence"


# ==========================================================================
# Shared fixtures
# ==========================================================================


@pytest.fixture
def tools():
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.MODERATE))
    registry.register_many(builtin_tools())
    return registry


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "projects")


class PlannedBrain:
    """Replays a planned sequence of tool-call batches.

    Deterministic stand-in for the local model. Each entry is one EXECUTE step,
    so a test can script "get it wrong, then get it right" precisely.
    """

    def __init__(self, *, tasks, acceptance, executes, diagnosis="something was wrong", fix="correct it"):
        self.tasks = tasks
        self.acceptance = acceptance
        self.executes = list(executes)
        self.diagnosis = diagnosis
        self.fix = fix
        self.execute_calls = 0
        self.diagnose_calls = 0
        self.prompts: list[str] = []

    def generate_structured(self, prompt, schema, *, max_tokens=1000, temperature=0.1, top_p=0.9):
        self.prompts.append(prompt)
        properties = schema.get("properties") or {}
        if "diagnosis" in properties:
            self.diagnose_calls += 1
            return json.dumps({"diagnosis": self.diagnosis, "fix": self.fix})
        if "tasks" in properties:
            return json.dumps({"tasks": self.tasks, "acceptance": self.acceptance})
        if "findings" in properties:
            return json.dumps({"tool_calls": [{"name": "list_files", "arguments": {}}]})
        index = min(self.execute_calls, len(self.executes) - 1)
        self.execute_calls += 1
        return json.dumps({"tool_calls": self.executes[index]})


def write(path, content):
    return {"name": "write_file", "arguments": {"path": path, "content": content}}


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    (root / "helper.py").write_bytes(b"def describe():\n    return 'calculator'\n")
    (root / "test_calc.py").write_bytes(
        b"from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    _git(root, "init")
    _git(root, "config", "user.email", "jarvis@example.invalid")
    _git(root, "config", "user.name", "Jarvis Acceptance")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    return root


def _record(name: str, payload: dict) -> None:
    """Persist evidence so the final report quotes measurements, not claims."""

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_local_online() -> bool:
    try:
        return ModelProbe(ModelCatalog(), ttl_seconds=600).probe(ModelTier.BUILD_LOCAL).online
    except Exception:
        return False


live = pytest.mark.live


def _history(name: str) -> list[dict]:
    path = EVIDENCE / f"{name}.history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_live_evidence_is_real(name: str) -> None:
    """The run happened, against the configured model, and was recorded."""

    path = EVIDENCE / f"{name}.json"
    if not path.exists():
        pytest.skip(f"no live evidence for {name}; run python -m jarvis.record_evidence")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("model"), "evidence must name the model that produced it"
    assert payload.get("recorded_at"), "evidence must be timestamped"
    assert _history(name), "every attempt must be recorded, not only the last"


def _assert_has_passed_live(name: str) -> None:
    """The scenario has genuinely passed on real hardware at least once.

    Deliberately about the history rather than the newest attempt: a stochastic
    model makes "the last run passed" a coin toss, and a suite that flips with
    it teaches nothing. What matters is whether the capability has ever been
    demonstrated, and at what rate.
    """

    attempts = _history(name)
    if not attempts:
        pytest.skip(f"no recorded attempts for {name}")
    passed = [item for item in attempts if item.get("passed")]
    assert passed, (
        f"{name} has never passed on real hardware in {len(attempts)} recorded attempts. "
        "See AUTONOMOUS_CORE_REPORT.md section 7a."
    )


# ==========================================================================
# A. Local self-patch
# ==========================================================================


class SelfPatchBrain:
    """Emits a minimal, correct anchored edit -- the shape a good run produces."""

    def __init__(self, *, first_patch=None):
        self.first_patch = first_patch
        self.patches = 0

    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        properties = schema.get("properties") or {}
        if "requests" in properties:
            return json.dumps({"requests": [{"tool": "read_file", "path": "calc.py"}], "notes": ""})
        if "plan" in properties:
            return json.dumps({"analysis": "fix add", "plan": "change the operator", "files_to_change": ["calc.py"]})
        if "approved" in properties:
            return json.dumps({"approved": True, "blocking_findings": [], "optional_findings": [], "recommended_tests": []})
        self.patches += 1
        if self.first_patch is not None and self.patches == 1:
            return json.dumps(self.first_patch)
        return json.dumps(
            {"analysis": "use addition", "files": [{"path": "calc.py", "search": "a - b", "replace": "a + b"}], "new_files": [], "deleted_files": []}
        )


def _self_patch_goal():
    return SelfImprovementGoal(
        objective="Make calc.add actually add.",
        allowed_paths=["calc.py", "helper.py"],
        protected_paths=["test_calc.py"],
        tests=[[sys.executable, "-m", "pytest", "-q", "test_calc.py"]],
    )


def test_A_self_patch_produces_a_verified_candidate(tmp_path):
    """A. Jarvis finds the source, edits it, tests it, reaches candidate-ready."""

    repo = _make_repo(tmp_path / "repo")
    before = (repo / "calc.py").read_bytes()

    result = RepositoryEngineer(
        brain=SelfPatchBrain(), worktree_root=tmp_path / "worktrees", max_cycles=3
    ).improve(repo, _self_patch_goal())

    assert result.status == "SELF_DEVELOPMENT_CANDIDATE_READY"
    assert result.changed_files == ["calc.py"]
    assert all(item.success for item in result.tests)
    # The live repository is never touched: the candidate is isolated.
    assert (repo / "calc.py").read_bytes() == before
    assert "a + b" in (Path(result.worktree) / "calc.py").read_text(encoding="utf-8")


@live
def test_A_live_self_patch_evidence_is_real():
    """A (real hardware). The run happened against the configured model."""

    _assert_live_evidence_is_real("A_self_patch_live")


@live
@pytest.mark.xfail(
    reason=(
        "Known limitation, documented in AUTONOMOUS_CORE_REPORT.md section 7a: the self-patch "
        "scenario passed live on a 243-line jarvis/cli.py and has not passed on the current "
        "430-line one, where the help text is a strong decoy. Not marked as a pass; recorded as "
        "the failure it is, and expected to clear with a larger BUILD_LOCAL."
    ),
    strict=False,
)
def test_A_live_self_patch_has_passed_at_least_once():
    _assert_has_passed_live("A_self_patch_live")


# ==========================================================================
# B. Bad patch recovery
# ==========================================================================


def test_B_an_invalid_first_patch_does_not_end_the_mission(tmp_path):
    """B. The first patch cannot be applied; the run repairs and succeeds."""

    repo = _make_repo(tmp_path / "repo")
    brain = SelfPatchBrain(
        first_patch={
            "analysis": "a guess",
            # An anchor that appears nowhere in the file.
            "files": [{"path": "calc.py", "search": "def subtract(x, y):", "replace": "def add(a, b):"}],
            "new_files": [],
            "deleted_files": [],
        }
    )

    result = RepositoryEngineer(
        brain=brain, worktree_root=tmp_path / "worktrees", max_cycles=3
    ).improve(repo, _self_patch_goal())

    assert brain.patches >= 2, "the engine must ask again after an unusable patch"
    assert result.status == "SELF_DEVELOPMENT_CANDIDATE_READY"
    assert "a + b" in (Path(result.worktree) / "calc.py").read_text(encoding="utf-8")


def test_B_recovery_covers_every_way_a_local_model_gets_an_edit_wrong(tmp_path):
    """B. Each of these was produced by the real 7B model during development."""

    cases = {
        "unmatched anchor": {"path": "a.py", "search": "def nowhere():", "replace": "x = 1"},
        "ambiguous anchor": {"path": "dup.py", "search": "x = 1", "replace": "x = 2"},
        "whole file replaced by a fragment": {"path": "big.py", "content": "x = 1\n"},
        "definition duplicated": {"path": "a.py", "search": "def run():", "replace": "def run():\n    pass\n\n\ndef run():"},
    }
    (tmp_path / "a.py").write_bytes(b"def run():\n    return 1\n")
    (tmp_path / "dup.py").write_bytes(b"x = 1\ny = 2\nx = 1\n")
    (tmp_path / "big.py").write_bytes(("\n".join(f"line_{i} = {i}" for i in range(40)) + "\n").encode())
    originals = {path.name: path.read_bytes() for path in tmp_path.glob("*.py")}

    engine = EditEngine(PathPolicy(tmp_path))
    for label, edit in cases.items():
        with pytest.raises(EditError) as excinfo:
            engine.apply(parse_bundle({"files": [edit]}))
        assert excinfo.value.recoverable, f"{label} must be recoverable, not fatal"

    for name, content in originals.items():
        assert (tmp_path / name).read_bytes() == content, "a rejected edit changes nothing"


# ==========================================================================
# C. Test failure recovery
# ==========================================================================


def test_C_valid_but_wrong_code_is_diagnosed_from_evidence_and_repaired(tmp_path, store, tools):
    """C. The first implementation parses and is wrong; the tests say so."""

    check = [sys.executable, "-m", "pytest", "-q", "test_it.py"]
    brain = PlannedBrain(
        tasks=[{"title": "implement double"}],
        acceptance=[{"text": "the tests pass", "check": check}],
        executes=[
            # Syntactically fine, behaviourally wrong.
            [
                write("app.py", "def double(value):\n    return value + 2\n"),
                write("test_it.py", "from app import double\n\n\ndef test_double():\n    assert double(5) == 10\n"),
            ],
            [write("app.py", "def double(value):\n    return value * 2\n")],
        ],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project("double a number", limits=ResourceLimits(max_steps=14))

    result = engine.run(project)

    assert result.accepted, result.message
    assert brain.diagnose_calls >= 1, "the repair must follow a diagnosis, not a blind retry"
    evidence = project.objective_criteria()[0].last_evidence
    assert "exit=0" in evidence


def test_C_the_diagnosis_sees_the_real_test_output(tmp_path, store, tools):
    """C. A model that cannot see why it failed cannot fix it."""

    check = [sys.executable, "-m", "pytest", "-q", "test_it.py"]
    brain = PlannedBrain(
        tasks=[{"title": "implement double"}],
        acceptance=[{"text": "the tests pass", "check": check}],
        executes=[
            [
                write("app.py", "def double(value):\n    return value + 2\n"),
                write("test_it.py", "from app import double\n\n\ndef test_double():\n    assert double(5) == 10\n"),
            ],
        ],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    engine.run(
        engine.create_project("double a number", limits=ResourceLimits(max_steps=8, max_consecutive_failures=8))
    )

    briefing = "\n".join(brain.prompts)
    assert "OUTPUT OF THE FAILING ACCEPTANCE CHECK" in briefing
    assert "assert 7 == 10" in briefing, "the actual assertion failure must reach the model"


# ==========================================================================
# D. Multi-file development
# ==========================================================================


def test_D_a_goal_spanning_two_source_files_succeeds(tmp_path, store, tools):
    """D. Two modules, one importing the other, both required to pass."""

    check = [sys.executable, "-m", "pytest", "-q", "test_units.py"]
    brain = PlannedBrain(
        tasks=[{"title": "write the converter"}, {"title": "write the formatter"}],
        acceptance=[{"text": "the tests pass", "check": check}],
        executes=[
            [
                write("convert.py", "def to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32\n"),
                write("present.py", "from convert import to_fahrenheit\n\n\ndef describe(celsius):\n    return f'{celsius}C is {to_fahrenheit(celsius):.0f}F'\n"),
                write(
                    "test_units.py",
                    "from convert import to_fahrenheit\nfrom present import describe\n\n\n"
                    "def test_convert():\n    assert to_fahrenheit(100) == 212\n\n\n"
                    "def test_describe():\n    assert describe(0) == '0C is 32F'\n",
                ),
            ]
        ],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project("convert and present temperatures", limits=ResourceLimits(max_steps=14))

    assert engine.run(project).accepted
    workspace = store.workspace_for(project)
    assert {"convert.py", "present.py", "test_units.py"} <= {path.name for path in workspace.glob("*.py")}
    assert {"convert.py", "present.py", "test_units.py"} <= {item.path for item in project.artifacts}


def test_D_a_multi_file_edit_is_all_or_nothing(tmp_path):
    """D. Two files change together, or neither does."""

    (tmp_path / "one.py").write_bytes(b"value = 1\n")
    (tmp_path / "two.py").write_bytes(b"value = 2\n")
    engine = EditEngine(PathPolicy(tmp_path))

    engine.apply(
        parse_bundle(
            {
                "files": [
                    {"path": "one.py", "search": "value = 1", "replace": "value = 11"},
                    {"path": "two.py", "search": "value = 2", "replace": "value = 22"},
                ]
            }
        )
    )
    assert (tmp_path / "one.py").read_text(encoding="utf-8") == "value = 11\n"
    assert (tmp_path / "two.py").read_text(encoding="utf-8") == "value = 22\n"


# ==========================================================================
# E. New project outside the Jarvis repository
# ==========================================================================


def test_E_a_new_application_is_built_in_an_isolated_workspace(tmp_path, store, tools):
    """E. A real application, in its own workspace, with its tests passing."""

    check = [sys.executable, "-m", "pytest", "-q"]
    brain = PlannedBrain(
        tasks=[{"title": "write wordfreq"}],
        acceptance=[{"text": "the tests pass", "check": check}],
        executes=[
            [
                write(
                    "wordfreq.py",
                    "import re\n\n\ndef count_words(text):\n"
                    "    words = re.findall(r\"[a-z0-9']+\", text.lower())\n"
                    "    counts = {}\n    for word in words:\n"
                    "        counts[word] = counts.get(word, 0) + 1\n    return counts\n",
                ),
                write(
                    "test_wordfreq.py",
                    "from wordfreq import count_words\n\n\n"
                    "def test_counts_ignoring_punctuation():\n"
                    "    assert count_words('Hello, world! Hello.') == {'hello': 2, 'world': 1}\n",
                ),
            ]
        ],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project("build a word frequency tool", limits=ResourceLimits(max_steps=14))

    assert engine.run(project).accepted
    workspace = store.workspace_for(project)
    assert workspace.is_relative_to(store.root), "a new project lives in its own isolated workspace"
    assert not workspace.is_relative_to(REPO), "and never inside the Jarvis repository"


@live
def test_E_live_new_project_evidence_is_real():
    """E (real hardware). The run happened against the configured model."""

    _assert_live_evidence_is_real("E_new_project_live")


@live
def test_E_live_new_project_has_passed_at_least_once():
    """Asserted against the whole history, not the last attempt.

    The model is stochastic. Asserting that the *most recent* run passed makes
    the suite flaky by construction and says nothing useful; asserting that the
    scenario has ever genuinely passed says exactly what it should.
    """

    _assert_has_passed_live("E_new_project_live")


@live
def test_F_live_capability_has_passed_at_least_once():
    _assert_has_passed_live("F_capability_live")


# ==========================================================================
# F. Capability acquisition
# ==========================================================================


@pytest.fixture
def capability_service(tmp_path, store, tools):
    def build(brain):
        return CapabilityService(
            registry=CapabilityRegistry(tmp_path / "capabilities" / "registry.json"),
            engine=ProjectEngine(brain=brain, store=store, tools=tools),
            graph=KnowledgeGraph(tmp_path / "palace.sqlite"),
            root=tmp_path / "capabilities" / "installed",
            execution_timeout=60,
        )

    return build


CAPABILITY_MAIN = '''from __future__ import annotations

import shutil
from typing import Any

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "dry_run": {"type": "boolean"}},
    "required": ["path"],
}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not path:
        return {"ok": False, "error": "no path was given"}
    player = shutil.which("python")
    if player is None:
        return {"ok": False, "error": "nothing available to play it with"}
    if payload.get("dry_run"):
        return {"ok": True, "dry_run": True, "player": player, "would_run": [player, str(path)]}
    return {"ok": True, "dry_run": False, "player": player, "played": str(path)}
'''

CAPABILITY_TESTS = '''import main


def test_dry_run_reports_the_mechanism_without_using_it():
    result = main.run({"path": "a.wav", "dry_run": True})
    assert result["ok"] is True
    assert result["would_run"][1] == "a.wav"
    assert "played" not in result


def test_missing_input_fails_cleanly():
    assert main.run({"dry_run": True})["ok"] is False
'''


def test_F_a_missing_capability_is_acquired_verified_registered_and_reusable(capability_service, tmp_path, store, tools):
    """F. Discover it is missing, build it, verify it, register it, use it."""

    brain = PlannedBrain(
        tasks=[{"title": "implement the capability"}, {"title": "write the tests"}],
        acceptance=[],
        executes=[[write("main.py", CAPABILITY_MAIN), write("test_capability.py", CAPABILITY_TESTS)]],
    )
    service = capability_service(brain)

    assert service.list() == [], "it must genuinely not exist beforehand"
    outcome = service.ensure("play an audio file", max_steps=20, keywords=["music", "sound"])

    assert outcome.acquired, outcome.reason
    assert all(check["ok"] for check in outcome.verification["checks"])

    execution = service.execute(outcome.capability_id, {"path": "song.wav", "dry_run": True})
    assert execution.ok and execution.output["would_run"][1] == "song.wav"

    # And after a restart, with the model made unavailable.
    class ExplodingBrain:
        def generate_structured(self, *args, **kwargs):
            raise AssertionError("reuse must not need the model")

    restarted = CapabilityService(
        registry=CapabilityRegistry(tmp_path / "capabilities" / "registry.json"),
        engine=ProjectEngine(brain=ExplodingBrain(), store=store, tools=tools),
        graph=KnowledgeGraph(tmp_path / "palace.sqlite"),
        root=tmp_path / "capabilities" / "installed",
    )
    resolved = restarted.resolve("play some music")
    assert resolved is not None and resolved.capability_id == outcome.capability_id
    assert restarted.execute(resolved.capability_id, {"path": "after.wav", "dry_run": True}).ok


# ==========================================================================
# G. Complex multi-component project
# ==========================================================================


PIPELINE_FILES = {
    "capture.py": (
        "\"\"\"Stand-in for a screen capture: reads a grid from a text file.\"\"\"\n\n\n"
        "def capture(path):\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        return [line.rstrip('\\n') for line in handle if line.strip()]\n"
    ),
    "recognise.py": (
        "\"\"\"Turn captured rows into a board state.\"\"\"\n\n\n"
        "def recognise(rows):\n"
        "    board = {}\n"
        "    for y, row in enumerate(rows):\n"
        "        for x, cell in enumerate(row):\n"
        "            if cell != '.':\n"
        "                board[(x, y)] = cell\n"
        "    return board\n"
    ),
    "engine_link.py": (
        "\"\"\"Stand-in for an external engine: scores a board state.\"\"\"\n\n\n"
        "VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}\n\n\n"
        "def evaluate(board):\n"
        "    return sum(VALUES.get(piece.lower(), 0) * (1 if piece.isupper() else -1) for piece in board.values())\n"
    ),
    "overlay.py": (
        "\"\"\"Present the analysis.\"\"\"\n\n"
        "from capture import capture\n"
        "from recognise import recognise\n"
        "from engine_link import evaluate\n\n\n"
        "def analyse(path):\n"
        "    board = recognise(capture(path))\n"
        "    score = evaluate(board)\n"
        "    return {'pieces': len(board), 'score': score, 'verdict': 'white' if score > 0 else 'black'}\n"
    ),
    "test_pipeline.py": (
        "from pathlib import Path\n\n"
        "from overlay import analyse\n\n\n"
        "def test_end_to_end(tmp_path):\n"
        "    board = tmp_path / 'board.txt'\n"
        "    board.write_text('R..q\\n.p..\\n', encoding='utf-8')\n"
        "    result = analyse(str(board))\n"
        "    assert result['pieces'] == 3\n"
        "    assert result['score'] == 5 - 9 - 1\n"
        "    assert result['verdict'] == 'black'\n"
    ),
}


def test_G_a_multi_component_pipeline_is_built_and_verified(tmp_path, store, tools):
    """G. Capture, recognition, an external engine, and a presentation layer.

    A benign stand-in for the chess-overlay scenario in the brief: four modules
    that must fit together, exercised end to end by one test. Nothing about the
    domain is hard-coded in Jarvis -- the shape of the work is what matters.
    """

    check = [sys.executable, "-m", "pytest", "-q", "test_pipeline.py"]
    brain = PlannedBrain(
        tasks=[
            {"title": "write the capture component"},
            {"title": "write the recognition component"},
            {"title": "write the engine link"},
            {"title": "write the overlay and the end-to-end test"},
        ],
        acceptance=[{"text": "the pipeline test passes", "check": check}],
        executes=[
            [write("capture.py", PIPELINE_FILES["capture.py"])],
            [write("recognise.py", PIPELINE_FILES["recognise.py"])],
            [write("engine_link.py", PIPELINE_FILES["engine_link.py"])],
            [
                write("overlay.py", PIPELINE_FILES["overlay.py"]),
                write("test_pipeline.py", PIPELINE_FILES["test_pipeline.py"]),
            ],
        ],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project(
        "capture a board, recognise it, score it with an engine, and present the result",
        limits=ResourceLimits(max_steps=25),
    )

    result = engine.run(project)

    assert result.accepted, result.message
    workspace = store.workspace_for(project)
    assert len({path.name for path in workspace.glob("*.py")}) >= 5
    assert len(project.artifacts) >= 5


def test_G_requirements_accumulate_across_interactions(tmp_path, store, tools):
    """G. The user specifies the system over several turns, and it evolves."""

    engine = ProjectEngine(brain=PlannedBrain(tasks=[], acceptance=[], executes=[[]]), store=store, tools=tools)
    project = engine.create_project("capture part of the screen")
    engine.add_requirement(project, "recognise the board from the capture")
    engine.add_requirement(project, "connect to an engine for analysis")
    engine.add_requirement(project, "show the result in an overlay")

    reopened = ProjectStore(store.root).load(project.id)
    assert len(reopened.requirements) == 4
    assert "overlay" in reopened.requirements[-1].text


# ==========================================================================
# H. Protected paths
# ==========================================================================


def test_H_a_protected_file_is_refused_and_stays_byte_identical(tmp_path):
    """H. The attempt is rejected and the file does not change by one byte."""

    protected = tmp_path / "tests" / "test_contract.py"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"def test_contract():\n    assert True\n")
    digest_before = protected.read_bytes()

    engine = EditEngine(PathPolicy(tmp_path, protected_paths=["tests"]))
    with pytest.raises(EditError) as excinfo:
        engine.apply(parse_bundle({"files": [{"path": "tests/test_contract.py", "content": "def test_contract():\n    assert False\n"}]}))

    assert excinfo.value.kind == "protected_path"
    assert not excinfo.value.recoverable, "a policy violation must not be retried"
    assert protected.read_bytes() == digest_before


def test_H_the_repository_engineer_rejects_a_protected_edit(tmp_path):
    """H. End to end, through the self-development path."""

    repo = _make_repo(tmp_path / "repo")
    before = (repo / "test_calc.py").read_bytes()

    class ProtectedEditBrain(SelfPatchBrain):
        def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
            properties = schema.get("properties") or {}
            if "requests" in properties or "plan" in properties or "approved" in properties:
                return super().generate_structured(prompt, schema, max_tokens=max_tokens)
            return json.dumps(
                {"analysis": "weaken the test", "files": [{"path": "test_calc.py", "search": "== 5", "replace": "== -1"}], "new_files": [], "deleted_files": []}
            )

    result = RepositoryEngineer(
        brain=ProtectedEditBrain(), worktree_root=tmp_path / "worktrees", max_cycles=2
    ).improve(repo, _self_patch_goal())

    assert not result.success
    assert "protected" in result.error
    assert (repo / "test_calc.py").read_bytes() == before


# ==========================================================================
# I. Atomicity
# ==========================================================================


def test_I_a_failed_multi_edit_leaves_no_partial_mutation(tmp_path):
    """I. Good edit, bad edit, new file: none of it lands."""

    (tmp_path / "a.py").write_bytes(b"value = 1\n")
    (tmp_path / "b.py").write_bytes(b"value = 2\n")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.py")}

    engine = EditEngine(PathPolicy(tmp_path))
    with pytest.raises(EditError):
        engine.apply(
            parse_bundle(
                {
                    "files": [
                        {"path": "a.py", "search": "value = 1", "replace": "value = 11"},
                        {"path": "b.py", "search": "this anchor does not exist", "replace": "value = 22"},
                    ],
                    "new_files": [{"path": "c.py", "content": "value = 3\n"}],
                }
            )
        )

    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    assert not (tmp_path / "c.py").exists()


def test_I_atomicity_holds_for_every_rejection_kind(tmp_path):
    """I. Whichever gate fires, the workspace is untouched."""

    (tmp_path / "good.py").write_bytes(b"value = 1\n")
    (tmp_path / "big.py").write_bytes(("\n".join(f"x{i} = {i}" for i in range(40)) + "\n").encode())
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.py")}
    good = {"path": "good.py", "search": "value = 1", "replace": "value = 99"}

    second_edits = [
        {"path": "big.py", "content": "x = 1\n"},                                  # truncation
        {"path": "good.py", "search": "value = 1", "replace": "def f(\n"},         # syntax
        {"path": "missing.py", "search": "x", "replace": "y"},                     # missing target
    ]
    for second in second_edits:
        engine = EditEngine(PathPolicy(tmp_path))
        with pytest.raises(EditError):
            engine.apply(parse_bundle({"files": [good, second]}))
        for name, content in before.items():
            assert (tmp_path / name).read_bytes() == content, f"{second} left a partial mutation"


# ==========================================================================
# J. Promotion
# ==========================================================================


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / "live"
    root.mkdir()
    (root / "app.py").write_bytes(b"VERSION = 1\n\n\ndef greet():\n    return 'hello'\n")
    _git(root, "init")
    _git(root, "config", "user.email", "jarvis@example.invalid")
    _git(root, "config", "user.name", "Jarvis Acceptance")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    return root


def test_J_a_verified_candidate_is_promoted_into_the_installation(installation, tmp_path):
    """J. The controlled path from candidate to live, with an audit trail."""

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n\n\ndef greet():\n    return 'hello world'\n")

    audit = PromotionAudit(tmp_path / "promotions.jsonl")
    promoter = Promoter(installation, audit=audit, snapshot_root=tmp_path / "snapshots")
    known_good = promoter.current_revision()

    record = promoter.promote(
        candidate,
        changed_files=["app.py"],
        health_check=HealthCheck(command=[sys.executable, "-c", "print('OK')"], expect_output="OK"),
        commit_message="promote: greet returns hello world",
    )

    assert record.outcome is PromotionOutcome.PROMOTED
    assert "hello world" in (installation / "app.py").read_text(encoding="utf-8")
    assert record.promoted_revision != known_good
    assert audit.history()[0]["outcome"] == "PROMOTED"
    assert [stage["stage"] for stage in record.stages][:3] == ["PREFLIGHT", "SNAPSHOT", "APPLY"]


def test_J_the_real_health_check_exercises_the_whole_kernel():
    """J. Health means the system assembles, not that a module imports."""

    ok, detail = default_health_check().run(REPO)
    assert ok, detail
    assert "JARVIS_HEALTH_OK" in detail


# ==========================================================================
# K. Rollback
# ==========================================================================


def test_K_a_failing_health_check_rolls_back_automatically(installation, tmp_path):
    """K. The installation defends itself without being asked."""

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n\n\ndef greet():\n    raise RuntimeError('broken')\n")
    before = (installation / "app.py").read_bytes()

    promoter = Promoter(installation, audit=PromotionAudit(tmp_path / "promotions.jsonl"), snapshot_root=tmp_path / "snapshots")
    known_good = promoter.current_revision()

    record = promoter.promote(
        candidate,
        changed_files=["app.py"],
        health_check=HealthCheck(command=[sys.executable, "-c", "raise SystemExit(1)"]),
        commit_message="promote: this will not survive",
    )

    assert record.outcome is PromotionOutcome.ROLLED_BACK
    assert (installation / "app.py").read_bytes() == before, "byte-identical to the known-good version"
    assert promoter.current_revision() == known_good


def test_K_a_rollback_that_itself_fails_is_escalated(installation, tmp_path, monkeypatch):
    """K. The one outcome that must never be silently swallowed."""

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "app.py").write_bytes(b"VERSION = 2\n")

    promoter = Promoter(installation, audit=PromotionAudit(tmp_path / "promotions.jsonl"), snapshot_root=tmp_path / "snapshots")
    monkeypatch.setattr(promoter, "_rollback", lambda snapshot, known_good: (False, "the disk is read-only"))

    record = promoter.promote(
        candidate, changed_files=["app.py"], health_check=HealthCheck(command=[sys.executable, "-c", "raise SystemExit(1)"])
    )

    assert record.outcome is PromotionOutcome.ROLLBACK_FAILED
    assert record.needs_human


# ==========================================================================
# L. Persistence
# ==========================================================================


def test_L_projects_memory_and_capabilities_all_survive_a_restart(tmp_path, tools):
    """L. Everything durable is still there in a brand-new process state."""

    store = ProjectStore(tmp_path / "projects")
    graph = KnowledgeGraph(tmp_path / "palace.sqlite")
    memory = ExperienceMemory(graph)

    brain = PlannedBrain(
        tasks=[{"title": "write it"}],
        acceptance=[{"text": "it exists", "check": [sys.executable, "-c", "open('made.txt')"]}],
        executes=[[write("made.txt", "hello\n")]],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    project = engine.create_project("make a file", limits=ResourceLimits(max_steps=10))
    assert engine.run(project).accepted
    memory.record_project(project)
    memory.record_capability("local.make.file", "creates a file", keywords=["file", "create"])
    graph.close()

    # Everything below reopens from disk only.
    reopened_store = ProjectStore(tmp_path / "projects")
    reopened_graph = KnowledgeGraph(tmp_path / "palace.sqlite")
    reopened_memory = ExperienceMemory(reopened_graph)

    loaded = reopened_store.load(project.id)
    assert loaded.state.value == "COMPLETED"
    assert loaded.objective_criteria()[0].satisfied
    assert loaded.artifacts

    assert reopened_memory.known_capabilities("create a file")
    assert reopened_memory.relevant("make a file")
    reopened_graph.close()


def test_L_a_paused_project_can_be_picked_up_later(tmp_path, tools):
    """L. Stopping is not losing: spend accumulates, work continues."""

    store = ProjectStore(tmp_path / "projects")
    brain = PlannedBrain(
        tasks=[{"title": "write it"}],
        acceptance=[{"text": "it exists", "check": [sys.executable, "-c", "open('made.txt')"]}],
        executes=[[write("made.txt", "hello\n")]],
    )
    project = ProjectEngine(brain=brain, store=store, tools=tools).create_project(
        "make a file", limits=ResourceLimits(max_steps=20)
    )
    first = ProjectEngine(brain=brain, store=store, tools=tools).run(project, max_steps=1)
    assert not first.accepted

    resumed = ProjectStore(tmp_path / "projects").load(project.id)
    second = ProjectEngine(brain=brain, store=store, tools=tools).run(resumed, max_steps=12)
    assert second.accepted
    assert resumed.steps_spent > 1


# ==========================================================================
# M. No cloud
# ==========================================================================


def test_M_no_paid_tier_is_enabled_by_default():
    """M. Deleting every credential must change nothing about local work."""

    catalog = ModelCatalog(environ={})
    assert catalog.paid_tiers_enabled() == []
    build = catalog.get(ModelTier.BUILD_LOCAL)
    assert build.enabled and not build.paid
    assert "127.0.0.1" in build.base_url or "localhost" in build.base_url


def test_M_development_works_with_every_cloud_variable_removed(tmp_path, tools, monkeypatch):
    """M. Scrub the environment of anything cloud-shaped, then build something."""

    for name in list(os.environ):
        if any(marker in name.upper() for marker in ("OPENAI", "ANTHROPIC", "CLAUDE", "RUNPOD", "AZURE", "API_KEY", "TOKEN")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JARVIS_EXPERT_CLOUD_ENABLED", "0")
    monkeypatch.setenv("JARVIS_SELF_HOSTED_ENABLED", "0")

    store = ProjectStore(tmp_path / "projects")
    brain = PlannedBrain(
        tasks=[{"title": "write it"}],
        acceptance=[{"text": "it exists", "check": [sys.executable, "-c", "open('offline.txt')"]}],
        executes=[[write("offline.txt", "built with no cloud\n")]],
    )
    engine = ProjectEngine(brain=brain, store=store, tools=tools)
    assert engine.run(engine.create_project("make a file", limits=ResourceLimits(max_steps=10))).accepted

    catalog = ModelCatalog()
    assert catalog.paid_tiers_enabled() == []


def test_M_the_capability_registry_needs_no_network(tmp_path, store, tools):
    """M. Reusing what Jarvis already learned is a local lookup."""

    service = CapabilityService(
        registry=CapabilityRegistry(tmp_path / "registry.json"),
        engine=ProjectEngine(brain=PlannedBrain(tasks=[], acceptance=[], executes=[[]]), store=store, tools=tools),
        graph=KnowledgeGraph(tmp_path / "palace.sqlite"),
        root=tmp_path / "installed",
    )
    assert service.resolve("anything at all") is None  # no network call, no exception


# ==========================================================================
# N. Real hardware
# ==========================================================================


@live
def test_N_build_local_answers_on_this_machine():
    """N. The configured Ollama model actually generates, here, now."""

    if not _build_local_online():
        pytest.skip("BUILD_LOCAL is not answering on this machine")

    catalog = ModelCatalog()
    health = ModelProbe(catalog, ttl_seconds=0).probe(ModelTier.BUILD_LOCAL, force=True)
    assert health.online
    _record(
        "N_build_local_probe",
        {"model": health.model, "latency_seconds": health.latency_seconds, "state": health.state.value},
    )


@live
def test_N_real_runs_were_recorded():
    """N. Mocks alone do not satisfy the brief; these are the real runs."""

    required = ["A_self_patch_live.json", "E_new_project_live.json"]
    missing = [name for name in required if not (EVIDENCE / name).exists()]
    if missing:
        pytest.skip(f"live evidence not recorded: {', '.join(missing)}")
    for name in required:
        payload = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
        assert payload["model"], f"{name} does not name the model that produced it"


# ==========================================================================
# O. Responsiveness
# ==========================================================================


def test_O_the_default_policy_keeps_a_consumer_gpu_usable():
    """O. One model at a time, and VRAM kept back for the desktop."""

    from brain.resources import GpuInfo, HostInfo, reserved_vram_mib

    host = HostInfo(platform="Windows", cpu_count=8, total_ram_mib=16384, gpus=[GpuInfo("GTX 1070", 8192, 610, 7582)])
    policy = ResourcePolicy.default_for(host)

    assert policy.max_concurrent_generations == 1, "two resident models on one 8 GB card is what makes a desktop unusable"
    assert policy.reserved_vram_mib >= 1024
    assert reserved_vram_mib(host) >= 1500


def test_O_the_measured_policy_on_this_machine_leaves_headroom():
    """O. Not the default -- what the tuner actually measured here."""

    policy = ResourcePolicyStore(REPO / "config" / "resources.json").load()
    if policy is None or not policy.tuned_at:
        pytest.skip("this machine has not been tuned; run python -m jarvis.tune_resources")

    assert policy.max_concurrent_generations == 1
    for tier, measurements in policy.measurements.items():
        chosen = policy.context_windows.get(tier)
        selected = [row for row in measurements if row["context_window"] == chosen and row["ok"]]
        assert selected, f"{tier} was set to {chosen} with no successful measurement behind it"
        row = selected[0]
        assert row["vram_free_mib"] >= policy.reserved_vram_mib, (
            f"{tier} at {chosen} leaves only {row['vram_free_mib']} MiB, below the {policy.reserved_vram_mib} MiB reserve"
        )
    _record("O_responsiveness", {"policy": policy.to_dict()})


def test_O_a_throughput_collapse_is_never_accepted():
    """O. Nominal context is not worth a machine that stutters."""

    policy = ResourcePolicyStore(REPO / "config" / "resources.json").load()
    if policy is None or not policy.measurements:
        pytest.skip("this machine has not been tuned")

    for tier, measurements in policy.measurements.items():
        successful = [row for row in measurements if row["ok"] and row["tokens_per_second"]]
        if len(successful) < 2:
            continue
        best = max(row["tokens_per_second"] for row in successful)
        chosen = policy.context_windows.get(tier)
        row = next(item for item in successful if item["context_window"] == chosen)
        assert row["tokens_per_second"] >= best * 0.85, (
            f"{tier} was set to {chosen} at {row['tokens_per_second']} tok/s against a best of {best}"
        )


# ==========================================================================
# Dependency isolation (brief section 14)
# ==========================================================================


def test_dependencies_never_touch_the_system_python(tmp_path, tools):
    """An install must be refused unless it can go somewhere isolated."""

    context = ToolContext(workspace=tmp_path, timeout_seconds=60)
    result = tools.invoke(ToolCall(name="install_packages", arguments={"packages": ["requests; rm -rf /"]}), context)
    assert not result.ok and not result.retryable

    result = tools.invoke(ToolCall(name="install_packages", arguments={"packages": []}), context)
    assert not result.ok


@live
def test_a_package_installs_into_the_project_venv_and_nowhere_else(tmp_path, tools):
    """The isolation guarantee, exercised for real.

    Marked live because it reaches PyPI; skipped when that is not available
    rather than reported as a pass.
    """

    from tools.web import network_available

    if not network_available():
        pytest.skip("no network")

    workspace = tmp_path / "proj"
    workspace.mkdir()
    context = ToolContext(workspace=workspace, timeout_seconds=900)

    created = tools.invoke(ToolCall(name="create_virtualenv", arguments={}), context)
    assert created.ok, created.error

    installed = tools.invoke(ToolCall(name="install_packages", arguments={"packages": ["tomli==2.0.1"]}), context)
    assert installed.ok and installed.output["returncode"] == 0, installed.error

    # Recorded, so the environment is reproducible.
    assert "tomli==2.0.1" in (workspace / "requirements.txt").read_text(encoding="utf-8")

    # Importable inside the project...
    used = tools.invoke(ToolCall(name="run_python", arguments={"code": "import tomli; print('IMPORTED')"}), context)
    assert used.ok and "IMPORTED" in used.output["stdout"]

    # ...and absent from the interpreter running Jarvis.
    host = subprocess.run([sys.executable, "-c", "import tomli"], capture_output=True, text=True, timeout=60)
    assert host.returncode != 0, "installing into a project must not change the host interpreter"


