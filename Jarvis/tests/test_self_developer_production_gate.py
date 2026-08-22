from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from brain.providers import OpenAICompatibleBrainProvider, OpenAICompatibleConfig, ProviderError
from development.repository_engineer import (
    ModelRequestBudget,
    ProtectionState,
    RepositoryEngineer,
    SelfDeveloperCheckpoint,
    SelfImprovementGoal,
)


class VLLMGateState:
    def __init__(self) -> None:
        self.calls = 0
        self.injected_502 = False
        self.max_observed_tokens = 0
        self.prompts: list[str] = []


def _start_vllm_gate_server(state: VLLMGateState, *, context_window: int = 8192):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state.calls += 1
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = payload["messages"][-1]["content"]
            max_tokens = int(payload.get("max_tokens", 0))
            estimated = int(len(prompt) / 4) + max_tokens + 256
            state.max_observed_tokens = max(state.max_observed_tokens, estimated)
            state.prompts.append(prompt)
            if estimated > context_window:
                self._send(400, {"error": {"message": "maximum context length exceeded"}})
                return
            if "Role: Repository Architect. Decide" in prompt and not state.injected_502:
                state.injected_502 = True
                self._send_raw(502, b"temporary upstream failure")
                return
            if "Return OK." in prompt:
                self._send(200, {"choices": [{"message": {"content": "OK"}}]})
                return
            response = self._response_for(prompt, payload.get("guided_json"))
            self._send(200, {"choices": [{"message": {"content": json.dumps(response)}}], "usage": {"completion_tokens": 20, "total_tokens": estimated}})

        def _response_for(self, prompt: str, schema):
            props = (schema or {}).get("properties") or {}
            if "ok" in props:
                return {"ok": True}
            if "requests" in props:
                return {
                    "requests": [
                        {"tool": "tree", "path": "pkg"},
                        {"tool": "search_text", "query": "add_numbers", "path": "pkg"},
                        {"tool": "read_file", "path": "pkg/mathops.py"},
                        {"tool": "read_file", "path": "tests/test_mathops.py"},
                    ],
                    "notes": "Inspect scoped package and tests.",
                }
            if "plan" in props:
                return {
                    "analysis": "The add_numbers implementation subtracts.",
                    "plan": "Change pkg/mathops.py and preserve public API.",
                    "files_to_change": ["pkg/mathops.py"],
                    "tests_to_run": ["python -m unittest tests.test_mathops"],
                }
            if "approved" in props:
                return {"approved": True, "blocking_findings": [], "optional_findings": [], "recommended_tests": []}
            if "Repository Repairer" in prompt:
                content = "def add_numbers(a, b):\n    return a + b\n\ndef stable_value():\n    return 7\n"
            else:
                content = "def add_numbers(a, b):\n    return a * b\n\ndef stable_value():\n    return 7\n"
            return {"analysis": "Modify math operation.", "files": [{"path": "pkg/mathops.py", "content": content}], "new_files": [], "deleted_files": []}

        def _send(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_raw(self, status: int, body: bytes):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _make_medium_repo(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "mathops.py").write_text("def add_numbers(a, b):\n    return a - b\n\ndef stable_value():\n    return 7\n", encoding="utf-8")
    for idx in range(30):
        (root / "pkg" / f"noise_{idx}.py").write_text(("def noise():\n    return 'x'\n\n" * 80), encoding="utf-8")
    (root / "tests" / "test_mathops.py").write_text(
        "import unittest\nfrom pkg.mathops import add_numbers, stable_value\n\nclass MathTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add_numbers(2, 3), 5)\n    def test_stable(self):\n        self.assertEqual(stable_value(), 7)\n\nif __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    (root / "bench.py").write_text("from pkg.mathops import add_numbers\nprint('CAPABILITY_ACQUISITION_SUCCESS_RATE:', 0.667 if add_numbers(2, 3) == 5 else 0.0)\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "config", "user.email", "jarvis@example.invalid"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "config", "user.name", "Jarvis Test"], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, timeout=20)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, capture_output=True, text=True, timeout=20)
    return root


def test_self_developer_offline_8192_context_end_to_end(tmp_path):
    repo = _make_medium_repo(tmp_path / "repo")
    state = VLLMGateState()
    server = _start_vllm_gate_server(state)
    try:
        provider = OpenAICompatibleBrainProvider(
            OpenAICompatibleConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                api_key="",
                model="qwen-test",
                timeout=5,
                retries=2,
                backoff_seconds=0.01,
                context_window=8192,
            )
        )
        checkpoint = SelfDeveloperCheckpoint(tmp_path / "run")
        goal = SelfImprovementGoal(
            objective="Improve capability acquisition reliability in math operations.",
            allowed_paths=["pkg"],
            protected_paths=["tests"],
            tests=[["python", "-m", "unittest", "tests.test_mathops"]],
            full_tests=[["python", "-m", "unittest", "tests.test_mathops"]],
            benchmarks=[["python", "bench.py"]],
            require_benchmark_improvement=True,
            metric_name="CAPABILITY_ACQUISITION_SUCCESS_RATE",
            metric_minimums={"CAPABILITY_ACQUISITION_SUCCESS_RATE": 0.667},
        )
        engineer = RepositoryEngineer(
            brain=provider,
            worktree_root=tmp_path / "external_worktrees",
            checkpoint=checkpoint,
            context_budget=ModelRequestBudget(context_window=8192),
            max_cycles=3,
        )

        preflight = engineer.preflight()
        result = engineer.improve(repo, goal)

        assert preflight["structured_generation"] == "OK"
        assert state.injected_502 is True
        assert state.max_observed_tokens <= 8192
        assert result.status == "SELF_DEVELOPMENT_CANDIDATE_READY"
        assert result.benchmarks_before[0].metrics["CAPABILITY_ACQUISITION_SUCCESS_RATE"] == 0.0
        assert result.benchmarks_after[0].metrics["CAPABILITY_ACQUISITION_SUCCESS_RATE"] == 0.667
        assert result.tests and all(item.success for item in result.tests)
        assert result.full_tests and all(item.success for item in result.full_tests)
        assert result.review and result.review.approved
        assert result.protection_state == ProtectionState.PRISTINE
        assert any("tree\", \"path\": \"pkg" not in prompt for prompt in state.prompts)
        assert checkpoint.state.get("BENCHMARK_BEFORE_COMPLETE")
        assert checkpoint.state.get("PLAN_COMPLETE")
        assert checkpoint.state.get("EVALUATION_COMPLETE")
    finally:
        server.shutdown()


class PauseAfterInvestigationBrain:
    provider_name = "pause_after_investigation"
    model_name = "mock"

    def __init__(self) -> None:
        self.plan_calls = 0

    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        props = schema.get("properties") or {}
        if "requests" in props:
            return json.dumps({"requests": [{"tool": "read_file", "path": "pkg/mathops.py"}]})
        if "plan" in props:
            self.plan_calls += 1
            raise ProviderError(kind="server_error", status=502, message="temporary", model="mock", attempt=1)
        return json.dumps({"ok": True})


def test_self_developer_resume_skips_completed_before_benchmark(tmp_path):
    repo = _make_medium_repo(tmp_path / "repo")
    checkpoint = SelfDeveloperCheckpoint(tmp_path / "run")
    goal = SelfImprovementGoal(
        objective="Improve mathops.",
        allowed_paths=["pkg"],
        tests=[["python", "-m", "unittest", "tests.test_mathops"]],
        benchmarks=[["python", "bench.py"]],
    )
    first = RepositoryEngineer(
        brain=PauseAfterInvestigationBrain(),
        worktree_root=tmp_path / "external",
        checkpoint=checkpoint,
        context_budget=ModelRequestBudget(context_window=8192),
        resume_command="python -m jarvis.self_develop --resume run",
    ).improve(repo, goal)

    assert first.status == "SELF_DEVELOPMENT_PAUSED"
    before_events = len([event for event in checkpoint.state["events"] if event["stage"] == "BENCHMARK_BEFORE_COMPLETE"])

    state = VLLMGateState()
    server = _start_vllm_gate_server(state)
    try:
        provider = OpenAICompatibleBrainProvider(
            OpenAICompatibleConfig(base_url=f"http://127.0.0.1:{server.server_port}", api_key="", model="qwen-test", timeout=5, retries=2, backoff_seconds=0.01)
        )
        second = RepositoryEngineer(
            brain=provider,
            worktree_root=tmp_path / "external",
            checkpoint=SelfDeveloperCheckpoint(tmp_path / "run"),
            context_budget=ModelRequestBudget(context_window=8192),
            max_cycles=3,
        ).improve(repo, goal)

        assert second.success
        after_events = len([event for event in checkpoint.load()["events"] if event["stage"] == "BENCHMARK_BEFORE_COMPLETE"])
        assert after_events == before_events
    finally:
        server.shutdown()


class WriteProtectedPathBrain:
    """Deterministic brain that always proposes writing into tests/test_mathops.py."""

    provider_name = "write_protected_path"
    model_name = "mock"

    def generate_structured(self, prompt, schema, *, max_tokens=4000, temperature=0.2, top_p=0.9):
        props = schema.get("properties") or {}
        if "requests" in props:
            return json.dumps({"requests": []})
        if "plan" in props:
            return json.dumps({"analysis": "a", "plan": "p", "files_to_change": ["tests/test_mathops.py"]})
        if "files" in props:
            return json.dumps(
                {
                    "analysis": "rewrite the test",
                    "files": [{"path": "tests/test_mathops.py", "content": "attacker controlled content\n"}],
                    "new_files": [],
                    "deleted_files": [],
                }
            )
        if "approved" in props:
            return json.dumps({"approved": True, "blocking_findings": [], "optional_findings": [], "recommended_tests": []})
        return json.dumps({"ok": True})


def test_self_developer_resume_keeps_original_goal_protected_paths(tmp_path):
    """A resumed run must keep enforcing the ORIGINAL goal's protected_paths,
    even if the caller reconstructs a differently-configured SelfImprovementGoal
    (e.g. forgetting --protected-path) for the --resume invocation."""
    repo = _make_medium_repo(tmp_path / "repo")
    checkpoint = SelfDeveloperCheckpoint(tmp_path / "run")
    original_goal = SelfImprovementGoal(
        objective="Improve mathops.",
        allowed_paths=["."],
        protected_paths=["tests"],
        tests=[["python", "-m", "unittest", "tests.test_mathops"]],
    )
    first = RepositoryEngineer(
        brain=PauseAfterInvestigationBrain(),
        worktree_root=tmp_path / "external",
        checkpoint=checkpoint,
        context_budget=ModelRequestBudget(context_window=8192),
        resume_command="python -m jarvis.self_develop --resume run",
    ).improve(repo, original_goal)
    assert first.status == "SELF_DEVELOPMENT_PAUSED"

    # Simulate an operator resuming without repeating --protected-path.
    resumed_goal = SelfImprovementGoal(objective="Improve mathops.", allowed_paths=["."])
    second = RepositoryEngineer(
        brain=WriteProtectedPathBrain(),
        worktree_root=tmp_path / "external",
        checkpoint=SelfDeveloperCheckpoint(tmp_path / "run"),
        context_budget=ModelRequestBudget(context_window=8192),
        max_cycles=1,
    ).improve(repo, resumed_goal)

    assert not second.success
    assert "protected repository path" in second.error
    worktree = Path(second.worktree)
    assert (worktree / "tests" / "test_mathops.py").read_text(encoding="utf-8") == (
        repo / "tests" / "test_mathops.py"
    ).read_text(encoding="utf-8")


def test_self_developer_protected_directory_detects_added_modified_deleted(tmp_path):
    repo = _make_medium_repo(tmp_path / "repo")
    goal = SelfImprovementGoal(objective="x", protected_paths=["tests"])
    engineer = RepositoryEngineer(brain=PauseAfterInvestigationBrain(), worktree_root=tmp_path / "external")
    manifest = engineer._hash_protected(repo, goal.protected_paths)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "clone", str(repo), str(worktree)], capture_output=True, text=True, timeout=20)

    assert engineer._protected_state(repo, worktree, manifest) == ProtectionState.PRISTINE
    (worktree / "tests" / "test_mathops.py").write_text("modified\n", encoding="utf-8")
    assert engineer._protected_state(repo, worktree, manifest) == ProtectionState.MODIFIED
    (worktree / "tests" / "test_mathops.py").write_text((repo / "tests" / "test_mathops.py").read_text(encoding="utf-8"), encoding="utf-8")
    (worktree / "tests" / "new_test.py").write_text("added\n", encoding="utf-8")
    assert engineer._protected_state(repo, worktree, manifest) == ProtectionState.MODIFIED
    (worktree / "tests" / "new_test.py").unlink()
    (worktree / "tests" / "test_mathops.py").unlink()
    assert engineer._protected_state(repo, worktree, manifest) == ProtectionState.MODIFIED
