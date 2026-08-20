from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from brain.providers import LocalTransformersBrainProvider, OpenAICompatibleBrainProvider, OpenAICompatibleConfig
from environments.coding.actions import ActionCandidate, ActionType
from learning.representations.action_encoding import SemanticActionEncoder
from learning.representations.semantic import LightweightLocalEmbeddingProvider
from runtime.action_generator import QwenActionGenerator
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.coding_brain_v03_demo import CodingBrainV03Config, run_coding_brain_v03_demo
from training.coding_curriculum import CodingTaskFactory, DatasetSplit


class FakeLocalBrain:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)

    def ask(self, system_prompt, user_prompt, max_tokens=700, temperature=0.2, top_p=None):
        return "local-ok"


class FakeProvider:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self):
        self.generate_calls = 0
        self.model = torch.nn.Linear(1, 1)

    def generate(self, prompt, *, max_tokens=700, temperature=0.2, top_p=None):
        self.generate_calls += 1
        return json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])

    def generate_coding(self, prompt, *, max_tokens=450, temperature=0.6, top_p=0.9):
        self.generate_calls += 1
        return json.dumps([{"action_type": "RUN_TESTS", "arguments": {}}])

    def health_check(self):
        return {"ok": True}

    def capabilities(self):
        return {"coding": True}


def test_local_provider_works_through_common_interface():
    provider = LocalTransformersBrainProvider(brain=FakeLocalBrain(), model_id="fake-local")
    assert provider.provider_name == "local_transformers"
    assert provider.model_name == "fake-local"
    assert provider.generate("hello") == "local-ok"
    assert provider.generate_coding("code") == "local-ok"
    assert provider.health_check()["ok"] is True


def test_openai_compatible_provider_works_with_mocked_http_server():
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured["authorization"] = self.headers.get("Authorization")
            captured["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))
            body = json.dumps(
                {
                    "choices": [{"message": {"content": "[{\"action_type\":\"RUN_TESTS\",\"arguments\":{}}]"}}],
                    "usage": {"completion_tokens": 7, "total_tokens": 21},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleBrainProvider(
            OpenAICompatibleConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                api_key="secret-test-key",
                model="remote-test",
                timeout=5,
            )
        )
        assert "RUN_TESTS" in provider.generate_coding("generate candidates", max_tokens=20)
        assert captured["payload"]["model"] == "remote-test"
        assert captured["authorization"] == "Bearer secret-test-key"
        assert provider.last_metadata["generated_tokens"] == 7
    finally:
        server.shutdown()


def test_provider_secrets_do_not_enter_observation_or_generator_metadata(tmp_path):
    secret = "super-secret-api-key"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"choices": [{"message": {"content": "[{\"action_type\":\"RUN_TESTS\",\"arguments\":{}}]"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleBrainProvider(
            OpenAICompatibleConfig(base_url=f"http://127.0.0.1:{server.server_port}", api_key=secret, model="remote-test", timeout=5)
        )
        task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
        from environments.coding.environment import CodingEnvironment
        from environments.coding.sandbox_backend import LocalTestSandboxBackend

        observation = CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe()
        generator = QwenActionGenerator(provider, num_candidates=1)
        candidates = generator.generate(task.description, observation)
        assert candidates
        assert secret not in observation.to_text()
        assert secret not in json.dumps(generator.last_generation_metadata)
        assert secret not in json.dumps([candidate.to_dict() for candidate in candidates])
    finally:
        server.shutdown()


def test_switching_providers_requires_no_runtime_changes(tmp_path):
    for provider in [FakeProvider(), FakeProvider()]:
        runtime = JarvisRuntime(
            brain=provider,
            config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, num_action_candidates=1),
            data_dir=tmp_path / provider.provider_name / str(id(provider)),
            mode=RuntimeMode.EVAL,
        )
        assert isinstance(runtime.action_generator, QwenActionGenerator)
        assert isinstance(runtime.text_encoder, LightweightLocalEmbeddingProvider)


def test_foundation_brain_is_not_called_for_embeddings(tmp_path):
    provider = FakeProvider()
    runtime = JarvisRuntime(
        brain=provider,
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, num_action_candidates=1),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    task.max_steps = 1
    runtime.run_episode(task, RuntimeMode.EVAL)
    assert provider.generate_calls == 1
    assert isinstance(runtime.text_encoder, LightweightLocalEmbeddingProvider)
    assert runtime.text_encoder.requests > 0


def test_embedding_provider_batches_candidate_texts_and_cache_works():
    encoder = LightweightLocalEmbeddingProvider(embedding_dim=16)
    action_encoder = SemanticActionEncoder(encoder, action_embedding_dim=8, hidden_dim=8)
    actions = [
        ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a-b", "new": "return a+b"}),
        ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a-b", "new": "return a*b"}),
    ]
    first = action_encoder.raw_features_batch(actions)
    second = action_encoder.raw_features_batch(actions)
    assert first.shape[0] == 2
    assert encoder.batch_requests == 2
    assert encoder.cache_hits > 0
    assert not torch.allclose(first[0], first[1])
    assert torch.allclose(first, second)


def test_timing_and_smoke_mode_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = run_coding_brain_v03_demo(CodingBrainV03Config(smoke=True, train_episodes=2, mock_brain=True, quiet=True, seed=3))
    if not metrics.get("SANDBOX_AVAILABLE"):
        return
    assert metrics["TRAIN_TASK_COUNT"] <= 2
    assert metrics["VALIDATION_TASK_COUNT"] == 1
    assert metrics["HOLDOUT_TASK_COUNT"] == 1
    assert metrics["CONFIG"]["coding_max_tokens"] == 300
    assert "PERFORMANCE" in metrics
    assert "brain_candidate_generation" in metrics["PERFORMANCE"]["timings"]
