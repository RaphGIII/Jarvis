from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from brain.providers import LocalTransformersBrainProvider, OpenAICompatibleBrainProvider, OpenAICompatibleConfig, ProviderError
from environments.coding.actions import ActionCandidate, ActionType
from learning.representations.action_encoding import SemanticActionEncoder
from learning.representations.semantic import LightweightLocalEmbeddingProvider
from runtime.action_generator import QwenActionGenerator, action_candidate_json_schema, fallback_candidates, parse_action_candidates
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig, ScoredAction
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


class _StubHandler(BaseHTTPRequestHandler):
    """Base for the fake inference servers in this module.

    Two details that are easy to get wrong and produced intermittent failures
    on Windows before they were fixed:

    ``protocol_version = "HTTP/1.1"`` keeps the connection open after a
    response, so a client can still read an error body after the handler
    returns. Under the HTTP/1.0 default the server tears the socket down first
    and the client's read of a 4xx body fails with WinError 10053 -- which made
    a test about error *classification* fail for reasons of socket timing.

    ``drain()`` reads the request body even when the handler ignores it. With
    keep-alive, bytes left unread stay in the buffer and are parsed as the
    start of the next request, which breaks any test that makes two calls on
    one connection.
    """

    protocol_version = "HTTP/1.1"

    def drain(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return None


def test_local_provider_works_through_common_interface():
    provider = LocalTransformersBrainProvider(brain=FakeLocalBrain(), model_id="fake-local")
    assert provider.provider_name == "local_transformers"
    assert provider.model_name == "fake-local"
    assert provider.generate("hello") == "local-ok"
    assert provider.generate_coding("code") == "local-ok"
    assert provider.health_check()["ok"] is True
    assert provider.capabilities()["structured_generation"] is False
    try:
        provider.generate_structured("hello", action_candidate_json_schema())
    except NotImplementedError:
        pass
    else:
        raise AssertionError("local provider should not claim guided JSON support")


def test_qwen_action_generator_falls_back_when_local_provider_has_no_guided_json(tmp_path):
    provider = LocalTransformersBrainProvider(brain=FakeLocalBrain(), model_id="fake-local")
    provider.brain.ask = lambda system_prompt, user_prompt, max_tokens=700, temperature=0.2, top_p=None: json.dumps(
        [{"action_type": "RUN_TESTS", "arguments": {}}]
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    observation = CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe()
    generator = QwenActionGenerator(provider, num_candidates=1)
    candidates = generator.generate(task.description, observation)
    assert candidates[0].action_type == ActionType.RUN_TESTS
    assert generator.last_generation_metadata["structured_generation_requests"] == 0


def test_openai_compatible_provider_works_with_mocked_http_server():
    captured = {}

    class Handler(_StubHandler):
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


def test_openai_compatible_provider_sends_guided_json_schema_to_server():
    captured = {}

    class Handler(_StubHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured["payload"] = json.loads(self.rfile.read(length).decode("utf-8"))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "{\"candidates\":[{\"action_type\":\"RUN_TESTS\",\"arguments\":{}}]}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 11, "total_tokens": 31},
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
            OpenAICompatibleConfig(base_url=f"http://127.0.0.1:{server.server_port}", api_key="", model="remote-test", timeout=5)
        )
        schema = action_candidate_json_schema()
        response = provider.generate_structured("generate candidates", schema, max_tokens=30)
        assert "RUN_TESTS" in response
        assert captured["payload"]["guided_json"] == schema
        assert provider.capabilities()["structured_generation"] is True
        assert provider.last_metadata["finish_reason"] == "stop"
        assert provider.last_metadata["generated_tokens"] == 11
        assert provider.last_metadata["total_tokens"] == 31
        assert provider.last_metadata["attempts"] == 1
    finally:
        server.shutdown()


def test_openai_compatible_provider_classifies_context_overflow_without_retry():
    calls = {"count": 0}

    class Handler(_StubHandler):
        def do_POST(self):
            self.drain()
            calls["count"] += 1
            body = b'{"error":{"message":"maximum context length exceeded"}}'
            self.send_response(400)
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
            OpenAICompatibleConfig(base_url=f"http://127.0.0.1:{server.server_port}", api_key="secret", model="remote-test", timeout=5, retries=3)
        )
        try:
            provider.generate_structured("too large", {"type": "object"}, max_tokens=100)
            raise AssertionError("expected ProviderError")
        except ProviderError as exc:
            assert exc.kind == "context_overflow"
            assert exc.status == 400
            assert exc.attempt == 1
            assert "secret" not in exc.message
        assert calls["count"] == 1
    finally:
        server.shutdown()


def test_openai_compatible_provider_retries_transient_502():
    calls = {"count": 0}

    class Handler(_StubHandler):
        def do_POST(self):
            self.drain()
            calls["count"] += 1
            if calls["count"] == 1:
                body = b"temporary upstream failure"
                self.send_response(502)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8")
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
            OpenAICompatibleConfig(base_url=f"http://127.0.0.1:{server.server_port}", api_key="", model="remote-test", timeout=5, retries=2, backoff_seconds=0.01)
        )
        assert provider.generate_structured("ok", {"type": "object"}, max_tokens=20) == "{\"ok\": true}"
        assert calls["count"] == 2
    finally:
        server.shutdown()


def test_provider_secrets_do_not_enter_observation_or_generator_metadata(tmp_path):
    secret = "super-secret-api-key"

    class Handler(_StubHandler):
        def do_POST(self):
            self.drain()
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


def test_qwen_action_generator_backfills_partial_candidate_sets(tmp_path):
    provider = FakeProvider()
    generator = QwenActionGenerator(provider, num_candidates=3)
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    observation = CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe()
    candidates = generator.generate(task.description, observation)
    assert len(candidates) == 3
    assert generator.last_generation_metadata["valid_candidates"] == 1
    assert generator.last_generation_metadata["fallback_backfill_count"] == 2
    assert len({json.dumps(candidate.to_dict(), sort_keys=True) for candidate in candidates}) == 3
    assert "4-8" not in generator._prompt(task.description, observation)
    assert "Return exactly 3 candidates" in generator._prompt(task.description, observation)
    assert "do not guess a PATCH_FILE old string" in generator._prompt(task.description, observation)


def test_fallback_candidates_prioritize_unread_implementation_source(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    observation = CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe()
    candidates = fallback_candidates(observation)
    assert candidates[0].action_type == ActionType.READ_FILE
    assert candidates[0].arguments["path"] == "solution.py"
    assert all(candidate.arguments.get("path") != "test_public.py" for candidate in candidates)
    assert [candidate.action_type for candidate in candidates[:2]] == [ActionType.READ_FILE, ActionType.RUN_TESTS]


def test_fallback_candidates_after_tests_prefer_read_over_repeated_tests(tmp_path):
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    env = CodingEnvironment(task, backend=LocalTestSandboxBackend(), terminate_on_public_success=False)
    env.step(ActionCandidate(ActionType.RUN_TESTS))
    candidates = fallback_candidates(env.observe())
    assert candidates[0].action_type == ActionType.READ_FILE
    assert candidates[0].arguments["path"] == "solution.py"
    assert ActionType.LIST_FILES not in [candidate.action_type for candidate in candidates[:2]]


def test_parse_diagnostics_count_parse_and_schema_errors():
    parsed, metadata = parse_action_candidates("not json", return_metadata=True)
    assert parsed == []
    assert metadata["parse_error_count"] == 1
    parsed, metadata = parse_action_candidates('[{"action_type":"NOPE"}, 3]', return_metadata=True)
    assert parsed == []
    assert metadata["schema_invalid_candidates"] == 2


def test_parser_accepts_legacy_top_level_list_and_path_fields():
    parsed = parse_action_candidates('[{"action_type":"READ_FILE","path":"solution.py","reason":"inspect"}]')
    assert len(parsed) == 1
    assert parsed[0].action_type == ActionType.READ_FILE
    assert parsed[0].arguments["path"] == "solution.py"


def test_parser_enforces_action_specific_required_arguments():
    parsed, metadata = parse_action_candidates(
        json.dumps(
            {
                "candidates": [
                    {"action_type": "READ_FILE", "arguments": {}},
                    {"action_type": "PATCH_FILE", "arguments": {"path": "solution.py", "old": "x"}},
                    {"action_type": "RUN_TESTS", "arguments": {"path": "test_public.py"}},
                    {"action_type": "SEARCH_TEXT", "arguments": {"query": "needle"}},
                ]
            }
        ),
        return_metadata=True,
    )
    assert [candidate.action_type for candidate in parsed] == [ActionType.SEARCH_TEXT]
    assert metadata["schema_invalid_candidates"] == 3


def test_guided_json_schema_uses_only_vllm_compatible_structural_keywords():
    unsupported_keywords = {
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "format",
        "multipleOf",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in unsupported_keywords
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    schema = action_candidate_json_schema()
    walk(schema)
    assert schema["required"] == ["candidates"]


class SequentialGenerationProvider:
    provider_name = "sequential"
    model_name = "sequential-model"

    def __init__(self, responses, structured: bool = False, fail_structured: bool = False):
        self.responses = list(responses)
        self.generate_calls = 0
        self.structured_calls = 0
        self.structured = structured
        self.fail_structured = fail_structured
        self.last_metadata = {}

    def capabilities(self):
        return {"coding": True, "structured_generation": self.structured}

    def generate_coding(self, prompt, *, max_tokens=450, temperature=0.6, top_p=0.9):
        self.generate_calls += 1
        self.last_metadata = {"finish_reason": "stop", "generated_tokens": 5, "total_tokens": 10, "attempts": 1}
        return self.responses[min(self.generate_calls + self.structured_calls - 1, len(self.responses) - 1)]

    def generate_structured(self, prompt, schema, *, max_tokens=450, temperature=0.6, top_p=0.9):
        self.structured_calls += 1
        if self.fail_structured:
            raise RuntimeError("guided JSON unavailable")
        assert schema["required"] == ["candidates"]
        self.last_metadata = {"finish_reason": "length", "generated_tokens": 9, "total_tokens": 18, "attempts": 1}
        return self.responses[min(self.generate_calls + self.structured_calls - 1, len(self.responses) - 1)]


def _holdout_observation(tmp_path):
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    return task, CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe()


def test_qwen_generator_regenerates_once_after_malformed_first_response(tmp_path):
    valid = json.dumps({"candidates": [{"action_type": "RUN_TESTS", "arguments": {}}]})
    provider = SequentialGenerationProvider(["not json", valid])
    task, observation = _holdout_observation(tmp_path)
    generator = QwenActionGenerator(provider, num_candidates=1)
    candidates = generator.generate(task.description, observation)
    assert candidates[0].action_type == ActionType.RUN_TESTS
    assert provider.generate_calls == 2
    assert generator.last_generation_metadata["candidate_regeneration_count"] == 1
    assert generator.last_generation_metadata["candidate_regeneration_success_count"] == 1


def test_qwen_generator_regeneration_failure_is_bounded_and_not_cached(tmp_path):
    provider = SequentialGenerationProvider(["not json", "{\"candidates\":[{\"action_type\":\"NOPE\"}]}"])
    task, observation = _holdout_observation(tmp_path)
    generator = QwenActionGenerator(provider, num_candidates=2)
    first = generator.generate(task.description, observation)
    assert len(first) == 2
    assert provider.generate_calls == 2
    assert generator.last_generation_metadata["candidate_regeneration_count"] == 1
    assert generator.last_generation_metadata["candidate_regeneration_success_count"] == 0
    assert generator.last_generation_metadata["zero_valid_qwen_candidates"] == 1
    second = generator.generate(task.description, observation)
    assert len(second) == 2
    assert provider.generate_calls == 4
    assert generator.last_generation_metadata["cache_hit"] is False


def test_qwen_generator_valid_generation_is_cached(tmp_path):
    valid = json.dumps({"candidates": [{"action_type": "RUN_TESTS", "arguments": {}}]})
    provider = SequentialGenerationProvider([valid])
    task, observation = _holdout_observation(tmp_path)
    generator = QwenActionGenerator(provider, num_candidates=1)
    assert generator.generate(task.description, observation)
    assert generator.generate(task.description, observation)
    assert provider.generate_calls == 1
    assert generator.last_generation_metadata["cache_hit"] is True


def test_qwen_generator_uses_structured_generation_and_falls_back_on_failure(tmp_path):
    valid = json.dumps({"candidates": [{"action_type": "RUN_TESTS", "arguments": {}}]})
    task, observation = _holdout_observation(tmp_path)

    structured_provider = SequentialGenerationProvider([valid], structured=True)
    structured = QwenActionGenerator(structured_provider, num_candidates=1)
    assert structured.generate(task.description, observation)[0].action_type == ActionType.RUN_TESTS
    assert structured_provider.structured_calls == 1
    assert structured.last_generation_metadata["structured_generation_requests"] == 1
    assert structured.last_generation_metadata["generation_length_truncation_count"] == 1

    fallback_provider = SequentialGenerationProvider([valid], structured=True, fail_structured=True)
    fallback = QwenActionGenerator(fallback_provider, num_candidates=1)
    assert fallback.generate(task.description, observation)[0].action_type == ActionType.RUN_TESTS
    assert fallback_provider.structured_calls == 1
    assert fallback_provider.generate_calls == 1
    assert fallback.last_generation_metadata["structured_generation_failures"] == 1


def test_qwen_generation_metadata_counts_zero_valid_candidates(tmp_path):
    class InvalidProvider(FakeProvider):
        def generate_coding(self, prompt, *, max_tokens=450, temperature=0.6, top_p=0.9):
            self.generate_calls += 1
            return json.dumps([{"action_type": "NOPE"}])

    generator = QwenActionGenerator(InvalidProvider(), num_candidates=2)
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    observation = CodingEnvironment(task, backend=LocalTestSandboxBackend()).observe()
    candidates = generator.generate(task.description, observation)
    assert len(candidates) == 2
    assert generator.last_generation_metadata["zero_valid_qwen_candidates"] == 1
    assert generator.last_generation_metadata["schema_invalid_candidates"] == 2
    assert generator.last_generation_metadata["candidate_regeneration_count"] == 1
    assert generator.last_generation_metadata["candidate_regeneration_success_count"] == 0
    assert generator.last_generation_metadata["fallback_backfill_count"] == 2


def test_runtime_feasibility_and_cold_start_weight(tmp_path):
    runtime = JarvisRuntime(
        brain=FakeProvider(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, warmup_experiences=5, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    observation = runtime.start_task(task, RuntimeMode.EVAL)
    assert runtime._learned_weight() == 0.0
    finish = ActionCandidate(ActionType.FINISH, confidence=1.0, estimated_cost=0.1)
    assert runtime._validate_candidate_feasibility(finish, observation) == (False, "public tests are not passing")
    missing = ActionCandidate(ActionType.READ_FILE, {"path": "missing.py"}, confidence=1.0)
    assert runtime._validate_candidate_feasibility(missing, observation) == (False, "path does not exist")
    protected = ActionCandidate(ActionType.WRITE_FILE, {"path": "test_public.py", "content": ""}, confidence=1.0)
    assert runtime._validate_candidate_feasibility(protected, observation) == (False, "path is protected")


def test_runtime_requires_read_before_patch_or_existing_overwrite(tmp_path):
    runtime = JarvisRuntime(
        brain=FakeProvider(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    observation = runtime.start_task(task, RuntimeMode.EVAL)
    patch = ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a - b", "new": "return a + b"})
    assert runtime._validate_candidate_feasibility(patch, observation) == (False, "read file before patching")
    overwrite = ActionCandidate(ActionType.WRITE_FILE, {"path": "solution.py", "content": "x = 1\n"})
    assert runtime._validate_candidate_feasibility(overwrite, observation) == (False, "read file before overwriting")
    create = ActionCandidate(ActionType.WRITE_FILE, {"path": "notes.py", "content": "x = 1\n"})
    assert runtime._validate_candidate_feasibility(create, observation) == (True, "")
    runtime.environment.step(ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}))
    observation = runtime.environment.observe()
    assert runtime._validate_candidate_feasibility(patch, observation) == (True, "")
    bad_patch = ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "not in file", "new": "x"})
    assert runtime._validate_candidate_feasibility(bad_patch, observation) == (False, "old text unavailable")


def test_profiler_breakdown_excludes_nested_totals():
    from runtime.profiling import PerformanceProfiler

    profiler = PerformanceProfiler()
    profiler.add_time("brain_candidate_generation", 4.0)
    profiler.add_time("docker_execution", 1.0)
    profiler.add_time("total_step", 6.0)
    profiler.add_time("total_episode", 10.0)
    summary = profiler.summary()
    assert summary["wall_time_seconds"] == 10.0
    assert summary["breakdown"]["brain_candidate_generation"]["percent"] == 40.0
    assert summary["breakdown"]["docker_execution"]["percent"] == 10.0
    assert summary["breakdown"]["other"]["percent"] == 50.0
    assert summary["timings"]["total_step"]["percent"] == 0.0


def test_public_run_failure_is_classified_as_infrastructure(tmp_path):
    from environments.coding.environment import CodingEnvironment
    from environments.coding.sandbox_backend import DisabledSandboxBackend

    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.HOLDOUT, 1)[0]
    env = CodingEnvironment(task, backend=DisabledSandboxBackend(), terminate_on_public_success=False)
    result = env.step(ActionCandidate(ActionType.RUN_TESTS))
    assert result.action_result.data["failure_kind"] == "infrastructure"
    assert result.done is False


def _scored(action_type: ActionType, score: float, feasible: bool) -> ScoredAction:
    return ScoredAction(
        candidate=ActionCandidate(action_type),
        score=score,
        policy_score=0.0,
        q_value=0.0,
        predicted_reward=0.0,
        expected_information_gain=0.0,
        risk=0.0,
        uncertainty=0.0,
        novelty=0.0,
        feasible=feasible,
        feasibility_reason="" if feasible else "infeasible",
    )


def test_select_epsilon_exploration_never_selects_infeasible_when_feasible_exists(tmp_path):
    runtime = JarvisRuntime(
        config=JarvisRuntimeConfig(train_exploration_epsilon=1.0, epsilon_min=1.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.TRAIN,
    )
    scored = [_scored(ActionType.PATCH_FILE, 999.0, False), _scored(ActionType.READ_FILE, 0.0, True)]
    for _ in range(25):
        assert runtime._select(scored).feasible is True


def test_select_score_sampling_never_selects_infeasible_when_feasible_exists(tmp_path):
    runtime = JarvisRuntime(
        config=JarvisRuntimeConfig(train_exploration_epsilon=0.0, epsilon_min=0.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.TRAIN,
    )
    scored = [_scored(ActionType.PATCH_FILE, 1000.0, False), _scored(ActionType.READ_FILE, -1000.0, True)]
    for _ in range(25):
        assert runtime._select(scored).feasible is True


def test_select_eval_never_selects_infeasible_when_feasible_exists(tmp_path):
    runtime = JarvisRuntime(data_dir=tmp_path / "runtime", mode=RuntimeMode.EVAL)
    scored = [_scored(ActionType.PATCH_FILE, 1000.0, False), _scored(ActionType.RUN_TESTS, -1.0, True)]
    selected = runtime._select(scored)
    assert selected.feasible is True
    assert selected.candidate.action_type == ActionType.RUN_TESTS


def test_select_all_infeasible_does_not_crash(tmp_path):
    runtime = JarvisRuntime(data_dir=tmp_path / "runtime", mode=RuntimeMode.EVAL)
    scored = [_scored(ActionType.PATCH_FILE, 2.0, False), _scored(ActionType.FINISH, 1.0, False)]
    selected = runtime._select(scored)
    assert selected.feasible is False
    assert selected.candidate.action_type == ActionType.PATCH_FILE


class InfeasiblePatchGenerator:
    last_generation_metadata = {"valid_candidates": 2}

    def generate(self, goal, observation):
        return [
            ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "guessed old", "new": "new"}),
            ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "another guess", "new": "new"}),
        ]


class FixedSequenceGenerator:
    last_generation_metadata = {}

    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def generate(self, goal, observation):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return [action]


def _runtime_with_actions(tmp_path, actions, mode=RuntimeMode.EVAL):
    from environments.coding.sandbox_backend import LocalTestSandboxBackend

    return JarvisRuntime(
        action_generator=FixedSequenceGenerator(actions),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=20, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=mode,
        sandbox_backend=LocalTestSandboxBackend(),
    )


def _score_action_types(runtime, observation, actions):
    features = runtime._observation_features(observation)
    latent = runtime._encode_features(features)
    return runtime._score_candidates(latent, actions, observation)


def test_runtime_patch_makes_run_tests_dominate_until_tests_run(tmp_path):
    runtime = _runtime_with_actions(
        tmp_path,
        [
            ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}),
            ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a - b", "new": "return a + b"}),
        ],
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.start_task(task, RuntimeMode.EVAL)
    runtime.step(task.description)
    runtime.step(task.description)
    observation = runtime.environment.observe()
    scored = _score_action_types(
        runtime,
        observation,
        [
            ActionCandidate(ActionType.RUN_TESTS, confidence=0.2, estimated_cost=2.0),
            ActionCandidate(ActionType.LIST_FILES, confidence=1.0, estimated_cost=0.1),
            ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}, confidence=1.0, estimated_cost=0.1),
        ],
    )
    assert runtime._code_changed_since_last_test() is True
    assert scored[0].candidate.action_type == ActionType.RUN_TESTS


def test_runtime_write_file_makes_run_tests_dominate_until_tests_run(tmp_path):
    runtime = _runtime_with_actions(
        tmp_path,
        [ActionCandidate(ActionType.WRITE_FILE, {"path": "notes.py", "content": "changed = True\n"})],
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.start_task(task, RuntimeMode.EVAL)
    runtime.step(task.description)
    observation = runtime.environment.observe()
    scored = _score_action_types(
        runtime,
        observation,
        [
            ActionCandidate(ActionType.RUN_TESTS, confidence=0.2, estimated_cost=2.0),
            ActionCandidate(ActionType.LIST_FILES, confidence=1.0, estimated_cost=0.1),
        ],
    )
    assert runtime._code_changed_since_last_test() is True
    assert scored[0].candidate.action_type == ActionType.RUN_TESTS


def test_runtime_repeated_run_tests_without_code_change_is_penalized(tmp_path):
    runtime = _runtime_with_actions(tmp_path, [ActionCandidate(ActionType.RUN_TESTS)])
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.start_task(task, RuntimeMode.EVAL)
    runtime.step(task.description)
    observation = runtime.environment.observe()
    assert runtime._stagnation_penalty(ActionCandidate(ActionType.RUN_TESTS), observation) > 0.0


def test_runtime_retest_after_modification_is_not_penalized(tmp_path):
    runtime = _runtime_with_actions(
        tmp_path,
        [
            ActionCandidate(ActionType.READ_FILE, {"path": "solution.py"}),
            ActionCandidate(ActionType.PATCH_FILE, {"path": "solution.py", "old": "return a - b", "new": "return a + b"}),
        ],
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    runtime.start_task(task, RuntimeMode.EVAL)
    runtime.step(task.description)
    runtime.step(task.description)
    observation = runtime.environment.observe()
    assert runtime._code_changed_since_last_test() is True
    assert runtime._stagnation_penalty(ActionCandidate(ActionType.RUN_TESTS), observation) == 0.0


def test_runtime_all_generated_infeasible_uses_safe_feasible_fallback(tmp_path):
    runtime = JarvisRuntime(
        action_generator=InfeasiblePatchGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    task.max_steps = 1
    metrics = runtime.run_episode(task, RuntimeMode.EVAL)
    selected = runtime.state.trajectory.transitions[0].metadata["scoring"]
    assert metrics["steps"] == 1
    assert selected["feasible"] is True
    assert selected["candidate"]["action_type"] == "READ_FILE"
    assert runtime.profiler.counters["post_feasibility_fallback_count"] > 0


def test_runtime_never_executes_infeasible_patch(tmp_path):
    runtime = JarvisRuntime(
        action_generator=InfeasiblePatchGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, train_exploration_epsilon=0.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    original = (task.workspace / "solution.py").read_text(encoding="utf-8")
    task.max_steps = 1
    runtime.run_episode(task, RuntimeMode.EVAL)
    action = runtime.state.trajectory.transitions[0].metadata["action"]
    assert action["action_type"] != "PATCH_FILE"
    assert (task.workspace / "solution.py").read_text(encoding="utf-8") == original


def test_runtime_training_epsilon_cannot_bypass_post_feasibility_fallback(tmp_path):
    runtime = JarvisRuntime(
        action_generator=InfeasiblePatchGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10, train_exploration_epsilon=1.0, epsilon_min=1.0),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.TRAIN,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    task.max_steps = 1
    runtime.run_episode(task, RuntimeMode.TRAIN)
    selected = runtime.state.trajectory.transitions[0].metadata["scoring"]
    assert selected["feasible"] is True
    assert selected["candidate"]["action_type"] != "PATCH_FILE"


def test_runtime_eval_cannot_bypass_post_feasibility_fallback(tmp_path):
    runtime = JarvisRuntime(
        action_generator=InfeasiblePatchGenerator(),
        config=JarvisRuntimeConfig(latent_dim=8, hidden_dim=8, replay_capacity=10),
        data_dir=tmp_path / "runtime",
        mode=RuntimeMode.EVAL,
    )
    task = CodingTaskFactory(tmp_path / "tasks").make_v03_split_tasks(DatasetSplit.TRAIN, 1)[0]
    task.max_steps = 1
    runtime.run_episode(task, RuntimeMode.EVAL)
    selected = runtime.state.trajectory.transitions[0].metadata["scoring"]
    assert selected["feasible"] is True
    assert selected["candidate"]["action_type"] == "READ_FILE"


def test_timing_and_smoke_mode_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = run_coding_brain_v03_demo(CodingBrainV03Config(smoke=True, train_episodes=2, mock_brain=True, quiet=True, seed=3))
    if not metrics.get("SANDBOX_AVAILABLE"):
        return
    assert metrics["TRAIN_TASK_COUNT"] <= 2
    assert metrics["VALIDATION_TASK_COUNT"] == 1
    assert metrics["HOLDOUT_TASK_COUNT"] == 1
    assert metrics["CONFIG"]["coding_max_tokens"] == 300
    assert metrics["CONFIG"]["max_steps"] == 6
    assert "PERFORMANCE" in metrics
    assert "brain_candidate_generation" in metrics["PERFORMANCE"]["timings"]
