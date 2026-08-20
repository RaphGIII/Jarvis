from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from brain.providers import LocalTransformersBrainProvider, OpenAICompatibleBrainProvider, make_brain_provider_from_env
from brain.registry import ModelRegistry
from environments.coding.sandbox_backend import DockerSandboxBackend, SandboxPolicy
from learning.objectives.optimizer import set_global_seeds
from learning.representations.semantic import DeterministicTextEncoder, LightweightLocalEmbeddingProvider
from runtime.action_generator import QwenActionGenerator
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.coding_benchmark import CodingBenchmark
from training.coding_curriculum import CodingTaskFactory, DatasetSplit


@dataclass
class CodingBrainV03Config:
    smoke: bool = False
    quick: bool = False
    seed: int = 41
    train_episodes: int = 30
    candidate_count: int = 6
    coding_max_tokens: int = 450
    persistent: bool = False
    resume: bool = False
    mock_brain: bool = False
    quiet: bool = False
    brain_profile: str | None = None
    brain_provider: str | None = None
    trace_actions: bool = False
    max_steps: int | None = None


class MockAutonomousPatchBrain:
    """Test-only Qwen stand-in that emits structured concrete patch candidates."""

    def __init__(self) -> None:
        self.calls = 0

    def think(self, user_prompt: str, max_tokens: int = 1200) -> str:
        self.calls += 1
        prompt = user_prompt.lower()
        excerpts = prompt.split("excerpts:", 1)[-1]
        candidates: list[dict[str, Any]] = [
            {
                "reason": "Run objective tests before editing.",
                "action_type": "RUN_TESTS",
                "arguments": {},
                "expected_effect": "Collect public failure signal.",
                "confidence": 0.45,
                "estimated_cost": 2.0,
            },
            {
                "reason": "Inspect the implementation file.",
                "action_type": "READ_FILE",
                "path": "solution.py",
                "arguments": {"path": "solution.py"},
                "expected_effect": "Expose source for patching.",
                "confidence": 0.55,
                "estimated_cost": 0.5,
            },
        ]
        if "solution.py:" in excerpts and "return a - b" in excerpts:
            candidates.insert(
                0,
                {
                    "reason": "Patch arithmetic operation in the implementation.",
                    "action_type": "PATCH_FILE",
                    "path": "solution.py",
                    "arguments": {"old": "return a - b", "new": "return a + b"},
                    "expected_effect": "Addition should pass public and hidden arithmetic tests.",
                    "confidence": 0.8,
                    "estimated_cost": 2.0,
                },
            )
        if "solution.py:" in excerpts and "return text.split(',')" in excerpts:
            candidates.insert(
                0,
                {
                    "reason": "Convert split strings into integers.",
                    "action_type": "PATCH_FILE",
                    "path": "solution.py",
                    "arguments": {"old": "return text.split(',')", "new": "return [int(part) for part in text.split(',')]"},
                    "expected_effect": "Parsed scores become integers.",
                    "confidence": 0.75,
                    "estimated_cost": 2.5,
                },
            )
        if "solution.py:" in excerpts and "class counter" in excerpts and "self.count -= 1" in excerpts:
            candidates.insert(
                0,
                {
                    "reason": "Increment should increase count and value should expose it.",
                    "action_type": "WRITE_FILE",
                    "path": "solution.py",
                    "arguments": {
                        "path": "solution.py",
                        "content": "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count += 1\n\n    def value(self):\n        return self.count\n",
                    },
                    "expected_effect": "Counter accumulates increments.",
                    "confidence": 0.72,
                    "estimated_cost": 4.0,
                },
            )
        if "solution.py:" in excerpts and "return a / b" in excerpts:
            candidates.insert(
                0,
                {
                    "reason": "Guard division by zero.",
                    "action_type": "WRITE_FILE",
                    "path": "solution.py",
                    "arguments": {
                        "path": "solution.py",
                        "content": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
                    },
                    "expected_effect": "Zero division returns None while valid division works.",
                    "confidence": 0.7,
                    "estimated_cost": 4.0,
                },
            )
        return json.dumps(candidates[:6])


def _snapshot(runtime: JarvisRuntime) -> dict[str, list[torch.Tensor]]:
    modules = {
        "ObservationEncoder": runtime.encoder,
        "ActionEncoder": runtime.action_encoder,
        "WorldModel": runtime.world_model,
        "Value": runtime.value_function,
        "Q Network": runtime.action_value,
        "Policy": runtime.policy,
    }
    return {name: [parameter.detach().clone() for parameter in module.parameters()] for name, module in modules.items()}


def _changed(before: dict[str, list[torch.Tensor]], runtime: JarvisRuntime) -> dict[str, str]:
    modules = {
        "ObservationEncoder": runtime.encoder,
        "ActionEncoder": runtime.action_encoder,
        "WorldModel": runtime.world_model,
        "Value": runtime.value_function,
        "Q Network": runtime.action_value,
        "Policy": runtime.policy,
    }
    result = {}
    for name, module in modules.items():
        result[name] = "YES" if any(not torch.allclose(old, new) for old, new in zip(before[name], module.parameters())) else "NO"
    result["Qwen"] = "NO"
    return result


def _make_brain(config: CodingBrainV03Config):
    if config.mock_brain:
        return MockAutonomousPatchBrain(), DeterministicTextEncoder(embedding_dim=64)
    if config.brain_profile:
        profile = ModelRegistry().get(config.brain_profile)
        os.environ.setdefault("JARVIS_BRAIN_PROVIDER", profile.provider)
        os.environ.setdefault("JARVIS_BRAIN_MODEL", profile.model)
    if config.brain_provider:
        os.environ["JARVIS_BRAIN_PROVIDER"] = config.brain_provider
    try:
        provider = make_brain_provider_from_env()
        return provider, LightweightLocalEmbeddingProvider(embedding_dim=128)
    except Exception as exc:
        raise RuntimeError("Autonomous coding generation was requested, but the configured brain provider could not be loaded.") from exc


def _make_runtime(config: CodingBrainV03Config, data_dir: Path, brain, semantic_encoder) -> JarvisRuntime:
    runtime = JarvisRuntime(
        brain=brain,
        semantic_text_encoder=semantic_encoder,
        config=JarvisRuntimeConfig(
            latent_dim=32 if config.quick else 64,
            hidden_dim=32 if config.quick else 96,
            replay_capacity=300 if config.quick else 1200,
            num_action_candidates=config.candidate_count,
            coding_generation_max_tokens=config.coding_max_tokens,
            train_exploration_epsilon=0.35,
            q_score_weight=1.3,
            world_reward_weight=0.4,
            confidence_weight=0.8,
            cost_weight=0.08,
            risk_weight=0.25,
            seed=config.seed,
            load_latest_checkpoints=config.resume,
            tensorboard_subdir="tensorboard/coding_v03",
            trace_actions=config.trace_actions,
        ),
        data_dir=data_dir,
        mode=RuntimeMode.TRAIN,
        sandbox_backend=DockerSandboxBackend(policy=SandboxPolicy(timeout_seconds=20.0)),
    )
    runtime.scheduler.config.value_policy_batch_size = 1 if config.quick else 4
    runtime.scheduler.config.world_model_batch_size = 1 if config.quick else 4
    runtime.scheduler.config.value_policy_train_every_n_steps = 1
    runtime.scheduler.config.world_model_train_every_n_steps = 1
    return runtime


def _counts(config: CodingBrainV03Config) -> tuple[int, int, int, int]:
    if config.smoke:
        return min(config.train_episodes, 2), 2, 1, 1
    if config.quick:
        return config.train_episodes, 4, 2, 3
    return config.train_episodes, 30, 10, 20


def _qwen_trainable(runtime: JarvisRuntime) -> bool:
    model = getattr(runtime.brain, "model", None)
    if model is None:
        return False
    return any(parameter.requires_grad for parameter in model.parameters())


def run_coding_brain_v03_demo(config: CodingBrainV03Config | None = None) -> dict[str, Any]:
    config = config or CodingBrainV03Config()
    if config.smoke and config.coding_max_tokens == 450:
        config.coding_max_tokens = 300
    if config.smoke and config.candidate_count == 6:
        config.candidate_count = 2
    if config.smoke and config.max_steps is None:
        config.max_steps = 6
    set_global_seeds(config.seed)
    random.seed(config.seed)
    if not DockerSandboxBackend.is_available():
        return {
            "SANDBOX_AVAILABLE": False,
            "MESSAGE": "Docker sandbox unavailable; unsafe host fallback is disabled.",
        }

    if config.persistent:
        data_dir = Path("data")
        dataset_root = Path("data") / "datasets" / "coding_v03"
        results_dir = Path("data") / "benchmark_results"
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="jarvis_coding_v03_"))
        data_dir = temp_root / "runtime"
        dataset_root = temp_root / "datasets"
        results_dir = temp_root / "benchmark_results"
    dataset_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    brain, semantic_encoder = _make_brain(config)
    runtime = _make_runtime(config, data_dir, brain, semantic_encoder)
    if not isinstance(runtime.action_generator, QwenActionGenerator):
        raise RuntimeError("v0.3 requires QwenActionGenerator in the productive runtime path.")

    train_episodes, _, validation_count, holdout_count = _counts(config)
    train_factory = CodingTaskFactory(dataset_root / "train")
    validation_factory = CodingTaskFactory(dataset_root / "validation")
    baseline_holdout_factory = CodingTaskFactory(dataset_root / "holdout_baseline")
    final_holdout_factory = CodingTaskFactory(dataset_root / "holdout_final")
    benchmark = CodingBenchmark()
    before_parameters = _snapshot(runtime)

    baseline_holdout_tasks = baseline_holdout_factory.make_v03_split_tasks(DatasetSplit.HOLDOUT, holdout_count)
    train_tasks = train_factory.make_v03_split_tasks(DatasetSplit.TRAIN, train_episodes)
    validation_tasks = validation_factory.make_v03_split_tasks(DatasetSplit.VALIDATION, validation_count)
    if config.max_steps is not None:
        for task in [*baseline_holdout_tasks, *train_tasks, *validation_tasks]:
            task.max_steps = min(task.max_steps, config.max_steps or task.max_steps)

    def log(message: str) -> None:
        if not config.quiet:
            print(message, flush=True)

    log(f"[SETUP] provider={getattr(brain, 'provider_name', brain.__class__.__name__)} model={getattr(brain, 'model_name', brain.__class__.__name__)}")
    log(f"[SETUP] train={len(train_tasks)} validation={len(validation_tasks)} holdout={len(baseline_holdout_tasks)} candidates={config.candidate_count} tokens={config.coding_max_tokens}")

    log(f"[BASELINE 1/{len(baseline_holdout_tasks)}] evaluating pristine holdout tasks...")
    runtime.trace_label = "BASELINE"
    baseline_holdout = benchmark.evaluate(runtime, baseline_holdout_tasks)
    for episode, task in enumerate(train_tasks):
        log(f"[TRAIN {episode + 1}/{len(train_tasks)}] task={task.task_id} | running episode...")
        runtime.trace_label = f"TRAIN {episode + 1}/{len(train_tasks)}"
        runtime.run_episode(task, RuntimeMode.TRAIN)
        log(
            f"[TRAIN {episode + 1}/{len(train_tasks)}] steps={runtime.state.step_count} "
            f"| reward={runtime.state.total_reward:.3f} | replay={len(runtime.replay_buffer)}"
        )
        runtime.tensorboard.log_scalar("training/reward", runtime.state.total_reward, episode + 1)
        runtime.tensorboard.log_scalar("training/success", 1.0 if runtime.state.latest_metrics.get("success") else 0.0, episode + 1)
        runtime.tensorboard.log_scalar("training/tests_passed", float(runtime.state.latest_metrics.get("tests_passed", 0)), episode + 1)
        runtime.tensorboard.log_scalar("training/invalid_action_rate", 1.0 if runtime.state.latest_metrics.get("invalid_action") else 0.0, episode + 1)
        runtime.tensorboard.log_scalar("training/episode_length", float(runtime.state.step_count), episode + 1)

    log(f"[VALIDATION 1/{len(validation_tasks)}] evaluating no-grad validation tasks...")
    runtime.trace_label = "VALIDATION"
    validation = benchmark.evaluate(runtime, validation_tasks)
    runtime.tensorboard.log_scalar("evaluation/validation_success_rate", validation.success_rate, runtime.scheduler.runtime_steps)
    runtime.tensorboard.log_scalar("evaluation/validation_reward", validation.mean_reward, runtime.scheduler.runtime_steps)

    latest_paths = runtime.save_checkpoints(validation.to_dict(), category="latest")
    best_metrics = runtime.checkpoint_manager.best_metrics()
    promoted = runtime.checkpoint_manager.should_promote(validation.to_dict(), best_metrics)
    if promoted:
        runtime.save_checkpoints(validation.to_dict(), category="best")

    final_holdout_tasks = final_holdout_factory.make_v03_split_tasks(DatasetSplit.HOLDOUT, holdout_count)
    if config.max_steps is not None:
        for task in final_holdout_tasks:
            task.max_steps = min(task.max_steps, config.max_steps or task.max_steps)
    log(f"[FINAL 1/{len(final_holdout_tasks)}] evaluating fresh pristine holdout tasks...")
    runtime.trace_label = "HOLDOUT"
    final_holdout = benchmark.evaluate(runtime, final_holdout_tasks)
    runtime.tensorboard.log_scalar("evaluation/holdout_success_rate", final_holdout.success_rate, runtime.scheduler.runtime_steps)
    runtime.tensorboard.log_scalar("evaluation/holdout_reward", final_holdout.mean_reward, runtime.scheduler.runtime_steps)

    result = {
        "SANDBOX_AVAILABLE": True,
        "MODEL": getattr(brain, "model_name", "MockAutonomousPatchBrain" if config.mock_brain else brain.__class__.__name__),
        "BRAIN_PROVIDER": getattr(brain, "provider_name", brain.__class__.__name__),
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "ACTION_GENERATOR": runtime.action_generator.__class__.__name__,
        "SEMANTIC_ENCODER": runtime.text_encoder.__class__.__name__,
        "QWEN_TRAINABLE": _qwen_trainable(runtime),
        "TRAIN_TASK_COUNT": len(train_tasks),
        "VALIDATION_TASK_COUNT": len(validation_tasks),
        "HOLDOUT_TASK_COUNT": len(final_holdout_tasks),
        "PERSISTENCE_MODE": "persistent" if config.persistent else "temporary",
        "SEED": config.seed,
        "BASELINE": baseline_holdout.to_dict(),
        "VALIDATION": validation.to_dict(),
        "FINAL": final_holdout.to_dict(),
        "DELTA": {
            "holdout_success_rate": final_holdout.success_rate - baseline_holdout.success_rate,
            "mean_reward": final_holdout.mean_reward - baseline_holdout.mean_reward,
            "mean_steps": final_holdout.mean_steps_to_solution - baseline_holdout.mean_steps_to_solution,
        },
        "BASELINE_HOLDOUT_SUCCESS_RATE": baseline_holdout.success_rate,
        "FINAL_HOLDOUT_SUCCESS_RATE": final_holdout.success_rate,
        "ABSOLUTE_DELTA": final_holdout.success_rate - baseline_holdout.success_rate,
        "PARAMETERS_CHANGED": _changed(before_parameters, runtime),
        "PROMOTED_BEST": promoted,
        "REPLAY_SIZE": len(runtime.replay_buffer),
        "PERSISTENT_EXPERIENCE": runtime.experience_store.count(),
        "TENSORBOARD": runtime.learning_summary()["tensorboard"],
        "PERFORMANCE": runtime.profiler.summary(),
        "PERFORMANCE_TEXT": runtime.profiler.format_summary(),
        "CONFIG": asdict(config),
    }
    output_path = results_dir / f"coding_brain_v03_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["RESULT_PATH"] = str(output_path)
    return result


def _parse_args() -> CodingBrainV03Config:
    parser = argparse.ArgumentParser(description="Run JARVIS Coding Brain v0.3 autonomous patch synthesis benchmark.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny integration smoke benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run a small local benchmark.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--coding-max-tokens", type=int, default=None)
    parser.add_argument("--persistent", action="store_true", help="Use repository-local data/ paths.")
    parser.add_argument("--resume", action="store_true", help="Resume latest checkpoints from the selected data directory.")
    parser.add_argument("--mock-brain", action="store_true", help="Use test-only structured mock brain instead of loading Qwen.")
    parser.add_argument("--quiet", action="store_true", help="Suppress live progress output.")
    parser.add_argument("--brain-profile", default=None)
    parser.add_argument("--brain-provider", choices=["local_transformers", "openai_compatible"], default=None)
    parser.add_argument("--trace-actions", action="store_true", help="Print safe per-step action scoring diagnostics.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override per-task step budget for smoke/diagnostics.")
    args = parser.parse_args()
    train_episodes = args.train_episodes if args.train_episodes is not None else (1 if args.smoke else (4 if args.quick else 30))
    candidate_count = args.candidate_count
    if args.smoke and args.candidate_count == 6:
        candidate_count = 2
    coding_max_tokens = args.coding_max_tokens if args.coding_max_tokens is not None else (300 if args.smoke else 450)
    return CodingBrainV03Config(
        smoke=args.smoke,
        quick=args.quick,
        seed=args.seed,
        train_episodes=train_episodes,
        candidate_count=candidate_count,
        coding_max_tokens=coding_max_tokens,
        persistent=args.persistent,
        resume=args.resume,
        mock_brain=args.mock_brain,
        quiet=args.quiet,
        brain_profile=args.brain_profile,
        brain_provider=args.brain_provider,
        trace_actions=args.trace_actions,
        max_steps=args.max_steps,
    )


def main() -> None:
    config = _parse_args()
    metrics = run_coding_brain_v03_demo(config)
    for key in [
        "SANDBOX_AVAILABLE",
        "MODEL",
        "DEVICE",
        "ACTION_GENERATOR",
        "BRAIN_PROVIDER",
        "SEMANTIC_ENCODER",
        "QWEN_TRAINABLE",
        "TRAIN_TASK_COUNT",
        "VALIDATION_TASK_COUNT",
        "HOLDOUT_TASK_COUNT",
        "PERSISTENCE_MODE",
        "SEED",
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    print(f"BASELINE HOLDOUT SUCCESS RATE: {metrics.get('BASELINE_HOLDOUT_SUCCESS_RATE', 0.0):.3f}")
    print(f"FINAL HOLDOUT SUCCESS RATE:    {metrics.get('FINAL_HOLDOUT_SUCCESS_RATE', 0.0):.3f}")
    print(f"ABSOLUTE DELTA:                {metrics.get('ABSOLUTE_DELTA', 0.0):+.3f}")
    if "PERFORMANCE_TEXT" in metrics:
        print(metrics["PERFORMANCE_TEXT"])
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
