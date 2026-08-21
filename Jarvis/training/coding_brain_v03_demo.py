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
from training.benchmark_progress import BenchmarkProgressStore
from training.coding_benchmark import BenchmarkResult, CodingBenchmark
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
    eval_controller: str = "full"
    eval_controller_suite: bool = False
    eval_controller_suite_only: bool = False
    checkpoint_category: str = "best"
    benchmark_dir: str | None = None
    validation_count: int | None = None
    holdout_count: int | None = None
    compact_output: bool = False


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
            load_latest_checkpoints=False,
            tensorboard_subdir="tensorboard/coding_v03",
            trace_actions=config.trace_actions,
            eval_controller=config.eval_controller if config.eval_controller != "all" else "full",
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


def _validate_count(name: str, value: int, maximum: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    if value > maximum:
        raise ValueError(f"{name}={value} exceeds available v0.3 curriculum size {maximum}.")
    return value


def _counts(config: CodingBrainV03Config) -> tuple[int, int, int, int]:
    if config.smoke:
        train_default, validation_default, holdout_default = config.train_episodes, 1, 1
    elif config.quick:
        train_default, validation_default, holdout_default = config.train_episodes, 2, 3
    else:
        train_default, validation_default, holdout_default = config.train_episodes, 10, 20
    train_count = _validate_count("train_episodes", train_default, 30)
    validation_count = _validate_count("validation_count", config.validation_count if config.validation_count is not None else validation_default, 10)
    holdout_count = _validate_count("holdout_count", config.holdout_count if config.holdout_count is not None else holdout_default, 20)
    return train_count, train_count, validation_count, holdout_count


def _fresh_tasks(factory: CodingTaskFactory, split: DatasetSplit, count: int, max_steps: int | None):
    tasks = factory.make_v03_split_tasks(split, count)
    if max_steps is not None:
        for task in tasks:
            task.max_steps = min(task.max_steps, max_steps)
    return tasks


def _qwen_trainable(runtime: JarvisRuntime) -> bool:
    model = getattr(runtime.brain, "model", None)
    if model is None:
        return False
    return any(parameter.requires_grad for parameter in model.parameters())


def _benchmark_result_from_dict(data: dict[str, Any]) -> BenchmarkResult:
    allowed = {key for key in BenchmarkResult.__dataclass_fields__}
    return BenchmarkResult(**{key: value for key, value in data.items() if key in allowed})


def _latest_mean_gate(runtime: JarvisRuntime) -> float:
    trajectory = runtime.state.trajectory
    if trajectory is None or not trajectory.transitions:
        return 0.0
    scores = trajectory.transitions[-1].metadata.get("candidate_scores") or []
    gates = [float(score.get("controller_gate", 0.0)) for score in scores]
    return sum(gates) / len(gates) if gates else 0.0


def _restore_progress_python_rng(progress: BenchmarkProgressStore) -> None:
    BenchmarkProgressStore.restore_rng_snapshot(progress.state.get("committed_rng"))


def _restore_committed_rng_after_runtime(progress: BenchmarkProgressStore, runtime: JarvisRuntime) -> None:
    if _resume_point_uses_committed_eval_rng(progress):
        BenchmarkProgressStore.restore_rng_snapshot(progress.state.get("committed_rng"), runtime=runtime)


def _resume_point_uses_committed_eval_rng(progress: BenchmarkProgressStore) -> bool:
    stage = str(progress.state.get("stage", ""))
    if stage == "interrupted":
        stage = str(progress.state.get("interrupted_from", ""))
    if stage in {"baseline", "baseline_complete", "validation", "validation_complete", "final_holdout", "final_holdout_complete"}:
        return True
    return False


def _stage_result_from_progress(benchmark: CodingBenchmark, progress: BenchmarkProgressStore, key: str) -> BenchmarkResult | None:
    metrics = progress.state.get("metrics") or {}
    if metrics.get(key):
        return _benchmark_result_from_dict(metrics[key])
    partial = metrics.get(f"{key}_partial")
    if partial and partial.get("accumulators"):
        return benchmark.result_from_accumulators(partial["accumulators"])
    return None


def _evaluate_resumable_stage(
    *,
    benchmark: CodingBenchmark,
    runtime: JarvisRuntime,
    tasks,
    progress: BenchmarkProgressStore,
    result_key: str,
    progress_completed_key: str,
    stage_name: str,
    controller: str,
    resume: bool,
) -> BenchmarkResult:
    existing = _stage_result_from_progress(benchmark, progress, result_key) if resume else None
    completed = int(progress.state.get(progress_completed_key, 0)) if resume else 0
    if existing is not None and existing.episodes >= len(tasks):
        return existing
    initial_accumulators = existing.accumulators if existing is not None else None
    result = benchmark.evaluate(
        runtime,
        tasks,
        eval_controller=controller,
        start_index=completed,
        initial_accumulators=initial_accumulators,
        after_episode=lambda index, partial: progress.save(
            runtime=runtime,
            commit_rng=True,
            stage=stage_name,
            **{progress_completed_key: index + 1},
            metrics={**(progress.state.get("metrics") or {}), f"{result_key}_partial": partial.to_dict()},
        ),
    )
    progress.save(
        runtime=runtime,
        commit_rng=True,
        stage=f"{stage_name}_complete",
        **{progress_completed_key: len(tasks)},
        metrics={**(progress.state.get("metrics") or {}), result_key: result.to_dict()},
    )
    return result


def _ablation_table(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for mode, metrics in results.items():
        diagnostics = metrics.get("controller_diagnostics") or {}
        table.append(
            {
                "mode": mode,
                "success_rate": metrics.get("success_rate", 0.0),
                "mean_reward": metrics.get("mean_reward", 0.0),
                "mean_steps": metrics.get("mean_steps_to_solution", 0.0),
                "regression_rate": metrics.get("regression_rate", 0.0),
                "controller_mean_gate": diagnostics.get("mean_gate", 0.0),
                "disagreement_vs_heuristic": diagnostics.get("action_selection_disagreement_rate_vs_heuristic", 0.0),
                "success_when_controller_overrides_heuristic": diagnostics.get("success_rate_when_learned_changed_episode", 0.0),
            }
        )
    return table


def _runtime_architecture(runtime: JarvisRuntime) -> dict[str, Any]:
    return {
        "latent_dim": runtime.config.latent_dim,
        "hidden_dim": runtime.config.hidden_dim,
        "replay_capacity": runtime.config.replay_capacity,
        "value_policy_batch_size": runtime.scheduler.config.value_policy_batch_size,
        "world_model_batch_size": runtime.scheduler.config.world_model_batch_size,
        "learned_gate_min_experiences": runtime.config.learned_gate_min_experiences,
        "learned_gate_warmup_experiences": runtime.config.learned_gate_warmup_experiences,
    }


def _format_ablation_table(table: list[dict[str, Any]]) -> str:
    lines = ["CONTROLLER_ABLATION_TABLE", "MODE                SUCCESS  STEPS   REWARD    GATE  DISAGREE"]
    if not table:
        lines.append("(not run)")
        return "\n".join(lines)
    for row in table:
        lines.append(
            f"{row.get('mode', ''):<19}"
            f"{float(row.get('success_rate', 0.0)):>7.3f}"
            f"{float(row.get('mean_steps', 0.0)):>7.2f}"
            f"{float(row.get('mean_reward', 0.0)):>9.3f}"
            f"{float(row.get('controller_mean_gate', 0.0)):>8.3f}"
            f"{float(row.get('disagreement_vs_heuristic', 0.0)):>10.3f}"
        )
    return "\n".join(lines)


def _print_compact_metrics(metrics: dict[str, Any]) -> None:
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
    architecture = metrics.get("RUNTIME_ARCHITECTURE")
    if architecture:
        print(
            "RUNTIME_ARCHITECTURE: "
            f"latent_dim={architecture.get('latent_dim')} "
            f"hidden_dim={architecture.get('hidden_dim')} "
            f"replay_capacity={architecture.get('replay_capacity')}"
        )
    if "BASELINE" in metrics:
        baseline = metrics["BASELINE"]
        print(f"BASELINE: success={baseline.get('success_rate', 0.0):.3f} reward={baseline.get('mean_reward', 0.0):.3f} steps={baseline.get('mean_steps_to_solution', 0.0):.2f}")
    if "VALIDATION" in metrics:
        validation = metrics["VALIDATION"]
        print(f"VALIDATION: success={validation.get('success_rate', 0.0):.3f} reward={validation.get('mean_reward', 0.0):.3f} steps={validation.get('mean_steps_to_solution', 0.0):.2f}")
    if "FINAL" in metrics:
        final = metrics["FINAL"]
        print(f"FINAL: success={final.get('success_rate', 0.0):.3f} reward={final.get('mean_reward', 0.0):.3f} steps={final.get('mean_steps_to_solution', 0.0):.2f}")
    print(f"REPLAY_SIZE: {metrics.get('REPLAY_SIZE', 0)}")
    table = metrics.get("CONTROLLER_ABLATION_TABLE") or []
    gates = [float(row.get("controller_mean_gate", 0.0)) for row in table]
    mean_gate = sum(gates) / len(gates) if gates else 0.0
    print(f"MEAN_CONTROLLER_GATE: {mean_gate:.3f}")
    if "PERFORMANCE_TEXT" in metrics:
        print(metrics["PERFORMANCE_TEXT"])
    print(_format_ablation_table(table))
    if "RESULT_PATH" in metrics:
        print(f"RESULT_PATH: {metrics['RESULT_PATH']}")
    if "PROGRESS_PATH" in metrics:
        print(f"PROGRESS_PATH: {metrics['PROGRESS_PATH']}")


def _load_suite_checkpoint(runtime: JarvisRuntime, category: str) -> dict[str, bool]:
    if category == "latest":
        return runtime.load_latest_checkpoints()
    return runtime.load_best_checkpoints()


def run_coding_brain_v03_demo(config: CodingBrainV03Config | None = None) -> dict[str, Any]:
    config = config or CodingBrainV03Config()
    try:
        return _run_coding_brain_v03_demo_impl(config)
    except KeyboardInterrupt:
        root = Path(config.benchmark_dir) if config.benchmark_dir else (Path("data") if config.persistent else Path.cwd())
        progress = BenchmarkProgressStore(root / "benchmark_results" / "coding_brain_v03_progress.json")
        progress.save(stage="interrupted", interrupted=True, interrupted_from=progress.state.get("stage"), commit_rng=False, capture_live_rng=True)
        summary = {
            "INTERRUPTED": True,
            "PROGRESS_PATH": str(progress.path),
            "PARTIAL_STATE": progress.state,
        }
        if not config.quiet:
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


def _run_coding_brain_v03_demo_impl(config: CodingBrainV03Config) -> dict[str, Any]:
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

    if config.benchmark_dir:
        run_root = Path(config.benchmark_dir)
        data_dir = run_root / "runtime"
        dataset_root = run_root / "datasets"
        results_dir = run_root / "benchmark_results"
    elif config.persistent:
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
    progress = BenchmarkProgressStore(results_dir / "coding_brain_v03_progress.json")
    progress.save(stage=progress.state.get("stage") if config.resume else "setup", config=asdict(config))

    brain, semantic_encoder = _make_brain(config)
    runtime = _make_runtime(config, data_dir, brain, semantic_encoder)
    if not isinstance(runtime.action_generator, QwenActionGenerator):
        raise RuntimeError("v0.3 requires QwenActionGenerator in the productive runtime path.")
    if config.resume:
        train_completed = int(progress.state.get("train_completed", 0))
        checkpoint_path = progress.state.get("resume_checkpoint")
        if train_completed > 0:
            if not checkpoint_path:
                raise RuntimeError("Progress has completed training episodes but no exact resume checkpoint reference.")
            runtime.load_training_resume_checkpoint(checkpoint_path, expected_train_completed=train_completed)
        elif checkpoint_path:
            runtime.load_training_resume_checkpoint(checkpoint_path, expected_train_completed=train_completed)
        _restore_committed_rng_after_runtime(progress, runtime)

    train_episodes, _, validation_count, holdout_count = _counts(config)
    train_factory = CodingTaskFactory(dataset_root / "train")
    validation_factory = CodingTaskFactory(dataset_root / "validation")
    baseline_holdout_factory = CodingTaskFactory(dataset_root / "holdout_baseline")
    final_holdout_factory = CodingTaskFactory(dataset_root / "holdout_final")
    benchmark = CodingBenchmark()
    before_parameters = _snapshot(runtime)

    if config.eval_controller_suite_only:
        suite_resume_checkpoint = progress.state.get("resume_checkpoint")
        suite_train_completed = int(progress.state.get("train_completed", 0))
        if suite_resume_checkpoint:
            runtime.load_training_resume_checkpoint(suite_resume_checkpoint, expected_train_completed=suite_train_completed)
        loaded = _load_suite_checkpoint(runtime, config.checkpoint_category)
        if not any(loaded.values()):
            raise RuntimeError(f"No {config.checkpoint_category} checkpoint available for --eval-controller-suite-only.")
        suite_before_parameters = _snapshot(runtime)
        suite_before_steps = dict(runtime.learning_summary()["parameters_updated"])
        suite_results = benchmark.evaluate_controller_suite(
            runtime,
            lambda mode: _fresh_tasks(
                CodingTaskFactory(dataset_root / f"holdout_suite_only_{mode}"),
                DatasetSplit.HOLDOUT,
                holdout_count,
                config.max_steps,
            ),
        )
        result = {
            "SANDBOX_AVAILABLE": True,
            "SUITE_ONLY": True,
            "CHECKPOINT_CATEGORY": config.checkpoint_category,
            "LOADED_CHECKPOINTS": loaded,
            "CONTROLLER_ABLATIONS": suite_results,
            "CONTROLLER_ABLATION_TABLE": _ablation_table(suite_results),
            "PARAMETERS_CHANGED": _changed(suite_before_parameters, runtime),
            "TRAINING_STEPS_BEFORE": suite_before_steps,
            "TRAINING_STEPS_AFTER": dict(runtime.learning_summary()["parameters_updated"]),
            "REPLAY_SIZE": len(runtime.replay_buffer),
            "RUNTIME_ARCHITECTURE": _runtime_architecture(runtime),
            "CONFIG": asdict(config),
        }
        output_path = results_dir / f"coding_brain_v03_suite_only_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result["RESULT_PATH"] = str(output_path)
        result["PROGRESS_PATH"] = str(progress.path)
        progress.save(stage="suite_only_complete", result_path=str(output_path), metrics={**(progress.state.get("metrics") or {}), "suite_only": result})
        return result

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
    controller = config.eval_controller if config.eval_controller != "all" else "full"
    baseline_holdout = _evaluate_resumable_stage(
        benchmark=benchmark,
        runtime=runtime,
        tasks=baseline_holdout_tasks,
        progress=progress,
        result_key="baseline",
        progress_completed_key="baseline_completed",
        stage_name="baseline",
        controller=controller,
        resume=config.resume,
    )
    train_start = int(progress.state.get("train_completed", 0)) if config.resume else 0
    for episode, task in enumerate(train_tasks[train_start:], start=train_start):
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
        runtime.tensorboard.log_scalar("training/mean_gate", _latest_mean_gate(runtime), episode + 1)
        if len(runtime.replay_buffer) % 25 == 0 or episode + 1 == len(train_tasks):
            curve = {
                "episode": episode + 1,
                "replay_size": len(runtime.replay_buffer),
                "reward": runtime.state.total_reward,
                "success": bool(runtime.state.latest_metrics.get("success")),
                "world_loss": runtime.state.latest_metrics.get("world_loss"),
                "value_loss": runtime.state.latest_metrics.get("value_loss"),
                "q_loss": runtime.state.latest_metrics.get("q_loss"),
                "policy_loss": runtime.state.latest_metrics.get("policy_loss"),
                "gradient_norm": runtime.state.latest_metrics.get("gradient_norm"),
                "world_gradient_norm": runtime.state.latest_metrics.get("world_gradient_norm"),
                "mean_gate": _latest_mean_gate(runtime),
            }
            curves = list((progress.state.get("metrics") or {}).get("training_curves") or [])
            curves.append(curve)
            progress.metric("training_curves", curves)
        resume_checkpoint = runtime.save_training_resume_checkpoint(
            results_dir / "resume_checkpoints" / f"train_episode_{episode + 1:04d}.pt",
            train_completed=episode + 1,
        )
        progress.save(
            stage="train",
            train_completed=episode + 1,
            resume_checkpoint=str(resume_checkpoint),
            latest_training_metrics=dict(runtime.state.latest_metrics),
            replay_size=len(runtime.replay_buffer),
        )

    log(f"[VALIDATION 1/{len(validation_tasks)}] evaluating no-grad validation tasks...")
    runtime.trace_label = "VALIDATION"
    validation = _evaluate_resumable_stage(
        benchmark=benchmark,
        runtime=runtime,
        tasks=validation_tasks,
        progress=progress,
        result_key="validation",
        progress_completed_key="validation_completed",
        stage_name="validation",
        controller=controller,
        resume=config.resume,
    )
    runtime.tensorboard.log_scalar("evaluation/validation_success_rate", validation.success_rate, runtime.scheduler.runtime_steps)
    runtime.tensorboard.log_scalar("evaluation/validation_reward", validation.mean_reward, runtime.scheduler.runtime_steps)

    validation_metrics = {**validation.to_dict(), "split": "validation"}
    latest_paths = runtime.save_checkpoints(validation_metrics, category="latest")
    best_metrics = runtime.checkpoint_manager.best_metrics()
    promoted = runtime.checkpoint_manager.should_promote(validation_metrics, best_metrics)
    if promoted:
        runtime.save_checkpoints(validation_metrics, category="best")

    final_holdout_tasks = final_holdout_factory.make_v03_split_tasks(DatasetSplit.HOLDOUT, holdout_count)
    if config.max_steps is not None:
        for task in final_holdout_tasks:
            task.max_steps = min(task.max_steps, config.max_steps or task.max_steps)
    log(f"[FINAL 1/{len(final_holdout_tasks)}] evaluating fresh pristine holdout tasks...")
    runtime.trace_label = "HOLDOUT"
    final_holdout = _evaluate_resumable_stage(
        benchmark=benchmark,
        runtime=runtime,
        tasks=final_holdout_tasks,
        progress=progress,
        result_key="final",
        progress_completed_key="final_holdout_completed",
        stage_name="final_holdout",
        controller=controller,
        resume=config.resume,
    )
    runtime.tensorboard.log_scalar("evaluation/holdout_success_rate", final_holdout.success_rate, runtime.scheduler.runtime_steps)
    runtime.tensorboard.log_scalar("evaluation/holdout_reward", final_holdout.mean_reward, runtime.scheduler.runtime_steps)
    controller_ablations = {}
    if config.eval_controller == "all" or config.eval_controller_suite:
        controller_ablations = benchmark.evaluate_controller_suite(
            runtime,
            lambda mode: _fresh_tasks(
                CodingTaskFactory(dataset_root / f"holdout_controller_{mode}"),
                DatasetSplit.HOLDOUT,
                holdout_count,
                config.max_steps,
            ),
        )
        progress.metric("controller_ablations", controller_ablations)

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
        "EVAL_CONTROLLER": controller,
        "CONTROLLER_ABLATIONS": controller_ablations,
        "CONTROLLER_ABLATION_TABLE": _ablation_table(controller_ablations),
        "REPLAY_SIZE": len(runtime.replay_buffer),
        "PERSISTENT_EXPERIENCE": runtime.experience_store.count(),
        "TENSORBOARD": runtime.learning_summary()["tensorboard"],
        "PERFORMANCE": runtime.profiler.summary(),
        "PERFORMANCE_TEXT": runtime.profiler.format_summary(),
        "RUNTIME_ARCHITECTURE": _runtime_architecture(runtime),
        "CONFIG": asdict(config),
    }
    output_path = results_dir / f"coding_brain_v03_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["RESULT_PATH"] = str(output_path)
    result["PROGRESS_PATH"] = str(progress.path)
    progress.save(stage="complete", result_path=str(output_path), metrics={**(progress.state.get("metrics") or {}), "result": result})
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
    parser.add_argument(
        "--eval-controller",
        choices=["heuristic", "policy", "policy_q", "policy_q_value", "full", "all"],
        default="full",
        help="Evaluation-only controller ablation mode.",
    )
    parser.add_argument("--eval-controller-suite", action="store_true", help="Evaluate all controller modes on fresh holdout tasks.")
    parser.add_argument("--eval-controller-suite-only", action="store_true", help="Load an existing checkpoint and run only controller ablations.")
    parser.add_argument("--checkpoint-category", choices=["best", "latest"], default="best", help="Checkpoint category for suite-only ablation.")
    parser.add_argument("--benchmark-dir", default=None, help="Reusable benchmark directory for temporary/resumable runs.")
    parser.add_argument("--validation-count", type=int, default=None, help="Override validation task count without changing runtime architecture.")
    parser.add_argument("--holdout-count", type=int, default=None, help="Override holdout task count without changing runtime architecture.")
    parser.add_argument("--compact-output", action="store_true", help="Print compact benchmark summaries while writing full diagnostics to RESULT_PATH.")
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
        eval_controller=args.eval_controller,
        eval_controller_suite=args.eval_controller_suite,
        eval_controller_suite_only=args.eval_controller_suite_only,
        checkpoint_category=args.checkpoint_category,
        benchmark_dir=args.benchmark_dir,
        validation_count=args.validation_count,
        holdout_count=args.holdout_count,
        compact_output=args.compact_output,
    )


def main() -> None:
    config = _parse_args()
    metrics = run_coding_brain_v03_demo(config)
    if config.compact_output:
        _print_compact_metrics(metrics)
        return
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
    if "CONTROLLER_ABLATION_TABLE" in metrics:
        print(_format_ablation_table(metrics["CONTROLLER_ABLATION_TABLE"]))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
