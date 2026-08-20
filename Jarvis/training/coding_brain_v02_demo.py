from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from environments.coding.sandbox_backend import DockerSandboxBackend
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.coding_benchmark import CodingBenchmark
from training.coding_curriculum import CodingTaskFactory, DatasetSplit
from training.generic_mutation_generator import GenericMutationActionGenerator


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


def run_coding_brain_v02_demo(train_episodes: int = 36) -> dict[str, object]:
    if not DockerSandboxBackend.is_available():
        return {
            "SANDBOX_AVAILABLE": False,
            "MESSAGE": "Docker sandbox unavailable; unsafe host fallback is disabled.",
            "PARAMETERS CHANGED": {
                "ObservationEncoder": "NO",
                "ActionEncoder": "NO",
                "WorldModel": "NO",
                "Value": "NO",
                "Q Network": "NO",
                "Policy": "NO",
                "Qwen": "NO",
            },
        }

    with tempfile.TemporaryDirectory(prefix="jarvis_coding_v02_", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        factory = CodingTaskFactory(root / "tasks")
        runtime = JarvisRuntime(
            action_generator=GenericMutationActionGenerator(num_candidates=6),
            config=JarvisRuntimeConfig(
                latent_dim=32,
                hidden_dim=32,
                replay_capacity=400,
                train_exploration_epsilon=0.55,
                policy_score_weight=1.4,
                q_score_weight=1.2,
                world_reward_weight=0.2,
                confidence_weight=0.0,
                cost_weight=0.05,
                risk_weight=0.1,
                seed=33,
            ),
            data_dir=root / "runtime",
            mode=RuntimeMode.TRAIN,
            sandbox_backend=DockerSandboxBackend(),
        )
        runtime.scheduler.config.value_policy_batch_size = 1
        runtime.scheduler.config.world_model_batch_size = 1
        runtime.scheduler.config.value_policy_train_every_n_steps = 1
        runtime.scheduler.config.world_model_train_every_n_steps = 1

        benchmark = CodingBenchmark()
        holdout_before = benchmark.evaluate(runtime, factory.make_split_tasks(DatasetSplit.HOLDOUT, count=4))
        before_parameters = _snapshot(runtime)

        for episode in range(train_episodes // 2):
            task = factory.make_split_tasks(DatasetSplit.TRAIN, count=1)[0]
            task.task_id = f"train_a_{episode}"
            runtime.run_episode(task, RuntimeMode.TRAIN)

        validation = benchmark.evaluate(runtime, factory.make_split_tasks(DatasetSplit.VALIDATION, count=4))

        for episode in range(train_episodes // 2, train_episodes):
            task = factory.make_split_tasks(DatasetSplit.TRAIN, count=1)[0]
            task.task_id = f"train_b_{episode}"
            runtime.run_episode(task, RuntimeMode.TRAIN)

        holdout_after = benchmark.evaluate(runtime, factory.make_split_tasks(DatasetSplit.HOLDOUT, count=4))
        parameter_changes = _changed(before_parameters, runtime)
        runtime.save_checkpoints(
            {
                "success_rate": holdout_after.success_rate,
                "mean_reward": holdout_after.mean_reward,
                "regression_rate": holdout_after.regression_rate,
            }
        )
        return {
            "SANDBOX_AVAILABLE": True,
            "BASELINE": holdout_before.to_dict(),
            "VALIDATION": validation.to_dict(),
            "FINAL": holdout_after.to_dict(),
            "DELTA": {
                "success_rate": holdout_after.success_rate - holdout_before.success_rate,
                "mean_reward": holdout_after.mean_reward - holdout_before.mean_reward,
                "mean_steps": holdout_after.mean_steps_to_solution - holdout_before.mean_steps_to_solution,
                "world_prediction_error": holdout_after.world_model_prediction_loss - holdout_before.world_model_prediction_loss,
                "q_error": holdout_after.value_prediction_error - holdout_before.value_prediction_error,
            },
            "PARAMETERS CHANGED": parameter_changes,
            "PARAMETERS UPDATED": runtime.learning_summary()["parameters_updated"],
            "NOT UPDATED": runtime.learning_summary()["not_updated"],
            "REPLAY SIZE": len(runtime.replay_buffer),
            "PERSISTENT EXPERIENCE": runtime.experience_store.count(),
            "TENSORBOARD": runtime.learning_summary()["tensorboard"],
        }


if __name__ == "__main__":
    metrics = run_coding_brain_v02_demo()
    for section, value in metrics.items():
        print(f"{section}: {value}")
