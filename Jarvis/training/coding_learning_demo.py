from __future__ import annotations

import tempfile
from pathlib import Path

from learning.curriculum.curriculum import CurriculumManager
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode
from training.coding_benchmark import CodingBenchmark
from training.coding_curriculum import CodingTaskFactory


def _make_tasks(factory: CodingTaskFactory, prefix: str, count: int):
    tasks = []
    for index in range(count):
        if index % 4 == 3:
            tasks.append(factory.make_syntax_bug_task(f"{prefix}_syntax_{index}"))
        else:
            tasks.append(factory.make_addition_bug_task(f"{prefix}_addition_{index}", variant=index))
    return tasks


def run_coding_learning_demo(episodes: int = 50) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="jarvis_coding_demo_", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        factory = CodingTaskFactory(root / "tasks")
        runtime = JarvisRuntime(
            config=JarvisRuntimeConfig(
                latent_dim=32,
                hidden_dim=32,
                replay_capacity=300,
                train_exploration_epsilon=0.5,
                policy_score_weight=2.0,
                world_reward_weight=0.0,
                confidence_weight=0.0,
                cost_weight=0.0,
                risk_weight=0.0,
                seed=21,
            ),
            data_dir=root / "runtime_data",
            mode=RuntimeMode.TRAIN,
        )
        runtime.scheduler.config.world_model_batch_size = 1
        runtime.scheduler.config.value_policy_batch_size = 1
        runtime.scheduler.config.world_model_train_every_n_steps = 1
        runtime.scheduler.config.value_policy_train_every_n_steps = 1
        benchmark = CodingBenchmark()
        before = benchmark.evaluate(runtime, _make_tasks(factory, "before", 4))

        curriculum = CurriculumManager()
        training_metrics = []
        for index in range(episodes):
            if index % 3 == 0:
                task = factory.select_task(curriculum, count=4)
                task.task_id = f"train_selected_{index}"
            elif index % 5 == 0:
                task = factory.make_syntax_bug_task(f"train_syntax_{index}")
            else:
                task = factory.make_addition_bug_task(f"train_addition_{index}", variant=index)
            metrics = runtime.run_episode(task, RuntimeMode.TRAIN)
            training_metrics.append(metrics)

        last_train_metrics = dict(runtime.state.latest_metrics)
        after = benchmark.evaluate(runtime, _make_tasks(factory, "after", 4))
        summary = runtime.learning_summary()
        train_success_rate = sum(1 for item in training_metrics if item["success"]) / max(1, len(training_metrics))
        result = {
            "Episodes": episodes,
            "Before": before.to_dict(),
            "After": after.to_dict(),
            "Success Rate": after.success_rate,
            "Mean Reward": after.mean_reward,
            "World Model Loss": last_train_metrics.get("world_loss", 0.0),
            "Value Loss": last_train_metrics.get("value_loss", 0.0),
            "Policy Loss": last_train_metrics.get("policy_loss", 0.0),
            "Replay Size": len(runtime.replay_buffer),
            "Persistent Experience": runtime.experience_store.count(),
            "Capability Trend": {
                name: estimate.trend for name, estimate in runtime.self_model.capabilities.items()
            },
            "Parameters Updated": summary["parameters_updated"],
            "Not Updated": summary["not_updated"],
            "Training Success Rate": train_success_rate,
            "Training Mean Reward": sum(float(item["reward"]) for item in training_metrics) / max(1, len(training_metrics)),
        }
        runtime.save_checkpoints(
            {
                "success_rate_after": after.success_rate,
                "mean_reward_after": after.mean_reward,
            }
        )
        return result


if __name__ == "__main__":
    metrics = run_coding_learning_demo()
    for key, value in metrics.items():
        print(f"{key}: {value}")
