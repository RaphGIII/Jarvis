from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import torch

from environments.coding.task import CodingTask
from runtime.jarvis_runtime import JarvisRuntime
from runtime.runtime_state import RuntimeMode


@dataclass
class BenchmarkResult:
    episodes: int
    success_rate: float
    mean_reward: float
    mean_steps_to_solution: float
    tests_passed_delta: float
    regression_rate: float
    invalid_action_rate: float
    world_model_prediction_loss: float
    value_prediction_error: float
    q_prediction_error: float = 0.0
    hidden_verifier_runs: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_steps_to_solution": self.mean_steps_to_solution,
            "tests_passed_delta": self.tests_passed_delta,
            "regression_rate": self.regression_rate,
            "invalid_action_rate": self.invalid_action_rate,
            "world_model_prediction_loss": self.world_model_prediction_loss,
            "value_prediction_error": self.value_prediction_error,
            "q_prediction_error": self.q_prediction_error,
            "hidden_verifier_runs": self.hidden_verifier_runs,
        }


class CodingBenchmark:
    def evaluate(self, runtime: JarvisRuntime, tasks: list[CodingTask]) -> BenchmarkResult:
        rewards = []
        steps = []
        successes = []
        tests_passed = []
        regressions = []
        invalids = []
        prediction_losses = []
        value_errors = []
        q_errors = []
        hidden_runs = []
        previous_mode = runtime.state.mode
        modules = [
            runtime.encoder,
            runtime.action_encoder,
            runtime.world_model,
            runtime.value_function,
            runtime.action_value,
            runtime.policy,
        ]
        previous_training_modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        with torch.no_grad():
            for task in tasks:
                metrics = runtime.run_episode(task, RuntimeMode.EVAL)
                hidden_result = runtime.final_hidden_verification()
                external_success = bool(hidden_result.get("success", metrics["success"]))
                rewards.append(float(metrics["reward"]))
                steps.append(float(metrics["steps"]))
                successes.append(1.0 if external_success else 0.0)
                hidden_runs.append(float(hidden_result.get("runs", 0)))
                latest = runtime.state.latest_metrics
                tests_passed.append(float(latest.get("tests_passed", 0)))
                regressions.append(1.0 if float(latest.get("tests_failed", 0)) > 0 and external_success is False else 0.0)
                invalids.append(1.0 if latest.get("invalid_action", False) else 0.0)
                transition_prediction = [
                    float(transition.metadata.get("prediction_error", 0.0))
                    for transition in runtime.state.trajectory.transitions
                ]
                transition_td = [
                    abs(float(transition.metadata.get("td_error", 0.0)))
                    for transition in runtime.state.trajectory.transitions
                ]
                transition_q = [
                    abs(float((transition.metadata.get("scoring") or {}).get("q_value", 0.0) - transition.reward))
                    for transition in runtime.state.trajectory.transitions
                ]
                prediction_losses.append(mean(transition_prediction) if transition_prediction else 0.0)
                value_errors.append(mean(transition_td) if transition_td else 0.0)
                q_errors.append(mean(transition_q) if transition_q else 0.0)
        for module, was_training in zip(modules, previous_training_modes):
            module.train(was_training)
        runtime.state.mode = previous_mode
        return BenchmarkResult(
            episodes=len(tasks),
            success_rate=mean(successes) if successes else 0.0,
            mean_reward=mean(rewards) if rewards else 0.0,
            mean_steps_to_solution=mean(steps) if steps else 0.0,
            tests_passed_delta=mean(tests_passed) if tests_passed else 0.0,
            regression_rate=mean(regressions) if regressions else 0.0,
            invalid_action_rate=mean(invalids) if invalids else 0.0,
            world_model_prediction_loss=mean(prediction_losses) if prediction_losses else 0.0,
            value_prediction_error=mean(value_errors) if value_errors else 0.0,
            q_prediction_error=mean(q_errors) if q_errors else 0.0,
            hidden_verifier_runs=mean(hidden_runs) if hidden_runs else 0.0,
        )
