from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import mean
from typing import Any, Callable

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
    episodes_with_invalid_action_rate: float = 0.0
    controller_diagnostics: dict[str, Any] = field(default_factory=dict)
    accumulators: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
            "episodes_with_invalid_action_rate": self.episodes_with_invalid_action_rate,
            "controller_diagnostics": self.controller_diagnostics,
            "accumulators": self.accumulators,
        }


class CodingBenchmark:
    CONTROLLER_MODES = ["heuristic", "policy", "policy_q", "policy_q_value", "full"]

    def evaluate(
        self,
        runtime: JarvisRuntime,
        tasks: list[CodingTask],
        *,
        eval_controller: str | None = None,
        after_episode: Callable[[int, BenchmarkResult], None] | None = None,
        start_index: int = 0,
        initial_accumulators: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        accumulators = self._empty_accumulators()
        if initial_accumulators:
            accumulators.update({key: list(value) for key, value in initial_accumulators.items()})
        rewards = accumulators["rewards"]
        steps = accumulators["steps"]
        successes = accumulators["successes"]
        tests_passed = accumulators["tests_passed"]
        regressions = accumulators["regressions"]
        invalids = accumulators["invalids"]
        episode_invalids = accumulators["episode_invalids"]
        prediction_losses = accumulators["prediction_losses"]
        value_errors = accumulators["value_errors"]
        q_errors = accumulators["q_errors"]
        hidden_runs = accumulators["hidden_runs"]
        episode_changed = accumulators["episode_changed"]
        episode_success_when_changed = accumulators["episode_success_when_changed"]
        all_controller_scores = accumulators["controller_scores"]
        previous_mode = runtime.state.mode
        previous_config = runtime.config
        if eval_controller is not None:
            runtime.config = replace(runtime.config, eval_controller=eval_controller)
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
            for index, task in enumerate(tasks[start_index:], start=start_index):
                metrics = runtime.run_episode(task, RuntimeMode.EVAL)
                public_success = bool(metrics["success"])
                if public_success and task.hidden_test_command is not None:
                    hidden_result = runtime.final_hidden_verification()
                    hidden_success = bool(hidden_result.get("success", False))
                elif public_success:
                    hidden_result = {"runs": 0}
                    hidden_success = True
                else:
                    hidden_result = {"runs": 0}
                    hidden_success = False
                external_success = public_success and hidden_success
                rewards.append(float(metrics["reward"]))
                steps.append(float(metrics["steps"]))
                successes.append(1.0 if external_success else 0.0)
                hidden_runs.append(float(hidden_result.get("runs", 0)))
                latest = runtime.state.latest_metrics
                tests_passed.append(float(latest.get("tests_passed", 0)))
                regressions.append(1.0 if float(latest.get("tests_failed", 0)) > 0 and external_success is False else 0.0)
                transition_invalids = [
                    1.0
                    if (transition.metadata.get("objective_metrics") or {}).get("invalid_action", False)
                    else 0.0
                    for transition in runtime.state.trajectory.transitions
                ]
                invalids.extend(transition_invalids)
                episode_invalids.append(1.0 if any(transition_invalids) else 0.0)
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
                changed = any(
                    bool((transition.metadata.get("scoring") or {}).get("controller_changed_heuristic", False))
                    for transition in runtime.state.trajectory.transitions
                )
                episode_changed.append(1.0 if changed else 0.0)
                if changed:
                    episode_success_when_changed.append(1.0 if external_success else 0.0)
                for transition in runtime.state.trajectory.transitions:
                    all_controller_scores.extend(transition.metadata.get("candidate_scores") or [])
                if after_episode is not None:
                    after_episode(index, self._build_result(
                        rewards,
                        steps,
                        successes,
                        tests_passed,
                        regressions,
                        invalids,
                        prediction_losses,
                        value_errors,
                        q_errors,
                        hidden_runs,
                        episode_invalids,
                        all_controller_scores,
                        episode_changed,
                        episode_success_when_changed,
                    ))
        for module, was_training in zip(modules, previous_training_modes):
            module.train(was_training)
        runtime.state.mode = previous_mode
        runtime.config = previous_config
        return self._build_result(
            rewards,
            steps,
            successes,
            tests_passed,
            regressions,
            invalids,
            prediction_losses,
            value_errors,
            q_errors,
            hidden_runs,
            episode_invalids,
            all_controller_scores,
            episode_changed,
            episode_success_when_changed,
        )

    def evaluate_controller_suite(
        self,
        runtime: JarvisRuntime,
        task_builder: Callable[[str], list[CodingTask]],
        modes: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        results = {}
        for mode in modes or self.CONTROLLER_MODES:
            results[mode] = self.evaluate(runtime, task_builder(mode), eval_controller=mode).to_dict()
        return results

    def result_from_accumulators(self, accumulators: dict[str, Any] | None) -> BenchmarkResult:
        accumulators = accumulators or self._empty_accumulators()
        return self._build_result(
            list(accumulators.get("rewards", [])),
            list(accumulators.get("steps", [])),
            list(accumulators.get("successes", [])),
            list(accumulators.get("tests_passed", [])),
            list(accumulators.get("regressions", [])),
            list(accumulators.get("invalids", [])),
            list(accumulators.get("prediction_losses", [])),
            list(accumulators.get("value_errors", [])),
            list(accumulators.get("q_errors", [])),
            list(accumulators.get("hidden_runs", [])),
            list(accumulators.get("episode_invalids", [])),
            list(accumulators.get("controller_scores", [])),
            list(accumulators.get("episode_changed", [])),
            list(accumulators.get("episode_success_when_changed", [])),
        )

    def _build_result(
        self,
        rewards,
        steps,
        successes,
        tests_passed,
        regressions,
        invalids,
        prediction_losses,
        value_errors,
        q_errors,
        hidden_runs,
        episode_invalids,
        all_controller_scores,
        episode_changed,
        episode_success_when_changed,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            episodes=len(rewards),
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
            episodes_with_invalid_action_rate=mean(episode_invalids) if episode_invalids else 0.0,
            controller_diagnostics=self._controller_diagnostics(
                all_controller_scores,
                episode_changed,
                episode_success_when_changed,
            ),
            accumulators={
                "rewards": list(rewards),
                "steps": list(steps),
                "successes": list(successes),
                "tests_passed": list(tests_passed),
                "regressions": list(regressions),
                "invalids": list(invalids),
                "prediction_losses": list(prediction_losses),
                "value_errors": list(value_errors),
                "q_errors": list(q_errors),
                "hidden_runs": list(hidden_runs),
                "episode_invalids": list(episode_invalids),
                "controller_scores": list(all_controller_scores),
                "episode_changed": list(episode_changed),
                "episode_success_when_changed": list(episode_success_when_changed),
            },
        )

    @staticmethod
    def _empty_accumulators() -> dict[str, list[Any]]:
        return {
            "rewards": [],
            "steps": [],
            "successes": [],
            "tests_passed": [],
            "regressions": [],
            "invalids": [],
            "prediction_losses": [],
            "value_errors": [],
            "q_errors": [],
            "hidden_runs": [],
            "episode_invalids": [],
            "controller_scores": [],
            "episode_changed": [],
            "episode_success_when_changed": [],
        }

    def _controller_diagnostics(
        self,
        candidate_scores: list[dict[str, Any]],
        episode_changed: list[float],
        episode_success_when_changed: list[float],
    ) -> dict[str, Any]:
        components = ["heuristic_score", "policy_score", "q_score", "value_score", "world_score", "learned_score"]
        mean_abs = {}
        for component in components:
            values = [abs(float(score.get(component, 0.0))) for score in candidate_scores]
            mean_abs[component] = mean(values) if values else 0.0
        gates = [float(score.get("controller_gate", 0.0)) for score in candidate_scores]
        heuristic_entries = [score for score in candidate_scores if bool(score.get("heuristic_winner", False))]
        changed_actions = [1.0 if bool(score.get("controller_changed_heuristic", False)) else 0.0 for score in heuristic_entries]
        disagreement_rate = mean(changed_actions) if changed_actions else 0.0
        return {
            "mean_abs_contribution": mean_abs,
            "mean_gate": mean(gates) if gates else 0.0,
            "action_selection_disagreement_rate_vs_heuristic": disagreement_rate,
            "learned_changed_heuristic_winner_rate": disagreement_rate,
            "episodes_with_learned_override_rate": mean(episode_changed) if episode_changed else 0.0,
            "success_rate_when_learned_changed_episode": mean(episode_success_when_changed) if episode_success_when_changed else 0.0,
            "candidate_count": len(candidate_scores),
        }
