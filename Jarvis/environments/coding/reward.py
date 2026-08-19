from __future__ import annotations

from dataclasses import dataclass

from environments.base import EnvironmentStep
from environments.coding.actions import ActionCandidate, ActionType
from environments.coding.observation import CodingObservation
from learning.rewards.reward_model import MultiObjectiveRewardModel, RewardSignal


@dataclass
class CodingRewardResult:
    signal: RewardSignal
    total: float
    components: dict[str, float]


class CodingRewardEngine:
    """Objective reward from tests, regressions, invalid actions, cost, and completion."""

    def __init__(self, reward_model: MultiObjectiveRewardModel | None = None) -> None:
        self.reward_model = reward_model or MultiObjectiveRewardModel()

    def compute(
        self,
        previous_observation: CodingObservation,
        action: ActionCandidate,
        environment_step: EnvironmentStep,
    ) -> CodingRewardResult:
        previous_tests = previous_observation.test_state
        current_tests = environment_step.observation.test_state
        prev_passed = float(previous_tests.get("passed", 0))
        current_passed = float(current_tests.get("passed", 0))
        prev_failed = float(previous_tests.get("failed", 0))
        current_failed = float(current_tests.get("failed", 0))
        total_tests = max(1.0, float(current_tests.get("total", previous_tests.get("total", 1))))

        passed_delta = current_passed - prev_passed
        failed_delta = current_failed - prev_failed
        invalid = 0.0 if environment_step.action_result.ok else 1.0
        regression = max(0.0, -passed_delta) + max(0.0, failed_delta)
        all_tests_passed = bool(current_tests.get("ran", False) and current_tests.get("failed", 0) == 0 and current_tests.get("passed", 0) > 0)
        completion = 1.0 if action.action_type == ActionType.FINISH and environment_step.success else 0.0
        step_cost = min(1.0, max(0.0, action.estimated_cost / 10.0))

        r_tests = passed_delta + (2.0 if all_tests_passed and passed_delta >= 0 else 0.0)
        r_task_progress = max(0.0, passed_delta) + (0.2 if environment_step.action_result.ok else 0.0)
        r_errors = invalid + max(0.0, current_failed - prev_failed)
        useful_action = action.action_type in {ActionType.READ_FILE, ActionType.PATCH_FILE, ActionType.RUN_TESTS, ActionType.INSPECT_ERROR}
        r_efficiency = (
            max(0.0, 1.0 - environment_step.observation.step_number / max(1.0, environment_step.observation.step_number + environment_step.observation.remaining_budget))
            if useful_action or environment_step.success
            else 0.0
        )

        signal = RewardSignal(
            task_success=1.0 if environment_step.success else 0.0,
            correctness=max(0.0, min(1.0, current_passed / total_tests)),
            efficiency=r_efficiency,
            novelty=0.0,
            learning_progress=max(0.0, passed_delta),
            error_penalty=r_errors,
            risk_penalty=regression + invalid * 0.5 + step_cost * 0.1,
        )
        total = self.reward_model.score(signal)
        components = {
            "R_tests": float(r_tests),
            "R_task_progress": float(r_task_progress),
            "R_regression": float(-regression),
            "R_errors": float(-r_errors),
            "R_efficiency": float(r_efficiency),
            "R_completion": float(completion),
            "R_user": 0.0,
            "R_intrinsic": 0.0,
        }
        return CodingRewardResult(signal=signal, total=total, components=components)
