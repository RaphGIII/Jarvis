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
        invalid = 1.0 if environment_step.action_result.data.get("invalid_action", False) else 0.0
        regression = max(0.0, -passed_delta) + max(0.0, failed_delta)
        all_tests_passed = bool(
            current_tests.get("ran", False)
            and current_tests.get("failed", 0) == 0
            and current_tests.get("passed", 0) > 0
        )
        hidden_passed = bool(current_tests.get("hidden_passed", 0))
        completion = 1.0 if action.action_type == ActionType.FINISH and environment_step.success else 0.0
        step_cost = min(1.0, max(0.0, action.estimated_cost / 10.0))
        error_text = f"{environment_step.action_result.stderr}\n{environment_step.action_result.stdout}".lower()
        syntax_or_runtime_error = 1.0 if any(token in error_text for token in ["syntaxerror", "traceback", "exception"]) else 0.0
        changed = action.action_type in {ActionType.PATCH_FILE, ActionType.WRITE_FILE}
        unchanged_or_passive = 1.0 if action.action_type in {ActionType.LIST_FILES, ActionType.INSPECT_ERROR, ActionType.FINISH} and not environment_step.success else 0.0

        r_tests = 2.0 * passed_delta + (4.0 if all_tests_passed else 0.0)
        r_task_progress = max(0.0, passed_delta) + (0.5 if changed and environment_step.action_result.ok else 0.0)
        r_regression = -5.0 * regression
        r_errors = -4.0 * invalid - 2.0 * syntax_or_runtime_error
        r_efficiency = -0.2 - 0.15 * step_cost - 0.3 * unchanged_or_passive
        r_completion = 20.0 if environment_step.success else 0.0
        r_hidden = 5.0 if hidden_passed else 0.0
        r_public_only = -6.0 if all_tests_passed and not environment_step.success else 0.0

        signal = RewardSignal(
            task_success=1.0 if environment_step.success else 0.0,
            correctness=max(0.0, min(1.0, current_passed / total_tests)),
            efficiency=max(0.0, 1.0 - environment_step.observation.step_number / max(1.0, environment_step.observation.step_number + environment_step.observation.remaining_budget)),
            novelty=0.0,
            learning_progress=max(0.0, passed_delta),
            error_penalty=invalid + syntax_or_runtime_error,
            risk_penalty=regression + invalid * 0.5 + step_cost * 0.1,
        )
        shaped_total = r_completion + r_hidden + r_tests + r_task_progress + r_regression + r_errors + r_efficiency + r_public_only
        total = float(shaped_total)
        components = {
            "R_tests": float(r_tests),
            "R_task_progress": float(r_task_progress),
            "R_regression": float(r_regression),
            "R_errors": float(r_errors),
            "R_efficiency": float(r_efficiency),
            "R_completion": float(r_completion),
            "R_hidden": float(r_hidden),
            "R_public_only_penalty": float(r_public_only),
            "R_user": 0.0,
            "R_intrinsic": 0.0,
        }
        return CodingRewardResult(signal=signal, total=total, components=components)
