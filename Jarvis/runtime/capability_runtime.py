from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capabilities.executor import CapabilityExecutor
from capabilities.models import (
    AcquisitionStage,
    CapabilityAcquisitionResult,
    CapabilityResolution,
    SkillSpecification,
)
from capabilities.permissions import PermissionPolicy
from capabilities.promotion import SkillPromoter
from capabilities.registry import CapabilityRegistry
from capabilities.resolver import CapabilityResolver
from capabilities.specification import SkillSpecificationGenerator
from capabilities.trajectory import AcquisitionTrajectory, AcquisitionTrajectoryStore
from capabilities.workspace import SkillWorkspaceManager
from environments.coding.sandbox_backend import DisabledSandboxBackend, DockerSandboxBackend, SandboxBackend, SandboxPolicy
from learning.representations.semantic import DeterministicTextEncoder
from runtime.jarvis_runtime import JarvisRuntime, JarvisRuntimeConfig
from runtime.runtime_state import RuntimeMode


@dataclass
class CapabilityRuntimeConfig:
    data_dir: str = "data/capabilities"
    skills_root: str = "skills"
    max_build_steps: int = 14
    num_action_candidates: int = 6
    use_docker: bool = True
    production_controller: str = "heuristic"
    learned_controller_mode: str = "shadow"
    seed: int = 404


class CapabilityAcquisitionRuntime:
    """v0.4 lifecycle for acquiring, promoting, and executing local capabilities."""

    def __init__(
        self,
        *,
        brain: Any | None = None,
        backend: SandboxBackend | None = None,
        config: CapabilityRuntimeConfig | None = None,
    ) -> None:
        self.config = config or CapabilityRuntimeConfig()
        self.root = Path(self.config.data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or self._default_backend()
        self.brain = brain
        self.registry = CapabilityRegistry(self.root / "registry.json")
        self.resolver = CapabilityResolver(self.registry, brain=brain)
        self.spec_generator = SkillSpecificationGenerator(brain=brain)
        self.permission_policy = PermissionPolicy()
        skills_root = Path(self.config.skills_root)
        self.workspace_manager = SkillWorkspaceManager(skills_root / "_staging")
        self.promoter = SkillPromoter(skills_root / "installed", self.registry)
        self.executor = CapabilityExecutor(self.root / "execution", self.backend)
        self.trajectory_store = AcquisitionTrajectoryStore(self.root / "acquisition_trajectories.jsonl")

    def handle_goal(
        self,
        goal: str,
        *,
        request_payload: dict[str, Any] | None = None,
        expected_output: dict[str, Any] | None = None,
        spec: SkillSpecification | None = None,
        hidden_workspace: Path | None = None,
        hidden_test_command: list[str] | None = None,
    ) -> CapabilityAcquisitionResult:
        payload = dict(request_payload or {})
        trajectory = AcquisitionTrajectory(goal)
        trajectory.record(AcquisitionStage.GOAL.value, {"goal": goal, "request_payload": payload})

        resolution = self.resolver.resolve(goal)
        trajectory.record(AcquisitionStage.GAP.value, resolution.to_dict())
        if resolution.status == "available" and resolution.manifest is not None:
            execution = self.executor.execute(resolution.manifest, payload)
            execution_ok = self._execution_matches(execution.success, execution.output, expected_output)
            result = CapabilityAcquisitionResult(
                goal=goal,
                success=execution_ok,
                resolution=resolution,
                capability_id=resolution.capability_id,
                execution_success=execution_ok,
                second_call_success=execution_ok,
                trajectory_id=trajectory.trajectory_id,
                output=execution.output,
                error=execution.error,
            )
            trajectory.record(AcquisitionStage.EXECUTE.value, execution.__dict__)
            self.trajectory_store.save(trajectory, result.to_dict())
            return result

        skill_spec = spec or self.spec_generator.generate(goal)
        spec_errors = skill_spec.validate()
        trajectory.record(AcquisitionStage.SPEC.value, {"specification": skill_spec.to_dict(), "validation_errors": spec_errors})
        if spec_errors:
            result = self._failed(goal, resolution, trajectory, "Invalid skill specification: " + "; ".join(spec_errors))
            self.trajectory_store.save(trajectory, result.to_dict())
            return result

        permission = self.permission_policy.evaluate(skill_spec)
        if not permission.allowed:
            trajectory.record(AcquisitionStage.SPEC.value, {"permission_block": permission.__dict__})
            result = self._failed(goal, resolution, trajectory, permission.reason)
            self.trajectory_store.save(trajectory, result.to_dict())
            return result

        staged = self.workspace_manager.create(skill_spec, uuid.uuid4().hex[:10])
        trajectory.record(
            AcquisitionStage.BUILD.value,
            {"workspace": str(staged.root), "public_tests": str(staged.public_tests_path), "protected_hashes": staged.protected_hashes},
        )
        task = staged.to_task(
            self._task_description(goal, skill_spec),
            hidden_workspace=hidden_workspace,
            hidden_test_command=hidden_test_command or ([sys.executable, "hidden_verifier.py"] if hidden_workspace is not None else None),
            max_steps=self.config.max_build_steps,
        )
        build_runtime = self._make_build_runtime(staged.root)
        metrics = build_runtime.run_episode(task, RuntimeMode.EVAL)
        public_success = bool(metrics.get("success", False))
        hidden_result = build_runtime.final_hidden_verification() if public_success else {"ran": False, "success": False, "runs": 0}
        hidden_success = bool(hidden_result.get("success", False))
        transitions = [transition.metadata for transition in build_runtime.state.trajectory.transitions] if build_runtime.state.trajectory else []
        trajectory.record(
            AcquisitionStage.TEST.value,
            {
                "public_success": public_success,
                "hidden_success": hidden_success,
                "metrics": metrics,
                "hidden_runs": hidden_result.get("runs", 0),
            },
        )
        trajectory.record(AcquisitionStage.REPAIR.value, {"transitions": transitions})
        trajectory.record(AcquisitionStage.VERIFY.value, {"hidden_result": {"ran": hidden_result.get("ran", False), "success": hidden_success}})

        promotion = self.promoter.promote(skill_spec, staged, public_success=public_success, hidden_success=hidden_success)
        trajectory.record(AcquisitionStage.PROMOTE.value, {"promoted": promotion.promoted, "errors": promotion.errors, "manifest": promotion.manifest.to_dict() if promotion.manifest else None})
        if not promotion.promoted or promotion.manifest is None:
            result = CapabilityAcquisitionResult(
                goal=goal,
                success=False,
                resolution=resolution,
                capability_id=skill_spec.capability_id,
                public_success=public_success,
                hidden_success=hidden_success,
                steps_to_acquisition=int(metrics.get("steps", 0)),
                repair_iterations=self._repair_iterations(transitions),
                invalid_action_rate=self._invalid_action_rate(transitions),
                trajectory_id=trajectory.trajectory_id,
                error="; ".join(promotion.errors),
            )
            self.trajectory_store.save(trajectory, result.to_dict())
            return result

        execution = self.executor.execute(promotion.manifest, payload)
        execution_ok = self._execution_matches(execution.success, execution.output, expected_output)
        trajectory.record(AcquisitionStage.EXECUTE.value, execution.__dict__)
        second_resolution = self.resolver.resolve(goal)
        second_execution = (
            self.executor.execute(second_resolution.manifest, payload)
            if second_resolution.status == "available" and second_resolution.manifest is not None
            else None
        )
        second_execution_ok = bool(
            second_execution
            and self._execution_matches(second_execution.success, second_execution.output, expected_output)
        )
        trajectory.record(
            AcquisitionStage.SECOND_CALL.value,
            {
                "resolution": second_resolution.to_dict(),
                "execution": second_execution.__dict__ if second_execution is not None else None,
            },
        )
        result = CapabilityAcquisitionResult(
            goal=goal,
            success=bool(public_success and hidden_success and execution_ok and second_execution_ok),
            resolution=resolution,
            capability_id=promotion.manifest.capability_id,
            promoted=True,
            public_success=public_success,
            hidden_success=hidden_success,
            execution_success=execution_ok,
            second_call_success=second_execution_ok,
            steps_to_acquisition=int(metrics.get("steps", 0)),
            repair_iterations=self._repair_iterations(transitions),
            invalid_action_rate=self._invalid_action_rate(transitions),
            trajectory_id=trajectory.trajectory_id,
            output=execution.output,
            error=execution.error,
        )
        self.trajectory_store.save(trajectory, result.to_dict())
        return result

    def _make_build_runtime(self, data_dir: Path) -> JarvisRuntime:
        return JarvisRuntime(
            brain=self.brain,
            semantic_text_encoder=DeterministicTextEncoder(embedding_dim=64),
            sandbox_backend=self.backend,
            data_dir=data_dir / ".runtime",
            mode=RuntimeMode.EVAL,
            config=JarvisRuntimeConfig(
                latent_dim=32,
                hidden_dim=32,
                replay_capacity=200,
                num_action_candidates=self.config.num_action_candidates,
                train_exploration_epsilon=0.0,
                eval_controller=self.config.production_controller,
                production_controller=self.config.production_controller,
                learned_controller_mode=self.config.learned_controller_mode,
                load_latest_checkpoints=False,
                seed=self.config.seed,
                tensorboard_subdir="tensorboard/capability_v04",
            ),
        )

    def _default_backend(self) -> SandboxBackend:
        if self.config.use_docker and DockerSandboxBackend.is_available():
            return DockerSandboxBackend(policy=SandboxPolicy(timeout_seconds=20.0))
        return DisabledSandboxBackend()

    @staticmethod
    def _task_description(goal: str, spec: SkillSpecification) -> str:
        public_spec = spec.to_dict()
        return (
            "Acquire the missing Jarvis capability as a local Python skill.\n"
            "Create main.py with def run(payload: dict) -> dict. Do not edit tests or skill_spec.json.\n"
            "Hidden verifier contents are unavailable and must not be guessed.\n"
            f"Original user goal: {goal}\n"
            f"Skill specification:\n{json.dumps(public_spec, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _repair_iterations(transitions: list[dict[str, Any]]) -> int:
        count = 0
        for metadata in transitions:
            action = metadata.get("action") or {}
            metrics = metadata.get("objective_metrics") or {}
            if action.get("action_type") == "RUN_TESTS" and metrics.get("tests_failed", 0):
                count += 1
        return count

    @staticmethod
    def _invalid_action_rate(transitions: list[dict[str, Any]]) -> float:
        if not transitions:
            return 0.0
        invalid = [
            1.0
            if (metadata.get("objective_metrics") or {}).get("invalid_action", False)
            else 0.0
            for metadata in transitions
        ]
        return sum(invalid) / len(invalid)

    @staticmethod
    def _execution_matches(success: bool, output: dict[str, Any], expected_output: dict[str, Any] | None) -> bool:
        if not success:
            return False
        if expected_output is None:
            return True
        return output == expected_output

    @staticmethod
    def _failed(goal: str, resolution: CapabilityResolution, trajectory: AcquisitionTrajectory, error: str) -> CapabilityAcquisitionResult:
        return CapabilityAcquisitionResult(
            goal=goal,
            success=False,
            resolution=resolution,
            trajectory_id=trajectory.trajectory_id,
            error=error,
        )
