from __future__ import annotations

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
from capabilities.research import CapabilityResearcher
from capabilities.resolver import CapabilityResolver
from capabilities.specification import SkillSpecificationGenerator
from capabilities.trajectory import AcquisitionTrajectory, AcquisitionTrajectoryStore
from capabilities.workspace import SkillWorkspaceManager
from development.memory import DevelopmentMemory
from development.software_engineer import AutonomousSoftwareEngineer, ProjectRequest
from environments.coding.environment import CodingEnvironment
from environments.coding.sandbox_backend import DisabledSandboxBackend, DockerSandboxBackend, SandboxBackend, SandboxPolicy


@dataclass
class CapabilityRuntimeConfig:
    data_dir: str = "data/capabilities"
    skills_root: str = "skills"
    max_build_steps: int = 14
    num_action_candidates: int = 6
    max_repair_cycles: int = 4
    max_blind_repair_cycles: int = 2
    use_docker: bool = True
    production_controller: str = "heuristic"
    learned_controller_mode: str = "shadow"
    seed: int = 404
    trace: bool = False
    enable_research: bool = True


class CapabilityAcquisitionRuntime:
    """v0.4 lifecycle for acquiring, promoting, and executing local capabilities."""

    def __init__(
        self,
        *,
        brain: Any | None = None,
        backend: SandboxBackend | None = None,
        config: CapabilityRuntimeConfig | None = None,
        researcher: CapabilityResearcher | None = None,
    ) -> None:
        self.config = config or CapabilityRuntimeConfig()
        self.root = Path(self.config.data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or self._default_backend()
        self.brain = brain
        self.registry = CapabilityRegistry(self.root / "registry.json")
        self.resolver = CapabilityResolver(self.registry, brain=brain)
        self.spec_generator = SkillSpecificationGenerator(brain=brain)
        self.researcher = researcher if researcher is not None else (CapabilityResearcher() if self.config.enable_research else None)
        self.permission_policy = PermissionPolicy()
        skills_root = Path(self.config.skills_root)
        self.workspace_manager = SkillWorkspaceManager(skills_root / "_staging")
        self.promoter = SkillPromoter(skills_root / "installed", self.registry)
        self.executor = CapabilityExecutor(self.root / "execution", self.backend)
        self.trajectory_store = AcquisitionTrajectoryStore(self.root / "acquisition_trajectories.jsonl")
        self.development_memory = DevelopmentMemory(self.root / "development_memory.jsonl")
        self.software_engineer = AutonomousSoftwareEngineer(
            brain=brain,
            backend=self.backend,
            memory=self.development_memory,
            trace=self.config.trace,
        )

    def handle_goal(
        self,
        goal: str,
        *,
        request_payload: dict[str, Any] | None = None,
        expected_output: dict[str, Any] | None = None,
        second_goal: str | None = None,
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

        research_note = None
        if spec is None and self.researcher is not None:
            research_note = self.researcher.research(goal)
            if research_note is not None:
                trajectory.record(AcquisitionStage.RESEARCH.value, research_note.to_dict())

        skill_spec = spec or self.spec_generator.generate(goal, research_note=research_note)
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
        project_request = ProjectRequest(
            goal=goal,
            specification=skill_spec,
            workspace=staged.root,
            test_command=["python", "-m", "unittest", "test_public.py"],
            protected_paths={"test_public.py", "skill_spec.json"},
            permissions=list(skill_spec.permissions),
            dependency_restrictions=list(skill_spec.constraints),
            max_repair_cycles=self.config.max_repair_cycles,
            max_blind_repair_cycles=self.config.max_blind_repair_cycles,
        )
        build_result = self.software_engineer.build(project_request)
        public_success = bool(build_result.public_test_result and build_result.public_test_result.success)
        internal_success = bool(build_result.internal_verification_success)
        reviewer_approved = bool(build_result.reviewer_approved)
        verifier_task = staged.to_task(
            "External hidden verifier for staged capability.",
            hidden_workspace=hidden_workspace,
            hidden_test_command=hidden_test_command or ([sys.executable, "hidden_verifier.py"] if hidden_workspace is not None else None),
            max_steps=1,
        )
        verifier_environment = CodingEnvironment(verifier_task, backend=self.backend)
        can_run_hidden = public_success and internal_success and reviewer_approved
        hidden_result = verifier_environment.run_final_hidden_verifier() if can_run_hidden else {"ran": False, "success": False, "runs": 0}
        hidden_success = bool(hidden_result.get("success", False))
        blind_repair_success = False
        blind_cycles = 0
        while can_run_hidden and not hidden_success and blind_cycles < self.config.max_blind_repair_cycles:
            blind_cycles += 1
            trajectory.record(
                AcquisitionStage.REPAIR.value,
                {"phase": "blind_hidden_repair", "cycle": blind_cycles, "message": "external acceptance verification failed"},
            )
            build_result = self.software_engineer.blind_generalization_repair(project_request, build_result)
            public_success = bool(build_result.public_test_result and build_result.public_test_result.success)
            internal_success = bool(build_result.internal_verification_success)
            reviewer_approved = bool(build_result.reviewer_approved)
            can_run_hidden = public_success and internal_success and reviewer_approved
            if not can_run_hidden:
                hidden_result = {"ran": False, "success": False, "runs": hidden_result.get("runs", 0)}
                hidden_success = False
                break
            verifier_environment = CodingEnvironment(verifier_task, backend=self.backend)
            hidden_result = verifier_environment.run_final_hidden_verifier()
            hidden_success = bool(hidden_result.get("success", False))
            blind_repair_success = hidden_success
        trajectory.record(AcquisitionStage.PLAN.value, {"plan": build_result.plan, "summary": build_result.summary})
        trajectory.record(AcquisitionStage.IMPLEMENT.value, {"files": [item.to_dict() for item in build_result.files], "llm_calls": build_result.llm_calls})
        trajectory.record(
            AcquisitionStage.TEST.value,
            {
                "public_success": public_success,
                "internal_verification_success": internal_success,
                "reviewer_approved": reviewer_approved,
                "hidden_success": hidden_success,
                "test_result": build_result.public_test_result.to_dict() if build_result.public_test_result else None,
                "internal_test_result": build_result.internal_test_result.to_dict() if build_result.internal_test_result else None,
                "hidden_runs": hidden_result.get("runs", 0),
            },
        )
        trajectory.record(AcquisitionStage.REPAIR.value, {"repairs": build_result.repairs, "failures": build_result.failures})
        trajectory.record(AcquisitionStage.VERIFY.value, {"hidden_result": {"ran": hidden_result.get("ran", False), "success": hidden_success}})

        promotion = self.promoter.promote(
            skill_spec,
            staged,
            public_success=public_success,
            internal_success=internal_success,
            reviewer_approved=reviewer_approved,
            hidden_success=hidden_success,
        )
        trajectory.record(AcquisitionStage.PROMOTE.value, {"promoted": promotion.promoted, "errors": promotion.errors, "manifest": promotion.manifest.to_dict() if promotion.manifest else None})
        if not promotion.promoted or promotion.manifest is None:
            result = CapabilityAcquisitionResult(
                goal=goal,
                success=False,
                resolution=resolution,
                capability_id=skill_spec.capability_id,
                public_success=public_success,
                internal_verification_success=internal_success,
                reviewer_approved=reviewer_approved,
                hidden_success=hidden_success,
                blind_repair_success=blind_repair_success,
                steps_to_acquisition=1 + build_result.repair_cycles,
                repair_iterations=build_result.repair_cycles + build_result.blind_repair_cycles,
                initial_implementation_pass=bool(build_result.success and build_result.repair_cycles == 0),
                llm_calls=build_result.llm_calls,
                token_usage=build_result.token_usage,
                development_state=build_result.final_state.value,
                trajectory_id=trajectory.trajectory_id,
                error="; ".join(promotion.errors),
            )
            self.software_engineer.record_lifecycle_memory(
                project_request,
                build_result,
                public_success=public_success,
                internal_verification_success=internal_success,
                reviewer_approved=reviewer_approved,
                hidden_success=hidden_success,
                promotion_success=False,
                execution_success=False,
                second_call_success=False,
            )
            self.trajectory_store.save(trajectory, result.to_dict())
            return result

        execution = self.executor.execute(promotion.manifest, payload)
        execution_ok = self._execution_matches(execution.success, execution.output, expected_output)
        trajectory.record(AcquisitionStage.EXECUTE.value, execution.__dict__)
        second_resolution = self.resolver.resolve(second_goal or goal)
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
            internal_verification_success=internal_success,
            reviewer_approved=reviewer_approved,
            hidden_success=hidden_success,
            blind_repair_success=blind_repair_success,
            execution_success=execution_ok,
            second_call_success=second_execution_ok,
            steps_to_acquisition=1 + build_result.repair_cycles,
            repair_iterations=build_result.repair_cycles + build_result.blind_repair_cycles,
            initial_implementation_pass=bool(build_result.success and build_result.repair_cycles == 0),
            llm_calls=build_result.llm_calls,
            token_usage=build_result.token_usage,
            development_state=build_result.final_state.value,
            trajectory_id=trajectory.trajectory_id,
            output=execution.output,
            error=execution.error,
        )
        self.software_engineer.record_lifecycle_memory(
            project_request,
            build_result,
            public_success=public_success,
            internal_verification_success=internal_success,
            reviewer_approved=reviewer_approved,
            hidden_success=hidden_success,
            promotion_success=True,
            execution_success=execution_ok,
            second_call_success=second_execution_ok,
        )
        self.trajectory_store.save(trajectory, result.to_dict())
        return result

    def _default_backend(self) -> SandboxBackend:
        if self.config.use_docker and DockerSandboxBackend.is_available():
            return DockerSandboxBackend(policy=SandboxPolicy(timeout_seconds=20.0))
        return DisabledSandboxBackend()

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
