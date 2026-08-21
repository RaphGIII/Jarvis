from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AcquisitionStage(str, Enum):
    GOAL = "goal"
    GAP = "gap_detection"
    SPEC = "specification"
    PLAN = "plan"
    IMPLEMENT = "implement"
    BUILD = "build"
    TEST = "test"
    REPAIR = "repair"
    VERIFY = "verify"
    PROMOTE = "promote"
    EXECUTE = "execute"
    SECOND_CALL = "second_call"


@dataclass
class CapabilityManifest:
    capability_id: str
    description: str
    version: str = "1.0.0"
    status: str = "active"
    entrypoint: str = "main.py"
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    permissions_required: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    source_location: str = ""
    tests_location: str = ""
    creation_metadata: dict[str, Any] = field(default_factory=dict)
    validation_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityManifest":
        return cls(
            capability_id=str(data["capability_id"]),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0.0")),
            status=str(data.get("status", "active")),
            entrypoint=str(data.get("entrypoint", "main.py")),
            input_schema=dict(data.get("input_schema") or {}),
            output_schema=dict(data.get("output_schema") or {}),
            permissions_required=list(data.get("permissions_required") or []),
            dependencies=list(data.get("dependencies") or []),
            source_location=str(data.get("source_location", "")),
            tests_location=str(data.get("tests_location", "")),
            creation_metadata=dict(data.get("creation_metadata") or {}),
            validation_status=dict(data.get("validation_status") or {}),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.capability_id.strip():
            errors.append("capability_id is required")
        if not self.description.strip():
            errors.append("description is required")
        if not self.version.strip():
            errors.append("version is required")
        if not self.entrypoint.strip():
            errors.append("entrypoint is required")
        if not isinstance(self.input_schema, dict):
            errors.append("input_schema must be an object")
        if not isinstance(self.output_schema, dict):
            errors.append("output_schema must be an object")
        return errors


@dataclass
class SkillSpecification:
    capability_id: str
    objective: str
    functional_requirements: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    allowed_dependencies: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    public_tests: list[dict[str, Any]] = field(default_factory=list)
    proposed_file_structure: list[str] = field(default_factory=lambda: ["main.py"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSpecification":
        return cls(
            capability_id=str(data["capability_id"]),
            objective=str(data.get("objective", "")),
            functional_requirements=[str(item) for item in data.get("functional_requirements", [])],
            inputs=dict(data.get("inputs") or {}),
            outputs=dict(data.get("outputs") or {}),
            constraints=[str(item) for item in data.get("constraints", [])],
            allowed_dependencies=[str(item) for item in data.get("allowed_dependencies", [])],
            permissions=[str(item) for item in data.get("permissions", [])],
            acceptance_criteria=[str(item) for item in data.get("acceptance_criteria", [])],
            public_tests=list(data.get("public_tests") or []),
            proposed_file_structure=[str(item) for item in data.get("proposed_file_structure", ["main.py"])],
            metadata=dict(data.get("metadata") or {}),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.capability_id.strip():
            errors.append("capability_id is required")
        if not self.objective.strip():
            errors.append("objective is required")
        if not self.acceptance_criteria:
            errors.append("acceptance_criteria are required")
        if not self.public_tests:
            errors.append("public_tests are required")
        if "main.py" not in self.proposed_file_structure:
            errors.append("main.py must be part of proposed_file_structure")
        return errors

    def to_manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            capability_id=self.capability_id,
            description=self.objective,
            version=str(self.metadata.get("version", "1.0.0")),
            entrypoint="main.py",
            input_schema=self.inputs,
            output_schema=self.outputs,
            permissions_required=list(self.permissions),
            dependencies=list(self.allowed_dependencies),
            creation_metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "capability_acquisition_v04",
                **dict(self.metadata),
            },
        )


@dataclass
class CapabilityResolution:
    status: str
    capability_id: str | None = None
    reason: str = ""
    confidence: float = 0.0
    manifest: CapabilityManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.manifest is not None:
            data["manifest"] = self.manifest.to_dict()
        return data


@dataclass
class CapabilityAcquisitionResult:
    goal: str
    success: bool
    resolution: CapabilityResolution
    capability_id: str | None = None
    promoted: bool = False
    public_success: bool = False
    hidden_success: bool = False
    execution_success: bool = False
    second_call_success: bool = False
    steps_to_acquisition: int = 0
    repair_iterations: int = 0
    invalid_action_rate: float = 0.0
    initial_implementation_pass: bool = False
    llm_calls: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    development_state: str = ""
    trajectory_id: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolution"] = self.resolution.to_dict()
        return data
