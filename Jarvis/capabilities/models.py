from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AcquisitionStage(str, Enum):
    GOAL = "goal"
    GAP = "gap_detection"
    RESEARCH = "research"
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


class CapabilityLifecycle(str, Enum):
    DISCOVERED = "DISCOVERED"
    DRAFT = "DRAFT"
    TESTING = "TESTING"
    VERIFIED = "VERIFIED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    BROKEN = "BROKEN"
    DISABLED = "DISABLED"
    SUPERSEDED = "SUPERSEDED"


class CapabilityHealth(str, Enum):
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    AT_RISK = "AT_RISK"
    BROKEN = "BROKEN"


class CapabilityResolutionStatus(str, Enum):
    FOUND = "FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING = "MISSING"
    BROKEN = "BROKEN"


class RuntimeBrain(str, Enum):
    NONE = "none"
    LOCAL = "local"
    CODEX = "codex"


@dataclass
class GoalEnvelope:
    """A normalized owner goal for capability resolution.

    The resolver is allowed to classify cheaply, but it is not allowed to write
    code or plan tools. This object is the narrow contract it reasons over.
    """

    normalized_goal: str
    goal_type: str = ""
    target_type: str = ""
    target: str = ""
    constraints: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_text(cls, text: str) -> "GoalEnvelope":
        import re

        raw = " ".join(str(text or "").strip().split())
        folded = _fold(raw)
        goal_type = ""
        target_type = ""
        target = ""
        confidence = 0.0
        if re.search(r"\b(spiel\w*|play|pause|resume|weiter|stopp?\w*)\b", folded):
            goal_type = "PLAY_MEDIA"
            target_type = "MEDIA_QUERY"
            target = re.sub(r"\b(zeus|jarvis|bitte|please|spiel\w*|play|mach\w*|an|auf|von|by)\b", " ", folded).strip()
            confidence = 0.75
        elif re.search(r"\b(oeffne\w*|open|starte?|start)\b", folded) and re.search(r"\b(web|website|seite|wikipedia|github|youtube|https?://|www\.)\b", folded):
            goal_type = "OPEN_WEB_RESOURCE"
            target_type = "WEB_RESOURCE"
            target = raw
            confidence = 0.75
        elif re.search(r"\b(wie\s+spaet|uhrzeit|time|what\s+time)\b", folded):
            goal_type = "TIME_QUERY"
            target_type = "TIME"
            target = raw
            confidence = 0.7
        elif re.search(r"\b(gr(?:oe|o)ss\w*|biggest|largest|meisten\s+(?:platz|speicher)|frisst)\b", folded) and re.search(r"\b(ordner|folder|directory|datei|file)\b", folded):
            goal_type = "FILESYSTEM_LARGEST_CHILD"
            target_type = "DIRECTORY"
            target = raw
            confidence = 0.75
        return cls(raw, goal_type=goal_type, target_type=target_type, target=target, confidence=confidence)


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
    #: RUNTIME HEALTH, kept apart from ``status`` (INSTALLATION STATE).
    #: ``status == "active"`` says the capability is installed and enabled;
    #: it says nothing about whether the last real call worked.  ``health``
    #: is written by every execution: state (healthy | degraded | failing |
    #: unverified), consecutive failures, last ok/error and its text, call
    #: count, last_used.  A verified runtime failure changes it at once.
    health: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    family: str = ""
    goal_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    anti_examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    security_level: int = 0
    latency_class: str = "unknown"
    runtime_dependencies: list[str] = field(default_factory=list)
    last_verified: str = ""
    success_rate: float = 0.0
    failure_count: int = 0
    source: str = "codex_generated"
    implementation_path: str = ""
    runtime_brain: str = RuntimeBrain.NONE.value
    codex_required: bool = False
    created_by: str = "codex"
    supersedes: list[str] = field(default_factory=list)
    semantic_signature: str = ""
    lifecycle: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lifecycle"] = self.lifecycle_state().value
        data["health"] = self.health_view()
        data["health_state"] = self.health_state().value
        data["family"] = self.family or self.capability_id.split(".", 1)[0]
        data["implementation_path"] = self.implementation_path or self.source_location
        data["runtime_dependencies"] = self.runtime_dependencies or self.dependencies
        data["codex_required"] = bool(self.codex_required)
        data["runtime_brain"] = self.runtime_brain or RuntimeBrain.NONE.value
        data["semantic_signature"] = self.semantic_signature or self.compute_semantic_signature()
        return data

    def health_view(self) -> dict[str, Any]:
        base = {"state": "unverified", "health": CapabilityHealth.UNKNOWN.value, "consecutive_failures": 0, "calls": 0, "last_ok_at": "", "last_error_at": "", "last_error": "",
                "last_used": "", "last_verified": self.last_verified, "success_rate": self.success_rate, "failure_count": self.failure_count, "repairs": []}
        base.update(self.health or {})
        base["health"] = self.health_state().value
        return base

    def lifecycle_state(self) -> CapabilityLifecycle:
        raw = (self.lifecycle or self.status or "").strip().upper()
        aliases = {
            "ACTIVE": CapabilityLifecycle.ACTIVE,
            "ENABLED": CapabilityLifecycle.ACTIVE,
            "VERIFIED": CapabilityLifecycle.VERIFIED,
            "DISABLED": CapabilityLifecycle.DISABLED,
            "DEPRECATED": CapabilityLifecycle.SUPERSEDED,
            "SUPERSEDED": CapabilityLifecycle.SUPERSEDED,
            "DEGRADED": CapabilityLifecycle.DEGRADED,
            "BROKEN": CapabilityLifecycle.BROKEN,
            "FAILING": CapabilityLifecycle.BROKEN,
            "TESTING": CapabilityLifecycle.TESTING,
            "DRAFT": CapabilityLifecycle.DRAFT,
            "DISCOVERED": CapabilityLifecycle.DISCOVERED,
        }
        return aliases.get(raw, CapabilityLifecycle.ACTIVE if str(self.status).lower() == "active" else CapabilityLifecycle.DISABLED)

    def health_state(self) -> CapabilityHealth:
        raw = str((self.health or {}).get("health") or (self.health or {}).get("state") or "").strip().upper()
        aliases = {
            "HEALTHY": CapabilityHealth.HEALTHY,
            "OK": CapabilityHealth.HEALTHY,
            "AT_RISK": CapabilityHealth.AT_RISK,
            "DEGRADED": CapabilityHealth.AT_RISK,
            "FAILING": CapabilityHealth.BROKEN,
            "BROKEN": CapabilityHealth.BROKEN,
            "FAILED": CapabilityHealth.BROKEN,
            "UNVERIFIED": CapabilityHealth.UNKNOWN,
            "UNKNOWN": CapabilityHealth.UNKNOWN,
            "": CapabilityHealth.UNKNOWN,
        }
        return aliases.get(raw, CapabilityHealth.UNKNOWN)

    def is_active(self) -> bool:
        return self.lifecycle_state() in {CapabilityLifecycle.ACTIVE, CapabilityLifecycle.VERIFIED} and str(self.status).lower() == "active"

    def is_broken(self) -> bool:
        return self.lifecycle_state() is CapabilityLifecycle.BROKEN or self.health_state() is CapabilityHealth.BROKEN

    @property
    def declares_subject(self) -> bool:
        """Whether this capability says, in types, what it is for.

        Only such a capability can be compared with another one for sameness.
        A record that declares no goal or target type is not "about nothing" --
        it is undeclared, which is a different thing, and guessing that two
        undeclared capabilities are duplicates of each other would retire
        working functionality on no evidence at all.
        """

        return bool(self.goal_types or self.target_types)

    def compute_semantic_signature(self) -> str:
        """A hash of what this capability is FOR, not of what it is called.

        The identifier is deliberately excluded. Including it -- as the first
        version of this did -- makes the signature unique per capability by
        construction, which reads like duplicate detection and can never once
        detect a duplicate. Aliases are excluded for the mirror reason: they
        are learned phrasings that change as the capability is used, and an
        identity that changes every time something is said differently is not
        an identity.
        """

        import hashlib

        parts = [
            self.family or self.capability_id.split(".", 1)[0],
            " ".join(sorted(self.goal_types)),
            " ".join(sorted(self.target_types)),
            " ".join(sorted(self.side_effects)),
            " ".join(sorted(self.runtime_dependencies or self.dependencies)),
        ]
        if not self.declares_subject:
            # Undeclared: the signature stays unique so nothing is ever
            # deduplicated against it.
            parts = [self.capability_id, *parts, (self.description or "").strip()[:200]]
        subject = "|".join(part.strip().lower() for part in parts if part.strip())
        if not subject:
            subject = (self.description or self.capability_id).strip().lower()
        return hashlib.sha256(subject.encode("utf-8")).hexdigest()[:24]

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
            health=dict(data.get("health") or {}),
            name=str(data.get("name", "")),
            family=str(data.get("family", "")),
            goal_types=[str(item) for item in data.get("goal_types", [])],
            target_types=[str(item) for item in data.get("target_types", [])],
            examples=[str(item) for item in data.get("examples", [])],
            anti_examples=[str(item) for item in data.get("anti_examples", [])],
            aliases=[str(item) for item in data.get("aliases", [])],
            preconditions=[str(item) for item in data.get("preconditions", [])],
            side_effects=[str(item) for item in data.get("side_effects", [])],
            verification=dict(data.get("verification") or {}),
            security_level=int(data.get("security_level", 0) or 0),
            latency_class=str(data.get("latency_class", "unknown")),
            runtime_dependencies=[str(item) for item in data.get("runtime_dependencies", data.get("dependencies", []))],
            last_verified=str(data.get("last_verified", "")),
            success_rate=float(data.get("success_rate", 0.0) or 0.0),
            failure_count=int(data.get("failure_count", 0) or 0),
            source=str(data.get("source", "codex_generated")),
            implementation_path=str(data.get("implementation_path", data.get("source_location", ""))),
            runtime_brain=str(data.get("runtime_brain", RuntimeBrain.NONE.value) or RuntimeBrain.NONE.value),
            codex_required=bool(data.get("codex_required", False)),
            created_by=str(data.get("created_by", "codex")),
            supersedes=[str(item) for item in data.get("supersedes", [])],
            semantic_signature=str(data.get("semantic_signature", "")),
            lifecycle=str(data.get("lifecycle", "")),
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
        if self.runtime_brain not in {item.value for item in RuntimeBrain}:
            errors.append("runtime_brain must be none, local, or codex")
        if self.codex_required and self.runtime_brain != RuntimeBrain.CODEX.value:
            errors.append("codex_required capabilities must declare runtime_brain='codex'")
        if self.security_level < 0 or self.security_level > 3:
            errors.append("security_level must be between 0 and 3")
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
        else:
            for index, case in enumerate(self.public_tests):
                if not isinstance(case, dict) or not isinstance(case.get("input"), dict):
                    errors.append(f"public_tests[{index}] must be an object with an 'input' object")
                elif "expected" not in case and not case.get("expected_keys") and not case.get("raises"):
                    errors.append(
                        f"public_tests[{index}] must set 'expected', 'expected_keys', or 'raises'"
                    )
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
            family=str(self.metadata.get("family", self.capability_id.split(".", 1)[0])),
            goal_types=[str(item) for item in self.metadata.get("goal_types", [])],
            target_types=[str(item) for item in self.metadata.get("target_types", [])],
            examples=[str(item) for item in self.metadata.get("examples", [])],
            anti_examples=[str(item) for item in self.metadata.get("anti_examples", [])],
            aliases=[str(item) for item in self.metadata.get("aliases", [])],
            preconditions=[str(item) for item in self.metadata.get("preconditions", [])],
            side_effects=[str(item) for item in self.metadata.get("side_effects", [])],
            verification=dict(self.metadata.get("verification") or {}),
            security_level=int(self.metadata.get("security_level", 0) or 0),
            latency_class=str(self.metadata.get("latency_class", "unknown")),
            runtime_dependencies=list(self.allowed_dependencies),
            source=str(self.metadata.get("source", "codex_generated")),
            runtime_brain=str(self.metadata.get("runtime_brain", RuntimeBrain.NONE.value) or RuntimeBrain.NONE.value),
            codex_required=bool(self.metadata.get("codex_required", False)),
            created_by=str(self.metadata.get("created_by", "codex")),
            lifecycle=CapabilityLifecycle.VERIFIED.value,
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
    result: str = ""
    goal: GoalEnvelope | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def outcome(self) -> CapabilityResolutionStatus:
        raw = (self.result or self.status or "").strip().upper()
        aliases = {
            "AVAILABLE": CapabilityResolutionStatus.FOUND,
            "FOUND": CapabilityResolutionStatus.FOUND,
            "AMBIGUOUS": CapabilityResolutionStatus.AMBIGUOUS,
            "MISSING": CapabilityResolutionStatus.MISSING,
            "UNKNOWN": CapabilityResolutionStatus.MISSING,
            "BROKEN": CapabilityResolutionStatus.BROKEN,
            "FAILING": CapabilityResolutionStatus.BROKEN,
        }
        return aliases.get(raw, CapabilityResolutionStatus.MISSING)

    @property
    def found(self) -> bool:
        return self.outcome is CapabilityResolutionStatus.FOUND and self.manifest is not None

    @property
    def needs_engineer(self) -> bool:
        return self.outcome in {
            CapabilityResolutionStatus.MISSING,
            CapabilityResolutionStatus.BROKEN,
            CapabilityResolutionStatus.AMBIGUOUS,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.manifest is not None:
            data["manifest"] = self.manifest.to_dict()
        if self.goal is not None:
            data["goal"] = self.goal.to_dict()
        data["result"] = self.outcome.value
        return data


@dataclass
class CapabilityAcquisitionResult:
    goal: str
    success: bool
    resolution: CapabilityResolution
    capability_id: str | None = None
    promoted: bool = False
    public_success: bool = False
    internal_verification_success: bool = False
    reviewer_approved: bool = False
    hidden_success: bool = False
    blind_repair_success: bool = False
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
    codex_state: str = ""
    codex_checked: bool = False
    queued: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolution"] = self.resolution.to_dict()
        return data


def _fold(text: str) -> str:
    import unicodedata

    lowered = (text or "").lower().replace("ß", "ss")
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(source, target)
    return "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
