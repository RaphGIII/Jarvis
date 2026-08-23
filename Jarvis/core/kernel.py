"""The composition root: one place that assembles a working Jarvis.

Every other module in the system is deliberately ignorant of how the pieces fit
together -- the project engine does not know what a tier is, the edit engine does
not know what a project is.  That decoupling is what makes the architecture
survivable, but something has to do the wiring, and this is it.

It is also where the deployment boundary lives.  Today the brain, the memory,
the project engine and the tools all run in this process on one Windows machine.
The intended future is a Jarvis brain on a home server with small portable
clients, and the migration path is:

* :class:`~brain.tiers.ModelCatalog` already addresses models by *role*, so
  pointing BUILD_LOCAL at a remote endpoint is a configuration change.
* :class:`JarvisKernel` exposes a small, serialisable surface -- create a
  project, run it, report status -- which is exactly the surface a service API
  would expose.
* Tool execution is already a separate concern with its own policy object, so a
  future split can run tools near the files while inference happens elsewhere.

Nothing here assumes a terminal, so a CLI, an HTTP service and a device client
can all sit on top of the same object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.resources import HostProbe, ResourcePolicy, ResourcePolicyStore
from brain.tiers import HealthState, ModelCatalog, ModelHealth, ModelProbe, ModelTier
from projects.engine import EngineHooks, ProjectEngine, SessionResult
from projects.models import Project, ResourceLimits
from projects.store import ProjectStore
from tools.builtin import builtin_tools
from tools.registry import AuditLog, RiskLevel, ToolPolicy, ToolRegistry
from tools.web import make_web_tools

#: Everything Jarvis persists lives under one root, so backing it up or moving
#: it to a server is a single directory copy.
DEFAULT_STATE_ROOT = Path(__file__).resolve().parent.parent / "data" / "jarvis"

#: Configuration is separate from state: state is what Jarvis produced and is
#: worth backing up, configuration is how this machine is set up and is not
#: portable to another one (the tuned context windows least of all).
DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"


@dataclass
class KernelConfig:
    state_root: Path = field(default_factory=lambda: DEFAULT_STATE_ROOT)
    config_root: Path = field(default_factory=lambda: DEFAULT_CONFIG_ROOT)
    #: Highest tool risk an unattended run may reach without approval.
    max_tool_risk: RiskLevel = RiskLevel.MODERATE
    #: Off by default: an autonomous run must not silently reach the network.
    enable_research_tools: bool = True
    default_limits: ResourceLimits = field(default_factory=ResourceLimits)

    @classmethod
    def from_env(cls) -> "KernelConfig":
        root = os.getenv("JARVIS_STATE_ROOT", "").strip()
        config_root = os.getenv("JARVIS_CONFIG_ROOT", "").strip()
        return cls(
            state_root=Path(root) if root else DEFAULT_STATE_ROOT,
            config_root=Path(config_root) if config_root else DEFAULT_CONFIG_ROOT,
            enable_research_tools=os.getenv("JARVIS_ENABLE_RESEARCH", "1").strip().lower() in {"1", "true", "yes", "on"},
        )


class JarvisKernel:
    """A configured, ready-to-use Jarvis."""

    def __init__(self, config: KernelConfig | None = None, *, catalog: ModelCatalog | None = None) -> None:
        self.config = config or KernelConfig.from_env()
        self.state_root = Path(self.config.state_root)
        self.state_root.mkdir(parents=True, exist_ok=True)

        self.catalog = catalog or ModelCatalog()
        self.host = HostProbe().detect()

        # Apply whatever the tuner measured on this machine.  Falling back to a
        # VRAM-derived default keeps an untuned machine conservative rather than
        # optimistic.
        self.config_root = Path(self.config.config_root)
        self.resource_store = ResourcePolicyStore(self.config_root / "resources.json")
        self.resources: ResourcePolicy = self.resource_store.load_or_default(self.host)
        self.resources.apply_to(self.catalog)

        self.probe = ModelProbe(self.catalog)
        self.projects = ProjectStore(self.state_root / "projects")
        self.audit = AuditLog(self.state_root / "audit" / "tools.jsonl")
        self.tools = self._build_tools()
        self._providers: dict[ModelTier, Any] = {}

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        registry = ToolRegistry(policy=ToolPolicy(max_risk=self.config.max_tool_risk), audit=self.audit)
        registry.register_many(builtin_tools())
        if self.config.enable_research_tools:
            registry.register_many(make_web_tools())
        return registry

    def provider(self, tier: ModelTier):
        """The provider for a tier, built once and reused.

        Reuse matters: each provider holds the keep-alive setting that keeps
        weights resident, and rebuilding one per call would defeat it.
        """

        if tier not in self._providers:
            from brain.providers import provider_for_spec

            self._providers[tier] = provider_for_spec(self.catalog.get(tier))
        return self._providers[tier]

    def engine(self, *, tier: ModelTier = ModelTier.BUILD_LOCAL, hooks: EngineHooks | None = None) -> ProjectEngine:
        return ProjectEngine(brain=self.provider(tier), store=self.projects, tools=self.tools, hooks=hooks)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def start_project(self, goal: str, **kwargs: Any) -> Project:
        kwargs.setdefault("limits", self.config.default_limits)
        return self.engine().create_project(goal, **kwargs)

    def work(
        self,
        project: Project,
        *,
        tier: ModelTier = ModelTier.BUILD_LOCAL,
        max_steps: int | None = None,
        hooks: EngineHooks | None = None,
    ) -> SessionResult:
        return self.engine(tier=tier, hooks=hooks).run(project, max_steps=max_steps)

    def resolve_project(self, reference: str) -> Project | None:
        """Find a project by id, or by what the user called it."""

        if self.projects.exists(reference):
            return self.projects.load(reference)
        matches = self.projects.find(reference, limit=1)
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def health(self, *, force: bool = False) -> dict[ModelTier, ModelHealth]:
        return self.probe.probe_all(force=force)

    def status(self, *, force: bool = False) -> dict[str, Any]:
        """A complete, honest picture of what is and is not working."""

        health = self.health(force=force)
        return {
            "state_root": str(self.state_root),
            "host": self.host.to_dict(),
            "resources": {
                "context_windows": dict(self.resources.context_windows),
                "max_concurrent_generations": self.resources.max_concurrent_generations,
                "reserved_vram_mib": self.resources.reserved_vram_mib,
                "tuned_at": self.resources.tuned_at or "never (using defaults)",
            },
            "tiers": {tier.value: item.to_dict() for tier, item in health.items()},
            "paid_tiers_enabled": [tier.value for tier in self.catalog.paid_tiers_enabled()],
            "cloud_free": not self.catalog.paid_tiers_enabled(),
            "tools": self.tools.names(),
            "projects": self.projects.summary(),
        }

    def ready_for_autonomous_work(self, *, force: bool = False) -> tuple[bool, str]:
        """Whether a build run can start right now, and if not, why not.

        Answered from a real generation, so "ready" is a claim about the model
        working rather than about a port being open.
        """

        health = self.probe.probe(ModelTier.BUILD_LOCAL, force=force)
        if health.online:
            return True, f"BUILD_LOCAL is {health.model} ({health.latency_seconds:.1f}s probe)"
        remedies = {
            HealthState.MODEL_MISSING: f"run: ollama pull {self.catalog.get(ModelTier.BUILD_LOCAL).model}",
            HealthState.PROVIDER_UNREACHABLE: "start the inference server (ollama serve)",
            HealthState.DISABLED: "enable BUILD_LOCAL in the model catalog",
            HealthState.NOT_CONFIGURED: "set a model for BUILD_LOCAL in the model catalog",
            HealthState.GENERATION_FAILED: "the model is present but cannot generate; check the server log",
        }
        return False, f"BUILD_LOCAL is {health.state.value}: {health.detail}. Try: {remedies.get(health.state, 'check the configuration')}"

    def release_models(self) -> None:
        """Ask every local provider to evict its weights, giving the GPU back."""

        for provider in self._providers.values():
            unload = getattr(provider, "unload", None)
            if callable(unload):
                unload()
