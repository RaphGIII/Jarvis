"""The learning loop, from the owner's request down to the registry.

Codex is the engineer, not the runtime. These tests pin the two halves of that
sentence: a request for something ZEUS already knows never reaches an engineer
and never even asks whether one is available, and a request for something it
does not know reaches one *before* anything is generated locally.

The live end-to-end proof (real Codex, real restart, real paraphrase) is in
``docs``/the sprint report; what is here is the part that must keep holding on
every commit, in seconds, without a network.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from capabilities.codex import CapabilityRequestQueue, CodexAvailabilityState, StaticCodexAvailability
from capabilities.models import (
    CapabilityHealth,
    CapabilityLifecycle,
    CapabilityManifest,
    CapabilityResolutionStatus,
)
from capabilities.registry import CapabilityRegistry
from capabilities.resolver import CapabilityResolver
from capabilities.service import CapabilityService
from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from service.core import JarvisCore


# --------------------------------------------------------------------------
# Registry: migration, deduplication, learned phrasings
# --------------------------------------------------------------------------


def _manifest(capability_id: str, description: str, **kwargs: Any) -> CapabilityManifest:
    fields: dict[str, Any] = {
        "lifecycle": CapabilityLifecycle.ACTIVE.value,
        "health": {"state": "healthy", "health": CapabilityHealth.HEALTHY.value},
    }
    fields.update(kwargs)
    return CapabilityManifest(capability_id, description, **fields)


def test_a_v1_registry_is_migrated_on_read_and_rewritten(tmp_path: Path) -> None:
    """A pre-migration record has none of the fields the resolver ranks on."""

    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "capabilities": {
                    "archive.zip.create": {
                        "capability_id": "archive.zip.create",
                        "description": "Package a folder into a single .zip archive.",
                        "status": "active",
                        "version": "1.0.0",
                        "source_location": "skills/installed/archive.zip.create/1.0.0",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    registry = CapabilityRegistry(path)
    manifest = registry.get("archive.zip.create")

    assert manifest is not None
    assert manifest.family == "archive"
    assert manifest.semantic_signature
    assert manifest.implementation_path == "skills/installed/archive.zip.create/1.0.0"
    assert manifest.lifecycle_state() is CapabilityLifecycle.ACTIVE
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2, "the migration is persisted, not recomputed forever"


def test_a_duplicate_subject_supersedes_the_older_capability_and_inherits_its_phrasings(tmp_path: Path) -> None:
    """Two names for one thing is a resolver that can only guess."""

    registry = CapabilityRegistry(tmp_path / "registry.json")
    first = registry.register(
        _manifest("files.checksum", "Compute a checksum of a file.", family="files",
                  goal_types=["FILE_CHECKSUM"], target_types=["FILE"], aliases=["hash a file"])
    )
    registry.learn_alias(first.capability_id, "Was ist die Pruefsumme der Datei?")

    second = registry.register(
        _manifest("files.checksum_v2", "Compute a checksum of a file, faster.", family="files",
                  goal_types=["FILE_CHECKSUM"], target_types=["FILE"])
    )

    assert registry.get("files.checksum").lifecycle_state() is CapabilityLifecycle.SUPERSEDED
    assert not registry.get("files.checksum").is_active()
    assert second.supersedes == ["files.checksum"]
    assert "hash a file" in second.aliases
    assert "Was ist die Pruefsumme der Datei?" in second.aliases, "the paraphrases the old one learned are not lost with it"
    assert registry.get("files.checksum").validation_status["superseded_by"] == "files.checksum_v2"


def test_a_learned_phrasing_makes_a_paraphrase_resolvable(tmp_path: Path) -> None:
    """The whole point of storing an alias: the next wording costs nothing."""

    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_manifest("files.sha256", "Compute the SHA-256 digest of a file."))

    paraphrase = "bilde mir den fingerprint der datei"
    assert registry.find(paraphrase) == []

    registry.learn_alias("files.sha256", "bilde mir den fingerprint der datei bericht.txt")

    assert [m.capability_id for m in registry.find(paraphrase)] == ["files.sha256"]


def test_an_alias_is_stored_without_its_particulars_once_and_survives_a_reload(tmp_path: Path) -> None:
    """The wording is kept; the file it happened to name is not.

    Storing "Pruefsumme von bericht.txt" verbatim would make this capability
    answer to the words *bericht* and *txt*, so "oeffne bericht.txt" would
    resolve to a checksum. Measured before this was fixed: confidence 0.57.
    """

    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_manifest("files.sha256", "Compute the SHA-256 digest of a file."))
    registry.learn_alias("files.sha256", "Pruefsumme von bericht.txt")
    registry.learn_alias("files.sha256", "pruefsumme von urlaub.txt")

    reloaded = CapabilityRegistry(tmp_path / "registry.json")

    aliases = reloaded.get("files.sha256").aliases
    assert len(aliases) == 1, "two requests about different files are one phrasing"
    assert aliases[0].startswith("Pruefsumme von ")
    assert "bericht" not in aliases[0] and "urlaub" not in aliases[0]


# --------------------------------------------------------------------------
# The service: one routing decision, no model, no engineer
# --------------------------------------------------------------------------


def _service(tmp_path: Path) -> CapabilityService:
    return _core(tmp_path).capabilities


def test_resolve_request_answers_found_missing_and_broken(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.registry.register(
        _manifest("files.sha256", "Compute the SHA-256 digest of a file.",
                  goal_types=["FILE_CHECKSUM"], target_types=["FILE"],
                  examples=["sha256 digest of a file"], aliases=["sha256 checksum"])
    )

    found = service.resolve_request("compute the sha256 checksum of bericht.txt")
    assert found.outcome is CapabilityResolutionStatus.FOUND
    assert found.capability_id == "files.sha256"
    assert service.resolve("compute the sha256 checksum of bericht.txt") is not None

    missing = service.resolve_request("dim the kitchen lights to forty percent")
    assert missing.outcome is CapabilityResolutionStatus.MISSING
    assert missing.manifest is None
    assert service.resolve("dim the kitchen lights to forty percent") is None

    service.registry.note_execution("files.sha256", False, "boom")
    service.registry.note_execution("files.sha256", False, "boom again")
    broken = service.resolve_request("compute the sha256 checksum of bericht.txt")
    assert broken.outcome is CapabilityResolutionStatus.BROKEN
    assert broken.capability_id == "files.sha256"


def test_resolution_carries_the_runners_up_as_evidence(tmp_path: Path) -> None:
    """A routing decision that cannot be second-guessed is not evidence."""

    service = _service(tmp_path)
    service.registry.register(_manifest("files.sha256", "Compute the SHA-256 digest of a file.", aliases=["sha256 checksum"]))
    service.registry.register(_manifest("files.line_count", "Count the lines in a text file.", aliases=["count file lines"]))

    resolution = service.resolve_request("compute the sha256 checksum of bericht.txt")

    assert resolution.candidates
    assert resolution.candidates[0]["capability_id"] == "files.sha256"
    assert "confidence" in resolution.candidates[0] and "reason" in resolution.candidates[0]


# --------------------------------------------------------------------------
# The queue: what ZEUS still owes when Codex is not there
# --------------------------------------------------------------------------


def test_a_queued_request_stops_being_queued_once_it_is_served(tmp_path: Path) -> None:
    queue = CapabilityRequestQueue(tmp_path / "requests.jsonl")
    first = queue.enqueue("compute a sha256 checksum", reason="Codex is OFFLINE")
    queue.enqueue("dim the lights", reason="Codex is OFFLINE")

    assert [r.goal for r in queue.list()] == ["compute a sha256 checksum", "dim the lights"]

    queue.resolve(first.request_id, detail="served by files.sha256")

    assert [r.goal for r in queue.list()] == ["dim the lights"]
    served = [r for r in queue.list(status="served")]
    assert len(served) == 1 and served[0].goal == "compute a sha256 checksum"
    assert served[0].reason == "served by files.sha256"


# --------------------------------------------------------------------------
# The owner path: FOUND runs locally, MISSING asks the engineer first
# --------------------------------------------------------------------------


class _NoModel:
    """Any model call from the capability path is the defect under test."""

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("routing a known capability must not call a model")

    def generate_stream(self, *args: Any, **kwargs: Any):
        raise AssertionError("routing a known capability must not call a model")


def _core(tmp_path: Path) -> JarvisCore:
    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state"))
    kernel.provider = lambda tier: _NoModel()  # type: ignore[assignment]
    kernel.catalog.get = lambda tier: type("S", (), {"model": "stub"})()  # type: ignore[assignment]
    return JarvisCore(kernel=kernel, identity=Identity())


def _install(core: JarvisCore, capability_id: str, body: str, **fields: Any) -> CapabilityManifest:
    """A capability on disk exactly as an acquisition leaves one."""

    root = Path(core.kernel.state_root) / "capabilities" / "installed" / capability_id / "1.0.0"
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(body, encoding="utf-8")
    manifest = _manifest(
        capability_id,
        fields.pop("description", "A capability installed for this test."),
        source_location=str(root),
        implementation_path=str(root),
        entrypoint="main.py",
        **{"input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}},
           "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
           **fields},
    )
    return core.capabilities.registry.register(manifest)


def test_a_known_capability_is_dispatched_locally_and_never_checks_codex(tmp_path: Path) -> None:
    core = _core(tmp_path)
    _install(
        core,
        "files.sha256",
        "def run(payload):\n    return {'ok': True, 'detail': 'digest computed'}\n",
        description="Compute the SHA-256 digest of a file.",
        aliases=["sha256 checksum of a file"],
    )
    checked: list[str] = []
    core.capabilities.ensure = lambda *a, **k: checked.append("ensure")  # type: ignore[assignment]

    events: list[dict[str, Any]] = []
    core.emit = lambda kind, payload, scope="": events.append({"kind": kind, **payload})  # type: ignore[assignment]
    delivered: list[str] = []
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]

    core._route_capability_goal("compute the sha256 checksum of bericht.txt",
                                "Zeus, compute the sha256 checksum of bericht.txt", "test")

    routing = [e for e in events if "capability_routing" in e]
    assert routing, "the routing decision must be recorded as evidence"
    assert routing[0]["capability_routing"]["result"] == "FOUND"
    assert routing[0]["capability_routing"]["capability_id"] == "files.sha256"
    assert routing[0]["capability_routing"]["codex_checked"] is False
    assert checked == [], "a known capability must not start an acquisition"
    assert delivered and "digest computed" in delivered[0]


def test_a_verified_run_teaches_the_registry_the_owner_s_own_wording(tmp_path: Path) -> None:
    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True, 'detail': 'digest computed'}\n",
             description="Compute the SHA-256 digest of a file.", aliases=["sha256 checksum of a file"])
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    core._deliver = lambda *a, **k: None  # type: ignore[assignment]

    phrase = "compute the sha256 checksum of bericht.txt"
    core._route_capability_goal(phrase, phrase, "test")

    aliases = core.capabilities.registry.get("files.sha256").aliases
    assert "compute the sha256 checksum of a file" in aliases
    assert not any("bericht" in alias for alias in aliases), "the file it was asked about is not what it answers to"


def test_a_missing_capability_reaches_the_engineer_and_nothing_else(tmp_path: Path) -> None:
    core = _core(tmp_path)
    started: list[tuple[str, str]] = []
    core._start_capability_engineering = (  # type: ignore[assignment]
        lambda goal, text, scope, capability_id="", repair="": started.append((goal, repair))
    )
    events: list[dict[str, Any]] = []
    core.emit = lambda kind, payload, scope="": events.append({"kind": kind, **payload})  # type: ignore[assignment]
    core._deliver = lambda *a, **k: None  # type: ignore[assignment]

    core._route_capability_goal("dim the kitchen lights to forty percent",
                                "Zeus, dim the kitchen lights to forty percent", "test")

    assert [e["capability_routing"]["result"] for e in events if "capability_routing" in e] == ["MISSING"]
    assert started == [("dim the kitchen lights to forty percent", "")]


def test_a_broken_capability_is_repaired_rather_than_executed_or_rebuilt_blind(tmp_path: Path) -> None:
    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n",
             description="Compute the SHA-256 digest of a file.", aliases=["sha256 checksum of a file"])
    core.capabilities.registry.note_execution("files.sha256", False, "hashlib is missing")
    core.capabilities.registry.note_execution("files.sha256", False, "hashlib is missing")
    started: list[tuple[str, str]] = []
    core._start_capability_engineering = (  # type: ignore[assignment]
        lambda goal, text, scope, capability_id="", repair="": started.append((capability_id, repair))
    )
    events: list[dict[str, Any]] = []
    core.emit = lambda kind, payload, scope="": events.append({"kind": kind, **payload})  # type: ignore[assignment]
    core._deliver = lambda *a, **k: None  # type: ignore[assignment]

    core._route_capability_goal("compute the sha256 checksum of bericht.txt",
                                "Zeus, compute the sha256 checksum of bericht.txt", "test")

    assert [e["capability_routing"]["result"] for e in events if "capability_routing" in e] == ["BROKEN"]
    assert started == [("files.sha256", "hashlib is missing")], "the defect is the repair brief, not a blank page"


def test_learning_something_already_learned_runs_it_instead_of_acquiring_it_again(tmp_path: Path) -> None:
    """"Lerne X" is a request about a capability; the registry decides, not the wording."""

    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True, 'detail': 'digest computed'}\n",
             description="Compute the SHA-256 digest of a file.",
             aliases=["sha256 checksum of a file", "lerne wie man den sha256 checksum einer datei bildet"])
    created: list[str] = []
    core.missions.create = lambda *a, **k: created.append("mission")  # type: ignore[assignment]
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    delivered: list[str] = []
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]

    core._answer_by_acquisition("Lerne wie man den sha256 checksum einer Datei bildet", "test")

    assert created == [], "an acquisition mission for something already installed is wasted work"
    assert delivered and "digest computed" in delivered[0]


def test_the_capability_request_queue_is_visible_through_the_api(tmp_path: Path) -> None:
    core = _core(tmp_path)
    queue = core._capability_request_queue()
    queue.enqueue("dim the kitchen lights", reason="Codex is OFFLINE")

    payload = core.capability_requests()

    assert payload["ok"] and payload["count"] == 1
    assert payload["requests"][0]["goal"] == "dim the kitchen lights"
    assert payload["requests"][0]["status"] == "queued"


def test_serving_a_queued_goal_closes_its_request(tmp_path: Path) -> None:
    core = _core(tmp_path)
    core._capability_request_queue().enqueue("compute the sha256 checksum of a file", reason="Codex is OFFLINE")
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n",
             description="Compute the SHA-256 digest of a file.",
             aliases=["compute the sha256 checksum of a file"])

    resolved = core._resolve_capability_requests("compute the sha256 checksum of a file")

    assert len(resolved) == 1
    assert core.capability_requests()["count"] == 0


# --------------------------------------------------------------------------
# The engineer's own interface: flags the installed CLI accepts, and a
# refusal that becomes a queued request rather than a failed build
# --------------------------------------------------------------------------


def test_the_cli_is_driven_with_flags_the_installed_codex_accepts(tmp_path: Path) -> None:
    """Measured against codex-cli 0.153.4: ``--full-auto`` does not exist.

    ``codex exec`` rejects it and exits immediately, which in the acquisition
    log is indistinguishable from an expert that tried and failed. The flags
    asserted here are the ones that were observed to start a run.
    """

    from experts.codex import CodexExpert

    expert = CodexExpert(executable="codex")
    recorded: dict[str, Any] = {}

    class _Completed:
        returncode = 0
        stdout = "done"
        stderr = ""

    def _fake_run(command, **kwargs):
        recorded["command"] = list(command)
        recorded["kwargs"] = kwargs
        return _Completed()

    import subprocess

    from experts.contracts import ExpertJob

    real_run = subprocess.run
    subprocess.run = _fake_run  # type: ignore[assignment]
    try:
        expert.execute(ExpertJob(goal="build it", workspace=tmp_path / "ws"))
    finally:
        subprocess.run = real_run  # type: ignore[assignment]

    command = recorded["command"]
    assert command[:2] == ["codex", "exec"]
    assert "--full-auto" not in command, "rejected by the installed CLI"
    assert command[2:5] == ["--sandbox", "workspace-write", "--skip-git-repo-check"]
    assert "-C" in command
    assert recorded["kwargs"]["stdin"] is subprocess.DEVNULL, "a service has no terminal to read from"


def test_a_spent_allowance_is_remembered_so_availability_stops_saying_ready() -> None:
    """``codex --version`` answers from a binary and knows nothing about the account."""

    from experts.codex import CodexExpert

    expert = CodexExpert(executable="codex")
    expert.note_quota_exhausted("You've hit your usage limit. try again at 5:29 AM", seconds=600.0)

    assert expert._quota_block_active()
    assert "usage limit" in expert._quota_detail


def test_a_quota_refusal_queues_the_request_instead_of_reporting_a_failed_build(tmp_path: Path) -> None:
    from experts.contracts import ExpertResult, ExpertStatus, QuotaState
    from service.acquisition import AcquisitionMission

    class _Service:
        def __init__(self, root: Path) -> None:
            self.root = root

        def ensure(self, *a: Any, **k: Any) -> None:
            raise AssertionError("codex_first must not fall back to BUILD_LOCAL")

        def _start_project(self, goal: str, capability_id: str, **kwargs: Any) -> Any:
            workspace = self.root / "ws" / capability_id
            workspace.mkdir(parents=True, exist_ok=True)
            return type("Project", (), {"workspace": str(workspace), "constraints": [], "acceptance": []})()

        @staticmethod
        def suggest_id(goal: str) -> str:
            return "files.checksum.sha256"

    class _Gateway:
        def status(self) -> dict[str, Any]:
            return {"expert_available": True, "state": "AVAILABLE"}

        def submit(self, job: Any) -> ExpertResult:
            return ExpertResult(status=ExpertStatus.UNAVAILABLE, provider="codex",
                                blocker="You've hit your usage limit",
                                quota=QuotaState(exhausted=True, detail="You've hit your usage limit"))

    class _Kernel:
        state_root = tmp_path

    mission = AcquisitionMission(
        service=_Service(tmp_path), kernel=_Kernel(), gateway=_Gateway(),
        codex_availability=StaticCodexAvailability(CodexAvailabilityState.READY, "ready"),
        memory=None, ledger=None, missions=None,
    )

    result = mission.run("compute a sha256 checksum", codex_first=True)

    assert not result.acquired
    assert result.codex_state == "QUOTA_EXHAUSTED"
    assert result.queued and result.queue_id
    assert "usage limit" in result.reason
    queued = CapabilityRequestQueue(tmp_path / "capabilities" / "requests.jsonl").list()
    assert [r.goal for r in queued] == ["compute a sha256 checksum"]


# --------------------------------------------------------------------------
# What the engineer costs, and what it is allowed to see
# --------------------------------------------------------------------------


def test_a_spent_subscription_never_falls_back_to_a_metered_channel() -> None:
    """The failure mode this whole policy exists for: "it was busy" becoming a bill."""

    from runtime.cost_policy import CostPolicy, SpendChannel

    policy = CostPolicy.load()

    assert policy.permits(SpendChannel.SUBSCRIPTION_CLI)
    assert not policy.permits(SpendChannel.PAID_API)
    assert SpendChannel.PAID_API not in policy.fallbacks_for(SpendChannel.SUBSCRIPTION_CLI)
    assert all(channel is SpendChannel.LOCAL_MODEL for channel in policy.fallbacks_for(SpendChannel.SUBSCRIPTION_CLI))
    assert policy.metered_channels_enabled == []


def test_the_only_registered_engineer_is_the_subscription_cli(tmp_path: Path) -> None:
    """A second provider on a metered channel would be a fallback nobody chose."""

    from runtime.cost_policy import SpendChannel
    from service.acquisition import AcquisitionMission

    class _Kernel:
        state_root = tmp_path

    gateway = AcquisitionMission(service=None, kernel=_Kernel()).gateway

    assert [provider.name for provider in gateway.providers] == ["codex"]
    assert [provider.channel for provider in gateway.providers] == [SpendChannel.SUBSCRIPTION_CLI]


def test_the_owner_password_is_not_reachable_by_the_engineer(tmp_path: Path) -> None:
    """What is actually guaranteed, stated exactly.

    Codex runs with ``--sandbox workspace-write``: it may only WRITE inside the
    capability workspace, but like any local process it can read the disk. So
    "the engineer cannot see the password" is not a claim about the sandbox --
    it is a claim about there being no password anywhere to see. What is
    stored is a scrypt verifier under a random salt (DPAPI-wrapped on Windows),
    and the plaintext exists only inside the frame that checks it.

    The two things the sandbox does guarantee are asserted alongside it: the
    engineer's environment carries no credential, and the only paths it is
    asked to touch are the capability's own two files.
    """

    from experts.contracts import ExpertJob
    from experts.codex import CodexExpert

    password = "a-password-the-engineer-must-never-see"
    core = _core(tmp_path)
    state_root = Path(core.kernel.state_root).resolve()
    core.security.setup(password)

    project = core.capabilities._start_project("build something", "files.sha256", max_steps=4)
    workspace = Path(str(project.workspace)).resolve()
    job = ExpertJob(goal="build something", workspace=workspace,
                    allowed_paths=["main.py", "test_capability.py"])

    assert job.allowed_paths == ["main.py", "test_capability.py"]
    assert password not in job.brief()
    leaked = [
        path
        for path in state_root.rglob("*")
        if path.is_file() and password in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert leaked == [], f"the password is stored in plaintext at {leaked}"
    assert core.security.unlock(password, "PROJECT_DELETE").get("ok"), "and it still verifies"
    assert not any(_SECRETISH(name) for name in CodexExpert(executable="")._environment())


def _SECRETISH(name: str) -> bool:
    from experts.codex import _SECRET_EXCEPTIONS, _SECRET_NAME

    return bool(_SECRET_NAME.search(name)) and name.upper() not in _SECRET_EXCEPTIONS


# --------------------------------------------------------------------------
# The universe: internal work is not an owner project
# --------------------------------------------------------------------------


def test_capability_and_selfdev_work_stays_out_of_the_default_universe(tmp_path: Path) -> None:
    from projects.models import Project

    core = _core(tmp_path)
    owner = Project(id="proj_owner", goal="Schach", kind="software", title="Schach")
    acquisition = Project(id="proj_acq", goal="Build a reusable capability that can: compute a sha256 checksum",
                          kind="capability", title="files.sha256")
    acquisition.metadata["capability_id"] = "files.sha256"
    core.kernel.projects.save(owner)
    core.kernel.projects.save(acquisition)

    default = core.project_graph()
    everything = core.project_graph(everything=True)

    labels = {node["label"] for node in default["nodes"] if node["kind"] == "project"}
    assert "Schach" in labels
    assert "files.sha256" not in labels, "an acquisition attempt is ZEUS's own work, not an owner project"
    assert not [node for node in default["nodes"] if node["kind"] == "capability"]
    assert [node for node in everything["nodes"] if node["kind"] == "capability"], "'show everything' still reveals it"


# --------------------------------------------------------------------------
# Generalization: what is learned is the kind of request, not the instance
# --------------------------------------------------------------------------


def test_the_particulars_of_a_request_are_not_part_of_what_gets_built() -> None:
    from capabilities.generalize import generalize, generic_keywords

    shape = generalize("Berechne mir die SHA-256-Pruefsumme der Datei zeus_acceptance.txt")

    assert "zeus_acceptance.txt" not in shape.goal
    assert shape.particulars == ["zeus_acceptance.txt"]
    assert "einer Datei" in shape.goal
    assert "sha" in shape.goal.lower()
    assert generic_keywords(["berechne", "sha", "pruefsumme", "datei", "zeus_acceptance", "txt"], shape.particulars) == [
        "berechne", "sha", "pruefsumme", "datei",
    ]


def test_paths_drives_and_quoted_literals_are_all_particulars() -> None:
    from capabilities.generalize import generalize

    assert generalize(r"Mach einen Screenshot von D:\Bilder\urlaub.png").particulars == [r"D:\Bilder\urlaub.png"]
    assert "a file" in generalize("compute the sha256 checksum of report.txt").goal
    assert generalize('Suche nach "Rammstein" in der Bibliothek').particulars == ["Rammstein"]
    assert generalize("Wie viele Dateien liegen auf D:?").particulars == ["D:"]


def test_a_request_with_no_particulars_is_left_exactly_as_it_is() -> None:
    from capabilities.generalize import generalize

    shape = generalize("Berechne die SHA-256-Pruefsumme einer Datei")

    assert not shape.changed
    assert shape.goal == "Berechne die SHA-256-Pruefsumme einer Datei"


def test_two_requests_about_different_files_generalize_to_the_same_capability() -> None:
    """The property that stops the registry filling with near-duplicates."""

    from capabilities.generalize import generalize

    first = generalize("Berechne die SHA-256-Pruefsumme der Datei bericht.txt")
    second = generalize("Berechne die SHA-256-Pruefsumme der Datei urlaub.txt")

    assert first.goal == second.goal


# --------------------------------------------------------------------------
# The registry decides before the model does — and must not overreach
# --------------------------------------------------------------------------


def test_a_known_capability_is_dispatched_before_the_planner_is_asked(tmp_path: Path) -> None:
    """Measured: FAST_LOCAL read "welchen sha256 Fingerprint hat die Datei X?"
    as file.open with confidence 0.95. The model has never seen the registry.
    """

    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True, 'detail': 'digest computed'}\n",
             description="Compute the SHA-256 digest of a file.",
             aliases=["berechne die sha-256 pruefsumme einer datei"])
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    delivered: list[str] = []
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]

    handled = core._dispatch_known_capability("Zeus, welchen sha256 Fingerprint hat die Datei?", "test")

    assert handled is True
    assert delivered and "digest computed" in delivered[0]


def test_the_pre_planner_gate_declines_everything_it_is_not_sure_about(tmp_path: Path) -> None:
    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n",
             description="Compute the SHA-256 digest of a file.",
             aliases=["berechne die sha-256 pruefsumme einer datei"])
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    core._deliver = lambda *a, **k: None  # type: ignore[assignment]

    for unrelated in ("Zeus, wie spaet ist es?", "Zeus, spiel Rammstein", "Zeus, oeffne Wikipedia"):
        assert core._dispatch_known_capability(unrelated, "test") is False, unrelated


def test_an_unhealthy_capability_is_not_dispatched_by_the_pre_planner_gate(tmp_path: Path) -> None:
    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n",
             description="Compute the SHA-256 digest of a file.",
             aliases=["berechne die sha-256 pruefsumme einer datei"],
             health={"state": "degraded", "health": CapabilityHealth.AT_RISK.value})
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    core._deliver = lambda *a, **k: None  # type: ignore[assignment]

    assert core._dispatch_known_capability("Zeus, berechne die sha256 pruefsumme einer datei", "test") is False


def test_one_broken_capability_does_not_answer_broken_to_everything(tmp_path: Path) -> None:
    """The boost that makes BROKEN visible must not make it universal.

    Lifting a broken candidate's score unconditionally promoted every broken
    capability that shared one word with the request to the top of the
    ranking, so a single defective registry entry would send unrelated
    requests into a repair of the wrong thing.
    """

    from capabilities.resolver import CapabilityResolver

    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_manifest(
        "media.spotify.play", "Start playback of a requested artist in Spotify.",
        aliases=["spiel musik"], health={"state": "failing", "health": CapabilityHealth.BROKEN.value},
    ))
    resolver = CapabilityResolver(registry)

    assert resolver.resolve("spiel musik").outcome is CapabilityResolutionStatus.BROKEN
    for unrelated in ("Zeus, berechne die sha256 pruefsumme einer datei",
                      "Zeus, wie spaet ist es?",
                      "Zeus, oeffne Wikipedia"):
        assert resolver.resolve(unrelated).outcome is CapabilityResolutionStatus.MISSING, unrelated


# --------------------------------------------------------------------------
# The answer is what the capability produced
# --------------------------------------------------------------------------


def test_a_value_returning_capability_says_the_value(tmp_path: Path) -> None:
    """"ran" is not an answer to "what is the checksum of this file"."""

    core = _core(tmp_path)
    _install(
        core, "files.sha256",
        "def run(payload):\n    return {'ok': True, 'sha256': 'deadbeef', 'algorithm': 'sha256'}\n",
        description="Compute the SHA-256 digest of a file.",
        aliases=["berechne die sha-256 pruefsumme einer datei"],
    )
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    delivered: list[str] = []
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]

    core._route_capability_goal("berechne die sha256 pruefsumme einer datei",
                                "Zeus, berechne die sha256 pruefsumme einer datei", "test")

    assert delivered and "deadbeef" in delivered[0], delivered
    assert "algorithm: sha256" in delivered[0]


def test_bookkeeping_keys_are_not_read_back_to_the_owner() -> None:
    from service.core import _capability_result_lines

    assert _capability_result_lines({"ok": True, "detail": "done", "dry_run": False}, "done") == []
    assert _capability_result_lines({"ok": True, "sha256": "abc"}, "SHA-256 ist abc") == [], "already said"
    assert _capability_result_lines({"ok": True, "count": 12}, "ran") == ["", "count: 12"]


def test_a_capability_whose_input_key_is_unrecognised_still_gets_the_named_file(tmp_path: Path) -> None:
    """The mapping cannot know every name a capability might choose."""

    core = _core(tmp_path)
    workspace = Path(core.kernel.state_root) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "bericht.txt").write_bytes(b"hello")
    manifest = _install(
        core, "files.sha256", "def run(payload):\n    return {'ok': True}\n",
        description="Compute the SHA-256 digest of a file.",
    )
    manifest.input_schema = {"type": "object", "properties": {"source_document": {"type": "string"}},
                             "required": ["source_document"]}

    payload, unmet = core._capability_payload(manifest, "digest of bericht.txt", "Zeus, digest of bericht.txt")

    assert unmet == []
    assert payload["source_document"].endswith("bericht.txt")


def test_two_unrecognised_required_slots_are_still_reported_rather_than_guessed(tmp_path: Path) -> None:
    core = _core(tmp_path)
    workspace = Path(core.kernel.state_root) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "bericht.txt").write_bytes(b"hello")
    manifest = _install(core, "files.diff", "def run(payload):\n    return {'ok': True}\n",
                        description="Compare two documents.")
    manifest.input_schema = {"type": "object",
                             "properties": {"left_document": {"type": "string"}, "right_document": {"type": "string"}},
                             "required": ["left_document", "right_document"]}

    payload, unmet = core._capability_payload(manifest, "compare bericht.txt", "Zeus, compare bericht.txt")

    assert sorted(unmet) == ["left_document", "right_document"], "which file goes where is a guess"


def test_a_question_with_no_action_verb_still_reaches_the_capability(tmp_path: Path) -> None:
    """The last fork before prose.

    "Welchen sha256 Fingerprint hat die Datei X?" carries nothing for a routing
    table to recognise as an action, so it fell through to a conversational
    answer -- a model answering from memory something it could have computed.
    """

    core = _core(tmp_path)
    workspace = Path(core.kernel.state_root) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "bericht.txt").write_bytes(b"hello")
    _install(
        core, "files.sha256",
        "def run(payload):\n    return {'ok': True, 'sha256': 'deadbeef'}\n",
        description="Compute the SHA-256 digest of a file.",
        aliases=["berechne die sha-256 pruefsumme einer datei"],
        input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
    )
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    delivered: list[str] = []
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]

    handled = core._dispatch_known_capability(
        "Welchen sha256 Fingerprint hat die Datei bericht.txt?", "test")

    assert handled is True
    assert "deadbeef" in delivered[0]


def test_a_full_alias_list_keeps_learning_by_forgetting_the_oldest(tmp_path: Path) -> None:
    """Appending at a cap stops learning silently, which is the worst way."""

    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_manifest("files.sha256", "Compute the SHA-256 digest of a file."))
    for index in range(60):
        registry.learn_alias("files.sha256", f"pruefsumme variante {index} einer datei")

    aliases = registry.get("files.sha256").aliases

    assert len(aliases) == 48
    assert aliases[0] == "pruefsumme variante 59 einer datei", "the newest wording is kept"
    assert not any("variante 0 " in alias for alias in aliases), "the oldest is what gives way"


# --------------------------------------------------------------------------
# ZEUS's own working notes are not the owner's projects
# --------------------------------------------------------------------------


def test_an_acquisition_attempt_is_never_offered_as_an_owner_project(tmp_path: Path) -> None:
    """Found live, 2026-09-08, and it produced a wrong answer, not just clutter.

    A capability acquisition creates a project titled after the capability it
    is building. Asked to compute a checksum, FAST_LOCAL was handed the failed
    attempt ``local.berechne.sha.256_fsumme`` in its project hints, chose
    ``project.open`` at confidence 0.98, and ZEUS answered "Projekt
    local.berechne.sha.256_fsumme ist offen". The grounding check waved it
    through because the target really did exist -- as ZEUS's own record of
    failing to build the thing being asked for.
    """

    from projects.models import Project

    core = _core(tmp_path)
    owner = Project(id="proj_owner", goal="Schach", kind="software", title="Schach")
    attempt = Project(id="proj_attempt",
                      goal="Build a reusable capability that can: berechne eine sha256 pruefsumme",
                      kind="capability", title="local.berechne.sha.256_pruefsumme")
    attempt.metadata["capability_id"] = "local.berechne.sha.256_pruefsumme"
    core.kernel.projects.save(owner)
    core.kernel.projects.save(attempt)

    titles = {row["title"] for row in core.owner_projects()}

    assert titles == {"Schach"}
    assert "local.berechne.sha.256_pruefsumme" in {row["title"] for row in core.list_projects()}, \
        "nothing is hidden from the full listing; it is only not an owner project"
    assert core._target_grounded("project.open", "local.berechne.sha.256_pruefsumme", "berechne eine pruefsumme") is False
    assert core._target_grounded("project.open", "Schach", "mach das Schach-Projekt auf") is True


def test_a_capability_that_only_reads_is_not_failed_for_the_age_of_its_input(tmp_path: Path) -> None:
    """Found live, 2026-09-08, on the first capability Codex ever taught ZEUS.

    It returned the correct SHA-256 digest of a file, echoed that file's path,
    and the receipt came back FAILED — because the file the owner had asked
    about was older than five minutes. The freshness check exists to catch a
    capability reporting a path it did not write; a path handed IN is not such
    a claim.
    """

    import time

    core = _core(tmp_path)
    old_file = tmp_path / "bericht.txt"
    old_file.write_bytes(b"hello")
    ancient = time.time() - 86_400
    import os

    os.utime(old_file, (ancient, ancient))

    read_only = core._verify_capability_output(
        {"ok": True, "sha256": "abc", "path": str(old_file)}, {"file_path": str(old_file)})
    assert [c.check for c in read_only] == ["the reported file exists"]
    assert all(c.passed for c in read_only)

    # A path the capability chose for itself is still held to the old bar.
    produced = core._verify_capability_output({"ok": True, "path": str(old_file)}, {"dry_run": False})
    assert [c.check for c in produced] == ["the reported file exists", "it was produced just now, not found"]
    assert produced[1].passed is False

    fresh = tmp_path / "screenshot.png"
    fresh.write_bytes(b"\x89PNG")
    assert all(c.passed for c in core._verify_capability_output({"ok": True, "path": str(fresh)}, {}))


def test_the_owner_s_name_for_zeus_is_not_a_search_term(tmp_path: Path) -> None:
    """Found live: ``_singular`` turned "zeus" into "zeu".

    "Zeus" is filtered as an address term at the end of tokenisation, but the
    singulariser had already produced "zeu" from it -- a token in no filter
    list, present in every request the owner speaks, matching nothing. Every
    score was divided by one more term than the request contained, and the
    paraphrase that should have resolved came in at 0.479 against a 0.48 bar.
    """

    from capabilities.resolver import _terms

    assert "zeu" not in _terms("Zeus, berechne die pruefsumme")
    assert "zeus" not in _terms("Zeus, berechne die pruefsumme")
    assert "pruefsumme" in _terms("Zeus, berechne die pruefsumme")


def test_a_longer_sentence_about_the_same_thing_is_not_punished_for_being_longer(tmp_path: Path) -> None:
    """Coverage of the capability's subject, not only of the request's words."""

    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_manifest(
        "local.sha256", "berechne SHA-256-Pruefsumme einer Datei",
        creation_metadata={"keywords": ["berechne", "pruefsumme", "datei", "sha"]},
    ))
    resolver_for = CapabilityResolver(registry)

    terse = resolver_for.resolve("berechne die sha256 pruefsumme einer datei")
    wordy = resolver_for.resolve("Zeus, wie lautet die sha256 Pruefsumme von zeus_acceptance.txt?")

    assert terse.outcome is CapabilityResolutionStatus.FOUND
    assert wordy.outcome is CapabilityResolutionStatus.FOUND, "same subject, more words, same answer"


def test_sharing_only_the_word_for_a_file_is_not_a_match(tmp_path: Path) -> None:
    """Found live, 2026-09-08, and it produced a confidently wrong answer.

    Asked "bestimme mir die Entropie in Bits pro Byte der Datei X", ZEUS
    answered with the SHA-256 checksum of that file at confidence 0.50 —
    because *datei* was the only word the request and the capability shared.
    Every capability that works on a file says "Datei"; a word every capability
    says cannot tell two of them apart, which is what BOILERPLATE is for, and
    the English *file* and *path* were already in it.

    It compounds, too: the wrong answer was then learned as one of the checksum
    capability's phrasings, so the next wrong match would have been easier.
    """

    registry = CapabilityRegistry(tmp_path / "registry.json")
    registry.register(_manifest(
        "local.sha256", "berechne SHA-256-Pruefsumme einer Datei",
        aliases=["Zeus, berechne mir die SHA-256-Pruefsumme einer Datei"],
        creation_metadata={"keywords": ["berechne", "pruefsumme", "datei", "sha"]},
    ))
    resolver = CapabilityResolver(registry)

    entropy = resolver.resolve("Zeus, bestimme mir die Entropie in Bits pro Byte der Datei bericht.txt")
    assert entropy.outcome is CapabilityResolutionStatus.MISSING, \
        "a different question about the same file is a different capability"

    lines = resolver.resolve("Zeus, zaehle die Zeilen in der Datei bericht.txt")
    assert lines.outcome is CapabilityResolutionStatus.MISSING

    # What must keep working: a real paraphrase of the same subject.
    same = resolver.resolve("Zeus, wie lautet die sha256 Pruefsumme von bericht.txt?")
    assert same.outcome is CapabilityResolutionStatus.FOUND
    assert same.capability_id == "local.sha256"


def test_an_unavailable_engineer_is_explained_in_a_sentence_not_a_json_dump() -> None:
    """What the owner read, live: `Nicht gelernt: Codex is ERROR: {"checked_at":
    1788845136.37, "expert_available": false, "policy": {"allow_browser…`.

    The gateway had already asked the provider why, and that answer is a
    sentence. Serialising the whole status object put policy flags and
    timestamps into a chat message.
    """

    from capabilities.codex import _from_gateway_status

    status = {
        "state": "UNKNOWN",
        "expert_available": False,
        "quota_exhausted": False,
        "checked_at": 1788845136.37,
        "policy": {"allow_local_models": True, "allow_paid_api": False},
        "providers": [{"name": "codex", "permitted": True, "available": False,
                       "detail": "codex is not signed in"}],
    }

    availability = _from_gateway_status(status)

    assert availability.detail == "codex is not signed in"
    assert "policy" not in availability.detail and "checked_at" not in availability.detail


def test_a_sentence_that_names_nothing_is_not_a_capability_to_build() -> None:
    """Observed live: "Ja, bitte lerne das." became a capability goal.

    The owner answered an offer to learn something; the confirmation did not
    replay what was being confirmed, and that sentence went to the engineer as
    the thing to build. A goal that names nothing can only produce a capability
    that does nothing.
    """

    from service.core import JarvisCore

    assert JarvisCore._names_a_subject("Ja, bitte lerne das.") is False
    assert JarvisCore._names_a_subject("ja") is False
    assert JarvisCore._names_a_subject("Lerne das bitte") is False
    assert JarvisCore._names_a_subject("Lerne, wie man die SHA-256-Pruefsumme einer Datei berechnet") is True
    assert JarvisCore._names_a_subject("Entropie berechnen") is True


def test_a_subjectless_learn_request_asks_instead_of_building(tmp_path: Path) -> None:
    core = _core(tmp_path)
    created: list[str] = []
    core.missions.create = lambda *a, **k: created.append("mission")  # type: ignore[assignment]
    core.emit = lambda *a, **k: None  # type: ignore[assignment]
    delivered: list[str] = []
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]

    core._answer_by_acquisition("Ja, bitte lerne das.", "test")

    assert created == []
    assert delivered and ("Was genau" in delivered[0] or "What exactly" in delivered[0])
