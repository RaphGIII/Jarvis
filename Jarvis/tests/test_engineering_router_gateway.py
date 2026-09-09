"""Engineer selection before execution, and an API engineer with hands.

The brief: never "cheap paid model → fail → frontier".  The engineering
router ranks Codex (subscription, zero marginal cost) and the metered API
engineers on learned reliability per engineering class and picks the cheapest
one that clears the bar.  A large new subsystem goes to the frontier engineer
directly -- when the owner is in BUILD mode, has enabled paid billing, and the
budget allows; otherwise Codex or the queue, said plainly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experts.api_engineer import ApiEngineerExpert
from experts.contracts import ExpertJob, ExpertStatus
from experts.gateway import ExpertGateway
from gateway.config import GatewayConfig
from gateway.modes import ChatMode
from gateway.secrets import CredentialStore
from gateway.task import TaskClass, TaskFacts, rule_based
from runtime.cost_policy import CostLedger, CostPolicy
from service.engineering import Engineer, EngineeringNeed, choose_engineer
from test_model_gateway import FakeNetwork, anthropic_reply, make_gateway


class Availability:
    def __init__(self, ready: bool) -> None:
        self._ready = ready

    def status(self):
        return SimpleNamespace(state=SimpleNamespace(value="READY" if self._ready else "NOT_INSTALLED"),
                               detail="stub", ready=self._ready)


@pytest.fixture
def cfg() -> GatewayConfig:
    config = GatewayConfig.defaults()
    for name in ("gemini", "openai", "anthropic"):
        config = config.with_provider_enabled(name, True)
    return config


@pytest.fixture
def creds(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "creds.json", use_dpapi=False)
    store.set("anthropic", "sk-ant-anthropictestkey00000000000000")
    store.set("openai", "sk-openaitestkey00000000000000000")
    store.set("gemini", "AIzaSyTESTKEYgemini0000000000000000000000")
    return store


def small_task():
    return rule_based(TaskFacts(text="Korrigiere den Tippfehler in der Fehlermeldung", is_engineering=True, estimated_files_changed=1))


def large_task():
    text = ("Beobachte meinen Bildschirm nur beim Schach, zeichne Partien auf, analysiere sie mit Stockfish, "
            "kategorisiere meine Fehler, führe ein Langzeitprofil, erstelle Training und ein Projekt Schach Training.")
    return rule_based(TaskFacts(text=text, is_engineering=True, new_subsystem=True, subsystems=5, needs_screen=True))


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_codex_takes_a_small_change_at_zero_cost_when_ready(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(True), task=small_task(),
                               gateway=gateway, mode=ChatMode.BUILD)
    assert decision.engineer is Engineer.CODEX and decision.provider_name == "codex"
    assert decision.task_class == TaskClass.ENGINEERING_SMALL.value and decision.estimated_eur == 0.0
    assert decision.q >= decision.tau


def test_a_large_new_subsystem_goes_directly_to_the_frontier_engineer_in_build_mode(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(True), task=large_task(),
                               gateway=gateway, mode=ChatMode.BUILD)
    assert decision.engineer is Engineer.API and decision.role == "engineer.frontier"
    assert decision.provider_name == "engineer.frontier"
    assert decision.estimated_eur > 0.0 and decision.estimate_range_eur[1] > decision.estimate_range_eur[0]
    codex = next(c for c in decision.candidates if c["role"] == "engineer.codex")
    standard = next(c for c in decision.candidates if c["role"] == "engineer.standard")
    assert codex["q"] < decision.tau and standard["q"] < decision.tau, "the cheaper engineers were skipped, not tried"
    assert "€" in decision.owner_sentence(german=True)


def test_outside_build_mode_a_large_task_stays_with_codex_and_says_why(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(True), task=large_task(),
                               gateway=gateway, mode=ChatMode.AUTO)
    assert decision.engineer is Engineer.CODEX
    assert "no engineer met the reliability bar" in decision.reason
    assert all(c["eligible"] is False for c in decision.candidates if c["role"] in {"engineer.standard", "engineer.frontier"})


def test_without_the_owner_enabling_paid_billing_no_metered_engineer_is_chosen(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork(), paid_api=False)
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(False), task=large_task(),
                               gateway=gateway, mode=ChatMode.BUILD)
    assert decision.engineer is Engineer.NONE and decision.queued
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(True), task=large_task(),
                               gateway=gateway, mode=ChatMode.BUILD)
    assert decision.engineer is Engineer.CODEX


def test_codex_unavailable_and_build_mode_picks_the_standard_engineer_for_a_small_change(tmp_path, cfg, creds):
    gateway = make_gateway(tmp_path, cfg, creds, FakeNetwork())
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(False), task=small_task(),
                               gateway=gateway, mode=ChatMode.BUILD)
    assert decision.engineer is Engineer.API and decision.role == "engineer.standard"


def test_the_classic_rule_still_applies_without_a_gateway():
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(True))
    assert decision.engineer is Engineer.CODEX and decision.provider_name == "codex"
    decision = choose_engineer(EngineeringNeed.CORE_ENGINEERING, availability=Availability(False))
    assert decision.engineer is Engineer.NONE


# --------------------------------------------------------------------------
# The API engineer edits a real worktree through the gateway
# --------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    jarvis = root / "Jarvis"
    (jarvis / "service").mkdir(parents=True)
    (jarvis / "service" / "greet.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "jarvis@example.invalid")
    _git(root, "config", "user.name", "Jarvis Test")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "baseline")
    return jarvis


GOOD_DIFF = (
    "diff --git a/service/greet.py b/service/greet.py\n"
    "--- a/service/greet.py\n"
    "+++ b/service/greet.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def greet():\n"
    "-    return 'hello'\n"
    "+    return 'hello world'\n"
)
BAD_DIFF = (
    "diff --git a/service/greet.py b/service/greet.py\n"
    "--- a/service/greet.py\n"
    "+++ b/service/greet.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def greet():\n"
    "-    return 'goodbye'\n"
    "+    return 'hello world'\n"
)


def _job(worktree: Path) -> ExpertJob:
    return ExpertJob(goal="Make greet() return 'hello world'", workspace=worktree,
                     acceptance=[("greet says hello world", [sys.executable, "-c",
                                  "import sys; sys.path.insert(0, '.'); from service.greet import greet; assert greet() == 'hello world'; print('OK')"])],
                     metadata={"files": ["service/greet.py"], "task_id": "job-1"})


def test_the_api_engineer_applies_the_models_diff_and_the_gateway_verifies_it(tmp_path, cfg, creds, worktree):
    net = FakeNetwork()
    net.responses["api.anthropic.com"] = anthropic_reply(json.dumps({"summary": "greet now says hello world", "diff": GOOD_DIFF}))
    gateway = make_gateway(tmp_path, cfg, creds, net)
    expert = ApiEngineerExpert(gateway, "engineer.frontier")
    assert expert.availability().available
    experts = ExpertGateway([expert], policy=CostPolicy(allow_paid_api=True), ledger=CostLedger(CostPolicy(allow_paid_api=True)))

    result = experts.submit(_job(worktree), provider_name="engineer.frontier")

    assert result.status is ExpertStatus.COMPLETED and result.verified, (result.blocker, result.test_evidence)
    assert result.files_changed == ["service/greet.py"]
    assert (worktree / "service" / "greet.py").read_text(encoding="utf-8").endswith("'hello world'\n")
    sent = net.requests[-1]["body"]
    assert "def greet" in json.dumps(sent), "the engineer saw the file it had to change"
    assert "sk-ant" not in json.dumps(sent["messages"])
    assert result.raw["cost_eur"] > 0.0 and gateway.governor.summary().month == pytest.approx(result.raw["cost_eur"])


def test_a_diff_that_does_not_apply_gets_one_repair_round_then_fails_honestly(tmp_path, cfg, creds, worktree):
    net = FakeNetwork()
    net.responses["api.anthropic.com"] = anthropic_reply(json.dumps({"summary": "x", "diff": BAD_DIFF}))
    gateway = make_gateway(tmp_path, cfg, creds, net)
    expert = ApiEngineerExpert(gateway, "engineer.standard")

    result = expert.execute(_job(worktree))

    assert result.status is ExpertStatus.FAILED and "did not apply after two attempts" in result.blocker
    assert len(net.requests) == 2
    assert "PREVIOUS DIFF DID NOT APPLY" in net.requests[-1]["body"]["messages"][0]["content"]
    assert (worktree / "service" / "greet.py").read_text(encoding="utf-8").endswith("'hello'\n"), "nothing half-applied"
    assert _git(worktree.parent, "status", "--porcelain").stdout.strip() == ""


def test_the_api_engineer_is_refused_in_free_mode_before_any_call(tmp_path, cfg, creds, worktree):
    net = FakeNetwork()
    gateway = make_gateway(tmp_path, cfg, creds, net)
    expert = ApiEngineerExpert(gateway, "engineer.frontier", mode=ChatMode.FREE)
    result = expert.execute(_job(worktree))
    assert result.status is ExpertStatus.REFUSED and net.requests == []


def test_the_api_engineer_reports_missing_key_and_owner_policy_as_unavailability(tmp_path, cfg, worktree):
    empty = CredentialStore(tmp_path / "none.json", use_dpapi=False)
    gateway = make_gateway(tmp_path, cfg, empty, FakeNetwork())
    assert ApiEngineerExpert(gateway, "engineer.frontier").availability().state == "NOT_AUTHENTICATED"
    store = CredentialStore(tmp_path / "some.json", use_dpapi=False)
    store.set("anthropic", "sk-ant-anthropictestkey00000000000000")
    gateway = make_gateway(tmp_path, cfg, store, FakeNetwork(), paid_api=False)
    availability = ApiEngineerExpert(gateway, "engineer.frontier").availability()
    assert not availability.available and "owner spending policy" in availability.detail


def test_codex_unavailability_never_falls_through_to_a_metered_engineer(tmp_path, cfg, creds, worktree):
    """The expert gateway iterates providers; the API engineers must not be on that path."""

    from experts.gateway import ProviderAvailability

    class DownCodex:
        name, channel, explicit_only = "codex", __import__("runtime.cost_policy", fromlist=["SpendChannel"]).SpendChannel.SUBSCRIPTION_CLI, False

        def availability(self):
            return ProviderAvailability(available=False, detail="not installed", state="NOT_INSTALLED")

        def execute(self, job):  # pragma: no cover - never reached
            raise AssertionError("codex is down")

    net = FakeNetwork()
    net.responses["api.anthropic.com"] = anthropic_reply(json.dumps({"summary": "x", "diff": GOOD_DIFF}))
    gateway = make_gateway(tmp_path, cfg, creds, net)
    experts = ExpertGateway([DownCodex(), ApiEngineerExpert(gateway, "engineer.standard"), ApiEngineerExpert(gateway, "engineer.frontier")],
                            policy=CostPolicy(allow_paid_api=True), ledger=CostLedger(CostPolicy(allow_paid_api=True)))
    result = experts.submit(_job(worktree))  # no provider named: the classic path
    assert result.status is not ExpertStatus.COMPLETED and net.requests == []
    assert (worktree / "service" / "greet.py").read_text(encoding="utf-8").endswith("'hello'\n")


# --------------------------------------------------------------------------
# The whole flow: estimate -> owner starts the build -> engineer -> verify -> promotion gate
# --------------------------------------------------------------------------

def test_a_metered_build_waits_for_the_owner_then_engineers_and_parks_for_promotion(tmp_path, cfg, creds, worktree, monkeypatch):
    from owner.security_gate import SecurityGate
    from service.selfdev import SelfDevMission, SelfDevRunner, SelfDevStore

    monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    net = FakeNetwork()
    net.responses["api.anthropic.com"] = anthropic_reply(json.dumps({"summary": "greet now says hello world", "diff": GOOD_DIFF}))
    gateway = make_gateway(tmp_path, cfg, creds, net)
    experts = ExpertGateway([ApiEngineerExpert(gateway, "engineer.standard"), ApiEngineerExpert(gateway, "engineer.frontier")],
                            policy=CostPolicy(allow_paid_api=True), ledger=CostLedger(CostPolicy(allow_paid_api=True)))
    gate = SecurityGate(tmp_path / "state" / "owner" / "auth.json")
    gate.setup("the-owner-types-this-into-the-ui")
    runner = SelfDevRunner(
        repository=worktree, store=SelfDevStore(tmp_path / "state" / "selfdev"),
        kernel=SimpleNamespace(state_root=tmp_path / "state", gateway=gateway),
        owner=SimpleNamespace(read=lambda _name: {}), lifecycle=SimpleNamespace(supervised=False), gateway=experts,
        availability=Availability(False), security=gate, emit=lambda *a, **k: None, set_state=lambda *a, **k: None,
    )
    runner.health_command = [sys.executable, "-c", "print('JARVIS_HEALTH_OK')"]
    mission = SelfDevMission(request="Zeus, lass greet() 'hello world' zurückgeben", chat_mode="BUILD")
    mission.investigation = {"files": ["service/greet.py"], "tests": [], "terms": ["greet"]}
    monkeypatch.setattr(runner, "_investigate", lambda m: None)

    parked = runner.run(mission)

    assert parked.phase == "AWAITING_BUILD" and parked.outcome == "awaiting_build_authorization"
    assert parked.engineering["engineer"] == "API" and parked.engineering["role"] == "engineer.standard"
    assert parked.engineering["hard_max_eur"] > 0 and "EUR" in parked.reason
    assert net.requests == [] and gateway.governor.summary().month == 0.0, "nothing spent before the owner's go"
    assert parked.finished

    built = runner.start_build(parked)

    assert built.phase == "AWAITING_AUTHORIZATION", (built.reason, [e for e in built.events if e["phase"] in {"ESCALATE", "VERIFY", "FAILED"}])
    assert built.outcome == "verified_awaiting_authorization"
    assert built.changed_files == ["service/greet.py"]
    assert built.expert["provider"] == "engineer.standard" and built.expert["cost_eur"] > 0
    assert gateway.governor.summary().month == pytest.approx(built.expert["cost_eur"])
    assert Path(built.worktree, "service", "greet.py").read_text(encoding="utf-8").endswith("'hello world'\n")
    assert (worktree / "service" / "greet.py").read_text(encoding="utf-8").endswith("'hello'\n"), "the live tree waits for the password"
