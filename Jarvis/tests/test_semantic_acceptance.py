"""Semantic acceptance: OWNER GOALS, not function returns.

Each test states an owner goal and checks what the *product* did against it --
what exists afterwards, what was refused, what the verdict says -- through the
same core the interface uses.  The planner is a fake FAST_LOCAL provider that
answers with a fixed plan, so the tests pin the semantics around the model
(constraints, roles, replanning, goal evaluation, health, gating) and never
the model's own judgement.
"""

from __future__ import annotations

from _audio import speech_wav

import json
import time
from pathlib import Path

import pytest

from service.core import JarvisCore


class PlanProvider:
    """FAST_LOCAL stand-in: returns the plan JSON for the first call, a replan for the next."""

    def __init__(self, *plans: dict):
        self.plans = list(plans)
        self.prompts: list[str] = []

    def generate(self, prompt, **_):
        self.prompts.append(prompt)
        plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        return json.dumps(plan)

    def generate_stream(self, prompt, **_):
        yield "ok"


class Kernel:
    def __init__(self, root: Path, provider) -> None:
        self.state_root = root
        self._provider = provider
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        return self._provider


def make_core(tmp_path, *plans):
    provider = PlanProvider(*plans)
    core = JarvisCore(kernel=Kernel(tmp_path, provider))
    core.language = "de"
    return core, provider


def compose(core, text):
    """Run the composition path synchronously and return the delivered text."""

    delivered = []
    core._deliver = lambda text, **kw: delivered.append((text, kw))  # type: ignore[assignment]
    handled = core._answer_by_composition(text, "", allow_single=True)
    return handled, delivered


KNOWLEDGE_REQUEST = ("Zeus, speichere diesen technischen Befund im Knowledge und verknüpfe ihn mit ZEUS / Voice / Wakeword: "
                     "Der Voice-Studio-Test teilte PCM durch 32768. Erstelle dabei ausdrücklich kein neues Projekt und keine "
                     "Ersatz-Notiz, sondern melde ehrlich, falls dein Knowledge-System das nicht kann.")


# --------------------------------------------------------------------------
# TEST A — KNOWLEDGE: node + relations, persisted, searchable, GOAL_SATISFIED
# --------------------------------------------------------------------------

def test_a_knowledge_goal_is_met_by_a_graph_node_with_relations(tmp_path):
    core, provider = make_core(tmp_path, {"mode": "doing", "steps": [
        {"step": "knowledge.search", "query": "Wakeword", "role": "optional"},
        {"step": "knowledge.create", "title": "Voice-Studio PCM bug", "text": "PCM was divided by 32768", "type": "technical_finding", "links": "ZEUS, Voice, Wakeword"},
    ]})
    handled, delivered = compose(core, KNOWLEDGE_REQUEST)

    assert handled and delivered
    kinds = [r.kind for r in core.receipts.all()] if hasattr(core.receipts, "all") else [r.kind for r in core._session_receipts]
    assert "project.create" not in kinds and "note.create" not in kinds and "file.write" not in kinds
    assert "knowledge.create" in kinds
    node = core.knowledge_read("Voice-Studio PCM bug")
    assert node["ok"]
    outgoing = {(it["edge"]["type"], it["node"]["title"]) for it in node.get("outgoing", [])}
    assert {("relates_to", "ZEUS"), ("relates_to", "Voice"), ("relates_to", "Wakeword")} <= outgoing
    assert core.knowledge_backlinks("Wakeword")["backlinks"][0]["title"] == "Voice-Studio PCM bug"
    # persisted: a fresh core on the same state root finds it by search
    again = JarvisCore(kernel=Kernel(tmp_path, provider))
    hits = again.knowledge_graph(query="PCM bug")["nodes"]
    assert any(n["title"] == "Voice-Studio PCM bug" for n in hits)
    assert "Ziel erreicht" in delivered[-1][0]
    mission = core.list_missions()["missions"][0]
    assert mission["state"] == "completed"


# --------------------------------------------------------------------------
# TEST B — NEGATIVE CONSTRAINT: the planner cannot choose file.write / note.create
# --------------------------------------------------------------------------

def test_b_a_forbidden_fallback_is_refused_before_it_runs(tmp_path):
    core, _ = make_core(tmp_path, {"mode": "doing", "steps": [
        {"step": "note.create", "title": "Ersatz", "text": "x"},
        {"step": "file.write", "path": "notizen/x.md", "content": "x"},
        {"step": "knowledge.create", "title": "Befund", "text": "x", "type": "technical_finding", "links": "ZEUS"},
    ]})
    handled, delivered = compose(core, KNOWLEDGE_REQUEST)

    assert handled
    kinds = [r.kind for r in core._session_receipts]
    assert "note.create" not in kinds and "file.write" not in kinds
    assert not list((tmp_path / "workspace").rglob("*.md")) if (tmp_path / "workspace").exists() else True
    assert core.knowledge_read("Befund")["ok"]
    assert "⛔ note.create" in delivered[-1][0]


def test_b_when_every_step_is_forbidden_nothing_is_created_and_the_owner_is_told(tmp_path):
    core, _ = make_core(tmp_path, {"mode": "doing", "steps": [{"step": "note.create", "title": "Ersatz", "text": "x"}]})
    handled, delivered = compose(core, "Store this in Knowledge; do not create a file or a note as fallback.")

    assert handled
    assert core._session_receipts == []
    assert "ruled out" in delivered[-1][0] or "ausgeschlossen" in delivered[-1][0]


# --------------------------------------------------------------------------
# TEST C — IRRELEVANT CAPABILITY: the vocative "Zeus" is not a subject
# --------------------------------------------------------------------------

def test_c_a_word_counter_is_not_offered_for_a_knowledge_request(tmp_path):
    from capabilities.models import CapabilityManifest
    from capabilities.registry import CapabilityRegistry

    registry = CapabilityRegistry(tmp_path / "registry.json")
    src = tmp_path / "cap"
    src.mkdir()
    (src / "main.py").write_text("def run(p): return {'ok': True}\n", encoding="utf-8")
    registry.register(CapabilityManifest(
        capability_id="learned.ausgabe_dateipfad_zeilen", description="Count the words, lines and characters of a text file.",
        source_location=str(src), input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
        creation_metadata={"keywords": ["anzahl", "ausgabe", "dateipfad", "textdatei", "zeilen", "zeus", "wörter"]}))

    assert registry.find(KNOWLEDGE_REQUEST) == [], "one shared address word is not a subject match"
    assert registry.find("Zeus, zähle die Zeilen in der Textdatei plan.txt"), "the real subject still matches"


# --------------------------------------------------------------------------
# TEST D — CAPABILITY FAILURE: optional failure survives, required failure replans
# --------------------------------------------------------------------------

def test_d_an_optional_step_failing_does_not_abort_the_goal(tmp_path):
    core, _ = make_core(tmp_path, {"mode": "doing", "steps": [
        {"step": "knowledge.read", "title": "does-not-exist", "role": "optional"},
        {"step": "knowledge.create", "title": "Erkenntnis", "text": "x", "type": "note", "links": "ZEUS"},
    ]})
    handled, delivered = compose(core, "Speichere die Erkenntnis im Knowledge")

    assert handled
    assert core.knowledge_read("Erkenntnis")["ok"]
    assert "Ziel erreicht" in delivered[-1][0]


def test_d_a_required_step_failing_leads_to_one_replan_that_completes_the_goal(tmp_path):
    core, provider = make_core(tmp_path,
        {"mode": "doing", "steps": [{"step": "knowledge.link", "source": "nichts", "target": "niemand", "relation": "concerns"},
                                   {"step": "say", "text": "done"}]},
        {"mode": "doing", "steps": [{"step": "knowledge.create", "title": "Ersatzweg", "text": "x", "type": "note", "links": "ZEUS"}]})
    handled, delivered = compose(core, "Speichere das im Knowledge")

    assert handled
    assert len(provider.prompts) == 2 and "just failed: knowledge.link" in provider.prompts[1]
    assert core.knowledge_read("Ersatzweg")["ok"]
    assert "Ziel erreicht" in delivered[-1][0]


def test_d_without_a_usable_replan_the_verdict_is_blocked_not_success(tmp_path):
    core, _ = make_core(tmp_path,
        {"mode": "doing", "steps": [{"step": "knowledge.link", "source": "nichts", "target": "niemand", "relation": "concerns"},
                                   {"step": "knowledge.create", "title": "Nachher", "text": "x", "type": "note"}]},
        {"mode": "answering"})
    handled, delivered = compose(core, "Speichere das im Knowledge")

    assert handled
    assert not core.knowledge_read("Nachher")["ok"], "the remainder was not run blindly"
    assert "Nicht geschafft" in delivered[-1][0]


# --------------------------------------------------------------------------
# TEST E — HEALTH: a real failure degrades; repeated success restores
# --------------------------------------------------------------------------

def test_e_a_verified_runtime_failure_changes_health_at_once_and_success_restores_it(tmp_path):
    from capabilities.models import CapabilityManifest
    from capabilities.registry import CapabilityRegistry
    from service.doctor import Doctor

    registry = CapabilityRegistry(tmp_path / "registry.json")
    src = tmp_path / "cap"
    src.mkdir()
    (src / "main.py").write_text("def run(p): return {'ok': True}\n", encoding="utf-8")
    registry.register(CapabilityManifest(capability_id="x.y", description="does x", source_location=str(src)))

    registry.note_execution("x.y", False, "File not found")
    assert registry.get("x.y").health_view()["state"] == "degraded"
    registry.note_execution("x.y", False, "File not found")
    assert registry.get("x.y").health_view()["state"] == "failing"
    assert CapabilityRegistry(tmp_path / "registry.json").get("x.y").health_view()["state"] == "failing", "persisted"
    assert registry.get("x.y").status == "active", "installation state is a different fact"

    class Core:
        capabilities = type("S", (), {"registry": registry})()

    check = Doctor(Core(), repository=tmp_path)._capability_health()
    assert check.level == "error" and "x.y" in check.detail

    registry.note_execution("x.y", True, "ok")
    assert registry.get("x.y").health_view()["state"] == "degraded", "one good call is not a clean bill"
    registry.note_execution("x.y", True, "ok")
    assert registry.get("x.y").health_view()["state"] == "healthy"


# --------------------------------------------------------------------------
# TEST F — PROJECTS: one owner project, acquisition attempts grouped
# --------------------------------------------------------------------------

def test_f_the_projects_overview_shows_owner_projects_and_groups_internal_attempts(tmp_path):
    core, _ = make_core(tmp_path, {"mode": "answering"})
    core.kernel.projects = __import__("projects.store", fromlist=["ProjectStore"]).ProjectStore(tmp_path / "projects")
    from projects.models import Project

    def add(title, kind, capability_id=""):
        p = Project(goal=f"Build {title}", title=title, kind=kind, metadata={"capability_id": capability_id} if capability_id else {})
        core.kernel.projects.save(p)
        return p

    add("ZEUS Acceptance Lab", "software")
    add("music.provider.spotify", "capability", "music.provider.spotify")
    add("music.provider.spotify", "capability", "music.provider.spotify")
    add("archive.zip.create", "capability", "archive.zip.create")
    overview = core.projects_overview()

    assert [p["title"] for p in overview["projects"]] == ["ZEUS Acceptance Lab"]
    assert overview["counts"] == {"owner": 1, "internal_attempts": 3, "families": 2, "unclassified": 0}
    spotify = next(f for f in overview["internal"] if f["capability_id"] == "music.provider.spotify")
    assert spotify["count"] == 2


# --------------------------------------------------------------------------
# TEST G — MISSION ATTEMPTS: the same request groups; titles are concise; states are distinct
# --------------------------------------------------------------------------

def test_g_attempts_of_the_same_request_share_a_family_and_get_a_durable_title(tmp_path):
    core, _ = make_core(tmp_path, {"mode": "answering"})
    goal = "Zeus, repariere deinen Desktop- und Supervisor-Lifecycle. Wenn ich ZEUS.exe starte soll genau ein Fenster erscheinen und dann noch viel mehr Text der hier nicht in den Titel gehört."
    a = core.missions.create(goal, kind="complex")
    core.missions.transition(a, "FAILED", "x")
    b = core.missions.create(goal + " ", kind="complex")
    rows = core.list_missions()["missions"]
    mine = [r for r in rows if r["id"] in {a.mission_id, b.mission_id}]

    assert len({r["family"] for r in mine}) == 1 and all(r["attempts"] == 2 for r in mine)
    assert mine[0]["title"] == "Repariere deinen Desktop- und Supervisor-Lifecycle"
    states = {r["id"]: r["state"] for r in mine}
    assert states[a.mission_id] == "failed" and states[b.mission_id] == "active"
    assert core.list_missions(status="failed")["count"] >= 1 and all(r["state"] == "failed" for r in core.list_missions(status="failed")["missions"])


# --------------------------------------------------------------------------
# TEST H — EXPERT QUOTA: a deterministic adapter's exhaustion shows as a state, no paid fallback
# --------------------------------------------------------------------------

def test_h_quota_exhaustion_is_a_visible_state_and_never_reaches_a_paid_channel(tmp_path):
    from experts.contracts import ExpertJob, ExpertResult, ExpertStatus, QuotaState
    from experts.gateway import ExpertGateway, ProviderAvailability
    from runtime.cost_policy import SpendChannel
    from service.doctor import Doctor

    class Exhausted:
        name = "fake-subscription"
        channel = SpendChannel.SUBSCRIPTION_CLI

        def availability(self):
            return ProviderAvailability(True, "subscription CLI available", version="1.0")

        def execute(self, job):
            return ExpertResult(status=ExpertStatus.UNAVAILABLE, provider=self.name, blocker="usage limit reached",
                                quota=QuotaState(exhausted=True, detail="usage limit reached · limit will reset at 21:00"))

    gateway = ExpertGateway(providers=[Exhausted()])
    assert gateway.status()["state"] == "AVAILABLE"
    result = gateway.submit(ExpertJob(goal="x", workspace=tmp_path))
    assert result.status is ExpertStatus.UNAVAILABLE
    status = gateway.status()
    assert status["state"] == "QUOTA_EXHAUSTED" and status["quota_exhausted"] and not status["expert_available"]
    assert all(row["channel"] == SpendChannel.SUBSCRIPTION_CLI.value for row in status["providers"])

    class Core:
        experts = gateway

    check = Doctor(Core(), repository=tmp_path)._expert()
    assert check.level == "warn" and "QUOTA_EXHAUSTED" in check.detail and "never PAYG" in check.remedy


# --------------------------------------------------------------------------
# TEST I — AMBIENT SPEECH: fragments outside a session create nothing
# --------------------------------------------------------------------------

def test_i_ambient_fragments_create_no_message_action_receipt_or_mission(tmp_path):
    from service.voice import VoiceService
    from speech.contracts import Audio, Transcript

    core, _ = make_core(tmp_path, {"mode": "answering"})
    heard = iter(["Toys.", "So is...", "Jarvis, Toys.", "Toys, Toys, Toys.", "Mach das Licht an"])

    class Engine:
        def status(self): return {"available": True, "voices": []}
        def transcribe(self, audio, *, language=""): return Transcript(text=next(heard), confidence=0.5)

    core._voice = VoiceService(core.bus, engine_factory=Engine)
    wav = speech_wav()
    results = [core.hear(wav, wake=0.9) for _ in range(4)]          # fragments after a (false) wake
    results.append(core.hear(wav))                                    # a real sentence, but no session at all

    assert all(r["ok"] is False and r.get("ignored") for r in results), results
    assert core.history == []
    assert core._session_receipts == []
    assert core.list_missions()["count"] == 0
    assert len(core.voice.gate.rejected) == 5


# --------------------------------------------------------------------------
# TEST J — VOICE CONFIG: volume never touches the wake score; sensitivity is the threshold
# --------------------------------------------------------------------------

def test_j_volume_does_not_change_the_wake_score_and_sensitivity_only_moves_the_threshold(tmp_path):
    import numpy as np

    from speech.wake_eval import score_pcm
    from speech.wake_zeus import ZeusDetector

    class Features:  # the level of the last frame decides the score
        def __init__(self): self.levels = []
        def __call__(self, frame): self.levels.append(float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 1000.0)
        def get_features(self, n=16):
            w = np.zeros((n, 96), dtype=np.float32); w[:, 0] = ([0.0] * n + self.levels)[-n:]; return w[None]
        def reset(self): self.levels.clear()

    def detector(threshold=0.5):
        W = np.zeros(16 * 96, dtype=np.float32); W[15 * 96] = 8.0
        return ZeusDetector({"W": W, "b": np.float32(-4.0), "mean": np.zeros(16 * 96, np.float32), "std": np.ones(16 * 96, np.float32)}, threshold=threshold, features=Features())

    loud = lambda: np.full(1280, 1000, dtype=np.int16)
    quiet = lambda: np.zeros(1280, dtype=np.int16)
    core, _ = make_core(tmp_path, {"mode": "answering"})
    audio = np.concatenate([quiet(), loud(), loud(), quiet()]).astype(np.float32)
    core.voice_settings(volume=1.0)
    first = score_pcm(detector(threshold=0.5), audio)
    core.voice_settings(volume=0.05)
    second = score_pcm(detector(threshold=0.5), audio)
    assert first["frames"] == second["frames"], "the wake score is a function of audio and weights only"

    core.voice_settings(wake_sensitivity=0.55)
    assert core.wake_effective_threshold() == (0.55, "owner")
    core.voice_settings(wake_sensitivity=0.8)
    assert core.wake_effective_threshold() == (0.8, "owner")
    assert core.voice.settings.volume == 0.05, "sensitivity changes touched nothing else"


# --------------------------------------------------------------------------
# TEST K — GOAL VS ACTION: a verified file write does not satisfy a knowledge goal
# --------------------------------------------------------------------------

def test_k_execution_verified_is_not_goal_satisfied(tmp_path):
    from service.composer import Composer, evaluate_goal, extract_constraints

    plan = Composer().parse("Speichere den Befund im Knowledge",
                            json.dumps({"mode": "doing", "steps": [{"step": "file.write", "path": "befund.md", "content": "x"}]}))
    assert plan.constraints.required_outcome == ["knowledge.create", "knowledge.link"]

    class Receipt:
        id, kind, ok, verified = "r1", "file.write", True, True

    plan.steps[0].status, plan.steps[0].receipt_id = "done", "r1"
    verdict = evaluate_goal(plan, [Receipt()])

    assert verdict.executed and verdict.execution_verified
    assert verdict.goal_satisfied is False
    assert any("required outcome not produced" in r for r in verdict.reasons)
    assert verdict.to_dict()["EXECUTION_VERIFIED"] is True and verdict.to_dict()["GOAL_SATISFIED"] is False
