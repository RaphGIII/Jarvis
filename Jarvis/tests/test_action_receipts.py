"""A model saying "done" must not be able to make anything be done.

These tests exist because of a specific observed failure in the running
product.  Asked to create a file with exact contents, verify it, and only then
confirm, ZEUS answered:

    Datei "zeus_test.txt" wurde erstellt ... Existenz geprueft: Datei existiert
    und enthaelt den erwarteten Inhalt. Bestaetigung erfolgt.

Nothing had been executed.  The path it named -- ``/home/user/Projekte/...`` --
does not exist and could not exist; it is a Linux path invented on a Windows
machine.  The system prompt already contained "Never claim an action was
performed unless it actually was", ranked last so no persona could crowd it
out.  It made no difference, and no wording would have: a model asked to
confirm will confirm.

So the tests below do not check that the model behaves.  They assume it does
not.  Everything here is real -- the tool registry, the edit engine, the project
store, the capability registry, the HTTP server, the event stream -- except the
model, which is replaced by one that lies as fluently as possible.  What is
asserted is that the lie cannot reach the user as a success.

The client talks HTTP to a real socket rather than calling ``JarvisCore``
directly, because the defect was reported against the running product and an
in-process test would not have covered the transport, the routing or the
serialisation the browser actually uses.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from runtime.receipts import Receipt, ReceiptLedger, Verification
from service.actions import ActionExecutor, ActionPlan, compose
from service.claims import find_claim
from service.core import JarvisCore
from service.http import JarvisHTTPServer
from service.intent import Intent, classify
from service.state import JarvisState
from tools.registry import ToolCall, ToolRegistry, ToolResult

TOKEN = "test-token"

#: Things a model says when it is pretending to have done something.  Every one
#: of these was either observed in the live failure or is the obvious next
#: phrasing of it.
FABRICATIONS = (
    'Datei "zeus_test.txt" wurde erstellt und gespeichert.',
    "Ich habe die Datei erstellt und den Inhalt geprueft.",
    "Ich habe die Datei erstellt und den Inhalt geprüft.",
    "Das Projekt wurde erfolgreich angelegt.",
    "Existenz geprueft: Datei existiert.",
    "I have created the file and verified its contents.",
    "The file has been created successfully.",
    "Successfully saved the document.",
    "The file now exists at that path.",
    "Done.",
)


# --------------------------------------------------------------------------
# Fixtures: everything real but the model
# --------------------------------------------------------------------------

class LyingProvider:
    """A model that always reports success, whatever it was asked."""

    def __init__(self, text: str = FABRICATIONS[0]) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt, **_):
        self.prompts.append(prompt)
        return self.text

    def generate_stream(self, prompt, **_):
        self.prompts.append(prompt)
        # Chunked the way a real stream arrives, so the guard is exercised
        # mid-stream rather than against one convenient blob.
        for word in self.text.split(" "):
            yield word + " "


class PlanningProvider:
    """Answers the planner with JSON and everything else with a fabrication."""

    def __init__(self, plan: dict, chatter: str = FABRICATIONS[0]) -> None:
        self.plan = plan
        self.chatter = chatter

    def generate(self, prompt, **_):
        if "machine-readable action" in prompt:
            return json.dumps(self.plan)
        return self.chatter

    def generate_stream(self, prompt, **_):
        yield self.chatter


def build_core(tmp_path: Path, provider) -> JarvisCore:
    """A real kernel on a temporary state root, with a controlled model."""

    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state"))
    kernel.provider = lambda tier: provider  # type: ignore[assignment]
    kernel.catalog.get = lambda tier: type("S", (), {"model": "stub"})()  # type: ignore[assignment]
    core = JarvisCore(kernel=kernel, identity=Identity())
    return core


class Client:
    """The browser, minus the rendering."""

    def __init__(self, server: JarvisHTTPServer) -> None:
        self.base = f"http://{server.host}:{server.port}"
        self.events: list[dict] = []
        self._thread: threading.Thread | None = None

    def call(self, path: str, **payload):
        request = urllib.request.Request(
            f"{self.base}{path}", data=json.dumps(payload).encode(), method="POST"
        )
        request.add_header("X-Jarvis-Token", TOKEN)
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())

    def watch(self) -> None:
        def run() -> None:
            request = urllib.request.Request(f"{self.base}/events?since=0")
            request.add_header("X-Jarvis-Token", TOKEN)
            try:
                with urllib.request.urlopen(request, timeout=60) as stream:
                    for raw in stream:
                        line = raw.decode("utf-8", "replace").strip()
                        if line.startswith("data:"):
                            try:
                                self.events.append(json.loads(line[5:].strip()))
                            except ValueError:
                                pass
            except Exception:
                pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(0.3)

    def say(self, text: str, timeout: float = 30.0) -> dict:
        seen = len(self.events)
        self.call("/api/message", text=text)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for event in self.events[seen:]:
                if event.get("type") == "message":
                    return event["payload"]
            time.sleep(0.05)
        raise AssertionError(f"no reply to {text!r} within {timeout}s")

    def of_type(self, type_: str) -> list[dict]:
        return [event["payload"] for event in self.events if event.get("type") == type_]

    def states(self) -> list[str]:
        return [payload.get("state", "") for payload in self.of_type("state")]

    def settled(self, expected: str, timeout: float = 10.0) -> str:
        """The last state, once it stops changing.

        `say` returns on the MESSAGE event, and the state that follows it
        arrives a moment later on the same stream. Reading immediately is a
        race that passes alone and fails under load -- which is exactly how it
        behaved: green in isolation, red in a full run.
        """

        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self.states()
            if current and current[-1] == expected:
                return current[-1]
            time.sleep(0.05)
        return (self.states() or [""])[-1]


@pytest.fixture
def live(tmp_path):
    """A running server, a watching client, and a factory for the model."""

    made: list[JarvisHTTPServer] = []

    def start(provider):
        core = build_core(tmp_path, provider)
        server = JarvisHTTPServer(core, port=0, token=TOKEN)
        server.start()
        made.append(server)
        client = Client(server)
        client.watch()
        return core, client

    yield start
    for server in made:
        server.stop()


# --------------------------------------------------------------------------
# The central guarantee: prose cannot manufacture a success
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fabrication", FABRICATIONS)
def test_no_wording_lets_the_model_report_a_success(live, fabrication):
    """The whole defect, one phrasing at a time.

    Parametrised over the phrasings rather than testing one, because the
    original fix for this class of bug is always "handle the sentence that was
    reported" and the next sentence then walks straight through.
    """

    _core, client = live(LyingProvider(fabrication))

    reply = client.say("Wie ist das Wetter heute?")

    assert fabrication.strip(" .") not in reply["text"], (
        "a success claim with no receipt behind it reached the user verbatim"
    )
    assert "receipt" in reply["text"].lower() or "beleg" in reply["text"].lower()


def test_a_receipts_evidence_does_not_leak_into_the_next_prompt(live, tmp_path):
    """An action turn shows the user every check and shows the model one line.

    Observed live: after three action turns, the transcript carried three
    blocks full of "erstellt / geschrieben / ok", and the next ordinary answer
    to "Wie geht es dir?" came back making completion claims -- which the guard
    then intercepted. The model was parroting the evidence back. Receipts are
    for the reader; the transcript gets the fact.
    """

    plan = {"action": "file.write", "path": "a.txt", "content": "x"}
    core, client = live(PlanningProvider(plan))
    client.say("Erstelle die Datei a.txt mit dem Inhalt x")

    shown = core.history[-1].text
    # `_compose_prompt` drops the final history entry because the live path
    # appends the current user turn before calling it; mimic that order rather
    # than testing the method out of sequence.
    from service.core import ConversationTurn

    core._history.append(ConversationTurn(role="user", text="Wie geht es dir?"))
    in_prompt = core._compose_prompt("Wie geht es dir?")

    assert "content matches exactly" in shown, "the user must still see every check"
    assert "content matches exactly" not in in_prompt, "the evidence leaked into the prompt"
    assert "[executed file.write" in in_prompt


def test_the_intercepted_claim_is_recorded_as_a_receipt(live):
    core, client = live(LyingProvider())

    client.say("Erzaehl mir etwas ueber Zeus")

    blocked = [item for item in core.receipts.all() if item.kind == "claim.blocked"]
    assert blocked, "the interception left no durable record"
    assert blocked[-1].ok is False
    assert blocked[-1].verified is False


def test_the_claim_is_cut_out_of_the_stream_not_only_the_final_message(live):
    """A lie that was printed and spoken has been believed.

    The final message replaces the streamed text in the UI, so correcting it
    afterwards fixes the transcript. It does not fix the several seconds during
    which the user read -- or heard -- the claim. So the chunk that completes
    the claim is never emitted at all.
    """

    _core, client = live(LyingProvider("Die Datei wurde erstellt und geprueft."))

    client.say("Hallo")

    streamed = "".join(payload.get("text", "") for payload in client.of_type("token"))
    assert "erstellt" not in streamed, f"the claim was streamed to the user: {streamed!r}"


# --------------------------------------------------------------------------
# ... but a claim that IS backed must get through
# --------------------------------------------------------------------------

def test_a_claim_about_something_really_done_is_allowed_and_cited(live):
    """Blocking a true statement would teach the user the guard is noise.

    Having genuinely written zeus_test.txt, the assistant must be able to say
    so on the next turn. The guard's question is not "did anything happen this
    turn" but "did *this* happen".
    """

    plan = {"action": "file.write", "path": "zeus_test.txt", "content": "ZEUS funktioniert"}
    provider = PlanningProvider(plan)
    core, client = live(provider)
    client.say("Erstelle die Datei zeus_test.txt mit dem Inhalt ZEUS funktioniert")

    provider.chatter = "Ja, die Datei zeus_test.txt wurde erstellt."
    reply = client.say("Hast du das wirklich gemacht?")

    assert "zeus_test.txt wurde erstellt" in reply["text"]
    assert "receipt rcpt_" in reply["text"], "a permitted claim must carry its evidence"


def test_a_real_receipt_does_not_licence_an_unrelated_claim(live):
    """The obvious way to break the refinement: one true thing, one invented one."""

    plan = {"action": "file.write", "path": "zeus_test.txt", "content": "x"}
    provider = PlanningProvider(plan)
    _core, client = live(provider)
    client.say("Erstelle die Datei zeus_test.txt mit dem Inhalt x")

    provider.chatter = "Ich habe auch deine E-Mails geloescht und die Datenbank gespeichert."
    reply = client.say("Und sonst so?")

    assert "E-Mails geloescht" not in reply["text"]
    assert "receipt" in reply["text"].lower() or "beleg" in reply["text"].lower()


def test_a_claim_naming_nothing_concrete_stays_blocked(live):
    """"Done." can match no receipt, so it is refused however much has happened."""

    plan = {"action": "file.write", "path": "zeus_test.txt", "content": "x"}
    provider = PlanningProvider(plan)
    _core, client = live(provider)
    client.say("Erstelle die Datei zeus_test.txt mit dem Inhalt x")

    provider.chatter = "Done."
    reply = client.say("Und?")

    assert reply["text"].strip() != "Done."


def test_only_a_verified_receipt_can_support_a_claim():
    from runtime.receipts import supporting

    unverified = Receipt(
        kind="file.write", executor="t", ok=True,
        evidence={"relative_path": "zeus_test.txt"}, verifications=(),
    )

    assert supporting([unverified], "zeus_test.txt wurde erstellt") is None


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------

def test_an_action_that_verified_nothing_is_not_verified():
    """The vacuous-truth failure: a guard with nothing to check reports green."""

    receipt = Receipt(kind="file.write", executor="test", ok=True, verifications=())

    assert receipt.ok is True
    assert receipt.verified is False


def test_one_failed_check_is_enough_to_withhold_the_verdict():
    receipt = Receipt(
        kind="file.write", executor="test", ok=True,
        verifications=(
            Verification("file exists on disk", True, observed="C:/x.txt"),
            Verification("content matches exactly", False, observed="''", expected="'hi'"),
        ),
    )

    assert receipt.verified is False


def test_compose_never_says_success_for_an_unverified_receipt():
    """The sentence the user reads is built here, so this is the last gate."""

    receipt = Receipt(kind="file.write", executor="test", ok=True, verifications=())

    text = compose(receipt).lower()

    assert "not treating that as success" in text
    assert "written, then independently read back" not in text


def test_the_ledger_survives_the_process(tmp_path):
    """A receipt that only exists in memory is not evidence of anything."""

    path = tmp_path / "receipts.jsonl"
    ReceiptLedger(path).record(Receipt(kind="file.write", executor="t", ok=True, detail="d"))

    reloaded = ReceiptLedger(path).all()

    assert len(reloaded) == 1
    assert reloaded[0].detail == "d"


def test_one_corrupt_line_does_not_hide_the_rest_of_the_history(tmp_path):
    path = tmp_path / "receipts.jsonl"
    ledger = ReceiptLedger(path)
    ledger.record(Receipt(kind="a", executor="t", ok=True))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
    ledger.record(Receipt(kind="b", executor="t", ok=True))

    assert [receipt.kind for receipt in ReceiptLedger(path).all()] == ["a", "b"]


# --------------------------------------------------------------------------
# File actions: real write, independent readback
# --------------------------------------------------------------------------

def test_a_file_action_writes_a_real_file_and_reads_it_back(live, tmp_path):
    plan = {"action": "file.write", "path": "zeus_test.txt", "content": "ZEUS funktioniert"}
    core, client = live(PlanningProvider(plan))

    reply = client.say("Erstelle die Datei zeus_test.txt mit dem Inhalt ZEUS funktioniert")

    receipt = core.receipts.all()[-1]
    assert receipt.verified, receipt.to_dict()
    written = Path(receipt.evidence["path"])
    # Read off the filesystem, not through anything that produced the receipt.
    assert written.read_text(encoding="utf-8") == "ZEUS funktioniert"
    assert str(written) in reply["text"]


def test_a_write_whose_readback_disagrees_is_not_a_success(live, tmp_path):
    """The writer says it worked; the file says otherwise. The file wins.

    This is why the readback is a separate step rather than the tool's return
    value: a writer that reports success and a reader that trusts the writer
    agree with each other about a file that is not there.
    """

    core = build_core(tmp_path, PlanningProvider({}))
    registry = ToolRegistry()

    def wrong_content(arguments, context):
        target = Path(context.workspace) / arguments["path"]
        target.write_text("something else entirely", encoding="utf-8")
        return {"applied": [arguments["path"]]}

    from tools.registry import ToolSpec

    registry.register(
        ToolSpec(name="write_file", purpose="w",
                 input_schema={"path": "string", "content": "string"}, adapter=wrong_content)
    )
    executor = ActionExecutor(core.kernel, tools=registry)

    receipt = executor.execute(
        ActionPlan("file.write", {"path": "a.txt", "content": "expected"}), request="x"
    )

    assert receipt.ok is False
    assert receipt.verified is False
    assert any("content matches" in check.check for check in receipt.failures)
    assert "not treating that as success" in compose(receipt).lower() or "failed" in compose(receipt).lower()


def test_a_failing_file_tool_produces_an_honest_failure(live, tmp_path):
    """Acceptance D, as a test: the tool genuinely fails, not a mocked verdict."""

    core = build_core(tmp_path, PlanningProvider({}))
    registry = ToolRegistry()

    from tools.registry import ToolError, ToolSpec

    def always_fails(arguments, context):
        raise ToolError("disk is on fire", kind="io_error", retryable=False)

    registry.register(
        ToolSpec(name="write_file", purpose="w",
                 input_schema={"path": "string", "content": "string"}, adapter=always_fails)
    )
    executor = ActionExecutor(core.kernel, tools=registry)

    receipt = executor.execute(
        ActionPlan("file.write", {"path": "a.txt", "content": "x"}), request="create a.txt"
    )
    text = compose(receipt)

    assert receipt.ok is False
    assert "disk is on fire" in receipt.detail
    assert text.lower().startswith("that failed")
    for word in ("created", "erstellt", "success", "verified"):
        assert word not in text.lower().replace("that failed.", "")


def test_a_directory_in_the_way_is_named_as_such(tmp_path):
    """The forced-failure path reported "existing edit target does not exist"
    about a path that plainly did exist -- as a directory. A failure message
    that misnames the problem is one nobody can act on.
    """

    core = build_core(tmp_path, PlanningProvider({}))
    executor = ActionExecutor(core.kernel)
    (executor.workspace / "blocked.txt").mkdir(parents=True, exist_ok=True)

    receipt = executor.execute(
        ActionPlan("file.write", {"path": "blocked.txt", "content": "x"}), request="x"
    )

    assert receipt.ok is False
    assert "is a directory" in receipt.detail
    assert "does not exist" not in receipt.detail


@pytest.mark.parametrize(
    "path",
    [
        "../../escaped.txt",
        "C:/Windows/Temp/evil.txt",   # absolute wins over `/` on Windows
        "D:/evil.txt",                # ...including a different drive
        "/etc/passwd",
        "sub/../../../escaped.txt",
    ],
)
def test_an_action_cannot_write_outside_its_workspace(tmp_path, path):
    """Containment is checked after resolution, not by inspecting the string.

    ``a/../../b``, a symlink and ``D:/x`` all look harmless as text, and on
    Windows an absolute right-hand side silently wins the join: ``Path("C:/a")
    / "D:/evil.txt"`` is ``D:/evil.txt``.
    """

    core = build_core(tmp_path, PlanningProvider({}))
    executor = ActionExecutor(core.kernel)

    receipt = executor.execute(ActionPlan("file.write", {"path": path, "content": "x"}), request="x")

    assert receipt.ok is False
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_a_subdirectory_inside_the_workspace_is_allowed(tmp_path):
    """Containment must not be so blunt that ordinary paths stop working."""

    core = build_core(tmp_path, PlanningProvider({}))
    executor = ActionExecutor(core.kernel)

    receipt = executor.execute(
        ActionPlan("file.write", {"path": "notes/today.txt", "content": "hi"}), request="x"
    )

    assert receipt.verified, receipt.to_dict()
    assert Path(receipt.evidence["path"]).read_text(encoding="utf-8") == "hi"


def test_an_executor_that_raises_becomes_a_failed_receipt_not_a_success(tmp_path):
    """An executor bug must not be able to look like a success."""

    core = build_core(tmp_path, PlanningProvider({}))
    executor = ActionExecutor(core.kernel)
    executor._write_file = lambda plan, request: 1 / 0  # type: ignore[assignment]

    receipt = executor.execute(ActionPlan("file.write", {"path": "a.txt", "content": "x"}))

    assert receipt.ok is False
    assert receipt.verified is False
    assert "ZeroDivisionError" in receipt.detail


def test_an_unsupported_action_is_declined_rather_than_narrated(tmp_path):
    core = build_core(tmp_path, PlanningProvider({}))
    executor = ActionExecutor(core.kernel)

    receipt = executor.execute(ActionPlan("none", reason="I cannot send email"), request="send mail")

    assert receipt.ok is False
    assert "cannot send email" in receipt.detail


# --------------------------------------------------------------------------
# Projects: real persistence, reload, and the API the UI reads
# --------------------------------------------------------------------------

def test_creating_a_project_persists_it_and_reloads_it_from_disk(live, tmp_path):
    plan = {"action": "project.create", "name": "Zeus Testprojekt"}
    core, client = live(PlanningProvider(plan))

    reply = client.say("Erstelle ein Projekt mit dem Namen Zeus Testprojekt")

    receipt = core.receipts.all()[-1]
    assert receipt.verified, receipt.to_dict()

    # Reloaded through a store this test builds itself, over the same directory.
    from projects.store import ProjectStore

    store = ProjectStore(Path(core.kernel.state_root) / "projects")
    reloaded = store.try_load(receipt.evidence["project_id"])
    assert reloaded is not None
    assert reloaded.title == "Zeus Testprojekt"

    # And through the same endpoint the Projects panel calls.
    listed = client.call("/api/projects")["projects"]
    assert any(item["title"] == "Zeus Testprojekt" for item in listed)
    # The owner reads one natural sentence; the id and the checks stay in
    # the receipt (Activity / Beleg), which the same turn published.
    assert "Zeus Testprojekt" in reply["text"] and reply["text"].startswith("Erledigt")
    assert any(e.get("receipt", {}).get("id") == receipt.id for e in client.of_type("tool"))


def test_a_project_the_model_merely_described_does_not_appear(live):
    """The reported failure, exactly: claimed created, Projects panel unchanged."""

    _core, client = live(LyingProvider("Projekt 'Zeus Testprojekt' wurde erstellt und gespeichert."))

    before = client.call("/api/projects")["projects"]
    reply = client.say("Erzaehl mir von deinen Faehigkeiten als Entwickler")
    after = client.call("/api/projects")["projects"]

    assert len(after) == len(before)
    assert "wurde erstellt" not in reply["text"]


# --------------------------------------------------------------------------
# Capability answers come from the registry
# --------------------------------------------------------------------------

def test_the_capability_registry_is_read_from_the_right_path(tmp_path):
    """It was given the directory, not the file, and failed silently forever.

    ``CapabilityRegistry`` raises when handed a directory; the caller's bare
    ``except`` turned that into ``[]``, which is indistinguishable from "there
    are no capabilities". A registry with entries in it reported none.
    """

    core = build_core(tmp_path, PlanningProvider({}))
    registry_path = Path(core.kernel.state_root) / "capabilities" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps({
            "schema_version": 1,
            "capabilities": {
                "play_music": {
                    "capability_id": "play_music", "description": "plays music",
                    "version": "1.0.0", "status": "active",
                }
            },
        }),
        encoding="utf-8",
    )

    listed = core.list_capabilities()
    report = core.capability_report()

    assert [item["capability_id"] for item in listed] == ["play_music"]
    assert report["error"] == ""
    assert [item["capability_id"] for item in report["active"]] == ["play_music"]


def test_a_capability_question_is_answered_from_the_registry_not_the_model(live, tmp_path):
    core, client = live(LyingProvider("Ich kann Musik abspielen, E-Mails senden und Auto fahren."))

    reply = client.say("Welche Faehigkeiten sind verifiziert?")

    assert "E-Mails" not in reply["text"], "the model's invented list reached the user"
    assert "registry" in reply["text"].lower() or "registry" in reply["text"]
    # And it agrees with the endpoint Diagnostics reads.
    assert core.capability_report()["active"] == []


def test_asking_it_to_learn_something_starts_a_mission_not_a_claim(live, monkeypatch):
    """A present-tense capability claim slips past a guard that watches for
    completed actions, so this one is never asked of the model at all: the
    request becomes a durable acquisition mission, and the only sentence the
    owner reads promises a report, never an ability.
    """

    from service import acquisition as acq

    ran = []
    monkeypatch.setattr(acq.AcquisitionMission, "run", lambda self, goal, **kw: ran.append(goal) or acq.AcquisitionResult(goal=goal, reason="stub"))
    core, client = live(LyingProvider("Ich kann jetzt Musik abspielen."))

    reply = client.say("Lerne wie man Musik abspielt.")

    assert "Ich kann jetzt Musik abspielen" not in reply["text"]
    assert "Mission m_" in reply["text"] and "lerne ich jetzt" in reply["text"]
    deadline = time.time() + 10
    while time.time() < deadline and not ran:
        time.sleep(0.05)
    assert ran == ["Lerne wie man Musik abspielt."]
    missions = core.missions.store.list()
    assert missions and missions[-1].kind == "capability"


def test_an_unreadable_registry_says_so_instead_of_reporting_none(tmp_path):
    core = build_core(tmp_path, PlanningProvider({}))
    path = Path(core.kernel.state_root) / "capabilities" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")

    report = core.capability_report()

    assert report["error"], "an unreadable registry silently became 'no capabilities'"


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Erstelle die Datei zeus_test.txt mit dem Inhalt ZEUS funktioniert", Intent.ACTION),
        ("create a file called notes.txt", Intent.ACTION),
        ("Speichere das bitte ab", Intent.ACTION),
        ("Loesche die Datei", Intent.ACTION),
        ("Erstelle ein Projekt namens Zeus Testprojekt", Intent.ACTION),
        ("Welche Faehigkeiten sind verifiziert?", Intent.READ),
        ("what can you do?", Intent.READ),
        ("Welche Projekte gibt es?", Intent.READ),
        ("Wer bist du?", Intent.CONVERSATION),
        ("Wie geht es dir?", Intent.CONVERSATION),
        ("Erklaer mir Rekursion", Intent.CONVERSATION),
        ("implementiere einen Sortieralgorithmus", Intent.PROJECT),
        ("lerne wie man Musik abspielt", Intent.CAPABILITY),
    ],
)
def test_requests_are_classified_before_anything_answers_them(text, expected):
    assert classify(text).intent is expected


def test_a_bare_filename_is_treated_as_an_action():
    """No verb, but "zeus_test.txt" is not something to have a conversation about."""

    assert classify("zeus_test.txt bitte mit dem Inhalt ZEUS funktioniert").intent is Intent.ACTION


def test_capability_questions_beat_capability_acquisition():
    """"was kannst du" is a question, not a request to go and learn something."""

    assert classify("Was kannst du alles?").intent is Intent.READ


# --------------------------------------------------------------------------
# The conversation path stays the fast path
# --------------------------------------------------------------------------

def test_ordinary_conversation_executes_nothing_and_stays_on_the_fast_path(live):
    core, client = live(LyingProvider("Mir geht es gut, danke der Nachfrage."))

    client.say("Wie geht es dir?")

    assert core.receipts.all() == []
    assert client.of_type("token"), "conversation must still stream"
    assert "thinking" in client.states()
    assert "working" not in client.states()


def test_an_action_turn_shows_working_then_verifying(live):
    plan = {"action": "file.write", "path": "a.txt", "content": "x"}
    _core, client = live(PlanningProvider(plan))

    client.say("Erstelle die Datei a.txt mit dem Inhalt x")
    states = client.states()

    assert "working" in states
    assert "verifying" in states
    assert states.index("working") < states.index("verifying")


def test_a_failed_action_leaves_the_interface_in_an_error_state(live, tmp_path):
    """A failure that returns straight to idle looks like a success from across
    the room -- the same defect as a reassuring sentence, in the interface."""

    plan = {"action": "file.write", "path": "blocked.txt", "content": "x"}
    core, client = live(PlanningProvider(plan))
    (Path(core.kernel.state_root) / "workspace").mkdir(parents=True, exist_ok=True)
    (Path(core.kernel.state_root) / "workspace" / "blocked.txt").mkdir()

    client.say("Erstelle die Datei blocked.txt mit dem Inhalt x")

    assert core.receipts.all()[-1].verified is False
    assert client.settled("error") == "error"


def test_a_new_conversation_clears_the_transcript_but_never_the_record(live, tmp_path):
    """Hiding the transcript is a convenience. Hiding the evidence would be the
    thing this system exists to prevent."""

    plan = {"action": "file.write", "path": "kept.txt", "content": "x"}
    core, client = live(PlanningProvider(plan))
    client.say("Erstelle die Datei kept.txt mit dem Inhalt x")
    receipts_before = len(core.receipts.all())
    activity_before = len(core.activity.recent())

    client.call("/api/new")

    assert core.history == []
    # Settled rather than read: /api/new sets IDLE, and a trailing STATE from
    # the turn before it arrives a moment later on the same stream. Reading
    # immediately is the same race `7e76681` fixed for the other state
    # assertions in this file, and it behaved the same way -- green alone, red
    # in a full run.
    assert client.settled("idle") == "idle"
    assert len(core.receipts.all()) == receipts_before
    assert len(core.activity.recent()) == activity_before


def test_a_verified_action_returns_to_idle(live):
    plan = {"action": "file.write", "path": "fine.txt", "content": "x"}
    _core, client = live(PlanningProvider(plan))

    client.say("Erstelle die Datei fine.txt mit dem Inhalt x")

    assert client.settled("idle") == "idle"


def test_an_action_turn_does_not_stream_model_prose(live):
    """There is no window in which the model's words are the answer."""

    plan = {"action": "file.write", "path": "a.txt", "content": "x"}
    _core, client = live(PlanningProvider(plan, chatter="I already created everything."))

    reply = client.say("Erstelle die Datei a.txt mit dem Inhalt x")

    assert "already created" not in reply["text"]
    assert not client.of_type("token")


def test_a_declined_plan_falls_back_to_conversation_rather_than_refusing(live):
    """The classifier is biased toward ACTION, so it over-triggers by design.

    "Schreibe mir ein Gedicht" trips the verb list. The planner is the second
    opinion, and when it says there is no action here the right answer is to
    have the conversation, not to refuse a legitimate request.
    """

    plan = {"action": "none", "reason": "this asks for a poem, not a file"}
    _core, client = live(PlanningProvider(plan, chatter="Es war einmal ein Blitz."))

    reply = client.say("Schreibe mir ein Gedicht ueber Zeus")

    assert "Blitz" in reply["text"]


# --------------------------------------------------------------------------
# The status badge must not be the most expensive thing running
# --------------------------------------------------------------------------

class RecordingProbe:
    """A probe that records which tiers were asked about and generates nothing."""

    def __init__(self, tiers=()):
        self.probed = []
        self.tiers = tuple(tiers)

    def probe(self, tier, *, force=False):
        self.probed.append(tier)
        return type("H", (), {
            "online": True, "summary": lambda self: "ok", "checked_at": 0.0,
            "to_dict": lambda self: {"state": "online", "online": True},
        })()

    def probe_all(self, *, force=False):
        return {tier: self.probe(tier, force=force) for tier in self.tiers}

    def cached_all(self):
        return {}


def test_the_health_badge_never_probes_the_build_model(tmp_path):
    """Measured on this machine, before the fix:

        after a FAST_LOCAL generation : qwen3:4b-instruct resident
        after the BUILD_LOCAL probe (47.1s): qwen2.5-coder:7b resident
        next FAST_LOCAL generation costs 28.3s

    The badge ran that probe on a 120-second timer, so conversation paid a
    28-second model reload every two minutes for the life of the process.
    """

    from brain.tiers import ModelTier

    core = build_core(tmp_path, LyingProvider())
    probe = RecordingProbe()
    core.kernel.probe = probe

    core._probe_health()

    assert probe.probed == [ModelTier.FAST_LOCAL]
    assert ModelTier.BUILD_LOCAL not in probe.probed


def test_reading_diagnostics_does_not_change_what_it_reports(tmp_path):
    """A diagnostic that degrades the thing it diagnoses is worse than none.

    ``kernel.status()`` probes every tier, and a probe is a real generation --
    so drawing the panel loaded the 7B coder and left the next sentence 28
    seconds slower. Measuring is now opt-in.
    """

    core = build_core(tmp_path, LyingProvider())
    probe = RecordingProbe()
    core.kernel.probe = probe

    payload = core.diagnostics()

    assert probe.probed == [], "reading diagnostics ran generations"
    assert payload["measured"] is False
    states = {tier["state"] for tier in payload["kernel"]["tiers"].values()}
    assert states == {"unmeasured"}, "an unmeasured tier must not be reported as offline"


def test_diagnostics_can_still_be_asked_to_measure(tmp_path):
    core = build_core(tmp_path, LyingProvider())
    probe = RecordingProbe(core.kernel.catalog.tiers())
    core.kernel.probe = probe

    payload = core.diagnostics(refresh=True)

    assert probe.probed, "an explicit refresh must actually probe"
    assert payload["measured"] is True


def test_no_probe_starts_while_a_turn_is_in_flight(tmp_path):
    """A status light must not queue ahead of the user's own generation."""

    from service.state import JarvisState

    core = build_core(tmp_path, LyingProvider())
    probe = RecordingProbe()
    core.kernel.probe = probe
    core.state.set(JarvisState.THINKING, detail="answering")

    core._cached_health()

    assert probe.probed == []


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def test_the_prompt_never_asks_the_model_to_answer_as_jarvis(tmp_path):
    """`--persona Jarvis` was the default, and the prompt ended "Jarvis:"."""

    core = build_core(tmp_path, LyingProvider())

    prompt = core._compose_prompt("Wer bist du?")

    assert "JARVIS" not in prompt
    assert prompt.rstrip().endswith("Zeus:")


def test_the_builtin_personas_carry_no_hard_coded_name():
    from persona.profiles import builtin_personas

    for persona in builtin_personas().values():
        assert "JARVIS" not in persona.character
        assert "Zeus" == Identity().assistant_name
        assert Identity().assistant_name in persona.system_prompt(assistant="Zeus")


def test_the_provider_system_prompt_follows_the_configured_identity():
    """This one is attached to the provider, so it reached every inference path."""

    from config import system_prompt

    assert "You are Zeus" in system_prompt(Identity())
    assert "You are Athena" in system_prompt(Identity(assistant_name="Athena"))
    assert "JARVIS" not in system_prompt(Identity())


def test_a_persona_selection_saved_before_the_rename_still_resolves(tmp_path):
    from persona.profiles import PersonaStore

    path = tmp_path / "personas.json"
    path.write_text(json.dumps({"active": "jarvis", "personas": []}), encoding="utf-8")

    store = PersonaStore(path)

    assert store.active().name == "default"


# --------------------------------------------------------------------------
# The receipt API
# --------------------------------------------------------------------------

def test_receipts_are_reachable_over_http_for_a_client_that_was_not_watching(live):
    plan = {"action": "file.write", "path": "a.txt", "content": "x"}
    core, client = live(PlanningProvider(plan))
    client.say("Erstelle die Datei a.txt mit dem Inhalt x")

    listed = client.call("/api/receipts")["receipts"]
    fetched = client.call("/api/receipt", id=listed[-1]["id"])

    assert listed[-1]["kind"] == "file.write"
    assert listed[-1]["verified"] is True
    assert fetched["id"] == listed[-1]["id"]


def test_the_receipt_reaches_the_ui_as_an_event_with_its_checks(live):
    plan = {"action": "file.write", "path": "a.txt", "content": "x"}
    _core, client = live(PlanningProvider(plan))

    client.say("Erstelle die Datei a.txt mit dem Inhalt x")

    receipts = [payload for payload in client.of_type("tool") if payload.get("receipt")]
    assert receipts
    receipt = receipts[-1]["receipt"]

    # The exact fields ui/app.js::addReceipt reads. Asserted by name because
    # this is a contract between two files that cannot import each other, and
    # the failure mode is a silently blank panel rather than an error.
    for field in ("kind", "verified", "ok", "detail", "id", "verifications"):
        assert field in receipt, f"the UI reads receipt.{field}"

    checks = receipt["verifications"]
    assert {"file exists on disk", "content matches exactly"} <= {item["check"] for item in checks}
    for item in checks:
        assert {"check", "passed", "observed"} <= set(item), "the UI reads all three"
        assert item["observed"], "a check with no observation is not evidence"


# --------------------------------------------------------------------------
# The claim detector itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", FABRICATIONS)
def test_every_fabrication_is_detected(text):
    assert find_claim(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Soll ich die Datei fuer dich erstellen?",
        "Ich kann eine Datei mit diesem Inhalt anlegen, wenn du willst.",
        "Would you like me to create that file?",
        "I can write that to a file if you want.",
        "Rekursion bedeutet, dass eine Funktion sich selbst aufruft.",
        "Der Unterschied zwischen einer Liste und einem Tupel ist die Veraenderbarkeit.",
    ],
)
def test_offering_to_act_is_not_a_claim_of_having_acted(text):
    """Refusing to let the assistant offer help would be its own kind of broken."""

    assert find_claim(text) is None


def test_umlauts_do_not_hide_a_claim():
    """"geprüft" and "geprueft" are the same claim and both must be caught."""

    assert find_claim("Ich habe die Datei erstellt und geprüft.") is not None
    assert find_claim("Ich habe die Datei erstellt und geprueft.") is not None


def test_only_the_installed_product_develops_the_product(live):
    """A core with a temporary state root (this test's) asked to change ZEUS must not
    build in a worktree of the live repository. On 2026-08-27 a test's core did, called
    the real expert and promoted commit 546d43f into the live tree."""

    core, client = live(LyingProvider("done"))
    reply = client.say("Zeus, aendere dein Auge: mach es im Leerlauf groesser.")
    assert "not available here" in reply["text"] or "nicht verf" in reply["text"]
    assert core.selfdev_store.active() is None and core.selfdev_store.list() == []
    assert core.resume_selfdev("nothing")["ok"] is False
