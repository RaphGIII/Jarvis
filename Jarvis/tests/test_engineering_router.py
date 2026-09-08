"""One engineer-selection path, for every way an owner can ask for code.

Reproduces the failed live acceptance of 2026-09-08. The owner said:

    "Zeus, ich möchte, dass du ab jetzt standardmäßig randlos im Vollbild
     startest. […] Wenn du das noch nicht kannst, bring dir diese Fähigkeit bei
     und setze sie anschließend direkt um."

and mission ``d1309425e9`` answered by running the local coder for eighteen
minutes, having its candidate rejected, and only then asking Codex -- which did
the work in fourteen. It then promoted seven files into the live tree and
restarted ZEUS, with no entry in the owner's audit log.

Two separate defects, pinned separately here: the engineer was chosen in the
wrong order, and the promotion had no gate.
"""

from __future__ import annotations

from typing import Any

import pytest

from capabilities.codex import CodexAvailabilityState, StaticCodexAvailability
from service.engineering import (
    Engineer,
    EngineeringNeed,
    choose_engineer,
    owner_authorized_local_build,
)


# --------------------------------------------------------------------------
# The decision itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("need", list(EngineeringNeed))
def test_codex_is_the_engineer_whenever_it_is_ready(need: EngineeringNeed) -> None:
    availability = StaticCodexAvailability(CodexAvailabilityState.READY, "ready")

    decision = choose_engineer(need, availability=availability)

    assert decision.engineer is Engineer.CODEX
    assert decision.codex_checked is True
    assert decision.to_dict()["build_local_invocations"] == 0
    assert availability.calls == 1, "availability is asked before anything is built"


@pytest.mark.parametrize(
    "state",
    [
        CodexAvailabilityState.QUOTA_EXHAUSTED,
        CodexAvailabilityState.OFFLINE,
        CodexAvailabilityState.BUSY,
        CodexAvailabilityState.AUTH_REQUIRED,
        CodexAvailabilityState.NOT_INSTALLED,
        CodexAvailabilityState.ERROR,
    ],
)
def test_the_local_coder_is_never_an_automatic_fallback(state: CodexAvailabilityState) -> None:
    """The rule mission d1309425e9 broke, in every way Codex can be away."""

    decision = choose_engineer(
        EngineeringNeed.CORE_ENGINEERING,
        availability=StaticCodexAvailability(state, "not now"),
    )

    assert decision.engineer is Engineer.NONE
    assert decision.queued is True
    assert decision.to_dict()["build_local_invocations"] == 0
    assert state.value in decision.reason


def test_the_local_coder_runs_only_when_the_owner_said_so() -> None:
    decision = choose_engineer(
        EngineeringNeed.CORE_ENGINEERING,
        availability=StaticCodexAvailability(CodexAvailabilityState.OFFLINE, "no network"),
        owner_authorized_local=True,
    )

    assert decision.engineer is Engineer.BUILD_LOCAL
    assert decision.owner_authorized_local is True
    assert "explicitly authorized" in decision.reason


def test_a_missing_availability_source_does_not_become_permission_to_build_locally() -> None:
    decision = choose_engineer(EngineeringNeed.CAPABILITY_MISSING, availability=None)

    assert decision.engineer is Engineer.NONE
    assert decision.codex_state == "NOT_CONFIGURED"


def test_an_availability_probe_that_raises_is_a_state_not_a_crash() -> None:
    class _Broken:
        def status(self) -> Any:
            raise RuntimeError("the CLI is wedged")

    decision = choose_engineer(EngineeringNeed.CAPABILITY_BROKEN, availability=_Broken())

    assert decision.engineer is Engineer.NONE
    assert decision.codex_state == "ERROR"
    assert "wedged" in decision.codex_detail


# --------------------------------------------------------------------------
# What counts as the owner authorizing the local coder
# --------------------------------------------------------------------------


AUTHORIZES = [
    "Zeus, mach das lokal",
    "bau es lokal",
    "nimm das lokale Modell",
    "use the local coder for this",
    "BUILD_LOCAL bitte",
]

DOES_NOT_AUTHORIZE = [
    "Wenn du das noch nicht kannst, bring dir diese Fähigkeit bei",
    "Programmier dir eine Vollbild-Umschaltung",
    "Bring dir bei, wie man Dateien vergleicht",
    "Ändere dich so, dass du randlos startest",
    "Füge dir einen Dunkelmodus hinzu",
    "Repariere deine Funktion für Screenshots",
    "entwickle das selbst",
    "mach das selbstständig",
]


@pytest.mark.parametrize("text", AUTHORIZES)
def test_an_explicit_local_authorization_is_recognised(text: str) -> None:
    assert owner_authorized_local_build(text) is True


@pytest.mark.parametrize("text", DOES_NOT_AUTHORIZE)
def test_asking_for_a_capability_is_not_permission_to_build_it_locally(text: str) -> None:
    """"Bring dir das bei" asks for the capability, not for the weaker engineer."""

    assert owner_authorized_local_build(text) is False


# --------------------------------------------------------------------------
# The production path: every phrasing reaches the same router
# --------------------------------------------------------------------------


OWNER_PHRASINGS = [
    pytest.param(
        "Zeus, ich möchte, dass du ab jetzt standardmäßig randlos im Vollbild startest. "
        "F11 soll zwischen Vollbild und Fenstermodus wechseln, Alt+Tab muss weiterhin funktionieren "
        "und die Einstellung soll Neustarts überleben. Wenn du das noch nicht kannst, bring dir "
        "diese Fähigkeit bei und setze sie anschließend direkt um.",
        id="the-live-fullscreen-request",
    ),
    pytest.param("Zeus, programmier dir eine Vollbild-Umschaltung", id="programmier-dir-X"),
    pytest.param("Zeus, bring dir bei, wie man zwei Dateien vergleicht", id="bring-dir-X-bei"),
    pytest.param("Zeus, ändere dich so, dass du randlos startest", id="aendere-dich-so-dass-X"),
    pytest.param("Zeus, füge dir einen Dunkelmodus hinzu", id="fuege-dir-X-hinzu"),
    pytest.param("Zeus, repariere deine Funktion für Screenshots", id="repariere-deine-funktion-X"),
]


@pytest.mark.parametrize("request_text", OWNER_PHRASINGS)
def test_no_phrasing_authorizes_the_local_coder(request_text: str) -> None:
    """None of these are permission; all of them must reach Codex first."""

    assert owner_authorized_local_build(request_text) is False

    decision = choose_engineer(
        EngineeringNeed.CORE_ENGINEERING,
        availability=StaticCodexAvailability(CodexAvailabilityState.READY, "ready"),
        owner_authorized_local=owner_authorized_local_build(request_text),
    )
    assert decision.engineer is Engineer.CODEX


@pytest.mark.parametrize("request_text", OWNER_PHRASINGS)
def test_no_phrasing_falls_back_to_the_local_coder_when_codex_is_away(request_text: str) -> None:
    decision = choose_engineer(
        EngineeringNeed.CORE_ENGINEERING,
        availability=StaticCodexAvailability(CodexAvailabilityState.QUOTA_EXHAUSTED, "spent"),
        owner_authorized_local=owner_authorized_local_build(request_text),
    )

    assert decision.engineer is Engineer.NONE
    assert decision.queued is True


def test_the_owner_sentence_says_what_actually_happened() -> None:
    queued = choose_engineer(
        EngineeringNeed.CORE_ENGINEERING,
        availability=StaticCodexAvailability(CodexAvailabilityState.QUOTA_EXHAUSTED, "spent"),
    )
    text = queued.owner_sentence(german=True)

    assert "QUOTA_EXHAUSTED" in text
    assert "nicht ersatzweise lokal" in text, "it must say what it did NOT do"


# --------------------------------------------------------------------------
# The runner itself: order of operations, and the promotion gate
# --------------------------------------------------------------------------


class _Gate:
    """The Owner Security Gate, as far as the runner can see it."""

    def __init__(self, *, configured: bool = True, token: str = "") -> None:
        self.configured = configured
        self._token = token
        self.asked: list[tuple[str, str]] = []

    def authorized(self, authorization: str, scope: str) -> bool:
        self.asked.append((authorization, scope))
        return bool(self._token) and authorization == self._token


def _runner(tmp_path, *, availability, gate, gateway=None):
    from service.selfdev import SelfDevRunner, SelfDevStore

    repository = tmp_path / "repo"
    (repository / "Jarvis").mkdir(parents=True, exist_ok=True)
    return SelfDevRunner(
        repository=repository,
        store=SelfDevStore(tmp_path / "selfdev"),
        kernel=type("K", (), {"state_root": tmp_path / "state"})(),
        owner=type("O", (), {"read": staticmethod(lambda _name: {})})(),
        lifecycle=None,
        gateway=gateway,
        availability=availability,
        security=gate,
        emit=lambda *a, **k: None,
        set_state=lambda *a, **k: None,
    )


def test_the_runner_asks_the_router_before_it_builds_anything(tmp_path) -> None:
    """Mission d1309425e9 ran BUILD_LOCAL for 18 minutes before asking anyone."""

    from service.selfdev import SelfDevMission

    availability = StaticCodexAvailability(CodexAvailabilityState.READY, "ready")
    runner = _runner(tmp_path, availability=availability, gate=_Gate())

    decision = runner._choose_engineer(SelfDevMission(request="Zeus, bring dir Vollbild bei"))

    assert decision.engineer is Engineer.CODEX
    assert availability.calls == 1


def test_the_runner_selects_the_local_coder_only_on_an_explicit_authorization(tmp_path) -> None:
    from service.selfdev import SelfDevMission

    runner = _runner(
        tmp_path,
        availability=StaticCodexAvailability(CodexAvailabilityState.OFFLINE, "no network"),
        gate=_Gate(),
    )

    plain = runner._choose_engineer(SelfDevMission(request="Zeus, bring dir Vollbild bei"))
    explicit = runner._choose_engineer(SelfDevMission(request="Zeus, bring dir Vollbild bei, mach das lokal"))

    assert plain.engineer is Engineer.NONE
    assert explicit.engineer is Engineer.BUILD_LOCAL


def test_a_chat_mission_carries_no_authorization_and_cannot_promote(tmp_path) -> None:
    """d1309425e9 promoted seven files with nothing in the owner's audit log."""

    from service.selfdev import SelfDevMission

    gate = _Gate(configured=True, token="a-real-token")
    runner = _runner(tmp_path, availability=StaticCodexAvailability(CodexAvailabilityState.READY, ""), gate=gate)
    mission = SelfDevMission(request="Zeus, bring dir Vollbild bei")

    assert mission.authorization == "", "a chat request mints no token"
    assert runner._promotion_authorized(mission) is False
    assert gate.asked == [("", "SELFDEV_PROMOTE")]

    settled = runner._await_authorization(mission)
    assert settled.phase == "AWAITING_AUTHORIZATION"
    assert settled.outcome == "verified_awaiting_authorization"
    assert "password" in settled.reason


def test_a_mission_carrying_the_owners_token_may_promote(tmp_path) -> None:
    from service.selfdev import SelfDevMission

    gate = _Gate(configured=True, token="a-real-token")
    runner = _runner(tmp_path, availability=StaticCodexAvailability(CodexAvailabilityState.READY, ""), gate=gate)
    mission = SelfDevMission(request="Zeus, bring dir Vollbild bei")
    mission.authorization = "a-real-token"

    assert runner._promotion_authorized(mission) is True


def test_no_password_configured_means_nothing_to_prove(tmp_path) -> None:
    """The gate's existing rule everywhere else, followed rather than tightened."""

    from service.selfdev import SelfDevMission

    runner = _runner(
        tmp_path,
        availability=StaticCodexAvailability(CodexAvailabilityState.READY, ""),
        gate=_Gate(configured=False),
    )

    assert runner._promotion_authorized(SelfDevMission(request="x")) is True


def test_a_gate_that_cannot_answer_has_not_said_yes(tmp_path) -> None:
    from service.selfdev import SelfDevMission

    class _Broken:
        configured = True

        def authorized(self, *a: Any) -> bool:
            raise RuntimeError("gate unreadable")

    runner = _runner(tmp_path, availability=StaticCodexAvailability(CodexAvailabilityState.READY, ""), gate=_Broken())

    assert runner._promotion_authorized(SelfDevMission(request="x")) is False
