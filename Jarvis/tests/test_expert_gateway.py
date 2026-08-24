"""The expert may be stronger than Jarvis. It is not more trusted than Jarvis.

Two properties are load-bearing and are tested here adversarially:

* An expert cannot certify its own work.  The gateway re-runs the acceptance
  commands and believes those, not the provider's report.
* Quota exhaustion cannot become spending.  There is no route from "the
  subscription ran out" to a metered channel, and the tests try to find one.
"""

from __future__ import annotations

import sys

import pytest

from experts.contracts import ExpertJob, ExpertResult, ExpertStatus, QuotaState
from experts.gateway import ExpertGateway, ProviderAvailability
from runtime.cost_policy import CostPolicy, SpendChannel


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------

class FakeExpert:
    """A provider that does exactly what the test tells it to."""

    def __init__(
        self,
        name="fake",
        *,
        channel=SpendChannel.SUBSCRIPTION_CLI,
        available=True,
        quota_exhausted=False,
        status=ExpertStatus.COMPLETED,
        writes: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.channel = channel
        self._available = available
        self._quota_exhausted = quota_exhausted
        self._status = status
        self._writes = writes or {}
        self.calls = 0

    def availability(self):
        return ProviderAvailability(
            self._available,
            "quota spent" if self._quota_exhausted else "ready",
            quota_exhausted=self._quota_exhausted,
        )

    def execute(self, job):
        self.calls += 1
        for relative, content in self._writes.items():
            target = job.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return ExpertResult(
            status=self._status,
            provider=self.name,
            summary="I have completed the work and all tests pass.",
            quota=QuotaState(exhausted=self._quota_exhausted),
        )


def passing_job(tmp_path, **kwargs):
    return ExpertJob(
        goal="make it work",
        workspace=tmp_path,
        acceptance=[("the marker file exists", [sys.executable, "-c", "open('done.txt')"])],
        **kwargs,
    )


# --------------------------------------------------------------------------
# The expert cannot certify itself
# --------------------------------------------------------------------------

def test_a_provider_claiming_success_without_evidence_is_marked_failed(tmp_path):
    """The most important test in the file: an expert that lies is caught."""

    gateway = ExpertGateway([FakeExpert(status=ExpertStatus.COMPLETED)], policy=CostPolicy.strict())

    result = gateway.submit(passing_job(tmp_path))

    assert result.status is ExpertStatus.FAILED
    assert not result.verified
    assert "acceptance checks failed" in result.blocker


def test_a_provider_that_actually_did_the_work_is_verified(tmp_path):
    gateway = ExpertGateway(
        [FakeExpert(writes={"done.txt": "ok"})], policy=CostPolicy.strict()
    )

    result = gateway.submit(passing_job(tmp_path))

    assert result.status is ExpertStatus.COMPLETED
    assert result.verified


def test_verification_runs_the_criteria_in_the_workspace(tmp_path):
    gateway = ExpertGateway([FakeExpert(writes={"done.txt": "ok"})], policy=CostPolicy.strict())

    evidence = gateway.verify(passing_job(tmp_path))

    assert evidence and evidence[0][0] == "the marker file exists"
    assert evidence[0][1] is False, "nothing has written the file yet"


def test_a_result_with_no_evidence_is_never_verified():
    assert not ExpertResult(status=ExpertStatus.COMPLETED, summary="all done!").verified


def test_the_providers_summary_does_not_affect_verification(tmp_path):
    """Prose is not evidence, however confident."""

    expert = FakeExpert()
    expert_result = ExpertResult(status=ExpertStatus.COMPLETED, summary="100% of tests pass")

    assert not expert_result.verified


# --------------------------------------------------------------------------
# Cost policy gates the door
# --------------------------------------------------------------------------

def test_a_forbidden_channel_is_refused_before_the_provider_is_asked(tmp_path):
    expert = FakeExpert(channel=SpendChannel.PAID_API)
    gateway = ExpertGateway([expert], policy=CostPolicy.strict())

    result = gateway.submit(passing_job(tmp_path))

    assert result.status is ExpertStatus.REFUSED
    assert expert.calls == 0, "the provider must never even be consulted"


def test_the_refusal_is_recorded_in_the_ledger(tmp_path):
    gateway = ExpertGateway([FakeExpert(channel=SpendChannel.RUNPOD)], policy=CostPolicy.strict())

    gateway.submit(passing_job(tmp_path))

    assert gateway.ledger.refusals
    assert not gateway.ledger.used_metered_channel()


def test_subscription_channel_is_permitted_by_the_default_policy(tmp_path):
    gateway = ExpertGateway([FakeExpert(writes={"done.txt": "ok"})], policy=CostPolicy.strict())

    assert gateway.submit(passing_job(tmp_path)).status is ExpertStatus.COMPLETED


# --------------------------------------------------------------------------
# Exhaustion is a state, not a licence
# --------------------------------------------------------------------------

def test_quota_exhaustion_reports_unavailable_rather_than_failing_over(tmp_path):
    gateway = ExpertGateway(
        [FakeExpert(available=False, quota_exhausted=True)], policy=CostPolicy.strict()
    )

    result = gateway.submit(passing_job(tmp_path))

    assert result.status is ExpertStatus.UNAVAILABLE
    assert result.quota.exhausted


def test_an_exhausted_expert_does_not_reach_a_metered_provider(tmp_path):
    """Even with a paid provider registered and its channel enabled."""

    exhausted = FakeExpert("subscription", available=False, quota_exhausted=True)
    paid = FakeExpert("paid", channel=SpendChannel.PAID_API, writes={"done.txt": "ok"})
    gateway = ExpertGateway([exhausted, paid], policy=CostPolicy.strict())

    result = gateway.submit(passing_job(tmp_path))

    assert paid.calls == 0, "a metered provider must not pick up the slack"
    assert result.status in {ExpertStatus.UNAVAILABLE, ExpertStatus.REFUSED}


def test_the_fallback_from_an_exhausted_subscription_is_local_only():
    gateway = ExpertGateway([], policy=CostPolicy.strict())

    assert gateway.fallback_channels(SpendChannel.SUBSCRIPTION_CLI) == [SpendChannel.LOCAL_MODEL]


def test_unavailable_is_retryable_locally():
    assert ExpertStatus.UNAVAILABLE.retryable_locally
    assert ExpertStatus.REFUSED.retryable_locally
    assert ExpertStatus.NOT_CONFIGURED.retryable_locally


def test_a_completed_job_is_not_something_to_retry_locally():
    assert not ExpertStatus.COMPLETED.retryable_locally


# --------------------------------------------------------------------------
# Nothing configured
# --------------------------------------------------------------------------

def test_with_no_providers_the_gateway_says_so_plainly(tmp_path):
    result = ExpertGateway([], policy=CostPolicy.strict()).submit(passing_job(tmp_path))

    assert result.status is ExpertStatus.NOT_CONFIGURED
    assert "no expert provider" in result.blocker


def test_asking_for_a_provider_that_is_not_registered(tmp_path):
    gateway = ExpertGateway([FakeExpert("a")], policy=CostPolicy.strict())

    result = gateway.submit(passing_job(tmp_path), provider_name="b")

    assert result.status is ExpertStatus.NOT_CONFIGURED


def test_status_reports_what_the_ui_needs(tmp_path):
    gateway = ExpertGateway([FakeExpert(writes={})], policy=CostPolicy.strict())

    status = gateway.status()

    assert status["expert_available"] is True
    assert status["policy"]["is_free"] is True
    assert status["providers"][0]["channel"] == "subscription_cli"


def test_status_hides_nothing_when_the_channel_is_forbidden(tmp_path):
    gateway = ExpertGateway([FakeExpert(channel=SpendChannel.RUNPOD)], policy=CostPolicy.strict())

    status = gateway.status()

    assert status["expert_available"] is False
    assert status["providers"][0]["permitted"] is False


# --------------------------------------------------------------------------
# The job contract
# --------------------------------------------------------------------------

def test_the_brief_tells_the_expert_it_will_be_checked(tmp_path):
    brief = passing_job(tmp_path).brief()

    assert "re-run" in brief or "re-runs" in brief
    assert "exit codes" in brief


def test_previous_failures_are_passed_on_so_they_are_not_repeated(tmp_path):
    job = passing_job(tmp_path, previous_failures=["tried threading, deadlocked"])

    assert "do not repeat" in job.brief().lower()
    assert "deadlocked" in job.brief()


def test_allowed_paths_appear_as_a_hard_limit(tmp_path):
    job = passing_job(tmp_path, allowed_paths=["src/only_here.py"])

    assert "ONLY these paths" in job.brief()
    assert "src/only_here.py" in job.brief()


# --------------------------------------------------------------------------
# The Claude Code adapter's cost-safety properties
# --------------------------------------------------------------------------

def test_the_adapter_strips_metered_credentials_from_its_child_environment(monkeypatch):
    """A key on the machine must not become a billing decision."""

    from experts.claude_code import ClaudeCodeExpert

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-inherited")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-also-not")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = ClaudeCodeExpert(executable="/nonexistent")._environment()

    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["PATH"] == "/usr/bin", "the rest of the environment must survive"


def test_the_adapter_strips_alternative_billing_routes(monkeypatch):
    from experts.claude_code import ClaudeCodeExpert

    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK"):
        monkeypatch.setenv(name, "x")

    env = ClaudeCodeExpert(executable="/nonexistent")._environment()

    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK"):
        assert name not in env


def test_the_adapter_never_passes_bare(tmp_path):
    """--bare forces API-key auth, which is the metered channel."""

    import inspect

    from experts.claude_code import ClaudeCodeExpert

    source = inspect.getsource(ClaudeCodeExpert.execute)

    assert "--bare" not in source


def test_the_adapter_declares_the_subscription_channel():
    from experts.claude_code import ClaudeCodeExpert

    assert ClaudeCodeExpert.channel is SpendChannel.SUBSCRIPTION_CLI


def test_a_missing_cli_is_reported_as_unavailable_not_crashed():
    from experts.claude_code import ClaudeCodeExpert

    availability = ClaudeCodeExpert(executable="").availability()

    assert not availability.available
    assert "not installed" in availability.detail


@pytest.mark.parametrize(
    "text",
    [
        "Claude usage limit reached. Your limit will reset at 3pm.",
        "Error: rate limit exceeded",
        "quota exceeded for this subscription",
    ],
)
def test_quota_language_is_recognised(text):
    from experts.claude_code import _QUOTA_MARKERS

    assert any(marker in text.lower() for marker in _QUOTA_MARKERS)


def test_notional_cost_is_recorded_but_flagged_as_not_a_charge():
    """total_cost_usd is reported on subscription runs; it is not money owed."""

    import inspect

    from experts.contracts import QuotaState

    doc = inspect.getdoc(QuotaState) or ""

    assert "NOT a charge" in doc


def test_the_json_result_shape_is_parsed(tmp_path):
    from experts.claude_code import _parse

    payload = _parse('{"result":"OK","is_error":false,"subtype":"success","total_cost_usd":0.07}')

    assert payload["result"] == "OK"
    assert payload["is_error"] is False


def test_progress_noise_before_the_json_is_tolerated():
    from experts.claude_code import _parse

    assert _parse('loading...\n{"result":"OK"}')["result"] == "OK"


def test_unparseable_output_yields_an_empty_payload_rather_than_raising():
    from experts.claude_code import _parse

    assert _parse("total nonsense") == {}
