"""The one rule that must never bend: no silent usage-based cost.

These tests are adversarial on purpose.  The realistic way a system like this
starts billing its owner is not a decision -- it is a fallback: the subscription
hits a limit, an API key happens to be in the environment, and a reasonable
`except` clause turns "the expert is busy" into a metered call.  So the tests
below mostly try to *reach* a metered channel by the routes that would actually
be taken.
"""

from __future__ import annotations

import json

import pytest

from runtime.cost_policy import (
    EXPERT_UNAVAILABLE,
    CostLedger,
    CostPolicy,
    CostPolicyViolation,
    SpendChannel,
)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

def test_the_shipped_default_cannot_spend_anything():
    policy = CostPolicy.strict()

    assert policy.is_free
    assert policy.metered_channels_enabled == []


def test_local_and_subscription_are_on_by_default():
    policy = CostPolicy.strict()

    assert policy.permits(SpendChannel.LOCAL_MODEL)
    assert policy.permits(SpendChannel.SUBSCRIPTION_CLI)


@pytest.mark.parametrize(
    "channel",
    [SpendChannel.PAID_API, SpendChannel.USAGE_CREDITS, SpendChannel.RUNPOD],
)
def test_every_metered_channel_is_off_by_default(channel):
    assert not CostPolicy.strict().permits(channel)


def test_browser_automation_of_ai_chat_is_off_by_default():
    assert not CostPolicy.strict().permits(SpendChannel.BROWSER_AI_AUTOMATION)


def test_loading_with_no_config_and_no_env_yields_the_strict_policy(tmp_path):
    policy = CostPolicy.load(config_dir=tmp_path, environ={})

    assert policy.is_free
    assert policy.to_dict()["allow_paid_api"] is False


# --------------------------------------------------------------------------
# A credential is not consent
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "variable",
    ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RUNPOD_API_KEY", "OPENROUTER_API_KEY"],
)
def test_a_key_in_the_environment_does_not_enable_billing(tmp_path, variable):
    """The exact accident: a key is present, so something assumes permission."""

    policy = CostPolicy.load(config_dir=tmp_path, environ={variable: "sk-live-not-a-permission"})

    assert not policy.permits(SpendChannel.PAID_API)
    assert policy.is_free


def test_normal_operation_needs_no_api_key_at_all(tmp_path):
    policy = CostPolicy.load(config_dir=tmp_path, environ={})

    assert policy.permits(SpendChannel.LOCAL_MODEL)
    assert policy.permits(SpendChannel.SUBSCRIPTION_CLI)


# --------------------------------------------------------------------------
# Refusal is loud
# --------------------------------------------------------------------------

def test_require_raises_rather_than_returning_false():
    """A returned False can be ignored; an exception cannot."""

    with pytest.raises(CostPolicyViolation) as caught:
        CostPolicy.strict().require(SpendChannel.PAID_API)

    assert caught.value.channel is SpendChannel.PAID_API


def test_the_refusal_says_what_to_do_about_it():
    message = CostPolicy.strict().explain(SpendChannel.PAID_API)

    assert "usage-based" in message
    assert "cost_policy.json" in message


def test_the_browser_refusal_names_the_real_objection():
    message = CostPolicy.strict().explain(SpendChannel.BROWSER_AI_AUTOMATION)

    assert "terms" in message


def test_require_is_silent_when_the_channel_is_allowed():
    CostPolicy.strict().require(SpendChannel.LOCAL_MODEL)
    CostPolicy.strict().require(SpendChannel.SUBSCRIPTION_CLI)


# --------------------------------------------------------------------------
# Quota exhaustion must not become a bill
# --------------------------------------------------------------------------

def test_an_exhausted_subscription_falls_back_only_to_local():
    """The single most dangerous path in the whole system."""

    fallbacks = CostPolicy.strict().fallbacks_for(SpendChannel.SUBSCRIPTION_CLI)

    assert fallbacks == [SpendChannel.LOCAL_MODEL]


def test_no_fallback_is_ever_a_metered_channel():
    """True even for a policy that has metered channels switched on."""

    permissive = CostPolicy(allow_paid_api=True, allow_usage_credits=True, allow_runpod=True)

    for channel in SpendChannel:
        for fallback in permissive.fallbacks_for(channel):
            assert not fallback.metered, f"{channel} fell back to metered {fallback}"


def test_with_local_models_disabled_exhaustion_yields_no_fallback_at_all():
    """Nothing left is the correct answer; inventing a paid one is not."""

    policy = CostPolicy(allow_local_models=False)

    assert policy.fallbacks_for(SpendChannel.SUBSCRIPTION_CLI) == []


def test_expert_unavailable_is_a_state_not_an_error_code():
    assert EXPERT_UNAVAILABLE == "EXPERT_UNAVAILABLE"


# --------------------------------------------------------------------------
# Turning something on has to be deliberate
# --------------------------------------------------------------------------

def test_a_config_file_can_enable_a_metered_channel(tmp_path):
    (tmp_path / "cost_policy.json").write_text(
        json.dumps({"allow_paid_api": True}), encoding="utf-8"
    )

    policy = CostPolicy.load(config_dir=tmp_path, environ={})

    assert policy.permits(SpendChannel.PAID_API)
    assert not policy.is_free
    assert str(tmp_path) in policy.source


def test_an_explicit_env_switch_can_enable_a_metered_channel(tmp_path):
    policy = CostPolicy.load(config_dir=tmp_path, environ={"JARVIS_ALLOW_RUNPOD": "1"})

    assert policy.permits(SpendChannel.RUNPOD)


def test_an_env_switch_can_also_turn_something_off(tmp_path):
    policy = CostPolicy.load(config_dir=tmp_path, environ={"JARVIS_ALLOW_SUBSCRIPTION_CLI": "0"})

    assert not policy.permits(SpendChannel.SUBSCRIPTION_CLI)


def test_unknown_keys_in_the_config_are_ignored(tmp_path):
    (tmp_path / "cost_policy.json").write_text(
        json.dumps({"allow_everything": True, "allow_paid_api": False}), encoding="utf-8"
    )

    assert CostPolicy.load(config_dir=tmp_path, environ={}).is_free


def test_a_corrupt_config_falls_back_to_strict_rather_than_open(tmp_path):
    """Failing open on a parse error would be the worst possible default."""

    (tmp_path / "cost_policy.json").write_text("{not json", encoding="utf-8")

    assert CostPolicy.load(config_dir=tmp_path, environ={}).is_free


def test_a_policy_is_immutable():
    policy = CostPolicy.strict()

    with pytest.raises(Exception):
        policy.allow_paid_api = True  # type: ignore[misc]


# --------------------------------------------------------------------------
# The audit trail
# --------------------------------------------------------------------------

def test_the_ledger_records_refusals():
    ledger = CostLedger(CostPolicy.strict())

    assert not ledger.check(SpendChannel.PAID_API, reason="escalating a hard task")

    assert len(ledger.refusals) == 1
    assert ledger.refusals[0].channel is SpendChannel.PAID_API


def test_the_ledger_can_substantiate_that_nothing_was_billed():
    ledger = CostLedger(CostPolicy.strict())
    ledger.check(SpendChannel.LOCAL_MODEL)
    ledger.check(SpendChannel.SUBSCRIPTION_CLI)
    ledger.check(SpendChannel.PAID_API)

    assert not ledger.used_metered_channel()
    assert ledger.summary()["refusals"] == 1


def test_the_ledger_notices_when_a_metered_channel_was_permitted():
    ledger = CostLedger(CostPolicy(allow_paid_api=True))

    ledger.check(SpendChannel.PAID_API)

    assert ledger.used_metered_channel()


def test_the_ledger_does_not_grow_without_bound():
    ledger = CostLedger(CostPolicy.strict(), limit=10)

    for _ in range(100):
        ledger.check(SpendChannel.LOCAL_MODEL)

    assert len(ledger.decisions) == 10


def test_ledger_require_raises_and_still_records():
    ledger = CostLedger(CostPolicy.strict())

    with pytest.raises(CostPolicyViolation):
        ledger.require(SpendChannel.RUNPOD)

    assert ledger.refusals[0].channel is SpendChannel.RUNPOD


# --------------------------------------------------------------------------
# Channel classification
# --------------------------------------------------------------------------

def test_subscription_and_local_are_not_metered():
    assert not SpendChannel.SUBSCRIPTION_CLI.metered
    assert not SpendChannel.LOCAL_MODEL.metered


def test_paid_api_credits_and_rented_gpus_are_metered():
    assert SpendChannel.PAID_API.metered
    assert SpendChannel.USAGE_CREDITS.metered
    assert SpendChannel.RUNPOD.metered
