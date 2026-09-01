"""The Owner Security Gate: scrypt verifier, scoped short-lived tokens, no leaks."""

from __future__ import annotations

import json

import pytest

from owner.security_gate import SCOPES, SecurityGate


@pytest.fixture
def gate(tmp_path):
    return SecurityGate(tmp_path / "auth.json", token_seconds=300)


def test_setup_stores_a_verifier_never_the_password(gate, tmp_path):
    assert gate.configured is False
    assert gate.setup("korrektes-pferd-batterie")["ok"] is True
    raw = (tmp_path / "auth.json").read_bytes()
    assert b"korrektes-pferd-batterie" not in raw
    if raw[:1] == b"{":  # non-DPAPI fallback: inspect the record
        record = json.loads(raw)
        assert record["kdf"] == "scrypt" and record["salt"] and record["verifier"]
        assert "korrektes" not in json.dumps(record)


def test_wrong_password_denied_correct_password_scoped(gate):
    gate.setup("korrektes-pferd-batterie")
    assert gate.unlock("falsch", "PERSONALITY_EDIT")["ok"] is False
    out = gate.unlock("korrektes-pferd-batterie", "PERSONALITY_EDIT")
    assert out["ok"] and out["scope"] == "PERSONALITY_EDIT"
    token = out["authorization"]
    assert gate.authorized(token, "PERSONALITY_EDIT") is True
    assert gate.authorized(token, "SECURITY_CONFIG") is False, "a token authorizes exactly its scope"
    assert gate.authorized("made-up-by-a-model", "PERSONALITY_EDIT") is False


def test_tokens_expire_and_lock_drops_them(gate):
    gate.setup("korrektes-pferd-batterie")
    out = gate.unlock("korrektes-pferd-batterie", "PROJECT_DELETE", seconds=0.0)
    token = out["authorization"]
    import time

    time.sleep(0.05)
    assert gate.authorized(token, "PROJECT_DELETE") is False, "expired"
    out2 = gate.unlock("korrektes-pferd-batterie", "PROJECT_DELETE")
    assert gate.lock() >= 1
    assert gate.authorized(out2["authorization"], "PROJECT_DELETE") is False


def test_restart_locks_everything_but_keeps_the_password(gate, tmp_path):
    gate.setup("korrektes-pferd-batterie")
    token = gate.unlock("korrektes-pferd-batterie", "PERSONALITY_EDIT")["authorization"]
    fresh = SecurityGate(tmp_path / "auth.json")
    assert fresh.configured is True
    assert fresh.authorized(token, "PERSONALITY_EDIT") is False, "tokens are memory-only"
    assert fresh.unlock("korrektes-pferd-batterie", "PERSONALITY_EDIT")["ok"] is True


def test_changing_the_password_requires_the_current_one(gate):
    gate.setup("korrektes-pferd-batterie")
    assert gate.setup("neues-passwort-123", current="falsch")["ok"] is False
    assert gate.setup("neues-passwort-123", current="korrektes-pferd-batterie")["ok"] is True
    assert gate.unlock("korrektes-pferd-batterie", "PERSONALITY_EDIT")["ok"] is False
    assert gate.unlock("neues-passwort-123", "PERSONALITY_EDIT")["ok"] is True


def test_consume_is_single_use_for_critical_operations(gate):
    gate.setup("korrektes-pferd-batterie")
    token = gate.unlock("korrektes-pferd-batterie", "AUTH_ADMIN")["authorization"]
    assert gate.consume(token, "AUTH_ADMIN") is True
    assert gate.consume(token, "AUTH_ADMIN") is False


def test_five_failures_back_off(gate):
    gate.setup("korrektes-pferd-batterie")
    for _ in range(5):
        assert gate.verify("falsch") is False
    assert gate.verify("korrektes-pferd-batterie") is False, "backoff window"


def test_short_passwords_are_refused_and_scopes_are_closed(gate):
    assert gate.setup("kurz")["ok"] is False
    gate.setup("korrektes-pferd-batterie")
    assert gate.unlock("korrektes-pferd-batterie", "EVERYTHING")["ok"] is False
    assert set(gate.status()["scopes"]) == set(SCOPES)


def test_status_never_contains_secret_material(gate):
    gate.setup("korrektes-pferd-batterie")
    text = json.dumps(gate.status())
    assert "korrektes" not in text and "verifier" not in text and "salt" not in text
