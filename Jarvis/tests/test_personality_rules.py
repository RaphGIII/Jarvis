"""Owner personality rules: classified, merged, persisted through the owner core, compiled into the
provider-independent contract, and created from a protected chat sentence after the password."""

from __future__ import annotations

import time

import pytest

from service.events import EventType
from test_actionability import make


@pytest.fixture
def isolated_owner(tmp_path, monkeypatch):
    """The owner documents of this test live under tmp_path -- both for the core and for the contract."""

    import owner.core as owner_core
    from persona.contract import _CACHE

    store = owner_core.OwnerCore(tmp_path / "owner_cfg", tmp_path / "owner_state")
    monkeypatch.setattr(owner_core, "_current", store)
    _CACHE.clear()
    yield store
    _CACHE.clear()


def drain(core, text, *, meta=None, wait=6.0):
    with core.bus.subscribe(replay=False) as sub:
        result = core.send_message(text, meta=meta)
        deadline = time.time() + wait
        events = []
        while time.time() < deadline:
            events.extend(sub.drain())
            if any(e.type is EventType.MESSAGE for e in events):
                break
            time.sleep(0.05)
    return result, events


def test_rule_model_classifies_normalises_and_merges():
    from persona.rules import classify_category, make_rule, merge_rule, owner_text

    assert classify_category("Merke dir, dass du auch mein Freund bist und mir zuhörst und Rat gibst.") == "relationship"
    assert classify_category("Du darfst nie ohne Rückfrage Dateien löschen.") == "boundary"
    assert classify_category("Wenn ich lerne, halte dich kurz.") == "situational"
    assert classify_category("Sprich mit mir immer locker.") == "communication_style"
    assert classify_category("Antworte bei Medizinfragen immer mit klinischer Relevanz.") == "domain_preference"
    facing = owner_text("Merke dir, dass du auch mein Freund bist und mir zuhörst und Rat gibst.")
    assert facing == "ZEUS ist auch Raphaels Freund, hört Raphael zu und gibt Rat.", facing
    assert owner_text("Merke dir, dass du mein Freund und Berater bist.") == "ZEUS ist Raphaels Freund und Berater."
    assert owner_text("Merke dir, dass du für mich da bist.") == "ZEUS ist für Raphael da."
    assert owner_text("Merke dir, dass du nie ohne Rückfrage Dateien löschst.") == "ZEUS löscht nie ohne Rückfrage Dateien."
    assert owner_text("Sprich mit mir immer locker und per du.") == "ZEUS spricht mit Raphael immer locker und per du."
    assert owner_text("Wenn ich lerne, halte dich kurz.") == "Wenn Raphael lernt, hält ZEUS sich kurz."
    assert owner_text("Erinnere mich abends an meine Medikamente.") == "ZEUS erinnert Raphael abends an Raphaels Medikamente."
    first = make_rule("Merke dir, dass du auch mein Freund bist und mir zuhörst und Rat gibst.", source="OWNER_CHAT", protected=True)
    rules, stored, how = merge_rule([], first)
    assert how == "added" and len(rules) == 1 and stored["protected"] and stored["version"] == 1
    again = make_rule("Merke dir, dass du mein Freund bist, mir zuhörst und Rat gibst.", source="OWNER_CHAT", protected=True)
    rules, stored, how = merge_rule(rules, again)
    assert how == "updated" and len(rules) == 1 and stored["version"] == 2, "an equivalent rule merges instead of accumulating"
    other = make_rule("Du darfst nie ohne Rückfrage Dateien löschen.", source="OWNER_UI")
    rules, _, how = merge_rule(rules, other)
    assert how == "added" and len(rules) == 2


def test_a_protected_relationship_sentence_becomes_a_rule_after_the_password(tmp_path, isolated_owner):
    core, provider = make(tmp_path)
    core.security.setup("korrektes-passwort-1")
    text = "Merke dir, dass du auch mein Freund bist und mir zuhörst und Rat gibst."
    held, _ = drain(core, text)
    assert held.get("held") is True, "without the password the sentence is held"
    minted = core.security.unlock("korrektes-passwort-1", "PROTECTED_MEMORY")
    result, events = drain(core, text, meta={"authorization": minted["authorization"], "source": "text"})
    assert result.get("stored") is True and result.get("rule_id"), result
    answers = [e.payload["text"] for e in events if e.type is EventType.MESSAGE]
    assert answers and answers[-1].startswith("Gemerkt – als Regel unter Persönlichkeit (Beziehung zu Raphael)"), answers
    assert provider.prompts == [], "the rule is written directly; no model is asked"
    # persisted in the owner personality document, protected, categorised, with provenance
    view = core.personality_view()
    rules = view["rules"]
    assert len(rules) == 1 and rules[0]["category"] == "relationship" and rules[0]["protected"] and rules[0]["source"] == "OWNER_CHAT"
    assert rules[0]["source_text"] == text and rules[0]["version"] == 1 and rules[0]["created_at"]
    assert isolated_owner.read("personality")["rules"][0]["id"] == rules[0]["id"], "it is the owner document, not a side store"
    # and in the NEXT contract, for every provider alike
    from persona.contract import current_contract

    contract = current_contract(scope="chat")
    assert "Freund" in contract.blocks["rules"] and "relationship to Raphael" in contract.blocks["rules"]
    assert "Freund" in current_contract(scope="role").text
    from config import conversation_prompt, system_prompt

    assert "Freund" in conversation_prompt(text="Was bist du noch?", language="de") and "Freund" in system_prompt()
    # the same sentence again merges, it does not duplicate
    minted = core.security.unlock("korrektes-passwort-1", "PROTECTED_MEMORY")
    result, _ = drain(core, "Merke dir, dass du mein Freund bist, mir zuhörst und Rat gibst.", meta={"authorization": minted["authorization"], "source": "text"})
    assert result.get("how") == "updated" and len(core.personality_view()["rules"]) == 1


def test_a_fact_about_the_owner_stays_a_protected_note(tmp_path, isolated_owner):
    core, provider = make(tmp_path)
    core.security.setup("korrektes-passwort-1")
    minted = core.security.unlock("korrektes-passwort-1", "PROTECTED_MEMORY")
    result, _ = drain(core, "Merke dir: mein Geburtstag ist der 3. März.", meta={"authorization": minted["authorization"], "source": "text"})
    assert result.get("stored") is True and result.get("node_id") and not result.get("rule_id")
    assert core.personality_view()["rules"] == []


def test_the_question_what_else_are_you_is_not_intercepted_deterministically():
    """'Was bist du noch?' must reach the model with the contract -- it is not the identity firewall's question."""

    from persona.smalltalk import identity_answer

    assert identity_answer("Was bist du noch?", language="de", assistant="ZEUS", creator="Raphael") is None
    assert identity_answer("Wer bist du?", language="de", assistant="ZEUS", creator="Raphael") is not None


def test_protected_rules_need_the_password_to_be_removed_or_rewritten(tmp_path, isolated_owner):
    core, provider = make(tmp_path)
    core.security.setup("korrektes-passwort-1")
    stored = core.personality_rule_add("Merke dir, dass du auch mein Freund bist und mir zuhörst.", source="OWNER_CHAT", protected=True)
    assert stored["ok"]
    rule = stored["rule"]
    # removing it without authorization is refused
    refused = core.personality_save({"rules": []}, reason="test")
    assert refused.get("ok") is False and refused.get("needs_auth") == "PERSONALITY_EDIT"
    # disabling it without authorization is refused too
    refused = core.personality_save({"rules": [{**rule, "enabled": False}]}, reason="test")
    assert refused.get("ok") is False and refused.get("needs_auth") == "PERSONALITY_EDIT"
    # an unprotected UI rule can be added freely, the protected one stays -- and the editor cannot
    # declare its own rule protected
    added = core.personality_save({"rules": [rule, {"text": "Bei Programmierung kurz und lösungsorientiert.", "enabled": True, "protected": True}]}, reason="test")
    assert added.get("ok") is True and len(added["rules"]) == 2
    assert added["rules"][1]["category"] == "domain_preference" and added["rules"][1]["protected"] is False and added["rules"][1]["source"] == "OWNER_UI"
    assert added["rules"][0]["protected"] is True and added["rules"][0]["source"] == "OWNER_CHAT"
    # with the authorization the protected rule can go
    token = core.security.unlock("korrektes-passwort-1", "PERSONALITY_EDIT")["authorization"]
    gone = core.personality_save({"rules": [added["rules"][1]]}, reason="test", authorization=token)
    assert gone.get("ok") is True and len(gone["rules"]) == 1


def test_owner_ui_preferences_persist_on_the_server(tmp_path):
    core, provider = make(tmp_path)
    assert core.ui_preferences()["preferences"]["ui.show_spend"] is True
    assert core.ui_preference_set("ui.show_spend", False)["ok"]
    assert core.ui_preferences()["preferences"]["ui.show_spend"] is False
    assert core.ui_preference_set("ui.nonsense", 1)["ok"] is False


def test_the_follow_up_question_reaches_the_model_carrying_the_stored_rule(tmp_path, isolated_owner):
    """No canned answer: 'Was bist du noch?' goes to the model, and what the model receives holds the rule."""

    from test_actionability import Provider

    class Recording(Provider):
        def __init__(self):
            super().__init__()
            self.systems = []

        def generate_stream(self, prompt, **kw):
            self.systems.append(kw.get("system", ""))
            self.prompts.append(prompt)
            yield "Auch dein Freund."

    rec = Recording()
    core, _ = make(tmp_path, provider=rec)
    core.security.setup("korrektes-passwort-1")
    minted = core.security.unlock("korrektes-passwort-1", "PROTECTED_MEMORY")
    stored, _ = drain(core, "Merke dir, dass du auch mein Freund bist und mir zuhörst und Rat gibst.",
                      meta={"authorization": minted["authorization"], "source": "text"})
    assert stored.get("rule_id") and rec.prompts == []
    _, events = drain(core, "Was bist du noch?", wait=20.0)
    answers = [e.payload for e in events if e.type is EventType.MESSAGE]
    assert answers and answers[-1]["text"] == "Auch dein Freund.", answers
    carried = "\n".join(rec.systems + rec.prompts)
    assert "ist auch Raphaels Freund, hört Raphael zu und gibt Rat." in carried and any("Owner rules" in s for s in rec.systems), carried[:2000]
