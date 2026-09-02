"""ZEUS's personality: owner-controlled, protected core, fixed prompt order, natural small talk."""

from __future__ import annotations

import time

import pytest

from config import conversation_prompt
from owner.core import DEFAULTS, OwnerCore
from persona.smalltalk import is_small_talk, small_talk_answer
from service.core import JarvisCore


class StubKernel:
    def __init__(self, reply="Ich bin ein autonomer, lokal laufender Engineering-Assistent."):
        self.reply = reply
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        reply = self.reply

        class P:
            def generate_stream(self, prompt, **_):
                yield reply
        return P()


def settle(core, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(t.role == "assistant" for t in core.history):
            time.sleep(0.1)
            return
        time.sleep(0.05)


# --------------------------------------------------------------------------
# Prompt architecture
# --------------------------------------------------------------------------

def test_the_conversation_prompt_has_the_required_order():
    prompt = conversation_prompt(language="de", guidance=["- prefer short answers"], task_style="patient and explanatory", text="hallo")
    i_identity = prompt.index("You are Zeus")
    i_core = prompt.index("Character:")
    i_honesty = prompt.index("Never claim an action was performed")
    i_prefs = prompt.index("Owner preferences:")
    i_lang = prompt.index("The owner is speaking German")
    i_guid = prompt.index("The owner has said")
    i_task = prompt.index("Style for this task")
    assert i_identity < i_core < i_honesty < i_prefs < i_lang < i_guid < i_task
    assert "You are Zeus, an autonomous engineering assistant" not in prompt and "Engineering-Assistent" not in prompt
    assert "not a language model" in prompt


def test_the_protected_core_carries_the_zeus_character_and_the_emotional_rule():
    core = DEFAULTS["personality"]["core"]
    assert "never sycophantic" in core["character"] and "never childish" in core["character"]
    assert any("Wie geht es dir" in line for line in core["emotional_language"])
    assert any("truthfully" in line for line in core["emotional_language"])


def test_dials_change_the_preferences_block_only(tmp_path):
    owner = OwnerCore(tmp_path)
    blocks = dict(owner.personality_blocks())
    assert "keep answers very short" in blocks["preferences"]  # conciseness 70 -> high
    tx = owner.propose({"personality": {"preferences": {"conciseness": 10, "humour": 90}}}, reason="t", origin="ui")
    owner.approve(tx.transaction_id)
    after = dict(owner.personality_blocks())
    assert "answer at comfortable length" in after["preferences"] and "dry humour is welcome" in after["preferences"]
    assert after["core"] == blocks["core"], "the core did not move"


# --------------------------------------------------------------------------
# Owner protection
# --------------------------------------------------------------------------

def test_the_core_is_refused_without_an_explicit_unlock(tmp_path):
    owner = OwnerCore(tmp_path)
    with pytest.raises(PermissionError):
        owner.propose({"personality": {"core": {"character": ["cheerful"]}}}, reason="model wrote this", origin="model")
    tx = owner.propose({"personality": {"core": {"character": ["cheerful", "calm"]}}}, reason="owner", origin="ui", unlock_core=True)
    owner.approve(tx.transaction_id)
    assert owner.read("personality")["core"]["character"] == ["cheerful", "calm"]
    assert owner.read("personality")["core"]["conversation"], "untouched core keys survive"


def test_the_core_service_never_passes_unlock_from_a_non_ui_origin(tmp_path):
    core = JarvisCore(kernel=StubKernel())
    core._owner = OwnerCore(tmp_path)
    # isolate the security gate too: on a machine where the real owner has
    # set a password, the un-isolated gate would answer needs_auth first and
    # this test would stop testing the origin check
    from owner.security_gate import SecurityGate

    core._security = SecurityGate(tmp_path / "auth.json")
    refused = core.owner_propose({"personality": {"core": {"character": ["x"]}}}, reason="selfdev", origin="selfdev", unlock_core=True)
    assert refused["ok"] is False and refused.get("protected")


def test_persona_paths_are_protected():
    from owner.protected import PROTECTED_PATHS, is_protected

    assert "persona" in PROTECTED_PATHS
    assert is_protected("persona/profiles.py") and is_protected("config/owner/personality.json")


def test_a_legacy_flat_personality_file_migrates(tmp_path):
    (tmp_path / "personality.json").write_text('{"traits": ["calm", "precise"], "humour": "none, strictly factual"}', encoding="utf-8")
    owner = OwnerCore(tmp_path)
    p = owner.read("personality")
    assert p["core"]["character"] == ["calm", "precise"] and p["preferences"]["humour"] == 0
    assert p["core"]["emotional_language"], "defaults fill what the old file never had"


# --------------------------------------------------------------------------
# Natural conversation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Wie geht es dir?", "wie geht's dir", "Hallo Zeus, wie geht es dir?", "Alles klar bei dir?", "How are you?"])
def test_ordinary_greetings_are_small_talk(text):
    assert is_small_talk(text)


@pytest.mark.parametrize("text", ["Was bist du technisch?", "Hast du wirklich menschliche Gefühle?", "Wie geht es dir mit dem Projekt Voice?",
                                  "Mach einen Screenshot.", "Der Screenshot ist falsch.", "Do you really feel anything?"])
def test_real_questions_are_not_small_talk(text):
    assert not is_small_talk(text)


def test_wie_geht_es_dir_gets_a_natural_zeus_answer_not_a_self_description():
    core = JarvisCore(kernel=StubKernel())
    core.language = "de"
    core.send_message("Wie geht es dir?")
    settle(core)
    reply = core.history[-1]
    assert reply.backend == "personality"
    assert "Assistent" not in reply.text and "Emotion" not in reply.text and "Bewusst" not in reply.text
    assert "Was steht an?" in reply.text or "Was brauchst du?" in reply.text or "Womit fange ich an?" in reply.text
    assert len(reply.text) < 120


def test_the_answer_uses_real_state():
    text = small_talk_answer("Wie geht es dir?", language="de", active_missions=2, uptime_seconds=7200, humour=90)
    assert "2 Missionen aktiv" in text and "Kaffee" in text
    quiet = small_talk_answer("How are you?", language="en", active_missions=0, humour=0)
    assert "Systems running" in quiet and "coffee" not in quiet


def test_technical_and_literal_questions_reach_the_model_with_the_personality():
    core = JarvisCore(kernel=StubKernel(reply="Technisch bin ich ein lokales System."))
    core.language = "de"
    prompt = core._compose_prompt("Was bist du technisch?")
    assert "give technical detail when asked what you are technically" in prompt
    prompt = core._compose_prompt("Hast du wirklich menschliche Gefühle?")
    assert "answer truthfully and briefly" in prompt
    core.send_message("Was bist du technisch?")
    settle(core)
    assert core.history[-1].backend == "stub"


def test_actions_and_corrections_keep_their_routes():
    from service.intent import classify

    assert classify("Mach einen Screenshot.").intent.value != "conversation"
    assert not is_small_talk("Der Screenshot ist falsch.")
