"""The product name as a setting rather than a spelling.

"Jarvis" was a placeholder and the product is ZEUS. The wrong way to do that is
a global find-and-replace: module paths, class names, state directories and
saved project files all contain the old string, and renaming them would break
everything already on disk to change what appears on a screen.

The one thing configuration cannot make true is the wake word. A detector is
trained weights, not a string, and openWakeWord has never heard of "Zeus". The
tests below pin that the gap is *reported* rather than papered over -- finding
out at the microphone that nothing is listening is the worst way to learn it.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from core.identity import BUILTIN_WAKE_MODELS, Identity, current, set_current


# --------------------------------------------------------------------------
# The names
# --------------------------------------------------------------------------

def test_the_product_is_zeus():
    identity = Identity()

    assert identity.product_name == "ZEUS"
    assert identity.assistant_name == "Zeus"


def test_the_persona_preamble_uses_the_assistant_name():
    preamble = Identity(assistant_name="Zeus").persona_preamble()

    assert "You are Zeus" in preamble
    assert "language model" in preamble, "the anti-disclaimer line must survive"


def test_another_name_needs_no_code_change():
    identity = Identity(product_name="ATHENA", assistant_name="Athena")

    assert "You are Athena" in identity.persona_preamble()
    assert identity.to_dict()["product_name"] == "ATHENA"


# --------------------------------------------------------------------------
# The wake word gap, reported rather than hidden
# --------------------------------------------------------------------------

def test_zeus_has_no_trained_wake_model():
    """The honest state of affairs, asserted so it cannot be forgotten."""

    identity = Identity(wake_word="Zeus")

    assert not identity.wake_word_available
    assert identity.resolved_wake_model == "hey_jarvis"


def test_the_gap_is_explained_in_a_sentence():
    note = Identity(wake_word="Zeus").wake_word_note()

    assert "Zeus" in note
    assert "hey jarvis" in note
    assert "openWakeWord" in note, "it should say how to fix it"


def test_a_word_with_a_model_reports_no_gap():
    identity = Identity(wake_word="jarvis")

    assert identity.wake_word_available
    assert identity.wake_word_note() == ""


def test_an_explicit_model_is_respected():
    identity = Identity(wake_word="Zeus", wake_model="zeus_custom_v1")

    assert identity.resolved_wake_model == "zeus_custom_v1"


def test_a_custom_model_matching_the_word_closes_the_gap():
    identity = Identity(wake_word="zeus", wake_model="zeus")

    assert identity.wake_word_available


def test_the_builtin_list_is_what_openwakeword_actually_ships():
    assert "hey_jarvis" in BUILTIN_WAKE_MODELS
    assert "zeus" not in BUILTIN_WAKE_MODELS


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_defaults_apply_with_no_config(tmp_path):
    identity = Identity.load(config_dir=tmp_path, environ={})

    assert identity.product_name == "ZEUS"


def test_a_config_file_can_rename_the_product(tmp_path):
    (tmp_path / "identity.json").write_text(
        json.dumps({"product_name": "ATHENA", "assistant_name": "Athena"}), encoding="utf-8"
    )

    identity = Identity.load(config_dir=tmp_path, environ={})

    assert identity.assistant_name == "Athena"
    assert str(tmp_path) in identity.source


def test_the_environment_can_override(tmp_path):
    identity = Identity.load(config_dir=tmp_path, environ={"JARVIS_ASSISTANT_NAME": "Hermes"})

    assert identity.assistant_name == "Hermes"


def test_a_corrupt_config_falls_back_to_defaults(tmp_path):
    """A typo in a name file must not stop the system starting."""

    (tmp_path / "identity.json").write_text("{not json", encoding="utf-8")

    assert Identity.load(config_dir=tmp_path, environ={}).product_name == "ZEUS"


def test_unknown_keys_are_ignored(tmp_path):
    (tmp_path / "identity.json").write_text(
        json.dumps({"colour": "blue", "assistant_name": "Zeus"}), encoding="utf-8"
    )

    assert Identity.load(config_dir=tmp_path, environ={}).assistant_name == "Zeus"


def test_an_identity_is_immutable():
    with pytest.raises(Exception):
        Identity().product_name = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------
# Internal names are deliberately left alone
# --------------------------------------------------------------------------

def test_internal_class_names_are_not_renamed():
    """Churning module paths to change a label is risk spent on nothing."""

    from service.core import JarvisCore

    assert JarvisCore.__name__ == "JarvisCore"


def test_the_core_reports_the_product_name_anyway(tmp_path):
    from service.core import JarvisCore

    class Kernel:
        state_root = tmp_path
        catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": ""})())})()

        def provider(self, tier):
            raise RuntimeError("not needed")

    core = JarvisCore(kernel=Kernel())

    assert core.persona_name == "Zeus"
    assert core.status()["product"] == "ZEUS"


def test_the_prompt_introduces_the_assistant_by_the_configured_name(tmp_path):
    from service.core import JarvisCore

    class Kernel:
        state_root = tmp_path
        catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": ""})())})()

        def provider(self, tier):
            raise RuntimeError("not needed")

    core = JarvisCore(kernel=Kernel(), identity=Identity(assistant_name="Athena"))

    assert "You are Athena" in core._compose_prompt("hello")


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------

@pytest.fixture()
def server(tmp_path):
    from service.core import JarvisCore
    from service.http import JarvisHTTPServer

    class Kernel:
        state_root = tmp_path
        catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": ""})())})()

        def provider(self, tier):
            raise RuntimeError("not needed")

    core = JarvisCore(kernel=Kernel())
    instance = JarvisHTTPServer(core, port=0, token="tok")
    instance.start()
    yield instance
    instance.stop()


def test_the_page_is_branded_at_serve_time(server):
    """Substituted rather than baked in, so renaming stays configuration and
    the page stays one document with no build step."""

    with urllib.request.urlopen(f"http://{server.host}:{server.port}/?token=tok", timeout=15) as response:
        page = response.read().decode()

    assert "<title>ZEUS</title>" in page
    assert 'class="brand">ZEUS<' in page
    assert 'window.ASSISTANT_NAME = "Zeus"' in page


def test_no_placeholder_survives_into_the_served_page(server):
    with urllib.request.urlopen(f"http://{server.host}:{server.port}/?token=tok", timeout=15) as response:
        page = response.read().decode()

    assert "__PRODUCT_NAME__" not in page
    assert "__ASSISTANT_NAME__" not in page


def test_diagnostics_expose_the_whole_identity(tmp_path):
    from service.core import JarvisCore

    class Kernel:
        state_root = tmp_path
        catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": ""})())})()

        def provider(self, tier):
            raise RuntimeError("not needed")

    payload = JarvisCore(kernel=Kernel()).diagnostics()

    assert payload["identity"]["product_name"] == "ZEUS"
    assert payload["identity"]["wake_word_available"] is False


# --------------------------------------------------------------------------
# The listener
# --------------------------------------------------------------------------

def test_the_listener_takes_its_model_from_the_identity():
    from speech.listener import ListenerConfig, WakeListener

    listener = WakeListener(ListenerConfig(token="x"))

    assert listener.config.wake_model == "hey_jarvis"
    assert "Zeus" in listener._identity_note


def test_an_explicit_model_beats_the_identity():
    from speech.listener import ListenerConfig, WakeListener

    listener = WakeListener(ListenerConfig(token="x", wake_model="alexa"))

    assert listener.config.wake_model == "alexa"
    assert listener._identity_note == ""
