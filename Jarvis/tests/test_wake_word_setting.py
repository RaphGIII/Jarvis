"""Hands-free wake-word listening is an owner switch, off by default, live at runtime.

Nothing of the wake model, the VAD, the listener or Voice Studio is removed:
with ``wake_word_enabled`` off the supervisor simply does not start the
listener, the microphone stays closed unless the owner uses it, and text
chat is untouched.  Switching it on in Voice Studio starts the listener
within the supervisor's watch interval -- no restart, no rebuild.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from service.voice import VoiceSettings
from zeus_supervisor import supervisor as supervisor_module
from zeus_supervisor.config import SupervisorConfig
from zeus_supervisor.supervisor import Supervisor


# --------------------------------------------------------------------------
# The setting
# --------------------------------------------------------------------------

def test_wake_word_listening_is_off_by_default_and_reversible(tmp_path):
    settings = VoiceSettings()
    assert settings.wake_word_enabled is False
    assert settings.enabled is False and settings.speak_replies is True, "the other voice settings are untouched"
    assert settings.apply({"wake_word_enabled": "true"}) == {} and settings.wake_word_enabled is True
    assert settings.apply({"wake_word_enabled": "off"}) == {} and settings.wake_word_enabled is False
    settings.apply({"wake_word_enabled": True})
    path = tmp_path / "voice" / "settings.json"
    settings.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["wake_word_enabled"] is True
    assert VoiceSettings.load(path).wake_word_enabled is True
    assert VoiceSettings.WAKE_INPUTS == ("wake_sensitivity",), "the wake model still reads exactly what it read before"


def test_the_owner_setting_reaches_voice_studio_through_the_normal_endpoint(tmp_path):
    from test_intelligence_flow import ScriptedNetwork, make_world

    core, kernel, local, executed = make_world(tmp_path, ScriptedNetwork(lambda p: {"primary_goal": "none"}))
    status = core.voice_settings()
    assert status["ok"] and status["settings"]["wake_word_enabled"] is False
    changed = core.voice_settings(wake_word_enabled=True)
    assert changed["ok"] and changed["settings"]["wake_word_enabled"] is True
    saved = json.loads((tmp_path / "state" / "voice" / "settings.json").read_text(encoding="utf-8"))
    assert saved["wake_word_enabled"] is True and saved["wake_sensitivity"] == VoiceSettings().wake_sensitivity


def test_text_chat_is_unaffected_by_the_wake_switch(tmp_path):
    from test_gateway_integration import answer_text, ask
    from test_intelligence_flow import ScriptedNetwork, make_world

    core, kernel, local, executed = make_world(tmp_path, ScriptedNetwork(lambda p: {"primary_goal": "none"}))
    assert core.voice.settings.wake_word_enabled is False
    events = ask(core, "Warum ist der Himmel blau?", wait=30)
    assert answer_text(events) == "Das tut mir leid. Nächstes Mal klappt es."


# --------------------------------------------------------------------------
# The supervisor
# --------------------------------------------------------------------------

class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, command, **kwargs) -> None:
        self.command = list(command)
        self.kwargs = kwargs
        self.pid = 4242 + len(FakePopen.instances)
        self.returncode = None
        self.terminated = False
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9



@pytest.fixture
def sup(tmp_path: Path, monkeypatch):
    repo = tmp_path / "Jarvis"
    (repo / "jarvis").mkdir(parents=True)
    (repo / "jarvis" / "serve.py").write_text("print('serve')\n", encoding="utf-8")
    (repo / ".venv-speech" / "Scripts").mkdir(parents=True)
    (repo / ".venv-speech" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    (repo / "data" / "jarvis" / "supervisor" / "logs").mkdir(parents=True)
    FakePopen.instances = []
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", FakePopen)
    # The stale-listener sweep shells out to PowerShell through subprocess.run; not the point here.
    monkeypatch.setattr(Supervisor, "_kill_stale_listeners", lambda self: 0)
    lines: list[str] = []
    config = SupervisorConfig(repository=repo, python="python", open_browser=False, voice=True, port=0)
    supervisor = Supervisor(config, log=lines.append)
    supervisor.token = "t"
    return supervisor, repo, lines


def _set(repo: Path, enabled: bool) -> None:
    path = repo / "data" / "jarvis" / "voice" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = VoiceSettings().to_dict()
    data["wake_word_enabled"] = enabled
    path.write_text(json.dumps(data), encoding="utf-8")


def test_the_listener_is_not_started_when_wake_word_listening_is_off(sup):
    supervisor, repo, lines = sup
    assert supervisor.wake_word_enabled() is False, "no settings file: off"
    supervisor._launch_listener()
    assert FakePopen.instances == [] and supervisor.listener is None
    assert any("wake_word_enabled=false" in line for line in lines)
    _set(repo, False)
    supervisor._launch_listener()
    assert FakePopen.instances == []
    assert sum("not started" in line for line in lines) == 1, "said once, not on every check"


def test_the_listener_starts_when_the_owner_switches_wake_word_listening_on(sup):
    supervisor, repo, lines = sup
    _set(repo, True)
    assert supervisor.wake_word_enabled() is True
    supervisor._launch_listener()
    assert len(FakePopen.instances) == 1 and "speech.listener" in FakePopen.instances[0].command
    assert supervisor.listener is FakePopen.instances[0]
    assert any("launched listener" in line for line in lines)


def test_the_switch_is_live_the_listener_stops_and_starts_with_it(sup):
    supervisor, repo, lines = sup
    _set(repo, True)
    supervisor._sync_listener()
    assert len(FakePopen.instances) == 1 and supervisor.listener.poll() is None
    _set(repo, False)
    supervisor._sync_listener()
    assert FakePopen.instances[0].terminated and any("switched off by the owner" in line for line in lines)
    supervisor._sync_listener()
    assert len(FakePopen.instances) == 1, "off stays off"
    _set(repo, True)
    supervisor._sync_listener()
    assert len(FakePopen.instances) == 2, "on again: a fresh listener, no restart of ZEUS"


def test_a_supervisor_without_voice_never_starts_a_listener_whatever_the_switch_says(sup):
    supervisor, repo, lines = sup
    supervisor.config.voice = False
    _set(repo, True)
    supervisor._launch_listener()
    supervisor._sync_listener()
    assert FakePopen.instances == [] and supervisor.listener is None
