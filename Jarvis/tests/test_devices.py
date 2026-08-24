"""Identity and pairing for the things that talk to Jarvis Core.

The shared token this replaces had one fatal property: it could not be revoked.
Losing a phone meant changing the secret every other client already used. These
tests pin the properties per-device credentials buy — revocation in isolation,
enrolment as a human decision rather than anything that can reach the port, and
presence tracked separately from pairing.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from devices.gateway import PAIRING_TTL_SECONDS, Device, DeviceGateway, PairingRequest
from service.core import JarvisCore
from service.http import JarvisHTTPServer


@pytest.fixture()
def gateway(tmp_path):
    return DeviceGateway(tmp_path / "devices.json")


def pair(gateway, name="kitchen box", **kwargs):
    request = gateway.request_pairing(name, **kwargs)
    device = gateway.approve(request.code)
    return request, device


# --------------------------------------------------------------------------
# Enrolment is a decision
# --------------------------------------------------------------------------

def test_a_device_cannot_pair_itself(gateway):
    """Anything that can reach the port would otherwise be able to join."""

    request = gateway.request_pairing("uninvited")

    assert not request.approved
    assert gateway.devices() == []


def test_approval_issues_a_credential(gateway):
    _, device = pair(gateway)

    assert device is not None
    assert device.token
    assert device.id.startswith("dev_")


def test_the_device_collects_its_own_token_once(gateway):
    """A token that can be collected twice can be collected by someone else."""

    request = gateway.request_pairing("box")
    gateway.approve(request.code)

    first = gateway.collect(request.code)
    second = gateway.collect(request.code)

    assert first["status"] == "paired" and first["token"]
    assert second["status"] == "unknown"


def test_collecting_before_approval_reports_pending(gateway):
    request = gateway.request_pairing("box")

    assert gateway.collect(request.code)["status"] == "pending"


def test_a_denied_request_yields_nothing(gateway):
    request = gateway.request_pairing("box")

    assert gateway.deny(request.code)

    assert gateway.collect(request.code)["status"] == "denied"
    assert gateway.approve(request.code) is None


def test_an_expired_request_cannot_be_approved(gateway, monkeypatch):
    """A code that lives for hours is a code an attacker has hours to use."""

    request = gateway.request_pairing("box")
    request.requested_at -= PAIRING_TTL_SECONDS + 1

    assert gateway.approve(request.code) is None


def test_an_unknown_code_is_refused(gateway):
    assert gateway.approve("000000") is None
    assert gateway.collect("000000")["status"] == "unknown"


def test_the_code_is_short_enough_to_read_aloud(gateway):
    """It has to be read off one screen and typed into another."""

    code = gateway.request_pairing("box").code

    assert len(code) == 6 and code.isdigit()


def test_pending_requests_are_listed_for_a_human(gateway):
    gateway.request_pairing("box one")
    gateway.request_pairing("box two")

    assert {item.name for item in gateway.pending()} == {"box one", "box two"}


def test_an_approved_request_leaves_the_pending_list(gateway):
    request = gateway.request_pairing("box")
    gateway.approve(request.code)

    assert gateway.pending() == []


# --------------------------------------------------------------------------
# Authentication and revocation
# --------------------------------------------------------------------------

def test_a_paired_device_authenticates(gateway):
    _, device = pair(gateway)

    assert gateway.authenticate(device.id, device.token) is not None


def test_a_wrong_token_does_not(gateway):
    _, device = pair(gateway)

    assert gateway.authenticate(device.id, "not-the-token") is None


def test_an_unknown_device_does_not(gateway):
    assert gateway.authenticate("dev_nope", "anything") is None


def test_revoking_one_device_leaves_the_others_working(gateway):
    """The property the shared token could not provide."""

    _, first = pair(gateway, "phone")
    _, second = pair(gateway, "kitchen box")

    assert gateway.revoke(first.id)

    assert gateway.authenticate(first.id, first.token) is None
    assert gateway.authenticate(second.id, second.token) is not None


def test_a_revoked_device_is_not_listed(gateway):
    _, device = pair(gateway)
    gateway.revoke(device.id)

    assert gateway.devices() == []
    assert len(gateway.devices(include_revoked=True)) == 1


def test_revoking_destroys_the_credential(gateway):
    _, device = pair(gateway)
    gateway.revoke(device.id)

    assert gateway.get(device.id).token == ""


def test_forgetting_removes_it_entirely(gateway):
    _, device = pair(gateway)

    assert gateway.forget(device.id)
    assert gateway.get(device.id) is None


def test_a_token_is_never_serialised_to_a_client(gateway):
    _, device = pair(gateway)

    assert "token" not in device.to_dict()
    assert "token" not in json.dumps(gateway.status())


# --------------------------------------------------------------------------
# Presence is not pairing
# --------------------------------------------------------------------------

def test_a_freshly_paired_device_is_absent_until_it_speaks(gateway):
    """Enrolled-and-unplugged is a different state from never-enrolled."""

    _, device = pair(gateway)

    assert not device.present


def test_a_heartbeat_makes_it_present(gateway):
    _, device = pair(gateway)

    gateway.touch(device.id)

    assert gateway.get(device.id).present


def test_a_silent_device_becomes_absent(gateway):
    _, device = pair(gateway)
    gateway.touch(device.id)
    gateway.get(device.id).last_seen = "2020-01-01T00:00:00+00:00"

    assert not gateway.get(device.id).present


def test_a_revoked_device_is_never_present(gateway):
    _, device = pair(gateway)
    gateway.touch(device.id)
    gateway.revoke(device.id)

    assert not gateway.get(device.id).present


def test_a_corrupt_timestamp_reads_as_absent(gateway):
    _, device = pair(gateway)
    gateway.get(device.id).last_seen = "not a date"

    assert not gateway.get(device.id).present


# --------------------------------------------------------------------------
# Commands to a device
# --------------------------------------------------------------------------

def test_a_command_waits_until_the_device_collects_it(gateway):
    """A display command that vanishes while the TV sleeps never worked."""

    _, device = pair(gateway)
    gateway.send(device.id, "display", {"text": "hello"})

    commands = gateway.drain(device.id)

    assert commands[0]["command"] == "display"
    assert commands[0]["payload"]["text"] == "hello"


def test_draining_clears_the_queue(gateway):
    _, device = pair(gateway)
    gateway.send(device.id, "display", {})
    gateway.drain(device.id)

    assert gateway.drain(device.id) == []


def test_the_queue_is_bounded(gateway):
    """A device that never collects must not grow the core's memory forever."""

    _, device = pair(gateway)
    for index in range(500):
        gateway.send(device.id, "display", {"n": index})

    assert len(gateway.drain(device.id)) <= 50


def test_a_command_to_an_unknown_device_is_refused(gateway):
    assert not gateway.send("dev_nope", "display", {})


def test_a_command_to_a_revoked_device_is_refused(gateway):
    _, device = pair(gateway)
    gateway.revoke(device.id)

    assert not gateway.send(device.id, "display", {})


def test_broadcast_can_target_a_capability(gateway):
    """Do not send a display command to something with no screen."""

    _, screen = pair(gateway, "tv", capabilities=["display"])
    _, speaker = pair(gateway, "speaker", capabilities=["audio_out"])

    sent = gateway.broadcast("display", {"text": "x"}, capability="display")

    assert sent == 1
    assert gateway.drain(screen.id)
    assert gateway.drain(speaker.id) == []


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_devices_survive_a_restart(tmp_path):
    first = DeviceGateway(tmp_path / "devices.json")
    _, device = pair(first)

    reloaded = DeviceGateway(tmp_path / "devices.json")

    assert reloaded.authenticate(device.id, device.token) is not None


def test_a_corrupt_registry_does_not_crash(tmp_path):
    (tmp_path / "devices.json").write_text("{not json", encoding="utf-8")

    assert DeviceGateway(tmp_path / "devices.json").devices() == []


def test_pairing_requests_do_not_survive_a_restart(tmp_path):
    """They expire in five minutes; persisting them would outlive their point."""

    first = DeviceGateway(tmp_path / "devices.json")
    first.request_pairing("box")

    assert DeviceGateway(tmp_path / "devices.json").pending() == []


# --------------------------------------------------------------------------
# Over HTTP
# --------------------------------------------------------------------------

class StubKernel:
    def __init__(self, root):
        self.state_root = root
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        raise RuntimeError("not needed")


@pytest.fixture()
def server(tmp_path):
    core = JarvisCore(kernel=StubKernel(tmp_path))
    instance = JarvisHTTPServer(core, port=0, token="core-token")
    instance.start()
    yield instance
    instance.stop()


def call(server, path, body, *, token="core-token", device=""):
    headers = {"Content-Type": "application/json", "X-Jarvis-Token": token}
    if device:
        headers["X-Jarvis-Device"] = device
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode())


def test_the_whole_pairing_flow_over_http(server):
    asked = call(server, "/api/device/pair", {"name": "kitchen box", "kind": "box"})
    approved = call(server, "/api/device/approve", {"code": asked["code"]})
    collected = call(server, "/api/device/collect", {"code": asked["code"]})

    assert approved["ok"]
    assert collected["status"] == "paired"
    assert collected["token"]


def test_a_device_credential_authenticates_the_api(server):
    asked = call(server, "/api/device/pair", {"name": "box"})
    call(server, "/api/device/approve", {"code": asked["code"]})
    collected = call(server, "/api/device/collect", {"code": asked["code"]})

    # No core token at all from here: the device stands on its own credential.
    status = call(
        server, "/api/status", {}, token=collected["token"], device=collected["device_id"]
    )

    assert status["persona"] == "Jarvis"


def test_a_revoked_device_loses_access(server):
    asked = call(server, "/api/device/pair", {"name": "box"})
    call(server, "/api/device/approve", {"code": asked["code"]})
    collected = call(server, "/api/device/collect", {"code": asked["code"]})
    call(server, "/api/device/revoke", {"device_id": collected["device_id"]})

    with pytest.raises(urllib.error.HTTPError) as caught:
        call(server, "/api/status", {}, token=collected["token"], device=collected["device_id"])

    assert caught.value.code == 401


def test_a_heartbeat_returns_state_and_pending_commands(server):
    asked = call(server, "/api/device/pair", {"name": "tv", "capabilities": ["display"]})
    call(server, "/api/device/approve", {"code": asked["code"]})
    collected = call(server, "/api/device/collect", {"code": asked["code"]})
    device_id = collected["device_id"]
    call(server, "/api/device/display", {"device_id": device_id, "command": "display", "payload": {"text": "hi"}})

    beat = call(server, "/api/device/heartbeat", {"device_id": device_id},
                token=collected["token"], device=device_id)

    assert beat["ok"]
    assert beat["state"]["state"] == "idle"
    assert beat["commands"][0]["payload"]["text"] == "hi"


def test_pairing_still_requires_the_core_token(server):
    """A device with no credential may ask, but only from something trusted."""

    with pytest.raises(urllib.error.HTTPError) as caught:
        call(server, "/api/device/pair", {"name": "intruder"}, token="wrong")

    assert caught.value.code == 401
