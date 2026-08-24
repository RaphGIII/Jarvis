"""The core service, its event stream, and the HTTP surface.

The properties worth testing here are the ones that only show up under a real
client: a browser tab that stops draining must not stall an autonomous build, a
page opened mid-mission must not render "idle", and the token must actually be
required rather than merely present.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from service.core import JarvisCore
from service.events import Event, EventBus, EventType
from service.http import JarvisHTTPServer
from service.state import JarvisState, StateMachine


# --------------------------------------------------------------------------
# Event bus
# --------------------------------------------------------------------------

def test_a_subscriber_receives_what_is_published():
    bus = EventBus()
    with bus.subscribe() as subscription:
        bus.publish(EventType.TOKEN, {"text": "hello"})

        event = subscription.get(timeout=1)

    assert event is not None
    assert event.type is EventType.TOKEN
    assert event.payload["text"] == "hello"


def test_every_event_gets_an_increasing_sequence_number():
    bus = EventBus()

    first = bus.publish(EventType.STATE, {})
    second = bus.publish(EventType.STATE, {})

    assert second.seq == first.seq + 1


def test_two_subscribers_both_receive_the_same_event():
    bus = EventBus()
    with bus.subscribe() as a, bus.subscribe() as b:
        bus.publish(EventType.MESSAGE, {"text": "x"})

        assert a.get(timeout=1).payload["text"] == "x"
        assert b.get(timeout=1).payload["text"] == "x"


def test_a_client_that_stops_draining_cannot_stall_the_core():
    """A paused browser tab must not be able to freeze an autonomous build."""

    bus = EventBus()
    with bus.subscribe(maxsize=5) as stalled:
        for index in range(500):
            bus.publish(EventType.TOKEN, {"n": index})

        assert stalled.dropped > 0, "the slow subscriber must lose events, not block"
        assert bus.sequence == 500, "publishing must have completed regardless"


def test_the_oldest_events_are_the_ones_dropped():
    bus = EventBus()
    with bus.subscribe(maxsize=3) as subscription:
        for index in range(10):
            bus.publish(EventType.TOKEN, {"n": index})

        remaining = [event.payload["n"] for event in subscription.drain()]

    assert remaining == [7, 8, 9], f"kept the wrong window: {remaining}"


def test_a_late_client_is_caught_up_from_the_replay_buffer():
    """A page opened mid-mission must not render a blank screen."""

    bus = EventBus()
    bus.publish(EventType.STATE, {"state": "working"})
    bus.publish(EventType.TOKEN, {"text": "already happening"})

    with bus.subscribe(replay=True) as late:
        events = late.drain()

    assert [event.payload for event in events] == [
        {"state": "working"},
        {"text": "already happening"},
    ]


def test_replay_can_be_declined():
    bus = EventBus()
    bus.publish(EventType.STATE, {})

    with bus.subscribe(replay=False) as subscription:
        assert subscription.drain() == []


def test_the_replay_buffer_is_bounded():
    bus = EventBus(replay=5)
    for index in range(50):
        bus.publish(EventType.TOKEN, {"n": index})

    assert len(bus.history(limit=1000)) == 5


def test_history_can_resume_from_a_sequence_number():
    bus = EventBus()
    for index in range(10):
        bus.publish(EventType.TOKEN, {"n": index})

    resumed = bus.history(since=7)

    assert [event.payload["n"] for event in resumed] == [7, 8, 9]


def test_unsubscribing_stops_delivery():
    bus = EventBus()
    subscription = bus.subscribe()
    subscription.close()

    bus.publish(EventType.TOKEN, {})

    assert bus.subscriber_count == 0


def test_a_broken_watcher_cannot_break_publishing():
    bus = EventBus()
    bus.watch(lambda event: (_ for _ in ()).throw(RuntimeError("boom")))
    seen = []
    bus.watch(seen.append)

    bus.publish(EventType.STATE, {})

    assert len(seen) == 1, "the second watcher must still have run"


def test_publishing_is_thread_safe():
    bus = EventBus()

    def spam():
        for _ in range(200):
            bus.publish(EventType.TOKEN, {})

    threads = [threading.Thread(target=spam) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert bus.sequence == 800


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

def test_state_changes_notify():
    seen = []
    machine = StateMachine(on_change=seen.append)

    machine.set(JarvisState.THINKING, detail="working it out")

    assert seen[-1].state is JarvisState.THINKING
    assert seen[-1].detail == "working it out"


def test_an_identical_state_does_not_notify_again():
    seen = []
    machine = StateMachine(on_change=seen.append)
    machine.set(JarvisState.THINKING, detail="x")

    machine.set(JarvisState.THINKING, detail="x")

    assert len(seen) == 1


def test_the_start_time_survives_a_detail_update():
    """"Thinking for 40 seconds" must stay true across progress lines."""

    machine = StateMachine()
    machine.set(JarvisState.WORKING, detail="step 1")
    original = machine.snapshot.since

    machine.set(JarvisState.WORKING, detail="step 2")

    assert machine.snapshot.since == original


def test_a_real_state_change_resets_the_start_time():
    machine = StateMachine()
    machine.set(JarvisState.WORKING)
    first = machine.snapshot.since
    time.sleep(0.01)

    machine.set(JarvisState.SPEAKING)

    assert machine.snapshot.since != first


def test_barge_in_is_possible_while_speaking():
    """The most important thing barge-in must support."""

    assert JarvisState.SPEAKING.accepts_speech


def test_speech_is_ignored_only_when_broken_or_offline():
    assert not JarvisState.OFFLINE.accepts_speech
    assert not JarvisState.ERROR.accepts_speech


def test_busy_states_are_the_ones_that_take_time():
    assert JarvisState.THINKING.busy and JarvisState.CODING.busy
    assert not JarvisState.IDLE.busy and not JarvisState.WAITING.busy


def test_the_context_manager_returns_to_idle():
    machine = StateMachine()

    with machine.busy_with(JarvisState.RESEARCHING, "reading docs"):
        assert machine.state is JarvisState.RESEARCHING

    assert machine.state is JarvisState.IDLE


def test_an_exception_inside_the_context_manager_lands_in_error():
    machine = StateMachine()

    with pytest.raises(ValueError):
        with machine.busy_with(JarvisState.CODING):
            raise ValueError("no")

    assert machine.state is JarvisState.ERROR


def test_every_state_the_ui_knows_about_exists():
    """The UI's animation table and the server's vocabulary must agree."""

    required = {
        "idle", "listening", "transcribing", "thinking", "speaking",
        "working", "researching", "coding", "waiting", "error", "offline",
    }

    assert {state.value for state in JarvisState} == required


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

class StubProvider:
    def __init__(self, chunks=("Hel", "lo")):
        self.chunks = chunks

    def generate_stream(self, prompt, **_):
        yield from self.chunks


class StubKernel:
    def __init__(self):
        self.catalog = type("C", (), {"get": staticmethod(lambda tier: type("S", (), {"model": "stub"})())})()
        self._provider = StubProvider()

    def provider(self, tier):
        return self._provider


def test_constructing_the_core_touches_nothing():
    """Starting the UI must not depend on a model being warm."""

    core = JarvisCore()

    assert core.state.state is JarvisState.IDLE
    assert core._kernel is None, "the kernel must not be built eagerly"


def test_a_message_streams_tokens_then_a_final_message():
    core = JarvisCore(kernel=StubKernel())
    with core.bus.subscribe(replay=False) as subscription:
        core.send_message("hi")
        deadline = time.time() + 5
        events = []
        while time.time() < deadline:
            event = subscription.get(timeout=0.2)
            if event:
                events.append(event)
                if event.type is EventType.MESSAGE:
                    break

    kinds = [event.type for event in events]
    assert EventType.USER_MESSAGE in kinds
    assert EventType.TOKEN in kinds
    assert EventType.MESSAGE in kinds

    final = next(event for event in events if event.type is EventType.MESSAGE)
    assert final.payload["text"] == "Hello"


def test_an_empty_message_is_rejected_without_starting_work():
    core = JarvisCore(kernel=StubKernel())

    assert core.send_message("   ")["ok"] is False


def test_the_backend_is_recorded_but_not_part_of_the_persona():
    """Identity must survive a backend change."""

    core = JarvisCore(kernel=StubKernel())
    core.send_message("hi")
    deadline = time.time() + 5
    while time.time() < deadline and len(core.history) < 2:
        time.sleep(0.05)

    reply = core.history[-1]
    assert reply.backend == "stub"
    assert "Qwen" not in reply.text


def test_the_prompt_states_the_identity_rather_than_a_costume():
    core = JarvisCore(kernel=StubKernel())

    prompt = core._compose_prompt("who are you?")

    from core.identity import current

    # Asserted against the configured name rather than a literal, so this
    # survives the next rename as well as this one.
    assert f"You are {current().assistant_name}" in prompt
    assert "do not describe yourself as a language model" in prompt


def test_diagnostics_survive_every_subsystem_being_broken():
    """The diagnostics view is most needed exactly when things are broken."""

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("everything is on fire")

    core = JarvisCore(kernel=Broken())

    payload = core.diagnostics()

    assert "error" in payload["kernel"]
    assert payload["state"]["state"] == "idle"


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

@pytest.fixture()
def server():
    core = JarvisCore(kernel=StubKernel())
    instance = JarvisHTTPServer(core, port=0, token="test-token")
    instance.start()
    yield instance
    instance.stop()


def fetch(server, path, body=None, token="test-token"):
    url = f"http://{server.host}:{server.port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if data else "GET",
        headers={"Content-Type": "application/json", "X-Jarvis-Token": token},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode())


def test_the_api_requires_the_token(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch(server, "/api/status", token="wrong")

    assert caught.value.code == 401


def test_status_is_served_with_the_token(server):
    status, payload = fetch(server, "/api/status")

    assert status == 200
    from core.identity import current

    assert payload["persona"] == current().assistant_name


def test_the_ui_is_served(server):
    url = f"http://{server.host}:{server.port}/?token=test-token"
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode()

    assert "<canvas id=\"eye\"" in body
    assert "__JARVIS_TOKEN__" not in body, "the token placeholder must be substituted"
    assert "test-token" in body


def test_a_path_escaping_the_ui_root_is_refused(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(
            f"http://{server.host}:{server.port}/../core/kernel.py", timeout=10
        )

    assert caught.value.code in {403, 404}


def test_an_unknown_endpoint_says_so(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        fetch(server, "/api/nonsense", {})

    assert caught.value.code == 404


def test_a_non_json_body_is_rejected(server):
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}/api/message",
        data=b"not json",
        method="POST",
        headers={"Content-Type": "application/json", "X-Jarvis-Token": "test-token"},
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)

    assert caught.value.code == 400


def test_the_event_stream_sends_the_current_state_immediately(server):
    """A page opened mid-mission renders the truth, not 'idle'."""

    server.core.state.set(JarvisState.WORKING, detail="building something")

    url = f"http://{server.host}:{server.port}/events?token=test-token"
    collected = ""
    # Read line by line: a fixed-size read blocks until the buffer fills, and
    # the opening burst is deliberately small.
    with urllib.request.urlopen(url, timeout=10) as response:
        for _ in range(20):
            line = response.readline()
            if not line:
                break
            collected += line.decode("utf-8", errors="replace")
            if "working" in collected:
                break

    assert "event: state" in collected
    assert "working" in collected


def test_the_event_stream_requires_the_token(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"http://{server.host}:{server.port}/events", timeout=5)

    assert caught.value.code == 401


def test_a_posted_message_reaches_the_core(server):
    status, payload = fetch(server, "/api/message", {"text": "hello there"})

    assert status == 200 and payload["ok"] is True
    deadline = time.time() + 5
    while time.time() < deadline and not server.core.history:
        time.sleep(0.05)
    assert server.core.history[0].text == "hello there"


def test_the_url_carries_the_token(server):
    assert "token=test-token" in server.url


def test_tv_mode_is_the_same_page_not_a_second_one(server):
    """A separate TV frontend would drift out of step with this one, and the
    first divergence would be a bug nobody notices until it is on the wall."""

    url = f"http://{server.host}:{server.port}/?token=test-token&tv=1"
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode()

    assert "body.tv" in body, "the kiosk styles live in the same document"
    assert 'id="eye"' in body, "and so does everything else"
