from brain.router import (
    BrainRouter,
    BrainTier,
    RemoteBrainUnavailable,
)


class FakeBrain:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return f"ANSWER:{prompt}"


def test_normal_chat_routes_fast_local():
    fast = FakeBrain()
    router = BrainRouter(fast_brain=fast, remote_enabled=False)

    decision = router.route("Erkläre mir die Photosynthese.")

    assert decision.tier is BrainTier.FAST_LOCAL


def test_build_goal_routes_remote():
    fast = FakeBrain()
    router = BrainRouter(fast_brain=fast, remote_enabled=False)

    decision = router.route(
        "Implementiere ein neues Feature in diesem Repository."
    )

    assert decision.tier is BrainTier.BUILD_REMOTE


def test_fast_local_executes_local_brain():
    fast = FakeBrain()
    router = BrainRouter(fast_brain=fast, remote_enabled=False)

    answer, decision = router.respond("Hallo Jarvis")

    assert answer == "ANSWER:Hallo Jarvis"
    assert decision.tier is BrainTier.FAST_LOCAL
    assert len(fast.calls) == 1


def test_remote_task_does_not_silently_use_local_or_cloud():
    fast = FakeBrain()
    router = BrainRouter(fast_brain=fast, remote_enabled=False)

    try:
        router.respond(
            "Implementiere ein neues Feature in diesem Repository."
        )
    except RemoteBrainUnavailable:
        pass
    else:
        raise AssertionError("Expected RemoteBrainUnavailable")

    assert fast.calls == []


def test_enabled_remote_dispatches_to_remote_brain():
    fast = FakeBrain()
    remote = FakeBrain()

    router = BrainRouter(
        fast_brain=fast,
        remote_brain=remote,
        remote_enabled=True,
    )

    answer, decision = router.respond(
        "Implementiere ein neues Feature in diesem Repository."
    )

    assert decision.tier is BrainTier.BUILD_REMOTE
    assert answer.startswith("ANSWER:")
    assert fast.calls == []
    assert len(remote.calls) == 1
