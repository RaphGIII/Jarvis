"""Ollama as a managed service: no boot path may hang, spawn twice, or storm.

The live defect: the frozen supervisor crashed in preflight when Ollama was
not running (``AttributeError: _ollama_exe`` after a cached binary check),
and its crash handler waited in a modal message box -- a "hang".  These
tests drive the lifecycle with fakes for the HTTP probe, the process probe
and the spawner, and a fake clock, so every timeout is exercised in
milliseconds and every claim below is about the code, not about this
machine's Ollama.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from zeus_supervisor.config import SupervisorConfig
from zeus_supervisor.ollama import OllamaService, OllamaState
from zeus_supervisor.preflight import Check, Preflight, PreflightCache, find_ollama_exe


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeProcess:
    def __init__(self, pid: int = 4242, *, exits_with: int | None = None, after: int = 0) -> None:
        self.pid = pid
        self.exits_with = exits_with
        self.after = after
        self.polls = 0
        self.returncode: int | None = None

    def poll(self) -> int | None:
        self.polls += 1
        if self.exits_with is not None and self.polls > self.after:
            self.returncode = self.exits_with
            return self.exits_with
        return None


class World:
    """The machine: whether the API answers, whether a server process exists, what spawning does."""

    def __init__(self, *, api_up: bool = False, process_exists: bool = False, ready_after_spawn: float | None = 2.0,
                 spawn_error: Exception | None = None, exe: str = r"C:\fake\ollama.exe") -> None:
        self.api_up = api_up
        self.process_exists = process_exists
        self.ready_after_spawn = ready_after_spawn
        self.spawn_error = spawn_error
        self.exe = exe
        self.clock = FakeClock()
        self.spawns: list[tuple[list[str], dict[str, str]]] = []
        self.spawned_at: float | None = None
        self.process: FakeProcess | None = None
        self.probes = 0

    def probe(self, url: str, timeout: float) -> str:
        self.probes += 1
        if self.api_up:
            return "0.30.10"
        if self.spawned_at is not None and self.ready_after_spawn is not None and self.clock() - self.spawned_at >= self.ready_after_spawn:
            self.api_up = True
            return "0.30.10"
        raise ConnectionRefusedError("refused")

    def port(self, url: str) -> bool:
        return self.api_up

    def processes(self) -> bool:
        spawned = getattr(self, "spawned_process", None)
        spawned_alive = spawned is not None and spawned.poll() is None
        return self.process_exists or self.api_up or spawned_alive

    def spawn(self, command: list[str], env: dict[str, str]) -> FakeProcess:
        if self.spawn_error is not None:
            raise self.spawn_error
        self.spawns.append((command, env))
        self.spawned_at = self.clock()
        self.process = self.process or FakeProcess()
        self.spawned_process = self.process
        return self.process

    def service(self, **kw) -> OllamaService:
        defaults = dict(url="http://127.0.0.1:11434", exe_finder=lambda: self.exe, models_dir=r"D:\store", start_timeout=45,
                        spawn_cooldown=30, max_spawns=3, spawn_window=600, probe=self.probe, port_probe=self.port,
                        process_probe=self.processes, spawner=self.spawn, clock=self.clock, sleep=self.clock.sleep)
        defaults.update(kw)
        return OllamaService(**defaults)


# --------------------------------------------------------------------------
# 1-7: the lifecycle
# --------------------------------------------------------------------------

def test_1_ollama_already_running_means_no_spawn():
    world = World(api_up=True)
    status = world.service().ensure()
    assert status.state is OllamaState.RUNNING and status.version == "0.30.10"
    assert world.spawns == [] and status.started_by_supervisor is False


def test_2_ollama_stopped_spawns_exactly_once_detached_with_the_configured_store():
    world = World()
    status = world.service().ensure()
    assert status.state is OllamaState.RUNNING and status.started_by_supervisor and status.pid == 4242
    assert len(world.spawns) == 1
    command, env = world.spawns[0]
    assert command == [r"C:\fake\ollama.exe", "serve"]
    assert env["OLLAMA_MODELS"] == r"D:\store"


def test_3_a_spawned_server_that_keeps_running_lets_preflight_continue():
    """The child is long-running: the service never waits on it, only on the API."""

    world = World(ready_after_spawn=1.0)
    service = world.service()
    status = service.ensure()
    assert status.ok
    assert world.process is not None and world.process.returncode is None, "ollama serve is still running"
    # and a second ensure() is a no-op
    assert service.ensure().ok and len(world.spawns) == 1


def test_4_delayed_readiness_is_polled_within_the_bound():
    world = World(ready_after_spawn=20.0)
    before = world.clock()
    status = world.service(start_timeout=45).ensure()
    assert status.ok
    assert 20.0 <= world.clock() - before < 45.0


def test_5_never_ready_times_out_and_reports_failed_without_hanging():
    world = World(ready_after_spawn=None)
    before = world.clock()
    status = world.service(start_timeout=45).ensure()
    assert status.state is OllamaState.FAILED, status
    assert "did not answer" in status.reason and "45s" in status.reason
    assert world.clock() - before <= 46.0, "bounded by the timeout"
    assert len(world.spawns) == 1


def test_5b_a_child_that_dies_early_is_failed_with_its_exit_code():
    world = World(ready_after_spawn=None)
    world.process = FakeProcess(exits_with=1, after=2)
    status = world.service().ensure()
    assert status.state is OllamaState.FAILED and "exited with code 1" in status.reason


def test_6_spawn_failure_is_a_degraded_state_not_an_exception():
    world = World(spawn_error=OSError("access denied"))
    status = world.service().ensure()
    assert status.state is OllamaState.FAILED and "could not start ollama serve" in status.reason
    world2 = World(exe="")
    status2 = world2.service().ensure()
    assert status2.state is OllamaState.MISSING and world2.spawns == []


def test_7_two_simultaneous_startup_checks_spawn_once():
    world = World(ready_after_spawn=0.5)
    service = world.service()
    results: list = []

    def go() -> None:
        results.append(service.ensure())

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(r.ok for r in results) and len(results) == 6
    assert len(world.spawns) == 1


def test_7b_an_existing_server_process_is_waited_for_not_duplicated():
    world = World(process_exists=True, ready_after_spawn=None)
    world.api_up = False
    service = world.service(start_timeout=5)
    status = service.ensure()
    assert world.spawns == [], "a process that exists is never doubled"
    assert status.state is OllamaState.FAILED and "did not answer" in status.reason


def test_7c_the_spawn_budget_and_cooldown_stop_a_process_storm():
    world = World(ready_after_spawn=None)
    service = world.service(start_timeout=5, spawn_cooldown=30, max_spawns=3, spawn_window=600)
    for _ in range(3):
        world.process = FakeProcess(exits_with=1, after=1)
        service.ensure()
        world.clock.sleep(31)
    world.process = FakeProcess(exits_with=1, after=1)
    status = service.ensure()
    assert len(world.spawns) == 3
    assert status.state is OllamaState.FAILED and "never stayed up" in status.reason
    # cool-down: right after a spawn nothing is spawned again
    world2 = World(ready_after_spawn=None)
    service2 = world2.service(start_timeout=2, spawn_cooldown=30)
    world2.process = FakeProcess(exits_with=1, after=1)
    service2.ensure()
    world2.process = FakeProcess(exits_with=1, after=1)
    service2.ensure()
    assert len(world2.spawns) == 1


def test_status_words_carry_the_reason():
    assert World(api_up=True).service().status().state is OllamaState.RUNNING
    assert World(process_exists=True).service().status().state is OllamaState.STARTING
    assert World(exe="").service().status().state is OllamaState.MISSING
    world = World()
    assert world.service().status().state is OllamaState.UNAVAILABLE
    service = World(spawn_error=OSError("x")).service()
    service.ensure()
    assert service.status().state is OllamaState.FAILED and "could not start" in service.status().reason


# --------------------------------------------------------------------------
# the regression: a cached binary check must not break the server check
# --------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "Jarvis"
    (root / "jarvis").mkdir(parents=True)
    (root / "service").mkdir()
    (root / "jarvis" / "serve.py").write_text("# stub", encoding="utf-8")
    (root / "service" / "core.py").write_text("# stub", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"], cwd=root, check=True)
    return root


def test_regression_cached_binary_check_with_ollama_down_does_not_raise(repo: Path, tmp_path: Path):
    config = SupervisorConfig(repository=repo, python="python", ollama_url="http://127.0.0.1:1", ollama_start_timeout=1)
    cache = PreflightCache(tmp_path / "cache.json")
    fingerprint = cache.fingerprint(config)
    cache.save(fingerprint, [Check("ollama.binary", True, r"C:\fake\ollama.exe (cached) (cached)"), Check("python", True, "3.14")])
    world = World(ready_after_spawn=None, exe=r"C:\fake\ollama.exe")
    pre = Preflight(config, log=lambda _m: None, cache=cache, ollama=world.service(start_timeout=1))
    report = pre.run(generation=False)  # must not raise
    names = {c.name: c for c in report.checks}
    assert names["ollama.binary"].detail.count("(cached)") == 1, "the cached suffix does not grow"
    assert names["ollama.server"].ok is False and names["ollama.server"].detail.startswith("FAILED:")
    assert report.ollama_state == "FAILED" and report.blocker.name == "ollama.server"
    assert len(world.spawns) == 1, "the cached check left the boot able to start Ollama"


def test_a_raising_check_becomes_a_failed_check_not_a_crash(repo: Path):
    config = SupervisorConfig(repository=repo, python="python")
    pre = Preflight(config, log=lambda _m: None)

    def boom(report):
        raise RuntimeError("kaboom")

    boom.__name__ = "_check_ollama_server"
    pre._check_ollama_server = boom  # type: ignore[assignment]
    report = pre.run(generation=False)
    server = next(c for c in report.checks if c.name == "ollama.server")
    assert server.ok is False and "kaboom" in server.detail


def test_diagnostics_observe_without_starting_ollama(repo: Path):
    config = SupervisorConfig(repository=repo, python="python")
    world = World()
    pre = Preflight(config, log=lambda _m: None, ollama=world.service())
    report = pre.run(generation=False, start_services=False)
    assert world.spawns == []
    assert report.ollama_state == "UNAVAILABLE"


def test_find_ollama_exe_prefers_an_explicit_existing_path(tmp_path: Path):
    fake = tmp_path / "ollama.exe"
    fake.write_bytes(b"MZ")
    assert find_ollama_exe(str(fake)) == str(fake)
    assert isinstance(find_ollama_exe(str(tmp_path / "missing.exe")), str)


def test_config_env_overrides_exist_for_failure_path_testing(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEUS_OLLAMA_URL", "http://127.0.0.1:11999")
    monkeypatch.setenv("ZEUS_OLLAMA_EXE", r"C:\nowhere\ollama.exe")
    monkeypatch.setenv("ZEUS_OLLAMA_START_TIMEOUT", "7")
    config = SupervisorConfig.load(repo)
    assert config.ollama_url == "http://127.0.0.1:11999" and config.ollama_exe == r"C:\nowhere\ollama.exe" and config.ollama_start_timeout == 7.0


# --------------------------------------------------------------------------
# 8-9: the supervisor -- hold without deadlock, recovery on restart
# --------------------------------------------------------------------------

def _supervisor(repo: Path, **cfg):
    from zeus_supervisor.supervisor import Supervisor

    config = SupervisorConfig(repository=repo, python="python", port=0, open_browser=False, voice=False, hold_retry_interval=0.05, **cfg)
    return Supervisor(config, log=None)


def test_8_a_held_supervisor_stops_on_a_shutdown_request_and_does_not_deadlock(repo: Path):
    sup = _supervisor(repo)
    sup.status_page.start = lambda: True  # type: ignore[assignment]
    sup.status_page.stop = lambda: None  # type: ignore[assignment]
    sup.control.request("shutdown", reason="test")
    started = time.monotonic()
    assert sup._hold(retry=lambda: False) is False
    assert time.monotonic() - started < 5
    assert sup.phase == "stopped"


def test_8b_the_status_page_quit_route_ends_a_held_supervisor(repo: Path):
    sup = _supervisor(repo)
    # a real status page on an ephemeral port
    sup.status_page.port = 0
    assert sup.status_page.start()
    port = sup.status_page._server.server_address[1]
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", data=b"{}", method="POST", headers={"X-Jarvis-Token": "wrong"})
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", data=b"{}", method="POST", headers={"X-Jarvis-Token": sup.token})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read())["stopping"] is True
        assert sup._stop.is_set()
        assert sup._hold(retry=None) is False
    finally:
        sup.status_page.stop()


def test_9_hold_retries_and_the_boot_continues_when_ollama_comes_back(repo: Path):
    sup = _supervisor(repo)
    sup.status_page.start = lambda: True  # type: ignore[assignment]
    sup.status_page.stop = lambda: None  # type: ignore[assignment]
    attempts = []

    def retry() -> bool:
        attempts.append(1)
        return len(attempts) >= 3

    started = time.monotonic()
    assert sup._hold(retry=retry) is True
    assert len(attempts) == 3 and time.monotonic() - started < 5


def test_9b_the_watchdog_recovers_a_dead_ollama_once_at_a_time(repo: Path):
    sup = _supervisor(repo, ollama_watch_interval=0)
    world = World(ready_after_spawn=0.1)
    world.api_up = True
    sup.ollama = world.service(start_timeout=5)
    sup._ollama_was_running = True
    sup._watch_ollama()
    assert world.spawns == [], "a healthy Ollama is left alone"
    world.api_up = False
    world.spawned_at = None
    sup._watch_ollama()
    sup._ollama_recovery.join(timeout=10)
    assert len(world.spawns) == 1 and sup.ollama.last.ok
    # while a recovery thread is alive, a second watch does not start another
    sup._ollama_recovery = threading.Thread(target=lambda: time.sleep(0.3))
    sup._ollama_recovery.start()
    world.api_up = False
    sup._watch_ollama()
    assert len(world.spawns) == 1
    sup._ollama_recovery.join()


def test_preflight_failure_holds_and_recovers_instead_of_crashing(repo: Path, monkeypatch: pytest.MonkeyPatch):
    """The boot path end to end: preflight fails (Ollama down), the supervisor holds,
    Ollama appears, the retry passes, and the loop proceeds to start."""

    sup = _supervisor(repo)
    sup.status_page.start = lambda: True  # type: ignore[assignment]
    sup.status_page.stop = lambda: None  # type: ignore[assignment]
    outcomes = iter([False, False, True])
    seen = []
    monkeypatch.setattr(sup, "_preflight", lambda: (seen.append(1), next(outcomes))[1])
    monkeypatch.setattr(sup, "_open_window_early", lambda: None)
    monkeypatch.setattr(sup, "_start_and_verify", lambda pending: "stopped")
    code = sup._main_loop()
    assert code == 0 and len(seen) == 3
    assert sup.phase in {"starting", "stopped"}
