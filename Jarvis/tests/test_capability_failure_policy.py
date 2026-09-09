"""What a failed capability run means, and what ZEUS does about it.

Two live production failures on 2026-09-09 are pinned here.

The owner asked for the SHA-256 checksum of a real file on their Desktop.
``local.berechne.sha.256_pruefsumme`` was found, was correct, and never got to
run: ZEUS's own path resolver refused the file for lying "outside the workspace
D:\\...\\data\\jarvis\\workspace" -- a rule written for what a capability may
WRITE, applied to what it may READ.  The mission failed three steps in a row,
the registry recorded none of them, and the only thing that then happened was a
background INSIGHT: "Fähigkeit local.berechne.sha.256_pruefsumme ist
ausgefallen", quoting an ``AttributeError`` about ``hex_digest`` that Codex had
repaired the day before and that no longer existed in the installed code.

So: reads and writes are separated, failures are classified before they are
counted, and a real implementation defect goes to the repair path rather than
to a notice board.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from capabilities.models import CapabilityHealth, CapabilityLifecycle, CapabilityManifest
from core.identity import Identity
from core.kernel import JarvisKernel, KernelConfig
from runtime.failure_kind import DEFECT, INPUT, PERMISSION, TRANSIENT, UNSUPPORTED, classify_failure
from runtime.paths import (
    PathError,
    is_write_key,
    resolve_argument_path,
    resolve_read_path,
    resolve_workspace_path,
)
from service.core import JarvisCore
from service.intent import absolute_paths


# ==========================================================================
# The filesystem policy: reading a file the owner named is not a write
# ==========================================================================


def test_a_write_target_still_may_not_leave_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "elsewhere" / "report.txt"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(PathError) as exc:
        resolve_argument_path(workspace, "output_path", str(outside))
    assert "outside the workspace" in str(exc.value)


def test_a_read_input_may_be_any_file_the_owner_named(tmp_path: Path) -> None:
    """The live refusal, inverted: this is the case that failed in production."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    desktop = tmp_path / "Users" / "rapha" / "OneDrive" / "Desktop" / "Keine Ahnung Python oder so"
    desktop.mkdir(parents=True)
    target = desktop / "version.txt"
    target.write_text("7.0.1", encoding="utf-8")

    resolved = resolve_argument_path(workspace, "file_path", str(target))
    assert resolved == target
    assert resolved.read_text(encoding="utf-8") == "7.0.1"


def test_which_argument_names_mean_a_write() -> None:
    assert is_write_key("output_path") and is_write_key("destination") and is_write_key("target")
    assert not is_write_key("file_path") and not is_write_key("source") and not is_write_key("path")


def test_a_relative_path_still_means_the_workspace_and_still_cannot_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hi", encoding="utf-8")
    assert resolve_read_path(workspace, "notes.txt") == workspace / "notes.txt"
    with pytest.raises(PathError):
        resolve_read_path(workspace, "../secrets.txt")


def test_a_credential_store_is_refused_for_reads_too(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = tmp_path / "app" / "secrets"
    store.mkdir(parents=True)
    (store / "spotify.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PathError) as exc:
        resolve_read_path(workspace, str(store / "spotify.json"))
    assert "secret store" in str(exc.value)

    key = tmp_path / "id_rsa"
    key.write_text("-----BEGIN", encoding="utf-8")
    with pytest.raises(PathError) as exc:
        resolve_read_path(workspace, str(key))
    assert "credential file" in str(exc.value)


def test_zeus_own_owner_directory_is_not_readable_by_a_capability(tmp_path: Path) -> None:
    """auth.json holds the verifier for the password that promotes code."""

    from runtime.paths import protected_roots

    state = tmp_path / "state"
    workspace = state / "workspace"
    workspace.mkdir(parents=True)
    (state / "owner").mkdir()
    verifier = state / "owner" / "auth.json"
    verifier.write_text('{"kdf": "scrypt"}', encoding="utf-8")

    roots = protected_roots(state)
    with pytest.raises(PathError) as exc:
        resolve_argument_path(workspace, "file_path", str(verifier), denied_roots=roots)
    assert "does not read out" in str(exc.value)

    # And nothing else on the machine became unreadable along with it.
    elsewhere = tmp_path / "notes.txt"
    elsewhere.write_text("plain", encoding="utf-8")
    assert resolve_argument_path(workspace, "file_path", str(elsewhere), denied_roots=roots) == elsewhere


def test_a_read_input_that_does_not_exist_is_named_before_anything_runs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PathError) as exc:
        resolve_read_path(workspace, str(tmp_path / "absent.txt"))
    assert "no such file" in str(exc.value)


def test_a_unicode_path_survives_the_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "Prüfsummen" / "größe_日本.txt"
    target.parent.mkdir()
    target.write_text("ü", encoding="utf-8")
    assert resolve_argument_path(workspace, "file_path", str(target)) == target


# ==========================================================================
# Reading the request: an absolute path IS a named file
# ==========================================================================


def test_the_owner_s_own_sentence_names_its_file(tmp_path: Path) -> None:
    """The exact live wording, whose path contains four spaces and a sentence after it."""

    folder = tmp_path / "Desktop" / "Keine Ahnung Python oder so" / "ffmpeg-7.0.1"
    folder.mkdir(parents=True)
    text = (f"nimm infolgendem ornder {folder} die datei Version und berechne mir "
            f"für diese datei den SHA-256-Hash")
    assert absolute_paths(text) == [str(folder)]


def test_a_path_that_is_not_there_is_still_reported_recognisably() -> None:
    found = absolute_paths(r"hash bitte D:\nirgends\gibt.es.nicht und dann fertig")
    assert found == [r"D:\nirgends\gibt.es.nicht"]


def test_a_sentence_with_no_path_names_nothing() -> None:
    assert absolute_paths("berechne mir irgendeine pruefsumme") == []
    # Quotes alone do not make a path.
    assert absolute_paths('Zeus, sag "hallo" und dann "bis gleich"') == []


# ==========================================================================
# Classifying the failure
# ==========================================================================


@pytest.mark.parametrize(
    "error,kind,health,repairable",
    [
        # The live defect, as the runtime saw it.
        ("AttributeError: '_hashlib.HASH' object has no attribute 'hex_digest'", DEFECT, True, True),
        ("Traceback (most recent call last):\n  File ...\nNameError: x", DEFECT, True, True),
        ("Capability did not produce valid output: JSONDecodeError", DEFECT, True, True),
        # The live refusal. Refusing correctly is not a defect and must not
        # count against the capability's health.
        (r"C:\Users\rapha\Desktop\x lies outside the workspace D:\ws", PERMISSION, False, False),
        ("x lies under a secret store (secrets); reading it is not allowed", PERMISSION, False, False),
        ("PermissionError: [Errno 13] Permission denied", PERMISSION, False, False),
        # Bad or absent input: ask the owner, do not rebuild the capability.
        ("file not found: bericht.txt", INPUT, False, False),
        ("file_path is required", INPUT, False, False),
        ("not a file: D:/some/directory", INPUT, False, False),
        # Outside and temporary.
        ("ConnectionError: connection refused", TRANSIENT, True, False),
        ("the request timed out after 30s", TRANSIENT, True, False),
        # Genuinely out of scope.
        ("NotImplementedError: only zip is supported", UNSUPPORTED, False, False),
    ],
)
def test_a_failure_is_named_before_it_is_counted(error: str, kind: str, health: bool, repairable: bool) -> None:
    result = classify_failure(error)
    assert result.kind == kind, f"{error!r} -> {result.kind} ({result.reason})"
    assert result.counts_against_health is health
    assert result.repairable is repairable


def test_a_silent_crash_is_a_defect() -> None:
    result = classify_failure("", return_code=1)
    assert result.kind == DEFECT and result.repairable


# ==========================================================================
# Health: a DEFECT is BROKEN at once, a refusal is not health at all
# ==========================================================================


def _core(tmp_path: Path) -> JarvisCore:
    class _NoModel:
        def generate(self, *a: Any, **k: Any) -> str:
            raise AssertionError("no model may be called on this path")

    kernel = JarvisKernel(KernelConfig(state_root=tmp_path / "state"))
    kernel.provider = lambda tier: _NoModel()  # type: ignore[assignment]
    kernel.catalog.get = lambda tier: type("S", (), {"model": "stub"})()  # type: ignore[assignment]
    return JarvisCore(kernel=kernel, identity=Identity())


def _install(core: JarvisCore, capability_id: str, body: str, **fields: Any) -> CapabilityManifest:
    root = Path(core.kernel.state_root) / "capabilities" / "installed" / capability_id / "1.0.0"
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(body, encoding="utf-8")
    manifest = CapabilityManifest(
        capability_id,
        fields.pop("description", "A capability installed for this test."),
        lifecycle=CapabilityLifecycle.ACTIVE.value,
        health={"state": "healthy", "health": CapabilityHealth.HEALTHY.value},
        source_location=str(root),
        implementation_path=str(root),
        entrypoint="main.py",
        input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        **fields,
    )
    return core.capabilities.registry.register(manifest)


def test_one_defect_is_enough_to_call_a_capability_broken(tmp_path: Path) -> None:
    """Waiting for a second identical traceback costs a second owner request."""

    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n")
    core.capabilities.registry.note_execution(
        "files.sha256", False, "AttributeError: no attribute 'hex_digest'", kind=DEFECT)
    manifest = core.capabilities.registry.get("files.sha256")
    assert manifest.health_state() is CapabilityHealth.BROKEN
    assert manifest.is_broken()


def test_two_refusals_do_not_make_a_capability_broken(tmp_path: Path) -> None:
    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n")
    for _ in range(3):
        classification = classify_failure("x lies outside the workspace D:\\ws")
        assert not classification.counts_against_health
    assert core.capabilities.registry.get("files.sha256").health_state() is CapabilityHealth.HEALTHY


# ==========================================================================
# The trigger: an INSIGHT may accompany a repair, it may not replace one
# ==========================================================================


def _wire(core: JarvisCore) -> tuple[list[dict[str, Any]], list[str], list[tuple[str, str]]]:
    events: list[dict[str, Any]] = []
    delivered: list[str] = []
    repairs: list[tuple[str, str]] = []
    core.emit = lambda kind, payload, scope="": events.append({"kind": kind, **payload})  # type: ignore[assignment]
    core._deliver = lambda text, **kwargs: delivered.append(text)  # type: ignore[assignment]
    core._start_capability_engineering = (  # type: ignore[assignment]
        lambda goal, text, scope, capability_id="", repair="": repairs.append((capability_id, repair))
    )
    return events, delivered, repairs


def test_a_capability_that_crashes_reaches_codex_and_not_a_notice_board(tmp_path: Path) -> None:
    """The acceptance the live screenshot failed: failure -> BROKEN -> repair."""

    core = _core(tmp_path)
    manifest = _install(
        core, "local.berechne.sha.256_pruefsumme",
        "import hashlib\n"
        "def run(payload):\n"
        "    return {'ok': False, 'error': \"AttributeError: '_hashlib.HASH' object has no attribute 'hex_digest'\"}\n",
        description="Compute the SHA-256 checksum of a file.",
        aliases=["sha-256 pruefsumme einer datei"],
    )
    events, delivered, repairs = _wire(core)
    # A file that really is there, so the run fails inside the capability
    # rather than on its input -- which is a different kind of failure and is
    # tested separately below.
    target = tmp_path / "x.bin"
    target.write_bytes(b"data")
    goal = f"berechne die sha-256 pruefsumme von {target}"

    core._execute_capability(manifest, goal, f"Zeus, {goal}", "test")

    failures = [e["capability_failure"] for e in events if "capability_failure" in e]
    assert failures and failures[0]["kind"] == DEFECT
    assert failures[0]["repairable"] is True
    health = [e["capability_health"] for e in events if "capability_health" in e]
    assert health and health[0]["broken"] is True
    assert len(repairs) == 1, "an INSIGHT is not a repair; the repair path must actually start"
    repaired_id, brief = repairs[0]
    assert repaired_id == "local.berechne.sha.256_pruefsumme", "the repair must name the SAME capability"
    assert "hex_digest" in brief, "the defect itself is the repair brief, not a blank page"


def test_a_refused_path_is_explained_and_never_sent_to_an_engineer(tmp_path: Path) -> None:
    core = _core(tmp_path)
    manifest = _install(
        core, "files.sha256",
        "def run(payload):\n"
        "    return {'ok': False, 'error': 'x lies under a secret store (secrets); reading it is not allowed'}\n",
    )
    events, delivered, repairs = _wire(core)
    target = tmp_path / "token.json"
    target.write_text("{}", encoding="utf-8")
    goal = f"hash {target}"

    core._execute_capability(manifest, goal, f"Zeus, {goal}", "test")

    failures = [e["capability_failure"] for e in events if "capability_failure" in e]
    assert failures and failures[0]["kind"] == PERMISSION
    assert repairs == [], "a correct refusal is not an engineering task"
    assert core.capabilities.registry.get("files.sha256").health_state() is CapabilityHealth.HEALTHY
    assert delivered and ("nicht erlaubt" in delivered[-1] or "not permitted" in delivered[-1])


def test_a_missing_target_asks_the_owner_rather_than_rebuilding_anything(tmp_path: Path) -> None:
    core = _core(tmp_path)
    manifest = _install(
        core, "files.sha256",
        "def run(payload):\n    return {'ok': False, 'error': 'file not found: bericht.txt'}\n",
    )
    events, delivered, repairs = _wire(core)
    target = tmp_path / "bericht.txt"
    target.write_text("x", encoding="utf-8")
    goal = f"hash {target}"

    core._execute_capability(manifest, goal, f"Zeus, {goal}", "test")

    failures = [e["capability_failure"] for e in events if "capability_failure" in e]
    assert failures and failures[0]["kind"] == INPUT
    assert repairs == []
    assert core.capabilities.registry.get("files.sha256").health_state() is CapabilityHealth.HEALTHY


# ==========================================================================
# The real installed capability, on the nine cases the owner can produce
# ==========================================================================


INSTALLED = (Path(__file__).resolve().parent.parent / "data" / "jarvis" / "capabilities" / "installed"
             / "local_berechne_sha_256_pruefsumme")


def _installed_module() -> Any:
    import importlib.util

    versions = sorted((p for p in INSTALLED.glob("*/main.py")), key=lambda p: p.parent.name)
    if not versions:
        pytest.skip("the sha-256 capability is not installed in this checkout")
    spec = importlib.util.spec_from_file_location("installed_sha_capability", versions[-1])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_installed_checksum_capability_answers_every_shape_of_real_request(tmp_path: Path) -> None:
    """Python's hashlib contract is ``hexdigest``; the live error said ``hex_digest``."""

    import hashlib

    module = _installed_module()

    small = tmp_path / "klein.txt"
    small.write_bytes(b"hallo zeus\n")
    unicode_file = tmp_path / "Prüfsummen ✓" / "größe_日本.txt"
    unicode_file.parent.mkdir(parents=True)
    unicode_file.write_bytes("unicode äöü\n".encode("utf-8"))
    # Larger than the capability's own chunk size, so the streaming path runs.
    large = tmp_path / "gross.bin"
    large.write_bytes(os.urandom(1) * (3 * 1024 * 1024 + 7))

    for target in (small, unicode_file, large):
        result = module.run({"file_path": str(target)})
        assert result["ok"], result
        assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert result["bytes"] == target.stat().st_size

    absent = module.run({"file_path": str(tmp_path / "gibt-es-nicht.txt")})
    assert not absent["ok"] and "not found" in absent["error"]

    directory = module.run({"file_path": str(tmp_path)})
    assert not directory["ok"] and "not a file" in directory["error"]

    nothing = module.run({})
    assert not nothing["ok"] and "required" in nothing["error"]


# ==========================================================================
# The insight that outlived its defect
# ==========================================================================


def test_a_repaired_capability_stops_quoting_the_error_of_the_version_before_it(tmp_path: Path) -> None:
    """The 2026-09-09 screenshot named a defect that had been fixed on 2026-09-08."""

    core = _core(tmp_path)
    _install(core, "files.sha256", "def run(payload):\n    return {'ok': True}\n")
    registry = core.capabilities.registry
    registry.note_execution("files.sha256", False,
                            "AttributeError: '_hashlib.HASH' object has no attribute 'hex_digest'", kind=DEFECT)
    assert registry.get("files.sha256").health_state() is CapabilityHealth.BROKEN

    registry.note_execution("files.sha256", True, "")
    registry.note_execution("files.sha256", True, "")

    health = registry.get("files.sha256").health_view()
    assert health["state"] == "healthy"
    assert health["last_error"] == "", "a repaired capability has no current error to report"
    assert not health.get("last_failure_kind")


def test_a_warning_about_a_capability_that_recovered_is_closed(tmp_path: Path) -> None:
    """One thought per capability, and it ends when the condition does."""

    from runtime.thoughts import ThoughtStore, capability_degradation

    broken = [{"capability_id": "files.sha256", "version": "1.0.0",
               "health": {"state": "failing", "last_error": "AttributeError: no attribute 'hex_digest'",
                          "consecutive_failures": 3, "last_failure_kind": "DEFECT"}}]
    thoughts = capability_degradation(broken)
    assert len(thoughts) == 1
    assert thoughts[0].key == "capability|files.sha256", "the key must not carry the state it is about"
    assert "1.0.0" in thoughts[0].text, "which version is failing is part of the claim"

    store = ThoughtStore(tmp_path / "thoughts.json")
    stored, outcome = store.offer(thoughts[0])
    assert outcome == "new" and stored is not None and stored.status in {"NEW", "IMPORTANT"}

    # Codex repairs it; the registry says healthy; the warning is no longer true.
    assert capability_degradation([{"capability_id": "files.sha256", "version": "1.0.1",
                                    "health": {"state": "healthy"}}]) == []
    assert store.retire({"capability|files.sha256"}) == 1
    assert store.get(stored.thought_id).status == "ACTED_ON"
    assert store.retire({"capability|files.sha256"}) == 0, "retiring twice changes nothing"
