"""The doctor answers from what exists; it never crashes and never probes a model."""

from __future__ import annotations

from pathlib import Path

from test_action_receipts import PlanningProvider, build_core


def test_doctor_runs_every_check_without_a_model(tmp_path: Path, monkeypatch):
    core = build_core(tmp_path, PlanningProvider(plan={"kind": "none"}))
    probes = []
    monkeypatch.setattr(core.kernel, "status", lambda force=False, probe=False: probes.append(probe) or {"tiers": {}})
    report = core.doctor()
    names = [c["name"] for c in report["checks"]]
    for expected in ("supervisor", "core", "fast_local", "build_local", "expert", "ollama", "gpu", "voice", "duplicates",
                     "window", "revision", "release", "capabilities", "missions", "rollback", "isolation"):
        assert expected in names, names
    assert all(c["level"] in {"ok", "warn", "error"} for c in report["checks"])
    assert not any("check crashed" in c["detail"] for c in report["checks"]), [c for c in report["checks"] if "crashed" in c["detail"]]
    assert probes and not any(probes), "the doctor must never ask the kernel to probe a tier"
    # A test core has no generation and no supervisor: those are findings, not crashes.
    core_check = next(c for c in report["checks"] if c["name"] == "core")
    assert core_check["level"] == "error" and "not READY" in core_check["detail"]
    assert report["healthy"] is False and "core" in report["errors"]
    assert report["seconds"] < 30
