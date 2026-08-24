"""Run the acceptance scenarios that need real inference, and record what happened.

    python -m jarvis.record_evidence            # every scenario
    python -m jarvis.record_evidence --only A   # just the self-patch

Written as a separate command rather than folded into the test suite because
each scenario costs several minutes of real model time, and a suite people are
reluctant to run is a suite that stops being run.  The tests in
``tests/test_acceptance.py`` marked ``live`` read the JSON this produces, so the
evidence is checked automatically while the expensive part happens on demand.

Nothing here fabricates a result.  Every file it writes is the outcome of a real
run against the configured BUILD_LOCAL model, including the failures.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from brain.tiers import ModelCatalog, ModelProbe, ModelTier
from runtime.deadline import Deadline, DeadlineExceeded
from runtime.heartbeat import Heartbeat, check_liveness
from core.kernel import JarvisKernel, KernelConfig
from development.repository_engineer import RepositoryEngineer, SelfImprovementGoal
from projects.models import ResourceLimits

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "data" / "acceptance_evidence"


def _record(name: str, payload: dict) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "recorded_at": datetime.now(timezone.utc).isoformat()}
    path = EVIDENCE / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _build_local():
    catalog = ModelCatalog()
    from brain.resources import ResourcePolicyStore

    policy = ResourcePolicyStore(REPO / "config" / "resources.json").load()
    if policy is not None:
        policy.apply_to(catalog)
    return catalog.get(ModelTier.BUILD_LOCAL)


def _require_online() -> None:
    health = ModelProbe(ModelCatalog(), ttl_seconds=0).probe(ModelTier.BUILD_LOCAL, force=True)
    if not health.online:
        raise SystemExit(f"BUILD_LOCAL is not usable: {health.summary()}")
    print(f"BUILD_LOCAL: {health.model} ({health.latency_seconds:.1f}s probe)\n", flush=True)


# --------------------------------------------------------------------------
# A. Self-patch against this repository
# --------------------------------------------------------------------------

def scenario_a(run_root: Path, budget: Deadline) -> dict:
    """Jarvis modifies its own repository, in an isolated candidate worktree."""

    from brain.providers import provider_for_spec
    from development.repository_engineer import ModelRequestBudget, SelfDeveloperCheckpoint, SelfImprovementMemory

    spec = _build_local()
    brain = provider_for_spec(spec)
    # The goal must be something the CLI does not already do, and the check
    # must prove behaviour rather than the presence of a substring. An earlier
    # version asked for /bye, which a later CLI rewrite had already added -- so
    # the scenario passed its own check before the model did anything.
    goal = SelfImprovementGoal(
        objective=(
            "Add /goodbye to the interactive console as another way to exit, alongside the words "
            "that already quit. Change only jarvis/cli.py."
        ),
        allowed_paths=["jarvis/cli.py"],
        protected_paths=["tests", "development", "brain", "projects", "capabilities", "tools"],
        tests=[
            [
                sys.executable,
                "-c",
                # Passes only if "/goodbye" ends up in the SAME set literal as
                # "/quit" -- i.e. it really is treated as an exit word.
                "import ast, pathlib; "
                "tree = ast.parse(pathlib.Path('jarvis/cli.py').read_text(encoding='utf-8')); "
                "ok = any(isinstance(n, ast.Set) and "
                "{getattr(e, 'value', None) for e in n.elts} >= {'/quit', '/goodbye'} "
                "for n in ast.walk(tree)); "
                "raise SystemExit(0 if ok else 1)",
            ]
        ],
    )

    benchmark_root = run_root / "self_patch"
    shutil.rmtree(benchmark_root, ignore_errors=True)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    engineer = RepositoryEngineer(
        brain=brain,
        worktree_root=run_root / "worktrees" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        memory=SelfImprovementMemory(benchmark_root / "trajectories.jsonl"),
        timeout_seconds=180.0,
        max_cycles=4,
        context_budget=ModelRequestBudget.from_env(spec.context_window),
        checkpoint=SelfDeveloperCheckpoint(benchmark_root),
        max_seconds=budget.remaining,
        model_call_timeout_seconds=300.0,
        heartbeat=Heartbeat(benchmark_root / "heartbeat.json", run="scenario_a").start(),
    )

    started = time.perf_counter()
    engineer.preflight()
    result = engineer.improve(REPO, goal, goal.tests)
    elapsed = time.perf_counter() - started

    diff = ""
    if result.diff_path and Path(result.diff_path).exists():
        diff = Path(result.diff_path).read_text(encoding="utf-8", errors="replace")[:4000]

    corrections = [
        event["payload"]
        for event in _events(result)
        if event.get("stage") == "DIAGNOSE" and event.get("payload", {}).get("kind")
    ]

    return {
        "scenario": "A_self_patch_live",
        "model": spec.model,
        "context_window": spec.context_window,
        "status": result.status,
        "success": result.success,
        "cycles": result.cycles,
        "changed_files": result.changed_files,
        "targeted_tests_passed": all(item.success for item in result.tests) if result.tests else None,
        "protection_state": result.protection_state,
        "elapsed_seconds": round(elapsed, 1),
        "recovery_events": corrections,
        "diff": diff,
    }


def _events(result) -> list[dict]:
    if not result.result_path or not Path(result.result_path).exists():
        return []
    try:
        payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload.get("trajectory", {}).get("events", [])


# --------------------------------------------------------------------------
# E. A brand-new project outside the repository
# --------------------------------------------------------------------------

def scenario_e(run_root: Path, budget: Deadline) -> dict:
    """A real application built from a natural-language goal, in isolation."""

    from projects.engine import EngineHooks

    state = run_root / "new_project"
    shutil.rmtree(state, ignore_errors=True)
    kernel = JarvisKernel(KernelConfig(state_root=state, enable_research_tools=False))

    steps: list[dict] = []
    project = kernel.start_project(
        "Create a small Python command-line tool called wordfreq. It must expose a function "
        "count_words(text) in wordfreq.py that returns a dict mapping each lowercase word to how "
        "many times it appears, ignoring punctuation. Also write tests for it in test_wordfreq.py "
        "using pytest.",
        limits=ResourceLimits(
            max_steps=30,
            max_seconds=min(2400.0, budget.remaining),
            max_consecutive_failures=6,
            step_timeout_seconds=300,
        ),
        acceptance=[("the tests pass", [sys.executable, "-m", "pytest", "-q"])],
    )

    def on_step(current, step):
        steps.append(
            {"index": step.index, "phase": step.phase.value, "success": step.success, "summary": step.summary[:160]}
        )
        print(f"  {step.index:>3} {step.phase.value:<12} {'ok  ' if step.success else 'FAIL'} {step.summary[:90]}", flush=True)

    started = time.perf_counter()
    result = kernel.work(project, hooks=EngineHooks(on_step=on_step))
    elapsed = time.perf_counter() - started

    workspace = kernel.projects.workspace_for(project)
    produced = {
        path.name: path.read_text(encoding="utf-8", errors="replace")[:2000]
        for path in sorted(workspace.glob("*.py"))
    }
    kernel.release_models()

    return {
        "scenario": "E_new_project_live",
        "model": kernel.catalog.get(ModelTier.BUILD_LOCAL).model,
        "context_window": kernel.catalog.get(ModelTier.BUILD_LOCAL).context_window,
        "accepted": result.accepted,
        "stop_reason": result.stop_reason.value,
        "steps": result.steps,
        "elapsed_seconds": round(elapsed, 1),
        "workspace": str(workspace),
        "acceptance": [
            {"text": item.text, "check": item.check, "satisfied": item.satisfied, "evidence": item.last_evidence[-800:]}
            for item in project.acceptance
        ],
        "files": produced,
        "trajectory": steps,
    }


# --------------------------------------------------------------------------
# F. A practical capability
# --------------------------------------------------------------------------

def scenario_f(run_root: Path, budget: Deadline) -> dict:
    """Acquire a capability the machine does not have, then use it."""

    from capabilities.registry import CapabilityRegistry
    from capabilities.service import CapabilityService
    from knowledge.graph import KnowledgeGraph
    from projects.engine import EngineHooks

    state = run_root / "capability"
    shutil.rmtree(state, ignore_errors=True)
    kernel = JarvisKernel(KernelConfig(state_root=state, enable_research_tools=False))

    steps: list[dict] = []

    def on_step(current, step):
        steps.append({"index": step.index, "phase": step.phase.value, "success": step.success, "summary": step.summary[:160]})
        print(f"  {step.index:>3} {step.phase.value:<12} {'ok  ' if step.success else 'FAIL'} {step.summary[:90]}", flush=True)

    engine = kernel.engine(hooks=EngineHooks(on_step=on_step))
    service = CapabilityService(
        registry=CapabilityRegistry(state / "capabilities" / "registry.json"),
        engine=engine,
        graph=KnowledgeGraph(state / "knowledge" / "palace.sqlite"),
        root=state / "capabilities" / "installed",
        execution_timeout=90,
    )

    goal = (
        "play an audio file on this Windows computer, given the path to a .wav file. "
        "Discover which player is actually available on the machine and use it."
    )
    started = time.perf_counter()
    outcome = service.ensure(
        goal,
        max_steps=60,
        keywords=["music", "song", "sound", "playback"],
        max_seconds=budget.remaining,
    )
    elapsed = time.perf_counter() - started

    payload = {
        "scenario": "F_capability_live",
        "model": kernel.catalog.get(ModelTier.BUILD_LOCAL).model,
        "goal": goal,
        "status": outcome.status,
        "acquired": outcome.acquired,
        "capability_id": outcome.capability_id,
        "verification": outcome.verification,
        "elapsed_seconds": round(elapsed, 1),
        "trajectory": steps,
    }

    if outcome.usable:
        sample = state / "sample.wav"
        _write_wav(sample)
        request = {"dry_run": True}
        for name, details in (outcome.manifest.input_schema.get("properties") or {}).items():
            if name != "dry_run" and str(details.get("type", "string")) == "string":
                request[name] = str(sample)
        execution = service.execute(outcome.capability_id, request)
        payload["input_schema"] = outcome.manifest.input_schema
        payload["execution"] = {"request": request, "ok": execution.ok, "output": execution.output, "error": execution.error[:400]}
        payload["implementation"] = Path(outcome.manifest.source_location, "main.py").read_text(encoding="utf-8")[:3000]

        # And prove it is reusable after a restart, with the model made unavailable.
        class ExplodingBrain:
            def generate_structured(self, *args, **kwargs):
                raise AssertionError("reuse after restart must not need the model")

        restarted = CapabilityService(
            registry=CapabilityRegistry(state / "capabilities" / "registry.json"),
            engine=kernel.engine(),
            graph=KnowledgeGraph(state / "knowledge" / "palace.sqlite"),
            root=state / "capabilities" / "installed",
        )
        restarted.engine.brain = ExplodingBrain()
        resolved = restarted.resolve("play some music")
        payload["after_restart"] = {
            "resolved_from": "play some music",
            "capability_id": resolved.capability_id if resolved else None,
            "executed_ok": bool(resolved and restarted.execute(resolved.capability_id, request).ok),
        }

    kernel.release_models()
    return payload


def _write_wav(path: Path) -> None:
    """A real, playable 0.2 s tone, so nothing depends on a fixture file."""

    import math
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * index / 22050)))
                for index in range(int(22050 * 0.2))
            )
        )


SCENARIOS = {"A": scenario_a, "E": scenario_e, "F": scenario_f}


def build_parser() -> argparse.ArgumentParser:
    """Separate from main() so the defaults can be asserted in a test.

    A wall-clock limit that only exists in a shell wrapper is not a limit; a
    test that reads the parser is how it stays one.
    """

    parser = argparse.ArgumentParser(description="Run the live acceptance scenarios and record the evidence.")
    parser.add_argument("--only", action="append", choices=sorted(SCENARIOS), default=[])
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--scenario-seconds",
        type=float,
        default=1800.0,
        help="hard wall-clock limit per scenario (default 1800)",
    )
    parser.add_argument(
        "--total-seconds",
        type=float,
        default=7200.0,
        help="hard wall-clock limit for the whole run (default 7200)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    _require_online()
    run_root = Path(args.run_root) if args.run_root else Path(tempfile.gettempdir()) / "jarvis_evidence"
    run_root.mkdir(parents=True, exist_ok=True)

    selected = args.only or sorted(SCENARIOS)

    # Hard wall-clock limits, and a heartbeat, because this command is what gets
    # left running unattended. Previously the only bound was a shell `timeout`
    # wrapper, which is not part of the program and disappears the moment
    # someone runs it a different way.
    overall = Deadline.of(args.total_seconds, name="evidence run")
    heartbeat = Heartbeat(run_root / "heartbeat.json", run="record_evidence", interval=5.0).start()
    heartbeat.set_budget(args.total_seconds)
    print(f"Heartbeat: {run_root / 'heartbeat.json'}  (python -m jarvis.doctor --run {run_root})\n", flush=True)

    try:
        for letter in selected:
            if overall.expired:
                print(f"=== scenario {letter} SKIPPED: the {args.total_seconds:.0f}s total budget is gone ===\n", flush=True)
                _record(f"{letter}_skipped", {"scenario": letter, "skipped": "total time budget exhausted"})
                continue

            print(f"=== scenario {letter} ===", flush=True)
            heartbeat.beat(f"scenario_{letter}", "starting", progress=True)
            started = time.perf_counter()
            budget = overall.child(args.scenario_seconds, name=f"scenario {letter}")
            try:
                payload = SCENARIOS[letter](run_root, budget)
            except DeadlineExceeded as exc:
                payload = {"scenario": letter, "error": str(exc), "timed_out": True, "model": _build_local().model}
            except Exception as exc:  # a failed scenario is still evidence
                payload = {"scenario": letter, "error": f"{type(exc).__name__}: {exc}", "model": _build_local().model}
            payload.setdefault("budget_seconds", round(budget.budget, 1))
            path = _record(payload.get("scenario", letter), payload)
            verdict = payload.get("success") or payload.get("accepted") or payload.get("acquired")
            elapsed = time.perf_counter() - started
            heartbeat.beat(f"scenario_{letter}", "PASS" if verdict else "did not pass", progress=True)
            print(f"    -> {'PASS' if verdict else 'did not pass'} in {elapsed:.0f}s; recorded to {path}\n", flush=True)
    finally:
        heartbeat.finish("complete")


if __name__ == "__main__":
    main()
