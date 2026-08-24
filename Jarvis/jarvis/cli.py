"""The interactive Jarvis console.

A thin shell over :class:`~core.kernel.JarvisKernel`.  Everything it can do, a
future HTTP service or a portable device client can do too, because the CLI
holds no logic of its own -- it parses a line, calls the kernel, and prints what
came back.  That is the point: the terminal is one client, not the system.

Two behaviours are deliberate and worth stating.

*The status line does not flatter.*  A tier is reported ONLINE only after a real
generation succeeded.  On this machine ``GET /v1/models`` returns HTTP 200 for a
model that is not pulled, so anything cheaper than a generation would be a lie
with a green tick next to it.

*Work becomes a project, not a reply.*  Asked to build something, Jarvis does
not answer with a description of what it would build.  It opens a durable
project and starts working, and the session can be closed and resumed later.
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

from brain.router import BrainRouter, BrainTier, BrainUnavailable
from brain.tiers import ModelTier
from capabilities.registry import CapabilityRegistry
from capabilities.service import CapabilityService
from core.kernel import JarvisKernel, KernelConfig
from knowledge.graph import KnowledgeGraph
from knowledge.memory import ExperienceMemory
from persona.profiles import Persona, PersonaStore
from projects.engine import EngineHooks
from projects.models import Project, ProjectState

BANNER = r"""
     _   _   _____   _____  _   _ _____ ____
    | | / \ |  _  \ \ \   / / | / /_   _/ ___|
 _  | |/ _ \| |_| |  \ \ / /| |/ /  | | \___ \
| |_| / ___ \  _  /   \ V / |   <   | |  ___) |
 \___/_/   \_\_| \_\   \_/  |_|\_\ |___||____/
"""

HELP = """
Commands
  <anything>            talk to Jarvis, or describe work to be done
  /status               honest health of every model tier and the machine
  /projects             list projects
  /project <id|words>   show one project in detail
  /work <id|words>      keep working on a project (add --steps N)
  /new <goal>           start a project explicitly
  /say <goal>           add a requirement to the most recent project
  /capabilities         list installed capabilities
  /learn <goal>         acquire a new capability
  /use <id> [json]      run an installed capability
  /remember <text>      write a note into the knowledge graph
  /recall <query>       search everything Jarvis knows
  /persona [name]       show or switch persona
  /tune                 benchmark this machine and store a resource policy
  /help                 this list
  /quit  /exit  /bye    leave
"""


class JarvisConsole:
    """The REPL.  Holds no domain logic -- only parsing and presentation."""

    def __init__(self, kernel: JarvisKernel | None = None) -> None:
        self.kernel = kernel or JarvisKernel()
        self.router = BrainRouter(catalog=self.kernel.catalog, probe=self.kernel.probe)
        self.personas = PersonaStore(self.kernel.state_root / "personas.json")
        self.graph = KnowledgeGraph(self.kernel.state_root / "knowledge" / "palace.sqlite")
        self.memory = ExperienceMemory(self.graph)
        self.capabilities = CapabilityService(
            registry=CapabilityRegistry(self.kernel.state_root / "capabilities" / "registry.json"),
            engine=self.kernel.engine(),
            graph=self.graph,
            root=self.kernel.state_root / "capabilities" / "installed",
        )
        self._last_project_id = ""

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def banner(self) -> None:
        print(BANNER)
        health = self.kernel.health()
        for tier in (ModelTier.FAST_LOCAL, ModelTier.BUILD_LOCAL):
            item = health[tier]
            print(f"  {tier.value:<14} {item.summary()}")
        gpu = self.kernel.host.primary_gpu
        if gpu:
            print(f"  {'GPU':<14} {gpu.name}, {gpu.free_mib} of {gpu.total_mib} MiB free")
        windows = self.kernel.resources.context_windows
        if windows:
            print(f"  {'context':<14} " + ", ".join(f"{k}={v}" for k, v in sorted(windows.items())))
        if self.kernel.catalog.paid_tiers_enabled():
            print(f"  {'paid tiers':<14} {', '.join(t.value for t in self.kernel.catalog.paid_tiers_enabled())}")
        else:
            print(f"  {'cost':<14} local only, no paid service enabled")
        print(f"  {'persona':<14} {self.personas.active().name}")
        print("\n  /help for commands\n")

    def status(self) -> None:
        payload = self.kernel.status(force=True)
        print()
        print("Models")
        for name, item in sorted(payload["tiers"].items()):
            mark = "ok " if item["online"] else "-- "
            detail = item["detail"][:90] if not item["online"] else f"{item['model']} ({item['latency_seconds']:.1f}s)"
            print(f"  {mark} {name:<16} {item['state']:<22} {detail}")
        print("\nMachine")
        host = payload["host"]
        print(f"  {host['platform']}, {host['cpu_count']} CPUs, {host['total_ram_mib']} MiB RAM")
        for gpu in host["gpus"]:
            print(f"  {gpu['name']}: {gpu['free_mib']} of {gpu['total_mib']} MiB free")
        resources = payload["resources"]
        print(f"  context windows: {resources['context_windows']}")
        print(f"  concurrency: {resources['max_concurrent_generations']}, reserved VRAM: {resources['reserved_vram_mib']} MiB")
        print(f"  tuned: {resources['tuned_at']}")
        print("\nState")
        print(f"  root: {payload['state_root']}")
        print(f"  projects: {payload['projects']['by_state'] or 'none yet'}")
        print(f"  capabilities: {len(self.capabilities.list())}")
        print(f"  tools: {len(payload['tools'])}")
        print(f"  cloud-free: {payload['cloud_free']}")
        print()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> None:
        projects = self.kernel.projects.list_projects(limit=20)
        if not projects:
            print("\nNo projects yet. Describe something you want built.\n")
            return
        print()
        for project in projects:
            progress = project.progress()
            print(
                f"  {project.id}  {project.state.value:<13} "
                f"{progress['acceptance_satisfied']}/{progress['acceptance_objective']} checks  "
                f"{project.title[:60]}"
            )
        print()

    def show_project(self, reference: str) -> None:
        project = self.kernel.resolve_project(reference)
        if project is None:
            print(f"\nNo project matches {reference!r}.\n")
            return
        self._last_project_id = project.id
        print(f"\n{project.id}  [{project.state.value}]  {project.title}")
        print(f"  goal:      {textwrap.shorten(project.goal, 160)}")
        print(f"  workspace: {project.workspace}")
        print(f"  spent:     {project.steps_spent} steps, {project.seconds_spent:.0f}s")
        if project.requirements:
            print("  requirements:")
            for item in project.requirements:
                print(f"    - {textwrap.shorten(item.text, 110)}")
        if project.acceptance:
            print("  acceptance:")
            for item in project.acceptance:
                mark = "x" if item.satisfied else " "
                check = " ".join(item.check) if item.check else "NO RUNNABLE CHECK"
                print(f"    [{mark}] {textwrap.shorten(item.text, 80)}   ({textwrap.shorten(check, 70)})")
        if project.tasks:
            print("  tasks:")
            for task in project.tasks[-12:]:
                print(f"    [{task.status.value:<11}] {textwrap.shorten(task.title, 90)}")
        blockers = project.active_blockers()
        if blockers:
            print("  blocked on:")
            for blocker in blockers:
                print(f"    ! {blocker.text}{'  (needs you)' if blocker.needs_user else ''}")
        if project.artifacts:
            print("  produced:")
            for artifact in project.artifacts[:12]:
                print(f"    - {artifact.path}")
        print()

    def work(self, reference: str, *, steps: int | None = None) -> None:
        project = self.kernel.resolve_project(reference) if reference else self._current_project()
        if project is None:
            print(f"\nNo project matches {reference!r}.\n")
            return

        ready, why = self.kernel.ready_for_autonomous_work()
        if not ready:
            print(f"\nCannot work: {why}\n")
            return

        self._last_project_id = project.id
        print(f"\nWorking on {project.id}: {project.title}")
        print(f"Model: {self.kernel.catalog.get(ModelTier.BUILD_LOCAL).model}  (Ctrl+C to stop)\n")

        def on_step(current: Project, step: Any) -> None:
            mark = "ok  " if step.success else "FAIL"
            print(f"  {step.index:>3} {step.phase.value:<12} {mark} {step.duration_seconds:>6.1f}s  {step.summary[:88]}")

        try:
            result = self.kernel.work(project, max_steps=steps, hooks=EngineHooks(on_step=on_step))
        except KeyboardInterrupt:
            self.kernel.projects.save(project)
            print("\n\nStopped. The project is saved; /work to continue.\n")
            return

        print(f"\n  {result.stop_reason.value}: {result.message}")
        if result.accepted:
            print("  Every acceptance check passes.")
        else:
            print("  Not finished. /project to see where it got to, /work to continue.")
        self.memory.record_project(project)
        print()

    def start_project(self, goal: str) -> None:
        project = self.kernel.start_project(goal)
        self._last_project_id = project.id
        print(f"\nStarted {project.id}: {project.title}")
        prior = self.memory.brief_for(goal)
        if prior:
            print("\nRelevant past experience:\n" + textwrap.indent(prior, "  "))
        print()
        self.work(project.id)

    def add_requirement(self, text: str) -> None:
        project = self._current_project()
        if project is None:
            print("\nNo project to add to. Use /new <goal> first.\n")
            return
        self.kernel.engine().add_requirement(project, text)
        print(f"\nAdded to {project.id}. It will be picked up on the next /work.\n")

    def _current_project(self) -> Project | None:
        if self._last_project_id and self.kernel.projects.exists(self._last_project_id):
            return self.kernel.projects.load(self._last_project_id)
        active = self.kernel.projects.active()
        return active[0] if active else None

    # ------------------------------------------------------------------
    # Capabilities and knowledge
    # ------------------------------------------------------------------

    def list_capabilities(self) -> None:
        installed = self.capabilities.list()
        if not installed:
            print("\nNo capabilities yet. /learn <goal> to acquire one.\n")
            return
        print()
        for manifest in installed:
            keys = ", ".join((manifest.input_schema.get("properties") or {}))
            print(f"  {manifest.capability_id:<40} v{manifest.version}")
            print(f"      {textwrap.shorten(manifest.description, 96)}")
            print(f"      takes: {keys or '(nothing)'}")
        print()

    def learn(self, goal: str) -> None:
        ready, why = self.kernel.ready_for_autonomous_work()
        if not ready:
            print(f"\nCannot learn: {why}\n")
            return
        print(f"\nAcquiring a capability for: {goal}\n")

        def on_step(current: Project, step: Any) -> None:
            mark = "ok  " if step.success else "FAIL"
            print(f"  {step.index:>3} {step.phase.value:<12} {mark} {step.duration_seconds:>6.1f}s  {step.summary[:88]}")

        self.capabilities.engine.hooks = EngineHooks(on_step=on_step)
        outcome = self.capabilities.ensure(goal)
        print(f"\n  {outcome.status}: {outcome.reason}")
        for check in outcome.verification.get("checks", []):
            print(f"    {'PASS' if check['ok'] else 'FAIL'}  {check['name']}")
        if outcome.usable:
            print(f"  Registered as {outcome.capability_id}. /use {outcome.capability_id} {{...}}")
        print()

    def use(self, argument: str) -> None:
        parts = argument.split(None, 1)
        capability_id = parts[0] if parts else ""
        payload: dict[str, Any] = {}
        if len(parts) > 1:
            try:
                payload = json.loads(parts[1])
            except json.JSONDecodeError as exc:
                print(f"\nThat is not valid JSON: {exc}\n")
                return
        manifest = self.capabilities.registry.get(capability_id)
        if manifest is None:
            print(f"\nNo capability called {capability_id!r}. /capabilities to see the list.\n")
            return
        result = self.capabilities.execute(capability_id, payload)
        print(f"\n  ok={result.ok}  ({result.duration_seconds:.1f}s)")
        print(textwrap.indent(json.dumps(result.output, indent=2, default=str)[:2000], "  "))
        if result.error:
            print(f"  error: {result.error[:400]}")
        print()

    def remember(self, text: str) -> None:
        node = self.graph.note(text[:90], text, provenance="user", confidence=1.0)
        print(f"\nNoted ({node.id}).\n")

    def recall(self, query: str) -> None:
        hits = self.graph.search(query, limit=8)
        if not hits:
            print("\nNothing found.\n")
            return
        print()
        for hit in hits:
            print(f"  [{hit.node.type.value:<10}] {textwrap.shorten(hit.node.title, 88)}")
            print(f"      {hit.how}, score {hit.score:.2f}, from {hit.node.provenance or 'unknown'}")
        print()

    def persona(self, name: str) -> None:
        if not name:
            print(f"\nActive: {self.personas.active().name}")
            for available in self.personas.names():
                print(f"  {available:<14} {self.personas.get(available).description}")
            print()
            return
        try:
            persona = self.personas.activate(name.strip())
        except KeyError as exc:
            print(f"\n{exc}\n")
            return
        print(f"\nPersona is now {persona.name}.\n")

    def tune(self) -> None:
        from brain.resources import ResourceTuner

        print("\nBenchmarking. Each context size loads the model once, so this takes a few minutes.\n")
        policy = ResourceTuner(self.kernel.catalog).tune()
        self.kernel.resource_store.save(policy)
        policy.apply_to(self.kernel.catalog)
        self.kernel.resources = policy
        for note in policy.notes:
            print(f"  {note}")
        print("\nStored. Restart to be sure every component picks it up.\n")

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def chat(self, message: str) -> None:
        decision = self.router.route(message)
        if decision.is_project:
            existing = self.kernel.projects.find(message, limit=1)
            if existing and not existing[0].state.terminal:
                print(f"\nThis looks like {existing[0].id} ({existing[0].title}).")
                print("  /work to continue it, or /new to start a separate project.\n")
                self._last_project_id = existing[0].id
                return
            print(f"\n[{decision.reason}] Opening a project rather than answering.")
            self.start_project(message)
            return

        try:
            brain = self.router.brain(BrainTier.FAST_LOCAL)
        except BrainUnavailable as exc:
            print(f"\n{exc}\n")
            return

        system = self.personas.system_prompt()
        context = self.memory.brief_for(message)
        prompt = f"{context}\n\n{message}" if context else message
        try:
            if hasattr(brain, "system_prompt"):
                brain.system_prompt = system
            answer = brain.generate(prompt, max_tokens=768, temperature=0.3)
        except Exception as exc:
            print(f"\nThe model could not answer: {type(exc).__name__}: {exc}\n")
            return
        print(f"\n{answer.strip()}\n")

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Handle one input line.  Returns False when the session should end."""

        command = line.strip()
        if not command:
            return True

        lowered = command.lower()
        if lowered in {"/quit", "/exit", "/bye", "quit", "exit", "bye"}:
            return False
        if lowered in {"/help", "help", "?"}:
            print(HELP)
            return True
        if lowered == "/status":
            self.status()
            return True
        if lowered == "/projects":
            self.list_projects()
            return True
        if lowered == "/capabilities":
            self.list_capabilities()
            return True
        if lowered == "/tune":
            self.tune()
            return True

        verb, _, rest = command.partition(" ")
        rest = rest.strip()
        actions = {
            "/project": lambda: self.show_project(rest),
            "/new": lambda: self.start_project(rest),
            "/say": lambda: self.add_requirement(rest),
            "/learn": lambda: self.learn(rest),
            "/use": lambda: self.use(rest),
            "/remember": lambda: self.remember(rest),
            "/recall": lambda: self.recall(rest),
            "/persona": lambda: self.persona(rest),
        }
        if verb.lower() == "/work":
            steps, reference = _extract_steps(rest)
            self.work(reference, steps=steps)
            return True
        action = actions.get(verb.lower())
        if action is not None:
            if not rest and verb.lower() in {"/new", "/say", "/learn", "/use", "/remember", "/recall"}:
                print(f"\n{verb} needs something after it. /help for the list.\n")
                return True
            action()
            return True
        if command.startswith("/"):
            print(f"\nUnknown command {verb}. /help for the list.\n")
            return True

        self.chat(command)
        return True

    def run(self) -> None:
        self.banner()
        while True:
            try:
                line = input("jarvis> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return
            try:
                if not self.handle(line):
                    print("Goodbye.")
                    return
            except BrainUnavailable as exc:
                print(f"\n{exc}\n")
            except KeyboardInterrupt:
                print("\n(interrupted)\n")
            except Exception as exc:  # a bad command must not end the session
                print(f"\n{type(exc).__name__}: {exc}\n")


def _extract_steps(argument: str) -> tuple[int | None, str]:
    """Pull ``--steps N`` out of an argument string."""

    parts = argument.split()
    steps: int | None = None
    remaining: list[str] = []
    index = 0
    while index < len(parts):
        if parts[index] == "--steps" and index + 1 < len(parts):
            try:
                steps = int(parts[index + 1])
            except ValueError:
                pass
            index += 2
            continue
        remaining.append(parts[index])
        index += 1
    return steps, " ".join(remaining)


def main() -> None:
    JarvisConsole().run()


if __name__ == "__main__":
    main()
