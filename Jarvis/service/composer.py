"""Capability composition and the general action planner.

"Begin my study session" is not a capability to build; it is a plan over
capabilities that exist: open a file, start a timer, play music.  Before ZEUS
develops anything new, this module asks whether the goal can be *composed*
from what it already has -- built-in actions, the music service, registered
capabilities -- and only names the primitive that is genuinely missing.

The planner speaks in typed steps, never in shell commands.  The model is
shown a menu of primitives (name, purpose, inputs) and answers with a JSON
list of steps that reference the menu by name; anything else is rejected
before it can run.  Execution is sequential, each step yields a receipt, and
the receipts become the mission's evidence.  The plan distinguishes what
it is: answering (no steps), doing (steps over primitives), learning (a
missing primitive to acquire), developing (a change to ZEUS -- never planned
here, that is the router's SELF_DEVELOPMENT).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class Primitive:
    name: str
    purpose: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    verification: str = ""
    risk: str = "low"  # low | medium | high
    requires: list[str] = field(default_factory=list)  # device/context requirements: screen, speaker, microphone, network
    provider: str = "builtin"  # builtin | music | capability:<id>

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def menu_line(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.inputs.items())
        return f'  {{"step": "{self.name}", {args}}}  -- {self.purpose}' if args else f'  {{"step": "{self.name}"}}  -- {self.purpose}'


BUILTIN_PRIMITIVES: tuple[Primitive, ...] = (
    Primitive("file.write", "write a text file in the workspace", {"path": "<file name>", "content": "<text>"}, {"path": "written file"},
              effects=["a file exists with the content"], verification="read back", risk="low"),
    Primitive("file.read", "read a text file from the workspace", {"path": "<file name>"}, {"content": "the text"}, verification="the file was read", risk="low"),
    Primitive("file.open", "open a file or folder with its default program", {"path": "<file or folder>"}, effects=["a program window opens"],
              verification="process started", risk="low", requires=["screen"]),
    Primitive("project.create", "create a durable project record", {"name": "<project name>"}, {"project_id": "id"}, verification="record exists", risk="low"),
    Primitive("music.play", "play music by name through the preferred provider", {"query": "<track, artist or playlist>"}, effects=["music is playing"],
              verification="media session reports playing", risk="low", requires=["speaker"], provider="music"),
    Primitive("music.pause", "pause music", verification="media session reports paused", provider="music", requires=["speaker"]),
    Primitive("music.resume", "resume music", verification="media session reports playing", provider="music", requires=["speaker"]),
    Primitive("knowledge.search", "search the owner's knowledge graph", {"query": "<terms>"}, {"nodes": "matching nodes"}, risk="low"),
    Primitive("timer.start", "start a countdown timer that announces its end", {"minutes": "<number>", "label": "<what for>"},
              effects=["a timer runs; an announcement at the end"], verification="timer registered", risk="low"),
    Primitive("note.create", "keep a short note in the workspace", {"title": "<title>", "text": "<text>"}, {"path": "note file"}, verification="read back"),
    Primitive("window.hide", "hide the ZEUS window", verification="window not visible", requires=["screen"]),
    Primitive("say", "say something to the owner (speech or text)", {"text": "<what to say>"}, verification="delivered"),
)

_COMPOUND = re.compile(r"\b(und dann|danach|anschließend|anschliessend|dann|außerdem|ausserdem|sowie|and then|then|afterwards|also|as well as)\b|,\s*(?:und|and)\s", re.I)
_LIST_VERBS = re.compile(r"\b(starte|start|öffne|oeffne|open|spiel|play|schreib|write|leg|lege|erstell|create|zeig|show|such|search|pausier|pause|sag|say|erinner|remind|stell|set)\w*", re.I)


def looks_compound(text: str) -> bool:
    """More than one thing to do: a connector, or two different action verbs."""

    if _COMPOUND.search(text or ""):
        return True
    verbs = {m.group(1).lower()[:4] for m in _LIST_VERBS.finditer(text or "")}
    return len(verbs) >= 2


@dataclass
class Step:
    step: str
    arguments: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    status: str = "planned"  # planned | done | failed | skipped | missing
    receipt_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    goal: str
    #: answering | doing | learning | developing | researching
    mode: str
    steps: list[Step] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # primitives the goal needs and ZEUS lacks
    risk: str = "low"
    reason: str = ""
    raw: str = ""

    @property
    def executable(self) -> bool:
        return self.mode == "doing" and bool(self.steps) and not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "mode": self.mode, "steps": [s.to_dict() for s in self.steps], "missing": list(self.missing),
                "risk": self.risk, "reason": self.reason}


PLANNER_PROMPT = """You turn the owner's goal into a short plan of typed steps. You do not perform them.

Every step must be one of these primitives, with its inputs filled in from the goal:
{menu}

If the goal needs something none of the primitives can do, add a step
  {{"step": "MISSING", "primitive": "<short name for the missing ability>", "purpose": "<what it must do>"}}
instead of inventing one.

If the goal is a question, an opinion or conversation, answer with {{"mode": "answering"}}.
Otherwise answer with {{"mode": "doing", "steps": [ ...steps in order... ]}}.

Rules:
- Reply with one JSON object and nothing else.
- Copy names, titles, file names and text exactly as the owner wrote them.
- Never invent a file name or a track the owner did not name.
- Prefer fewer steps. Never repeat a step.

Owner's goal:
{goal}

JSON:"""


class Composer:
    """Menu → plan → execution, with a gap named when composition cannot cover the goal."""

    def __init__(self, *, capabilities: Iterable[dict[str, Any]] = (), extra: Iterable[Primitive] = (),
                 context_requirements: Iterable[str] = ("screen", "speaker", "microphone")) -> None:
        self.primitives: dict[str, Primitive] = {p.name: p for p in BUILTIN_PRIMITIVES}
        for p in extra:
            self.primitives[p.name] = p
        for manifest in capabilities:
            prim = primitive_from_manifest(manifest)
            if prim is not None:
                self.primitives[prim.name] = prim
        self.available_context = set(context_requirements)

    def menu(self) -> str:
        return "\n".join(p.menu_line() for p in self.primitives.values())

    def plan(self, goal: str, provider: Any, *, guidance: str = "") -> Plan:
        prompt = PLANNER_PROMPT.format(menu=self.menu(), goal=goal.strip())
        if guidance.strip():
            prompt = prompt.replace("Owner's goal:", "The owner has corrected earlier readings; these outrank your guess:\n" + guidance.strip() + "\n\nOwner's goal:")
        try:
            try:
                raw = provider.generate(prompt, max_tokens=600, temperature=0.0)
            except TypeError:
                raw = provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return Plan(goal, "answering", reason=f"the planner could not be reached: {exc}")
        return self.parse(goal, str(raw))

    def parse(self, goal: str, raw: str) -> Plan:
        from brain.json_utils import lenient_json_loads

        try:
            payload = lenient_json_loads(raw)
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict):
            return Plan(goal, "answering", reason="the planner did not return a usable plan", raw=raw[:400])
        mode = str(payload.get("mode", "doing" if payload.get("steps") else "answering")).lower()
        if mode != "doing":
            return Plan(goal, "answering", reason=str(payload.get("reason", "")), raw=raw[:400])
        steps: list[Step] = []
        missing: list[str] = []
        risk = "low"
        seen: set[str] = set()
        for row in payload.get("steps") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("step") or row.get("action") or "").strip()
            if name == "MISSING":
                missing.append(str(row.get("primitive") or row.get("purpose") or "unknown")[:80])
                steps.append(Step("MISSING", {"purpose": str(row.get("purpose", ""))[:200]}, status="missing"))
                continue
            prim = self.primitives.get(name)
            if prim is None:
                # A step the menu never offered is a gap, not an instruction.
                missing.append(name[:80] or "unnamed")
                steps.append(Step(name or "MISSING", {}, status="missing", detail="not on the menu"))
                continue
            args = {k: v for k, v in row.items() if k not in {"step", "action", "purpose"}}
            key = name + json.dumps(args, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            unmet = [r for r in prim.requires if r not in self.available_context]
            if unmet:
                missing.append(f"{name} needs {', '.join(unmet)} on this device")
                steps.append(Step(name, args, status="missing", detail=f"needs {', '.join(unmet)}"))
                continue
            if prim.risk == "high":
                risk = "high"
            elif prim.risk == "medium" and risk == "low":
                risk = "medium"
            steps.append(Step(name, args, purpose=str(row.get("purpose", ""))[:120]))
        if not steps:
            return Plan(goal, "answering", reason="no steps", raw=raw[:400])
        return Plan(goal, "learning" if missing and all(s.status == "missing" for s in steps) else "doing", steps, missing, risk, raw=raw[:400])

    def execute(self, plan: Plan, run_step: Callable[[Step], Any], *, on_step: Callable[[Step, Any], None] | None = None) -> list[Any]:
        """Run the steps in order; stop at the first failure.  ``run_step`` returns a receipt."""

        receipts: list[Any] = []
        for step in plan.steps:
            if step.status == "missing":
                step.status = "skipped"
                continue
            try:
                receipt = run_step(step)
            except Exception as exc:  # noqa: BLE001
                step.status, step.detail = "failed", f"{type(exc).__name__}: {exc}"[:300]
                break
            receipts.append(receipt)
            ok = bool(getattr(receipt, "ok", False))
            step.status = "done" if ok else "failed"
            step.receipt_id = str(getattr(receipt, "id", ""))
            step.detail = str(getattr(receipt, "detail", ""))[:300]
            if on_step is not None:
                on_step(step, receipt)
            if not ok:
                break
        return receipts


def primitive_from_manifest(manifest: dict[str, Any] | Any) -> Primitive | None:
    """A registered capability as a menu primitive, from its manifest."""

    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    cid = str(data.get("capability_id", "")).strip()
    if not cid or str(data.get("status", "active")) not in {"active", "degraded"}:
        return None
    schema = data.get("input_schema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null"))
        except ValueError:
            schema = {}
    props = (schema or {}).get("properties", {}) if isinstance(schema, dict) else {}
    inputs = {k: f"<{v.get('type', 'value')}>" if isinstance(v, dict) else "<value>" for k, v in props.items() if k not in {"client_id", "client_secret", "dry_run", "output"}}
    purpose = str(data.get("description", "")).split("\n")[0][:120]
    return Primitive(f"capability:{cid}", purpose or cid, inputs, verification=str((data.get("validation_status") or {}).get("summary", ""))[:80] if isinstance(data.get("validation_status"), dict) else "registry checks",
                     risk=str(data.get("risk", "low")), provider=f"capability:{cid}",
                     requires=list(data.get("requires", []) or []))
