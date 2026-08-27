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
    Primitive("knowledge.search", "search the owner's knowledge graph (a lookup, not a way to store anything)", {"query": "<terms>"}, {"nodes": "matching nodes"}, risk="low"),
    Primitive("knowledge.create", "store a fact, finding, decision or note as a node in the owner's knowledge graph, linked to what it concerns",
              {"title": "<short title>", "text": "<the content>", "type": "<technical_finding|note|decision|concept|document>",
               "links": "<comma-separated names it concerns, e.g. ZEUS, Voice, Wakeword>"},
              {"node_id": "the stored node"}, effects=["a knowledge node exists and is searchable", "relations to the named concepts exist"],
              verification="read back from the graph", risk="low"),
    Primitive("knowledge.link", "relate two existing knowledge nodes", {"source": "<title>", "target": "<title>", "relation": "<applies_to|concerns|part_of|relates_to|supports|contradicts>"},
              {"edge_id": "the relation"}, effects=["a typed relation exists"], verification="read back", risk="low"),
    Primitive("knowledge.read", "read one knowledge node and its relations", {"title": "<title or id>"}, {"node": "the node with links"}, risk="low"),
    Primitive("timer.start", "start a countdown timer that announces its end", {"minutes": "<number>", "label": "<what for>"},
              effects=["a timer runs; an announcement at the end"], verification="timer registered", risk="low"),
    Primitive("note.create", "write a markdown note FILE in the workspace (not knowledge; use knowledge.create to store knowledge)",
              {"title": "<title>", "text": "<text>"}, {"path": "note file"}, effects=["a file exists"], verification="read back"),
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


#: What a step is for.  A failed OPTIONAL step never sinks the goal; a failed
#: REQUIRED step asks for a replan; VERIFICATION steps check, they do not do.
ROLES = ("required", "optional", "alternative", "verification")

#: Primitives that observe rather than change anything.  Unless the planner
#: says otherwise they are optional: a lookup that finds nothing is not a
#: reason to abandon the goal it was looking things up for.
OBSERVING = frozenset({"knowledge.search", "knowledge.read", "file.read", "say"})


@dataclass
class Step:
    step: str
    arguments: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    status: str = "planned"  # planned | done | failed | skipped | missing | forbidden
    receipt_id: str = ""
    detail: str = ""
    role: str = "required"

    @property
    def required(self) -> bool:
        return self.role in {"required", "alternative"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Which primitives an object word in a prohibition names.  "keine Datei"
#: forbids every file-producing primitive, "keine Notiz" the note, and so on.
_OBJECT_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "file": ("file.write", "note.create"), "datei": ("file.write", "note.create"), "dateien": ("file.write", "note.create"),
    "files": ("file.write", "note.create"),
    "note": ("note.create",), "notes": ("note.create",), "notiz": ("note.create",), "notizen": ("note.create",),
    "ersatznotiz": ("note.create",), "ersatz-notiz": ("note.create",), "markdown": ("note.create", "file.write"),
    "project": ("project.create",), "projects": ("project.create",), "projekt": ("project.create",), "projekte": ("project.create",),
    "timer": ("timer.start",), "music": ("music.play", "music.resume"), "musik": ("music.play", "music.resume"),
    "window": ("window.hide",), "fenster": ("window.hide",),
}

#: What the owner wants to exist afterwards, by the family of primitive that
#: produces it.  A goal that names knowledge is met by a verified knowledge
#: write, not by a file that happens to contain the same words.
_OUTCOME_FAMILIES: dict[str, tuple[str, ...]] = {
    "knowledge": ("knowledge.create", "knowledge.link"), "wissen": ("knowledge.create", "knowledge.link"),
    "knowledge-graph": ("knowledge.create", "knowledge.link"), "graph": ("knowledge.create", "knowledge.link"),
    "wissensgraph": ("knowledge.create", "knowledge.link"),
    "datei": ("file.write",), "file": ("file.write",), "notiz": ("note.create", "file.write"), "note": ("note.create", "file.write"),
    "projekt": ("project.create",), "project": ("project.create",), "timer": ("timer.start",),
    "musik": ("music.play", "music.resume"), "music": ("music.play", "music.resume"),
}

_NEGATION = re.compile(
    r"\b(?:kein|keine|keinen|keiner|keinem|nicht|ohne|niemals|nie|no|not|never|without|don'?t|do not|avoid|vermeide)\b"
    r"((?:\s+[\w-]+){1,5})", re.I)
_FALLBACK_WORDS = re.compile(r"\b(fallback|ersatz|ersatzweise|stattdessen|instead|as a fallback|als ersatz)\b", re.I)
_OUTCOME = re.compile(r"\b(?:im|ins|in|into|to|im|in the|in den|in das|into the|ins)\s+(\w+(?:-\w+)?)", re.I)


@dataclass
class PlanConstraints:
    """What the owner said must not happen, and what must hold at the end.

    Extracted from the goal deterministically -- before any model sees it --
    and enforced in :meth:`Composer.parse` and :func:`evaluate_goal`.  Prose
    in a prompt can be ignored; a forbidden step that becomes ``status ==
    "forbidden"`` cannot be run.
    """

    forbidden_actions: list[str] = field(default_factory=list)
    forbidden_effects: list[str] = field(default_factory=list)
    required_outcome: list[str] = field(default_factory=list)
    allowed_fallbacks: list[str] = field(default_factory=list)
    fallbacks_forbidden: bool = False
    evidence: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.forbidden_actions or self.forbidden_effects or self.required_outcome or self.fallbacks_forbidden)

    def forbids(self, step_name: str) -> bool:
        if step_name in self.forbidden_actions:
            return True
        family = step_name.split(".", 1)[0]
        return family in self.forbidden_effects

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_section(self) -> str:
        if self.empty:
            return ""
        lines = ["HARD CONSTRAINTS from the owner (a plan that breaks one is rejected before it runs):"]
        if self.forbidden_actions:
            lines.append("- never use these steps: " + ", ".join(sorted(set(self.forbidden_actions))))
        if self.forbidden_effects:
            lines.append("- nothing of these kinds may be created: " + ", ".join(sorted(set(self.forbidden_effects))))
        if self.fallbacks_forbidden:
            lines.append("- no fallback or substitute of any kind; if the real thing cannot be done, add a MISSING step and say so")
        if self.required_outcome:
            lines.append("- the goal is only met by: " + ", ".join(sorted(set(self.required_outcome))))
        return "\n".join(lines)


def extract_constraints(goal: str) -> PlanConstraints:
    """Negations and outcome words in the owner's sentence, as typed constraints."""

    out = PlanConstraints()
    text = goal or ""
    negated: list[str] = []
    for match in _NEGATION.finditer(text):
        # The negation reaches to the end of its clause, not into the next one.
        span = re.split(r"[.;:!?]|\b(?:sondern|aber|but|however)\b", match.group(0), maxsplit=1)[0].lower()
        negated.append(span)
        # Every object word inside the negated span counts, not only the last.
        for word in re.findall(r"[\w-]+", span):
            prims = _OBJECT_PRIMITIVES.get(word)
            if prims:
                for prim in prims:
                    if prim not in out.forbidden_actions:
                        out.forbidden_actions.append(prim)
                if span.strip() not in out.evidence:
                    out.evidence.append(span.strip())
    if _FALLBACK_WORDS.search(text) and negated:
        out.fallbacks_forbidden = True
    lowered = text.lower()
    negated_text = " ".join(negated)
    for word, prims in _OUTCOME_FAMILIES.items():
        pattern = r"\b" + re.escape(word) + r"\b"
        if not re.search(pattern, lowered) or any(p in out.forbidden_actions for p in prims):
            continue
        if len(re.findall(pattern, negated_text)) >= len(re.findall(pattern, lowered)):
            continue  # the word only occurs inside a prohibition
        if True:
            # An outcome is required only when the owner asks to *put* something
            # there ("speichere ... im Knowledge", "store ... in Knowledge").
            if re.search(r"\b(speicher\w*|store|save|leg\w*|put|schreib\w*|write|erstell\w*|create|verknüpf\w*|link|merk\w*|remember|halte fest)\b", lowered):
                for prim in prims:
                    if prim not in out.required_outcome:
                        out.required_outcome.append(prim)
    return out


@dataclass
class GoalEvaluation:
    """ACTION_EXECUTED != EXECUTION_VERIFIED != GOAL_SATISFIED, kept apart on purpose."""

    executed: bool
    execution_verified: bool
    goal_satisfied: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ACTION_EXECUTED": self.executed, "EXECUTION_VERIFIED": self.execution_verified,
                "GOAL_SATISFIED": self.goal_satisfied, "reasons": list(self.reasons)}


def evaluate_goal(plan: "Plan", receipts: list[Any]) -> GoalEvaluation:
    """Did the owner's goal hold afterwards -- not merely "did the steps run"."""

    ran = [s for s in plan.steps if s.status in {"done", "failed"}]
    executed = bool(ran)
    by_id = {str(getattr(r, "id", "")): r for r in receipts}
    verified_steps = [s for s in plan.steps if s.status == "done" and getattr(by_id.get(s.receipt_id), "verified", False)]
    required = [s for s in plan.steps if s.required and s.status not in {"forbidden", "skipped", "missing"}]
    reasons: list[str] = []
    execution_verified = bool(required) and all(s in verified_steps for s in required)
    if not required:
        reasons.append("no required step ran")
    for s in required:
        if s.status == "failed":
            reasons.append(f"required step {s.step} failed: {s.detail[:80]}")
        elif s not in verified_steps:
            reasons.append(f"required step {s.step} ran but was not verified")
    c = plan.constraints
    kinds = [str(getattr(r, "kind", "")) for r in receipts]
    for kind in kinds:
        if c.forbids(kind) and any(getattr(r, "ok", False) for r in receipts if getattr(r, "kind", "") == kind):
            reasons.append(f"forbidden action happened: {kind}")
    outcome_ok = True
    if c.required_outcome:
        met = [s for s in verified_steps if s.step in c.required_outcome]
        if not met:
            outcome_ok = False
            reasons.append("required outcome not produced: " + ", ".join(c.required_outcome))
    for s in plan.steps:
        if s.status == "forbidden":
            reasons.append(f"planned but refused (forbidden): {s.step}")
    forbidden_happened = any(r.startswith("forbidden action happened") for r in reasons)
    goal_satisfied = execution_verified and outcome_ok and not forbidden_happened and not plan.missing
    if plan.missing:
        reasons.append("missing primitive(s): " + ", ".join(plan.missing))
    return GoalEvaluation(executed, execution_verified, goal_satisfied, reasons)


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
    constraints: PlanConstraints = field(default_factory=PlanConstraints)
    replans: int = 0

    @property
    def forbidden(self) -> list[str]:
        return [s.step for s in self.steps if s.status == "forbidden"]

    @property
    def executable(self) -> bool:
        return self.mode == "doing" and any(s.status == "planned" for s in self.steps) and not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "mode": self.mode, "steps": [s.to_dict() for s in self.steps], "missing": list(self.missing),
                "risk": self.risk, "reason": self.reason, "constraints": self.constraints.to_dict(), "forbidden": self.forbidden,
                "replans": self.replans}


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
- Choose a step by what it DOES (its purpose, inputs and effects), never because a word in its name appears in the goal.
- A step may carry "role": "optional" (nice to have; the goal survives its failure) or "role": "verification" (checks the result). Everything else is required.
- "Store in Knowledge" means knowledge.create, never a file or a note.
{constraints}
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

    def plan(self, goal: str, provider: Any, *, guidance: str = "", constraints: PlanConstraints | None = None,
             avoid: Iterable[str] = ()) -> Plan:
        constraints = constraints if constraints is not None else extract_constraints(goal)
        section = constraints.prompt_section()
        avoid = [a for a in avoid if a]
        if avoid:
            section += ("\n" if section else "") + "Do NOT use these steps; they just failed: " + ", ".join(avoid)
        prompt = PLANNER_PROMPT.format(menu=self.menu(), goal=goal.strip(), constraints=(section + "\n") if section else "")
        if guidance.strip():
            prompt = prompt.replace("Owner's goal:", "The owner has corrected earlier readings; these outrank your guess:\n" + guidance.strip() + "\n\nOwner's goal:")
        try:
            try:
                raw = provider.generate(prompt, max_tokens=600, temperature=0.0)
            except TypeError:
                raw = provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return Plan(goal, "answering", reason=f"the planner could not be reached: {exc}", constraints=constraints)
        return self.parse(goal, str(raw), constraints=constraints)

    def replan(self, plan: Plan, failed: Step, provider: Any, *, guidance: str = "") -> Plan | None:
        """One bounded second attempt for the remainder after a required step failed.

        The planner is told what already happened and what not to use again.
        Returns the new plan for the rest, or None when nothing usable came
        back -- the caller then reports BLOCKED with the real reason.
        """

        if plan.replans >= 1:
            return None
        done = [s for s in plan.steps if s.status == "done"]
        remainder_goal = (f"{plan.goal}\n\nAlready done (do not repeat): " + "; ".join(f"{s.step} {json.dumps(s.arguments, ensure_ascii=False)[:60]}" for s in done)
                          + f"\nThe step {failed.step} failed: {failed.detail[:160]}. Reach the goal another way, or say MISSING.")
        fresh = self.plan(remainder_goal, provider, guidance=guidance, constraints=plan.constraints, avoid=[failed.step])
        fresh.goal = plan.goal
        fresh.replans = plan.replans + 1
        if fresh.mode != "doing" or not any(s.status == "planned" for s in fresh.steps):
            return None
        return fresh

    def parse(self, goal: str, raw: str, *, constraints: PlanConstraints | None = None) -> Plan:
        constraints = constraints if constraints is not None else extract_constraints(goal)
        from brain.json_utils import lenient_json_loads

        try:
            payload = lenient_json_loads(raw)
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict):
            return Plan(goal, "answering", reason="the planner did not return a usable plan", raw=raw[:400], constraints=constraints)
        mode = str(payload.get("mode", "doing" if payload.get("steps") else "answering")).lower()
        if mode != "doing":
            return Plan(goal, "answering", reason=str(payload.get("reason", "")), raw=raw[:400], constraints=constraints)
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
            args = {k: v for k, v in row.items() if k not in {"step", "action", "purpose", "role", "optional"}}
            key = name + json.dumps(args, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            role = str(row.get("role") or ("optional" if row.get("optional") else "")).lower()
            if role not in ROLES:
                role = "optional" if name in OBSERVING else "required"
            if constraints.forbids(name):
                # The owner said no.  The step stays visible as what the planner
                # wanted, and can never run.
                steps.append(Step(name, args, purpose=str(row.get("purpose", ""))[:120], status="forbidden",
                                  detail="forbidden by the owner's request", role=role))
                continue
            unmet = [r for r in prim.requires if r not in self.available_context]
            if unmet:
                missing.append(f"{name} needs {', '.join(unmet)} on this device")
                steps.append(Step(name, args, status="missing", detail=f"needs {', '.join(unmet)}"))
                continue
            if prim.risk == "high":
                risk = "high"
            elif prim.risk == "medium" and risk == "low":
                risk = "medium"
            steps.append(Step(name, args, purpose=str(row.get("purpose", ""))[:120], role=role))
        if not steps:
            return Plan(goal, "answering", reason="no steps", raw=raw[:400], constraints=constraints)
        if steps and all(s.status == "forbidden" for s in steps):
            return Plan(goal, "doing", steps, missing, risk, raw=raw[:400], constraints=constraints,
                        reason="every planned step is forbidden by the owner's request")
        # The last remaining step carries the goal; a plan whose only doing
        # steps are observations would otherwise "succeed" by looking.
        planned = [s for s in steps if s.status == "planned"]
        if planned and all(s.role == "optional" for s in planned):
            planned[-1].role = "required"
        return Plan(goal, "learning" if missing and all(s.status == "missing" for s in steps) else "doing", steps, missing, risk, raw=raw[:400],
                    constraints=constraints)

    def execute(self, plan: Plan, run_step: Callable[[Step], Any], *, on_step: Callable[[Step, Any], None] | None = None,
                replan: Callable[[Plan, Step], Plan | None] | None = None) -> list[Any]:
        """Run the steps in order.  ``run_step`` returns a receipt.

        A failed *optional* step is recorded and the plan continues.  A failed
        *required* step asks ``replan`` (once) for a new plan of the
        remainder; its steps are appended to ``plan.steps`` and run.  Without
        a usable replan the plan stops there -- with the failed step named,
        not with the goal silently pretended.  Forbidden and missing steps
        never run.
        """

        receipts: list[Any] = []
        index = 0
        while index < len(plan.steps):
            step = plan.steps[index]
            index += 1
            if step.status == "missing":
                step.status = "skipped"
                continue
            if step.status != "planned":
                continue
            try:
                receipt = run_step(step)
            except Exception as exc:  # noqa: BLE001
                receipt = None
                step.status, step.detail = "failed", f"{type(exc).__name__}: {exc}"[:300]
            if receipt is not None:
                receipts.append(receipt)
                ok = bool(getattr(receipt, "ok", False))
                step.status = "done" if ok else "failed"
                step.receipt_id = str(getattr(receipt, "id", ""))
                step.detail = str(getattr(receipt, "detail", ""))[:300]
                if on_step is not None:
                    on_step(step, receipt)
            if step.status != "failed":
                continue
            if not step.required:
                continue  # optional: noted, not fatal
            fresh = replan(plan, step) if replan is not None else None
            if fresh is None:
                for later in plan.steps[index:]:
                    if later.status == "planned":
                        later.status, later.detail = "skipped", f"not run: {step.step} failed first"
                break
            plan.replans = fresh.replans
            plan.missing = list(dict.fromkeys(plan.missing + fresh.missing))
            for later in plan.steps[index:]:
                if later.status == "planned":
                    later.status, later.detail = "skipped", "replaced by the replan"
            plan.steps.extend(fresh.steps)
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
