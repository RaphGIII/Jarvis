"""Personas: who Jarvis is when it talks, kept out of the code that thinks.

The brief asks for the architecture, not for a gallery of characters, and is
explicit that little time should go on writing personalities.  So this module is
small on purpose.  What matters is the separation: nothing in the project
engine, the edit engine or the tool runtime knows a persona exists.  A persona
contributes a system prompt and a few style preferences; it can never change
what a tool is permitted to do or what counts as acceptance.  That boundary is
what stops "be more decisive" from turning into weaker safety gates.

Language is handled the same way -- as a preference, not a feature.  A persona
that names a language adds one line to the system prompt.  Supporting German
therefore costs nothing beyond what the underlying model already knows, and
adding Japanese costs the same nothing, which is the point of doing it here
rather than as a capability per language.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Instructions that hold whatever persona is active.  A persona may set the
#: tone of these sentences but never remove them, because they are the
#: difference between a helpful assistant and a confident liar.
INVARIANT_RULES = (
    "Never claim an action was performed unless it actually was.",
    "Report failures plainly, with the evidence.",
    "Prefer saying you do not know over inventing an answer.",
)


@dataclass
class Persona:
    """A named communication style."""

    name: str
    description: str = ""
    #: Free-text character notes folded into the system prompt.
    character: str = ""
    #: e.g. "concise", "detailed", "formal", "playful".
    style: str = "concise"
    #: BCP-47-ish tag, or "auto" to mirror whatever the user wrote in.
    language: str = "auto"
    #: How the user is addressed, where the language distinguishes it.
    address: str = ""
    #: Extra lines appended verbatim to the system prompt.
    extra_instructions: list[str] = field(default_factory=list)

    def system_prompt(self, *, base: str = "", assistant: str = "") -> str:
        """Assemble the system prompt for this persona.

        ``assistant`` fills the :data:`NAME` placeholder.  Substituted with
        ``replace`` rather than ``str.format`` because a user-defined persona is
        free-text and may legitimately contain braces; a persona that crashed
        the prompt by mentioning a dict would be an absurd failure mode.
        """

        if not assistant:
            from core.identity import current

            assistant = current().assistant_name

        parts: list[str] = []
        if base:
            parts.append(base.strip())
        if self.character:
            parts.append(self.character.strip().replace(NAME, assistant))
        if self.style:
            parts.append(f"Communication style: {self.style}.")
        if self.language and self.language != "auto":
            parts.append(f"Always reply in {self.language}, whatever language the user writes in.")
        else:
            parts.append("Reply in the same language the user writes in.")
        if self.address:
            parts.append(f"Address the user as: {self.address}.")
        parts.extend(self.extra_instructions)
        # Appended last so they are the most recent thing the model reads, and
        # so a persona cannot displace them by being verbose.
        parts.extend(INVARIANT_RULES)
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


#: Written into a persona's character text wherever the assistant's own name
#: belongs.  Substituted at prompt time from :mod:`core.identity`.
#:
#: These personas used to spell the name out: *"You are JARVIS, an autonomous
#: engineering assistant..."*.  The product was renamed to ZEUS by setting
#: :class:`~core.identity.Identity`, and the identity preamble duly said "You
#: are Zeus" -- and then the persona said "You are JARVIS" immediately after it,
#: so the live product introduced itself as JARVIS to a user who asked it who it
#: was.  A name that is configuration in one file and a literal in another is
#: not configuration.
NAME = "{assistant}"


def builtin_personas() -> dict[str, Persona]:
    """A deliberately short list: enough to prove the architecture works."""

    return {
        "default": Persona(
            name="default",
            description="The default: Zeus as the owner's personality document defines him.",
            # Identity and character come from the owner core; a persona only
            # carries task style.  ("an autonomous engineering assistant"
            # used to live here and outranked the owner's document by
            # position -- that sentence is what the owner heard back.)
            character="You are {assistant}.",
            style="natural, concise; technical when the question is technical",
        ),
        "mentor": Persona(
            name="mentor",
            description="Explains the reasoning, for when the user is learning.",
            character=(
                f"You are {NAME} in teaching mode. Explain why, not just what. "
                "Name the trade-offs you considered and the one you chose."
            ),
            style="patient and explanatory",
        ),
        "terse": Persona(
            name="terse",
            description="Answers only, for when the user already knows the context.",
            character=f"You are {NAME}. Answer in as few words as the question allows. No preamble.",
            style="minimal",
        ),
        "default_de": Persona(
            name="default_de",
            description="The default persona, always answering in German.",
            character="Du bist {assistant}.",
            style="natuerlich, knapp; technisch wenn die Frage technisch ist",
            language="German",
        ),
    }


#: Personas that used to carry the old name, so a stored ``active`` selection
#: from before the rename still resolves instead of falling over.
RENAMED = {"jarvis": "default", "jarvis_de": "default_de"}


class PersonaStore:
    """Loads, saves and selects personas, including per-project overrides."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._personas: dict[str, Persona] = builtin_personas()
        self._active = "default"
        #: project id -> persona name.  A long-running project can keep its own
        #: voice without the user re-selecting it every session.
        self._project_overrides: dict[str, str] = {}
        self.load()

    # -- persistence -----------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt persona file must not stop Jarvis starting; the
            # built-ins are always a working fallback.
            return
        for data in payload.get("personas") or []:
            if isinstance(data, dict) and data.get("name"):
                persona = Persona.from_dict(data)
                self._personas[persona.name] = persona
        active = payload.get("active")
        if isinstance(active, str):
            # A selection saved before the rename still names "jarvis".
            active = RENAMED.get(active, active)
            if active in self._personas:
                self._active = active
        overrides = payload.get("project_overrides")
        if isinstance(overrides, dict):
            self._project_overrides = {str(k): str(v) for k, v in overrides.items()}

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": self._active,
            "project_overrides": self._project_overrides,
            # Built-ins are re-created from code on load, so only genuine
            # customisations are written out.
            "personas": [
                persona.to_dict()
                for name, persona in sorted(self._personas.items())
                if name not in builtin_personas() or persona != builtin_personas()[name]
            ],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    # -- selection -------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._personas)

    def get(self, name: str) -> Persona:
        # Old names keep resolving. Renaming a persona must not turn a stored
        # preference, a config file or a --persona flag into a crash.
        name = RENAMED.get(name, name)
        if name not in self._personas:
            raise KeyError(f"unknown persona: {name}. Available: {', '.join(self.names())}")
        return self._personas[name]

    def active(self, *, project_id: str | None = None) -> Persona:
        if project_id and project_id in self._project_overrides:
            override = self._project_overrides[project_id]
            if override in self._personas:
                return self._personas[override]
        return self._personas[self._active]

    def activate(self, name: str) -> Persona:
        persona = self.get(name)
        # The resolved name, not the one asked for: activating "jarvis" and
        # then storing "jarvis" as active would write a name that no longer
        # exists, and the next load would raise on it.
        self._active = persona.name
        self.save()
        return persona

    def set_for_project(self, project_id: str, name: str) -> Persona:
        persona = self.get(name)
        self._project_overrides[project_id] = name
        self.save()
        return persona

    def clear_for_project(self, project_id: str) -> None:
        self._project_overrides.pop(project_id, None)
        self.save()

    def define(self, persona: Persona) -> Persona:
        """Add or replace a persona.  This is how Jarvis gains a new voice."""

        if not persona.name.strip():
            raise ValueError("a persona needs a name")
        self._personas[persona.name] = persona
        self.save()
        return persona

    def system_prompt(self, *, project_id: str | None = None, base: str = "", assistant: str = "") -> str:
        return self.active(project_id=project_id).system_prompt(base=base, assistant=assistant)
