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

    def system_prompt(self, *, base: str = "") -> str:
        """Assemble the system prompt for this persona."""

        parts: list[str] = []
        if base:
            parts.append(base.strip())
        if self.character:
            parts.append(self.character.strip())
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


def builtin_personas() -> dict[str, Persona]:
    """A deliberately short list: enough to prove the architecture works."""

    return {
        "jarvis": Persona(
            name="jarvis",
            description="The default: a capable, direct engineering assistant.",
            character=(
                "You are JARVIS, an autonomous engineering assistant running locally on the user's own machine. "
                "You understand goals, decompose them, use tools, verify results with real evidence, "
                "and learn from what worked and what did not."
            ),
            style="concise and technical",
        ),
        "mentor": Persona(
            name="mentor",
            description="Explains the reasoning, for when the user is learning.",
            character=(
                "You are JARVIS in teaching mode. Explain why, not just what. "
                "Name the trade-offs you considered and the one you chose."
            ),
            style="patient and explanatory",
        ),
        "terse": Persona(
            name="terse",
            description="Answers only, for when the user already knows the context.",
            character="You are JARVIS. Answer in as few words as the question allows. No preamble.",
            style="minimal",
        ),
        "jarvis_de": Persona(
            name="jarvis_de",
            description="The default persona, always answering in German.",
            character=(
                "Du bist JARVIS, ein autonomer Engineering-Assistent, der lokal auf dem Rechner des Nutzers laeuft. "
                "Du verstehst Ziele, zerlegst sie, benutzt Werkzeuge und pruefst Ergebnisse mit echten Belegen."
            ),
            style="praezise und technisch",
            language="German",
        ),
    }


class PersonaStore:
    """Loads, saves and selects personas, including per-project overrides."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._personas: dict[str, Persona] = builtin_personas()
        self._active = "jarvis"
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
        if isinstance(active, str) and active in self._personas:
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
        self._active = name
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

    def system_prompt(self, *, project_id: str | None = None, base: str = "") -> str:
        return self.active(project_id=project_id).system_prompt(base=base)
