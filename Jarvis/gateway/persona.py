"""ZEUS stays ZEUS, whoever computed the answer.

Two halves.  The system prompt every cloud role receives carries the owner's
persona (the same :func:`config.system_prompt` the local tiers use) plus an
identity clause: the model speaks as ZEUS, does not name the vendor behind it,
and does not claim the owner trained a foundation model.  The output guard
then catches the sentences that slip through anyway -- "Ich bin Gemini, ein
Sprachmodell von Google" -- and rewrites only those, leaving the rest of the
answer untouched.  Provider identity remains available in diagnostics; it is
only the conversational voice that keeps it out.
"""

from __future__ import annotations

import re
from typing import Any

_PROVIDER_NAMES = r"(?:Gemini|Google(?:\s+DeepMind)?|ChatGPT|GPT(?:-?\d[\w.\-]*)?|OpenAI|Claude|Anthropic|Bard|Copilot|Llama|Meta\s+AI|Mistral)"
_SELF = r"(?:I am|I'm|Ich bin|ich bin|I was|Ich wurde|ich wurde|Ich heiße|I, )"

# Sentences where the assistant introduces itself as a vendor's model.
_SELF_ID = re.compile(
    rf"(?P<lead>{_SELF})\s+(?:(?:ein|eine|a|an|the)\s+)?(?:(?:großes|large|multimodales|multimodal|KI-|AI\s+)?"
    rf"(?:Sprachmodell|language\s+model|Modell|model|Assistent|assistant|KI|AI)\s*,?\s*)?"
    rf"(?:namens\s+|called\s+|named\s+)?{_PROVIDER_NAMES}\b[^.!?\n]*",
    re.IGNORECASE,
)
# "... developed/trained/created by Google/OpenAI/Anthropic"
_BY_VENDOR = re.compile(
    rf"(?:,?\s*(?:developed|trained|created|made|built|entwickelt|trainiert|erstellt|gebaut)\s+(?:by|von)\s+{_PROVIDER_NAMES}"
    rf"(?:\s+(?:DeepMind|AI|Inc\.?|LLC))?)",
    re.IGNORECASE,
)
# "as an AI model from Google" / "als KI-Modell von OpenAI"
_AS_VENDOR_MODEL = re.compile(
    rf"(?P<lead>\b(?:als|as))\s+(?:ein|eine|a|an)?\s*(?:KI|AI|Sprach|language)?[\w\- ]{{0,20}}?(?:Modell|model|Assistent|assistant)\s+"
    rf"(?:von|from|by|of)\s+{_PROVIDER_NAMES}\b",
    re.IGNORECASE,
)


def identity_clause(assistant: str = "ZEUS", *, owner: str = "") -> str:
    who = owner or "the owner"
    return (
        f"\nIdentity: You are {assistant}, {who}'s personal AI system, designed and orchestrated by {who}. "
        f"You are one component of {assistant}; the underlying model vendor is an implementation detail. "
        f"Never introduce yourself as, or claim to be, a vendor's model or assistant (Gemini, GPT, ChatGPT, Claude or similar), "
        f"and never name the vendor of the model that produced this answer unless the user explicitly asks a technical question "
        f"about which model is running. Never claim that {who} trained a foundation model; {who} designed and orchestrates {assistant}.\n"
    )


def system_prompt_for_role(role: str, *, assistant: str | None = None, extra: str = "") -> str:
    """The owner's persona prompt plus the identity clause, for a cloud role."""

    try:
        from config import system_prompt

        base = system_prompt()
    except Exception:  # noqa: BLE001 - a persona that cannot load must not block the answer
        base = f"You are {assistant or 'ZEUS'}, this user's personal AI system."
    name = assistant
    if not name:
        try:
            from core.identity import current

            name = current().assistant_name
        except Exception:  # noqa: BLE001
            name = "ZEUS"
    text = base.rstrip() + identity_clause(name)
    if extra:
        text += "\n" + extra.strip() + "\n"
    return text


def guard_identity(text: str, *, assistant: str = "ZEUS") -> tuple[str, int]:
    """Rewrite self-identification as a vendor model.  Returns (text, replacements)."""

    if not text:
        return text, 0
    count = 0

    def _self(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        lead = match.group("lead")
        return f"{lead} {assistant}"

    out, n = _SELF_ID.subn(_self, text)
    out, n2 = _BY_VENDOR.subn("", out)
    out, n3 = _AS_VENDOR_MODEL.subn(lambda m: f"{m.group('lead')} {assistant}", out)
    count += n2 + n3
    return out, count


def diagnostics_identity(role: str, provider: str, model: str) -> dict[str, Any]:
    """Where provider identity *is* allowed to show: the technical view."""

    return {"role": role, "provider": provider, "model": model}
