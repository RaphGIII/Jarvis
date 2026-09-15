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

_PROVIDER_NAMES = (r"(?:Gemini|Google(?:\s+DeepMind)?|ChatGPT|GPT(?:-?\d[\w.\-]*)?|OpenAI|Claude|Anthropic|Bard|Copilot|Llama|Meta\s+AI|Mistral"
                   r"|Qwen|Ollama|Groq|Cerebras|OpenRouter|DeepSeek|Grok|xAI|Perplexity|Cohere|Alibaba(?:\s+Cloud)?)")
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
# Sentence leads where the assistant speaks AS a vendor: "As Gemini, ...",
# "Als Claude kann ich ...", "As an OpenAI model, ...", "I am Google's AI".
_APPOSITIVE = (rf"(?:\s*,\s*(?:ein|eine|a|an|the)\s+[^,.!?\n]{{0,40}}?(?:Sprachmodell|language\s+model|Modell|model|KI|AI|assistant|Assistent)"
               rf"(?:\s+(?:von|by|from|of)\s+{_PROVIDER_NAMES}(?:\s+(?:DeepMind|AI|Cloud|Inc\.?))?)?)?")
_AS_VENDOR_LEAD = re.compile(
    rf"(?P<lead>(?:^|(?<=[.!?\n]\s)|(?<=^\s))(?:As|Als))\s+(?:(?:an?|ein|eine)\s+)?(?:(?:{_PROVIDER_NAMES})(?:'s|s)?\s+)?"
    rf"(?:(?:AI|KI|language\s+)?(?:model|Modell|assistant|Assistent)\s+)?(?:{_PROVIDER_NAMES})\b(?:'s\s+(?:AI|model))?{_APPOSITIVE}\s*,?",
    re.IGNORECASE,
)
_OWNED_AI = re.compile(rf"(?P<lead>\b(?:I am|I'm|Ich bin|This is|Hier spricht|Hier ist))\s+(?:{_PROVIDER_NAMES})(?:'s|s)?\s+(?:AI|KI|assistant|Assistent|model|Modell)\b",
                       re.IGNORECASE)
# "as an AI model from Google" / "als KI-Modell von OpenAI"
_AS_VENDOR_MODEL = re.compile(
    rf"(?P<lead>\b(?:als|as))\s+(?:ein|eine|a|an)?\s*(?:KI|AI|Sprach|language)?[\w\- ]{{0,20}}?(?:Modell|model|Assistent|assistant)\s+"
    rf"(?:von|from|by|of)\s+{_PROVIDER_NAMES}\b",
    re.IGNORECASE,
)


def identity_clause(assistant: str = "ZEUS", *, owner: str = "") -> str:
    """One compact sentence: who designed and orchestrates ZEUS, and what the engine is not."""

    who = owner or "the owner"
    return (f"\nIdentity: You are {assistant}, designed and orchestrated by {who}; {who} never trained a foundation model. "
            f"Never introduce yourself as a vendor's model or assistant; the engine behind an answer is infrastructure.\n")


def system_prompt_for_role(role: str, *, assistant: str | None = None, extra: str = "") -> str:
    """The compact PersonalityContract plus the identity clause, for a cloud role.

    The same contract every provider receives: identity, character, the
    invariants, the owner's preferences and rules -- a few hundred tokens,
    never the owner's documents read aloud.
    """

    name = assistant
    creator = ""
    if not name:
        try:
            from core.identity import current

            identity = current()
            name, creator = identity.assistant_name, getattr(identity, "creator", "")
        except Exception:  # noqa: BLE001
            name = "ZEUS"
    else:
        try:
            from core.identity import current

            creator = getattr(current(), "creator", "")
        except Exception:  # noqa: BLE001
            creator = ""
    try:
        from persona.contract import current_contract

        base = current_contract(scope="role").text
        if name and not base.startswith(f"You are {name}"):
            base = f"You are {name}. " + base
    except Exception:  # noqa: BLE001 - a persona that cannot load must not block the answer
        base = f"You are {name}, this user's personal AI system."
    text = base.rstrip() + identity_clause(name, owner=creator)
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
    out, n4 = _AS_VENDOR_LEAD.subn(lambda m: f"{m.group('lead')} {assistant},", out)
    out, n5 = _OWNED_AI.subn(lambda m: f"{m.group('lead')} {assistant}", out)
    count += n2 + n3 + n4 + n5
    return out, count


def diagnostics_identity(role: str, provider: str, model: str) -> dict[str, Any]:
    """Where provider identity *is* allowed to show: the technical view."""

    return {"role": role, "provider": provider, "model": model}
