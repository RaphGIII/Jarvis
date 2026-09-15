MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.3

#: Everything in the system prompt that is not the assistant's name.  Split out
#: because this text used to open with "You are JARVIS", and it is attached to
#: the provider -- so it reached *every* inference path, under every persona,
#: regardless of what config/identity.json said.  Renaming the product left this
#: behind, and the running product duly introduced itself as JARVIS.
#:
#: Rule 5 stays, but it is no longer the mechanism.  A model that is asked to
#: confirm will confirm; what actually stops a fabricated success now is that
#: side-effecting requests never reach a free-text generation at all -- see
#: :mod:`service.actions`.  The sentence is kept because it is true and costs
#: nothing, not because it is load-bearing.
BEHAVIOUR_PROMPT = """
Your job is to:
1. Understand the user's goal.
2. Reason about the problem.
3. Create a concrete plan.
4. Decide which capabilities are required.
5. Never claim an action was performed unless it actually was.
6. Learn from successes and failures.
"""


def system_prompt(identity=None) -> str:
    """The provider-level system prompt, named from the current identity."""

    if identity is None:
        from core.identity import current

        identity = current()
    # The owner's personality reaches every route through this one function
    # as the compact PersonalityContract, so chat and engineering describe
    # the same character at a few hundred tokens.
    try:
        from persona.contract import current_contract

        personality = "\n" + current_contract(scope="system", identity=identity).text + "\n"
    except Exception:
        personality = f"\nYou are {identity.assistant_name}, this user's personal AI system.\n"
    return f"{personality}{BEHAVIOUR_PROMPT}"


def conversation_prompt(*, language: str = "", guidance: list[str] | None = None, task_style: str = "",
                        transcript: str = "", text: str = "", assistant: str = "", identity=None) -> str:
    """The ordinary-conversation prompt in its fixed order.

    1. protected identity + core personality (owner core)      -- nothing outranks it
    2. honesty invariants
    3. owner preferences (the dials)
    4. conversation context: language, the owner's corrections
    5. task-specific style (persona style / extra instructions)  -- last, weakest
    then the transcript and the owner's words.
    """

    from core.identity import current as current_identity
    from owner.core import current as owner_core
    from persona.contract import compile_contract

    identity = identity or current_identity()
    contract = compile_contract(assistant=identity.assistant_name, product=identity.product_name, creator=getattr(identity, "creator", ""),
                                personality=owner_core().read("personality"), scope="chat")
    blocks = contract.blocks
    parts = [blocks.get("identity", ""), blocks.get("character", ""), blocks.get("invariants", ""), blocks.get("preferences", ""),
             blocks.get("answers", ""), blocks.get("rules", "")]
    if language:
        from persona.language import language_name

        parts.append(f"The owner is speaking {language_name(language)}; reply in that language.")
    if guidance:
        parts.append("The owner has said, and it applies here:\n" + "\n".join(guidance))
    if task_style:
        parts.append(f"Style for this task (never overrides who you are): {task_style}")
    system = "\n\n".join(p for p in parts if p)
    name = assistant or identity.assistant_name
    return system + "\n\n" + (f"Recent conversation:\n{transcript}\n\n" if transcript else "") + f"user: {text}\n{name}:"


def __getattr__(name: str):
    """``SYSTEM_PROMPT`` resolved on access rather than frozen at import.

    Kept as a module attribute so the several existing ``from config import
    SYSTEM_PROMPT`` call sites keep working, but computed through
    :func:`system_prompt` so that changing the identity changes the prompt.
    """

    if name == "SYSTEM_PROMPT":
        return system_prompt()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")