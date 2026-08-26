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
    # The owner's personality reaches every route through this one function,
    # so FAST_LOCAL chat and BUILD_LOCAL work describe the same character.
    try:
        from owner.core import current as owner_core

        personality = "\n" + owner_core().personality_prompt() + "\n"
    except Exception:
        personality = ""
    return f"\nYou are {identity.assistant_name}, this user's personal AI system.\n{personality}{BEHAVIOUR_PROMPT}"


def __getattr__(name: str):
    """``SYSTEM_PROMPT`` resolved on access rather than frozen at import.

    Kept as a module attribute so the several existing ``from config import
    SYSTEM_PROMPT`` call sites keep working, but computed through
    :func:`system_prompt` so that changing the identity changes the prompt.
    """

    if name == "SYSTEM_PROMPT":
        return system_prompt()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")