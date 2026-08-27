"""What the product is called, in one place.

"Jarvis" was a placeholder and the name is now ZEUS.  The wrong way to do that
is a global find-and-replace: module paths, class names, state directories and
saved project files all contain the old string, and renaming them would break
compatibility with everything already on disk to change what a user reads on a
screen.

So identity is a *setting*, not a spelling.  Three names, each doing a different
job:

``product_name``
    The system, in headings and documents.  "ZEUS".
``assistant_name``
    Who the user is talking to, in conversation and speech.  "Zeus".
``wake_word``
    What is said out loud to get its attention.

Internal names stay as they are.  ``service/core.py`` still defines
``JarvisCore``, the state directory is still ``data/jarvis``, and nothing on
disk has to move -- because none of that is user-facing, and churning it would
be risk spent on nothing.

The one thing this cannot make true by configuration is the wake word.  A
wake-word detector is a trained model, not a string: openWakeWord ships
``hey_jarvis`` and has never heard of "Zeus", so setting ``wake_word`` here
changes what the UI says while :attr:`wake_model` decides what is actually
detected.  :meth:`Identity.wake_word_available` reports whether the two agree,
so the gap is visible rather than a surprise at the microphone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

#: Wake models openWakeWord ships pretrained.  Anything else has to be trained.
BUILTIN_WAKE_MODELS = ("hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy", "timer", "weather")


def trained_wake_model_path(word: str) -> Path:
    """Where :mod:`speech.wake_training` writes the classifier for ``word``."""

    return Path(__file__).resolve().parent.parent / "data" / "models" / "wake" / f"{word}.npz"


def trained_wake_model_exists(word: str) -> bool:
    return trained_wake_model_path(word).is_file()


@dataclass(frozen=True)
class Identity:
    """The names the user sees and says."""

    product_name: str = "ZEUS"
    assistant_name: str = "Zeus"
    #: What the user says out loud.  Aspirational until a matching model exists.
    wake_word: str = "Zeus"
    #: The openWakeWord model actually used for detection.  Empty means "pick
    #: one that matches the wake word if there is one".
    wake_model: str = ""
    #: Shown under the name in the interface, when there is room.
    tagline: str = "personal AI"
    #: Where the identity came from, for diagnostics.
    source: str = "defaults"

    # -- the wake word gap -----------------------------------------------

    @property
    def resolved_wake_model(self) -> str:
        """The model that will actually be listened for."""

        if self.wake_model:
            return self.wake_model
        candidate = self.wake_word.strip().lower().replace(" ", "_")
        for name in BUILTIN_WAKE_MODELS:
            if name == candidate or name == f"hey_{candidate}":
                return name
        # A model trained here (speech.wake_training) for exactly this word.
        if trained_wake_model_exists(candidate):
            return candidate
        # No trained model for this word. hey_jarvis is what exists, and saying
        # so is better than silently listening for nothing.
        return "hey_jarvis"

    @property
    def wake_word_available(self) -> bool:
        """Whether the spoken wake word matches the model that is listening."""

        model = self.resolved_wake_model
        spoken = self.wake_word.strip().lower().replace(" ", "_")
        return model in {spoken, f"hey_{spoken}"}

    def wake_word_note(self) -> str:
        """A plain sentence about the gap, or "" when there is none."""

        if self.wake_word_available:
            return ""
        return (
            f'No wake-word model exists for "{self.wake_word}". '
            f'Listening for "{self.resolved_wake_model.replace("_", " ")}" instead. '
            "Train a custom model with openWakeWord and set wake_model to use the real name."
        )

    # -- how the assistant refers to itself ------------------------------

    def persona_preamble(self) -> str:
        """The identity sentence that opens a system prompt."""

        return (
            f"You are {self.assistant_name}, this user's personal AI system. "
            "You are not a chat assistant demo and you do not describe yourself as a "
            "language model."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_name": self.product_name,
            "assistant_name": self.assistant_name,
            "wake_word": self.wake_word,
            "wake_model": self.resolved_wake_model,
            "wake_word_available": self.wake_word_available,
            "wake_word_note": self.wake_word_note(),
            "tagline": self.tagline,
            "source": self.source,
        }

    # -- construction ----------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        config_dir: str | Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> "Identity":
        """Defaults, then ``config/identity.json``, then the environment."""

        identity = cls()
        source = "defaults"

        path = Path(config_dir or _default_config_dir()) / "identity.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                fields = {
                    key: str(value)
                    for key, value in data.items()
                    if key in _FIELDS and isinstance(value, str)
                }
                if fields:
                    identity = replace(identity, **fields)
                    source = str(path)

        env = os.environ if environ is None else environ
        overrides = {
            key: env[f"JARVIS_{key.upper()}"]
            for key in _FIELDS
            if f"JARVIS_{key.upper()}" in env
        }
        if overrides:
            identity = replace(identity, **overrides)
            source = f"{source}+env"

        return replace(identity, source=source)


_FIELDS = ("product_name", "assistant_name", "wake_word", "wake_model", "tagline")


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


#: The process-wide identity.  A module-level default rather than a singleton
#: class: every call site can accept an Identity for testing, and this is just
#: what they get when nobody passes one.
_current: Identity | None = None


def current() -> Identity:
    global _current
    if _current is None:
        _current = Identity.load()
    return _current


def set_current(identity: Identity) -> Identity:
    """Override the process identity.  Used by tests and by the launcher."""

    global _current
    _current = identity
    return identity
