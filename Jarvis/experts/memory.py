"""Remembering how a hard problem was solved, so it is not bought twice.

Escalation costs subscription quota that cannot be replaced when it runs out.
An expert solving a problem is therefore an expensive event, and the expensive
part is not the tokens -- it is that the *approach* existed nowhere before and
will exist nowhere afterwards unless something writes it down.

So a solved escalation records more than the answer:

*What the local model tried and how it failed.*  This is the half usually
thrown away, and it is the half that stops the next attempt from walking into
the same wall. "Threading deadlocked here" is worth more to a future run than
the working code, because the working code will need adapting anyway and the
dead end will not.

*What actually worked, and how it was proved.*  Not the expert's description of
its work -- the acceptance commands and their exit codes, which are the only
part that was ever verified.

Retrieval happens *before* escalation, not after. A future task that matches a
remembered lesson gets it in the local model's context and may well succeed
without an expert at all, which is the entire point: the subscription buys a
lesson once and the system keeps it.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Lesson:
    """One problem, solved the hard way, written down."""

    #: Coarse class from :func:`experts.escalation.classify_goal`.
    task_class: str
    goal: str
    #: What the local model tried, and how each attempt failed.
    failed_approaches: list[str] = field(default_factory=list)
    #: What the expert did, in its own words.  Context, not proof.
    successful_approach: str = ""
    #: Files the solution touched, so a similar task knows where to look.
    files: list[str] = field(default_factory=list)
    #: Commands that were run, including the ones that verified it.
    commands: list[str] = field(default_factory=list)
    #: (criterion, passed) for each acceptance check Jarvis re-ran itself.
    verification: list[dict[str, Any]] = field(default_factory=list)
    #: The reusable shape, if one was stated.
    pattern: str = ""
    provider: str = ""
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def verified(self) -> bool:
        """Whether Jarvis independently confirmed this worked.

        An unverified lesson is a rumour: the expert said it worked, nothing
        checked, and teaching that to a future run would spread the mistake.
        """

        return bool(self.verification) and all(item.get("passed") for item in self.verification)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verified"] = self.verified
        return data

    def as_context(self) -> str:
        """The lesson as something to put in a prompt."""

        lines = [f"A previous task of this kind ({self.task_class}) was solved as follows.", ""]
        lines.append(f"Goal: {self.goal}")
        if self.failed_approaches:
            lines.append("")
            lines.append("These approaches were tried locally and FAILED -- do not repeat them:")
            lines += [f"  - {item}" for item in self.failed_approaches[:6]]
        if self.successful_approach:
            lines += ["", "What worked:", f"  {self.successful_approach[:900]}"]
        if self.files:
            lines += ["", f"Files involved: {', '.join(self.files[:10])}"]
        if self.pattern:
            lines += ["", f"Reusable pattern: {self.pattern[:400]}"]
        return "\n".join(lines)


class ExpertMemory:
    """Lessons from escalations, stored as JSONL and retrieved by similarity.

    Lexical retrieval rather than embeddings: the corpus is tens of entries, not
    thousands, and a deterministic overlap score is inspectable when it picks
    the wrong lesson. An embedding model would add a dependency and a failure
    mode to solve a problem this size does not have.
    """

    #: How many distinctive terms a goal must share with a lesson before it is
    #: recalled. One is coincidence; two is a signal.
    MIN_TERMS = 2

    def __init__(self, path: str | Path, *, keep: int = 500) -> None:
        self.path = Path(path)
        self.keep = keep
        self._lock = threading.Lock()
        self._lessons: list[Lesson] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines[-self.keep :]:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            data.pop("verified", None)   # derived, never stored as truth
            try:
                self._lessons.append(Lesson(**data))
            except TypeError:
                continue

    def record(self, lesson: Lesson) -> Lesson:
        with self._lock:
            self._lessons.append(lesson)
            del self._lessons[: max(0, len(self._lessons) - self.keep)]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(lesson)) + "\n")
            except OSError:
                pass
        return lesson

    def all(self, *, task_class: str = "") -> list[Lesson]:
        with self._lock:
            return [
                item for item in self._lessons
                if not task_class or item.task_class == task_class
            ]

    def recall(self, goal: str, *, task_class: str = "", limit: int = 2) -> list[Lesson]:
        """Lessons most likely to help with this goal.

        Only verified ones. An unverified lesson is what the expert claimed, and
        putting a claim into a future prompt as though it were knowledge is how
        one bad answer becomes several.
        """

        # Every verified lesson is a candidate, whatever its class. Filtering by
        # class first would make cross-class recall impossible, and a debugging
        # lesson genuinely can help a build task that fails the same way --
        # class is a scoring signal, not a gate.
        candidates = [item for item in self.all() if item.verified]
        if not candidates:
            return []

        wanted = _terms(goal)
        if len(wanted) < self.MIN_TERMS:
            return []

        scored = []
        for lesson in candidates:
            lesson_terms = _terms(lesson.goal)
            shared = wanted & lesson_terms
            if not shared:
                continue
            # One shared word is usually coincidence, so two are normally
            # required. The exception is a lesson whose goal is short enough
            # that its every distinctive term is present -- "add a retry" has
            # only "retry" to offer, and demanding two would make short lessons
            # permanently unreachable.
            if len(shared) < self.MIN_TERMS and shared != lesson_terms:
                continue
            bonus = 2 if task_class and lesson.task_class == task_class else 0
            scored.append((len(shared) + bonus, lesson))

        scored.sort(key=lambda pair: -pair[0])
        return [lesson for _, lesson in scored[:limit]]

    def context_for(self, goal: str, *, task_class: str = "", limit: int = 2) -> str:
        """Remembered lessons as prompt text, or "" when there are none.

        Returning empty rather than a heading with nothing under it matters: a
        prompt that says "previous lessons:" followed by nothing invites the
        model to invent some.
        """

        lessons = self.recall(goal, task_class=task_class, limit=limit)
        if not lessons:
            return ""
        return "\n\n".join(lesson.as_context() for lesson in lessons) + "\n\n"

    def summary(self) -> dict[str, Any]:
        lessons = self.all()
        return {
            "lessons": len(lessons),
            "verified": sum(1 for item in lessons if item.verified),
            "classes": sorted({item.task_class for item in lessons}),
        }


def lesson_from_escalation(
    *,
    goal: str,
    task_class: str,
    result: Any,
    failed_approaches: list[str] | None = None,
    pattern: str = "",
) -> Lesson:
    """Build a lesson from an :class:`~experts.contracts.ExpertResult`.

    Verification is taken from what Jarvis re-ran, never from the provider's
    own account of its work -- which is recorded as context and nothing more.
    """

    return Lesson(
        task_class=task_class,
        goal=goal,
        failed_approaches=list(failed_approaches or []),
        successful_approach=str(getattr(result, "summary", ""))[:2000],
        files=list(getattr(result, "files_changed", []) or []),
        commands=list(getattr(result, "commands_run", []) or []),
        verification=[
            {"criterion": text, "passed": bool(passed)}
            for text, passed, _output in (getattr(result, "test_evidence", []) or [])
        ],
        pattern=pattern,
        provider=str(getattr(result, "provider", "")),
    )


#: Words that carry no signal about WHAT a task was about.  Pronouns and
#: placeholder nouns matter as much as articles here: "please can you fix the
#: thing" and "please can you add the other thing" share "you" and "thing" and
#: nothing else, and recalling one for the other puts noise in the prompt.
_STOPWORDS = frozenset(
    """the a an and or of to in on for with from by is are was were be been it this that
    these those as at into make made add adds added fix fixes fixed use uses using do does
    please can could should would jarvis you your yours we our us me my mine they them
    thing things stuff item items other others some any more most very just also then
    something anything everything one two new old""".split()
)


def _terms(text: str) -> set[str]:
    """The words in a goal that say what it is about.

    A goal here is a sentence of subject followed by a contract -- payload
    keys, return shapes, "never fabricate", "it is checked from outside". The
    contract is written in the same house style for every capability, so two
    goals with nothing in common share most of it.

    Measured live on 2026-08-26, recalling for a goal about packaging a folder
    into a zip: the two lessons returned were both about capturing the screen,
    matched on *absent, absolute, accept, bytes, checked, choose, error, every,
    exists, fabricate, failure, false, file, int, its, location, merely, must,
    never, not, optional* -- twenty terms, every one of them contract
    vocabulary and not one about zips or screens. The screen-capture lesson was
    then prepended to the build brief, and the model spent its first cycles
    diagnosing a test called ``test_capture_screen`` that had nothing to do
    with the task.

    That is the same substitution the capability registry was repaired for in
    ``e26f723``: term overlap standing in for "is about the same thing". So the
    same two defences apply here -- read the subject sentence rather than the
    whole brief, and drop the vocabulary every contract contains.
    """

    from capabilities.registry import BOILERPLATE, _subject_sentence

    words = re.findall(r"[a-z0-9_]{3,}", _subject_sentence(text).lower())
    return {word for word in words if word not in _STOPWORDS and word not in BOILERPLATE}
