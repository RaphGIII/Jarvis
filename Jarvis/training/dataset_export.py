"""Turning verified work into training data -- without training on failure.

The brief is explicit that Jarvis must not update weights after every
interaction, and it is right to be: continual online training on your own
output is the fastest known route to a model that confidently repeats its own
mistakes.  What this module does instead is collect the material, so that a
deliberate, offline, reviewable fine-tune is possible later without redesigning
anything.

The quality bar is the whole point:

*Only verified trajectories.*  A sample is exported when the project's objective
acceptance criteria actually passed.  A model's own belief that it did well is
not evidence, and training on it would be training on wishful thinking.

*Failures are kept, but labelled.*  A repair sample -- broken state, diagnosis,
fix that provably worked -- is arguably the most valuable kind of training data
there is, and is completely different from a sample of an unfixed failure.

*Secrets never leave.*  Everything passes through a redactor before it is
written, because a dataset is exactly the artefact people copy around.

Formats are ordinary JSONL: a chat-style ``messages`` form for instruction
tuning, and a richer trajectory form that keeps the tool calls, so a future
agent-style fine-tune has something to work from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from projects.models import Phase, Project, ProjectState, TaskStatus


@dataclass
class TrainingSample:
    """One exportable example, with the evidence that justifies it."""

    #: "solution" (goal to verified result) or "repair" (failure to fix).
    kind: str
    goal: str
    context: str
    response: str
    #: What proves this sample is worth learning from.
    evidence: str = ""
    #: 0..1.  Deterministic, from the objective outcome, never from a model.
    score: float = 0.0
    project_id: str = ""
    tags: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_chat(self, *, system_prompt: str = "") -> dict[str, Any]:
        """The ``messages`` form most instruction-tuning pipelines expect."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"{self.goal}\n\n{self.context}".strip()})
        messages.append({"role": "assistant", "content": self.response})
        return {"messages": messages, "score": self.score, "kind": self.kind, "project_id": self.project_id}


#: Patterns for things that must never reach a dataset file.  Deliberately
#: aggressive: a false positive costs one redacted sample, a false negative
#: costs a leaked credential in a file designed to be shared.
_SECRET_PATTERNS = (
    re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b"),
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,})\b"),
    re.compile(r"\b(AKIA[0-9A-Z]{12,})\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|bearer)\b\s*[:=]\s*[\"']?([^\s\"',]{6,})"),
    re.compile(r"\b(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),
)


def redact(text: str) -> str:
    """Strip anything that looks like a credential."""

    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(lambda match: match.group(0).replace(match.groups()[-1], "<redacted>"), cleaned)
    return cleaned


class DatasetExporter:
    """Builds training samples from projects that objectively succeeded."""

    def __init__(self, *, min_score: float = 0.5) -> None:
        self.min_score = min_score

    # -- extraction ------------------------------------------------------

    def samples_from(self, project: Project) -> list[TrainingSample]:
        """Everything worth learning from one project.

        A project that never proved anything contributes nothing, however busy
        it looked.
        """

        if not project.acceptance_satisfied():
            return []

        samples: list[TrainingSample] = []
        solution = self._solution_sample(project)
        if solution is not None:
            samples.append(solution)
        samples.extend(self._repair_samples(project))
        return [sample for sample in samples if sample.score >= self.min_score]

    def _solution_sample(self, project: Project) -> TrainingSample | None:
        """Goal in, verified result out."""

        completed = [task for task in project.tasks if task.status is TaskStatus.DONE]
        if not completed:
            return None

        evidence = "\n".join(
            f"$ {' '.join(item.check)}\n{item.last_evidence[-600:]}"
            for item in project.objective_criteria()
            if item.satisfied
        )
        response = "\n".join(
            [
                "Plan:",
                *[f"  {index}. {task.title}" for index, task in enumerate(completed, start=1)],
                "",
                "Result:",
                *[f"  - {artifact.path}: {artifact.description}" for artifact in project.artifacts[:10]],
            ]
        )
        return TrainingSample(
            kind="solution",
            goal=redact(project.goal),
            context=redact(self._context_of(project)),
            response=redact(response),
            evidence=redact(evidence),
            score=self._score(project),
            project_id=project.id,
            tags=[project.kind, "verified"],
        )

    def _repair_samples(self, project: Project) -> list[TrainingSample]:
        """Broken state, diagnosis, and the fix that provably worked.

        Only exported from a project that ultimately passed, so the fix is known
        to have been correct rather than merely different.
        """

        samples: list[TrainingSample] = []
        steps = project.steps
        for index, step in enumerate(steps):
            if step.phase is not Phase.DIAGNOSE or not step.success:
                continue
            failure = next((item for item in reversed(steps[:index]) if not item.success), None)
            recovery = next(
                (item for item in steps[index + 1 :] if item.phase is Phase.EXECUTE and item.success), None
            )
            if failure is None or recovery is None:
                continue

            diagnosis = str(step.detail.get("diagnosis", "")).strip()
            fix = str(step.detail.get("fix", "")).strip()
            if not diagnosis:
                continue

            failure_text = json.dumps(failure.detail, default=str)[:1200] or failure.summary
            samples.append(
                TrainingSample(
                    kind="repair",
                    goal=redact(f"Something failed while working on: {project.goal}"),
                    context=redact(f"FAILURE:\n{failure.summary}\n{failure_text}"),
                    response=redact(f"Diagnosis: {diagnosis}\n\nFix: {fix}\n\nApplied: {recovery.summary}"),
                    evidence=redact(recovery.summary),
                    score=self._score(project) * 0.9,
                    project_id=project.id,
                    tags=[project.kind, "repair"],
                    tool_calls=[
                        {"name": call.get("name"), "ok": call.get("ok")} for call in recovery.tool_calls[:6]
                    ],
                )
            )
        return samples

    def _context_of(self, project: Project) -> str:
        parts = []
        if project.constraints:
            parts.append("Constraints:\n" + "\n".join(f"  - {item}" for item in project.constraints))
        objective = project.objective_criteria()
        if objective:
            parts.append(
                "Acceptance:\n" + "\n".join(f"  - {item.text} (check: {' '.join(item.check)})" for item in objective)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _score(project: Project) -> float:
        """A deterministic quality score, from the outcome only.

        Never from a model's self-assessment: the point of the score is to rank
        samples by how much they are worth learning from, and a model grading
        its own homework would make the ranking meaningless.
        """

        objective = project.objective_criteria()
        if not objective:
            return 0.0
        passed = sum(1 for item in objective if item.satisfied) / len(objective)
        if passed < 1.0:
            return 0.0

        # Efficiency is a mild bonus: a goal reached in five steps is a better
        # demonstration than the same goal reached in fifty.
        steps = max(1, project.steps_spent)
        efficiency = max(0.0, min(1.0, 20.0 / steps))
        abandoned = sum(1 for task in project.tasks if task.status is TaskStatus.ABANDONED)
        penalty = min(0.3, abandoned * 0.1)
        return round(max(0.0, min(1.0, 0.7 + 0.3 * efficiency - penalty)), 3)

    # -- writing ---------------------------------------------------------

    def export(
        self,
        projects: Iterable[Project],
        destination: str | Path,
        *,
        chat_format: bool = True,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Write every qualifying sample to a JSONL file."""

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)

        samples: list[TrainingSample] = []
        considered = 0
        for project in projects:
            considered += 1
            samples.extend(self.samples_from(project))

        samples.sort(key=lambda item: item.score, reverse=True)

        with target.open("w", encoding="utf-8") as handle:
            for sample in samples:
                payload = sample.to_chat(system_prompt=system_prompt) if chat_format else sample.to_dict()
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

        by_kind: dict[str, int] = {}
        for sample in samples:
            by_kind[sample.kind] = by_kind.get(sample.kind, 0) + 1

        return {
            "path": str(target),
            "projects_considered": considered,
            "samples_written": len(samples),
            "by_kind": by_kind,
            "mean_score": round(sum(item.score for item in samples) / len(samples), 3) if samples else 0.0,
            "format": "chat" if chat_format else "trajectory",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_from_store(self, store: Any, destination: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Export every completed project a :class:`ProjectStore` holds."""

        projects = [project for project in store.iter_projects() if project.state is ProjectState.COMPLETED]
        return self.export(projects, destination, **kwargs)
