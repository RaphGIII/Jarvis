"""Experience memory: what Jarvis learned by doing, stored so it can be found.

The project store already keeps everything about *one* project.  This module is
about the value that only appears *across* projects: that a particular library
worked, that a particular approach did not, that a capability now exists.  It
writes those into the knowledge graph, where they can be retrieved by a future
project that has never heard of the one that learned them.

The distinction that makes this useful rather than decorative is what gets
recorded.  Not the conversation -- dumping transcripts into a prompt is how
context windows die without improving answers.  What gets recorded is:

* **outcomes**, with whether they worked and what the evidence was,
* **lessons**, especially from failures, because avoiding a known dead end is
  worth more than rediscovering a known success,
* **capabilities**, so "can I already do this?" is a lookup rather than a guess.

Retrieval is scoped to the question being asked, so a prompt gets the six things
that matter rather than everything that ever happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge.graph import EdgeType, KnowledgeGraph, Node, NodeType
from projects.models import Project, TaskStatus


@dataclass
class Lesson:
    """One transferable thing learned, with the evidence that supports it."""

    text: str
    worked: bool
    evidence: str = ""
    source_project: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "worked": self.worked,
            "evidence": self.evidence,
            "source_project": self.source_project,
            "tags": list(self.tags),
        }


class ExperienceMemory:
    """Writes finished work into the graph, and reads it back for new work."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    # -- writing ---------------------------------------------------------

    def record_project(self, project: Project) -> Node:
        """Fold a project's durable knowledge into the graph.

        Called when a session ends, whatever the outcome.  A failed project is
        often more informative than a successful one, so nothing is skipped on
        the grounds of not having worked.
        """

        project_node = self.graph.remember(
            NodeType.PROJECT,
            title=project.title or project.goal[:70],
            body=project.goal,
            tags=[project.kind, project.state.value.lower()],
            metadata={
                "project_id": project.id,
                "state": project.state.value,
                "workspace": project.workspace,
                "steps_spent": project.steps_spent,
                "accepted": project.state.value == "COMPLETED",
            },
            provenance=f"project:{project.id}",
        )

        for decision in project.decisions:
            node = self.graph.remember(
                NodeType.DECISION,
                title=decision.text[:90],
                body=f"{decision.text}\n\nRationale: {decision.rationale}",
                tags=["decision"],
                metadata={"alternatives": decision.alternatives},
                provenance=f"project:{project.id}",
            )
            self.graph.link(project_node, node, EdgeType.PRODUCED)

        for experiment in project.experiments:
            node = self.graph.remember(
                NodeType.EXPERIMENT,
                title=experiment.hypothesis[:90] or "experiment",
                body=(
                    f"Hypothesis: {experiment.hypothesis}\n"
                    f"Method: {experiment.method}\n"
                    f"Outcome: {experiment.outcome}\n"
                    f"Lesson: {experiment.lesson}"
                ),
                tags=["experiment", "succeeded" if experiment.succeeded else "failed"],
                metadata={"succeeded": experiment.succeeded},
                provenance=f"project:{project.id}",
            )
            self.graph.link(project_node, node, EdgeType.PRODUCED)

        # Findings sourced from real observation are worth keeping; a model's
        # unverified guesses are not, and mixing them would poison retrieval.
        for finding in project.findings:
            if finding.source not in {"observation", "research", "engine"} or finding.confidence < 0.8:
                continue
            node = self.graph.remember(
                NodeType.NOTE,
                title=finding.text[:90],
                body=finding.text,
                tags=["finding", finding.source],
                metadata={"confidence": finding.confidence},
                provenance=finding.reference or f"project:{project.id}",
            )
            self.graph.link(project_node, node, EdgeType.MENTIONS)

        for reference in project.references:
            node = self.graph.remember(
                NodeType.SOURCE,
                title=str(reference.get("title") or reference.get("url", ""))[:90],
                body=str(reference.get("summary", "")),
                tags=["reference"],
                metadata={"url": reference.get("url"), "retrieved_at": reference.get("retrieved_at")},
                provenance=str(reference.get("url", "")),
            )
            self.graph.link(project_node, node, EdgeType.DERIVED_FROM)

        for artifact in project.artifacts:
            node = self.graph.remember(
                NodeType.FILE,
                title=artifact.path,
                body=artifact.description,
                tags=["artifact"],
                metadata={"kind": artifact.kind, "workspace": project.workspace},
                provenance=f"project:{project.id}",
            )
            self.graph.link(project_node, node, EdgeType.PRODUCED)

        for lesson in self.lessons_from(project):
            self.record_lesson(lesson, project_node)

        return project_node

    def record_lesson(self, lesson: Lesson, project_node: Node | None = None) -> Node:
        node = self.graph.remember(
            NodeType.CONCEPT,
            title=lesson.text[:90],
            body=lesson.text + (f"\n\nEvidence: {lesson.evidence}" if lesson.evidence else ""),
            tags=["lesson", "worked" if lesson.worked else "failed", *lesson.tags],
            metadata={"worked": lesson.worked, "source_project": lesson.source_project},
            provenance=lesson.source_project or "experience",
        )
        if project_node is not None:
            self.graph.link(project_node, node, EdgeType.PRODUCED)
        return node

    def record_capability(
        self, capability_id: str, description: str, *, keywords: list[str] | None = None, **metadata: Any
    ) -> Node:
        """Register a capability so a future goal can find it.

        ``keywords`` matter more than they look.  Lexical retrieval can bridge
        inflection ("play"/"plays") but not vocabulary: a capability described
        as "plays an audio file" is unreachable from "play some music", because
        nothing lexical connects *music* to *audio*.  A real embedding model
        would bridge it, and :class:`~knowledge.graph.OllamaEmbedder` is
        supported for when one is configured -- but a capability that states the
        words people will actually use to ask for it does not need one.
        """

        terms = list(keywords or [])
        return self.graph.remember(
            NodeType.CAPABILITY,
            title=capability_id,
            body=" ".join([description, *terms]).strip(),
            tags=["capability", *terms],
            metadata={**metadata, "keywords": terms, "description": description},
            provenance="capability_registry",
        )

    @staticmethod
    def lessons_from(project: Project) -> list[Lesson]:
        """Extract the transferable lessons from a finished project."""

        lessons: list[Lesson] = []

        for experiment in project.experiments:
            if experiment.lesson:
                lessons.append(
                    Lesson(
                        text=experiment.lesson,
                        worked=experiment.succeeded,
                        evidence=experiment.outcome[:400],
                        source_project=project.id,
                        tags=(project.kind,),
                    )
                )

        for task in project.tasks:
            if task.status is TaskStatus.ABANDONED and task.last_error:
                lessons.append(
                    Lesson(
                        text=f"{task.title} could not be done: {task.last_error[:200]}",
                        worked=False,
                        evidence=task.last_error[:400],
                        source_project=project.id,
                        tags=(project.kind, "abandoned"),
                    )
                )

        if project.state.value == "COMPLETED":
            passing = [item for item in project.objective_criteria() if item.satisfied]
            if passing:
                lessons.append(
                    Lesson(
                        text=f"{project.goal[:150]} was achieved and verified",
                        worked=True,
                        evidence="; ".join(" ".join(item.check) for item in passing)[:400],
                        source_project=project.id,
                        tags=(project.kind, "solution"),
                    )
                )
        return lessons

    # -- reading ---------------------------------------------------------

    def relevant(self, goal: str, *, limit: int = 6) -> list[Node]:
        return self.graph.context_for(goal, limit=limit)

    def prior_failures(self, goal: str, *, limit: int = 5) -> list[Node]:
        """Dead ends already found, so they are not walked into again."""

        hits = self.graph.search(goal, limit=limit * 4)
        failures = [hit.node for hit in hits if "failed" in hit.node.tags]
        return failures[:limit]

    def prior_solutions(self, goal: str, *, limit: int = 5) -> list[Node]:
        hits = self.graph.search(goal, limit=limit * 4)
        solutions = [hit.node for hit in hits if "worked" in hit.node.tags or "solution" in hit.node.tags]
        return solutions[:limit]

    def known_capabilities(self, goal: str, *, limit: int = 5) -> list[Node]:
        return [hit.node for hit in self.graph.search(goal, type=NodeType.CAPABILITY, limit=limit)]

    def brief_for(self, goal: str, *, limit: int = 5) -> str:
        """A compact, prompt-ready summary of what past work says about ``goal``.

        Bounded and sectioned rather than a dump: a small model reads the top of
        a prompt far more reliably than the middle, so the few genuinely useful
        lines have to be short and labelled.
        """

        sections: list[str] = []

        capabilities = self.known_capabilities(goal, limit=limit)
        if capabilities:
            sections.append(
                "CAPABILITIES YOU ALREADY HAVE:\n"
                + "\n".join(f"  - {node.title}: {node.body[:120]}" for node in capabilities)
            )

        solutions = self.prior_solutions(goal, limit=limit)
        if solutions:
            sections.append(
                "WHAT WORKED BEFORE:\n" + "\n".join(f"  - {node.title[:140]}" for node in solutions)
            )

        failures = self.prior_failures(goal, limit=limit)
        if failures:
            sections.append(
                "WHAT FAILED BEFORE (do not repeat):\n" + "\n".join(f"  - {node.title[:140]}" for node in failures)
            )

        return "\n\n".join(sections)

    def export(self, path: str | Path) -> Path:
        """Write the whole graph out as JSON, for backup or inspection."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "stats": self.graph.stats(),
            "nodes": [node.to_dict() for node in self.graph.nodes(limit=1_000_000)],
            "edges": [
                edge.to_dict()
                for node in self.graph.nodes(limit=1_000_000)
                for edge in self.graph.edges_from(node.id)
            ],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return target
