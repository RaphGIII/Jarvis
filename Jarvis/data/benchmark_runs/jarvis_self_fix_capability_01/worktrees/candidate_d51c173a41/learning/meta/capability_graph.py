from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityNode:
    name: str
    competence: float = 0.0
    confidence: float = 0.0
    attempts: int = 0
    dependencies: set[str] = field(default_factory=set)
    learning_progress: float = 0.0


class CapabilityGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, CapabilityNode] = {}

    def add_capability(self, name: str, dependencies: set[str] | None = None) -> CapabilityNode:
        node = self.nodes.setdefault(name, CapabilityNode(name=name))
        if dependencies:
            node.dependencies.update(dependencies)
        return node

    def update(self, name: str, competence: float, confidence: float, learning_progress: float = 0.0) -> CapabilityNode:
        node = self.add_capability(name)
        node.competence = float(competence)
        node.confidence = float(confidence)
        node.learning_progress = float(learning_progress)
        node.attempts += 1
        return node

    def trainable_capabilities(self, dependency_threshold: float = 0.6) -> list[CapabilityNode]:
        trainable = []
        for node in self.nodes.values():
            if all(self.nodes.get(dep, CapabilityNode(dep)).competence >= dependency_threshold for dep in node.dependencies):
                trainable.append(node)
        return trainable

    def next_training_target(self) -> CapabilityNode | None:
        candidates = self.trainable_capabilities()
        if not candidates:
            return None
        return min(candidates, key=lambda node: (node.competence, -node.learning_progress))
