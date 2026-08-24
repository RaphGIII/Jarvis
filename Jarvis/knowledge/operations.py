"""Doing things to the knowledge graph by asking.

"Connect this note to the project", "what contradicts what", "how are these two
related" -- these are graph operations, and the temptation is to let a model
perform them by generating whatever it likes.  That is how a personal knowledge
base quietly acquires nodes nobody wrote and edges nobody meant.

So the split here is the same one that runs through the rest of this system:
the model decides *what* the user wants and which nodes they mean; deterministic
code decides whether that is allowed and then does it.  Every operation is a
real method with real arguments, and the model's only job is to pick one and
name its targets.

Two consequences worth stating.

*Destructive operations are separated from the rest.*  ``delete`` is reachable
only when the caller passes ``confirm=True``, because "remove the old notes"
from a model that mildly misunderstood is not recoverable, and an undo stack
for a SQLite graph is a bigger promise than this needs to make.

*Resolution is reported, not assumed.*  When "the chess project" matches three
nodes, the operation says so and does nothing rather than picking one.  A wrong
guess here writes a wrong edge that nobody will notice until it misleads a
retrieval months later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Operations the model may request, and what each needs.  This IS the
#: vocabulary -- an operation not listed here cannot be requested, which is what
#: stops "do whatever seems right" from being an available action.
OPERATIONS = {
    "open": ("show one node in full", ("target",)),
    "neighbours": ("show what a node is connected to", ("target",)),
    "connect": ("create a relationship between two nodes", ("target", "other")),
    "disconnect": ("remove a relationship between two nodes", ("target", "other")),
    "note": ("create a new note", ("title",)),
    "append": ("add text to an existing note", ("target", "text")),
    "rename": ("change a node's title", ("target", "title")),
    "tag": ("add a tag to a node", ("target", "text")),
    "path": ("show how two nodes are connected", ("target", "other")),
    "contradictions": ("find notes that appear to disagree", ()),
    "search": ("find nodes matching a query", ("text",)),
    "delete": ("remove a node permanently", ("target",)),
}

#: Operations that cannot be undone.  Gated separately from the rest.
DESTRUCTIVE = frozenset({"delete", "disconnect"})


@dataclass
class OperationResult:
    ok: bool = False
    operation: str = ""
    detail: str = ""
    #: Nodes the operation acted on or found.
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    #: Set when a reference matched several nodes and nothing was done.
    ambiguous: list[dict[str, Any]] = field(default_factory=list)
    needs_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "operation": self.operation,
            "detail": self.detail,
            "nodes": self.nodes,
            "edges": self.edges,
            "ambiguous": self.ambiguous,
            "needs_confirmation": self.needs_confirmation,
        }


_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": sorted(OPERATIONS)},
        "target": {"type": "string"},
        "other": {"type": "string"},
        "title": {"type": "string"},
        "text": {"type": "string"},
        "relation": {"type": "string"},
    },
    "required": ["operation"],
}


class GraphOperator:
    """Natural-language operations on the knowledge graph."""

    def __init__(self, graph: Any, *, brain: Any = None) -> None:
        self.graph = graph
        self.brain = brain

    # -- resolving what the user meant -----------------------------------

    def resolve(self, reference: str, *, selected: str = "") -> tuple[Any, list[Any]]:
        """Find the node a phrase refers to.

        Returns ``(node, candidates)``.  When ``node`` is None and
        ``candidates`` is non-empty the reference was ambiguous and the caller
        must ask rather than choose -- a wrong guess writes a wrong edge that
        nobody notices until it misleads a retrieval months later.
        """

        reference = (reference or "").strip()

        # A node the user has selected in the UI wins over anything textual:
        # "connect this to the project" means the thing they are looking at.
        if selected and reference.lower() in {"", "this", "it", "that", "the selected node", "here"}:
            node = self.graph.get(selected)
            if node is not None:
                return node, []

        if not reference:
            return None, []

        # An exact id is unambiguous by construction.
        node = self.graph.get(reference)
        if node is not None:
            return node, []

        # Then an exact title, checked directly rather than through search.
        # Search is lexical and skips short or common words, so a node called
        # "A" or "Ollama" could be unfindable by its own name -- which is the
        # one way a user is guaranteed to refer to it.
        for node_type in self._node_types():
            found = self.graph.find_by_title(node_type, reference)
            if found is not None:
                return found, []

        hits = self.graph.search(reference, limit=6)
        exact = [hit.node for hit in hits if hit.node.title.strip().lower() == reference.lower()]
        if len(exact) == 1:
            return exact[0], []
        if exact:
            return None, exact

        strong = [hit.node for hit in hits]
        if len(strong) == 1:
            return strong[0], []
        return None, strong[:5]

    @staticmethod
    def _node_types() -> list[Any]:
        from knowledge.graph import NodeType

        return list(NodeType)

    # -- interpreting a request ------------------------------------------

    def interpret(self, request: str, *, selected: str = "") -> dict[str, Any]:
        """Turn a sentence into an operation and its arguments.

        The model chooses from a closed list. It cannot invent an operation,
        which is what keeps "do whatever seems right" from being available.
        """

        if self.brain is None:
            return {"operation": "search", "text": request}

        catalogue = "\n".join(f"  {name}: {purpose}" for name, (purpose, _) in sorted(OPERATIONS.items()))
        prompt = (
            "Return JSON only. Decide which knowledge-graph operation the user is asking for.\n"
            "Use ONLY these operations:\n"
            f"{catalogue}\n\n"
            "'target' and 'other' name nodes, by title or by the word 'this' for the selected one.\n"
            "If nothing fits, use 'search'.\n\n"
            f"Selected node: {selected or '(none)'}\n"
            f"Request: {request}\n"
        )
        try:
            payload = json.loads(self.brain.generate_structured(prompt, _INTENT_SCHEMA, max_tokens=250))
        except Exception:
            return {"operation": "search", "text": request}
        if payload.get("operation") not in OPERATIONS:
            return {"operation": "search", "text": request}
        return payload

    # -- performing it ---------------------------------------------------

    def perform(
        self,
        request: str,
        *,
        selected: str = "",
        confirm: bool = False,
    ) -> OperationResult:
        """Interpret a request and carry it out."""

        intent = self.interpret(request, selected=selected)
        return self.execute(intent, selected=selected, confirm=confirm, request=request)

    def execute(
        self,
        intent: dict[str, Any],
        *,
        selected: str = "",
        confirm: bool = False,
        request: str = "",
    ) -> OperationResult:
        operation = str(intent.get("operation", "")).strip()
        if operation not in OPERATIONS:
            return OperationResult(operation=operation, detail=f"unknown operation {operation!r}")

        result = OperationResult(operation=operation)

        if operation in DESTRUCTIVE and not confirm:
            # Not refused -- described, and offered back for confirmation.
            # "Remove the old notes" from a model that mildly misunderstood is
            # not recoverable, and an undo stack is a bigger promise than this
            # needs to make.
            result.needs_confirmation = True
            result.detail = f"{operation} is permanent; confirm to proceed"
            node, candidates = self.resolve(str(intent.get("target", "")), selected=selected)
            if node is not None:
                result.nodes = [node.to_dict()]
            elif candidates:
                result.ambiguous = [item.to_dict() for item in candidates]
            return result

        handler = getattr(self, f"_do_{operation}", None)
        if handler is None:  # pragma: no cover - OPERATIONS and methods agree
            return OperationResult(operation=operation, detail="not implemented")

        try:
            return handler(intent, selected=selected, request=request)
        except Exception as exc:
            return OperationResult(operation=operation, detail=f"{type(exc).__name__}: {exc}")

    # -- the operations themselves ---------------------------------------

    def _target(self, intent: dict[str, Any], key: str, selected: str) -> tuple[Any, OperationResult | None]:
        node, candidates = self.resolve(str(intent.get(key, "")), selected=selected)
        if node is not None:
            return node, None
        failure = OperationResult(operation=str(intent.get("operation", "")))
        if candidates:
            failure.ambiguous = [item.to_dict() for item in candidates]
            failure.detail = f"{intent.get(key)!r} matches {len(candidates)} nodes; which one?"
        else:
            failure.detail = f"nothing matches {intent.get(key)!r}"
        return None, failure

    def _do_open(self, intent, *, selected, request):
        node, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        detail = self.graph.node_detail(node.id)
        return OperationResult(
            ok=True, operation="open", detail=f"opened {node.title}",
            nodes=[detail["node"]],
            edges=[item["edge"] for item in detail["outgoing"] + detail["incoming"]],
        )

    def _do_neighbours(self, intent, *, selected, request):
        node, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        neighbours = self.graph.neighbours(node.id, depth=1, limit=40)
        return OperationResult(
            ok=True, operation="neighbours",
            detail=f"{len(neighbours)} connected to {node.title}",
            nodes=[item.to_dict() for item in neighbours],
        )

    def _do_connect(self, intent, *, selected, request):
        source, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        target, failure = self._target(intent, "other", selected)
        if failure:
            return failure
        if source.id == target.id:
            return OperationResult(operation="connect", detail="a node cannot be connected to itself")

        edge_type = self._relation(intent.get("relation"))
        edge = self.graph.link(source.id, target.id, edge_type)
        return OperationResult(
            ok=True, operation="connect",
            detail=f"{source.title} -> {target.title} ({edge_type.value})",
            nodes=[source.to_dict(), target.to_dict()],
            edges=[edge.to_dict()] if edge else [],
        )

    def _do_disconnect(self, intent, *, selected, request):
        source, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        target, failure = self._target(intent, "other", selected)
        if failure:
            return failure

        removed = 0
        for edge in self.graph.edges_from(source.id):
            if edge.target == target.id:
                self.graph.delete_edge(edge.id)
                removed += 1
        for edge in self.graph.edges_from(target.id):
            if edge.target == source.id:
                self.graph.delete_edge(edge.id)
                removed += 1
        return OperationResult(
            ok=removed > 0, operation="disconnect",
            detail=f"removed {removed} link(s)" if removed else "they were not connected",
            nodes=[source.to_dict(), target.to_dict()],
        )

    def _do_note(self, intent, *, selected, request):
        from knowledge.graph import NodeType

        title = str(intent.get("title") or "").strip() or (request or "note")[:80]
        body = str(intent.get("text") or "").strip()
        node = self.graph.remember(NodeType.NOTE, title, body, provenance="user")
        return OperationResult(ok=True, operation="note", detail=f"created {node.title}", nodes=[node.to_dict()])

    def _do_append(self, intent, *, selected, request):
        node, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        text = str(intent.get("text") or "").strip()
        if not text:
            return OperationResult(operation="append", detail="nothing to append")
        node.body = (node.body + "\n\n" + text).strip()
        self.graph.add_node(node)
        return OperationResult(ok=True, operation="append", detail=f"appended to {node.title}", nodes=[node.to_dict()])

    def _do_rename(self, intent, *, selected, request):
        node, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        title = str(intent.get("title") or "").strip()
        if not title:
            return OperationResult(operation="rename", detail="no new title given")
        was = node.title
        node.title = title
        self.graph.add_node(node)
        return OperationResult(ok=True, operation="rename", detail=f"{was} -> {title}", nodes=[node.to_dict()])

    def _do_tag(self, intent, *, selected, request):
        node, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        tag = str(intent.get("text") or "").strip()
        if not tag:
            return OperationResult(operation="tag", detail="no tag given")
        if tag not in node.tags:
            node.tags.append(tag)
            self.graph.add_node(node)
        return OperationResult(ok=True, operation="tag", detail=f"tagged {node.title} with {tag}", nodes=[node.to_dict()])

    def _do_path(self, intent, *, selected, request):
        source, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        target, failure = self._target(intent, "other", selected)
        if failure:
            return failure

        route = self._shortest_path(source.id, target.id)
        if not route:
            return OperationResult(
                operation="path", detail=f"no connection between {source.title} and {target.title}"
            )
        nodes = [self.graph.get(node_id) for node_id in route]
        return OperationResult(
            ok=True, operation="path",
            detail=" -> ".join(node.title for node in nodes if node),
            nodes=[node.to_dict() for node in nodes if node],
        )

    def _do_contradictions(self, intent, *, selected, request):
        from research.agent import _opposed

        nodes = self.graph.nodes(limit=300)
        found: list[dict[str, Any]] = []
        for index, first in enumerate(nodes):
            for second in nodes[index + 1 :]:
                if _opposed(first.title, second.title):
                    found.append({"a": first.to_dict(), "b": second.to_dict()})
                    if len(found) >= 10:
                        break
            if len(found) >= 10:
                break
        return OperationResult(
            ok=True, operation="contradictions",
            detail=f"{len(found)} apparent disagreement(s)",
            nodes=[item["a"] for item in found] + [item["b"] for item in found],
        )

    def _do_search(self, intent, *, selected, request):
        query = str(intent.get("text") or intent.get("target") or request or "").strip()
        hits = self.graph.search(query, limit=12) if query else []
        return OperationResult(
            ok=bool(hits), operation="search",
            detail=f"{len(hits)} match(es) for {query!r}",
            nodes=[hit.node.to_dict() for hit in hits],
        )

    def _do_delete(self, intent, *, selected, request):
        node, failure = self._target(intent, "target", selected)
        if failure:
            return failure
        title = node.title
        self.graph.delete_node(node.id)
        return OperationResult(ok=True, operation="delete", detail=f"deleted {title}")

    # -- helpers ---------------------------------------------------------

    def _relation(self, name: Any) -> Any:
        from knowledge.graph import EdgeType

        wanted = str(name or "").strip().lower()
        for candidate in EdgeType:
            if candidate.value == wanted or candidate.name.lower() == wanted:
                return candidate
        return EdgeType.RELATES_TO

    def _shortest_path(self, start: str, end: str, *, limit: int = 6) -> list[str]:
        """Breadth-first, so the answer is the shortest route rather than any route."""

        if start == end:
            return [start]
        seen = {start}
        queue: list[list[str]] = [[start]]
        while queue:
            route = queue.pop(0)
            if len(route) > limit:
                continue
            current = route[-1]
            for edge in self.graph.edges_from(current) + self.graph.edges_to(current):
                for neighbour in (edge.target, edge.source):
                    if neighbour in seen or neighbour == current:
                        continue
                    if neighbour == end:
                        return route + [neighbour]
                    seen.add(neighbour)
                    queue.append(route + [neighbour])
        return []
