"""Operating on the knowledge graph by asking, without it filling up with junk.

The temptation is to let a model perform graph edits directly. That is how a
personal knowledge base quietly acquires nodes nobody wrote and edges nobody
meant. Here the model picks an operation from a closed list and names its
targets; deterministic code decides whether that is allowed and does it.

The two properties that keep it honest: an ambiguous reference does nothing and
says so, and a destructive operation needs confirmation.
"""

from __future__ import annotations

import json

import pytest

from knowledge.graph import EdgeType, KnowledgeGraph, NodeType
from knowledge.operations import DESTRUCTIVE, OPERATIONS, GraphOperator, OperationResult


@pytest.fixture()
def graph(tmp_path):
    instance = KnowledgeGraph(tmp_path / "graph.db")
    yield instance
    instance.close()


@pytest.fixture()
def populated(graph):
    jarvis = graph.note("Project Jarvis", "the personal AI system")
    ollama = graph.note("Ollama", "local model runtime")
    whisper = graph.note("Whisper", "speech recognition")
    bread = graph.note("Sourdough", "unrelated baking notes")
    graph.link(jarvis.id, ollama.id, EdgeType.DEPENDS_ON)
    graph.link(ollama.id, whisper.id, EdgeType.RELATES_TO)
    return graph, {"jarvis": jarvis, "ollama": ollama, "whisper": whisper, "bread": bread}


class ScriptedBrain:
    """Returns whatever intent the test scripted."""

    def __init__(self, intent):
        self.intent = intent
        self.prompts = []

    def generate_structured(self, prompt, schema, **_):
        self.prompts.append(prompt)
        return json.dumps(self.intent)


# --------------------------------------------------------------------------
# The vocabulary is closed
# --------------------------------------------------------------------------

def test_the_model_cannot_invent_an_operation(populated):
    graph, _ = populated
    operator = GraphOperator(graph, brain=ScriptedBrain({"operation": "drop_everything"}))

    intent = operator.interpret("do whatever seems right")

    assert intent["operation"] in OPERATIONS


def test_an_unusable_model_falls_back_to_searching(populated):
    class Broken:
        def generate_structured(self, *a, **k):
            raise RuntimeError("model down")

    graph, _ = populated
    operator = GraphOperator(graph, brain=Broken())

    assert operator.interpret("anything")["operation"] == "search"


def test_every_listed_operation_has_a_handler(populated):
    graph, _ = populated
    operator = GraphOperator(graph)

    for name in OPERATIONS:
        assert hasattr(operator, f"_do_{name}"), f"{name} has no implementation"


# --------------------------------------------------------------------------
# Ambiguity does nothing, loudly
# --------------------------------------------------------------------------

def test_an_ambiguous_reference_writes_nothing(graph):
    """A wrong guess writes a wrong edge nobody notices for months."""

    graph.note("Chess board detection", "a")
    graph.note("Chess engine integration", "b")
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "open", "target": "chess"})

    assert not result.ok
    assert result.ambiguous
    assert "which one" in result.detail


def test_an_exact_title_is_not_ambiguous(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "open", "target": "Ollama"})

    assert result.ok


def test_an_id_resolves_directly(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "open", "target": nodes["ollama"].id})

    assert result.ok


def test_a_reference_matching_nothing_says_so(populated):
    graph, _ = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "open", "target": "nothing like this exists"})

    assert not result.ok
    assert "nothing matches" in result.detail


def test_this_means_the_selected_node(populated):
    """"Connect this to the project" means the thing they are looking at."""

    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute(
        {"operation": "open", "target": "this"}, selected=nodes["whisper"].id
    )

    assert result.ok
    assert result.nodes[0]["title"] == "Whisper"


# --------------------------------------------------------------------------
# Destructive operations are gated
# --------------------------------------------------------------------------

def test_delete_needs_confirmation(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "delete", "target": "Sourdough"})

    assert result.needs_confirmation
    assert not result.ok
    assert graph.get(nodes["bread"].id) is not None, "nothing may be deleted without confirmation"


def test_the_confirmation_prompt_names_what_would_go(populated):
    graph, _ = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "delete", "target": "Sourdough"})

    assert result.nodes and result.nodes[0]["title"] == "Sourdough"


def test_delete_with_confirmation_removes_it(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "delete", "target": "Sourdough"}, confirm=True)

    assert result.ok
    assert graph.get(nodes["bread"].id) is None


def test_disconnect_is_also_gated(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "disconnect", "target": "Project Jarvis", "other": "Ollama"})

    assert result.needs_confirmation
    assert graph.edges_from(nodes["jarvis"].id), "the link must still be there"


def test_the_destructive_set_matches_what_is_gated():
    assert DESTRUCTIVE <= set(OPERATIONS)


# --------------------------------------------------------------------------
# The operations
# --------------------------------------------------------------------------

def test_connect_creates_a_relationship(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "connect", "target": "Whisper", "other": "Project Jarvis"})

    assert result.ok
    assert graph.edges_from(nodes["whisper"].id)


def test_connect_honours_a_named_relation(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    operator.execute(
        {"operation": "connect", "target": "Whisper", "other": "Project Jarvis", "relation": "part_of"}
    )

    assert graph.edges_from(nodes["whisper"].id)[0].type is EdgeType.PART_OF


def test_an_unknown_relation_falls_back_rather_than_failing(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    operator.execute(
        {"operation": "connect", "target": "Whisper", "other": "Ollama", "relation": "is_vaguely_about"}
    )

    assert graph.edges_from(nodes["whisper"].id)[0].type is EdgeType.RELATES_TO


def test_a_node_cannot_be_connected_to_itself(populated):
    graph, _ = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "connect", "target": "Ollama", "other": "Ollama"})

    assert not result.ok


def test_disconnect_with_confirmation_removes_the_link(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute(
        {"operation": "disconnect", "target": "Project Jarvis", "other": "Ollama"}, confirm=True
    )

    assert result.ok
    assert not graph.edges_from(nodes["jarvis"].id)


def test_disconnecting_unrelated_nodes_says_so(populated):
    graph, _ = populated
    operator = GraphOperator(graph)

    result = operator.execute(
        {"operation": "disconnect", "target": "Sourdough", "other": "Whisper"}, confirm=True
    )

    assert not result.ok
    assert "not connected" in result.detail


def test_a_note_can_be_created(graph):
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "note", "title": "A new idea", "text": "the body"})

    assert result.ok
    assert graph.find_by_title(NodeType.NOTE, "A new idea") is not None


def test_text_can_be_appended(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    operator.execute({"operation": "append", "target": "Ollama", "text": "and it is fast"})

    assert "and it is fast" in graph.get(nodes["ollama"].id).body


def test_appending_nothing_is_refused(populated):
    graph, _ = populated

    assert not GraphOperator(graph).execute({"operation": "append", "target": "Ollama", "text": ""}).ok


def test_a_node_can_be_renamed(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    result = operator.execute({"operation": "rename", "target": "Ollama", "title": "Ollama runtime"})

    assert result.ok
    assert graph.get(nodes["ollama"].id).title == "Ollama runtime"


def test_a_node_can_be_tagged(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    operator.execute({"operation": "tag", "target": "Ollama", "text": "infrastructure"})

    assert "infrastructure" in graph.get(nodes["ollama"].id).tags


def test_tagging_twice_does_not_duplicate(populated):
    graph, nodes = populated
    operator = GraphOperator(graph)

    operator.execute({"operation": "tag", "target": "Ollama", "text": "infra"})
    operator.execute({"operation": "tag", "target": "Ollama", "text": "infra"})

    assert graph.get(nodes["ollama"].id).tags.count("infra") == 1


def test_neighbours_are_listed(populated):
    graph, _ = populated

    result = GraphOperator(graph).execute({"operation": "neighbours", "target": "Ollama"})

    assert result.ok
    assert {item["title"] for item in result.nodes} >= {"Project Jarvis", "Whisper"}


def test_search_finds_nodes(populated):
    graph, _ = populated

    result = GraphOperator(graph).execute({"operation": "search", "text": "speech"})

    assert result.ok
    assert any("Whisper" in item["title"] for item in result.nodes)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def test_a_path_between_two_connected_nodes(populated):
    """Jarvis -> Ollama -> Whisper, found without being told the route."""

    graph, _ = populated

    result = GraphOperator(graph).execute(
        {"operation": "path", "target": "Project Jarvis", "other": "Whisper"}
    )

    assert result.ok
    assert [item["title"] for item in result.nodes] == ["Project Jarvis", "Ollama", "Whisper"]


def test_no_path_is_reported_rather_than_invented(populated):
    graph, _ = populated

    result = GraphOperator(graph).execute(
        {"operation": "path", "target": "Sourdough", "other": "Whisper"}
    )

    assert not result.ok
    assert "no connection" in result.detail


def test_the_path_is_the_shortest_one(graph):
    a = graph.note("A", "")
    b = graph.note("B", "")
    c = graph.note("C", "")
    d = graph.note("D", "")
    graph.link(a.id, b.id)
    graph.link(b.id, d.id)
    graph.link(a.id, c.id)
    graph.link(c.id, b.id)

    result = GraphOperator(graph).execute({"operation": "path", "target": "A", "other": "D"})

    assert len(result.nodes) == 3, "A -> B -> D, not the longer way round"


# --------------------------------------------------------------------------
# End to end through the model
# --------------------------------------------------------------------------

def test_a_sentence_becomes_an_operation(populated):
    graph, nodes = populated
    brain = ScriptedBrain({"operation": "connect", "target": "this", "other": "Project Jarvis"})
    operator = GraphOperator(graph, brain=brain)

    result = operator.perform("connect this to the Jarvis project", selected=nodes["whisper"].id)

    assert result.ok
    assert graph.edges_from(nodes["whisper"].id)


def test_the_selected_node_is_offered_to_the_model(populated):
    graph, nodes = populated
    brain = ScriptedBrain({"operation": "open", "target": "this"})
    operator = GraphOperator(graph, brain=brain)

    operator.perform("open this", selected=nodes["ollama"].id)

    assert nodes["ollama"].id in brain.prompts[0]


def test_a_failing_operation_reports_rather_than_raises(populated):
    graph, _ = populated

    class Exploding:
        def get(self, *a, **k):
            raise RuntimeError("database on fire")

        def search(self, *a, **k):
            raise RuntimeError("database on fire")

    result = GraphOperator(Exploding()).execute({"operation": "open", "target": "x"})

    assert not result.ok
    assert "on fire" in result.detail


def test_the_result_serialises_for_the_ui():
    payload = OperationResult(ok=True, operation="connect", detail="a -> b").to_dict()

    assert payload["operation"] == "connect"
    assert "needs_confirmation" in payload
