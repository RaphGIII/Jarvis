"""The knowledge graph as something a client can actually draw.

A visualisation needs different things from a query engine: every edge it is
given must have both endpoints present, or it draws a line into nothing; and
"everything about project X" has to mean the neighbourhood of X rather than the
nodes whose text happens to match, or the constellation comes back with no
lines in it at all.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from knowledge.graph import EdgeType, KnowledgeGraph, NodeType
from service.core import JarvisCore
from service.http import JarvisHTTPServer


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
    unrelated = graph.note("Sourdough", "bread notes")
    graph.link(jarvis.id, ollama.id, EdgeType.DEPENDS_ON)
    graph.link(jarvis.id, whisper.id, EdgeType.DEPENDS_ON)
    return graph, {"jarvis": jarvis, "ollama": ollama, "whisper": whisper, "unrelated": unrelated}


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def test_an_empty_graph_exports_cleanly(graph):
    payload = graph.export()

    assert payload["nodes"] == [] and payload["edges"] == []


def test_everything_is_exported_when_no_query_is_given(populated):
    graph, nodes = populated

    payload = graph.export()

    assert len(payload["nodes"]) == 4
    assert len(payload["edges"]) == 2


def test_a_query_returns_a_neighbourhood_not_just_the_matches(populated):
    """The interesting part of "everything about X" is what X connects to."""

    graph, nodes = populated

    payload = graph.export(query="Jarvis")

    titles = {node["title"] for node in payload["nodes"]}
    assert "Project Jarvis" in titles
    assert "Ollama" in titles, "a match with no neighbours is not a constellation"
    assert payload["edges"], "the links are the point"


def test_an_unrelated_node_is_left_out_of_a_focused_export(populated):
    graph, nodes = populated

    payload = graph.export(query="Jarvis")

    assert "Sourdough" not in {node["title"] for node in payload["nodes"]}


def test_every_exported_edge_has_both_endpoints_present(populated):
    """Otherwise the client draws a line into nothing."""

    graph, _ = populated

    payload = graph.export(query="Jarvis")
    present = {node["id"] for node in payload["nodes"]}

    for edge in payload["edges"]:
        assert edge["source"] in present
        assert edge["target"] in present


def test_a_limit_is_respected_and_reported(graph):
    for index in range(30):
        graph.note(f"Note {index}", "body")

    payload = graph.export(limit=10)

    assert len(payload["nodes"]) == 10
    assert payload["truncated"] is True


def test_an_export_that_fits_is_not_marked_truncated(populated):
    graph, _ = populated

    assert graph.export(limit=100)["truncated"] is False


def test_exported_nodes_carry_what_the_view_needs(populated):
    graph, _ = populated

    node = graph.export()["nodes"][0]

    for key in ("id", "title", "type", "body"):
        assert key in node


# --------------------------------------------------------------------------
# Node inspection
# --------------------------------------------------------------------------

def test_a_node_reports_both_directions(populated):
    graph, nodes = populated

    detail = graph.node_detail(nodes["ollama"].id)

    assert detail["incoming"], "Ollama is depended on by Jarvis"
    assert detail["degree"] == 1


def test_the_inspector_names_the_node_at_the_other_end(populated):
    graph, nodes = populated

    detail = graph.node_detail(nodes["jarvis"].id)

    titles = {item["node"]["title"] for item in detail["outgoing"]}
    assert titles == {"Ollama", "Whisper"}


def test_the_edge_type_is_available_for_the_label(populated):
    graph, nodes = populated

    detail = graph.node_detail(nodes["jarvis"].id)

    assert detail["outgoing"][0]["edge"]["type"] == EdgeType.DEPENDS_ON.value


def test_an_unknown_node_says_so_rather_than_raising(graph):
    assert "error" in graph.node_detail("nope")


def test_an_isolated_node_has_degree_zero(populated):
    graph, nodes = populated

    assert graph.node_detail(nodes["unrelated"].id)["degree"] == 0


# --------------------------------------------------------------------------
# Over the service
# --------------------------------------------------------------------------

class StubKernel:
    def __init__(self, state_root):
        self.state_root = state_root
        self.catalog = type("C", (), {"get": staticmethod(lambda t: type("S", (), {"model": "stub"})())})()

    def provider(self, tier):
        raise RuntimeError("not needed")


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "state"
    (root / "knowledge").mkdir(parents=True)
    with KnowledgeGraph(root / "knowledge" / "graph.db") as seeded:
        first = seeded.note("Project Jarvis", "the system")
        second = seeded.note("Ollama", "runtime")
        seeded.link(first.id, second.id, EdgeType.DEPENDS_ON)

    core = JarvisCore(kernel=StubKernel(root))
    instance = JarvisHTTPServer(core, port=0, token="tok")
    instance.start()
    yield instance
    instance.stop()


def call(server, path, body):
    request = urllib.request.Request(
        f"http://{server.host}:{server.port}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Jarvis-Token": "tok"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode())


def test_the_graph_is_served_to_the_client(server):
    payload = call(server, "/api/knowledge/graph", {"limit": 100})

    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 1


def test_a_node_can_be_inspected_over_http(server):
    graph_payload = call(server, "/api/knowledge/graph", {"limit": 100})
    node_id = graph_payload["nodes"][0]["id"]

    detail = call(server, "/api/knowledge/node", {"id": node_id})

    assert detail["node"]["id"] == node_id


def test_a_missing_graph_reports_an_error_rather_than_crashing(tmp_path):
    core = JarvisCore(kernel=StubKernel(tmp_path / "nothing-here"))

    payload = core.knowledge_graph()

    assert payload["nodes"] == []


def test_the_graph_script_is_served(server):
    with urllib.request.urlopen(f"http://{server.host}:{server.port}/graph.js", timeout=10) as response:
        body = response.read().decode()

    assert "KnowledgeStarfield" in body


def test_the_page_loads_the_graph_script(server):
    url = f"http://{server.host}:{server.port}/?token=tok"
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode()

    assert 'src="graph.js"' in body
    assert 'id="graphCanvas"' in body
