from __future__ import annotations

import json
import subprocess
import sys

import pytest

from knowledge.graph import EdgeType, KnowledgeGraph, LexicalEmbedder, Node, NodeType
from knowledge.memory import ExperienceMemory, Lesson
from projects.models import Project, ProjectState, TaskStatus


@pytest.fixture
def graph(tmp_path):
    with KnowledgeGraph(tmp_path / "palace.sqlite") as instance:
        yield instance


# ------------------------------------------------------------------ embedder


def test_the_embedder_is_stable_across_processes(tmp_path):
    """Vectors written today must still match after a restart.

    Python randomises string hashing per process, so an embedder built on
    hash() would silently stop matching its own stored vectors -- the one thing
    a persistent index must never do.
    """

    script = (
        "import sys, json; sys.path.insert(0, r'"
        + str(tmp_path.parent.parent)
        + "');"
        "from knowledge.graph import LexicalEmbedder;"
        "print(json.dumps(LexicalEmbedder().embed('screen capture chess board')))"
    )
    here = LexicalEmbedder().embed("screen capture chess board")
    completed = subprocess.run(
        [sys.executable, "-c", script.replace(str(tmp_path.parent.parent), _package_root())],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == here


def _package_root():
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = LexicalEmbedder()
    target = embedder.embed("capture the screen and recognise a chess board")
    close = embedder.embed("recognise a chess board from a screen capture")
    far = embedder.embed("compute payroll deductions for quarterly tax filing")
    similarity = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert similarity(target, close) > similarity(target, far)


# ------------------------------------------------------------------ nodes


def test_nodes_persist_across_reopening(tmp_path):
    path = tmp_path / "palace.sqlite"
    with KnowledgeGraph(path) as graph:
        node = graph.note("Stockfish path", "stockfish.exe lives in C:/tools", tags=["chess"])
        node_id = node.id

    with KnowledgeGraph(path) as reopened:
        loaded = reopened.get(node_id)
        assert loaded is not None
        assert loaded.title == "Stockfish path"
        assert loaded.tags == ["chess"]


def test_remember_merges_instead_of_duplicating(graph):
    first = graph.remember(NodeType.CONCEPT, "screen capture", "use mss")
    second = graph.remember(NodeType.CONCEPT, "Screen Capture", "it is fast", tags=["vision"])
    assert first.id == second.id
    assert "use mss" in second.body and "it is fast" in second.body
    assert graph.nodes(type=NodeType.CONCEPT) and len(graph.nodes(type=NodeType.CONCEPT)) == 1


def test_provenance_and_confidence_are_kept(graph):
    node = graph.note("pytest exits 5 when no tests are collected", provenance="docs.pytest.org", confidence=0.95)
    assert graph.get(node.id).provenance == "docs.pytest.org"
    assert graph.get(node.id).confidence == pytest.approx(0.95)


# ------------------------------------------------------------------ edges


def test_edges_are_queryable_from_both_ends(graph):
    project = graph.remember(NodeType.PROJECT, "chess overlay")
    concept = graph.remember(NodeType.CONCEPT, "board recognition")
    graph.link(project, concept, EdgeType.MENTIONS)

    assert [edge.target for edge in graph.edges_from(project.id)] == [concept.id]
    assert [edge.source for edge in graph.edges_to(concept.id)] == [project.id]
    assert [node.id for node in graph.backlinks(concept.id)] == [project.id]


def test_duplicate_links_are_idempotent(graph):
    a = graph.remember(NodeType.NOTE, "a")
    b = graph.remember(NodeType.NOTE, "b")
    graph.link(a, b, EdgeType.RELATES_TO)
    graph.link(a, b, EdgeType.RELATES_TO, weight=2.0)
    edges = graph.edges_from(a.id)
    assert len(edges) == 1 and edges[0].weight == 2.0


def test_deleting_a_node_leaves_no_dangling_edges(graph):
    a = graph.remember(NodeType.NOTE, "a")
    b = graph.remember(NodeType.NOTE, "b")
    graph.link(a, b)
    graph.delete_node(b.id)
    assert graph.edges_from(a.id) == []
    assert graph.get(b.id) is None


def test_neighbours_walks_both_directions(graph):
    a = graph.remember(NodeType.PROJECT, "root")
    b = graph.remember(NodeType.CONCEPT, "child")
    c = graph.remember(NodeType.SOURCE, "grandchild")
    graph.link(a, b)
    graph.link(c, b)  # points the other way

    identifiers = {node.id for node in graph.neighbours(a.id, depth=2)}
    assert b.id in identifiers and c.id in identifiers


# ------------------------------------------------------------------ search


def test_keyword_search_finds_the_thing_you_named(graph):
    graph.note("Stockfish engine integration", "connect via UCI over stdin")
    graph.note("Payroll tax rules", "quarterly filing")
    hits = graph.search_keyword("stockfish")
    assert hits and hits[0].node.title == "Stockfish engine integration"


def test_semantic_search_finds_the_thing_you_described(graph):
    graph.note("Stockfish engine integration", "connect to the chess engine using the UCI protocol")
    graph.note("Payroll tax rules", "quarterly filing deadlines for employers")
    hits = graph.search_semantic("chess engine protocol")
    assert hits and hits[0].node.title == "Stockfish engine integration"


def test_merged_search_rewards_agreement_between_both_signals(graph):
    both = graph.note("chess engine UCI protocol", "connect to the chess engine over UCI")
    keyword_only = graph.note("chess", "a board game")
    hits = {hit.node.id: hit for hit in graph.search("chess engine UCI protocol")}
    assert hits[both.id].score > hits[keyword_only.id].score
    assert hits[both.id].how == "keyword+semantic"


def test_search_can_be_restricted_to_a_type(graph):
    graph.remember(NodeType.CAPABILITY, "audio.play_file", "plays an audio file")
    graph.remember(NodeType.NOTE, "audio notes", "some thoughts about audio")
    hits = graph.search("audio", type=NodeType.CAPABILITY)
    assert [hit.node.type for hit in hits] == [NodeType.CAPABILITY]


def test_context_includes_the_neighbourhood_of_a_hit(graph):
    decision = graph.remember(NodeType.DECISION, "use mss for screen capture")
    experiment = graph.remember(NodeType.EXPERIMENT, "tried pyautogui screenshot", "too slow at 4 fps")
    graph.link(decision, experiment, EdgeType.DERIVED_FROM)

    titles = {node.title for node in graph.context_for("screen capture")}
    assert "use mss for screen capture" in titles
    assert "tried pyautogui screenshot" in titles, "a decision matters together with the experiment behind it"


def test_changing_the_embedder_is_detected_not_silently_mixed(graph):
    graph.note("something", "content")

    class OtherEmbedder(LexicalEmbedder):
        version = "other-v2"

    graph.embedder = OtherEmbedder()
    assert graph.stats()["stale_vectors"] == 1
    assert graph.reindex() == 1
    assert graph.stats()["stale_vectors"] == 0


# ------------------------------------------------------------------ memory


def _finished_project():
    project = Project(goal="Build a word frequency counter", kind="software")
    project.state = ProjectState.COMPLETED
    project.add_decision("use collections.Counter", rationale="standard library, no dependency")
    project.add_experiment(
        "count words with a manual dict",
        method="write_file",
        outcome="worked but verbose",
        succeeded=True,
        lesson="a manual dict works for word counting",
    )
    project.add_experiment(
        "parse with a regex first",
        method="apply_edits",
        outcome="the anchor never matched",
        succeeded=False,
        lesson="the search anchor must be copied verbatim from the current file",
    )
    project.add_finding("pytest exits 5 when no tests are collected", source="observation", confidence=0.95)
    project.add_finding("I think this might need numpy", source="model", confidence=0.4)
    project.add_artifact("wordfreq.py", description="the implementation")
    criterion = project.add_acceptance("tests pass", check=["python", "-m", "pytest", "-q"])
    criterion.satisfied = True
    task = project.add_task("install a package that does not exist")
    task.status = TaskStatus.ABANDONED
    task.last_error = "install_packages: no matching distribution"
    return project


def test_a_finished_project_is_folded_into_the_graph(graph):
    memory = ExperienceMemory(graph)
    node = memory.record_project(_finished_project())

    linked = {item.title for item in graph.neighbours(node.id, depth=1, limit=50)}
    assert any("collections.Counter" in title for title in linked)
    assert any("wordfreq.py" == title for title in linked)


def test_unverified_model_guesses_are_not_stored_as_facts(graph):
    """Mixing guesses with observations would poison every later retrieval."""

    ExperienceMemory(graph).record_project(_finished_project())
    titles = " ".join(node.title for node in graph.nodes(limit=500))
    assert "pytest exits 5" in titles
    assert "might need numpy" not in titles


def test_failures_are_retrievable_so_they_are_not_repeated(graph):
    memory = ExperienceMemory(graph)
    memory.record_project(_finished_project())

    failures = memory.prior_failures("search anchor did not match the file")
    assert failures
    assert any("verbatim" in node.title for node in failures)


def test_solutions_are_retrievable(graph):
    memory = ExperienceMemory(graph)
    memory.record_project(_finished_project())
    assert memory.prior_solutions("word frequency counter")


def test_the_brief_is_sectioned_and_bounded(graph):
    memory = ExperienceMemory(graph)
    memory.record_project(_finished_project())
    memory.record_capability("text.word_count", "counts words in a string")

    brief = memory.brief_for("count the words in a document")

    assert "CAPABILITIES YOU ALREADY HAVE" in brief
    assert "text.word_count" in brief
    assert len(brief) < 4000, "a brief that does not fit in a small model's attention is not a brief"


def test_knowledge_survives_a_restart(tmp_path):
    """The point of all of it: a later process benefits from earlier work."""

    path = tmp_path / "palace.sqlite"
    with KnowledgeGraph(path) as graph:
        ExperienceMemory(graph).record_project(_finished_project())

    with KnowledgeGraph(path) as reopened:
        memory = ExperienceMemory(reopened)
        assert memory.prior_failures("anchor did not match")
        assert memory.relevant("word frequency")


def test_export_writes_nodes_and_edges(graph, tmp_path):
    memory = ExperienceMemory(graph)
    memory.record_project(_finished_project())
    path = memory.export(tmp_path / "export.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nodes"] and payload["edges"]
    assert payload["stats"]["nodes"] == len(payload["nodes"])


def test_capabilities_are_findable_by_the_words_people_use(graph):
    """A capability must be reachable from how it would actually be asked for.

    Lexical retrieval bridges inflection but not vocabulary: nothing lexical
    connects "music" to "audio". Rather than pretend otherwise, a capability
    declares the words people will use, which is knowledge the acquisition
    pipeline has anyway.
    """

    memory = ExperienceMemory(graph)
    memory.record_capability(
        "audio.play_file",
        "plays an audio file through the system speakers",
        keywords=["music", "song", "sound", "playback"],
    )
    found = memory.known_capabilities("play some music")
    assert found and found[0].title == "audio.play_file"


def test_inflection_alone_does_not_hide_a_capability(graph):
    """The part lexical retrieval genuinely can fix: play / plays / playing."""

    memory = ExperienceMemory(graph)
    memory.record_capability("audio.play_file", "plays an audio file through the speakers")
    assert memory.known_capabilities("play an audio file")
    assert memory.known_capabilities("playing audio files")


def test_the_graph_can_be_used_from_more_than_one_thread(tmp_path):
    """It could not, and the failure was expensive and misread.

    SQLite refuses a connection used from a thread other than the one that
    created it. The graph now lives on a long-lived service object that several
    threads reach: the HTTP handler builds it lazily, the answering thread
    queries it, the acquisition thread writes to it.

    A live capability repair died instantly with "SQLite objects created in a
    thread can only be used in that same thread", the mission counted that as a
    failed local attempt, and escalated over it -- so the local tier never ran
    at all and the ledger recorded a failure it had not earned.
    """

    import threading

    from knowledge.graph import KnowledgeGraph, Node, NodeType

    graph = KnowledgeGraph(tmp_path / "palace.sqlite")
    graph.add_node(Node(type=NodeType.NOTE, title="written on the main thread"))

    failures: list[str] = []

    def worker() -> None:
        try:
            graph.add_node(Node(type=NodeType.NOTE, title="written on a worker"))
            graph.export(limit=10)
        except Exception as exc:  # noqa: BLE001 - the point is to report it
            failures.append(repr(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert failures == []
    assert len(graph.export(limit=10)["nodes"]) == 2
    graph.close()


def test_closing_releases_connections_opened_on_other_threads(tmp_path):
    import threading

    from knowledge.graph import KnowledgeGraph, Node, NodeType

    graph = KnowledgeGraph(tmp_path / "palace.sqlite")

    def worker() -> None:
        graph.add_node(Node(type=NodeType.NOTE, title="worker"))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(graph._connections) >= 2
    graph.close()
    assert graph._connections == []


def test_a_graph_survives_being_closed_and_used_again(tmp_path):
    """close() is not a one-way door; a fresh connection is made on demand."""

    from knowledge.graph import KnowledgeGraph, Node, NodeType

    graph = KnowledgeGraph(tmp_path / "palace.sqlite")
    graph.add_node(Node(type=NodeType.NOTE, title="before"))
    graph.close()

    graph.add_node(Node(type=NodeType.NOTE, title="after"))

    assert len(graph.export(limit=10)["nodes"]) == 2
