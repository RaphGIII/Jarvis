"""Studium "Sterne": the library as a sky -- documents not chunks, subjects as clusters, capped meaningful edges,
a deterministic and stable layout, and the view's pure geometry (run in node)."""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pytest

from study import galaxy
from study.service import StudyService


def doc(i: int | str, **kw) -> dict:
    base = {"id": f"d{i}", "title": f"Dokument {i}", "filename": f"dokument_{i}.pdf", "source_type": "pdf", "provider": "upload",
            "units": 10, "subject": "", "topic": "", "course": "", "semester": "", "module": "", "confirmed": [],
            "updated_at": 1000.0 + (i if isinstance(i, int) else 0), "opened_at": None, "flags": {}}
    base.update(kw)
    return base


def library(n: int, subjects: int = 8, seed: int = 3) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        s = rnd.randrange(subjects)
        out.append(doc(i, subject=f"Fach {s}" if s else "", topic=f"Thema {rnd.randrange(6)}", course="Medizin", units=rnd.randrange(1, 300)))
    return out


def positions(payload: dict) -> dict[str, tuple[float, float]]:
    return {n["id"]: (n["x"], n["y"]) for n in payload["nodes"]}


# ---------------------------------------------------------------------------------------------------------- nodes


def test_nodes_are_documents_with_the_owners_classification_winning():
    docs = [
        doc(1, subject="Physiologie", confirmed=["subject"], inferred={"subject": "Anatomie"}),
        doc(2, subject="", inferred={"subject": "Biochemie", "topic": "Citratzyklus"}),
        doc(3, subject="", confirmed=["subject"], inferred={"subject": "Anatomie"}),   # the owner cleared it on purpose
        doc(4, source_type="pptx", units=8, flags={"goodnotes": False}),
        doc(5, provider="goodnotes", flags={"goodnotes": True}),
    ]
    payload = galaxy.build(docs + [doc(1, title="doppelt")])
    nodes = {n["id"]: n for n in payload["nodes"]}
    assert len(payload["nodes"]) == 5, "one node per document, duplicates collapsed"
    assert nodes["d1"]["subject"] == "Physiologie" and nodes["d1"]["confirmed"] == ["subject"]
    assert nodes["d2"]["subject"] == "Biochemie" and nodes["d2"]["topic"] == "Citratzyklus", "inferred fills what the owner did not set"
    assert nodes["d3"]["subject"] == "", "a confirmed empty subject is not overridden by an inference"
    assert nodes["d4"]["units"] == 8 and nodes["d4"]["source_type"] == "pptx"
    assert nodes["d5"]["goodnotes"] is True and nodes["d5"]["provider"] == "goodnotes"
    assert all(0 <= n["weight"] <= 1 and n["state"] == "ready" for n in nodes.values())


def test_the_service_draws_documents_not_chunks(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    long_text = "\n\n".join(f"Absatz {i}: Der Frank-Starling-Mechanismus beschreibt die Vorlast und das Schlagvolumen. " * 6 for i in range(30))
    (src / "Herz.md").write_text("# Herz\n\n" + long_text, encoding="utf-8")
    (src / "Niere.md").write_text("# Niere\n\nAldosteron fördert die Natriumrückresorption.\n", encoding="utf-8")
    svc = StudyService(tmp_path / "study", tmp_path / "store", embeddings=False, slide_renderer=False)
    for path in sorted(src.iterdir()):
        assert svc.import_path(path, copy=True)["ok"]
    assert svc.index.stats()["chunks"] > 2, "the long document has many chunks"
    payload = svc.galaxy()
    assert payload["ok"] and len(payload["nodes"]) == 2
    assert {n["title"] for n in payload["nodes"]} == {d["title"] for d in svc.documents()}
    assert payload["stats"]["semantic"] is False, "no provider: no semantic edges, no error"


# ---------------------------------------------------------------------------------------------------------- clusters


def test_clusters_by_subject_then_topic_and_the_unassigned():
    docs = [doc(i, subject="Physiologie", topic="Herz" if i < 8 else "Niere") for i in range(14)]
    docs += [doc(100 + i, subject="Anatomie") for i in range(3)]
    docs += [doc(200), doc(201, subject="  physiologie ")]
    payload = galaxy.build(docs)
    subjects = {c["label"]: c for c in payload["clusters"] if c["kind"] == "subject"}
    assert set(subjects) == {"Physiologie", "Anatomie", "Ohne Zuordnung"}, "labels fold case and spaces into one subject"
    assert len(subjects["Physiologie"]["members"]) == 15
    assert subjects["Ohne Zuordnung"]["members"] == ["d200"]
    topics = {c["label"]: c for c in payload["clusters"] if c["kind"] == "topic"}
    assert set(topics) == {"Herz", "Niere"} and all(t["parent"] == subjects["Physiologie"]["id"] for t in topics.values())
    assert not any(c["kind"] == "topic" and c["parent"] == subjects["Anatomie"]["id"] for c in payload["clusters"]), "a small subject is not divided"
    rest = [c for c in payload["clusters"] if c["kind"] == "rest"]
    assert rest and rest[0]["members"] == ["d201"], "the members without a topic keep a place of their own"
    order = [c["label"] for c in payload["clusters"] if c["kind"] == "subject"]
    assert order[-1] == "Ohne Zuordnung" and order[0] == "Physiologie", "largest first, unassigned last"
    # every member lies inside its disc
    nodes = {n["id"]: n for n in payload["nodes"]}
    for cluster in payload["clusters"]:
        for member in cluster["members"]:
            n = nodes[member]
            assert math.hypot(n["x"] - cluster["x"], n["y"] - cluster["y"]) <= cluster["r"] + 1e-4


def test_subject_discs_do_not_overlap():
    payload = galaxy.build(library(900, subjects=30))
    discs = [c for c in payload["clusters"] if c["kind"] == "subject"]
    for i, a in enumerate(discs):
        for b in discs[i + 1:]:
            assert math.hypot(a["x"] - b["x"], a["y"] - b["y"]) >= a["r"] + b["r"] - 1e-4, (a["label"], b["label"])


# ---------------------------------------------------------------------------------------------------------- edges


def test_edges_are_meaningful_capped_and_never_all_pairs():
    docs = [doc(i, subject="Physiologie", topic="Herz", course="Medizin") for i in range(40)]
    docs += [doc(100 + i, subject="Physiologie", topic="Niere", course="Medizin") for i in range(10)]
    docs += [doc(200 + i) for i in range(5)]   # no subject, no topic: nothing to connect them
    payload = galaxy.build(docs, limit_edges_per_node=3)
    degree: dict[str, int] = {}
    for edge in payload["edges"]:
        assert edge["kind"] in {"topic", "course", "semantic"} and 0 < edge["weight"] <= 1
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1
    assert max(degree.values()) <= 3, "capped per node"
    assert len(payload["edges"]) <= 3 * len(docs) // 2
    assert len(payload["edges"]) < 55 * 54 // 2 // 10, "far from all pairs"
    assert not any(e["source"].startswith("d20") or e["target"].startswith("d20") for e in payload["edges"] if len(e["source"]) == 4)
    assert any(e["kind"] == "topic" for e in payload["edges"])
    pairs = {(e["source"], e["target"]) for e in payload["edges"]}
    assert len(pairs) == len(payload["edges"]) and not any((b, a) in pairs for a, b in pairs), "one edge per pair"


def _concept_vector(concept: int, noise: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(noise)
    v = np.zeros(dim, dtype=np.float32)
    v[concept] = 3.0
    v += rng.normal(scale=0.15, size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_semantic_edges_come_from_centroids_above_a_threshold():
    docs = [doc(i) for i in range(6)]
    vectors = {"d0": _concept_vector(1, 0), "d1": _concept_vector(1, 1), "d2": _concept_vector(1, 2),
               "d3": _concept_vector(5, 3), "d4": _concept_vector(9, 4), "d5": _concept_vector(13, 5)}
    payload = galaxy.build(docs, vectors_by_doc=vectors)
    semantic = [e for e in payload["edges"] if e["kind"] == "semantic"]
    connected = {frozenset((e["source"], e["target"])) for e in semantic}
    assert connected and all(pair <= {"d0", "d1", "d2"} for pair in connected), connected
    nodes = {n["id"]: n for n in payload["nodes"]}
    assert set(nodes["d0"]["similar"]) == {"d1", "d2"} and nodes["d3"]["similar"] == []
    assert payload["stats"]["semantic"] and galaxy.SEMANTIC_FLOOR <= payload["stats"]["semantic_threshold"] < 1, "calibrated against this library's own pairs"
    two = galaxy.build(docs[:2], vectors_by_doc={k: vectors[k] for k in ("d0", "d1")})
    assert two["stats"]["semantic_threshold"] == galaxy.SEMANTIC_FIXED and two["nodes"][0]["similar"] == ["d1"], "two documents: a fixed, cautious threshold"
    # an explicit, impossible threshold: nothing is similar
    assert not [e for e in galaxy.build(docs, vectors_by_doc=vectors, semantic_threshold=0.999)["edges"] if e["kind"] == "semantic"]


def test_centroids_are_the_mean_of_a_documents_chunk_vectors():
    rowids = [10, 11, 12, 13]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    result = galaxy.centroids(rowids, vectors, {10: "a", 11: "a", 12: "b"})   # chunk 13 belongs to no known document
    assert set(result) == {"a", "b"}
    assert np.allclose(result["a"], np.array([1, 1, 0]) / math.sqrt(2)) and np.allclose(result["b"], [0, 0, 1])
    assert galaxy.centroids([], np.zeros((0, 0)), {}) == {}


def test_a_large_library_calibrates_its_threshold_and_stays_top_k():
    docs = library(1200, subjects=12)
    rng = np.random.default_rng(1)
    bases = rng.normal(size=(12, 96)).astype(np.float32)
    vectors = {d["id"]: bases[zlib.crc32(d["subject"].encode()) % 12] + 0.9 * rng.normal(size=96).astype(np.float32) for d in docs}
    payload = galaxy.build(docs, vectors_by_doc=vectors, limit_edges_per_node=3)
    assert galaxy.SEMANTIC_FLOOR <= payload["stats"]["semantic_threshold"] < 1
    assert all(len(n["similar"]) <= galaxy.SEMANTIC_TOP_K for n in payload["nodes"])
    assert len(payload["edges"]) <= 3 * 1200 // 2


# ---------------------------------------------------------------------------------------------------------- layout


def test_layout_is_deterministic_normalised_and_stable_when_a_document_arrives():
    docs = library(240)
    first, again = galaxy.build(docs), galaxy.build([dict(d) for d in docs])
    assert positions(first) == positions(again), "the same library draws the same sky"
    xs = [n["x"] for n in first["nodes"]]
    ys = [n["y"] for n in first["nodes"]]
    assert -1.2 <= min(xs) and max(xs) <= 1.2 and -1.2 <= min(ys) and max(ys) <= 1.2, "roughly [-1, 1]"
    assert max(xs) - min(xs) > 0.8, "and it uses the space"
    grown = galaxy.build(docs + [doc("neu", subject="Fach 1", topic="Thema 2", course="Medizin")])
    before, after = positions(first), positions(grown)
    moved = max(math.hypot(before[k][0] - after[k][0], before[k][1] - after[k][1]) for k in before)
    assert moved < 0.02, f"a new document moves the others by a hair, not a reshuffle ({moved:.4f})"
    shuffled = docs[:]
    random.Random(9).shuffle(shuffled)
    assert positions(galaxy.build(shuffled)) == positions(first), "the order documents arrive in does not matter"


def test_a_single_document_sits_in_the_middle():
    payload = galaxy.build([doc(1, subject="Physiologie")])
    node = payload["nodes"][0]
    assert (node["x"], node["y"]) == (0.0, 0.0) and payload["clusters"][0]["r"] > 0


def test_index_states_pass_through():
    docs = [doc(i) for i in range(5)]
    payload = galaxy.build(docs, index_states={"d0": "queued", "d1": "processing", "d2": "failed", "d3": "nonsense"})
    states = {n["id"]: n["state"] for n in payload["nodes"]}
    assert states == {"d0": "queued", "d1": "processing", "d2": "failed", "d3": "ready", "d4": "ready"}


def test_the_service_builds_centroids_from_the_stored_chunk_vectors(tmp_path):
    from test_study_semantic import ConceptEmbeddings

    src = tmp_path / "src"
    src.mkdir()
    (src / "Vorlast.md").write_text("# Vorlast\n\nDie Vorlast ist die Füllung des Ventrikels; mehr Vorlast, mehr Schlagvolumen (Frank-Starling).\n", encoding="utf-8")
    (src / "Füllung.md").write_text("# Füllung\n\nMehr Blut strömt zurück: die Vordehnung steigt, das Herz wirft mehr aus.\n", encoding="utf-8")
    (src / "Niere.md").write_text("# Niere\n\nAldosteron fördert die Natriumrückresorption in der Niere.\n", encoding="utf-8")
    svc = StudyService(tmp_path / "study", tmp_path / "store", embeddings=ConceptEmbeddings(), slide_renderer=False)
    for path in sorted(src.iterdir()):
        assert svc.import_path(path, copy=True)["ok"]
    svc.embed_pending()
    payload = svc.galaxy()
    by_title = {n["title"]: n for n in payload["nodes"]}
    assert len(by_title) == 3 and payload["stats"]["semantic"], payload["stats"]
    vorlast, fullung, niere = (by_title[t] for t in ("Vorlast", "Füllung", "Niere"))
    assert fullung["id"] in vorlast["similar"] and niere["id"] not in vorlast["similar"]


def test_the_service_passes_index_states_when_it_has_them(tmp_path):
    svc = StudyService(tmp_path / "study", tmp_path / "store", embeddings=False, slide_renderer=False)
    note = tmp_path / "Notiz.md"
    note.write_text("# Notiz\n\nDer Citratzyklus.\n", encoding="utf-8")
    assert svc.import_path(note, copy=True)["ok"]
    doc_id = svc.documents()[0]["id"]
    assert svc.galaxy()["nodes"][0]["state"] == "ready"
    svc.index_states = lambda: {doc_id: "processing"}  # type: ignore[attr-defined]
    assert svc.galaxy()["nodes"][0]["state"] == "processing"


# ---------------------------------------------------------------------------------------------------------- budget


def test_five_thousand_documents_build_well_under_a_second():
    docs = library(5000, subjects=40)
    rng = np.random.default_rng(2)
    bases = rng.normal(size=(40, 768)).astype(np.float32)
    vectors = {d["id"]: bases[i % 40] + 0.8 * rng.normal(size=768).astype(np.float32) for i, d in enumerate(docs)}
    galaxy.build(docs[:50], vectors_by_doc={k: vectors[k] for k in list(vectors)[:50]})   # warm numpy
    started = time.perf_counter()
    plain = galaxy.build(docs)
    plain_seconds = time.perf_counter() - started
    started = time.perf_counter()
    semantic = galaxy.build(docs, vectors_by_doc=vectors)
    semantic_seconds = time.perf_counter() - started
    print(f"galaxy 5000 docs: layout+metadata edges {plain_seconds * 1000:.0f} ms, with 768-d semantic edges {semantic_seconds * 1000:.0f} ms")
    assert len(plain["nodes"]) == 5000 and semantic["stats"]["semantic"]
    assert plain_seconds < 0.8, plain_seconds
    assert semantic_seconds < 1.5, semantic_seconds   # a loaded machine; measured ~0.6 s
    assert len(json.dumps(plain)) < 3_500_000, "the payload stays a few megabytes at most"


# ---------------------------------------------------------------------------------------------------------- route + UI


def test_the_galaxy_route(tmp_path):
    from service.http import JarvisHTTPServer
    from test_actionability import make

    core, _ = make(tmp_path)
    note = tmp_path / "Herz.md"
    note.write_text("# Herz\n\nDer Frank-Starling-Mechanismus.\n", encoding="utf-8")
    assert core.study.import_path(note, copy=True)["ok"]
    server = JarvisHTTPServer(core, port=0, token="tok")
    server.start()
    try:
        request = urllib.request.Request(f"http://{server.host}:{server.port}/api/study/galaxy", headers={"X-Jarvis-Token": "tok"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
        assert payload["ok"] and [n["title"] for n in payload["nodes"]] == [core.study.documents()[0]["title"]]
        assert payload["clusters"] and "edges" in payload
    finally:
        server.stop()


def test_the_sky_geometry_in_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, str(root / "ui" / "tests" / "study_galaxy.test.mjs")], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=120)
    assert completed.returncode == 0 and "ALL OK" in completed.stdout, completed.stdout[-2000:] + completed.stderr[-500:]
