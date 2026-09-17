"""Studium as a sky: every study document one quiet point of light, clustered by subject.

``build()`` turns the document list (plus, when semantic search is set up, one centroid vector per
document) into the payload the "Sterne" view draws:

- **nodes**: one per document, never per chunk, with the owner's classification winning over the
  inferred one, a size weight, a recency and an index state.
- **clusters**: one per subject ("Ohne Zuordnung" for the rest); inside a large subject, one per topic
  that has enough members.
- **edges**: only meaningful ones -- consecutive documents of a topic, of a course+subject, and the
  semantic neighbours of each document's centroid above a calibrated threshold.  Candidates are
  O(n·k), never all pairs, and every node keeps at most ``limit_edges_per_node`` of them.
- **layout**: deterministic and stable.  Cluster discs (area proportional to their members) sit on
  a golden-angle spiral by size; a document's place inside its disc comes from a hash of its id, so
  the same library draws the same sky every time and a new document moves the others by a hair
  (its cluster grows by one member's area), never a reshuffle.  Coordinates are scaled by the
  library size, not by the outermost point, so one far document cannot rescale everything.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Iterable

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
UNASSIGNED = "Ohne Zuordnung"
STATES = {"queued", "processing", "ready", "failed"}
CATEGORY_KEYS = ("course", "subject", "semester", "module", "topic")
TOPIC_SPLIT_MIN_SUBJECT = 12      # a subject is divided into topics only from this many documents
TOPIC_SPLIT_MIN_MEMBERS = 3       # ... and only topics with at least this many documents get a disc of their own
SEMANTIC_TOP_K = 5
SEMANTIC_FLOOR = 0.35             # below this, no model calls two documents related
SEMANTIC_FIXED = 0.85             # two documents (no spread to calibrate against): a fixed, cautious threshold
SEMANTIC_QUANTILE = 0.97          # larger libraries: only the top 3 % of pair similarities count
EXACT_SEMANTIC_MAX = 800          # up to this many documents similarities are exact in the model's full dimension
PROJECTED_DIMS = 64
_SCALE = 1.0 / 2.2                # world units per sqrt(document) -> roughly [-1, 1]


def _hash01(text: str) -> tuple[float, float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    a = int.from_bytes(digest[:4], "little") / 4294967296.0
    b = int.from_bytes(digest[4:], "little") / 4294967296.0
    return a, b


def _key(label: str) -> str:
    return " ".join(str(label or "").split()).casefold()


def _category(doc: dict[str, Any], key: str) -> str:
    """The owner's choice wins; an inferred value (``doc["inferred"]`` or the column itself) is the fallback."""

    confirmed = set(doc.get("confirmed") or [])
    value = str(doc.get(key) or "").strip()
    if key in confirmed:
        return value
    inferred = doc.get("inferred") if isinstance(doc.get("inferred"), dict) else {}
    return value or str(inferred.get(key) or "").strip()


def _node(doc: dict[str, Any], states: dict[str, str]) -> dict[str, Any]:
    flags = doc.get("flags") if isinstance(doc.get("flags"), dict) else {}
    units = int(doc.get("units") or 0)
    state = str(states.get(doc["id"], "ready"))
    return {
        "id": str(doc["id"]),
        "title": str(doc.get("title") or doc.get("filename") or "Dokument"),
        "filename": str(doc.get("filename") or ""),
        "source_type": str(doc.get("source_type") or ""),
        "provider": str(doc.get("provider") or ""),
        "goodnotes": bool(flags.get("goodnotes")) or doc.get("provider") == "goodnotes",
        **{key: _category(doc, key) for key in CATEGORY_KEYS},
        "confirmed": sorted(set(doc.get("confirmed") or []) & set(CATEGORY_KEYS)),
        "units": units,
        "weight": round(min(1.0, math.log1p(max(units, 1)) / math.log1p(400)), 4),
        "recency": float(max(float(doc.get("opened_at") or 0), float(doc.get("updated_at") or 0), float(doc.get("indexed_at") or 0))),
        "state": state if state in STATES else "ready",
        "status": str(doc.get("status") or ""),
    }


# ---------------------------------------------------------------------------------------------- centroids


def centroids(rowids: Iterable[int], vectors: Any, document_of: dict[int, str]) -> dict[str, Any]:
    """The mean of each document's chunk vectors, normalised: one vector per document."""

    import numpy as np

    rowids = list(rowids)
    matrix = np.asarray(vectors, dtype=np.float32)
    if not rowids or matrix.ndim != 2 or matrix.shape[0] != len(rowids):
        return {}
    docs = [document_of.get(int(r)) for r in rowids]
    names = sorted({d for d in docs if d})
    if not names:
        return {}
    position = {d: i for i, d in enumerate(names)}
    keep = np.fromiter((d is not None for d in docs), dtype=bool, count=len(docs))
    groups = np.fromiter((position[d] for d in docs if d is not None), dtype=np.int64)
    sums = np.zeros((len(names), matrix.shape[1]), dtype=np.float64)
    np.add.at(sums, groups, matrix[keep])
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = (sums / norms).astype(np.float32)
    return {name: unit[i] for i, name in enumerate(names)}


# ---------------------------------------------------------------------------------------------- clusters + layout


def _spiral_pack(discs: list[tuple[str, float]], *, gap: float) -> dict[str, tuple[float, float]]:
    """Discs (key, radius), largest first, on a golden-angle spiral; pushed outward only as far as needed to not overlap."""

    placed: list[tuple[float, float, float]] = []
    centres: dict[str, tuple[float, float]] = {}
    if not discs:
        return centres
    # overlap checks through a uniform grid (a disc is filed in every cell its box touches): O(n) for thousands of discs
    radii = sorted(r for _, r in discs)
    cell = max(radii[len(radii) // 2] * 2 + gap, 1e-6)
    grid: dict[tuple[int, int], list[int]] = {}

    def cells(x: float, y: float, reach: float) -> Iterable[tuple[int, int]]:
        for gx in range(math.floor((x - reach) / cell), math.floor((x + reach) / cell) + 1):
            for gy in range(math.floor((y - reach) / cell), math.floor((y + reach) / cell) + 1):
                yield gx, gy

    def free(x: float, y: float, radius: float) -> bool:
        seen: set[int] = set()
        for key in cells(x, y, radius + gap):
            for i in grid.get(key, ()):
                if i in seen:
                    continue
                seen.add(i)
                px, py, pr = placed[i]
                if (x - px) ** 2 + (y - py) ** 2 < (radius + pr + gap) ** 2:
                    return False
        return True

    area = 0.0
    for index, (key, radius) in enumerate(discs):
        if index == 0:
            x = y = 0.0
        else:
            angle = index * GOLDEN_ANGLE
            distance = math.sqrt((area + (radius + gap / 2) ** 2) * 1.7)
            while True:
                x, y = math.cos(angle) * distance, math.sin(angle) * distance
                if free(x, y, radius):
                    break
                distance += (radius + gap) * 0.5
        placed.append((x, y, radius))
        for cell_key in cells(x, y, radius):
            grid.setdefault(cell_key, []).append(len(placed) - 1)
        centres[key] = (x, y)
        area += (radius + gap / 2) ** 2
    return centres


def _disc_radius(count: int) -> float:
    return math.sqrt(max(count, 1)) * 1.0


def _clusters(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for node in nodes:
        label = node["subject"] or UNASSIGNED
        key = _key(label) if node["subject"] else "~unassigned"
        by_subject.setdefault(key, []).append(node)
        labels.setdefault(key, label)
    # largest subjects nearest the centre; "Ohne Zuordnung" always last; ties by label
    order = sorted(by_subject, key=lambda k: (k == "~unassigned", -len(by_subject[k]), labels[k].casefold()))
    radii = {k: _disc_radius(len(by_subject[k])) * 1.18 for k in order}
    # subjects keep air between them in proportion to their size, so a large subject never reads as part of its neighbour
    centres = _spiral_pack([(k, radii[k] * 1.15) for k in order], gap=1.8)
    clusters: list[dict[str, Any]] = []
    for key in order:
        members = by_subject[key]
        cx, cy = centres[key]
        subject_id = f"subject:{key}"
        subject = {"id": subject_id, "kind": "subject", "label": labels[key], "parent": "", "members": [m["id"] for m in members],
                   "x": cx, "y": cy, "r": radii[key]}
        clusters.append(subject)
        topics: dict[str, list[dict[str, Any]]] = {}
        topic_labels: dict[str, str] = {}
        if len(members) >= TOPIC_SPLIT_MIN_SUBJECT:
            for member in members:
                if member["topic"]:
                    topics.setdefault(_key(member["topic"]), []).append(member)
                    topic_labels.setdefault(_key(member["topic"]), member["topic"])
            topics = {k: v for k, v in topics.items() if len(v) >= TOPIC_SPLIT_MIN_MEMBERS}
        if len(topics) >= 2 or (len(topics) == 1 and len(next(iter(topics.values()))) < len(members)):
            grouped = {m["id"] for group in topics.values() for m in group}
            rest = [m for m in members if m["id"] not in grouped]
            parts = sorted(topics, key=lambda k: (-len(topics[k]), topic_labels[k].casefold()))
            discs = [(k, _disc_radius(len(topics[k]))) for k in parts]
            if rest:
                discs.append(("~rest", _disc_radius(len(rest))))
            inner = _spiral_pack(discs, gap=0.35)
            # the packed topics are scaled into the subject disc around its centre
            extent = max(math.hypot(*inner[k]) + r for k, r in discs) or 1.0
            fit = min(1.0, radii[key] * 0.94 / extent)
            for k, r in discs:
                ix, iy = inner[k]
                group = topics[k] if k != "~rest" else rest
                cluster = {"id": f"topic:{key}:{k}", "kind": "topic" if k != "~rest" else "rest",
                           "label": topic_labels.get(k, ""), "parent": subject_id, "members": [m["id"] for m in group],
                           "x": cx + ix * fit, "y": cy + iy * fit, "r": r * fit}
                clusters.append(cluster)
                for m in group:
                    m["cluster"] = subject_id
                    m["topic_cluster"] = cluster["id"]
                    _place(m, cluster)
        else:
            for m in members:
                m["cluster"] = subject_id
                m["topic_cluster"] = ""
                _place(m, {"x": cx, "y": cy, "r": radii[key] * 0.86 if len(members) > 1 else 0.0})
    return clusters


def _place(node: dict[str, Any], disc: dict[str, Any]) -> None:
    """A stable place in the disc from the id alone: uniform over the area, the same on every call."""

    a, b = _hash01(node["id"])
    angle = a * 2 * math.pi
    distance = math.sqrt(b) * disc["r"]
    node["x"] = disc["x"] + math.cos(angle) * distance
    node["y"] = disc["y"] + math.sin(angle) * distance


# ---------------------------------------------------------------------------------------------- edges


def _semantic_candidates(ids: list[str], vectors: dict[str, Any], *, top_k: int, threshold: float | None) -> tuple[list[tuple[str, str, float]], dict[str, list[str]], float]:
    import numpy as np

    have = [i for i in ids if i in vectors]
    if len(have) < 2:
        return [], {}, threshold if threshold is not None else SEMANTIC_FIXED
    dims = {int(np.asarray(vectors[i]).shape[-1]) for i in have}
    if len(dims) != 1:
        return [], {}, threshold if threshold is not None else SEMANTIC_FIXED
    matrix = np.vstack([np.asarray(vectors[i], dtype=np.float32) for i in have])
    n = len(have)
    rng = np.random.default_rng(n)
    if n > EXACT_SEMANTIC_MAX and matrix.shape[1] > PROJECTED_DIMS:
        # A large library: the similarity is measured in the library's own dominant directions (a seeded randomized range
        # finder over a sample, uncentred so cosines keep their meaning -- two small QRs, no full SVD, which is seconds here).
        # n x n products in 64 dimensions instead of 768: a tenth of the cost.
        sample = matrix[np.sort(rng.choice(n, size=min(n, 1024), replace=False))]
        sample = sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-9)
        probe = rng.standard_normal((sample.shape[1], PROJECTED_DIMS)).astype(np.float32)
        range_basis, _ = np.linalg.qr(sample @ probe)
        basis, _ = np.linalg.qr(sample.T @ range_basis)
        matrix = matrix @ basis
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    k = min(top_k, n - 1)
    if threshold is None:
        if n < 3:
            threshold = SEMANTIC_FIXED   # one pair says nothing about how this model spreads its similarities
        elif n < 12:
            # a handful of documents: related means clearly above this library's own pair similarities (models differ in band)
            sims = matrix @ matrix.T
            values = sims[np.triu_indices(n, 1)]
            threshold = max(SEMANTIC_FLOOR, float(values.mean() + values.std()))
        else:
            sample = matrix[np.sort(rng.choice(n, size=min(n, 160), replace=False))]
            sims = sample @ matrix.T
            values = sims[sims < 0.99995]   # not a document with itself
            threshold = max(SEMANTIC_FLOOR, float(np.quantile(values, SEMANTIC_QUANTILE))) if values.size else SEMANTIC_FIXED
    pairs: list[tuple[str, str, float]] = []
    similar: dict[str, list[str]] = {}
    block = 1024
    for start in range(0, n, block):
        sims = matrix[start:start + block] @ matrix.T
        count = sims.shape[0]
        sims[np.arange(count), np.arange(count) + start] = -2.0
        # the k best of each row by k row-wise argmax passes (k is small): linear, no n x n sort or partition
        line = np.arange(count)
        best_cols = np.empty((count, k), dtype=np.int64)
        best_vals = np.empty((count, k), dtype=np.float32)
        for rank in range(k):
            chosen = np.argmax(sims, axis=1)
            best_cols[:, rank] = chosen
            best_vals[:, rank] = sims[line, chosen]
            sims[line, chosen] = -2.0
        for r, cols, vals in zip(line.tolist(), best_cols.tolist(), best_vals.tolist()):
            for c, v in zip(cols, vals):
                if v < threshold:
                    break
                source, target = have[start + r], have[c]
                similar.setdefault(source, []).append(target)
                pairs.append((source, target, v))
    return pairs, similar, float(threshold)


def _chain(group: list[dict[str, Any]]) -> list[tuple[str, str]]:
    ordered = sorted(group, key=lambda n: (n["recency"], n["title"].casefold(), n["id"]))
    return [(a["id"], b["id"]) for a, b in zip(ordered, ordered[1:])]


def _edges(nodes: list[dict[str, Any]], vectors: dict[str, Any] | None, *, cap: int, threshold: float | None) -> tuple[list[dict[str, Any]], dict[str, list[str]], float | None]:
    candidates: dict[tuple[str, str], tuple[float, str]] = {}

    def offer(a: str, b: str, weight: float, kind: str) -> None:
        if a == b:
            return
        pair = (a, b) if a < b else (b, a)
        if pair not in candidates or candidates[pair][0] < weight:
            candidates[pair] = (weight, kind)

    topics: dict[tuple[str, str], list[dict[str, Any]]] = {}
    courses: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in nodes:
        if node["topic"]:
            topics.setdefault((_key(node["subject"]), _key(node["topic"])), []).append(node)
        if node["course"] and node["subject"]:
            courses.setdefault((_key(node["course"]), _key(node["subject"])), []).append(node)
    for group in topics.values():
        for a, b in _chain(group):
            offer(a, b, 0.62, "topic")
    for group in courses.values():
        for a, b in _chain(group):
            offer(a, b, 0.4, "course")
    similar: dict[str, list[str]] = {}
    used_threshold: float | None = None
    if vectors:
        pairs, similar, used_threshold = _semantic_candidates([n["id"] for n in nodes], vectors, top_k=SEMANTIC_TOP_K, threshold=threshold)
        for a, b, score in pairs:
            offer(a, b, 0.5 + 0.5 * score, "semantic")   # a real likeness outranks mere shared metadata
    degree: dict[str, int] = {}
    edges: list[dict[str, Any]] = []
    for (a, b), (weight, kind) in sorted(candidates.items(), key=lambda item: (-item[1][0], item[0])):
        if degree.get(a, 0) >= cap or degree.get(b, 0) >= cap:
            continue
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
        edges.append({"source": a, "target": b, "kind": kind, "weight": round(weight, 4)})
    return edges, similar, used_threshold


# ---------------------------------------------------------------------------------------------- build


def build(documents: Iterable[dict[str, Any]], *, vectors_by_doc: dict[str, Any] | None = None, index_states: dict[str, str] | None = None,
          limit_edges_per_node: int = 3, semantic_threshold: float | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    states = {str(k): str(v) for k, v in (index_states or {}).items()}
    seen: set[str] = set()
    nodes: list[dict[str, Any]] = []
    for doc in documents:
        if not doc or not doc.get("id") or str(doc["id"]) in seen:
            continue
        seen.add(str(doc["id"]))
        nodes.append(_node(doc, states))
    clusters = _clusters(nodes)
    scale = _SCALE / math.sqrt(max(len(nodes), 1))
    for node in nodes:
        node["x"] = round(node["x"] * scale, 5)
        node["y"] = round(node["y"] * scale, 5)
    for cluster in clusters:
        cluster["x"] = round(cluster["x"] * scale, 5)
        cluster["y"] = round(cluster["y"] * scale, 5)
        cluster["r"] = round(cluster["r"] * scale, 5)
    edges, similar, threshold = _edges(nodes, vectors_by_doc, cap=max(0, int(limit_edges_per_node)), threshold=semantic_threshold)
    for node in nodes:
        node["similar"] = similar.get(node["id"], [])
    return {"ok": True, "nodes": nodes, "clusters": clusters, "edges": edges,
            "stats": {"documents": len(nodes), "clusters": sum(1 for c in clusters if c["kind"] == "subject"), "edges": len(edges),
                      "semantic": bool(similar), "semantic_threshold": threshold, "build_ms": round((time.perf_counter() - started) * 1000, 1)}}
