"""The knowledge graph: notes, concepts, people, projects, capabilities, links.

This is the backend a graph UI would later sit on -- deliberately backend only,
per the brief.  The shape is the one an Obsidian-like tool needs: typed nodes
with stable identifiers, typed edges with automatic backlinks, provenance on
everything, and two retrieval paths (keyword and semantic) because neither
alone is sufficient.

Storage is SQLite.  Unlike the project store, where one JSON document per
project keeps things hand-inspectable, a graph is all about joins -- "what links
here", "what does this project know about X" -- and doing those over a directory
of JSON files means loading everything into memory on every query.  SQLite gives
indexed edges, transactional writes and a single portable file.

Two design points worth stating:

*Backlinks are not optional.*  Every edge is queryable from both ends.  A note
that nothing links to is nearly worthless, and discovering that requires the
reverse index to be as cheap as the forward one.

*Semantic search must work offline with no model loaded.*  The default embedder
is a deterministic lexical vectoriser -- no downloads, no GPU, no startup cost.
When a real embedding model is configured it slots in behind the same interface
and the stored vectors are versioned so a change of embedder is detected rather
than silently mixing incompatible vectors.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol


class NodeType(str, Enum):
    """The entity kinds the palace knows about."""

    NOTE = "note"
    CONCEPT = "concept"
    PERSON = "person"
    PROJECT = "project"
    SOURCE = "source"
    FILE = "file"
    CAPABILITY = "capability"
    TASK = "task"
    IDEA = "idea"
    EXPERIMENT = "experiment"
    DECISION = "decision"


class EdgeType(str, Enum):
    """How entities relate.  Kept small; specificity lives in the payload."""

    MENTIONS = "mentions"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    DERIVED_FROM = "derived_from"
    IMPLEMENTS = "implements"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    PRODUCED = "produced"
    ABOUT = "about"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Node:
    """One entity."""

    type: NodeType
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    body: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Where this came from -- a file path, a URL, a project id, "user".
    provenance: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def text(self) -> str:
        """Everything searchable about this node, as one string."""

        return " ".join([self.title, self.body, " ".join(self.tags)])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass
class Edge:
    """A typed, directed link.  Queryable from both ends."""

    source: str
    target: str
    type: EdgeType = EdgeType.RELATES_TO
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass
class SearchHit:
    node: Node
    score: float
    how: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node.to_dict(), "score": round(self.score, 4), "how": self.how}


class Embedder(Protocol):
    """Turns text into a vector.  Swappable; the version guards the store."""

    version: str

    def embed(self, text: str) -> list[float]:
        ...


class LexicalEmbedder:
    """A deterministic bag-of-words vectoriser, hashed into fixed dimensions.

    Chosen as the default precisely because it is unexciting: no model to
    download, no VRAM, no cold-start latency, identical results on every
    machine, and it works with the network unplugged.  It captures topical
    overlap, which is most of what "find me things about X" needs; a real
    embedding model can replace it when one is configured, and the version
    string below makes that transition visible.
    """

    #: Bumped whenever tokenisation changes, so stored vectors built by an
    #: older scheme are detected as stale rather than compared against new ones.
    version = "lexical-v2-256"
    dimensions = 256

    _WORD = re.compile(r"[a-z0-9_]+")
    _STOPWORDS = frozenset(
        """
        the a an and or but if then else for of to in on at by with from into over under
        is are was were be been being do does did have has had this that these those it its
        as not no so such than too very can will just don should now
        der die das und oder aber wenn dann sonst fuer von zu in an auf bei mit aus ueber
        ist sind war waren sein nicht kein so ein eine einen dem den des
        """.split()
    )

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, weight in self._tokens(text):
            vector[self._bucket(token)] += weight
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    @staticmethod
    def _bucket(token: str) -> int:
        """Map a token to a dimension, identically in every process.

        Emphatically not ``hash()``: Python randomises string hashing per
        process, so vectors written today would land in different dimensions
        tomorrow and every stored embedding would silently stop matching after
        a restart -- the exact failure a persistent store must not have.
        """

        return zlib.crc32(token.encode("utf-8")) % LexicalEmbedder.dimensions

    def _tokens(self, text: str) -> Iterable[tuple[str, float]]:
        words = [stem(word) for word in self._WORD.findall(text.lower()) if len(word) > 2 and word not in self._STOPWORDS]
        for word in words:
            yield word, 1.0
        # Adjacent pairs give a little word-order sensitivity, which is what
        # separates "test runner" from "run the tests" in practice.
        for first, second in zip(words, words[1:]):
            yield f"{first}_{second}", 0.5


def stem(word: str) -> str:
    """Crude suffix stripping, so ``plays`` and ``playing`` reach ``play``.

    Not a linguistic stemmer, and not trying to be.  Its whole job is to stop a
    lookup failing on inflection alone -- a capability registered as "plays an
    audio file" was invisible to the query "play a file" purely because of the
    trailing s.  Words of four characters or fewer are left alone, since that is
    where a naive stemmer does more harm than good.
    """

    if len(word) <= 4:
        return word
    for suffix, minimum in (("ingly", 7), ("edly", 7), ("ing", 6), ("ers", 6), ("er", 5), ("ed", 5), ("es", 5), ("s", 4)):
        if word.endswith(suffix) and len(word) > minimum:
            trimmed = word[: -len(suffix)]
            # "running" -> "runn" -> "run"; "pressing" -> "press" stays.
            if len(trimmed) > 2 and trimmed[-1] == trimmed[-2] and trimmed[-1] not in "sl":
                trimmed = trimmed[:-1]
            return trimmed
    return word


class OllamaEmbedder:
    """Real embeddings from a local Ollama model, when one is configured."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 60.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.version = f"ollama-{model}"

    def embed(self, text: str) -> list[float]:
        import urllib.request

        request = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=json.dumps({"model": self.model, "prompt": text[:8000]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vector = [float(value) for value in payload.get("embedding") or []]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    provenance TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS nodes_title ON nodes(title);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    provenance TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(source, target, type)
);
CREATE INDEX IF NOT EXISTS edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS edges_target ON edges(target);

CREATE TABLE IF NOT EXISTS vectors (
    node_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    vector TEXT NOT NULL
);
"""


class KnowledgeGraph:
    """Typed nodes, typed edges, backlinks, keyword and semantic retrieval."""

    def __init__(self, path: str | Path, *, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder: Embedder = embedder or LexicalEmbedder()
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "KnowledgeGraph":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- writing ---------------------------------------------------------

    def add_node(self, node: Node) -> Node:
        node.updated_at = _now()
        self._connection.execute(
            """
            INSERT INTO nodes (id, type, title, body, tags, metadata, provenance, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type=excluded.type, title=excluded.title, body=excluded.body, tags=excluded.tags,
                metadata=excluded.metadata, provenance=excluded.provenance, confidence=excluded.confidence,
                updated_at=excluded.updated_at
            """,
            (
                node.id,
                node.type.value,
                node.title,
                node.body,
                json.dumps(node.tags),
                json.dumps(node.metadata, default=str),
                node.provenance,
                float(node.confidence),
                node.created_at,
                node.updated_at,
            ),
        )
        self._index(node)
        self._connection.commit()
        return node

    def note(self, title: str, body: str = "", **kwargs: Any) -> Node:
        return self.add_node(Node(type=NodeType.NOTE, title=title, body=body, **kwargs))

    def remember(self, type: NodeType, title: str, body: str = "", **kwargs: Any) -> Node:
        """Add a node, reusing an existing one with the same type and title.

        Autonomous runs revisit the same concepts constantly.  Without this the
        graph fills with near-duplicates and every backlink query returns the
        same idea five times.
        """

        existing = self.find_by_title(type, title)
        if existing is not None:
            if body and body not in existing.body:
                existing.body = f"{existing.body}\n\n{body}".strip()
            existing.tags = sorted(set(existing.tags) | set(kwargs.get("tags") or []))
            existing.metadata.update(kwargs.get("metadata") or {})
            if kwargs.get("provenance"):
                existing.provenance = kwargs["provenance"]
            return self.add_node(existing)
        return self.add_node(Node(type=type, title=title, body=body, **kwargs))

    def link(
        self,
        source: str | Node,
        target: str | Node,
        type: EdgeType = EdgeType.RELATES_TO,
        **kwargs: Any,
    ) -> Edge:
        edge = Edge(
            source=source.id if isinstance(source, Node) else source,
            target=target.id if isinstance(target, Node) else target,
            type=type,
            **kwargs,
        )
        self._connection.execute(
            """
            INSERT INTO edges (id, source, target, type, weight, metadata, provenance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, target, type) DO UPDATE SET
                weight=excluded.weight, metadata=excluded.metadata, provenance=excluded.provenance
            """,
            (
                edge.id,
                edge.source,
                edge.target,
                edge.type.value,
                float(edge.weight),
                json.dumps(edge.metadata, default=str),
                edge.provenance,
                edge.created_at,
            ),
        )
        self._connection.commit()
        return edge

    def delete_node(self, node_id: str) -> None:
        """Remove a node and every edge touching it, leaving no dangling links."""

        self._connection.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
        self._connection.execute("DELETE FROM vectors WHERE node_id = ?", (node_id,))
        self._connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._connection.commit()

    # -- reading ---------------------------------------------------------

    def get(self, node_id: str) -> Node | None:
        row = self._connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _node_from_row(row) if row else None

    def find_by_title(self, type: NodeType, title: str) -> Node | None:
        row = self._connection.execute(
            "SELECT * FROM nodes WHERE type = ? AND lower(title) = lower(?) LIMIT 1", (type.value, title)
        ).fetchone()
        return _node_from_row(row) if row else None

    def nodes(self, *, type: NodeType | None = None, limit: int = 200) -> list[Node]:
        if type is None:
            rows = self._connection.execute("SELECT * FROM nodes ORDER BY updated_at DESC LIMIT ?", (limit,))
        else:
            rows = self._connection.execute(
                "SELECT * FROM nodes WHERE type = ? ORDER BY updated_at DESC LIMIT ?", (type.value, limit)
            )
        return [_node_from_row(row) for row in rows]

    def edges_from(self, node_id: str, *, type: EdgeType | None = None) -> list[Edge]:
        query = "SELECT * FROM edges WHERE source = ?"
        params: list[Any] = [node_id]
        if type is not None:
            query += " AND type = ?"
            params.append(type.value)
        return [_edge_from_row(row) for row in self._connection.execute(query, params)]

    def edges_to(self, node_id: str, *, type: EdgeType | None = None) -> list[Edge]:
        query = "SELECT * FROM edges WHERE target = ?"
        params: list[Any] = [node_id]
        if type is not None:
            query += " AND type = ?"
            params.append(type.value)
        return [_edge_from_row(row) for row in self._connection.execute(query, params)]

    def backlinks(self, node_id: str) -> list[Node]:
        """Every node that points here.  The reverse index, cheaply."""

        rows = self._connection.execute(
            "SELECT n.* FROM nodes n JOIN edges e ON n.id = e.source WHERE e.target = ? ORDER BY n.updated_at DESC",
            (node_id,),
        )
        return [_node_from_row(row) for row in rows]

    def neighbours(self, node_id: str, *, depth: int = 1, limit: int = 50) -> list[Node]:
        """Nodes reachable within ``depth`` hops, in either direction."""

        seen = {node_id}
        frontier = {node_id}
        collected: list[Node] = []
        for _ in range(max(1, depth)):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._connection.execute(
                f"SELECT DISTINCT target AS other FROM edges WHERE source IN ({placeholders}) "
                f"UNION SELECT DISTINCT source AS other FROM edges WHERE target IN ({placeholders})",
                (*frontier, *frontier),
            ).fetchall()
            frontier = {row["other"] for row in rows} - seen
            seen |= frontier
            for identifier in frontier:
                node = self.get(identifier)
                if node is not None:
                    collected.append(node)
                if len(collected) >= limit:
                    return collected
        return collected

    # -- retrieval -------------------------------------------------------

    def search_keyword(self, query: str, *, type: NodeType | None = None, limit: int = 10) -> list[SearchHit]:
        """Term-overlap search.  Exact, explainable, and always available."""

        terms = _terms(query)
        if not terms:
            return []
        hits: list[SearchHit] = []
        for node in self.nodes(type=type, limit=5000):
            node_terms = _terms(node.text())
            if not node_terms:
                continue
            overlap = len(terms & node_terms)
            if not overlap:
                continue
            score = overlap / len(terms)
            if query.strip().lower() in node.title.lower():
                score += 0.5
            hits.append(SearchHit(node=node, score=score, how="keyword"))
        hits.sort(key=lambda item: (item.score, item.node.updated_at), reverse=True)
        return hits[:limit]

    def search_semantic(self, query: str, *, type: NodeType | None = None, limit: int = 10) -> list[SearchHit]:
        """Vector similarity over stored embeddings."""

        target = self.embedder.embed(query)
        if not any(target):
            return []
        rows = self._connection.execute(
            "SELECT v.node_id, v.vector, v.version FROM vectors v WHERE v.version = ?", (self.embedder.version,)
        ).fetchall()
        hits: list[SearchHit] = []
        for row in rows:
            node = self.get(row["node_id"])
            if node is None or (type is not None and node.type is not type):
                continue
            similarity = _cosine(target, json.loads(row["vector"]))
            if similarity > 0.05:
                hits.append(SearchHit(node=node, score=similarity, how="semantic"))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def search(self, query: str, *, type: NodeType | None = None, limit: int = 10) -> list[SearchHit]:
        """Both retrievers, merged.

        Keyword search finds the thing you named; semantic search finds the
        thing you described.  Running both and merging is more useful than
        either alone, and costs nothing extra offline.
        """

        merged: dict[str, SearchHit] = {}
        for hit in self.search_keyword(query, type=type, limit=limit * 2):
            merged[hit.node.id] = hit
        for hit in self.search_semantic(query, type=type, limit=limit * 2):
            existing = merged.get(hit.node.id)
            if existing is None:
                merged[hit.node.id] = hit
            else:
                # Agreement between two independent signals is worth more than
                # a strong score from one.
                existing.score += hit.score * 0.5
                existing.how = "keyword+semantic"
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def context_for(self, query: str, *, limit: int = 6, depth: int = 1) -> list[Node]:
        """What Jarvis should be reminded of when working on ``query``.

        Search results plus their immediate neighbourhood: a relevant decision
        usually matters together with the experiment that produced it.
        """

        seeds = self.search(query, limit=limit)
        collected: dict[str, Node] = {hit.node.id: hit.node for hit in seeds}
        for hit in seeds:
            for neighbour in self.neighbours(hit.node.id, depth=depth, limit=4):
                collected.setdefault(neighbour.id, neighbour)
        return list(collected.values())[: limit * 3]

    # -- maintenance -----------------------------------------------------

    def _index(self, node: Node) -> None:
        vector = self.embedder.embed(node.text())
        self._connection.execute(
            "INSERT INTO vectors (node_id, version, vector) VALUES (?, ?, ?) "
            "ON CONFLICT(node_id) DO UPDATE SET version=excluded.version, vector=excluded.vector",
            (node.id, self.embedder.version, json.dumps(vector)),
        )

    def reindex(self) -> int:
        """Rebuild every vector, after changing the embedder."""

        count = 0
        for node in self.nodes(limit=1_000_000):
            self._index(node)
            count += 1
        self._connection.commit()
        return count

    def stats(self) -> dict[str, Any]:
        node_counts = {
            row["type"]: row["n"]
            for row in self._connection.execute("SELECT type, COUNT(*) AS n FROM nodes GROUP BY type")
        }
        edge_counts = {
            row["type"]: row["n"]
            for row in self._connection.execute("SELECT type, COUNT(*) AS n FROM edges GROUP BY type")
        }
        stale = self._connection.execute(
            "SELECT COUNT(*) AS n FROM vectors WHERE version != ?", (self.embedder.version,)
        ).fetchone()["n"]
        return {
            "path": str(self.path),
            "nodes": sum(node_counts.values()),
            "edges": sum(edge_counts.values()),
            "by_node_type": node_counts,
            "by_edge_type": edge_counts,
            "embedder": self.embedder.version,
            "stale_vectors": stale,
        }


def _node_from_row(row: sqlite3.Row) -> Node:
    return Node(
        id=row["id"],
        type=NodeType(row["type"]),
        title=row["title"],
        body=row["body"],
        tags=json.loads(row["tags"]),
        metadata=json.loads(row["metadata"]),
        provenance=row["provenance"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _edge_from_row(row: sqlite3.Row) -> Edge:
    return Edge(
        id=row["id"],
        source=row["source"],
        target=row["target"],
        type=EdgeType(row["type"]),
        weight=row["weight"],
        metadata=json.loads(row["metadata"]),
        provenance=row["provenance"],
        created_at=row["created_at"],
    )


def _terms(text: str) -> set[str]:
    """Searchable terms, stemmed so keyword search matches inflected forms.

    Uses the same stemming as the embedder; if the two disagreed, a node could
    be found by one retriever and be invisible to the other.
    """

    return {
        stem(word)
        for word in re.split(r"[^a-z0-9_]+", text.lower())
        if len(word) > 2 and word not in LexicalEmbedder._STOPWORDS
    }


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))
