"""Answering a technical question from public sources, with the receipts.

Search and fetch already exist as tools.  What is missing between them and a
useful answer is everything that makes research trustworthy rather than merely
fast, and each part of it is here for a specific failure it prevents.

*Provenance is structural, not optional.*  A :class:`Finding` cannot exist
without a URL and the excerpt it came from.  A research system that returns
prose the user cannot check has produced something worse than nothing -- it
looks like knowledge and is untraceable.

*Source ranking is deterministic.*  Which of eight results is authoritative is
decided by the domain, not by asking a 4B model to judge credibility.  Official
documentation outranks a blog post about it, and that ordering is a fact about
the URL rather than an opinion.

*Contradictions are surfaced, not resolved.*  When two sources disagree the
honest output says so and cites both.  Picking a winner is a judgement the user
is better placed to make, and quietly dropping the loser is how a research tool
becomes confidently wrong.

*It works offline, badly but honestly.*  With no network the report says it had
no sources rather than answering from the model's memory, which is exactly the
case where a plausible hallucination is most likely and least detectable.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

#: Domains whose documentation is primary.  Ordered by how close they sit to
#: the thing being documented: the project's own site beats a mirror, a mirror
#: beats a Q&A site, and a Q&A site beats a content farm.
AUTHORITY: tuple[tuple[int, tuple[str, ...]], ...] = (
    (100, ("docs.python.org", "peps.python.org", "developer.mozilla.org", "w3.org",
           "rfc-editor.org", "ietf.org", "iso.org", "unicode.org")),
    (90, ("github.com", "gitlab.com", "readthedocs.io", "readthedocs.org",
          "godoc.org", "pkg.go.dev", "docs.rs", "crates.io", "pypi.org", "npmjs.com")),
    (80, ("microsoft.com", "learn.microsoft.com", "developer.apple.com",
          "android.com", "developer.android.com", "kernel.org", "gnu.org",
          "postgresql.org", "sqlite.org", "nginx.org", "docker.com", "kubernetes.io")),
    (60, ("wikipedia.org", "arxiv.org", "stackoverflow.com", "superuser.com",
          "serverfault.com", "askubuntu.com")),
)

#: Below this, a source is used only when nothing better was found.
WEAK_SOURCE = 30


def authority_of(url: str) -> int:
    """How primary a source is, from its domain alone.

    Deterministic on purpose.  Asking a small model which of eight results is
    trustworthy invites exactly the confident-and-wrong answer research is
    supposed to eliminate, and the answer is mostly a property of the domain.
    """

    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if not host:
        return 0
    for score, domains in AUTHORITY:
        for domain in domains:
            if host == domain or host.endswith("." + domain):
                return score
    # A documentation-shaped path is weak evidence, but better than nothing.
    path = urllib.parse.urlparse(url).path.lower()
    if any(marker in path for marker in ("/docs/", "/documentation/", "/reference/", "/manual/")):
        return 50
    return WEAK_SOURCE


@dataclass
class Finding:
    """One claim, and where it came from.

    ``source_url`` and ``excerpt`` are required rather than optional: a finding
    without them is an assertion, and assertions are what this module exists to
    avoid producing.
    """

    claim: str
    source_url: str
    excerpt: str
    source_title: str = ""
    authority: int = 0
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Source:
    url: str
    title: str = ""
    authority: int = 0
    ok: bool = True
    chars: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchReport:
    question: str
    findings: list[Finding] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    offline: bool = False
    seconds: float = 0.0

    @property
    def grounded(self) -> bool:
        """True when there is at least one finding backed by a real source."""

        return any(item.source_url and item.excerpt for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "summary": self.summary,
            "findings": [item.to_dict() for item in self.findings],
            "sources": [item.to_dict() for item in self.sources],
            "queries": self.queries,
            "contradictions": self.contradictions,
            "offline": self.offline,
            "grounded": self.grounded,
            "seconds": round(self.seconds, 1),
        }

    def as_text(self) -> str:
        """The report as prose a model or a person can read, citations included."""

        if not self.findings:
            return f"No sources answered: {self.question}\n" + (
                "The network was unreachable." if self.offline else "Nothing relevant was found."
            )
        lines = [f"RESEARCH: {self.question}", ""]
        if self.summary:
            lines += [self.summary, ""]
        lines.append("FINDINGS (each with its source):")
        for index, item in enumerate(self.findings, start=1):
            lines.append(f"{index}. {item.claim}")
            lines.append(f"   source: {item.source_url}")
            lines.append(f"   quote: \"{item.excerpt[:220].strip()}\"")
        if self.contradictions:
            lines += ["", "SOURCES DISAGREE:"]
            for item in self.contradictions:
                lines.append(f"- {item['topic']}: {item['a']} vs {item['b']}")
        return "\n".join(lines)


_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        }
    },
    "required": ["queries"],
}

_FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["claim", "quote"],
            },
        }
    },
    "required": ["findings"],
}


class ResearchAgent:
    """Turns a question into cited findings, using free sources only."""

    def __init__(
        self,
        *,
        brain: Any = None,
        search: Any = None,
        fetcher: Any = None,
        graph: Any = None,
    ) -> None:
        self.brain = brain
        self._search = search
        self._fetcher = fetcher
        self.graph = graph

    # -- lazily built so constructing an agent touches no network ---------

    @property
    def search(self) -> Any:
        if self._search is None:
            from tools.web import DuckDuckGoBackend

            self._search = DuckDuckGoBackend()
        return self._search

    @property
    def fetcher(self) -> Any:
        if self._fetcher is None:
            from tools.web import DocumentFetcher

            self._fetcher = DocumentFetcher()
        return self._fetcher

    # -- the pipeline ----------------------------------------------------

    def research(
        self,
        question: str,
        *,
        max_sources: int = 4,
        max_queries: int = 3,
        max_seconds: float = 180.0,
    ) -> ResearchReport:
        started = time.perf_counter()
        report = ResearchReport(question=question.strip())
        deadline = started + max_seconds

        report.queries = self._queries_for(question, limit=max_queries)

        hits: list[Any] = []
        for query in report.queries:
            if time.perf_counter() > deadline:
                break
            try:
                hits.extend(self.search.search(query, limit=6))
            except Exception as exc:
                report.offline = True
                report.summary = f"search failed: {exc}"

        ranked = self._rank(hits)
        if not ranked:
            report.offline = report.offline or True
            report.seconds = time.perf_counter() - started
            if not report.summary:
                report.summary = "no sources were reachable"
            return report

        for hit in ranked[:max_sources]:
            if time.perf_counter() > deadline:
                break
            url = getattr(hit, "url", "")
            document = self._fetch(url)
            source = Source(
                url=url,
                title=getattr(hit, "title", "") or "",
                authority=authority_of(url),
                ok=bool(document and document.ok),
                chars=len(document.text) if document and document.ok else 0,
                error="" if (document and document.ok) else (document.error if document else "not fetched"),
            )
            report.sources.append(source)
            if not source.ok or not document.text.strip():
                continue
            report.findings.extend(
                self._extract(question, document.text, source)
            )

        report.contradictions = self._contradictions(report.findings)
        report.summary = report.summary or self._summarise(question, report.findings)
        report.seconds = time.perf_counter() - started

        if self.graph is not None:
            self._remember(report)
        return report

    # -- steps -----------------------------------------------------------

    def _queries_for(self, question: str, *, limit: int) -> list[str]:
        """Search queries for a question.

        The question itself is always included.  A model that returns nothing
        usable must not be able to prevent the search from happening at all,
        and the literal question is a perfectly good query.
        """

        queries = [question.strip()]
        if self.brain is None:
            return queries[:limit]

        prompt = (
            "Return JSON only. Turn this technical question into short web search queries.\n"
            "Prefer terms that would appear in official documentation. No prose, no quotes.\n\n"
            f"Question: {question}\n"
        )
        try:
            raw = self.brain.generate_structured(prompt, _QUERY_SCHEMA, max_tokens=200)
            payload = json.loads(raw)
        except Exception:
            return queries[:limit]

        for item in payload.get("queries", []):
            text = str(item).strip()
            if text and text.lower() not in {q.lower() for q in queries}:
                queries.append(text)
        return queries[:limit]

    def _rank(self, hits: Iterable[Any]) -> list[Any]:
        """Best sources first, duplicates removed, weak ones last."""

        seen: set[str] = set()
        unique = []
        for hit in hits:
            url = getattr(hit, "url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(hit)
        return sorted(unique, key=lambda hit: -authority_of(getattr(hit, "url", "")))

    def _fetch(self, url: str) -> Any:
        try:
            return self.fetcher.fetch(url)
        except Exception:
            return None

    def _extract(self, question: str, text: str, source: Source) -> list[Finding]:
        """Pull claims out of one document, each tied to a quote from it."""

        if self.brain is None:
            return []

        prompt = (
            "Return JSON only. Extract facts from the DOCUMENT that help answer the QUESTION.\n"
            "Every finding must include a 'quote' copied VERBATIM from the document -- if you "
            "cannot quote it, do not claim it.\n"
            "Extract nothing if the document does not address the question.\n\n"
            f"QUESTION: {question}\n\n"
            f"DOCUMENT ({source.url}):\n{text[:8000]}\n"
        )
        try:
            payload = json.loads(self.brain.generate_structured(prompt, _FINDINGS_SCHEMA, max_tokens=900))
        except Exception:
            return []

        findings: list[Finding] = []
        for item in payload.get("findings", [])[:6]:
            claim = str(item.get("claim", "")).strip()
            quote = str(item.get("quote", "")).strip()
            if not claim or not quote:
                continue
            # The quote must really be in the document. Without this the
            # citation is decorative and a hallucinated claim arrives wearing
            # a source URL, which is worse than an uncited one.
            if not _contains_quote(text, quote):
                continue
            findings.append(
                Finding(
                    claim=claim,
                    source_url=source.url,
                    source_title=source.title,
                    excerpt=quote,
                    authority=source.authority,
                )
            )
        return findings

    def _contradictions(self, findings: list[Finding]) -> list[dict[str, Any]]:
        """Findings that assert opposite things about the same subject.

        Deliberately shallow: it looks for a negation difference between claims
        that otherwise share most of their words. Anything cleverer would be a
        judgement, and the point is to SHOW the disagreement rather than settle
        it.
        """

        found: list[dict[str, Any]] = []
        for index, first in enumerate(findings):
            for second in findings[index + 1 :]:
                if first.source_url == second.source_url:
                    continue
                if _opposed(first.claim, second.claim):
                    found.append(
                        {
                            "topic": _shared_subject(first.claim, second.claim),
                            "a": f"{first.claim} ({first.source_url})",
                            "b": f"{second.claim} ({second.source_url})",
                        }
                    )
        return found[:5]

    def _summarise(self, question: str, findings: list[Finding]) -> str:
        if not findings:
            return "nothing was found that answers this"
        if self.brain is None:
            return f"{len(findings)} finding(s) from {len({f.source_url for f in findings})} source(s)"
        joined = "\n".join(f"- {item.claim}" for item in findings[:12])
        prompt = (
            "Answer the question in two or three sentences using ONLY the findings below. "
            "Do not add anything they do not say. If they are insufficient, say so.\n\n"
            f"Question: {question}\n\nFindings:\n{joined}\n"
        )
        try:
            return str(self.brain.generate(prompt, max_tokens=300)).strip()
        except Exception:
            return f"{len(findings)} finding(s) from {len({f.source_url for f in findings})} source(s)"

    def _remember(self, report: ResearchReport) -> None:
        """Persist the report so the next question can start from it."""

        try:
            from knowledge.graph import EdgeType, NodeType

            topic = self.graph.remember(
                NodeType.NOTE,
                f"Research: {report.question[:90]}",
                report.as_text(),
                provenance="research",
                tags=["research"],
            )
            for source in report.sources:
                if not source.ok:
                    continue
                node = self.graph.remember(
                    NodeType.SOURCE,
                    source.title or source.url,
                    source.url,
                    provenance=source.url,
                    metadata={"authority": source.authority},
                )
                self.graph.link(topic.id, node.id, EdgeType.DERIVED_FROM)
        except Exception:
            # Persistence is a convenience; losing it must not lose the report.
            pass


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains_quote(document: str, quote: str) -> bool:
    """Whether a quote really appears in the document, ignoring whitespace."""

    needle = _normalise(quote)
    if len(needle) < 12:
        return False
    return needle in _normalise(document)


_NEGATIONS = ("not ", "no ", "never ", "cannot ", "can't ", "isn't ", "doesn't ", "unsupported", "removed")


def _opposed(first: str, second: str) -> bool:
    a, b = _normalise(first), _normalise(second)
    if a == b:
        return False
    words_a = set(a.split())
    words_b = set(b.split())
    overlap = len(words_a & words_b) / max(1, min(len(words_a), len(words_b)))
    if overlap < 0.6:
        return False
    negated_a = any(marker in a for marker in _NEGATIONS)
    negated_b = any(marker in b for marker in _NEGATIONS)
    return negated_a != negated_b


def _shared_subject(first: str, second: str) -> str:
    words = [w for w in _normalise(first).split() if w in set(_normalise(second).split())]
    return " ".join(words[:6]) or "the same point"
