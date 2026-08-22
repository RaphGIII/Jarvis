"""Minimal external-knowledge research step for capability acquisition.

This is deliberately scoped: it does not implement a general web-browsing
agent. It resolves a small set of well-known technical topics that appear in
a goal to an authoritative public documentation URL, fetches the page with
the standard library only, and extracts a bounded, goal-relevant text
snippet. That snippet is attached to the generated ``SkillSpecification`` so
the implementer/repairer prompts see it automatically (they already dump the
full specification, including ``metadata``, into their prompts).

Research is always best-effort: if no topic is recognized, or the fetch
fails (no network, DNS failure, timeout, non-200 response), acquisition
proceeds without research context rather than blocking on it.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable

FetchFn = Callable[[str], str]


@dataclass
class ResearchNote:
    query: str
    source: str
    summary: str
    fetched: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "source": self.source,
            "summary": self.summary,
            "fetched": self.fetched,
            "error": self.error,
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)


def extract_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return ""
    return " ".join(parser.chunks)


def default_fetch(url: str, *, timeout: float = 8.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "JarvisResearchAgent/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (deliberate, docs-only fetch)
        raw = response.read()
    charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


class CapabilityResearcher:
    """Resolves goal keywords to public documentation and fetches a snippet."""

    TOPIC_SOURCES: dict[str, tuple[str, str]] = {
        "sha-256": ("sha-256 checksum hexdigest", "https://docs.python.org/3/library/hashlib.html"),
        "sha256": ("sha-256 checksum hexdigest", "https://docs.python.org/3/library/hashlib.html"),
        "hashlib": ("hashlib module", "https://docs.python.org/3/library/hashlib.html"),
        "checksum": ("checksum hexdigest", "https://docs.python.org/3/library/hashlib.html"),
        "base64": ("base64 encoding", "https://docs.python.org/3/library/base64.html"),
        "uuid": ("uuid generation", "https://docs.python.org/3/library/uuid.html"),
        "iso 8601": ("iso 8601 date format", "https://docs.python.org/3/library/datetime.html"),
        "regular expression": ("regular expression syntax", "https://docs.python.org/3/library/re.html"),
        "regex": ("regular expression syntax", "https://docs.python.org/3/library/re.html"),
    }

    def __init__(self, *, fetch: FetchFn | None = None, max_chars: int = 1200) -> None:
        self.fetch = fetch or default_fetch
        self.max_chars = max_chars

    def topic_for_goal(self, goal: str) -> tuple[str, str] | None:
        lowered = goal.lower()
        for keyword, source in self.TOPIC_SOURCES.items():
            if keyword in lowered:
                return source
        return None

    def research(self, goal: str) -> ResearchNote | None:
        topic = self.topic_for_goal(goal)
        if topic is None:
            return None
        query, url = topic
        try:
            html = self.fetch(url)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return ResearchNote(query=query, source=url, summary="", fetched=False, error=str(exc))
        text = extract_text(html)
        summary = _relevant_snippet(text, query, self.max_chars)
        return ResearchNote(query=query, source=url, summary=summary, fetched=bool(summary))


def _relevant_snippet(text: str, query: str, max_chars: int) -> str:
    if not text:
        return ""
    lowered = text.lower()
    for term in query.lower().split():
        index = lowered.find(term)
        if index != -1:
            start = max(0, index - 100)
            return text[start : start + max_chars].strip()
    return text[:max_chars].strip()
