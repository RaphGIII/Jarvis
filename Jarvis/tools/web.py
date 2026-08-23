"""Research tools: fetching and searching public technical documentation.

Constraints from the mission that shape every decision in this file:

*No paid search API may be required.*  Search therefore goes through DuckDuckGo's
public HTML endpoint, which needs no key and no account.  A paid backend can be
slotted in behind :class:`SearchBackend` later, but nothing here depends on one.

*Offline must degrade, not break.*  Every adapter reports "the network is not
reachable" as an ordinary tool failure, so the agent loop can note the blocker
and carry on with local knowledge instead of aborting the project.

*Research is evidence, not vibes.*  Results carry their URL and retrieval time so
the project record can distinguish "the docs say" from "the model believes".
"""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from tools.registry import RiskLevel, ToolContext, ToolError, ToolSpec

USER_AGENT = "Mozilla/5.0 (compatible; JarvisResearch/1.0; +local-autonomous-agent)"

#: Hosts that serve technical documentation and are safe to fetch unattended.
#: The list is advisory: it ranks results, it does not gate them.
DOCUMENTATION_HOSTS = (
    "docs.python.org",
    "peps.python.org",
    "pypi.org",
    "readthedocs.io",
    "github.com",
    "raw.githubusercontent.com",
    "developer.mozilla.org",
    "stackoverflow.com",
    "wikipedia.org",
    "man7.org",
    "ollama.com",
)


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        ...


class DuckDuckGoBackend:
    """Keyless web search via DuckDuckGo's HTML endpoint.

    Scraping HTML is fragile by nature, so parsing failures degrade to an empty
    result list rather than an exception: "found nothing" is a state the agent
    can reason about, a traceback is not.
    """

    name = "duckduckgo_html"
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, *, timeout: float = 20.0, opener=None) -> None:
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def search(self, query: str, *, limit: int = 8) -> list[SearchHit]:
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={"User-Agent": USER_AGENT, "Accept": "text/html", "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise ToolError(f"search endpoint returned HTTP {exc.code}", kind="search_unavailable") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ToolError(f"search is unreachable (offline?): {exc}", kind="offline") from None
        return parse_duckduckgo_html(body, limit=limit)


_RESULT_ANCHOR = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.DOTALL | re.IGNORECASE
)
_SNIPPET = re.compile(r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<text>.*?)</a>', re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")


def parse_duckduckgo_html(body: str, *, limit: int = 8) -> list[SearchHit]:
    titles = _RESULT_ANCHOR.findall(body)
    snippets = [_clean(match) for match in _SNIPPET.findall(body)]
    hits: list[SearchHit] = []
    for index, (href, title) in enumerate(titles[:limit]):
        url = _unwrap_redirect(html.unescape(href))
        if not url.startswith("http"):
            continue
        hits.append(SearchHit(title=_clean(title), url=url, snippet=snippets[index] if index < len(snippets) else ""))
    return hits


def _clean(fragment: str) -> str:
    return html.unescape(_TAGS.sub("", fragment)).strip()


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps results in ``/l/?uddg=<encoded>``; unwrap to the target."""

    if "duckduckgo.com/l/" not in href and not href.startswith("/l/"):
        return href
    query = urllib.parse.urlparse(href).query
    target = urllib.parse.parse_qs(query).get("uddg")
    return urllib.parse.unquote(target[0]) if target else href


@dataclass
class FetchedDocument:
    url: str
    ok: bool
    status: int = 0
    content_type: str = ""
    text: str = ""
    error: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "ok": self.ok,
            "status": self.status,
            "content_type": self.content_type,
            "text": self.text,
            "error": self.error,
            "fetched_at": self.fetched_at,
        }


class DocumentFetcher:
    """Fetches a URL and reduces HTML to readable text."""

    def __init__(self, *, timeout: float = 25.0, max_chars: int = 20000, opener=None) -> None:
        self.timeout = timeout
        self.max_chars = max_chars
        self._opener = opener or urllib.request.urlopen

    def fetch(self, url: str) -> FetchedDocument:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return FetchedDocument(url=url, ok=False, error="only http(s) URLs may be fetched")
        # Blocking loopback and link-local addresses keeps a model-chosen URL
        # from being aimed at services on this machine or the local network.
        if _is_local_host(parsed.hostname or ""):
            return FetchedDocument(url=url, ok=False, error="refusing to fetch a local or private address")

        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"})
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read(self.max_chars * 8)
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "") if hasattr(response, "headers") else ""
        except urllib.error.HTTPError as exc:
            return FetchedDocument(url=url, ok=False, status=exc.code, error=f"HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return FetchedDocument(url=url, ok=False, error=f"unreachable (offline?): {exc}")

        text = raw.decode("utf-8", errors="replace")
        if "html" in content_type.lower() or text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
            text = html_to_text(text)
        return FetchedDocument(
            url=url,
            ok=True,
            status=status,
            content_type=content_type,
            text=text[: self.max_chars],
        )


_SCRIPT_STYLE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_BREAKS = re.compile(r"</(p|div|li|tr|h[1-6]|pre|section|article)>", re.IGNORECASE)
_BLANKS = re.compile(r"\n{3,}")


def html_to_text(document: str) -> str:
    """Reduce HTML to text, keeping code blocks and paragraph structure."""

    body = _SCRIPT_STYLE.sub(" ", document)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = _BREAKS.sub("\n\n", body)
    body = _TAGS.sub("", body)
    body = html.unescape(body)
    lines = [line.rstrip() for line in body.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(line for line in lines if line.strip() or True)).strip()


def _is_local_host(hostname: str) -> bool:
    lowered = hostname.lower()
    if lowered in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    if lowered.endswith(".local") or lowered.endswith(".internal"):
        return True
    return bool(re.match(r"^(10\.|127\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)", lowered))


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def make_web_tools(
    *, backend: SearchBackend | None = None, fetcher: DocumentFetcher | None = None
) -> list[ToolSpec]:
    """Build the research toolset, with injectable backends for testing."""

    search_backend = backend or DuckDuckGoBackend()
    document_fetcher = fetcher or DocumentFetcher()

    def web_search(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = str(arguments["query"]).strip()
        if not query:
            raise ToolError("query must not be empty", kind="invalid_arguments")
        limit = max(1, min(int(arguments.get("limit") or 6), 15))
        hits = search_backend.search(query, limit=limit)
        ranked = sorted(hits, key=lambda hit: (0 if _is_documentation(hit.url) else 1))
        return {
            "query": query,
            "backend": search_backend.name,
            "count": len(ranked),
            "results": [hit.to_dict() for hit in ranked],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    def fetch_url(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        document = document_fetcher.fetch(str(arguments["url"]))
        if not document.ok:
            raise ToolError(document.error or "fetch failed", kind="fetch_failed")
        max_chars = int(arguments.get("max_chars") or context.max_output_chars)
        payload = document.to_dict()
        payload["text"] = document.text[:max_chars]
        return payload

    return [
        ToolSpec(
            name="web_search",
            purpose="Search the public web for technical documentation. Needs no API key.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
            adapter=web_search,
            risk=RiskLevel.MODERATE,
            tags=("research", "network"),
            example='{"name": "web_search", "arguments": {"query": "python mss screen capture example"}}',
        ),
        ToolSpec(
            name="fetch_url",
            purpose="Fetch a public documentation page and return it as readable text.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
                "required": ["url"],
            },
            adapter=fetch_url,
            risk=RiskLevel.MODERATE,
            tags=("research", "network"),
            example='{"name": "fetch_url", "arguments": {"url": "https://docs.python.org/3/library/json.html"}}',
        ),
    ]


def _is_documentation(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(host == known or host.endswith("." + known) or known in host for known in DOCUMENTATION_HOSTS)


def network_available(*, timeout: float = 5.0, opener=None) -> bool:
    """Cheap connectivity check, so the agent can plan for being offline."""

    open_url = opener or urllib.request.urlopen
    request = urllib.request.Request("https://duckduckgo.com/", method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with open_url(request, timeout=timeout):
            return True
    except Exception:
        return False
