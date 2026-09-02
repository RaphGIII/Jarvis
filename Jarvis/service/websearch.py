"""A small, honest web search: Bing's HTML results page, stdlib only.

Used by the deterministic ``web.search`` intent so "such im Internet nach X"
answers in about a second with real result titles, URLs and snippets — and
says plainly when the network or the engine is not reachable.  No API key,
no browser.  (DuckDuckGo's HTML endpoint was tried first and serves an
anomaly page to non-browser clients; Bing serves parseable results.)

The caller composes the answer; this module only fetches and parses.  It
never invents a result: everything returned was present in the response.
"""

from __future__ import annotations

import base64
import html
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

ENDPOINT = "https://www.bing.com/search"
TIMEOUT_SECONDS = 8.0
MAX_RESULTS = 6

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
_ALGO = re.compile(r'<li class="b_algo".*?</li>', re.S)
_LINK = re.compile(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAGS = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    return html.unescape(_TAGS.sub("", fragment or "")).strip()


def _real_url(href: str) -> str:
    """Bing wraps results as /ck/a?…&u=a1<base64url>&…; unwrap to the target."""

    href = html.unescape(href)
    parsed = urllib.parse.urlparse(href)
    if "/ck/" in parsed.path:
        query = urllib.parse.parse_qs(parsed.query)
        packed = (query.get("u") or [""])[0]
        if packed.startswith("a1"):
            payload = packed[2:]
            payload += "=" * (-len(payload) % 4)
            try:
                return base64.urlsafe_b64decode(payload).decode("utf-8", "replace")
            except (ValueError, UnicodeDecodeError):
                return href
    return href


def search(query: str, *, limit: int = MAX_RESULTS) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"ok": False, "error": "leere Suchanfrage"}
    url = ENDPOINT + "?" + urllib.parse.urlencode({"q": query, "setlang": "de"})
    request = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept-Language": "de-DE,de;q=0.9,en;q=0.7"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(900_000).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"ok": False, "error": f"Suche nicht erreichbar: {exc}", "offline": True}
    results = []
    for block in _ALGO.findall(body):
        m = _LINK.search(block)
        if not m:
            continue
        target = _real_url(m.group(1))
        title = _text(m.group(2))
        sm = _SNIPPET.search(block)
        snippet = _text(sm.group(1)) if sm else ""
        if not title or not target.startswith("http"):
            continue
        results.append({"title": title[:160], "url": target[:400], "snippet": snippet[:280]})
        if len(results) >= max(1, int(limit)):
            break
    if not results:
        return {"ok": False, "error": "keine Ergebnisse gefunden (oder die Ergebnisseite hat sich geändert)", "query": query}
    return {"ok": True, "query": query, "results": results}
