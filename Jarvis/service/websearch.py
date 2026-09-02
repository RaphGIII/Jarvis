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


# --------------------------------------------------------------------------
# Canonical URLs for spoken site names
# --------------------------------------------------------------------------

#: "Öffne Wikipedia" names a site, not a URL.  The canonical map turns the
#: obvious names into their real addresses deterministically; anything with a
#: dot is treated as an address; everything else opens a results page — the
#: browser shows something useful either way, never a dead end.
KNOWN_SITES = {
    "wikipedia": "https://de.wikipedia.org", "wiki": "https://de.wikipedia.org",
    "google": "https://www.google.com", "youtube": "https://www.youtube.com",
    "github": "https://github.com", "gmail": "https://mail.google.com",
    "google maps": "https://maps.google.com", "maps": "https://maps.google.com",
    "amazon": "https://www.amazon.de", "ebay": "https://www.ebay.de",
    "reddit": "https://www.reddit.com", "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "chatgpt": "https://chatgpt.com", "openai": "https://openai.com",
    "anthropic": "https://www.anthropic.com", "claude": "https://claude.ai",
    "netflix": "https://www.netflix.com", "twitch": "https://www.twitch.tv",
    "twitter": "https://x.com", "x": "https://x.com",
    "instagram": "https://www.instagram.com", "facebook": "https://www.facebook.com",
    "whatsapp": "https://web.whatsapp.com", "whatsapp web": "https://web.whatsapp.com",
    "outlook": "https://outlook.live.com", "linkedin": "https://www.linkedin.com",
    "dropbox": "https://www.dropbox.com", "google drive": "https://drive.google.com",
    "drive": "https://drive.google.com", "translate": "https://translate.google.com",
    "google translate": "https://translate.google.com", "deepl": "https://www.deepl.com",
    "spotify": "https://open.spotify.com", "spotify web": "https://open.spotify.com",
    "chess": "https://www.chess.com", "chess com": "https://www.chess.com",
    "lichess": "https://lichess.org", "amboss": "https://www.amboss.com/de",
    "doccheck": "https://www.doccheck.com", "duden": "https://www.duden.de",
    "leo": "https://www.leo.org", "pubmed": "https://pubmed.ncbi.nlm.nih.gov",
    "moodle": "https://moodle.org", "discord": "https://discord.com/app",
    "twitter x": "https://x.com", "bing": "https://www.bing.com",
    "duckduckgo": "https://duckduckgo.com", "tagesschau": "https://www.tagesschau.de",
    "spiegel": "https://www.spiegel.de", "zeit": "https://www.zeit.de",
    "heise": "https://www.heise.de", "hackernews": "https://news.ycombinator.com",
    "hacker news": "https://news.ycombinator.com",
}


def resolve_site(name: str) -> tuple[str, str]:
    """(url, how) for a spoken website name.

    how: "url" (was already an address), "known" (canonical map),
    "search" (unknown name — a results page for it, never a dead end).
    """

    raw = str(name or "").strip().strip(" .!?\"'„“”")
    if not raw:
        return "", ""
    if re.match(r"^https?://", raw, re.I):
        return raw, "url"
    if re.match(r"^(?:www\.)?[\w-]{2,}(?:\.[a-z]{2,})+(?:/\S*)?$", raw, re.I):
        return "https://" + raw, "url"
    key = re.sub(r"\s+", " ", raw.lower().replace("-", " ").replace(".", " ")).strip()
    for probe in (key, key.replace(" ", "")):
        if probe in KNOWN_SITES:
            return KNOWN_SITES[probe], "known"
    return ENDPOINT + "?" + urllib.parse.urlencode({"q": raw, "setlang": "de"}), "search"


def known_site(name: str) -> bool:
    url, how = resolve_site(name)
    return how in {"known", "url"}
