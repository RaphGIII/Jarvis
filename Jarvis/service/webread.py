"""Real page reading: fetch an article and make it summarizable.

Search snippets cannot answer "fass mir den Inhalt zusammen" — this module
fetches the actual page, strips it down to readable text (paragraph-first,
script/style/nav removed) and hands it to FAST_LOCAL with a bounded retry
ladder: full text, then half, then a quarter — a timeout shrinks the work
instead of killing the answer.  Stdlib only; a site that blocks or fails is
reported so the caller can try the next source.
"""

from __future__ import annotations

import html as html_mod
import re
import urllib.error
import urllib.request
from typing import Any

TIMEOUT_SECONDS = 14.0
MAX_BYTES = 900_000
MAX_TEXT = 24_000

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
_DROP = re.compile(r"<(script|style|noscript|svg|nav|header|footer|aside|form|iframe)\b.*?</\1>", re.S | re.I)
_COMMENTS = re.compile(r"<!--.*?-->", re.S)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_ARTICLE = re.compile(r"<(article|main)\b[^>]*>(.*?)</\1>", re.S | re.I)
_PARAGRAPH = re.compile(r"<(p|h1|h2|h3|li)\b[^>]*>(.*?)</\1>", re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return html_mod.unescape(_TAGS.sub(" ", fragment)).replace("\xa0", " ").strip()


def fetch_readable(url: str, *, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    """{"ok", "url", "title", "text"} — real page content or an honest error."""

    request = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                   "Accept-Language": "de-DE,de;q=0.9,en;q=0.7"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_BYTES).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}
    body = _COMMENTS.sub("", _DROP.sub("", body))
    title = _clean(_TITLE.search(body).group(1)) if _TITLE.search(body) else ""
    scope = body
    m = _ARTICLE.search(body)
    if m and len(m.group(2)) > 800:
        scope = m.group(2)
    blocks = [_clean(b) for _, b in _PARAGRAPH.findall(scope)]
    blocks = [b for b in blocks if len(b) >= 40 or (b and b == title)]
    text = "\n".join(blocks)[:MAX_TEXT]
    if len(text) < 300:
        # paragraph extraction failed (unusual markup): fall back to all text
        text = re.sub(r"\s+", " ", _clean(scope))[:MAX_TEXT]
    if len(text) < 200:
        return {"ok": False, "url": url, "title": title, "error": "kein lesbarer Inhalt gefunden (Seite blockiert oder leer)"}
    return {"ok": True, "url": url, "title": title, "text": text}


SUMMARY_PROMPT = """Fasse den folgenden Artikel in 4-6 Sätzen auf Deutsch zusammen.
Nur was im Text steht — nichts erfinden. Nenne die wichtigsten Fakten zuerst.

Titel: {title}

Artikel:
{text}

Zusammenfassung:"""


def summarize_with_retry(provider: Any, *, title: str, text: str, max_tokens: int = 400) -> dict[str, Any]:
    """Summarize with a shrinking-context ladder: full → 1/2 → 1/4."""

    attempts = []
    for fraction, label in ((1.0, "voll"), (0.5, "halb"), (0.25, "viertel")):
        chunk = text[: max(600, int(len(text) * fraction))]
        prompt = SUMMARY_PROMPT.format(title=title or "(ohne Titel)", text=chunk)
        try:
            answer = provider.generate(prompt, max_tokens=max_tokens, temperature=0.2)
            summary = str(answer).strip()
            if summary:
                return {"ok": True, "summary": summary, "context": label, "chars": len(chunk)}
            attempts.append(f"{label}: leere Antwort")
        except Exception as exc:  # noqa: BLE001 - shrink and try again
            attempts.append(f"{label}: {type(exc).__name__}")
            continue
    return {"ok": False, "error": "Zusammenfassung fehlgeschlagen: " + "; ".join(attempts)}
