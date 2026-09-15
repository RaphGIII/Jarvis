"""§24, §39, §51 -- the ordinary owner-facing product never names a vendor, a model, an internal role or a raw failure.

Every occurrence of the words below is classified:
  A. advanced technical diagnostics -- allowed (Settings › Erweitert: owner.js, diagnostics.js, release.js)
  B. developer source and tests -- allowed (Python source comments/identifiers, tests)
  C. normal owner-facing product -- NOT allowed: target count zero.

Class C is what this test enforces: the visible strings of the ordinary UI
(JavaScript string literals and HTML text) and the sentences the core hands
the owner.  Identifiers, comments and regular expressions that only classify
text are not owner-facing and are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"

#: Files whose visible strings may name vendors: the advanced surfaces (class A).
ADVANCED = {"ui/views/owner.js", "ui/views/diagnostics.js", "ui/views/release.js"}
#: Words that must not appear in ordinary owner-facing strings.
LEAK = re.compile(r"\b(Google|Gemini|OpenAI|GPT|ChatGPT|Anthropic|Claude|Groq|Cerebras|OpenRouter|Qwen|Ollama|Jarvis|FAST_LOCAL|BUILD_LOCAL"
                  r"|provider_unavailable|reasoning\.free|reasoning\.smart|reasoning\.deep|engineer\.standard|engineer\.frontier|local\.fast"
                  r"|GoalSpec|PlanSpec|SelfDev|HTTP 429|HTTP 503|quota_exhausted|Traceback)\b")
#: A string literal in JavaScript: "...", '...' or `...` (no nesting needed for a scan).
JS_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`', re.S)
#: Strings that are not owner-facing even inside a normal view.
NOT_VISIBLE = re.compile(r"^(?:[\"'`])(?:/api/|\.\./|\./|#|\w+\.js|[A-Z_]{3,}$|\W*$|X-[A-Za-z-]+[\"'`]$)")


def _visible_strings(text: str) -> list[str]:
    # strip comments first: // ... and /* ... */
    code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    code = re.sub(r"(^|[^:\"'`])//[^\n]*", r"\1", code)
    out = []
    for match in JS_STRING.finditer(code):
        literal = match.group(0)
        if NOT_VISIBLE.match(literal):
            continue
        # regular expressions and class lists are not text the owner reads
        body = literal[1:-1]
        if re.search(r"\\b|\\w|\\s|\|\w+\|", body) and "/" not in body[:1]:
            if "|" in body and " " not in body.replace("|", ""):
                continue
        out.append(body)
    return out


def _leaks_in_js(path: Path) -> list[str]:
    hits = []
    for literal in _visible_strings(path.read_text(encoding="utf-8", errors="replace")):
        if LEAK.search(literal):
            # the knowledge view's ZEUS_HINT regex and the files shortcut path are classification/path strings, not sentences
            if "wakeword" in literal or "Jarvis_recovery" in literal:
                continue
            hits.append(f"{path.relative_to(ROOT).as_posix()}: {literal[:90]!r}")
    return hits


def test_ordinary_ui_strings_name_no_vendor_model_role_or_raw_failure():
    offenders: list[str] = []
    for path in sorted(UI.rglob("*.js")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ADVANCED or "/tests/" in rel:
            continue
        offenders += _leaks_in_js(path)
    html = (UI / "index.html").read_text(encoding="utf-8")
    visible_html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    visible_html = re.sub(r"<script.*?</script>|<style.*?</style>", "", visible_html, flags=re.S)
    visible_html = re.sub(r"<[^>]+>", " ", visible_html)
    visible_html = re.sub(r'content="[^"]*"', "", visible_html)
    for word in LEAK.findall(visible_html):
        offenders.append(f"ui/index.html: {word}")
    assert offenders == [], "class C (ordinary owner-facing) must be empty:\n" + "\n".join(offenders)


def test_advanced_surfaces_are_the_only_place_vendors_appear():
    """Class A exists and is explicit: the advanced views may name providers; nothing routes an ordinary user there by accident."""

    owner = (UI / "views" / "owner.js").read_text(encoding="utf-8")
    assert LEAK.search(owner), "the advanced provider panel legitimately names vendors"
    settings = (UI / "views" / "settings.js").read_text(encoding="utf-8")
    assert 'views.open("settings", { tab: id })' in settings and '"advanced", "Erweitert"' in settings
    assert not LEAK.search("\n".join(_visible_strings(settings))), "the settings hub itself stays vendor-free; owner.js is mounted inside Erweitert"


def test_owner_facing_sentences_from_the_core_are_zeus_level():
    """The sentences the core hands the owner: no vendor, no role, no raw status."""

    core = (ROOT / "service" / "core.py").read_text(encoding="utf-8")
    sentences = re.findall(r'_deliver\(\(?\s*f?"([^"\n]{12,})"', core) + re.findall(r'(?:FREE_UNAVAILABLE_(?:DE|EN)|INTELLIGENCE_UNAVAILABLE_(?:DE|EN)) = \(?"([^"\n]+)"', core)
    offenders = [s for s in sentences if LEAK.search(s) or re.search(r"\blokale KI\b|\bStream\b", s)]
    assert offenders == [], offenders
    from persona.smalltalk import identity_answer

    for question in ("Wer bist du?", "Welches Modell bist du?", "Bist du Gemini?", "Wer hat dich gebaut?"):
        answer = identity_answer(question, language="de", assistant="ZEUS", creator="Raphael") or ""
        assert not LEAK.search(answer), (question, answer)


def test_no_mode_chips_and_one_performance_selector():
    app = (UI / "app.js").read_text(encoding="utf-8")
    index = (UI / "index.html").read_text(encoding="utf-8")
    assert "modebar" not in index and "core/gateway.js" not in app
    assert 'id="perf"' in index and 'id="sidebar"' in index and 'id="workPill"' in index
    perf = (UI / "core" / "performance.js").read_text(encoding="utf-8")
    for label in ("Automatisch", "Ohne Kosten", "Mehr Leistung", "Maximale Leistung", "Entwicklung"):
        assert label in perf
    assert not LEAK.search("\n".join(_visible_strings(perf)))
