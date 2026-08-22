from __future__ import annotations

from capabilities.research import CapabilityResearcher, extract_text


SAMPLE_HTML = """
<html><head><style>body{color:red}</style><script>var x=1;</script></head>
<body><h1>hashlib</h1><p>Return a sha256 hash object; optionally initialized with data.</p></body></html>
"""


def test_extract_text_strips_script_and_style():
    text = extract_text(SAMPLE_HTML)
    assert "hashlib" in text
    assert "sha256 hash object" in text
    assert "color:red" not in text
    assert "var x=1" not in text


def test_topic_for_goal_matches_known_keyword():
    researcher = CapabilityResearcher()
    topic = researcher.topic_for_goal("Compute the SHA-256 checksum of a string.")
    assert topic is not None
    query, url = topic
    assert "hashlib.html" in url


def test_topic_for_goal_returns_none_for_unrelated_goal():
    researcher = CapabilityResearcher()
    assert researcher.topic_for_goal("Double an integer number.") is None


def test_research_fetches_and_summarizes_recognized_topic():
    researcher = CapabilityResearcher(fetch=lambda url: SAMPLE_HTML)

    note = researcher.research("Compute the SHA-256 checksum of a string.")

    assert note is not None
    assert note.fetched
    assert "hashlib.html" in note.source
    assert "sha256" in note.summary.lower() or "hashlib" in note.summary.lower()


def test_research_returns_none_for_unrecognized_goal():
    researcher = CapabilityResearcher(fetch=lambda url: SAMPLE_HTML)
    assert researcher.research("Double an integer number.") is None


def test_research_reports_fetch_failure_without_raising():
    def _boom(url: str) -> str:
        raise OSError("network unreachable")

    researcher = CapabilityResearcher(fetch=_boom)

    note = researcher.research("Compute a sha256 checksum of a string.")

    assert note is not None
    assert not note.fetched
    assert "network unreachable" in note.error
