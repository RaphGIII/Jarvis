"""Research that can be checked, using free sources only.

The failure this module is built against is a research tool that returns fluent
prose the user cannot verify. Every property tested here exists to make that
impossible: a finding cannot exist without a source and a verbatim quote, the
quote must really be in the document, source ranking is a fact about the domain
rather than a model's opinion, and with no network the report says so instead of
answering from the model's memory.
"""

from __future__ import annotations

import json

import pytest

from research.agent import (
    AUTHORITY,
    WEAK_SOURCE,
    Finding,
    ResearchAgent,
    ResearchReport,
    Source,
    authority_of,
)


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------

class Hit:
    def __init__(self, url, title=""):
        self.url = url
        self.title = title


class FakeSearch:
    def __init__(self, hits, fail=False):
        self.hits = hits
        self.fail = fail
        self.queries = []

    def search(self, query, *, limit=8):
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("no network")
        return list(self.hits)


class Doc:
    def __init__(self, text, ok=True, error=""):
        self.text = text
        self.ok = ok
        self.error = error


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages

    def fetch(self, url):
        return self.pages.get(url, Doc("", ok=False, error="404"))


class FakeBrain:
    """Returns whatever the test scripted for each schema."""

    def __init__(self, *, queries=None, findings=None, summary="a summary"):
        self._queries = queries or []
        self._findings = findings or []
        self._summary = summary
        self.prompts = []

    def generate_structured(self, prompt, schema, **_):
        self.prompts.append(prompt)
        if "queries" in (schema.get("properties") or {}):
            return json.dumps({"queries": self._queries})
        return json.dumps({"findings": self._findings})

    def generate(self, prompt, **_):
        self.prompts.append(prompt)
        return self._summary


PYTHON_DOC = (
    "The subprocess module allows you to spawn new processes. "
    "subprocess.run() waits for the command to complete by default. "
    "Use the timeout argument to avoid waiting forever."
)


# --------------------------------------------------------------------------
# Source ranking is a fact, not an opinion
# --------------------------------------------------------------------------

def test_official_documentation_outranks_a_blog():
    assert authority_of("https://docs.python.org/3/library/subprocess.html") > authority_of(
        "https://someblog.example.com/python-subprocess-tips"
    )


def test_a_project_host_outranks_a_qa_site():
    assert authority_of("https://github.com/psf/requests") > authority_of(
        "https://stackoverflow.com/questions/123"
    )


def test_a_documentation_shaped_path_beats_an_unknown_one():
    assert authority_of("https://example.com/docs/getting-started") > authority_of(
        "https://example.com/blog/post"
    )


def test_an_unknown_domain_is_weak_but_usable():
    assert authority_of("https://random.example.org/page") == WEAK_SOURCE


def test_a_malformed_url_scores_nothing():
    assert authority_of("not a url") == 0


def test_subdomains_inherit_authority():
    assert authority_of("https://mail.python.org/x") == authority_of("https://python.org/x")


def test_the_authority_table_is_ordered_highest_first():
    scores = [score for score, _ in AUTHORITY]
    assert scores == sorted(scores, reverse=True)


def test_sources_are_visited_best_first():
    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "c", "quote": "subprocess.run() waits for the command"}]),
        search=FakeSearch([Hit("https://blog.example.com/x"), Hit("https://docs.python.org/3/y")]),
        fetcher=FakeFetcher({
            "https://docs.python.org/3/y": Doc(PYTHON_DOC),
            "https://blog.example.com/x": Doc(PYTHON_DOC),
        }),
    )

    report = agent.research("how does subprocess.run work?", max_sources=2)

    assert report.sources[0].url.startswith("https://docs.python.org")


# --------------------------------------------------------------------------
# A claim without a checkable quote is not a finding
# --------------------------------------------------------------------------

def test_a_finding_carries_its_source_and_quote():
    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "run() waits", "quote": "subprocess.run() waits for the command"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/library/subprocess.html", "subprocess")]),
        fetcher=FakeFetcher({"https://docs.python.org/3/library/subprocess.html": Doc(PYTHON_DOC)}),
    )

    report = agent.research("does subprocess.run block?")

    assert report.grounded
    assert report.findings[0].source_url.startswith("https://docs.python.org")
    assert "waits for the command" in report.findings[0].excerpt


def test_a_quote_that_is_not_in_the_document_is_discarded():
    """Otherwise a hallucinated claim arrives wearing a real source URL."""

    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "it uses telepathy", "quote": "subprocess uses telepathy internally"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/x")]),
        fetcher=FakeFetcher({"https://docs.python.org/3/x": Doc(PYTHON_DOC)}),
    )

    report = agent.research("how does it work?")

    assert report.findings == []
    assert not report.grounded


def test_whitespace_differences_do_not_reject_a_real_quote():
    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "c", "quote": "subprocess.run()   waits\nfor the command"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/x")]),
        fetcher=FakeFetcher({"https://docs.python.org/3/x": Doc(PYTHON_DOC)}),
    )

    assert agent.research("q").findings


def test_a_trivially_short_quote_is_rejected():
    """Two words appear in every document and prove nothing."""

    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "c", "quote": "the"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/x")]),
        fetcher=FakeFetcher({"https://docs.python.org/3/x": Doc(PYTHON_DOC)}),
    )

    assert agent.research("q").findings == []


def test_a_finding_without_a_claim_is_dropped():
    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "", "quote": "subprocess.run() waits for the command"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/x")]),
        fetcher=FakeFetcher({"https://docs.python.org/3/x": Doc(PYTHON_DOC)}),
    )

    assert agent.research("q").findings == []


# --------------------------------------------------------------------------
# Offline, and other unhappy paths
# --------------------------------------------------------------------------

def test_with_no_network_it_says_so_rather_than_inventing():
    """The case where a plausible hallucination is likeliest and least detectable."""

    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "anything", "quote": "anything at all here"}]),
        search=FakeSearch([], fail=True),
        fetcher=FakeFetcher({}),
    )

    report = agent.research("what is the answer?")

    assert report.offline
    assert not report.grounded
    assert "no sources" in report.as_text().lower() or "search failed" in report.summary


def test_a_source_that_will_not_fetch_is_recorded_as_failed():
    agent = ResearchAgent(
        brain=FakeBrain(),
        search=FakeSearch([Hit("https://docs.python.org/3/gone")]),
        fetcher=FakeFetcher({}),
    )

    report = agent.research("q")

    assert report.sources[0].ok is False
    assert report.sources[0].error


def test_one_dead_source_does_not_stop_the_others():
    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "c", "quote": "subprocess.run() waits for the command"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/gone"), Hit("https://github.com/x")]),
        fetcher=FakeFetcher({"https://github.com/x": Doc(PYTHON_DOC)}),
    )

    report = agent.research("q", max_sources=2)

    assert report.grounded


def test_a_broken_model_does_not_prevent_searching():
    """The literal question is always a usable query."""

    class Broken:
        def generate_structured(self, *a, **k):
            raise RuntimeError("model down")

        def generate(self, *a, **k):
            raise RuntimeError("model down")

    search = FakeSearch([])
    agent = ResearchAgent(brain=Broken(), search=search, fetcher=FakeFetcher({}))

    agent.research("how do I do the thing?")

    assert search.queries == ["how do I do the thing?"]


def test_the_question_itself_is_always_one_of_the_queries():
    search = FakeSearch([])
    agent = ResearchAgent(brain=FakeBrain(queries=["other query"]), search=search, fetcher=FakeFetcher({}))

    agent.research("the original question")

    assert "the original question" in search.queries


def test_duplicate_urls_are_visited_once():
    fetcher = FakeFetcher({"https://docs.python.org/3/x": Doc(PYTHON_DOC)})
    agent = ResearchAgent(
        brain=FakeBrain(findings=[{"claim": "c", "quote": "subprocess.run() waits for the command"}]),
        search=FakeSearch([Hit("https://docs.python.org/3/x"), Hit("https://docs.python.org/3/x")]),
        fetcher=fetcher,
    )

    report = agent.research("q", max_sources=5)

    assert len(report.sources) == 1


# --------------------------------------------------------------------------
# Disagreement is shown, not settled
# --------------------------------------------------------------------------

def test_contradicting_sources_are_surfaced():
    report = ResearchReport(question="q")
    report.findings = [
        Finding("the flag is supported on Windows", "https://a.example/1", "quote one here"),
        Finding("the flag is not supported on Windows", "https://b.example/2", "quote two here"),
    ]
    agent = ResearchAgent()

    contradictions = agent._contradictions(report.findings)

    assert contradictions
    assert "a.example" in contradictions[0]["a"]
    assert "b.example" in contradictions[0]["b"]


def test_two_findings_from_the_same_source_are_not_a_contradiction():
    findings = [
        Finding("it is supported", "https://a.example/1", "q1 long enough"),
        Finding("it is not supported", "https://a.example/1", "q2 long enough"),
    ]

    assert ResearchAgent()._contradictions(findings) == []


def test_unrelated_findings_are_not_contradictions():
    findings = [
        Finding("python uses indentation for blocks", "https://a.example/1", "q1 long enough"),
        Finding("rust does not use a garbage collector", "https://b.example/2", "q2 long enough"),
    ]

    assert ResearchAgent()._contradictions(findings) == []


def test_contradictions_appear_in_the_readable_report():
    report = ResearchReport(question="q")
    report.findings = [
        Finding("the api is available", "https://a.example/1", "quote one here"),
        Finding("the api is not available", "https://b.example/2", "quote two here"),
    ]
    report.contradictions = ResearchAgent()._contradictions(report.findings)

    assert "SOURCES DISAGREE" in report.as_text()


# --------------------------------------------------------------------------
# The report a project or a model consumes
# --------------------------------------------------------------------------

def test_the_readable_report_cites_every_finding():
    report = ResearchReport(question="q", summary="s")
    report.findings = [Finding("a claim", "https://docs.python.org/3/x", "a verbatim quote here")]

    text = report.as_text()

    assert "https://docs.python.org/3/x" in text
    assert "a verbatim quote here" in text


def test_an_empty_report_says_nothing_was_found():
    assert "No sources answered" in ResearchReport(question="q").as_text()


def test_the_report_serialises_for_the_ui():
    report = ResearchReport(question="q")
    report.findings = [Finding("c", "https://x.example/1", "a long enough quote")]

    payload = report.to_dict()

    assert payload["grounded"] is True
    assert payload["findings"][0]["source_url"] == "https://x.example/1"


def test_findings_record_when_they_were_retrieved():
    assert Finding("c", "https://x/1", "quote long enough").retrieved_at


def test_a_report_with_no_findings_is_not_grounded():
    assert not ResearchReport(question="q", summary="confident prose").grounded
