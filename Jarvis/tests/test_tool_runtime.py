from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.builtin import builtin_tools
from tools.registry import (
    AuditLog,
    RiskLevel,
    ToolCall,
    ToolContext,
    ToolError,
    ToolPolicy,
    ToolRegistry,
    ToolSpec,
)
from tools.web import DocumentFetcher, SearchHit, html_to_text, make_web_tools, parse_duckduckgo_html


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "app.py").write_bytes(b"def add(a, b):\n    return a + b\n")
    (root / "notes.md").write_bytes(b"# notes\nsearch me\n")
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_bytes(b"class Helper:\n    pass\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
    return root


@pytest.fixture
def registry(tmp_path):
    reg = ToolRegistry(
        policy=ToolPolicy(max_risk=RiskLevel.MODERATE),
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    reg.register_many(builtin_tools())
    return reg


@pytest.fixture
def context(workspace):
    return ToolContext(workspace=workspace, timeout_seconds=120.0, allowed_paths=[], protected_paths=[])


def call(registry, context, tool_name, /, **arguments):
    """Invoke a tool.

    ``tool_name`` is positional-only so a tool whose own argument happens to be
    called ``name`` (``which``) does not collide with this helper's parameter.
    """

    return registry.invoke(ToolCall(name=tool_name, arguments=arguments), context)


# ------------------------------------------------------------------ registry


def test_unknown_tool_returns_a_failed_result_not_an_exception(registry, context):
    result = call(registry, context, "no_such_tool")
    assert not result.ok
    assert result.error_kind == "unknown_tool"
    assert "Available:" in result.error


def test_unknown_tool_suggests_a_close_name(registry, context):
    result = call(registry, context, "read_fil", path="app.py")
    assert "read_file" in result.error


def test_missing_required_argument_is_reported_as_retryable(registry, context):
    result = call(registry, context, "read_file")
    assert not result.ok and result.retryable
    assert "missing required argument" in result.error


def test_arguments_are_coerced_for_sloppy_models(registry, context):
    result = call(registry, context, "read_file", path="app.py", start_line="1", end_line="1")
    assert result.ok
    assert result.output["content"] == "def add(a, b):"


def test_adapter_exception_becomes_a_result(context):
    def explode(arguments, ctx):
        raise KeyError("boom")

    registry = ToolRegistry()
    registry.register(ToolSpec(name="explode", purpose="", input_schema={}, adapter=explode))
    result = call(registry, context, "explode")
    assert not result.ok and result.error_kind == "adapter_exception"
    assert "boom" in result.error


def test_tool_call_parses_the_dialects_models_actually_emit():
    assert ToolCall.from_payload({"name": "read_file", "arguments": {"path": "a"}}).arguments == {"path": "a"}
    assert ToolCall.from_payload({"tool": "read_file", "args": {"path": "a"}}).arguments == {"path": "a"}
    assert ToolCall.from_payload({"name": "read_file", "arguments": '{"path": "a"}'}).arguments == {"path": "a"}
    # Arguments inlined next to the name, with the model's own commentary dropped.
    assert ToolCall.from_payload({"name": "read_file", "path": "a", "thought": "hmm"}).arguments == {"path": "a"}


def test_malformed_tool_payload_is_audited_as_a_failure(tmp_path, context):
    registry = ToolRegistry(audit=AuditLog(tmp_path / "audit.jsonl"))
    result = registry.invoke_payload({"arguments": {}}, context)
    assert not result.ok and result.error_kind == "invalid_call"
    assert registry.audit.entries


# ------------------------------------------------------------------ policy


def test_risk_above_the_ceiling_is_denied_without_an_approver(context):
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.SAFE))
    registry.register_many(builtin_tools())
    result = call(registry, context, "run_tests")
    assert not result.ok
    assert result.error_kind == "permission_denied"
    assert not result.retryable


def test_approver_can_raise_the_ceiling_for_one_run(context):
    asked = []

    def approve(spec, call_):
        asked.append(spec.name)
        return True

    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.SAFE, approve=approve))
    registry.register_many(builtin_tools())
    result = call(registry, context, "check_syntax")
    assert result.ok or result.error_kind != "permission_denied"

    denied = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.SAFE, approve=lambda spec, c: False))
    denied.register_many(builtin_tools())
    assert denied.invoke(ToolCall(name="run_tests"), context).error_kind == "permission_denied"


def test_deny_list_wins_over_everything(context):
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.CRITICAL, deny=["run_command"]))
    registry.register_many(builtin_tools())
    assert call(registry, context, "run_command", command=["python", "-c", "print(1)"]).error_kind == "permission_denied"


def test_description_hides_tools_the_policy_would_refuse(context):
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.SAFE))
    registry.register_many(builtin_tools())
    described = {item["name"] for item in registry.describe()}
    assert "read_file" in described
    assert "run_command" not in described, "offering a tool that will be refused wastes a model cycle"


def test_prompt_rendering_lists_signatures(registry):
    rendered = registry.render_for_prompt(tags=["investigate"])
    assert "read_file(" in rendered
    assert "search_text(" in rendered


# ------------------------------------------------------------------ audit


def test_every_invocation_is_audited(registry, context, tmp_path):
    call(registry, context, "read_file", path="app.py")
    call(registry, context, "read_file", path="does-not-exist.py")
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["ok"] is True
    assert json.loads(lines[1])["ok"] is False


def test_secrets_are_redacted_from_the_audit_trail():
    redacted = AuditLog.redact({"url": "https://x", "api_key": "sk-secret", "nested": {"auth_token": "t"}})
    assert redacted["api_key"] == "<redacted>"
    assert redacted["nested"]["auth_token"] == "<redacted>"
    assert redacted["url"] == "https://x"


# ------------------------------------------------------------------ filesystem


def test_list_files_skips_caches_and_binaries(registry, context):
    files = call(registry, context, "list_files").output["files"]
    assert "app.py" in files and "pkg/util.py" in files
    assert not any("__pycache__" in item for item in files)


def test_search_text_finds_content(registry, context):
    output = call(registry, context, "search_text", query="search me").output
    assert output["count"] == 1
    assert output["matches"][0]["path"] == "notes.md"


def test_search_text_rejects_a_broken_regex(registry, context):
    result = call(registry, context, "search_text", query="(unclosed", regex=True)
    assert not result.ok and result.error_kind == "invalid_arguments"


def test_find_definition_distinguishes_definition_from_mention(registry, context, workspace):
    (workspace / "uses.py").write_bytes(b"from pkg.util import Helper\nh = Helper()\n")
    output = call(registry, context, "find_definition", symbol="Helper").output
    assert output["count"] == 1
    assert output["definitions"][0]["path"] == "pkg/util.py"


@pytest.mark.parametrize("bad", ["../escape.py", "pkg/../../escape.py"])
def test_reads_cannot_escape_the_workspace(registry, context, bad):
    result = call(registry, context, "read_file", path=bad)
    assert not result.ok
    assert result.error_kind in {"path_denied", "missing_file"}


def test_apply_edits_goes_through_the_edit_engine(registry, context, workspace):
    result = call(
        registry,
        context,
        "apply_edits",
        files=[{"path": "app.py", "search": "a + b", "replace": "a * b"}],
    )
    assert result.ok
    assert (workspace / "app.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a * b\n"


def test_apply_edits_respects_protected_paths(registry, workspace):
    context = ToolContext(workspace=workspace, protected_paths=["app.py"])
    before = (workspace / "app.py").read_bytes()
    result = call(registry, context, "apply_edits", files=[{"path": "app.py", "search": "a + b", "replace": "a * b"}])
    assert not result.ok and result.error_kind == "protected_path"
    assert not result.retryable
    assert (workspace / "app.py").read_bytes() == before


def test_failed_multi_file_edit_leaves_nothing_changed(registry, context, workspace):
    before = (workspace / "app.py").read_bytes()
    result = call(
        registry,
        context,
        "apply_edits",
        files=[
            {"path": "app.py", "search": "a + b", "replace": "a * b"},
            {"path": "pkg/util.py", "search": "this anchor is absent", "replace": "x"},
        ],
    )
    assert not result.ok
    assert (workspace / "app.py").read_bytes() == before


def test_write_file_creates_then_updates(registry, context, workspace):
    assert call(registry, context, "write_file", path="new.py", content="x = 1\n").ok
    assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 1\n"
    assert call(registry, context, "write_file", path="new.py", content="x = 2\n").ok
    assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 2\n"


# ------------------------------------------------------------------ process


def test_run_command_executes_and_captures_output(registry, context):
    output = call(registry, context, "run_command", command=["python", "-c", "print('hello tools')"]).output
    assert output["success"]
    assert "hello tools" in output["stdout"]


def test_run_command_reports_a_nonzero_exit_as_data(registry, context):
    output = call(registry, context, "run_command", command=["python", "-c", "raise SystemExit(3)"]).output
    assert output["returncode"] == 3
    assert output["success"] is False


def test_run_command_enforces_the_executable_allowlist(registry, context):
    result = call(registry, context, "run_command", command=["curl", "http://example.com"])
    assert not result.ok and result.error_kind == "command_denied"
    assert not result.retryable


def test_run_command_times_out_instead_of_hanging(registry, context):
    result = call(
        registry,
        context,
        "run_command",
        command=["python", "-c", "import time; time.sleep(30)"],
        timeout_seconds=2,
    )
    assert not result.ok and result.error_kind == "timeout"


def test_subprocess_environment_excludes_credentials(registry, context, monkeypatch):
    monkeypatch.setenv("JARVIS_SECRET_API_KEY", "sk-do-not-leak")
    output = call(
        registry,
        context,
        "run_command",
        command=["python", "-c", "import os; print(os.environ.get('JARVIS_SECRET_API_KEY', 'ABSENT'))"],
    ).output
    assert "ABSENT" in output["stdout"]
    assert "sk-do-not-leak" not in output["stdout"]


def test_check_syntax_finds_a_broken_file(registry, context, workspace):
    (workspace / "broken.py").write_bytes(b"def f(:\n")
    output = call(registry, context, "check_syntax", paths=["broken.py"]).output
    assert not output["ok"]
    assert output["errors"][0]["path"] == "broken.py"


def test_run_tests_reports_a_parsed_summary(registry, context, workspace):
    (workspace / "test_sample.py").write_bytes(b"def test_ok():\n    assert True\n\ndef test_bad():\n    assert False\n")
    output = call(registry, context, "run_tests", target="test_sample.py").output
    assert output["summary"].get("passed") == 1
    assert output["summary"].get("failed") == 1


def test_git_write_commands_are_refused_by_default(registry, context):
    result = call(registry, context, "git", args=["commit", "-m", "nope"])
    assert not result.ok and result.error_kind == "command_denied"


def test_git_read_commands_are_allowed(registry, context):
    result = call(registry, context, "git", args=["status", "--short"])
    assert result.ok  # a non-repo still exits non-zero, but the tool ran


# ------------------------------------------------------------------ packages


def test_install_packages_refuses_a_suspicious_specifier(registry, context):
    result = call(registry, context, "install_packages", packages=["requests; rm -rf /"])
    assert not result.ok and result.error_kind == "invalid_arguments"
    assert not result.retryable


def test_install_packages_requires_a_list(registry, context):
    assert not call(registry, context, "install_packages", packages=[]).ok


def test_which_reports_a_definitely_present_program(registry, context):
    output = call(registry, context, "which", name=Path(sys.executable).stem).output
    assert output["found"] in {True, False}  # environment dependent
    assert not call(registry, context, "which", name="definitely-not-installed-xyz").output["found"]


# ------------------------------------------------------------------ research


class StubBackend:
    name = "stub"

    def __init__(self, hits=None, error=None):
        self._hits = hits or []
        self._error = error

    def search(self, query, *, limit):
        if self._error:
            raise self._error
        return self._hits[:limit]


class StubFetcher(DocumentFetcher):
    def __init__(self, document):
        self._document = document

    def fetch(self, url):
        return self._document


def _research_registry(backend=None, fetcher=None):
    registry = ToolRegistry(policy=ToolPolicy(max_risk=RiskLevel.MODERATE))
    registry.register_many(make_web_tools(backend=backend, fetcher=fetcher))
    return registry


def test_search_ranks_documentation_hosts_first(context):
    backend = StubBackend(
        [
            SearchHit("Random blog", "https://example.com/post"),
            SearchHit("Python docs", "https://docs.python.org/3/library/json.html"),
        ]
    )
    output = call(_research_registry(backend=backend), context, "web_search", query="json").output
    assert output["results"][0]["url"].startswith("https://docs.python.org")
    assert output["retrieved_at"]


def test_offline_search_is_an_ordinary_tool_failure(context):
    backend = StubBackend(error=ToolError("search is unreachable (offline?)", kind="offline"))
    result = call(_research_registry(backend=backend), context, "web_search", query="anything")
    assert not result.ok
    assert result.error_kind == "offline"
    assert result.retryable, "being offline is temporary; the agent may try again later"


def test_fetch_refuses_local_addresses(context):
    """A model-chosen URL must not be aimed at services on this machine."""

    registry = _research_registry()
    for url in ("http://127.0.0.1:11434/api/tags", "http://192.168.1.5/admin", "http://localhost/"):
        result = call(registry, context, "fetch_url", url=url)
        assert not result.ok, url
        assert "local or private" in result.error


def test_fetch_refuses_non_http_schemes(context):
    result = call(_research_registry(), context, "fetch_url", url="file:///etc/passwd")
    assert not result.ok


def test_fetched_html_is_reduced_to_readable_text(context):
    from tools.web import FetchedDocument

    document = FetchedDocument(
        url="https://docs.python.org/x",
        ok=True,
        status=200,
        content_type="text/html",
        text=html_to_text("<html><body><script>bad()</script><p>Hello</p><pre>code()</pre></body></html>"),
    )
    output = call(_research_registry(fetcher=StubFetcher(document)), context, "fetch_url", url="https://docs.python.org/x").output
    assert "Hello" in output["text"]
    assert "code()" in output["text"]
    assert "bad()" not in output["text"]
    assert output["fetched_at"], "provenance must be recorded so research is citable"


def test_duckduckgo_html_parsing_unwraps_redirects():
    body = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">Python <b>docs</b></a>'
        '<a class="result__snippet">The official docs</a>'
    )
    hits = parse_duckduckgo_html(body)
    assert hits[0].url == "https://docs.python.org/3/"
    assert hits[0].title == "Python docs"
    assert hits[0].snippet == "The official docs"


def test_duckduckgo_parsing_degrades_to_empty_on_layout_change():
    assert parse_duckduckgo_html("<html><body>totally different markup</body></html>") == []
