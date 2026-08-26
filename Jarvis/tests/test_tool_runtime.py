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


def test_write_file_is_idempotent(registry, context, workspace):
    """Re-asserting content a file already has is success, not failure.

    write_file declares desired content. Treating a no-op as an error made an
    agent that correctly re-stated a finished file look like it was failing,
    which then triggered pointless repair cycles -- seen in a live run.
    """

    assert call(registry, context, "write_file", path="same.py", content="x = 1\n").ok
    result = call(registry, context, "write_file", path="same.py", content="x = 1\n")
    assert result.ok
    assert result.output["unchanged"] is True
    assert (workspace / "same.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_edits_still_rejects_a_no_op_search_replace(registry, context):
    """The no-op guard still matters where it was designed to: anchored edits."""

    result = call(
        registry,
        context,
        "apply_edits",
        files=[{"path": "app.py", "search": "return a + b", "replace": "return a + b"}],
    )
    assert not result.ok and result.error_kind == "no_effective_edit"


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
    output = call(registry, context, "find_program", name=Path(sys.executable).stem).output
    assert output["found"] in {True, False}  # environment dependent
    assert not call(registry, context, "find_program", name="definitely-not-installed-xyz").output["found"]


# -- find_program answers about availability, not about PATH ----------------
#
# It used to run shutil.which and nothing else, so "is pyautogui available
# here" came back {found: false, path: ""} -- true of the PATH, false of the
# machine, and read by the model as the second. Measured live during the
# screen-capture acquisition of 2026-08-26: pyautogui was installed and
# importable, the project's own briefing said so, and the tool said no.


def test_an_importable_package_is_not_reported_as_absent(registry, context):
    """The defect, stated as a test. `json` is importable in every interpreter
    that can run this suite and is on no PATH anywhere."""

    output = call(registry, context, "find_program", name="json").output

    assert output["found"] is True
    assert output["kind"] == "python_package"
    assert output["importable"] is True


def test_a_python_package_offers_no_path_to_execute(registry, context):
    """A path is what a caller hands to subprocess. There isn't one, and
    returning something plausible would be worse than returning nothing."""

    output = call(registry, context, "find_program", name="json").output

    assert output["path"] == ""
    assert "import json" in output["answer"]


def test_an_executable_is_still_reported_as_one(registry, context):
    output = call(registry, context, "find_program", name=Path(sys.executable).stem).output

    if output["found"] and output["kind"] == "executable":
        assert output["path"]
        assert "subprocess" in output["answer"]


def test_something_that_is_neither_says_so_in_words(registry, context):
    output = call(registry, context, "find_program", name="definitely-not-installed-xyz").output

    assert output["found"] is False
    assert output["kind"] == "absent"
    assert output["importable"] is False
    assert "neither" in output["answer"]


def test_a_name_that_cannot_be_a_module_does_not_raise(registry, context):
    """A hyphen makes a name unimportable rather than an error. Every question
    this tool is asked has an answer; none of them is a traceback."""

    output = call(registry, context, "find_program", name="not-an-identifier").output

    assert output["found"] is False
    assert output["kind"] == "absent"


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


# ------------------------------------- the two write tools have distinct jobs

def test_apply_edits_requires_an_anchor_and_points_at_write_file(registry, context, workspace):
    """A local model reaches for "replace the whole file" almost every time.

    Left free to do that through apply_edits, an edit meant to add one import
    arrived as a file containing only that import. Refusing the shape and
    naming the right tool works where prompting did not.
    """

    before = (workspace / "app.py").read_bytes()
    result = call(registry, context, "apply_edits", files=[{"path": "app.py", "replace": "import shutil\n"}])

    assert not result.ok
    assert result.error_kind == "empty_search"
    assert result.retryable
    assert "write_file" in result.error
    assert (workspace / "app.py").read_bytes() == before


def test_write_file_is_still_allowed_to_set_whole_contents(registry, context, workspace):
    """The escape hatch apply_edits points at must actually work."""

    body = "\n".join(f"line_{index} = {index}" for index in range(30)) + "\n"
    assert call(registry, context, "write_file", path="app.py", content=body).ok
    assert (workspace / "app.py").read_text(encoding="utf-8") == body


def test_write_file_cannot_gut_a_real_module(registry, context, workspace):
    """Observed live: main.py reduced to a single `import shutil`."""

    body = "\n".join(f"line_{index} = {index}" for index in range(30)) + "\n"
    call(registry, context, "write_file", path="app.py", content=body)

    result = call(registry, context, "write_file", path="app.py", content="import shutil\n")

    assert not result.ok
    assert result.error_kind == "rewrite_truncates_file"
    assert (workspace / "app.py").read_text(encoding="utf-8") == body


def test_small_files_can_still_be_replaced_wholesale(registry, context, workspace):
    call(registry, context, "write_file", path="tiny.py", content="a = 1\nb = 2\n")
    assert call(registry, context, "write_file", path="tiny.py", content="a = 9\n").ok


def test_an_anchorless_edit_may_fill_an_empty_file(registry, context, workspace):
    """An anchor is only required where there is something to lose.

    A live run created an empty test file and then spent three cycles being
    refused permission to put the tests in it.
    """

    (workspace / "empty.py").write_bytes(b"")
    result = call(
        registry, context, "apply_edits", files=[{"path": "empty.py", "replace": "def test_x():\n    assert True\n"}]
    )
    assert result.ok, result.error
    assert "assert True" in (workspace / "empty.py").read_text(encoding="utf-8")


def test_an_anchorless_edit_may_create_a_missing_file(registry, context, workspace):
    assert call(registry, context, "apply_edits", files=[{"path": "brand_new.py", "replace": "x = 1\n"}]).ok
    assert (workspace / "brand_new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_an_anchorless_edit_is_still_refused_on_a_file_with_content(registry, context, workspace):
    before = (workspace / "app.py").read_bytes()
    result = call(registry, context, "apply_edits", files=[{"path": "app.py", "replace": "import shutil\n"}])
    assert not result.ok and result.error_kind == "empty_search"
    assert (workspace / "app.py").read_bytes() == before


def test_a_repeated_anchor_miss_is_told_to_rewrite_the_file(registry, context, workspace):
    """Observed live during a capability build: the model diagnosed the bug
    correctly three times running and missed the same anchor every time.

    The advice that would have unblocked it existed but was gated on file
    size, and the file was nine lines over the threshold. Repetition is the
    stronger signal -- a second miss on one path means the text being matched
    against is not what the file holds, so no further anchor invented from
    that picture will match either. It also needs no threshold to mistune.
    """

    long_file = "\n".join(f"line {index}" for index in range(200)) + "\n"
    (workspace / "long.py").write_text(long_file, encoding="utf-8")
    edit = {"path": "long.py", "search": "def absent():\n    return 1", "replace": "x"}

    first = call(registry, context, "apply_edits", files=[dict(edit)])
    second = call(registry, context, "apply_edits", files=[dict(edit)])

    assert not first.ok and not second.ok
    assert "read_file" not in first.error, "one miss is not yet evidence of drift"
    assert "2 failed anchors" in second.error
    assert "read_file" in second.error


def test_a_short_file_is_told_to_rewrite_on_the_first_miss(registry, context, workspace):
    """Size is still a signal: a file small enough to reproduce reliably does
    not need a second failure to prove the point."""

    (workspace / "small.py").write_text("x = 1\ny = 2\n", encoding="utf-8")

    result = call(
        registry, context, "apply_edits",
        files=[{"path": "small.py", "search": "def absent():\n    return 1", "replace": "z"}],
    )

    assert not result.ok
    assert "write_file" in result.error
    assert "only 2 lines" in result.error


def test_anchor_misses_are_counted_per_file_not_globally(registry, context, workspace):
    """Drift in one file says nothing about another."""

    (workspace / "a.py").write_text("\n".join(f"a{i}" for i in range(200)), encoding="utf-8")
    (workspace / "b.py").write_text("\n".join(f"b{i}" for i in range(200)), encoding="utf-8")
    absent = {"search": "def absent():\n    return 1", "replace": "x"}

    call(registry, context, "apply_edits", files=[{"path": "a.py", **absent}])
    other = call(registry, context, "apply_edits", files=[{"path": "b.py", **absent}])

    assert "write_file" not in other.error


def test_a_long_file_is_never_told_to_rewrite_itself_from_memory(registry, context, workspace):
    """The advice that unblocks a sixty-line file destroys a seven-hundred-line one.

    Observed during a live capability repair: told to "call write_file with the
    complete corrected contents", the model replaced a 688-line module with a
    37-line sketch, and only the edit engine's shrink guard stopped a working
    implementation from being lost to fix one number. The drift is the same;
    the remedy is to go and look, not to retype.
    """

    long_file = "\n".join(f"line {index}" for index in range(700)) + "\n"
    (workspace / "big.py").write_text(long_file, encoding="utf-8")
    edit = {"path": "big.py", "search": "def absent():\n    return 1", "replace": "x"}

    call(registry, context, "apply_edits", files=[dict(edit)])
    second = call(registry, context, "apply_edits", files=[dict(edit)])

    assert "write_file" not in second.error, "a 700-line file cannot be retyped"
    assert "read_file" in second.error
    assert "700 lines" in second.error


def test_the_two_remedies_do_not_get_mixed_up(registry, context, workspace):
    """Small files get 'rewrite it'; long ones get 'go and read it'."""

    (workspace / "tiny.py").write_text("a = 1\n", encoding="utf-8")
    (workspace / "huge.py").write_text("\n".join(f"l{i}" for i in range(500)), encoding="utf-8")
    absent = {"search": "def absent():\n    return 1", "replace": "z"}

    tiny = call(registry, context, "apply_edits", files=[{"path": "tiny.py", **absent}])
    call(registry, context, "apply_edits", files=[{"path": "huge.py", **absent}])
    huge = call(registry, context, "apply_edits", files=[{"path": "huge.py", **absent}])

    assert "write_file" in tiny.error and "read_file" not in tiny.error
    assert "read_file" in huge.error and "write_file" not in huge.error


# -- replace_definition: the door a large file never had --------------------
#
# Anchored editing has a hole in the middle of it and every large repair falls
# in. On a short file, a model that cannot land an anchor is told to send the
# whole file back, and that works. On a long one it cannot: retyping seven
# hundred lines from memory produces a sketch, and the shrink guard rightly
# refuses it. Both doors close. Measured on the Spotify latency repair -- eight
# consecutive anchor failures against a 708-line module, one write_file refused
# for shrinking it to 61 lines, and the attempt ended having changed nothing,
# with a correct diagnosis attached to every failure.

MODULE = "\n".join([
    '"""A module longer than anyone will retype."""',
    "",
    "import subprocess",
    "",
    "",
    "CONSTANT = 1",
    "",
    "",
    "def _powershell(verb, timeout=60.0):",
    '    """The old one."""',
    "    return subprocess.run(['powershell', '-Command', verb], timeout=timeout)",
    "",
    "",
    "class Player:",
    "    def play(self, track):",
    "        return _powershell('play ' + track)",
    "",
    "    def pause(self):",
    "        return _powershell('pause')",
    "",
    "",
    "def tail():",
    "    return CONSTANT",
    "",
])

FASTER = "\n".join([
    "def _powershell(verb, timeout=60.0):",
    "    script = _script_path()",
    "    return subprocess.run(['powershell', '-File', str(script)], timeout=timeout)",
    "",
])


@pytest.fixture
def module(context):
    (Path(context.workspace) / "big.py").write_text(MODULE, encoding="utf-8")
    return Path(context.workspace) / "big.py"


def test_a_function_is_replaced_without_any_anchor(registry, context, module):
    result = call(registry, context, "replace_definition",
                  path="big.py", name="_powershell", source=FASTER)

    assert result.ok, result.error
    body = module.read_text(encoding="utf-8")
    assert "-File" in body
    assert "The old one." not in body


def test_everything_outside_the_definition_survives(registry, context, module):
    """This is the whole reason the tool is safe to name in the advice: a model
    that reaches for it too eagerly cannot lose the rest of the file."""

    call(registry, context, "replace_definition",
         path="big.py", name="_powershell", source=FASTER)
    body = module.read_text(encoding="utf-8")

    assert "CONSTANT = 1" in body
    assert "def tail():" in body
    assert "class Player:" in body
    assert '"""A module longer than anyone will retype."""' in body


def test_a_method_keeps_the_indentation_of_its_class(registry, context, module):
    result = call(registry, context, "replace_definition", path="big.py", name="Player.pause",
                  source="def pause(self):\n    return _powershell('stop')\n")

    assert result.ok, result.error
    body = module.read_text(encoding="utf-8")
    assert "    def pause(self):" in body
    assert "        return _powershell('stop')" in body
    assert "    def play(self, track):" in body


def test_the_name_is_taken_from_the_replacement_when_the_given_one_is_wrong(registry, context, module):
    """The replacement says what it defines, in code, and it has to define
    exactly one thing anyway -- so `name` restates a fact already established,
    and a redundant parameter is somewhere to be wrong.

    Observed live during the archive-count repair: the model sent a perfectly
    good `def run(payload)` under the name "run.payload.get('source')", having
    over-generalised the dotted form documented for methods, and the edit was
    refused for the one part of the call that carried no information.
    """

    result = call(registry, context, "replace_definition", path="big.py",
                  name="_powershell.verb.timeout", source=FASTER)

    assert result.ok, result.error
    body = module.read_text(encoding="utf-8")
    assert "-File" in body
    assert "CONSTANT = 1" in body and "def tail():" in body


def test_a_replacement_naming_something_absent_is_still_refused(registry, context, module):
    """Inference finds the definition the replacement declares. It does not
    invent one that is not in the file."""

    result = call(registry, context, "replace_definition", path="big.py",
                  name="whatever", source="def not_in_the_file():\n    pass\n")

    assert not result.ok
    assert module.read_text(encoding="utf-8") == MODULE


def test_a_replacement_carrying_other_statements_is_refused(registry, context, module):
    """Only def/class statements were counted, so a module-level assignment
    travelled along with the definition and was spliced into the middle of the
    file. Observed live during the archive-count repair: a replacement carrying
    both `INPUT_SCHEMA = {...}` and `def run(...)`, refused downstream by the
    duplicate-definition guard with a message about INPUT_SCHEMA -- not what the
    model thought it was doing, and not where it would have looked."""

    result = call(registry, context, "replace_definition", path="big.py", name="tail",
                  source="CONSTANT = 99\n\n\ndef tail():\n    return CONSTANT\n")

    assert not result.ok
    assert "ONLY the definition" in result.error
    assert "NOTHING WAS WRITTEN" in result.error
    assert module.read_text(encoding="utf-8") == MODULE


def test_a_definition_with_a_decorator_is_still_one_statement(registry, context, module):
    """A decorator is part of the definition, not another statement."""

    result = call(registry, context, "replace_definition", path="big.py", name="tail",
                  source="@staticmethod\ndef tail():\n    return 7\n")

    assert result.ok, result.error
    assert "@staticmethod" in module.read_text(encoding="utf-8")


def test_a_name_that_is_not_there_lists_the_ones_that_are(registry, context, module):
    """An error a model can act on names the alternatives; a similarity score
    cannot be edited into a correct call."""

    result = call(registry, context, "replace_definition",
                  path="big.py", name="nope", source="def nope():\n    pass\n")

    assert not result.ok
    assert "_powershell" in result.error and "Player.pause" in result.error


def test_a_replacement_defining_something_else_is_refused(registry, context, module):
    result = call(registry, context, "replace_definition", path="big.py", name="tail",
                  source="def other():\n    pass\n")

    assert not result.ok
    assert "NOTHING WAS WRITTEN" in result.error
    assert module.read_text(encoding="utf-8") == MODULE


def test_a_replacement_that_does_not_parse_is_refused_before_writing(registry, context, module):
    result = call(registry, context, "replace_definition", path="big.py", name="tail",
                  source="def tail(:\n")

    assert not result.ok
    assert "NOTHING WAS WRITTEN" in result.error
    assert module.read_text(encoding="utf-8") == MODULE


def test_two_definitions_in_one_replacement_are_refused(registry, context, module):
    result = call(registry, context, "replace_definition", path="big.py", name="tail",
                  source="def tail():\n    pass\n\n\ndef extra():\n    pass\n")

    assert not result.ok
    assert module.read_text(encoding="utf-8") == MODULE


def test_replacing_a_definition_with_itself_is_not_silent_success(registry, context, module):
    """The same reasoning as the edit engine's no-op guard: an edit that changes
    nothing means the model's picture of the file and the file have diverged,
    and reporting success would send it looking in the wrong place next."""

    result = call(registry, context, "replace_definition", path="big.py", name="tail",
                  source="def tail():\n    return CONSTANT\n")

    assert not result.ok
    assert "already has exactly that body" in result.error


def test_a_file_that_does_not_parse_says_so_rather_than_guessing(registry, context):
    (Path(context.workspace) / "broken.py").write_text("def run(:\n", encoding="utf-8")

    result = call(registry, context, "replace_definition", path="broken.py", name="run",
                  source="def run():\n    pass\n")

    assert not result.ok
    assert "does not currently parse" in result.error


def test_a_missing_file_is_not_created_by_a_replacement(registry, context):
    result = call(registry, context, "replace_definition", path="absent.py", name="run",
                  source="def run():\n    pass\n")

    assert not result.ok
    assert not (Path(context.workspace) / "absent.py").exists()
