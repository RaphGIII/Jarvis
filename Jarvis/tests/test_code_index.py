"""Adversarial fixtures for deterministic repository navigation.

Every fixture here is built so that plain lexical search picks the WRONG line:
the decoy mentions the search term more often, earlier in the file, and in
longer runs than the real implementation does.  These are not hypothetical --
they are distilled from the four consecutive live runs in which a 7B model was
asked to add an exit word to the CLI and four times edited the help string,
because the help string is genuinely the strongest lexical match in the file.

The contract under test: the correct executable region is selected anyway.
"""

from __future__ import annotations

import textwrap

import pytest

from development.code_index import CodeIndex, Role


def write(root, relative, source):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The exact shape that defeated the lexical selector four times running
# --------------------------------------------------------------------------

CLI_WITH_DECOY_HELP = '''
    """Console for Jarvis.

    Commands:
        /quit /exit /bye     leave the console
        /help                show /quit /exit /bye and friends
    """

    HELP = """
    Available commands:
      /quit      leave the console
      /exit      leave the console
      /bye       leave the console
      /status    show status
    Type /quit or /exit or /bye to leave.
    """

    BANNER = "type /quit, /exit or /bye to leave"


    def show_help():
        """Print the help text listing /quit, /exit and /bye."""
        # The user can type /quit, /exit or /bye here.
        print(HELP)


    def describe_exit_words():
        """Documentation helper mentioning /quit /exit /bye repeatedly."""
        return "The exit words are /quit, /exit and /bye."


    def handle(command):
        lowered = command.strip().lower()
        if lowered in {"/quit", "/exit", "/bye"}:
            return "exit"
        if lowered == "/help":
            show_help()
            return "handled"
        return "unknown"
'''


@pytest.fixture()
def decoy_repo(tmp_path):
    write(tmp_path, "cli.py", CLI_WITH_DECOY_HELP)
    return CodeIndex(tmp_path)


def test_the_branch_outranks_the_help_text_that_mentions_it_more_often(decoy_repo):
    """The decoy wins on every lexical measure.  It must still lose."""

    occurrences = decoy_repo.find_literal("/quit")

    assert len(occurrences) > 5, "the fixture must genuinely be ambiguous"

    best = occurrences[0]
    assert best.role is Role.CONTROL_FLOW
    assert "lowered in" in best.text, f"picked {best.text!r} instead of the branch"
    assert best.symbol == "handle"


def test_help_text_is_classified_as_text_not_code(decoy_repo):
    index = decoy_repo.index("cli.py")
    help_lines = [n for n, line in enumerate(index.lines, 1) if "/status" in line]

    assert help_lines, "fixture sanity"
    for line in help_lines:
        assert not index.role_at(line).executable, f"line {line} must not read as code"


def test_docstrings_and_comments_never_rank_as_executable(decoy_repo):
    """Prose is never code, even when it sits inside a function body."""

    index = decoy_repo.index("cli.py")
    prose = [
        n
        for n, line in enumerate(index.lines, 1)
        if line.lstrip().startswith("#") or "Print the help text" in line or "Documentation helper" in line
    ]

    assert prose, "fixture sanity"
    for line in prose:
        assert index.role_at(line) is Role.DOCUMENTATION, f"line {line}: {index.lines[line - 1]!r}"


def test_a_helper_that_returns_help_text_ranks_below_the_branch(decoy_repo):
    """`return "the exit words are /quit..."` really is executable -- just not the point.

    The classifier should not pretend otherwise; it should rank it correctly.
    """

    occurrences = decoy_repo.find_literal("/quit", executable_only=True)
    positions = {item.symbol: rank for rank, item in enumerate(occurrences)}

    assert positions["handle"] == 0
    assert positions.get("describe_exit_words", 99) > 0


def test_the_region_handed_to_the_model_contains_the_set_literal(decoy_repo):
    regions = decoy_repo.regions_for_terms(["/quit", "/bye"], budget_chars=2000)

    assert regions
    assert '{"/quit", "/exit", "/bye"}' in regions[0].text
    assert regions[0].symbol == "handle"


def test_the_region_is_a_whole_function_so_the_anchor_is_unambiguous(decoy_repo):
    regions = decoy_repo.regions_for_terms(["/quit"], budget_chars=2000)

    assert regions[0].text.startswith("def handle(")
    assert 'return "unknown"' in regions[0].text


# --------------------------------------------------------------------------
# Large file, real target far from the strongest lexical decoy
# --------------------------------------------------------------------------

@pytest.fixture()
def haystack_repo(tmp_path):
    """400 lines of decoy, the implementation at the very bottom."""

    filler = "\n".join(
        f'# note {n}: the /reload command reloads.  See /reload docs.\n'
        f'SAMPLE_{n} = "mentions /reload for documentation"'
        for n in range(120)
    )
    documented = "/reload is documented here.  " * 20
    source = (
        '"""Module docstring describing /reload at length.  /reload /reload."""\n\n'
        f'DOCS = """\n{documented}\n"""\n\n'
        f"{filler}\n\n\n"
        "def unrelated_but_plausible(command):\n"
        '    """Looks like a dispatcher and mentions /reload."""\n'
        "    return command\n\n\n"
        "def dispatch(command):\n"
        '    if command == "/reload":\n'
        "        return reload_everything()\n"
        "    return None\n"
    )
    (tmp_path / "big.py").write_text(source, encoding="utf-8")
    return CodeIndex(tmp_path)


def test_target_at_the_bottom_of_a_large_file_still_wins(haystack_repo):
    occurrences = haystack_repo.find_literal("/reload")

    assert len(occurrences) > 100, "fixture must be a genuine haystack"
    assert occurrences[0].symbol == "dispatch"
    assert occurrences[0].role is Role.CONTROL_FLOW


def test_large_file_regions_respect_the_budget(haystack_repo):
    regions = haystack_repo.regions_for_terms(["/reload"], budget_chars=1500)

    assert sum(region.char_count for region in regions) <= 1500
    assert "dispatch" in regions[0].text


# --------------------------------------------------------------------------
# Several plausible functions; only one is the dispatcher
# --------------------------------------------------------------------------

@pytest.fixture()
def many_candidates_repo(tmp_path):
    write(
        tmp_path,
        "app.py",
        '''
        def log_command(command):
            """Log a command such as /save."""
            print(f"got {command}")


        def validate_command(command):
            """Reject unknown commands like /save."""
            return isinstance(command, str)


        def format_command(command):
            """Format /save for display."""
            return command.upper()


        COMMAND_HELP = {"/save": "save the session", "/load": "load a session"}


        def execute(command):
            if command == "/save":
                return save_session()
            return None
        ''',
    )
    return CodeIndex(tmp_path)


def test_the_dispatcher_beats_three_plausible_neighbours(many_candidates_repo):
    best = many_candidates_repo.find_literal("/save")[0]

    assert best.symbol == "execute"
    assert best.role is Role.CONTROL_FLOW


def test_a_module_level_lookup_table_is_not_control_flow(many_candidates_repo):
    index = many_candidates_repo.index("app.py")
    table_line = next(n for n, line in enumerate(index.lines, 1) if "COMMAND_HELP" in line)

    assert index.role_at(table_line) is Role.CONSTANT_TEXT


# --------------------------------------------------------------------------
# Tests referencing the term must not outrank the implementation
# --------------------------------------------------------------------------

def test_implementation_outranks_the_test_that_exercises_it(tmp_path):
    write(tmp_path, "pkg/feature.py", '''
        def toggle(flag):
            if flag == "verbose":
                return True
            return False
    ''')
    write(tmp_path, "tests/test_feature.py", '''
        def test_verbose():
            assert toggle("verbose") is True
            assert toggle("verbose") is not None
            # verbose verbose verbose
    ''')
    index = CodeIndex(tmp_path)

    regions = index.regions_for_terms(["verbose"], budget_chars=4000)

    assert regions[0].path == "pkg/feature.py", f"got {regions[0].path}"


# --------------------------------------------------------------------------
# Structural queries
# --------------------------------------------------------------------------

def test_symbols_carry_their_line_spans(decoy_repo):
    symbols = {symbol.name: symbol for symbol in decoy_repo.index("cli.py").symbols}

    assert set(symbols) == {"show_help", "describe_exit_words", "handle"}
    assert symbols["handle"].start_line < symbols["handle"].end_line


def test_methods_are_qualified_by_their_class(tmp_path):
    write(tmp_path, "m.py", '''
        class Console:
            def handle(self, command):
                return command
    ''')
    index = CodeIndex(tmp_path)

    assert index.find_symbol("handle")[0].qualname == "Console.handle"
    assert index.find_symbol("handle")[0].kind == "method"


def test_references_exclude_the_definition_itself(tmp_path):
    write(tmp_path, "a.py", "def helper():\n    return 1\n")
    write(tmp_path, "b.py", "from a import helper\n\ndef use():\n    return helper()\n")
    index = CodeIndex(tmp_path)

    references = index.references("helper")

    assert all(item.path != "a.py" or item.line != 1 for item in references)
    assert any(item.path == "b.py" for item in references)


def test_import_edges_are_available_in_both_directions(tmp_path):
    write(tmp_path, "core.py", "VALUE = 1\n")
    write(tmp_path, "user.py", "import core\n")
    index = CodeIndex(tmp_path)

    assert "core" in index.imports_of("user.py")
    assert "user.py" in index.importers_of("core")


def test_a_file_that_does_not_parse_still_yields_comment_roles(tmp_path):
    write(tmp_path, "broken.py", "# a comment about /quit\ndef oops(:\n    pass\n")
    index = CodeIndex(tmp_path)

    file_index = index.index("broken.py")

    assert file_index.parse_error
    assert file_index.role_at(1) is Role.DOCUMENTATION


def test_indexing_ignores_virtualenvs_and_caches(tmp_path):
    write(tmp_path, "real.py", "x = 1\n")
    write(tmp_path, ".venv/lib/site.py", "y = 2\n")
    write(tmp_path, "__pycache__/cached.py", "z = 3\n")
    index = CodeIndex(tmp_path)

    found = {index.relative(path) for path in index.python_files()}

    assert found == {"real.py"}


def test_describe_literal_labels_code_and_text_distinctly(decoy_repo):
    report = decoy_repo.describe_literal("/quit")

    assert "[CODE/CONTROL_FLOW]" in report
    assert "[TEXT/" in report
    assert report.index("[CODE/") < report.index("[TEXT/"), "code must be listed first"


def test_a_term_that_is_absent_says_so_plainly(decoy_repo):
    assert "does not appear" in decoy_repo.describe_literal("/nonexistent")


def test_match_case_dispatch_counts_as_control_flow(tmp_path):
    write(tmp_path, "matcher.py", '''
        NOTES = "the /stop command is documented here"


        def route(command):
            match command:
                case "/stop":
                    return "stopping"
                case _:
                    return "unknown"
    ''')
    index = CodeIndex(tmp_path)

    best = index.find_literal("/stop")[0]

    assert best.role is Role.CONTROL_FLOW
    assert best.symbol == "route"
