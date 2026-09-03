"""Filesystem questions must route to filesystem tools — never screen.capture."""

from __future__ import annotations

from service.intents import TopIntent, parse_fs_operation, understand
from service.system_context import SystemContext


def op(text):
    a = parse_fs_operation(text)
    return (a.operation, a.arguments) if a else (None, {})


def test_largest_folder_is_fs_not_screen_capture():
    o, args = op("Zeig mir den größten Ordner auf D:")
    assert o == "fs.largest" and args["drive"] == "D:\\"
    o2, _ = op("Welcher Ordner frisst auf D: am meisten Platz?")
    assert o2 == "fs.largest"


def test_count_resolves_a_bare_name():
    o, args = op("wieviele unterordner hat jarvis")
    assert o == "fs.count" and args["name"].lower() == "jarvis"


def test_self_reference_repo_counts_and_lists():
    o, args = op("Wie viele direkte Unterordner hat dein eigenes Repo?")
    assert o == "fs.count" and args["self_ref"]
    o2, args2 = op("Was ist direkt in deinem Repo drin?")
    assert o2 == "fs.list" and args2["self_ref"]


def test_open_repo_by_self_reference():
    o, args = op("Bring mich zu deinem Quellcode.")
    assert o == "fs.open" and args["self_ref"]
    o2, args2 = op("Öffne dein eigenes Repo.")
    assert o2 == "fs.open" and args2["self_ref"]


def test_the_understanding_routes_fs_to_system_control():
    u = understand("Zeig mir den größten Ordner auf D:")
    assert u.top is TopIntent.SYSTEM_CONTROL and u.action.operation == "fs.largest"


def test_a_screenshot_request_is_still_a_screenshot():
    # the fs changes must not swallow real screenshot intents
    u = understand("Mach einen Screenshot.")
    assert u.action is None or u.action.operation != "fs.largest"


def test_system_context_resolves_real_paths():
    sc = SystemContext()
    key, path = sc.resolve_self_reference("was ist in deinem repo")
    assert key == "repository_root" and path
    key2, path2 = sc.resolve_self_reference("zeig mir deine modelle")
    assert key2 == "model_root"
    assert sc.resolve_self_reference("wie spät ist es") == ("", "")
