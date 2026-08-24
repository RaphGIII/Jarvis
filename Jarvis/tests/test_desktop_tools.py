"""Tools that reach outside the workspace.

Everything in tools.builtin is bounded by a project directory, which is what
makes it safe to hand to a small model. These are not, so the tests here are
mostly about blast radius: what a wrong guess can and cannot do, and whether a
side effect can be inspected before it happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.desktop import (
    AUDIO_SUFFIXES,
    clipboard_write,
    desktop_tools,
    find_applications,
    find_media,
    launch_application,
    media_control,
    media_folders,
    notify,
    open_path,
    open_url,
    running_processes,
    screenshot,
)
from tools.registry import RiskLevel, ToolContext


@pytest.fixture()
def context(tmp_path):
    return ToolContext(workspace=tmp_path)


# --------------------------------------------------------------------------
# Risk is declared honestly
# --------------------------------------------------------------------------

def test_every_desktop_tool_declares_a_risk_and_an_example():
    for spec in desktop_tools():
        assert spec.example, f"{spec.name} has no example"
        assert spec.purpose.strip(), f"{spec.name} has no purpose"


def test_reading_the_screen_or_launching_things_is_high_risk():
    risks = {spec.name: spec.risk for spec in desktop_tools()}

    assert risks["launch_application"] is RiskLevel.HIGH
    assert risks["open_path"] is RiskLevel.HIGH
    assert risks["screenshot"] is RiskLevel.HIGH


def test_writing_the_clipboard_is_high_risk():
    """The clipboard routinely holds a password the user copied a moment ago."""

    risks = {spec.name: spec.risk for spec in desktop_tools()}

    assert risks["clipboard_write"] is RiskLevel.HIGH


def test_looking_around_is_safe():
    risks = {spec.name: spec.risk for spec in desktop_tools()}

    for name in ("running_processes", "find_applications", "media_folders", "find_media"):
        assert risks[name] is RiskLevel.SAFE, name


def test_opening_a_url_is_not_merely_low_risk():
    """It can reach the network and start a program."""

    risks = {spec.name: spec.risk for spec in desktop_tools()}

    assert risks["open_url"] >= RiskLevel.MODERATE


# --------------------------------------------------------------------------
# Every side effect can be rehearsed
# --------------------------------------------------------------------------

def test_launching_supports_a_dry_run(context):
    program = "notepad" if sys.platform == "win32" else "ls"

    result = launch_application({"program": program, "dry_run": True}, context)

    assert result["ok"] and result["dry_run"]
    assert "would_run" in result


def test_opening_a_path_supports_a_dry_run(tmp_path, context):
    target = tmp_path / "song.mp3"
    target.write_bytes(b"not really audio")

    result = open_path({"path": str(target), "dry_run": True}, context)

    assert result["ok"] and result["would_open"] == str(target)


def test_media_keys_support_a_dry_run(context):
    result = media_control({"action": "playpause", "dry_run": True}, context)

    assert result["ok"] and result["would_send"] == "playpause"


def test_notifications_support_a_dry_run(context):
    result = notify({"message": "hello", "dry_run": True}, context)

    assert result["ok"] and "hello" in result["would_notify"]


def test_screenshots_support_a_dry_run(context):
    result = screenshot({"path": "shot.png", "dry_run": True}, context)

    assert result["ok"] and result["would_write"].endswith("shot.png")


def test_clipboard_supports_a_dry_run(context):
    result = clipboard_write({"text": "secret", "dry_run": True}, context)

    assert result["ok"] and result["would_write"] == 6
    assert "secret" not in str(result), "a dry run must not echo the payload back"


# --------------------------------------------------------------------------
# What a wrong guess cannot do
# --------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".msi", ".ps1", ".vbs", ".reg", ".scr"])
def test_dangerous_file_types_are_refused(tmp_path, context, suffix):
    target = tmp_path / f"installer{suffix}"
    target.write_text("x", encoding="utf-8")

    result = open_path({"path": str(target)}, context)

    assert not result["ok"]
    assert "refusing" in result["error"]


@pytest.mark.parametrize("url", ["file:///C:/Windows/System32", "javascript:alert(1)", "ftp://host/x"])
def test_only_http_urls_may_be_opened(context, url):
    """file:// would turn "open a link" into "read anything on the disk"."""

    result = open_url({"url": url}, context)

    assert not result["ok"]
    assert "http" in result["error"]


def test_a_missing_program_is_reported_not_launched(context):
    result = launch_application({"program": "definitely-not-installed-xyzzy"}, context)

    assert not result["ok"]
    assert "not found" in result["error"]


def test_a_missing_path_is_reported(context):
    result = open_path({"path": "/no/such/file/anywhere"}, context)

    assert not result["ok"]


def test_a_screenshot_cannot_escape_the_workspace(context):
    result = screenshot({"path": "../../escaped.png"}, context)

    assert not result["ok"]
    assert "workspace" in result["error"]


def test_an_unknown_media_action_lists_what_is_supported(context):
    result = media_control({"action": "teleport"}, context)

    assert not result["ok"]
    assert "playpause" in result["supported"]


def test_an_empty_path_is_rejected(context):
    assert not open_path({"path": "  "}, context)["ok"]


# --------------------------------------------------------------------------
# Discovery actually works on this machine
# --------------------------------------------------------------------------

def test_processes_can_be_listed(context):
    result = running_processes({"limit": 5}, context)

    assert result["ok"]
    assert result["processes"], "no processes at all is not plausible"


def test_processes_can_be_filtered(context):
    result = running_processes({"contains": "python", "limit": 10}, context)

    assert result["ok"]
    assert all("python" in name.lower() for name in result["processes"])


def test_an_installed_program_is_found_and_a_fictional_one_is_not(context):
    known = "notepad" if sys.platform == "win32" else "ls"

    result = find_applications({"names": [known, "totally-fictional-program-xyzzy"]}, context)

    assert known in result["found"], f"{known} should exist on this machine"
    assert "totally-fictional-program-xyzzy" in result["missing"]


def test_media_folders_are_reported_with_counts(context):
    result = media_folders({}, context)

    assert result["ok"]
    for folder in result["folders"]:
        assert "path" in folder and "audio_files" in folder


def test_media_search_finds_audio_and_ignores_everything_else(tmp_path, context):
    (tmp_path / "song.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "clip.flac").write_bytes(b"x")

    result = find_media({"paths": [str(tmp_path)]}, context)

    names = {item["name"] for item in result["matches"]}
    assert names == {"song", "clip"}


def test_media_search_can_filter_by_query(tmp_path, context):
    (tmp_path / "bach-prelude.mp3").write_bytes(b"x")
    (tmp_path / "unrelated.mp3").write_bytes(b"x")

    result = find_media({"paths": [str(tmp_path)], "query": "bach"}, context)

    assert [item["name"] for item in result["matches"]] == ["bach-prelude"]


def test_media_search_respects_a_limit(tmp_path, context):
    for index in range(30):
        (tmp_path / f"track{index}.mp3").write_bytes(b"x")

    result = find_media({"paths": [str(tmp_path)], "limit": 5}, context)

    assert len(result["matches"]) == 5
    assert result["truncated"]


def test_the_audio_suffix_list_covers_the_common_formats():
    for suffix in (".mp3", ".flac", ".wav", ".m4a", ".ogg"):
        assert suffix in AUDIO_SUFFIXES


def test_a_nonexistent_search_root_is_skipped_not_fatal(context):
    result = find_media({"paths": ["/no/such/directory"]}, context)

    assert result["ok"] and result["matches"] == []


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_the_kernel_registers_the_desktop_pack():
    from core.kernel import KernelConfig

    assert KernelConfig().enable_desktop_tools is True


def test_the_pack_can_be_turned_off():
    from core.kernel import KernelConfig

    assert KernelConfig(enable_desktop_tools=False).enable_desktop_tools is False
