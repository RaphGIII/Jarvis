"""The deterministic basics: library files are real and fenced, the new
system-control intents parse the owner's actual phrasings, Bing redirect
URLs decode, and the app index folds names the way people say them."""

from __future__ import annotations

from pathlib import Path

from service.apps import _fold
from service.intents import parse_system_control
from service.library import Library
from service.websearch import _real_url


# -- library: real files under one fenced root ------------------------------

def test_library_folder_and_note_are_real_files(tmp_path):
    lib = Library(tmp_path / "shelf")
    made = lib.create_folder("Studium/Anatomie")
    assert made["ok"] and Path(made["path"]).is_dir()
    note = lib.write_note("Studium/Anatomie", "Herz", "Vier Kammern.")
    assert note["ok"] and note["read_back"]
    on_disk = Path(note["path"]).read_text(encoding="utf-8")
    assert "Vier Kammern." in on_disk and on_disk.startswith("# Herz")


def test_library_never_escapes_its_root(tmp_path):
    lib = Library(tmp_path / "shelf")
    lib.create_folder("ok")
    assert lib.move("../outside", "ok")["ok"] is False
    assert lib.read_note("../../etc/passwd")["ok"] is False
    # a crafted folder name is sanitised INTO the root, never above it
    made = lib.create_folder("../evil")
    assert made["ok"] is False or Path(made["path"]).resolve().is_relative_to((tmp_path / "shelf").resolve())


def test_library_import_copies_the_file(tmp_path):
    src = tmp_path / "doc.txt"
    src.write_text("inhalt", encoding="utf-8")
    lib = Library(tmp_path / "shelf")
    out = lib.import_file(str(src))
    assert out["ok"] and Path(out["path"]).read_text(encoding="utf-8") == "inhalt"
    assert src.exists(), "import copies, never moves the owner's file"


# -- the deterministic basics parse without a model -------------------------

def test_time_and_date_questions_become_typed_intents():
    assert parse_system_control("Wie spät ist es?").operation == "system.tell_time"
    assert parse_system_control("Zeus, wie viel Uhr haben wir?").operation == "system.tell_time"
    assert parse_system_control("Welches Datum haben wir heute?").operation == "system.tell_date"


def test_open_app_is_deterministic_and_views_stay_views():
    intent = parse_system_control("Öffne Spotify")
    assert intent.operation == "app.open" and intent.target == "Spotify"
    assert parse_system_control("Öffne die Datei bericht.pdf") is None, "file-ish opens stay with the file paths"


def test_open_url_and_web_search_parse():
    url = parse_system_control("Öffne youtube.com")
    assert url.operation == "web.open" and url.target == "https://youtube.com"
    search = parse_system_control("Such im Internet nach aktuellen GPU Preisen")
    assert search.operation == "web.search" and "GPU Preisen" in search.target
    assert parse_system_control("Google mal nach Piper TTS").operation == "web.search"


# -- pure helpers -----------------------------------------------------------

def test_bing_redirect_urls_decode_to_their_target():
    packed = "https://www.bing.com/ck/a?!&&p=x&u=a1aHR0cHM6Ly9kZS53aWtpcGVkaWEub3Jn&ntb=1"
    assert _real_url(packed) == "https://de.wikipedia.org"
    assert _real_url("https://example.org/page") == "https://example.org/page"


def test_app_name_folding_matches_spoken_german():
    assert _fold("Blender 4.3") == "blender 4.3"
    assert _fold("RECHNER") == "rechner"
    assert _fold("Städte-Übersicht") == "staedte-uebersicht"
