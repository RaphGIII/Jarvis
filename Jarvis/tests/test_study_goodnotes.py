"""GoodNotes as an honest source adapter: exports are recognised, native notebooks are inspected but never faked."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from study import goodnotes
from study.goodnotes import NATIVE_MESSAGE, GoodNotesSourceAdapter
from study.parsers import source_type_for
from study_fixtures import make_pdf


@pytest.fixture()
def adapter():
    return GoodNotesSourceAdapter()


# --------------------------------------------------------------------------- PDF exports

def test_a_pdf_export_is_recognised_by_its_producer(tmp_path, adapter):
    for producer in ("GoodNotes 5", "Goodnotes 6", "GOODNOTES iOS PDF"):
        path = tmp_path / f"{producer}.pdf"
        path.write_bytes(make_pdf([["Biochemie Mitschrift"]], producer=producer))
        found = adapter.detect(path)
        assert found and found["kind"] == "pdf" and found["origin"] == "goodnotes_export"
        assert "metadata" in found["evidence"] and found["confidence"] >= 0.9
    other = tmp_path / "skript.pdf"
    other.write_bytes(make_pdf([["Skript"]], producer="Microsoft Word"))
    assert adapter.detect(other) is None
    assert adapter.detect(tmp_path / "x.pdf", metadata={"creator": "Goodnotes 6.3"})["kind"] == "pdf"


def test_a_pdf_in_a_goodnotes_folder_is_an_export(tmp_path, adapter):
    folder = tmp_path / "OneDrive" / "GoodNotes Backup" / "Studium"
    folder.mkdir(parents=True)
    path = folder / "Physiologie.pdf"
    path.write_bytes(make_pdf([["Herz"]], producer="Quartz PDFContext"))
    found = adapter.detect(path)
    assert found["evidence"] == ["folder"] and found["notebook"] == "Physiologie" and found["page"] is None
    assert 0.5 < found["confidence"] < 0.9


# ------------------------------------------------------------------------- image exports

def test_image_exports_are_recognised_by_metadata_or_folder(tmp_path, adapter):
    Image = pytest.importorskip("PIL.Image")
    from PIL import PngImagePlugin

    png = tmp_path / "Seite.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "GoodNotes 5")
    Image.new("RGB", (8, 8), "white").save(png, pnginfo=info)
    found = adapter.detect(png)
    assert found["kind"] == "image" and found["evidence"] == ["metadata"]

    jpeg = tmp_path / "Biochemie 3.jpg"
    exif = Image.Exif()
    exif[305] = "Goodnotes 6"
    Image.new("RGB", (8, 8), "white").save(jpeg, exif=exif)
    found = adapter.detect(jpeg)
    assert found["kind"] == "image" and found["page"] == 3 and found["notebook"] == "Biochemie"

    plain = tmp_path / "Foto.png"
    Image.new("RGB", (8, 8), "white").save(plain)
    assert adapter.detect(plain) is None

    folder = tmp_path / "Goodnotes" / "Anatomie"
    folder.mkdir(parents=True)
    webp = folder / "Anatomie 12.webp"
    Image.new("RGB", (8, 8), "white").save(webp)
    found = adapter.detect(webp)
    assert found["evidence"] == ["folder"] and found["notebook"] == "Anatomie" and found["page"] == 12

    # a year is not a page
    assert adapter.detect(tmp_path / "Klausur 2024.png", metadata={"Software": "GoodNotes"})["page"] is None


def test_images_map_to_the_image_parser_but_native_notebooks_do_not():
    for name in ("a.png", "a.jpg", "a.jpeg", "a.webp", "A.WEBP"):
        assert source_type_for(name) == "image"
    assert source_type_for("Notizbuch.goodnotes") == ""


# ------------------------------------------------------------------------------- native

def _native(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_a_native_notebook_is_inspected_and_honestly_unsupported(tmp_path, adapter):
    path = _native(tmp_path / "Biochemie.goodnotes", {
        "index.notes.pb": b"\x0a\x03abc",
        "schema.pb": b"\x08\x01",
        "notes/6F1C-1": b"\x12\x34 ink strokes",
        "notes/6F1C-2": b"\x12\x34 more strokes",
        "attachments/AB12": make_pdf([["Vorlesungsfolie"]]),
        "attachments/CD34": b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
        "thumbnails/thumbnail.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 32,
        "search/6F1C-1": b"recognition data",
    })
    assert adapter.detect(path) == {"origin": "goodnotes_export", "kind": "native", "notebook": "Biochemie", "page": None,
                                    "confidence": 1.0, "evidence": ["suffix"]}
    report = adapter.inspect_native(path)
    assert report["supported"] is False and report["message"] == NATIVE_MESSAGE
    assert report["message"] == ("Diese GoodNotes-Datei kann ZEUS derzeit nicht direkt lesen. "
                                 "Exportiere sie als PDF oder verbinde deinen GoodNotes-Backup-Ordner.")
    assert report["container"] == "zip" and report["members"] == 8
    groups = report["groups"]
    assert groups["notes"]["count"] == 2 and groups["attachments"]["count"] == 2
    assert groups["thumbnails"]["count"] == 1 and groups["search"]["count"] == 1 and groups["index"]["count"] == 2
    assert sorted((a["member"], a["type"]) for a in report["readable_attachments"]) == [("attachments/AB12", "pdf"), ("attachments/CD34", "png")]
    assert [t["type"] for t in report["thumbnails"]] == ["jpeg"]
    assert report["handwriting_readable"] is False and report["help"]["export_pdf"]
    # nothing was extracted next to the source
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Biochemie.goodnotes"]

    extracted = adapter.extract_attachment(path, "attachments/AB12")
    try:
        assert extracted is not None and extracted.read_bytes().startswith(b"%PDF-")
        assert tmp_path not in extracted.parents
    finally:
        if extracted is not None:
            import shutil

            shutil.rmtree(extracted.parent, ignore_errors=True)
    assert adapter.extract_attachment(path, "notes/6F1C-1") is None
    assert adapter.extract_attachment(path, "../../evil") is None


def test_a_native_file_that_is_not_a_zip_is_unsupported_with_the_same_calm_message(tmp_path, adapter):
    path = tmp_path / "Alt.goodnotes"
    path.write_bytes(b"\x00\x01binary goodnotes data")
    report = adapter.inspect_native(path)
    assert report["supported"] is False and report["container"] == "unknown" and report["message"] == NATIVE_MESSAGE
    missing = adapter.inspect_native(tmp_path / "fehlt.goodnotes")
    assert missing["supported"] is False and missing["container"] == "missing"


def test_the_zip_bomb_guard_stops_before_reading(tmp_path, adapter):
    many = _native(tmp_path / "Viele.goodnotes", {f"notes/{i}": b"x" for i in range(50)})
    report = adapter.inspect_native(many, max_members=20)
    assert report["limits_exceeded"] == "members" and report["supported"] is False and report["readable_attachments"] == []

    big = _native(tmp_path / "Gross.goodnotes", {"attachments/a": b"\x00" * 5_000_000})
    report = adapter.inspect_native(big, max_uncompressed_bytes=1_000_000)
    assert report["limits_exceeded"] == "uncompressed_size" and report["groups"] == {}

    # a highly compressed member is flagged, not opened
    report = adapter.inspect_native(big)
    assert report["limits_exceeded"] is None and report["suspicious_members"] == ["attachments/a"]

    report = adapter.inspect_native(big, max_file_bytes=100)
    assert report["limits_exceeded"] == "file_size"


# ----------------------------------------------------------------------------- discovery

def _fake_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial", "LOCALAPPDATA", "APPDATA"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    return home


def test_discovery_finds_goodnotes_folders_under_the_cloud_roots(tmp_path, monkeypatch, adapter):
    home = _fake_home(tmp_path, monkeypatch)
    business = tmp_path / "OneDrive - Uni Wien"
    monkeypatch.setenv("OneDriveCommercial", str(business))
    backup = home / "OneDrive" / "Apps" / "GoodNotes 5 Backup"
    backup.mkdir(parents=True)
    make = lambda p, data=b"x": (p.parent.mkdir(parents=True, exist_ok=True), p.write_bytes(data))  # noqa: E731
    make(backup / "Biochemie.pdf")
    make(backup / "Studium" / "Anatomie.pdf")
    make(backup / "Seite 1.png")
    make(backup / "Alt.goodnotes")
    make(business / "Goodnotes" / "Physik.pdf")
    icloud = home / "iCloudDrive" / "iCloud~com~goodnotesapp~goodnotes"
    make(icloud / "Chemie.goodnotes")
    dropbox_root = tmp_path / "Dropbox (Persönlich)"
    make(dropbox_root / "Notizen" / "goodnotes-export" / "Mathe.pdf")
    local = tmp_path / "local"
    make(local / "Dropbox" / "info.json", ('{"personal": {"path": "' + str(dropbox_root).replace("\\", "\\\\") + '"}}').encode("utf-8"))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    too_deep = home / "OneDrive" / "a" / "b" / "c" / "d" / "e" / "GoodNotes"
    make(too_deep / "tief.pdf")

    found = adapter.discover_folders(include_drives=False)
    by_name = {Path(f["path"]).name: f for f in found}
    assert set(by_name) == {"GoodNotes 5 Backup", "Goodnotes", "iCloud~com~goodnotesapp~goodnotes", "goodnotes-export"}
    main = by_name["GoodNotes 5 Backup"]
    assert (main["pdfs"], main["images"], main["native_files"], main["provider"]) == (2, 1, 1, "onedrive")
    assert main["last_modified"]
    assert by_name["iCloud~com~goodnotesapp~goodnotes"]["provider"] == "icloud"
    assert by_name["goodnotes-export"]["provider"] == "dropbox"
    assert by_name["Goodnotes"]["provider"] == "onedrive"


def test_discovery_never_follows_a_junction_loop(tmp_path, monkeypatch, adapter):
    home = _fake_home(tmp_path, monkeypatch)
    root = home / "Dropbox"
    (root / "GoodNotes").mkdir(parents=True)
    (root / "GoodNotes" / "a.pdf").write_bytes(b"x")
    loop = root / "GoodNotes" / "loop"
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(root), str(loop))
        else:
            os.symlink(root, loop, target_is_directory=True)
    except (OSError, AttributeError):
        pytest.skip("cannot create a junction or symlink here")
    found = adapter.discover_folders(include_drives=False)
    assert [(Path(f["path"]).name, f["pdfs"]) for f in found] == [("GoodNotes", 1)]


def test_the_old_module_functions_still_work(tmp_path, monkeypatch):
    home = _fake_home(tmp_path, monkeypatch)
    folder = home / "Dropbox" / "GoodNotes"
    folder.mkdir(parents=True)
    (folder / "Mathe.pdf").write_bytes(b"x")
    (folder / "Mathe.goodnotes").write_bytes(b"x")
    monkeypatch.setattr(goodnotes.ADAPTER, "cloud_roots", lambda include_drives=True: [{"path": home / "Dropbox", "provider": "dropbox"}])

    assert goodnotes.is_goodnotes_pdf({"producer": "GoodNotes 5"}) and not goodnotes.is_goodnotes_pdf({})
    assert goodnotes.notebook_name("x/Mathe.pdf") == "Mathe"
    assert goodnotes.NATIVE_SUFFIX == ".goodnotes"
    assert goodnotes.cloud_roots() == [home / "Dropbox"]
    (entry,) = goodnotes.discover()
    assert entry["pdfs"] == 1 and entry["native_files"] == 1
    assert goodnotes.native_files(folder) == [str(folder / "Mathe.goodnotes")]
    status = goodnotes.status()
    assert status["direct_api"] is False and status["discovered"][0]["path"] == str(folder)
    assert goodnotes.detect(folder / "Mathe.goodnotes")["kind"] == "native"
    assert goodnotes.inspect_native(folder / "Mathe.goodnotes")["supported"] is False
    help_text = goodnotes.export_help()
    assert "PDF" in help_text["summary"] and any("Automatisches Backup" in step for step in help_text["auto_backup"])
