"""Korrigieren: classification from the owner's words, scoped retrieval,
overrides that outrank the plan, and owner-only storage."""

from __future__ import annotations

from pathlib import Path

from service.corrections import (
    CorrectionStore, OwnerCorrection, apply_overrides, classify_correction, domain_of, guidance_lines, rule_for,
)


def test_classification_from_owner_vocabulary() -> None:
    assert classify_correction("Ich meinte eine Frage, keine Datei.")[0] == "INTENT_ERROR"
    assert classify_correction("Das war die falsche Datei.")[0:2] == ("ENTITY_RESOLUTION_ERROR", "ENTITY_SPECIFIC")
    assert classify_correction("Notizen gehören künftig immer in den Ordner notizen", request="leg eine notiz an")[0:2] == \
        ("OWNER_PREFERENCE", "DOMAIN_SPECIFIC")
    assert classify_correction("Nur diesmal bitte als markdown")[1] == "THIS_REQUEST"
    assert classify_correction("Das hat nicht funktioniert, Fehler beim Schreiben")[0:2] == ("EXECUTION_FAILURE", "THIS_REQUEST")
    assert classify_correction("Du hast behauptet es läuft, aber es lief gar nicht")[0] == "VERIFICATION_DEFECT"
    assert classify_correction("Spotify hat den falschen Track gespielt, die Fähigkeit ist kaputt")[0] in {"CAPABILITY_DEFECT", "EXECUTION_FAILURE"}
    assert classify_correction("Always answer in English from now on")[0:2] == ("OWNER_PREFERENCE", "GLOBAL_OWNER_PREFERENCE")


def test_rule_extracts_directory_override() -> None:
    when, then = rule_for("Notizen gehören künftig immer in den Ordner notizen", request="leg eine notiz an: Milch kaufen",
                          classification="OWNER_PREFERENCE", scope="DOMAIN_SPECIFIC")
    assert when == {"domain": "files"}
    assert then["overrides"] == {"directory": "notizen"}


def test_store_retrieval_respects_scope(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "c.jsonl")
    store.add(OwnerCorrection(original_request="leg eine notiz an", what_was_wrong="notes always in notizen/",
                              classification="OWNER_PREFERENCE", scope="DOMAIN_SPECIFIC",
                              when={"domain": "files"}, then={"note": "notes go to notizen/", "overrides": {"directory": "notizen"}}))
    store.add(OwnerCorrection(original_request="x", what_was_wrong="only once", classification="PARAMETER_ERROR",
                              scope="THIS_REQUEST", then={"note": "once"}))
    store.add(OwnerCorrection(original_request="spiel musik", what_was_wrong="always spotify", classification="OWNER_PREFERENCE",
                              scope="GLOBAL_OWNER_PREFERENCE", then={"note": "prefer spotify"}))
    store.add(OwnerCorrection(original_request="spiel Bohemian Rhapsody", what_was_wrong="wrong track",
                              classification="ENTITY_RESOLUTION_ERROR", scope="ENTITY_SPECIFIC",
                              when={"domain": "music", "terms": ["bohemian"]}, then={"note": "the Queen one"}))

    files = store.relevant("erstelle eine notiz: Milch kaufen")
    notes = [c.then["note"] for c in files]
    assert "notes go to notizen/" in notes and "prefer spotify" in notes and "once" not in notes
    assert "the Queen one" not in notes

    music = store.relevant("spiel bohemian rhapsody")
    assert [c.then["note"] for c in music][0] == "the Queen one", "narrowest scope first"
    assert domain_of("spiel bohemian rhapsody") == "music"

    args, applied = apply_overrides({"path": "milch.txt", "content": "x"}, files)
    assert args["path"] == "notizen/milch.txt" and applied
    assert guidance_lines(files)[0].startswith("- [")


def test_edit_and_delete(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "c.jsonl")
    row = store.add(OwnerCorrection(original_request="a", what_was_wrong="b", classification="OWNER_PREFERENCE",
                                    scope="GLOBAL_OWNER_PREFERENCE", then={"note": "b"}))
    assert store.update(row.correction_id, active=False).active is False
    assert store.relevant("anything") == []
    assert store.delete(row.correction_id) and store.get(row.correction_id) is None
