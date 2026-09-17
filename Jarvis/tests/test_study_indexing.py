"""Studium background indexing: persisted jobs, real progress, resume, fingerprints, caches, and the owner's files are safe."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from study.indexer import job_fraction
from study.service import StudyService
from study_fixtures import HEART_PAGES, make_pdf, make_pptx
from test_study_semantic import ConceptEmbeddings


def service(tmp_path, **kw) -> StudyService:
    events: list[tuple[str, dict]] = []
    svc = StudyService(tmp_path / "study", tmp_path / "store", embeddings=kw.pop("embeddings", False), slide_renderer=kw.pop("slide_renderer", False),
                       emit=lambda k, p: events.append((k, p)), **kw)
    svc.events = events  # type: ignore[attr-defined]
    return svc


def phases(svc, job_id):
    seen = []
    for kind, payload in svc.events:
        if kind == "study.progress" and payload["job_id"] == job_id and (not seen or seen[-1] != payload["phase"]):
            seen.append(payload["phase"])
    return seen


def long_pdf(pages: int = 40) -> bytes:
    return make_pdf([[f"Kapitel {n}", "Der Frank-Starling-Mechanismus verbindet Vorlast und Schlagvolumen." if n == 17 else f"Seite {n} über Herzphysiologie."]
                     for n in range(1, pages + 1)])


def test_an_import_is_a_persisted_job_with_real_phases_and_counters(tmp_path):
    svc = service(tmp_path, embeddings=ConceptEmbeddings())
    result = svc.import_bytes("Herz.pdf", long_pdf(40), wait=False)
    assert result["queued"] and svc.index.job(result["job_id"])["state"] == "queued"
    assert svc.indexer.run_pending() == 1
    job = svc.index.job(result["job_id"])
    assert job["state"] == "ready" and job["pages_total"] == 40 and job["pages_completed"] == 40
    assert job["chunks_completed"] == job["chunks_total"] > 0
    assert job["embedding_chunks_completed"] == job["embedding_chunks_total"] == job["chunks_total"]
    assert phases(svc, result["job_id"])[:1] == ["queued"] and {"parsing", "chunking", "indexing", "embedding", "ready"} <= set(phases(svc, result["job_id"]))
    fractions = [p["fraction"] for k, p in svc.events if k == "study.progress" and p["job_id"] == result["job_id"]]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0, "progress only moves forward and ends at 100 %"
    assert svc.search("Frank-Starling-Mechanismus")["results"][0]["best"]["location"]["page"] == 17


def test_indexing_continues_without_the_interface_and_the_summary_reports_it(tmp_path):
    """The owner leaves Studium: nothing in the interface drives the work; the worker does, and progress is read back from the index."""

    svc = service(tmp_path)
    svc.indexer.start()
    try:
        queued = [svc.import_bytes(f"Skript {n}.pdf", long_pdf(12 + n), wait=False) for n in range(3)]
        summary = svc.indexing()
        assert summary["documents_total"] == 3 and summary["active"] in {True, False}
        for _ in range(200):
            summary = svc.indexing()
            if not summary["active"]:
                break
            time.sleep(0.05)
        assert not summary["active"] and summary["documents_completed"] == 3 and summary["percent"] == 100
        assert all(svc.index.job(q["job_id"])["state"] == "ready" for q in queued)
    finally:
        svc.indexer.stopping = True
    # a fresh service on the same data (a reload, a restart) sees the finished jobs; later the batch collapses
    again = service(tmp_path)
    recent = again.indexer.summary(recent_seconds=3600)
    assert recent["documents_completed"] == 3 and not recent["active"]
    assert again.indexer.summary(recent_seconds=0)["documents_total"] == 0, "when everything is done the progress line has nothing to show"


def test_an_interrupted_job_resumes_after_a_restart_without_redoing_embeddings(tmp_path):
    provider = ConceptEmbeddings()
    svc = service(tmp_path, embeddings=provider)
    first = svc.import_bytes("Herz.pdf", long_pdf(30), wait=True)
    assert first["ok"]
    doc_id = first["document"]["id"]
    assert svc.embed_pending()["documents"] == 1
    calls = provider.calls
    # simulate a restart in the middle of a re-index: a running job in the index, no worker alive
    job = svc.indexer.new_job(kind="import", name="Herz.pdf", path=first["document"]["stored_path"], fingerprint=first["document"]["content_hash"])
    job.update(state="running", phase="embedding", provider="upload", document_id=doc_id)
    svc.index.save_job(job)
    restarted = service(tmp_path, embeddings=provider)
    for stale in restarted.index.jobs(states=("running",)):
        stale.update(state="queued")
        restarted.index.save_job(stale)
    assert restarted.indexer.run_pending() == 1
    resumed = restarted.index.job(job["job_id"])
    assert resumed["state"] == "ready" and resumed["embedding_chunks_completed"] == resumed["embedding_chunks_total"]
    assert provider.calls == calls, "every passage came from the vector cache; nothing was embedded twice"


def test_the_batch_keeps_finished_documents_while_others_run_so_progress_never_goes_back(tmp_path):
    """Seen live: finished jobs aged out after 20 s, the batch shrank and "35 %" fell back to "2 %"."""

    svc = service(tmp_path)
    first = svc.import_bytes("Eins.pdf", long_pdf(10), wait=False)
    second = svc.import_bytes("Zwei.pdf", long_pdf(11), wait=False)
    dup = svc.import_bytes("Eins Kopie.pdf", long_pdf(10), wait=False)
    svc.indexer.run(svc.index.job(first["job_id"]))
    # both were added ten minutes ago; Eins finished long ago while Zwei still waits
    for job_id, finished in ((first["job_id"], time.time() - 600), (second["job_id"], None)):
        job = svc.index.job(job_id)
        job["created_at"] = time.time() - 700
        if finished:
            job["finished_at"] = job["updated_at"] = finished
        svc.index.save_job(job)
    summary = svc.indexing()
    assert summary["active"] and summary["documents_total"] == 2 and summary["documents_completed"] == 1, summary
    assert summary["percent"] >= 50
    assert dup.get("already") or dup.get("duplicate") or svc.index.job(dup["job_id"])["state"] in {"queued", "skipped"}


def test_the_indexer_start_requeues_what_was_running(tmp_path):
    svc = service(tmp_path)
    job = svc.indexer.new_job(kind="import", name="x.md", path=str(tmp_path / "x.md"))
    job["state"] = "running"
    svc.index.save_job(job)
    fresh = service(tmp_path)
    fresh.indexer._loop = lambda: None  # do not work, only resume
    fresh.indexer.start()
    assert fresh.index.job(job["job_id"])["state"] == "queued" and fresh.index.job(job["job_id"])["attempts"] == 1


def test_fingerprints_skip_duplicates_and_identical_queued_work(tmp_path):
    svc = service(tmp_path)
    data = long_pdf(5)
    one = svc.import_bytes("A.pdf", data, wait=False)
    two = svc.import_bytes("A Kopie.pdf", data, wait=False)
    assert two.get("already") and two["job_id"] == one["job_id"], "the same bytes queued twice are one job"
    svc.indexer.run_pending()
    three = svc.import_bytes("A noch einmal.pdf", data)
    assert three["duplicate"] and svc.index.job(three["job_id"])["state"] == "skipped"
    assert sorted(p.name for p in (tmp_path / "store").iterdir()) == ["A.pdf"]


def test_a_changed_file_is_reindexed_and_only_changed_passages_are_embedded(tmp_path):
    provider = ConceptEmbeddings()
    folder = tmp_path / "Physiologie"
    folder.mkdir()
    target = folder / "Herz.pdf"
    target.write_bytes(make_pdf([["Vorlast", "Frank-Starling-Mechanismus"], ["Nachlast", "Afterload"]]))
    svc = service(tmp_path, embeddings=provider)
    source = svc.connect_folder(folder)["source"]
    assert svc.scan_source(source["id"])["indexed"] == 1
    svc.embed_pending()
    assert svc.scan_source(source["id"])["unchanged"] == 1, "an unchanged file is not parsed again"
    calls = provider.calls
    time.sleep(1.1)
    target.write_bytes(make_pdf([["Vorlast", "Frank-Starling-Mechanismus"], ["Nachlast", "Nachlast erhöht den Druck"]]))
    report = svc.scan_source(source["id"])
    assert report["indexed"] == 1
    before = provider.calls
    svc.embed_pending()
    assert provider.calls - before == 1, "only the changed page's passage is embedded; the unchanged one comes from the cache"
    assert calls <= before


def test_an_embedding_set_larger_than_one_batch_is_marked_complete(tmp_path):
    """The live bug: a 12-chunk document never counted as embedded because completeness was checked per batch."""

    provider = ConceptEmbeddings()
    svc = service(tmp_path, embeddings=provider)
    doc = svc.import_bytes("Lang.pdf", long_pdf(12))["document"]
    assert len(svc.index.chunks_of(doc["id"])) > 8
    svc.embed_pending()
    assert svc.index.unembedded(provider.id) == [] and svc.index.embedding_stats(provider.id)["embedded_documents"] == 1


def test_job_fraction_is_computed_from_work_not_time():
    base = {"state": "running", "phase": "ocr", "pages_total": 100, "pages_completed": 100, "ocr_pages_total": 100, "ocr_pages_completed": 25,
            "chunks_total": 0, "chunks_completed": 0, "embedding_chunks_total": 0, "embedding_chunks_completed": 0, "slides_total": 0, "slides_completed": 0}
    early = job_fraction(base)
    later = job_fraction({**base, "ocr_pages_completed": 75})
    assert 0 < early < later < 1
    assert job_fraction({**base, "state": "ready"}) == 1.0
    assert job_fraction({**base, "phase": "queued", "pages_completed": 0, "ocr_pages_completed": 0}) < early


def test_failures_are_reported_calmly_and_leave_no_stray_upload(tmp_path):
    svc = service(tmp_path)
    assert svc.import_bytes("Tabelle.xlsx", b"PK...")["reason"] == "unsupported"
    assert svc.import_path(tmp_path / "gibt-es-nicht.pdf")["reason"] == "missing"
    broken = svc.import_bytes("Kaputt.pdf", b"%PDF-1.4 broken", wait=False)
    svc.indexer.run_pending()
    job = svc.index.job(broken["job_id"])
    assert job["state"] == "failed" and job["reason"] == "parse_failed" and "gelesen" in job["error"]
    assert not (tmp_path / "store" / "Kaputt.pdf").exists(), "an upload Studium could not read is removed again"
    native = svc.import_bytes("Herz.goodnotes", b"PK\x03\x04 native notebook")
    assert native["reason"] == "goodnotes_native" and "Exportiere sie als PDF" in native["error"]
    assert svc.indexing()["failed"], "recent failures stay visible for the interface"


def test_the_same_refused_file_is_one_failure_not_one_per_attempt(tmp_path):
    svc = service(tmp_path)
    native = b"PK" + bytes(60)
    first = svc.import_bytes("Notizbuch.goodnotes", native, wait=False)
    again = svc.import_bytes("Notizbuch.goodnotes", native, wait=False)
    assert first["reason"] == again["reason"] == "goodnotes_native" and first["job_id"] == again["job_id"]
    assert len([j for j in svc.index.jobs(states=("failed",)) if j["name"] == "Notizbuch.goodnotes"]) == 1


def test_an_encrypted_pdf_says_so(tmp_path):
    from pypdf import PdfReader, PdfWriter

    plain = tmp_path / "plain.pdf"
    plain.write_bytes(make_pdf(HEART_PAGES))
    writer = PdfWriter(clone_from=PdfReader(str(plain)))
    writer.encrypt("geheim")
    locked = tmp_path / "Geschützt.pdf"
    with locked.open("wb") as handle:
        writer.write(handle)
    svc = service(tmp_path)
    result = svc.import_path(locked)
    assert result["ok"] is False and result["reason"] == "encrypted" and "Passwort" in result["error"]


def test_removing_from_studium_never_deletes_the_owners_file(tmp_path):
    folder = tmp_path / "GoodNotes"
    folder.mkdir()
    own = folder / "Herz.pdf"
    own.write_bytes(make_pdf(HEART_PAGES))
    svc = service(tmp_path)
    source = svc.connect_folder(folder, provider="goodnotes")["source"]
    svc.scan_source(source["id"])
    doc = svc.documents()[0]
    assert svc.delete(doc["id"])["file_kept"] and own.exists()
    svc.scan_source(source["id"])
    doc = svc.documents()[0]
    refused = svc.delete(doc["id"], delete_file=True)
    assert own.exists() and refused["file_note"], "a file outside ZEUS's Studium folder is never deleted, even when asked"
    upload = svc.import_bytes("Notiz.md", "# Niere\n\nRenin.".encode("utf-8"))
    stored = Path(upload["document"]["stored_path"])
    assert svc.delete(upload["document"]["id"])["file_kept"] and stored.exists()


def test_a_moved_file_is_found_again_and_a_vanished_one_is_marked_missing(tmp_path):
    folder = tmp_path / "Kurs"
    (folder / "alt").mkdir(parents=True)
    first = folder / "alt" / "Herz.pdf"
    first.write_bytes(make_pdf(HEART_PAGES))
    other = folder / "alt" / "Niere.pdf"
    other.write_bytes(make_pdf([["Niere", "Aldosteron"]]))
    svc = service(tmp_path)
    source = svc.connect_folder(folder)["source"]
    svc.scan_source(source["id"])
    (folder / "neu").mkdir()
    first.rename(folder / "neu" / "Herz.pdf")
    other.unlink()
    report = svc.scan_source(source["id"])
    assert report["moved"] == 1 and report["missing"] == 1
    names = {d["filename"]: d for d in svc.documents()}
    assert names["Herz.pdf"]["stored_path"].endswith(str(Path("neu") / "Herz.pdf"))
    assert names["Niere.pdf"]["flags"].get("missing") is True, "the index keeps the entry and says the file is gone"


def test_slides_render_only_in_the_background_worker(tmp_path):
    class Counting:
        calls = 0

        def available(self):
            return True

        def render_slides(self, path, out, *, width=1600, timeout=120.0):
            Counting.calls += 1
            return {"ok": False, "error": "test"}

    pptx = tmp_path / "Neuro.pptx"
    make_pptx(pptx, [("Motorik", ["Kortex"], "")])
    svc = service(tmp_path, slide_renderer=Counting())
    assert svc.import_path(pptx)["ok"]
    assert Counting.calls == 0, "an inline import (tests, chat attachments) never starts PowerPoint"
