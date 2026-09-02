"""Owner speech corpus: storage and the metrics every STT decision rests on."""

from __future__ import annotations

from speech.corpus import PHRASES, SpeechCorpus, cer, entity_accuracy, wer


def test_wer_zero_on_identical_and_case_insensitive():
    assert wer("Öffne Wikipedia.", "öffne wikipedia") == 0.0


def test_wer_counts_substitutions():
    assert wer("öffne bitte wikipedia", "öffne bitte wikimedia") == 1 / 3


def test_cer_is_finer_than_wer():
    truth, hyp = "Stockfish", "Stockfisch"
    assert wer(truth, hyp) == 1.0
    assert cer(truth, hyp) < 0.2


def test_entity_accuracy_only_counts_present_entities():
    assert entity_accuracy("Öffne Spotify und Wikipedia", "öffne spotify und wikipedia") == 1.0
    assert entity_accuracy("Öffne Spotify und Wikipedia", "öffne sportify und wikipedia") == 0.5
    assert entity_accuracy("Guten Morgen", "guten morgen") is None


def test_corpus_add_list_delete(tmp_path):
    corpus = SpeechCorpus(tmp_path)
    entry = corpus.add(b"RIFFxxxx", ext="wav", ground_truth="Öffne Wikipedia.", category="befehl")
    assert (tmp_path / "recordings").is_dir()
    entries = corpus.list()
    assert len(entries) == 1 and entries[0]["ground_truth"] == "Öffne Wikipedia."
    assert corpus.stats()["count"] == 1
    assert corpus.delete(entry["id"]) is True
    assert corpus.list() == []


def test_corpus_refuses_empty_truth(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        SpeechCorpus(tmp_path).add(b"x", ext="wav", ground_truth="  ")


def test_phrase_script_covers_the_required_categories():
    cats = {c for c, _ in PHRASES}
    for needed in {"normal", "befehl", "technik", "projekt", "medizin", "gemischt", "zahlen", "url", "app"}:
        assert needed in cats
    assert len(PHRASES) >= 30
