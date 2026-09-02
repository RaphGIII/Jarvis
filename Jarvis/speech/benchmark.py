"""STT benchmark on the OWNER's corpus: measured on this machine, this voice.

Run inside the speech venv (it needs faster-whisper):

    .venv-speech\\Scripts\\python -m speech.benchmark --models small,base --out report.json

For every candidate model each corpus recording is transcribed with the SAME
options production uses (language, initial_prompt vocabulary, beam size) and
scored against the owner-verified transcript: WER, CER, entity accuracy and
latency.  The report is JSON; choosing a production model from anything but
such a report is guessing.

CUDA is measured dead on this GTX 1070 (Pascal: int8/fp16 refused, float32
loads ~50 s then fails on cublas64_12.dll) — the default device is CPU int8, the
same as production.  Do not re-probe CUDA from here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech.corpus import SpeechCorpus, cer, entity_accuracy, wer  # noqa: E402

PRODUCTION_VOCABULARY = "Jarvis, Zeus, Ollama, Qwen, Stockfish, Spotify, Wikipedia, Repository, Capability, Physikum."


def bench_model(model_name: str, entries: list[dict], *, device: str, compute: str) -> dict:
    from faster_whisper import WhisperModel

    load_started = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute)
    load_seconds = time.perf_counter() - load_started

    rows = []
    for entry in entries:
        audio = entry["audio"]
        if not Path(audio).is_file():
            continue
        started = time.perf_counter()
        try:
            segments, _info = model.transcribe(audio, language="de", beam_size=5,
                                               initial_prompt=PRODUCTION_VOCABULARY, vad_filter=True)
            hypothesis = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            rows.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        latency = time.perf_counter() - started
        rows.append({
            "id": entry["id"], "category": entry.get("category", ""),
            "truth": entry["ground_truth"], "hypothesis": hypothesis,
            "wer": round(wer(entry["ground_truth"], hypothesis), 4),
            "cer": round(cer(entry["ground_truth"], hypothesis), 4),
            "entity_accuracy": entity_accuracy(entry["ground_truth"], hypothesis),
            "latency_seconds": round(latency, 2),
        })
    scored = [r for r in rows if "error" not in r]
    entity_rows = [r for r in scored if r["entity_accuracy"] is not None]
    summary = {
        "model": model_name, "device": device, "compute": compute,
        "load_seconds": round(load_seconds, 1), "utterances": len(scored),
        "wer": round(sum(r["wer"] for r in scored) / len(scored), 4) if scored else None,
        "cer": round(sum(r["cer"] for r in scored) / len(scored), 4) if scored else None,
        "entity_accuracy": round(sum(r["entity_accuracy"] for r in entity_rows) / len(entity_rows), 4) if entity_rows else None,
        "median_latency_seconds": sorted(r["latency_seconds"] for r in scored)[len(scored) // 2] if scored else None,
        "errors": len(rows) - len(scored),
    }
    try:
        del model
    except Exception:  # noqa: BLE001
        pass
    return {"summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m speech.benchmark")
    parser.add_argument("--corpus", default=str(Path(__file__).resolve().parent.parent / "data" / "jarvis" / "speech_corpus"))
    parser.add_argument("--models", default="small", help="comma-separated faster-whisper model names")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute", default="int8")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--held-out-only", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    corpus = SpeechCorpus(args.corpus)
    entries = corpus.list()
    if args.held_out_only:
        entries = [e for e in entries if e.get("held_out")]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print(json.dumps({"ok": False, "error": "the corpus is empty — record owner phrases first (Voice Studio → Spracherkennung trainieren)"}))
        return 1

    report = {"ok": True, "corpus": args.corpus, "utterances": len(entries),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": []}
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"benchmarking {model_name} on {len(entries)} utterance(s)…", file=sys.stderr)
        report["results"].append(bench_model(model_name, entries, device=args.device, compute=args.compute))
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload, encoding="utf-8")
        print(json.dumps({"ok": True, "out": args.out,
                          "summaries": [r["summary"] for r in report["results"]]}, ensure_ascii=False))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
