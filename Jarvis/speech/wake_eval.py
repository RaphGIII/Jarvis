"""Score recordings through the real detector, and evaluate the owner's samples.

Two jobs, one code path, so the Voice Studio test and the calibration report
cannot disagree with each other or with the listener:

* :func:`score_wav` -- one WAV through :class:`speech.wake_zeus.ZeusDetector`,
  frame by frame exactly as the listener feeds the microphone.  Returns the
  best per-frame score and whether the detector *fired* (two consecutive
  frames over the threshold).  Used by ``/api/voice/wake/test``.

* :func:`evaluate` -- every owner recording under ``data/wake/{positive,
  negative}`` through the same function; per-clip scores, recall and
  rejection at candidate thresholds, and a recommended threshold.  Written to
  ``data/models/wake/zeus_eval.json`` for Voice Studio and the doctor.

Preprocessing contract (the same one training uses): 16 kHz mono int16.  Other
rates are resampled, stereo is averaged.  Nothing about the owner's TTS volume
or any other setting reaches this module -- a score is a function of audio,
weights and nothing else.

Why the tail padding: a wizard recording stops at a fixed time, so the word
may end on the last frame.  The microphone keeps running after a real "Zeus";
the detector needs one more frame to confirm.  Clips are therefore followed by
``TAIL_SECONDS`` of low-level noise -- *noise*, because appending digital
silence is unrealistic (a microphone never delivers exact zeros) and the
transition into it measured as a false 0.99 on one otherwise silent clip.

Honesty about the numbers: the recordings under data/wake are the ones the
model was trained on unless training held some out (the manifest's
``owner_holdout`` says so).  In-sample recall is an upper bound; the report
labels which it is.  Hard negatives (Jesus, Servus, ...) are only evaluated
when the owner recorded them under ``data/wake/hard_negative`` -- synthetic
voices are not the owner's and are not reported as real-world acceptance.

    .venv-speech\\Scripts\\python -m speech.wake_eval [--threshold 0.55]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
FRAME = 1280
TAIL_SECONDS = 0.6
#: RMS of the padding noise; a quiet room's dither, far below any speech.
TAIL_NOISE_RMS = 4.0
#: A recording whose loudest frame is below this never contained a word.
SILENT_RMS = 25.0
CANDIDATE_THRESHOLDS = (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAKE_DIR = ROOT / "data" / "wake"
DEFAULT_EVAL = ROOT / "data" / "models" / "wake" / "zeus_eval.json"


def load_pcm(data: bytes | str | Path) -> np.ndarray:
    """WAV (bytes or path) -> float32 samples in the int16 range, 16 kHz mono."""

    handle = wave.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data), "rb")
    with handle as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif width == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 65536.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        import scipy.signal as ss

        x = ss.resample_poly(x, SAMPLE_RATE, rate).astype(np.float32)
    return x


def with_tail(x: np.ndarray, *, seed: int = 0) -> np.ndarray:
    tail = np.random.default_rng(seed).normal(0.0, TAIL_NOISE_RMS, int(TAIL_SECONDS * SAMPLE_RATE)).astype(np.float32)
    return np.concatenate([x.astype(np.float32), tail])


def score_pcm(detector: Any, x: np.ndarray, *, tail: bool = True) -> dict[str, Any]:
    """Stream ``x`` through ``detector`` from a clean state; best score, fired?, per-frame trace."""

    detector.reset()
    audio = with_tail(x) if tail else x.astype(np.float32)
    scores: list[float] = []
    fired_at: list[int] = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        if detector.feed(audio[i:i + FRAME]):
            fired_at.append(i + FRAME)
        scores.append(round(float(detector.last_score), 4))
    peak = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    loudest = max((float(np.sqrt(np.mean(np.square(x[i:i + FRAME])))) for i in range(0, max(1, len(x) - FRAME + 1), FRAME)), default=0.0)
    return {"score": max(scores, default=0.0), "detected": bool(fired_at), "fired_at_seconds": [round(f / SAMPLE_RATE, 2) for f in fired_at],
            "frames": scores, "threshold": float(detector.threshold), "seconds": round(len(x) / SAMPLE_RATE, 2),
            "rms": round(peak, 1), "loudest_frame_rms": round(loudest, 1), "silent": loudest < SILENT_RMS}


def score_wav(detector: Any, data: bytes | str | Path, **kwargs: Any) -> dict[str, Any]:
    return score_pcm(detector, load_pcm(data), **kwargs)


def _fires(frames: list[float], threshold: float, hits: int = 2) -> bool:
    streak = 0
    for score in frames:
        streak = streak + 1 if score >= threshold else 0
        if streak >= hits:
            return True
    return False


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "count": 0}
    return {"min": round(min(values), 3), "median": round(statistics.median(values), 3), "max": round(max(values), 3), "count": len(values)}


def recommend(positives: list[list[float]], negatives: list[list[float]], *, candidates=CANDIDATE_THRESHOLDS, hits: int = 2) -> dict[str, Any]:
    """Recall/rejection per candidate threshold and the recommendation.

    Rule, stated so the owner can disagree with it: among thresholds at which
    every negative is rejected, take those with the highest recall; of them,
    the one nearest the midpoint between the loudest negative peak and the
    quietest detected positive peak -- equal margin against a false
    activation and a missed word in a room the samples did not cover.  If no
    threshold rejects every negative, the recommendation is the one with the
    fewest false activations, and the report says the samples do not separate.
    """

    rows = []
    for t in candidates:
        tp = sum(1 for f in positives if _fires(f, t, hits))
        fp = sum(1 for f in negatives if _fires(f, t, hits))
        rows.append({"threshold": t, "recall": round(tp / len(positives), 3) if positives else None, "positives_detected": tp,
                     "false_activations": fp, "rejection": round(1 - fp / len(negatives), 3) if negatives else None})
    clean = [r for r in rows if r["false_activations"] == 0 and r["recall"] is not None]
    if clean:
        top = max(r["recall"] for r in clean)
        tied = [r for r in clean if r["recall"] == top]
        neg_peak = max((max(f) for f in negatives if f), default=0.0)
        detected = [max(f) for f in positives if f and _fires(f, tied[0]["threshold"], hits)]
        pos_floor = min(detected) if detected else 1.0
        midpoint = (neg_peak + pos_floor) / 2
        best = min(tied, key=lambda r: (abs(r["threshold"] - midpoint), -r["threshold"]))
        separates = True
    elif rows:
        best = min(rows, key=lambda r: (r["false_activations"], -(r["recall"] or 0)))
        separates = False
    else:
        return {"rows": [], "recommended": None, "separates": False}
    return {"rows": rows, "recommended": best["threshold"], "separates": separates}


def evaluate(detector: Any, wake_dir: Path = DEFAULT_WAKE_DIR, *, threshold: float | None = None, log=print) -> dict[str, Any]:
    started = time.time()
    clips: dict[str, list[dict[str, Any]]] = {}
    for kind in ("positive", "negative", "hard_negative"):
        folder = wake_dir / kind
        clips[kind] = []
        for path in sorted(folder.glob("*.wav")) if folder.is_dir() else []:
            try:
                result = score_wav(detector, path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
                clips[kind].append({"file": path.name, "error": str(exc)})
                continue
            clips[kind].append({"file": path.name, **{k: result[k] for k in ("score", "detected", "fired_at_seconds", "seconds", "rms", "loudest_frame_rms", "silent")},
                                "frames": result["frames"]})
    usable_pos = [c for c in clips["positive"] if "frames" in c and not c["silent"]]
    silent_pos = [c["file"] for c in clips["positive"] if c.get("silent")]
    negatives = [c for c in clips["negative"] + clips["hard_negative"] if "frames" in c]
    rec = recommend([c["frames"] for c in usable_pos], [c["frames"] for c in negatives])
    effective = float(threshold if threshold is not None else detector.threshold)
    at_effective = next((r for r in rec["rows"] if abs(r["threshold"] - effective) < 1e-9), None)
    if at_effective is None:
        tp = sum(1 for c in usable_pos if _fires(c["frames"], effective))
        fp = sum(1 for c in negatives if _fires(c["frames"], effective))
        at_effective = {"threshold": effective, "recall": round(tp / len(usable_pos), 3) if usable_pos else None, "positives_detected": tp,
                        "false_activations": fp, "rejection": round(1 - fp / len(negatives), 3) if negatives else None}
    report = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_fingerprint": getattr(detector, "fingerprint", ""),
        "effective_threshold": effective,
        "counts": {"positive": len(clips["positive"]), "positive_usable": len(usable_pos), "negative": len(clips["negative"]),
                   "hard_negative": len(clips["hard_negative"])},
        "silent_positives": silent_pos,
        "positive_scores": _summary([c["score"] for c in usable_pos]),
        "negative_scores": _summary([c["score"] for c in negatives]),
        "at_effective_threshold": at_effective,
        "thresholds": rec["rows"],
        "recommended_threshold": rec["recommended"],
        "separates": rec["separates"],
        "hard_negatives_evaluated": bool(clips["hard_negative"]),
        "clips": {kind: [{k: v for k, v in c.items() if k != "frames"} for c in rows] for kind, rows in clips.items()},
        "seconds": round(time.time() - started, 1),
    }
    log(f"  positives {report['positive_scores']}  negatives {report['negative_scores']}")
    for row in rec["rows"]:
        log(f"  threshold {row['threshold']}: recall {row['recall']} ({row['positives_detected']}/{len(usable_pos)}), "
            f"false activations {row['false_activations']}/{len(negatives)}")
    log(f"  recommended {rec['recommended']} ({'separates' if rec['separates'] else 'DOES NOT separate'}); "
        f"silent positives skipped: {silent_pos or 'none'}; hard negatives: {len(clips['hard_negative'])}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the owner's wake-word recordings through the real detector")
    parser.add_argument("--model", default="")
    parser.add_argument("--wake-dir", default=str(DEFAULT_WAKE_DIR))
    parser.add_argument("--threshold", type=float, default=None, help="the effective threshold to report against")
    parser.add_argument("--out", default=str(DEFAULT_EVAL))
    parser.add_argument("--json", action="store_true", help="print the report as JSON (last line)")
    args = parser.parse_args(argv)

    from speech.wake_zeus import DEFAULT_MODEL, ZeusDetector

    detector = ZeusDetector.load(args.model or DEFAULT_MODEL, threshold=args.threshold)
    report = evaluate(detector, Path(args.wake_dir), threshold=args.threshold, log=(lambda *_: None) if args.json else print)
    # The recordings on disk are the ones training used; held-out figures live in the manifest.
    report["in_sample"] = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report))
    else:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
