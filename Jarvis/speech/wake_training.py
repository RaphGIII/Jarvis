"""Train a local "Zeus" wake-word detector.  Runs inside .venv-speech.

    .venv-speech\\Scripts\\python -m speech.wake_training [--out data/models/wake/zeus.npz]

How it works, and what it is not.  openWakeWord ships a frozen feature
backbone (melspectrogram + a speech-embedding model, both ONNX) that turns 80 ms
frames into 96-dimensional embeddings; every one of its wake models is a small
classifier over a window of 16 of those.  There is no "Zeus" classifier, and
renaming ``hey_jarvis`` does not make one -- measured here, ``hey_jarvis``
scores 0.0003 on a spoken "Zeus".

So this trains the missing classifier, on this machine, from local material:

*Positives.*  "Zeus" spoken by every piper voice installed under data/voices,
at several speeds, pitches and gains, placed at a random point in a clip with
silence and noise around it.  Owner recordings under data/wake/positive/*.wav
are added when present and weighted more, because a synthetic voice is not the
owner's.

*Negatives.*  Other words and sentences from the same voices -- including near
neighbours ("Zeit", "Zeh", "Zoo", "Deus", "juice", "seuss", "hey jarvis") --
silence, noise, recordings under data/wake/negative/*.wav, and, importantly,
every window of a *positive* clip in which the word has not finished yet.

*Labels the way the listener sees them.*  Each clip is streamed frame by frame
through the same feature pipeline the microphone uses, and every 16-frame
window gets a label: positive only when the word ended inside the last 400 ms
of that window.  Training on end-of-clip windows alone measured 142 false
activations an hour; this is what fixed it.

*Model.*  A small MLP over the flattened 16x96 window (scikit-learn), exported
to plain numpy weights so the listener needs nothing but numpy to score a
frame.  Measured on a held-out split of *clips* (never of windows, which would
leak) and reported honestly: recall per clip, false activations per hour of
negative audio streamed through the real detector, and per-frame latency.

The limit is stated rather than hidden: synthetic positives cover the *word*,
not the owner's voice.  The report says how many owner samples were used; with
none, expect the real-microphone recall to be lower than the held-out figure.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME = 1280
CLIP_SAMPLES = SAMPLE_RATE * 2  # every training clip is 2.0 s
WINDOW_FRAMES = 16
#: A window is positive when the word ended within this many seconds before
#: the window's end -- the detector should fire as the word finishes.
POSITIVE_TAIL_SECONDS = 0.40

POSITIVE_PHRASES = ("Zeus", "Zeus.", "Zeus!", "Hey Zeus", "Zeus,")
NEGATIVE_PHRASES = (
    "Zeit", "Zeh", "Zoo", "Deus", "Zeug", "Zaun", "Zeus ist ein Gott", "Zeugnis", "Zeile",
    "juice", "seuss", "choose", "news", "loose", "hey jarvis", "Jarvis", "alexa", "computer", "Siri",
    "Guten Morgen", "Wie spät ist es", "Mach das Licht an", "Spiel Musik", "Was ist das Wetter",
    "Good morning", "what time is it", "turn on the light", "play some music", "how is the weather",
    "Ich habe heute viel zu tun", "Das ist eine gute Idee", "I will call you later", "the meeting starts at nine",
    "eins zwei drei vier", "one two three four", "Danke", "thank you", "ja", "nein", "yes", "no",
    "Kannst du mir helfen", "Ich gehe jetzt nach Hause", "Das Essen ist fertig", "Bitte leiser",
    "Let me think about it", "That sounds reasonable", "Where did I put my keys", "See you tomorrow",
)


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    if rate != SAMPLE_RATE:
        import scipy.signal as ss

        x = ss.resample_poly(x, SAMPLE_RATE, rate)
    return x


def _voices(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.onnx") if (p.with_suffix(".onnx.json")).exists())


def _synth(voice, text: str, length_scale: float) -> np.ndarray:
    from piper import SynthesisConfig

    chunks = list(voice.synthesize(text, SynthesisConfig(length_scale=length_scale)))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32) * 32767.0
    rate = chunks[0].sample_rate
    if rate != SAMPLE_RATE:
        import scipy.signal as ss

        x = ss.resample_poly(x, SAMPLE_RATE, rate)
    return x


def _trim(x: np.ndarray) -> np.ndarray:
    """Cut leading/trailing silence so the word's end is where the sound ends."""

    if len(x) == 0:
        return x
    level = np.abs(x)
    threshold = max(200.0, float(level.max()) * 0.02)
    idx = np.nonzero(level > threshold)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - 400): min(len(x), idx[-1] + 800)]


def _pitch(x: np.ndarray, factor: float) -> np.ndarray:
    import scipy.signal as ss

    n = max(1, int(round(len(x) / factor)))
    return ss.resample(x, n).astype(np.float32)


def _place(x: np.ndarray, rng: random.Random, *, noise_db: float, gain: float) -> tuple[np.ndarray, int]:
    """The word at a random position in a 2 s clip with noise; returns (clip, word_end_sample)."""

    clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    x = x[:CLIP_SAMPLES] * gain
    start = rng.randint(0, max(0, CLIP_SAMPLES - len(x)))
    clip[start:start + len(x)] += x
    end = start + len(x)
    if noise_db > -80:
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0.0, 1.0, CLIP_SAMPLES).astype(np.float32)
        noise = np.convolve(noise, np.ones(4) / 4, mode="same")
        clip += noise * (10 ** (noise_db / 20)) * 32767.0
    return np.clip(clip, -32767, 32767), end


def build_dataset(voices_dir: Path, wake_dir: Path, *, per_voice: int, seed: int, log=print):
    """Clips with (audio, is_positive, word_end_sample, weight)."""

    from piper import PiperVoice

    rng = random.Random(seed)
    clips: list[tuple[np.ndarray, int, int, float]] = []
    meta = {"voices": [], "owner_positive": 0, "owner_negative": 0}
    for path in _voices(voices_dir):
        try:
            voice = PiperVoice.load(str(path))
        except Exception as exc:
            log(f"  skip {path.name}: {exc}")
            continue
        meta["voices"].append(path.name)
        base_pos = [_trim(_synth(voice, p, ls)) for p in POSITIVE_PHRASES for ls in (0.8, 1.0, 1.25)]
        base_neg = [_trim(_synth(voice, p, ls)) for p in NEGATIVE_PHRASES for ls in (0.9, 1.1)]
        for _ in range(per_voice):
            x = _pitch(rng.choice(base_pos), rng.uniform(0.86, 1.14))
            clip, end = _place(x, rng, noise_db=rng.choice((-80, -80, -45, -35, -28)), gain=rng.uniform(0.3, 1.0))
            clips.append((clip, 1, end, 1.0))
        for _ in range(per_voice):
            x = _pitch(rng.choice(base_neg), rng.uniform(0.86, 1.14))
            clip, end = _place(x, rng, noise_db=rng.choice((-80, -80, -45, -35, -28)), gain=rng.uniform(0.3, 1.0))
            clips.append((clip, 0, end, 1.0))
        log(f"  {path.name}: {len(base_pos)} positive / {len(base_neg)} negative base clips")
    for _ in range(max(40, per_voice // 2)):
        clip, _end = _place(np.zeros(1, dtype=np.float32), rng, noise_db=rng.choice((-80, -50, -35, -25)), gain=1.0)
        clips.append((clip, 0, 0, 1.0))
    for kind, label, weight, copies in (("positive", 1, 3.0, 8), ("negative", 0, 1.0, 3)):
        folder = wake_dir / kind
        for path in sorted(folder.glob("*.wav")) if folder.is_dir() else []:
            x = _trim(_load_wav(path))
            for _ in range(copies):
                clip, end = _place(_pitch(x, rng.uniform(0.95, 1.05)), rng, noise_db=rng.choice((-80, -40)), gain=rng.uniform(0.6, 1.0))
                clips.append((clip, label, end, weight))
            meta[f"owner_{kind}"] += 1
    rng.shuffle(clips)
    return clips, meta


def stream_windows(af, clip: np.ndarray, is_positive: int, end: int):
    """Every 16-frame window of a clip, labelled as the detector will see it."""

    af.reset()
    xs, ys = [], []
    audio = clip.astype(np.int16)
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        af(audio[i:i + FRAME])
        window = af.get_features(WINDOW_FRAMES)[0]
        if window.shape[0] < WINDOW_FRAMES or i < FRAME * 4:
            continue
        window_end = i + FRAME
        positive = bool(is_positive) and (end <= window_end) and (window_end - end) <= POSITIVE_TAIL_SECONDS * SAMPLE_RATE
        xs.append(window.reshape(-1).astype(np.float32))
        ys.append(1 if positive else 0)
    return xs, ys


def extract(af, clips, log=print):
    t = time.time()
    X, y, w, clip_index = [], [], [], []
    for index, (clip, label, end, weight) in enumerate(clips):
        xs, ys = stream_windows(af, clip, label, end)
        X.extend(xs); y.extend(ys); w.extend([weight] * len(xs)); clip_index.extend([index] * len(xs))
    log(f"  {len(X)} windows from {len(clips)} clips in {time.time() - t:.1f}s")
    return np.stack(X), np.array(y), np.array(w), np.array(clip_index)


def train(clips, af, *, seed: int, log=print):
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(seed)
    n = len(clips)
    holdout = set(rng.choice(n, size=max(1, n // 4), replace=False).tolist())
    X, y, w, ci = extract(af, clips, log=log)
    train_mask = np.array([c not in holdout for c in ci])
    mean, std = X[train_mask].mean(axis=0), X[train_mask].std(axis=0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=300, random_state=seed, early_stopping=True)
    # Positive windows are rare; repeat them so the MLP does not learn "never".
    Xtr, ytr = X[train_mask], y[train_mask]
    pos = np.nonzero(ytr == 1)[0]
    repeat = max(1, int((ytr == 0).sum() / max(1, len(pos)) / 2))
    idx = np.concatenate([np.arange(len(ytr))] + [pos] * (repeat - 1))
    t = time.time()
    clf.fit((Xtr[idx] - mean) / std, ytr[idx])
    log(f"  MLP trained on {len(idx)} windows ({repeat}x positives) in {time.time() - t:.1f}s")
    weights = {"W1": clf.coefs_[0].astype(np.float32), "b1": clf.intercepts_[0].astype(np.float32),
               "W2": clf.coefs_[1].astype(np.float32), "b2": clf.intercepts_[1].astype(np.float32),
               "W3": clf.coefs_[2].astype(np.float32), "b3": clf.intercepts_[2].astype(np.float32),
               "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return weights, sorted(holdout)


def evaluate(weights, af, clips, holdout, thresholds=(0.5, 0.7, 0.85), log=print):
    """Per clip, through the real detector: did it fire on the word, and on nothing else?"""

    from speech.wake_zeus import ZeusDetector

    report = {}
    for threshold in thresholds:
        detector = ZeusDetector.from_weights(weights, threshold=threshold, hits=2, cooldown=0.0, features=af)
        tp = fn = fp = 0
        neg_seconds = 0.0
        t = time.time(); frames = 0
        for index in holdout:
            clip, label, end, _w = clips[index]
            detector.reset()
            audio = clip.astype(np.int16)
            fired_at = []
            for i in range(0, len(audio) - FRAME + 1, FRAME):
                frames += 1
                if detector.feed(audio[i:i + FRAME]):
                    fired_at.append(i + FRAME)
            if label:
                good = [f for f in fired_at if end - 0.2 * SAMPLE_RATE <= f <= end + 0.8 * SAMPLE_RATE]
                tp += 1 if good else 0
                fn += 0 if good else 1
                fp += len(fired_at) - len(good)
            else:
                fp += len(fired_at)
                neg_seconds += len(audio) / SAMPLE_RATE
        latency = (time.time() - t) / max(1, frames) * 1000
        hours = max(1e-6, neg_seconds / 3600)
        report[str(threshold)] = {"recall": round(tp / max(1, tp + fn), 4), "positives": tp + fn,
                                  "false_activations": fp, "false_per_hour_negative_audio": round(fp / hours, 1),
                                  "negative_audio_seconds": round(neg_seconds, 1), "latency_ms_per_frame": round(latency, 2)}
        log(f"  threshold {threshold}: recall {report[str(threshold)]['recall']} on {tp + fn} clips; "
            f"{fp} false activations in {neg_seconds:.0f}s of negative audio ({fp / hours:.1f}/h); {latency:.1f} ms/frame")
    return report


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train the local 'Zeus' wake-word classifier")
    parser.add_argument("--voices", default=str(root / "data" / "voices"))
    parser.add_argument("--wake-dir", default=str(root / "data" / "wake"), help="owner recordings: positive/*.wav, negative/*.wav")
    parser.add_argument("--out", default=str(root / "data" / "models" / "wake" / "zeus.npz"))
    parser.add_argument("--per-voice", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args(argv)

    from openwakeword.utils import AudioFeatures

    af = AudioFeatures(inference_framework="onnx")
    started = time.time()
    print("building the dataset")
    clips, meta = build_dataset(Path(args.voices), Path(args.wake_dir), per_voice=args.per_voice, seed=args.seed)
    positives = sum(1 for c in clips if c[1] == 1)
    print(f"  {positives} positive, {len(clips) - positives} negative clips; owner samples: {meta['owner_positive']}+/{meta['owner_negative']}-")
    print("training")
    weights, holdout = train(clips, af, seed=args.seed)
    print(f"evaluating on {len(holdout)} held-out clips through the streaming detector")
    report = evaluate(weights, af, clips, holdout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **weights)
    manifest = {"model": str(out), "threshold": args.threshold, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": round(time.time() - started, 1), "dataset": meta, "report": report,
                "clips": {"positive": positives, "negative": len(clips) - positives, "holdout": len(holdout)}}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.json')} in {time.time() - started:.0f}s")
    if meta["owner_positive"] == 0:
        print("NOTE: no owner recordings under data/wake/positive -- the held-out figures are for synthetic voices only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
