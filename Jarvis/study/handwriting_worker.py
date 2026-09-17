"""The handwriting worker: a TrOCR line recogniser in its own low-priority process.

Runs as ``python -m study.handwriting_worker <model_dir> <threads> [idle_seconds]``.  Importing torch + transformers
and loading a 558M-parameter model costs 20-60 s and ~2.3 GB of memory; none of that may live in the server process,
and the CPU work must never make the conversation stutter, so the process runs at BELOW_NORMAL priority with a small
thread count and ends itself after ``idle_seconds`` without a request.

Protocol: one JSON object per line on stdin, one JSON answer per line on stdout.

    {"op": "ping"}                                        -> {"ok": true, "model": "<dir name>", "seconds": load time}
    {"op": "recognize", "images": [base64 PNG, ...],
     "max_tokens": 64}                                    -> {"ok": true, "lines": [{"text": ..., "confidence": 0..1}]}

``confidence`` is the mean probability of the generated tokens (greedy decoding), a usable if optimistic signal.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path


def _lower_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001 - priority is a courtesy, not a requirement
        pass


def main() -> int:
    model_dir = Path(sys.argv[1])
    threads = max(1, int(sys.argv[2])) if len(sys.argv) > 2 else 3
    idle = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    _lower_priority()
    for key, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("TRANSFORMERS_VERBOSITY", "error"),
                       ("HF_HUB_DISABLE_TELEMETRY", "1"), ("OMP_NUM_THREADS", str(threads)), ("MKL_NUM_THREADS", str(threads))):
        os.environ[key] = value
    last = [time.monotonic()]

    def watchdog() -> None:
        while True:
            time.sleep(5.0)
            if time.monotonic() - last[0] > idle:
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True, name="handwriting-idle").start()
    started = time.monotonic()
    import torch
    from PIL import Image

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    processor = TrOCRProcessor.from_pretrained(str(model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir)).eval()
    load_seconds = round(time.monotonic() - started, 1)
    last[0] = time.monotonic()

    def recognize(images: list[Image.Image], max_tokens: int) -> list[dict[str, object]]:
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        with torch.inference_mode():
            out = model.generate(pixel_values, max_new_tokens=max_tokens, num_beams=1, do_sample=False, use_cache=True,
                                 output_scores=True, return_dict_in_generate=True)
            scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
        pad = model.generation_config.pad_token_id
        lines = []
        for row, text in enumerate(texts):
            generated = out.sequences[row, 1:]
            keep = torch.isfinite(scores[row])
            if pad is not None:
                keep = keep & (generated[: scores.shape[1]] != pad)
            probs = torch.exp(scores[row][keep]) if bool(keep.any()) else torch.tensor([0.0])
            lines.append({"text": str(text).strip(), "confidence": round(float(probs.mean()), 4),
                          "min_confidence": round(float(probs.min()), 4)})
        return lines

    out_stream = sys.stdout.buffer
    for raw in sys.stdin.buffer:
        last[0] = time.monotonic()
        try:
            request = json.loads(raw.decode("utf-8"))
            if request.get("op") == "ping":
                answer = {"ok": True, "model": model_dir.name, "seconds": load_seconds, "threads": threads}
            elif request.get("op") == "recognize":
                images = [Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB") for item in request.get("images") or []]
                max_tokens = max(8, min(160, int(request.get("max_tokens") or 64)))
                answer = {"ok": True, "lines": recognize(images, max_tokens) if images else []}
            else:
                answer = {"ok": False, "error": f"unknown op {request.get('op')!r}"}
        except Exception as exc:  # noqa: BLE001 - one bad request never ends the worker
            answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        last[0] = time.monotonic()
        out_stream.write((json.dumps(answer, ensure_ascii=True) + "\n").encode("ascii"))
        out_stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
