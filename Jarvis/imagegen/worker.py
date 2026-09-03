"""The persistent image worker: load once, generate many, hold the GPU briefly.

The one-shot design (imagegen/generate.py) paid 200–370 s of model load per
image — the owner's "25 second generation" became minutes of perceived delay.
This worker is a long-lived process (in the image venv) speaking JSON lines
over stdin/stdout:

    {"op": "generate", "id": "...", "prompt": "...", "out": "...", ...}
    {"op": "ping"} / {"op": "quit"}

It loads SD-Turbo ONCE into CPU RAM (~2.5 GB), and for each job moves the
pipeline to the GPU, generates, saves, and moves it BACK to CPU, freeing the
VRAM.  Phases are streamed as events so the product can show real progress:

    {"event": "phase", "id": ..., "phase": "loading_model|to_gpu|generating|saving|to_cpu", "at": 1.2}
    {"event": "result", "id": ..., "ok": true, "file": ..., timings...}

Typical steady-state cost measured on the GTX 1070: to_gpu ~3–6 s,
generation ~9–13 s, to_cpu ~2–3 s — the GPU is held for well under half a
minute instead of the whole model lifetime.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("ZEUS_IMAGE_MODEL", "stabilityai/sd-turbo")

_pipe = None
_pipe_model = ""
_torch = None


def _say(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _ensure_pipe(model: str, notify) -> None:
    global _pipe, _pipe_model, _torch
    if _pipe is not None and _pipe_model == model:
        return
    os.environ.setdefault("HF_HOME", r"D:\JarvisLocal\hf_cache")
    notify("loading_model")
    import torch
    from diffusers import AutoPipelineForText2Image

    _torch = torch
    if _pipe is not None:
        # switching models: free the old one first (8 GB is tight)
        try:
            _pipe.to("cpu")
            del _pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        _pipe = None
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, safety_checker=None)
    _pipe.enable_attention_slicing()
    _pipe_model = model


def _generate(req: dict) -> dict:
    rid = req.get("id", "")
    started = time.monotonic()
    timings: dict[str, float] = {}

    def notify(phase: str) -> None:
        timings[phase] = round(time.monotonic() - started, 2)
        _say({"event": "phase", "id": rid, "phase": phase, "at": timings[phase]})

    try:
        width, height = (int(v) for v in str(req.get("size", "512x512")).lower().split("x", 1))
    except ValueError:
        return {"event": "result", "id": rid, "ok": False, "error": f"bad size {req.get('size')!r}"}
    try:
        model = str(req.get("model") or DEFAULT_MODEL)
        _ensure_pipe(model, notify)
        torch = _torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            notify("to_gpu")
            _pipe.to(device)
            torch.cuda.reset_peak_memory_stats()
        seed = int(req.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "little")
        notify("generating")
        generator = torch.Generator(device=device).manual_seed(seed)
        # turbo/LCM models want cfg 0; real SD wants CFG (7-8) to FOLLOW the
        # prompt -- guidance_scale 0 is exactly why sd-turbo drew a park for
        # "black hole".  The caller passes the right cfg for the model.
        cfg = float(req.get("cfg", 0.0))
        image = _pipe(prompt=str(req.get("prompt", "")), negative_prompt=str(req.get("negative", "")) or None,
                      width=width, height=height, num_inference_steps=max(1, int(req.get("steps", 2))),
                      guidance_scale=cfg, generator=generator).images[0]
        notify("saving")
        out = Path(str(req.get("out", "")))
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        vram = int(torch.cuda.max_memory_allocated() / (1 << 20)) if device == "cuda" else 0
        if device == "cuda":
            notify("to_cpu")
            _pipe.to("cpu")
            torch.cuda.empty_cache()
        timings["total"] = round(time.monotonic() - started, 2)
        ok = out.is_file() and out.stat().st_size > 0
        return {"event": "result", "id": rid, "ok": ok, "file": str(out),
                "bytes": out.stat().st_size if ok else 0, "model": model,
                "device": device, "seed": seed, "width": width, "height": height,
                "steps": max(1, int(req.get("steps", 2))), "cfg": cfg, "vram_peak_mib": vram,
                "timings": timings,
                "error": "" if ok else "the file was not written"}
    except Exception as exc:  # noqa: BLE001 - the caller needs the reason
        try:
            if _pipe is not None and _torch is not None and _torch.cuda.is_available():
                _pipe.to("cpu")
                _torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return {"event": "result", "id": rid, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:400], "timings": timings}


def main() -> int:
    _say({"event": "hello", "pid": os.getpid(), "model": DEFAULT_MODEL})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            _say({"event": "error", "error": "bad json"})
            continue
        op = req.get("op")
        if op == "quit":
            break
        if op == "ping":
            _say({"event": "pong", "loaded": _pipe is not None})
            continue
        if op == "generate":
            _say(_generate(req))
            continue
        _say({"event": "error", "error": f"unknown op {op!r}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
