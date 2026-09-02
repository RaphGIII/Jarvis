"""Local image generation on the GTX 1070 — one shot, then the VRAM is free.

Runs inside the image venv (D:\\JarvisLocal\\venv-image):

    venv-image\\Scripts\\python -m imagegen.generate --prompt "..." --out x.png

Model: SD-Turbo (stabilityai/sd-turbo), chosen FOR this GPU: fp16 fits in
8 GB Pascal, and 1–4 denoising steps mean the generation itself takes
seconds — the model load dominates.  The process exits after writing the
image, so the chat model's VRAM residency is disturbed for the shortest
possible time (see the resource note in service/imagegen.py).

Prints one JSON line with the real result: file, seconds, VRAM peak, seed.
Never claims success without the file existing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "stabilityai/sd-turbo"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m imagegen.generate")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--model", default=os.environ.get("ZEUS_IMAGE_MODEL", DEFAULT_MODEL))
    args = parser.parse_args(argv)

    # the HF cache belongs on D: — C: has 17 GB free and the model is ~2.5 GB
    os.environ.setdefault("HF_HOME", r"D:\JarvisLocal\hf_cache")

    try:
        width, height = (int(v) for v in args.size.lower().split("x", 1))
    except ValueError:
        print(json.dumps({"ok": False, "error": f"bad size {args.size!r}; use e.g. 512x512"}))
        return 1

    started = time.perf_counter()
    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None)
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        load_seconds = time.perf_counter() - started

        seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), "little")
        generator = torch.Generator(device=device).manual_seed(seed)
        gen_started = time.perf_counter()
        image = pipe(prompt=args.prompt, negative_prompt=args.negative or None,
                     width=width, height=height, num_inference_steps=max(1, args.steps),
                     guidance_scale=0.0, generator=generator).images[0]
        gen_seconds = time.perf_counter() - gen_started

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        vram_mib = int(torch.cuda.max_memory_allocated() / (1 << 20)) if device == "cuda" else 0
        result = {"ok": out.is_file() and out.stat().st_size > 0, "file": str(out),
                  "bytes": out.stat().st_size if out.is_file() else 0,
                  "model": args.model, "device": device, "seed": seed,
                  "width": width, "height": height, "steps": max(1, args.steps),
                  "load_seconds": round(load_seconds, 1), "generate_seconds": round(gen_seconds, 1),
                  "vram_peak_mib": vram_mib}
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 1
    except Exception as exc:  # noqa: BLE001 - the caller needs the reason, not a stack trace
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
