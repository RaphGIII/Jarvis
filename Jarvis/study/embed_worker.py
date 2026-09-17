"""The embedding worker: a local ONNX text-embedding model in its own process.

Runs as ``python -m study.embed_worker <model_dir> <pooling> <threads>`` so the model's
memory and CPU work never live inside the server process, and at a lower priority, so
indexing a whole semester never makes the conversation stutter.  Protocol: one JSON
object per line on stdin, one JSON answer per line on stdout.

    {"op": "ping"}                              -> {"ok": true, "dim": D}
    {"op": "embed", "texts": [...]}             -> {"ok": true, "dim": D, "vectors": base64(float32 N*D)}

Vectors are L2-normalised, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

MAX_TOKENS = 512
BATCH = 8


def _lower_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001 - priority is a courtesy, not a requirement
        pass


def main() -> int:
    model_dir = Path(sys.argv[1])
    pooling = sys.argv[2] if len(sys.argv) > 2 else "mean"
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    _lower_priority()
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    onnx_file = next((p for p in (model_dir / "onnx" / "model_quantized.onnx", model_dir / "onnx" / "model.onnx", model_dir / "model.onnx") if p.is_file()), None)
    if onnx_file is None:
        raise SystemExit(f"no ONNX model in {model_dir}")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    pad_id = tokenizer.token_to_id("<pad>")
    pad_id = 0 if pad_id is None else pad_id
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_file), options, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}

    def embed(texts: list[str]) -> "np.ndarray":
        parts = []
        for start in range(0, len(texts), BATCH):
            encoded = tokenizer.encode_batch(texts[start:start + BATCH])
            width = max(1, max(len(e.ids) for e in encoded))
            ids = np.full((len(encoded), width), pad_id, dtype=np.int64)
            mask = np.zeros((len(encoded), width), dtype=np.int64)
            for row, item in enumerate(encoded):
                ids[row, :len(item.ids)] = item.ids
                mask[row, :len(item.ids)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = session.run(None, feeds)[0]
            if pooling == "cls":
                vectors = hidden[:, 0]
            else:
                vectors = (hidden * mask[..., None]).sum(axis=1) / np.maximum(1, mask.sum(axis=1, keepdims=True))
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            parts.append((vectors / np.maximum(norms, 1e-12)).astype(np.float32))
        return np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)

    dim = int(embed(["ping"]).shape[1])
    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line.decode("utf-8"))
            if request.get("op") == "ping":
                answer = {"ok": True, "dim": dim}
            elif request.get("op") == "embed":
                texts = [str(t) for t in request.get("texts") or []]
                vectors = embed(texts) if texts else np.zeros((0, dim), dtype=np.float32)
                answer = {"ok": True, "dim": dim, "count": len(texts), "vectors": base64.b64encode(vectors.tobytes()).decode("ascii")}
            else:
                answer = {"ok": False, "error": f"unknown op {request.get('op')!r}"}
        except Exception as exc:  # noqa: BLE001 - one bad request never ends the worker
            answer = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out.write((json.dumps(answer) + "\n").encode("ascii"))
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
