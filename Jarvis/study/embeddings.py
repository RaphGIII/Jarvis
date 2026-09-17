"""Semantic embeddings for Studium: which provider turns text into vectors, and the rule that keeps material private.

The provider is configuration, not code (``config/study.json`` → ``embeddings``):

    {"provider": "onnx",   "model": "multilingual-e5-base", "model_dir": "D:\\JarvisLocal\\embeddings\\multilingual-e5-base",
     "pooling": "mean", "query_prefix": "query: ", "passage_prefix": "passage: ", "threads": 4}
    {"provider": "ollama", "model": "bge-m3", "url": "http://127.0.0.1:11434"}
    {"provider": "none"}

Every provider says whether it is local.  The owner's study material leaves the machine
only when the configuration explicitly says ``"allow_external": true`` -- a provider that
is not local is otherwise refused and search stays keyword-only, saying so.

A provider's ``id`` names model and pooling; vectors stored under one id are never
compared with vectors of another, so changing the provider re-embeds, it never mixes.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "study.json"
DEFAULT_MODEL_DIR = Path(r"D:\JarvisLocal\embeddings\multilingual-e5-base")
DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "onnx", "model": "multilingual-e5-base", "model_dir": str(DEFAULT_MODEL_DIR), "pooling": "mean",
    "query_prefix": "query: ", "passage_prefix": "passage: ", "threads": 4, "allow_external": False,
}


class EmbeddingProvider(Protocol):
    id: str
    local: bool

    def available(self) -> bool: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def status(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class NoEmbeddings:
    """Semantic search switched off (or refused): search is keyword and fragment matching only."""

    local = True

    def __init__(self, reason: str = "") -> None:
        self.id = "none"
        self.reason = reason

    def available(self) -> bool:
        return False

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("no embedding provider")

    def embed_query(self, text: str) -> np.ndarray:
        raise RuntimeError("no embedding provider")

    def status(self) -> dict[str, Any]:
        return {"provider": "none", "available": False, "local": True, "reason": self.reason}

    def close(self) -> None:
        return None


class OnnxEmbeddings:
    """A local ONNX model (default: multilingual-e5-base, int8) in a low-priority worker process."""

    local = True

    def __init__(self, *, model: str, model_dir: str | Path, pooling: str = "mean", query_prefix: str = "", passage_prefix: str = "",
                 threads: int = 4, timeout: float = 120.0) -> None:
        self.model = model
        self.model_dir = Path(model_dir)
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.threads = threads
        self.timeout = timeout
        self.id = f"onnx:{model}:{pooling}"
        self.dim = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.Lock()
        self._reason = ""

    def _files_present(self) -> bool:
        onnx = any((self.model_dir / rel).is_file() for rel in ("onnx/model_quantized.onnx", "onnx/model.onnx", "model.onnx"))
        return onnx and (self.model_dir / "tokenizer.json").is_file()

    def _start(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        if self._reason:
            return False
        if not self._files_present():
            self._reason = f"Modell fehlt: {self.model_dir}"
            return False
        try:
            import importlib.util

            missing = [name for name in ("onnxruntime", "tokenizers") if importlib.util.find_spec(name) is None]
            if missing:
                self._reason = "Python-Pakete fehlen: " + ", ".join(missing)
                return False
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen([sys.executable, "-m", "study.embed_worker", str(self.model_dir), self.pooling, str(self.threads)],
                                             cwd=str(_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            self._reason = f"Worker startet nicht: {exc}"
            return False
        self._lines = queue.Queue()
        stdout = self._process.stdout

        def pump() -> None:
            assert stdout is not None
            for line in stdout:
                self._lines.put(line)
            self._lines.put(b"")

        threading.Thread(target=pump, daemon=True, name="study-embed-worker").start()
        answer = self._call_locked({"op": "ping"}, timeout=90.0)
        if not answer.get("ok"):
            self._reason = f"Worker antwortet nicht: {answer.get('error', '')}"
            self.close()
            return False
        self.dim = int(answer.get("dim") or 0)
        return True

    def _call_locked(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            return {"ok": False, "error": "worker not running"}
        try:
            process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            process.stdin.flush()
            line = self._lines.get(timeout=timeout or self.timeout)
        except (OSError, queue.Empty) as exc:
            self.close()
            return {"ok": False, "error": f"worker failed: {type(exc).__name__}"}
        if not line:
            self.close()
            return {"ok": False, "error": "worker exited"}
        try:
            return json.loads(line.decode("ascii"))
        except ValueError:
            return {"ok": False, "error": "unreadable worker answer"}

    def available(self) -> bool:
        with self._lock:
            return self._start()

    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            if not self._start():
                raise RuntimeError(self._reason or "embedding worker unavailable")
            answer = self._call_locked({"op": "embed", "texts": texts})
        if not answer.get("ok"):
            raise RuntimeError(str(answer.get("error") or "embedding failed"))
        dim = int(answer["dim"])
        vectors = np.frombuffer(base64.b64decode(answer["vectors"]), dtype=np.float32)
        return vectors.reshape(len(texts), dim) if texts else np.zeros((0, dim), dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def status(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        return {"provider": "onnx", "id": self.id, "model": self.model, "model_dir": str(self.model_dir), "local": True,
                "files_present": self._files_present(), "running": running, "dim": self.dim, "available": running or (self._files_present() and not self._reason),
                "reason": self._reason}

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass


class OllamaEmbeddings:
    """An embedding model served by Ollama (``/api/embed``).  Local only when the URL is this machine."""

    def __init__(self, *, model: str, url: str = "http://127.0.0.1:11434", query_prefix: str = "", passage_prefix: str = "", timeout: float = 120.0) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.timeout = timeout
        self.id = f"ollama:{model}"
        host = (urllib.parse.urlparse(self.url).hostname or "").lower()
        self.local = host in {"127.0.0.1", "localhost", "::1"}
        self._reason = ""

    def _embed(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(f"{self.url}/api/embed", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured local endpoint
            data = json.loads(response.read().decode("utf-8"))
        vectors = np.asarray(data.get("embeddings") or [], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) if vectors.size else np.ones((0, 1), dtype=np.float32)
        return vectors / np.maximum(norms, 1e-12)

    def available(self) -> bool:
        try:
            self._embed(["ping"])
            return True
        except Exception as exc:  # noqa: BLE001
            self._reason = f"{type(exc).__name__}: {exc}"
            return False

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def status(self) -> dict[str, Any]:
        return {"provider": "ollama", "id": self.id, "model": self.model, "url": self.url, "local": self.local, "reason": self._reason}

    def close(self) -> None:
        return None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """``config/study.json`` → ``embeddings``, over the defaults; ``ZEUS_EMBEDDINGS`` = ``none`` switches semantic search off."""

    config = dict(DEFAULT_CONFIG)
    target = Path(path) if path else CONFIG_PATH
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
        config.update((loaded.get("embeddings") or {}) if isinstance(loaded, dict) else {})
    except (OSError, ValueError):
        pass
    if os.environ.get("ZEUS_EMBEDDINGS_DIR"):
        config["model_dir"] = os.environ["ZEUS_EMBEDDINGS_DIR"]
    if os.environ.get("ZEUS_EMBEDDINGS", "").lower() == "none":
        config["provider"] = "none"
    return config


def create_provider(config: dict[str, Any] | None = None) -> EmbeddingProvider:
    """The configured provider -- or none, with the reason, when it would send material off this machine without permission."""

    config = config if config is not None else load_config()
    kind = str(config.get("provider") or "none").lower()
    if kind == "onnx":
        provider: EmbeddingProvider = OnnxEmbeddings(model=str(config.get("model") or "model"), model_dir=config.get("model_dir") or DEFAULT_MODEL_DIR,
                                                     pooling=str(config.get("pooling") or "mean"), query_prefix=str(config.get("query_prefix") or ""),
                                                     passage_prefix=str(config.get("passage_prefix") or ""), threads=int(config.get("threads") or 4))
    elif kind == "ollama":
        provider = OllamaEmbeddings(model=str(config.get("model") or "bge-m3"), url=str(config.get("url") or "http://127.0.0.1:11434"),
                                    query_prefix=str(config.get("query_prefix") or ""), passage_prefix=str(config.get("passage_prefix") or ""))
    else:
        return NoEmbeddings("semantische Suche ist ausgeschaltet")
    if not provider.local and not config.get("allow_external"):
        return NoEmbeddings(f"{provider.id} liegt nicht auf diesem Rechner; Studienmaterial verlässt den Rechner nur mit allow_external")
    return provider


def passage_text(chunk_text: str, *, title: str = "", heading: str = "") -> str:
    """What a chunk is embedded as: where it stands (document, heading) and what it says."""

    head = " · ".join(part for part in (title.strip(), heading.strip()) if part)
    return f"{head}\n{chunk_text}" if head else chunk_text


class VectorMatrix:
    """All vectors of one provider id, held in memory for a brute-force cosine search (a semester is a few thousand chunks)."""

    def __init__(self, rowids: list[int], vectors: np.ndarray) -> None:
        self.rowids = np.asarray(rowids, dtype=np.int64)
        self.vectors = vectors.astype(np.float32) if vectors.size else np.zeros((0, 0), dtype=np.float32)
        self.loaded_at = time.time()

    def __len__(self) -> int:
        return int(self.rowids.shape[0])

    def top(self, query: np.ndarray, k: int = 40, *, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if not len(self):
            return []
        sims = self.vectors @ query.astype(np.float32)
        if allowed is not None:
            mask = np.isin(self.rowids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
            sims = np.where(mask, sims, -1.0)
        k = min(k, len(self))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(self.rowids[i]), float(sims[i])) for i in idx if sims[i] > -1.0]

    def similarity_of(self, rowids: list[int], query: np.ndarray) -> dict[int, float]:
        if not len(self) or not rowids:
            return {}
        if not hasattr(self, "_position"):
            self._position = {int(r): i for i, r in enumerate(self.rowids)}
        found = [(r, self._position[r]) for r in rowids if r in self._position]
        if not found:
            return {}
        sims = self.vectors[[i for _, i in found]] @ query.astype(np.float32)
        return {r: float(s) for (r, _), s in zip(found, sims)}
