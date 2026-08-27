"""Stockfish over UCI (python-chess): the five best moves for a FEN.

One engine process for the tool's lifetime; MultiPV 5; a bounded think time
so the overlay follows a live game.  Scores are reported from the side to
move's point of view as the engine gives them, converted to a human string
("+0.35", "M3").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

DEFAULT_STOCKFISH_CANDIDATES = (
    Path(r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"),
    Path(r"D:\stockfish\stockfish.exe"),
)


def default_stockfish_path() -> Path | None:
    for candidate in DEFAULT_STOCKFISH_CANDIDATES:
        if candidate.is_file():
            return candidate
    import shutil

    found = shutil.which("stockfish")
    return Path(found) if found else None


@dataclass
class Line:
    rank: int
    move_uci: str
    move_san: str
    score: str
    centipawns: int | None
    mate: int | None
    depth: int
    pv: str

    @property
    def san(self) -> str:
        return self.move_san

    @property
    def uci(self) -> str:
        return self.move_uci

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "uci": self.move_uci, "san": self.move_san, "score": self.score, "cp": self.centipawns,
                "mate": self.mate, "depth": self.depth, "pv": self.pv}


def format_score(score: chess.engine.PovScore) -> tuple[str, int | None, int | None]:
    pov = score.pov(score.turn)
    if pov.is_mate():
        m = pov.mate()
        return (f"M{m}" if m and m > 0 else f"-M{abs(m or 0)}"), None, m
    cp = pov.score(mate_score=100000)
    return f"{(cp or 0) / 100:+.2f}", cp, None


class Analyzer:
    def __init__(self, stockfish_path: str | Path, *, threads: int = 2, hash_mb: int = 128) -> None:
        self.path = Path(stockfish_path)
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: chess.engine.SimpleEngine | None = None

    def _open(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
            try:
                self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
            except chess.engine.EngineError:
                pass
        return self._engine

    def best_moves(self, fen: str, *, count: int = 5, seconds: float = 0.8) -> list[Line]:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"invalid position: {board.status()!r}")
        engine = self._open()
        infos = engine.analyse(board, chess.engine.Limit(time=seconds), multipv=count)
        lines: list[Line] = []
        for index, info in enumerate(infos if isinstance(infos, list) else [infos], start=1):
            pv = info.get("pv") or []
            if not pv:
                continue
            text, cp, mate = format_score(info["score"])
            san = board.san(pv[0])
            pv_san = board.variation_san(pv[:6])
            lines.append(Line(index, pv[0].uci(), san, text, cp, mate, int(info.get("depth", 0)), pv_san))
        return lines

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001
                pass
            self._engine = None
