"""Schach Analyse: watch the screen for a chessboard, analyse it, show the five best moves.

    python -m tools.chess_analysis [--stockfish PATH] [--model PATH] [--interval 0.5] [--status FILE]

Loop (in a worker thread; Tk owns the main thread):

    capture the screen -> find the board (cached; re-found when it disappears)
    -> crop -> recognise pieces (YOLO, CPU) -> grid -> FEN -> side to move
    -> Stockfish MultiPV 5 (only when the position changed) -> overlay + status file

The status file is what ZEUS reads for its own view: board rectangle, FEN,
side (with the reason), the lines, timings, and the last error.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .board import BoardRect, crop, find_board, validate
from .engine import Analyzer, default_stockfish_path
from .overlay import OverlayWindow, format_lines, side_text
from .position import SideTracker, detect_orientation, full_fen, grid_to_board_fen, plausible
from .recognize import PositionSmoother, YoloRecognizer, default_model_path


def capture_screen() -> np.ndarray:
    import mss

    with mss.mss() as sct:
        monitor = sct.monitors[0]  # the whole virtual screen
        shot = sct.grab(monitor)
        frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(frame), (monitor["left"], monitor["top"])


class AnalysisLoop(threading.Thread):
    def __init__(self, args: argparse.Namespace, updates: "queue.Queue[dict[str, Any]]") -> None:
        super().__init__(daemon=True, name="chess-analysis")
        self.args = args
        self.updates = updates
        self.stop = threading.Event()
        self.tracker = SideTracker()
        self.status: dict[str, Any] = {"running": True, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
                                       "board": None, "fen": "", "side": "", "side_source": "", "orientation": "", "lines": [],
                                       "frames": 0, "analyses": 0, "last_error": "", "recognizer": str(args.model), "stockfish": str(args.stockfish)}
        self.rect: BoardRect | None = None
        self.smoother = PositionSmoother(window=3)
        self._last_board_fen = ""
        self.flip_requested = threading.Event()

    def write_status(self) -> None:
        try:
            path = Path(self.args.status)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**self.status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    def run(self) -> None:
        recognizer = YoloRecognizer(self.args.model, confidence=self.args.confidence)
        analyzer = Analyzer(self.args.stockfish)
        self.updates.put({"head": "Schach Analyse — suche Brett…", "status": "Bildschirm wird beobachtet"})
        try:
            while not self.stop.is_set():
                started = time.perf_counter()
                try:
                    self.step(recognizer, analyzer)
                    self.status["last_error"] = ""
                except Exception as exc:  # noqa: BLE001 - one bad frame must not end the tool
                    self.status["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
                    self.updates.put({"head": "Schach Analyse — Fehler", "status": self.status["last_error"]})
                self.status["frames"] += 1
                self.status["frame_ms"] = int((time.perf_counter() - started) * 1000)
                self.write_status()
                self.stop.wait(max(0.05, self.args.interval - (time.perf_counter() - started)))
        finally:
            analyzer.close()
            self.status["running"] = False
            self.write_status()

    def step(self, recognizer: YoloRecognizer, analyzer: Analyzer) -> None:
        frame, origin = capture_screen()
        if self.rect is None or not validate(frame, self.rect):
            self.rect = find_board(frame)
            self.smoother.reset()
            self.status["board"] = self.rect.to_dict() if self.rect else None
            if self.rect is None:
                self.updates.put({"head": "Schach Analyse — kein Brett sichtbar", "side_text": "", "lines": [], "status": f"Bild {self.status['frames']}: kein 8×8-Brett gefunden"})
                return
        board_img = crop(frame, self.rect)
        grid = self.smoother.push(recognizer.detect(board_img), recognizer.last_detections)
        orientation = detect_orientation(grid)
        board_fen = grid_to_board_fen(grid, orientation)
        ok, why = plausible(board_fen)
        if not ok:
            self.updates.put({"head": "Schach Analyse — Stellung unklar", "status": f"{why} (Erkennung: {len(recognizer.last_detections)} Figuren)"})
            self.status.update({"fen": "", "orientation": orientation})
            return
        if self.flip_requested.is_set():
            self.flip_requested.clear()
            self.tracker.flip()
            self._last_board_fen = ""  # force a re-analysis with the flipped side
        side, source = self.tracker.update(board_fen)
        fen = full_fen(board_fen, side)
        self.status.update({"orientation": orientation, "side": side, "side_source": source, "fen": fen, "detections": len(recognizer.last_detections)})
        if board_fen == self._last_board_fen and self.status.get("lines"):
            return
        self._last_board_fen = board_fen
        t = time.perf_counter()
        lines = [l.to_dict() for l in analyzer.best_moves(fen, count=5, seconds=self.args.think)]
        self.status["lines"] = lines
        self.status["analyses"] += 1
        self.status["engine_ms"] = int((time.perf_counter() - t) * 1000)
        self.updates.put({"head": f"Schach Analyse — Brett {self.rect.size}px", "side_text": side_text(side, source, orientation),
                          "lines": format_lines(lines), "status": f"FEN {board_fen}  ·  Tiefe {lines[0]['depth'] if lines else '?'}  ·  {self.status['engine_ms']} ms"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch the screen for a chessboard and show Stockfish's five best moves")
    parser.add_argument("--stockfish", default=str(default_stockfish_path() or ""))
    parser.add_argument("--model", default=str(default_model_path() or ""))
    parser.add_argument("--interval", type=float, default=0.6, help="seconds between screen captures")
    parser.add_argument("--think", type=float, default=0.8, help="engine time per position in seconds")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--status", default=str(Path(__file__).resolve().parents[2] / "data" / "jarvis" / "tools" / "chess_analysis.json"))
    parser.add_argument("--headless", action="store_true", help="no overlay window (tests)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (tests)")
    args = parser.parse_args(argv)
    if not args.stockfish or not Path(args.stockfish).is_file():
        print("no Stockfish executable found; pass --stockfish", file=sys.stderr)
        return 2
    if not args.model or not Path(args.model).is_file():
        print("no YOLO piece model found; pass --model", file=sys.stderr)
        return 2
    updates: "queue.Queue[dict[str, Any]]" = queue.Queue()
    loop = AnalysisLoop(args, updates)
    loop.start()
    if args.headless:
        deadline = time.time() + 600
        while loop.is_alive() and time.time() < deadline and (not args.max_frames or loop.status["frames"] < args.max_frames):
            time.sleep(0.2)
        loop.stop.set()
        loop.join(timeout=10)
        print(json.dumps({k: v for k, v in loop.status.items() if k != "lines"}))
        return 0
    window = OverlayWindow(updates, on_flip=loop.flip_requested.set, on_quit=lambda: (loop.stop.set(), updates.put({"quit": True})))
    try:
        window.run()
    finally:
        loop.stop.set()
        loop.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
