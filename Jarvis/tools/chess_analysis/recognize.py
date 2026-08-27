"""Pieces on the board: the owner's trained YOLO detector, on the CPU.

The Chessaru project (D:\\Chessaru) trained a small YOLO on the owner's own
screen boards; its weights are the most reliable recogniser available for
this machine, so this module uses them unchanged.  Inference is pinned to the
CPU on purpose: the GPU holds the conversation model, and a second model on it
evicts that one for the next chat turn (measured earlier at ~28 s).  A 640 px
board crop takes well under a second on the CPU, which is enough for a game.

Each detection's centre lands in one of the 8x8 cells; the most confident
detection per cell wins.  The result is a screen-oriented grid (row 0 = top
of the screen) of FEN letters, which :mod:`position` turns into a FEN.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .position import CLASS_TO_FEN, Grid

DEFAULT_MODEL_CANDIDATES = (
    Path(r"D:\Chessaru\runs\detect\chess_pieces_orientation_boosted\weights\best.pt"),
    Path(r"D:\Chessaru\runs\detect\chess_pieces_black_view_final\weights\best.pt"),
    Path(r"D:\Chessaru\runs\detect\chess_pieces_user\weights\best.pt"),
)


def default_model_path() -> Path | None:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


class YoloRecognizer:
    def __init__(self, model_path: str | Path, *, confidence: float = 0.15, image_size: int = 640, tta: bool = True) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size
        #: Test-time augmentation (flips/scales, averaged).  Measured on the
        #: owner's start position on screen: without it both kings were
        #: missed or misread (white king empty, black king "n"); with it they
        #: score 0.80 / 0.77 in the right cells.  Costs ~3x CPU per frame.
        self.tta = tta
        self._model: Any = None
        self.last_detections: list[dict[str, Any]] = []

    def _load(self) -> Any:
        if self._model is None:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(self.model_path.parent))
            os.environ.setdefault("MPLCONFIGDIR", str(self.model_path.parent))
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, board_bgr: np.ndarray) -> Grid:
        """Screen-oriented 8x8 grid of FEN letters (None = empty) for a square board crop."""

        model = self._load()
        results = model.predict(board_bgr, device="cpu", verbose=False, conf=self.confidence, imgsz=self.image_size, augment=self.tta)
        names = model.names
        h, w = board_bgr.shape[:2]
        grid: Grid = [[None] * 8 for _ in range(8)]
        best = [[-1.0] * 8 for _ in range(8)]
        detections = []
        for box in results[0].boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            piece = CLASS_TO_FEN.get(str(names[int(box.cls[0].item())]))
            if piece is None:
                continue
            conf = float(box.conf[0].item())
            col = min(7, max(0, int(cx / (w / 8))))
            row = min(7, max(0, int(cy / (h / 8))))
            detections.append({"piece": piece, "col": col, "row": row, "confidence": round(conf, 3)})
            if conf > best[row][col]:
                best[row][col] = conf
                grid[row][col] = piece
        self.last_detections = detections
        return grid


def grid_from_detections(detections: list[dict[str, Any]]) -> Grid:
    """The same cell assignment for already computed detections (tests, replays)."""

    grid: Grid = [[None] * 8 for _ in range(8)]
    best = [[-1.0] * 8 for _ in range(8)]
    for d in detections:
        r, c, conf = int(d["row"]), int(d["col"]), float(d.get("confidence", 1.0))
        if conf > best[r][c]:
            best[r][c] = conf
            grid[r][c] = d["piece"]
    return grid


class PositionSmoother:
    """Per-cell majority over the last ``window`` frames, weighted by confidence.

    A live board is seen again every fraction of a second; a piece that one
    frame misses and the next two see is a piece.  Votes are keyed by the
    board rectangle so a moved window starts fresh.
    """

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self.frames: list[tuple[Grid, list[dict[str, Any]]]] = []

    def reset(self) -> None:
        self.frames.clear()

    def push(self, grid: Grid, detections: list[dict[str, Any]]) -> Grid:
        self.frames.append((grid, detections))
        del self.frames[:-self.window]
        conf = {}
        for _grid, dets in self.frames:
            for d in dets:
                conf[(d["row"], d["col"], d["piece"])] = conf.get((d["row"], d["col"], d["piece"]), 0.0) + float(d.get("confidence", 1.0))
        out: Grid = [[None] * 8 for _ in range(8)]
        n = len(self.frames)
        for r in range(8):
            for c in range(8):
                votes = {p: v for (rr, cc, p), v in conf.items() if rr == r and cc == c}
                if not votes:
                    continue
                piece, weight = max(votes.items(), key=lambda kv: kv[1])
                seen = sum(1 for g, _ in self.frames if g[r][c] == piece)
                # a piece needs to be seen in more than half the frames, or
                # with a strong single confidence when the window is young
                if seen * 2 > n or (n < self.window and weight >= 0.5):
                    out[r][c] = piece
        return out
