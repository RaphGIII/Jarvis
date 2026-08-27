"""Find a chessboard on a screen image.

No manual region: the board is found from what a board *is* -- a large square
region cut by nine equally spaced vertical and nine equally spaced horizontal
edges (every square boundary is a colour edge), whose 8x8 cells alternate in
brightness.  Both facts are checked; either alone is fooled by spreadsheets
(lines, no chequer) or photos (chequer-ish, no grid).

Returns a :class:`BoardRect` in screen coordinates plus a confidence.  The
caller caches the rectangle and re-validates it cheaply (:func:`validate`)
every frame, so the expensive search runs only when the board moved or
disappeared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A board smaller than this (in pixels) has squares too small to classify.
MIN_SIZE = 200
#: Tolerance on the position of an expected grid line, as a fraction of the square size.
LINE_TOLERANCE = 0.18


@dataclass(frozen=True)
class BoardRect:
    x: int
    y: int
    size: int
    confidence: float = 0.0

    @property
    def square(self) -> float:
        return self.size / 8.0

    def cell(self, col: int, row: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of screen cell ``col`` (0 = left), ``row`` (0 = top)."""

        s = self.square
        return (int(self.x + col * s), int(self.y + row * s), int(self.x + (col + 1) * s), int(self.y + (row + 1) * s))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "size": self.size, "confidence": round(self.confidence, 3)}


def _cluster(values: np.ndarray, gap: float) -> list[float]:
    """Sorted 1-D values merged when closer than ``gap``; returns cluster centres."""

    if values.size == 0:
        return []
    values = np.sort(values)
    groups: list[list[float]] = [[float(values[0])]]
    for v in values[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(float(v))
        else:
            groups.append([float(v)])
    return [float(np.mean(g)) for g in groups]


def _grids(positions: list[float], min_square: float) -> list[tuple[float, float, int]]:
    """Every (start, spacing, matched) with >= 7 of 9 evenly spaced lines present.

    ``start`` and ``spacing`` are refitted to the lines that actually matched
    (least squares), not taken from the first line and one difference: a
    2-3 px error in the spacing is 16-24 px at the far edge of the board,
    enough to push edge pieces into the wrong cell.
    """

    out = []
    pos = np.array(positions, dtype=np.float64)
    if pos.size < 7:
        return out
    diffs = np.diff(pos)
    candidates = sorted({round(float(d), 1) for d in diffs if d >= min_square})
    seen: set[tuple[int, int]] = set()
    for spacing in candidates:
        tol = spacing * LINE_TOLERANCE
        for start in pos:
            expected = start + spacing * np.arange(9)
            hits = [(k, float(pos[np.argmin(np.abs(pos - e))])) for k, e in enumerate(expected) if np.any(np.abs(pos - e) <= tol)]
            if len(hits) < 7:
                continue
            ks = np.array([k for k, _ in hits], dtype=np.float64)
            vs = np.array([v for _, v in hits], dtype=np.float64)
            fitted_spacing, fitted_start = np.polyfit(ks, vs, 1)
            key = (int(round(fitted_start)), int(round(fitted_spacing * 8)))
            if key in seen:
                continue
            seen.add(key)
            out.append((float(fitted_start), float(fitted_spacing), len(hits)))
    return out


def chequer_contrast(gray: np.ndarray, rect: BoardRect) -> float:
    """How strongly the 8x8 cells alternate: |mean(light cells) - mean(dark cells)| in the cell centres."""

    means = np.zeros((8, 8), dtype=np.float64)
    s = rect.square
    for r in range(8):
        for c in range(8):
            x0, y0, x1, y1 = rect.cell(c, r)
            # the centre third of the cell: pieces sit there too, but the
            # alternation still dominates over 64 cells
            cx0, cx1 = int(x0 + s * 0.35), int(x1 - s * 0.35)
            cy0, cy1 = int(y0 + s * 0.35), int(y1 - s * 0.35)
            patch = gray[max(0, cy0):max(cy0 + 1, cy1), max(0, cx0):max(cx0 + 1, cx1)]
            means[r, c] = float(patch.mean()) if patch.size else 0.0
    parity = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0
    return abs(float(means[parity].mean()) - float(means[~parity].mean()))


def outside_alternation(gray: np.ndarray, rect: BoardRect) -> float:
    """How much the strips just outside the rectangle keep alternating.

    A real board ends at its edge: the strip above, below, left and right is
    background.  A candidate shifted by one square has real board squares
    on one side, alternating with the same period -- this is the number that
    tells the two apart when both match nine lines and both chequer inside.
    """

    s = rect.square
    h, w = gray.shape[:2]
    scores = []
    for side in ("top", "bottom", "left", "right"):
        vals = []
        for k in range(8):
            if side in ("top", "bottom"):
                x0 = int(rect.x + k * s + s * 0.35); x1 = int(rect.x + (k + 1) * s - s * 0.35)
                y0 = int(rect.y - s + s * 0.35) if side == "top" else int(rect.y + rect.size + s * 0.35)
                y1 = int(rect.y - s * 0.35) if side == "top" else int(rect.y + rect.size + s - s * 0.35)
                parity = (k + (-1 if side == "top" else 8)) % 2
            else:
                y0 = int(rect.y + k * s + s * 0.35); y1 = int(rect.y + (k + 1) * s - s * 0.35)
                x0 = int(rect.x - s + s * 0.35) if side == "left" else int(rect.x + rect.size + s * 0.35)
                x1 = int(rect.x - s * 0.35) if side == "left" else int(rect.x + rect.size + s - s * 0.35)
                parity = (k + (-1 if side == "left" else 8)) % 2
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                vals = []
                break
            vals.append((parity, float(gray[y0:y1, x0:x1].mean())))
        if len(vals) == 8:
            even = [v for p_, v in vals if p_ == 0]
            odd = [v for p_, v in vals if p_ == 1]
            scores.append(abs(sum(even) / len(even) - sum(odd) / len(odd)))
    return max(scores) if scores else 0.0


def find_board(image_bgr: np.ndarray) -> BoardRect | None:
    """The most board-like square on the image, or None."""

    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 60, 160)
    min_len = int(max(MIN_SIZE * 0.6, min(h, w) * 0.12))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=min_len, maxLineGap=6)
    if lines is None:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        if abs(x1 - x2) <= 2 and abs(y2 - y1) >= min_len:
            xs.append((x1 + x2) / 2.0)
        elif abs(y1 - y2) <= 2 and abs(x2 - x1) >= min_len:
            ys.append((y1 + y2) / 2.0)
    vx = _cluster(np.array(xs), 4.0)
    hy = _cluster(np.array(ys), 4.0)
    min_square = MIN_SIZE / 8.0
    best: BoardRect | None = None
    best_key: tuple = ()
    for x0, sx, mx in _grids(vx, min_square):
        for y0, sy, my in _grids(hy, min_square):
            if abs(sx - sy) > 0.08 * max(sx, sy):
                continue
            size = int(round((sx + sy) / 2 * 8))
            rect = BoardRect(int(round(x0)), int(round(y0)), size)
            if rect.x < -2 or rect.y < -2 or rect.x + size > w + 2 or rect.y + size > h + 2:
                continue
            contrast = chequer_contrast(gray, rect)
            if contrast < 12:
                continue
            confidence = min(1.0, (mx + my) / 18.0) * min(1.0, contrast / 60.0)
            candidate = BoardRect(max(0, rect.x), max(0, rect.y), size, confidence)
            # More matched grid lines first: a candidate shifted by one square
            # matches 8 of 9 lines and still alternates, but never all 9.
            # Then the stronger chequer (a shifted candidate includes a row
            # of background), then size.
            outside = outside_alternation(gray, candidate)
            key = (mx + my, -round(outside / 10), contrast, size)
            if best is None or key > best_key:
                best, best_key = candidate, key
    return best


def validate(image_bgr: np.ndarray, rect: BoardRect) -> bool:
    """Cheap per-frame check that the cached rectangle still holds a board."""

    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape[:2]
    if rect.x + rect.size > w or rect.y + rect.size > h or rect.size < MIN_SIZE:
        return False
    return chequer_contrast(gray, rect) >= 12


def crop(image_bgr: np.ndarray, rect: BoardRect) -> np.ndarray:
    return image_bgr[rect.y:rect.y + rect.size, rect.x:rect.x + rect.size]
