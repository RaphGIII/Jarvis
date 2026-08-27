"""Schach Analyse: board finding, position/side logic, the recogniser and the engine.

The finder is tested on a rendered board pasted into a screen-sized image;
the recogniser on the owner's own labelled board screenshots when they exist
(D:\\Chessaru); the engine on the local Stockfish when it exists.  Anything
that needs a file outside the repository skips honestly instead of failing.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from tools.chess_analysis.board import BoardRect, chequer_contrast, find_board, validate
from tools.chess_analysis.engine import Analyzer, default_stockfish_path
from tools.chess_analysis.overlay import format_lines, side_text
from tools.chess_analysis.position import (START_FEN_BOARD, SideTracker, detect_orientation, full_fen, grid_to_board_fen,
                                           last_mover, plausible, side_from_check)
from tools.chess_analysis.recognize import default_model_path, grid_from_detections

cv2 = pytest.importorskip("cv2")


def render_board(size=560, origin=(300, 120), screen=(1080, 1920), light=(215, 230, 240), dark=(90, 130, 170)):
    """A screen with a chequered board and a few blobs for pieces."""

    image = np.full((screen[0], screen[1], 3), (40, 40, 40), dtype=np.uint8)
    # some distractors: window chrome and a text block
    cv2.rectangle(image, (0, 0), (screen[1], 40), (70, 70, 70), -1)
    cv2.rectangle(image, (1000, 200), (1800, 900), (245, 245, 245), -1)
    for i in range(20):
        cv2.line(image, (1020, 230 + i * 30), (1780, 230 + i * 30), (200, 200, 200), 1)
    s = size // 8
    for r in range(8):
        for c in range(8):
            colour = light if (r + c) % 2 == 0 else dark
            cv2.rectangle(image, (origin[0] + c * s, origin[1] + r * s), (origin[0] + (c + 1) * s, origin[1] + (r + 1) * s), colour, -1)
    for r, c in ((0, 0), (0, 4), (7, 4), (7, 7), (3, 3)):
        cv2.circle(image, (origin[0] + c * s + s // 2, origin[1] + r * s + s // 2), s // 3, (20, 20, 20) if r < 4 else (250, 250, 250), -1)
    return image


# --------------------------------------------------------------------------
# Board finder
# --------------------------------------------------------------------------

def test_the_board_is_found_on_a_screen_with_distractors():
    image = render_board()
    rect = find_board(image)

    assert rect is not None
    assert abs(rect.x - 300) <= 6 and abs(rect.y - 120) <= 6 and abs(rect.size - 560) <= 12
    assert rect.confidence > 0.5
    assert validate(image, rect)


def test_no_board_means_none():
    image = np.full((600, 800, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (500, 400), (200, 200, 200), -1)
    assert find_board(image) is None


def test_a_spreadsheet_grid_is_not_a_board():
    image = np.full((900, 1200, 3), 250, dtype=np.uint8)
    for i in range(0, 700, 40):
        cv2.line(image, (100 + i, 100), (100 + i, 740), (180, 180, 180), 1)
        cv2.line(image, (100, 100 + i), (740, 100 + i), (180, 180, 180), 1)
    rect = find_board(image)
    assert rect is None, "evenly spaced lines without alternating cells are not a chessboard"


def test_chequer_contrast_distinguishes_a_board_from_a_flat_area():
    image = render_board()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert chequer_contrast(gray, BoardRect(300, 120, 560)) > 40
    assert chequer_contrast(gray, BoardRect(1100, 300, 400)) < 12


# --------------------------------------------------------------------------
# Position, orientation, side to move
# --------------------------------------------------------------------------

def start_grid(orientation="white"):
    rows = ["rnbqkbnr", "pppppppp", "", "", "", "", "PPPPPPPP", "RNBQKBNR"]
    grid = [[(row[c] if row else None) for c in range(8)] for row in rows]
    if orientation == "black":
        grid = [list(reversed(r)) for r in reversed(grid)]
    return grid


def test_the_start_position_becomes_the_standard_fen_in_both_orientations():
    assert grid_to_board_fen(start_grid("white"), "white") == START_FEN_BOARD
    assert detect_orientation(start_grid("white")) == "white"
    assert detect_orientation(start_grid("black")) == "black"
    assert grid_to_board_fen(start_grid("black"), "black") == START_FEN_BOARD


def test_detections_map_to_cells_by_confidence():
    grid = grid_from_detections([{"piece": "K", "row": 7, "col": 4, "confidence": 0.9}, {"piece": "Q", "row": 7, "col": 4, "confidence": 0.5},
                                 {"piece": "k", "row": 0, "col": 4, "confidence": 0.8}])
    assert grid[7][4] == "K" and grid[0][4] == "k"
    assert plausible(grid_to_board_fen(grid))[0]


def test_implausible_positions_are_named():
    ok, why = plausible("8/8/8/8/8/8/8/8")
    assert not ok and "king" in why


def test_the_side_in_check_must_be_the_side_to_move():
    # white king on e1 attacked by a black rook on e8: white to move
    assert side_from_check("4r2k/8/8/8/8/8/8/4K3") == "w"
    assert side_from_check("4k3/8/8/8/8/8/8/4R2K") == "b"
    assert side_from_check(START_FEN_BOARD) is None


def test_the_last_mover_is_read_from_one_legal_move():
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    assert last_mover(START_FEN_BOARD, after_e4) == "w"
    after_e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    assert last_mover(after_e4, after_e5) == "b"
    assert last_mover(START_FEN_BOARD, after_e5) is None, "two moves apart: no single mover"


def test_the_tracker_follows_a_game_and_explains_itself():
    tracker = SideTracker()
    assert tracker.update(START_FEN_BOARD) == ("w", "prior: start position")
    side, source = tracker.update("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    assert side == "b" and source.startswith("motion: white moved last")
    side, source = tracker.update("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR")
    assert side == "w" and "black moved last" in source
    # a check overrides everything
    side, source = tracker.update("4r2k/8/8/8/8/8/8/4K3")
    assert side == "w" and source.startswith("legality")
    # the owner's override holds while nothing proves otherwise
    tracker.update("8/8/8/8/8/3k4/8/4K3")
    tracker.flip()
    assert tracker.side in {"w", "b"} and tracker.source == "owner override"


def test_full_fen_claims_castling_only_for_pieces_at_home():
    assert full_fen(START_FEN_BOARD, "w") == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert full_fen("4k3/8/8/8/8/8/8/4K3", "b").split()[2] == "-"


def test_temporal_voting_removes_single_frame_flicker():
    from tools.chess_analysis.recognize import PositionSmoother

    smoother = PositionSmoother(window=3)
    seen = [{"piece": "K", "row": 7, "col": 4, "confidence": 0.9}]
    grid_with = grid_from_detections(seen)
    smoother.push(grid_with, seen)
    smoother.push(grid_with, seen)
    flicker = smoother.push([[None] * 8 for _ in range(8)], [])      # one frame misses the king
    assert flicker[7][4] == "K"
    gone = smoother.push([[None] * 8 for _ in range(8)], [])
    assert gone[7][4] is None, "two of three frames without it: the piece is gone"


def test_overlay_text_is_human():
    lines = format_lines([{"rank": 1, "san": "e4", "score": "+0.30", "pv": "1. e4 e5 2. Nf3"}])
    assert lines[0].startswith("1. e4") and "+0.30" in lines[0]
    assert "Weiß am Zug" in side_text("w", "prior: start position", "white")


# --------------------------------------------------------------------------
# The engine and the recogniser: real local resources when present
# --------------------------------------------------------------------------

@pytest.mark.skipif(default_stockfish_path() is None, reason="no local Stockfish")
def test_stockfish_returns_five_ranked_lines():
    analyzer = Analyzer(default_stockfish_path())
    try:
        lines = analyzer.best_moves(full_fen(START_FEN_BOARD, "w"), count=5, seconds=0.3)
    finally:
        analyzer.close()
    assert len(lines) == 5 and [l.rank for l in lines] == [1, 2, 3, 4, 5]
    assert all(l.san and l.score for l in lines)
    assert lines[0].san in {"e4", "d4", "Nf3", "c4", "e3", "g3", "Nc3"}


def _labelled_boards():
    root = Path(r"D:\Chessaru\datasets\chess_pieces_user")
    out = []
    for image in sorted(glob.glob(str(root / "images" / "val" / "*.*")))[:3]:
        label = root / "labels" / "val" / (Path(image).stem + ".txt")
        if label.is_file():
            out.append((image, label))
    return out


@pytest.mark.skipif(default_model_path() is None or not _labelled_boards(), reason="no owner piece model / labelled boards")
def test_the_owner_model_recognises_the_owners_boards():
    from tools.chess_analysis.recognize import YoloRecognizer

    names = ["white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
             "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"]
    from tools.chess_analysis.position import CLASS_TO_FEN

    recognizer = YoloRecognizer(default_model_path())
    agreements, cells = 0, 0
    for image_path, label_path in _labelled_boards():
        image = cv2.imread(image_path)
        expected = [[None] * 8 for _ in range(8)]
        for line in Path(label_path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy = int(parts[0]), float(parts[1]), float(parts[2])
            expected[min(7, int(cy * 8))][min(7, int(cx * 8))] = CLASS_TO_FEN[names[cls]]
        got = recognizer.detect(image)
        for r in range(8):
            for c in range(8):
                cells += 1
                agreements += got[r][c] == expected[r][c]
    assert cells and agreements / cells >= 0.9, f"{agreements}/{cells} cells agree (measured 179/192 on these augmented boards)"
