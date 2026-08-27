"""From a recognised 8x8 grid to a FEN, with orientation and side to move.

Orientation: the side whose pieces sit nearer the bottom of the screen is the
player ("white" = white at the bottom).  Cheap and right whenever both
colours are on the board.

Side to move is the hard part -- a screenshot does not say it.  Three
independent sources, in order of trust:

1. **Legality.**  The side *not* to move can never be in check.  If white is
   in check, white is to move (and vice versa).  Deterministic, from
   python-chess, decisive whenever a check is on the board.
2. **Motion.**  The tool watches continuously: when the position changes by
   one legal move from the previous one, the mover is known and the other
   side is to move.  This tracks a live game exactly.
3. **Prior.**  The starting position is white to move; otherwise the last
   known side is kept, and the owner can flip it from the overlay.

Every inference reports its source so the overlay can say "black to move
(moved last: white)" rather than pretending certainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess

START_FEN_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

#: YOLO class name -> FEN letter (the Chessaru model's classes).
CLASS_TO_FEN = {
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B", "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b", "black_rook": "r", "black_queen": "q", "black_king": "k",
}

Grid = list[list[Optional[str]]]  # [screen_row][screen_col] -> FEN letter or None


def detect_orientation(grid: Grid) -> str:
    """'white' when white's pieces are nearer the bottom of the screen, else 'black'."""

    white_rows = [r for r in range(8) for c in range(8) if grid[r][c] and grid[r][c].isupper()]
    black_rows = [r for r in range(8) for c in range(8) if grid[r][c] and grid[r][c].islower()]
    if not white_rows or not black_rows:
        return "white"
    return "white" if sum(white_rows) / len(white_rows) >= sum(black_rows) / len(black_rows) else "black"


def grid_to_board_fen(grid: Grid, orientation: str = "white") -> str:
    """The piece-placement field of a FEN (rank 8 first) from the screen grid."""

    rows = []
    for rank_from_top in range(8):
        if orientation == "black":
            cells = [grid[7 - rank_from_top][7 - c] for c in range(8)]
        else:
            cells = [grid[rank_from_top][c] for c in range(8)]
        text, empty = "", 0
        for cell in cells:
            if cell:
                if empty:
                    text += str(empty)
                    empty = 0
                text += cell
            else:
                empty += 1
        if empty:
            text += str(empty)
        rows.append(text)
    return "/".join(rows)


def plausible(board_fen: str) -> tuple[bool, str]:
    """A recognised position that could be a chess position: two kings, sane counts."""

    try:
        board = chess.Board(board_fen + " w - - 0 1")
    except ValueError as exc:
        return False, f"unreadable: {exc}"
    if len(board.pieces(chess.KING, chess.WHITE)) != 1 or len(board.pieces(chess.KING, chess.BLACK)) != 1:
        return False, "not exactly one king per side"
    for colour, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        if len(board.pieces(chess.PAWN, colour)) > 8:
            return False, f"{name} has more than eight pawns"
        if len(chess.SquareSet(board.occupied_co[colour])) > 16:
            return False, f"{name} has more than sixteen pieces"
    return True, ""


def side_from_check(board_fen: str) -> str | None:
    """The side that must be to move because the other is in check; None when no check."""

    white_board = chess.Board(board_fen + " w - - 0 1")
    black_board = chess.Board(board_fen + " b - - 0 1")
    white_in_check = white_board.is_check()          # white to move and in check: fine
    black_in_check = black_board.is_check()
    if white_in_check and not black_in_check:
        return "w"
    if black_in_check and not white_in_check:
        return "b"
    return None


def last_mover(previous_fen: str, current_fen: str) -> str | None:
    """Which colour moved between two piece placements, if exactly one legal move explains it."""

    for side in ("w", "b"):
        try:
            board = chess.Board(f"{previous_fen} {side} KQkq - 0 1")
        except ValueError:
            continue
        for move in board.legal_moves:
            board.push(move)
            same = board.board_fen() == current_fen
            board.pop()
            if same:
                return side
    return None


@dataclass
class SideTracker:
    """Keeps the side to move across frames; explains every answer."""

    side: str = "w"
    source: str = "prior: start position"
    previous_board_fen: str = ""
    override: str | None = None
    history: list[str] = field(default_factory=list)

    def flip(self) -> str:
        self.override = "b" if self.side == "w" else "w"
        self.side, self.source = self.override, "owner override"
        return self.side

    def update(self, board_fen: str) -> tuple[str, str]:
        if board_fen == self.previous_board_fen:
            return self.side, self.source
        forced = side_from_check(board_fen)
        mover = last_mover(self.previous_board_fen, board_fen) if self.previous_board_fen else None
        if forced is not None:
            self.side, self.source = forced, "legality: the other side is in check"
            self.override = None
        elif mover is not None:
            self.side, self.source = ("b" if mover == "w" else "w"), f"motion: {'white' if mover == 'w' else 'black'} moved last"
            self.override = None
        elif board_fen == START_FEN_BOARD:
            self.side, self.source = "w", "prior: start position"
            self.override = None
        elif self.override is not None:
            self.side, self.source = self.override, "owner override"
        else:
            self.source = f"kept: {self.source.split(':')[0]} (no evidence in this change)"
        self.previous_board_fen = board_fen
        self.history.append(f"{self.side}:{self.source}")
        del self.history[:-20]
        return self.side, self.source


def full_fen(board_fen: str, side: str) -> str:
    """A complete FEN.  Castling rights are unknowable from a picture; they are
    claimed only for kings and rooks still on their home squares."""

    board = chess.Board(board_fen + " w - - 0 1")
    rights = ""
    if board.piece_at(chess.E1) == chess.Piece.from_symbol("K"):
        if board.piece_at(chess.H1) == chess.Piece.from_symbol("R"):
            rights += "K"
        if board.piece_at(chess.A1) == chess.Piece.from_symbol("R"):
            rights += "Q"
    if board.piece_at(chess.E8) == chess.Piece.from_symbol("k"):
        if board.piece_at(chess.H8) == chess.Piece.from_symbol("r"):
            rights += "k"
        if board.piece_at(chess.A8) == chess.Piece.from_symbol("r"):
            rights += "q"
    return f"{board_fen} {side} {rights or '-'} - 0 1"
