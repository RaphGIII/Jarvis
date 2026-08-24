"""Generate chess board images with known ground truth.

These are the *requirements* for the guided chess project, not part of its
solution: the user supplies board pictures and the FEN each one really shows,
and Jarvis has to build something that recovers the second from the first.

Rendered rather than photographed on purpose. A synthetic board has exact
ground truth -- every square is where the generator put it -- so the project's
accuracy can be measured instead of eyeballed. Photographs of a real board are
the harder problem and the right *next* fixture set; starting there would mean
the first failure is ambiguous between "the detector is wrong" and "the
photograph is bad".

Two deliberate variations, because a pipeline that only works on one image has
not been shown to work at all:

* a margin around the board, so the code cannot assume the board fills the frame
* a different square size, so it cannot hard-code 64 pixels
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

#: Unicode chess pieces, drawn as text rather than sprites so the fixture
#: generator needs no asset files.
GLYPHS = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)

#: Positions worth analysing, with the FEN each image really shows.
POSITIONS = [
    (
        "starting_position",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "the initial position",
    ),
    (
        "italian_game",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "after 1.e4 e5 2.Nf3 Nc6 3.Bc4",
    ),
    (
        "mate_in_one",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "white mates with Ra8",
    ),
    (
        "endgame_kpk",
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
        "king and pawn against king",
    ),
]


def board_from_fen(fen: str) -> list[list[str]]:
    """Rank 8 first, as the board is drawn."""

    rows = []
    for rank in fen.split()[0].split("/"):
        row: list[str] = []
        for char in rank:
            if char.isdigit():
                row.extend([""] * int(char))
            else:
                row.append(char)
        rows.append(row)
    return rows


def render(fen: str, path: Path, *, square: int = 64, margin: int = 0) -> None:
    size = square * 8 + margin * 2
    image = Image.new("RGB", (size, size), (30, 30, 34))
    draw = ImageDraw.Draw(image)

    for rank in range(8):
        for file in range(8):
            x0 = margin + file * square
            y0 = margin + rank * square
            colour = LIGHT if (rank + file) % 2 == 0 else DARK
            draw.rectangle([x0, y0, x0 + square, y0 + square], fill=colour)

    from PIL import ImageFont

    try:
        # A font with chess glyphs. Segoe UI Symbol ships with Windows.
        font = ImageFont.truetype("seguisym.ttf", int(square * 0.72))
    except OSError:
        font = ImageFont.load_default()

    for rank, row in enumerate(board_from_fen(fen)):
        for file, piece in enumerate(row):
            if not piece:
                continue
            glyph = GLYPHS[piece]
            x = margin + file * square + square // 2
            y = margin + rank * square + square // 2
            colour = (250, 250, 250) if piece.isupper() else (20, 20, 20)
            draw.text((x, y), glyph, font=font, fill=colour, anchor="mm")

    image.save(path)


def main() -> int:
    here = Path(__file__).resolve().parent
    manifest = []

    for index, (name, fen, description) in enumerate(POSITIONS):
        # Vary the geometry so a solution cannot hard-code it.
        square = 64 if index % 2 == 0 else 80
        margin = 0 if index % 2 == 0 else 24
        filename = f"{name}.png"
        render(fen, here / filename, square=square, margin=margin)
        manifest.append(
            {
                "image": filename,
                "fen": fen,
                "description": description,
                "square_pixels": square,
                "margin_pixels": margin,
            }
        )
        print(f"wrote {filename}  ({square}px squares, {margin}px margin)")

    (here / "fixtures.json").write_text(json.dumps({"positions": manifest}, indent=2), encoding="utf-8")
    print(f"wrote fixtures.json with {len(manifest)} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
