"""Letting a text model find out what is in an image.

This exists because of a specific, instructive failure. Asked to reconstruct a
chess position from a board image, BUILD_LOCAL wrote:

    def position(path) -> str:
        # Placeholder implementation
        return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'

The obvious reading is that the model gave up. The truer one is that it had no
choice: it is a text model, it had never seen the image, and it was being asked
to write pixel-recognition code for glyphs it could not look at. Guessing the
most common position was the only move available.

The architecture already has an answer for this shape of problem -- deterministic
software finds the facts, the model reasons about them -- and it was simply
missing for images. These tools close that: measure a region, summarise a grid,
compare two crops. The model still writes the recognition code, but now it can
*check its assumptions against the actual pixels* the way it checks a file's
contents with read_file.

Deliberately statistical rather than a vision model. A description from a VLM
would be prose about a picture, which is exactly the kind of unverifiable input
the rest of this system refuses; a mean colour and an ink fraction are numbers,
and code written against numbers can be tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.registry import RiskLevel, ToolContext, ToolSpec

#: Above this fraction of non-background pixels, a cell has something in it.
DEFAULT_INK_THRESHOLD = 0.02


def _open(path: str, context: ToolContext) -> Any:
    from PIL import Image

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path(context.workspace) / target
    if not target.is_file():
        raise FileNotFoundError(f"no such image: {target}")
    return Image.open(target).convert("RGB")


def image_info(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Size and overall colour makeup of an image."""

    try:
        image = _open(str(payload.get("path", "")), context)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    colours = image.getcolors(maxcolors=1_000_000) or []
    colours.sort(key=lambda pair: -pair[0])
    return {
        "ok": True,
        "width": image.width,
        "height": image.height,
        "distinct_colours": len(colours),
        "dominant": [
            {"rgb": list(rgb), "fraction": round(count / (image.width * image.height), 4)}
            for count, rgb in colours[:6]
        ],
    }


def image_region(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Describe one rectangle: its colours, and how much is not background.

    ``ink_fraction`` is the useful number for "is there something here": on a
    chess board it separates an empty square from an occupied one without
    knowing anything about chess.
    """

    try:
        image = _open(str(payload.get("path", "")), context)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    try:
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        width = int(payload.get("width", 0)) or image.width
        height = int(payload.get("height", 0)) or image.height
    except (TypeError, ValueError):
        return {"ok": False, "error": "x, y, width and height must be integers"}

    crop = image.crop((x, y, min(x + width, image.width), min(y + height, image.height)))
    if crop.width == 0 or crop.height == 0:
        return {"ok": False, "error": "the region is empty"}

    colours = crop.getcolors(maxcolors=1_000_000) or []
    colours.sort(key=lambda pair: -pair[0])
    total = crop.width * crop.height
    background = colours[0][1] if colours else (0, 0, 0)

    # Anything far from the most common colour counts as ink.
    ink = sum(
        count for count, rgb in colours
        if sum(abs(a - b) for a, b in zip(rgb, background)) > 60
    )
    return {
        "ok": True,
        "region": {"x": x, "y": y, "width": crop.width, "height": crop.height},
        "background_rgb": list(background),
        "ink_fraction": round(ink / total, 4),
        "distinct_colours": len(colours),
        "dominant": [
            {"rgb": list(rgb), "fraction": round(count / total, 4)} for count, rgb in colours[:4]
        ],
        "mean_rgb": [round(sum(c[1][i] * c[0] for c in colours) / total) for i in range(3)],
    }


def image_grid(payload: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Split a region into a grid and summarise every cell.

    General, not chess-specific: any tiled layout -- a board, a sprite sheet, a
    contact sheet, a calendar -- is easier to reason about as a grid of measured
    cells than as a picture.
    """

    try:
        image = _open(str(payload.get("path", "")), context)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    try:
        rows = max(1, int(payload.get("rows", 8)))
        columns = max(1, int(payload.get("columns", 8)))
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        cell = int(payload.get("cell", 0))
    except (TypeError, ValueError):
        return {"ok": False, "error": "rows, columns, x, y and cell must be integers"}

    if not cell:
        cell = min((image.width - x) // columns, (image.height - y) // rows)
    if cell <= 0:
        return {"ok": False, "error": "the grid does not fit inside the image"}

    cells = []
    for row in range(rows):
        for column in range(columns):
            crop = image.crop((
                x + column * cell, y + row * cell,
                x + (column + 1) * cell, y + (row + 1) * cell,
            ))
            colours = crop.getcolors(maxcolors=1_000_000) or []
            colours.sort(key=lambda pair: -pair[0])
            total = max(1, crop.width * crop.height)
            background = colours[0][1] if colours else (0, 0, 0)
            ink = sum(
                count for count, rgb in colours
                if sum(abs(a - b) for a, b in zip(rgb, background)) > 60
            )
            # The mean colour of the ink itself, which is what distinguishes a
            # white piece from a black one on either shade of square.
            ink_pixels = [(count, rgb) for count, rgb in colours
                          if sum(abs(a - b) for a, b in zip(rgb, background)) > 60]
            ink_mean = (
                [round(sum(c * rgb[i] for c, rgb in ink_pixels) / max(1, ink)) for i in range(3)]
                if ink_pixels else None
            )
            cells.append({
                "row": row,
                "column": column,
                "background_rgb": list(background),
                "ink_fraction": round(ink / total, 4),
                "occupied": (ink / total) >= float(payload.get("ink_threshold", DEFAULT_INK_THRESHOLD)),
                "ink_mean_rgb": ink_mean,
            })

    occupied = sum(1 for item in cells if item["occupied"])
    return {
        "ok": True,
        "rows": rows,
        "columns": columns,
        "cell": cell,
        "origin": {"x": x, "y": y},
        "occupied_cells": occupied,
        "cells": cells,
    }


def vision_tools() -> list[ToolSpec]:
    """Reading tools: they measure an image and never change one."""

    return [
        ToolSpec(
            name="image_info",
            purpose="Size and dominant colours of an image file.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            adapter=image_info,
            risk=RiskLevel.SAFE,
            tags=("vision", "investigate"),
            example='{"name": "image_info", "arguments": {"path": "board.png"}}',
        ),
        ToolSpec(
            name="image_region",
            purpose=(
                "Measure one rectangle of an image: its colours and how much of it is not "
                "background. Use it to check whether something is present at a position."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
                "required": ["path"],
            },
            adapter=image_region,
            risk=RiskLevel.SAFE,
            tags=("vision", "investigate"),
            example='{"name": "image_region", "arguments": {"path": "board.png", "x": 0, "y": 0, "width": 64, "height": 64}}',
        ),
        ToolSpec(
            name="image_grid",
            purpose=(
                "Split an image into a grid and measure every cell: which are occupied and "
                "what colour the content is. Use it for any tiled layout."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "rows": {"type": "integer"},
                    "columns": {"type": "integer"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "cell": {"type": "integer"},
                },
                "required": ["path"],
            },
            adapter=image_grid,
            risk=RiskLevel.SAFE,
            tags=("vision", "investigate"),
            example='{"name": "image_grid", "arguments": {"path": "board.png", "rows": 8, "columns": 8}}',
        ),
    ]
