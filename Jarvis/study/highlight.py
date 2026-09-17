"""Highlight geometry: every match on a page, as few calm rectangles as the text allows.

The engines report text as runs -- a PDF selection is one polygon per text run ("Frank",
"-", "Starling-Mechanismus"), OCR is one box per word, a slide is one box per shape.
The viewer wants one rectangle per line segment of a match, no rectangle drawn twice,
and one match marked as the primary: the place the search result pointed at.

Rectangles are ``[x, y, w, h]`` in any consistent unit (points or page fractions).
"""

from __future__ import annotations

from typing import Any


def merge_line_rects(rects: list[list[float]]) -> list[list[float]]:
    """Runs on the same text line that touch or nearly touch become one rectangle."""

    ordered = sorted((list(map(float, r)) for r in rects if r and r[2] > 0 and r[3] > 0), key=lambda r: (round(r[1] + r[3] / 2, 1), r[0]))
    merged: list[list[float]] = []
    for rect in ordered:
        if merged:
            last = merged[-1]
            height = max(last[3], rect[3])
            same_line = abs((last[1] + last[3] / 2) - (rect[1] + rect[3] / 2)) <= 0.5 * height
            gap = rect[0] - (last[0] + last[2])
            if same_line and gap <= 0.6 * height:
                left = min(last[0], rect[0])
                top = min(last[1], rect[1])
                right = max(last[0] + last[2], rect[0] + rect[2])
                bottom = max(last[1] + last[3], rect[1] + rect[3])
                merged[-1] = [left, top, right - left, bottom - top]
                continue
        merged.append(rect)
    return merged


def overlap(a: list[float], b: list[float]) -> float:
    """Intersection over the smaller rectangle's area: 1.0 when one lies inside the other."""

    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    return inter / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


def arrange(matches: list[dict[str, Any]], *, at: int | None = None) -> list[dict[str, Any]]:
    """Matches ``[{"phrase", "start", "rects"}]`` in reading order, rectangles merged per line, duplicates dropped, one primary.

    The primary is the occurrence of the first phrase (the focus) nearest to ``at`` -- the character offset the
    search result pointed at -- or its first occurrence.  A later match whose rectangles are already covered by
    an earlier one (the word "Vorlast" inside the marked phrase "Vorlast und Nachlast") is not drawn again.
    """

    cleaned: list[dict[str, Any]] = []
    for order, match in enumerate(matches):
        rects = merge_line_rects(match.get("rects") or [])
        if rects:
            cleaned.append({"phrase": match.get("phrase", ""), "start": int(match.get("start", -1) if match.get("start") is not None else -1),
                            "rank": int(match.get("rank", order)), "rects": rects})
    if not cleaned:
        return []
    focus_rank = min(m["rank"] for m in cleaned)
    focus = [m for m in cleaned if m["rank"] == focus_rank]
    if at is not None and any(m["start"] >= 0 for m in focus):
        primary = min(focus, key=lambda m: abs(m["start"] - at) if m["start"] >= 0 else 10**9)
    else:
        primary = focus[0]
    kept: list[dict[str, Any]] = [primary]
    for match in sorted(cleaned, key=lambda m: (m["rank"], m["start"])):
        if match is primary:
            continue
        fresh = [r for r in match["rects"] if not any(overlap(r, k) > 0.6 for other in kept for k in other["rects"])]
        if fresh:
            kept.append({**match, "rects": fresh})
    out = []
    for match in sorted(kept, key=lambda m: (m["rects"][0][1], m["rects"][0][0])):
        out.append({"phrase": match["phrase"], "start": match["start"], "rects": [[round(v, 5) for v in r] for r in match["rects"]],
                    "primary": match is primary})
    return out
