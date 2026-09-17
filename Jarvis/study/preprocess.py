"""Image preparation for OCR: pure functions on numpy images, and ``prepare`` that chains them.

Every step works on a derived copy -- the file on disk and the caller's array are never changed -- and every
geometric step (orientation, perspective, deskew, crop, scaling) contributes a 3x3 matrix.  ``Prepared.matrix``
maps ORIGINAL pixel coordinates to PREPARED pixel coordinates; ``Prepared.to_original`` inverts it, so a box the OCR
engine finds on the prepared image lands on the right place of the original image as fractions [0..1].

"Original" means the image as a viewer shows it: EXIF orientation from a phone camera is applied on load, exactly
as a browser applies it, and everything else is relative to that.

The steps are conservative on purpose.  A colour diagram must not be destroyed by a threshold, a scan must not be
"perspective corrected" into a trapezoid, and a handwritten page must not be rotated by 90 degrees because an OSD
guess was weak: each step applies only when its evidence is strong, and ``Prepared.steps`` records what happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_LONG_SIDE = 3400        # huge phone photos are scaled down to this
MIN_LONG_SIDE = 1000        # tiny scans are scaled up at least to this
TARGET_TEXT_HEIGHT = 32.0   # median text component height (px) Tesseract and line crops read best at
MAX_SKEW = 12.0             # degrees searched by deskew


# ------------------------------------------------------------------------------- loading + colour

def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    """An RGB or grayscale uint8 array: files through Pillow (Unicode paths, WebP, EXIF orientation), arrays copied."""

    if isinstance(source, np.ndarray):
        array = source.copy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
    from PIL import Image, ImageOps

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        image = image.convert("RGB") if image.mode != "L" else image
        return np.array(image)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()


# ------------------------------------------------------------------------------- photometric steps

def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a blurred background estimate: a shadow across a photographed page becomes an even white."""

    h, w = gray.shape[:2]
    k = max(15, (min(h, w) // 20) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 3)
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return flat


def normalize_contrast(gray: np.ndarray, *, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE: faint pencil and grey ink get contrast without blowing out the paper."""

    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def denoise(gray: np.ndarray, *, strength: int = 1) -> np.ndarray:
    """Light denoising that keeps thin strokes: median 3 for sensor/JPEG speckle, NL-means only when asked (slow)."""

    if strength <= 0:
        return gray.copy()
    out = cv2.medianBlur(gray, 3)
    if strength >= 2:
        out = cv2.fastNlMeansDenoising(out, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return out


def adaptive_threshold(gray: np.ndarray, *, block: int = 31, offset: int = 12) -> np.ndarray:
    """Black ink on white by local mean.  Optional: Tesseract binarises itself; use this for analysis and bad lighting."""

    block = max(3, block | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, offset)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = ink) robust to uneven light: adaptive threshold on the flattened page, specks removed."""

    flat = flatten_illumination(gray)
    _, otsu = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    local = 255 - adaptive_threshold(flat, block=max(15, (min(gray.shape[:2]) // 40) | 1), offset=15)
    mask = cv2.bitwise_and(otsu, local)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def rule_lines_mask(binary: np.ndarray) -> np.ndarray:
    """uint8 mask of ruled / squared paper lines and table rules inside a binary image (255 = dark).

    Long straight runs are rules outright.  Grid paper is subtler: its lines are cut into cell-sized pieces by the
    crossings and by the writing, each piece no longer than a tall letter -- but the pieces line up in the same column
    (or row) over a large part of the page, which handwriting never does."""

    h, w = binary.shape[:2]
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 12), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 12))))
    rules = cv2.bitwise_or(horizontal, vertical)
    short_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 60))))
    short_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 60), 1)))
    # a slightly skewed rule spreads over neighbouring columns / rows: widen before measuring coverage
    cols = cv2.dilate(short_v, np.ones((1, 5), np.uint8)).astype(bool).sum(axis=0) / float(h)
    rows = cv2.dilate(short_h, np.ones((5, 1), np.uint8)).astype(bool).sum(axis=1) / float(w)
    rule_cols = _periodic(cols > 0.3) if h >= 300 else np.zeros(w, bool)
    rule_rows = _periodic(rows > 0.3) if w >= 300 else np.zeros(h, bool)
    if rule_cols.any():
        rules[:, rule_cols] = cv2.bitwise_or(rules[:, rule_cols], short_v[:, rule_cols])
    if rule_rows.any():
        rules[rule_rows, :] = cv2.bitwise_or(rules[rule_rows, :], short_h[rule_rows, :])
    return rules


def _periodic(flags: np.ndarray) -> np.ndarray:
    """Positions flagged in thin, regularly spaced runs (paper rules), widened by 2; everything else cleared.

    Letter stems can line up in a column by chance; four or more thin runs at an even spacing do not."""

    out = np.zeros(len(flags), bool)
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    runs = [r for r in runs if r[1] - r[0] <= 8]
    if len(runs) < 4:
        return out
    centres = np.array([(a + b) / 2.0 for a, b in runs])
    spacing = np.diff(centres)
    typical = float(np.median(spacing))
    if typical < 12:
        return out
    regular = np.abs(spacing - typical) <= max(3.0, typical * 0.15)
    if regular.mean() < 0.6:
        return out
    for index, (a, b) in enumerate(runs):
        neighbours = ([regular[index - 1]] if index > 0 else []) + ([regular[index]] if index < len(regular) else [])
        if any(neighbours):
            out[max(0, a - 2):min(len(out), b + 2)] = True
    return out


def remove_rule_lines(mask: np.ndarray) -> np.ndarray:
    """Ink mask without horizontal/vertical rules (lined or squared paper, table borders)."""

    rules = cv2.dilate(rule_lines_mask(mask), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(mask, cv2.bitwise_not(rules))


def erase_rule_lines(flat: np.ndarray) -> tuple[np.ndarray, int]:
    """(copy with ruled / squared paper lines painted white, number of rule pixels / 1000 rounded up; 0 = none found).

    Ink crossing a rule stays: only pixels close to the rule's own brightness are erased.  Underlines and table rules go
    too, which only helps OCR; the original image is untouched."""

    candidates = (flat < 238).astype(np.uint8) * 255
    rules = rule_lines_mask(candidates)
    found = int(cv2.countNonZero(rules))
    if found < 200:
        return flat.copy(), 0
    mask = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    level = float(np.median(flat[rules > 0]))
    out = flat.copy()
    erase = (mask > 0) & (flat.astype(np.int16) > level - 45)
    out[erase] = 255
    return out, -(-found // 1000)


def text_height(gray: np.ndarray) -> float:
    """Typical height (px) of letter-sized connected components; 0 when the page has no readable ink.

    Ink-weighted median: paper texture, bleed-through and grid fragments are many but small, letters carry the ink."""

    # broken pencil strokes rejoin first, or a faint word counts as a dozen tiny letters
    mask = cv2.morphologyEx(remove_rule_lines(ink_mask(gray)), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    areas = stats[1:, cv2.CC_STAT_AREA]
    h, w = gray.shape[:2]
    keep = (heights >= 4) & (heights < h * 0.2) & (widths < w * 0.5) & (areas >= 8)
    if keep.sum() < 5:
        return 0.0
    kept_heights, kept_areas = heights[keep].astype(np.float64), areas[keep].astype(np.float64)
    order = np.argsort(kept_heights)
    cumulative = np.cumsum(kept_areas[order])
    return float(kept_heights[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


# ------------------------------------------------------------------------------- geometry helpers

def _translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def _scale(s: float) -> np.ndarray:
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)


def _as3(m: np.ndarray) -> np.ndarray:
    return m if m.shape == (3, 3) else np.vstack([m, [0, 0, 1]])


def _border_value(image: np.ndarray) -> Any:
    return 255 if image.ndim == 2 else (255,) * image.shape[2]


def rotate(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate counter-clockwise by ``angle`` degrees on an expanded white canvas; returns (image, 3x3 matrix)."""

    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    out = cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=_border_value(image))
    return out, _as3(m)


def rotate90(image: np.ndarray, degrees: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact rotation by 0/90/180/270 degrees clockwise; returns (image, 3x3 matrix)."""

    h, w = image.shape[:2]
    degrees %= 360
    if degrees == 0:
        return image.copy(), np.eye(3)
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180), np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=np.float64)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=np.float64)
    raise ValueError("degrees must be a multiple of 90")


def resize(image: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    out = cv2.resize(image, (nw, nh), interpolation=interpolation)
    return out, np.array([[nw / w, 0, 0], [0, nh / h, 0], [0, 0, 1]], dtype=np.float64)


# ------------------------------------------------------------------------------- deskew

def estimate_skew(gray: np.ndarray, *, max_angle: float = MAX_SKEW) -> float:
    """Small-angle skew in degrees (positive = text rises to the right, rotate by this to correct); 0.0 when unsure.

    Projection profiles: the rotation that makes text rows sharpest maximises the variance of row ink sums.
    Coarse 1-degree search, then 0.1-degree refinement, on a downscaled ink mask without rule lines."""

    h, w = gray.shape[:2]
    factor = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA) if factor < 1 else gray
    mask = ink_mask(small)
    if cv2.countNonZero(mask) < 50:
        return 0.0
    sh, sw = mask.shape[:2]
    centre = (sw / 2.0, sh / 2.0)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(mask, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        rows = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(rows))

    coarse = np.arange(-max_angle, max_angle + 0.001, 1.0)
    scores = [score(a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 1.0, best + 1.001, 0.1)
    fine_scores = [score(a) for a in fine]
    angle = float(fine[int(np.argmax(fine_scores))])
    baseline = score(0.0)
    if abs(angle) < 0.15 or max(fine_scores) < baseline * 1.02:
        return 0.0
    return round(angle, 2)


def deskew(image: np.ndarray, *, max_angle: float = MAX_SKEW) -> tuple[np.ndarray, np.ndarray, float]:
    angle = estimate_skew(to_gray(image), max_angle=max_angle)
    if angle == 0.0:
        return image.copy(), np.eye(3), 0.0
    out, m = rotate(image, angle)
    return out, m, angle


# ------------------------------------------------------------------------------- orientation

def orientation_osd(gray: np.ndarray, *, min_confidence: float = 3.0) -> tuple[int, float] | None:
    """(clockwise degrees to rotate so text reads upright, confidence) from Tesseract OSD; None when unavailable/unsure."""

    try:
        from study import ocr
    except ImportError:  # pragma: no cover
        return None
    if not ocr.available() or "osd" not in ocr.languages():
        return None
    try:
        import pytesseract
        from PIL import Image

        pytesseract.pytesseract.tesseract_cmd = ocr.tesseract_path()
        h, w = gray.shape[:2]
        factor = min(1.0, 2000.0 / max(h, w))
        small = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA) if factor < 1 else gray
        data = pytesseract.image_to_osd(Image.fromarray(small), config=(ocr._user_dir_flag() + " --psm 0").strip(),
                                        output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 - too little text for OSD is an exception in Tesseract
        return None
    rotate_by = int(data.get("rotate", 0) or 0) % 360
    confidence = float(data.get("orientation_conf", 0.0) or 0.0)
    if confidence < min_confidence:
        return None
    return rotate_by, confidence


def orientation_heuristic(gray: np.ndarray) -> tuple[int, float]:
    """0 or 90 with a ratio: text lines make row profiles much spikier than column profiles.  Cannot tell 90 from 270."""

    h, w = gray.shape[:2]
    factor = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA) if factor < 1 else gray
    mask = remove_rule_lines(ink_mask(small)).astype(np.float64)
    if mask.sum() < 255 * 50:
        return 0, 1.0
    rows = np.var(mask.sum(axis=1)) / max(1, mask.shape[1]) ** 2
    cols = np.var(mask.sum(axis=0)) / max(1, mask.shape[0]) ** 2
    if cols > rows * 2.5:
        return 90, float(cols / max(rows, 1e-9))
    return 0, float(rows / max(cols, 1e-9))


def detect_orientation(gray: np.ndarray, *, use_osd: bool = True) -> tuple[int, str]:
    """(clockwise degrees, how) -- "osd", "heuristic" or "none"."""

    if use_osd:
        found = orientation_osd(gray)
        if found is not None:
            return found[0], "osd"
    degrees, ratio = orientation_heuristic(gray)
    if degrees and ratio > 4.0:
        return degrees, "heuristic"
    return 0, "none"


# ------------------------------------------------------------------------------- perspective

def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype(np.float64)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float64)


def find_page_quad(image: np.ndarray) -> tuple[np.ndarray | None, float]:
    """The dominant page quadrilateral of a photographed page (TL, TR, BR, BL in pixels) and a confidence 0..1.

    Confident only when a convex 4-gon covers a large part of the photo, is not simply the image frame (a scan), its
    corners are near right angles after correction would be plausible, and the page is brighter than its surroundings."""

    gray = to_gray(image)
    h, w = gray.shape[:2]
    factor = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(gray, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA) if factor < 1 else gray
    sh, sw = small.shape[:2]
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edges = cv2.dilate(cv2.Canny(blurred, 40, 120), np.ones((3, 3), np.uint8))
    best: tuple[np.ndarray, float] | None = None
    for source in (bright, edges):
        contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area < 0.25 * sw * sh:
                break
            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = _order_quad(approx)
            ratio = cv2.contourArea(quad.astype(np.float32)) / float(sw * sh)
            margin = 0.015 * max(sw, sh)
            touching = sum(1 for x, y in quad if x < margin or y < margin or x > sw - margin or y > sh - margin)
            if ratio > 0.97 or touching >= 3:
                continue  # the frame of a scan, not a page lying on a table
            angles = []
            for i in range(4):
                a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
                v1, v2 = a - b, c - b
                cosang = abs(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)))
                angles.append(cosang)
            if max(angles) > 0.45:  # a corner below ~63 degrees is not a page seen at a sane angle
                continue
            inside = np.zeros_like(small)
            cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
            inner = float(cv2.mean(small, mask=inside)[0])
            outer_mask = cv2.bitwise_not(inside)
            outer = float(cv2.mean(small, mask=outer_mask)[0]) if cv2.countNonZero(outer_mask) else inner
            contrast = (inner - outer) / 255.0
            if contrast < 0.08:
                continue
            confidence = max(0.0, min(1.0, 0.4 + 0.6 * min(1.0, ratio / 0.5) * min(1.0, contrast / 0.25) - max(angles)))
            if best is None or confidence > best[1]:
                best = (quad / factor, confidence)
    if best is None:
        return None, 0.0
    return best[0], round(best[1], 3)


def warp_page(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The page quadrilateral warped to an upright rectangle; returns (image, 3x3 homography)."""

    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    out = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, matrix.astype(np.float64)


# ------------------------------------------------------------------------------- crop + scale

def content_box(gray: np.ndarray, *, pad: int = 16) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the inked area with padding; the whole image when nothing is inked."""

    h, w = gray.shape[:2]
    mask = ink_mask(gray)
    # a thin dark scanner edge is not content: ignore ink touching the border strip
    border = max(3, min(h, w) // 150)
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [stats[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if not boxes:
        return 0, 0, w, h
    x0 = min(int(b[0]) for b in boxes)
    y0 = min(int(b[1]) for b in boxes)
    x1 = max(int(b[0] + b[2]) for b in boxes)
    y1 = max(int(b[1] + b[3]) for b in boxes)
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def crop_margins(image: np.ndarray, *, pad: int = 16) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """(cropped copy, (dx, dy) offset of the crop in the input, 3x3 matrix)."""

    x0, y0, x1, y1 = content_box(to_gray(image), pad=pad)
    return image[y0:y1, x0:x1].copy(), (x0, y0), _translate(-x0, -y0)


def scale_factor(shape: tuple[int, ...], median_text_height: float) -> float:
    """Upscale small text towards TARGET_TEXT_HEIGHT, downscale huge photos; 1.0 when the size is already sensible."""

    h, w = shape[:2]
    long_side = max(h, w)
    factor = 1.0
    if median_text_height and median_text_height < TARGET_TEXT_HEIGHT * 0.6:
        factor = min(3.0, TARGET_TEXT_HEIGHT / median_text_height)
    elif long_side < MIN_LONG_SIDE:
        factor = MIN_LONG_SIDE / long_side
    if long_side * factor > MAX_LONG_SIDE * 1.4:
        factor = MAX_LONG_SIDE / long_side
    if median_text_height and median_text_height * factor > TARGET_TEXT_HEIGHT * 3:
        factor = min(factor, max(TARGET_TEXT_HEIGHT * 2 / median_text_height, MIN_LONG_SIDE / long_side))
    return 1.0 if 0.92 <= factor <= 1.08 else round(factor, 4)


# ------------------------------------------------------------------------------- prepare

@dataclass
class Prepared:
    """The OCR working copy and how it relates to the original image."""

    image: np.ndarray                    # grayscale uint8, enhanced (what Tesseract reads)
    rgb: np.ndarray                      # the same geometry in colour, unenhanced (what line recognisers crop from)
    matrix: np.ndarray                   # 3x3: original pixel -> prepared pixel
    original_size: tuple[int, int]       # (width, height) of the original image
    steps: list[str] = field(default_factory=list)
    angle: float = 0.0
    orientation: int = 0
    perspective: bool = False
    text_height: float = 0.0

    @property
    def size(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])

    def to_original_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
        back = homogeneous @ np.linalg.inv(self.matrix).T
        return back[:, :2] / back[:, 2:3]

    def to_original_box(self, box: list[float] | tuple[float, ...]) -> list[float]:
        """A prepared-pixel box [x, y, w, h] as fractions [x, y, w, h] of the original image, clipped to [0..1]."""

        x, y, w, h = (float(v) for v in box)
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        back = self.to_original_points(corners)
        ow, oh = self.original_size
        x0 = max(0.0, min(1.0, back[:, 0].min() / ow))
        y0 = max(0.0, min(1.0, back[:, 1].min() / oh))
        x1 = max(0.0, min(1.0, back[:, 0].max() / ow))
        y1 = max(0.0, min(1.0, back[:, 1].max() / oh))
        return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)]

    def from_original_box(self, fraction_box: list[float]) -> list[float]:
        """Fractions of the original image -> prepared pixel box (bounding rectangle)."""

        ow, oh = self.original_size
        x, y, w, h = fraction_box[0] * ow, fraction_box[1] * oh, fraction_box[2] * ow, fraction_box[3] * oh
        pts = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]], dtype=np.float64) @ self.matrix.T
        pts = pts[:, :2] / pts[:, 2:3]
        return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))]


def prepare(source: str | Path | np.ndarray, *, orientation: bool = True, use_osd: bool = True, perspective: bool = True,
            deskew_page: bool = True, crop: bool = True, scale: bool = True, enhance: bool = True, denoise_strength: int = 1,
            binarize: bool = False, erase_rules: bool = True, perspective_confidence: float = 0.6) -> Prepared:
    """Load (never modifying the source) and prepare a page for OCR; ``Prepared`` maps boxes back to the original."""

    rgb = to_rgb(load_image(source))
    oh, ow = rgb.shape[:2]
    matrix = np.eye(3)
    steps: list[str] = []
    result = Prepared(image=to_gray(rgb), rgb=rgb, matrix=matrix, original_size=(ow, oh))

    # 1. a huge photo is scaled down first so every later step is fast
    if max(oh, ow) > MAX_LONG_SIDE * 1.4:
        rgb, m = resize(rgb, MAX_LONG_SIDE / max(oh, ow))
        matrix = m @ matrix
        steps.append("downscale")

    # 2. a page photographed on a table: warp only with a confident quadrilateral
    if perspective:
        quad, confidence = find_page_quad(rgb)
        if quad is not None and confidence >= perspective_confidence:
            rgb, m = warp_page(rgb, quad)
            matrix = m @ matrix
            result.perspective = True
            steps.append(f"perspective({confidence:.2f})")

    # 3. quarter turns
    if orientation:
        degrees, how = detect_orientation(to_gray(rgb), use_osd=use_osd)
        if degrees:
            rgb, m = rotate90(rgb, degrees)
            matrix = m @ matrix
            result.orientation = degrees
            steps.append(f"orientation({degrees},{how})")

    # 4. small-angle skew
    if deskew_page:
        angle = estimate_skew(to_gray(rgb))
        if angle:
            rgb, m = rotate(rgb, angle)
            matrix = m @ matrix
            result.angle = angle
            steps.append(f"deskew({angle})")

    # 5. margins
    if crop:
        cropped, (dx, dy), m = crop_margins(rgb)
        if cropped.size and (dx or dy or cropped.shape[:2] != rgb.shape[:2]):
            rgb = cropped
            matrix = m @ matrix
            steps.append(f"crop({dx},{dy})")

    # 6. scale by text size
    gray = to_gray(rgb)
    height = text_height(gray)
    if scale:
        factor = scale_factor(rgb.shape, height)
        if factor != 1.0:
            rgb, m = resize(rgb, factor)
            matrix = m @ matrix
            height *= factor
            steps.append(f"scale({factor})")
            gray = to_gray(rgb)

    # 7. photometric: a working copy for the recogniser
    if enhance:
        gray = flatten_illumination(gray)
        if erase_rules:
            gray, erased = erase_rule_lines(gray)
            if erased:
                steps.append(f"rules({erased})")
        gray = normalize_contrast(gray)
        steps.append("illumination+clahe")
    if denoise_strength:
        gray = denoise(gray, strength=denoise_strength)
        steps.append("denoise")
    if binarize:
        gray = adaptive_threshold(gray)
        steps.append("threshold")

    result.image, result.rgb, result.matrix, result.steps, result.text_height = gray, rgb, matrix, steps, round(height, 1)
    return result
