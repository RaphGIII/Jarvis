"""PowerPoint slides: exact shape geometry from the file, real slide images from PowerPoint.

``slide_geometry`` reads the PPTX as the zip + XML it is.  A shape's box comes from its
own ``a:xfrm``; placeholders that do not carry one inherit it from the slide layout and
then the slide master, exactly as PowerPoint draws them.  Boxes are fractions of the
slide so the viewer can lay them over a rendered image of any size.

``render_slides`` exports every slide as PNG through PowerPoint's COM server, driven by
a short PowerShell script (no pywin32).  Office automation is treated as hostile:
it never attaches to a PowerPoint the owner has open, never shows a window, opens
read-only, closes without saving, and a hard timeout kills only the PowerPoint process
this render started.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"

EMU_PER_POINT = 12700

_TITLE_TYPES = {"title", "ctrTitle"}
_BODY_TYPES = {"body", "obj", "subTitle"}

Box = tuple[float, float, float, float]


# ------------------------------------------------------------------------------ package

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", "").replace("­", "")).strip()


def _read_xml(archive: zipfile.ZipFile, name: str) -> ET.Element | None:
    try:
        return ET.fromstring(archive.read(name))
    except (KeyError, ET.ParseError, zipfile.BadZipFile, OSError, ValueError):
        return None


def _rels(archive: zipfile.ZipFile, part: str) -> list[tuple[str, str, str]]:
    """(id, type, resolved part name) for every relationship of ``part``."""

    folder, base = posixpath.split(part)
    root = _read_xml(archive, posixpath.join(folder, "_rels", base + ".rels"))
    if root is None:
        return []
    out = []
    for rel in root.findall(f"{_REL}Relationship"):
        target = rel.get("Target", "")
        if not target or rel.get("TargetMode") == "External":
            continue
        resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
        out.append((rel.get("Id", ""), rel.get("Type", ""), resolved))
    return out


def _related(archive: zipfile.ZipFile, part: str, kind: str) -> str:
    for _, rel_type, target in _rels(archive, part):
        if rel_type.endswith("/" + kind):
            return target
    return ""


def slide_order(archive: zipfile.ZipFile) -> list[str]:
    """Slide part names in presentation order (the same numbering the parser uses)."""

    order: list[str] = []
    presentation = _read_xml(archive, "ppt/presentation.xml")
    if presentation is not None:
        targets = {rel_id: target for rel_id, _, target in _rels(archive, "ppt/presentation.xml")}
        for slide_id in presentation.iter(f"{_P}sldId"):
            target = targets.get(slide_id.get(f"{_R}id", ""), "")
            if target:
                order.append(target)
    if not order:
        names = archive.namelist()
        order = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
    return order


def slide_size(archive: zipfile.ZipFile) -> tuple[int, int] | None:
    presentation = _read_xml(archive, "ppt/presentation.xml")
    size = presentation.find(f"{_P}sldSz") if presentation is not None else None
    if size is None:
        return None
    try:
        cx, cy = int(size.get("cx", "0")), int(size.get("cy", "0"))
    except ValueError:
        return None
    return (cx, cy) if cx > 0 and cy > 0 else None


# ----------------------------------------------------------------------------- geometry

def _int(el: ET.Element | None, attr: str) -> int | None:
    if el is None:
        return None
    try:
        return int(el.get(attr, ""))
    except ValueError:
        return None


def _xfrm_of(shape: ET.Element) -> ET.Element | None:
    for props in (f"{_P}spPr", f"{_P}grpSpPr"):
        holder = shape.find(props)
        if holder is not None:
            return holder.find(f"{_A}xfrm")
    return shape.find(f"{_P}xfrm")  # graphicFrame


def _box_of(xfrm: ET.Element | None) -> Box | None:
    if xfrm is None:
        return None
    off, ext = xfrm.find(f"{_A}off"), xfrm.find(f"{_A}ext")
    x, y, w, h = _int(off, "x"), _int(off, "y"), _int(ext, "cx"), _int(ext, "cy")
    if None in (x, y, w, h):
        return None
    return (float(x), float(y), float(w), float(h))  # type: ignore[arg-type]


def _placeholder(shape: ET.Element) -> ET.Element | None:
    for nv in (f"{_P}nvSpPr", f"{_P}nvPicPr", f"{_P}nvGraphicFramePr"):
        holder = shape.find(nv)
        if holder is not None:
            nv_pr = holder.find(f"{_P}nvPr")
            return nv_pr.find(f"{_P}ph") if nv_pr is not None else None
    return None


def _category(ph_type: str | None) -> str:
    ph_type = ph_type or "obj"  # the specification's default placeholder type
    if ph_type in _TITLE_TYPES:
        return "title"
    if ph_type in _BODY_TYPES:
        return "body"
    return ph_type


def _find_placeholder(tree: ET.Element | None, ph_type: str | None, idx: str | None, *, by_idx: bool) -> ET.Element | None:
    """The placeholder shape in a layout/master that a slide placeholder inherits from."""

    if tree is None:
        return None
    wanted = _category(ph_type)
    candidates = [(shape, ph) for shape in tree.iter(f"{_P}sp") if (ph := _placeholder(shape)) is not None]
    if by_idx and idx is not None:
        for shape, ph in candidates:
            if ph.get("idx") == idx and _category(ph.get("type")) == wanted:
                return shape
    for shape, ph in candidates:
        if ph.get("type", "obj") == (ph_type or "obj"):
            return shape
    for shape, ph in candidates:
        if _category(ph.get("type")) == wanted:
            return shape
    if by_idx and idx is not None:
        for shape, ph in candidates:
            if ph.get("idx") == idx:
                return shape
    return None


class _Inheritance:
    """The layout and master a slide draws its placeholders from, parsed once per part."""

    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self._cache: dict[str, ET.Element | None] = {}

    def tree(self, part: str) -> ET.Element | None:
        if not part:
            return None
        if part not in self._cache:
            self._cache[part] = _read_xml(self.archive, part)
        return self._cache[part]

    def inherited_box(self, slide_part: str, ph: ET.Element) -> Box | None:
        ph_type, idx = ph.get("type"), ph.get("idx")
        layout_part = _related(self.archive, slide_part, "slideLayout")
        layout_shape = _find_placeholder(self.tree(layout_part), ph_type, idx, by_idx=True)
        if layout_shape is not None:
            box = _box_of(_xfrm_of(layout_shape))
            if box is not None:
                return box
            layout_ph = _placeholder(layout_shape)
            if layout_ph is not None:
                ph_type = layout_ph.get("type", ph_type)
        master_part = _related(self.archive, layout_part, "slideMaster") if layout_part else ""
        master_shape = _find_placeholder(self.tree(master_part), ph_type, idx, by_idx=False)
        return _box_of(_xfrm_of(master_shape)) if master_shape is not None else None


def _group_transform(own: Box, child: tuple[int, int, int, int], parent: Callable[[Box], Box] | None) -> Callable[[Box], Box]:
    """Map a box in a group's child coordinates (chOff/chExt) onto the group's own box, then onward."""

    cox, coy, cew, ceh = child
    sx = own[2] / cew if cew else 1.0
    sy = own[3] / ceh if ceh else 1.0

    def apply(b: Box) -> Box:
        mapped = (own[0] + (b[0] - cox) * sx, own[1] + (b[1] - coy) * sy, b[2] * sx, b[3] * sy)
        return parent(mapped) if parent is not None else mapped

    return apply


def _paragraphs(shape: ET.Element) -> list[str]:
    body = shape.find(f"{_P}txBody")
    if body is None:
        return []
    out = []
    for para in body.iter(f"{_A}p"):
        text = _clean("".join(t.text or "" for t in para.iter(f"{_A}t")))
        if text:
            out.append(text)
    return out


def shape_entries(archive: zipfile.ZipFile, slide_part: str, slide: ET.Element,
                  size: tuple[int, int] | None, inheritance: _Inheritance | None = None) -> list[tuple[ET.Element, dict[str, Any]]]:
    """Every text shape and picture of a parsed slide, in document order, with its box.

    Returns ``(element, shape)`` so a caller that walks the same tree can look boxes up
    by element.  ``box`` is None when neither the slide nor its layout/master place the
    shape, or when the slide size is unknown.
    """

    inheritance = inheritance or _Inheritance(archive)
    entries: list[tuple[ET.Element, dict[str, Any]]] = []
    tree = slide.find(f"{_P}cSld/{_P}spTree")
    if tree is None:
        return entries

    def to_fraction(box: Box | None) -> list[float] | None:
        if box is None or size is None:
            return None
        cx, cy = size
        x0, y0 = max(0.0, box[0] / cx), max(0.0, box[1] / cy)
        x1, y1 = min(1.0, (box[0] + box[2]) / cx), min(1.0, (box[1] + box[3]) / cy)
        if x1 < x0 or y1 < y0:
            return None
        return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)]

    def walk(container: ET.Element, transform: Callable[[Box], Box] | None, group_box: Box | None) -> None:
        for child in container:
            tag = child.tag
            if tag == f"{_P}grpSp":
                xfrm = _xfrm_of(child)
                own = _box_of(xfrm)
                if own is None:
                    # no group transform: children keep the enclosing coordinate space
                    walk(child, transform, group_box)
                    continue
                if transform is not None:
                    outer: Box = transform(own)
                elif group_box is not None:
                    outer = group_box
                else:
                    outer = own
                ch_off = xfrm.find(f"{_A}chOff") if xfrm is not None else None
                ch_ext = xfrm.find(f"{_A}chExt") if xfrm is not None else None
                cox, coy, cew, ceh = _int(ch_off, "x"), _int(ch_off, "y"), _int(ch_ext, "cx"), _int(ch_ext, "cy")
                if group_box is None and cox is not None and coy is not None and cew is not None and ceh is not None:
                    walk(child, _group_transform(own, (cox, coy, cew, ceh), transform), None)
                else:
                    walk(child, None, outer)  # child offsets not resolvable: every child gets the group's box
                continue
            if tag not in (f"{_P}sp", f"{_P}pic"):
                continue
            ph = _placeholder(child)
            box = _box_of(_xfrm_of(child))
            if box is not None:
                if transform is not None:
                    box = transform(box)
                elif group_box is not None:
                    box = group_box  # child offsets not resolvable: the group's own box
            elif group_box is not None:
                box = group_box
            elif ph is not None:
                box = inheritance.inherited_box(slide_part, ph)
            if tag == f"{_P}pic":
                entries.append((child, {"kind": "picture", "box": to_fraction(box), "text": "", "paragraphs": []}))
                continue
            paragraphs = _paragraphs(child)
            if not paragraphs:
                continue
            if ph is not None and _category(ph.get("type")) == "title":
                kind = "title"
            elif ph is not None and _category(ph.get("type")) == "body":
                kind = "body"
            else:
                kind = "text"
            entries.append((child, {"kind": kind, "box": to_fraction(box), "text": "\n".join(paragraphs), "paragraphs": paragraphs}))

    walk(tree, None, None)
    return entries


def slide_geometry(path: str | Path) -> dict[int, dict[str, Any]]:
    """Per 1-based slide number: the slide size in EMU and every shape with its box as fractions of the slide."""

    out: dict[int, dict[str, Any]] = {}
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return out
    with archive:
        size = slide_size(archive)
        inheritance = _Inheritance(archive)
        for number, part in enumerate(slide_order(archive), start=1):
            slide = _read_xml(archive, part)
            if slide is None:
                continue
            shapes = [shape for _, shape in shape_entries(archive, part, slide, size, inheritance)]
            out[number] = {"size": list(size) if size else None, "shapes": shapes}
    return out


# ------------------------------------------------------------------------------- render

_available: bool | None = None
_render_lock = threading.Lock()


def available() -> bool:
    """Windows with PowerPoint's COM server registered.  Cheap: a registry lookup, cached."""

    global _available
    if _available is None:
        _available = False
        if sys.platform == "win32":
            try:
                import winreg

                for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Classes\PowerPoint.Application\CLSID"),
                                  (winreg.HKEY_CLASSES_ROOT, r"PowerPoint.Application\CLSID")):
                    try:
                        with winreg.OpenKey(hive, key):
                            _available = True
                            break
                    except OSError:
                        continue
            except ImportError:
                _available = False
    return _available


def powerpoint_pids() -> set[int] | None:
    """PIDs of running POWERPNT.EXE processes; None when the process list cannot be read."""

    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq POWERPNT.EXE", "/FO", "CSV", "/NH"], capture_output=True,
                             timeout=20, creationflags=flags).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pids: set[int] = set()
    for line in raw.decode("ascii", errors="replace").splitlines():
        fields = [f.strip().strip('"') for f in line.split(",")]
        if len(fields) >= 2 and fields[0].lower() == "powerpnt.exe" and fields[1].isdigit():
            pids.add(int(fields[1]))
    return pids


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
    except OSError:
        pass


_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$src = $env:ZEUS_SLIDES_SRC
$out = $env:ZEUS_SLIDES_OUT
$resultFile = $env:ZEUS_SLIDES_RESULT
$width = [int]$env:ZEUS_SLIDES_WIDTH
$result = @{ ok = $false; slides = @(); width = $width; height = 0; error = '' }
$app = $null
$pres = $null
try {
    if (Get-Process -Name POWERPNT -ErrorAction SilentlyContinue) { throw 'POWERPNT_RUNNING' }
    $app = New-Object -ComObject PowerPoint.Application
    try { $app.DisplayAlerts = 1 } catch { }
    $pres = $app.Presentations.Open($src, -1, 0, 0)
    $sw = [double]$pres.PageSetup.SlideWidth
    $sh = [double]$pres.PageSetup.SlideHeight
    $height = [int][math]::Round($width * $sh / $sw)
    $result.height = $height
    $files = @()
    $count = $pres.Slides.Count
    for ($i = 1; $i -le $count; $i++) {
        $file = [System.IO.Path]::Combine($out, ('slide_' + $i + '.png'))
        $pres.Slides.Item($i).Export($file, 'PNG', $width, $height)
        $files += $file
    }
    $result.slides = $files
    $result.ok = $true
} catch {
    $result.error = [string]$_.Exception.Message
} finally {
    if ($pres -ne $null) { try { $pres.Saved = -1 } catch { } ; try { $pres.Close() } catch { } }
    if ($app -ne $null) {
        try { if ($app.Presentations.Count -eq 0) { $app.Quit() } } catch { }
        try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) } catch { }
    }
    $pres = $null
    $app = $null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    $json = $result | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($resultFile, $json, (New-Object System.Text.UTF8Encoding $false))
}
"""

_BUSY = "PowerPoint ist gerade geöffnet – die Folienbilder werden erstellt, sobald PowerPoint geschlossen ist."


def render_slides(path: str | Path, out_dir: str | Path, *, width: int = 1600, timeout: float = 120.0,
                  _pids: Callable[[], set[int] | None] | None = None) -> dict[str, Any]:
    """Every slide as ``out_dir/slide_{n}.png`` through PowerPoint, with all Office safety rules applied."""

    started = time.monotonic()
    width = max(64, min(int(width), 8000))
    result: dict[str, Any] = {"ok": False, "slides": [], "width": width, "height": 0, "seconds": 0.0, "error": "", "deferred": False}

    def finish(**changes: Any) -> dict[str, Any]:
        result.update(changes)
        result["seconds"] = round(time.monotonic() - started, 2)
        return result

    list_pids = _pids or powerpoint_pids
    source = Path(path).resolve()
    if not source.is_file():
        return finish(error=f"Datei nicht gefunden: {source.name}")
    if not available():
        return finish(error="PowerPoint ist auf diesem Rechner nicht verfügbar.")
    if not _render_lock.acquire(timeout=max(1.0, timeout)):
        return finish(error="Eine andere Folienumwandlung läuft noch.", deferred=True)
    work = ""
    try:
        before = list_pids()
        if before is None:
            return finish(error="Die Prozessliste ist nicht lesbar; PowerPoint wird nicht gestartet.", deferred=True)
        if before:
            return finish(error=_BUSY, deferred=True)
        target = Path(out_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.glob("slide_*.png"):
            if re.fullmatch(r"slide_\d+\.png", stale.name):
                try:
                    stale.unlink()
                except OSError:
                    pass
        work = tempfile.mkdtemp(prefix="zeus_slides_")
        script = Path(work) / "export_slides.ps1"
        script.write_text(_SCRIPT, encoding="utf-8-sig")
        result_file = Path(work) / "result.json"
        env = dict(os.environ, ZEUS_SLIDES_SRC=str(source), ZEUS_SLIDES_OUT=str(target), ZEUS_SLIDES_RESULT=str(result_file),
                   ZEUS_SLIDES_WIDTH=str(width))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                                       cwd=work, creationflags=flags)
        except OSError as exc:
            return finish(error=f"PowerShell ließ sich nicht starten: {exc}")
        remaining = max(1.0, timeout - (time.monotonic() - started))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            _kill_ours(before, list_pids)
            return finish(error=f"PowerPoint hat nicht innerhalb von {int(timeout)} s geantwortet; der Vorgang wurde abgebrochen.")
        # After Quit() PowerPoint needs a moment to leave.  One that stays is NOT ended: if the owner opened a
        # presentation meanwhile, Windows joined it to this very process, and ending it would lose their work.
        deadline = time.monotonic() + max(2.0, min(20.0, timeout - (time.monotonic() - started)))
        while time.monotonic() < deadline:
            now = list_pids()
            if now is None or not (now - before):
                break
            time.sleep(0.5)
        try:
            answer = json.loads(result_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return finish(error="PowerPoint hat kein Ergebnis geliefert.")
        if not answer.get("ok"):
            message = str(answer.get("error") or "")
            if "POWERPNT_RUNNING" in message:
                return finish(error=_BUSY, deferred=True)
            return finish(error=f"PowerPoint konnte die Folien nicht exportieren: {message}".strip())
        slides = answer.get("slides") or []
        if isinstance(slides, str):  # ConvertTo-Json flattens a one-element array
            slides = [slides]
        files = [str(p) for p in slides if Path(str(p)).is_file()]
        if len(files) != len(slides):
            return finish(error="Nicht alle Folienbilder wurden geschrieben.", slides=files)
        return finish(ok=True, slides=files, width=int(answer.get("width") or width), height=int(answer.get("height") or 0))
    finally:
        _render_lock.release()
        if work:
            shutil.rmtree(work, ignore_errors=True)


def visible_window_pids() -> set[int]:
    """Processes that own a visible top-level window: an Office process the owner is looking at."""

    if sys.platform != "win32":
        return set()
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        owners: set[int] = set()
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def visit(hwnd: int, _lparam: int) -> bool:
            if user32.IsWindowVisible(hwnd):
                proc_id = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
                owners.add(int(proc_id.value))
            return True

        user32.EnumWindows(callback_type(visit), 0)
        return owners
    except Exception:  # noqa: BLE001 - unknown means: assume the owner might be looking
        return {-1}


def _kill_ours(before: set[int], list_pids: Callable[[], set[int] | None], visible: Callable[[], set[int]] | None = None) -> None:
    """End only the PowerPoint processes that were not running before this render started and that show no window.

    A hung hidden render is ours; a PowerPoint with a visible window is the owner's (Windows joins a presentation
    the owner opens during a render into the same process) and is never ended."""

    now = list_pids()
    shown = (visible or visible_window_pids)()
    if -1 in shown:
        return
    for pid in (now or set()) - before:
        if pid not in shown:
            _kill_pid(pid)
