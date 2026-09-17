/* The Studium viewer: the original material, opened at the place that was asked for.

   PDF        the rendered page with every match marked (rectangles from the PDF
              engine, or from the OCR word boxes on a scanned page); the match the
              result pointed at is the primary mark.  Previous / next / go to page,
              zoom, search in the document, the page's text, "Ask ZEUS".
   PowerPoint the exact slide as PowerPoint draws it, the shape that holds the
              match marked, speaker notes below; while the image is being made
              (or without PowerPoint) the slide's own structure, calmly.
   Word, Markdown, text
              the whole document as one reading column with its outline; it opens
              at the section and paragraph of the result, the matches marked
              there.  Word has no page numbers of its own, so none are shown.
   image      the image itself

   Address: #study?doc=ID&unit=N&focus=PHRASE&hl=OTHER|PHRASES&at=OFFSET&para=P

   "Ask ZEUS" sends the question with the open place -- or only the selected
   text -- as its source, and shows the answer here, beside the page. */

import { el, clear } from "../core/dom.js";
import { api, audioUrl } from "../core/api.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";
import { render as renderMarkdown, renderPartial } from "../core/markdown.js";
import { findPhrases, primarySpan } from "../core/textmarks.js";
import { PageLoader, renderPlan, visibleTiles, visibleFraction } from "../core/pagerender.js";

let active = null;   // the mounted viewer's state; one at a time

const WORD = {
  pdf_pages: ["Seite", "Seiten"], pdf_native: ["Seite", "Seiten"], image: ["Seite", "Seiten"], slides: ["Folie", "Folien"], sections: ["Abschnitt", "Abschnitte"],
};
const ASK = {
  page: ["Erklär mir nur diesen Abschnitt.", "Fass diese Seite zusammen.", "Prüf mich zu dieser Seite."],
  slide: ["Erklär mir diese Folie.", "Fass diese Folie zusammen.", "Prüf mich zu dieser Folie."],
  section: ["Erklär mir nur diesen Abschnitt.", "Fass diesen Abschnitt zusammen.", "Prüf mich aus diesem Abschnitt."],
};

export async function mountViewer(pane, params) {
  const id = String(params.doc || "");
  const loading = el("div", { class: "vw-loading", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: "Öffnet die Stelle …" }));
  pane.append(loading);
  const info = await api("/api/study/document", { id });
  loading.remove();
  if (!info || info.ok === false) {
    pane.append(failure(info && info.transport ? "ZEUS ist gerade nicht erreichbar." : "Dieses Dokument ist nicht mehr im Studium.",
      info && info.transport ? "Die Verbindung kommt gleich wieder; dann öffnet sich die Stelle." : "Es wurde entfernt oder neu eingelesen.",
      [["Zurück zum Studium", () => views.open("study", {})]]));
    return;
  }
  const doc = info.document;
  const units = info.units || [];
  const total = Math.max(1, doc.units || units.length || 1);
  const phrases = [String(params.focus || ""), ...String(params.hl || "").split("|")].map((p) => p.trim()).filter(Boolean);
  const state = {
    id, doc, units, total, kind: info.viewer, slides: info.slides || null, filePresent: info.file_present !== false,
    unit: clampUnit(Number(params.unit) || 1, total), focus: String(params.focus || ""), phrases, at: numberOr(params.at, null),
    para: numberOr(params.para, null), zoom: Math.max(0.5, Math.min(3, numberOr(params.zoom, 1))), side: "", selection: "", askRequest: "", streamText: "", disposers: [],
    fig: numberOr(params.fig, null), figUnit: numberOr(params.fig, null) != null ? clampUnit(Number(params.unit) || 1, total) : null,
    matchCache: new Map(), sheet: null,
  };
  // page bitmaps: preview, then sharp, tiles when zoomed in; the server skips renders this viewer has moved past
  state.loader = new PageLoader({
    urlFor: (page, width, clip, seq, prio, session) => audioUrl(`/api/study/page.png?id=${encodeURIComponent(id)}&page=${page}&w=${width}`
      + `${clip ? `&clip=${clip.join(",")}` : ""}&seq=${seq}&prio=${prio}&session=${session}`),
  });
  state.disposers.push(() => { state.loader.controller?.abort(); state.loader.cache.keep.clear(); state.loader.cache.clear(); });
  active = state;

  const title = el("div", { class: "vw-title" }, el("span", { class: "vw-name", text: doc.filename || doc.title, title: doc.filename || doc.title }),
    el("span", { class: "vw-where" }));
  const pageInput = el("input", { class: "vw-page", type: "number", min: "1", max: String(total), "aria-label": unitWord(state) + " wählen",
                                  onChange: (e) => go(state, Number(e.target.value)) });
  const zoomLabel = el("span", { class: "vw-zoom", text: `${Math.round(state.zoom * 100)} %` });
  const zoomable = state.kind === "pdf_pages" || state.kind === "image" || state.kind === "slides";
  const bar = el("header", { class: "vw-bar" },
    el("button", { class: "vw-back", "aria-label": "Zurück zum Studium", onClick: () => views.open("study", {}) }, "←"),
    title,
    el("div", { class: "vw-nav", role: "group", "aria-label": "Navigation" },
      el("button", { class: "vw-btn", "aria-label": `Vorherige ${unitWord(state)}`, text: "‹", onClick: () => go(state, state.unit - 1) }),
      pageInput, el("span", { class: "vw-total", text: `/ ${total}` }),
      el("button", { class: "vw-btn", "aria-label": `Nächste ${unitWord(state)}`, text: "›", onClick: () => go(state, state.unit + 1) })),
    zoomable ? el("div", { class: "vw-nav vw-zoomgroup", role: "group", "aria-label": "Zoom" },
      el("button", { class: "vw-btn", "aria-label": "Verkleinern", text: "−", onClick: () => zoom(state, -0.15) }), zoomLabel,
      el("button", { class: "vw-btn", "aria-label": "Vergrößern", text: "+", onClick: () => zoom(state, 0.15) })) : null,
    el("div", { class: "vw-tools" },
      state.kind === "sections" ? el("button", { class: "vw-tool", text: "Gliederung", "aria-pressed": "true", onClick: (e) => toggleOutline(state, e.currentTarget) }) : null,
      el("button", { class: "vw-tool", text: "Suchen", "aria-pressed": "false", onClick: (e) => toggleSide(state, "search", e.currentTarget) }),
      state.kind === "pdf_pages" || state.kind === "slides" ? el("button", { class: "vw-tool", text: "Text", "aria-pressed": "false", onClick: (e) => toggleSide(state, "text", e.currentTarget) }) : null,
      el("button", { class: "vw-tool", text: "Original", title: "In der zugehörigen App öffnen", onClick: () => openOriginal(state) }),
      el("button", { class: "vw-ask", text: "Ask ZEUS", "aria-pressed": "false", onClick: (e) => toggleSide(state, "ask", e.currentTarget) })));
  const page = el("div", { class: "vw-page-area", tabindex: "0", "aria-label": "Dokument" });
  const side = el("aside", { class: "vw-side", hidden: true });
  const outline = state.kind === "sections" ? el("nav", { class: "vw-outline", "aria-label": "Gliederung" }) : null;
  const root = el("div", { class: "viewer", "data-kind": state.kind }, bar, el("div", { class: "vw-body" + (outline ? " with-outline" : "") }, outline, page, side));
  pane.append(root);
  Object.assign(state, { root, page, side, title, pageInput, zoomLabel, outline });

  const onKey = (e) => {
    if (active !== state || ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); go(state, state.unit + 1); }
    if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); go(state, state.unit - 1); }
    if ((e.key === "+" || e.key === "=") && zoomable) zoom(state, 0.15);
    if (e.key === "-" && zoomable) zoom(state, -0.15);
  };
  document.addEventListener("keydown", onKey);
  state.disposers.push(() => document.removeEventListener("keydown", onKey));
  const onSelect = () => {
    const selection = String(window.getSelection?.() || "").trim();
    if (selection && state.root.contains(window.getSelection().anchorNode)) {
      state.selection = selection.slice(0, 6000);
      if (state.side === "ask") renderAsk(state);
    }
  };
  document.addEventListener("selectionchange", onSelect);
  state.disposers.push(() => document.removeEventListener("selectionchange", onSelect));
  if (window.ResizeObserver && (state.kind === "pdf_pages" || state.kind === "slides")) {
    let timer = 0;
    let lastWidth = page.clientWidth;
    const observer = new ResizeObserver(() => {
      if (Math.abs(page.clientWidth - lastWidth) < 24) return;
      lastWidth = page.clientWidth;
      clearTimeout(timer);
      timer = setTimeout(() => (state.sheet && state.sheet.unit.number === state.unit ? rezoomPdf(state) : renderUnit(state, { keepScroll: true })), 220);
    });
    observer.observe(page);
    state.disposers.push(() => observer.disconnect());
  }
  if (state.kind === "pdf_pages") {
    let tileTimer = 0;
    const onScroll = () => { clearTimeout(tileTimer); tileTimer = setTimeout(() => updateTiles(state), 140); };
    page.addEventListener("scroll", onScroll, { passive: true });
    state.disposers.push(() => { clearTimeout(tileTimer); page.removeEventListener("scroll", onScroll); });
  }
  // "Mach größer" / "Zeig die ganze Seite" from the chat or voice
  state.disposers.push(bus.on("notification", (p) => {
    if (!p || p._replay || p.kind !== "study_view" || active !== state) return;
    if (p.params && p.params.doc && p.params.doc !== state.id) return;
    const command = p.command || {};
    if (command.fit) zoom(state, 0, { fit: true });
    else if (Number(command.zoom)) zoom(state, Number(command.zoom));
  }));
  if (state.kind === "sections") await mountSections(state);
  else await renderUnit(state);
  if (params.fit && zoomable && active === state) zoom(state, 0, { fit: true });
}

export function unmountViewer() {
  if (!active) return;
  for (const dispose of active.disposers) { try { dispose(); } catch { /* already gone */ } }
  active = null;
}

function numberOr(value, fallback) {
  const n = Number(value);
  return value !== undefined && value !== null && value !== "" && Number.isFinite(n) ? n : fallback;
}

function clampUnit(n, total) {
  return Math.max(1, Math.min(total, Math.round(n) || 1));
}

function unitWord(state, plural = false) {
  return (WORD[state.kind] || WORD.sections)[plural ? 1 : 0];
}

function failure(title, sub, actions = []) {
  return el("div", { class: "vw-failed", role: "status" },
    el("div", { class: "vw-failed-title", text: title }),
    sub ? el("div", { class: "vw-failed-sub", text: sub }) : null,
    actions.length ? el("div", { class: "vw-failed-actions" }, ...actions.map(([label, run]) => el("button", { class: "vw-link", text: label, onClick: run }))) : null);
}

async function openOriginal(state) {
  const answer = await api("/api/fs/open", { path: state.doc.stored_path });
  if (answer && answer.ok === false) {
    window.zeus?.toast?.(state.filePresent ? "Die Originaldatei ließ sich nicht öffnen." : "Die Originaldatei ist nicht mehr da.", "bad");
  }
}

function writeAddress(state) {
  const params = { doc: state.id, unit: String(state.unit) };
  if (state.focus) params.focus = state.focus;
  history.replaceState(history.state, "", `#study?${new URLSearchParams(params)}`);
}

async function go(state, number, { focus = "", phrases = null, at = null, para = null } = {}) {
  const next = clampUnit(number, state.total);
  if (next === state.unit && !focus && para == null) { state.pageInput.value = String(next); return; }
  state.unit = next;
  state.focus = focus;
  state.phrases = phrases || (focus ? [focus] : []);
  state.at = at;
  state.para = para;
  state.selection = "";
  if (state.figUnit !== next) state.fig = null;
  writeAddress(state);
  if (state.kind === "sections") { scrollToSection(state, { smooth: true }); return; }
  await renderUnit(state);
}

function zoom(state, delta, { fit = false } = {}) {
  if (!state.zoomLabel) return;
  if (fit) {
    // the whole page in view: as large as the pane allows, never larger than the reading width
    const unit = state.current || {};
    const ratio = unit.width && unit.height ? unit.width / unit.height : (state.kind === "slides" ? 16 / 9 : 595 / 842);
    const base = Math.min(Math.max(320, state.page.clientWidth - 48), state.kind === "slides" ? 1080 : 880);
    state.zoom = Math.max(0.5, Math.min(1, Math.floor(((state.page.clientHeight - 40) * ratio / base) * 100) / 100));
  } else {
    state.zoom = Math.max(0.5, Math.min(3, Math.round((state.zoom + delta) * 100) / 100));
  }
  state.zoomLabel.textContent = `${Math.round(state.zoom * 100)} %`;
  if (state.kind === "pdf_pages" && state.sheet && state.sheet.unit.number === state.unit) rezoomPdf(state, { top: fit });
  else renderUnit(state, { keepScroll: !fit });
  clearTimeout(state.zoomSync);
  state.zoomSync = setTimeout(() => remember(state), 600);
}

function whereLabel(unit) {
  if (!unit) return "";
  if (unit.kind === "page") return `Seite ${unit.number}`;
  if (unit.kind === "slide") return `Folie ${unit.number}${unit.title ? " · " + unit.title : ""}`;
  return unit.title || `Abschnitt ${unit.number}`;
}

function remember(state) {
  // the open place is the study focus: "Fass diese Seite zusammen" in the chat means this page
  api("/api/study/focus", { id: state.id, unit: state.unit, topic: state.focus, focus: state.focus, zoom: state.zoom, mode: state.side || "page",
                            paragraph: state.para, at: state.at, highlights: state.phrases.slice(0, 8) });
  try { localStorage.setItem("zeus.study.last", JSON.stringify({ doc: state.id, unit: state.unit })); } catch { /* private window */ }
}

async function renderUnit(state, { keepScroll = false } = {}) {
  const token = (state.renderToken = (state.renderToken || 0) + 1);
  state.pageInput.value = String(state.unit);
  const info = await api("/api/study/unit", { id: state.id, unit: state.unit });
  if (token !== state.renderToken || active !== state) return;
  if (!info || info.ok === false) {
    clear(state.page);
    state.page.append(failure("Diese Stelle lässt sich gerade nicht öffnen.", info && info.error ? info.error : "", [["Erneut versuchen", () => renderUnit(state)]]));
    return;
  }
  const unit = info.unit || { number: state.unit, kind: "page", text: "", blocks: [] };
  state.current = unit;
  state.figures = info.figures || [];
  state.sheet = null;
  state.title.querySelector(".vw-where").textContent = whereLabel(unit);
  const scroll = keepScroll ? state.page.scrollTop : 0;
  clear(state.page);
  if (!state.filePresent && state.kind !== "slides") {
    state.page.append(failure("Die Originaldatei ist nicht mehr da.", `${state.doc.stored_path || state.doc.filename} – der Text bleibt durchsuchbar.`,
      [["Text dieser Seite zeigen", () => toggleSide(state, "text", null)]]));
  } else if (state.kind === "pdf_pages") renderPdfPage(state, unit);
  else if (state.kind === "pdf_native") renderNativePdf(state);
  else if (state.kind === "image") renderImage(state);
  else if (state.kind === "slides") renderSlide(state, unit);
  state.page.scrollTop = scroll;
  if (state.side === "text") renderTextSide(state);
  if (state.side === "ask") renderAsk(state);
  remember(state);
}

/* 100 % is a comfortable reading width, not the whole pane; zoom scales from there. */
function sheetWidth(state, max = 880) {
  return Math.round(Math.min(Math.max(320, state.page.clientWidth - 48), max) * state.zoom);
}

function sheetShell(state, unit, { width, ratio, label }) {
  const loadingLine = el("div", { class: "vw-sheet-loading", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: label }));
  const sheet = el("div", { class: "vw-sheet", style: { width: `${width}px`, aspectRatio: ratio ? String(ratio) : "" } }, loadingLine);
  return sheet;
}

function renderPdfPage(state, unit) {
  const cssWidth = sheetWidth(state);
  const ratio = unit.width && unit.height ? unit.width / unit.height : 595 / 842;
  const sheet = sheetShell(state, unit, { width: cssWidth, ratio, label: `Seite ${unit.number} wird geladen …` });
  const img = el("img", { class: "vw-img", alt: `${state.doc.filename}, Seite ${unit.number}`, draggable: "false" });
  const tiles = el("div", { class: "vw-tiles", "aria-hidden": "true" });
  const marks = el("div", { class: "vw-marks", "aria-hidden": "true" });
  sheet.append(img, tiles, marks);
  state.page.append(sheet);
  state.sheet = { el: sheet, img, tiles, marks, unit, ratio, shown: 0, plan: null };
  img.addEventListener("load", () => {
    sheet.classList.add("loaded"); sheet.style.aspectRatio = ""; sheet.querySelector(".vw-sheet-loading")?.remove();
    if (state.sheet && state.sheet.el === sheet && state.sheet.plan && state.sheet.plan.tiles) updateTiles(state);
  });
  paintPdf(state);
  if (!unit.has_text) {
    state.page.append(el("div", { class: "vw-note", text: "Auf dieser Seite ist kein Text erkannt – sie ist geöffnet, aber nicht nach Inhalt durchsuchbar." }));
  } else if (unit.ocr) {
    state.page.append(el("div", { class: "vw-note", text: unit.handwriting ? "Text dieser Seite aus der Handschrifterkennung – die Seite selbst ist das Original." : "Text dieser Seite aus der Texterkennung." }));
  }
  if (state.phrases.length || state.fig != null) markMatches(state, unit, marks);
}

/* Preview at once, the sharp page when it is ready, tiles of the visible region when zoomed in, neighbours ahead.
   Highlights live in their own layer: a new mark never re-renders a bitmap, a new bitmap never moves a mark. */
async function paintPdf(state) {
  const view = state.sheet;
  if (!view) return;
  const loader = state.loader;
  loader.begin();
  const generation = loader.seq;
  const number = view.unit.number;
  const plan = renderPlan(parseFloat(view.el.style.width) || sheetWidth(state), window.devicePixelRatio || 1);
  view.plan = plan;
  const stale = () => active !== state || state.sheet !== view || loader.seq !== generation;
  const show = async (url, width) => {
    if (!url || stale() || view.shown >= width) return;
    if (view.img.getAttribute("src")) {
      // swap only a decoded bitmap: the page never flashes empty between preview and sharp
      const probe = new Image();
      probe.src = url;
      try { await probe.decode(); } catch { /* shown below anyway */ }
      if (stale() || view.shown >= width) return;
    }
    view.img.src = url;
    view.shown = width;
    loader.cache.keep = new Set([loader.key(number, width, null)]);
  };
  const cached = [...loader.cache.map.keys()].filter((k) => k.startsWith(`${number}:`) && k.endsWith(":page"))
    .map((k) => Number(k.split(":")[1])).sort((a, b) => b - a)[0];
  try {
    if (cached) await show(loader.cache.get(loader.key(number, cached, null)), cached);
    if (!view.shown && plan.preview < plan.sharp) await show(await loader.load(number, plan.preview), plan.preview);
    await show(await loader.load(number, plan.sharp), plan.sharp);
  } catch (err) {
    if (stale() || view.shown) return;
    view.el.classList.add("broken");
    clear(view.el);
    state.sheet = null;
    view.el.append(err && err.status === 404
      ? failure("Die Originaldatei ist nicht mehr da.", "Der Text bleibt durchsuchbar.", [["Text zeigen", () => toggleSide(state, "text", null)]])
      : failure("Diese Seite konnte nicht dargestellt werden.", "Ihr Text ist trotzdem da.", [["Text zeigen", () => toggleSide(state, "text", null)]]));
    return;
  }
  if (stale()) return;
  if (plan.tiles) updateTiles(state);
  // the neighbours, at low priority and not cancelled by the next page change: flipping ahead is instant
  const ahead = Math.min(plan.sharp, 1800);
  for (const n of [number + 1, number - 1]) {
    if (n < 1 || n > state.total || loader.cache.get(loader.key(n, ahead, null))) continue;
    loader.load(n, ahead, { prio: 1, controller: null }).catch(() => {});
  }
}

/* Zoomed beyond the sharp cap: the visible region in tiles at the exact scale, laid over the page. */
function updateTiles(state) {
  const view = state.sheet;
  if (!view || !view.plan || active !== state) return;
  const scale = view.plan.tiles ? view.plan.tileScale : 0;
  for (const tile of [...view.tiles.children]) if (Number(tile.dataset.scale) !== scale) tile.remove();
  if (!scale || !view.el.classList.contains("loaded")) return;
  const visible = visibleFraction(view.el.getBoundingClientRect(), state.page.getBoundingClientRect());
  if (!visible.w || !visible.h) return;
  const generation = state.loader.seq;
  const present = new Set([...view.tiles.children].map((t) => t.dataset.key));
  for (const tile of visibleTiles(visible, scale, view.ratio)) {
    if (present.has(tile.key)) continue;
    const [x, y, w, h] = tile.clip;
    const img = el("img", { class: "vw-tile", alt: "", draggable: "false", "data-key": tile.key, "data-scale": String(scale),
                            style: { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` } });
    view.tiles.append(img);
    state.loader.load(view.unit.number, scale, { clip: tile.clip }).then((url) => {
      if (!url || state.sheet !== view || state.loader.seq !== generation) { img.remove(); return; }
      img.src = url;
    }).catch(() => img.remove());
  }
  // bounded: the oldest tiles go first (their bitmaps stay in the LRU)
  for (const tile of [...view.tiles.children].slice(0, Math.max(0, view.tiles.children.length - 24))) tile.remove();
}

/* A new zoom or pane width: the same page, re-laid-out; the scaled bitmap stays until the sharp one replaces it. */
function rezoomPdf(state, { top = false } = {}) {
  const view = state.sheet;
  const page = state.page;
  const cx = (page.scrollLeft + page.clientWidth / 2) / Math.max(1, page.scrollWidth);
  const cy = (page.scrollTop + page.clientHeight / 2) / Math.max(1, page.scrollHeight);
  view.el.style.width = `${sheetWidth(state)}px`;
  clear(view.tiles);
  if (top) page.scrollTop = 0;
  else {
    page.scrollLeft = Math.max(0, cx * page.scrollWidth - page.clientWidth / 2);
    page.scrollTop = Math.max(0, cy * page.scrollHeight - page.clientHeight / 2);
  }
  paintPdf(state);
}

async function markMatches(state, unit, marks) {
  const token = state.renderToken;
  let answer = { ok: true, matches: [] };
  if (state.phrases.length) {
    // one question per place: zooming or resizing never asks the engine again
    const key = `${unit.number}|${state.at}|${state.phrases.join("|")}`;
    answer = state.matchCache.get(key) || await api("/api/study/locate", { id: state.id, page: unit.number, phrases: state.phrases, at: state.at });
    if (answer && answer.ok !== false) {
      state.matchCache.set(key, answer);
      if (state.matchCache.size > 40) state.matchCache.delete(state.matchCache.keys().next().value);
    }
  }
  if (token !== state.renderToken || active !== state) return;
  const matches = (answer && answer.matches) || [];
  let primary = null;
  const figure = state.fig != null && state.figUnit === unit.number
    ? (state.figures || []).find((f) => Number(f.number) === Number(state.fig) && Array.isArray(f.box)) : null;
  if (figure) {
    const [x, y, w, h] = figure.box;
    const region = el("span", { class: "vw-mark vw-figure" + (matches.length ? "" : " primary"), title: figure.caption || `Abbildung ${figure.number}`,
                                style: { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` } });
    marks.append(region);
    if (!matches.length) primary = region;
  }
  if (!matches.length && !figure) {
    if (answer && answer.ok !== false && state.focus) marks.parentElement?.setAttribute("data-unmarked", "true");
    return;
  }
  for (const match of matches) {
    for (const [x, y, w, h] of match.rects) {
      const mark = el("span", { class: "vw-mark" + (match.primary ? " primary" : ""), title: match.phrase,
                                style: { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` } });
      marks.append(mark);
      if (match.primary && !primary) primary = mark;
    }
  }
  const count = matches.length;
  if (unit.kind) state.title.querySelector(".vw-where").textContent = `${whereLabel(unit)}${count > 1 ? ` · ${count} Treffer` : ""}`;
  const reduced = document.body.classList.contains("reduced-motion") || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const target = primary || marks.firstElementChild;
  requestAnimationFrame(() => target?.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" }));
}

function renderNativePdf(state) {
  // without the page engine: the browser's own PDF viewer, opened at the page
  state.page.append(el("iframe", { class: "vw-frame", title: state.doc.filename,
    src: audioUrl(`/api/study/original?id=${encodeURIComponent(state.id)}`) + `#page=${state.unit}` }));
}

function renderImage(state) {
  const sheet = sheetShell(state, {}, { width: sheetWidth(state), ratio: 0, label: "Bild wird geladen …" });
  const img = el("img", { class: "vw-img", alt: state.doc.filename });
  img.addEventListener("load", () => { sheet.classList.add("loaded"); sheet.querySelector(".vw-sheet-loading")?.remove(); }, { once: true });
  img.addEventListener("error", () => { clear(sheet); sheet.append(failure("Das Bild konnte nicht geladen werden.", "")); }, { once: true });
  img.src = audioUrl(`/api/study/original?id=${encodeURIComponent(state.id)}`);
  const marks = el("div", { class: "vw-marks", "aria-hidden": "true" });
  sheet.append(img, marks);
  state.page.append(sheet);
  // a photographed or handwritten page: the recognised words are marked on the image itself
  if (state.phrases.length) markMatches(state, { number: 1 }, marks);
}

/* ---------------------------------------------------------------- slides */

function renderSlide(state, unit) {
  const slides = state.slides || { state: "unavailable" };
  const ratio = unit.width && unit.height ? unit.width / unit.height : 16 / 9;
  if (slides.state === "ready") {
    const cssWidth = sheetWidth(state, 1080);
    const sheet = sheetShell(state, unit, { width: cssWidth, ratio, label: `Folie ${unit.number} wird geladen …` });
    sheet.classList.add("slide");
    const img = el("img", { class: "vw-img", alt: `${state.doc.filename}, Folie ${unit.number}${unit.title ? ": " + unit.title : ""}`, draggable: "false" });
    const marks = el("div", { class: "vw-marks", "aria-hidden": "true" });
    sheet.append(img, marks);
    img.addEventListener("load", () => { sheet.classList.add("loaded"); sheet.querySelector(".vw-sheet-loading")?.remove(); }, { once: true });
    img.addEventListener("error", () => { state.slides = { state: "failed" }; renderUnit(state, { keepScroll: true }); }, { once: true });
    img.src = audioUrl(`/api/study/page.png?id=${encodeURIComponent(state.id)}&page=${unit.number}`);
    state.page.append(sheet);
    const notes = (unit.blocks || []).filter((b) => b.kind === "notes");
    if (notes.length) {
      state.page.append(el("aside", { class: "vw-slide-notes", style: { width: `${cssWidth}px` } }, el("span", { class: "vw-notes-label", text: "Notizen" }),
        ...notes.map((b) => highlighted(el("p"), b.text, state.phrases))));
    }
    if (state.phrases.length) markMatches(state, unit, marks);
    return;
  }
  const note = {
    rendering: "Folienbild wird erstellt …", pending: "Folienbild wird erstellt …",
    deferred: slides.note || "PowerPoint ist gerade geöffnet – das Folienbild entsteht, sobald es geschlossen ist.",
    failed: slides.note || "Das Folienbild ließ sich nicht erstellen; die Folie steht hier als Text.",
    unavailable: slides.note || "Ohne PowerPoint zeigt Studium Folien als Text.",
  }[slides.state] || "";
  renderText(state, unit, { slide: true, note });
  if (["rendering", "pending", "deferred"].includes(slides.state)) watchSlides(state);
}

function watchSlides(state) {
  if (state.slideWatch) return;
  if (state.slides?.state === "pending") api("/api/study/render_slides", { id: state.id });
  state.slideWatch = setInterval(async () => {
    if (active !== state) { clearInterval(state.slideWatch); return; }
    const info = await api("/api/study/document", { id: state.id });
    const next = info && info.slides;
    if (next && next.state !== state.slides?.state) {
      state.slides = next;
      if (!["rendering", "pending", "deferred"].includes(next.state)) { clearInterval(state.slideWatch); state.slideWatch = 0; }
      renderUnit(state, { keepScroll: true });
    }
  }, 2500);
  state.disposers.push(() => clearInterval(state.slideWatch));
}

/* ---------------------------------------------------------------- text marking */

function highlighted(node, text, phrases, { primaryAt = null, offset = 0 } = {}) {
  const spans = phrases.length ? findPhrases(text, phrases) : [];
  const primary = primaryAt != null ? primarySpan(spans, primaryAt, offset) : null;
  let at = 0;
  for (const span of spans) {
    const [start, end, rank] = span;
    if (start > at) node.append(document.createTextNode(text.slice(at, start)));
    node.append(el("mark", { class: "vw-hit" + (rank === 0 ? " focus" : "") + (span === primary ? " primary" : ""), text: text.slice(start, end) }));
    at = end;
  }
  node.append(document.createTextNode(text.slice(at)));
  return node;
}

function renderText(state, unit, { slide = false, note = "" } = {}) {
  const article = el("article", { class: "vw-text" + (slide ? " slide" : "") });
  if (note) article.append(el("div", { class: "vw-quiet", role: "status" }, slide && /erstellt/.test(note) ? el("span", { class: "vw-loading-line", "aria-hidden": "true" }) : null, el("span", { text: note })));
  const blocks = unit.blocks && unit.blocks.length ? unit.blocks : [{ text: unit.text, kind: "paragraph" }];
  for (const block of blocks) {
    const tag = block.kind === "heading" || block.kind === "slide_title" ? "h2" : block.kind === "notes" ? "aside" : block.kind === "table" ? "pre" : "p";
    const node = el(tag, { class: `vw-block ${block.kind}` });
    if (block.kind === "notes") node.append(el("span", { class: "vw-notes-label", text: "Notizen" }));
    node.append(highlighted(el("span"), block.text || "", state.phrases, { primaryAt: state.at, offset: block.char_start || 0 }));
    article.append(node);
  }
  if (!unit.text) article.append(el("p", { class: "vw-note", text: "Dieser Abschnitt enthält keinen Text." }));
  state.page.append(article);
  const hit = article.querySelector(".vw-hit.primary") || article.querySelector(".vw-hit");
  if (hit) requestAnimationFrame(() => hit.scrollIntoView({ block: "center" }));
}

/* ---------------------------------------------------------------- Word, Markdown, text: one reading column */

async function mountSections(state) {
  const loading = el("div", { class: "vw-loading", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: "Dokument wird geladen …" }));
  state.page.append(loading);
  const info = await api("/api/study/sections", { id: state.id });
  loading.remove();
  if (active !== state) return;
  if (!info || info.ok === false) {
    state.page.append(failure("Dieses Dokument lässt sich gerade nicht öffnen.", (info && info.error) || "", [["Erneut versuchen", () => { clear(state.page); mountSections(state); }]]));
    return;
  }
  state.sections = info.units || [];
  const article = el("article", { class: "vw-text vw-document" });
  const outline = state.outline;
  clear(outline);
  outline.append(el("div", { class: "vw-outline-head", text: "Gliederung" }));
  for (const unit of state.sections) {
    const section = el("section", { class: "vw-section", "data-unit": String(unit.number), "aria-label": unit.title || `Abschnitt ${unit.number}` });
    const target = unit.number === state.unit;
    for (const block of unit.blocks || []) {
      const level = block.kind === "heading" ? Math.min(4, Math.max(2, (block.heading_path || []).length + 1)) : 0;
      const tag = level ? `h${level}` : block.kind === "table" ? "pre" : "p";
      const node = el(tag, { class: `vw-block ${block.kind}`, "data-paragraph": block.paragraph != null ? String(block.paragraph) : "" });
      // matches are marked in the section the result points at; the rest of the document reads clean
      highlighted(node, block.text || "", target ? state.phrases : [], { primaryAt: state.at, offset: block.char_start || 0 });
      section.append(node);
    }
    article.append(section);
    const depth = Math.max(0, String(unit.title || "").split(" › ").length - 1);
    const leaf = String(unit.title || `Abschnitt ${unit.number}`).split(" › ").pop();
    outline.append(el("button", { class: "vw-outline-item", "data-unit": String(unit.number), style: { paddingLeft: `${12 + depth * 14}px` },
      title: unit.title || "", text: leaf, onClick: () => go(state, unit.number) }));
  }
  if (!info.complete) article.append(el("p", { class: "vw-note", text: "Das Dokument ist sehr lang; hier steht der Anfang. Die Suche findet alles." }));
  state.page.append(article);
  state.article = article;
  requestAnimationFrame(() => scrollToSection(state, { smooth: false }));
  let timer = 0;
  const onScroll = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (state.scrolling) return;
      const top = state.page.getBoundingClientRect().top + 80;
      let current = state.sections[0]?.number || 1;
      for (const section of article.querySelectorAll(".vw-section")) {
        if (section.getBoundingClientRect().top <= top) current = Number(section.dataset.unit);
      }
      if (current !== state.unit) { state.unit = current; state.focus = ""; state.phrases = []; setCurrentSection(state); writeAddress(state); remember(state); }
    }, 140);
  };
  state.page.addEventListener("scroll", onScroll, { passive: true });
  state.disposers.push(() => state.page.removeEventListener("scroll", onScroll));
  setCurrentSection(state);
  remember(state);
}

function setCurrentSection(state) {
  state.pageInput.value = String(state.unit);
  const unit = (state.sections || []).find((u) => u.number === state.unit);
  state.current = unit ? { ...unit, kind: "section", text: (unit.blocks || []).map((b) => b.text).join("\n\n") } : null;
  state.title.querySelector(".vw-where").textContent = unit ? whereLabel({ ...unit, kind: "section" }) : "";
  for (const item of state.outline?.querySelectorAll(".vw-outline-item") || []) {
    const on = Number(item.dataset.unit) === state.unit;
    item.classList.toggle("on", on);
    item.setAttribute("aria-current", on ? "location" : "false");
  }
  if (state.side === "ask") renderAsk(state);
}

function scrollToSection(state, { smooth }) {
  const article = state.article;
  if (!article) return;
  const section = article.querySelector(`.vw-section[data-unit="${state.unit}"]`);
  if (!section) return;
  if (state.phrases.length) {
    // a newly chosen place: mark it (and only it)
    for (const node of article.querySelectorAll(".vw-hit")) node.replaceWith(document.createTextNode(node.textContent));
    for (const block of section.querySelectorAll(".vw-block")) {
      const text = block.textContent;
      clear(block);
      highlighted(block, text, state.phrases, { primaryAt: state.at });
    }
  }
  const paragraph = state.para != null ? section.querySelector(`[data-paragraph="${state.para}"]`) : null;
  const target = section.querySelector(".vw-hit.primary") || paragraph?.querySelector(".vw-hit") || paragraph || section.querySelector(".vw-hit") || section;
  for (const node of article.querySelectorAll(".vw-block.target")) node.classList.remove("target");
  paragraph?.classList.add("target");
  const reduced = document.body.classList.contains("reduced-motion") || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  state.scrolling = true;
  target.scrollIntoView({ block: target === section ? "start" : "center", behavior: smooth && !reduced ? "smooth" : "auto" });
  setTimeout(() => { state.scrolling = false; }, smooth && !reduced ? 700 : 60);
  setCurrentSection(state);
  remember(state);
}

function toggleOutline(state, button) {
  const closed = state.root.querySelector(".vw-body").classList.toggle("outline-closed");
  button.setAttribute("aria-pressed", String(!closed));
}

/* ---------------------------------------------------------------- side panel */

function toggleSide(state, which, button) {
  const opening = state.side !== which;
  state.side = opening ? which : "";
  for (const b of state.root.querySelectorAll(".vw-tool, .vw-ask")) if (b.textContent !== "Gliederung") b.setAttribute("aria-pressed", "false");
  if (opening && button) button.setAttribute("aria-pressed", "true");
  if (state.side) openSide(state);
  else state.root.querySelector(".vw-side").hidden = true;
}

function openSide(state) {
  const side = state.root.querySelector(".vw-side");
  side.hidden = false;
  if (state.side === "search") renderSearch(state);
  if (state.side === "text") renderTextSide(state);
  if (state.side === "ask") renderAsk(state);
}

function closeSide(state) {
  state.side = "";
  state.root.querySelector(".vw-side").hidden = true;
  for (const b of state.root.querySelectorAll(".vw-tool, .vw-ask")) if (b.textContent !== "Gliederung") b.setAttribute("aria-pressed", "false");
}

function sideHead(state, title) {
  return el("header", { class: "vw-side-head" }, el("h3", { text: title }),
    el("button", { class: "vw-side-close", "aria-label": "Schließen", text: "×", onClick: () => closeSide(state) }));
}

function renderSearch(state) {
  const side = state.root.querySelector(".vw-side");
  clear(side);
  const results = el("div", { class: "vw-hits", role: "list" });
  const input = el("input", { class: "vw-search", placeholder: "In diesem Dokument suchen …", "aria-label": "In diesem Dokument suchen",
    onKeydown: async (e) => {
      if (e.key !== "Enter" || !e.target.value.trim()) return;
      clear(results);
      results.append(el("div", { class: "vw-quiet", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: "Sucht …" })));
      const found = await api("/api/study/search", { query: e.target.value.trim(), document_id: state.id, limit: 1 });
      clear(results);
      if (!found || found.ok === false) { results.append(el("div", { class: "vw-empty", text: "Die Suche ist gerade nicht erreichbar." })); return; }
      const top = (found.results && found.results[0]) || null;
      const hits = top ? [top.best, ...(top.more || [])] : [];
      if (!hits.length) { results.append(el("div", { class: "vw-empty", text: "Nichts gefunden." })); return; }
      for (const hit of hits) {
        results.append(el("button", { class: "vw-hit-row", role: "listitem",
          onClick: () => go(state, hit.location.unit, { focus: hit.focus || found.topic, phrases: hit.phrases, at: hit.at, para: hit.location.paragraph }) },
          el("span", { class: "vw-hit-where", text: hit.location.label }), el("span", { class: "vw-hit-text", text: hit.excerpt })));
      }
    } });
  side.append(sideHead(state, "Suchen"), input, results);
  setTimeout(() => input.focus(), 20);
}

function renderTextSide(state) {
  const side = state.root.querySelector(".vw-side");
  clear(side);
  const unit = state.current || {};
  const body = el("div", { class: "vw-side-text" });
  highlighted(body, unit.text || "", state.phrases, { primaryAt: state.at });
  side.append(sideHead(state, `Text · ${whereLabel(unit) || unitWord(state) + " " + state.unit}`),
    el("p", { class: "vw-hint", text: "Markiere einen Abschnitt und frag ZEUS dazu." }), body);
}

function renderAsk(state) {
  const side = state.root.querySelector(".vw-side");
  const keep = side.querySelector(".vw-answer");
  const typed = side.querySelector(".vw-ask-input")?.value || "";
  clear(side);
  const unit = state.current || {};
  const kind = unit.kind === "slide" ? "slide" : unit.kind === "section" ? "section" : "page";
  const noun = { page: "Diese Seite", slide: "Diese Folie", section: "Dieser Abschnitt" }[kind];
  const scope = el("div", { class: "vw-scope" },
    el("span", { text: state.selection ? "Nur die Markierung" : noun }),
    state.selection ? el("q", { text: state.selection.length > 160 ? state.selection.slice(0, 160) + " …" : state.selection })
                    : el("span", { class: "vw-scope-where", text: whereLabel(unit) }),
    state.selection ? el("button", { class: "vw-link", text: "Markierung aufheben", onClick: () => { state.selection = ""; window.getSelection()?.removeAllRanges(); renderAsk(state); } }) : null);
  const answer = keep || el("div", { class: "vw-answer md", "aria-live": "polite" });
  const input = el("textarea", { class: "vw-ask-input", rows: 2, placeholder: `Frag ZEUS zu ${kind === "slide" ? "dieser Folie" : kind === "section" ? "diesem Abschnitt" : "dieser Seite"} …`,
                                 "aria-label": "Frage an ZEUS", value: typed });
  const send = async () => {
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    clear(answer);
    answer.append(el("div", { class: "vw-q", text: question }),
      el("div", { class: "vw-thinking", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: `ZEUS liest ${kind === "slide" ? "die Folie" : kind === "section" ? "den Abschnitt" : "die Seite"} …` })));
    const chat = await import("./chat.js");
    const reply = await chat.send(question, "text", { study_context: { document_id: state.id, unit: state.unit, selection: state.selection }, stayInView: true });
    if (reply && reply.ok === false) {
      answer.querySelector(".vw-thinking")?.remove();
      answer.append(el("div", { class: "vw-empty", text: "ZEUS ist gerade nicht erreichbar. Versuch es gleich noch einmal." }));
      return;
    }
    state.askRequest = (reply && reply.request_id) || "";
    state.streamText = "";
    listenForAnswer(state, answer, question);
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  side.append(sideHead(state, "Ask ZEUS"), scope,
    el("div", { class: "vw-ask-row" }, input, el("button", { class: "vw-send", "aria-label": "Fragen", text: "↑", onClick: send })),
    el("div", { class: "vw-quick" }, ...ASK[kind].map((q) => el("button", { class: "vw-chip", text: q, onClick: () => { input.value = q; send(); } }))),
    answer);
  if (!keep) setTimeout(() => input.focus(), 20);
}

function listenForAnswer(state, answer, question) {
  delete answer.dataset.done;
  const offToken = bus.on("token", (p) => {
    if (p._replay || !state.askRequest || active !== state) return;
    if (p.reset) {
      // the route that was answering stopped mid-answer and another one answers: say so instead of going blank
      state.streamText = "";
      answer.querySelector(".vw-a")?.remove();
      if (!answer.querySelector(".vw-thinking")) {
        answer.append(el("div", { class: "vw-thinking", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: "ZEUS setzt neu an …" })));
      }
      return;
    }
    answer.querySelector(".vw-thinking")?.remove();
    state.streamText += p.text || "";
    let body = answer.querySelector(".vw-a");
    if (!body) { body = el("div", { class: "vw-a" }); answer.append(body); }
    body.innerHTML = renderPartial(state.streamText);
  });
  const offMessage = bus.on("message", (p) => {
    const meta = p.meta || {};
    if (p._replay || !state.askRequest || meta.request_id !== state.askRequest) return;
    clear(answer);
    answer.dataset.done = "1";
    answer.append(el("div", { class: "vw-q", text: question }), el("div", { class: "vw-a", html: renderMarkdown(p.text || "") }));
    state.askRequest = "";
    offToken?.();
    offMessage?.();
  });
  state.disposers.push(() => { offToken?.(); offMessage?.(); });
}
