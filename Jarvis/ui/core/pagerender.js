/* Sharp pages, quickly: the rendering plan of the Studium PDF viewer, and its bounded bitmap cache.

   The principle: never render more pixels than the screen shows.

     preview   a small render of the page, shown at once
     sharp     the page at CSS width × devicePixelRatio (capped), swapped in when it is decoded
     tiles     beyond the cap (zoomed in), only the visible region, in tiles at the exact scale

   Requests carry the viewer's session and its view generation (it rises with every page change
   or zoom, and every request of one view shares it); the server drops a queued render of a
   generation the viewer has already moved past, and the viewer aborts its own fetches
   when the page changes.  Neighbour pages are fetched ahead at low priority.  Bitmaps are
   object URLs in a small LRU; the evicted ones are revoked.

   Pure functions + one small class, no DOM: node tests cover them (ui/tests/study_pdf.test.mjs). */

export const SHARP_CAP = 3200;     // widest full-page render (px)
export const TILE_PX = 1024;       // tile edge at the render scale
export const TILE_MAX_PAGE_PX = 12000;

const step = (px, by) => Math.max(by, Math.round(px / by) * by);

/* Widths for a page shown at `cssWidth` CSS pixels. */
export function renderPlan(cssWidth, dpr = 1) {
  const exact = Math.max(1, cssWidth) * Math.max(1, Math.min(3, dpr || 1));
  const sharp = Math.min(SHARP_CAP, step(exact, 100));
  const preview = Math.min(sharp, step(Math.min(cssWidth, 900), 100));
  const tiles = exact > SHARP_CAP * 1.05;
  return { preview, sharp, tiles, tileScale: tiles ? Math.min(TILE_MAX_PAGE_PX, step(exact, 200)) : 0 };
}

/* The tiles (page fractions) that cover `visible` ({x, y, w, h} fractions) at a page render width of `pagePx`. */
export function visibleTiles(visible, pagePx, aspect, margin = 0.02) {
  if (!pagePx) return [];
  const heightPx = pagePx / Math.max(0.05, aspect);   // aspect = width / height
  const nx = Math.max(1, Math.ceil(pagePx / TILE_PX));
  const ny = Math.max(1, Math.ceil(heightPx / TILE_PX));
  const x0 = Math.max(0, visible.x - margin), y0 = Math.max(0, visible.y - margin);
  const x1 = Math.min(1, visible.x + visible.w + margin), y1 = Math.min(1, visible.y + visible.h + margin);
  const tiles = [];
  for (let j = Math.floor(y0 * ny); j < Math.min(ny, Math.ceil(y1 * ny)); j++) {
    for (let i = Math.floor(x0 * nx); i < Math.min(nx, Math.ceil(x1 * nx)); i++) {
      tiles.push({ key: `${pagePx}:${i}:${j}`, clip: [i / nx, j / ny, 1 / nx, 1 / ny].map((v) => Math.round(v * 10000) / 10000) });
    }
  }
  return tiles;
}

/* The part of a sheet (a rectangle on screen) inside a scrolling viewport, as page fractions. */
export function visibleFraction(sheetRect, viewportRect) {
  const w = Math.max(1, sheetRect.width), h = Math.max(1, sheetRect.height);
  const left = Math.max(sheetRect.left, viewportRect.left), right = Math.min(sheetRect.right, viewportRect.right);
  const top = Math.max(sheetRect.top, viewportRect.top), bottom = Math.min(sheetRect.bottom, viewportRect.bottom);
  if (right <= left || bottom <= top) return { x: 0, y: 0, w: 0, h: 0 };
  return { x: (left - sheetRect.left) / w, y: (top - sheetRect.top) / h, w: (right - left) / w, h: (bottom - top) / h };
}

/* A small LRU of object URLs; `keep` protects what is on screen. */
export class BitmapCache {
  constructor(limit = 16, revoke = (url) => URL.revokeObjectURL(url)) {
    this.limit = limit;
    this.revoke = revoke;
    this.map = new Map();
    this.keep = new Set();
  }
  get(key) {
    if (!this.map.has(key)) return null;
    const value = this.map.get(key);
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }
  set(key, url) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, url);
    for (const [old, value] of this.map) {
      if (this.map.size <= this.limit) break;
      if (this.keep.has(old)) continue;
      this.map.delete(old);
      this.revoke(value);
    }
  }
  clear() {
    for (const [key, value] of this.map) if (!this.keep.has(key)) { this.map.delete(key); this.revoke(value); }
  }
}

/* Loads page bitmaps for one viewer: sequence numbers, abort on page change, the cache. */
export class PageLoader {
  constructor({ urlFor, fetchImpl = (...args) => fetch(...args), cache = new BitmapCache() }) {  // unbound: window.fetch called as a method throws "Illegal invocation"
    this.urlFor = urlFor;            // (page, width, clip, seq, prio, session) -> url
    this.fetchImpl = fetchImpl;
    this.cache = cache;
    this.seq = 0;
    this.controller = null;
    this.session = Math.random().toString(36).slice(2, 10);
  }
  /* A new page (or zoom) is wanted: everything still loading for the old one stops. */
  begin() {
    this.seq += 1;
    this.controller?.abort();
    this.controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    return this.controller;
  }
  key(page, width, clip) {
    return `${page}:${width}:${clip ? clip.join(",") : "page"}`;
  }
  async load(page, width, { clip = null, prio = 0, controller = this.controller } = {}) {
    const key = this.key(page, width, clip);
    const hit = this.cache.get(key);
    if (hit) return hit;
    let response;
    try {
      response = await this.fetchImpl(this.urlFor(page, width, clip, this.seq, prio, this.session), controller ? { signal: controller.signal } : {});
    } catch (err) {
      if (err && err.name === "AbortError") return null;
      throw err;
    }
    if (response.status === 204) return null;          // superseded on the server: the viewer moved on
    if (!response.ok) { const error = new Error(`HTTP ${response.status}`); error.status = response.status; throw error; }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    this.cache.set(key, url);
    return url;
  }
}
