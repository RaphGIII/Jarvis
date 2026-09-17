/* Studium, "Sterne": the library as a quiet sky.

   Every document is one point of light, clustered by subject (the layout comes
   from the server, study/galaxy.py, and is stable: the map never moves by
   itself).  One canvas, no DOM per document; drawn only when something changes
   -- a pan, a zoom, a hover, new data, or a short eased camera move.

   Hover: a small note (title, kind, subject, size, index state).  Click: a thin
   ring and the document's strongest connections as hairlines.  Double-click,
   Enter or "Öffnen": the viewer.  Right-click or "···": the few things to do
   with it.  Keyboard: Tab to the sky, arrows move between documents, Enter
   opens, Escape lets go. */

import { el, clear } from "../core/dom.js";
import { api } from "../core/api.js";
import * as sf from "../core/starfield.js";

const IVORY = "235,231,221";
const CHAMPAGNE = "200,176,132";
const MUTED_WARM = "185,138,120";
const LABEL = "172,167,156";
const FONT = '"Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
const HIT_PX = 12;
const EDGES_SHOWN = 6;
const MAX_DOC_LABELS = 160;

function reducedMotion() {
  try {
    return document.body.classList.contains("reduced-motion") || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch { return false; }
}

function ellipsis(text, max) {
  const s = String(text || "");
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

export function mountGalaxy(container, callbacks = {}) {
  const canvas = el("canvas", { class: "sky-canvas", tabindex: "0", role: "application", "aria-roledescription": "Sternenkarte",
                               "aria-label": "Sterne: dein Material als Karte. Pfeiltasten wählen ein Dokument, Enter öffnet, Escape hebt die Auswahl auf." });
  const tip = el("div", { class: "sky-tip", hidden: true, "aria-hidden": "true" });
  const menu = el("div", { class: "sky-menu", hidden: true, role: "menu", "aria-label": "Dokument" });
  const bar = el("div", { class: "sky-bar", hidden: true });
  const focusChip = el("div", { class: "sky-focus", hidden: true });
  const empty = el("div", { class: "sky-empty", hidden: true });
  const live = el("span", { class: "sky-live", "aria-live": "polite" });
  const fitButton = el("button", { class: "sky-fit", type: "button", text: "Alles zeigen", onClick: () => { fitAll(true); canvas.focus(); } });
  const wrap = el("div", { class: "sky" }, canvas, focusChip, fitButton, bar, tip, menu, empty, live);
  container.append(wrap);
  const ctx = canvas.getContext("2d");

  let nodes = [];
  let clusters = [];
  let byId = new Map();
  let clusterById = new Map();
  let links = new Map();          // id -> [{ node, weight, kind }] strongest first
  let grid = null;
  let camera = { x: 0, y: 0, zoom: 300 };
  let fitZoom = 300;
  let spacing = 0.05;
  let view = { w: 0, h: 0 };
  let dpr = 1;
  let hovered = null;
  let selected = null;
  let emphasis = null;            // Set of ids, or null
  let fitted = false;
  let pendingFit = null;          // a camera move waiting for the canvas to have a size
  let frame = 0;
  let anim = 0;
  let destroyed = false;
  let lastDrawMs = 0;

  /* ---------------------------------------------------------------- data */

  function load(payload) {
    const keepSelected = selected && selected.id;
    nodes = (payload && payload.nodes) || [];
    clusters = (payload && payload.clusters) || [];
    byId = new Map(nodes.map((n) => [n.id, n]));
    clusterById = new Map(clusters.map((c) => [c.id, c]));
    const order = [...nodes].sort((a, b) => (a.recency || 0) - (b.recency || 0));
    order.forEach((n, i) => { n._rec = nodes.length > 1 ? i / (nodes.length - 1) : 1; });
    for (const c of clusters) c._extent = 0;
    for (const n of nodes) {
      n._cluster = clusterById.get(n.topic_cluster) || clusterById.get(n.cluster) || null;
      // a label sits above where the documents are, not above the empty rim of a disc drawn for growth
      for (const c of [clusterById.get(n.cluster), clusterById.get(n.topic_cluster)]) {
        if (c) c._extent = Math.max(c._extent, Math.hypot(n.x - c.x, n.y - c.y));
      }
    }
    // the typical distance between neighbouring documents (world units): points are sized against it, so a dense sky stays points
    const area = clusters.filter((c) => c.kind === "subject").reduce((sum, c) => sum + Math.PI * c.r * c.r, 0);
    spacing = nodes.length ? Math.sqrt(area / nodes.length) || 0.05 : 0.05;
    links = new Map();
    for (const e of (payload && payload.edges) || []) {
      const a = byId.get(e.source), b = byId.get(e.target);
      if (!a || !b) continue;
      if (!links.has(a.id)) links.set(a.id, []);
      if (!links.has(b.id)) links.set(b.id, []);
      links.get(a.id).push({ node: b, weight: e.weight, kind: e.kind });
      links.get(b.id).push({ node: a, weight: e.weight, kind: e.kind });
    }
    for (const list of links.values()) list.sort((x, y) => y.weight - x.weight);
    grid = sf.buildGrid(nodes, sf.gridCellFor(nodes));
    selected = keepSelected ? byId.get(keepSelected) || null : null;
    hovered = null;
    if (emphasis) emphasis = new Set([...emphasis].filter((id) => byId.has(id)));
    empty.hidden = nodes.length > 0;
    if (!nodes.length) {
      clear(empty);
      empty.append(el("p", { text: "Noch keine Sterne." }), el("p", { class: "sky-empty-sub", text: "Jedes Dokument, das du hinzufügst, wird hier ein Punkt – geordnet nach Fach." }));
    }
    const summary = `${nodes.length === 1 ? "1 Dokument" : `${nodes.length} Dokumente`} in ${countSubjects()}`;
    canvas.setAttribute("aria-label", `Sterne: ${summary}. Pfeiltasten wählen ein Dokument, Enter öffnet, Escape hebt die Auswahl auf.`);
    updateBar();
    if (!fitted && view.w > 0) fitAll(false);
    invalidate();
  }

  function countSubjects() {
    const n = clusters.filter((c) => c.kind === "subject").length;
    return n === 1 ? "1 Fach" : `${n} Fächern`;
  }

  async function refresh() {
    const data = await api("/api/study/galaxy");
    if (destroyed) return;
    if (!data || data.ok === false) {
      nodes = [];
      empty.hidden = false;
      clear(empty);
      empty.append(el("p", { text: data && data.transport ? "ZEUS ist gerade nicht erreichbar." : "Die Sterne lassen sich gerade nicht zeichnen." }));
      invalidate();
      return;
    }
    load(data);
  }

  /* ---------------------------------------------------------------- camera */

  function setCamera(next) {
    camera = { x: next.x, y: next.y, zoom: sf.clampZoom(next.zoom) };
    invalidate();
  }

  function moveCamera(target, animate) {
    cancelAnimationFrame(anim);
    anim = 0;
    if (!animate || reducedMotion()) { setCamera(target); return; }
    const from = { ...camera };
    const started = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - started) / 480);
      camera = sf.lerpCamera(from, target, t);
      draw();
      anim = t < 1 ? requestAnimationFrame(step) : 0;
    };
    anim = requestAnimationFrame(step);
  }

  function fitAll(animate) {
    if (view.w <= 0) { pendingFit = "all"; return; }
    const discs = clusters.filter((c) => c.kind === "subject");
    const target = sf.fitCamera(discs.length ? discs : nodes, view, { margin: 56, maxZoom: Math.min(view.w, view.h) * 0.9 });
    fitZoom = target.zoom;
    fitted = true;
    moveCamera(target, animate);
  }

  function fitTo(list, animate) {
    if (!list.length) return;
    if (view.w <= 0) { pendingFit = list; return; }
    const pad = 0.6 / Math.max(fitZoom, 1) * 60;   // a little sky around even a single point
    const target = sf.fitCamera(list.map((n) => ({ x: n.x, y: n.y, r: pad })), view, { margin: 72, maxZoom: fitZoom * 4 });
    moveCamera(target, animate);
  }

  /* ---------------------------------------------------------------- drawing */

  function invalidate() {
    if (frame || destroyed) return;
    frame = requestAnimationFrame(() => { frame = 0; draw(); });
  }

  function radiusOf(n) {
    // a fifth of the gap to the neighbours, within calm bounds; larger documents a little larger
    const base = Math.min(2.6, Math.max(0.9, spacing * camera.zoom * 0.2));
    return base * (0.8 + 0.55 * (n.weight || 0)) * (emphasis && emphasis.has(n.id) ? 1.4 : 1);
  }

  function alphaOf(n) {
    let a = 0.3 + 0.38 * (n._rec ?? 1) + 0.12 * (n.weight || 0);
    if (n.state === "queued") a *= 0.4;
    return sf.emphasisAlpha(n.id, emphasis, Math.min(0.82, a));
  }

  function draw() {
    if (destroyed || view.w <= 0) return;
    const t0 = performance.now();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);
    if (!nodes.length) { lastDrawMs = performance.now() - t0; return; }
    const b = sf.viewBounds(camera, view, 30);
    const focus = selected || hovered;

    // the focused document's strongest connections: hairlines, nothing else
    for (const target of [selected, hovered !== selected ? hovered : null]) {
      if (!target) continue;
      const p = sf.worldToScreen(camera, view, target.x, target.y);
      const list = (links.get(target.id) || []).slice(0, EDGES_SHOWN);
      ctx.lineWidth = 0.8;
      for (const link of list) {
        const q = sf.worldToScreen(camera, view, link.node.x, link.node.y);
        ctx.strokeStyle = `rgba(${CHAMPAGNE},${(0.16 + 0.34 * Math.min(1, link.weight)) * (target === selected ? 1 : 0.7)})`;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
        ctx.stroke();
      }
    }

    // points, batched by colour and brightness so 5 000 documents are a few dozen fills
    const buckets = new Map();
    const shown = [];
    for (const n of nodes) {
      if (n.x < b.minX || n.x > b.maxX || n.y < b.minY || n.y > b.maxY) continue;
      shown.push(n);
      const alpha = alphaOf(n);
      const warm = emphasis && emphasis.has(n.id);
      const key = `${warm ? CHAMPAGNE : IVORY}|${Math.round(alpha * 20)}`;
      let bucket = buckets.get(key);
      if (!bucket) { bucket = []; buckets.set(key, bucket); }
      bucket.push(n);
    }
    for (const [key, list] of buckets) {
      const [rgb, level] = key.split("|");
      ctx.fillStyle = `rgba(${rgb},${Number(level) / 20})`;
      ctx.beginPath();
      for (const n of list) {
        const p = sf.worldToScreen(camera, view, n.x, n.y);
        const r = radiusOf(n);
        ctx.moveTo(p.x + r, p.y);
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }

    // index states: a thin partial ring while being read, a warm-muted dot when it failed
    for (const n of shown) {
      if (n.state !== "processing" && n.state !== "failed") continue;
      const p = sf.worldToScreen(camera, view, n.x, n.y);
      const r = radiusOf(n);
      if (n.state === "processing") {
        ctx.strokeStyle = `rgba(${CHAMPAGNE},0.55)`;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2.6, -Math.PI / 2, Math.PI);
        ctx.stroke();
      } else {
        ctx.fillStyle = `rgba(${MUTED_WARM},0.85)`;
        ctx.beginPath();
        ctx.arc(p.x + r + 2.2, p.y - r - 1.2, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (hovered && hovered !== selected) ring(hovered, 3.8, `rgba(${IVORY},0.32)`);
    if (selected) ring(selected, 5, `rgba(${CHAMPAGNE},0.9)`);

    // subject (and, closer, topic) names in the Studium label style, only where the cluster is a place on screen
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "center";
    for (const c of clusters) {
      if (!sf.clusterLabelVisible(c, camera.zoom)) continue;
      const centre = sf.worldToScreen(camera, view, c.x, c.y);
      const rpx = c.r * camera.zoom;
      if (centre.x + rpx < 0 || centre.x - rpx > view.w || centre.y + rpx < 0 || centre.y - rpx > view.h) continue;
      const lit = !emphasis || !emphasis.size || c.members.some((id) => emphasis.has(id));
      const subject = c.kind === "subject";
      ctx.font = `600 ${subject ? 10.5 : 9.5}px ${FONT}`;
      if ("letterSpacing" in ctx) ctx.letterSpacing = subject ? "2.3px" : "1.8px";
      ctx.fillStyle = `rgba(${LABEL},${(subject ? 0.62 : 0.42) * (lit ? 1 : 0.4)})`;
      const y = centre.y - (c._extent || 0) * camera.zoom - (subject ? 16 : 9);
      if (y < 14 || y > view.h - 8) continue;   // zoomed into the cluster: its name has left the view with its rim
      ctx.fillText(String(c.label || "").toUpperCase(), Math.max(60, Math.min(view.w - 60, centre.x)), y);
    }
    if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";

    // document names: the focused one always, the others once there is room between them
    ctx.textAlign = "left";
    let labels = 0;
    for (const n of shown) {
      const isFocus = n === focus;
      const c = n._cluster;
      if (!sf.nodeLabelVisible({ hovered: n === hovered, selected: n === selected, clusterRadius: c ? c.r : 0, clusterSize: c ? c.members.length : 1, zoom: camera.zoom })) continue;
      if (!isFocus && labels >= MAX_DOC_LABELS) continue;
      if (n === hovered && n !== selected) continue;   // the note says it already
      labels += 1;
      const p = sf.worldToScreen(camera, view, n.x, n.y);
      const dim = emphasis && emphasis.size && !emphasis.has(n.id);
      ctx.font = `${isFocus ? 12 : 11.5}px ${FONT}`;
      ctx.fillStyle = isFocus ? `rgba(${IVORY},0.95)` : `rgba(${LABEL},${dim ? 0.3 : 0.78})`;
      ctx.fillText(ellipsis(n.title, 34), p.x + radiusOf(n) + 7, p.y + 4);
    }
    lastDrawMs = performance.now() - t0;
  }

  function ring(n, gap, colour) {
    const p = sf.worldToScreen(camera, view, n.x, n.y);
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radiusOf(n) + gap, 0, Math.PI * 2);
    ctx.stroke();
  }

  /* ---------------------------------------------------------------- surfaces: note, action row, menu */

  function describe(n) {
    return [sf.typeWord(n), n.subject || "Ohne Zuordnung", sf.unitsWord(n)].join(" · ");
  }

  function showTip(n, sx, sy) {
    if (!n) { tip.hidden = true; return; }
    clear(tip);
    tip.append(el("div", { class: "sky-tip-title", text: n.title }), el("div", { class: "sky-tip-meta", text: describe(n) }),
      el("div", { class: "sky-tip-state" + (n.state === "failed" ? " failed" : ""), text: sf.stateWord(n) }));
    tip.hidden = false;
    const w = tip.offsetWidth || 220, h = tip.offsetHeight || 60;
    let x = sx + 14, y = sy + 14;
    if (x + w > view.w - 8) x = sx - w - 14;
    if (y + h > view.h - 8) y = sy - h - 14;
    tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  function updateBar() {
    clear(bar);
    if (!selected) { bar.hidden = true; return; }
    const n = selected;
    bar.append(el("span", { class: "sky-bar-title", text: n.title, title: n.title }),
      el("span", { class: "sky-bar-meta", text: describe(n) }),
      el("button", { class: "sky-act", type: "button", text: "Öffnen", onClick: () => act("open", n) }),
      el("button", { class: "sky-act more", type: "button", text: "···", "aria-haspopup": "menu", "aria-label": `Mehr zu ${n.title}`,
        onClick: (e) => { const r = e.currentTarget.getBoundingClientRect(); const w = wrap.getBoundingClientRect(); openMenu(n, r.left - w.left, r.top - w.top, { above: true }); } }));
    bar.hidden = false;
  }

  const ACTIONS = [
    ["open", "Öffnen"], ["search", "Im Studium suchen"], ["chat", "Im Chat verwenden"], ["similar", "Ähnliche Unterlagen"], ["assign", "Zu Fach zuordnen"],
  ];

  function openMenu(n, x, y, { above = false } = {}) {
    clear(menu);
    for (const [key, label] of ACTIONS) {
      menu.append(el("button", { class: "sky-menu-item", type: "button", role: "menuitem", text: label, onClick: () => { closeMenu(); act(key, n); } }));
    }
    menu.hidden = false;
    const w = menu.offsetWidth || 200, h = menu.offsetHeight || 180;
    const left = Math.max(8, Math.min(view.w - w - 8, x));
    const top = above ? Math.max(8, y - h - 6) : Math.max(8, Math.min(view.h - h - 8, y));
    menu.style.transform = `translate(${left}px, ${top}px)`;
    menu.querySelector("button")?.focus();
  }

  function closeMenu({ refocus = false } = {}) {
    if (menu.hidden) return;
    menu.hidden = true;
    if (refocus) canvas.focus();
  }

  menu.addEventListener("keydown", (e) => {
    const items = [...menu.querySelectorAll("button")];
    const at = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[(at + 1) % items.length]?.focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); items[(at - 1 + items.length) % items.length]?.focus(); }
    else if (e.key === "Escape" || e.key === "Tab") { e.preventDefault(); e.stopPropagation(); closeMenu({ refocus: true }); }
  });

  function act(key, n) {
    const doc = { id: n.id, title: n.title, filename: n.filename, subject: n.subject, source_type: n.source_type };
    if (key === "open") callbacks.onOpen?.(doc);
    else if (key === "search") callbacks.onSearchInStudy?.(doc);
    else if (key === "chat") callbacks.onSearchInChat?.(doc);
    else if (key === "similar") showSimilar(n);
    else if (key === "assign") callbacks.onAssign?.(doc);
  }

  function showSimilar(n) {
    let ids = (n.similar || []).filter((id) => byId.has(id));
    if (!ids.length) ids = (links.get(n.id) || []).map((l) => l.node.id);
    setEmphasis([n.id, ...ids], ids.length ? `Ähnlich wie „${ellipsis(n.title, 40)}“` : `Nichts Ähnliches zu „${ellipsis(n.title, 40)}“`);
    announce(ids.length ? `${ids.length === 1 ? "1 ähnliche Unterlage" : `${ids.length} ähnliche Unterlagen`} hervorgehoben` : "Keine ähnlichen Unterlagen gefunden");
    callbacks.onSimilar?.({ id: n.id, title: n.title }, ids);
  }

  function announce(text) {
    live.textContent = "";
    setTimeout(() => { live.textContent = text; }, 30);
  }

  function select(n, { reveal = false, say = true } = {}) {
    selected = n || null;
    updateBar();
    if (selected && reveal) {
      const p = sf.worldToScreen(camera, view, selected.x, selected.y);
      if (p.x < 40 || p.y < 40 || p.x > view.w - 40 || p.y > view.h - 90) moveCamera({ ...camera, x: selected.x, y: selected.y }, true);
    }
    if (selected && say) announce(`${selected.title}, ${describe(selected)}, ${sf.stateWord(selected)}`);
    invalidate();
  }

  function setEmphasis(ids, topicLabel = "") {
    const list = (ids || []).map(String);
    if (!list.length) {
      emphasis = null;
      focusChip.hidden = true;
      invalidate();
      return;
    }
    emphasis = new Set(list.filter((id) => byId.has(id)));
    clear(focusChip);
    focusChip.append(el("span", { class: "sky-focus-label", text: topicLabel || "Hervorgehoben" }),
      el("span", { class: "sky-focus-count", text: emphasis.size === 1 ? "1 Dokument" : `${emphasis.size} Dokumente` }),
      el("button", { class: "sky-focus-clear", type: "button", "aria-label": "Hervorhebung aufheben", text: "×",
        onClick: () => { setEmphasis(null); callbacks.onClearEmphasis?.(); canvas.focus(); } }));
    focusChip.hidden = false;
    const members = [...emphasis].map((id) => byId.get(id));
    if (members.length) fitTo(members, true);
    invalidate();
  }

  /* ---------------------------------------------------------------- input */

  const pointers = new Map();
  let drag = null;
  let pinch = null;

  function local(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function hit(sx, sy) {
    if (!grid || !nodes.length) return null;
    const w = sf.screenToWorld(camera, view, sx, sy);
    return sf.nearestInGrid(grid, w.x, w.y, HIT_PX / camera.zoom);
  }

  function onPointerDown(e) {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    closeMenu();
    const p = local(e);
    pointers.set(e.pointerId, p);
    try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { distance: Math.hypot(a.x - b.x, a.y - b.y) || 1 };
      drag = null;
    } else {
      drag = { x: p.x, y: p.y, moved: 0 };
    }
    cancelAnimationFrame(anim);
    anim = 0;
  }

  function onPointerMove(e) {
    const p = local(e);
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
    if (pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const distance = Math.hypot(a.x - b.x, a.y - b.y) || 1;
      camera = sf.zoomAt(camera, view, (a.x + b.x) / 2, (a.y + b.y) / 2, distance / pinch.distance);
      pinch.distance = distance;
      tip.hidden = true;
      invalidate();
      return;
    }
    if (drag) {
      const dx = p.x - drag.x, dy = p.y - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      drag.x = p.x; drag.y = p.y;
      if (drag.moved > 3) {
        camera = { ...camera, x: camera.x - dx / camera.zoom, y: camera.y - dy / camera.zoom };
        canvas.classList.add("panning");
        tip.hidden = true;
        invalidate();
      }
      return;
    }
    const found = hit(p.x, p.y);
    if (found !== hovered) {
      hovered = found;
      canvas.classList.toggle("over", Boolean(found));
      invalidate();
    }
    showTip(found, p.x, p.y);
  }

  function onPointerUp(e) {
    const p = local(e);
    pointers.delete(e.pointerId);
    canvas.classList.remove("panning");
    if (pinch) { if (pointers.size < 2) pinch = null; drag = null; return; }
    if (drag && drag.moved <= 3) select(hit(p.x, p.y), { say: false });
    drag = null;
  }

  function onLeave() {
    if (drag) return;
    hovered = null;
    tip.hidden = true;
    canvas.classList.remove("over");
    invalidate();
  }

  function onDoubleClick(e) {
    const p = local(e);
    const found = hit(p.x, p.y);
    if (found) { select(found, { say: false }); act("open", found); }
  }

  function onContextMenu(e) {
    e.preventDefault();
    const p = local(e);
    const found = hit(p.x, p.y);
    if (!found) { closeMenu(); return; }
    select(found, { say: false });
    tip.hidden = true;
    openMenu(found, p.x + 4, p.y + 4);
  }

  function onWheel(e) {
    e.preventDefault();
    const p = local(e);
    const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    cancelAnimationFrame(anim);
    anim = 0;
    camera = sf.zoomAt(camera, view, p.x, p.y, Math.exp(-delta * 0.0016));
    tip.hidden = true;
    invalidate();
  }

  function onKey(e) {
    if (e.target !== canvas) return;
    if (e.key.startsWith("Arrow")) {
      e.preventDefault();
      const shown = sf.visibleNodes(nodes, camera, view, 0);
      if (!selected) {
        const centre = sf.nearestTo(shown.length ? shown : nodes, camera.x, camera.y);
        if (centre) select(centre, { reveal: true });
        return;
      }
      const next = sf.nearestInDirection(nodes, selected, e.key);
      if (next) select(next, { reveal: true });
      return;
    }
    if (e.key === "Enter" && selected) { e.preventDefault(); act("open", selected); return; }
    if (e.key === "Escape") {
      // Escape lets go of what the sky holds first; only an idle sky passes it on (the app closes the view then)
      if (!menu.hidden) { e.preventDefault(); e.stopPropagation(); closeMenu(); return; }
      if (selected) { e.preventDefault(); e.stopPropagation(); select(null, { say: false }); announce("Auswahl aufgehoben"); }
      return;
    }
    if ((e.key === "ContextMenu" || (e.key === "F10" && e.shiftKey)) && selected) {
      e.preventDefault();
      const p = sf.worldToScreen(camera, view, selected.x, selected.y);
      openMenu(selected, p.x + 8, p.y + 8);
      return;
    }
    if (e.key === "+" || e.key === "=") { e.preventDefault(); moveCamera(sf.zoomAt(camera, view, view.w / 2, view.h / 2, 1.4), true); return; }
    if (e.key === "-" || e.key === "_") { e.preventDefault(); moveCamera(sf.zoomAt(camera, view, view.w / 2, view.h / 2, 1 / 1.4), true); return; }
    if (e.key === "0") { e.preventDefault(); fitAll(true); }
  }

  function onDocumentPointer(e) {
    if (!menu.hidden && !menu.contains(e.target)) closeMenu();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("dblclick", onDoubleClick);
  canvas.addEventListener("contextmenu", onContextMenu);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("keydown", onKey);
  document.addEventListener("pointerdown", onDocumentPointer, true);

  const resize = () => {
    const rect = wrap.getBoundingClientRect();
    const w = Math.round(rect.width), h = Math.round(rect.height);
    if (w <= 0 || h <= 0) return;
    dpr = window.devicePixelRatio || 1;
    view = { w, h };
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    if (pendingFit === "all" || (!fitted && nodes.length)) { pendingFit = null; fitAll(false); }
    else if (Array.isArray(pendingFit)) { const list = pendingFit; pendingFit = null; fitTo(list, false); }
    draw();
  };
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(resize) : null;
  observer?.observe(wrap);
  resize();
  refresh();

  return {
    setEmphasis,
    refresh,
    load,                                   // in-memory data (tests, synthetic libraries)
    select: (id) => select(byId.get(id) || null),
    lastDrawMs: () => lastDrawMs,
    redraw: () => { draw(); return lastDrawMs; },
    camera: () => ({ ...camera }),
    focus: () => canvas.focus(),
    destroy() {
      destroyed = true;
      cancelAnimationFrame(frame);
      cancelAnimationFrame(anim);
      observer?.disconnect();
      document.removeEventListener("pointerdown", onDocumentPointer, true);
      wrap.remove();
    },
  };
}
