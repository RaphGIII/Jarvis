/* Files: the OS universe.

   The successful Project Galaxy engine (views/projects.js exports Galaxy)
   rendering the owner's real filesystem: drives are the outermost view,
   folders are stars coloured by category, files are small satellites that
   only appear when the camera is close.  Double-click ENTERS a folder
   (re-roots the galaxy, breadcrumb back), wheel zoom past a threshold
   expands the hovered folder in place, zooming far out collapses the
   expansions again — semantic zoom, not a 100k-node dump.

   Everything shown is a real path from /api/fs/list; nothing is invented.
   The backend watches the current root with ReadDirectoryChangesW and the
   view refreshes (camera preserved) when a debounced change event arrives.
   Visual classification (category colour, pin, hide) never moves a folder
   on disk. */

import { el, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";
import { Galaxy, warp } from "./projects.js";

const CATEGORY_HUE = {
  PROJECTS: [102, 201, 255], DEVELOPMENT: [127, 224, 180], GAMES: [190, 140, 250],
  MEDIA: [240, 140, 190], DOCUMENTS: [240, 200, 110], AI_MODELS: [160, 130, 250],
  TOOLS: [120, 210, 220], SYSTEM: [130, 145, 170], OTHER: [150, 170, 200],
};
const CATEGORY_COLOUR = Object.fromEntries(Object.entries(CATEGORY_HUE).map(([k, [r, g, b]]) => [k, `rgb(${r},${g},${b})`]));
const ENTER_ZOOM = 1.75;   // zooming closer than this over a folder expands it in place
const COLLAPSE_ZOOM = 0.8; // zooming further out than this collapses expansions
const MAX_EXPANDED = 10;

let galaxy = null;
let active = false;
let root = null;               // current path, null = drives overview
let expanded = new Set();      // paths expanded in place
let watching = null;
let refreshTimer = 0;
let zoomTimer = 0;
const cams = new Map();        // path -> saved camera

const REDUCED_MOTION = () => document.body.classList.contains("reduced-motion");

const store = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
const save = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
let pins = new Set(store("zeus.files.pins", []));
let hiddenPaths = new Set(store("zeus.files.hidden", []));

export const view = {
  id: "files",
  title: "Files",
  async mount(pane, params) {
    active = true;
    // an explicit empty path is the universe (drives) level; only a missing
    // parameter falls back to the last root the owner was in
    root = "path" in params ? (params.path || null) : store("zeus.files.root", "D:\\") || null;
    await render(pane);
    if (!view._bused) { view._bused = true; bus.on("diagnostic", onFsEvent); }
    if (!view._esc) {
      // ESC = one level up (drive → drives → close); the engine's own ESC
      // (menu close, zoom reset) keeps precedence via its guards below
      view._esc = (e) => {
        if (!active || e.key !== "Escape" || !galaxy) return;
        if (galaxy.menu || galaxy.dim) return;
        if (Math.abs(galaxy.cam.z - 1) > 0.15) return;
        e.stopPropagation();
        goUp();
      };
      window.addEventListener("keydown", view._esc, true);
    }
  },
  unmount() {
    active = false;
    if (galaxy) cams.set(root || "", { ...galaxy.cam });
    galaxy?.destroy(); galaxy = null;
    document.body.classList.remove("galaxy-full");
    clearInterval(zoomTimer); zoomTimer = 0;
    if (watching) { api("/api/fs/unwatch", { path: watching }).catch(() => {}); watching = null; }
  },
};

function onFsEvent(payload) {
  if (!active || !payload || !payload.fs || payload._replay) return;
  const changed = payload.changed || [];
  if (root && !changed.some((c) => String(c.path || "").toLowerCase().startsWith(root.toLowerCase()))) return;
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { if (active) views.open("files", { path: root || "" }, { push: false }); }, 400);
}

/* ---- building the galaxy from real listings --------------------------- */
const fsId = (path) => "fs:" + String(path).toLowerCase();

function dirNode(entry, { sub = false } = {}) {
  const category = entry.category || "OTHER";
  const pinned = pins.has(entry.path.toLowerCase());
  const kids = entry.children_count || 0;
  return {
    id: fsId(entry.path), kind: "project", label: entry.name, hue: CATEGORY_HUE[category] || CATEGORY_HUE.OTHER,
    importance: pinned ? "PINNED" : ["PROJECTS", "DEVELOPMENT", "AI_MODELS"].includes(category) ? "ACTIVE" : "NORMAL",
    // a folder's size on screen is what it holds: sqrt so a 400-entry node
    // dominates without dwarfing everything else; rings mark the heavyweights
    baseR: sub ? 0 : 8 + (pinned ? 5 : 0) + Math.min(14, Math.sqrt(kids) * 1.5),
    ringed: kids >= 60,
    tasks: 0, sub, data: { ...entry, isFs: true }, health: { state: "HEALTHY", reason: "" },
  };
}
function fileNode(entry) {
  return { id: fsId(entry.path), kind: "capability", label: entry.name, data: { ...entry, isFs: true }, attempts: 1 };
}

async function listing(path) {
  const out = await api("/api/fs/list", { path });
  if (out.ok === false) return { dirs: [], files: [], error: out.error, truncated: false };
  const visible = (out.entries || []).filter((e) => !hiddenPaths.has(e.path.toLowerCase()));
  return {
    dirs: visible.filter((e) => e.type === "dir"),
    files: visible.filter((e) => e.type === "file"),
    truncated: Boolean(out.truncated), error: null,
  };
}

async function buildGraph() {
  if (!root) {
    // the universe: drives are the great systems, and the owner's Projects
    // galaxy is embedded as a system of its own — one cosmos, not two apps
    const { drives } = await api("/api/fs/roots");
    const nodes = [{ id: "fsroot", kind: "self", label: "Dein Rechner" }];
    const edges = [];
    for (const d of drives || []) {
      const share = d.total_bytes ? Math.min(1, (d.total_bytes - (d.free_bytes || 0)) / d.total_bytes) : 0.4;
      nodes.push({
        ...dirNode({ name: d.label || d.path, path: d.path, category: d.primary ? "PROJECTS" : "SYSTEM", type: "dir", children_count: 0 }),
        baseR: d.primary ? 30 : 16 + share * 8, ringed: true,
      });
      if (d.primary) nodes.at(-1).importance = "PINNED";
      edges.push({ source: "fsroot", target: fsId(d.path), type: "relates" });
    }
    nodes.push({
      id: "galaxy:projects", kind: "project", label: "Projekte", hue: [225, 235, 255],
      importance: "FOCUS", baseR: 24, ringed: true, tasks: 0,
      data: { isGalaxyLink: true, name: "Projekte" }, health: { state: "HEALTHY", reason: "" },
    });
    edges.push({ source: "fsroot", target: "galaxy:projects", type: "relates" });
    return { nodes, edges, counts: `${(drives || []).length} Laufwerke · 1 Projekt-Galaxie` };
  }
  const { dirs, files, truncated, error } = await listing(root);
  const nodes = [{ id: "fsroot", kind: "self", label: root }];
  const edges = [];
  for (const d of dirs) { nodes.push(dirNode(d)); edges.push({ source: "fsroot", target: fsId(d.path), type: "relates" }); }
  for (const f of files.slice(0, 80)) { nodes.push(fileNode(f)); edges.push({ source: "fsroot", target: fsId(f.path), type: "uses" }); }
  const counts = `${dirs.length} Ordner · ${files.length} Dateien` + (truncated ? " · gekürzt" : "") + (error ? ` · ${error}` : "");
  return { nodes, edges, counts };
}

/* ---- semantic zoom: expand in place, collapse when far ---------------- */
async function expandInPlace(n) {
  const path = n.data?.path;
  if (!path || expanded.has(path) || expanded.size >= MAX_EXPANDED || !galaxy) return;
  expanded.add(path);
  const { dirs, files } = await listing(path);
  if (!galaxy || !galaxy.byId.has(n.id)) return;
  const nodes = [], edges = [];
  for (const d of dirs.slice(0, 24)) { nodes.push({ ...dirNode(d, { sub: true }), r: 6, depth: 0.92 }); edges.push({ source: n.id, target: fsId(d.path), type: "subproject_of" }); }
  for (const f of files.slice(0, 24)) { nodes.push(fileNode(f)); edges.push({ source: n.id, target: fsId(f.path), type: "uses" }); }
  galaxy.addGraph(nodes, edges);
}

function collapseAll() {
  if (!galaxy || !expanded.size) return;
  for (const path of expanded) galaxy.removeSubtree(fsId(path));
  expanded = new Set();
}

/* ---- the view --------------------------------------------------------- */
async function render(pane) {
  expanded = new Set();
  save("zeus.files.root", root || "D:\\");
  const crumbs = el("div", { class: "fs-crumbs" });
  const parts = root ? root.replace(/\\+$/, "").split("\\") : [];
  crumbs.append(el("button", { class: "chip" + (root ? "" : " on"), text: "◈ Universum", onClick: () => go(null) }));
  parts.forEach((part, i) => {
    const target = parts.slice(0, i + 1).join("\\") + "\\";
    crumbs.append(el("span", { class: "sep", text: "›" }), el("button", { class: "chip" + (i === parts.length - 1 ? " on" : ""), text: part || "\\", onClick: () => go(target) }));
  });
  const search = el("input", { placeholder: "Fokus… (Ordner, Datei)", style: { maxWidth: "220px" } });
  const shortcuts = el("span", { style: { display: "inline-flex", gap: "4px" } });
  for (const [label, target] of [["D:\\", "D:\\"], ["C:\\", "C:\\"], ["ZEUS", "D:\\Jarvis_recovery_20260823\\"]]) {
    shortcuts.append(el("button", { class: "chip" + (target === root ? " on" : ""), text: label, onClick: () => go(target) }));
  }
  const counts = el("span", { class: "empty", style: { padding: 0 } });
  pane.append(el("div", { class: "toolbar galaxy-overlay" }, crumbs, search, shortcuts, counts));

  const wrap = el("div", { class: "galaxy-wrap" });
  const canvas = el("canvas", { id: "constellation", class: "galaxy files" });
  const legend = el("div", { class: "galaxy-legend" },
    ...Object.entries(CATEGORY_COLOUR).map(([k, c]) => el("span", { style: { "--c": c }, text: k.toLowerCase().replace("_", "/") })));
  const hint = el("div", { class: "galaxy-hint", text: root ? "rauszoomen → größere ebene · doppelklick betreten · rechtsklick menü · esc zurück" : "reinzoomen/doppelklick → system betreten · die projekte-galaxie ist teil dieses universums" });
  wrap.append(canvas, legend, hint);
  pane.append(wrap);

  const graph = await buildGraph();
  counts.textContent = graph.counts;
  galaxy = new Galaxy(canvas, wrap, graph, {
    mode: "GALAXY", persistDrag: false,
    // the universe allows a far wider pull-back than a single folder level
    minZoom: root ? 0.35 : 0.2,
    // pulling out past the floor rises one semantic level: folder → parent →
    // drive → universe; the universe itself is the outermost truth
    onZoomOutBeyond: root ? () => goUp() : undefined,
    onSelect: (n) => {
      if (n.data?.isGalaxyLink || (n.data?.isFs && n.data.type === "dir")) return choiceMenu(n);
      inspectEntry(n);
    },
    onDoubleClick: (n) => {
      if (n.data?.isGalaxyLink) return enterProjects();
      if (n.data?.isFs && n.data.type === "dir") go(n.data.path);
    },
    onContext: (n, x, y) => contextMenu(n, x, y),
    // soft category nebulas UNDER the bodies: the sectors of this system
    drawUnder: (ctx, g) => {
      const groups = new Map();
      for (const n of g.nodes) {
        if (n.kind !== "project" || n.sub || !n.data?.category || !n.visible) continue;
        if (!groups.has(n.data.category)) groups.set(n.data.category, []);
        groups.get(n.data.category).push(n);
      }
      for (const [cat, ns] of groups) {
        if (ns.length < 2) continue;
        let sx = 0, sy = 0;
        for (const n of ns) { const [x, y] = g.toScreen(n); sx += x; sy += y; }
        const cx = sx / ns.length, cy = sy / ns.length;
        let spread = 60;
        for (const n of ns) { const [x, y] = g.toScreen(n); spread = Math.max(spread, Math.hypot(x - cx, y - cy) + n.r * g.cam.z + 30); }
        const [r, gg, b] = CATEGORY_HUE[cat] || CATEGORY_HUE.OTHER;
        const neb = ctx.createRadialGradient(cx, cy, spread * 0.15, cx, cy, spread);
        neb.addColorStop(0, `rgba(${r},${gg},${b},.075)`); neb.addColorStop(0.7, `rgba(${r},${gg},${b},.035)`); neb.addColorStop(1, "transparent");
        ctx.fillStyle = neb; ctx.beginPath(); ctx.arc(cx, cy, spread, 0, Math.PI * 2); ctx.fill();
      }
    },
    // category sector names float over their folders (view-only grouping)
    drawExtras: (ctx, g) => {
      if (!root) return;
      const groups = new Map();
      for (const n of g.nodes) {
        if (n.kind !== "project" || n.sub || !n.data?.category || !n.visible) continue;
        if (!groups.has(n.data.category)) groups.set(n.data.category, []);
        groups.get(n.data.category).push(n);
      }
      ctx.textAlign = "center"; ctx.font = "600 10px Segoe UI, sans-serif";
      for (const [cat, ns] of groups) {
        if (ns.length < 2) continue;
        let sx = 0, sy = 0, top = Infinity;
        for (const n of ns) { const [x, y] = g.toScreen(n); sx += x; sy += y; top = Math.min(top, y - n.r * g.cam.z); }
        const [r, gg, b] = CATEGORY_HUE[cat] || CATEGORY_HUE.OTHER;
        ctx.fillStyle = `rgba(${r},${gg},${b},.4)`;
        ctx.fillText(cat.replace("_", " / "), sx / ns.length, top - 26);
      }
    },
  });
  // category galaxies: the same real folders, spatially grouped by category,
  // then relaxed apart so no two bodies open on top of each other
  if (root) {
    const folders = galaxy.nodes.filter((n) => n.kind === "project" && !n.sub && n.data?.category);
    const cats = [...new Set(folders.map((n) => n.data.category))];
    const W = galaxy.W(), H = galaxy.H();
    cats.forEach((cat, ci) => {
      const a = (ci / Math.max(1, cats.length)) * Math.PI * 2 - Math.PI / 2;
      const spread = cats.length > 1 ? 1 : 0;
      const cx = W / 2 + Math.cos(a) * W * 0.32 * spread;
      const cy = H / 2 + Math.sin(a) * H * 0.3 * spread;
      const members = folders.filter((n) => n.data.category === cat);
      members.forEach((n, i) => {
        const ga = i * 2.399963 + ci;
        const rr = 34 + 30 * Math.sqrt(i);
        n.tx = cx + Math.cos(ga) * rr; n.ty = cy + Math.sin(ga) * rr * 0.75;
      });
    });
    for (let it = 0; it < 80; it++) {
      let moved = false;
      for (const a of folders) for (const b of folders) {
        if (a === b) continue;
        const dx = a.tx - b.tx, dy = a.ty - b.ty, dd = Math.max(1, Math.hypot(dx, dy));
        const min = a.r + b.r + 34;
        if (dd < min) {
          const push = (min - dd) / 2, ux = dx / dd, uy = dy / dd;
          a.tx += ux * push; a.ty += uy * push; b.tx -= ux * push; b.ty -= uy * push; moved = true;
        }
      }
      for (const n of folders) { const R = n.r + 14; n.tx = Math.max(R, Math.min(W - R, n.tx)); n.ty = Math.max(R + 30, Math.min(H - R - 24, n.ty)); }
      if (!moved) break;
    }
  }
  const saved = cams.get(root || "");
  if (saved) Object.assign(galaxy.cam, saved);
  else if (!REDUCED_MOTION()) {
    // arriving in a system: a short settle from slightly outside, so entering
    // a galaxy reads as travel, not as a page swap
    galaxy.cam.z = 0.74;
    galaxy.flyTo({ x: galaxy.W() / 2, y: galaxy.H() / 2 }, 1);
  }
  window.zeusGalaxy = galaxy; // console/test access to the live scene
  search.oninput = () => galaxy?.focusText(search.value);

  // watch the current root live (D:\ and C:\ roots stay unwatched: too broad)
  if (watching && watching !== root) { api("/api/fs/unwatch", { path: watching }).catch(() => {}); watching = null; }
  if (root && root.length > 3) { const r = await api("/api/fs/watch", { path: root }); if (r.ok) watching = root; }

  // zoom drives depth: close over a folder expands it, far away collapses
  clearInterval(zoomTimer);
  zoomTimer = setInterval(() => {
    if (!galaxy || !active) return;
    if (galaxy.cam.z <= COLLAPSE_ZOOM) { collapseAll(); return; }
    if (galaxy.cam.z >= ENTER_ZOOM) {
      const target = galaxy.hover || galaxy.byId.get([...galaxy.selected][0]);
      if (target?.data?.isGalaxyLink) { enterProjects(); return; }
      if (target?.data?.isFs && target.data.type === "dir" && !target.sub) expandInPlace(target);
    }
  }, 350);

  // project ↔ folder links: real workspaces only
  try {
    const overview = await api("/api/projects/overview");
    for (const p of overview.projects || []) {
      const ws = String(p.workspace || p.metadata?.workspace || "").toLowerCase();
      if (!ws) continue;
      const node = galaxy?.byId.get(fsId(ws)) || galaxy?.byId.get(fsId(ws.replace(/\\+$/, "")));
      if (node) { node.projectRef = p; node.label += " ◆"; }
    }
  } catch {}
}

function go(path) {
  if (galaxy) {
    cams.set(root || "", { ...galaxy.cam });
    // rising = the target is an ancestor (or the universe); diving = deeper in
    const rising = !path || (root && String(root).toLowerCase().startsWith(String(path).toLowerCase()));
    warp(galaxy.canvas, rising ? "rise" : "dive");
  }
  root = path;
  views.open("files", { path: path || "" });
}

function goUp() {
  if (!root) { views.close(); return; }
  const trimmed = root.replace(/[\\/]+$/, "");
  const idx = trimmed.lastIndexOf("\\");
  go(idx > 1 ? trimmed.slice(0, idx + 1) : null);
}

/* The Projects galaxy is a system inside this universe: entering it is the
   same gesture as entering a drive. */
function enterProjects() {
  if (galaxy) { cams.set(root || "", { ...galaxy.cam }); warp(galaxy.canvas, "dive"); }
  views.open("projects", {});
}

/* Clicking a system asks the one real question instead of burying it in an
   inspector: travel into it, or open it in the Explorer. */
function choiceMenu(n) {
  if (!galaxy) return;
  galaxy.closeMenu();
  const d = n.data || {};
  const [sx, sy] = galaxy.toScreen(n);
  const item = (label, fn, cls = "") => el("button", { class: cls, text: label, onClick: async () => { galaxy?.closeMenu(); await fn(); } });
  const menu = el("div", { class: "galaxy-menu choice" }, el("h6", { text: d.name || n.label }),
    item("🚀  Galaxy bereisen", () => (d.isGalaxyLink ? enterProjects() : go(d.path))),
    d.isGalaxyLink ? null : item("📁  Im Explorer öffnen", () => api("/api/fs/open", { path: d.path })),
    d.isGalaxyLink ? null : item("Inspizieren", () => inspectEntry(n)));
  // beside the body, never under the cursor: a double-click's second click
  // must still reach the canvas, not the menu that the first click opened
  const off = (n.r || 10) * galaxy.cam.z + 22;
  menu.style.left = Math.max(8, Math.min(sx + off, galaxy.W() - 240)) + "px";
  menu.style.top = Math.max(8, Math.min(sy - 20, galaxy.H() - 160)) + "px";
  galaxy.wrap.append(menu); galaxy.menu = menu;
}

/* ---- inspector and context menu --------------------------------------- */
function inspectEntry(n) {
  const d = n.data || {};
  if (!d.isFs) return;
  const size = d.size >= 1 << 20 ? `${(d.size / (1 << 20)).toFixed(1)} MB` : d.size >= 1024 ? `${(d.size / 1024).toFixed(0)} KB` : d.size != null ? `${d.size} B` : "—";
  views.inspect(d.name || n.label,
    el("div", { class: "meta" }, badge(d.type === "dir" ? "ORDNER" : "DATEI", "blue"), " ", badge(d.category || "OTHER", "dim"),
      n.projectRef ? badge("PROJEKT: " + (n.projectRef.title || ""), "amber") : ""),
    section("Pfad", kv("pfad", d.path), kv("geändert", (d.modified || "").slice(0, 19).replace("T", " ")),
      d.type === "dir" ? kv("einträge", String(d.children_count ?? "—")) : kv("größe", size)),
    el("div", { class: "toolbar" },
      d.type === "dir" ? button("Galaxie betreten", () => go(d.path), "primary") : null,
      button("Im Explorer öffnen", () => api("/api/fs/open", { path: d.path })),
      button("Pfad kopieren", () => navigator.clipboard?.writeText(d.path)),
      n.projectRef ? button("Projekt öffnen", () => views.open("projects", { id: n.projectRef.id })) : null,
      button(pins.has(d.path.toLowerCase()) ? "Loslösen" : "Anheften", () => { togglePin(d.path); views.open("files", { path: root || "" }, { push: false }); }),
      button("Zeus fragen", () => chat.send(`Was liegt in „${d.path}“ und wofür nutze ich es?`, "galaxy"))));
}

function togglePin(path) { const k = path.toLowerCase(); pins.has(k) ? pins.delete(k) : pins.add(k); save("zeus.files.pins", [...pins]); }
function toggleHide(path) { const k = path.toLowerCase(); hiddenPaths.has(k) ? hiddenPaths.delete(k) : hiddenPaths.add(k); save("zeus.files.hidden", [...hiddenPaths]); }

function contextMenu(n, sx, sy) {
  if (!galaxy) return;
  galaxy.closeMenu();
  const d = n.data || {};
  if (!d.isFs) return;
  const item = (label, fn) => el("button", { text: label, onClick: async () => { galaxy?.closeMenu(); await fn(); } });
  const menu = el("div", { class: "galaxy-menu" }, el("h6", { text: d.name || n.label }),
    d.type === "dir" ? item("Galaxie betreten", () => go(d.path)) : null,
    item("Im Explorer öffnen", () => api("/api/fs/open", { path: d.path })),
    item("Pfad kopieren", () => navigator.clipboard?.writeText(d.path)),
    item(pins.has(d.path.toLowerCase()) ? "Loslösen" : "Anheften", () => { togglePin(d.path); views.open("files", { path: root || "" }, { push: false }); }),
    item("Ausblenden (nur visuell)", () => { toggleHide(d.path); views.open("files", { path: root || "" }, { push: false }); }),
    item("Inspizieren", () => inspectEntry(n)),
    item("Zeus fragen", () => chat.send(`Was liegt in „${d.path}“ und wofür nutze ich es?`, "galaxy")));
  menu.style.left = Math.min(sx + 6, galaxy.W() - 210) + "px";
  menu.style.top = Math.min(sy + 6, galaxy.H() - 280) + "px";
  galaxy.wrap.append(menu); galaxy.menu = menu;
}
