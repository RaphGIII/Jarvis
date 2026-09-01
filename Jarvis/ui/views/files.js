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
import { Galaxy } from "./projects.js";

const CATEGORY_HUE = {
  PROJECTS: [102, 201, 255], DEVELOPMENT: [127, 224, 180], GAMES: [190, 140, 250],
  MEDIA: [240, 140, 190], DOCUMENTS: [240, 200, 110], SYSTEM: [130, 145, 170], OTHER: [150, 170, 200],
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

const store = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
const save = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
let pins = new Set(store("zeus.files.pins", []));
let hiddenPaths = new Set(store("zeus.files.hidden", []));

export const view = {
  id: "files",
  title: "Files",
  async mount(pane, params) {
    active = true;
    root = params.path || store("zeus.files.root", "D:\\");
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
  return {
    id: fsId(entry.path), kind: "project", label: entry.name, hue: CATEGORY_HUE[category],
    importance: pinned ? "PINNED" : ["PROJECTS", "DEVELOPMENT"].includes(category) ? "ACTIVE" : "NORMAL",
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
    const { drives } = await api("/api/fs/roots");
    const nodes = [{ id: "fsroot", kind: "self", label: "Dein Rechner" }];
    const edges = [];
    for (const d of drives || []) {
      nodes.push(dirNode({ name: d.label || d.path, path: d.path, category: d.primary ? "PROJECTS" : "SYSTEM", type: "dir", children_count: 0 }));
      if (d.primary) nodes.at(-1).importance = "PINNED";
      edges.push({ source: "fsroot", target: fsId(d.path), type: "relates" });
    }
    return { nodes, edges, counts: `${(drives || []).length} Laufwerke` };
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
  crumbs.append(el("button", { class: "chip", text: "◈ Rechner", onClick: () => go(null) }));
  parts.forEach((part, i) => {
    const target = parts.slice(0, i + 1).join("\\") + "\\";
    crumbs.append(el("span", { class: "sep", text: "›" }), el("button", { class: "chip" + (i === parts.length - 1 ? " on" : ""), text: part || "\\", onClick: () => go(target) }));
  });
  const search = el("input", { placeholder: "Fokus… (Ordner, Datei)", style: { maxWidth: "220px" } });
  const shortcuts = el("span", { style: { display: "inline-flex", gap: "4px" } });
  for (const [label, target] of [["D:\\", "D:\\"], ["C:\\", "C:\\"], ["ZEUS", "D:\\Jarvis_recovery_20260823\\"]]) {
    shortcuts.append(el("button", { class: "chip", text: label, onClick: () => go(target) }));
  }
  const immersive = el("button", { class: "chip", text: "⛶ immersiv", onClick: () => document.body.classList.toggle("galaxy-full") });
  const counts = el("span", { class: "empty", style: { padding: 0 } });
  pane.append(el("div", { class: "toolbar galaxy-overlay" }, crumbs, search, shortcuts, immersive, counts));

  const wrap = el("div", { class: "galaxy-wrap" });
  const canvas = el("canvas", { id: "constellation", class: "galaxy files" });
  const legend = el("div", { class: "galaxy-legend" },
    ...Object.entries(CATEGORY_COLOUR).map(([k, c]) => el("span", { style: { "--c": c }, text: k.toLowerCase() })));
  const hint = el("div", { class: "galaxy-hint", text: "doppelklick betreten · rechtsklick menü · zoom = tiefe · esc zurück" });
  wrap.append(canvas, legend, hint);
  pane.append(wrap);

  const graph = await buildGraph();
  counts.textContent = graph.counts;
  galaxy = new Galaxy(canvas, wrap, graph, {
    mode: "GALAXY", persistDrag: false,
    onSelect: (n) => inspectEntry(n),
    onDoubleClick: (n) => { if (n.data?.isFs && n.data.type === "dir") go(n.data.path); },
    onContext: (n, x, y) => contextMenu(n, x, y),
    // faint category region names over their real folders (view-only grouping)
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
        let sx = 0, sy = 0;
        for (const n of ns) { const [x, y] = g.toScreen(n); sx += x; sy += y; }
        const [r, gg, b] = CATEGORY_HUE[cat] || CATEGORY_HUE.OTHER;
        ctx.fillStyle = `rgba(${r},${gg},${b},.32)`;
        ctx.fillText(cat, sx / ns.length, sy / ns.length - 56);
      }
    },
  });
  // category galaxies: the same real folders, spatially grouped by category
  if (root) {
    const folders = galaxy.nodes.filter((n) => n.kind === "project" && !n.sub && n.data?.category);
    const cats = [...new Set(folders.map((n) => n.data.category))];
    const W = galaxy.W(), H = galaxy.H();
    cats.forEach((cat, ci) => {
      const a = (ci / Math.max(1, cats.length)) * Math.PI * 2 - Math.PI / 2;
      const spread = cats.length > 1 ? 1 : 0;
      const cx = W / 2 + Math.cos(a) * W * 0.3 * spread;
      const cy = H / 2 + Math.sin(a) * H * 0.28 * spread;
      const members = folders.filter((n) => n.data.category === cat);
      members.forEach((n, i) => {
        const ga = i * 2.399963 + ci;
        const rr = 26 + 22 * Math.sqrt(i);
        n.tx = cx + Math.cos(ga) * rr; n.ty = cy + Math.sin(ga) * rr * 0.75;
      });
    });
  }
  const saved = cams.get(root || "");
  if (saved) Object.assign(galaxy.cam, saved);
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
  if (galaxy) cams.set(root || "", { ...galaxy.cam });
  root = path;
  views.open("files", { path: path || "" });
}

function goUp() {
  if (!root) { views.close(); return; }
  const trimmed = root.replace(/[\\/]+$/, "");
  const idx = trimmed.lastIndexOf("\\");
  go(idx > 1 ? trimmed.slice(0, idx + 1) : null);
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
