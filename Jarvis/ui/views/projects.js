/* Projects: the owner's project universe.

   A galaxy, not a card list.  Each owner project is a STAR SYSTEM: a layered
   star (size from importance and work, colour and halo from health), its
   subprojects and missions on faint orbit rings, the capabilities it uses as
   small satellites on an outer orbit.  Knowledge is a soft nebula.  ZEUS's
   own thoughts are pulses travelling along the relation they concern.  A
   three-layer parallax star field and a restrained particle drift give the
   canvas depth; the camera eases; semantic zoom shows systems from afar and
   missions/thoughts up close.

   The whole canvas is used: systems are laid out on a golden-angle spiral
   scaled to the canvas and relaxed apart, so four projects are four distinct
   systems rather than a cluster in the middle.

   Owner intent wins over the layout engine: a dragged system becomes
   OWNER_POSITIONED and is persisted (/api/project/update); LOCKED systems
   never move; box-select, multi-select and group move; layout modes never
   touch owner-saved positions.  Importance (PINNED FOCUS ACTIVE NORMAL
   LOW_PRIORITY DORMANT TEST ARCHIVED) decides what dominates; TEST and
   ARCHIVED stay out of the default view ("show everything" is explicit).

   Right-click (or the ⋯ button) opens the context menu: Focus, Pin, Hide,
   Archive, Importance, Create Mission, Ask Zeus, Local Graph. */

import { $, el, clear, kv, section, badge, button, ago, seconds } from "../core/dom.js";
import { api } from "../core/api.js";
import { state } from "../core/state.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const IMPORTANCE = ["PINNED", "FOCUS", "ACTIVE", "NORMAL", "LOW_PRIORITY", "DORMANT", "TEST", "ARCHIVED"];
const IMPORTANCE_WEIGHT = { PINNED: 1.0, FOCUS: 0.95, ACTIVE: 0.8, NORMAL: 0.6, LOW_PRIORITY: 0.4, DORMANT: 0.3, TEST: 0.25, ARCHIVED: 0.2 };
const HEALTH_COLOUR = { HEALTHY: "#7fe0b4", AT_RISK: "#f0c674", BLOCKED: "#ff7b7b", DORMANT: "#6f7f99", COMPLETE: "#66c9ff" };
const HEALTH_HUE = { HEALTHY: [120, 230, 180], AT_RISK: [240, 200, 110], BLOCKED: [255, 120, 120], DORMANT: [120, 135, 165], COMPLETE: [110, 200, 255] };
const KIND_COLOUR = { self: "#9fdcff", mission: "#b8c9e6", capability: "#8fa1c2", knowledge: "#9b8cf0", thought: "#f0c674" };
const MODES = ["GALAXY", "DEPENDENCY", "TIMELINE", "HIERARCHY", "KNOWLEDGE", "MISSION FLOW"];
const STATE_TONE = { active: "active", running: "active", working: "active", executing: "active", blocked: "blocked", failed: "blocked",
                     accepted: "done", complete: "done", completed: "done", paused: "idle", draft: "idle" };
const REDUCED = () => Boolean(state.ui.reducedMotion);

let galaxy = null;
let savedCam = null; // the owner's camera survives leaving the view
let lastParams = null;   // what the parked scene was showing
let lastDigest = "";     // fingerprint of the data the parked scene renders
let isDeep = false;      // detail page (params.id) — never resumed, always fresh

// a cheap stable fingerprint: enough to notice "the data changed while the
// owner was away" without diffing graphs
export function digestOf(obj) {
  const s = JSON.stringify(obj);
  let h = 0;
  for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) | 0;
  return s.length + ":" + h;
}

export const view = {
  id: "projects",
  title: "Projects",
  async mount(pane, params) {
    lastParams = params;
    isDeep = Boolean(params.id);
    if (params.id) return deep(pane, params.id);
    const everything = params.everything === "1" || params.everything === true;
    const [graph, overview] = await Promise.all([api("/api/projects/graph", { everything }), api("/api/projects/overview")]);
    lastDigest = digestOf(graph.nodes?.map((n) => [n.id, n.updated_at, n.importance, n.health?.state]) || graph);
    const wrap = el("div", { class: "galaxy-wrap" });
    const canvas = el("canvas", { id: "constellation", class: "galaxy" });
    const search = el("input", { placeholder: "Focus… (project, mission, capability)", value: params.focus || params.q || "", style: { maxWidth: "240px" } });
    const everyToggle = el("label", { class: "empty", style: { padding: 0, cursor: "pointer" } }, el("input", { type: "checkbox", checked: everything }), " show everything");
    const counts = el("span", { class: "empty", style: { padding: 0 } });
    const chips = el("span", { class: "galaxy-toolbar", style: { display: "inline-flex", gap: "4px" } });
    const levels = ["ALL", "FOCUS", "ACTIVE", "BLOCKED"];
    let level = params.level || "ALL";
    for (const l of levels) chips.append(el("button", { class: "chip" + (level === l ? " on" : ""), text: l.toLowerCase(), onClick: (ev) => {
      level = l; for (const c of chips.querySelectorAll(".chip")) c.classList.toggle("on", c === ev.currentTarget); galaxy?.setLevel(level); } }));
    // immersive: everything floats OVER the galaxy; nothing frames it.
    // The alternative layout modes stay reachable via ?mode=… but the select
    // is gone: one strong default view beats six half-views in a dropdown.
    const toolbar = el("div", { class: "toolbar galaxy-overlay" }, search, chips, everyToggle, counts);
    const legend = el("div", { class: "galaxy-legend" },
      el("span", { style: { "--c": HEALTH_COLOUR.HEALTHY }, text: "healthy" }), el("span", { style: { "--c": HEALTH_COLOUR.AT_RISK }, text: "at risk" }),
      el("span", { style: { "--c": HEALTH_COLOUR.BLOCKED }, text: "blocked" }), el("span", { style: { "--c": KIND_COLOUR.mission }, text: "mission" }),
      el("span", { style: { "--c": KIND_COLOUR.capability }, text: "capability" }), el("span", { style: { "--c": KIND_COLOUR.knowledge }, text: "knowledge" }));
    const hint = el("div", { class: "galaxy-hint", text: "rauszoomen → universum · doppelklick eintauchen · ziehen · rechtsklick menü · esc zurück" });
    wrap.append(toolbar, canvas, legend, hint);
    wrap.append(focusDrawer(overview, graph, params));
    pane.append(wrap);
    galaxy = new Galaxy(canvas, wrap, graph, {
      onSelect: (n) => inspect(n, graph, () => views.open("projects", params)),
      // double-click = dive INTO the system (camera + context dim), no page swap
      onDoubleClick: (n) => {
        if (!galaxy) return;
        if (n.kind === "project") {
          const keep = new Set([n.id]);
          for (const e of galaxy.edges) { if (e.a === n) keep.add(e.b.id); if (e.b === n) keep.add(e.a.id); }
          galaxy.dim = keep; galaxy.focusId = n.id; galaxy.flyTo(n, 2.05);
        } else { galaxy.focusText(n.label); }
      },
      mode: params.mode || "GALAXY", filter: params.filter || "", uses: params.uses || "", idleDays: Number(params.idle_days || 0), connected: params.connected || "", level,
      // this galaxy is one zoom level of a larger cosmos: pulling out past the
      // floor rises into the Files universe (drives), same engine, same space
      minZoom: 0.5,
      onZoomOutBeyond: () => {
        savedCam = null; warp(canvas, "rise");
        // the scene is parked, not destroyed: reset its camera so returning
        // later doesn't resume at the zoom floor
        if (galaxy) Object.assign(galaxy.cam, { x: 0, y: 0, z: 1, vx: 0, vy: 0 });
        views.open("files", { path: "" });
      } });
    window.zeusGalaxy = galaxy; // console/test access to the live scene
    if (savedCam && !params.focus && !params.id) Object.assign(galaxy.cam, savedCam);
    const nP = graph.nodes.filter((n) => n.kind === "project").length, nM = graph.nodes.filter((n) => n.kind === "mission").length;
    counts.textContent = `${nP} project${nP === 1 ? "" : "s"} · ${nM} mission${nM === 1 ? "" : "s"} · ${graph.nodes.filter((n) => n.kind === "capability").length} capabilities · ${graph.nodes.filter((n) => n.kind === "thought").length} thoughts` + (graph.hidden ? ` · ${graph.hidden} hidden` : "");
    search.oninput = () => galaxy.focusText(search.value);
    if (search.value) galaxy.focusText(search.value);
    everyToggle.firstChild.onchange = (e) => views.open("projects", { ...params, everything: e.target.checked ? "1" : "" });
    if (params.focus) galaxy.focusText(params.focus, true);
  },
  unmount() { if (galaxy) savedCam = { ...galaxy.cam }; galaxy?.destroy(); galaxy = null; },

  /* keep-alive: leaving Projects parks the whole scene; coming back is
     instant (same DOM, same camera, no refetch, no relayout). A background
     check refetches the graph and forces a rebuild only if data changed. */
  suspend() { galaxy?.stop(); },
  resume(params) {
    if (!galaxy || isDeep) return false;
    const same = (k) => String(params[k] || "") === String(lastParams?.[k] || "");
    if (params.id || params.focus || params.q) return false;
    if (!same("everything") || !same("mode") || !same("filter") || !same("level")) return false;
    lastParams = params;
    galaxy.resize();
    galaxy.start();
    window.zeusGalaxy = galaxy;
    const everything = params.everything === "1" || params.everything === true;
    api("/api/projects/graph", { everything }).then((graph) => {
      const digest = digestOf(graph.nodes?.map((n) => [n.id, n.updated_at, n.importance, n.health?.state]) || graph);
      if (digest !== lastDigest && views.currentView()?.id === "projects") {
        views.open("projects", params, { push: false, force: true });
      }
    }).catch(() => {});
    return true;
  },
};

/* The focus information lives in a drawer over the galaxy: one summary chip
   line, the full panel on click. It never takes surface away from space. */
function focusDrawer(overview, graph, params) {
  const panel = focusPanel(overview, graph, params);
  const projects = (overview.projects || []).filter((p) => !p.hidden && !["TEST", "ARCHIVED"].includes(p.importance));
  const blocked = projects.filter((p) => p.health?.state === "BLOCKED").length;
  const today = projects.filter((p) => Date.now() - new Date(p.updated_at || 0) < 864e5).length;
  const waiting = (overview.missions || []).filter((m) => m.state === "waiting" || m.owner_input_required).length;
  const drawer = el("div", { class: "focus-drawer", dataset: { open: "false" } });
  const toggle = el("div", { class: "focus-toggle" },
    el("b", { text: "Überblick" }),
    el("span", { text: `heute ${today}` }),
    el("span", { text: `blockiert ${blocked}` }),
    el("span", { text: `wartet auf dich ${waiting}` }));
  toggle.onclick = () => { drawer.dataset.open = drawer.dataset.open === "true" ? "false" : "true"; };
  drawer.append(panel, toggle);
  return drawer;
}

/* ---- the Focus panel: what deserves attention ------------------------- */
function focusPanel(overview, graph, params) {
  const projects = (overview.projects || []).filter((p) => !p.hidden && !["TEST", "ARCHIVED"].includes(p.importance));
  const missions = overview.missions || [];
  const now = Date.now();
  const today = projects.filter((p) => now - new Date(p.updated_at || 0) < 864e5);
  const blocked = projects.filter((p) => p.health?.state === "BLOCKED").concat(missions.filter((m) => m.state === "blocked").map((m) => ({ ...m, isMission: true })));
  const recent = projects.filter((p) => now - new Date(p.updated_at || 0) < 3 * 864e5 && !today.includes(p));
  const needsOwner = missions.filter((m) => m.state === "waiting" || m.owner_input_required);
  const suggests = graph.nodes.filter((n) => n.kind === "thought" && ["NEW", "IMPORTANT"].includes(n.data?.status));
  const col = (title, items, render) => el("div", { class: "focus-col" }, el("h5", { text: title }),
    items.length ? el("div", {}, ...items.slice(0, 5).map(render)) : el("div", { class: "empty", style: { padding: "2px 0" }, text: "—" }));
  const projectRow = (p) => el("div", { class: "focus-row", onClick: () => views.open("projects", { id: p.id }) }, el("span", { class: "dot", style: { background: HEALTH_COLOUR[p.health?.state] || "#6b7c93" } }), el("span", { text: p.title || p.goal?.slice(0, 40) }));
  const missionRow = (m) => el("div", { class: "focus-row", onClick: () => views.open("missions", { mission: m.id }) }, el("span", { class: "dot", style: { background: "#e0b04a" } }), el("span", { text: m.title || m.goal?.slice(0, 40) }));
  const thoughtRow = (t) => el("div", { class: "focus-row", onClick: () => views.open("thoughts") }, el("span", { class: "dot", style: { background: KIND_COLOUR.thought } }), el("span", { text: t.label }));
  return el("div", { class: "focus-panel" },
    col("Today", today, projectRow), col("Blocked", blocked, (x) => (x.isMission ? missionRow(x) : projectRow(x))), col("Recently active", recent, projectRow),
    col("Needs owner", needsOwner, missionRow), col("ZEUS suggests", suggests, thoughtRow));
}

/* ---- the galaxy engine -------------------------------------------------
   Shared with the File Galaxy (views/files.js): the class is exported and
   gains only additive hooks (custom hue, double-click/context callbacks,
   dynamic sub-graphs). The Projects rendering itself is unchanged. */
export class Galaxy {
  constructor(canvas, wrap, graph, opts) {
    this.canvas = canvas; this.wrap = wrap; this.ctx = canvas.getContext("2d"); this.opts = opts;
    this.nodes = []; this.edges = []; this.byId = new Map();
    this.cam = { x: 0, y: 0, z: 1, vx: 0, vy: 0 };
    this.selected = new Set(); this.hover = null; this.drag = null; this.box = null; this.focusId = null; this.dim = null;
    this.frames = 0; this.settled = false; this.raf = 0; this.mode = opts.mode || "GALAXY"; this.t = 0; this.level = opts.level || "ALL";
    this.menu = null;
    // three parallax layers of stars and a sparse particle drift
    const rnd = mulberry(7);
    this.layers = [0.35, 0.7, 1.15].map((depth, i) => ({ depth, stars: Array.from({ length: [260, 140, 60][i] }, () => ({ x: rnd(), y: rnd(), s: [0.7, 1.1, 1.7][i] * (0.7 + rnd() * 0.6), a: 0.25 + rnd() * 0.6, tw: rnd() * 6.28, hue: rnd() })) }));
    this.motes = Array.from({ length: 48 }, () => ({ x: rnd(), y: rnd(), vx: (rnd() - 0.5) * 0.00012, vy: (rnd() - 0.5) * 0.00012, a: 0.08 + rnd() * 0.2, s: 0.8 + rnd() * 1.4 }));
    this.build(graph);
    this.applyFilters();
    this.resize(); this.bind();
    this.layout(true);
    // the first resize can run before flex/grid has settled the canvas box
    // (clientWidth 0 -> the 900px fallback); measure again on the next frame
    requestAnimationFrame(() => { this.resize(); this.layout(true); });
    this.start();
  }

  build(graph) {
    const W = this.W(), H = this.H();
    for (const raw of graph.nodes) {
      const n = { ...raw, x: W / 2, y: H / 2, vx: 0, vy: 0, r: 6, depth: 1, fixed: false, locked: false, ownerPlaced: false, visible: true, seed: hash(raw.id) };
      if (n.kind === "project") {
        const w = IMPORTANCE_WEIGHT[n.importance] || 0.6;
        n.r = raw.baseR || 9 + w * 16 + Math.min(9, (n.tasks || 0) * 0.5); n.depth = 1;
        const layout = n.layout || {};
        if (layout.state === "OWNER_POSITIONED" || layout.state === "LOCKED") { n.x = layout.x; n.y = layout.y; n.ownerPlaced = true; n.locked = layout.state === "LOCKED"; }
      } else if (n.kind === "self") { n.r = 14; n.depth = 1; n.x = W / 2; n.y = H / 2; n.fixed = true; }
      else if (n.kind === "mission") { n.r = 3.2 + (n.state === "active" ? 1.2 : 0); n.depth = 0.85; }
      else if (n.kind === "capability") { n.r = 3.5 + Math.min(3, (n.attempts || 1) * 0.4); n.depth = 0.75; }
      else if (n.kind === "knowledge") { n.r = 34; n.depth = 0.55; }
      else if (n.kind === "thought") { n.r = 3; n.depth = 0.95; }
      this.nodes.push(n); this.byId.set(n.id, n);
    }
    this.edges = graph.edges.filter((e) => this.byId.has(e.source) && this.byId.has(e.target)).map((e) => ({ ...e, a: this.byId.get(e.source), b: this.byId.get(e.target) }));
    for (const e of this.edges) {
      if (e.type === "mission_of" || e.type === "thought" || e.type === "uses" || e.type === "subproject_of") { e.b.parent = e.b.parent || e.a; }
      if (e.type === "subproject_of") { e.b.sub = true; e.b.depth = 0.92; e.b.r = Math.max(6, e.b.r * 0.55); }
    }
  }

  applyFilters() {
    const { filter, uses, idleDays, connected } = this.opts;
    for (const n of this.nodes) n.visible = true;
    if (filter === "blocked" || this.level === "BLOCKED") for (const n of this.nodes) if (n.kind === "project" && n.health?.state !== "BLOCKED") n.visible = false;
    if (this.level === "FOCUS") for (const n of this.nodes) if (n.kind === "project" && !["PINNED", "FOCUS"].includes(n.importance)) n.visible = false;
    if (this.level === "ACTIVE") for (const n of this.nodes) if (n.kind === "project" && !["PINNED", "FOCUS", "ACTIVE"].includes(n.importance)) n.visible = false;
    if (idleDays) for (const n of this.nodes) if (n.kind === "project" && Date.now() - new Date(n.updated_at || 0) < idleDays * 864e5) n.visible = false;
    if (uses) for (const n of this.nodes) if (n.kind === "project" && !this.edges.some((e) => e.a === n && e.b.kind === "capability" && e.b.label.includes(uses))) n.visible = false;
    if (connected) {
      const anchor = this.find(connected);
      if (anchor) { const keep = new Set([anchor.id]); for (const e of this.edges) { if (e.a === anchor) keep.add(e.b.id); if (e.b === anchor) keep.add(e.a.id); } for (const n of this.nodes) n.visible = keep.has(n.id) || n.kind === "self"; }
    }
    // children follow their parent's visibility
    for (const n of this.nodes) if (n.parent && n.parent.kind === "project" && !n.parent.visible) n.visible = false;
  }

  setLevel(level) { this.level = level; this.applyFilters(); this.layout(false); this.start(); }

  W() { return this.canvas.clientWidth || 900; }
  H() { return this.canvas.clientHeight || 480; }
  resize() {
    const w = Math.round(this.W() * devicePixelRatio), h = Math.round(this.H() * devicePixelRatio);
    if (this.canvas.width === w && this.canvas.height === h) return; // no-op resizes must not loop the observer
    this.canvas.width = w; this.canvas.height = h; this.settled = false; this.frames = 0;
  }

  /* ---- layout: systems on a golden-angle spiral, relaxed apart --------- */
  layout(initial) {
    const W = this.W(), H = this.H();
    const systems = this.nodes.filter((n) => n.kind === "project" && n.visible && !n.sub);
    const zeus = this.byId.get("zeus");
    const place = (n, x, y) => { if (!n.ownerPlaced && !n.locked) { if (initial || this.mode !== "GALAXY") { n.x = x; n.y = y; } else { n.tx = x; n.ty = y; } } };
    const systemRadius = (n) => n.r + 30 + Math.min(96, 16 * this.childrenOf(n).length);
    if (this.mode === "GALAXY") {
      systems.sort((a, b) => (IMPORTANCE_WEIGHT[b.importance] || 0) - (IMPORTANCE_WEIGHT[a.importance] || 0) || (a.seed - b.seed));
      const cx = W / 2, cy = H / 2, golden = Math.PI * (3 - Math.sqrt(5));
      const rx = W * 0.40, ry = H * 0.38, count = Math.max(1, systems.length);
      systems.forEach((n, i) => {
        // i=0 nearest the core (but never on it); radius grows with sqrt so area is used evenly
        const f = Math.sqrt((i + 0.85) / (count + 0.5));
        const a = i * golden + (n.seed % 1000) / 1000 * 0.35;
        place(n, cx + Math.cos(a) * rx * f * (0.55 + 0.45 * f) + Math.cos(a) * 60, cy + Math.sin(a) * ry * f * (0.55 + 0.45 * f) + Math.sin(a) * 40);
      });
      // relax: no two systems overlap (their orbit radii included); owner-placed stay
      for (let it = 0; it < 120; it++) {
        let moved = false;
        for (const a of systems) for (const b of systems) {
          if (a === b) continue;
          const ax = a.tx ?? a.x, ay = a.ty ?? a.y, bx = b.tx ?? b.x, by = b.ty ?? b.y;
          const dx = ax - bx, dy = ay - by, d = Math.max(1, Math.hypot(dx, dy)), min = systemRadius(a) + systemRadius(b) + 44;
          if (d < min) {
            const push = (min - d) / 2, ux = dx / d, uy = dy / d;
            if (!a.ownerPlaced && !a.locked) { if (initial) { a.x += ux * push; a.y += uy * push; } else { a.tx = (a.tx ?? a.x) + ux * push; a.ty = (a.ty ?? a.y) + uy * push; } moved = true; }
            if (!b.ownerPlaced && !b.locked) { if (initial) { b.x -= ux * push; b.y -= uy * push; } else { b.tx = (b.tx ?? b.x) - ux * push; b.ty = (b.ty ?? b.y) - uy * push; } moved = true; }
          }
        }
        for (const a of systems) {
          if (a.ownerPlaced || a.locked) continue;
          const R = systemRadius(a) + 10;
          if (initial) { a.x = Math.max(R, Math.min(W - R, a.x)); a.y = Math.max(R, Math.min(H - R - 16, a.y)); }
          else if (a.tx !== undefined) { a.tx = Math.max(R, Math.min(W - R, a.tx)); a.ty = Math.max(R, Math.min(H - R - 16, a.ty)); }
        }
        if (!moved) break;
      }
      if (zeus) { zeus.x = W / 2; zeus.y = H / 2; }
    } else if (this.mode === "DEPENDENCY") {
      const caps = this.nodes.filter((n) => n.kind === "capability" && n.visible);
      caps.forEach((n, i) => { n.x = 80 + (i + 0.5) * (W - 160) / Math.max(1, caps.length); n.y = H * 0.85; n.placedOnce = true; n.orbit = null; });
      systems.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, systems.length), H * 0.28));
    } else if (this.mode === "TIMELINE") {
      const dated = systems.filter((n) => n.updated_at).sort((a, b) => new Date(a.data?.created_at || a.updated_at) - new Date(b.data?.created_at || b.updated_at));
      dated.forEach((n, i) => place(n, 70 + (i + 0.5) * (W - 140) / Math.max(1, dated.length), H * (0.35 + 0.3 * ((i % 2) ? 1 : 0))));
    } else if (this.mode === "HIERARCHY") {
      systems.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, systems.length), H * 0.3));
      if (zeus) { zeus.x = W / 2; zeus.y = H * 0.08; }
    } else if (this.mode === "MISSION FLOW") {
      const ms = this.nodes.filter((n) => n.kind === "mission" && n.visible).sort((a, b) => new Date(a.updated_at || 0) - new Date(b.updated_at || 0));
      ms.forEach((n, i) => { n.tx = 60 + (i + 0.5) * (W - 120) / Math.max(1, ms.length); n.ty = H * 0.55; n.flow = true; });
      systems.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, systems.length), H * 0.2));
    } else if (this.mode === "KNOWLEDGE") {
      const k = this.byId.get("knowledge"); if (k) { k.x = W / 2; k.y = H / 2; k.r = 70; }
      systems.forEach((n, i) => { const a = (i / Math.max(1, systems.length)) * Math.PI * 2; place(n, W / 2 + Math.cos(a) * W * 0.34, H / 2 + Math.sin(a) * H * 0.36); });
    }
    // infrastructure without a project: capability satellites on ZEUS's outer orbit
    const loose = this.nodes.filter((n) => n.kind === "capability" && n.visible && !n.parent);
    loose.forEach((n, i) => { if (!n.placedOnce && zeus) { n.parent = zeus; n.orbit = { a: Math.PI * 0.15 + (i / Math.max(1, loose.length)) * Math.PI * 2, d: 58 + (i % 2) * 14, speed: 0.06 + (n.seed % 7) * 0.01, tilt: 0.55 }; n.placedOnce = true; } });
    // children on orbit rings around their parents: subprojects innermost, missions, then capabilities
    for (const n of this.nodes) {
      if (n.parent && !n.flow && (initial || !n.placedOnce) && (n.kind === "mission" || n.kind === "thought" || n.kind === "capability" || n.sub)) {
        const siblings = this.childrenOf(n.parent).filter((c) => c.kind === n.kind && Boolean(c.sub) === Boolean(n.sub));
        const idx = Math.max(0, siblings.indexOf(n));
        const ring = n.sub ? 1 : n.kind === "mission" ? 2 : n.kind === "thought" ? 2.6 : 3;
        const d = n.parent.r + 18 + ring * 19 + (idx % 2) * 6;
        const a = (idx / Math.max(1, siblings.length)) * Math.PI * 2 + (n.seed % 100) / 100 * 0.6;
        n.orbit = { a, d, speed: (0.05 + (n.seed % 5) * 0.012) * (n.kind === "mission" && n.state === "active" ? 1.8 : 1) * (n.sub ? 0.4 : 1), tilt: 0.62 };
        n.x = n.parent.x + Math.cos(a) * d; n.y = n.parent.y + Math.sin(a) * d * n.orbit.tilt;
        n.placedOnce = true;
      }
      if (n.kind === "knowledge" && this.mode !== "KNOWLEDGE") { n.x = W * 0.84; n.y = H * 0.22; }
    }
    this.settled = false; this.frames = 0;
  }

  childrenOf(p) { return this.nodes.filter((c) => c.parent === p); }
  setMode(mode) { this.mode = mode; this.layout(false); this.start(); }

  /* ---- physics: a short relaxation, then rest ------------------------- */
  step() {
    const movers = this.nodes.filter((n) => n.visible && !n.fixed && !n.locked && n.kind === "project" && !n.sub);
    for (const a of movers) {
      if (a.tx !== undefined) { a.vx += (a.tx - a.x) * 0.04; a.vy += (a.ty - a.y) * 0.04; }
      a.vx *= 0.78; a.vy *= 0.78; a.x += a.vx; a.y += a.vy;
    }
    const slow = REDUCED() ? 0 : 1;
    for (const n of this.nodes) {
      if (n.orbit && n.parent && !n.flow) {
        // hovering or selecting a body freezes its orbit locally, so slow
        // cinematic motion never makes clicking a moving target
        if (n === this.hover || this.selected.has(n.id) || n.parent === this.hover) { continue; }
        n.orbit.a += n.orbit.speed * 0.003 * slow;
        n.x = n.parent.x + Math.cos(n.orbit.a) * n.orbit.d; n.y = n.parent.y + Math.sin(n.orbit.a) * n.orbit.d * (n.orbit.tilt || 1);
      } else if (n.flow && n.tx !== undefined) { n.x += (n.tx - n.x) * 0.1; n.y += (n.ty - n.y) * 0.1; }
    }
    for (const m of this.motes) { m.x = (m.x + m.vx * slow + 1) % 1; m.y = (m.y + m.vy * slow + 1) % 1; }
    this.cam.x += this.cam.vx; this.cam.y += this.cam.vy; this.cam.vx *= 0.88; this.cam.vy *= 0.88;
    this.frames += 1;
    const moving = movers.some((n) => Math.hypot(n.vx, n.vy) > 0.05) || Math.hypot(this.cam.vx, this.cam.vy) > 0.05;
    if (this.frames > 60 && !moving) this.settled = true;
  }

  start() { if (this.raf) return; const tick = () => { if (!this.settled || this.hasMotion()) this.step(); this.draw(); this.t += 1; this.raf = requestAnimationFrame(tick); }; this.raf = requestAnimationFrame(tick); }
  hasMotion() { return !REDUCED(); }
  // stop pauses the frame loop but keeps the whole scene (nodes, camera,
  // layout) alive — the keep-alive views park with stop() and wake with start()
  stop() { cancelAnimationFrame(this.raf); this.raf = 0; }
  destroy() { this.stop(); this.closeMenu(); window.removeEventListener("keydown", this._onKey, true); window.removeEventListener("resize", this._onResize); }

  /* ---- drawing -------------------------------------------------------- */
  toScreen(n) { return [(n.x - this.cam.x) * this.cam.z + this.W() / 2 * (1 - this.cam.z), (n.y - this.cam.y) * this.cam.z + this.H() / 2 * (1 - this.cam.z)]; }
  fromScreen(sx, sy) { return [(sx - this.W() / 2 * (1 - this.cam.z)) / this.cam.z + this.cam.x, (sy - this.H() / 2 * (1 - this.cam.z)) / this.cam.z + this.cam.y]; }
  lod() { return this.cam.z < 0.85 ? 0 : this.cam.z < 1.45 ? 1 : 2; }

  draw() {
    const ctx = this.ctx, W = this.W(), H = this.H(), z = this.cam.z, t = this.t, lod = this.lod();
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    // deep space: a slow gradient wash with two faint nebular tints
    const bg = ctx.createRadialGradient(W * 0.5, H * 0.45, 20, W * 0.5, H * 0.45, Math.max(W, H) * 0.75);
    bg.addColorStop(0, "#0b1424"); bg.addColorStop(0.55, "#060a14"); bg.addColorStop(1, "#02040a");
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    for (const [fx, fy, col] of [[0.18, 0.72, "rgba(60,90,160,.10)"], [0.8, 0.3, "rgba(120,80,170,.08)"]]) {
      const g = ctx.createRadialGradient(W * fx - this.cam.x * 0.03, H * fy - this.cam.y * 0.03, 0, W * fx, H * fy, Math.max(W, H) * 0.45);
      g.addColorStop(0, col); g.addColorStop(1, "transparent"); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
    // parallax star layers
    for (const layer of this.layers) {
      const px = this.cam.x * 0.06 * layer.depth, py = this.cam.y * 0.06 * layer.depth;
      for (const s of layer.stars) {
        const x = ((s.x * W - px) % W + W) % W, y = ((s.y * H - py) % H + H) % H;
        const tw = REDUCED() ? 1 : 0.75 + 0.25 * Math.sin(t / 40 + s.tw);
        ctx.fillStyle = s.hue > 0.85 ? `rgba(200,215,255,${s.a * tw})` : s.hue < 0.08 ? `rgba(255,225,190,${s.a * tw})` : `rgba(170,195,235,${s.a * tw})`;
        ctx.beginPath(); ctx.arc(x, y, s.s * (0.8 + 0.2 * z), 0, Math.PI * 2); ctx.fill();
      }
    }
    // particle drift
    for (const m of this.motes) { ctx.fillStyle = `rgba(150,190,240,${m.a})`; ctx.beginPath(); ctx.arc(m.x * W, m.y * H, m.s, 0, Math.PI * 2); ctx.fill(); }
    // under-layer (category nebulas in the File universe) sits behind bodies
    if (this.opts.drawUnder) this.opts.drawUnder(ctx, this);
    // knowledge nebula
    const k = this.byId.get("knowledge");
    if (k && k.visible) {
      const [kx, ky] = this.toScreen(k);
      for (let i = 0; i < 4; i++) {
        const ox = Math.cos(t / 400 + i * 1.7) * 14 * z, oy = Math.sin(t / 360 + i * 2.1) * 10 * z, rr = k.r * (1.6 + i * 0.5) * z;
        const g = ctx.createRadialGradient(kx + ox, ky + oy, 0, kx + ox, ky + oy, rr);
        g.addColorStop(0, i % 2 ? "rgba(140,110,240,.16)" : "rgba(90,130,240,.12)"); g.addColorStop(1, "transparent");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(kx + ox, ky + oy, rr, 0, Math.PI * 2); ctx.fill();
      }
    }
    // orbit rings (faint ellipses) for every system with children
    for (const p of this.nodes) {
      if (!p.visible || (p.kind !== "project" && p.kind !== "self")) continue;
      const rings = new Set(this.childrenOf(p).filter((c) => c.visible && c.orbit).map((c) => Math.round(c.orbit.d)));
      if (!rings.size) continue;
      const [px, py] = this.toScreen(p);
      const dimmed = this.dim && !this.dim.has(p.id);
      for (const d of rings) {
        ctx.strokeStyle = dimmed ? "rgba(140,170,220,.04)" : "rgba(140,170,220,.10)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.ellipse(px, py, d * z, d * z * 0.62, 0, 0, Math.PI * 2); ctx.stroke();
      }
    }
    // relations: energy paths for active relations, faint otherwise; thoughts pulse along them
    for (const e of this.edges) {
      if (!e.a.visible || !e.b.visible) continue;
      if (e.type === "mission_of" || e.type === "uses" || e.type === "subproject_of") { if (lod < 2 && !e.active) continue; }
      const [ax, ay] = this.toScreen(e.a), [bx, by] = this.toScreen(e.b);
      const dimmed = this.dim && !this.dim.has(e.a.id) && !this.dim.has(e.b.id);
      ctx.strokeStyle = e.type === "thought" ? `rgba(240,198,116,${dimmed ? .05 : .28})` : e.active ? `rgba(127,224,180,${dimmed ? .05 : .32})` : `rgba(110,160,230,${dimmed ? .03 : .10})`;
      ctx.lineWidth = e.active ? 1.3 : 0.8; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      if (e.active && !REDUCED()) { const p = (t / 90) % 1; ctx.fillStyle = "rgba(127,224,180,.85)"; ctx.beginPath(); ctx.arc(ax + (bx - ax) * p, ay + (by - ay) * p, 1.8, 0, Math.PI * 2); ctx.fill(); }
      if (e.type === "thought" && !REDUCED()) { const p = (t / 140 + e.b.seed % 100 / 100) % 1; ctx.fillStyle = "rgba(240,198,116,.9)"; ctx.beginPath(); ctx.arc(ax + (bx - ax) * p, ay + (by - ay) * p, 2, 0, Math.PI * 2); ctx.fill(); }
    }
    // bodies, far to near
    const labels = [];
    for (const n of [...this.nodes].sort((a, b) => a.depth - b.depth)) {
      if (!n.visible) continue;
      if (n.kind === "knowledge") { const [x, y] = this.toScreen(n); labels.push({ n, x, y: y + 6, size: 11, text: n.label }); continue; }
      // semantic LOD: detail fades IN as the camera closes, never pops
      const ramp = (a, b) => Math.max(0, Math.min(1, (z - a) / (b - a)));
      const detail = (n.kind === "project" || n.kind === "self") ? 1
        : n.sub ? ramp(0.7, 1.0)
        : n.kind === "capability" ? ramp(0.8, 1.15)
        : n.kind === "mission" ? ramp(0.95, 1.3)
        : n.kind === "thought" ? ramp(1.1, 1.5)
        : ramp(1.25, 1.6);
      const emphasized = n === this.hover || this.selected.has(n.id);
      if (detail <= 0.02 && !emphasized) continue;
      const [x, y] = this.toScreen(n); const r = n.r * z * (0.6 + 0.4 * n.depth);
      const dimmed = this.dim && !this.dim.has(n.id);
      ctx.globalAlpha = dimmed ? 0.18 : emphasized ? 1 : detail;
      if (n.kind === "project") this.drawStar(ctx, n, x, y, r, t);
      else if (n.kind === "self") this.drawCore(ctx, x, y, r, t);
      else this.drawBody(ctx, n, x, y, r, t);
      ctx.globalAlpha = 1;
      // labels arrive later than bodies: names first, detail labels near.
      // A view with hundreds of same-kind bodies (Wissen) supplies labelFor
      // to keep the sky readable: names appear on approach, not all at once.
      const wantedLabel = this.opts.labelFor ? this.opts.labelFor(n, z, emphasized)
        : (n.kind === "project" || n.kind === "self" || detail > 0.55 || emphasized);
      if (wantedLabel && !dimmed) labels.push({ n, x, y: y + r + 5, size: n.kind === "self" ? 12 : n.kind === "project" ? (n.sub ? 10 : 11.5) : 9.5, text: n.kind === "capability" ? `${n.label}` : n.label });
    }
    // labels without collisions (priority: self, systems, then the rest)
    ctx.textAlign = "center"; const placed = [];
    for (const l of labels.sort((a, b) => b.size - a.size)) {
      ctx.font = `${l.size}px Segoe UI, sans-serif`; const w = ctx.measureText(l.text).width + 8, h = l.size + 4;
      const box = { x: l.x - w / 2, y: l.y, w, h };
      if (placed.some((b) => !(box.x + box.w < b.x || b.x + b.w < box.x || box.y + box.h < b.y || b.y + b.h < box.y))) { if (l.n.kind !== "project" && l.n.kind !== "self") continue; }
      placed.push(box);
      ctx.fillStyle = "rgba(4,7,14,.55)"; ctx.fillRect(box.x, box.y + 1, box.w, box.h);
      ctx.fillStyle = l.n === this.hover ? "#ffffff" : l.n.kind === "project" ? "#d5e1f2" : l.n.kind === "knowledge" ? "#c6bcf5" : "#93a4bd";
      ctx.fillText(l.text, l.x, l.y + l.size);
    }
    if (this.box) { ctx.strokeStyle = "rgba(143,211,255,.7)"; ctx.setLineDash([4, 3]); ctx.strokeRect(this.box.x, this.box.y, this.box.w, this.box.h); ctx.setLineDash([]); }
    if (this.opts.drawExtras) this.opts.drawExtras(ctx, this);
  }

  drawStar(ctx, n, x, y, r, t) {
    const hs = n.health?.state || "HEALTHY", [cr, cg, cb] = n.hue || HEALTH_HUE[hs] || HEALTH_HUE.HEALTHY;
    const faded = ["DORMANT", "ARCHIVED", "LOW_PRIORITY", "TEST"].includes(n.importance);
    const pulse = (["ACTIVE", "FOCUS", "PINNED"].includes(n.importance) && !REDUCED()) ? 1 + Math.sin(t / 28 + n.seed) * 0.05 : 1;
    const sel = this.selected.has(n.id) || n === this.hover;
    // planetary ring: seeded per body so it is stable; larger bodies only
    const ringed = (n.ringed ?? (n.seed % 5 === 0)) && r > 8;
    const ringTilt = 0.26 + (n.seed % 37) / 130, ringRot = ((n.seed % 63) / 100) - 0.31;
    // health halo (wide, soft)
    let g = ctx.createRadialGradient(x, y, r * 0.6, x, y, r * 3.2 * pulse);
    g.addColorStop(0, `rgba(${cr},${cg},${cb},${faded ? .12 : .22})`); g.addColorStop(0.5, `rgba(${cr},${cg},${cb},${faded ? .04 : .08})`); g.addColorStop(1, "transparent");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3.2 * pulse, 0, Math.PI * 2); ctx.fill();
    // corona
    g = ctx.createRadialGradient(x, y, r * 0.2, x, y, r * 1.55);
    g.addColorStop(0, `rgba(255,255,255,${faded ? .55 : .95})`); g.addColorStop(0.35, `rgba(${cr},${cg},${cb},${faded ? .55 : .9})`); g.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 1.55, 0, Math.PI * 2); ctx.fill();
    // the far half of the ring passes BEHIND the body
    if (ringed) {
      ctx.strokeStyle = `rgba(${cr},${cg},${cb},${faded ? .12 : .22})`; ctx.lineWidth = Math.max(1, r * 0.10);
      ctx.beginPath(); ctx.ellipse(x, y, r * 1.8, r * 1.8 * ringTilt, ringRot, Math.PI, Math.PI * 2); ctx.stroke();
    }
    // core
    g = ctx.createRadialGradient(x - r * 0.25, y - r * 0.25, 0, x, y, r);
    g.addColorStop(0, "#ffffff"); g.addColorStop(0.55, `rgb(${Math.min(255, cr + 60)},${Math.min(255, cg + 60)},${Math.min(255, cb + 60)})`); g.addColorStop(1, `rgba(${cr},${cg},${cb},.85)`);
    ctx.fillStyle = sel ? "#ffffff" : g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    // terminator: a soft shadow on the side away from the light gives the
    // disc a sphere's face instead of a flat dot
    if (r > 5 && !sel) {
      ctx.save(); ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.clip();
      g = ctx.createRadialGradient(x - r * 0.55, y - r * 0.55, r * 0.2, x, y, r * 1.45);
      g.addColorStop(0, "rgba(3,6,14,0)"); g.addColorStop(0.68, "rgba(3,6,14,0)"); g.addColorStop(1, "rgba(3,6,14,.55)");
      ctx.fillStyle = g; ctx.fillRect(x - r, y - r, r * 2, r * 2);
      ctx.restore();
    }
    // the near half of the ring passes IN FRONT
    if (ringed) {
      ctx.strokeStyle = `rgba(${cr},${cg},${cb},${faded ? .2 : .4})`; ctx.lineWidth = Math.max(1, r * 0.10);
      ctx.beginPath(); ctx.ellipse(x, y, r * 1.8, r * 1.8 * ringTilt, ringRot, 0, Math.PI); ctx.stroke();
    }
    // diffraction spikes for the important systems
    if (["PINNED", "FOCUS", "ACTIVE"].includes(n.importance) && !faded) {
      ctx.strokeStyle = `rgba(255,255,255,${.18 * pulse})`; ctx.lineWidth = 1;
      for (const a of [0, Math.PI / 2]) { ctx.beginPath(); ctx.moveTo(x + Math.cos(a) * r * 2.6, y + Math.sin(a) * r * 2.6); ctx.lineTo(x - Math.cos(a) * r * 2.6, y - Math.sin(a) * r * 2.6); ctx.stroke(); }
    }
    if (hs === "BLOCKED") { ctx.strokeStyle = `rgba(255,120,120,${.35 + .25 * Math.sin(t / 18)})`; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, r * 1.9, 0, Math.PI * 2); ctx.stroke(); }
    if (n.locked) { ctx.strokeStyle = "#dce5f0"; ctx.lineWidth = 1; ctx.strokeRect(x - 3, y - r - 10, 6, 5); }
    if (n.importance === "PINNED") { ctx.fillStyle = "#f0c674"; ctx.beginPath(); ctx.arc(x + r * 0.95, y - r * 0.95, 2.6, 0, Math.PI * 2); ctx.fill(); }
    if (sel) { ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.arc(x, y, r + 6, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
    // progress arc: tasks done
    if (n.tasks) { const f = (n.tasks_done || 0) / n.tasks; ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, r + 3, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * f); ctx.stroke(); }
  }

  drawCore(ctx, x, y, r, t) {
    const pulse = REDUCED() ? 1 : 1 + Math.sin(t / 22) * 0.06;
    let g = ctx.createRadialGradient(x, y, 0, x, y, r * 4 * pulse);
    g.addColorStop(0, "rgba(159,220,255,.35)"); g.addColorStop(0.4, "rgba(79,195,247,.12)"); g.addColorStop(1, "transparent");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 4 * pulse, 0, Math.PI * 2); ctx.fill();
    g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, "#ffffff"); g.addColorStop(0.5, "#bfe8ff"); g.addColorStop(1, "rgba(79,195,247,.6)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(159,220,255,.35)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r * 1.8, t / 60, t / 60 + Math.PI * 1.3); ctx.stroke();
  }

  drawBody(ctx, n, x, y, r, t) {
    const colour = KIND_COLOUR[n.kind] || "#6b7c93";
    const sel = this.selected.has(n.id) || n === this.hover;
    if (n.kind === "mission" && n.state === "active") { const g = ctx.createRadialGradient(x, y, 0, x, y, r * 4); g.addColorStop(0, "rgba(127,224,180,.35)"); g.addColorStop(1, "transparent"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 4, 0, Math.PI * 2); ctx.fill(); }
    if (n.kind === "thought") { const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3.5); g.addColorStop(0, `rgba(240,198,116,${.35 + .25 * Math.sin(t / 20 + n.seed)})`); g.addColorStop(1, "transparent"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3.5, 0, Math.PI * 2); ctx.fill(); }
    const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
    g.addColorStop(0, "#ffffff"); g.addColorStop(0.6, colour); g.addColorStop(1, colour + "aa");
    ctx.fillStyle = sel ? "#ffffff" : g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    // even the small moons get a face: a hint of shadow away from the light
    if (r > 4 && !sel) {
      ctx.save(); ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.clip();
      ctx.fillStyle = "rgba(3,6,14,.4)"; ctx.beginPath(); ctx.arc(x + r * 0.55, y + r * 0.55, r, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }
    if (n.kind === "capability") { ctx.strokeStyle = "rgba(143,161,194,.6)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r + 2.5, 0.3, Math.PI * 1.4); ctx.stroke(); }
    if (n.kind === "mission" && ["blocked", "failed"].includes(n.state)) { ctx.strokeStyle = "rgba(255,120,120,.7)"; ctx.beginPath(); ctx.arc(x, y, r + 2, 0, Math.PI * 2); ctx.stroke(); }
  }

  /* ---- interaction ---------------------------------------------------- */
  hit(sx, sy) {
    const [x, y] = this.fromScreen(sx, sy); const lod = this.lod(); let best = null;
    for (const n of this.nodes) {
      if (!n.visible) continue;
      const shown = n.kind === "project" || n.kind === "self" || n.kind === "knowledge" || (lod >= 1 && (n.kind === "capability" || n.sub)) || lod >= 2;
      if (!shown) continue;
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < n.r + 7 / this.cam.z && (!best || n.depth > best.depth)) best = n;
    }
    return best;
  }
  find(text) { const q = String(text || "").toLowerCase(); return this.nodes.find((n) => n.visible && n.label.toLowerCase().includes(q)) || null; }
  focusText(text, open = false) {
    const n = this.find(text);
    if (!text) { this.dim = null; this.focusId = null; return; }
    if (!n) return;
    const keep = new Set([n.id]); for (const e of this.edges) { if (e.a === n) keep.add(e.b.id); if (e.b === n) keep.add(e.a.id); }
    this.dim = keep; this.focusId = n.id; this.flyTo(n, 1.7);
    if (open) this.opts.onSelect(n);
  }
  flyTo(n, z) {
    // centres the GIVEN POINT: the camera origin is the world point at the
    // top-left, so centring n means flying to n minus half a screen
    const tx = n.x - this.W() / 2, ty = n.y - this.H() / 2;
    const steps = REDUCED() ? 1 : 34; const sx = this.cam.x, sy = this.cam.y, sz = this.cam.z; let i = 0;
    const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const go = () => { i += 1; const t = ease(i / steps); this.cam.x = sx + (tx - sx) * t; this.cam.y = sy + (ty - sy) * t; this.cam.z = sz + (z - sz) * t; if (i < steps) requestAnimationFrame(go); };
    go(); this.settled = false;
  }
  bind() {
    const c = this.canvas;
    c.onmousemove = (e) => {
      const r = c.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top;
      if (this.drag) {
        const [x, y] = this.fromScreen(sx, sy);
        if (this.drag.kind === "pan") { this.cam.x = this.drag.camX - (sx - this.drag.sx) / this.cam.z; this.cam.y = this.drag.camY - (sy - this.drag.sy) / this.cam.z; this.cam.vx = -(e.movementX) / this.cam.z * 0.25; this.cam.vy = -(e.movementY) / this.cam.z * 0.25; }
        else if (this.drag.kind === "node") { for (const n of this.drag.nodes) { if (n.locked) continue; n.x = x + n.dx; n.y = y + n.dy; n.tx = n.ty = undefined; n.vx = n.vy = 0; } this.drag.moved = true; }
        else if (this.drag.kind === "box") { this.box = { x: Math.min(this.drag.sx, sx), y: Math.min(this.drag.sy, sy), w: Math.abs(sx - this.drag.sx), h: Math.abs(sy - this.drag.sy) }; }
        this.settled = false; return;
      }
      this.hover = this.hit(sx, sy); c.style.cursor = this.hover ? "pointer" : "grab"; this.settled = false;
    };
    c.onmousedown = (e) => {
      if (e.button === 2) return;
      this.closeMenu();
      const r = c.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top; const n = this.hit(sx, sy);
      if (e.shiftKey && !n) { this.drag = { kind: "box", sx, sy }; return; }
      if (n && n.kind === "project") {
        if (!this.selected.has(n.id) && !e.ctrlKey) this.selected = new Set([n.id]); else this.selected.add(n.id);
        const [x, y] = this.fromScreen(sx, sy);
        const nodes = [...this.selected].map((id) => this.byId.get(id)).filter(Boolean); for (const m of nodes) { m.dx = m.x - x; m.dy = m.y - y; }
        this.drag = { kind: "node", nodes, moved: false }; c.style.cursor = "grabbing"; return;
      }
      this.drag = { kind: "pan", sx, sy, camX: this.cam.x, camY: this.cam.y }; c.style.cursor = "grabbing";
    };
    const finish = async (e) => {
      const d = this.drag; this.drag = null; c.style.cursor = "grab";
      if (!d) return;
      if (d.kind === "box" && this.box) {
        const b = this.box; this.box = null; this.selected = new Set();
        for (const n of this.nodes) { if (n.kind !== "project" || !n.visible) continue; const [x, y] = this.toScreen(n); if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) this.selected.add(n.id); }
        return;
      }
      if (d.kind === "node") {
        if (d.moved) { for (const n of d.nodes) { if (n.locked) continue; n.ownerPlaced = true; n.layout = { ...(n.layout || {}), x: n.x, y: n.y, state: "OWNER_POSITIONED" }; if (this.opts.persistDrag !== false) await api("/api/project/update", { id: n.id, layout: { x: n.x, y: n.y, state: "OWNER_POSITIONED" } }); } }
        else if (d.nodes.length === 1) this.opts.onSelect(d.nodes[0]);
        return;
      }
      if (d.kind === "pan") { const r = c.getBoundingClientRect(); const n = this.hit(e.clientX - r.left, e.clientY - r.top); if (n && Math.abs(e.clientX - r.left - d.sx) < 3 && Math.abs(e.clientY - r.top - d.sy) < 3) this.opts.onSelect(n); }
    };
    c.onmouseup = finish; c.onmouseleave = () => { if (this.drag?.kind === "pan") this.drag = null; this.hover = null; };
    c.ondblclick = (e) => {
      const r = c.getBoundingClientRect(); const n = this.hit(e.clientX - r.left, e.clientY - r.top);
      if (n && this.opts.onDoubleClick) { this.opts.onDoubleClick(n); return; }
      if (n?.kind === "project") views.open("projects", { id: n.id }); else if (n) this.focusText(n.label);
    };
    c.onwheel = (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const minZ = this.opts.minZoom ?? 0.45, maxZ = this.opts.maxZoom ?? 3.4;
      const [wx, wy] = this.fromScreen(sx, sy);
      // semantic zoom: pressing on past the floor is a deliberate gesture and
      // rises to the next larger level (folder → drive → universe → …)
      if (e.deltaY > 0 && this.cam.z <= minZ + 0.001) {
        this._outPressure = (this._outPressure || 0) + 1;
        if (this._outPressure >= 3 && this.opts.onZoomOutBeyond) { this._outPressure = 0; this.opts.onZoomOutBeyond(); return; }
      } else this._outPressure = 0;
      const z = Math.max(minZ, Math.min(maxZ, this.cam.z * (e.deltaY < 0 ? 1.1 : 0.91)));
      this.cam.z = z; const [nx, ny] = this.fromScreen(sx, sy); this.cam.x += wx - nx; this.cam.y += wy - ny; this.settled = false;
    };
    c.oncontextmenu = (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect(); const n = this.hit(e.clientX - r.left, e.clientY - r.top);
      if (n && this.opts.onContext) { if (!this.selected.has(n.id)) this.selected = new Set([n.id]); this.opts.onContext(n, e.clientX - r.left, e.clientY - r.top); return; }
      if (n && n.kind === "project") { if (!this.selected.has(n.id)) this.selected = new Set([n.id]); this.openMenu(n, e.clientX - r.left, e.clientY - r.top); } else this.closeMenu();
    };
    // stopImmediatePropagation: the File view's own ESC handler sits on the
    // same window node and would otherwise see the menu already closed and
    // navigate up one level on the same keypress
    this._onKey = (e) => { if (e.key === "Escape") { if (this.menu) { this.closeMenu(); e.stopImmediatePropagation(); return; } if (this.dim || this.cam.z !== 1) { this.dim = null; this.focusId = null; this.flyTo({ x: this.W() / 2, y: this.H() / 2 }, 1); e.stopImmediatePropagation(); } } };
    window.addEventListener("keydown", this._onKey, true);
    this._onResize = () => this.resize(); window.addEventListener("resize", this._onResize);
    // NO ResizeObserver on the canvas: its backing-store writes feed back
    // into an auto-height flex chain and inflate the canvas without bound.
    // Layout-driven size changes are avoided instead (the inspector floats
    // over space rather than shrinking it).
  }
  /* ---- dynamic sub-graphs (the File Galaxy's semantic zoom) ----------- */
  addGraph(nodes, edges, { around = null } = {}) {
    const added = [];
    for (const raw of nodes) {
      if (this.byId.has(raw.id)) continue;
      const base = around || this.byId.get(raw.parentId) || { x: this.W() / 2, y: this.H() / 2, r: 20 };
      const a = Math.random() * Math.PI * 2;
      const n = { visible: true, fixed: false, locked: false, ownerPlaced: false, vx: 0, vy: 0, depth: 0.9, r: 6,
                  seed: hash(raw.id), ...raw, x: base.x + Math.cos(a) * (base.r + 30), y: base.y + Math.sin(a) * (base.r + 22) };
      this.nodes.push(n); this.byId.set(n.id, n);
      added.push(n);
    }
    for (const e of edges || []) {
      if (!this.byId.has(e.source) || !this.byId.has(e.target)) continue;
      const edge = { ...e, a: this.byId.get(e.source), b: this.byId.get(e.target) };
      this.edges.push(edge);
      if (e.type === "mission_of" || e.type === "uses" || e.type === "subproject_of" || e.type === "thought") edge.b.parent = edge.b.parent || edge.a;
      if (e.type === "subproject_of") { edge.b.sub = true; }
    }
    // place the new children on orbit rings around their parent
    for (const n of added) {
      if (!n.parent || n.orbit) continue;
      const siblings = this.childrenOf(n.parent).filter((c) => c.kind === n.kind && Boolean(c.sub) === Boolean(n.sub));
      const idx = Math.max(0, siblings.indexOf(n));
      const ring = n.sub ? 1 : n.kind === "capability" ? 2.2 : 1.6;
      const d = n.parent.r + 18 + ring * 19 + (idx % 3) * 7;
      const a = (idx / Math.max(1, siblings.length)) * Math.PI * 2 + (n.seed % 100) / 100 * 0.6;
      n.orbit = { a, d, speed: (0.05 + (n.seed % 5) * 0.012) * (n.sub ? 0.4 : 0.8), tilt: 0.62 };
      n.x = n.parent.x + Math.cos(a) * d; n.y = n.parent.y + Math.sin(a) * d * n.orbit.tilt;
      n.placedOnce = true;
    }
    this.settled = false;
    return added;
  }

  removeSubtree(id, { keepRoot = true } = {}) {
    const doomed = new Set();
    const collect = (nodeId) => {
      for (const e of this.edges) {
        if (e.a?.id === nodeId && (e.type === "subproject_of" || e.type === "uses" || e.type === "mission_of")) {
          if (!doomed.has(e.b.id)) { doomed.add(e.b.id); collect(e.b.id); }
        }
      }
    };
    collect(id);
    if (!keepRoot) doomed.add(id);
    if (!doomed.size) return 0;
    this.nodes = this.nodes.filter((n) => !doomed.has(n.id));
    for (const gone of doomed) this.byId.delete(gone);
    this.edges = this.edges.filter((e) => !doomed.has(e.a?.id) && !doomed.has(e.b?.id));
    this.selected = new Set([...this.selected].filter((s) => !doomed.has(s)));
    this.settled = false;
    return doomed.size;
  }

  lock(n, locked) { n.locked = locked; n.layout = { ...(n.layout || {}), x: n.x, y: n.y, state: locked ? "LOCKED" : "OWNER_POSITIONED" }; return api("/api/project/update", { id: n.id, layout: n.layout }); }
  release(n) { n.ownerPlaced = false; n.locked = false; n.layout = { state: "AUTO_POSITIONED" }; this.layout(false); return api("/api/project/update", { id: n.id, layout: { state: "AUTO_POSITIONED" } }); }

  /* ---- context menu --------------------------------------------------- */
  closeMenu() { if (this.menu) { this.menu.remove(); this.menu = null; } }
  openMenu(n, sx, sy) {
    this.closeMenu();
    const p = n.data || {};
    const reload = () => views.open("projects", {});
    const item = (label, fn) => el("button", { text: label, onClick: async () => { this.closeMenu(); await fn(); } });
    const menu = el("div", { class: "galaxy-menu" }, el("h6", { text: p.title || n.label }),
      item("Focus", () => this.focusText(n.label)),
      item("Open", () => views.open("projects", { id: n.id })),
      item(p.importance === "PINNED" ? "Unpin" : "Pin", async () => { await api("/api/project/update", { id: n.id, importance: p.importance === "PINNED" ? "NORMAL" : "PINNED" }); reload(); }),
      item(n.locked ? "Release position" : "Lock position", async () => { if (n.locked) await this.release(n); else await this.lock(n, true); reload(); }),
      item(p.hidden ? "Unhide" : "Hide", async () => { await api("/api/project/update", { id: n.id, hidden: !p.hidden }); reload(); }),
      item("Archive", async () => { await api("/api/project/update", { id: n.id, importance: "ARCHIVED" }); reload(); }),
      el("div", { class: "sep" }), el("h6", { text: "Importance" }),
      el("div", { class: "row" }, ...IMPORTANCE.map((i) => el("button", { class: i === (p.importance || n.importance) ? "on" : "", text: i.toLowerCase().replace("_", " "), onClick: async () => { this.closeMenu(); await api("/api/project/update", { id: n.id, importance: i }); reload(); } }))),
      el("div", { class: "sep" }),
      item("Create mission", () => chat.send(`Zeus, starte eine Mission für das Projekt „${p.title || n.label}“: nächster sinnvoller Schritt.`, "galaxy")),
      item("Ask Zeus", () => chat.send(`Wie steht das Projekt „${p.title || n.label}“? Was blockiert es und was ist der nächste Schritt?`, "galaxy")),
      item("Local graph", () => views.open("projects", { connected: p.title || n.label })));
    const W = this.W(), H = this.H();
    menu.style.left = Math.min(sx + 6, W - 210) + "px"; menu.style.top = Math.min(sy + 6, H - 330) + "px";
    this.wrap.append(menu); this.menu = menu;
  }
}

/* A brief cinematic hand-over between zoom levels: the outgoing canvas is
   frozen as a ghost that scales past the viewer (dive) or shrinks away
   (rise) while the next level mounts underneath. Decoration only — it must
   never be able to block navigation. */
export function warp(canvas, direction = "rise") {
  if (REDUCED() || !canvas || !canvas.width) return;
  try {
    const ghost = document.createElement("canvas");
    ghost.width = canvas.width; ghost.height = canvas.height;
    ghost.getContext("2d").drawImage(canvas, 0, 0);
    const r = canvas.getBoundingClientRect();
    Object.assign(ghost.style, {
      position: "fixed", left: r.left + "px", top: r.top + "px", width: r.width + "px", height: r.height + "px",
      zIndex: 60, pointerEvents: "none", opacity: "1",
      transition: "transform 420ms cubic-bezier(.55,0,.18,1), opacity 420ms cubic-bezier(.55,0,.18,1)",
    });
    document.body.append(ghost);
    requestAnimationFrame(() => { ghost.style.opacity = "0"; ghost.style.transform = direction === "dive" ? "scale(1.65)" : "scale(0.5)"; });
    setTimeout(() => ghost.remove(), 470);
  } catch { /* a failed ghost is a skipped effect, nothing more */ }
}

function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function hash(s) { let h = 2166136261; for (const ch of String(s)) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0) % 100000; }

/* ---- inspector -------------------------------------------------------- */
async function inspect(n, graph, reload) {
  if (n.kind === "mission") return views.open("missions", { mission: n.id });
  if (n.kind === "capability") return views.open("capabilities", { id: n.label });
  if (n.kind === "thought") return views.open("thoughts");
  if (n.kind === "knowledge") return views.open("knowledge");
  if (n.kind === "self") return views.open("missions");
  const p = n.data || {};
  const [detail, knowledge] = await Promise.all([api("/api/project", { id: n.id }), api("/api/knowledge/graph", { query: (p.title || n.label || "").slice(0, 40), limit: 8 }).catch(() => ({}))]);
  const tasks = detail.tasks || [];
  const current = tasks.find((t) => ["active", "running", "working"].includes(String(t.status)));
  const blocked = tasks.filter((t) => ["blocked", "failed"].includes(String(t.status)));
  const next = tasks.find((t) => ["todo", "pending", "open", "planned"].includes(String(t.status)));
  const done = tasks.filter((t) => ["done", "complete", "completed", "accepted"].includes(String(t.status))).length;
  const related = graph.edges.filter((e) => e.source === n.id || e.target === n.id).map((e) => graph.nodes.find((x) => x.id === (e.source === n.id ? e.target : e.source))).filter(Boolean);
  const importance = el("select", {}, ...IMPORTANCE.map((i) => el("option", { value: i, text: i.toLowerCase().replace("_", " "), selected: (p.importance || n.importance) === i })));
  importance.onchange = async () => { await api("/api/project/update", { id: n.id, importance: importance.value }); reload(); };
  const note = el("input", { placeholder: "owner note…" });
  const meta = detail.metadata || {};
  const knowledgeNodes = (knowledge && knowledge.nodes) || [];
  views.inspect(p.title || n.label,
    el("div", { class: "meta" }, badge(p.health?.state || "?", p.health?.state === "HEALTHY" ? "ok" : p.health?.state === "BLOCKED" ? "bad" : p.health?.state === "COMPLETE" ? "blue" : "warn"), " ", badge(p.importance || n.importance || "NORMAL", "dim"), " ", badge(p.state || detail.state || "", STATE_TONE[String(p.state).toLowerCase()] || "idle")),
    tasks.length ? el("div", { class: "bar green", style: { margin: "6px 0 10px" } }, el("i", { style: { width: `${(done / tasks.length) * 100}%` } })) : null,
    section("Goal", kv("goal", p.goal || detail.goal), kv("owner said", meta.owner_request), kv("parent", meta.parent_title), kv("deadline", meta.deadline)),
    section("Status", kv("progress", tasks.length ? `${done}/${tasks.length} tasks` : "no tasks"), kv("health", `${p.health?.state || "?"} — ${p.health?.reason || ""}`),
      el("div", { class: "kv" }, el("span", { class: "k", text: "importance" }), el("span", { class: "v" }, importance)),
      kv("last activity", ago(p.updated_at || detail.updated_at)), kv("created", (detail.created_at || "").slice(0, 10))),
    section("Now", kv("current mission", current ? current.title : (related.find((r) => r.kind === "mission" && r.state === "active") || {}).label || "—"),
      kv("blockers", blocked.length ? blocked.map((t) => t.title).join("; ") : (detail.blockers || []).map((b) => b.text).join("; ") || "none"),
      kv("next action", next ? next.title : "—"), kv("risks", p.health?.state === "AT_RISK" ? p.health.reason : "none known")),
    section("Connected", kv("missions", related.filter((r) => r.kind === "mission").map((r) => r.label).join("\n") || "—"),
      kv("subprojects", related.filter((r) => r.kind === "project").map((r) => r.label).join(", ") || "—"),
      kv("capabilities", related.filter((r) => r.kind === "capability").map((r) => r.label).join(", ") || "—"),
      kv("ZEUS thoughts", related.filter((r) => r.kind === "thought").map((r) => r.label).join("\n") || "—")),
    section("Knowledge", knowledgeNodes.length ? el("div", {}, ...knowledgeNodes.slice(0, 6).map((k) => el("div", { class: "focus-row", onClick: () => views.open("knowledge", { q: k.title }) }, el("span", { class: "dot", style: { background: KIND_COLOUR.knowledge } }), el("span", { text: k.title })))) : el("div", { class: "empty", style: { padding: "2px 0" }, text: "no related notes yet" })),
    (detail.artifacts || []).length ? section("Documents", ...detail.artifacts.slice(-6).map((a) => kv(a.kind, a.path))) : null,
    (detail.decisions || []).length ? section("Decisions", ...detail.decisions.slice(-5).map((d) => kv((d.at || "").slice(0, 10), d.text))) : null,
    (detail.acceptance || []).length ? section("Acceptance", ...detail.acceptance.slice(0, 5).map((a) => kv(a.satisfied ? "✓" : "·", a.text))) : null,
    (p.notes || []).length ? section("ZEUS notes", ...p.notes.map((x) => kv((x.at || "").slice(0, 10), x.title || x.text))) : null,
    section("Owner notes", note, el("div", { class: "toolbar" }, button("Add note", async () => { if (note.value.trim()) { await api("/api/project/update", { id: n.id, note: note.value }); note.value = ""; } }))),
    el("div", { class: "toolbar" },
      button("Open", () => views.open("projects", { id: n.id }), "primary"),
      button("Focus", () => galaxy?.focusText(n.label)),
      button(n.locked ? "Release position" : "Lock position", async () => { if (n.locked) await galaxy.release(n); else await galaxy.lock(n, true); reload(); }),
      button(p.importance === "PINNED" ? "Unpin" : "Pin", async () => { await api("/api/project/update", { id: n.id, importance: p.importance === "PINNED" ? "NORMAL" : "PINNED" }); reload(); }),
      button("Archive", async () => { await api("/api/project/update", { id: n.id, importance: "ARCHIVED" }); reload(); }, "ghost"),
      button(p.hidden ? "Unhide" : "Hide", async () => { await api("/api/project/update", { id: n.id, hidden: !p.hidden }); reload(); }, "ghost"),
      button("Create mission", () => chat.send(`Zeus, starte eine Mission für das Projekt „${p.title || n.label}“: ${next ? next.title : "nächster sinnvoller Schritt"}.`, "galaxy")),
      button("Ask ZEUS", () => chat.send(`Wie steht das Projekt „${p.title || n.label}“? Was blockiert es und was ist der nächste Schritt?`, "galaxy")),
      button("Local graph", () => views.open("projects", { connected: p.title || n.label }))));
}

/* ---- deep view: the project's own system -------------------------------- */
async function deep(pane, id) {
  // the deep page is a reading surface, not a spatial one
  $("app").classList.remove("spatial");
  const [detail, graph, timeline] = await Promise.all([api("/api/project", { id }), api("/api/projects/graph", { everything: true }), api("/api/project/timeline", { id })]);
  if (detail.error) { pane.append(el("div", { class: "empty", text: detail.error })); return; }
  const node = graph.nodes.find((n) => n.id === id) || { id, label: detail.title || detail.goal, data: {} };
  views.breadcrumb([{ label: "Projects", onClick: () => views.open("projects") }, { label: (node.label || id).slice(0, 40) }]);
  const tasks = detail.tasks || [];
  const done = tasks.filter((t) => ["done", "complete", "completed", "accepted"].includes(String(t.status))).length;
  const blocked = tasks.filter((t) => ["blocked", "failed"].includes(String(t.status)));
  const active = tasks.find((t) => ["active", "running", "working"].includes(String(t.status)));
  const health = node.health || { state: blocked.length ? "BLOCKED" : active ? "HEALTHY" : "DORMANT", reason: "" };
  pane.append(el("div", { class: "card" },
    el("div", { class: "title", text: node.label || detail.goal || "" }),
    el("div", { class: "meta" }, badge(health.state, health.state === "HEALTHY" ? "ok" : health.state === "COMPLETE" ? "blue" : health.state === "BLOCKED" ? "bad" : "warn"),
      badge(node.importance || "NORMAL", "dim"), badge(detail.state || "unknown", STATE_TONE[String(detail.state).toLowerCase()] || "idle"),
      el("span", { text: tasks.length ? `${done}/${tasks.length} tasks done` : "no tasks" }), el("span", { text: health.reason || "" })),
    tasks.length ? el("div", { class: "bar green", style: { marginTop: "8px" } }, el("i", { style: { width: `${(done / tasks.length) * 100}%` } })) : null));
  const wrap = el("div", { class: "galaxy-wrap" });
  const canvas = el("canvas", { id: "constellation", class: "galaxy", style: { height: "36vh", minHeight: "260px" } });
  wrap.append(canvas);
  pane.append(wrap);
  const local = { nodes: graph.nodes.filter((n) => n.id === id || n.id === "zeus" || graph.edges.some((e) => (e.source === id && e.target === n.id) || (e.target === id && e.source === n.id))), edges: graph.edges.filter((e) => e.source === id || e.target === id) };
  const centre = local.nodes.find((n) => n.id === id); if (centre) { centre.layout = { x: (canvas.clientWidth || 900) / 2, y: 150, state: "LOCKED" }; }
  galaxy = new Galaxy(canvas, wrap, local, { onSelect: (n) => inspect(n, graph, () => views.open("projects", { id })), mode: "GALAXY" });
  galaxy.cam.z = 1.6;
  const answer = (q, a) => el("div", { class: "kv" }, el("span", { class: "k", text: q }), el("span", { class: "v", text: a }));
  pane.append(section("At a glance",
    answer("What are we doing", detail.goal || "—"),
    answer("Where are we", tasks.length ? `${done} of ${tasks.length} tasks complete` : "no plan yet"),
    answer("Happening now", active ? active.title : ((detail.steps || []).at(-1)?.summary || "nothing running")),
    answer("Blocking us", blocked.length ? blocked.map((t) => t.title).join("; ") : "nothing"),
    answer("Next", (tasks.find((t) => ["todo", "pending", "open", "planned"].includes(String(t.status))) || {}).title || "—")));
  if ((detail.acceptance || []).length) pane.append(section("Acceptance", ...detail.acceptance.map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: a.satisfied ? "✓" : "·" }), el("span", { class: "v", text: a.text })))));
  if (tasks.length) pane.append(section("Tasks / dependencies", dependencyGraph(tasks)));
  if ((detail.decisions || []).length) pane.append(section("Decisions", ...detail.decisions.map((d) => el("div", { class: "kv" }, el("span", { class: "k", text: (d.at || "").slice(0, 10) }), el("span", { class: "v", text: d.text })))));
  if ((detail.artifacts || []).length) pane.append(section("Documents", ...detail.artifacts.map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: a.kind }), el("span", { class: "v", text: a.path })))));
  pane.append(section("Timeline (from Activity and the mission stores)", timelineView(timeline.events || [])));
  pane.append(el("div", { class: "toolbar" },
    button("Continue this project", () => chat.send(`Continue the project: ${detail.goal}`, "galaxy"), "primary"),
    button("Ask ZEUS about it", () => chat.send(`Wie steht das Projekt „${detail.title || detail.goal}“? Was blockiert es und was ist der nächste Schritt?`, "galaxy")),
    button("Open graph", () => views.open("knowledge", { q: detail.goal || "" }))));
}

function dependencyGraph(tasks) {
  const wrap = el("div", { class: "timeline" });
  for (const t of tasks) {
    const s = String(t.status);
    const cls = ["done", "complete", "completed", "accepted"].includes(s) ? "ok" : ["blocked", "failed"].includes(s) ? "bad" : ["active", "running", "working"].includes(s) ? "work" : "";
    wrap.append(el("div", { class: "tl " + cls }, el("span", { class: "when", text: s }), el("span", { class: "text", text: t.title }),
      t.attempts ? el("span", { class: "sub", text: `${t.attempts} attempts` }) : null, cls === "bad" ? el("span", { class: "sub", text: "cannot proceed: this task is unfinished" }) : null));
  }
  return wrap;
}

function timelineView(events) {
  const wrap = el("div", { class: "timeline" });
  const tone = (k) => k.includes("failed") ? "bad" : k.includes("promoted") || k.includes("completed") || k.includes("acquired") ? "ok" : k.includes("correction") ? "warn" : "work";
  for (const e of events.slice(-60).reverse()) wrap.append(el("div", { class: "tl " + tone(e.kind) }, el("span", { class: "when", text: (e.at || "").slice(0, 16).replace("T", " ") }), el("span", { class: "text", text: `${e.kind.replace(/_/g, " ")} — ${e.summary}` })));
  if (!events.length) wrap.append(el("div", { class: "empty", text: "No events yet." }));
  return wrap;
}
