/* Projects: the owner's project universe.

   A galaxy, not a card list: owner projects are stars (size from importance
   and work, halo from health), missions are small bodies orbiting the
   project they belong to (or ZEUS), capability families are collapsed
   satellites, Knowledge is a nebula, and ZEUS's own thoughts are pulses on
   the relations they concern. Depth, parallax and a settled layout that
   stops burning CPU once it has relaxed.

   Owner intent wins over the layout engine: a dragged node becomes
   OWNER_POSITIONED and is persisted (/api/project/update); LOCKED nodes never
   move; box-select and group move; layout modes (galaxy, dependency,
   timeline, hierarchy, mission flow) never touch owner-saved positions.
   Importance (PINNED FOCUS ACTIVE NORMAL LOW_PRIORITY DORMANT ARCHIVED)
   decides what dominates by default; "show everything" is explicit.

   The Focus panel answers "what deserves my attention?"; the inspector
   answers what a selected project is, where it stands and what is next. */

import { $, el, clear, kv, section, badge, button, ago, seconds } from "../core/dom.js";
import { api } from "../core/api.js";
import { state, setPref } from "../core/state.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const IMPORTANCE = ["PINNED", "FOCUS", "ACTIVE", "NORMAL", "LOW_PRIORITY", "DORMANT", "ARCHIVED"];
const IMPORTANCE_WEIGHT = { PINNED: 1.0, FOCUS: 0.95, ACTIVE: 0.8, NORMAL: 0.6, LOW_PRIORITY: 0.4, DORMANT: 0.3, ARCHIVED: 0.2 };
const HEALTH_COLOUR = { HEALTHY: "#66d9a0", AT_RISK: "#e0b04a", BLOCKED: "#ff6b6b", DORMANT: "#5b6b82", COMPLETE: "#4fc3f7" };
const KIND_COLOUR = { self: "#8fd3ff", mission: "#9fb7d9", capability: "#7a8aa8", knowledge: "#7f6fd9", thought: "#f0c674" };
const MODES = ["GALAXY", "DEPENDENCY", "TIMELINE", "HIERARCHY", "KNOWLEDGE", "MISSION FLOW"];
const STATE_TONE = { active: "active", running: "active", working: "active", executing: "active", blocked: "blocked", failed: "blocked",
                     accepted: "done", complete: "done", completed: "done", paused: "idle", draft: "idle" };

let galaxy = null;

export const view = {
  id: "projects",
  title: "Projects",
  async mount(pane, params) {
    if (params.id) return deep(pane, params.id);
    const everything = params.everything === "1" || params.everything === true;
    const [graph, overview] = await Promise.all([api("/api/projects/graph", { everything }), api("/api/projects/overview")]);
    const canvas = el("canvas", { id: "constellation", class: "galaxy" });
    const search = el("input", { placeholder: "Focus… (project, mission, capability)", value: params.focus || params.q || "", style: { maxWidth: "260px" } });
    const mode = el("select", {}, ...MODES.map((m) => el("option", { value: m, text: m.toLowerCase(), selected: (params.mode || state.ui.galaxyMode || "GALAXY") === m })));
    const everyToggle = el("label", { class: "empty", style: { padding: 0, cursor: "pointer" } }, el("input", { type: "checkbox", checked: everything }), " show everything");
    const counts = el("span", { class: "empty", style: { padding: 0 } });
    const toolbar = el("div", { class: "toolbar" }, search, mode, everyToggle, counts);
    pane.append(toolbar, canvas);
    const focus = focusPanel(overview, graph, params);
    pane.append(focus);
    galaxy = new Galaxy(canvas, graph, { onSelect: (n) => inspect(n, graph, () => views.open("projects", params)), mode: mode.value, filter: params.filter || "", uses: params.uses || "", idleDays: Number(params.idle_days || 0), connected: params.connected || "" });
    counts.textContent = `${graph.nodes.filter((n) => n.kind === "project").length} projects · ${graph.nodes.filter((n) => n.kind === "mission").length} missions · ${graph.nodes.filter((n) => n.kind === "capability").length} capability families · ${graph.nodes.filter((n) => n.kind === "thought").length} thoughts` + (graph.hidden ? ` · ${graph.hidden} hidden/archived` : "");
    search.oninput = () => galaxy.focusText(search.value);
    if (search.value) galaxy.focusText(search.value);
    mode.onchange = () => { setPref("galaxyMode", mode.value); galaxy.setMode(mode.value); };
    everyToggle.firstChild.onchange = (e) => views.open("projects", { ...params, everything: e.target.checked ? "1" : "" });
    if (params.focus) galaxy.focusText(params.focus, true);
  },
  unmount() { galaxy?.destroy(); galaxy = null; },
};

/* ---- the Focus panel: what deserves attention ------------------------- */
function focusPanel(overview, graph, params) {
  const projects = (overview.projects || []).filter((p) => !p.hidden);
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

/* ---- the galaxy ------------------------------------------------------- */
class Galaxy {
  constructor(canvas, graph, opts) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d"); this.opts = opts;
    this.nodes = []; this.edges = []; this.byId = new Map();
    this.cam = { x: 0, y: 0, z: 1, vx: 0, vy: 0 };
    this.selected = new Set(); this.hover = null; this.drag = null; this.box = null; this.focusId = null; this.dim = null;
    this.frames = 0; this.settled = false; this.raf = 0; this.mode = opts.mode || "GALAXY"; this.t = 0;
    this.stars = Array.from({ length: 220 }, (_, i) => ({ x: Math.random(), y: Math.random(), z: 0.2 + Math.random() * 0.8, s: Math.random() < 0.15 ? 1.6 : 0.9 }));
    this.build(graph);
    this.applyFilters();
    this.resize(); this.bind();
    this.layout(true);
    this.start();
  }

  build(graph) {
    const W = this.W(), H = this.H();
    for (const raw of graph.nodes) {
      const n = { ...raw, x: W / 2, y: H / 2, vx: 0, vy: 0, r: 6, depth: 1, fixed: false, locked: false, ownerPlaced: false, visible: true };
      if (n.kind === "project") {
        const w = IMPORTANCE_WEIGHT[n.importance] || 0.6;
        n.r = 10 + w * 18 + Math.min(10, (n.tasks || 0) * 0.6); n.depth = 1;
        const layout = n.layout || {};
        if (layout.state === "OWNER_POSITIONED" || layout.state === "LOCKED") { n.x = layout.x; n.y = layout.y; n.ownerPlaced = true; n.locked = layout.state === "LOCKED"; }
      } else if (n.kind === "self") { n.r = 22; n.depth = 1; n.x = W / 2; n.y = H / 2; n.fixed = true; }
      else if (n.kind === "mission") { n.r = 4 + (n.state === "active" ? 1.5 : 0); n.depth = 0.85; }
      else if (n.kind === "capability") { n.r = 6 + Math.min(6, (n.attempts || 1) * 0.6); n.depth = 0.75; }
      else if (n.kind === "knowledge") { n.r = 30; n.depth = 0.6; }
      else if (n.kind === "thought") { n.r = 4; n.depth = 0.95; }
      this.nodes.push(n); this.byId.set(n.id, n);
    }
    this.edges = graph.edges.filter((e) => this.byId.has(e.source) && this.byId.has(e.target)).map((e) => ({ ...e, a: this.byId.get(e.source), b: this.byId.get(e.target) }));
    for (const e of this.edges) if (e.type === "mission_of" || e.type === "thought" || e.type === "uses") { e.b.parent = e.b.parent || e.a; }
  }

  applyFilters() {
    const { filter, uses, idleDays, connected } = this.opts;
    for (const n of this.nodes) n.visible = true;
    if (filter === "blocked") for (const n of this.nodes) if (n.kind === "project" && n.health?.state !== "BLOCKED") n.visible = n.kind !== "project";
    if (idleDays) for (const n of this.nodes) if (n.kind === "project" && Date.now() - new Date(n.updated_at || 0) < idleDays * 864e5) n.visible = false;
    if (uses) for (const n of this.nodes) if (n.kind === "project" && !this.edges.some((e) => e.a === n && e.b.kind === "capability" && e.b.label.includes(uses))) n.visible = false;
    if (connected) {
      const anchor = this.find(connected);
      if (anchor) { const keep = new Set([anchor.id]); for (const e of this.edges) { if (e.a === anchor) keep.add(e.b.id); if (e.b === anchor) keep.add(e.a.id); } for (const n of this.nodes) n.visible = keep.has(n.id) || n.kind === "self"; }
    }
  }

  W() { return this.canvas.clientWidth || 900; }
  H() { return this.canvas.clientHeight || 420; }
  resize() { this.canvas.width = this.W() * devicePixelRatio; this.canvas.height = this.H() * devicePixelRatio; this.settled = false; this.frames = 0; }

  /* ---- layout modes (never move owner-placed nodes) ------------------- */
  layout(initial) {
    const W = this.W(), H = this.H();
    const projects = this.nodes.filter((n) => n.kind === "project" && n.visible);
    const zeus = this.byId.get("zeus");
    const place = (n, x, y) => { if (!n.ownerPlaced && !n.locked) { if (initial || this.mode !== "GALAXY") { n.x = x; n.y = y; } else { n.tx = x; n.ty = y; } } };
    if (this.mode === "GALAXY") {
      projects.sort((a, b) => (IMPORTANCE_WEIGHT[b.importance] || 0) - (IMPORTANCE_WEIGHT[a.importance] || 0));
      projects.forEach((n, i) => { const w = IMPORTANCE_WEIGHT[n.importance] || 0.6; const ring = 120 + (1 - w) * 240; const a = (i / Math.max(1, projects.length)) * Math.PI * 2 - Math.PI / 2; place(n, W / 2 + Math.cos(a) * ring * (W / 900), H / 2 + Math.sin(a) * ring * 0.62 * (H / 420)); });
    } else if (this.mode === "DEPENDENCY") {
      // capabilities in a bottom band, projects above, missions between
      const caps = this.nodes.filter((n) => n.kind === "capability" && n.visible);
      caps.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, caps.length), H * 0.85));
      projects.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, projects.length), H * 0.28));
    } else if (this.mode === "TIMELINE") {
      const dated = projects.filter((n) => n.updated_at).sort((a, b) => new Date(a.data?.created_at || a.updated_at) - new Date(b.data?.created_at || b.updated_at));
      dated.forEach((n, i) => place(n, 70 + (i + 0.5) * (W - 140) / Math.max(1, dated.length), H * (0.35 + 0.3 * ((i % 2) ? 1 : 0))));
    } else if (this.mode === "HIERARCHY") {
      projects.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, projects.length), H * 0.3));
      if (zeus) zeus.x = W / 2, zeus.y = H * 0.08;
    } else if (this.mode === "MISSION FLOW") {
      const ms = this.nodes.filter((n) => n.kind === "mission" && n.visible).sort((a, b) => new Date(a.updated_at || 0) - new Date(b.updated_at || 0));
      ms.forEach((n, i) => { n.tx = 60 + (i + 0.5) * (W - 120) / Math.max(1, ms.length); n.ty = H * 0.55; n.flow = true; });
      projects.forEach((n, i) => place(n, 80 + (i + 0.5) * (W - 160) / Math.max(1, projects.length), H * 0.2));
    } else if (this.mode === "KNOWLEDGE") {
      const k = this.byId.get("knowledge"); if (k) { k.x = W / 2; k.y = H / 2; k.r = 60; }
      projects.forEach((n, i) => { const a = (i / Math.max(1, projects.length)) * Math.PI * 2; place(n, W / 2 + Math.cos(a) * W * 0.32, H / 2 + Math.sin(a) * H * 0.34); });
    }
    if (zeus && this.mode === "GALAXY") { zeus.x = W / 2; zeus.y = H / 2; }
    // infrastructure without a project: capability families on an outer ring, spaced apart
    const loose = this.nodes.filter((n) => n.kind === "capability" && n.visible && !n.parent);
    loose.forEach((n, i) => { const a = Math.PI * 0.15 + (i / Math.max(1, loose.length)) * Math.PI * 1.7; const rx = W * 0.44, ry = H * 0.42; if (!n.placedOnce) { n.x = W / 2 + Math.cos(a) * rx; n.y = H / 2 + Math.sin(a) * ry; n.placedOnce = true; } });
    // children around their parents
    for (const n of this.nodes) {
      if ((n.kind === "mission" || n.kind === "thought" || n.kind === "capability") && n.parent && !n.flow && (initial || !n.placedOnce)) {
        const a = Math.random() * Math.PI * 2, d = n.parent.r + 26 + Math.random() * 30;
        n.x = n.parent.x + Math.cos(a) * d; n.y = n.parent.y + Math.sin(a) * d; n.orbit = { a, d, speed: (0.15 + Math.random() * 0.2) * (n.kind === "mission" && n.state === "active" ? 1.6 : 1) };
        n.placedOnce = true;
      }
      if (n.kind === "knowledge" && this.mode !== "KNOWLEDGE") { n.x = W * 0.86; n.y = H * 0.2; }
    }
    this.settled = false; this.frames = 0;
  }

  setMode(mode) { this.mode = mode; this.layout(false); this.start(); }

  /* ---- physics: a short relaxation, then rest ------------------------- */
  step() {
    const W = this.W(), H = this.H();
    const movers = this.nodes.filter((n) => n.visible && !n.fixed && !n.locked && n.kind !== "mission" && n.kind !== "thought");
    for (const a of movers) {
      if (a.ownerPlaced) continue;
      for (const b of movers) {
        if (a === b) continue;
        const dx = a.x - b.x, dy = a.y - b.y, d = Math.max(12, Math.hypot(dx, dy)), min = a.r + b.r + 34;
        if (d < min) { const f = (min - d) / d * 0.04; a.vx += dx * f; a.vy += dy * f; }
      }
      if (a.tx !== undefined) { a.vx += (a.tx - a.x) * 0.02; a.vy += (a.ty - a.y) * 0.02; }
      a.vx *= 0.8; a.vy *= 0.8; a.x += a.vx; a.y += a.vy;
      a.x = Math.max(a.r + 6, Math.min(W - a.r - 6, a.x)); a.y = Math.max(a.r + 6, Math.min(H - a.r - 18, a.y));
    }
    // orbiting bodies follow their parent; they never push it
    for (const n of this.nodes) {
      if (n.orbit && n.parent && !n.flow) {
        n.orbit.a += n.orbit.speed * 0.004 * (state.ui.reducedMotion ? 0 : 1);
        n.x = n.parent.x + Math.cos(n.orbit.a) * n.orbit.d; n.y = n.parent.y + Math.sin(n.orbit.a) * n.orbit.d;
      } else if (n.flow && n.tx !== undefined) { n.x += (n.tx - n.x) * 0.1; n.y += (n.ty - n.y) * 0.1; }
    }
    // camera inertia
    this.cam.x += this.cam.vx; this.cam.y += this.cam.vy; this.cam.vx *= 0.9; this.cam.vy *= 0.9;
    this.frames += 1;
    const moving = movers.some((n) => Math.hypot(n.vx, n.vy) > 0.05) || Math.hypot(this.cam.vx, this.cam.vy) > 0.05;
    if (this.frames > 60 && !moving) this.settled = true;
  }

  start() { if (this.raf) return; const tick = () => { if (!this.settled || this.hasMotion()) this.step(); this.draw(); this.t += 1; this.raf = requestAnimationFrame(tick); }; this.raf = requestAnimationFrame(tick); }
  hasMotion() { return !state.ui.reducedMotion && this.nodes.some((n) => (n.kind === "mission" && n.state === "active") || n.kind === "thought"); }
  destroy() { cancelAnimationFrame(this.raf); this.raf = 0; }

  /* ---- drawing -------------------------------------------------------- */
  toScreen(n) { return [(n.x - this.cam.x) * this.cam.z + this.W() / 2 * (1 - this.cam.z), (n.y - this.cam.y) * this.cam.z + this.H() / 2 * (1 - this.cam.z)]; }
  fromScreen(sx, sy) { return [(sx - this.W() / 2 * (1 - this.cam.z)) / this.cam.z + this.cam.x, (sy - this.H() / 2 * (1 - this.cam.z)) / this.cam.z + this.cam.y]; }

  draw() {
    const ctx = this.ctx, W = this.W(), H = this.H(), z = this.cam.z;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // parallax star field + nebula wash
    for (const s of this.stars) {
      const x = ((s.x * W - this.cam.x * 0.05 * s.z) % W + W) % W, y = ((s.y * H - this.cam.y * 0.05 * s.z) % H + H) % H;
      ctx.fillStyle = `rgba(160,190,230,${0.12 + s.z * 0.25})`; ctx.fillRect(x, y, s.s, s.s);
    }
    const k = this.byId.get("knowledge");
    if (k && k.visible) { const [kx, ky] = this.toScreen(k); const g = ctx.createRadialGradient(kx, ky, 0, kx, ky, k.r * 2.6 * z); g.addColorStop(0, "rgba(127,111,217,.20)"); g.addColorStop(1, "transparent"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(kx, ky, k.r * 2.6 * z, 0, Math.PI * 2); ctx.fill(); }
    // edges: energy paths for active relations, faint otherwise
    for (const e of this.edges) {
      if (!e.a.visible || !e.b.visible) continue;
      const [ax, ay] = this.toScreen(e.a), [bx, by] = this.toScreen(e.b);
      const dimmed = this.dim && !this.dim.has(e.a.id) && !this.dim.has(e.b.id);
      ctx.strokeStyle = e.type === "thought" ? `rgba(240,198,116,${dimmed ? .06 : .35})` : e.active ? `rgba(102,217,160,${dimmed ? .05 : .35})` : `rgba(79,195,247,${dimmed ? .04 : .14})`;
      ctx.lineWidth = e.active ? 1.4 : 1; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      if (e.active && !state.ui.reducedMotion) { const p = (this.t / 90) % 1; ctx.fillStyle = "rgba(102,217,160,.8)"; ctx.beginPath(); ctx.arc(ax + (bx - ax) * p, ay + (by - ay) * p, 1.8, 0, Math.PI * 2); ctx.fill(); }
      if (e.type === "thought" && !state.ui.reducedMotion) { const p = (this.t / 140) % 1; ctx.fillStyle = "rgba(240,198,116,.9)"; ctx.beginPath(); ctx.arc(ax + (bx - ax) * p, ay + (by - ay) * p, 2, 0, Math.PI * 2); ctx.fill(); }
    }
    // bodies, far to near
    const labels = [];
    for (const n of [...this.nodes].sort((a, b) => a.depth - b.depth)) {
      if (!n.visible) continue;
      const [x, y] = this.toScreen(n); const r = n.r * z * (0.6 + 0.4 * n.depth);
      const dimmed = this.dim && !this.dim.has(n.id);
      ctx.globalAlpha = dimmed ? 0.18 : 1;
      let colour = n.kind === "project" ? (HEALTH_COLOUR[n.health?.state] || "#9db0c8") : (KIND_COLOUR[n.kind] || "#6b7c93");
      if (n.kind === "project" && ["DORMANT", "ARCHIVED", "LOW_PRIORITY"].includes(n.importance)) colour = "#5b6b82";
      const halo = n.kind === "project" ? (n.importance === "ACTIVE" || n.importance === "FOCUS" || n.importance === "PINNED" ? 2.6 : 1.8) : n.kind === "self" ? 3 : 1.6;
      const pulse = (n.kind === "project" && n.importance === "ACTIVE" && !state.ui.reducedMotion) ? 1 + Math.sin(this.t / 25) * 0.08 : 1;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r * halo * pulse); g.addColorStop(0, colour + "66"); g.addColorStop(1, "transparent");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * halo * pulse, 0, Math.PI * 2); ctx.fill();
      if (n.kind === "project" && n.health?.state === "BLOCKED") { ctx.strokeStyle = "rgba(255,107,107,.55)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r * 1.35, 0, Math.PI * 2); ctx.stroke(); }
      ctx.fillStyle = this.selected.has(n.id) || n === this.hover ? "#ffffff" : colour;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      if (n.kind === "project" && n.locked) { ctx.strokeStyle = "#dce5f0"; ctx.lineWidth = 1; ctx.strokeRect(x - 3, y - r - 9, 6, 5); }
      if (n.kind === "project" && n.importance === "PINNED") { ctx.fillStyle = "#e0b04a"; ctx.beginPath(); ctx.arc(x + r * 0.8, y - r * 0.8, 2.5, 0, Math.PI * 2); ctx.fill(); }
      ctx.globalAlpha = 1;
      // semantic zoom: far = projects only; medium = + capabilities; near = missions/thoughts too
      const lod = z < 0.8 ? 0 : z < 1.4 ? 1 : 2;
      const wanted = n.kind === "project" || n.kind === "self" || n.kind === "knowledge" || (lod >= 1 && n.kind === "capability") || (lod >= 2) || n === this.hover || this.selected.has(n.id);
      if (wanted && !dimmed) labels.push({ n, x, y: y + r + 4, size: n.kind === "self" ? 12 : n.kind === "project" ? 11 : 10, text: n.kind === "capability" ? `${n.label} · ${n.attempts} attempts` : n.label });
    }
    // labels without collisions (priority: self, projects, then the rest)
    ctx.textAlign = "center"; const placed = [];
    for (const l of labels.sort((a, b) => b.size - a.size)) {
      ctx.font = `${l.size}px Segoe UI, sans-serif`; const w = ctx.measureText(l.text).width + 8, h = l.size + 4;
      const box = { x: l.x - w / 2, y: l.y, w, h };
      if (placed.some((b) => !(box.x + box.w < b.x || b.x + b.w < box.x || box.y + box.h < b.y || b.y + b.h < box.y))) { if (l.n.kind !== "project" && l.n.kind !== "self") continue; }
      placed.push(box); ctx.fillStyle = l.n === this.hover ? "#ffffff" : l.n.kind === "project" ? "#c8d6ea" : "#8a9bb3"; ctx.fillText(l.text, l.x, l.y + l.size);
    }
    if (this.box) { ctx.strokeStyle = "rgba(143,211,255,.7)"; ctx.setLineDash([4, 3]); ctx.strokeRect(this.box.x, this.box.y, this.box.w, this.box.h); ctx.setLineDash([]); }
  }

  /* ---- interaction ---------------------------------------------------- */
  hit(sx, sy) { const [x, y] = this.fromScreen(sx, sy); let best = null; for (const n of this.nodes) { if (!n.visible) continue; const d = Math.hypot(n.x - x, n.y - y); if (d < n.r + 6 / this.cam.z && (!best || n.depth > best.depth)) best = n; } return best; }
  find(text) { const q = String(text || "").toLowerCase(); return this.nodes.find((n) => n.visible && n.label.toLowerCase().includes(q)) || null; }
  focusText(text, open = false) {
    const n = this.find(text);
    if (!text) { this.dim = null; this.focusId = null; return; }
    if (!n) return;
    const keep = new Set([n.id]); for (const e of this.edges) { if (e.a === n) keep.add(e.b.id); if (e.b === n) keep.add(e.a.id); }
    this.dim = keep; this.focusId = n.id; this.flyTo(n, 1.6);
    if (open) this.opts.onSelect(n);
  }
  flyTo(n, z) { const steps = state.ui.reducedMotion ? 1 : 24; const sx = this.cam.x, sy = this.cam.y, sz = this.cam.z; let i = 0; const ease = (t) => 1 - Math.pow(1 - t, 3); const go = () => { i += 1; const t = ease(i / steps); this.cam.x = sx + (n.x - sx) * t; this.cam.y = sy + (n.y - sy) * t; this.cam.z = sz + (z - sz) * t; if (i < steps) requestAnimationFrame(go); }; go(); this.settled = false; }
  bind() {
    const c = this.canvas;
    c.onmousemove = (e) => {
      const r = c.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top;
      if (this.drag) {
        const [x, y] = this.fromScreen(sx, sy);
        if (this.drag.kind === "pan") { this.cam.x = this.drag.camX - (sx - this.drag.sx) / this.cam.z; this.cam.y = this.drag.camY - (sy - this.drag.sy) / this.cam.z; this.cam.vx = -(e.movementX) / this.cam.z * 0.3; this.cam.vy = -(e.movementY) / this.cam.z * 0.3; }
        else if (this.drag.kind === "node") { for (const n of this.drag.nodes) { if (n.locked) continue; n.x = x + n.dx; n.y = y + n.dy; n.vx = n.vy = 0; } this.drag.moved = true; }
        else if (this.drag.kind === "box") { this.box = { x: Math.min(this.drag.sx, sx), y: Math.min(this.drag.sy, sy), w: Math.abs(sx - this.drag.sx), h: Math.abs(sy - this.drag.sy) }; }
        this.settled = false; return;
      }
      this.hover = this.hit(sx, sy); c.style.cursor = this.hover ? "pointer" : "grab"; this.settled = false;
    };
    c.onmousedown = (e) => {
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
        if (d.moved) { for (const n of d.nodes) { if (n.locked) continue; n.ownerPlaced = true; n.layout = { ...(n.layout || {}), x: n.x, y: n.y, state: "OWNER_POSITIONED" }; await api("/api/project/update", { id: n.id, layout: { x: n.x, y: n.y, state: "OWNER_POSITIONED" } }); } }
        else if (d.nodes.length === 1) this.opts.onSelect(d.nodes[0]);
        return;
      }
      if (d.kind === "pan") { const r = c.getBoundingClientRect(); const n = this.hit(e.clientX - r.left, e.clientY - r.top); if (n && Math.abs(e.clientX - r.left - d.sx) < 3 && Math.abs(e.clientY - r.top - d.sy) < 3) this.opts.onSelect(n); }
    };
    c.onmouseup = finish; c.onmouseleave = () => { if (this.drag?.kind === "pan") this.drag = null; this.hover = null; };
    c.ondblclick = (e) => { const r = c.getBoundingClientRect(); const n = this.hit(e.clientX - r.left, e.clientY - r.top); if (n?.kind === "project") views.open("projects", { id: n.id }); else if (n) this.focusText(n.label); };
    c.onwheel = (e) => { e.preventDefault(); const r = c.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top; const [wx, wy] = this.fromScreen(sx, sy); const z = Math.max(0.45, Math.min(3.2, this.cam.z * (e.deltaY < 0 ? 1.12 : 0.89))); this.cam.z = z; const [nx, ny] = this.fromScreen(sx, sy); this.cam.x += wx - nx; this.cam.y += wy - ny; this.settled = false; };
    this._onKey = (e) => { if (e.key === "Escape" && (this.dim || this.cam.z !== 1)) { this.dim = null; this.focusId = null; this.flyTo({ x: this.W() / 2, y: this.H() / 2 }, 1); e.stopPropagation(); } };
    window.addEventListener("keydown", this._onKey, true);
    this._onResize = () => this.resize(); window.addEventListener("resize", this._onResize);
  }
  lock(n, locked) { n.locked = locked; n.layout = { ...(n.layout || {}), x: n.x, y: n.y, state: locked ? "LOCKED" : "OWNER_POSITIONED" }; return api("/api/project/update", { id: n.id, layout: n.layout }); }
  release(n) { n.ownerPlaced = false; n.locked = false; n.layout = { state: "AUTO_POSITIONED" }; this.layout(false); return api("/api/project/update", { id: n.id, layout: { state: "AUTO_POSITIONED" } }); }
}

/* ---- inspector -------------------------------------------------------- */
async function inspect(n, graph, reload) {
  if (n.kind === "mission") return views.open("missions", { mission: n.id });
  if (n.kind === "capability") return views.open("capabilities", { id: n.label });
  if (n.kind === "thought") return views.open("thoughts");
  if (n.kind === "knowledge") return views.open("knowledge");
  if (n.kind === "self") return views.open("missions");
  const p = n.data || {};
  const detail = await api("/api/project", { id: n.id });
  const tasks = detail.tasks || [];
  const current = tasks.find((t) => ["active", "running", "working"].includes(String(t.status)));
  const blocked = tasks.filter((t) => ["blocked", "failed"].includes(String(t.status)));
  const next = tasks.find((t) => ["todo", "pending", "open", "planned"].includes(String(t.status)));
  const related = graph.edges.filter((e) => e.source === n.id || e.target === n.id).map((e) => graph.nodes.find((x) => x.id === (e.source === n.id ? e.target : e.source))).filter(Boolean);
  const importance = el("select", {}, ...IMPORTANCE.map((i) => el("option", { value: i, text: i.toLowerCase(), selected: p.importance === i })));
  importance.onchange = async () => { await api("/api/project/update", { id: n.id, importance: importance.value }); reload(); };
  const note = el("input", { placeholder: "owner note…" });
  views.inspect(p.title || n.label,
    section("Status", kv("health", `${p.health?.state || "?"} — ${p.health?.reason || ""}`), kv("state", p.state), kv("importance", importance),
      kv("goal", p.goal), kv("last activity", ago(p.updated_at)), kv("progress", p.tasks ? `${p.tasks_done}/${p.tasks} tasks` : "no tasks")),
    section("Now", kv("current mission", current ? current.title : "—"), kv("blockers", blocked.length ? blocked.map((t) => t.title).join("; ") : "none"),
      kv("next action", next ? next.title : "—"), kv("risks", p.health?.state === "AT_RISK" ? p.health.reason : "none known")),
    section("Connected", kv("missions", related.filter((r) => r.kind === "mission").map((r) => r.label).join("\n") || "—"),
      kv("capabilities", related.filter((r) => r.kind === "capability").map((r) => r.label).join(", ") || "—"),
      kv("ZEUS thoughts", related.filter((r) => r.kind === "thought").map((r) => r.label).join("\n") || "—"),
      kv("knowledge", "open the graph for related notes")),
    (p.notes || []).length ? section("ZEUS notes", ...p.notes.map((x) => kv((x.at || "").slice(0, 10), x.title || x.text))) : null,
    (detail.acceptance || []).length ? section("Recent decisions / acceptance", ...detail.acceptance.slice(0, 5).map((a) => kv(a.satisfied ? "✓" : "·", a.text))) : null,
    section("Owner notes", note, el("div", { class: "toolbar" }, button("Add note", async () => { if (note.value.trim()) { await api("/api/project/update", { id: n.id, note: note.value }); note.value = ""; } }))),
    el("div", { class: "toolbar" },
      button("Open", () => views.open("projects", { id: n.id }), "primary"),
      button("Focus", () => galaxy?.focusText(n.label)),
      button(n.locked ? "Release position" : "Lock position", async () => { if (n.locked) await galaxy.release(n); else await galaxy.lock(n, true); reload(); }),
      button(p.importance === "PINNED" ? "Unpin" : "Pin", async () => { await api("/api/project/update", { id: n.id, importance: p.importance === "PINNED" ? "NORMAL" : "PINNED" }); reload(); }),
      button("Archive", async () => { await api("/api/project/update", { id: n.id, importance: "ARCHIVED" }); reload(); }, "ghost"),
      button(p.hidden ? "Unhide" : "Hide", async () => { await api("/api/project/update", { id: n.id, hidden: !p.hidden }); reload(); }, "ghost"),
      button("Create mission", () => chat.send(`Zeus, starte eine Mission für das Projekt „${p.title || n.label}“: ${next ? next.title : "nächster sinnvoller Schritt"}.`)),
      button("Ask ZEUS", () => chat.send(`What is the state of the project "${p.title || n.label}"? What blocks it and what is next?`)),
      button("Local graph", () => views.open("projects", { connected: p.title || n.label }))));
}

/* ---- deep view: the project's own system -------------------------------- */
async function deep(pane, id) {
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
  // the local system: this project at the centre, its related bodies around
  const canvas = el("canvas", { id: "constellation", class: "galaxy", style: { height: "34vh" } });
  pane.append(canvas);
  const local = { nodes: graph.nodes.filter((n) => n.id === id || n.id === "zeus" || graph.edges.some((e) => (e.source === id && e.target === n.id) || (e.target === id && e.source === n.id))), edges: graph.edges.filter((e) => e.source === id || e.target === id) };
  const centre = local.nodes.find((n) => n.id === id); if (centre) { centre.layout = { x: canvas.clientWidth / 2 || 450, y: 170, state: "LOCKED" }; }
  galaxy = new Galaxy(canvas, local, { onSelect: (n) => inspect(n, graph, () => views.open("projects", { id })), mode: "GALAXY" });
  const answer = (q, a) => el("div", { class: "kv" }, el("span", { class: "k", text: q }), el("span", { class: "v", text: a }));
  pane.append(section("At a glance",
    answer("What are we doing", detail.goal || "—"),
    answer("Where are we", tasks.length ? `${done} of ${tasks.length} tasks complete` : "no plan yet"),
    answer("Happening now", active ? active.title : ((detail.steps || []).at(-1)?.summary || "nothing running")),
    answer("Blocking us", blocked.length ? blocked.map((t) => t.title).join("; ") : "nothing"),
    answer("Next", (tasks.find((t) => ["todo", "pending", "open", "planned"].includes(String(t.status))) || {}).title || "—")));
  if ((detail.acceptance || []).length) pane.append(section("Acceptance", ...detail.acceptance.map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: a.satisfied ? "✓" : "·" }), el("span", { class: "v", text: a.text })))));
  if (tasks.length) pane.append(section("Tasks / dependencies", dependencyGraph(tasks)));
  pane.append(section("Timeline (from Activity and the mission stores)", timelineView(timeline.events || [])));
  pane.append(el("div", { class: "toolbar" },
    button("Continue this project", () => chat.send(`Continue the project: ${detail.goal}`), "primary"),
    button("Ask ZEUS about it", () => chat.send(`What is the state of the project "${detail.goal}"? What blocks it and what is next?`)),
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
