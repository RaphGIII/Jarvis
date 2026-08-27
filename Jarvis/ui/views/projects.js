/* Projects: the OWNER's projects are the first-class objects. Internal work
   — capability acquisition attempts, repair attempts, self-development
   missions — is not a project; it appears grouped under the capability it
   belongs to ("System work"), one row per family with an attempt count, and
   feeds the constellation only as secondary, collapsed nodes around the
   owner projects they relate to. Nothing is deleted or hidden: legacy
   records the classifier cannot place are listed as "unclassified". */

import { $, el, clear, kv, section, badge, button, ago, seconds } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const STATE_TONE = { active: "active", running: "active", working: "active", executing: "active", blocked: "blocked", failed: "blocked",
                     accepted: "done", complete: "done", completed: "done", paused: "idle", draft: "idle" };

let anim = 0;

export const view = {
  id: "projects",
  title: "Projects",
  async mount(pane, params) {
    if (params.id) return deep(pane, params.id);
    const data = await api("/api/projects/overview");
    const projects = data.projects || [];
    const internal = data.internal || [];
    const missions = data.missions || [];
    const canvas = el("canvas", { id: "constellation" });
    const focus = el("input", { placeholder: "Focus a project…", style: { maxWidth: "260px" } });
    const detail = el("label", { class: "empty", style: { padding: 0, cursor: "pointer" } }, el("input", { type: "checkbox" }), " show internal work in the constellation");
    pane.append(el("div", { class: "toolbar" }, focus, detail,
      el("span", { class: "empty", style: { padding: 0 }, text: `${projects.length} owner project(s) · ${data.counts?.families || 0} internal capability job(s) (${data.counts?.internal_attempts || 0} attempts) · ${missions.length} missions` })), canvas);
    const grid = el("div", { class: "grid" });
    if (!projects.length) grid.append(el("div", { class: "empty", text: "No owner projects yet. Describe something to build and ZEUS opens one." }));
    for (const p of projects) grid.append(card(p));
    pane.append(section("Owner projects", grid));
    if (internal.length) {
      const rows = el("div");
      for (const fam of internal) rows.append(familyRow(fam));
      pane.append(section("System work (capability acquisition — not owner projects)", rows));
    }
    if ((data.unclassified || []).length) {
      const rows = el("div", { class: "grid" });
      for (const p of data.unclassified) rows.append(card(p));
      pane.append(section("Unclassified legacy records", rows));
    }
    let showInternal = false;
    detail.firstChild.onchange = (e) => { showInternal = e.target.checked; cancelAnimationFrame(anim); constellation(canvas, projects, missions, showInternal ? internal : [], focus.value); };
    focus.oninput = () => { cancelAnimationFrame(anim); constellation(canvas, projects, missions, showInternal ? internal : [], focus.value); };
    constellation(canvas, projects, missions, [], "");
  },
  unmount() { cancelAnimationFrame(anim); },
};

function card(p) {
  const node = el("div", { class: "card click" },
    el("div", { class: "title", text: p.title || p.goal || "(no goal recorded)" }),
    el("div", { class: "meta" }, badge(p.state || "unknown", STATE_TONE[String(p.state).toLowerCase()] || "idle"),
      p.origin === "unclassified" ? badge("unclassified", "warn") : null,
      el("span", { text: `${p.tasks ?? 0} tasks` }), el("span", { text: `${p.steps ?? 0} steps` }), el("span", { text: ago(p.updated_at) })));
  node.onclick = () => views.open("projects", { id: p.id });
  return node;
}

function familyRow(fam) {
  const latest = fam.attempts[fam.attempts.length - 1] || {};
  const node = el("div", { class: "card" },
    el("div", { class: "title" }, fam.capability_id, " ", badge(`${fam.count} attempt${fam.count === 1 ? "" : "s"}`, "blue"), " ", badge(fam.latest_state || "?", STATE_TONE[String(fam.latest_state).toLowerCase()] || "idle")),
    el("div", { class: "meta" }, ...fam.attempts.map((a, i) => { const b = el("button", { class: "ghost", style: { padding: "0 6px", fontSize: "10px" }, text: `#${i + 1} ${a.state}` }); b.onclick = () => views.open("projects", { id: a.id }); return b; }),
      el("span", { text: ago(fam.updated_at) }), el("span", { text: `${latest.tasks ?? 0} tasks in the latest` })),
    el("div", { class: "toolbar" }, button("Capability", () => views.open("capabilities", { id: fam.capability_id })), button("Missions", () => views.open("missions", { filter: "all" }))));
  return node;
}

/* Constellation: owner projects are the primary nodes (size = task count);
   missions orbit ZEUS (the self-development target) or the project they
   name; internal capability families are secondary nodes shown only on
   request. Labels are laid out to avoid collisions; the layout relaxes for a
   bounded number of frames and then stops (no perpetual CPU). */
function constellation(canvas, projects, missions, internal, focusText) {
  const ctx = canvas.getContext("2d");
  const q = (focusText || "").toLowerCase();
  const nodes = projects.map((p) => ({ id: p.id, label: (p.title || p.goal || "").slice(0, 28), r: 12 + Math.min(24, (p.tasks || 0) * 2),
    tone: STATE_TONE[String(p.state).toLowerCase()] || "idle", x: 0, y: 0, vx: 0, vy: 0, kind: "project", data: p }));
  const zeus = { id: "zeus", label: "ZEUS", r: 26, tone: "self", x: 0, y: 0, vx: 0, vy: 0, kind: "self" };
  nodes.push(zeus);
  const edges = [];
  const byTitle = (text) => nodes.find((n) => n.kind === "project" && text && (text.toLowerCase().includes(String(n.data.title || "").toLowerCase()) && n.data.title));
  for (const m of missions.filter((m) => m.system !== "acquisition").slice(0, 14)) {
    const parent = byTitle(m.goal || "") || zeus;
    const node = { id: m.id, label: (m.title || m.goal || "").slice(0, 22), r: 5, tone: m.state === "completed" ? "done" : m.state === "failed" || m.state === "blocked" ? "blocked" : m.state === "active" ? "active" : "idle",
      x: 0, y: 0, vx: 0, vy: 0, kind: "mission", data: m, parent };
    nodes.push(node);
    edges.push([parent, node]);
  }
  for (const fam of internal.slice(0, 12)) {
    const node = { id: fam.capability_id, label: fam.capability_id.slice(0, 24) + ` (${fam.count})`, r: 7, tone: "idle", x: 0, y: 0, vx: 0, vy: 0, kind: "family", data: fam, parent: zeus };
    nodes.push(node);
    edges.push([zeus, node]);
  }
  const resize = () => { canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; };
  resize();
  const W = () => canvas.width / devicePixelRatio, H = () => canvas.height / devicePixelRatio;
  const primaries = nodes.filter((n) => n.kind === "project");
  primaries.forEach((n, i) => { const a = (i / Math.max(1, primaries.length)) * Math.PI * 2; n.x = W() / 2 + Math.cos(a) * W() * 0.3; n.y = H() / 2 + Math.sin(a) * H() * 0.32; });
  zeus.x = W() / 2; zeus.y = H() / 2;
  for (const n of nodes) if (n.parent) { n.x = n.parent.x + (Math.random() - 0.5) * 80; n.y = n.parent.y + (Math.random() - 0.5) * 80; }
  const colours = { active: "#66d9a0", blocked: "#ff6b6b", done: "#4fc3f7", idle: "#6b7c93", self: "#8fd3ff" };
  let hover = null;
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    hover = nodes.find((n) => Math.hypot(n.x - x, n.y - y) < n.r + 6) || null;
    canvas.style.cursor = hover ? "pointer" : "default";
    if (frames >= LIMIT) draw();
  };
  canvas.onclick = () => {
    if (!hover) return;
    if (hover.kind === "project") views.open("projects", { id: hover.id });
    else if (hover.kind === "mission") views.open("missions", { mission: hover.id });
    else if (hover.kind === "family") views.open("capabilities", { id: hover.id });
    else views.open("missions");
  };
  const LIMIT = 240;
  let frames = 0;
  const step = () => {
    for (const a of nodes) {
      for (const b of nodes) {
        if (a === b) continue;
        const dx = a.x - b.x, dy = a.y - b.y, d = Math.max(20, Math.hypot(dx, dy));
        const f = (a.r + b.r + 46 - d) / d;
        if (f > 0) { a.vx += dx * f * 0.02; a.vy += dy * f * 0.02; }
      }
      a.vx += (W() / 2 - a.x) * 0.0015; a.vy += (H() / 2 - a.y) * 0.0015;
      a.vx *= 0.85; a.vy *= 0.85;
      if (a !== zeus) { a.x = Math.max(a.r + 8, Math.min(W() - a.r - 8, a.x + a.vx)); a.y = Math.max(a.r + 8, Math.min(H() - a.r - 18, a.y + a.vy)); }
    }
    for (const [a, b] of edges) { const dx = a.x - b.x, dy = a.y - b.y, d = Math.max(1, Math.hypot(dx, dy)); const want = a === zeus ? 120 : 70; b.vx += dx * (d - want) / d * 0.02; b.vy += dy * (d - want) / d * 0.02; }
  };
  const draw = () => {
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, W(), H());
    ctx.strokeStyle = "rgba(79,195,247,.18)"; ctx.lineWidth = 1;
    for (const [a, b] of edges) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
    const placed = [];
    const dim = (n) => q && !`${n.label} ${n.data?.goal || ""}`.toLowerCase().includes(q) && n.kind !== "self";
    for (const n of nodes) {
      const c = colours[n.tone] || colours.idle;
      ctx.globalAlpha = dim(n) ? 0.25 : 1;
      const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 2.2);
      g.addColorStop(0, c + "55"); g.addColorStop(1, "transparent");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 2.2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = n === hover ? "#ffffff" : c; ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    // labels: primaries always, secondaries only when hovered or focused or when there is room
    ctx.textAlign = "center";
    for (const n of nodes.slice().sort((a, b) => b.r - a.r)) {
      const secondary = n.kind === "mission" || n.kind === "family";
      const wanted = !secondary || n === hover || (q && !dim(n));
      const size = n.kind === "self" ? 12 : secondary ? 10 : 11;
      ctx.font = `${size}px Segoe UI, sans-serif`;
      const w = ctx.measureText(n.label).width;
      const box = { x: n.x - w / 2 - 3, y: n.y + n.r + 4, w: w + 6, h: size + 4 };
      const collides = placed.some((b) => !(box.x + box.w < b.x || b.x + b.w < box.x || box.y + box.h < b.y || b.y + b.h < box.y));
      if (!wanted && (collides || dim(n))) continue;
      if (collides && wanted && secondary && n !== hover) continue;
      placed.push(box);
      ctx.globalAlpha = dim(n) ? 0.3 : 1;
      ctx.fillStyle = n === hover ? "#dce5f0" : secondary ? "#8a9bb3" : "#b8c7da";
      ctx.fillText(n.label, n.x, box.y + size);
      ctx.globalAlpha = 1;
    }
  };
  const tick = () => {
    step(); draw();
    frames += 1;
    if (frames < LIMIT) anim = requestAnimationFrame(tick);   // settle, then stop consuming CPU
  };
  tick();
}

/* ---- deep view ------------------------------------------------------ */

async function deep(pane, id) {
  const detail = await api("/api/project", { id });
  if (detail.error) { pane.append(el("div", { class: "empty", text: detail.error })); return; }
  views.breadcrumb([{ label: "Projects", onClick: () => views.open("projects") }, { label: (detail.title || detail.goal || id).slice(0, 40) }]);
  const tasks = detail.tasks || [];
  const done = tasks.filter((t) => ["done", "complete", "completed", "accepted"].includes(String(t.status))).length;
  const blocked = tasks.filter((t) => ["blocked", "failed"].includes(String(t.status)));
  const active = tasks.find((t) => ["active", "running", "working"].includes(String(t.status)));
  const health = blocked.length ? ["BLOCKED", "bad"] : active ? ["ON TRACK", "ok"] : done === tasks.length && tasks.length ? ["COMPLETE", "blue"] : ["IDLE", "dim"];
  const steps = detail.steps || [];
  const elapsed = steps.length ? (new Date(steps[steps.length - 1].at || 0) - new Date(steps[0].at || 0)) / 1000 : 0;

  pane.append(
    el("div", { class: "card" },
      el("div", { class: "title", text: detail.goal || "" }),
      el("div", { class: "meta" }, badge(health[0], health[1]), badge(detail.state || "unknown", STATE_TONE[String(detail.state).toLowerCase()] || "idle"),
        el("span", { text: tasks.length ? `${done}/${tasks.length} tasks done (derived from task status)` : "no tasks" }),
        elapsed ? el("span", { text: `elapsed ${seconds(elapsed)}` }) : null),
      tasks.length ? el("div", { class: "bar green", style: { marginTop: "8px" } }, el("i", { style: { width: `${(done / tasks.length) * 100}%` } })) : null),
  );
  const answer = (q, a) => el("div", { class: "kv" }, el("span", { class: "k", text: q }), el("span", { class: "v", text: a }));
  pane.append(section("At a glance",
    answer("What are we doing", detail.goal || "—"),
    answer("Where are we", tasks.length ? `${done} of ${tasks.length} tasks complete` : "no plan yet"),
    answer("Happening now", active ? active.title : (steps.at(-1)?.summary || "nothing running")),
    answer("Blocking us", blocked.length ? blocked.map((t) => t.title).join("; ") : "nothing"),
    answer("Next", (tasks.find((t) => ["todo", "pending", "open", "planned"].includes(String(t.status))) || {}).title || "—")));
  if ((detail.acceptance || []).length) pane.append(section("Acceptance", ...detail.acceptance.map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: a.satisfied ? "✓" : "·" }), el("span", { class: "v", text: a.text })))));
  if (tasks.length) pane.append(section("Tasks / dependencies", dependencyGraph(tasks)));
  pane.append(section("Timeline", timeline(steps)));
  pane.append(el("div", { class: "toolbar" },
    button("Continue this project", () => chat.send(`Continue the project: ${detail.goal}`), "primary"),
    button("Ask ZEUS about it", () => chat.send(`What is the state of the project "${detail.goal}"? What blocks it and what is next?`)),
    button("Open graph", () => views.open("knowledge", { q: detail.goal || "" }))));
}

/* Tasks as a dependency chain: what the store knows (order + status) drawn as
   a left-to-right graph; blocked tasks red, the current one blue. */
function dependencyGraph(tasks) {
  const wrap = el("div", { class: "timeline" });
  for (const t of tasks) {
    const s = String(t.status);
    const cls = ["done", "complete", "completed", "accepted"].includes(s) ? "ok" : ["blocked", "failed"].includes(s) ? "bad" : ["active", "running", "working"].includes(s) ? "work" : "";
    wrap.append(el("div", { class: "tl " + cls },
      el("span", { class: "when", text: s }), el("span", { class: "text", text: t.title }),
      t.attempts ? el("span", { class: "sub", text: `${t.attempts} attempts` }) : null,
      cls === "bad" ? el("span", { class: "sub", text: "cannot proceed: this task is unfinished" }) : null));
  }
  return wrap;
}

function timeline(steps) {
  const wrap = el("div", { class: "timeline" });
  for (const s of steps.slice(-40).reverse()) {
    wrap.append(el("div", { class: "tl " + (s.success ? "ok" : "bad") },
      el("span", { class: "when", text: (s.at || "").slice(11, 19) }), el("span", { class: "text", text: `${s.phase || ""} — ${s.summary || ""}` })));
  }
  if (!steps.length) wrap.append(el("div", { class: "empty", text: "No events yet." }));
  return wrap;
}
