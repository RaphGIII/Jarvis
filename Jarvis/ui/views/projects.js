/* Projects: a constellation of real project nodes (size from task count,
   links from real relations: a mission that belongs to a project, a
   capability a project produced), and a deep view answering "what are we
   doing, where are we, what is happening, what blocks us, what is next". */

import { $, el, clear, kv, section, badge, button, ago, seconds } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const STATE_TONE = { active: "active", running: "active", working: "active", blocked: "blocked", failed: "blocked",
                     accepted: "done", complete: "done", completed: "done" };

let anim = 0;

export const view = {
  id: "projects",
  title: "Projects",
  async mount(pane, params) {
    if (params.id) return deep(pane, params.id);
    const [data, selfdev] = await Promise.all([api("/api/projects"), api("/api/selfdev")]);
    const projects = data.projects || [];
    const missions = selfdev.missions || [];
    const canvas = el("canvas", { id: "constellation" });
    pane.append(canvas);
    const grid = el("div", { class: "grid" });
    if (!projects.length) grid.append(el("div", { class: "empty", text: "No projects yet. Describe something to build and ZEUS opens one." }));
    for (const p of projects) grid.append(card(p));
    pane.append(grid);
    constellation(canvas, projects, missions);
  },
  unmount() { cancelAnimationFrame(anim); },
};

function card(p) {
  const node = el("div", { class: "card click" },
    el("div", { class: "title", text: p.title || p.goal || "(no goal recorded)" }),
    el("div", { class: "meta" }, badge(p.state || "unknown", STATE_TONE[p.state] || "idle"),
      el("span", { text: `${p.tasks ?? 0} tasks` }), el("span", { text: `${p.steps ?? 0} steps` }), el("span", { text: ago(p.updated_at) })));
  node.onclick = () => views.open("projects", { id: p.id });
  return node;
}

/* Real relations only: ZEUS (the self-development target) is a node the
   selfdev missions orbit; every project is a node sized by its task count. */
function constellation(canvas, projects, missions) {
  const ctx = canvas.getContext("2d");
  const nodes = projects.map((p, i) => ({ id: p.id, label: (p.title || p.goal || "").slice(0, 28), r: 10 + Math.min(24, (p.tasks || 0) * 2),
    tone: STATE_TONE[p.state] || "idle", x: 0, y: 0, vx: 0, vy: 0, kind: "project", data: p }));
  const zeus = { id: "zeus", label: "ZEUS", r: 26, tone: "self", x: 0, y: 0, vx: 0, vy: 0, kind: "self" };
  nodes.push(zeus);
  const edges = [];
  for (const m of missions.slice(-12)) {
    const node = { id: m.mission_id, label: (m.request || "").slice(0, 24), r: 6, tone: m.outcome === "promoted" ? "active" : m.outcome === "failed" ? "blocked" : m.finished ? "idle" : "active",
      x: 0, y: 0, vx: 0, vy: 0, kind: "mission", data: m };
    nodes.push(node);
    edges.push([zeus, node]);
  }
  const resize = () => { canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; };
  resize();
  const W = () => canvas.width / devicePixelRatio, H = () => canvas.height / devicePixelRatio;
  nodes.forEach((n, i) => { const a = (i / nodes.length) * Math.PI * 2; n.x = W() / 2 + Math.cos(a) * W() * 0.28; n.y = H() / 2 + Math.sin(a) * H() * 0.3; });
  zeus.x = W() / 2; zeus.y = H() / 2;
  const colours = { active: "#66d9a0", blocked: "#ff6b6b", done: "#4fc3f7", idle: "#6b7c93", self: "#8fd3ff" };
  let hover = null;
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    hover = nodes.find((n) => Math.hypot(n.x - x, n.y - y) < n.r + 6) || null;
    canvas.style.cursor = hover ? "pointer" : "grab";
  };
  canvas.onclick = () => {
    if (!hover) return;
    if (hover.kind === "project") views.open("projects", { id: hover.id });
    else if (hover.kind === "mission") views.open("missions", { mission: hover.id });
    else views.open("missions");
  };
  let t = 0;
  const tick = () => {
    t += 0.008;
    // a gentle relaxation: repulsion between nodes, a pull toward the centre
    for (const a of nodes) {
      for (const b of nodes) {
        if (a === b) continue;
        const dx = a.x - b.x, dy = a.y - b.y, d = Math.max(20, Math.hypot(dx, dy));
        const f = (a.r + b.r + 40 - d) / d;
        if (f > 0) { a.vx += dx * f * 0.02; a.vy += dy * f * 0.02; }
      }
      a.vx += (W() / 2 - a.x) * 0.002; a.vy += (H() / 2 - a.y) * 0.002;
      a.vx *= 0.85; a.vy *= 0.85;
      if (a !== zeus) { a.x += a.vx; a.y += a.vy; }
    }
    for (const [a, b] of edges) { const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy); const want = 90; b.vx += dx * (d - want) / d * 0.02; b.vy += dy * (d - want) / d * 0.02; }
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, W(), H());
    ctx.strokeStyle = "rgba(79,195,247,.18)"; ctx.lineWidth = 1;
    for (const [a, b] of edges) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
    for (const n of nodes) {
      const c = colours[n.tone] || colours.idle;
      const pulse = n.kind === "mission" && !n.data.finished ? 1 + Math.sin(t * 6) * 0.15 : 1;
      const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 2.2 * pulse);
      g.addColorStop(0, c + "55"); g.addColorStop(1, "transparent");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 2.2 * pulse, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = n === hover ? "#ffffff" : c; ctx.beginPath(); ctx.arc(n.x, n.y, n.r * pulse, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = n === hover ? "#dce5f0" : "#9db0c8"; ctx.font = `${n.kind === "self" ? 12 : 11}px Segoe UI, sans-serif`; ctx.textAlign = "center";
      ctx.fillText(n.label, n.x, n.y + n.r + 14);
    }
    anim = requestAnimationFrame(tick);
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
      el("div", { class: "meta" }, badge(health[0], health[1]), badge(detail.state || "unknown", STATE_TONE[detail.state] || "idle"),
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
