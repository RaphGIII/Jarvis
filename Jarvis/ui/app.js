/*
 * The client entry point. Subscribes to the event stream, keeps the shared
 * state, drives the eye, and registers the views. Everything else lives in
 * its own module (ui/core, ui/views, ui/voice) so that ZEUS can change one
 * area of its interface without rewriting the whole of it.
 *
 * All state comes from the server. The UI never guesses what ZEUS is doing --
 * it renders the state event and nothing else. "Start thinking" is not
 * something the send button does locally.
 */

import { $, el } from "./core/dom.js";
import { api } from "./core/api.js";
import * as bus from "./core/bus.js";
import { state, set, setPref, category } from "./core/state.js";
import * as views from "./core/views.js";

import * as chat from "./views/chat.js";
import * as activity from "./views/activity.js";
import * as projects from "./views/projects.js";
import * as files from "./views/files.js";
import * as personality from "./views/personality.js";
import * as missions from "./views/missions.js";
import * as knowledge from "./views/knowledge.js";
import * as corrections from "./views/corrections.js";
import * as diagnostics from "./views/diagnostics.js";
import * as owner from "./views/owner.js";
import * as release from "./views/release.js";
import * as capabilities from "./views/capabilities.js";
import * as voiceStudio from "./views/voice.js";
import * as chessTool from "./views/chess.js";
import * as thoughts from "./views/thoughts.js";
import * as calendar from "./views/calendar.js";
import * as palette from "./views/palette.js";
import * as mic from "./voice/mic.js";
import * as playback from "./voice/playback.js";

let eye = null;
let stream = null;
let lastSeq = 0;
let reconnectDelay = 500;

const VIEW_MODULES = [missions, projects, files, knowledge, calendar, personality, activity, corrections, diagnostics, owner, release, capabilities, voiceStudio, chessTool, thoughts];

/* The cosmos behind the shell: a LIVING star field — slow drift, quiet
   twinkle — throttled to ~24fps over a few hundred dots, so depth costs
   almost nothing. Honest about restraint: with reduced motion (or the
   owner's cosmosMotion preference off, or a hidden tab) it paints once and
   stands still. */
const cosmos = { stars: [], w: 0, h: 0, raf: 0, last: 0 };

function seedCosmos() {
  const canvas = $("cosmos");
  if (!canvas) return;
  cosmos.w = canvas.width = window.innerWidth;
  cosmos.h = canvas.height = window.innerHeight;
  let seed = 9;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  cosmos.stars = Array.from({ length: 360 }, () => ({
    x: rnd() * cosmos.w, y: rnd() * cosmos.h, d: rnd(),
    warm: rnd() > 0.86, tw: rnd() * 6.28, sp: 0.02 + rnd() * 0.05,
    vx: (rnd() - 0.5) * 2.2, vy: (rnd() - 0.5) * 1.4, // px per SECOND — a drift, not a flight
  }));
}

function cosmosAlive() {
  return !state.ui.reducedMotion && state.ui.cosmosMotion !== false;
}

function drawCosmos(t) {
  const canvas = $("cosmos");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cosmos.w, cosmos.h);
  for (const s of cosmos.stars) {
    const tw = cosmosAlive() ? 0.7 + 0.3 * Math.sin(t * s.sp * 4 + s.tw) : 1;
    const a = (s.warm ? 0.12 + s.d * 0.4 : 0.1 + s.d * 0.42) * tw;
    ctx.fillStyle = s.warm ? `rgba(255,226,190,${a})` : `rgba(175,200,240,${a})`;
    ctx.beginPath(); ctx.arc(s.x, s.y, 0.4 + s.d * 1.1, 0, Math.PI * 2); ctx.fill();
  }
}

function tickCosmos(now) {
  cosmos.raf = requestAnimationFrame(tickCosmos);
  if (document.hidden || !cosmosAlive()) return;
  if (now - cosmos.last < 42) return; // ~24fps is plenty for a twinkle
  const dt = Math.min(0.2, (now - cosmos.last) / 1000) || 0.04;
  cosmos.last = now;
  for (const s of cosmos.stars) {
    s.x = (s.x + s.vx * dt + cosmos.w) % cosmos.w;
    s.y = (s.y + s.vy * dt + cosmos.h) % cosmos.h;
  }
  drawCosmos(now / 1000);
}

function paintCosmos() {
  seedCosmos();
  drawCosmos(0);
  if (!cosmos.raf) cosmos.raf = requestAnimationFrame(tickCosmos);
}
bus.on("state:ui", () => { if (!cosmosAlive()) drawCosmos(0); });

function startJarvis() {
  if (new URLSearchParams(location.search).has("tv")) {
    document.body.classList.add("tv");
    document.documentElement.requestFullscreen?.().catch(() => {});
  }
  if (state.ui.reducedMotion) document.body.classList.add("reduced-motion");
  paintCosmos();
  window.addEventListener("resize", () => paintCosmos());

  eye = new JarvisEye($("eye"));
  window.zeusEye = eye; // settings and tests reach the live eye here
  eye.start();
  import("./core/themes.js").then((themes) => themes.apply(state.ui.theme || "COSMOS", eye, state.ui.animIntensity || "NORMAL"));
  if (window.ResizeObserver) new ResizeObserver(() => eye.resize()).observe($("eye"));
  else window.addEventListener("resize", () => eye.resize());

  for (const mod of VIEW_MODULES) views.register(mod.view);
  chat.init({ eye });
  import("./core/workrail.js").then((workrail) => workrail.init({ toast, chat }));
  // the eye carries a background-work indicator whenever jobs are active
  bus.on("jobs:active", (n) => eye?.setBackgroundWork?.(Number(n) || 0));
  mic.init({ eye });
  playback.init({ eye });
  palette.init();
  wireShell();
  buildRail();
  connect();
  refreshStatus();
  refreshHealth();
  setInterval(refreshStatus, 15000);
  setInterval(refreshHealth, 5000);
  // during boot the veil deserves a live picture: poll fast until it lifts
  const bootPoll = setInterval(() => { if (veilLifted) clearInterval(bootPoll); else refreshHealth(); }, 1000);
  refreshGpu();
  setInterval(refreshGpu, 3000);
  setInterval(drawUptime, 1000);

  const target = views.fromHash();
  if (target) views.open(target.id, target.params, { push: false });
}

/* ------------------------------------------------------------------ */
/* transport                                                           */
/* ------------------------------------------------------------------ */

function connect() {
  if (stream) stream.close();
  stream = new EventSource(`/events?token=${encodeURIComponent(window.JARVIS_TOKEN)}&since=${lastSeq}`);
  stream.onopen = () => {
    reconnectDelay = 500;
    setPill("connected", "live");
    refreshStatus();
  };
  stream.onerror = () => {
    setPill("reconnecting", "warn");
    stream.close();
    reconnectDelay = Math.min(10000, reconnectDelay * 2);
    setTimeout(connect, reconnectDelay);
  };
  for (const type of ["state", "token", "message", "user_message", "transcript", "tool", "progress",
                      "notification", "error", "speech", "diagnostic", "knowledge", "job"]) {
    stream.addEventListener(type, (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      if (event.seq) lastSeq = Math.max(lastSeq, event.seq);
      const payload = event.payload || {};
      // Replayed history (a refresh, a reconnect from seq 0) is the past:
      // views render it as history and never play its audio or act on it.
      if (event.replay) payload._replay = true;
      payload._seq = event.seq;
      bus.emit(type, payload);
    });
  }
}

/* ZEUS opening a view on request ("Zeus, öffne meine Projekte"). Never from
   replayed history: a refresh must not re-open whatever was opened before. */
bus.on("notification", (payload) => {
  if (payload._replay) return;
  if (payload.kind === "open_view" && payload.view) { views.open(payload.view, payload.params || {}); return; }
  if (payload.kind === "needs_auth" && payload.scope) {
    // A protected change is waiting for the owner's manually typed password.
    import("./core/authgate.js").then(async (authgate) => {
      const token = await authgate.ensureAuth(payload.scope, { reason: payload.text });
      if (!token) return;
      const retry = payload.retry || {};
      if (retry.operation === "project.delete" && retry.target) {
        await api("/api/project/delete", { id: retry.target, authorization: token });
      }
    });
  }
});

/* ------------------------------------------------------------------ */
/* state -> eye, label, HUD                                            */
/* ------------------------------------------------------------------ */

bus.on("state", (payload) => {
  const name = payload.state || "idle";
  set("eye", payload);
  eye.setState(name);
  const label = $("stateLabel");
  label.textContent = name;
  label.dataset.cat = category(name);
  $("detail").textContent = payload.detail || "";
});

bus.on("speech", (payload) => {
  if (typeof payload.energy === "number") eye.setEnergy(payload.energy);
});

/* The mission HUD: real progress events only. */
bus.on("progress", (payload) => {
  if (payload.kind !== "selfdev" && payload.kind !== "capability" && payload.kind !== "mission") return;
  const phase = String(payload.phase || "");
  const finished = ["DONE", "FAILED", "CANCELLED"].includes(phase);
  if (finished) {
    set("mission", null);
    $("hud").hidden = true;
    return;
  }
  const mission = { ...(state.mission || {}), ...payload, started: state.mission?.started || Date.now() };
  set("mission", mission);
  $("hud").hidden = false;
  $("hudKind").textContent = payload.kind;
  $("hudGoal").textContent = payload.summary || payload.request || mission.request || "";
  $("hudPhase").textContent = phase;
  const stages = ["UNDERSTAND", "INVESTIGATE", "BUILD", "VERIFY", "ESCALATE", "PROMOTE", "RESTARTING"];
  const idx = stages.indexOf(phase);
  $("hudBar").style.width = (idx >= 0 ? ((idx + 1) / stages.length) * 100 : 10) + "%";
});
bus.on("notification", (payload) => {
  if (payload.kind === "selfdev" && /cancelled|failed|done|promoted/i.test(payload.text || "")) {
    set("mission", null);
    $("hud").hidden = true;
  }
  if (payload.text && (payload.kind || "").match(/selfdev|release|relaunch|restart|owner_config|correction|isolation/)) toast(payload.text, "note");
});
setInterval(() => {
  if (!state.mission) return;
  const s = Math.floor((Date.now() - state.mission.started) / 1000);
  $("hudTime").textContent = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}, 1000);

export function toast(text, tone = "") {
  const node = el("div", { class: "toast " + tone, text });
  $("toasts").append(node);
  setTimeout(() => node.remove(), 6000);
}

/* ------------------------------------------------------------------ */
/* shell wiring                                                        */
/* ------------------------------------------------------------------ */

function wireShell() {
  $("brand").onclick = () => views.close();
  $("btnWorkspaceClose").onclick = () => views.close();
  $("btnInspectorClose").onclick = () => views.closeInspector();
  $("hud").onclick = () => views.open("missions");
  $("btnPalette").onclick = () => palette.open();
  for (const button of document.querySelectorAll("[data-view-button]")) {
    button.onclick = () => {
      const id = button.dataset.viewButton;
      if (views.currentView()?.id === id) views.close(); else views.open(id);
    };
  }
  $("btnNew").onclick = async () => {
    chat.reset();
    await api("/api/new", {});
  };
  // the shell is fullscreen and frameless: these are the window's real
  // controls, backed by Win32 through the core (never fake CSS buttons)
  $("btnWinMin").onclick = () => api("/api/window", { action: "minimize", reason: "owner" });
  $("btnWinHide").onclick = () => api("/api/window", { action: "hide", reason: "owner" });

  document.addEventListener("keydown", (e) => {
    const inField = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); palette.open(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "p" && !e.shiftKey) { e.preventDefault(); palette.open("search: "); return; }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "p") { e.preventDefault(); views.open("projects"); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "m") { e.preventDefault(); views.open("missions"); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === ",") { e.preventDefault(); views.open("owner"); return; }
    if (e.key === "Escape") {
      if (palette.isOpen()) { palette.close(); return; }
      if ($("panel").classList.contains("open")) { $("panel").classList.remove("open"); return; }
      if ($("graphView").classList.contains("open")) { knowledge.closeGraph(); return; }
      if (views.inspectorIsOpen()) { views.closeInspector(); return; }
      if (views.isWorkspace()) { views.close(); return; }
      if (!inField || document.activeElement === $("input")) { api("/api/stop", {}); chat.endStreaming(); }
    }
  });
  $("btnClose").onclick = () => $("panel").classList.remove("open");
}

function buildRail() {
  const rail = $("rail");
  const groups = [
    ["Work", ["missions", "projects", "files", "activity"]],
    ["Mind", ["knowledge", "personality", "corrections", "capabilities"]],
    ["System", ["diagnostics", "release", "owner", "voice"]],
  ];
  for (const [title, ids] of groups) {
    rail.append(el("h5", { text: title }));
    for (const id of ids) {
      const view = views.get(id);
      if (!view) continue;
      rail.append(el("button", { dataset: { viewButton: id }, onClick: () => views.open(id) }, view.title));
    }
  }
}

/* ------------------------------------------------------------------ */
/* polls: status, health, gpu, uptime                                  */
/* ------------------------------------------------------------------ */

function setPill(text, tone) {
  const pill = $("connPill");
  pill.textContent = text;
  pill.className = "pill" + (tone ? " " + tone : "");
  set("connection", text);
}

async function refreshStatus() {
  const status = await api("/api/status");
  if (status.ok === false && status.transport) { setPill("offline", "bad"); return; }
  set("status", status);
  const conn = status.connection || "OFFLINE";
  setPill(conn.toLowerCase(), conn === "OFFLINE" ? "bad" : conn === "EXPERT QUOTA EXHAUSTED" ? "warn" : "live");
  noteUptime(status.uptime_seconds);
}

async function refreshHealth() {
  const health = await api("/api/health");
  if (health.ok === false) { updateVeil({}, null); return; }
  set("health", health);
  const rd = health.readiness || {};
  for (const span of document.querySelectorAll("#readiness span")) {
    span.classList.toggle("on", Boolean(rd[span.dataset.stage]));
  }
  updateVeil(rd, health);
}

/* The boot veil: the main UI is covered until INTERACTIVE_READY (core
   answering + conversation model warm).  The owner never sees the shell
   reconnecting to its own backend; voice keeps warming behind the veil's
   dissolve with its own honest light. */
let veilLifted = false;
function updateVeil(rd, health) {
  const veil = $("bootVeil");
  if (!veil || veilLifted) return;
  const lights = { core: rd.CORE_READY, ai: rd.AI_READY, voice: rd.VOICE_READY, universe: rd.CORE_READY };
  for (const li of veil.querySelectorAll(".bv-systems li")) li.classList.toggle("on", Boolean(lights[li.dataset.k]));
  const detail = String(health?.detail || "");
  const phase = !rd.CORE_READY ? "KERN WIRD GESTARTET"
    : !rd.AI_READY ? (/unavailable|unreachable/i.test(detail) ? "LOKALE INTELLIGENZ WIRD WIEDERHERGESTELLT" : "LOKALE INTELLIGENZ WIRD GELADEN")
    : "ZEUS ONLINE";
  const node = $("bvPhase");
  if (node && node.textContent !== phase) node.textContent = phase;
  if (rd.INTERACTIVE_READY) {
    veilLifted = true;
    veil.classList.add("lifting");
    setTimeout(() => { veil.hidden = true; }, 950);
    idlePreload();
  }
}

/* Idle warming (§P1): once interactive and the browser reports idle time,
   prefetch what the owner opens most — the responses prime the server-side
   caches so the first Projects/Calendar click is warm.  One shot, low cost,
   never during visible work. */
let preloaded = false;
function idlePreload() {
  if (preloaded) return;
  preloaded = true;
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 4000));
  idle(() => {
    api("/api/projects/overview").catch(() => {});
    api("/api/calendar/list", { start: new Date().toISOString(), end: new Date(Date.now() + 7 * 864e5).toISOString() });
    api("/api/jobs");
  }, { timeout: 15000 });
}

let uptimeAt = 0;
let uptimeSeconds = -1;
function noteUptime(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return;
  uptimeSeconds = value;
  uptimeAt = performance.now();
  drawUptime();
}
function drawUptime() {
  if (uptimeSeconds < 0) return;
  const total = Math.max(0, Math.floor(uptimeSeconds + (performance.now() - uptimeAt) / 1000));
  const d = Math.floor(total / 86400), h = Math.floor((total % 86400) / 3600), m = Math.floor((total % 3600) / 60);
  $("uptime").textContent = "up " + (d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : m ? `${m}m` : `${total}s`);
}

const GPU_BUSY_PERCENT = 45;
async function refreshGpu() {
  if (document.hidden) return;
  const data = await api("/api/gpu");
  set("gpu", data);
  const meter = $("gpuMeter");
  const load = data && data.available ? Number(data.utilization_percent) : NaN;
  if (!Number.isFinite(load)) { meter.hidden = true; return; }
  const percent = Math.max(0, Math.min(100, Math.round(load)));
  meter.hidden = false;
  meter.classList.toggle("busy", percent >= GPU_BUSY_PERCENT);
  $("gpuFill").style.height = percent + "%";
  $("gpuValue").textContent = percent + "%";
  meter.title = `GPU ${data.name || ""} — ${percent}%` + (data.memory_total_mib ? ` · ${data.memory_used_mib} of ${data.memory_total_mib} MiB` : "");
}

// A script error is shown, not swallowed: a blank pane with a console line
// nobody reads is the failure mode this replaces.
window.addEventListener("error", (e) => {
  // the ResizeObserver loop warning is a benign browser notice, not a fault
  if (String(e.message || "").includes("ResizeObserver loop")) return;
  toast(`UI error: ${e.message} (${(e.filename || "").split("/").pop()}:${e.lineno})`, "bad");
});
window.addEventListener("unhandledrejection", (e) => toast(`UI error: ${e.reason && e.reason.message ? e.reason.message : e.reason}`, "bad"));

window.startJarvis = startJarvis;
window.zeus = { views, bus, state, api, toast };
startJarvis();
