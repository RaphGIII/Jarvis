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
import * as palette from "./views/palette.js";
import * as mic from "./voice/mic.js";
import * as playback from "./voice/playback.js";

let eye = null;
let stream = null;
let lastSeq = 0;
let reconnectDelay = 500;

const VIEW_MODULES = [missions, projects, files, knowledge, personality, activity, corrections, diagnostics, owner, release, capabilities, voiceStudio, chessTool, thoughts];

/* The cosmos behind the shell: a static star field painted once (and on
   resize). No animation loop — depth for free, GPU untouched. */
function paintCosmos() {
  const canvas = $("cosmos");
  if (!canvas) return;
  const w = (canvas.width = window.innerWidth), h = (canvas.height = window.innerHeight);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  let seed = 9;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let i = 0; i < 340; i++) {
    const x = rnd() * w, y = rnd() * h, d = rnd();
    const warm = rnd() > 0.86;
    ctx.fillStyle = warm ? `rgba(255,226,190,${0.12 + d * 0.4})` : `rgba(175,200,240,${0.1 + d * 0.42})`;
    ctx.beginPath(); ctx.arc(x, y, 0.4 + d * 1.1, 0, Math.PI * 2); ctx.fill();
  }
}

function startJarvis() {
  if (new URLSearchParams(location.search).has("tv")) {
    document.body.classList.add("tv");
    document.documentElement.requestFullscreen?.().catch(() => {});
  }
  if (state.ui.reducedMotion) document.body.classList.add("reduced-motion");
  paintCosmos();
  window.addEventListener("resize", () => paintCosmos());

  eye = new JarvisEye($("eye"));
  eye.start();
  if (window.ResizeObserver) new ResizeObserver(() => eye.resize()).observe($("eye"));
  else window.addEventListener("resize", () => eye.resize());

  for (const mod of VIEW_MODULES) views.register(mod.view);
  chat.init({ eye });
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
                      "notification", "error", "speech", "diagnostic", "knowledge"]) {
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
  if (health.ok === false) return;
  set("health", health);
  const rd = health.readiness || {};
  for (const span of document.querySelectorAll("#readiness span")) {
    span.classList.toggle("on", Boolean(rd[span.dataset.stage]));
  }
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
