/*
 * The client entry point: the shell of ONE product.  Subscribes to the event
 * stream, keeps the shared state, drives the orb, registers the views and
 * builds the sidebar, the performance control and the work surface.
 *
 * All state comes from the server.  The UI never guesses what ZEUS is doing
 * -- it renders the state event and nothing else -- and it never names the
 * intelligence that computed an answer: that truth lives in Settings ›
 * Erweitert, and nowhere in the ordinary product.
 */

import { $, el } from "./core/dom.js";
import { api } from "./core/api.js";
import * as bus from "./core/bus.js";
import { state, set, setPref, category } from "./core/state.js";
import * as views from "./core/views.js";
import * as sidebar from "./core/sidebar.js";
import * as worksurface from "./core/worksurface.js";

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
import * as settings from "./views/settings.js";
import * as palette from "./views/palette.js";
import * as mic from "./voice/mic.js";
import * as playback from "./voice/playback.js";

let eye = null;
let stream = null;
let lastSeq = 0;
let reconnectDelay = 500;

const VIEW_MODULES = [missions, projects, files, knowledge, calendar, personality, activity, corrections, diagnostics, owner, release,
                      capabilities, voiceStudio, chessTool, thoughts, settings];

/* ------------------------------------------------------------------ */
/* appearance: dark / light / system, glass, motion                    */
/* ------------------------------------------------------------------ */

const systemDark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

export function applyAppearance(next) {
  // Dark is the product.  The light appearance exists for the owner who asks for it explicitly;
  // "system" never turns ZEUS light by itself.
  const wanted = next || state.ui.appearance || "dark";
  const dark = wanted !== "light";
  document.documentElement.dataset.appearance = dark ? "dark" : "light";
  document.documentElement.dataset.glass = state.ui.glass || "normal";
  document.body.classList.toggle("reduced-motion", Boolean(state.ui.reducedMotion));
  const intensity = { OFF: 0, LOW: 0.55, NORMAL: 1.0, HIGH: 1.35 }[state.ui.animIntensity || "NORMAL"] ?? 1.0;
  eye?.setIntensity?.(intensity);
  document.body.classList.toggle("anim-off", intensity === 0);
}
systemDark?.addEventListener?.("change", () => applyAppearance());

function startZeus() {
  if (new URLSearchParams(location.search).has("tv")) {
    document.body.classList.add("tv");
    document.documentElement.requestFullscreen?.().catch(() => {});
  }
  applyAppearance();
  // owner-level interface preferences live on the server, so they hold across profiles and restarts
  api("/api/ui/preferences").then((r) => {
    const prefs = (r && r.preferences) || {};
    if (typeof prefs["ui.show_spend"] === "boolean" && (state.ui.showSpend !== false) !== prefs["ui.show_spend"]) setPref("showSpend", prefs["ui.show_spend"]);
  }).catch(() => {});

  eye = new JarvisEye($("eye"));
  window.zeusEye = eye;
  eye.start();
  if (window.ResizeObserver) new ResizeObserver(() => eye.resize()).observe($("eye"));
  else window.addEventListener("resize", () => eye.resize());

  for (const mod of VIEW_MODULES) views.register(mod.view);
  chat.init({ eye });
  sidebar.init({ chat, toast });
  worksurface.init({ toast });
  bus.on("jobs:active", (n) => eye?.setBackgroundWork?.(Number(n) || 0));
  mic.init({ eye });
  playback.init({ eye });
  palette.init();
  wireShell();
  connect();
  refreshStatus();
  refreshHealth();
  setInterval(refreshStatus, 15000);
  setInterval(refreshHealth, 5000);
  const bootPoll = setInterval(() => { if (veilLifted) clearInterval(bootPoll); else refreshHealth(); }, 1000);

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
    setPill("verbunden", "live");
    refreshStatus();
  };
  stream.onerror = () => {
    setPill("verbinde neu", "warn");
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
      if (event.replay) payload._replay = true;
      payload._seq = event.seq;
      bus.emit(type, payload);
    });
  }
}

/* ZEUS opening a view on request.  Never from replayed history. */
bus.on("notification", (payload) => {
  if (payload._replay) return;
  if (payload.kind === "open_view" && payload.view) { views.open(payload.view, payload.params || {}); return; }
  if (payload.kind === "needs_auth" && payload.scope) {
    import("./core/authgate.js").then(async (authgate) => {
      const token = await authgate.ensureAuth(payload.scope, { reason: payload.text });
      if (!token) return;
      const retry = payload.retry || {};
      if (retry.operation === "project.delete" && retry.target) {
        await api("/api/project/delete", { id: retry.target, authorization: token });
      }
      // a protected memory write: the same words again, now with the owner's authorization
      if (retry.operation === "message" && retry.text) {
        await chat.send(retry.text, retry.source || "text", { authorization: token, mode: retry.mode });
      }
    });
  }
  if (payload.text && (payload.kind || "").match(/release|relaunch|restart|owner_config|correction|isolation/)) toast(payload.text, "note");
});

/* ------------------------------------------------------------------ */
/* state -> orb, label                                                 */
/* ------------------------------------------------------------------ */

const STATE_WORDS = {
  idle: "bereit", listening: "hört zu", transcribing: "versteht", thinking: "denkt", speaking: "spricht", waiting: "wartet auf dich",
  working: "arbeitet", verifying: "prüft", coding: "entwickelt", researching: "recherchiert", error: "unterbrochen", offline: "offline",
};

bus.on("state", (payload) => {
  const name = payload.state || "idle";
  set("eye", payload);
  eye.setState(name);
  const label = $("stateLabel");
  label.textContent = STATE_WORDS[name] || name;
  label.dataset.cat = category(name);
  label.dataset.state = name;
  $("detail").textContent = payload.detail || "";
});

bus.on("speech", (payload) => {
  if (typeof payload.energy === "number") eye.setEnergy(payload.energy);
});

export function toast(text, tone = "") {
  const node = el("div", { class: "toast glass " + tone, text });
  $("toasts").append(node);
  setTimeout(() => node.remove(), 6000);
}

/* ------------------------------------------------------------------ */
/* shell wiring                                                        */
/* ------------------------------------------------------------------ */

function wireShell() {
  $("btnWorkspaceClose").onclick = () => views.close();
  $("btnSearch").onclick = () => palette.open();
  // the app window's own controls: only inside the desktop shell (Chromium app mode), never in a browser tab
  // Chromium fullscreen reports display-mode: fullscreen, an --app window standalone; both are the desktop shell
  const standalone = window.matchMedia && (window.matchMedia("(display-mode: standalone)").matches || window.matchMedia("(display-mode: fullscreen)").matches);
  if (standalone) {
    $("winctl").hidden = false;
    $("btnWinMin").onclick = () => api("/api/window", { action: "minimize", reason: "owner" });
    $("btnWinClose").onclick = () => api("/api/window", { action: "close", reason: "owner" });
  }
  $("btnProfile").onclick = () => views.open("settings", {});
  $("btnPlus").onclick = () => palette.open();
  $("btnInspectorClose").onclick = () => views.closeInspector();
  $("btnSidebar").onclick = () => {
    const narrow = window.innerWidth <= 980;
    if (narrow) $("app").classList.toggle("sidebar-open");
    else { const collapsed = $("app").classList.toggle("sidebar-collapsed"); setPref("sidebarCollapsed", collapsed); }
  };
  if (state.ui.sidebarCollapsed) $("app").classList.add("sidebar-collapsed");
  bus.on("view:open", ({ id }) => {
    const view = views.get(id);
    $("topTitle").textContent = view ? view.title : "";
    $("btnWorkspaceClose").hidden = false;
    if (window.innerWidth <= 980) $("app").classList.remove("sidebar-open");
  });
  bus.on("view:close", () => {
    $("topTitle").textContent = $("app").classList.contains("conversing") ? "Chat" : (window.PRODUCT_NAME || "ZEUS");
    $("btnWorkspaceClose").hidden = true;
  });

  document.addEventListener("keydown", (e) => {
    const inField = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (e.key === "F11") {
      e.preventDefault(); e.stopPropagation();
      api("/api/window", { action: "toggle_fullscreen", reason: "f11" }).then((r) => { if (r && r.ok === false) toast(r.error || "Fenstermodus konnte nicht gewechselt werden", "warn"); });
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); palette.open(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "p" && !e.shiftKey) { e.preventDefault(); palette.open("search: "); return; }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "p") { e.preventDefault(); views.open("projects"); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "m") { e.preventDefault(); views.open("missions"); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === ",") { e.preventDefault(); views.open("settings"); return; }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); sidebar.newChat(); return; }
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

/* ------------------------------------------------------------------ */
/* polls: status, health                                               */
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
  const conn = String(status.connection || "OFFLINE");
  setPill(conn === "OFFLINE" ? "offline" : conn === "STARTING" ? "startet" : "verbunden", conn === "OFFLINE" ? "bad" : "live");
}

async function refreshHealth() {
  const health = await api("/api/health");
  if (health.ok === false) { updateVeil({}, null); return; }
  set("health", health);
  const rd = health.readiness || {};
  for (const span of document.querySelectorAll("#readiness span")) span.classList.toggle("on", Boolean(rd[span.dataset.stage]));
  updateVeil(rd, health);
}

/* The boot veil lifts on INTERACTIVE_READY: the core answers and its intelligence is wired. */
let veilLifted = false;
function updateVeil(rd, health) {
  const veil = $("bootVeil");
  if (!veil || veilLifted) return;
  const lights = { core: rd.CORE_READY, ai: rd.AI_READY, voice: rd.VOICE_READY };
  for (const li of veil.querySelectorAll(".bv-systems li")) li.classList.toggle("on", Boolean(lights[li.dataset.k]));
  const phase = !rd.CORE_READY ? "ZEUS WIRD GESTARTET" : !rd.AI_READY ? "INTELLIGENZ WIRD VERBUNDEN" : "ZEUS IST BEREIT";
  const node = $("bvPhase");
  if (node && node.textContent !== phase) node.textContent = phase;
  if (rd.INTERACTIVE_READY) {
    veilLifted = true;
    veil.classList.add("lifting");
    setTimeout(() => { veil.hidden = true; }, 950);
    idlePreload();
  }
}

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

window.addEventListener("error", (e) => {
  if (String(e.message || "").includes("ResizeObserver loop")) return;
  console.error("[zeus ui]", e.message, e.filename, e.lineno);
  toast("Ein Teil der Oberfläche hat einen Fehler gemeldet. Details in der Konsole.", "bad");
});
window.addEventListener("unhandledrejection", (e) => { console.error("[zeus ui]", e.reason); });

window.startJarvis = startZeus;
window.zeus = { views, bus, state, api, toast, applyAppearance, sidebar };
startZeus();
