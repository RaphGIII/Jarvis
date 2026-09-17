/* Active task surfaces: small, temporary, and only while something is happening.

   ZEUS stays calm until it acts.  When it acts, the one piece of interface the
   action needs appears above the composer and leaves again when it is done:

     music      "Play Spotify" -> a compact transport control (Windows media session)
     calendar   an event created or changed -> what was written, and the way to the calendar
     work       a job running (an image, an import) -> title, progress, cancel

   Study results are not a surface of their own: they arrive in the conversation
   as sources.  Every control here calls a real route. */

import { $, el, clear } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";
import * as views from "./views.js";

const live = new Map();          // kind -> {node, timer}
const HIDE_AFTER = { music: 45000, calendar: 14000, work: 4000 };

export function init() {
  bus.on("tool", (p) => {
    if (p._replay || !p.receipt) return;
    const kind = String(p.receipt.kind || "");
    if (kind.startsWith("music.")) showMusic(p.receipt);
    else if (kind.startsWith("calendar.")) showCalendar(p.receipt);
  });
  bus.on("job", (p) => { if (!p._replay && p.job_id) showWork(p); });
}

function host() {
  return $("surfaces");
}

function place(kind, node, { persistent = false } = {}) {
  const existing = live.get(kind);
  if (existing) { clearTimeout(existing.timer); existing.node.replaceWith(node); }
  else host()?.append(node);
  const entry = { node, timer: 0 };
  live.set(kind, entry);
  node.addEventListener("pointerenter", () => clearTimeout(entry.timer));
  node.addEventListener("pointerleave", () => schedule(kind));
  node.addEventListener("focusin", () => clearTimeout(entry.timer));
  if (!persistent) schedule(kind);
}

function schedule(kind, delay) {
  const entry = live.get(kind);
  if (!entry) return;
  clearTimeout(entry.timer);
  entry.timer = setTimeout(() => dismiss(kind), delay ?? HIDE_AFTER[kind] ?? 12000);
}

export function dismiss(kind) {
  const entry = live.get(kind);
  if (!entry) return;
  clearTimeout(entry.timer);
  entry.node.classList.add("leaving");
  setTimeout(() => entry.node.remove(), 260);
  live.delete(kind);
}

function closeButton(kind) {
  return el("button", { class: "sf-close", "aria-label": "Schließen", text: "×", onClick: () => dismiss(kind) });
}

/* ---------------------------------------------------------------- music */

function showMusic(receipt) {
  const evidence = receipt.evidence || {};
  const node = el("section", { class: "surface sf-music", "aria-label": "Wiedergabe" });
  const title = el("div", { class: "sf-title" });
  const sub = el("div", { class: "sf-sub" });
  const toggle = el("button", { class: "sf-btn primary", "aria-label": "Wiedergabe umschalten" });
  const render = (state) => {
    const track = state.title ? `${state.title}${state.artist ? " · " + state.artist : ""}` : (evidence.track || evidence.query || "Musik");
    title.textContent = track;
    sub.textContent = [state.app || evidence.app || "", state.status === "Playing" || state.playing ? "spielt" : state.status ? "pausiert" : ""].filter(Boolean).join(" · ");
    const playing = state.playing ?? String(state.status || evidence.status || "").toLowerCase() === "playing";
    toggle.textContent = playing ? "❚❚" : "▶";
    toggle.dataset.playing = playing ? "1" : "";
  };
  const control = async (action) => {
    const state = await api("/api/media/control", { action });
    if (state && state.ok !== false) render(state);
    schedule("music");
  };
  toggle.onclick = () => control(toggle.dataset.playing ? "pause" : "play");
  node.append(
    el("div", { class: "sf-text" }, title, sub),
    el("div", { class: "sf-controls" },
      el("button", { class: "sf-btn", "aria-label": "Vorheriger Titel", text: "⏮", onClick: () => control("previous") }),
      toggle,
      el("button", { class: "sf-btn", "aria-label": "Nächster Titel", text: "⏭", onClick: () => control("next") })),
    closeButton("music"));
  render({ status: evidence.status, app: evidence.app });
  place("music", node);
  if (receipt.ok !== false) api("/api/media/state").then((s) => { if (s && s.ok !== false) render(s); });
}

/* ---------------------------------------------------------------- calendar */

function showCalendar(receipt) {
  const node = el("section", { class: "surface sf-calendar", "aria-label": "Kalender" },
    el("div", { class: "sf-text" },
      el("div", { class: "sf-title", text: receipt.detail || "Kalender aktualisiert" }),
      el("div", { class: "sf-sub", text: receipt.verified ? "eingetragen" : receipt.ok ? "ausgeführt" : "nicht geschafft" })),
    el("button", { class: "sf-link", text: "Kalender öffnen", onClick: () => { dismiss("calendar"); views.open("calendar", {}); } }),
    closeButton("calendar"));
  place("calendar", node);
}

/* ---------------------------------------------------------------- work */

const FINISHED = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

function showWork(job) {
  const done = FINISHED.has(job.state);
  const existing = live.get("work");
  if (done && !existing) return;
  const percent = job.progress != null ? Math.round(Number(job.progress) * 100) : null;
  const node = el("section", { class: "surface sf-work" + (done ? " done" : ""), "aria-label": "Laufende Arbeit", "aria-live": "polite" },
    el("div", { class: "sf-text" },
      el("div", { class: "sf-title", text: job.title || "Arbeit" }),
      el("div", { class: "sf-sub", text: done ? (job.state === "COMPLETED" ? "fertig" : job.state === "CANCELLED" ? "abgebrochen" : "nicht geschafft")
                                             : percent != null ? `${percent} %` : "läuft" }),
      !done && percent != null ? el("div", { class: "sf-bar" }, el("i", { style: { width: `${percent}%` } })) : null),
    !done && job.cancellable ? el("button", { class: "sf-link", text: "Abbrechen", onClick: () => api("/api/jobs/cancel", { job_id: job.job_id }) }) : null);
  place("work", node, { persistent: !done });
  if (done) schedule("work");
}

export function clearAll() {
  for (const kind of [...live.keys()]) dismiss(kind);
  clear(host());
}
