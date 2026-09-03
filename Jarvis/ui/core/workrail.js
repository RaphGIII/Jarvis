/* The left dock: ACTIVE work, recent results, recent conversations.

   The owner must always know what ZEUS is doing.  A small edge tab shows the
   count of active jobs (from EventType.JOB events); opening it reveals the
   dock: running jobs with their live phase, recently completed work (image
   results as thumbnails, clickable), and the conversation archive grouped by
   day.  Restoring a conversation swaps the live transcript through
   /api/conversation/restore and re-renders it in the chat log. */

import { $, el, clear } from "./dom.js";
import { api, audioUrl } from "./api.js";
import * as bus from "./bus.js";

const jobs = new Map();           // job_id -> job dict
let dockOpen = false;
let conversations = [];
let toastFn = null;
let chatMod = null;

const store = (k, f) => { try { return JSON.parse(localStorage.getItem(k)) ?? f; } catch { return f; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

export function init({ toast, chat }) {
  toastFn = toast;
  chatMod = chat;
  dockOpen = Boolean(store("zeus.dock", false));
  const tab = el("button", { id: "dockTab", title: "Aktive Arbeit & Gespräche", onClick: toggle },
    el("span", { class: "dt-icon", text: "⚡" }), el("span", { id: "dockCount", class: "dt-count", hidden: true }));
  const dock = el("aside", { id: "leftDock", hidden: !dockOpen });
  document.body.append(tab, dock);

  bus.on("job", onJob);
  bus.on("notification", (p) => {
    if (p._replay) return;
    if (p.kind === "image" && p.file) {
      toastFn?.(p.text || "Bild fertig.", "note");
    }
  });
  refresh();
  setInterval(() => { if (jobs.size || dockOpen) render(); }, 5000);
}

async function refresh() {
  const [j, c] = await Promise.all([api("/api/jobs"), api("/api/conversations")]);
  if (j && j.active) {
    for (const job of [...(j.active || []), ...(j.recent || [])]) jobs.set(job.job_id, job);
  }
  if (c && c.conversations) conversations = c.conversations;
  render();
}

function onJob(p) {
  if (!p || !p.job_id) return;
  const known = jobs.get(p.job_id);
  jobs.set(p.job_id, p);
  if (!p._replay && p.event === "completed" && known?.state !== "COMPLETED") {
    toastFn?.(`✓ ${p.title}`, "note");
  }
  if (!p._replay && p.event === "failed" && known?.state !== "FAILED") {
    toastFn?.(`✗ ${p.title}: ${p.error || "fehlgeschlagen"}`, "warn");
  }
  render();
}

function toggle() {
  dockOpen = !dockOpen;
  save("zeus.dock", dockOpen);
  $("leftDock").hidden = !dockOpen;
  if (dockOpen) refresh();
  render();
}

function activeJobs() {
  return [...jobs.values()].filter((j) => !["COMPLETED", "FAILED", "CANCELLED"].includes(j.state))
    .sort((a, b) => a.created_at - b.created_at);
}
function doneJobs() {
  return [...jobs.values()].filter((j) => ["COMPLETED", "FAILED", "CANCELLED"].includes(j.state))
    .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0)).slice(0, 6);
}

function render() {
  const active = activeJobs();
  bus.emit("jobs:active", active.length);
  const count = $("dockCount");
  if (count) {
    count.hidden = active.length === 0;
    count.textContent = String(active.length);
  }
  $("dockTab")?.classList.toggle("working", active.length > 0);
  if (!dockOpen) return;
  const dock = $("leftDock");
  if (!dock) return;
  clear(dock);

  dock.append(el("header", {}, el("b", { text: "ZEUS ARBEITET" }),
    el("button", { class: "chip", text: "×", onClick: toggle })));

  const sec = (title) => { const s = el("section", {}, el("h5", { text: title })); dock.append(s); return s; };

  const act = sec("Aktiv");
  if (!active.length) act.append(el("div", { class: "empty", style: { padding: "2px 0" }, text: "Nichts läuft gerade." }));
  for (const job of active) {
    const row = el("div", { class: "dock-job" },
      el("div", { class: "dj-title", text: job.title }),
      el("div", { class: "dj-phase" },
        el("span", { class: "dj-pulse" }),
        el("span", { text: (job.phase || job.state.toLowerCase()) + (job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : "") })),
      job.progress != null ? el("div", { class: "dj-bar" }, el("i", { style: { width: `${Math.round(job.progress * 100)}%` } })) : null,
      job.cancellable ? el("button", { class: "chip", text: "Abbrechen", onClick: async () => {
        await api("/api/jobs/cancel", { job_id: job.job_id }); } }) : null);
    act.append(row);
  }

  const done = doneJobs();
  if (done.length) {
    const fin = sec("Zuletzt fertig");
    for (const job of done) {
      const ok = job.state === "COMPLETED";
      const row = el("div", { class: "dock-job done" },
        el("div", { class: "dj-title", text: (ok ? "✓ " : "✗ ") + job.title + (job.seconds ? ` · ${job.seconds}s` : "") }));
      if (ok && job.kind === "image" && job.result?.file) {
        row.append(el("img", { class: "dj-thumb", src: audioUrl("/api/image/file?path=" + encodeURIComponent(job.result.file)),
                               title: job.result.file, onClick: () => api("/api/fs/open", { path: job.result.file }) }));
      }
      if (!ok && job.error) row.append(el("div", { class: "empty", style: { padding: 0 }, text: job.error.slice(0, 90) }));
      fin.append(row);
    }
  }

  const conv = sec("Gespräche");
  conv.append(el("button", { class: "chip", text: "+ Neues Gespräch", onClick: async () => {
    await api("/api/new", {});
    chatMod?.reset();
    refresh();
  } }));
  if (!conversations.length) conv.append(el("div", { class: "empty", style: { padding: "2px 0" }, text: "Noch keine archivierten Gespräche." }));
  const today = new Date().toISOString().slice(0, 10);
  const yesterday = new Date(Date.now() - 864e5).toISOString().slice(0, 10);
  let lastGroup = "";
  for (const c of conversations) {
    const day = String(c.at || "").slice(0, 10);
    const group = day === today ? "Heute" : day === yesterday ? "Gestern" : "Früher";
    if (group !== lastGroup) { conv.append(el("div", { class: "dock-group", text: group })); lastGroup = group; }
    conv.append(el("div", { class: "dock-conv", title: c.summary || "", onClick: () => restoreConversation(c.id) },
      el("span", { class: "dc-title", text: c.title || "(ohne Titel)" }),
      el("span", { class: "dc-meta", text: `${c.turn_count || "?"} Beiträge` })));
  }
}

async function restoreConversation(id) {
  const out = await api("/api/conversation/restore", { id });
  if (out.ok === false) { toastFn?.(out.error || "Konnte das Gespräch nicht laden.", "warn"); return; }
  const views = await import("./views.js");
  views.close({ push: false });
  chatMod?.reset();
  for (const turn of out.turns || []) {
    chatMod?.addTurn(turn.role === "user" ? "user" : "jarvis",
                     turn.role === "user" ? "DU" : (window.ASSISTANT_NAME || "ZEUS"),
                     turn.text, { _replay: true });
  }
  toastFn?.(`Gespräch „${out.title || id}“ wiederhergestellt.`, "note");
  refresh();
}
