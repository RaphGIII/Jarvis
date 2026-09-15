/* The work surface: ZEUS never looks frozen.

   A pill in the top bar says what ZEUS is doing right now (idle, one job,
   waiting for the owner); opening it shows the running work -- jobs and
   long missions -- in owner terms: what, how far, waiting for what, done.
   Never implementation logs, never phases spelled in engineering vocabulary.
   Data: /api/jobs and /api/missions, kept live by job and progress events. */

import { $, el, clear } from "./dom.js";
import { api, audioUrl } from "./api.js";
import * as bus from "./bus.js";
import * as views from "./views.js";

const jobs = new Map();
let missions = [];
let open = false;
let toastFn = null;

const PHASE_WORDS = {
  QUEUED: "wartet", UNDERSTANDING: "versteht die Aufgabe", PLANNING: "plant", WAITING_FOR_RESOURCE: "wartet auf Ressourcen",
  EXECUTING: "arbeitet", VERIFYING: "prüft das Ergebnis", FINALIZING: "schließt ab", COMPLETED: "fertig", FAILED: "fehlgeschlagen", CANCELLED: "abgebrochen",
  UNDERSTAND: "versteht die Aufgabe", INVESTIGATE: "untersucht", SURVEY: "sichtet", ENGINEER: "entwickelt", BUILD: "baut", ESCALATE: "holt Verstärkung",
  VERIFY: "prüft", PROMOTE: "übernimmt", RESTARTING: "startet neu", DONE: "fertig", CREATED: "angelegt", PLAN: "plant", EXECUTE: "arbeitet",
  DIAGNOSE: "analysiert", COMPLETE: "fertig", SPECIFY: "spezifiziert", AWAITING_AUTHORIZATION: "wartet auf deine Freigabe",
  AWAITING_BUILD: "wartet auf den Build", WAITING: "wartet",
};

function phaseWord(raw) {
  const key = String(raw || "").toUpperCase();
  return PHASE_WORDS[key] || String(raw || "").toLowerCase().replace(/_/g, " ");
}

export function init({ toast }) {
  toastFn = toast;
  $("workPill").onclick = toggle;
  bus.on("job", onJob);
  bus.on("progress", () => refreshMissions());
  bus.on("notification", (p) => {
    if (p._replay) return;
    if (p.kind === "image" && p.file) toastFn?.(p.text || "Bild fertig.", "note");
    if (/selfdev|mission|release|relaunch|restart/.test(p.kind || "")) refreshMissions();
  });
  refresh();
  setInterval(() => { if (jobs.size || open || missions.length) refresh(); }, 6000);
}

async function refresh() {
  const [j, m] = await Promise.all([api("/api/jobs"), api("/api/missions", { status: "active" })]);
  if (j && j.active) for (const job of [...(j.active || []), ...(j.recent || [])]) jobs.set(job.job_id, job);
  if (m && m.missions) missions = m.missions;
  render();
}

async function refreshMissions() {
  const m = await api("/api/missions", { status: "active" });
  if (m && m.missions) missions = m.missions;
  render();
}

function onJob(p) {
  if (!p || !p.job_id) return;
  const known = jobs.get(p.job_id);
  jobs.set(p.job_id, p);
  if (!p._replay && p.event === "completed" && known?.state !== "COMPLETED") toastFn?.(`Fertig: ${p.title}`, "good");
  if (!p._replay && p.event === "failed" && known?.state !== "FAILED") toastFn?.(`Nicht geschafft: ${p.title}`, "warn");
  render();
}

function toggle() {
  open = !open;
  $("workPanel").hidden = !open;
  if (open) refresh(); else render();
}

function activeJobs() {
  return [...jobs.values()].filter((j) => !["COMPLETED", "FAILED", "CANCELLED"].includes(j.state)).sort((a, b) => a.created_at - b.created_at);
}
function doneJobs() {
  return [...jobs.values()].filter((j) => ["COMPLETED", "FAILED", "CANCELLED"].includes(j.state)).sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0)).slice(0, 5);
}
function activeMissions() {
  return missions.filter((m) => !m.finished && !["completed", "failed", "cancelled"].includes(m.state));
}

function render() {
  const active = activeJobs();
  const live = activeMissions();
  const waiting = live.filter((m) => m.owner_input_required || m.state === "waiting" || m.state === "blocked");
  bus.emit("jobs:active", active.length + live.length);
  const pill = $("workPill");
  if (pill) {
    const total = active.length + live.length;
    pill.classList.toggle("busy", total > 0 && !waiting.length);
    pill.classList.toggle("attention", waiting.length > 0);
    const txt = waiting.length ? "Wartet auf dich" : total === 0 ? "Bereit" : total === 1 ? (active[0]?.title || live[0]?.title || live[0]?.goal || "Arbeitet") : "Arbeitet";
    pill.querySelector(".txt").textContent = String(txt).slice(0, 48);
    pill.querySelector(".n").textContent = total > 1 ? String(total) : "";
  }
  if (!open) return;
  const panel = $("workPanel");
  clear(panel);
  panel.append(el("header", {}, el("b", { text: "ZEUS ARBEITET" }), el("button", { class: "chip", text: "×", onClick: toggle })));
  const sec = (title) => { const s = el("section", {}, el("h5", { text: title })); panel.append(s); return s; };

  const act = sec("Gerade");
  if (!active.length && !live.length) act.append(el("div", { class: "note", text: "Nichts läuft gerade." }));
  for (const m of live) {
    const wait = m.owner_input_required || m.state === "waiting" || m.state === "blocked";
    const row = el("div", { class: "wk" },
      el("div", { class: "wk-title", text: m.title || m.goal || "Arbeit" }),
      el("div", { class: "wk-phase" }, el("span", { class: "wk-dot" + (wait ? " wait" : "") }),
        el("span", { text: wait ? (m.next_action || "wartet auf deine Freigabe") : phaseWord(m.phase) })));
    if (m.tasks && m.tasks.total) row.append(el("div", { class: "wk-bar" }, el("i", { style: { width: `${Math.round(100 * (m.tasks.done || 0) / m.tasks.total)}%` } })));
    row.append(el("div", { class: "wk-actions" }, el("button", { class: "chip", text: wait ? "Freigeben" : "Öffnen", onClick: () => { toggle(); views.open("missions", { focus: m.id }); } })));
    act.append(row);
  }
  for (const job of active) {
    const row = el("div", { class: "wk" },
      el("div", { class: "wk-title", text: job.title }),
      el("div", { class: "wk-phase" }, el("span", { class: "wk-dot" }),
        el("span", { text: phaseWord(job.phase || job.state) + (job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : "") })),
      job.progress != null ? el("div", { class: "wk-bar" }, el("i", { style: { width: `${Math.round(job.progress * 100)}%` } })) : null,
      job.cancellable ? el("div", { class: "wk-actions" }, el("button", { class: "chip", text: "Abbrechen", onClick: async () => { await api("/api/jobs/cancel", { job_id: job.job_id }); } })) : null);
    act.append(row);
  }

  const done = doneJobs();
  if (done.length) {
    const fin = sec("Zuletzt");
    for (const job of done) {
      const ok = job.state === "COMPLETED";
      const row = el("div", { class: "wk done" }, el("div", { class: "wk-title", text: (ok ? "✓ " : "✗ ") + job.title + (job.seconds ? ` · ${job.seconds}s` : "") }));
      if (ok && job.kind === "image" && job.result?.file) {
        row.append(el("img", { class: "wk-thumb", src: audioUrl("/api/image/file?path=" + encodeURIComponent(job.result.file)), title: job.result.file,
                               onClick: () => api("/api/fs/open", { path: job.result.file }) }));
      }
      if (!ok && job.error) row.append(el("div", { class: "note", text: "Details unter Fortschritt." }));
      fin.append(row);
    }
  }
  panel.append(el("div", { class: "wk-actions" }, el("button", { class: "chip", text: "Alle Arbeiten", onClick: () => { toggle(); views.open("missions"); } }),
    el("button", { class: "chip", text: "Fortschritt", onClick: () => { toggle(); views.open("activity"); } })));
}
