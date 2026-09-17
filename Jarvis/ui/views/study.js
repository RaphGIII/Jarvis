/* Studium: ask about the material, get the place.

   One search line.  Results are places in the owner's material, ranked by how
   well they answer, never by file type -- each one opens the viewer at that
   page, slide or section with every match marked.  A question ("Was steht zu
   Aldosteron in meinen Notizen?") can go to ZEUS, which answers from those
   places and cites them.  Below: the few materials used last.  To the right,
   collapsible: all material by subject, with the categories ZEUS inferred and
   the owner can correct, and the connected sources.

   What ZEUS is doing with added material shows as one quiet line per file:
   reading, text recognition, indexing, preparing semantic search, drawing
   slides -- and, when something did not work, why.

   With a document in the parameters (doc, unit, focus) the view is the viewer. */

import { el, clear } from "../core/dom.js";
import { api, postBytes } from "../core/api.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";
import { sourceRef, viewParams } from "../core/sources.js";
import { mountViewer, unmountViewer } from "./study_viewer.js";
import { lineFor, jobRows } from "../core/indexing.js";

const QUESTION = /^(was|wie|warum|wieso|weshalb|erkl[äa]r|fass|vergleich|welche[rs]?\b.*\?|pr[üu]f)/i;
const TYPE_WORD = { pdf: "PDF", docx: "Word", pptx: "Folien", markdown: "Notiz", text: "Text", image: "Bild" };
const TYPE_FILTER = [["pdf", "PDF"], ["pptx", "Folien"], ["docx", "Word"], ["notes", "Notizen"]];
let offEvents = [];
let pollTimer = 0;
let indexingLine = null;   // the mounted workspace's indexing line

export const view = {
  id: "study",
  title: "Studium",
  async mount(pane, params) {
    stop();
    if (params.doc) {
      await mountViewer(pane, params);
      return;
    }
    await mountWorkspace(pane, params);
  },
  unmount() {
    stop();
    unmountViewer();
  },
};

function stop() {
  for (const off of offEvents) off();
  offEvents = [];
  clearInterval(pollTimer);
  pollTimer = 0;
  indexingLine?.dispose();
  indexingLine = null;
  hideGalaxy();
}

export function unitsLabel(doc) {
  const n = Number(doc.units) || 0;
  if (doc.source_type === "pptx") return n === 1 ? "1 Folie" : `${n} Folien`;
  if (["pdf", "image"].includes(doc.source_type)) return n === 1 ? "1 Seite" : `${n} Seiten`;
  return n === 1 ? "1 Abschnitt" : `${n} Abschnitte`;
}

async function mountWorkspace(pane, params) {
  const results = el("section", { class: "st-results", "aria-live": "polite" });
  const recent = el("section", { class: "st-recent" });
  const status = el("div", { class: "st-status", role: "status" });
  const activity = el("div", { class: "st-activity", "aria-live": "polite" });
  const library = el("aside", { class: "st-library", id: "studyLibrary", "aria-label": "Material" });
  const input = el("input", { class: "st-input", type: "search", placeholder: "Frag ZEUS zu deinen Unterlagen …", value: params.q || "",
                              "aria-label": "Frag ZEUS zu deinen Unterlagen", autocomplete: "off", spellcheck: "false" });
  const addMenu = el("div", { class: "st-add-menu", hidden: true, role: "menu" });
  const filePicker = el("input", { type: "file", multiple: true, hidden: true, accept: ".pdf,.docx,.pptx,.md,.markdown,.txt,.png,.jpg,.jpeg,.webp,.tif,.tiff,.goodnotes",
                                   onChange: (e) => { const files = [...e.target.files]; e.target.value = ""; upload(files, activity); } });
  const addButton = el("button", { class: "st-add", "aria-haspopup": "menu", "aria-expanded": "false", text: "Material hinzufügen",
                                   onClick: () => toggleAdd(addMenu, addButton, filePicker, activity, refresh) });
  const libraryToggle = el("button", { class: "st-lib-toggle", "aria-controls": "studyLibrary", "aria-expanded": "true", text: "Material",
    onClick: () => {
      const open = !root.classList.toggle("library-closed");
      libraryToggle.setAttribute("aria-expanded", String(open));
      try { localStorage.setItem("zeus.study.library", open ? "open" : "closed"); } catch { /* private window */ }
    } });
  const filters = { type: "", subject: "" };
  const form = el("form", { class: "st-search", role: "search", onSubmit: (e) => { e.preventDefault(); if (!searchInSky(root, input.value)) search(input.value, results, filters, library); } },
    el("span", { class: "st-search-icon", "aria-hidden": "true", html: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/></svg>' }),
    input,
    el("button", { class: "st-go", type: "submit", "aria-label": "Suchen", text: "↵" }));
  const main = el("div", { class: "st-main" },
    el("header", { class: "st-head" }, el("h1", { class: "st-title", text: "Studium" }), el("div", { class: "st-head-tools" }, addButton, libraryToggle, addMenu)),
    form, status, activity, results, recent, filePicker);
  const root = el("div", { class: "study" }, main, library);
  // Beside the page on a wide window; on a narrow one it would cover the page, so it waits to be asked for.
  let libraryOpen = !window.matchMedia("(max-width: 1180px)").matches;
  try { const stored = localStorage.getItem("zeus.study.library"); if (stored === "closed" || (stored === "open" && libraryOpen)) libraryOpen = stored === "open"; } catch { /* ignore */ }
  if (!libraryOpen) { root.classList.add("library-closed"); libraryToggle.setAttribute("aria-expanded", "false"); }
  mountViewSwitch(root, main, { input, activity, runListSearch: (text) => search(text, results, filters, library) });
  pane.append(root);

  // files dropped anywhere on the workspace become material
  root.addEventListener("dragover", (e) => { e.preventDefault(); root.classList.add("dropping"); });
  root.addEventListener("dragleave", (e) => { if (e.target === root) root.classList.remove("dropping"); });
  root.addEventListener("drop", (e) => { e.preventDefault(); root.classList.remove("dropping"); upload([...e.dataTransfer.files], activity); });

  let docsCache = [];
  const refresh = async () => {
    const [docs, state] = await Promise.all([api("/api/study/documents", { limit: 400, order: "recent" }), api("/api/study/status")]);
    docsCache = (docs && docs.documents) || [];
    renderRecent(recent, docsCache);
    renderLibrary(library, docsCache, (state && state.sources) || [], refresh);
    renderStatus(status, state);
    if (state && Object.keys(state.scanning || {}).length && !pollTimer) pollTimer = setInterval(refresh, 2500);
    if (state && !Object.keys(state.scanning || {}).length && pollTimer) { clearInterval(pollTimer); pollTimer = 0; }
  };
  root.libraryDocs = () => docsCache;
  offEvents.push(bus.on("tool", (p) => { if (!p._replay && p.source === "study") refresh(); }));
  indexingLine = mountIndexing(activity, () => refreshSoon(refresh));
  const jobStates = new Map();
  offEvents.push(bus.on("progress", (p) => {
    if (p._replay || p.source !== "study" || !p.study) return;
    indexingLine?.soon();
    // a document appears in the list as soon as it is being read, and changes when it is done
    const job = p.study;
    const mark = `${job.state}:${job.phase === "queued" ? "queued" : "started"}`;
    if (job.job_id && jobStates.get(job.job_id) !== mark) { jobStates.set(job.job_id, mark); refreshSoon(refresh); }
  }));
  offEvents.push(bus.on("study:changed", refresh));
  await refresh();
  if (params.q) search(params.q, results, filters, library);
  setTimeout(() => input.focus(), 30);
}

let refreshTimer = 0;
function refreshSoon(refresh) {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refresh, 400);
}

/* ---------------------------------------------------------------- indexing line */

/* One thin line under the search while ZEUS reads material: the real percentage from the persisted jobs, what is being read
   now.  Click: the compact job list.  It survives navigation and restarts because the server keeps the jobs; the line only
   shows them.  When everything is done it says so for a moment and folds away; a file that could not be read stays until
   it is dismissed. */
function mountIndexing(box, onFinished) {
  const text = el("span", { class: "st-ix-text" });
  const detail = el("span", { class: "st-ix-detail" });
  const line = el("button", { class: "st-ix", type: "button", "aria-expanded": "false", hidden: true, onClick: () => { ctl.open = !ctl.open; paint(); } },
    text, detail, el("span", { class: "st-ix-bar", "aria-hidden": "true" }, el("i")));
  const list = el("div", { class: "st-ix-list", hidden: true });
  box.append(line, list);
  let dismissed = new Set();
  try { dismissed = new Set(JSON.parse(localStorage.getItem("zeus.study.dismissed") || "[]")); } catch { /* private window */ }
  const ctl = { summary: null, uploading: 0, local: [], open: false, doneAt: 0, wasActive: false, fetching: false, again: false, timer: 0, soonTimer: 0, alive: true };

  const dismiss = (id) => {
    // dismissing a failure dismisses every job of that file failing the same way
    const gone = failures().find((j) => j.job_id === id);
    for (const j of [...ctl.local, ...(((ctl.summary && ctl.summary.failed) || []))]) if (gone && j.name === gone.name && j.reason === gone.reason) dismissed.add(j.job_id);
    dismissed.add(id);
    ctl.local = ctl.local.filter((j) => j.job_id !== id);
    try { localStorage.setItem("zeus.study.dismissed", JSON.stringify([...dismissed].slice(-60))); } catch { /* private window */ }
    paint();
  };
  const failures = () => {
    const seen = new Set();
    return [...ctl.local, ...(((ctl.summary && ctl.summary.failed) || []))].filter((j) => {
      const key = `${j.name}:${j.reason}`;
      if (dismissed.has(j.job_id) || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  function paint() {
    if (!ctl.alive) return;
    let shown = lineFor(ctl.summary, { uploading: ctl.uploading });
    if (shown && shown.done) {
      ctl.doneAt = ctl.doneAt || Date.now();
      if (Date.now() - ctl.doneAt > 5000 && !ctl.open) shown = null;
      else setTimeout(paint, 5200);
    } else if (shown && shown.busy) ctl.doneAt = 0;
    const failed = failures();
    if (!shown && failed.length) shown = { busy: false, failed: failed.length, text: `${failed.length === 1 ? "1 Datei" : `${failed.length} Dateien`} nicht gelesen`, detail: "Warum?", fraction: null };
    box.classList.toggle("on", Boolean(shown));
    line.hidden = !shown;
    if (!shown) { ctl.open = false; list.hidden = true; return; }
    line.classList.toggle("busy", Boolean(shown.busy));
    line.classList.toggle("failed", !shown.busy && Boolean(shown.failed));
    line.classList.toggle("measured", shown.fraction != null);
    line.setAttribute("aria-expanded", String(ctl.open));
    text.textContent = shown.text;
    detail.textContent = shown.detail || "";
    line.querySelector(".st-ix-bar i").style.width = shown.fraction == null ? "" : `${Math.round(shown.fraction * 100)}%`;
    list.hidden = !ctl.open;
    if (ctl.open) renderList();
  }

  function renderList() {
    clear(list);
    const summary = { ...(ctl.summary || {}), failed: failures() };
    const items = jobRows(summary, dismissed);
    if (!items.length) { list.append(el("div", { class: "st-ix-row quiet", text: "Nichts in Arbeit." })); return; }
    for (const item of items.slice(0, 30)) {
      list.append(el("div", { class: "st-ix-row", "data-state": item.state },
        el("span", { class: "st-ix-name", text: item.name, title: item.name }),
        el("span", { class: "st-ix-words", text: item.words }),
        el("span", { class: "st-ix-pct", text: item.state === "running" ? `${Math.round(item.fraction * 100)} %` : "" }),
        item.cancellable
          ? el("button", { class: "st-link", type: "button", text: "Abbrechen", onClick: async () => { await api("/api/study/jobs/cancel", { job_id: item.id }); fetchNow(); } })
          : item.state === "failed"
            ? el("button", { class: "st-ix-close", type: "button", "aria-label": "Ausblenden", text: "×", onClick: () => dismiss(item.id) })
            : el("span")));
    }
  }

  async function fetchNow() {
    if (!ctl.alive) return;
    if (ctl.fetching) { ctl.again = true; return; }
    ctl.fetching = true;
    const summary = await api("/api/study/indexing");
    ctl.fetching = false;
    if (!ctl.alive) return;
    if (summary && summary.ok !== false) {
      if (ctl.wasActive && !summary.active) onFinished();
      ctl.wasActive = Boolean(summary.active);
      ctl.summary = summary;
    }
    paint();
    clearTimeout(ctl.timer);
    if (ctl.again) { ctl.again = false; ctl.timer = setTimeout(fetchNow, 300); }
    else if ((summary && summary.active) || ctl.uploading) ctl.timer = setTimeout(fetchNow, 2000);
    else if (summary && summary.documents_total) ctl.timer = setTimeout(fetchNow, 6000);
  }

  fetchNow();
  return {
    soon() { if (!ctl.soonTimer) ctl.soonTimer = setTimeout(() => { ctl.soonTimer = 0; fetchNow(); }, 450); },
    uploading(delta) { ctl.uploading = Math.max(0, ctl.uploading + delta); paint(); if (delta < 0) fetchNow(); },
    refused(name, result) {
      ctl.local.push({ job_id: `local-${name}-${Date.now()}`, name, state: "failed", reason: result && result.reason, created_at: Date.now() / 1000,
                       error: result && result.transport ? "ZEUS ist gerade nicht erreichbar." : (result && result.error) || "nicht angenommen" });
      paint();
    },
    dispose() { ctl.alive = false; clearTimeout(ctl.timer); clearTimeout(ctl.soonTimer); },
  };
}

/* ---------------------------------------------------------------- search */

async function search(query, box, filters, library) {
  const text = String(query || "").trim();
  clear(box);
  if (!text) return;
  box.append(el("div", { class: "st-searching", role: "status" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: "Sucht in deinen Unterlagen …" })));
  const body = { query: text, limit: 8 };
  const chosen = {};
  if (filters.type === "notes") chosen.goodnotes = true;
  else if (filters.type) chosen.source_type = [filters.type];
  if (filters.subject) chosen.subject = filters.subject;
  if (Object.keys(chosen).length) body.filters = chosen;
  const found = await api("/api/study/search", body);
  clear(box);
  if (!found || found.ok === false) {
    box.append(el("div", { class: "st-none", text: found && found.transport ? "ZEUS ist gerade nicht erreichbar." : "Die Suche hat gerade nicht geklappt." }));
    return;
  }
  const items = found.results || [];
  const topic = found.topic || text;
  box.append(el("div", { class: "st-topic" }, el("span", { text: topic }),
    QUESTION.test(text) ? el("button", { class: "st-ask", text: "ZEUS antworten lassen",
      onClick: async () => { const chat = await import("./chat.js"); chat.send(text); } }) : null));
  box.append(filterRow(filters, library, () => search(text, box, filters, library)));
  if (!items.length) {
    const filtered = filters.type || filters.subject;
    box.append(el("div", { class: "st-none", text: filtered ? "Mit diesem Filter nichts gefunden." : "In deinen Unterlagen steht dazu nichts – oder das Material ist noch nicht hier." }));
    return;
  }
  const list = el("div", { class: "st-list", role: "list" });
  for (const result of items) {
    const doc = result.document;
    const best = result.best;
    const compact = { document_id: doc.id, title: doc.title, filename: doc.filename, source_type: doc.source_type, goodnotes: Boolean((doc.flags || {}).goodnotes),
                      location: best.location, excerpt: best.excerpt, highlights: best.highlights, focus: best.focus || "", phrases: best.phrases || [],
                      at: best.at, match: best.match, certainty: best.certainty, handwriting: best.handwriting, figure: best.figure };
    const ref = sourceRef(compact);
    ref.style.setProperty("--relevance", String(result.relevance));
    if ((result.more || []).length) {
      ref.append(el("div", { class: "src-more" }, el("span", { text: "Auch:" }),
        ...result.more.map((hit) => el("button", { class: "src-more-link", text: hit.location.label,
          onClick: () => views.open("study", viewParams({ ...compact, location: hit.location, focus: hit.focus || "", phrases: hit.phrases || [], at: hit.at, figure: hit.figure })) }))));
    }
    list.append(ref);
  }
  box.append(list);
}

function filterRow(filters, library, rerun) {
  const docs = (library.closest(".study")?.libraryDocs?.() || []);
  const subjects = [...new Set(docs.map((d) => d.subject).filter(Boolean))].sort();
  const types = new Set(docs.map((d) => ((d.flags || {}).goodnotes ? "notes" : d.source_type)));
  const typeChoices = TYPE_FILTER.filter(([key]) => types.has(key));
  if (typeChoices.length < 2 && subjects.length < 2) return el("span");
  const chip = (label, on, run) => el("button", { class: "st-filter" + (on ? " on" : ""), "aria-pressed": String(on), text: label, onClick: run });
  const row = el("div", { class: "st-filters", role: "group", "aria-label": "Filter" });
  if (typeChoices.length >= 2) {
    row.append(chip("Alle", !filters.type, () => { filters.type = ""; rerun(); }),
      ...typeChoices.map(([key, label]) => chip(label, filters.type === key, () => { filters.type = filters.type === key ? "" : key; rerun(); })));
  }
  if (subjects.length >= 2) {
    row.append(el("span", { class: "st-filter-sep", "aria-hidden": "true" }),
      ...subjects.map((s) => chip(s, filters.subject === s, () => { filters.subject = filters.subject === s ? "" : s; rerun(); })));
  }
  return row;
}

/* ---------------------------------------------------------------- lists */

function docRow(doc, { detail = true } = {}) {
  const kind = (doc.flags || {}).goodnotes ? "GoodNotes" : TYPE_WORD[doc.source_type] || "";
  return el("button", { class: "st-doc", onClick: () => views.open("study", { doc: doc.id, unit: "1" }) },
    el("span", { class: "st-doc-name", text: doc.title || doc.filename }),
    detail ? el("span", { class: "st-doc-meta", text: [doc.subject, kind, unitsLabel(doc)].filter(Boolean).join(" · ") }) : null,
    doc.status === "processing" ? el("span", { class: "st-doc-busy", text: "wird eingelesen" })
      : (doc.flags || {}).missing ? el("span", { class: "st-doc-warn", text: "Datei fehlt" })
        : doc.status === "no_text" ? el("span", { class: "st-doc-warn", text: "ohne Text" }) : null);
}

function renderRecent(box, docs) {
  clear(box);
  if (!docs.length) {
    box.append(el("div", { class: "st-empty" },
      el("p", { class: "st-empty-lead", text: "Noch kein Material." }),
      el("p", { text: "Füge Skripte, Vorlesungsfolien, Word-Dokumente, Notizen oder GoodNotes-Exporte hinzu. ZEUS liest sie Seite für Seite und findet danach jede Stelle wieder." })));
    return;
  }
  const opened = docs.filter((d) => d.opened_at).slice(0, 4);
  const shown = opened.length ? opened : docs.slice(0, 4);
  box.append(el("h2", { class: "st-label", text: opened.length ? "Zuletzt geöffnet" : "Zuletzt hinzugefügt" }),
    el("div", { class: "st-docs" }, ...shown.map((d) => docRow(d))));
}

function renderStatus(box, state) {
  clear(box);
  if (!state || state.ok === false) return;
  const scanning = Object.values(state.scanning || {});
  if (scanning.length) {
    const s = scanning[0];
    box.append(el("span", { class: "st-busy" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), `Liest Material … ${s.indexed || 0} neu, ${s.unchanged || 0} unverändert`));
  }
  const documents = (state.stats || {}).documents || 0;
  if (state.ocr && !state.ocr.available && documents) box.append(el("span", { class: "st-hint", text: state.ocr.note }));
  else if (state.ocr && state.ocr.note && documents) box.append(el("span", { class: "st-hint", text: state.ocr.note }));
  const semantic = state.embeddings || {};
  if (documents && semantic.provider && semantic.provider !== "none" && semantic.available === false && semantic.reason) {
    box.append(el("span", { class: "st-hint", text: `Semantische Suche nicht verfügbar (${semantic.reason}) – Stichwortsuche funktioniert.` }));
  }
  // semantic preparation is an embedding job: the indexing line under the search shows it, one line, not two
}

const CATEGORY_FIELDS = [["course", "Studiengang"], ["subject", "Fach"], ["semester", "Semester"], ["module", "Modul"], ["topic", "Thema"]];

function renderLibrary(box, docs, sources, refresh) {
  clear(box);
  box.append(el("header", { class: "st-lib-head" }, el("h2", { text: "Material" }), el("span", { class: "st-count", text: String(docs.length) })));
  if (!docs.length) { box.append(el("p", { class: "st-lib-empty", text: "Hier erscheint dein Material, geordnet nach Fach." })); }
  const groups = new Map();
  for (const doc of docs) {
    const key = [doc.course, doc.subject].filter(Boolean).join(" › ") || "Ohne Zuordnung";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(doc);
  }
  for (const [key, items] of [...groups.entries()].sort((a, b) => (a[0] === "Ohne Zuordnung") - (b[0] === "Ohne Zuordnung") || a[0].localeCompare(b[0]))) {
    const group = el("details", { class: "st-group", open: groups.size <= 4 }, el("summary", {}, el("span", { text: key }), el("span", { class: "st-count", text: String(items.length) })));
    for (const doc of items.sort((a, b) => String(a.title).localeCompare(String(b.title)))) {
      const row = el("div", { class: "st-lib-row" }, docRow(doc, { detail: false }),
        el("button", { class: "st-edit", "aria-label": `Zuordnung von ${doc.title} ändern`, text: "···", onClick: () => editCategories(row, doc, refresh) }));
      group.append(row);
    }
    box.append(group);
  }
  if (sources.length) {
    box.append(el("h3", { class: "st-lib-sub", text: "Verbundene Quellen" }));
    for (const source of sources) {
      box.append(el("div", { class: "st-source" },
        el("span", { class: "st-source-name", text: source.label || source.path, title: source.path }),
        el("span", { class: "st-doc-meta", text: `${source.provider === "goodnotes" ? "GoodNotes · " : source.provider === "zeus_files" ? "ZEUS · " : ""}${source.files || 0} Dateien` }),
        el("button", { class: "st-link", text: "Neu einlesen", onClick: async () => { await api("/api/study/scan", { source_id: source.id }); refresh(); } }),
        el("button", { class: "st-link", text: "Trennen", onClick: async () => {
          if (!confirm(`„${source.label || source.path}“ trennen? Das Material bleibt im Studium.`)) return;
          await api("/api/study/source/remove", { id: source.id }); refresh();
        } })));
    }
  }
}

function editCategories(row, doc, refresh) {
  row.parentElement.querySelector(".st-cat-form")?.remove();
  const inputs = {};
  const form = el("form", { class: "st-cat-form", onSubmit: async (e) => {
    e.preventDefault();
    const changes = {};
    for (const [key, input] of Object.entries(inputs)) if (input.value.trim() !== String(doc[key] || "")) changes[key] = input.value.trim();
    if (Object.keys(changes).length) await api("/api/study/categories", { id: doc.id, changes });
    refresh();
  } });
  for (const [key, label] of CATEGORY_FIELDS) {
    const input = el("input", { value: doc[key] || "", placeholder: label, "aria-label": label });
    inputs[key] = input;
    form.append(el("label", {}, el("span", { text: label + ((doc.confirmed || []).includes(key) ? "" : doc[key] ? " (vermutet)" : "") }), input));
  }
  form.append(el("div", { class: "st-cat-actions" },
    el("button", { class: "st-link danger", type: "button", text: "Entfernen …", onClick: () => removeChoice(form, doc, refresh) }),
    el("button", { class: "st-link", type: "button", text: "Abbrechen", onClick: () => form.remove() }),
    el("button", { class: "st-save", type: "submit", text: "Speichern" })));
  row.after(form);
  inputs.subject.focus();
}

/* Removing is two different things, and neither happens by accident: out of Studium (the file stays where it is),
   or also the file -- which the server only does for a file ZEUS itself stored, never one in a connected folder. */
function removeChoice(form, doc, refresh) {
  form.querySelector(".st-remove")?.remove();
  const note = el("p", { class: "st-remove-note", text: "„Aus Studium entfernen“ nimmt das Dokument aus Suche und Liste. Die Datei bleibt, wo sie ist." });
  const panel = el("div", { class: "st-remove", role: "group", "aria-label": "Entfernen" }, note,
    el("div", { class: "st-cat-actions" },
      el("button", { class: "st-link", type: "button", text: "Abbrechen", onClick: () => panel.remove() }),
      el("button", { class: "st-link danger", type: "button", text: "Auch die Datei löschen …", onClick: (e) => {
        note.textContent = `Die Datei „${doc.filename}“ wird endgültig gelöscht. Dateien aus verbundenen Ordnern bleiben immer erhalten.`;
        e.currentTarget.replaceWith(el("button", { class: "st-save danger", type: "button", text: "Datei endgültig löschen", onClick: async () => {
          const result = await api("/api/study/delete", { id: doc.id, delete_file: true });
          if (result && result.file_note) window.zeus?.toast?.(result.file_note, "note");
          refresh();
        } }));
      } }),
      el("button", { class: "st-save", type: "button", text: "Aus Studium entfernen", onClick: async () => { await api("/api/study/delete", { id: doc.id }); refresh(); } })));
  form.append(panel);
}

/* ---------------------------------------------------------------- adding material */

async function upload(files, activity) {
  // each file is handed over, then indexed in the background; the line under the search shows the real progress
  indexingLine?.uploading(files.length);
  for (const file of files) {
    const result = await postBytes(`/api/study/upload?wait=0&name=${encodeURIComponent(file.name)}`, await file.arrayBuffer());
    indexingLine?.uploading(-1);
    if ((!result || result.ok === false) && !(result && result.job_id)) indexingLine?.refused(file.name, result);
    if (result && result.duplicate) window.zeus?.toast?.(`„${file.name}“ ist schon im Studium.`, "note");
  }
  bus.emit("study:changed", {});
}

function toggleAdd(menu, button, picker, activity, refresh) {
  const opening = menu.hidden;
  menu.hidden = !opening;
  button.setAttribute("aria-expanded", String(opening));
  if (!opening) return;
  clear(menu);
  const pane = el("div", { class: "st-add-pane" });
  const close = () => { menu.hidden = true; button.setAttribute("aria-expanded", "false"); };
  menu.append(
    el("button", { class: "st-add-item", role: "menuitem", onClick: () => { close(); picker.click(); } },
      el("span", { text: "Datei hochladen" }), el("small", { text: "PDF, Word, PowerPoint, Markdown, Text, Bilder" })),
    el("button", { class: "st-add-item", role: "menuitem", onClick: () => folderForm(pane, "folder", close, refresh) },
      el("span", { text: "Ordner verbinden" }), el("small", { text: "wird eingelesen und bei „Neu einlesen“ aktualisiert" })),
    el("button", { class: "st-add-item", role: "menuitem", onClick: async () => { close(); await api("/api/study/import_zeus", {}); refresh(); } },
      el("span", { text: "ZEUS-Dateien importieren" }), el("small", { text: "alles Lesbare aus deiner ZEUS-Bibliothek" })),
    el("button", { class: "st-add-item", role: "menuitem", onClick: () => goodnotesForm(pane, picker, close, refresh) },
      el("span", { text: "GoodNotes-Export" }), el("small", { text: "PDF-Export oder Backup-Ordner" })),
    pane);
}

function folderForm(pane, provider, close, refresh) {
  clear(pane);
  const input = el("input", { class: "st-path", placeholder: provider === "goodnotes" ? "z. B. C:\\Users\\…\\OneDrive\\GoodNotes" : "z. B. D:\\Uni\\Semester 3", "aria-label": "Ordnerpfad" });
  pane.append(el("form", { class: "st-path-form", onSubmit: async (e) => {
    e.preventDefault();
    if (!input.value.trim()) return;
    const result = await api("/api/study/connect", { path: input.value.trim(), provider });
    if (result && result.ok) { close(); refresh(); } else pane.append(el("div", { class: "st-fail", text: result?.error || "Ordner nicht gefunden" }));
  } }, input, el("button", { class: "st-save", type: "submit", text: "Verbinden" })));
  input.focus();
}

async function goodnotesForm(pane, picker, close, refresh) {
  clear(pane);
  pane.append(el("p", { class: "st-note", text: "GoodNotes bietet keinen direkten Zugriff auf Notizbücher. ZEUS liest deine Notizbücher als PDF: exportiert oder automatisch als PDF gesichert (GoodNotes › Einstellungen › Automatisches Backup, Format PDF) in einen synchronisierten Ordner." }));
  const found = await api("/api/study/goodnotes");
  for (const folder of (found && found.discovered) || []) {
    pane.append(el("div", { class: "st-source" }, el("span", { class: "st-source-name", text: folder.path }),
      el("span", { class: "st-doc-meta", text: `${folder.pdfs} PDF${folder.native_files ? ` · ${folder.native_files} .goodnotes (nur als PDF lesbar)` : ""}` }),
      el("button", { class: "st-save", text: "Verbinden", onClick: async () => { await api("/api/study/connect", { path: folder.path, provider: "goodnotes", label: "GoodNotes" }); close(); refresh(); } })));
  }
  pane.append(el("div", { class: "st-add-row" },
    el("button", { class: "st-link", text: "PDF-Export hochladen", onClick: () => { close(); picker.click(); } }),
    el("button", { class: "st-link", text: "Backup-Ordner angeben", onClick: () => folderForm(pane, "goodnotes", close, refresh) })));
}

/* ---------------------------------------------------------------- Sterne: the library as a sky (study_galaxy.js)

   An optional second view.  "Liste" (the default, unchanged) and "Sterne" are two quiet words beside the
   heading; in Sterne the recent material and the material list step aside and the sky takes the page under
   the search line.  The search line keeps working: its results light up in the sky. */

let galaxy = null;   // { handle, area, hits, slot } while the sky is shown

function mountViewSwitch(root, main, { input, activity, runListSearch }) {
  const choose = (mode) => (mode === "sky" ? showGalaxy(root, { input, runListSearch }) : showList(root));
  const list = el("button", { class: "st-view", type: "button", text: "Liste", onClick: () => choose("list") });
  const sky = el("button", { class: "st-view", type: "button", text: "Sterne", onClick: () => choose("sky") });
  const group = el("div", { class: "st-view-switch", role: "group", "aria-label": "Ansicht" }, list, el("span", { class: "st-view-sep", "aria-hidden": "true", text: "·" }), sky);
  main.querySelector(".st-head .st-title").after(group);
  const area = el("section", { class: "st-sky-area", hidden: true, "aria-label": "Sterne" });
  activity.after(area);
  root.studySky = { list, sky, area };
  offEvents.push(bus.on("study:changed", () => galaxy?.handle?.refresh()));
  root.dataset.studyView = "list";
  list.setAttribute("aria-pressed", "true");
  sky.setAttribute("aria-pressed", "false");
  let stored = "";
  try { stored = localStorage.getItem("zeus.study.view") || ""; } catch { /* private window */ }
  if (stored === "sky") showGalaxy(root, { input, runListSearch, remember: false });
}

function markView(root, mode, remember) {
  const parts = root.studySky;
  root.dataset.studyView = mode;
  parts.list.setAttribute("aria-pressed", String(mode === "list"));
  parts.sky.setAttribute("aria-pressed", String(mode === "sky"));
  parts.area.hidden = mode !== "sky";
  if (remember) { try { localStorage.setItem("zeus.study.view", mode); } catch { /* private window */ } }
}

async function showGalaxy(root, { input, runListSearch, remember = true }) {
  if (!root.studySky || root.dataset.studyView === "sky") return;
  markView(root, "sky", remember);
  const area = root.studySky.area;
  clear(area);
  const hits = el("div", { class: "st-sky-hits", hidden: true, "aria-live": "polite" });
  const slot = el("div", { class: "st-sky-slot" });
  const stage = el("div", { class: "st-sky-stage" });
  area.append(hits, slot, stage);
  const { mountGalaxy } = await import("./study_galaxy.js");
  if (root.dataset.studyView !== "sky" || !area.isConnected) return;   // switched back, or the view left meanwhile
  hideGalaxy();
  const current = { area, hits, slot, handle: null, root };
  current.handle = mountGalaxy(stage, {
    onOpen: (doc) => views.open("study", { doc: doc.id, unit: "1" }),
    onSearchInStudy: (doc) => {
      input.value = doc.title;
      showList(root);
      runListSearch(doc.title);
    },
    onSearchInChat: async (doc) => {
      const chat = await import("./chat.js");
      views.close();
      chat.focusComposer(`Zu „${doc.title}“: `);
    },
    onAssign: (doc) => assignSubject(slot, doc),
    onClearEmphasis: () => { clear(hits); hits.hidden = true; },
  });
  galaxy = current;
  const pending = input.value.trim();
  if (pending) searchInSky(root, pending);
}

function showList(root) {
  if (!root.studySky || root.dataset.studyView === "list") return;
  markView(root, "list", true);
  hideGalaxy();
  clear(root.studySky.area);
}

function hideGalaxy() {
  if (!galaxy) return;
  try { galaxy.handle?.destroy(); } catch { /* already gone */ }
  galaxy = null;
}

/* In Sterne, a search lights up the documents that answer it; the places themselves stay one click away. */
function searchInSky(root, query) {
  if (!root || root.dataset.studyView !== "sky" || !galaxy) return false;
  runSkySearch(galaxy, String(query || "").trim());
  return true;
}

async function runSkySearch(current, text) {
  const { hits, handle } = current;
  clear(hits);
  if (!text) { handle.setEmphasis(null); hits.hidden = true; return; }
  hits.hidden = false;
  hits.append(el("span", { class: "st-searching" }, el("span", { class: "vw-loading-line", "aria-hidden": "true" }), el("span", { text: "Sucht in deinen Unterlagen …" })));
  const found = await api("/api/study/search", { query: text, limit: 24 });
  if (galaxy !== current) return;
  clear(hits);
  if (!found || found.ok === false) {
    hits.append(el("span", { class: "st-none", text: found && found.transport ? "ZEUS ist gerade nicht erreichbar." : "Die Suche hat gerade nicht geklappt." }));
    return;
  }
  const results = found.results || [];
  if (!results.length) {
    handle.setEmphasis(null);
    hits.append(el("span", { class: "st-none", text: "In deinen Unterlagen steht dazu nichts – oder das Material ist noch nicht hier." }));
    return;
  }
  const ids = [...new Set(results.map((r) => r.document.id))];
  handle.setEmphasis(ids, found.topic || text);
  for (const result of results.slice(0, 6)) {
    const doc = result.document;
    const best = result.best;
    const compact = { document_id: doc.id, location: best.location, focus: best.focus || "", phrases: best.phrases || [], at: best.at };
    hits.append(el("button", { class: "st-sky-hit", type: "button", title: `${doc.title || doc.filename} öffnen`,
      onClick: () => views.open("study", viewParams(compact)) },
      el("span", { class: "st-sky-hit-name", text: doc.title || doc.filename }),
      el("span", { class: "st-sky-hit-where", text: (best.location && best.location.label) || "" })));
  }
}

/* "Zu Fach zuordnen": one small line above the sky; the owner's word becomes a confirmed category. */
function assignSubject(slot, doc) {
  clear(slot);
  const field = el("input", { class: "st-path", value: doc.subject || "", placeholder: "Fach", "aria-label": `Fach für ${doc.title}` });
  const done = () => clear(slot);
  slot.append(el("form", { class: "st-sky-assign", onSubmit: async (e) => {
    e.preventDefault();
    const subject = field.value.trim();
    if (subject !== String(doc.subject || "")) await api("/api/study/categories", { id: doc.id, changes: { subject } });
    done();
    bus.emit("study:changed", {});
  } },
    el("span", { class: "st-sky-assign-label", text: `Fach für „${doc.title}“` }), field,
    el("button", { class: "st-link", type: "button", text: "Abbrechen", onClick: done }),
    el("button", { class: "st-save", type: "submit", text: "Zuordnen" })));
  field.focus();
  field.select();
}
