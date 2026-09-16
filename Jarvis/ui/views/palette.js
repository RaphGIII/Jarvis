/* The command palette (Ctrl+K) and universal search (Ctrl+P): ask ZEUS,
   open any view, run lifecycle actions, and search projects, missions,
   capabilities, corrections, knowledge and activity from one box. Results
   come from /api/search; commands are context-aware. */

import { $, el, clear } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import { state, setPref } from "../core/state.js";
import * as chat from "./chat.js";
import * as knowledge from "./knowledge.js";

let open_ = false;
let items = [];
let active = 0;
let timer = null;

/* A typed phrase that names a graph action: focus / open / connected to / using / untouched in N days. */
function phraseCommand(q) {
  const t = String(q || "").trim();
  let m;
  if ((m = t.match(/^(?:focus|fokus(?:siere)?)\s+(.+)$/i))) return { type: "graph", label: `Fokus ${m[1]}`, sub: "in der Galaxie anfliegen", run: () => views.open("projects", { focus: m[1] }) };
  if ((m = t.match(/^(?:open|öffne)\s+(.+)$/i))) return { type: "graph", label: `Öffne ${m[1]}`, sub: "das Projekt selbst", run: () => views.open("projects", { focus: m[1] }) };
  if ((m = t.match(/^show (?:everything )?connected to\s+(.+)$/i)) || (m = t.match(/^zeige (?:alles )?(?:was )?mit\s+(.+?)\s+verbunden/i))) return { type: "graph", label: `Alles rund um ${m[1]}`, sub: "Umgebung im Graphen", run: () => views.open("projects", { connected: m[1] }) };
  if ((m = t.match(/^show projects using\s+(.+)$/i))) return { type: "graph", label: `Projekte mit ${m[1]}`, sub: "nach Fähigkeit", run: () => views.open("projects", { uses: m[1] }) };
  if ((m = t.match(/(?:haven.?t touched|untouched|not touched) in (\d+) days/i))) return { type: "graph", label: `Projekte seit ${m[1]} Tagen unberührt`, sub: "ruhende Projekte", run: () => views.open("projects", { idle_days: m[1] }) };
  if (/^show blocked projects$/i.test(t)) return { type: "graph", label: "Blockierte Projekte zeigen", run: () => views.open("projects", { filter: "blocked" }) };
  if (/^hide archived$/i.test(t)) return { type: "graph", label: "Archivierte ausblenden", run: () => views.open("projects", {}) };
  return null;
}

function commands() {
  const current = views.currentView();
  const list = [
    { type: "view", label: "Missionen", sub: "was ZEUS gerade tut", run: () => views.open("missions"), keys: "Ctrl+M" },
    { type: "view", label: "Einstellungen", sub: "Persönlichkeit, Erscheinungsbild, Leistung", run: () => views.open("settings"), keys: "Ctrl+," },
    { type: "view", label: "Projekte", sub: "Überblick und Tiefe", run: () => views.open("projects"), keys: "Ctrl+Shift+P" },
    { type: "view", label: "Dateien", sub: "dein Rechner, live", run: () => views.open("files") },
    { type: "view", label: "Persönlichkeit", sub: "wie ZEUS spricht und sich verhält", run: () => views.open("settings", { tab: "personality" }) },
    { type: "view", label: "Wissen", sub: "Übersicht, Bibliothek, Bearbeiten", run: () => views.open("knowledge") },
    { type: "view", label: "Studium", sub: "Wissen als Ebenen", run: () => views.open("knowledge", { mode: "list" }) },
    { type: "view", label: "Wissensgraph", sub: "als Netz", run: () => knowledge.openGraph("") },
    { type: "view", label: "Fortschritt", sub: "was ZEUS getan und geprüft hat", run: () => views.open("activity") },
    { type: "view", label: "Korrekturen", sub: "was du korrigiert hast", run: () => views.open("corrections") },
    { type: "view", label: "Fähigkeiten", sub: "was ZEUS kann", run: () => views.open("capabilities") },
    { type: "view", label: "Diagnostics", sub: "technischer Zustand (Erweitert)", run: () => views.open("diagnostics") },
    { type: "view", label: "Versionen", sub: "Stände, Kandidaten, Rückkehr", run: () => views.open("release") },
    { type: "view", label: "Erweitert", sub: "Systembesitz, technische Diagnose", run: () => views.open("settings", { tab: "advanced" }) },
    { type: "view", label: "Voice", sub: "Wake-Wort, Mikrofon, Stimme", run: () => views.open("voice") },
    { type: "action", label: "Neuer Chat", sub: "ein frisches Gespräch", run: () => window.zeus?.sidebar?.newChat?.() },
    { type: "action", label: "Fenster ausblenden", sub: "ZEUS läuft weiter; ZEUS.exe holt es zurück", run: () => api("/api/window/hide", { reason: "palette" }) },
    { type: "action", label: "ZEUS neu starten", sub: "geplanter Neustart", run: async () => { if (confirm("ZEUS jetzt neu starten?")) api("/api/restart", { reason: "owner (palette)" }); } },
    { type: "action", label: "ZEUS vollständig beenden", sub: "Fenster, Stimme, Kern, alles", run: async () => { if (confirm("ZEUS vollständig beenden?")) api("/api/quit", { reason: "owner (palette)" }); } },
    { type: "action", label: state.ui.reducedMotion ? "Bewegung einschalten" : "Bewegung reduzieren", sub: "Barrierefreiheit", run: () => { setPref("reducedMotion", !state.ui.reducedMotion); document.body.classList.toggle("reduced-motion", state.ui.reducedMotion); } },
    { type: "action", label: "Kandidaten bauen und prüfen", sub: "Erweitert", run: () => api("/api/release/build", { verify: true }) },
    { type: "action", label: "Zurück zu ZEUS", sub: "zum Chat", run: () => views.close(), keys: "Esc" },
  ];
  list.push(
    { type: "graph", label: "Blockierte Projekte", sub: "nur blockierte", run: () => views.open("projects", { filter: "blocked" }) },
    { type: "graph", label: "Archivierte ausblenden", sub: "die normale Ansicht", run: () => views.open("projects", {}) },
    { type: "graph", label: "Alles zeigen", sub: "jedes Projekt, jeder Anlauf", run: () => views.open("projects", { everything: "1" }) },
    { type: "graph", label: "Projekte seit 30 Tagen unberührt", sub: "nur ruhende", run: () => views.open("projects", { idle_days: "30" }) },
    { type: "graph", label: "Projekte mit Bildschirmaufnahme", sub: "nach Fähigkeit", run: () => views.open("projects", { uses: "screen" }) },
    { type: "view", label: "Gedanken", sub: "was ZEUS von selbst bemerkt hat", run: () => views.open("thoughts") },
    { type: "view", label: "Schach-Analyse", sub: "Schachhilfe am Bildschirm", run: () => views.open("chess") },
  );
  if (current?.id === "projects" && current.params?.id) {
    list.unshift({ type: "project", label: "ZEUS zu diesem Projekt fragen", run: () => chat.send("Wie steht dieses Projekt, was blockiert es, und was kommt als Nächstes?") });
    list.unshift({ type: "project", label: "Projekt fortsetzen", run: () => chat.send("Setze das aktuelle Projekt fort.") });
  }
  return list;
}

export function init() {
  const input = $("paletteInput");
  input.addEventListener("input", () => refresh(input.value));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(items.length - 1, active + 1); paint(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(0, active - 1); paint(); }
    else if (e.key === "Enter") { e.preventDefault(); run(items[active]); }
    else if (e.key === "Escape") { close(); }
  });
  $("palette").addEventListener("click", (e) => { if (e.target === $("palette")) close(); });
}

export function isOpen() { return open_; }

export function open(prefill = "") {
  open_ = true;
  $("palette").classList.add("open");
  const input = $("paletteInput");
  input.value = prefill;
  input.focus();
  refresh(prefill);
}

export function close() {
  open_ = false;
  $("palette").classList.remove("open");
}

async function refresh(query) {
  const q = query.trim();
  const lowered = q.toLowerCase();
  const local = commands().filter((c) => !q || `${c.label} ${c.sub || ""}`.toLowerCase().includes(lowered.replace(/^search:\s*/, "")));
  items = [];
  const phrase = phraseCommand(q);
  if (phrase) items.push(phrase);
  if (q && !q.startsWith("search:")) items.push({ type: "ask", label: `Ask ZEUS: “${q}”`, sub: "send to the conversation", run: () => chat.send(q) });
  items.push(...local.slice(0, q ? 6 : 14));
  active = 0;
  paint();
  clearTimeout(timer);
  const term = q.replace(/^search:\s*/, "");
  if (term.length < 2) return;
  timer = setTimeout(async () => {
    const r = await api("/api/search", { q: term, limit: 24 });
    const hits = (r.results || []).map((h) => ({ type: h.type, label: h.title, sub: h.snippet || h.when || "", run: () => openHit(h) }));
    items = items.filter((i) => i.type === "ask").concat(hits, local.slice(0, 4));
    active = 0;
    paint();
  }, 180);
}

function openHit(h) {
  switch (h.type) {
    case "project": return views.open("projects", { id: h.id });
    case "mission": return views.open("missions", { mission: h.id });
    case "capability": return views.open("capabilities", { id: h.id });
    case "correction": return views.open("corrections");
    case "knowledge": return views.open("knowledge", { q: h.title });
    case "activity": return views.open("activity", { q: h.title.slice(0, 40) });
    case "receipt": return views.open("activity", { receipt: h.id });
    default: return views.open("activity", { q: h.title });
  }
}

function paint() {
  const list = clear($("paletteList"));
  items.forEach((item, i) => {
    const row = el("div", { class: "pal" + (i === active ? " active" : "") },
      el("span", { class: "type", text: item.type }), el("span", { class: "label", text: item.label }),
      item.sub ? el("span", { class: "sub", text: item.sub }) : null, item.keys ? el("span", { class: "hintk", text: item.keys }) : null);
    row.onmouseenter = () => { active = i; paint(); };
    row.onclick = () => run(item);
    list.append(row);
  });
  if (!items.length) list.append(el("div", { class: "empty", style: { padding: "12px" }, text: "Nichts passt." }));
}

function run(item) {
  if (!item) return;
  close();
  try { item.run(); } catch (err) { console.error(err); }
}
