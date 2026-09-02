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
  if ((m = t.match(/^(?:focus|fokus(?:siere)?)\s+(.+)$/i))) return { type: "graph", label: `Focus ${m[1]}`, sub: "fly to it in the galaxy", run: () => views.open("projects", { focus: m[1] }) };
  if ((m = t.match(/^(?:open|öffne)\s+(.+)$/i))) return { type: "graph", label: `Open ${m[1]}`, sub: "the project's own system", run: () => views.open("projects", { focus: m[1] }) };
  if ((m = t.match(/^show (?:everything )?connected to\s+(.+)$/i)) || (m = t.match(/^zeige (?:alles )?(?:was )?mit\s+(.+?)\s+verbunden/i))) return { type: "graph", label: `Everything connected to ${m[1]}`, sub: "local graph", run: () => views.open("projects", { connected: m[1] }) };
  if ((m = t.match(/^show projects using\s+(.+)$/i))) return { type: "graph", label: `Projects using ${m[1]}`, sub: "by capability", run: () => views.open("projects", { uses: m[1] }) };
  if ((m = t.match(/(?:haven.?t touched|untouched|not touched) in (\d+) days/i))) return { type: "graph", label: `Projects untouched for ${m[1]} days`, sub: "idle projects", run: () => views.open("projects", { idle_days: m[1] }) };
  if (/^show blocked projects$/i.test(t)) return { type: "graph", label: "Show blocked projects", run: () => views.open("projects", { filter: "blocked" }) };
  if (/^hide archived$/i.test(t)) return { type: "graph", label: "Hide archived", run: () => views.open("projects", {}) };
  return null;
}

function commands() {
  const current = views.currentView();
  const list = [
    { type: "view", label: "Mission Control", sub: "active missions, SelfDev center", run: () => views.open("missions"), keys: "Ctrl+M" },
    { type: "view", label: "Projects", sub: "constellation and deep views", run: () => views.open("projects"), keys: "Ctrl+Shift+P" },
    { type: "view", label: "Files", sub: "the real D: universe, live", run: () => views.open("files") },
    { type: "view", label: "Persönlichkeit", sub: "behaviour, rules, learning", run: () => views.open("personality") },
    { type: "view", label: "Wissen", sub: "Galaxy, Bibliothek, Bearbeiten", run: () => views.open("knowledge") },
    { type: "view", label: "Wissen: Ebenen-Liste", sub: "Strata-Ansicht", run: () => views.open("knowledge", { mode: "list" }) },
    { type: "view", label: "Knowledge graph (starfield overlay)", sub: "Overlay", run: () => knowledge.openGraph("") },
    { type: "view", label: "Activity", sub: "the operation log", run: () => views.open("activity") },
    { type: "view", label: "Korrekturen", sub: "what the owner corrected", run: () => views.open("corrections") },
    { type: "view", label: "Capabilities", sub: "acquired capabilities", run: () => views.open("capabilities") },
    { type: "view", label: "Diagnostics", sub: "is ZEUS healthy?", run: () => views.open("diagnostics") },
    { type: "view", label: "Versions", sub: "known-good, releases, rollback", run: () => views.open("release") },
    { type: "view", label: "Owner settings", sub: "identity, personality, policy", run: () => views.open("owner"), keys: "Ctrl+," },
    { type: "view", label: "Voice Studio", sub: "wake word, microphone, voice", run: () => views.open("voice") },
    { type: "action", label: "New conversation", sub: "clear the transcript", run: () => $("btnNew").click() },
    { type: "action", label: "Hide window", sub: "ZEUS keeps running; ZEUS.exe brings it back", run: () => api("/api/window/hide", { reason: "palette" }) },
    { type: "action", label: "Restart ZEUS", sub: "planned restart under the supervisor", run: async () => { if (confirm("Restart ZEUS now?")) api("/api/restart", { reason: "owner (palette)" }); } },
    { type: "action", label: "ZEUS vollständig beenden", sub: "window, voice, core, supervisor", run: async () => { if (confirm("ZEUS vollständig beenden?")) api("/api/quit", { reason: "owner (palette)" }); } },
    { type: "action", label: state.ui.reducedMotion ? "Enable motion" : "Reduce motion", sub: "accessibility", run: () => { setPref("reducedMotion", !state.ui.reducedMotion); document.body.classList.toggle("reduced-motion", state.ui.reducedMotion); } },
    { type: "action", label: "Build & verify a ZEUS.exe candidate", sub: "release pipeline", run: () => api("/api/release/build", { verify: true }) },
    { type: "action", label: "Back to ZEUS", sub: "presence mode", run: () => views.close(), keys: "Esc" },
  ];
  list.push(
    { type: "graph", label: "Show blocked projects", sub: "galaxy filtered to BLOCKED", run: () => views.open("projects", { filter: "blocked" }) },
    { type: "graph", label: "Hide archived", sub: "the default galaxy: archived and hidden projects stay out", run: () => views.open("projects", {}) },
    { type: "graph", label: "Show everything", sub: "every project, attempt and artifact", run: () => views.open("projects", { everything: "1" }) },
    { type: "graph", label: "Projects untouched for 30 days", sub: "idle projects only", run: () => views.open("projects", { idle_days: "30" }) },
    { type: "graph", label: "Projects using screen capture", sub: "by capability", run: () => views.open("projects", { uses: "screen" }) },
    { type: "view", label: "Thoughts", sub: "what ZEUS noticed on its own", run: () => views.open("thoughts") },
    { type: "view", label: "Schach Analyse", sub: "screen chess assistant", run: () => views.open("chess") },
  );
  if (current?.id === "projects" && current.params?.id) {
    list.unshift({ type: "project", label: "Ask ZEUS about this project", run: () => chat.send("What is the state of this project, what blocks it, and what is next?") });
    list.unshift({ type: "project", label: "Continue this project", run: () => chat.send("Continue the current project.") });
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
  if (!items.length) list.append(el("div", { class: "empty", style: { padding: "12px" }, text: "Nothing matches." }));
}

function run(item) {
  if (!item) return;
  close();
  try { item.run(); } catch (err) { console.error(err); }
}
