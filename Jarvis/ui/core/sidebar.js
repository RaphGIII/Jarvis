/* The persistent sidebar: one place to start a chat, find one, reach a
   project, a workspace surface or the system settings.

   Chats are the owner's conversations (the archive plus the live one),
   grouped by day, pinnable, renameable, deletable, searchable and
   assignable to a project.  Everything here is server truth read through
   the existing conversation and project routes; nothing about providers,
   models or internal phases is shown.  A chat opens immediately: the live
   transcript is swapped through /api/conversation/restore and rendered
   from the record. */

import { $, el, clear } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";
import * as views from "./views.js";

const WORKSPACE = [
  ["capabilities", "Fähigkeiten", "◇"],
  ["knowledge", "Wissen", "◈"],
  ["knowledge", "Studium", "▤", { mode: "list" }],
  ["activity", "Fortschritt", "◷"],
  ["calendar", "Kalender", "▦"],
  ["files", "Dateien", "▣"],
  ["missions", "Laufende Arbeit", "◐"],
];
const SYSTEM = [
  ["settings", "Persönlichkeit", "✦", { tab: "personality" }],
  ["settings", "Einstellungen", "⚙", {}],
];

let conversations = [];
let projects = [];
let liveTitle = "";
let liveId = "";
let chatMod = null;
let toastFn = null;
let filter = "";
let menuNode = null;
let renaming = "";

export function init({ chat, toast }) {
  chatMod = chat;
  toastFn = toast;
  build();
  refresh();
  bus.on("user_message", (p) => {
    if (p._replay) return;
    if (!liveTitle && p.text) { liveTitle = String(p.text).slice(0, 48); renderChats(); }
  });
  bus.on("view:open", renderNav);
  bus.on("view:close", renderNav);
  bus.on("diagnostic", (p) => { if (p && p.conversation_summarized) refresh(); });
  setInterval(refresh, 90_000);
  document.addEventListener("click", (e) => { if (menuNode && !menuNode.contains(e.target)) closeMenu(); });
}

export async function refresh() {
  const [c, p] = await Promise.all([api("/api/conversations", { limit: 60 }), api("/api/projects/overview")]);
  if (c && c.conversations) conversations = c.conversations;
  if (p && p.projects) projects = p.projects;
  renderChats();
  renderProjects();
}

export function newChat() {
  return startNew();
}

function build() {
  const side = $("sidebar");
  clear(side);
  side.append(
    el("div", { class: "sb-head" }, el("div", { class: "sb-brand", onClick: () => views.close() }, window.PRODUCT_NAME || "ZEUS",
      el("small", { text: "persönliche Intelligenz" }))),
    el("div", { class: "sb-actions" },
      el("button", { class: "sb-new", onClick: startNew }, el("span", { class: "plus", text: "+" }), "Neuer Chat"),
      el("label", { class: "sb-search" }, el("span", { text: "⌕" }),
        el("input", { id: "sbSearch", placeholder: "Suche in Chats …", onInput: (e) => { filter = e.target.value; searchChats(); },
                      onKeydown: (e) => { if (e.key === "Escape") { e.target.value = ""; filter = ""; renderChats(); } } }))),
    el("div", { class: "sb-body" },
      el("div", { class: "sb-group", id: "sbChats" }),
      el("div", { class: "sb-group", id: "sbProjects" }),
      el("div", { class: "sb-group", id: "sbWorkspace" }),
      el("div", { class: "sb-group", id: "sbSystem" })),
    el("div", { class: "sb-foot", id: "sbFoot" }),
  );
  renderNav();
}

async function startNew() {
  const r = await api("/api/new", {});
  chatMod?.reset();
  liveTitle = "";
  liveId = "";
  views.close();
  if (r && r.archived) toastFn?.("Chat gespeichert.", "note");
  refresh();
  $("input")?.focus();
}

function groupOf(iso) {
  const day = String(iso || "").slice(0, 10);
  const today = new Date();
  const d = (n) => new Date(today.getTime() - n * 864e5).toISOString().slice(0, 10);
  if (day === d(0)) return "Heute";
  if (day === d(1)) return "Gestern";
  if (day >= d(7)) return "Letzte 7 Tage";
  return "Älter";
}

function chatRow(c, { live = false } = {}) {
  const on = live ? !views.isWorkspace() && liveId === "" : liveId === c.id && !views.isWorkspace();
  const row = el("button", { class: "sb-item" + (on ? " on" : ""), title: c.summary || c.title || "",
                             onClick: () => (live ? views.close() : openChat(c.id)) });
  if (renaming && renaming === c.id) {
    const input = el("input", { class: "sb-rename", value: c.title || "" });
    const done = async (save) => {
      renaming = "";
      if (save && input.value.trim() && input.value.trim() !== c.title) {
        await api("/api/conversation/rename", { id: c.id, title: input.value.trim() });
      }
      refresh();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); });
    input.addEventListener("blur", () => done(true));
    row.append(input);
    setTimeout(() => input.focus(), 0);
    return row;
  }
  row.append(el("span", { class: "ico", text: live ? "◉" : "○" }),
             el("span", { class: "lbl", text: c.title || "Neuer Chat" }));
  if (c.pinned) row.append(el("span", { class: "pin", text: "★" }));
  if (!live) row.append(el("span", { class: "more", text: "⋯", title: "Optionen", onClick: (e) => { e.stopPropagation(); openMenu(e, c); } }));
  return row;
}

function renderChats() {
  const box = $("sbChats");
  if (!box) return;
  clear(box);
  box.append(el("h5", { text: "Chats" }));
  if (filter.trim()) return; // the search renders its own list
  const live = { id: "", title: liveTitle || "Aktueller Chat", summary: "" };
  box.append(chatRow(live, { live: true }));
  const pinned = conversations.filter((c) => c.pinned);
  const rest = conversations.filter((c) => !c.pinned);
  if (pinned.length) {
    box.append(el("div", { class: "sb-sub", text: "Angeheftet" }));
    for (const c of pinned) box.append(chatRow(c));
  }
  let lastGroup = "";
  for (const c of rest) {
    const group = groupOf(c.at);
    if (group !== lastGroup) { box.append(el("div", { class: "sb-sub", text: group })); lastGroup = group; }
    box.append(chatRow(c));
  }
  if (!conversations.length) box.append(el("div", { class: "sb-empty", text: "Noch keine gespeicherten Chats." }));
}

let searchTimer = null;
function searchChats() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const box = $("sbChats");
    if (!box) return;
    clear(box);
    box.append(el("h5", { text: "Suche" }));
    if (!filter.trim()) { renderChats(); return; }
    const r = await api("/api/conversations/search", { query: filter.trim(), limit: 30 });
    const results = (r && r.results) || [];
    if (!results.length) { box.append(el("div", { class: "sb-empty", text: "Nichts gefunden." })); return; }
    for (const c of results) {
      const row = chatRow(c);
      if (c.snippet) row.title = c.snippet;
      box.append(row);
    }
  }, 220);
}

function renderProjects() {
  const box = $("sbProjects");
  if (!box) return;
  clear(box);
  box.append(el("h5", { text: "Projekte" }));
  const shown = projects.slice(0, 8);
  for (const p of shown) {
    box.append(el("button", { class: "sb-item", title: p.goal || "", onClick: () => views.open("projects", { focus: p.title || p.id }) },
      el("span", { class: "ico", text: "▪" }), el("span", { class: "lbl", text: p.title || p.id }),
      p.tasks ? el("span", { class: "n", text: `${p.tasks_done || 0}/${p.tasks}` }) : null));
  }
  if (!projects.length) box.append(el("div", { class: "sb-empty", text: "Noch keine Projekte." }));
  if (projects.length > 8) box.append(el("button", { class: "sb-item", onClick: () => views.open("projects") },
    el("span", { class: "ico", text: "…" }), el("span", { class: "lbl", text: `Alle ${projects.length} Projekte` })));
}

function renderNav() {
  const ws = $("sbWorkspace");
  const sys = $("sbSystem");
  if (!ws || !sys) return;
  const current = views.currentView();
  const isOn = (id, params) => current && current.id === id && (!params || Object.entries(params).every(([k, v]) => (current.params || {})[k] === v))
    && (params || !Object.keys(current.params || {}).length || id !== "knowledge" && id !== "settings");
  clear(ws); clear(sys);
  ws.append(el("h5", { text: "Workspace" }));
  for (const [id, label, ico, params] of WORKSPACE) {
    ws.append(el("button", { class: "sb-item" + (isOn(id, params) ? " on" : ""), onClick: () => views.open(id, params || {}) },
      el("span", { class: "ico", text: ico }), el("span", { class: "lbl", text: label })));
  }
  sys.append(el("h5", { text: "System" }));
  for (const [id, label, ico, params] of SYSTEM) {
    sys.append(el("button", { class: "sb-item" + (isOn(id, params) ? " on" : ""), onClick: () => views.open(id, params || {}) },
      el("span", { class: "ico", text: ico }), el("span", { class: "lbl", text: label })));
  }
  renderChats();
}

async function openChat(id) {
  const out = await api("/api/conversation/restore", { id });
  if (!out || out.ok === false) { toastFn?.(out?.error || "Der Chat konnte nicht geladen werden.", "warn"); return; }
  views.close({ push: false });
  chatMod?.reset();
  for (const turn of out.turns || []) {
    chatMod?.addTurn(turn.role === "user" ? "user" : "jarvis", turn.role === "user" ? "Du" : (window.ASSISTANT_NAME || "ZEUS"), turn.text, { _replay: true });
  }
  liveId = id;
  liveTitle = out.title || "";
  $("topTitle").textContent = out.title || "Chat";
  refresh();
}

function closeMenu() {
  menuNode?.remove();
  menuNode = null;
}

function openMenu(e, c) {
  closeMenu();
  const menu = el("div", { class: "sb-menu glass-strong" });
  menu.append(el("h6", { text: (c.title || "Chat").slice(0, 40) }));
  menu.append(el("button", { text: c.pinned ? "Loslösen" : "Anheften", onClick: async () => { closeMenu(); await api("/api/conversation/pin", { id: c.id, pinned: !c.pinned }); refresh(); } }));
  menu.append(el("button", { text: "Umbenennen", onClick: () => { closeMenu(); renaming = c.id; renderChats(); } }));
  if (projects.length) {
    menu.append(el("div", { class: "sep" }));
    menu.append(el("h6", { text: "Projekt zuordnen" }));
    for (const p of projects.slice(0, 6)) {
      menu.append(el("button", { text: (c.project_id === p.id ? "✓ " : "") + (p.title || p.id), onClick: async () => {
        closeMenu(); await api("/api/conversation/assign", { id: c.id, project_id: c.project_id === p.id ? "" : p.id }); refresh(); } }));
    }
  }
  menu.append(el("div", { class: "sep" }));
  menu.append(el("button", { class: "danger", text: "Löschen", onClick: async () => {
    closeMenu();
    if (!confirm(`„${c.title || "Chat"}“ endgültig löschen?`)) return;
    await api("/api/conversation/delete", { id: c.id });
    if (liveId === c.id) { liveId = ""; liveTitle = ""; }
    refresh();
  } }));
  document.body.append(menu);
  const rect = e.currentTarget.getBoundingClientRect();
  menu.style.left = Math.min(window.innerWidth - 220, rect.left) + "px";
  menu.style.top = Math.min(window.innerHeight - 260, rect.bottom + 4) + "px";
  menuNode = menu;
}
