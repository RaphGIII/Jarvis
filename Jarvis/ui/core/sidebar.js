/* The persistent sidebar: one place to start a chat, find one, reach a
   project, a workspace surface or the system settings.

   Chats are the owner's conversations (the archive plus the live one),
   grouped by day, pinnable, renameable, deletable, searchable and
   assignable to a project.  Everything here is server truth read through
   the existing conversation and project routes; nothing about providers,
   models or internal phases is shown.  A chat opens immediately: the live
   transcript is swapped through /api/conversation/restore and rendered
   from the record.  At the bottom: the owner, and what ZEUS is doing, in
   one quiet line. */

import { $, el, clear } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";
import * as views from "./views.js";

/* thin monochrome icons, one stroke weight, drawn inline */
const ICON = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
  chat: '<path d="M4.5 5.5h15v10h-8l-4 3.5v-3.5h-3z"/>',
  live: '<path d="M4.5 5.5h15v10h-8l-4 3.5v-3.5h-3z"/><circle cx="12" cy="10.5" r="1.4" fill="currentColor" stroke="none"/>',
  project: '<path d="M4 7.5a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/>',
  knowledge: '<path d="M5 5.5h6.5v13H5zM12.5 5.5H19v13h-6.5z"/><path d="M7.5 9h1.5M7.5 12h1.5M15 9h1.5M15 12h1.5"/>',
  skills: '<path d="M12 3.5l2.2 5 5.3.6-4 3.6 1.2 5.3L12 15.4 7.3 18l1.2-5.3-4-3.6 5.3-.6z"/>',
  study: '<path d="M3.5 8.5 12 4.5l8.5 4L12 12.5z"/><path d="M6.5 10v5c0 1.5 2.5 3 5.5 3s5.5-1.5 5.5-3v-5"/><path d="M20.5 8.5v5"/>',
  files: '<path d="M7 3.5h7l4 4v13H7z"/><path d="M14 3.5v4h4"/>',
  calendar: '<rect x="3.5" y="5" width="17" height="15" rx="2.5"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
  missions: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  progress: '<path d="M4 18l5-6 4 3 7-9"/>',
  personality: '<circle cx="12" cy="9" r="3.5"/><path d="M5.5 19.5c1-3.5 3.6-5 6.5-5s5.5 1.5 6.5 5"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M12 3.5v2.3M12 18.2v2.3M3.5 12h2.3M18.2 12h2.3M6 6l1.6 1.6M16.4 16.4 18 18M6 18l1.6-1.6M16.4 7.6 18 6"/>',
  more: '<circle cx="6" cy="12" r="1.1" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.1" fill="currentColor" stroke="none"/><circle cx="18" cy="12" r="1.1" fill="currentColor" stroke="none"/>',
};
const icon = (name) => el("span", { class: "i", html: `<svg viewBox="0 0 24 24">${ICON[name] || ""}</svg>` });

const WORKSPACE = [
  ["knowledge", "Wissen", "knowledge"],
  ["capabilities", "Fähigkeiten", "skills"],
  ["knowledge", "Studium", "study", { mode: "list" }],
  ["files", "Dateien", "files"],
  ["calendar", "Kalender", "calendar"],
  ["missions", "Missionen", "missions"],
];
const SYSTEM = [
  ["settings", "Persönlichkeit", "personality", { tab: "personality" }],
  ["settings", "Einstellungen", "settings", {}],
];
const STATE_WORDS = {
  idle: "ZEUS ist bereit.", listening: "ZEUS hört zu.", transcribing: "ZEUS versteht.", thinking: "ZEUS denkt.", speaking: "ZEUS spricht.",
  waiting: "ZEUS wartet auf dich.", working: "ZEUS arbeitet.", verifying: "ZEUS prüft.", coding: "ZEUS entwickelt.", researching: "ZEUS recherchiert.",
  error: "ZEUS wurde unterbrochen.", offline: "ZEUS ist offline.",
};

let conversations = [];
let projects = [];
let liveTitle = "";
let liveId = "";
let chatMod = null;
let toastFn = null;
let filter = "";
let menuNode = null;
let renaming = "";
let ownerName = "Raphael";

export function init({ chat, toast }) {
  chatMod = chat;
  toastFn = toast;
  build();
  refresh();
  api("/api/diagnostics").then((d) => {
    const who = d && d.identity && (d.identity.creator || d.identity.owner_name);
    if (who) { ownerName = String(who); renderFoot(); }
  }).catch(() => {});
  bus.on("user_message", (p) => {
    if (p._replay) return;
    if (!liveTitle && p.text) { liveTitle = String(p.text).slice(0, 48); renderChats(); }
  });
  bus.on("state", (p) => renderFoot(p && p.state));
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
    el("div", { class: "sb-head", onClick: () => views.close() }, el("span", { class: "sb-mark" }), el("span", { class: "sb-brand", text: window.PRODUCT_NAME || "ZEUS" })),
    el("div", { class: "sb-actions" },
      el("button", { class: "sb-new", onClick: startNew }, icon("plus"), "Neuer Chat"),
      el("label", { class: "sb-search" }, icon("search"),
        el("input", { id: "sbSearch", placeholder: "Suche in Chats …", onInput: (e) => { filter = e.target.value; searchChats(); },
                      onKeydown: (e) => { if (e.key === "Escape") { e.target.value = ""; filter = ""; renderChats(); } } }))),
    el("div", { class: "sb-body" },
      el("div", { class: "sb-group", id: "sbChats" }),
      el("div", { class: "sb-group", id: "sbProjects" }),
      el("div", { class: "sb-group", id: "sbWorkspace" }),
      el("div", { class: "sb-group", id: "sbSystem" })),
    el("div", { class: "sb-foot", id: "sbFoot", onClick: () => views.open("settings", {}) }),
  );
  renderNav();
  renderFoot();
}

let lastState = "offline";
function renderFoot(stateName) {
  const foot = $("sbFoot");
  if (!foot) return;
  if (stateName) lastState = stateName;
  clear(foot);
  const tone = lastState === "error" ? "bad" : lastState === "offline" ? "" : lastState === "waiting" ? "warn" : "ok";
  foot.append(
    el("span", { class: "avatar", text: ownerName.slice(0, 1).toUpperCase() }),
    el("div", { class: "who" },
      el("div", { class: "name", text: ownerName }),
      el("div", { class: "st" }, el("span", { class: "dot " + tone }), el("span", { text: STATE_WORDS[lastState] || STATE_WORDS.idle }))),
  );
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
  row.append(icon(live ? "live" : "chat"), el("span", { class: "lbl", text: c.title || "Neuer Chat" }));
  if (c.pinned) row.append(el("span", { class: "pin", text: "●" }));
  if (!live) {
    const more = el("span", { class: "more", title: "Optionen", html: `<svg class="i" viewBox="0 0 24 24" style="width:14px;height:14px">${ICON.more}</svg>`,
                              onClick: (e) => { e.stopPropagation(); openMenu(e, c); } });
    row.append(more);
  }
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

/* The owner's projects -- never ZEUS's own engineering jobs. */
function renderProjects() {
  const box = $("sbProjects");
  if (!box) return;
  clear(box);
  box.append(el("h5", { text: "Projekte" }));
  const own = projects.filter((p) => (p.origin || "owner") === "owner");
  const shown = own.slice(0, 8);
  for (const p of shown) {
    box.append(el("button", { class: "sb-item", title: p.goal || "", onClick: () => views.open("projects", { focus: p.title || p.id }) },
      icon("project"), el("span", { class: "lbl", text: p.title || p.id }),
      p.tasks ? el("span", { class: "n", text: `${p.tasks_done || 0}/${p.tasks}` }) : null));
  }
  if (own.length > 8) box.append(el("button", { class: "sb-item quiet", onClick: () => views.open("projects") },
    el("span", { class: "i" }), el("span", { class: "lbl", text: `Alle ${own.length} Projekte` })));
  box.append(el("button", { class: "sb-item quiet", onClick: newProject }, icon("plus"), el("span", { class: "lbl", text: "Neues Projekt" })));
}

function newProject() {
  views.close();
  chatMod?.focusComposer?.("Lege ein neues Projekt an: ");
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
      icon(ico), el("span", { class: "lbl", text: label })));
  }
  sys.append(el("h5", { text: "System" }));
  for (const [id, label, ico, params] of SYSTEM) {
    sys.append(el("button", { class: "sb-item" + (isOn(id, params) ? " on" : ""), onClick: () => views.open(id, params || {}) },
      icon(ico), el("span", { class: "lbl", text: label })));
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
