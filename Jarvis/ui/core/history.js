/* The conversation archive, where Chat keeps it: a quiet panel, not a column.

   Every conversation stays reachable -- grouped by day, searchable, pinnable,
   renameable, deletable, assignable to a project -- through the existing
   conversation routes.  Opening one swaps the live transcript
   (/api/conversation/restore) and renders it from the record.  The panel is
   opened from the chat's top bar; the empty chat shows the latest few. */

import { $, el, clear } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";

let chatMod = null;
let toastFn = null;
let conversations = [];
let projects = [];
let liveTitle = "";
let liveId = "";
let filter = "";
let renaming = "";
let menuNode = null;
let panel = null;

export function init({ chat, toast }) {
  chatMod = chat;
  toastFn = toast;
  panel = el("aside", { id: "historyPanel", class: "history-panel", hidden: true, "aria-label": "Verlauf" });
  document.body.append(panel);
  bus.on("user_message", (p) => {
    if (p._replay) return;
    if (!liveTitle && p.text) liveTitle = String(p.text).slice(0, 48);
  });
  bus.on("diagnostic", (p) => { if (p && p.conversation_summarized) refresh(); });
  document.addEventListener("click", (e) => {
    if (menuNode && !menuNode.contains(e.target)) closeMenu();
    if (panel && !panel.hidden && !panel.contains(e.target) && !e.target.closest?.("[data-history-toggle]")) close();
  });
  refresh();
}

export async function refresh() {
  const [c, p] = await Promise.all([api("/api/conversations", { limit: 80 }), api("/api/projects/overview")]);
  if (c && c.conversations) conversations = c.conversations;
  if (p && p.projects) projects = p.projects;
  if (panel && !panel.hidden) render();
  bus.emit("history:changed", conversations);
}

export function recent(limit = 4) {
  const stamp = (c) => String(c.updated_at || c.at || "");
  return [...conversations].sort((a, b) => stamp(b).localeCompare(stamp(a))).slice(0, limit);
}

export function isOpen() {
  return Boolean(panel && !panel.hidden);
}

export function toggle() {
  if (isOpen()) close(); else open();
}

export function open() {
  panel.hidden = false;
  render();
  refresh();
  setTimeout(() => panel.querySelector("input")?.focus(), 30);
}

export function close() {
  if (panel) panel.hidden = true;
  closeMenu();
}

export async function newChat() {
  const r = await api("/api/new", {});
  chatMod?.reset();
  liveTitle = "";
  liveId = "";
  close();
  if (r && r.archived) toastFn?.("Chat gespeichert.", "note");
  refresh();
  bus.emit("mode:chat", {});
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

export async function openChat(id) {
  const out = await api("/api/conversation/restore", { id });
  if (!out || out.ok === false) { toastFn?.(out?.error || "Der Chat konnte nicht geladen werden.", "warn"); return; }
  chatMod?.reset();
  for (const turn of out.turns || []) {
    chatMod?.addTurn(turn.role === "user" ? "user" : "jarvis", turn.role === "user" ? "Du" : (window.ASSISTANT_NAME || "ZEUS"), turn.text, { _replay: true });
  }
  liveId = id;
  liveTitle = out.title || "";
  close();
  bus.emit("mode:chat", { title: out.title || "" });
  refresh();
}

function row(c) {
  const item = el("div", { class: "hp-item" + (liveId === c.id ? " on" : ""), role: "listitem" });
  if (renaming === c.id) {
    const input = el("input", { class: "hp-rename", value: c.title || "", "aria-label": "Neuer Titel" });
    const done = async (save) => {
      renaming = "";
      if (save && input.value.trim() && input.value.trim() !== c.title) await api("/api/conversation/rename", { id: c.id, title: input.value.trim() });
      refresh();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); });
    input.addEventListener("blur", () => done(true));
    item.append(input);
    setTimeout(() => input.focus(), 0);
    return item;
  }
  item.append(
    el("button", { class: "hp-open", title: c.summary || c.title || "", onClick: () => openChat(c.id) },
      el("span", { class: "hp-title", text: c.title || "Chat" }),
      c.pinned ? el("span", { class: "hp-pin", text: "angeheftet" }) : null,
      c.snippet ? el("span", { class: "hp-snippet", text: c.snippet }) : null),
    el("button", { class: "hp-more", "aria-label": `Optionen für ${c.title || "Chat"}`, text: "···", onClick: (e) => { e.stopPropagation(); openMenu(e, c); } }));
  return item;
}

let searchTimer = 0;
function render() {
  clear(panel);
  const search = el("input", { class: "hp-search", placeholder: "Gespräche durchsuchen …", value: filter, "aria-label": "Gespräche durchsuchen",
    onInput: (e) => { filter = e.target.value; clearTimeout(searchTimer); searchTimer = setTimeout(renderList, 220); },
    onKeydown: (e) => { if (e.key === "Escape") { if (filter) { filter = ""; e.target.value = ""; renderList(); } else close(); } } });
  panel.append(
    el("header", { class: "hp-head" }, el("h3", { text: "Verlauf" }),
      el("button", { class: "hp-new", text: "Neues Gespräch", onClick: newChat })),
    search,
    el("div", { class: "hp-list", id: "hpList", role: "list" }));
  renderList();
}

async function renderList() {
  const list = $("hpList");
  if (!list) return;
  clear(list);
  if (filter.trim()) {
    const r = await api("/api/conversations/search", { query: filter.trim(), limit: 30 });
    const results = (r && r.results) || [];
    if (!results.length) { list.append(el("div", { class: "hp-empty", text: "Nichts gefunden." })); return; }
    for (const c of results) list.append(row(c));
    return;
  }
  if (liveTitle) list.append(el("div", { class: "hp-live" }, el("span", { class: "hp-sub", text: "Jetzt" }), el("span", { text: liveTitle })));
  const stamp = (c) => String(c.updated_at || c.at || "");
  const ordered = [...conversations].sort((a, b) => stamp(b).localeCompare(stamp(a)));
  const pinned = ordered.filter((c) => c.pinned);
  if (pinned.length) {
    list.append(el("div", { class: "hp-sub", text: "Angeheftet" }));
    for (const c of pinned) list.append(row(c));
  }
  let last = "";
  for (const c of ordered.filter((c) => !c.pinned)) {
    const group = groupOf(c.updated_at || c.at);
    if (group !== last) { list.append(el("div", { class: "hp-sub", text: group })); last = group; }
    list.append(row(c));
  }
  if (!conversations.length) list.append(el("div", { class: "hp-empty", text: "Noch keine gespeicherten Gespräche." }));
}

function closeMenu() {
  menuNode?.remove();
  menuNode = null;
}

function openMenu(e, c) {
  closeMenu();
  const menu = el("div", { class: "sb-menu glass-strong", role: "menu" });
  menu.append(el("button", { text: c.pinned ? "Loslösen" : "Anheften", onClick: async () => { closeMenu(); await api("/api/conversation/pin", { id: c.id, pinned: !c.pinned }); refresh(); } }));
  menu.append(el("button", { text: "Umbenennen", onClick: () => { closeMenu(); renaming = c.id; render(); } }));
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
  menu.style.left = Math.min(window.innerWidth - 220, rect.left - 160) + "px";
  menu.style.top = Math.min(window.innerHeight - 260, rect.bottom + 4) + "px";
  menuNode = menu;
}
